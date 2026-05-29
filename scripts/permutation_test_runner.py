"""permutation_test_runner.py — Block-bootstrap permutation test for daily-dispatch
survivors.

Mission (council R4 task #54, 2026-05-29): given the 14+ daily-dispatch champions with
HoldSR > 1.5, test whether the live result is REAL edge or selection artifact. For each
survivor, reconstruct the winning SAP, generate N block-bootstrap permutations of the
ticker's daily returns (20-day blocks to preserve short-term autocorrelation), re-run
the SAP on each permutation, measure HoldSR on the holdout window (2025-01-01 onward).
Verdict:
  REAL     — survival_rate < 5%  of permuted runs produce HoldSR > 1.5 (live signal)
  SUSPECT  — 5% <= survival_rate < 20%
  ARTIFACT — survival_rate >= 20% (champion is noise)

Methodology notes (mandatory, statistical correctness):
  * Block bootstrap of LOG RETURNS in 20-day contiguous blocks, sampled with replacement,
    reconstructed via cumulative product to preserve cross-period serial structure.
  * OHLC reconstruction: we permute log close-to-close returns and rebuild close;
    high/low/open are scaled proportionally to close so that intra-bar geometry is
    preserved relative to close (so ATR / Donchian / VWAP etc. compute on a coherent
    OHLC). This is a documented limitation — true intra-bar permutation would require
    a joint bootstrap of all four series. We document this caveat in the report.
  * Volume is block-bootstrapped INDEPENDENTLY (it has different autocorrelation) and
    re-aligned to the same calendar bars. VWAP under the permuted volume is therefore
    valid but the price-volume correlation is broken (which is part of the null).
  * Calendar (timestamps, holdout cutoff) is preserved. The same calendar fraction of
    bars is used for the holdout window in each permutation.
  * Confidence interval reporting: 95% one-sided upper tail.

Public API:
  run_permutation_test(ticker, sap_id, n_permutations=1000, block_size_days=20) -> dict

Output layout:
  Per-ticker raw: /Volumes/ZG-2TB/zg/permutation_test/results/<utc>/<TICKER>_<sap_id>.json
  Summary CSV:    AI-Tools/reports/permutation_test_2026-05-29.csv
  Summary MD:     AI-Tools/reports/permutation_test_2026-05-29.md

Dependencies: pandas + numpy + lab.hypothesis_runner + lab.championship_search +
lab.indicator_hardening_runner (sibling modules).

Run:
  python permutation_test_runner.py --n-perms 500 --workers 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "lab"))

# Lab imports (sibling modules)
import hypothesis_runner as _hr  # noqa: E402
import championship_search as _cs  # noqa: E402
import indicator_hardening_runner as _ihr  # noqa: E402

# Storage tiers
RESULTS_LOCAL = Path("/Volumes/ZG-2TB/zg/permutation_test/results")
DRIVE_REPORTS = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools/reports"
)
POSTERIOR_DIR = _cs.POSTERIOR_DIR
DAILY_CACHE = _ihr.DRIVE_OHLC_DAILY

# Constants
# Holdout definition mirrors run_hypothesis_for_ticker EXACTLY: last 10% of bars
# (cut = int(0.9 * n_obs)). NOT a calendar-based cutoff.
# We support a calendar cutoff option for compatibility with the council brief, but the
# DEFAULT matches the live dispatcher so live HoldSR exactly reproduces.
HOLDOUT_FRACTION = 0.10  # last 10% of bars (matches hypothesis_runner cut = int(0.9 * n_obs))
HOLDOUT_CUTOFF_CALENDAR = "2025-01-01"  # alternative — not used by default
HOLDSR_THRESHOLD = 1.5  # council's "live result is real" bar
BLOCK_SIZE_DAYS = 20
BARS_PER_YEAR = 252
COST_PER_SIDE = 5e-4  # mirror hypothesis_runner

# RNG seed — different per ticker for reproducibility across runs
def _rng_seed_for(ticker: str, base: int = 20260529) -> int:
    # Stable hash so re-runs are reproducible
    h = 0
    for ch in ticker:
        h = (h * 31 + ord(ch)) & 0xFFFF_FFFF
    return (base ^ h) & 0x7FFF_FFFF


# ============================================================================
# Survivor identification
# ============================================================================

def load_survivors_from_dispatch(roll_up_path: Path) -> List[Tuple[str, str, float]]:
    """Parse the dispatch roll-up markdown for tickers with HoldSR > threshold.

    Returns list of (ticker, sap_id, live_holdsr).
    """
    txt = roll_up_path.read_text()
    out: List[Tuple[str, str, float]] = []
    in_table = False
    for line in txt.splitlines():
        if line.strip().startswith("| # | Ticker"):
            in_table = True
            continue
        if in_table:
            if not line.strip().startswith("|"):
                if out:  # already collected some, leave on blank line
                    break
                continue
            if line.strip().startswith("|---") or line.strip().startswith("|--"):
                continue
            parts = [c.strip() for c in line.split("|")[1:-1]]
            if len(parts) < 4:
                continue
            try:
                rank = int(parts[0])
                ticker = parts[1]
                sap_id = parts[2]
                holdsr = float(parts[3])
            except (ValueError, IndexError):
                continue
            if holdsr >= HOLDSR_THRESHOLD:
                out.append((ticker, sap_id, holdsr))
    return out


def reconstruct_hypothesis_from_posterior(ticker: str, sap_id: str) -> Optional[dict]:
    """Re-build the hypothesis dict from posterior history.

    Strategy: posterior.history[i] has parent_seed_id + perturb_params. Re-render via
    championship_search variant_generator deterministic ordering — find the variant
    with matching id, return its dict.
    """
    posterior_path = POSTERIOR_DIR / f"{ticker.upper()}.json"
    if not posterior_path.exists():
        return None
    try:
        posterior = json.loads(posterior_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    # Find matching history entry to get (parent_seed_id, perturb_params)
    matching = None
    for h in posterior.get("history", []):
        if h.get("sap_id") == sap_id:
            matching = h
            break
    if matching is None:
        return None
    parent_seed = matching.get("parent_seed_id")
    perturb = matching.get("perturb_params")
    if not parent_seed or not perturb:
        return None
    # Re-render via _SEED_TEMPLATES
    seed_tmpl = None
    for s in _cs._SEED_TEMPLATES:
        if s["seed_id"] == parent_seed:
            seed_tmpl = s
            break
    if seed_tmpl is None:
        return None
    tmpl = _cs._format_template(seed_tmpl, perturb)
    hypothesis = {
        "id": sap_id,
        "name": f"{parent_seed} @ {perturb}",
        "thesis": f"Reconstructed for permutation test ({ticker})",
        "parent_seed_id": parent_seed,
        "perturb_params": perturb,
        "regime_gate": tmpl.get("regime_gate", "TRUE"),
        "bias_filter": tmpl.get("bias_filter", "TRUE"),
        "trigger": tmpl.get("trigger", "FALSE"),
        "confirmation": tmpl.get("confirmation", "TRUE"),
        "timing": tmpl.get("timing", "TRUE"),
        "exit": tmpl.get("exit", "FALSE"),
        "no_trade": tmpl.get("no_trade", "FALSE"),
        "side": tmpl.get("side", "long"),
        "cost": "5bps_per_side",
        "universe": f"single_ticker:{ticker.upper()}",
        "timeframe": "1d",
        "data_sources": list(seed_tmpl.get("data_sources", [])),
    }
    if "child_hypotheses" in tmpl:
        hypothesis["child_hypotheses"] = tmpl["child_hypotheses"]
    if "alt_data_overlay" in tmpl:
        hypothesis["alt_data_overlay"] = tmpl["alt_data_overlay"]
    return hypothesis


# ============================================================================
# OHLC load + holdout split
# ============================================================================

def load_ohlc_dataframe(ticker: str) -> Optional[pd.DataFrame]:
    """Load OHLCV with date column from the yfinance daily cache."""
    p = DAILY_CACHE / f"{ticker}.parquet"
    if not p.exists():
        return None
    for attempt in range(3):
        try:
            df = pd.read_parquet(p)
            break
        except OSError as e:
            if attempt == 2:
                print(f"  [load] FAILED {ticker}: {e}", flush=True)
                return None
            time.sleep(1)
    needed = {"open", "high", "low", "close", "volume", "date"}
    if not needed.issubset(df.columns):
        return None
    df = df.dropna(subset=list(needed)).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def holdout_start_index_fraction(n: int, frac: float = HOLDOUT_FRACTION) -> int:
    """Mirror hypothesis_runner: cut = int((1 - frac) * n).

    For frac=0.10, this is the last 10% of bars.
    """
    return int((1.0 - frac) * n)


def holdout_start_index_calendar(df: pd.DataFrame, cutoff: str = HOLDOUT_CUTOFF_CALENDAR) -> int:
    """Return the index of the first bar >= cutoff date (calendar-based alternative)."""
    cut = pd.Timestamp(cutoff)
    mask = df["date"] >= cut
    if not mask.any():
        return len(df)
    return int(mask.idxmax())


# ============================================================================
# Block bootstrap
# ============================================================================

def block_bootstrap_indices(
    n: int, block_size: int, rng: np.random.Generator
) -> np.ndarray:
    """Generate length-n indices via circular block bootstrap.

    Sample contiguous blocks of size `block_size` with replacement, starting at
    random indices in [0, n) (wrap-around at the boundary), concatenate, truncate
    to length n. Preserves autocorrelation up to lag block_size-1.
    """
    n_blocks = (n + block_size - 1) // block_size
    starts = rng.integers(0, n, size=n_blocks)
    out = np.empty(n_blocks * block_size, dtype=np.int64)
    for i, s in enumerate(starts):
        # Circular wrap
        idx = (s + np.arange(block_size)) % n
        out[i * block_size:(i + 1) * block_size] = idx
    return out[:n]


def bootstrap_ohlcv(
    df: pd.DataFrame, block_size: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Return a block-bootstrapped OHLCV dataframe with the same calendar.

    Method:
      1. Compute close-to-close log returns r_t = log(C_t / C_{t-1}).
      2. Block-bootstrap r_t.
      3. Reconstruct C from cumulative product starting at the original first close.
      4. Compute scale factors per bar = C_new / C_old (relative to permuted close vs
         what the close would have been on the same bar without permutation — i.e. we
         take the SAME index's original OHL ratio to close and apply it to the new
         close to keep intra-bar geometry intact relative to the permuted close).
         Concretely: for each permuted bar i with original index j (the bar from which
         it was drawn), set:
             new_close[i] = (rebuilt from log-returns)
             new_open[i]  = new_close[i] * (open[j]  / close[j])
             new_high[i]  = new_close[i] * (high[j]  / close[j])
             new_low[i]   = new_close[i] * (low[j]   / close[j])
         This preserves the intra-bar HLOC ratios from the source bar.
      5. Volume is independently block-bootstrapped (different autocorr structure).
      6. Date column is the ORIGINAL calendar — only the bar content is permuted.

    Caveat: bootstrap of just returns leaves a degree of freedom — we use the
    intra-bar ratio from the source bar to anchor OHL relative to close. This
    means within each permuted bar, the bar's "shape" is the source bar's shape;
    but consecutive bars no longer share a real high/low handoff (gaps possible).
    For ATR/Donchian/VWAP this is acceptable; for tick-level strategies it's not.
    """
    n = len(df)
    close = df["close"].to_numpy(dtype=np.float64)
    open_ = df["open"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    volume = df["volume"].to_numpy(dtype=np.float64)

    # 1. Log returns of close
    log_ret = np.zeros(n)
    log_ret[1:] = np.log(close[1:] / close[:-1])

    # 2. Block-bootstrap RETURNS (not bars). idx[i] -> source bar for bar i.
    idx_returns = block_bootstrap_indices(n, block_size, rng)
    # We bootstrap the *return* series, then bar i takes its OHL-to-close ratios
    # from the same source bar i_src that contributed the return.
    perm_log_ret = log_ret[idx_returns]
    # First bar has return 0 conventionally
    perm_log_ret[0] = 0.0

    # 3. Reconstruct close via cumulative product
    new_close = np.empty(n)
    new_close[0] = close[0]
    new_close[1:] = close[0] * np.exp(np.cumsum(perm_log_ret[1:]))

    # 4. Intra-bar geometry from the source bar
    # ratios are open[j]/close[j], high[j]/close[j], low[j]/close[j] at source bar j
    safe_close = np.where(close > 0, close, 1.0)
    ratio_open = open_ / safe_close
    ratio_high = high / safe_close
    ratio_low = low / safe_close
    new_open = new_close * ratio_open[idx_returns]
    new_high = new_close * ratio_high[idx_returns]
    new_low = new_close * ratio_low[idx_returns]
    # Sanity: ensure high >= max(open, close), low <= min(open, close)
    new_high = np.maximum.reduce([new_high, new_open, new_close])
    new_low = np.minimum.reduce([new_low, new_open, new_close])

    # 5. Independent block-bootstrap of volume
    idx_vol = block_bootstrap_indices(n, block_size, rng)
    new_volume = volume[idx_vol]

    out = pd.DataFrame({
        "date": df["date"].to_numpy(),  # ORIGINAL calendar
        "open": new_open,
        "high": new_high,
        "low": new_low,
        "close": new_close,
        "volume": new_volume,
    })
    return out


# ============================================================================
# Per-permutation evaluation
# ============================================================================

def _bars_dict_from_df(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    return {k: df[k].to_numpy(dtype=np.float64)
            for k in ("open", "high", "low", "close", "volume")}


def _annualized_sharpe(rets: np.ndarray) -> float:
    """Annualized Sharpe; returns NaN if too few non-NaN obs, 0.0 if std==0 (no variance,
    e.g. strategy didn't trade in this window — realized HoldSR is zero, not undefined)."""
    rets = rets[~np.isnan(rets)]
    if rets.size < 50:
        return float("nan")
    sd = np.std(rets, ddof=1)
    if sd == 0:
        # Strategy did not trade in this window — realized HoldSR is 0 (no edge,
        # no losses). This is the correct null value for permutation testing:
        # a "no-trade" outcome is NOT a positive HoldSR and must not be excluded.
        return 0.0
    return float(np.mean(rets) / sd * np.sqrt(BARS_PER_YEAR))


def evaluate_sap_on_bars(
    hypothesis: dict, df: pd.DataFrame, holdout_idx: int
) -> Tuple[float, float, int, float]:
    """Run the SAP on a bar dataframe; return (full_sharpe, holdout_sharpe, n_trades, win_rate).

    Holdout is bars[holdout_idx:]. We avoid the full 6-step pipeline (walk-forward, PBO,
    DSR, stability) — those are not the per-permutation metric. We only need:
      - Holdout Sharpe (the threshold metric)
      - Win rate, n_trades (descriptive)
    """
    bars = _bars_dict_from_df(df)
    try:
        # NB: evaluate_hypothesis does NOT use alt_data_resolver here — the 5 seed
        # templates currently only reference OHLCV indicators in their realized
        # form (GOV_AWARE has alt_data_overlay as metadata but the *realized*
        # role strings are pure-tech, per championship_search seed template).
        # If a future seed adds alt-data tokens to the role strings, they'd be
        # passed through unchanged on permuted bars (which is intentional — alt
        # data IS unaffected by the OHLCV permutation since it's external).
        pos = _hr.evaluate_hypothesis(bars, hypothesis, alt_data_resolver=None)
    except Exception as e:
        return (float("nan"), float("nan"), 0, float("nan"))
    rets = _hr.returns_from_position(bars, pos)
    full_sr = _annualized_sharpe(rets)
    if holdout_idx >= len(rets):
        hold_sr = float("nan")
    else:
        hold_sr = _annualized_sharpe(rets[holdout_idx:])
    wr, n_trades = _hr.win_rate_from_position(bars, pos)
    return (full_sr, hold_sr, int(n_trades), float(wr))


# ============================================================================
# Main per-ticker permutation loop
# ============================================================================

def _single_perm_worker(args):
    """Worker for multiprocessing — runs one permutation."""
    ticker, hypothesis, df_pickled, holdout_idx, block_size, seed = args
    df = df_pickled  # already a dataframe
    rng = np.random.default_rng(seed)
    perm_df = bootstrap_ohlcv(df, block_size, rng)
    full_sr, hold_sr, n_trades, wr = evaluate_sap_on_bars(
        hypothesis, perm_df, holdout_idx
    )
    return {
        "seed": seed,
        "full_sharpe": full_sr,
        "holdout_sharpe": hold_sr,
        "n_trades": n_trades,
        "win_rate": wr,
    }


def run_permutation_test(
    ticker: str,
    sap_id: str,
    n_permutations: int = 1000,
    block_size_days: int = BLOCK_SIZE_DAYS,
    workers: int = 4,
    progress_every: int = 50,
) -> Dict[str, Any]:
    """Run the block-bootstrap permutation test for one ticker × SAP.

    Returns a dict with verdict + null distribution stats.
    """
    t0 = time.time()
    hypothesis = reconstruct_hypothesis_from_posterior(ticker, sap_id)
    if hypothesis is None:
        return {"ticker": ticker, "sap_id": sap_id, "status": "no_hypothesis",
                "verdict": "UNAVAILABLE", "wall_clock_s": 0.0}
    df = load_ohlc_dataframe(ticker)
    if df is None or len(df) < 300:
        return {"ticker": ticker, "sap_id": sap_id, "status": "no_bars",
                "verdict": "UNAVAILABLE", "wall_clock_s": 0.0}
    # Mirror live dispatcher's holdout = last 10% of bars (see hypothesis_runner.py L2020)
    holdout_idx = holdout_start_index_fraction(len(df), HOLDOUT_FRACTION)
    if holdout_idx >= len(df):
        return {"ticker": ticker, "sap_id": sap_id, "status": "no_holdout",
                "verdict": "UNAVAILABLE", "wall_clock_s": 0.0}

    # Live re-evaluation (sanity check vs reported HoldSR)
    live_full_sr, live_hold_sr, live_trades, live_wr = evaluate_sap_on_bars(
        hypothesis, df, holdout_idx
    )

    # Permutation runs
    base_seed = _rng_seed_for(ticker)
    seeds = [base_seed + i for i in range(n_permutations)]

    holdout_srs: List[float] = []
    full_srs: List[float] = []
    n_trades_list: List[int] = []
    wr_list: List[float] = []

    # Decide parallelism: each ticker × n_perms via Pool. We hold df + hypothesis fixed.
    # Pass df as a pickled object; worker re-builds bars.
    work_args = [
        (ticker, hypothesis, df, holdout_idx, block_size_days, s) for s in seeds
    ]

    if workers > 1:
        # Use multiprocessing
        with Pool(processes=workers) as pool:
            for i, res in enumerate(pool.imap_unordered(_single_perm_worker, work_args,
                                                       chunksize=10)):
                holdout_srs.append(res["holdout_sharpe"])
                full_srs.append(res["full_sharpe"])
                n_trades_list.append(res["n_trades"])
                wr_list.append(res["win_rate"])
                if (i + 1) % progress_every == 0:
                    print(f"  [{ticker}/{sap_id}] {i+1}/{n_permutations} "
                          f"perms done (live_hold={live_hold_sr:.3f})", flush=True)
    else:
        for i, args in enumerate(work_args):
            res = _single_perm_worker(args)
            holdout_srs.append(res["holdout_sharpe"])
            full_srs.append(res["full_sharpe"])
            n_trades_list.append(res["n_trades"])
            wr_list.append(res["win_rate"])
            if (i + 1) % progress_every == 0:
                print(f"  [{ticker}/{sap_id}] {i+1}/{n_permutations} "
                      f"perms done (live_hold={live_hold_sr:.3f})", flush=True)

    holdout_arr = np.asarray(holdout_srs, dtype=np.float64)
    valid = ~np.isnan(holdout_arr)
    n_valid = int(valid.sum())
    holdout_valid = holdout_arr[valid]

    # Null distribution stats
    n_above_1p5 = int((holdout_valid > HOLDSR_THRESHOLD).sum())
    n_above_live = int((holdout_valid > live_hold_sr).sum()) if not np.isnan(live_hold_sr) else None
    survival_rate = n_above_1p5 / n_valid if n_valid > 0 else float("nan")
    p_value_live = (n_above_live / n_valid) if (n_above_live is not None and n_valid > 0) else float("nan")
    perm_holdsr_p95 = float(np.percentile(holdout_valid, 95)) if n_valid > 0 else float("nan")
    perm_holdsr_p99 = float(np.percentile(holdout_valid, 99)) if n_valid > 0 else float("nan")
    perm_holdsr_max = float(np.max(holdout_valid)) if n_valid > 0 else float("nan")
    perm_holdsr_mean = float(np.mean(holdout_valid)) if n_valid > 0 else float("nan")
    perm_holdsr_std = float(np.std(holdout_valid, ddof=1)) if n_valid > 1 else float("nan")

    if survival_rate < 0.05:
        verdict = "REAL"
    elif survival_rate < 0.20:
        verdict = "SUSPECT"
    else:
        verdict = "ARTIFACT"

    elapsed = time.time() - t0
    return {
        "ticker": ticker,
        "sap_id": sap_id,
        "status": "ok",
        "live_holdout_sharpe": live_hold_sr,
        "live_full_sharpe": live_full_sr,
        "live_n_trades": live_trades,
        "live_win_rate": live_wr,
        "n_permutations": n_permutations,
        "n_valid_permutations": n_valid,
        "block_size_days": block_size_days,
        "holdout_cutoff": f"last {int(HOLDOUT_FRACTION*100)}% of bars",
        "holdout_idx_used": holdout_idx,
        "total_bars": len(df),
        "perm_holdsr_mean": perm_holdsr_mean,
        "perm_holdsr_std": perm_holdsr_std,
        "perm_holdsr_p95": perm_holdsr_p95,
        "perm_holdsr_p99": perm_holdsr_p99,
        "perm_holdsr_max": perm_holdsr_max,
        "n_perms_above_1p5": n_above_1p5,
        "n_perms_above_live": n_above_live,
        "survival_rate": survival_rate,
        "p_value_live": p_value_live,
        "verdict": verdict,
        "wall_clock_s": elapsed,
        # Sample of null distribution for histogramming
        "_null_distribution_sample": holdout_valid.tolist()[:200],
        "_full_null_distribution": holdout_valid.tolist(),
    }


# ============================================================================
# Reporting
# ============================================================================

def _text_histogram(arr: np.ndarray, bins: int = 20, width: int = 40) -> str:
    if arr.size == 0:
        return "(no data)"
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return "(all NaN)"
    counts, edges = np.histogram(arr, bins=bins)
    max_count = max(counts) if counts.size else 0
    lines = []
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        bar = "█" * int(round((c / max_count) * width)) if max_count else ""
        lines.append(f"  [{lo:+6.2f}, {hi:+6.2f}) {c:>5}  {bar}")
    return "\n".join(lines)


def write_csv(results: List[Dict[str, Any]], path: Path) -> None:
    cols = [
        "ticker", "sap_id", "status", "verdict",
        "live_holdout_sharpe", "live_full_sharpe", "live_n_trades", "live_win_rate",
        "n_permutations", "n_valid_permutations", "block_size_days", "holdout_cutoff",
        "perm_holdsr_mean", "perm_holdsr_std", "perm_holdsr_p95",
        "perm_holdsr_p99", "perm_holdsr_max",
        "n_perms_above_1p5", "n_perms_above_live",
        "survival_rate", "p_value_live", "wall_clock_s",
    ]
    rows = []
    for r in results:
        rows.append({c: r.get(c) for c in cols})
    df = pd.DataFrame(rows, columns=cols)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_markdown(results: List[Dict[str, Any]], path: Path,
                   meta: Dict[str, Any]) -> None:
    real = sum(1 for r in results if r.get("verdict") == "REAL")
    suspect = sum(1 for r in results if r.get("verdict") == "SUSPECT")
    artifact = sum(1 for r in results if r.get("verdict") == "ARTIFACT")
    unavail = sum(1 for r in results if r.get("verdict") == "UNAVAILABLE")
    n_total = len(results)

    if artifact >= n_total / 2:
        overall = "ARTIFACT"
        overall_text = (
            f"**{artifact}/{n_total} survivors are ARTIFACT.** The 24-champion claim "
            "is largely selection noise. Retire OHLCV-only program per the council kill "
            "criterion and pivot to alt-data overlays / multi-timeframe."
        )
    elif real >= n_total / 2:
        overall = "REAL"
        overall_text = (
            f"**{real}/{n_total} survivors test as REAL.** The live champions look like "
            "edge, not selection artifact. Proceed to alt-data overlay + cross-asset "
            "robustness checks per council R5."
        )
    else:
        overall = "MIXED"
        overall_text = (
            f"**Mixed:** {real} REAL, {suspect} SUSPECT, {artifact} ARTIFACT. "
            "Drop the ARTIFACT champions, scrutinize SUSPECT with a 2x-larger permutation "
            "test, advance REAL to the next gate."
        )

    lines = []
    lines.append(f"# Permutation Test — Block-Bootstrap Null on Daily-Dispatch Survivors")
    lines.append("")
    lines.append(f"_Generated: {datetime.now(timezone.utc).isoformat()}_")
    lines.append(f"_Block bootstrap: {meta['block_size_days']}-day blocks, "
                 f"{meta['n_permutations']} permutations per ticker_")
    lines.append(f"_Holdout cutoff: last {int(HOLDOUT_FRACTION*100)}% of bars "
                 f"(mirrors live `hypothesis_runner.run_hypothesis_for_ticker` cut)_")
    lines.append("")
    lines.append("## Overall Verdict")
    lines.append("")
    lines.append(f"**{overall}** — {overall_text}")
    lines.append("")
    lines.append("## Verdict table")
    lines.append("")
    lines.append("| Ticker | SAP | Live HoldSR | Perm Mean | Perm P95 | Perm Max | "
                 "Survival Rate | Verdict |")
    lines.append("|--------|-----|------------:|----------:|---------:|---------:|"
                 "--------------:|--------:|")
    # Sort by survival_rate ascending (most REAL first)
    sorted_results = sorted(
        [r for r in results if r.get("status") == "ok"],
        key=lambda r: r.get("survival_rate", 1.0),
    )
    for r in sorted_results:
        lines.append(
            f"| {r['ticker']} | {r['sap_id']} | "
            f"{r.get('live_holdout_sharpe', float('nan')):.3f} | "
            f"{r.get('perm_holdsr_mean', float('nan')):.3f} | "
            f"{r.get('perm_holdsr_p95', float('nan')):.3f} | "
            f"{r.get('perm_holdsr_max', float('nan')):.3f} | "
            f"{r.get('survival_rate', float('nan')):.1%} | "
            f"**{r.get('verdict')}** |"
        )
    # Unavailable tickers
    unavail_results = [r for r in results if r.get("status") != "ok"]
    if unavail_results:
        lines.append("")
        lines.append("### Unavailable")
        for r in unavail_results:
            lines.append(f"- {r['ticker']} / {r['sap_id']}: {r.get('status')}")

    # Null distribution histograms per ticker
    lines.append("")
    lines.append("## Null distributions (holdout Sharpe under permutation)")
    lines.append("")
    for r in sorted_results:
        lines.append(f"### {r['ticker']} ({r['sap_id']}) — verdict {r['verdict']}")
        lines.append("")
        full = np.asarray(r.get("_full_null_distribution", []), dtype=np.float64)
        lines.append("```")
        lines.append(_text_histogram(full, bins=20, width=40))
        lines.append("```")
        live_sr = r.get("live_holdout_sharpe", float("nan"))
        survival = r.get("survival_rate", float("nan"))
        lines.append(f"- Live HoldSR = **{live_sr:.3f}**, "
                     f"perm P95 = {r.get('perm_holdsr_p95', float('nan')):.3f}, "
                     f"perm max = {r.get('perm_holdsr_max', float('nan')):.3f}")
        lines.append(f"- {r.get('n_perms_above_1p5', 0)}/{r.get('n_valid_permutations', 0)} "
                     f"permutations cleared 1.5 → survival rate = {survival:.1%}")
        lines.append(f"- p-value vs live HoldSR: {r.get('p_value_live', float('nan')):.3f}")
        lines.append(f"- Wall-clock: {r.get('wall_clock_s', 0.0):.1f}s")
        lines.append("")

    # Methodology
    lines.append("## Methodology")
    lines.append("")
    lines.append(
        f"- Each ticker's winning SAP (the `current_best` from `data/posteriors/<T>.json`) "
        f"is reconstructed from its `parent_seed_id` + `perturb_params` via the same "
        f"`championship_search._format_template` path the live dispatcher uses."
    )
    lines.append(
        f"- For each permutation: block-bootstrap the close-to-close LOG RETURNS in "
        f"{meta['block_size_days']}-day contiguous blocks (with replacement), reconstruct "
        f"close via cumulative product, then attach OHL via the source bar's intra-bar "
        f"ratio. Volume is independently block-bootstrapped (different autocorrelation "
        f"structure). The calendar (date column, holdout cutoff) is preserved."
    )
    lines.append(
        f"- HoldSR is computed on the LAST {int(HOLDOUT_FRACTION*100)}% of bars of the "
        f"PERMUTED series — this exactly mirrors `hypothesis_runner.run_hypothesis_for_ticker` "
        f"(`cut = int(0.9 * n_obs)`), so the live HoldSR is the reference distribution we're "
        f"testing against."
    )
    lines.append(
        f"- The block size (20) preserves autocorrelation up to lag 19; longer-horizon "
        f"dependence (e.g. quarterly drift) is broken by design — that's the point of "
        f"the null."
    )
    lines.append(
        "- Verdict: REAL if survival_rate < 5%, SUSPECT if 5-20%, ARTIFACT if > 20%."
    )
    lines.append("")
    lines.append("### Caveats")
    lines.append("")
    lines.append(
        "- Intra-bar geometry uses the SOURCE bar's HLOC/close ratios applied to the "
        "rebuilt close. This is a documented limitation — true intra-bar permutation would "
        "require a joint bootstrap. For ATR/Donchian/VWAP (used by all 5 Mission 12 seeds), "
        "the consequence is acceptable."
    )
    lines.append(
        "- Volume is independently bootstrapped, breaking the price-volume correlation. "
        "This is intentional and part of the null: a signal that depends on real price-"
        "volume confluence (Volume > 1.2 × SMA(V,20) as confirmation) becomes a coin-flip "
        "under the null."
    )
    lines.append(
        "- Alt-data tokens (Form4, CongressBuy, etc.) are not present in the realized role "
        "strings of any current Mission 12 seed (GOV_AWARE's alt_data_overlay is metadata-"
        "only). If a future seed embeds alt-data tokens, those would resolve against the "
        "REAL alt-data calendar even under permutation — which is correct (the null only "
        "permutes OHLCV; alt-data is external)."
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roll-up", type=str,
                        default=str(DRIVE_REPORTS / "championship_roll_up_2026-05-29.md"))
    parser.add_argument("--n-perms", type=int, default=500)
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE_DAYS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ticker", type=str, default=None,
                        help="Run only this ticker (for testing)")
    parser.add_argument("--out-dir-local", type=str,
                        default=str(RESULTS_LOCAL))
    parser.add_argument("--out-csv", type=str,
                        default=str(DRIVE_REPORTS / "permutation_test_2026-05-29.csv"))
    parser.add_argument("--out-md", type=str,
                        default=str(DRIVE_REPORTS / "permutation_test_2026-05-29.md"))
    parser.add_argument("--status-path", type=str, default="/tmp/permutation_test_status.md")
    args = parser.parse_args()

    print(f"[permutation_test] starting at {datetime.now(timezone.utc).isoformat()}", flush=True)
    print(f"  roll_up = {args.roll_up}", flush=True)
    print(f"  n_perms = {args.n_perms}, block_size = {args.block_size}, "
          f"workers = {args.workers}", flush=True)

    # 1. Identify survivors
    survivors = load_survivors_from_dispatch(Path(args.roll_up))
    if args.ticker:
        survivors = [s for s in survivors if s[0] == args.ticker.upper()]
        if not survivors:
            print(f"  Ticker {args.ticker} not in survivors list", flush=True)
            return 1
    print(f"[permutation_test] {len(survivors)} survivors with HoldSR > {HOLDSR_THRESHOLD}", flush=True)
    for ticker, sap_id, holdsr in survivors:
        print(f"  - {ticker}: {sap_id} (live HoldSR {holdsr:.3f})", flush=True)

    # 2. Setup output dirs
    utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    local_out_dir = Path(args.out_dir_local) / utc
    local_out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[permutation_test] per-ticker raw results -> {local_out_dir}", flush=True)

    # 3. Run sequentially per ticker, parallel per permutation within ticker
    all_results: List[Dict[str, Any]] = []
    overall_t0 = time.time()
    for i, (ticker, sap_id, live_holdsr) in enumerate(survivors):
        print(f"\n[{i+1}/{len(survivors)}] === {ticker} ({sap_id}) ===", flush=True)
        try:
            res = run_permutation_test(
                ticker=ticker, sap_id=sap_id,
                n_permutations=args.n_perms,
                block_size_days=args.block_size,
                workers=args.workers,
            )
        except Exception as e:
            tb = traceback.format_exc()
            print(f"  ERROR: {e}\n{tb}", flush=True)
            res = {"ticker": ticker, "sap_id": sap_id, "status": "error",
                   "verdict": "UNAVAILABLE", "error": str(e), "traceback": tb,
                   "wall_clock_s": 0.0}
        all_results.append(res)

        # Persist per-ticker JSON
        ticker_out = local_out_dir / f"{ticker}_{sap_id}.json"
        try:
            ticker_out.write_text(json.dumps(res, indent=2, default=str))
        except OSError as e:
            print(f"  WARN: couldn't write {ticker_out}: {e}", flush=True)

        # Update status
        elapsed_total = time.time() - overall_t0
        status_lines = [
            f"# Permutation test progress\n",
            f"_Updated: {datetime.now(timezone.utc).isoformat()}_\n",
            f"- {i+1}/{len(survivors)} tickers done",
            f"- Elapsed: {elapsed_total:.0f}s",
            f"- Last ticker: {ticker} -> verdict {res.get('verdict')} "
            f"(survival_rate = {res.get('survival_rate', float('nan'))})",
            "",
            "## Verdicts so far",
        ]
        for r in all_results:
            sr = r.get("survival_rate", float("nan"))
            sr_str = f"{sr:.1%}" if isinstance(sr, (int, float)) and not np.isnan(sr) else "n/a"
            status_lines.append(
                f"- {r['ticker']} ({r['sap_id']}): {r.get('verdict')} (survival = {sr_str})"
            )
        try:
            Path(args.status_path).write_text("\n".join(status_lines))
        except OSError:
            pass

        print(f"  Ticker {ticker} verdict: {res.get('verdict')} "
              f"(survival_rate = {res.get('survival_rate', float('nan'))}, "
              f"wall-clock = {res.get('wall_clock_s', 0):.1f}s)", flush=True)

    # 4. Write CSV + markdown
    csv_path = Path(args.out_csv)
    md_path = Path(args.out_md)
    write_csv(all_results, csv_path)
    write_markdown(all_results, md_path, meta={
        "block_size_days": args.block_size,
        "n_permutations": args.n_perms,
    })
    print(f"\n[permutation_test] CSV   -> {csv_path}", flush=True)
    print(f"[permutation_test] MD    -> {md_path}", flush=True)
    print(f"[permutation_test] Local -> {local_out_dir}", flush=True)
    print(f"[permutation_test] DONE. Total wall-clock = "
          f"{time.time() - overall_t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
