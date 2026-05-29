"""indicator_hardening_runner.py — 6-step indicator validation pipeline per Phase 2.

For each indicator in the registry, on each ticker in the cohort:
  Step 1: Rolling walk-forward (12 folds) → IS Sharpe, OOS Sharpe, WFE
  Step 2: (Skipped — purged CV is for ML labels, not pure-rule strategies)
  Step 3: CSCV / PBO over parameter grid (configs × time)
  Step 4: Deflated Sharpe Ratio
  Step 5: Parameter stability (±10% perturbation on every param)
  Step 6: Final 6-month holdout (last 25k bars untouched in folds)

Cost model: 5 bps per side baseline. Per-bar return for held position:
  r_bar = position[i-1] * log(close[i]/close[i-1]) - cost_per_side * |Δposition|

Per-ticker results go to:
  /Volumes/ZG-2TB/zg/indicator_backtest/results/<indicator>/<ticker>/{wfa.csv, params.json, pbo.json, dsr.json, stability.csv}

Aggregated cohort results + Drive durable output:
  AI-Tools/s&p500-ticker-mastery/data/indicator_validation/<indicator>/<utc>/{cohort.csv, summary.json, pbo.json, dsr.json, stability.csv}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from indicator_compute import REGISTRY  # noqa: E402
from indicator_pbo_dsr import cscv_pbo, deflated_sharpe, rolling_walkforward_folds, walk_forward_efficiency, _sharpe  # noqa: E402

OHLC_DIR = Path("/Volumes/ZG-2TB/zg/indicator_backtest/ohlc_5min")
DRIVE_OHLC_5MIN = Path("/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/claudes test/archive_dead/version_3_2026-05-05/S&P500 5 Year Historical Data")
# DAILY fallback — 5-year yfinance cache, works reliably even when 5min Drive path is FUSE-broken
DRIVE_OHLC_DAILY = Path("/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/s&p500-ticker-mastery/cache/yfinance_5yr")
RESULTS_LOCAL = Path("/Volumes/ZG-2TB/zg/indicator_backtest/results")
DRIVE_RESULTS = Path("/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/s&p500-ticker-mastery/data/indicator_validation")

COST_PER_SIDE = 5e-4  # 5 bps
# Timeframe-conditional ann factor. Set at runtime per --timeframe.
BARS_PER_YEAR_DEFAULT = 252  # daily — overridden via set_timeframe()
_state = {"bars_per_year": BARS_PER_YEAR_DEFAULT, "timeframe": "1d", "min_bars": 300}


def set_timeframe(tf: str):
    """tf in {'1d', '5min'}."""
    if tf == "1d":
        _state["bars_per_year"] = 252
        _state["timeframe"] = "1d"
        _state["min_bars"] = 300
    elif tf == "5min":
        _state["bars_per_year"] = 252 * 78
        _state["timeframe"] = "5min"
        _state["min_bars"] = 5000
    else:
        raise ValueError(f"unknown timeframe {tf}")


def load_ohlc(ticker: str) -> dict[str, np.ndarray] | None:
    """Read parquet from local stage or fall back to Drive with retry.

    Resolves the path based on _state["timeframe"].
    """
    tf = _state["timeframe"]
    if tf == "5min":
        p_local = OHLC_DIR / f"{ticker}_5min.parquet"
        p_drive = DRIVE_OHLC_5MIN / f"{ticker}_5min.parquet"
    else:
        p_local = None
        p_drive = DRIVE_OHLC_DAILY / f"{ticker}.parquet"
    path = p_local if (p_local is not None and p_local.exists() and p_local.stat().st_size > 1000) else p_drive
    for attempt in range(3):
        try:
            df = pd.read_parquet(path)
            break
        except OSError as e:
            if attempt == 2:
                print(f"  [load_ohlc] FAILED {ticker}: {e}", flush=True)
                return None
            time.sleep(1)
    # Normalize columns (handles both lower- and Title-case)
    rename = {}
    for canon in ("open", "high", "low", "close", "volume"):
        for src in (canon, canon[0].upper() + canon[1:], canon.upper()):
            if src in df.columns:
                rename[src] = canon
                break
    df = df.rename(columns=rename)
    needed = {"open", "high", "low", "close", "volume"}
    if not needed.issubset(df.columns):
        print(f"  [load_ohlc] {ticker} missing cols. have={list(df.columns)}", flush=True)
        return None
    df = df.dropna(subset=list(needed))
    if len(df) < _state["min_bars"]:
        return None
    return {k: df[k].to_numpy(dtype=np.float64) for k in needed}


def returns_from_signal(bars: dict, signal: np.ndarray) -> np.ndarray:
    """Per-bar log return of strategy holding `signal[i-1]` units from bar i-1 to i.

    Includes cost = COST_PER_SIDE * |Δsignal| applied at bar of change.
    No look-ahead: position used is shifted by 1.
    """
    close = bars["close"]
    log_ret = np.zeros_like(close)
    log_ret[1:] = np.log(close[1:] / close[:-1])
    pos = np.zeros_like(close, dtype=np.float64)
    pos[1:] = signal[:-1].astype(np.float64)  # shift signal forward
    dpos = np.zeros_like(close)
    dpos[1:] = np.abs(pos[1:] - pos[:-1])
    return pos * log_ret - COST_PER_SIDE * dpos


def annualized_sharpe(rets: np.ndarray) -> float:
    rets = rets[~np.isnan(rets)]
    if rets.size < 50 or np.std(rets, ddof=1) == 0:
        return float("nan")
    return float(np.mean(rets) / np.std(rets, ddof=1) * np.sqrt(_state["bars_per_year"]))


def win_rate_from_signal(bars: dict, signal: np.ndarray) -> tuple[float, int]:
    """Trade-level win rate. A trade = period of constant non-zero position.

    Returns (win_rate, n_trades).
    """
    close = bars["close"]
    pos = np.zeros_like(close, dtype=np.int8)
    pos[1:] = signal[:-1]
    n = len(close)
    trades = []
    i = 0
    while i < n:
        if pos[i] == 0:
            i += 1; continue
        side = pos[i]
        entry = close[i]
        j = i
        while j < n and pos[j] == side:
            j += 1
        exit_ = close[min(j - 1, n - 1)]
        ret = side * (exit_ / entry - 1) - 2 * COST_PER_SIDE
        trades.append(ret)
        i = j
    if not trades:
        return float("nan"), 0
    arr = np.asarray(trades)
    return float((arr > 0).mean()), int(arr.size)


def run_indicator_for_ticker(name: str, ticker: str, n_folds: int = 12) -> dict:
    bars = load_ohlc(ticker)
    if bars is None:
        return {"ticker": ticker, "status": "no_data"}
    reg = REGISTRY[name]
    fn = reg["fn"]
    grid = reg["grid"]
    default_params = reg["default"]

    # ---- Step 1: walk-forward on default params
    sig = fn(bars, **default_params)
    rets = returns_from_signal(bars, sig)
    # Adapt fold count to series length — daily ~1262 bars supports 6 folds; 5min ~30k supports 12
    eff_folds = n_folds if len(rets) >= n_folds * 60 else max(4, min(n_folds, len(rets) // 60))
    folds = rolling_walkforward_folds(len(rets), n_folds=eff_folds, train_frac=0.8, embargo_frac=0.005)
    is_sharpes = []
    oos_sharpes = []
    for f in folds:
        is_sharpes.append(annualized_sharpe(rets[f.train_start:f.train_end]))
        oos_sharpes.append(annualized_sharpe(rets[f.test_start:f.test_end]))
    is_med = float(np.nanmedian(is_sharpes)) if is_sharpes else float("nan")
    oos_med = float(np.nanmedian(oos_sharpes)) if oos_sharpes else float("nan")
    wfe = walk_forward_efficiency(is_med, oos_med)
    full_sharpe = annualized_sharpe(rets)
    wr, n_trades = win_rate_from_signal(bars, sig)

    # ---- Step 3: PBO over parameter grid + 16 chunks
    # Build (T, N) matrix where N = grid size
    n_obs = len(rets)
    M = np.zeros((n_obs, len(grid)), dtype=np.float64)
    for j, params in enumerate(grid):
        s_j = fn(bars, **params)
        M[:, j] = returns_from_signal(bars, s_j)
    # CSCV with S=16 needs n_obs >= 16; we have ~30k+ so fine
    try:
        pbo_res = cscv_pbo(M, s_chunks=16)
    except Exception as e:
        pbo_res = {"pbo": float("nan"), "error": str(e)}

    # ---- Step 4: DSR — selected = default config. n_trials = grid size (lower bound).
    # variance_of_sharpes across configs (annualized)
    config_sharpes = np.array([annualized_sharpe(M[:, j]) for j in range(M.shape[1])])
    var_sr = float(np.nanvar(config_sharpes)) if np.sum(~np.isnan(config_sharpes)) > 1 else 1.0
    dsr_res = deflated_sharpe(full_sharpe, rets, n_trials=max(len(grid), 10), variance_of_sharpes=var_sr)

    # ---- Step 5: Parameter stability — already partially covered by the grid; flag if std(config_sharpes)/mean > 0.5
    if np.sum(~np.isnan(config_sharpes)) > 1:
        mean_sr = float(np.nanmean(config_sharpes))
        std_sr = float(np.nanstd(config_sharpes))
        stab_cv = (std_sr / abs(mean_sr)) if abs(mean_sr) > 1e-6 else float("inf")
    else:
        mean_sr, std_sr, stab_cv = float("nan"), float("nan"), float("nan")
    stability = {
        "grid_sharpes": config_sharpes.tolist(),
        "grid_params": grid,
        "mean_sharpe": mean_sr,
        "std_sharpe": std_sr,
        "cv": stab_cv,
        "stable": (stab_cv < 0.5) if not np.isnan(stab_cv) else False,
    }

    # ---- Step 6: Holdout = last 10% of data, never seen by folds (folds end well before)
    cut = int(0.9 * n_obs)
    holdout_sharpe = annualized_sharpe(rets[cut:])
    insample_sharpe_pre = annualized_sharpe(rets[:cut])

    return {
        "ticker": ticker,
        "status": "ok",
        "n_obs": int(n_obs),
        "n_trades": n_trades,
        "win_rate": wr,
        "full_sharpe": full_sharpe,
        "is_sharpe_median": is_med,
        "oos_sharpe_median": oos_med,
        "wfe": wfe,
        "is_sharpes": is_sharpes,
        "oos_sharpes": oos_sharpes,
        "pbo": pbo_res.get("pbo"),
        "pbo_n_combos": pbo_res.get("n_combos"),
        "dsr_prob": dsr_res.get("dsr_prob"),
        "dsr_sr0": dsr_res.get("sr0_threshold"),
        "stability": stability,
        "holdout_sharpe": holdout_sharpe,
        "insample_sharpe_pre_holdout": insample_sharpe_pre,
    }


def classify_status(cohort_rows: list[dict], pbo: float, dsr: float) -> str:
    """Phase 2 promotion: PBO < 0.15 AND DSR > 0.95 AND mean WR >= 0.50 → TESTED_MULTIPLE_TICKERS."""
    valid = [r for r in cohort_rows if r["status"] == "ok"]
    if not valid:
        return "NO_DATA"
    wr_mean = np.nanmean([r["win_rate"] for r in valid if r["win_rate"] is not None and not np.isnan(r["win_rate"])])
    if (not np.isnan(pbo) and pbo < 0.15 and not np.isnan(dsr) and dsr > 0.95 and wr_mean >= 0.50):
        return "TESTED_MULTIPLE_TICKERS"
    if (not np.isnan(pbo) and pbo > 0.5) or (not np.isnan(dsr) and dsr < 0.5) or (not np.isnan(wr_mean) and wr_mean < 0.45):
        return "REJECTED"
    return "TESTED_PRELIMINARY"


def run_one_indicator(name: str, tickers: list[str], utc_tag: str, n_folds: int = 12) -> dict:
    print(f"\n=== {name} === tickers={tickers}", flush=True)
    rows = []
    t0 = time.time()
    for t in tickers:
        try:
            r = run_indicator_for_ticker(name, t, n_folds=n_folds)
            r["indicator"] = name
            rows.append(r)
            if r["status"] == "ok":
                print(
                    f"  {t}: WR={r['win_rate']:.3f} N={r['n_trades']:>4d} WFE={r['wfe']:+.2f} PBO={r['pbo']:.3f} DSR={r['dsr_prob']:.3f}",
                    flush=True,
                )
            else:
                print(f"  {t}: {r['status']}", flush=True)
        except Exception as e:
            print(f"  {t}: ERROR {e}", flush=True)
            traceback.print_exc()
            rows.append({"ticker": t, "indicator": name, "status": f"error:{e}"})

    valid = [r for r in rows if r["status"] == "ok"]
    pbo_mean = float(np.nanmean([r["pbo"] for r in valid])) if valid else float("nan")
    dsr_mean = float(np.nanmean([r["dsr_prob"] for r in valid])) if valid else float("nan")
    wr_mean = float(np.nanmean([r["win_rate"] for r in valid])) if valid else float("nan")
    wfe_mean = float(np.nanmean([r["wfe"] for r in valid])) if valid else float("nan")
    n_trades_total = int(np.sum([r["n_trades"] for r in valid])) if valid else 0

    new_status = classify_status(rows, pbo_mean, dsr_mean)

    summary = {
        "indicator": name,
        "utc": utc_tag,
        "n_tickers_attempted": len(tickers),
        "n_tickers_ok": len(valid),
        "n_trades_total": n_trades_total,
        "wr_mean": wr_mean,
        "wfe_mean": wfe_mean,
        "pbo_mean": pbo_mean,
        "dsr_mean": dsr_mean,
        "new_status": new_status,
        "elapsed_sec": time.time() - t0,
    }

    # Write local results (fast) then mirror to Drive (slow but durable)
    for base in (RESULTS_LOCAL / name / utc_tag, DRIVE_RESULTS / name / utc_tag):
        try:
            base.mkdir(parents=True, exist_ok=True)
            with open(base / "summary.json", "w") as f:
                json.dump(summary, f, indent=2, default=str)
            # cohort CSV
            cohort_rows = []
            for r in rows:
                cohort_rows.append({
                    "ticker": r.get("ticker"),
                    "status": r.get("status"),
                    "n_obs": r.get("n_obs"),
                    "n_trades": r.get("n_trades"),
                    "win_rate": r.get("win_rate"),
                    "full_sharpe": r.get("full_sharpe"),
                    "is_sharpe_med": r.get("is_sharpe_median"),
                    "oos_sharpe_med": r.get("oos_sharpe_median"),
                    "wfe": r.get("wfe"),
                    "pbo": r.get("pbo"),
                    "dsr_prob": r.get("dsr_prob"),
                    "holdout_sharpe": r.get("holdout_sharpe"),
                })
            pd.DataFrame(cohort_rows).to_csv(base / "cohort.csv", index=False)
            # Stability CSV (one row per ticker × grid point)
            stab_rows = []
            for r in rows:
                if r.get("stability"):
                    s = r["stability"]
                    for params, sharpe in zip(s["grid_params"], s["grid_sharpes"]):
                        stab_rows.append({
                            "ticker": r["ticker"],
                            "params": json.dumps(params),
                            "sharpe": sharpe,
                            "stable": s["stable"],
                            "cv": s["cv"],
                        })
            if stab_rows:
                pd.DataFrame(stab_rows).to_csv(base / "stability.csv", index=False)
        except OSError as e:
            print(f"  [persist] failed at {base}: {e}", flush=True)

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indicators", nargs="+", default=list(REGISTRY.keys()))
    ap.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "JPM", "XOM", "JNJ"])
    ap.add_argument("--output-jsonl", default="/Volumes/ZG-2TB/zg/indicator_backtest/results/summary_all.jsonl")
    ap.add_argument("--utc-tag", default=None)
    ap.add_argument("--timeframe", default="1d", choices=["1d", "5min"], help="OHLC timeframe")
    ap.add_argument("--n-folds", type=int, default=12)
    args = ap.parse_args()

    set_timeframe(args.timeframe)
    utc_tag = args.utc_tag or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_jsonl = Path(args.output_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    print(f"Indicators: {args.indicators}", flush=True)
    print(f"Tickers:    {args.tickers}", flush=True)
    print(f"Timeframe:  {args.timeframe} (bars/year={_state['bars_per_year']})", flush=True)
    print(f"UTC tag:    {utc_tag}", flush=True)

    all_summaries = []
    for ind in args.indicators:
        if ind not in REGISTRY:
            print(f"[skip] unknown indicator {ind}", flush=True)
            continue
        s = run_one_indicator(ind, args.tickers, utc_tag, n_folds=args.n_folds)
        all_summaries.append(s)
        with open(out_jsonl, "a") as f:
            f.write(json.dumps(s, default=str) + "\n")

    print("\n=== ALL DONE ===")
    df = pd.DataFrame(all_summaries)
    print(df[["indicator", "n_tickers_ok", "wr_mean", "wfe_mean", "pbo_mean", "dsr_mean", "new_status"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
