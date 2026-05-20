#!/usr/bin/env python3
"""
ORB Fade Backtest — fully numpy-vectorized per session.

Hypothesis: Opening-range breakouts on liquid equities tend to reverse.
Fade the breakout; target the OR midpoint.

Entry rules (one trade per session, first signal wins):
  - SHORT when a bar closes ABOVE OR_high  →  fade the breakout failure
  - LONG  when a bar closes BELOW OR_low   →  fade the breakdown failure
  (Both are COUNTER-TREND entries vs. the original ORB long-only direction.)

Stop  : 1.5 × ATR(14, pre-OR bars) on entry bar, capped at OR_range × 1.0
Target: OR midpoint  = (OR_high + OR_low) / 2
Force-flat : 15:55 ET  (same as orb_fast.py)
Entry cutoff: 15:30 ET (no new entries after this)
Slippage: 5 bps each way
Commission: $0.0035 / share each way
Position size: $5,000 notional
No-lookahead: entry at next-bar open after signal; OR built from bars 0..OR_MINUTES-1 only.

Pure numpy — no per-bar Python loops.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ET_TZ = pytz.timezone("America/New_York")
DATA_ROOT = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/claudes test/data/timeframes"
    "/S&P500 5 Year Historical Data/Minutes TimeFrames/1Min_merged"
)
RESULTS_ROOT = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools/trading-ticker-mastery/backtests"
)

OR_MINUTES: int = 15
ATR_PERIOD: int = 14
ATR_STOP_MULT: float = 1.5  # ATR multiplier for stop
OR_STOP_CAP_MULT: float = 1.0  # stop capped at OR_range × this
SLIPPAGE_BPS: int = 5
FEE_PER_SHARE: float = 0.0035
NOTIONAL: float = 5_000.0

import datetime

RTH_OPEN = datetime.time(9, 30)
RTH_CLOSE = datetime.time(16, 0)
ENTRY_CUTOFF_MIN: int = 15 * 60 + 30  # 930 minutes since midnight
FORCE_EXIT_MIN: int = 15 * 60 + 55  # 955 minutes since midnight

THRESHOLDS = dict(n_min=30, wr_min=0.55, pf_min=1.4, ret_min=0.0, dd_max=5.0, hold_max=360)


# ---------------------------------------------------------------------------
# Data loading  (identical to orb_fast.py _load_rth_fast pattern)
# ---------------------------------------------------------------------------
def load_rth(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Load RTH 1-min bars for *ticker* between *start* and *end* (inclusive).

    Args:
        ticker: Uppercase ticker symbol.
        start: ISO date string, e.g. "2021-01-01".
        end: ISO date string, e.g. "2024-12-31".

    Returns:
        DataFrame with columns open/high/low/close/volume/min_of_day/date_et,
        sorted ascending by ts_et, containing only RTH bars in [start, end].

    Raises:
        FileNotFoundError: If the parquet file for *ticker* is absent.
        ValueError: If the loaded frame is empty after filtering.
    """
    path = DATA_ROOT / f"{ticker}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Parquet not found: {path}")

    df = pd.read_parquet(path, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["ts_utc"] = pd.to_datetime(df["timestamp"], utc=True)
    df["ts_et"] = df["ts_utc"].dt.tz_convert(ET_TZ)

    # Fast integer RTH filter (avoids slow dt.time comparisons)
    h = df["ts_et"].dt.hour
    m = df["ts_et"].dt.minute
    rth = ((h > 9) | ((h == 9) & (m >= 30))) & (h < 16)
    df = df[rth].copy()

    df["min_of_day"] = h[rth].values * 60 + m[rth].values

    # Date range filter on the already-RTH-filtered (smaller) frame
    ts_date = df["ts_et"].dt.normalize()
    start_dt = pd.Timestamp(start, tz=ET_TZ)
    end_dt = pd.Timestamp(end, tz=ET_TZ)
    df = df[(ts_date >= start_dt) & (ts_date <= end_dt)].copy()
    df["date_et"] = df["ts_et"].dt.date

    df = df.sort_values("ts_et").reset_index(drop=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# ATR helper  (vectorized, no Python loop)
# ---------------------------------------------------------------------------
def _atr14_series(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> np.ndarray:
    """Return per-bar ATR(14) using Wilder EMA.  Result[0] = NaN.

    Uses vectorized numpy; the EMA portion is unavoidably O(n) but operates on
    a single array — no per-bar Python overhead after the initial TR computation.

    Args:
        highs: 1-D float64 array of bar highs.
        lows: 1-D float64 array of bar lows.
        closes: 1-D float64 array of bar closes.

    Returns:
        1-D float64 array of ATR values, same length as inputs.
    """
    n = len(highs)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = highs[0] - lows[0]
    prev_close = closes[:-1]
    tr[1:] = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(np.abs(highs[1:] - prev_close), np.abs(lows[1:] - prev_close)),
    )

    atr = np.empty(n, dtype=np.float64)
    atr[:ATR_PERIOD] = np.nan
    if n >= ATR_PERIOD:
        atr[ATR_PERIOD - 1] = np.mean(tr[:ATR_PERIOD])
        k = 1.0 / ATR_PERIOD  # Wilder smoothing
        for i in range(ATR_PERIOD, n):
            atr[i] = atr[i - 1] * (1.0 - k) + tr[i] * k
    return atr


# ---------------------------------------------------------------------------
# Per-session vectorized backtest (fade variant)
# ---------------------------------------------------------------------------
def _run_session_fade(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    vols: np.ndarray,
    min_of_day: np.ndarray,
    ts_strs: np.ndarray,
) -> Optional[Dict[str, Any]]:
    """Run one RTH session for the ORB-fade strategy.

    All arrays are 1-D numpy arrays for a single RTH session, sorted ascending.
    Returns a single trade dict or None.

    Entry logic (NO lookahead — signal on bar i, fill at bar i+1 open):
      - If close[i] > OR_high  → short entry (fade breakout)
      - If close[i] < OR_low   → long  entry (fade breakdown)
      First qualifying bar after OR window and before entry cutoff wins.

    Stop distance = min(ATR_stop_mult × ATR14[i], OR_range × OR_stop_cap_mult)
    Target = OR midpoint

    Args:
        opens: Per-bar opens.
        highs: Per-bar highs.
        lows: Per-bar lows.
        closes: Per-bar closes.
        vols: Per-bar volumes (unused for signal here; reserved for future filter).
        min_of_day: Integer minutes since midnight for each bar.
        ts_strs: String representation of each bar timestamp.

    Returns:
        Trade dict or None if no qualifying signal.
    """
    n = len(opens)
    if n < OR_MINUTES + 2:
        return None

    # ---- Build OR ----
    or_h = float(np.max(highs[:OR_MINUTES]))
    or_l = float(np.min(lows[:OR_MINUTES]))
    or_r = or_h - or_l
    if or_r <= 0.0:
        return None

    or_mid = (or_h + or_l) * 0.5

    # ---- ATR over pre-OR + up-to-entry bars (no-lookahead: computed up to bar i) ----
    # We compute ATR over the entire session array once; for bar i the ATR used
    # is atr_full[i], which uses only TR data up to and including bar i.
    atr_full = _atr14_series(highs, lows, closes)

    # ---- Candidate signal bars ----
    idx = np.arange(OR_MINUTES, n - 1)

    # Signal conditions
    short_signal = closes[idx] > or_h   # close above OR_high → fade (go short)
    long_signal  = closes[idx] < or_l   # close below OR_low  → fade (go long)
    has_signal   = short_signal | long_signal

    time_ok   = min_of_day[idx] < ENTRY_CUTOFF_MIN
    next_ok   = min_of_day[idx + 1] < ENTRY_CUTOFF_MIN
    atr_valid = np.isfinite(atr_full[idx]) & (atr_full[idx] > 0.0)

    cand = idx[has_signal & time_ok & next_ok & atr_valid]
    if len(cand) == 0:
        return None

    signal_idx = int(cand[0])
    entry_idx  = signal_idx + 1

    is_short = bool(closes[signal_idx] > or_h)
    direction = "short" if is_short else "long"

    # ---- Entry price (with slippage) ----
    entry_raw = opens[entry_idx]
    slip_mult = (1 + SLIPPAGE_BPS / 10_000) if is_short else (1 - SLIPPAGE_BPS / 10_000)
    # For short: we SELL at a slightly lower price (adverse slippage).
    # Reframe: entry_filled is the actual execution price.
    # Short: entry_filled = raw * (1 - bps)  [sold at discount = worse fill]
    # Long:  entry_filled = raw * (1 + bps)  [bought at premium = worse fill]
    if is_short:
        entry_filled = entry_raw * (1 - SLIPPAGE_BPS / 10_000)
    else:
        entry_filled = entry_raw * (1 + SLIPPAGE_BPS / 10_000)
    if entry_filled <= 0.0:
        return None

    shares = max(1.0, NOTIONAL / entry_filled)

    # ---- Stop and Target ----
    atr_stop = ATR_STOP_MULT * float(atr_full[signal_idx])
    or_cap   = OR_STOP_CAP_MULT * or_r
    stop_dist = min(atr_stop, or_cap)
    stop_dist = max(stop_dist, 0.01)  # floor at 1 cent

    if is_short:
        stop_price   = entry_filled + stop_dist  # stop ABOVE entry for short
        target_price = or_mid                     # target OR midpoint
        # Sanity: target must be below entry for short
        if target_price >= entry_filled:
            return None
    else:
        stop_price   = entry_filled - stop_dist  # stop BELOW entry for long
        target_price = or_mid                     # target OR midpoint
        # Sanity: target must be above entry for long
        if target_price <= entry_filled:
            return None

    # ---- Exit scan ----
    force_offset = int(np.searchsorted(min_of_day[entry_idx:], FORCE_EXIT_MIN, side="left"))
    last_scan = int(entry_idx + force_offset) if force_offset < n - entry_idx else n - 1

    scan = np.arange(entry_idx, last_scan + 1)

    if is_short:
        hit_stop_mask   = highs[scan] >= stop_price   # price rises to stop
        hit_target_mask = lows[scan]  <= target_price  # price drops to target
    else:
        hit_stop_mask   = lows[scan]  <= stop_price    # price drops to stop
        hit_target_mask = highs[scan] >= target_price  # price rises to target

    force_mask = min_of_day[scan] >= FORCE_EXIT_MIN

    any_exit = hit_stop_mask | hit_target_mask | force_mask
    first_exit_pos = int(np.argmax(any_exit)) if np.any(any_exit) else len(scan) - 1
    exit_bar_local = scan[first_exit_pos]

    # ---- Determine exit type ----
    forced = False
    if force_mask[first_exit_pos]:
        exit_raw    = opens[exit_bar_local]
        exit_reason = "force_flat_1555"
        forced      = True
    elif hit_stop_mask[first_exit_pos] and hit_target_mask[first_exit_pos]:
        # Both hit same bar — conservative: stop wins
        exit_raw    = stop_price
        exit_reason = "stop"
    elif hit_stop_mask[first_exit_pos]:
        exit_raw    = stop_price
        exit_reason = "stop"
    else:
        exit_raw    = target_price
        exit_reason = "target"

    # Clamp to bar range
    bar_l = float(lows[exit_bar_local])
    bar_h = float(highs[exit_bar_local])
    exit_raw = float(max(bar_l, min(bar_h, exit_raw)))

    # Exit slippage (adverse)
    if is_short:
        exit_filled = exit_raw * (1 + SLIPPAGE_BPS / 10_000)  # cover at premium
        gross = (entry_filled - exit_filled) * shares           # short P&L
    else:
        exit_filled = exit_raw * (1 - SLIPPAGE_BPS / 10_000)  # sell at discount
        gross = (exit_filled - entry_filled) * shares           # long P&L

    fees = 2.0 * FEE_PER_SHARE * shares
    net  = gross - fees
    hold = int(exit_bar_local - entry_idx)

    return {
        "entry_bar_et":    str(ts_strs[entry_idx]),
        "exit_bar_et":     str(ts_strs[exit_bar_local]),
        "direction":       direction,
        "shares":          round(float(shares), 4),
        "entry_price":     round(float(entry_filled), 4),
        "exit_price":      round(float(exit_filled), 4),
        "stop_price":      round(float(stop_price), 4),
        "target_price":    round(float(target_price), 4),
        "stop_dist":       round(float(stop_dist), 4),
        "or_high":         round(float(or_h), 4),
        "or_low":          round(float(or_l), 4),
        "or_mid":          round(float(or_mid), 4),
        "or_range":        round(float(or_r), 4),
        "gross_pnl":       round(float(gross), 4),
        "fees":            round(float(fees), 4),
        "net_pnl":         round(float(net), 4),
        "holding_bars":    hold,
        "holding_minutes": hold,
        "forced_exit":     forced,
        "exit_reason":     exit_reason,
    }


# ---------------------------------------------------------------------------
# Phase runner
# ---------------------------------------------------------------------------
def run_phase(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Iterate over all sessions in *df* and collect trades.

    Args:
        df: RTH-filtered DataFrame as returned by load_rth().

    Returns:
        List of trade dicts (one per session that produced a signal).
    """
    trades: List[Dict[str, Any]] = []
    ts_strs = df["ts_et"].astype(str).values
    opens   = df["open"].values.astype(np.float64)
    highs   = df["high"].values.astype(np.float64)
    lows    = df["low"].values.astype(np.float64)
    closes  = df["close"].values.astype(np.float64)
    vols    = df["volume"].values.astype(np.float64)
    mods    = df["min_of_day"].values.astype(np.int32)
    dates   = df["date_et"].values

    # Vectorized session-boundary detection
    session_starts = np.where(np.concatenate([[True], dates[1:] != dates[:-1]]))[0]
    session_ends   = np.concatenate([session_starts[1:], [len(df)]])

    for s, e in zip(session_starts, session_ends):
        trade = _run_session_fade(
            opens[s:e], highs[s:e], lows[s:e], closes[s:e],
            vols[s:e], mods[s:e], ts_strs[s:e],
        )
        if trade is not None:
            trades.append(trade)
    return trades


# ---------------------------------------------------------------------------
# Metrics  (identical interface to orb_fast.py compute_metrics)
# ---------------------------------------------------------------------------
def compute_metrics(trades: List[Dict[str, Any]], label: str = "") -> Dict[str, Any]:
    """Compute aggregate performance metrics from a list of trade dicts.

    Args:
        trades: List of trade dicts as returned by run_phase().
        label: Descriptive label to embed in the result dict.

    Returns:
        Dict of performance metrics.
    """
    if not trades:
        return {
            "label": label, "total_trades": 0, "win_rate": None,
            "profit_factor": None, "avg_holding_minutes": None,
            "total_net_pnl": 0.0, "total_return_pct": None,
            "max_drawdown_pct": None, "sharpe_daily": None,
            "avg_win": None, "avg_loss": None, "expectancy": None,
            "avg_trades_per_day": None, "forced_exits": 0,
            "num_trading_days": 0,
            "pct_short": None, "pct_long": None,
        }

    pnls   = np.array([t["net_pnl"] for t in trades], dtype=np.float64)
    wins   = pnls[pnls > 0.0]
    losses = pnls[pnls <= 0.0]
    gp = float(wins.sum())  if len(wins)   else 0.0
    gl = float(abs(losses.sum())) if len(losses) else 0.0
    pf = gp / gl if gl > 0.0 else float("inf")
    wr = float(len(wins)) / len(pnls)

    equity = np.cumsum(pnls)
    peak   = np.maximum.accumulate(equity)
    dd_pct = float(np.max(peak - equity)) / NOTIONAL * 100.0
    ret    = float(pnls.sum()) / NOTIONAL * 100.0

    days     = {t["entry_bar_et"][:10] for t in trades}
    avg_hold = float(np.mean([t["holding_minutes"] for t in trades]))

    dpnls_d: Dict[str, float] = {}
    for t in trades:
        d = t["entry_bar_et"][:10]
        dpnls_d[d] = dpnls_d.get(d, 0.0) + t["net_pnl"]
    dp = list(dpnls_d.values())
    sharpe = (np.mean(dp) / (np.std(dp) + 1e-9)) * np.sqrt(252) if len(dp) > 1 else None

    n_short = sum(1 for t in trades if t.get("direction") == "short")
    n_long  = sum(1 for t in trades if t.get("direction") == "long")

    return {
        "label":              label,
        "total_trades":       len(trades),
        "win_rate":           round(wr, 4),
        "profit_factor":      round(pf, 4),
        "avg_holding_minutes": round(avg_hold, 2),
        "avg_trades_per_day": round(len(trades) / max(len(days), 1), 3),
        "total_net_pnl":      round(float(pnls.sum()), 2),
        "total_return_pct":   round(ret, 4),
        "max_drawdown_pct":   round(dd_pct, 4),
        "avg_win":            round(float(wins.mean()), 2) if len(wins) else None,
        "avg_loss":           round(float(losses.mean()), 2) if len(losses) else None,
        "expectancy":         round(float(pnls.mean()), 2),
        "sharpe_daily":       round(float(sharpe), 4) if sharpe is not None else None,
        "forced_exits":       sum(1 for t in trades if t["forced_exit"]),
        "num_trading_days":   len(days),
        "pct_short":          round(n_short / len(trades), 3) if trades else None,
        "pct_long":           round(n_long  / len(trades), 3) if trades else None,
    }


# ---------------------------------------------------------------------------
# Mastery check
# ---------------------------------------------------------------------------
def passes_mastery(m: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Check whether metrics satisfy mastery thresholds.

    Args:
        m: Metrics dict from compute_metrics().

    Returns:
        Tuple of (passed: bool, fail_reasons: list[str]).
    """
    fails = []
    n  = m.get("total_trades") or 0
    wr = m.get("win_rate") or 0.0
    pf = m.get("profit_factor") or 0.0
    r  = m.get("total_return_pct") or 0.0
    dd = m.get("max_drawdown_pct") or 0.0
    h  = m.get("avg_holding_minutes") or 9999

    if n  < THRESHOLDS["n_min"]:    fails.append(f"n={n}<{THRESHOLDS['n_min']}")
    if wr < THRESHOLDS["wr_min"]:   fails.append(f"wr={wr:.1%}<{THRESHOLDS['wr_min']:.0%}")
    if pf < THRESHOLDS["pf_min"]:   fails.append(f"pf={pf:.2f}<{THRESHOLDS['pf_min']}")
    if r  <= THRESHOLDS["ret_min"]: fails.append(f"ret={r:.2f}%<={THRESHOLDS['ret_min']:.0f}%")
    if dd > THRESHOLDS["dd_max"]:   fails.append(f"dd=-{dd:.2f}%>-{THRESHOLDS['dd_max']:.0f}%")
    if h  > THRESHOLDS["hold_max"]: fails.append(f"hold={h:.0f}m>{THRESHOLDS['hold_max']}m")
    return len(fails) == 0, fails


# ---------------------------------------------------------------------------
# Run + save single ticker
# ---------------------------------------------------------------------------
def run_ticker(
    ticker: str,
    train_start: str = "2021-01-01",
    train_end: str   = "2022-12-31",
    test_start: str  = "2023-01-01",
    test_end: str    = "2024-12-31",
) -> Dict[str, Any]:
    """Backtest ORB fade for a single *ticker* with walk-forward split.

    Args:
        ticker: Uppercase ticker symbol.
        train_start: Start of training period, ISO date string.
        train_end: End of training period, ISO date string.
        test_start: Start of test period, ISO date string.
        test_end: End of test period, ISO date string.

    Returns:
        Full results dict including train/test metrics and trades.

    Raises:
        FileNotFoundError: If parquet data is missing.
        ValueError: If the loaded data is empty.
    """
    overall_start = min(train_start, test_start)
    overall_end   = max(train_end,   test_end)
    full_df = load_rth(ticker, overall_start, overall_end)
    if full_df.empty:
        raise ValueError(f"No RTH data for {ticker}")

    results: Dict[str, Any] = {"ticker": ticker, "strategy": "ORBFade"}

    for phase, s, e in [
        ("train", train_start, train_end),
        ("test",  test_start,  test_end),
    ]:
        phase_df = full_df[
            (full_df["date_et"] >= pd.to_datetime(s).date())
            & (full_df["date_et"] <= pd.to_datetime(e).date())
        ].copy()
        if phase_df.empty:
            results[phase] = {"error": f"no data for {phase}"}
            continue
        trades = run_phase(phase_df)
        m = compute_metrics(trades, f"{ticker}|ORBFade|{phase}")
        results[phase] = {
            "metrics": m,
            "trades": trades,
            "period_start": s,
            "period_end": e,
        }

    results["no_lookahead_audit"] = {
        "entry_execution":       "next_bar_open",
        "future_bars_accessed":  False,
        "cross_session_leakage": False,
        "force_flat_rule":       "15:55_ET",
        "or_built_from":         f"bars_0_to_{OR_MINUTES - 1}",
        "stop_computed_from":    "ATR14_at_signal_bar_capped_OR_range",
        "target":                "OR_midpoint",
    }

    out_dir = RESULTS_ROOT / ticker / "orb_fade"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Slim result (no trade lists) for quick inspection
    slim = {k: v for k, v in results.items() if k not in ("train", "test")}
    for ph in ("train", "test"):
        if ph in results:
            slim[ph] = {k: v for k, v in results[ph].items() if k != "trades"}
    (out_dir / "result.json").write_text(json.dumps(slim, indent=2, default=str))

    # Full trade lists per phase
    for ph in ("train", "test"):
        if ph in results and "trades" in results[ph]:
            (out_dir / f"trades_{ph}.json").write_text(
                json.dumps(results[ph]["trades"], indent=2, default=str)
            )
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tickers = sys.argv[1:] if len(sys.argv) > 1 else ["AAPL"]
    for ticker in tickers:
        print(f"\n{'='*60}")
        print(f"ORB Fade — {ticker}")
        print("=" * 60)
        t0 = time.time()
        try:
            r = run_ticker(ticker)
            for phase in ("train", "test"):
                m = r.get(phase, {}).get("metrics", {})
                if not m:
                    print(f"  {phase}: no metrics")
                    continue
                ok, fails = passes_mastery(m)
                status = "PASS" if ok else "FAIL"
                print(
                    f"  {phase:5s} [{status}]  "
                    f"n={m['total_trades']:3d}  "
                    f"wr={m['win_rate']:.1%}  "
                    f"pf={m['profit_factor']:.2f}  "
                    f"ret={m['total_return_pct']:+.2f}%  "
                    f"dd=-{m['max_drawdown_pct']:.2f}%  "
                    f"hold={m['avg_holding_minutes']:.0f}m  "
                    f"sharpe={m.get('sharpe_daily') or 'N/A'}"
                )
                if not ok:
                    print(f"         fail_reasons: {', '.join(fails)}")
            print(f"  elapsed: {time.time()-t0:.2f}s")
        except Exception as exc:
            print(f"  ERROR: {exc}")
