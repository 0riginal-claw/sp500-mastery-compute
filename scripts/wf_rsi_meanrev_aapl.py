#!/usr/bin/env python
"""Walk-forward backtest: AAPL daily RSI(14)<30 mean-reversion, hold 21 bars.

WF config: 2yr IS / 1yr OOS / 1yr step. Reports per-fold n_trades, WR, PF, Ret%.
Spawn brief: c39a6daad5b74374 (autonomous_mode).
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import yfinance as yf

# autosolve_skip: autonomous_mode single-helper slice
os.environ.setdefault("AUTO_CLOUD_DISPATCH", "0")  # local mechanical task <60s


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    # Wilder's smoothing via EMA with alpha=1/period
    avg_up = up.ewm(alpha=1 / period, adjust=False).mean()
    avg_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def simulate(df: pd.DataFrame, hold: int = 21) -> pd.DataFrame:
    """Long-only: enter at close on RSI<30 day, exit at close hold-bars later.

    No stacking: skip new signals while position open.
    """
    closes = df["Close"].to_numpy()
    rsis = df["RSI"].to_numpy()
    dates = df.index.to_numpy()
    n = len(df)
    trades = []
    i = 0
    while i < n:
        if not np.isnan(rsis[i]) and rsis[i] < 30:
            entry_idx = i
            exit_idx = min(i + hold, n - 1)
            if exit_idx <= entry_idx:
                break
            ret = closes[exit_idx] / closes[entry_idx] - 1.0
            trades.append(
                {
                    "entry_date": pd.Timestamp(dates[entry_idx]).date().isoformat(),
                    "exit_date": pd.Timestamp(dates[exit_idx]).date().isoformat(),
                    "entry_px": float(closes[entry_idx]),
                    "exit_px": float(closes[exit_idx]),
                    "ret_pct": float(ret * 100),
                    "bars_held": exit_idx - entry_idx,
                    "entry_rsi": float(rsis[entry_idx]),
                }
            )
            i = exit_idx + 1  # skip to bar after exit (no re-entry overlap)
        else:
            i += 1
    return pd.DataFrame(trades)


def fold_metrics(trades: pd.DataFrame) -> dict:
    n = len(trades)
    if n == 0:
        return {"n_trades": 0, "win_rate": None, "profit_factor": None, "ret_pct": 0.0}
    wins = trades[trades["ret_pct"] > 0]["ret_pct"]
    losses = trades[trades["ret_pct"] <= 0]["ret_pct"]
    sum_win = wins.sum()
    sum_loss_abs = -losses.sum()  # positive
    wr = len(wins) / n
    if sum_loss_abs > 0:
        pf = sum_win / sum_loss_abs
    elif sum_win > 0:
        pf = float("inf")
    else:
        pf = 0.0
    # compounded equity curve over fold
    eq = (1 + trades["ret_pct"] / 100).prod() - 1
    return {
        "n_trades": int(n),
        "win_rate": round(wr, 4),
        "profit_factor": round(pf, 4) if np.isfinite(pf) else None,
        "ret_pct": round(float(eq * 100), 4),
        "sum_win_pct": round(float(sum_win), 4),
        "sum_loss_pct": round(float(losses.sum()), 4),
    }


def main() -> int:
    ticker = "AAPL"
    today = datetime.now(timezone.utc).date()
    # Folds: 4 OOS folds of 1yr each, step 1yr. IS = 2yr preceding each OOS.
    n_folds = 4
    oos_years = 1
    is_years = 2

    # Latest fold ends today
    fold_specs = []
    for k in range(n_folds):
        oos_end = today - timedelta(days=365 * (n_folds - 1 - k))
        oos_start = oos_end - timedelta(days=365 * oos_years)
        is_end = oos_start
        is_start = is_end - timedelta(days=365 * is_years)
        fold_specs.append(
            {"fold": k + 1, "is_start": is_start, "is_end": is_end, "oos_start": oos_start, "oos_end": oos_end}
        )

    earliest = min(f["is_start"] for f in fold_specs) - timedelta(days=30)
    latest = today

    print(f"Fetching {ticker} daily {earliest} → {latest}", file=sys.stderr)
    raw = yf.download(
        ticker,
        start=earliest.isoformat(),
        end=latest.isoformat(),
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        print("ERROR: no data from yfinance", file=sys.stderr)
        return 2
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw["RSI"] = rsi(raw["Close"], 14)

    fold_results = []
    all_trades = []
    for f in fold_specs:
        oos_mask = (raw.index.date >= f["oos_start"]) & (raw.index.date < f["oos_end"])
        oos_df = raw.loc[oos_mask].copy()
        if len(oos_df) < 30:
            fold_results.append({**{k: v.isoformat() if hasattr(v, "isoformat") else v for k, v in f.items()},
                                  "n_trades": 0, "win_rate": None, "profit_factor": None, "ret_pct": 0.0,
                                  "note": "insufficient_oos_bars"})
            continue
        trades = simulate(oos_df, hold=21)
        m = fold_metrics(trades)
        row = {**{k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in f.items()}, **m}
        fold_results.append(row)
        if not trades.empty:
            trades = trades.assign(fold=f["fold"])
            all_trades.append(trades)

    results_df = pd.DataFrame(fold_results)
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

    out_dir = "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/s&p500-ticker-mastery/backtests/AAPL"
    os.makedirs(out_dir, exist_ok=True)
    results_csv = os.path.join(out_dir, "wf_rsi30_meanrev_2yIS_1yOOS_hold21.csv")
    trades_csv = os.path.join(out_dir, "wf_rsi30_meanrev_2yIS_1yOOS_hold21_trades.csv")
    results_df.to_csv(results_csv, index=False)
    trades_df.to_csv(trades_csv, index=False)

    # promising flag: ALL folds with n>0 must hit WR>=52% AND PF>=1.10
    folds_with_trades = [r for r in fold_results if r["n_trades"] > 0]
    if folds_with_trades:
        promising_all = all(
            (r["win_rate"] is not None and r["win_rate"] >= 0.52)
            and (r["profit_factor"] is not None and r["profit_factor"] >= 1.10)
            for r in folds_with_trades
        )
        # aggregate-level: weighted by trades
        total_n = sum(r["n_trades"] for r in folds_with_trades)
        agg_wr = sum((r["win_rate"] or 0) * r["n_trades"] for r in folds_with_trades) / max(total_n, 1)
        agg_pf_num = sum((r.get("sum_win_pct") or 0) for r in folds_with_trades)
        agg_pf_den = -sum((r.get("sum_loss_pct") or 0) for r in folds_with_trades)
        agg_pf = (agg_pf_num / agg_pf_den) if agg_pf_den > 0 else (float("inf") if agg_pf_num > 0 else 0.0)
        promising_agg = agg_wr >= 0.52 and agg_pf >= 1.10
    else:
        promising_all = False
        promising_agg = False
        agg_wr = 0.0
        agg_pf = 0.0

    summary = {
        "ticker": ticker,
        "strategy": "daily RSI(14)<30 mean reversion, hold 21 bars",
        "wf_config": "2yr IS / 1yr OOS / 1yr step, 4 folds",
        "data_range": f"{earliest.isoformat()} -> {latest.isoformat()}",
        "n_bars": len(raw),
        "folds": fold_results,
        "aggregate": {
            "total_trades": int(sum(r["n_trades"] for r in fold_results)),
            "weighted_win_rate": round(agg_wr, 4),
            "aggregate_profit_factor": (round(agg_pf, 4) if np.isfinite(agg_pf) else None),
        },
        "promising_all_folds": promising_all,
        "promising_aggregate": promising_agg,
        "threshold": "WR>=52% AND PF>=1.10",
        "outputs": {"results_csv": results_csv, "trades_csv": trades_csv},
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
