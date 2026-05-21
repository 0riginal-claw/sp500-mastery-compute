# autosolve_skip: self-contained backtest script
"""Walk-forward backtest: daily RSI<30 mean-reversion on AAPL, hold 21 days.

Folds: 4 consecutive 1yr OOS windows (end-anchored at data end 2026-04-20).
Prior 2yr serves as IS / indicator warmup (RSI<30 is param-free, no tuning needed).

Entry: next bar after RSI(14) < 30 signal (close-to-close return).
Hold: 21 trading days, single-position (skip new signals while in trade).
Cost: 1 bp/side commission/slippage assumption.
"""
import json
import glob
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path("/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/state/autonomous_mode/reports")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_GLOB = "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/s&p500-ticker-mastery/cache/features/AAPL_v10_full_*.parquet"
HOLD = 21
RSI_THRESH = 30.0
COST_BPS_PER_SIDE = 1.0


def load():
    f = sorted(glob.glob(FEATURE_GLOB))[0]
    df = pd.read_parquet(f, columns=["close", "rsi_14"]).sort_index()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df, f


def fold_bounds(end_ts: pd.Timestamp, n_folds: int = 4):
    """Return [(oos_start, oos_end), ...] in chronological order, each 1yr."""
    bounds = []
    for k in range(n_folds, 0, -1):
        oos_end = end_ts - pd.DateOffset(years=n_folds - k)
        oos_start = oos_end - pd.DateOffset(years=1) + pd.Timedelta(days=1)
        bounds.append((oos_start.normalize(), oos_end.normalize()))
    return bounds


def backtest_fold(df: pd.DataFrame, oos_start: pd.Timestamp, oos_end: pd.Timestamp):
    """Return per-trade returns (net of round-trip cost) for trades ENTERED in [oos_start, oos_end]."""
    sub = df.loc[df.index <= oos_end + pd.Timedelta(days=HOLD * 2)].copy()
    sub["signal"] = sub["rsi_14"] < RSI_THRESH
    closes = sub["close"].values
    idx = sub.index
    n = len(sub)

    trades = []
    i = 0
    cost = 2 * COST_BPS_PER_SIDE / 1e4
    while i < n - HOLD - 1:
        entry_signal_day = idx[i]
        if sub["signal"].iat[i] and oos_start <= idx[i + 1] <= oos_end:
            entry_px = closes[i + 1]
            exit_px = closes[i + 1 + HOLD]
            r = exit_px / entry_px - 1.0 - cost
            trades.append({
                "signal_day": entry_signal_day,
                "entry_day": idx[i + 1],
                "exit_day": idx[i + 1 + HOLD],
                "entry": float(entry_px),
                "exit": float(exit_px),
                "ret": float(r),
            })
            i = i + 1 + HOLD + 1
        else:
            i += 1
    return trades


def stats(trades):
    if not trades:
        return dict(n_trades=0, win_rate=None, profit_factor=None, total_ret=0.0, avg_ret=None, median_ret=None)
    rets = np.array([t["ret"] for t in trades])
    wins = rets[rets > 0]
    losses = rets[rets < 0]
    pf = (wins.sum() / abs(losses.sum())) if losses.size and losses.sum() != 0 else float("inf")
    return dict(
        n_trades=int(rets.size),
        win_rate=float((rets > 0).mean()),
        profit_factor=float(pf) if np.isfinite(pf) else None,
        total_ret=float(rets.sum()),
        avg_ret=float(rets.mean()),
        median_ret=float(np.median(rets)),
    )


def main():
    df, src = load()
    end_ts = df.index.max()
    folds = fold_bounds(end_ts, n_folds=4)

    out = {
        "task": "WF backtest AAPL daily RSI<30 mean-reversion, hold 21d",
        "source_feature_file": src,
        "data_range": [str(df.index.min().date()), str(df.index.max().date())],
        "n_bars": int(len(df)),
        "params": {
            "rsi_lookback": 14,
            "rsi_thresh": RSI_THRESH,
            "hold_days": HOLD,
            "cost_bps_per_side": COST_BPS_PER_SIDE,
            "is_years": 2,
            "oos_years_per_fold": 1,
            "step_years": 1,
        },
        "folds": [],
    }
    all_trades = []
    for k, (s, e) in enumerate(folds, start=1):
        trs = backtest_fold(df, s, e)
        st = stats(trs)
        st.update({"fold": k, "oos_start": str(s.date()), "oos_end": str(e.date())})
        out["folds"].append(st)
        all_trades.extend([{**t, "fold": k} for t in trs])

    out["aggregate_oos"] = stats(all_trades)

    promising = []
    for f in out["folds"]:
        if f["win_rate"] is not None and f["profit_factor"] is not None:
            if f["win_rate"] >= 0.52 and f["profit_factor"] >= 1.10:
                promising.append(f["fold"])
    out["promising_folds"] = promising
    agg = out["aggregate_oos"]
    out["aggregate_promising"] = bool(
        agg["win_rate"] is not None
        and agg["profit_factor"] is not None
        and agg["win_rate"] >= 0.52
        and agg["profit_factor"] >= 1.10
    )

    (OUT_DIR / "wf_aapl_rsi_2026-05-20.json").write_text(json.dumps(out, indent=2, default=str))
    pd.DataFrame(all_trades).to_csv(OUT_DIR / "wf_aapl_rsi_2026-05-20_trades.csv", index=False)

    lines = []
    lines.append("# Walk-forward AAPL daily RSI<30 mean-reversion, hold 21d\n")
    lines.append(f"- Source: `{src.split('/')[-1]}`")
    lines.append(f"- Data range: {out['data_range'][0]} .. {out['data_range'][1]} ({out['n_bars']} bars)")
    lines.append(f"- Cost: {COST_BPS_PER_SIDE} bp/side round-trip")
    lines.append("")
    lines.append("| Fold | OOS start  | OOS end    | n_trades | WR     | PF     | Ret%   |")
    lines.append("|------|------------|------------|----------|--------|--------|--------|")
    for f in out["folds"]:
        wr = f"{f['win_rate']*100:.1f}%" if f["win_rate"] is not None else "—"
        pf = f"{f['profit_factor']:.2f}" if f["profit_factor"] is not None else "—"
        ret = f"{f['total_ret']*100:.2f}%"
        lines.append(f"| {f['fold']} | {f['oos_start']} | {f['oos_end']} | {f['n_trades']} | {wr} | {pf} | {ret} |")
    a = out["aggregate_oos"]
    wr = f"{a['win_rate']*100:.1f}%" if a["win_rate"] is not None else "—"
    pf = f"{a['profit_factor']:.2f}" if a["profit_factor"] is not None else "—"
    lines.append(f"| **agg** | — | — | {a['n_trades']} | {wr} | {pf} | {a['total_ret']*100:.2f}% |")
    lines.append("")
    lines.append(f"- Promising folds (WR≥52% & PF≥1.10): {out['promising_folds'] or 'none'}")
    lines.append(f"- Aggregate promising: {out['aggregate_promising']}")
    (OUT_DIR / "wf_aapl_rsi_2026-05-20.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
