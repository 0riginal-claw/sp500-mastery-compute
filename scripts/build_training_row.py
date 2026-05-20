#!/usr/bin/env python3
"""Build daily + rolling training rows for XGBoost/Mythos retrain.

Joins per-day feature snapshots (paper_trade/features/<DATE>/<TICKER>.parquet)
with realized P&L labels (paper_trade/state/<DATE>_state.json:closed_trades).
Writes paper_trade/training_rows/<DATE>.parquet AND all_rolling.parquet
(rolling concat of last 30 days).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

WORK = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-"
    "zachgladstone@gmail.com/My Drive/AI-Tools/s&p500-ticker-mastery"
)
PAPER_DIR = WORK / "paper_trade"
FEATURES_DIR = PAPER_DIR / "features"
STATE_DIR = PAPER_DIR / "state"
OUT_DIR = PAPER_DIR / "training_rows"


def _daterange(since: str, until: str):
    s = datetime.strptime(since, "%Y-%m-%d").date()
    u = datetime.strptime(until, "%Y-%m-%d").date()
    d = s
    while d <= u:
        yield d.strftime("%Y-%m-%d")
        d += timedelta(days=1)


def _load_labels(date_str: str) -> dict[str, dict]:
    """Return {ticker: {pnl, entry_price, exit_price, y_label}} for closed trades."""
    p = STATE_DIR / f"{date_str}_state.json"
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text())
    except Exception:
        return {}
    out: dict[str, dict] = {}
    for t in d.get("closed_trades") or []:
        if not isinstance(t, dict):
            continue
        ticker = t.get("ticker")
        if not ticker:
            continue
        pnl = t.get("pnl") or t.get("realized_pnl") or 0.0
        try:
            pnl_f = float(pnl)
        except (TypeError, ValueError):
            pnl_f = 0.0
        out[ticker] = {
            "pnl": pnl_f,
            "entry_price": t.get("entry_price") or t.get("entry") or None,
            "exit_price": t.get("exit_price") or t.get("exit") or None,
            "y_label": 1 if pnl_f > 0 else 0,
        }
    return out


def _build_day(date_str: str, pd) -> "pd.DataFrame | None":
    day_feat_dir = FEATURES_DIR / date_str
    if not day_feat_dir.exists():
        return None
    labels = _load_labels(date_str)
    rows = []
    for fp in sorted(day_feat_dir.glob("*.parquet")):
        ticker = fp.stem.upper()
        try:
            df = pd.read_parquet(fp)
        except Exception as e:
            print(f"[warn] failed read {fp}: {e}", file=sys.stderr)
            continue
        # Collapse multi-row feature parquet → final row (most recent snapshot).
        if df.empty:
            continue
        row = df.iloc[-1].to_dict()
        row["ticker"] = ticker
        row["date"] = date_str
        lab = labels.get(ticker)
        if lab:
            row.update(lab)
        else:
            row.update({"pnl": None, "entry_price": None, "exit_price": None, "y_label": None})
        rows.append(row)
    if not rows:
        return None
    return pd.DataFrame(rows)


def main() -> int:
    today = datetime.now().strftime("%Y-%m-%d")
    default_since = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--since", default=default_since,
                    help="Inclusive start date YYYY-MM-DD (default: today - 30d)")
    ap.add_argument("--until", default=today,
                    help="Inclusive end date YYYY-MM-DD (default: today)")
    args = ap.parse_args()

    try:
        import pandas as pd  # noqa
    except ImportError as e:
        print(f"[warn] pandas missing ({e}); exit 0", file=sys.stderr)
        return 0
    try:
        import pyarrow  # noqa: F401
    except ImportError as e:
        print(f"[warn] pyarrow missing ({e}); exit 0", file=sys.stderr)
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_day_frames = []
    days_processed = 0
    for d in _daterange(args.since, args.until):
        df = _build_day(d, pd)
        if df is None or df.empty:
            continue
        per_day_path = OUT_DIR / f"{d}.parquet"
        df.to_parquet(per_day_path, index=False)
        per_day_frames.append(df)
        days_processed += 1

    if per_day_frames:
        roll = pd.concat(per_day_frames, ignore_index=True, sort=False)
        roll_path = OUT_DIR / "all_rolling.parquet"
        roll.to_parquet(roll_path, index=False)
        print(f"[ok] days={days_processed} rolling_rows={len(roll)} → {roll_path}")
    else:
        print(f"[warn] no per-day rows built in window {args.since}..{args.until}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
