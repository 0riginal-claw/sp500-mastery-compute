#!/usr/bin/env python3
"""Build daily + rolling training rows for XGBoost retrain.

Joins per-day feature snapshots (paper_trade/features/<DATE>/<TICKER>.parquet)
with realized P&L labels (paper_trade/state/<DATE>_state.json:closed_trades).
Writes paper_trade/training_rows/<DATE>.parquet AND all_rolling.parquet
(rolling concat of last 30 days).

2026-05-21: Mythos 256-dim transformer dropped per OC audit rank #2. Any
legacy per-day snapshots that contain mythos_emb_* / emb_<0..255> columns are
stripped here so the rolling concat never resurfaces them. Restoration path:
unset MYTHOS_DROP_STRIP (or set =0) AND set MYTHOS_DISABLED=0 in env.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

# 2026-05-21: Mythos col cleanup. Default ON — strips emb_X / mythos_emb_X from
# any legacy snapshot. Set MYTHOS_DROP_STRIP=0 to disable (e.g. during a
# transitional restoration window where you want the old cols passed through).
_MYTHOS_STRIP_RAW = os.environ.get("MYTHOS_DROP_STRIP", "1").strip().lower()
MYTHOS_DROP_STRIP: bool = _MYTHOS_STRIP_RAW in ("1", "true", "yes", "on")
_MYTHOS_COL_RE = re.compile(r"^(mythos_emb_\d+|emb_\d+)$")


def _strip_mythos_cols(df):
    """Drop any mythos_emb_* / emb_<N> cols from a DataFrame. No-op if absent."""
    if not MYTHOS_DROP_STRIP:
        return df, []
    drop_cols = [c for c in df.columns if _MYTHOS_COL_RE.match(str(c))]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df, drop_cols

WORK = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-"
    "zachgladstone@gmail.com/My Drive/AI-Tools/s&p500-ticker-mastery"
)
PAPER_DIR = WORK / "paper_trade"
FEATURES_DIR = PAPER_DIR / "features"
STATE_DIR = PAPER_DIR / "state"
FILLS_DIR = PAPER_DIR / "fills"
OUT_DIR = PAPER_DIR / "training_rows"


def _daterange(since: str, until: str):
    s = datetime.strptime(since, "%Y-%m-%d").date()
    u = datetime.strptime(until, "%Y-%m-%d").date()
    d = s
    while d <= u:
        yield d.strftime("%Y-%m-%d")
        d += timedelta(days=1)


def _aggregate_fills_by_ticker(date_str: str) -> dict[str, dict]:
    """Aggregate SELL fills per ticker from paper_trade/fills/<DATE>.jsonl.

    Returns {ticker: {sell_qty, sell_proceeds, avg_sell_px}}. Used as a
    fallback exit_price source when closed_trades[].exit_price is None
    (a common occurrence: the LIVE_PAPER flatten path leaves
    exit_price=None + pending_ws_fill=True, and the reconciler matches
    by BUY-order_id which does NOT match the SELL-order_id in fills).

    Fix A (2026-05-21): per-trade pnl was always None -> y_label always 0.
    By deriving exit price from sell fills (aggregated per ticker), we can
    populate pnl = (avg_sell - entry_price) * sell_qty in _load_labels.
    """
    p = FILLS_DIR / f"{date_str}.jsonl"
    if not p.exists():
        return {}
    out: dict[str, dict] = {}
    try:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                side = str(r.get("side") or "")
                if "SELL" not in side.upper():
                    continue
                sym = r.get("symbol") or ""
                if not sym:
                    continue
                try:
                    qty = float(r.get("qty") or 0.0)
                    px = float(r.get("filled_avg_price") or 0.0)
                except (TypeError, ValueError):
                    continue
                if qty <= 0 or px <= 0:
                    continue
                slot = out.setdefault(sym, {"sell_qty": 0.0, "sell_proceeds": 0.0})
                slot["sell_qty"] += qty
                slot["sell_proceeds"] += qty * px
    except Exception:
        return {}
    for sym, slot in out.items():
        if slot["sell_qty"] > 0:
            slot["avg_sell_px"] = slot["sell_proceeds"] / slot["sell_qty"]
        else:
            slot["avg_sell_px"] = 0.0
    return out


def _load_labels(date_str: str) -> dict[str, dict]:
    """Return {ticker: {pnl, entry_price, exit_price, y_label}} for closed trades.

    Fix A (2026-05-21): when closed_trades[].pnl is None (the usual case
    because LIVE_PAPER flatten leaves it None + reconciler fails to match
    by BUY-order_id), derive exit_price from the day's SELL fills aggregated
    per ticker, then compute pnl = (avg_sell - entry_price) * sell_qty.
    """
    p = STATE_DIR / f"{date_str}_state.json"
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text())
    except Exception:
        return {}
    # Pre-load fills aggregate so we can backfill exit_price/pnl per ticker.
    fills_agg = _aggregate_fills_by_ticker(date_str)
    out: dict[str, dict] = {}
    for t in d.get("closed_trades") or []:
        if not isinstance(t, dict):
            continue
        ticker = t.get("ticker")
        if not ticker:
            continue
        # 1) Prefer explicit per-trade pnl if writer populated it.
        pnl_raw = t.get("pnl")
        entry_px = t.get("entry_price") or t.get("entry")
        exit_px = t.get("exit_price") or t.get("exit")
        qty = t.get("qty")
        # 2) If pnl missing, try entry/exit/qty arithmetic from state.
        if pnl_raw is None and entry_px is not None and exit_px is not None and qty is not None:
            try:
                pnl_raw = (float(exit_px) - float(entry_px)) * float(qty)
            except (TypeError, ValueError):
                pnl_raw = None
        # 3) If still missing, derive exit_price from fills aggregate
        #    (the canonical fallback for LIVE_PAPER closed_trades whose
        #    pending_ws_fill never reconciled).
        if pnl_raw is None and entry_px is not None and ticker in fills_agg:
            f = fills_agg[ticker]
            sell_qty = float(f.get("sell_qty") or 0.0)
            avg_sell = float(f.get("avg_sell_px") or 0.0)
            if sell_qty > 0 and avg_sell > 0:
                try:
                    pnl_raw = (avg_sell - float(entry_px)) * sell_qty
                    exit_px = avg_sell  # propagate derived exit_price
                    qty = sell_qty
                except (TypeError, ValueError):
                    pass
        # 4) Last-resort fallbacks.
        if pnl_raw is None:
            pnl_raw = t.get("realized_pnl") or 0.0
        try:
            pnl_f = float(pnl_raw)
        except (TypeError, ValueError):
            pnl_f = 0.0
        out[ticker] = {
            "pnl": pnl_f,
            "entry_price": entry_px,
            "exit_price": exit_px,
            "y_label": 1 if pnl_f > 0 else 0,
        }
    return out


def _build_day(date_str: str, pd) -> "pd.DataFrame | None":
    day_feat_dir = FEATURES_DIR / date_str
    if not day_feat_dir.exists():
        return None
    labels = _load_labels(date_str)
    rows = []
    stripped_total = 0
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
        # 2026-05-21: drop any legacy Mythos cols from per-ticker snapshots
        # before assembling the row dict. Most current snapshots have 0 such
        # cols (verified empty); the call is a safe no-op when none present.
        df, dropped = _strip_mythos_cols(df)
        stripped_total += len(dropped)
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
    if stripped_total > 0:
        print(
            f"[mythos-drop] {date_str}: stripped {stripped_total} legacy emb cols total",
            file=sys.stderr,
        )
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
        # Belt-and-braces: strip Mythos cols from final rolling DF in case a
        # legacy per-day parquet on disk slips through (sort=False preserves
        # union of cols across days; old days may have emb_* cols even if
        # today's _build_day stripped them).
        roll, rolled_dropped = _strip_mythos_cols(roll)
        if rolled_dropped:
            print(
                f"[mythos-drop] rolling concat stripped {len(rolled_dropped)} legacy emb cols",
                file=sys.stderr,
            )
        roll_path = OUT_DIR / "all_rolling.parquet"
        roll.to_parquet(roll_path, index=False)
        print(f"[ok] days={days_processed} rolling_rows={len(roll)} cols={len(roll.columns)} → {roll_path}")
    else:
        print(f"[warn] no per-day rows built in window {args.since}..{args.until}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
