"""backfill_yfinance_5yr.py — bulk daily OHLCV backfill for S&P 500.

Pulls 5-year daily OHLCV from yfinance in batches of 50 tickers per request.
Writes one parquet per ticker under cache/yfinance_5yr/{TICKER}.parquet.

Usage:
    python scripts/backfill_yfinance_5yr.py [--tickers TICKER1,TICKER2,...] [--years 5]

Default: reads sp500_tickers.txt (509 tickers) and pulls 5 years of daily data.
Free; no API key required. ~10 batches × 50 = full universe in <2 min.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

_ROOT = Path(__file__).resolve().parents[1]
TICKERS_PATH = _ROOT / "sp500_tickers.txt"
OUT_DIR = _ROOT / "cache" / "yfinance_5yr"
MANIFEST = _ROOT / "cache" / "yfinance_5yr" / "_manifest.json"


def load_tickers() -> list[str]:
    with open(TICKERS_PATH) as f:
        return [line.strip() for line in f if line.strip()]


def backfill_batch(tickers: list[str], years: int = 5) -> dict:
    """Download daily OHLCV for a batch and write per-ticker parquet."""
    end = datetime.utcnow().date()
    start = end - timedelta(days=365 * years + 10)
    try:
        df = yf.download(
            tickers=" ".join(tickers),
            start=start.isoformat(),
            end=end.isoformat(),
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            actions=False,
            progress=False,
            threads=True,
        )
    except Exception as e:
        return {"error": str(e), "tickers": tickers}

    out = {}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for t in tickers:
        try:
            if len(tickers) == 1:
                sub = df.copy()
            else:
                sub = df[t].copy() if t in df.columns.get_level_values(0) else None
            if sub is None or sub.empty:
                out[t] = {"rows": 0, "error": "empty"}
                continue
            sub = sub.reset_index()
            sub.columns = [str(c).lower() for c in sub.columns]
            sub["ticker"] = t
            p = OUT_DIR / f"{t}.parquet"
            sub.to_parquet(p, index=False)
            out[t] = {
                "rows": len(sub),
                "start": str(sub["date"].iloc[0])[:10] if "date" in sub.columns else None,
                "end": str(sub["date"].iloc[-1])[:10] if "date" in sub.columns else None,
                "path": str(p),
            }
        except Exception as e:
            out[t] = {"error": str(e)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default=None, help="Comma-separated subset")
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--smoke", action="store_true", help="Smoke test 5 tickers only")
    args = ap.parse_args()

    if args.smoke:
        tickers = ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL"]
    elif args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = load_tickers()

    print(f"[yfinance] backfilling {len(tickers)} tickers, {args.years}yr")
    full_manifest = {}
    t0 = time.time()
    for i in range(0, len(tickers), args.batch_size):
        batch = tickers[i : i + args.batch_size]
        print(f"  batch {i // args.batch_size + 1}: {len(batch)} tickers ({batch[0]}..{batch[-1]})")
        res = backfill_batch(batch, years=args.years)
        full_manifest.update(res)
        time.sleep(1.0)
    elapsed = time.time() - t0

    # Manifest
    summary = {
        "source": "yfinance",
        "ts": datetime.utcnow().isoformat() + "Z",
        "tickers_attempted": len(tickers),
        "tickers_succeeded": sum(1 for v in full_manifest.values() if "rows" in v and v.get("rows", 0) > 0),
        "rows_total": sum(v.get("rows", 0) for v in full_manifest.values()),
        "elapsed_sec": round(elapsed, 1),
        "years": args.years,
        "per_ticker": full_manifest,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[yfinance] DONE: {summary['tickers_succeeded']}/{summary['tickers_attempted']} OK, {summary['rows_total']} rows, {elapsed:.1f}s")
    print(f"[yfinance] manifest: {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
