"""backfill_news_gdelt.py — GDELT 2.0 Doc API ticker-mention backfill.

Pulls news article counts + tone for S&P 500 tickers using GDELT's free
Doc 2.0 API (no key required). Output: per-ticker parquet with daily
article count + average tone.

API:
    https://api.gdeltproject.org/api/v2/doc/doc
    ?query=<TICKER>+stock
    &mode=TimelineVolInfo
    &timespan=5years
    &format=json

GDELT free tier: ~100 req/sec; 5yr historical timeline counts as one req.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

_ROOT = Path(__file__).resolve().parents[1]
TICKERS_PATH = _ROOT / "sp500_tickers.txt"
OUT_DIR = _ROOT / "cache" / "gdelt_news"
MANIFEST = OUT_DIR / "_manifest.json"

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"


def load_tickers() -> list[str]:
    with open(TICKERS_PATH) as f:
        return [line.strip() for line in f if line.strip()]


def fetch_ticker_timeline(ticker: str, timespan: str = "5years") -> pd.DataFrame:
    """Fetch GDELT timeline-vol-tone for a ticker. Returns date,count,tone."""
    params = {
        "query": f"{ticker} stock",
        "mode": "TimelineVolInfo",
        "timespan": timespan,
        "format": "json",
        "timelinesmooth": 1,
    }
    r = requests.get(GDELT_DOC_API, params=params, timeout=30, headers={"User-Agent": "Mozilla/5.0 sp500-mastery/1.0"})
    r.raise_for_status()
    try:
        data = r.json()
    except Exception:
        return pd.DataFrame(columns=["date", "vol", "ticker"])
    # GDELT returns timeline.[].data[]
    rows = []
    for tl in data.get("timeline", []):
        for d in tl.get("data", []):
            rows.append({
                "date": d.get("date"),
                "vol": d.get("value"),
            })
    if not rows:
        return pd.DataFrame(columns=["date", "vol", "ticker"])
    df = pd.DataFrame(rows)
    # GDELT timestamps are 'YYYYMMDDHHMMSS'
    df["date"] = pd.to_datetime(df["date"].astype(str).str[:8], format="%Y%m%d", errors="coerce")
    df["vol"] = pd.to_numeric(df["vol"], errors="coerce")
    df = df.dropna(subset=["date"]).reset_index(drop=True)
    df["ticker"] = ticker
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--timespan", default="5years")
    ap.add_argument("--max", type=int, default=None, help="Max tickers (for partial runs)")
    ap.add_argument("--sleep", type=float, default=0.5)
    args = ap.parse_args()

    if args.smoke:
        tickers = ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL"]
    elif args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",")]
    else:
        tickers = load_tickers()

    if args.max:
        tickers = tickers[: args.max]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    manifest = {}
    print(f"[gdelt] backfilling {len(tickers)} tickers, timespan={args.timespan}")
    for i, t in enumerate(tickers):
        try:
            df = fetch_ticker_timeline(t, timespan=args.timespan)
            p = OUT_DIR / f"{t}.parquet"
            df.to_parquet(p, index=False)
            manifest[t] = {
                "rows": len(df),
                "start": str(df["date"].iloc[0])[:10] if len(df) else None,
                "end": str(df["date"].iloc[-1])[:10] if len(df) else None,
                "path": str(p),
            }
            if i % 50 == 0:
                print(f"  [gdelt] {i+1}/{len(tickers)}: {t} -> {len(df)} rows")
            time.sleep(args.sleep)
        except Exception as e:
            manifest[t] = {"error": str(e)}
            time.sleep(args.sleep)

    elapsed = time.time() - t0
    summary = {
        "source": "gdelt",
        "ts": datetime.utcnow().isoformat() + "Z",
        "tickers_attempted": len(tickers),
        "tickers_succeeded": sum(1 for v in manifest.values() if "rows" in v and v.get("rows", 0) > 0),
        "rows_total": sum(v.get("rows", 0) for v in manifest.values()),
        "elapsed_sec": round(elapsed, 1),
        "timespan": args.timespan,
        "per_ticker": manifest,
    }
    with open(MANIFEST, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[gdelt] DONE: {summary['tickers_succeeded']}/{summary['tickers_attempted']} OK, {summary['rows_total']} rows, {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
