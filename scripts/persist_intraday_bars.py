#!/usr/bin/env python3
"""Persist 1-minute SIP bars for every ticker traded today.

Run mode: cron-style after market close (16:10 ET).
For each ticker in today's state.positions + state.closed_trades, pull SIP 1m
bars from Alpaca historical API for [today_open_et - 15min, today_close_et + 5min]
and write to paper_trade/intraday_bars/<TICKER>/<DATE>.parquet.

Idempotent: skip existing files unless --force.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

WORK = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-"
    "zachgladstone@gmail.com/My Drive/AI-Tools/s&p500-ticker-mastery"
)
PAPER_DIR = WORK / "paper_trade"
BARS_DIR = PAPER_DIR / "intraday_bars"
STATE_DIR = PAPER_DIR / "state"
ENV_FILE = "/Users/orginal/.config/auto_signup/alpaca.env"


def load_env() -> None:
    if not os.path.exists(ENV_FILE):
        print(f"[warn] env file missing: {ENV_FILE}", file=sys.stderr)
        return
    for line in open(ENV_FILE):
        line = line.strip()
        if line.startswith("export ") and "=" in line:
            k, v = line[7:].split("=", 1)
            os.environ.setdefault(k, v.strip('"').strip("'"))


def tickers_from_state(date_str: str) -> list[str]:
    p = STATE_DIR / f"{date_str}_state.json"
    if not p.exists():
        print(f"[warn] no state file at {p}", file=sys.stderr)
        return []
    d = json.loads(p.read_text())
    out: set[str] = set()
    pos = d.get("positions")
    if isinstance(pos, dict):
        out.update(pos.keys())
    elif isinstance(pos, list):
        for x in pos:
            if isinstance(x, dict) and "ticker" in x:
                out.add(x["ticker"])
            elif isinstance(x, str):
                out.add(x)
    for t in d.get("closed_trades") or []:
        if isinstance(t, dict) and "ticker" in t:
            out.add(t["ticker"])
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"),
                    help="Trading date YYYY-MM-DD (default: today)")
    ap.add_argument("--tickers", default="",
                    help="Comma-separated tickers (default: today's state positions+closed)")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing parquet files")
    args = ap.parse_args()

    load_env()

    try:
        import pandas as pd  # noqa: F401
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from alpaca.data.enums import DataFeed
    except ImportError as e:
        print(f"[warn] missing deps ({e}); exiting 0 to not break orchestrator", file=sys.stderr)
        return 0

    api_key = os.environ.get("ALPACA_API_KEY") or os.environ.get("ALPACA_PAPER_API_KEY")
    api_sec = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("ALPACA_PAPER_SECRET_KEY")
    if not api_key or not api_sec:
        print("[warn] Alpaca creds not in env; exit 0", file=sys.stderr)
        return 0

    tickers = (
        [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        if args.tickers
        else tickers_from_state(args.date)
    )
    if not tickers:
        print("[warn] no tickers to process; exit 0", file=sys.stderr)
        return 0

    # ET window: 09:15 .. 16:05 ET = 13:15 .. 20:05 UTC (EDT). Use UTC-aware.
    day = datetime.strptime(args.date, "%Y-%m-%d")
    start = day.replace(hour=13, minute=15, tzinfo=timezone.utc) - timedelta(hours=0)
    end = day.replace(hour=20, minute=5, tzinfo=timezone.utc)

    client = StockHistoricalDataClient(api_key, api_sec)
    BARS_DIR.mkdir(parents=True, exist_ok=True)

    wrote = skipped = errored = 0
    for t in tickers:
        outdir = BARS_DIR / t
        outdir.mkdir(parents=True, exist_ok=True)
        outp = outdir / f"{args.date}.parquet"
        if outp.exists() and not args.force:
            skipped += 1
            continue
        try:
            req = StockBarsRequest(
                symbol_or_symbols=[t],
                timeframe=TimeFrame.Minute,
                start=start,
                end=end,
                feed=DataFeed.SIP,
            )
            bars = client.get_stock_bars(req)
            df = bars.df  # MultiIndex (symbol, timestamp)
            if df is None or df.empty:
                print(f"[warn] {t}: no bars returned", file=sys.stderr)
                errored += 1
                continue
            # Flatten: drop symbol level, reset index to column 't'
            if "symbol" in df.index.names:
                df = df.xs(t, level="symbol")
            df = df.reset_index().rename(columns={"timestamp": "t"})
            keep = [c for c in ("t", "open", "high", "low", "close", "volume", "vwap", "trade_count") if c in df.columns]
            df = df[keep].rename(columns={"open": "o", "high": "h", "low": "l", "close": "c", "volume": "v"})
            df.to_parquet(outp, index=False)
            wrote += 1
        except Exception as e:
            print(f"[error] {t}: {e}", file=sys.stderr)
            errored += 1

    print(f"[ok] wrote={wrote} skipped={skipped} errored={errored} date={args.date}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
