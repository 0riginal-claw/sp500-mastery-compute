#!/usr/bin/env python3
"""Persist latest SIP NBBO quote snapshots for a ticker list.

Run mode: 2x per day, called inline from cron OR by orchestrator at
09:29:55 ET (pre_open) + 15:54:55 ET (pre_close).
Appends rows to paper_trade/quotes/<DATE>.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

WORK = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-"
    "zachgladstone@gmail.com/My Drive/AI-Tools/s&p500-ticker-mastery"
)
PAPER_DIR = WORK / "paper_trade"
QUOTES_DIR = PAPER_DIR / "quotes"
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


def _spread_bps(bid: float, ask: float) -> float | None:
    if bid is None or ask is None or bid <= 0 or ask <= 0:
        return None
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return None
    return (ask - bid) / mid * 10000.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tickers", required=True,
                    help="Comma-separated tickers, e.g. AAPL,MSFT,NVDA")
    ap.add_argument("--phase", required=True,
                    choices=["pre_open", "pre_close", "test"],
                    help="Capture phase label")
    args = ap.parse_args()

    load_env()

    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestQuoteRequest
        from alpaca.data.enums import DataFeed
    except ImportError as e:
        print(f"[warn] missing deps ({e}); exit 0", file=sys.stderr)
        return 0

    api_key = os.environ.get("ALPACA_API_KEY") or os.environ.get("ALPACA_PAPER_API_KEY")
    api_sec = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("ALPACA_PAPER_SECRET_KEY")
    if not api_key or not api_sec:
        print("[warn] Alpaca creds not in env; exit 0", file=sys.stderr)
        return 0

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        print("[warn] no tickers; exit 0", file=sys.stderr)
        return 0

    client = StockHistoricalDataClient(api_key, api_sec)
    try:
        req = StockLatestQuoteRequest(symbol_or_symbols=tickers, feed=DataFeed.SIP)
        quotes = client.get_stock_latest_quote(req)
    except Exception as e:
        print(f"[error] alpaca call failed: {e}", file=sys.stderr)
        return 0

    QUOTES_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    outp = QUOTES_DIR / f"{today}.jsonl"
    now_iso = datetime.now(timezone.utc).isoformat()

    n = 0
    with open(outp, "a") as f:
        for ticker, q in (quotes or {}).items():
            bid = getattr(q, "bid_price", None)
            ask = getattr(q, "ask_price", None)
            row = {
                "ticker": ticker,
                "bid": bid,
                "ask": ask,
                "bid_size": getattr(q, "bid_size", None),
                "ask_size": getattr(q, "ask_size", None),
                "spread_bps": _spread_bps(bid, ask),
                "captured_at_utc": now_iso,
                "phase": args.phase,
            }
            f.write(json.dumps(row, default=str) + "\n")
            n += 1

    print(f"[ok] wrote {n} quote rows to {outp} phase={args.phase}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
