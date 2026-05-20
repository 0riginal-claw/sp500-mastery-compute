#!/usr/bin/env python3
"""Persist full Alpaca account snapshots at key phases of the trading day.

Phases: startup (pre-market boot), open (post 09:30 ET fill window),
flatten (post 15:55 ET liquidation), close (post 16:00 ET final).
Writes paper_trade/account/<DATE>_<phase>.json.
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
ACCT_DIR = PAPER_DIR / "account"
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


def _flatten(account_obj) -> dict:
    """Convert pydantic / alpaca SDK model to a plain JSON-safe dict."""
    # alpaca-py models expose .model_dump() (pydantic v2) or .dict()
    for attr in ("model_dump", "dict"):
        fn = getattr(account_obj, attr, None)
        if callable(fn):
            try:
                d = fn()
                return json.loads(json.dumps(d, default=str))
            except Exception:
                continue
    # Fallback: vars()
    return json.loads(json.dumps(vars(account_obj), default=str))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", required=True,
                    choices=["startup", "open", "flatten", "close"],
                    help="Snapshot phase label")
    args = ap.parse_args()

    load_env()

    try:
        from alpaca.trading.client import TradingClient
    except ImportError as e:
        print(f"[warn] missing deps ({e}); exit 0", file=sys.stderr)
        return 0

    api_key = os.environ.get("ALPACA_API_KEY") or os.environ.get("ALPACA_PAPER_API_KEY")
    api_sec = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("ALPACA_PAPER_SECRET_KEY")
    if not api_key or not api_sec:
        print("[warn] Alpaca creds not in env; exit 0", file=sys.stderr)
        return 0

    try:
        client = TradingClient(api_key, api_sec, paper=True)
        acct = client.get_account()
    except Exception as e:
        print(f"[error] alpaca get_account failed: {e}", file=sys.stderr)
        return 0

    payload = _flatten(acct)
    payload["_captured_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["_phase"] = args.phase

    ACCT_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    outp = ACCT_DIR / f"{today}_{args.phase}.json"
    outp.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[ok] wrote {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
