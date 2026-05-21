"""backfill_govtrades_5yr.py — Senate eFD + House STOCK Act 5yr backfill.

Pulls congressional trade disclosures using the open mirror at
unitedstates/congress-legislators + community CSV mirrors:
  - https://senatestockwatcher.com/api  (Senate eFD, free, no key)
  - https://housestockwatcher.com/api   (House STOCK Act, free)

These two endpoints provide a JSON dump of all disclosed trades for 5 years.
Writes one parquet per ticker (filtered) and a master parquet of all trades.
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
OUT_DIR = _ROOT / "cache" / "govtrades_extras"
MANIFEST = OUT_DIR / "_manifest.json"

SENATE_API = "https://senatestockwatcher.com/api/v2/transactions"
HOUSE_API = "https://housestockwatcher.com/api/v2/transactions"


def fetch_all(url: str, label: str) -> pd.DataFrame:
    print(f"  [{label}] fetching {url} ...")
    try:
        r = requests.get(url, timeout=120, headers={"User-Agent": "Mozilla/5.0 sp500-mastery/1.0"})
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data if isinstance(data, list) else data.get("transactions", data))
        print(f"  [{label}] got {len(df)} rows, cols={list(df.columns)[:8]}")
        return df
    except Exception as e:
        print(f"  [{label}] ERROR: {e}")
        return pd.DataFrame()


def normalize(df: pd.DataFrame, chamber: str) -> pd.DataFrame:
    """Normalize column names across senate/house feeds."""
    if df.empty:
        return df
    df = df.copy()
    df["chamber"] = chamber
    # Common aliases — keep best-effort
    for src, dst in [
        ("transaction_date", "txn_date"),
        ("date", "txn_date"),
        ("disclosure_date", "disclosure_date"),
        ("ticker", "ticker"),
        ("asset_ticker", "ticker"),
        ("amount", "amount_range"),
        ("type", "txn_type"),
        ("transaction_type", "txn_type"),
        ("senator", "filer"),
        ("representative", "filer"),
    ]:
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]
    keep = [c for c in ("txn_date", "disclosure_date", "ticker", "amount_range", "txn_type", "filer", "chamber") if c in df.columns]
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    if "txn_date" in df.columns:
        df["txn_date"] = pd.to_datetime(df["txn_date"], errors="coerce")
    return df[keep] if keep else df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    senate = normalize(fetch_all(SENATE_API, "senate"), "senate")
    house = normalize(fetch_all(HOUSE_API, "house"), "house")
    full = pd.concat([senate, house], ignore_index=True) if not (senate.empty and house.empty) else pd.DataFrame()

    if full.empty:
        print("[govtrades] no data fetched — endpoints may be down. Skipping.")
        summary = {
            "source": "govtrades",
            "ts": datetime.utcnow().isoformat() + "Z",
            "tickers_succeeded": 0,
            "rows_total": 0,
            "elapsed_sec": round(time.time() - t0, 1),
            "error": "no_data",
        }
        with open(MANIFEST, "w") as f:
            json.dump(summary, f, indent=2)
        return 1

    # Master parquet
    full.to_parquet(OUT_DIR / "_all_trades.parquet", index=False)

    # Per-ticker filter (only sp500)
    with open(TICKERS_PATH) as f:
        sp500 = set(line.strip() for line in f if line.strip())
    if args.smoke:
        sp500 = {"NVDA", "AAPL", "MSFT", "AMZN", "GOOGL"}

    manifest = {}
    for t in sp500:
        if "ticker" in full.columns:
            sub = full[full["ticker"] == t]
            if not sub.empty:
                p = OUT_DIR / f"{t}.parquet"
                sub.to_parquet(p, index=False)
                manifest[t] = {
                    "rows": len(sub),
                    "path": str(p),
                }

    elapsed = time.time() - t0
    summary = {
        "source": "govtrades",
        "ts": datetime.utcnow().isoformat() + "Z",
        "tickers_succeeded": len(manifest),
        "rows_total": sum(v.get("rows", 0) for v in manifest.values()),
        "rows_master": len(full),
        "elapsed_sec": round(elapsed, 1),
        "per_ticker": manifest,
    }
    with open(MANIFEST, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[govtrades] DONE: {len(manifest)} tickers with trades, {summary['rows_total']} rows, {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
