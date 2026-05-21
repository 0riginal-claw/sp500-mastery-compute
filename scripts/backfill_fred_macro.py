"""backfill_fred_macro.py — FRED macro indicator backfill.

Pulls 30 key macro indicators from FRED via the public CSV API (no key required).
Writes one parquet per series under cache/fred_macro/{SERIES}.parquet plus
a combined cache/fred_macro/_combined.parquet.

FRED public CSV endpoint:
    https://fred.stlouisfed.org/graph/fredgraph.csv?id=GDP&cosd=YYYY-MM-DD

Free. Rate limit ~120 req/min. Smoke <60s for 30 series.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = _ROOT / "cache" / "fred_macro"
MANIFEST = OUT_DIR / "_manifest.json"

# 30 key macro indicators
FRED_SERIES = {
    # Rates & yields
    "DGS10": "10Y Treasury",
    "DGS2": "2Y Treasury",
    "DGS30": "30Y Treasury",
    "T10Y2Y": "10Y-2Y Spread",
    "DFF": "Fed Funds Rate",
    "DFEDTARU": "Fed Funds Upper Target",
    # Inflation
    "CPIAUCSL": "CPI All Items",
    "CPILFESL": "Core CPI",
    "PCEPI": "PCE Price Index",
    "T10YIE": "10Y Breakeven Inflation",
    # Labor
    "UNRATE": "Unemployment Rate",
    "PAYEMS": "Nonfarm Payrolls",
    "ICSA": "Initial Claims",
    "JTSJOL": "Job Openings",
    # GDP & activity
    "GDP": "GDP",
    "GDPC1": "Real GDP",
    "INDPRO": "Industrial Production",
    "RSAFS": "Retail Sales",
    # Credit & liquidity
    "BAMLH0A0HYM2": "HY OAS",
    "BAA10Y": "Baa Spread",
    "TEDRATE": "TED Spread",
    "WALCL": "Fed Total Assets",
    "M2SL": "M2 Money Supply",
    # Sentiment & housing
    "UMCSENT": "UMich Sentiment",
    "HOUST": "Housing Starts",
    "PERMIT": "Building Permits",
    # FX & commodities (where FRED has them)
    "DTWEXBGS": "USD Broad Index",
    "DCOILWTICO": "WTI Crude",
    "GOLDAMGBD228NLBM": "Gold London Fix",
    "VIXCLS": "VIX Close",
}


def fetch_series(sid: str, start: str, retries: int = 3) -> pd.DataFrame:
    """Fetch via pandas.read_csv directly — bypasses requests TLS issue on macOS."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd={start}"
    last = None
    df = None
    for attempt in range(retries):
        try:
            df = pd.read_csv(url)
            break
        except Exception as e:
            last = e
            time.sleep(2 ** attempt)
    if df is None:
        raise last  # pragma: no cover
    # Schema: observation_date,<SID>   (or DATE,<SID> historically)
    date_col = "observation_date" if "observation_date" in df.columns else df.columns[0]
    df = df.rename(columns={date_col: "date", sid: "value"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date"]).reset_index(drop=True)
    df["series"] = sid
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    start = (datetime.utcnow().date() - timedelta(days=365 * args.years + 10)).isoformat()
    series = list(FRED_SERIES.items())
    if args.smoke:
        series = series[:5]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    manifest = {}
    combined_frames = []
    for sid, desc in series:
        try:
            df = fetch_series(sid, start)
            p = OUT_DIR / f"{sid}.parquet"
            df.to_parquet(p, index=False)
            manifest[sid] = {
                "desc": desc,
                "rows": len(df),
                "start": str(df["date"].iloc[0])[:10] if len(df) else None,
                "end": str(df["date"].iloc[-1])[:10] if len(df) else None,
                "path": str(p),
            }
            combined_frames.append(df)
            print(f"  [fred] {sid}: {len(df)} rows ({desc})")
            time.sleep(0.5)  # politeness
        except Exception as e:
            manifest[sid] = {"desc": desc, "error": str(e)}
            print(f"  [fred] {sid}: ERROR {e}")

    if combined_frames:
        combined = pd.concat(combined_frames, ignore_index=True)
        combined.to_parquet(OUT_DIR / "_combined.parquet", index=False)

    elapsed = time.time() - t0
    summary = {
        "source": "fred",
        "ts": datetime.utcnow().isoformat() + "Z",
        "series_attempted": len(series),
        "series_succeeded": sum(1 for v in manifest.values() if "rows" in v),
        "rows_total": sum(v.get("rows", 0) for v in manifest.values()),
        "elapsed_sec": round(elapsed, 1),
        "per_series": manifest,
    }
    with open(MANIFEST, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[fred] DONE: {summary['series_succeeded']}/{summary['series_attempted']} OK, {summary['rows_total']} rows, {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
