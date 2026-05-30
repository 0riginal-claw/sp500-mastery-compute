"""
Capacity-Floor Cohort OHLC Backfill — Task #84 step B

Fetches 5-year DAILY bars from Alpaca for the 200 cohort tickers and saves
them as single-file parquets under
``/Volumes/ZG-2TB/zg/cache/alpaca_5yr_capacity_floor/<TICKER>.parquet``.

Schema matches alpaca_5yr cache:
  timestamp (datetime64[ns, UTC]) | open | high | low | close | volume |
  trade_count | vwap   (all float64)

Run:
    python -m lab.capacity_floor_ohlc_fetch
"""
from __future__ import annotations

import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Same lab path setup
DRIVE_BASE = "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive"
LAB_BASE = f"{DRIVE_BASE}/AI-Tools/s&p500-ticker-mastery"
sys.path.insert(0, f"{LAB_BASE}/scripts")

from lab.capacity_floor_cohort import alpaca_credentials  # type: ignore

COHORT_CSV = Path(f"{LAB_BASE}/data/capacity_floor_cohort_2026-05-29.csv")
CACHE_DIR = Path("/Volumes/ZG-2TB/zg/cache/alpaca_5yr_capacity_floor")
STATUS_PATH = Path("/Volumes/ZG-2TB/zg/tmp/champ_003c/ohlc_status.json")


def load_cohort_tickers() -> List[str]:
    rows: List[str] = []
    with open(COHORT_CSV, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row["ticker"])
    return rows


def fetch_one_ticker(ticker: str, key: str, secret: str) -> Dict:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = StockHistoricalDataClient(api_key=key, secret_key=secret)
    end = datetime.now(timezone.utc) - timedelta(minutes=20)  # SIP non-realtime offset
    start = end - timedelta(days=5 * 365 + 5)  # ~5y + buffer

    out_path = CACHE_DIR / f"{ticker}.parquet"
    if out_path.exists():
        try:
            df = pd.read_parquet(out_path)
            if len(df) > 100:
                return {"ticker": ticker, "status": "skip_cached", "bars": len(df)}
        except Exception:
            pass

    try:
        req = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            feed="sip",
        )
        bars = client.get_stock_bars(req)
        df = bars.df
        if df is None or df.empty:
            return {"ticker": ticker, "status": "empty", "bars": 0}
        # multi-index (symbol, timestamp) -> reset to flat timestamp column
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
            df = df.drop(columns=[c for c in df.columns if c == "symbol"], errors="ignore")
        else:
            df = df.reset_index()
        # ensure timestamp UTC tz
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"])
            if ts.dt.tz is None:
                df["timestamp"] = ts.dt.tz_localize("UTC")
            else:
                df["timestamp"] = ts.dt.tz_convert("UTC")
        # column order match
        keep_cols = ["timestamp", "open", "high", "low", "close", "volume", "trade_count", "vwap"]
        for c in keep_cols:
            if c not in df.columns:
                df[c] = pd.NA
        df = df[keep_cols].copy()
        for c in ["open", "high", "low", "close", "volume", "trade_count", "vwap"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
        # write atomically
        tmp = out_path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        tmp.rename(out_path)
        return {
            "ticker": ticker,
            "status": "ok",
            "bars": int(len(df)),
            "date_min": str(df["timestamp"].min()) if len(df) else "",
            "date_max": str(df["timestamp"].max()) if len(df) else "",
        }
    except Exception as e:
        return {"ticker": ticker, "status": "error", "bars": 0, "error": str(e)[:200]}


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)

    tickers = load_cohort_tickers()
    print(f"Cohort tickers to fetch: {len(tickers)}")

    key, secret = alpaca_credentials()
    started = time.time()
    results: List[Dict] = []
    by_status: Dict[str, int] = {"ok": 0, "skip_cached": 0, "empty": 0, "error": 0}

    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(fetch_one_ticker, t, key, secret): t for t in tickers}
        completed = 0
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            by_status[r["status"]] = by_status.get(r["status"], 0) + 1
            completed += 1
            if completed % 20 == 0:
                print(f"  progress: {completed}/{len(tickers)} "
                      f"ok={by_status['ok']} cached={by_status['skip_cached']} "
                      f"empty={by_status['empty']} err={by_status['error']}")

    dur = time.time() - started

    summary = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "duration_sec": round(dur, 1),
        "cohort_count": len(tickers),
        "by_status": by_status,
        "results": results,
    }
    STATUS_PATH.write_text(json.dumps(summary, indent=2))

    print()
    print(f"DONE in {dur:.1f}s")
    print(f"  ok       : {by_status['ok']}")
    print(f"  cached   : {by_status['skip_cached']}")
    print(f"  empty    : {by_status['empty']}")
    print(f"  errors   : {by_status['error']}")
    if by_status["error"]:
        print("Sample errors:")
        for r in results:
            if r["status"] == "error":
                print(f"  {r['ticker']}: {r.get('error', '?')}")
                if sum(1 for x in results if x['status'] == 'error') >= 5:
                    break


if __name__ == "__main__":
    main()
