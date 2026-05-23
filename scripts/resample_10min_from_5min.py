"""resample_10min_from_5min.py — Build 10Min Cache B partition from 5Min parquets.

Reads:
    version_3 - Gabriel/Gabriel_Alpaca TimeFrames/Minutes TimeFrames/5Min/<TICKER>/YYYY-MM.parquet
Writes:
    version_3 - Gabriel/Gabriel_Alpaca TimeFrames/Minutes TimeFrames/10Min/<TICKER>/YYYY-MM.parquet

Aggregation (pandas .resample('10min')):
    open=first, high=max, low=min, close=last, volume=sum, trade_count=sum,
    vwap=volume-weighted mean (sum(vwap*volume)/sum(volume))

Conventions:
    closed='left', label='left' — bar timestamp = bar OPEN (matches Alpaca convention
    used elsewhere in Cache B). Drops NaN buckets (non-trading minutes).

Usage:
    # Single ticker, full history:
    python resample_10min_from_5min.py AAPL

    # All 502 tickers (sequential; ~30 min on Mac under load < 12):
    python resample_10min_from_5min.py --all

    # Specific tickers:
    python resample_10min_from_5min.py --tickers BEN META WMT

    # Dry-run (no writes):
    python resample_10min_from_5min.py AAPL --dry-run

2026-05-22: created as part of 10Min TF addition (slice 1).
Source: INTERNET solver iss_1779498988_0690ef4b — canonical pandas OHLCV resample pattern.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger("resample_10min")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DRIVE_ROOT = Path(
    "/Users/orginal/Library/CloudStorage/"
    "GoogleDrive-zachgladstone@gmail.com/My Drive"
)
CACHE_B_ROOT = DRIVE_ROOT / "version_3 - Gabriel" / "Gabriel_Alpaca TimeFrames"
SRC_ROOT = CACHE_B_ROOT / "Minutes TimeFrames" / "5Min"
DST_ROOT = CACHE_B_ROOT / "Minutes TimeFrames" / "10Min"

# Aggregation rules per canonical pandas OHLCV pattern.
OHLCV_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
    "trade_count": "sum",
}


def resample_5m_to_10m(df_5m: pd.DataFrame) -> pd.DataFrame:
    """Resample a 5Min OHLCV DataFrame to 10Min using canonical OHLCV agg.

    Input: DataFrame with columns [timestamp, open, high, low, close, volume,
           trade_count, vwap]. timestamp may be tz-aware or tz-naive.
    Output: same schema, halved row count (approx), timestamps aligned to :00, :10, :20...
    """
    df = df_5m.copy()
    if "timestamp" not in df.columns:
        raise ValueError("expected 'timestamp' column in 5Min input")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]

    # VWAP needs volume-weighted aggregation (not simple mean / max / min).
    # Compute notional first, then sum, then divide by summed volume post-resample.
    has_vwap = "vwap" in df.columns
    if has_vwap:
        df["_notional"] = df["vwap"] * df["volume"]

    agg = {k: v for k, v in OHLCV_AGG.items() if k in df.columns}
    if has_vwap:
        agg["_notional"] = "sum"

    # closed='left', label='left' = bar timestamp is bar OPEN (Alpaca convention).
    out = df.resample("10min", closed="left", label="left").agg(agg)

    if has_vwap:
        # Volume-weighted vwap; protect against div0.
        out["vwap"] = (out["_notional"] / out["volume"]).where(out["volume"] > 0)
        out = out.drop(columns=["_notional"])

    # Drop non-trading buckets (NaN open/high/low/close — no 5Min bar fell in that 10Min slot).
    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out.reset_index()  # restore timestamp column
    return out


def find_5min_parquets(ticker: str) -> list[Path]:
    ticker_dir = SRC_ROOT / ticker
    if not ticker_dir.exists():
        return []
    return sorted(p for p in ticker_dir.glob("*.parquet") if p.stat().st_size > 0)


def process_ticker(ticker: str, dry_run: bool = False, overwrite: bool = False) -> dict:
    """Resample one ticker's full 5Min history to 10Min. Returns stats dict."""
    src_files = find_5min_parquets(ticker)
    if not src_files:
        logger.warning("%s: no 5Min parquets found at %s", ticker, SRC_ROOT / ticker)
        return {"ticker": ticker, "status": "missing_src", "in_rows": 0, "out_rows": 0, "n_files": 0}

    dfs = []
    for p in src_files:
        try:
            dfs.append(pd.read_parquet(p))
        except Exception as e:
            logger.warning("%s: failed reading %s: %s", ticker, p, e)
            continue
    if not dfs:
        return {"ticker": ticker, "status": "all_unreadable", "in_rows": 0, "out_rows": 0, "n_files": 0}

    df_5m = pd.concat(dfs, ignore_index=True)
    df_10m = resample_5m_to_10m(df_5m)
    in_rows, out_rows = len(df_5m), len(df_10m)

    if dry_run:
        logger.info(
            "%s: DRY-RUN in=%d out=%d ratio=%.2f", ticker, in_rows, out_rows,
            (out_rows / in_rows) if in_rows else 0.0,
        )
        return {"ticker": ticker, "status": "dry_run", "in_rows": in_rows, "out_rows": out_rows, "n_files": len(src_files)}

    # Write monthly-partitioned output to mirror Cache B layout.
    dst_dir = DST_ROOT / ticker
    dst_dir.mkdir(parents=True, exist_ok=True)
    df_10m["_yyyymm"] = pd.to_datetime(df_10m["timestamp"], utc=True).dt.strftime("%Y-%m")
    n_written = 0
    for yyyymm, grp in df_10m.groupby("_yyyymm"):
        dst_path = dst_dir / f"{yyyymm}.parquet"
        if dst_path.exists() and not overwrite:
            continue
        grp = grp.drop(columns=["_yyyymm"])
        grp.to_parquet(dst_path, index=False, compression="snappy")
        n_written += 1
    logger.info(
        "%s: in=%d out=%d files_in=%d files_out=%d",
        ticker, in_rows, out_rows, len(src_files), n_written,
    )
    return {
        "ticker": ticker, "status": "ok",
        "in_rows": in_rows, "out_rows": out_rows,
        "n_files_in": len(src_files), "n_files_out": n_written,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker", nargs="?", default=None, help="single ticker (e.g. AAPL)")
    ap.add_argument("--tickers", nargs="+", help="multiple tickers")
    ap.add_argument("--all", action="store_true", help="process all 502 tickers")
    ap.add_argument("--dry-run", action="store_true", help="don't write parquets")
    ap.add_argument("--overwrite", action="store_true", help="overwrite existing 10Min parquets")
    args = ap.parse_args()

    if args.all:
        tickers = sorted(p.name for p in SRC_ROOT.iterdir() if p.is_dir())
    elif args.tickers:
        tickers = args.tickers
    elif args.ticker:
        tickers = [args.ticker]
    else:
        ap.error("specify a ticker, --tickers, or --all")

    logger.info("resample_10min: %d tickers, dry_run=%s, overwrite=%s",
                len(tickers), args.dry_run, args.overwrite)
    results = []
    for i, t in enumerate(tickers, 1):
        try:
            r = process_ticker(t, dry_run=args.dry_run, overwrite=args.overwrite)
            results.append(r)
            if i % 25 == 0:
                logger.info("progress: %d / %d", i, len(tickers))
        except Exception as e:
            logger.exception("%s: unhandled error: %s", t, e)
            results.append({"ticker": t, "status": "error", "err": str(e)})

    ok = sum(1 for r in results if r.get("status") == "ok")
    dry = sum(1 for r in results if r.get("status") == "dry_run")
    err = sum(1 for r in results if r.get("status") in ("error", "missing_src", "all_unreadable"))
    logger.info("done: ok=%d dry=%d err=%d total=%d", ok, dry, err, len(results))
    sys.exit(0 if (ok + dry) == len(results) else 1)


if __name__ == "__main__":
    main()
