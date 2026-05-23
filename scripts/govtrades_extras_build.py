#!/usr/bin/env python3
"""govtrades_extras_build.py — Per-ticker parquet builder for gov-trades extras.

# model_reason: mechanical SQLite -> parquet ETL; haiku-class work
# autosolve_skip: small focused build script

Reads Ph0tis Gov-Trades SQLite (39 MB) and emits per-ticker daily parquet files
under `data/govtrades_extras/<TICKER>.parquet` with 6 base columns:

  insider_buy_30d       — count of congress 'Purchase' transactions, 30-day rolling
  insider_sell_30d      — count of congress 'Sale' transactions, 30-day rolling
  congress_buy_30d      — USD min-amount of congress 'Purchase' txs, 30-day rolling
  congress_sell_30d     — USD min-amount of congress 'Sale' txs, 30-day rolling
  lobby_spend_30d       — USD lobbying spend (per ticker), 30-day rolling
  contract_award_30d    — USD federal contract awards, 30-day rolling

PIT (point-in-time) discipline — CRITICAL
-----------------------------------------
STOCK Act allows congress 45-day lag between trade and disclosure. Keying off
`transaction_date` would create LOOKAHEAD BIAS — we'd "know" about trades that
weren't publicly visible yet.

Mitigations (belt + braces):
  1. Key the rolling windows off `report_date` (congress) / `date` (lobbying,
     contracts) — the FILING/DISCLOSURE date, not the trade/action date.
  2. Apply `.shift(1)` to every output column so day-T row only sees data
     filed STRICTLY before day-T (no same-day leakage).

Per a575b384 + abbfc459 + a5aaf218 consensus.

Usage
-----
  python3 scripts/govtrades_extras_build.py --tickers AAPL --start 2024-01-01 --end 2024-06-01
  python3 scripts/govtrades_extras_build.py --tickers ALL --start 2021-01-01 --end 2026-05-22
  python3 scripts/govtrades_extras_build.py --tickers tickers.txt
"""
from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
from pathlib import Path

import pandas as pd

LOG = logging.getLogger("govtrades_extras_build")

DRIVE = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive"
)
DB_PATH = DRIVE / "Ph0tis/Gov-Trades/data/govtrades.db"
OUT_DIR = DRIVE / "AI-Tools/s&p500-ticker-mastery/data/govtrades_extras"

FEATURE_COLS = [
    "insider_buy_30d",
    "insider_sell_30d",
    "congress_buy_30d",
    "congress_sell_30d",
    "lobby_spend_30d",
    "contract_award_30d",
]


def _connect_ro(db_path: Path) -> sqlite3.Connection:
    """Open SQLite read-only (URI mode)."""
    uri = f"file:{db_path}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _empty_frame(dates: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.DataFrame(index=dates.copy())
    df.index.name = "date"
    for c in FEATURE_COLS:
        df[c] = 0.0
    return df


def _rolling_sum_by_date(
    events: pd.DataFrame, date_col: str, value_col: str, dates: pd.DatetimeIndex
) -> pd.Series:
    """Aggregate `value_col` summed per-day on `date_col`, reindex to `dates`,
    then rolling-30d sum + shift(1) for PIT."""
    if events.empty:
        return pd.Series(0.0, index=dates)
    ev = events.copy()
    ev[date_col] = pd.to_datetime(ev[date_col], errors="coerce")
    ev = ev.dropna(subset=[date_col])
    ev[value_col] = pd.to_numeric(ev[value_col], errors="coerce").fillna(0.0)
    daily = ev.groupby(ev[date_col].dt.normalize())[value_col].sum()
    daily = daily.reindex(dates, fill_value=0.0).astype(float)
    return daily.rolling(window=30, min_periods=1).sum().shift(1).fillna(0.0)


def build_for_ticker(
    conn: sqlite3.Connection, ticker: str, start: str, end: str
) -> pd.DataFrame:
    dates = pd.date_range(start=start, end=end, freq="D")
    out = _empty_frame(dates)

    # ---- congress_trades — key off report_date (PIT-safe disclosure date) ----
    ct = pd.read_sql_query(
        "SELECT report_date, transaction_type, amount_min "
        "FROM congress_trades WHERE ticker = ?",
        conn,
        params=(ticker,),
    )
    buys = ct[ct["transaction_type"].str.contains("Purchase", case=False, na=False)].copy()
    sells = ct[ct["transaction_type"].str.contains("Sale", case=False, na=False)].copy()
    buys["_ones"] = 1.0
    sells["_ones"] = 1.0
    out["insider_buy_30d"] = _rolling_sum_by_date(buys, "report_date", "_ones", dates).values
    out["insider_sell_30d"] = _rolling_sum_by_date(sells, "report_date", "_ones", dates).values
    out["congress_buy_30d"] = _rolling_sum_by_date(buys, "report_date", "amount_min", dates).values
    out["congress_sell_30d"] = _rolling_sum_by_date(sells, "report_date", "amount_min", dates).values

    # ---- lobbying — key off `date` (filing date) ----
    lob = pd.read_sql_query(
        "SELECT date, amount FROM lobbying WHERE ticker = ?", conn, params=(ticker,)
    )
    out["lobby_spend_30d"] = _rolling_sum_by_date(lob, "date", "amount", dates).values

    # ---- gov_contracts_awards — key off `date` (record date) ----
    # action_date is the contract action; `date` is the record date in this DB.
    awards = pd.read_sql_query(
        "SELECT date, amount FROM gov_contracts_awards WHERE ticker = ?",
        conn,
        params=(ticker,),
    )
    out["contract_award_30d"] = _rolling_sum_by_date(awards, "date", "amount", dates).values

    out = out.reset_index().rename(columns={"index": "date"})
    out["ticker"] = ticker
    return out[["date", "ticker", *FEATURE_COLS]]


def resolve_tickers(arg: str, conn: sqlite3.Connection) -> list[str]:
    if arg == "ALL":
        rows = conn.execute(
            "SELECT DISTINCT ticker FROM congress_trades WHERE ticker IS NOT NULL "
            "UNION SELECT DISTINCT ticker FROM lobbying WHERE ticker IS NOT NULL "
            "UNION SELECT DISTINCT ticker FROM gov_contracts_awards WHERE ticker IS NOT NULL"
        ).fetchall()
        return sorted({r[0] for r in rows if r[0]})
    p = Path(arg)
    if p.exists() and p.is_file():
        return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    return [t.strip().upper() for t in arg.split(",") if t.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tickers", required=True, help="ALL | path-to-file | comma list")
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2026-05-22")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if not DB_PATH.exists():
        LOG.error("DB not found at %s", DB_PATH)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = _connect_ro(DB_PATH)
    try:
        tickers = resolve_tickers(args.tickers, conn)
        LOG.info("Building %d tickers -> %s", len(tickers), out_dir)
        n_ok = 0
        for tk in tickers:
            try:
                df = build_for_ticker(conn, tk, args.start, args.end)
                out_path = out_dir / f"{tk}.parquet"
                df.to_parquet(out_path, index=False)
                nonzero = sum(int((df[c] != 0).any()) for c in FEATURE_COLS)
                LOG.info("  %s  rows=%d nonzero_cols=%d/%d  -> %s",
                         tk, len(df), nonzero, len(FEATURE_COLS), out_path.name)
                n_ok += 1
            except Exception as e:
                LOG.warning("  %s FAILED: %s", tk, e)
        LOG.info("Done: %d/%d tickers ok", n_ok, len(tickers))
        return 0 if n_ok else 1
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
