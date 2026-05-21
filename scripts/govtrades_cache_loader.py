# autosolve_skip: data loader infra (read-only SQLite mirror of govtrades.db)
"""
govtrades_cache_loader.py — Unified Congress/lobbying/contracts cache loader.

Source: Ph0tis/Gov-Trades/data/govtrades.db (39 MB SQLite, 8 tables).
Tables used:
  - congress_trades   55,740 rows  2012-2026  (representative, ticker, ...)
  - lobbying          20,080 rows  1999-2026  (date, amount, client, ticker)
  - gov_contracts_awards          (action_date, ticker, amount, agency)
  - gov_contracts_quarterly       (ticker, year, quarter, amount)

Public API (all `.shift(1)`-safe — caller provides bar dates, loader returns
event rows that the feature code filters strictly to date < bar_date):

  GovTradesCache.get_congress_trades(ticker, start=None, end=None) -> pd.DataFrame
  GovTradesCache.get_lobbying(ticker, start=None, end=None)        -> pd.DataFrame
  GovTradesCache.get_contracts(ticker, start=None, end=None)       -> pd.DataFrame
  GovTradesCache.get_contracts_quarterly(ticker)                   -> pd.DataFrame

Idempotent process-local cache. WAL-safe via sqlite3.backup(). Read-only.
"""
from __future__ import annotations

import logging
import os
import sqlite3
from typing import Optional

import pandas as pd

LOG = logging.getLogger(__name__)

GOVTRADES_DB_DRIVE = (
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/"
    "My Drive/Ph0tis/Gov-Trades/data/govtrades.db"
)
# NOTE: 2026-05-21 — switched to read-only URI direct over Drive (NO /tmp copy).
# Prior implementation copied the 39 MB DB via sqlite3.backup() which deadlocked
# on Drive-FUSE. Read-only URI avoids any write to the source AND skips the
# copy entirely; FUSE handles the random-access reads without lock contention.


class GovTradesCache:
    # per-ticker DataFrames keyed by (ticker, kind)
    _df_cache: dict = {}

    @classmethod
    def _connect(cls):
        if not os.path.exists(GOVTRADES_DB_DRIVE):
            raise FileNotFoundError(f"govtrades.db missing at {GOVTRADES_DB_DRIVE}")
        return sqlite3.connect(
            f"file:{GOVTRADES_DB_DRIVE}?mode=ro", uri=True, timeout=10.0
        )

    @classmethod
    def _query_df(cls, sql: str, params: tuple, parse_dates: Optional[list] = None) -> pd.DataFrame:
        try:
            with cls._connect() as con:
                df = pd.read_sql_query(sql, con, params=params, parse_dates=parse_dates)
            return df
        except Exception as e:
            LOG.warning("[govtrades_cache] query failed: %s", e)
            return pd.DataFrame()

    @classmethod
    def get_congress_trades(
        cls,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Congress STOCK Act trades for `ticker`. Columns include
        representative, party, transaction_date, transaction_type, amount_min,
        amount_range, excess_return, source.
        """
        if not ticker:
            return pd.DataFrame()
        key = ("ct", ticker.upper())
        if key not in cls._df_cache:
            sql = (
                "SELECT representative, bio_guide_id, chamber, party, "
                "ticker, transaction_type, transaction_date, report_date, "
                "amount_range, amount_min, excess_return, source "
                "FROM congress_trades WHERE ticker = ? "
                "ORDER BY transaction_date"
            )
            df = cls._query_df(sql, (ticker.upper(),), parse_dates=["transaction_date", "report_date"])
            cls._df_cache[key] = df
        df = cls._df_cache[key]
        if df.empty:
            return df
        if start:
            df = df[df["transaction_date"] >= pd.to_datetime(start)]
        if end:
            df = df[df["transaction_date"] <= pd.to_datetime(end)]
        return df.reset_index(drop=True)

    @classmethod
    def get_lobbying(
        cls,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Lobbying records for `ticker`. Columns: date, amount, client,
        issue, specific_issue, registrant.
        """
        if not ticker:
            return pd.DataFrame()
        key = ("lob", ticker.upper())
        if key not in cls._df_cache:
            sql = (
                "SELECT date, amount, client, issue, specific_issue, "
                "registrant, ticker FROM lobbying WHERE ticker = ? "
                "ORDER BY date"
            )
            df = cls._query_df(sql, (ticker.upper(),), parse_dates=["date"])
            cls._df_cache[key] = df
        df = cls._df_cache[key]
        if df.empty:
            return df
        if start:
            df = df[df["date"] >= pd.to_datetime(start)]
        if end:
            df = df[df["date"] <= pd.to_datetime(end)]
        return df.reset_index(drop=True)

    @classmethod
    def get_contracts(
        cls,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Gov-contract award rows: action_date, agency, description, amount."""
        if not ticker:
            return pd.DataFrame()
        key = ("contract", ticker.upper())
        if key not in cls._df_cache:
            sql = (
                "SELECT ticker, date, action_date, agency, description, amount "
                "FROM gov_contracts_awards WHERE ticker = ? "
                "ORDER BY action_date"
            )
            df = cls._query_df(sql, (ticker.upper(),), parse_dates=["date", "action_date"])
            cls._df_cache[key] = df
        df = cls._df_cache[key]
        if df.empty:
            return df
        if start:
            df = df[df["action_date"] >= pd.to_datetime(start)]
        if end:
            df = df[df["action_date"] <= pd.to_datetime(end)]
        return df.reset_index(drop=True)

    @classmethod
    def get_contracts_quarterly(cls, ticker: str) -> pd.DataFrame:
        """Quarterly aggregate contract amounts: year, quarter, amount."""
        if not ticker:
            return pd.DataFrame()
        key = ("contract_q", ticker.upper())
        if key not in cls._df_cache:
            sql = (
                "SELECT ticker, year, quarter, amount "
                "FROM gov_contracts_quarterly WHERE ticker = ? "
                "ORDER BY year, quarter"
            )
            df = cls._query_df(sql, (ticker.upper(),))
            cls._df_cache[key] = df
        return cls._df_cache[key].copy()


# Smoke
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    tk = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
    ct = GovTradesCache.get_congress_trades(tk, "2024-01-01", "2025-12-31")
    print(f"{tk} congress 2024-2025: {len(ct)} rows")
    if not ct.empty:
        print(ct[["transaction_date", "representative", "transaction_type", "amount_min"]].head(3))
    lob = GovTradesCache.get_lobbying(tk, "2024-01-01", "2025-12-31")
    print(f"{tk} lobbying 2024-2025: {len(lob)} rows")
    if not lob.empty:
        print(lob[["date", "amount", "client"]].head(3))
    con = GovTradesCache.get_contracts(tk, "2024-01-01", "2025-12-31")
    print(f"{tk} contracts 2024-2025: {len(con)} rows")
    q = GovTradesCache.get_contracts_quarterly(tk)
    print(f"{tk} contracts quarterly (all): {len(q)} rows")
