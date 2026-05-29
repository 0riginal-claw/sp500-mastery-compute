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

# Path resolution: post-2026-05-13 storage-tier migration moved the canonical
# govtrades.db off Drive (Ph0tis/Gov-Trades/) onto the 2TB external (/Volumes/ZG-2TB/).
# We walk a fallback chain so older callers / mirror configs still work.
#
# Order:
#   1. $GOVTRADES_DB env var (operator override)
#   2. /Volumes/ZG-2TB/zg/govtrades/data/govtrades.db (canonical post-migration, 334 MB)
#   3. /My Drive/Ph0tis/Gov-Trades/data/govtrades.db  (legacy pre-migration)
#   4. /My Drive/claudes test/govtrades_test/govtrades.db (audit mirror)
#
# The first path that opens cleanly AND contains a `politicians` table wins.
# Result is cached at class-level so we only probe once per process.
GOVTRADES_DB_FALLBACKS = [
    "/Volumes/ZG-2TB/zg/govtrades/data/govtrades.db",
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/"
    "My Drive/Ph0tis/Gov-Trades/data/govtrades.db",
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/"
    "My Drive/claudes test/govtrades_test/govtrades.db",
]

# Back-compat alias — some external callers still import this constant directly.
# Points at the first fallback (canonical post-migration).
GOVTRADES_DB_DRIVE = GOVTRADES_DB_FALLBACKS[0]


def _resolve_govtrades_db_path() -> str:
    """Walk the fallback chain; return first path that opens + has politicians table.

    Raises FileNotFoundError listing every path searched if all fall through.
    """
    candidates = []
    env_override = os.environ.get("GOVTRADES_DB")
    if env_override:
        candidates.append(env_override)
    candidates.extend(GOVTRADES_DB_FALLBACKS)

    errors = []
    for path in candidates:
        if not os.path.exists(path):
            errors.append(f"  - {path}  (missing)")
            continue
        try:
            con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5.0)
            try:
                cur = con.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name='politicians'"
                )
                if cur.fetchone() is None:
                    errors.append(f"  - {path}  (no politicians table)")
                    continue
            finally:
                con.close()
            LOG.info("[govtrades_cache] resolved DB -> %s", path)
            return path
        except sqlite3.Error as e:
            errors.append(f"  - {path}  (sqlite error: {e})")

    raise FileNotFoundError(
        "govtrades.db not found in any known location. Searched (in order):\n"
        + "\n".join(errors)
    )
# NOTE: 2026-05-21 — switched to read-only URI direct over Drive (NO /tmp copy).
# Prior implementation copied the 39 MB DB via sqlite3.backup() which deadlocked
# on Drive-FUSE. Read-only URI avoids any write to the source AND skips the
# copy entirely; FUSE handles the random-access reads without lock contention.
#
# 2026-05-28 — added fallback chain (env override → /Volumes/ZG-2TB → legacy Drive →
# audit mirror) so the wrapper survives the storage-tier migration.


class GovTradesCache:
    # per-ticker DataFrames keyed by (ticker, kind)
    _df_cache: dict = {}
    _db_path: Optional[str] = None  # resolved on first _connect() call

    @classmethod
    def _connect(cls):
        if cls._db_path is None:
            cls._db_path = _resolve_govtrades_db_path()
        return sqlite3.connect(
            f"file:{cls._db_path}?mode=ro", uri=True, timeout=10.0
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

    @classmethod
    def get_offexchange(
        cls,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Off-exchange / dark-pool short-volume series for `ticker`.
        Columns: ticker, date, otc_short, otc_total, dpi (Dark Pool Index =
        otc_short/otc_total). Sourced from QuiverQuant Hobbyist-tier
        live+historical/offexchange endpoints.
        """
        if not ticker:
            return pd.DataFrame()
        key = ("ox", ticker.upper())
        if key not in cls._df_cache:
            sql = (
                "SELECT ticker, date, otc_short, otc_total, dpi "
                "FROM offexchange WHERE ticker = ? ORDER BY date"
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
    def get_flights(
        cls,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Corporate-jet flight tracking for `ticker`. Columns: ticker, date,
        departure_city, arrival_city. Sourced from QuiverQuant live/flights.
        Sparse signal — null cities on many rows; treat as event flag rather
        than continuous series.
        """
        if not ticker:
            return pd.DataFrame()
        key = ("flights", ticker.upper())
        if key not in cls._df_cache:
            sql = (
                "SELECT ticker, date, departure_city, arrival_city "
                "FROM flights WHERE ticker = ? ORDER BY date"
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
    def get_politicians(cls) -> pd.DataFrame:
        """All Congressional politicians with bio_guide_id, party, chamber,
        state, image_url, trade_count, trade_volume, net_worth.
        """
        key = ("politicians", "_all")
        if key not in cls._df_cache:
            sql = (
                "SELECT bio_guide_id, candidate_id, name, party, chamber, "
                "state, image_url, trade_count, trade_volume, net_worth, "
                "last_updated FROM politicians"
            )
            df = cls._query_df(sql, ())
            cls._df_cache[key] = df
        return cls._df_cache[key].copy()

    @classmethod
    def get_corporate_donors(cls, ticker: str) -> pd.DataFrame:
        """Corporate PAC donations to politicians, keyed by ticker.
        Columns: bio_guide_id, ticker, company_cmte_id, committee_name,
        transaction_date, transaction_amount, transaction_type, cycle.
        """
        if not ticker:
            return pd.DataFrame()
        key = ("donors", ticker.upper())
        if key not in cls._df_cache:
            sql = (
                "SELECT bio_guide_id, ticker, company_cmte_id, company_cmte_nm, "
                "committee_name, transaction_date, transaction_amount, "
                "transaction_type, cycle FROM corporate_donors WHERE ticker = ?"
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
    ox = GovTradesCache.get_offexchange(tk, "2024-01-01", "2025-12-31")
    print(f"{tk} offexchange 2024-2025: {len(ox)} rows")
    if not ox.empty:
        print(ox[["date", "otc_short", "dpi"]].head(3))
    fl = GovTradesCache.get_flights(tk, "2024-01-01", "2025-12-31")
    print(f"{tk} flights 2024-2025: {len(fl)} rows")
    pols = GovTradesCache.get_politicians()
    print(f"politicians (all): {len(pols)} rows")
    donors = GovTradesCache.get_corporate_donors(tk)
    print(f"{tk} corporate donors: {len(donors)} rows")
