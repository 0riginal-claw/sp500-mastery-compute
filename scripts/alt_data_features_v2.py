"""
alt_data_features_v2.py
=======================
Extends alt_data_features.py with the missing EDGAR and Gov-Trades feature layers.

New layers (12 additional features):
--------------------------------------
EDGAR v2 (4 features):
  filing_count_def14a_365d   count of DEF 14A proxy filings in trailing 365 days
  filing_count_s1_365d       count of S-1 / S-1/A filings in trailing 365 days
  days_since_last_def14a     calendar days since most recent DEF 14A (9999 if none)
  filing_count_10k_365d      count of 10-K / 10-K/A filings in trailing 365 days

Government contracts (5 features):
  gov_contract_count_90d       count of contract award rows in trailing 90 days
  gov_contract_dollar_log_180d log1p of total contract USD value in trailing 180 days
  gov_contract_dollar_log_90d  log1p of total contract USD value in trailing 90 days
  quarterly_contract_qoq       quarter-over-quarter dollar change (most recent 2 quarters)
  quarterly_contract_dollar_log log1p of most-recent-quarter contract amount (point-in-time)

Lobbying issue signals (3 features):
  lobbying_antitrust_flag    1 if antitrust/labor-antitrust issue filed in trailing 180 days
  lobbying_tax_flag          1 if tax/IRS issue filed in trailing 180 days
  lobbying_trade_flag        1 if trade (domestic/foreign) issue filed in trailing 180 days

Gaps intentionally skipped (data not in local SQLite):
  - 13D/13G activist filings:  NOT in edgar.db (only 8-K/10-K/10-Q/DEF14A/S-1 present).
  - 13F institutional holdings: NOT in edgar.db.
  - Form 4 insider transactions: handled by insider_form4_features.py (SEC API).
  - XBRL fundamentals: not stored in edgar.db; would require live SEC XBRL API calls
    at ~0.12 s/request per ticker — skipped in this batch layer; document here.

Point-in-time safety
--------------------
All windows close on the LEFT of bar t:
  event_date <= t.date()   (future events excluded)
Gov-Contracts uses action_date (the award date, publicly visible).
Quarterly contracts uses year+quarter end date as the release boundary.

Database sources
----------------
  edgar.db     — /claudes test/data/edgar/data/edgar.db
  govtrades.db — /claudes test/data/gov_trades/data/govtrades.db
Both are copied to /tmp at first access (same pattern as alt_data_features.py).
"""

from __future__ import annotations

import os
import re
import shutil
import sqlite3
import time
from datetime import date, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_EDGAR_DB_DRIVE = (
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/claudes test/data/edgar/data/edgar.db"
)
_GOVTRADES_DB_DRIVE = (
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/claudes test/data/gov_trades/data/govtrades.db"
)
_TMP_EDGAR = "/tmp/alt_data_edgar.db"         # shared with alt_data_features.py
_TMP_GOVTRADES = "/tmp/alt_data_govtrades.db"  # shared with alt_data_features.py


# ---------------------------------------------------------------------------
# Module-level caches
# ---------------------------------------------------------------------------
_edgar_v2_cache: Dict[str, pd.DataFrame] = {}
_contracts_awards_cache: Dict[str, pd.DataFrame] = {}
_contracts_quarterly_cache: Dict[str, pd.DataFrame] = {}
_lobbying_issue_cache: Dict[str, pd.DataFrame] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_local_copies() -> None:
    """Copy Drive DBs to /tmp once per process (idempotent)."""
    if not os.path.exists(_TMP_EDGAR):
        shutil.copy2(_EDGAR_DB_DRIVE, _TMP_EDGAR)
    if not os.path.exists(_TMP_GOVTRADES):
        shutil.copy2(_GOVTRADES_DB_DRIVE, _TMP_GOVTRADES)


def _connect(db_path: str, retries: int = 3, delay: float = 2.0) -> sqlite3.Connection:
    for attempt in range(retries):
        try:
            conn = sqlite3.connect(db_path, timeout=30.0, check_same_thread=False)
            conn.execute("SELECT 1")
            return conn
        except sqlite3.OperationalError:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise


def _to_ordinal(dates) -> np.ndarray:
    return np.array([d.toordinal() for d in dates], dtype=np.int64)


def _rolling_count(bar_ords: np.ndarray, ev_ords: np.ndarray, window: int) -> np.ndarray:
    """Count events strictly in (bar - window, bar] for each bar."""
    if len(ev_ords) == 0:
        return np.zeros(len(bar_ords), dtype=np.float64)
    ev_sorted = np.sort(ev_ords)
    out = np.empty(len(bar_ords), dtype=np.float64)
    for i, d in enumerate(bar_ords):
        lo = d - window
        out[i] = float(
            np.searchsorted(ev_sorted, d, side="right")
            - np.searchsorted(ev_sorted, lo, side="right")
        )
    return out


def _days_since_last(bar_ords: np.ndarray, ev_ords: np.ndarray, sentinel: float = 9999.0) -> np.ndarray:
    """Calendar days since the most recent event <= bar. sentinel if none."""
    if len(ev_ords) == 0:
        return np.full(len(bar_ords), sentinel, dtype=np.float64)
    ev_sorted = np.sort(ev_ords)
    out = np.empty(len(bar_ords), dtype=np.float64)
    for i, d in enumerate(bar_ords):
        idx = np.searchsorted(ev_sorted, d, side="right") - 1
        out[i] = float(d - ev_sorted[idx]) if idx >= 0 else sentinel
    return out


def _quarter_end_ordinal(year: int, quarter: int) -> int:
    """Return the ordinal of the last day of the given quarter."""
    month_end = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}
    m, day = month_end[quarter]
    return date(year, m, day).toordinal()


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_edgar_v2(ticker: str) -> pd.DataFrame:
    """
    Returns DataFrame [filed_date (date), form (str)] for the forms
    we use in v2: DEF 14A, S-1, S-1/A, 10-K, 10-K/A.
    Cached per ticker.
    """
    if ticker in _edgar_v2_cache:
        return _edgar_v2_cache[ticker]

    _ensure_local_copies()
    conn = _connect(_TMP_EDGAR)
    try:
        query = """
            SELECT filed_at, form
            FROM filings
            WHERE ticker = ?
              AND form IN ('DEF 14A','S-1','S-1/A','10-K','10-K/A')
        """
        df = pd.read_sql_query(query, conn, params=(ticker,))
    finally:
        conn.close()

    if df.empty:
        df = pd.DataFrame(columns=["filed_date", "form"])
    else:
        df["filed_date"] = pd.to_datetime(df["filed_at"], utc=True).dt.normalize().dt.date
        df = df[["filed_date", "form"]].copy()

    _edgar_v2_cache[ticker] = df
    return df


def _load_contracts_awards(ticker: str) -> pd.DataFrame:
    """
    Returns DataFrame [award_date (date), amount (float)] from gov_contracts_awards.
    Uses action_date as the point-in-time boundary.
    """
    if ticker in _contracts_awards_cache:
        return _contracts_awards_cache[ticker]

    _ensure_local_copies()
    conn = _connect(_TMP_GOVTRADES)
    try:
        query = """
            SELECT COALESCE(action_date, date) AS raw_date, amount
            FROM gov_contracts_awards
            WHERE ticker = ?
        """
        df = pd.read_sql_query(query, conn, params=(ticker,))
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()

    if df.empty:
        df = pd.DataFrame(columns=["award_date", "amount"])
    else:
        df["award_date"] = pd.to_datetime(df["raw_date"], errors="coerce").dt.normalize().dt.date
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
        df = df.dropna(subset=["award_date"])[["award_date", "amount"]].copy()

    _contracts_awards_cache[ticker] = df
    return df


def _load_contracts_quarterly(ticker: str) -> pd.DataFrame:
    """
    Returns DataFrame [year (int), quarter (int), amount (float), quarter_end_ord (int)].
    quarter_end_ord is the ordinal of the last day of the quarter — used as the
    point-in-time release boundary.
    """
    if ticker in _contracts_quarterly_cache:
        return _contracts_quarterly_cache[ticker]

    _ensure_local_copies()
    conn = _connect(_TMP_GOVTRADES)
    try:
        query = """
            SELECT year, quarter, amount
            FROM gov_contracts_quarterly
            WHERE ticker = ?
            ORDER BY year, quarter
        """
        df = pd.read_sql_query(query, conn, params=(ticker,))
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()

    if df.empty:
        df = pd.DataFrame(columns=["year", "quarter", "amount", "quarter_end_ord"])
    else:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
        df["quarter_end_ord"] = df.apply(
            lambda r: _quarter_end_ordinal(int(r["year"]), int(r["quarter"])), axis=1
        )
        df = df.sort_values("quarter_end_ord").reset_index(drop=True)

    _contracts_quarterly_cache[ticker] = df
    return df


def _load_lobbying_issues(ticker: str) -> pd.DataFrame:
    """
    Returns DataFrame [lob_date (date), issue (str)].
    """
    if ticker in _lobbying_issue_cache:
        return _lobbying_issue_cache[ticker]

    _ensure_local_copies()
    conn = _connect(_TMP_GOVTRADES)
    try:
        query = """
            SELECT date, issue
            FROM lobbying
            WHERE ticker = ?
        """
        df = pd.read_sql_query(query, conn, params=(ticker,))
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()

    if df.empty:
        df = pd.DataFrame(columns=["lob_date", "issue"])
    else:
        df["lob_date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize().dt.date
        df = df.dropna(subset=["lob_date"])[["lob_date", "issue"]].copy()

    _lobbying_issue_cache[ticker] = df
    return df


# ---------------------------------------------------------------------------
# Feature functions
# ---------------------------------------------------------------------------

def add_edgar_v2_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Adds 4 EDGAR v2 features to a daily-indexed DataFrame.

    Columns added:
      filing_count_def14a_365d  count of DEF 14A filings in trailing 365 days
      filing_count_s1_365d      count of S-1/S-1/A filings in trailing 365 days
      days_since_last_def14a    days since most recent DEF 14A (9999 if none)
      filing_count_10k_365d     count of 10-K/10-K/A filings in trailing 365 days

    Point-in-time: uses filed_at (SEC receipt date).
    Data available: DEF 14A 3289 rows, S-1/S-1/A 48 rows, 10-K/10-K/A 3423 rows.
    MISSING from edgar.db: 13D, 13G, 13F — these form types are not stored; skip.
    """
    df = df.copy()
    filings = _load_edgar_v2(ticker)

    bar_dates = df.index.normalize().date
    bar_ords = _to_ordinal(bar_dates)

    if filings.empty:
        df["filing_count_def14a_365d"] = 0.0
        df["filing_count_s1_365d"] = 0.0
        df["days_since_last_def14a"] = 9999.0
        df["filing_count_10k_365d"] = 0.0
        return df

    def_14a = filings[filings["form"] == "DEF 14A"]["filed_date"].values
    s1 = filings[filings["form"].isin(["S-1", "S-1/A"])]["filed_date"].values
    k10 = filings[filings["form"].isin(["10-K", "10-K/A"])]["filed_date"].values

    def_14a_ords = _to_ordinal(def_14a) if len(def_14a) else np.array([], dtype=np.int64)
    s1_ords = _to_ordinal(s1) if len(s1) else np.array([], dtype=np.int64)
    k10_ords = _to_ordinal(k10) if len(k10) else np.array([], dtype=np.int64)

    df["filing_count_def14a_365d"] = _rolling_count(bar_ords, def_14a_ords, 365)
    df["filing_count_s1_365d"] = _rolling_count(bar_ords, s1_ords, 365)
    df["days_since_last_def14a"] = _days_since_last(bar_ords, def_14a_ords)
    df["filing_count_10k_365d"] = _rolling_count(bar_ords, k10_ords, 365)

    return df


def add_gov_contracts_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Adds 5 government-contract features to a daily-indexed DataFrame.

    Columns added:
      gov_contract_count_90d        count of contract award rows, trailing 90 days
      gov_contract_dollar_log_180d  log1p(total USD), trailing 180 days
      gov_contract_dollar_log_90d   log1p(total USD), trailing 90 days
      quarterly_contract_qoq        QoQ dollar change (most recent two visible quarters)
      quarterly_contract_dollar_log log1p(most recent visible quarter USD)

    Point-in-time:
      awards       — uses action_date (public award announcement)
      quarterly    — visible from quarter_end day onwards (conservative)

    Data available: gov_contracts_awards 60K rows (2025-11 to 2026-04),
                    gov_contracts_quarterly 17K rows (2008-Q1 to 2026-Q3).
    """
    df = df.copy()
    awards = _load_contracts_awards(ticker)
    quarterly = _load_contracts_quarterly(ticker)

    bar_dates = df.index.normalize().date
    bar_ords = _to_ordinal(bar_dates)
    n = len(bar_ords)

    # --- awards features ---
    if awards.empty:
        df["gov_contract_count_90d"] = 0.0
        df["gov_contract_dollar_log_180d"] = 0.0
        df["gov_contract_dollar_log_90d"] = 0.0
    else:
        aw_ords = _to_ordinal(awards["award_date"].values)
        aw_amounts = awards["amount"].values

        cnt_90 = np.zeros(n, dtype=np.float64)
        dol_180 = np.zeros(n, dtype=np.float64)
        dol_90 = np.zeros(n, dtype=np.float64)

        for i, d in enumerate(bar_ords):
            m90 = (aw_ords > d - 90) & (aw_ords <= d)
            m180 = (aw_ords > d - 180) & (aw_ords <= d)
            cnt_90[i] = float(m90.sum())
            dol_180[i] = float(np.log1p(aw_amounts[m180].sum()))
            dol_90[i] = float(np.log1p(aw_amounts[m90].sum()))

        df["gov_contract_count_90d"] = cnt_90
        df["gov_contract_dollar_log_180d"] = dol_180
        df["gov_contract_dollar_log_90d"] = dol_90

    # --- quarterly features ---
    if quarterly.empty or len(quarterly) < 2:
        df["quarterly_contract_qoq"] = 0.0
        df["quarterly_contract_dollar_log"] = 0.0
    else:
        qtr_ords = quarterly["quarter_end_ord"].values
        qtr_amounts = quarterly["amount"].values

        qoq_arr = np.zeros(n, dtype=np.float64)
        qtr_log_arr = np.zeros(n, dtype=np.float64)

        for i, d in enumerate(bar_ords):
            # Most recent quarter whose end date <= bar date
            visible = qtr_ords <= d
            if visible.sum() == 0:
                continue
            last_idx = int(np.where(visible)[0].max())
            qtr_log_arr[i] = float(np.log1p(qtr_amounts[last_idx]))
            if last_idx >= 1:
                qoq_arr[i] = float(qtr_amounts[last_idx] - qtr_amounts[last_idx - 1])

        df["quarterly_contract_qoq"] = qoq_arr
        df["quarterly_contract_dollar_log"] = qtr_log_arr

    return df


# Regex patterns for lobbying issue classification
_RE_ANTITRUST = re.compile(r"antitrust|labor\s*issues", re.I)
_RE_TAX = re.compile(r"taxation|internal\s*revenue|tax", re.I)
_RE_TRADE = re.compile(r"trade\s*\(domestic|foreign\)|trade\s*domestic|foreign\s*trade", re.I)


def add_lobbying_issue_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Adds 3 lobbying issue signal features to a daily-indexed DataFrame.

    Columns added:
      lobbying_antitrust_flag   1 if antitrust/labor-antitrust issue in trailing 180 days
      lobbying_tax_flag         1 if tax/IRS issue in trailing 180 days
      lobbying_trade_flag       1 if trade (domestic/foreign) issue in trailing 180 days

    Issue strings in govtrades.db can be multi-line (newline-separated topics per row).
    We match any token in the concatenated issue text.
    Point-in-time: uses lobbying.date (disclosure/report date).
    """
    df = df.copy()
    lob = _load_lobbying_issues(ticker)

    bar_dates = df.index.normalize().date
    bar_ords = _to_ordinal(bar_dates)
    n = len(bar_ords)

    if lob.empty:
        df["lobbying_antitrust_flag"] = 0.0
        df["lobbying_tax_flag"] = 0.0
        df["lobbying_trade_flag"] = 0.0
        return df

    lob_ords = _to_ordinal(lob["lob_date"].values)
    lob_issues = lob["issue"].fillna("").values

    # Pre-classify rows
    is_antitrust = np.array([bool(_RE_ANTITRUST.search(s)) for s in lob_issues])
    is_tax = np.array([bool(_RE_TAX.search(s)) for s in lob_issues])
    is_trade = np.array([bool(_RE_TRADE.search(s)) for s in lob_issues])

    anti_flag = np.zeros(n, dtype=np.float64)
    tax_flag = np.zeros(n, dtype=np.float64)
    trade_flag = np.zeros(n, dtype=np.float64)

    for i, d in enumerate(bar_ords):
        m180 = (lob_ords > d - 180) & (lob_ords <= d)
        anti_flag[i] = 1.0 if is_antitrust[m180].any() else 0.0
        tax_flag[i] = 1.0 if is_tax[m180].any() else 0.0
        trade_flag[i] = 1.0 if is_trade[m180].any() else 0.0

    df["lobbying_antitrust_flag"] = anti_flag
    df["lobbying_tax_flag"] = tax_flag
    df["lobbying_trade_flag"] = trade_flag
    return df


def add_all_v2_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Convenience wrapper: applies all three v2 feature functions in order.

    Returns DataFrame enriched with all 12 new columns:
      EDGAR v2 (4):         filing_count_def14a_365d, filing_count_s1_365d,
                            days_since_last_def14a, filing_count_10k_365d
      Gov contracts (5):    gov_contract_count_90d, gov_contract_dollar_log_180d,
                            gov_contract_dollar_log_90d, quarterly_contract_qoq,
                            quarterly_contract_dollar_log
      Lobbying issues (3):  lobbying_antitrust_flag, lobbying_tax_flag,
                            lobbying_trade_flag
    """
    df = add_edgar_v2_features(df, ticker)
    df = add_gov_contracts_features(df, ticker)
    df = add_lobbying_issue_features(df, ticker)
    return df


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from datetime import datetime, timezone

    test_dates = pd.date_range("2024-01-01", "2025-12-31", freq="B", tz="UTC")
    base_df = pd.DataFrame({"close": 100.0}, index=test_dates)

    results = {}
    for tk in ["AAPL", "NVDA", "XOM"]:
        print(f"\n=== {tk} ===")
        enriched = add_all_v2_features(base_df.copy(), tk)
        new_cols = [c for c in enriched.columns if c not in base_df.columns]
        results[tk] = new_cols
        print(f"  +{len(new_cols)} features: {new_cols}")
        for c in new_cols:
            col = enriched[c]
            nz = int((col != 0).sum())
            print(f"    {c}: {nz} non-zero rows ({nz/len(enriched)*100:.1f}%)")

    total = len(results.get("AAPL", []))
    print(f"\nSmoke test complete. +{total} features per ticker.")
