"""
alt_data_features.py
====================
Joins EDGAR filings, Congressional trade disclosures, and lobbying spend
onto a daily price DataFrame for S&P 500 backtesting.

All features are point-in-time safe:
  - EDGAR  : uses filed_at (the date the filing was received by SEC).
  - GovTrades: uses report_date (the public disclosure date, not transaction_date).
               Disclosure lag is ~59 days; transaction_date would introduce lookahead.
  - Lobbying : uses date (the disclosure / report date).

Rolling windows are closed on the LEFT of bar t, i.e. events with
event_date <= t.date() are included; future events are excluded.

Column units
------------
filing_count_30d         : count  — total filings in trailing 30 calendar days
filing_count_8k_30d      : count  — 8-K filings in trailing 30 calendar days
filing_count_10q_180d    : count  — 10-Q filings in trailing 180 calendar days
days_since_last_10q      : days   — calendar days since most recent 10-Q; 9999 if none
cong_net_buy_60d         : USD    — sum(buy midpoints) - sum(sell midpoints), trailing 60d
cong_n_unique_buyers_90d : count  — distinct congress members with buy trades, trailing 90d
cong_chamber_buy_ratio_30d: ratio — (House buys) / (total buys) in trailing 30d; 0 if no buys
lobbying_spend_30d       : USD    — disclosed lobbying spend, trailing 30 calendar days
lobbying_n_issues_90d    : count  — distinct lobbying issues filed, trailing 90 calendar days
"""

import os
import re
import shutil
import sqlite3
import time
from datetime import timedelta
from functools import lru_cache
from typing import Dict

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths — Drive originals (source of truth; never queried directly at runtime)
# ---------------------------------------------------------------------------
_EDGAR_DB_DRIVE = (
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/claudes test/data/edgar/data/edgar.db"
)
_GOVTRADES_DB_DRIVE = (
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/claudes test/data/gov_trades/data/govtrades.db"
)

# Local /tmp copies — zero Drive WAL contention at query time.
_TMP_EDGAR = "/tmp/alt_data_edgar.db"
_TMP_GOVTRADES = "/tmp/alt_data_govtrades.db"


def _ensure_local_copies() -> None:
    """Copy Drive DBs to /tmp once per process (skipped if already present).

    Drive-backed SQLite files suffer WAL-file races when read from subprocesses
    or concurrent threads, causing 180 s timeouts.  Copying to local disk
    eliminates the I/O indirection; reads then hit the OS page-cache directly.
    """
    if not os.path.exists(_TMP_EDGAR):
        shutil.copy2(_EDGAR_DB_DRIVE, _TMP_EDGAR)
    if not os.path.exists(_TMP_GOVTRADES):
        shutil.copy2(_GOVTRADES_DB_DRIVE, _TMP_GOVTRADES)

# ---------------------------------------------------------------------------
# Module-level caches (dict keyed by ticker; populated once per process run)
# ---------------------------------------------------------------------------
_edgar_cache: Dict[str, pd.DataFrame] = {}
_govtrades_cache: Dict[str, pd.DataFrame] = {}
_lobbying_cache: Dict[str, pd.DataFrame] = {}

# ---------------------------------------------------------------------------
# Amount-range parser for congressional trades
# ---------------------------------------------------------------------------
_DOLLAR_RE = re.compile(r"\$?([\d,]+)")


def _parse_amount_midpoint(amount_range: str) -> float:
    """
    Parse strings like '$1,001 - $15,000' or '$50,001 - $100,000' to their midpoint.
    Returns 0.0 for unparseable values.
    """
    if not isinstance(amount_range, str):
        return 0.0
    nums = _DOLLAR_RE.findall(amount_range)
    if len(nums) >= 2:
        lo = float(nums[0].replace(",", ""))
        hi = float(nums[1].replace(",", ""))
        return (lo + hi) / 2.0
    if len(nums) == 1:
        return float(nums[0].replace(",", ""))
    return 0.0


# ---------------------------------------------------------------------------
# Data loaders (cached per ticker)
# ---------------------------------------------------------------------------

def _connect_with_retry(db_path: str, retries: int = 3, delay: float = 2.0) -> sqlite3.Connection:
    """Open a SQLite connection with retries for transient Drive I/O errors."""
    for attempt in range(retries):
        try:
            conn = sqlite3.connect(db_path, timeout=30.0, check_same_thread=False)
            # Verify connection is usable
            conn.execute("SELECT 1")
            return conn
        except sqlite3.OperationalError:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise


def _load_edgar(ticker: str) -> pd.DataFrame:
    """
    Returns DataFrame with columns [filed_date (date), form (str)].
    Cached in _edgar_cache.
    """
    if ticker in _edgar_cache:
        return _edgar_cache[ticker]

    _ensure_local_copies()
    conn = _connect_with_retry(_TMP_EDGAR)
    try:
        query = """
            SELECT filed_at, form
            FROM filings
            WHERE ticker = ?
        """
        df = pd.read_sql_query(query, conn, params=(ticker,))
    finally:
        conn.close()

    if df.empty:
        df = pd.DataFrame(columns=["filed_date", "form"])
    else:
        df["filed_date"] = pd.to_datetime(df["filed_at"], utc=True).dt.normalize().dt.date
        df = df[["filed_date", "form"]].copy()

    _edgar_cache[ticker] = df
    return df


def _load_govtrades(ticker: str) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
      [report_date (date), transaction_type (str), amount_mid (float), chamber (str), representative (str)].
    Uses report_date for point-in-time correctness.
    Cached in _govtrades_cache.
    """
    if ticker in _govtrades_cache:
        return _govtrades_cache[ticker]

    _ensure_local_copies()
    conn = _connect_with_retry(_TMP_GOVTRADES)
    try:
        query = """
            SELECT report_date, transaction_type, amount_range, chamber, representative
            FROM congress_trades
            WHERE ticker = ?
        """
        df = pd.read_sql_query(query, conn, params=(ticker,))
    except Exception:
        # table may not exist for some builds
        df = pd.DataFrame()
    finally:
        conn.close()

    if df.empty:
        df = pd.DataFrame(columns=["report_date", "transaction_type", "amount_mid", "chamber", "representative"])
    else:
        df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce").dt.normalize().dt.date
        df["amount_mid"] = df["amount_range"].apply(_parse_amount_midpoint)
        df = df.dropna(subset=["report_date"])

    _govtrades_cache[ticker] = df
    return df


def _load_lobbying(ticker: str) -> pd.DataFrame:
    """
    Returns DataFrame with columns [lob_date (date), amount (float), issue (str)].
    Cached in _lobbying_cache.
    """
    if ticker in _lobbying_cache:
        return _lobbying_cache[ticker]

    _ensure_local_copies()
    conn = _connect_with_retry(_TMP_GOVTRADES)
    try:
        query = """
            SELECT date, amount, issue
            FROM lobbying
            WHERE ticker = ?
        """
        df = pd.read_sql_query(query, conn, params=(ticker,))
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()

    if df.empty:
        df = pd.DataFrame(columns=["lob_date", "amount", "issue"])
    else:
        df["lob_date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize().dt.date
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
        df = df.dropna(subset=["lob_date"])
        df = df[["lob_date", "amount", "issue"]].copy()

    _lobbying_cache[ticker] = df
    return df


# ---------------------------------------------------------------------------
# Rolling window helper
# ---------------------------------------------------------------------------

def _rolling_count_in_window(bar_dates, event_dates, window_days: int) -> np.ndarray:
    """
    For each bar date d in bar_dates, count events where
    (d - window_days) < event_date <= d.
    Both bar_dates and event_dates are Python date objects (or date-like).
    Returns numpy array aligned with bar_dates.
    """
    bar_arr = np.array([d.toordinal() for d in bar_dates], dtype=np.int64)
    if len(event_dates) == 0:
        return np.zeros(len(bar_arr), dtype=np.float64)
    ev_arr = np.sort(np.array([d.toordinal() for d in event_dates], dtype=np.int64))

    out = np.empty(len(bar_arr), dtype=np.float64)
    for i, d in enumerate(bar_arr):
        lo = d - window_days
        # searchsorted: count events in (lo, d] i.e. lo < ev <= d
        n = np.searchsorted(ev_arr, d, side="right") - np.searchsorted(ev_arr, lo, side="right")
        out[i] = float(n)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_edgar_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Adds EDGAR filing features to a daily price DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Daily-indexed DataFrame with a tz-aware UTC DatetimeIndex.
    ticker : str
        Ticker symbol (e.g. 'AAPL').

    Returns
    -------
    pd.DataFrame with additional columns:
        filing_count_30d      (count) : all filing types in trailing 30 calendar days
        filing_count_8k_30d   (count) : 8-K filings in trailing 30 calendar days
        filing_count_10q_180d (count) : 10-Q filings in trailing 180 calendar days
        days_since_last_10q   (days)  : calendar days since most recent 10-Q (9999 if none ever)

    All windows are point-in-time safe: event filed_at <= bar date.
    """
    df = df.copy()
    filings = _load_edgar(ticker)

    bar_dates = df.index.normalize().date  # array of Python date objects

    if filings.empty:
        df["filing_count_30d"] = 0.0
        df["filing_count_8k_30d"] = 0.0
        df["filing_count_10q_180d"] = 0.0
        df["days_since_last_10q"] = 9999.0
        return df

    all_dates = filings["filed_date"].values
    dates_8k = filings.loc[filings["form"].str.upper().str.startswith("8-K"), "filed_date"].values
    dates_10q = filings.loc[filings["form"].str.upper().str.startswith("10-Q"), "filed_date"].values

    df["filing_count_30d"] = _rolling_count_in_window(bar_dates, all_dates, 30)
    df["filing_count_8k_30d"] = _rolling_count_in_window(bar_dates, dates_8k, 30)
    df["filing_count_10q_180d"] = _rolling_count_in_window(bar_dates, dates_10q, 180)

    # days_since_last_10q: at bar t, most recent 10-Q with filed_date <= t
    if len(dates_10q) > 0:
        sorted_10q = np.sort(np.array([d.toordinal() for d in dates_10q], dtype=np.int64))
        bar_ord = np.array([d.toordinal() for d in bar_dates], dtype=np.int64)
        days_since = np.empty(len(bar_ord), dtype=np.float64)
        for i, d in enumerate(bar_ord):
            idx = np.searchsorted(sorted_10q, d, side="right") - 1
            if idx < 0:
                days_since[i] = 9999.0
            else:
                days_since[i] = float(d - sorted_10q[idx])
        df["days_since_last_10q"] = days_since
    else:
        df["days_since_last_10q"] = 9999.0

    return df


def add_govtrades_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Adds Congressional trading features to a daily price DataFrame.

    Uses report_date (public disclosure date), NOT transaction_date, to avoid
    lookahead bias. The typical disclosure lag is ~59 days.

    Parameters
    ----------
    df : pd.DataFrame
        Daily-indexed DataFrame with a tz-aware UTC DatetimeIndex.
    ticker : str
        Ticker symbol.

    Returns
    -------
    pd.DataFrame with additional columns:
        cong_net_buy_60d          (USD)   : sum(buy midpoints) - sum(sell midpoints),
                                            trailing 60 calendar days by report_date
        cong_n_unique_buyers_90d  (count) : distinct congress members with buy trades,
                                            trailing 90 calendar days by report_date
        cong_chamber_buy_ratio_30d (ratio): House buys / total buys in trailing 30d;
                                            0.0 if no buy trades in window

    Amount ranges (e.g. '$1,001 - $15,000') are converted to their midpoint.
    """
    df = df.copy()
    trades = _load_govtrades(ticker)

    bar_dates = df.index.normalize().date

    zero_cols = ["cong_net_buy_60d", "cong_n_unique_buyers_90d", "cong_chamber_buy_ratio_30d"]
    if trades.empty:
        for c in zero_cols:
            df[c] = 0.0
        return df

    # Classify buy vs sell
    trades["is_buy"] = trades["transaction_type"].str.lower().str.contains("purchase|buy", na=False)
    trades["is_sell"] = trades["transaction_type"].str.lower().str.contains("sale|sell", na=False)
    trades["is_house"] = trades["chamber"].str.lower().str.contains("house", na=False)

    buys = trades[trades["is_buy"]]
    sells = trades[trades["is_sell"]]

    buy_dates_ord = np.sort(np.array([d.toordinal() for d in buys["report_date"]], dtype=np.int64))
    sell_dates_ord = np.sort(np.array([d.toordinal() for d in sells["report_date"]], dtype=np.int64))

    n = len(bar_dates)
    net_buy_60d = np.zeros(n, dtype=np.float64)
    n_unique_buyers_90d = np.zeros(n, dtype=np.float64)
    chamber_ratio_30d = np.zeros(n, dtype=np.float64)

    bar_ord = np.array([d.toordinal() for d in bar_dates], dtype=np.int64)

    # Pre-compute ordinals once for all trade rows (vectorised)
    buys = buys.copy()
    sells = sells.copy()
    buys["_ord"] = np.array([d.toordinal() for d in buys["report_date"]], dtype=np.int64)
    sells["_ord"] = np.array([d.toordinal() for d in sells["report_date"]], dtype=np.int64)

    buy_ord = buys["_ord"].values
    sell_ord = sells["_ord"].values
    buy_amounts = buys["amount_mid"].values
    sell_amounts = sells["amount_mid"].values
    buy_reps = buys["representative"].values
    buy_is_house = buys["is_house"].values

    for i, d in enumerate(bar_ord):
        # 60-day net buy
        bm60 = (buy_ord > d - 60) & (buy_ord <= d)
        sm60 = (sell_ord > d - 60) & (sell_ord <= d)
        net_buy_60d[i] = buy_amounts[bm60].sum() - sell_amounts[sm60].sum()

        # 90-day unique buyers
        bm90 = (buy_ord > d - 90) & (buy_ord <= d)
        n_unique_buyers_90d[i] = len(set(buy_reps[bm90]))

        # 30-day chamber ratio
        bm30 = (buy_ord > d - 30) & (buy_ord <= d)
        total_30 = bm30.sum()
        if total_30 > 0:
            chamber_ratio_30d[i] = buy_is_house[bm30].sum() / total_30
        else:
            chamber_ratio_30d[i] = 0.0

    df["cong_net_buy_60d"] = net_buy_60d
    df["cong_n_unique_buyers_90d"] = n_unique_buyers_90d
    df["cong_chamber_buy_ratio_30d"] = chamber_ratio_30d

    return df


def add_lobbying_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Adds lobbying disclosure features to a daily price DataFrame.

    Uses lobbying.date as the disclosure/report date (point-in-time safe).

    Parameters
    ----------
    df : pd.DataFrame
        Daily-indexed DataFrame with a tz-aware UTC DatetimeIndex.
    ticker : str
        Ticker symbol.

    Returns
    -------
    pd.DataFrame with additional columns:
        lobbying_spend_30d    (USD)   : sum of disclosed lobbying spend, trailing 30 calendar days
        lobbying_n_issues_90d (count) : distinct lobbying issues filed, trailing 90 calendar days
    """
    df = df.copy()
    lob = _load_lobbying(ticker)

    bar_dates = df.index.normalize().date

    if lob.empty:
        df["lobbying_spend_30d"] = 0.0
        df["lobbying_n_issues_90d"] = 0.0
        return df

    n = len(bar_dates)
    spend_30d = np.zeros(n, dtype=np.float64)
    issues_90d = np.zeros(n, dtype=np.float64)

    bar_ord = np.array([d.toordinal() for d in bar_dates], dtype=np.int64)
    lob_ord = np.array([d.toordinal() for d in lob["lob_date"]], dtype=np.int64)
    lob_amounts = lob["amount"].values
    lob_issues = lob["issue"].values

    for i, d in enumerate(bar_ord):
        mask_30 = (lob_ord > d - 30) & (lob_ord <= d)
        spend_30d[i] = lob_amounts[mask_30].sum()

        mask_90 = (lob_ord > d - 90) & (lob_ord <= d)
        issues_90d[i] = len(set(lob_issues[mask_90]))

    df["lobbying_spend_30d"] = spend_30d
    df["lobbying_n_issues_90d"] = issues_90d

    return df


def add_all_alt_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Convenience wrapper: applies add_edgar_features, add_govtrades_features,
    and add_lobbying_features in order.

    Parameters
    ----------
    df : pd.DataFrame
        Daily-indexed DataFrame with a tz-aware UTC DatetimeIndex.
    ticker : str
        Ticker symbol.

    Returns
    -------
    pd.DataFrame enriched with all 9 alt-data columns:
        filing_count_30d, filing_count_8k_30d, filing_count_10q_180d,
        days_since_last_10q, cong_net_buy_60d, cong_n_unique_buyers_90d,
        cong_chamber_buy_ratio_30d, lobbying_spend_30d, lobbying_n_issues_90d
    """
    df = add_edgar_features(df, ticker)
    df = add_govtrades_features(df, ticker)
    df = add_lobbying_features(df, ticker)
    return df


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pandas as pd
    from datetime import datetime, timezone

    test_dates = pd.date_range("2024-01-01", "2024-12-31", freq="B", tz="UTC")
    df = pd.DataFrame({"close": 100.0}, index=test_dates)

    for tk in ["AAPL", "NVDA", "TSLA"]:
        print(f"\n=== {tk} ===")
        enriched = add_all_alt_features(df.copy(), tk)
        new_cols = [c for c in enriched.columns if c not in df.columns]
        print(f"  added cols: {new_cols}")
        print(f"  non-zero rows per col:")
        for c in new_cols:
            nz = (enriched[c] != 0).sum()
            print(f"    {c}: {nz} non-zero rows ({nz/len(enriched)*100:.1f}%)")
        print(f"  sample (mid-year):")
        print(enriched.loc["2024-06-01":"2024-06-07"][new_cols].to_string())
