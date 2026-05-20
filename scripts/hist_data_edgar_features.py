"""EDGAR raw-DB feature wrapper.

Source: claudes test/data/edgar/data/edgar.db (SQLite, 57,066 filings,
500 tickers, 2020-01-02 to 2026-04-24).

Emits 9 per-bar features derived from SEC filing recency / density.
.shift(1)-safe via merge_asof(direction='backward', allow_exact_matches=False)
and searchsorted side='left' (strict-prior boundary).

Distinct from `sec_edgar_features.py` (which is a repo-binding stub) — this
module reads the local indexed DB the EDGAR ingest pipeline maintains.
Mirrors the feature semantics of the pre-computed
OC-2/strategy_intelligence_system/edgar/features_by_ticker/<TICKER>_edgar_features.csv
which only existed for AAPL — this wrapper extends to all 500 tickers via
direct SQL.

Wired 2026-05-17 — fills the "raw EDGAR" gap (B1 in the 2026-05-17
historical-data-paths audit; see reports/historical_data_paths_2026-05-17.md).
"""
from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

EDGAR_DB_PATH = (
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/"
    "My Drive/claudes test/data/edgar/data/edgar.db"
)

EDGAR_FEATURE_NAMES = [
    "edgar_days_since_any_filing",
    "edgar_days_since_8k",
    "edgar_days_since_10q",
    "edgar_days_since_10k",
    "edgar_filing_flag_7d",
    "edgar_filing_flag_30d",
    "edgar_eightk_flag_7d",
    "edgar_filings_count_90d",
    "edgar_has_10k_this_year",
]

_CAP_DAYS = 9999
_FILINGS_CACHE: dict[str, pd.DataFrame] = {}


def _zero_fill(df: pd.DataFrame) -> pd.DataFrame:
    for c in EDGAR_FEATURE_NAMES:
        df[c] = 0
    return df


def _load_filings(ticker: str) -> Optional[pd.DataFrame]:
    """Load all filings for a ticker from edgar.db (cached per-process)."""
    if ticker in _FILINGS_CACHE:
        return _FILINGS_CACHE[ticker]
    if not os.path.exists(EDGAR_DB_PATH):
        LOG.warning("[edgar] DB not found at %s", EDGAR_DB_PATH)
        _FILINGS_CACHE[ticker] = pd.DataFrame(columns=["ticker", "form", "filed_at"])
        return _FILINGS_CACHE[ticker]
    try:
        with sqlite3.connect(f"file:{EDGAR_DB_PATH}?mode=ro", uri=True) as conn:
            q = (
                "SELECT ticker, form, filed_at FROM filings "
                "WHERE ticker = ? ORDER BY filed_at"
            )
            f = pd.read_sql_query(q, conn, params=(ticker,))
        if f.empty:
            _FILINGS_CACHE[ticker] = f
            return f
        f["filed_at"] = pd.to_datetime(
            f["filed_at"], utc=True, errors="coerce"
        ).dt.tz_convert(None)
        f = f.dropna(subset=["filed_at"]).reset_index(drop=True)
        _FILINGS_CACHE[ticker] = f
        return f
    except Exception as e:
        LOG.warning("[edgar] DB read failed for %s: %s", ticker, e)
        _FILINGS_CACHE[ticker] = pd.DataFrame(columns=["ticker", "form", "filed_at"])
        return _FILINGS_CACHE[ticker]


def _resolve_dates(df: pd.DataFrame) -> Optional[pd.Series]:
    """Resolve the bar-date series from df['date'] or DatetimeIndex; None on failure."""
    try:
        if "date" in df.columns:
            s = pd.to_datetime(df["date"], errors="coerce")
            if hasattr(s.dt, "tz") and s.dt.tz is not None:
                s = s.dt.tz_convert(None)
            return s
        if isinstance(df.index, pd.DatetimeIndex):
            idx = df.index.tz_convert(None) if df.index.tz is not None else df.index
            return pd.Series(idx, index=df.index)
    except Exception as e:
        LOG.warning("[edgar] date resolution failed: %s", e)
    return None


def add_edgar_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Add 9 EDGAR filing-recency/density features.

    Required input: df['date'] (datetime64) or df with DatetimeIndex.
    Output: df with EDGAR_FEATURE_NAMES columns appended. Always returns df
    (graceful zero-fill on any error). Producer emits past-only values via
    merge_asof(direction='backward', allow_exact_matches=False) and
    searchsorted side='left' — the consumer does NOT need an extra .shift(1).
    """
    df = df.copy()
    try:
        dates = _resolve_dates(df)
        if dates is None or dates.isna().all():
            LOG.warning("[edgar] %s: no usable 'date' col or DatetimeIndex; zero-filling", ticker)
            return _zero_fill(df)

        filings = _load_filings(ticker)
        if filings is None or filings.empty:
            return _zero_fill(df)

        # Build per-form sorted views (numpy datetime64[ns] arrays for searchsorted).
        form_upper = filings["form"].astype(str).str.upper()
        f_any = filings[["filed_at"]].sort_values("filed_at").reset_index(drop=True)
        f_8k = (
            filings.loc[form_upper == "8-K", ["filed_at"]]
            .sort_values("filed_at")
            .reset_index(drop=True)
        )
        f_10q = (
            filings.loc[form_upper.str.startswith("10-Q"), ["filed_at"]]
            .sort_values("filed_at")
            .reset_index(drop=True)
        )
        f_10k = (
            filings.loc[form_upper.str.startswith("10-K"), ["filed_at"]]
            .sort_values("filed_at")
            .reset_index(drop=True)
        )

        bars = (
            pd.DataFrame({"date": dates.values})
            .reset_index(drop=False)
            .rename(columns={"index": "_orig_idx"})
        )
        bars["date"] = pd.to_datetime(bars["date"])
        bars_sorted = bars.sort_values("date").reset_index(drop=True)

        def _days_since(bars_df: pd.DataFrame, filings_df: pd.DataFrame) -> pd.Series:
            if filings_df.empty:
                return pd.Series([_CAP_DAYS] * len(bars_df), dtype="int64")
            m = pd.merge_asof(
                bars_df[["date"]],
                filings_df.rename(columns={"filed_at": "_f"}),
                left_on="date",
                right_on="_f",
                direction="backward",
                allow_exact_matches=False,
            )
            d = (m["date"] - m["_f"]).dt.days
            return d.fillna(_CAP_DAYS).clip(upper=_CAP_DAYS).astype("int64")

        bars_sorted["edgar_days_since_any_filing"] = _days_since(bars_sorted, f_any)
        bars_sorted["edgar_days_since_8k"] = _days_since(bars_sorted, f_8k)
        bars_sorted["edgar_days_since_10q"] = _days_since(bars_sorted, f_10q)
        bars_sorted["edgar_days_since_10k"] = _days_since(bars_sorted, f_10k)
        bars_sorted["edgar_filing_flag_7d"] = (
            bars_sorted["edgar_days_since_any_filing"] <= 7
        ).astype("int8")
        bars_sorted["edgar_filing_flag_30d"] = (
            bars_sorted["edgar_days_since_any_filing"] <= 30
        ).astype("int8")
        bars_sorted["edgar_eightk_flag_7d"] = (
            bars_sorted["edgar_days_since_8k"] <= 7
        ).astype("int8")

        # 90-day count: strict-prior boundary using searchsorted side='left' on both ends.
        any_dates_np = f_any["filed_at"].values.astype("datetime64[ns]") if not f_any.empty else np.array([], dtype="datetime64[ns]")
        tenk_dates_np = f_10k["filed_at"].values.astype("datetime64[ns]") if not f_10k.empty else np.array([], dtype="datetime64[ns]")
        bars_dates_np = bars_sorted["date"].values.astype("datetime64[ns]")
        win_lo = bars_dates_np - np.timedelta64(90, "D")
        if any_dates_np.size > 0:
            hi_idx = np.searchsorted(any_dates_np, bars_dates_np, side="left")
            lo_idx = np.searchsorted(any_dates_np, win_lo, side="left")
            bars_sorted["edgar_filings_count_90d"] = (hi_idx - lo_idx).astype("int32")
        else:
            bars_sorted["edgar_filings_count_90d"] = np.int32(0)

        # YTD: # 10-Ks filed in current calendar year strictly before this bar
        years = pd.DatetimeIndex(bars_sorted["date"]).year.values
        year_starts = np.array(
            [np.datetime64(f"{int(y)}-01-01", "ns") for y in years]
        )
        if tenk_dates_np.size > 0:
            hi_idx_10k = np.searchsorted(tenk_dates_np, bars_dates_np, side="left")
            lo_idx_10k = np.searchsorted(tenk_dates_np, year_starts, side="left")
            bars_sorted["edgar_has_10k_this_year"] = (
                (hi_idx_10k - lo_idx_10k) > 0
            ).astype("int8")
        else:
            bars_sorted["edgar_has_10k_this_year"] = np.int8(0)

        # Restore original order
        bars_back = bars_sorted.sort_values("_orig_idx").reset_index(drop=True)
        for c in EDGAR_FEATURE_NAMES:
            df[c] = bars_back[c].values
        return df
    except Exception as e:
        LOG.warning("[edgar] add_edgar_features failed for %s: %s", ticker, e)
        return _zero_fill(df)


# WIRE_CANDIDATE marker for the consumer auto-wirer
WIRE_CANDIDATE = True
WIRE_MODULE_NAME = "hist_data_edgar_features"
WIRE_IMPORT_LINE = (
    "from hist_data_edgar_features import add_edgar_features, EDGAR_FEATURE_NAMES"
)
WIRE_CALL_LINE = "f = add_edgar_features(f, ticker)"
