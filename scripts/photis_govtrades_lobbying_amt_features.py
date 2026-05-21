"""
photis_govtrades_lobbying_amt_features.py — Lobbying DOLLAR AMOUNT features (v10 gap-fill).

v10's govtrades_features.py already produces `lobbying_filing_count_30d` (row count only).
This wrapper adds the dollar dimension that v10 omits.

Reads govtrades.db table:
  - lobbying(ticker, date, amount, ...)

Features added (prefix gt_lob_amt_*):
  - gt_lob_amt_30d_usd      : sum of lobbying spend in trailing 30 days
  - gt_lob_amt_qoq_growth   : QoQ growth in lobbying spend (compare 90-day windows)
  - gt_lob_amt_ttm_usd      : sum of lobbying spend in trailing 365 days

.shift(1) safety: only rows with date < bar_date are included.
Graceful failure: missing DB / ticker absent → zero-filled columns.
Idempotent: re-calling already-augmented df is a no-op.
"""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GOVTRADES_DB_DRIVE = (
    "/Users/orginal/Library/CloudStorage/"
    "GoogleDrive-zachgladstone@gmail.com/My Drive/"
    "Ph0tis/Gov-Trades/data/govtrades.db"
)
# 2026-05-21: NO /tmp copy — read-only URI direct over Drive.
GOVTRADES_DB_LOCAL = GOVTRADES_DB_DRIVE

FEATURE_NAMES: list[str] = [
    "gt_lob_amt_30d_usd",
    "gt_lob_amt_qoq_growth",
    "gt_lob_amt_ttm_usd",
]

_lob_amt_cache: dict[str, pd.DataFrame] = {}


def _ensure_local_db() -> bool:
    """2026-05-21: direct read-only URI over Drive, no /tmp copy."""
    if not os.path.exists(GOVTRADES_DB_DRIVE):
        logger.warning("[gt_lob_amt] source DB missing: %s", GOVTRADES_DB_DRIVE)
        return False
    try:
        with sqlite3.connect(
            f"file:{GOVTRADES_DB_DRIVE}?mode=ro", uri=True, timeout=10.0
        ) as con:
            con.execute("SELECT 1 FROM lobbying LIMIT 1").fetchone()
        return True
    except Exception as e:
        logger.warning("[gt_lob_amt] read-only probe failed: %s", e)
        return False


def _ro_connect():
    return sqlite3.connect(
        f"file:{GOVTRADES_DB_DRIVE}?mode=ro", uri=True, timeout=10.0
    )


def _load_lobbying(ticker: str) -> pd.DataFrame:
    if ticker in _lob_amt_cache:
        return _lob_amt_cache[ticker]
    if not _ensure_local_db():
        _lob_amt_cache[ticker] = pd.DataFrame()
        return _lob_amt_cache[ticker]
    try:
        with _ro_connect() as con:
            df = pd.read_sql_query(
                "SELECT date, amount FROM lobbying WHERE ticker = ?",
                con,
                params=(ticker,),
                parse_dates=["date"],
            )
        df = df.dropna(subset=["date"]).sort_values("date")
        df["date"] = pd.to_datetime(df["date"], utc=False).dt.normalize()
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
        _lob_amt_cache[ticker] = df
    except Exception as e:
        logger.warning("[gt_lob_amt] load failed for %s: %s", ticker, e)
        _lob_amt_cache[ticker] = pd.DataFrame()
    return _lob_amt_cache[ticker]


def add_photis_govtrades_lobbying_amt_features(
    df: pd.DataFrame,
    ticker: str | None = None,
    csv_path: str | None = None,
) -> pd.DataFrame:
    """Add lobbying dollar-amount features to df.

    Parameters
    ----------
    df:       OHLCV DataFrame with DatetimeIndex (UTC or tz-naive daily bars).
    ticker:   Ticker symbol (e.g. "AAPL"). Required.
    csv_path: Unused; kept for API parity with other photis wrappers.

    Returns df with FEATURE_NAMES columns appended.
    """
    # Idempotent check
    if all(c in df.columns for c in FEATURE_NAMES):
        return df

    zeros = {c: 0.0 for c in FEATURE_NAMES}
    if not ticker:
        return df.assign(**zeros)

    raw = _load_lobbying(ticker)

    if df.empty:
        return df.assign(**zeros)

    bar_dates = df.index.normalize()
    if bar_dates.tz is not None:
        bar_dates = bar_dates.tz_localize(None)

    amt_30d = np.zeros(len(df), dtype=np.float64)
    amt_ttm = np.zeros(len(df), dtype=np.float64)
    qoq_growth = np.full(len(df), np.nan, dtype=np.float64)

    if not raw.empty:
        dates_ord = raw["date"].values.astype("datetime64[D]").astype(np.int64)
        amounts = raw["amount"].values.astype(np.float64)

        for i, bar_d in enumerate(bar_dates):
            cutoff = np.datetime64(bar_d, "D").astype(np.int64)
            mask = dates_ord < cutoff
            if not mask.any():
                continue
            valid_dates = dates_ord[mask]
            valid_amounts = amounts[mask]

            # 30-day window
            mask_30 = valid_dates >= (cutoff - 30)
            amt_30d[i] = valid_amounts[mask_30].sum()

            # TTM
            mask_ttm = valid_dates >= (cutoff - 365)
            amt_ttm[i] = valid_amounts[mask_ttm].sum()

            # QoQ: compare current 90-day window vs prior 90-day window
            mask_curr_q = (valid_dates >= (cutoff - 90)) & (valid_dates < cutoff)
            mask_prev_q = (valid_dates >= (cutoff - 180)) & (valid_dates < (cutoff - 90))
            curr_q = valid_amounts[mask_curr_q].sum()
            prev_q = valid_amounts[mask_prev_q].sum()
            if prev_q != 0:
                qoq_growth[i] = (curr_q - prev_q) / abs(prev_q)
            elif curr_q > 0:
                qoq_growth[i] = 1.0  # new lobbying appeared

    qoq_filled = pd.Series(qoq_growth, index=df.index).fillna(0.0)

    return df.assign(
        gt_lob_amt_30d_usd=amt_30d,
        gt_lob_amt_qoq_growth=qoq_filled.values,
        gt_lob_amt_ttm_usd=amt_ttm,
    )
