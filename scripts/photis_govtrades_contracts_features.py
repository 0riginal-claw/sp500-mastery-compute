"""
photis_govtrades_contracts_features.py — Government contract award features (v10 gap-fill).

Reads govtrades.db tables:
  - gov_contracts_awards(ticker, action_date, agency, description, amount)
  - gov_contracts_quarterly(ticker, year, quarter, amount)

Features added (prefix gt_contracts_*):
  - gt_contracts_ttm_usd        : sum of award amounts in trailing 365 days (as-of safe)
  - gt_contracts_award_count_30d: count of contract awards in trailing 30 days
  - gt_contracts_qoq_growth     : (current_quarter / prev_quarter) - 1 from quarterly table

.shift(1) safety: only rows with action_date < bar_date are included.
Graceful failure: if DB missing / ticker absent → zero-filled columns returned.
Idempotent: re-calling on already-augmented df is a no-op.
"""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GOVTRADES_DB_DRIVE = (
    "/Users/orginal/Library/CloudStorage/"
    "GoogleDrive-zachgladstone@gmail.com/My Drive/"
    "Ph0tis/Gov-Trades/data/govtrades.db"
)
GOVTRADES_DB_LOCAL = "/tmp/govtrades_contracts_features.db"

FEATURE_NAMES: list[str] = [
    "gt_contracts_ttm_usd",
    "gt_contracts_award_count_30d",
    "gt_contracts_qoq_growth",
]

_contracts_cache: dict[str, pd.DataFrame] = {}
_quarterly_cache: dict[str, pd.DataFrame] = {}


def _ensure_local_db() -> bool:
    if os.path.exists(GOVTRADES_DB_LOCAL):
        try:
            with sqlite3.connect(GOVTRADES_DB_LOCAL, timeout=10.0) as con:
                con.execute("SELECT COUNT(*) FROM gov_contracts_awards").fetchone()
            return True
        except Exception:
            try:
                os.remove(GOVTRADES_DB_LOCAL)
            except OSError:
                pass

    if not os.path.exists(GOVTRADES_DB_DRIVE):
        logger.warning("[gt_contracts] source DB missing: %s", GOVTRADES_DB_DRIVE)
        return False

    try:
        src = sqlite3.connect(f"file:{GOVTRADES_DB_DRIVE}?mode=ro", uri=True, timeout=30.0)
        dst = sqlite3.connect(GOVTRADES_DB_LOCAL)
        with dst:
            src.backup(dst)
        src.close()
        dst.close()
        return True
    except Exception as e:
        logger.warning("[gt_contracts] sqlite-backup failed: %s — trying shutil", e)

    try:
        shutil.copy2(GOVTRADES_DB_DRIVE, GOVTRADES_DB_LOCAL)
        for sfx in ("-wal", "-shm"):
            side = GOVTRADES_DB_DRIVE + sfx
            if os.path.exists(side):
                shutil.copy2(side, GOVTRADES_DB_LOCAL + sfx)
        with sqlite3.connect(GOVTRADES_DB_LOCAL, timeout=10.0) as con:
            con.execute("SELECT COUNT(*) FROM gov_contracts_awards").fetchone()
        return True
    except Exception as e:
        logger.warning("[gt_contracts] all copy strategies failed: %s", e)
        return False


def _load_contracts(ticker: str) -> pd.DataFrame:
    if ticker in _contracts_cache:
        return _contracts_cache[ticker]
    if not _ensure_local_db():
        _contracts_cache[ticker] = pd.DataFrame()
        return _contracts_cache[ticker]
    try:
        with sqlite3.connect(GOVTRADES_DB_LOCAL, timeout=10.0) as con:
            df = pd.read_sql_query(
                "SELECT action_date, amount FROM gov_contracts_awards WHERE ticker = ?",
                con,
                params=(ticker,),
                parse_dates=["action_date"],
            )
        df = df.dropna(subset=["action_date"]).sort_values("action_date")
        df["action_date"] = pd.to_datetime(df["action_date"], utc=False).dt.normalize()
        _contracts_cache[ticker] = df
    except Exception as e:
        logger.warning("[gt_contracts] load failed for %s: %s", ticker, e)
        _contracts_cache[ticker] = pd.DataFrame()
    return _contracts_cache[ticker]


def _load_quarterly(ticker: str) -> pd.DataFrame:
    if ticker in _quarterly_cache:
        return _quarterly_cache[ticker]
    if not _ensure_local_db():
        _quarterly_cache[ticker] = pd.DataFrame()
        return _quarterly_cache[ticker]
    try:
        with sqlite3.connect(GOVTRADES_DB_LOCAL, timeout=10.0) as con:
            df = pd.read_sql_query(
                "SELECT year, quarter, amount FROM gov_contracts_quarterly WHERE ticker = ?",
                con,
                params=(ticker,),
            )
        df = df.sort_values(["year", "quarter"])
        df["q_idx"] = df["year"] * 4 + df["quarter"]
        df["qoq_growth"] = df["amount"].pct_change().replace([np.inf, -np.inf], np.nan)
        _quarterly_cache[ticker] = df
    except Exception as e:
        logger.warning("[gt_contracts] quarterly load failed for %s: %s", ticker, e)
        _quarterly_cache[ticker] = pd.DataFrame()
    return _quarterly_cache[ticker]


def add_photis_govtrades_contracts_features(
    df: pd.DataFrame,
    ticker: str | None = None,
    csv_path: str | None = None,
) -> pd.DataFrame:
    """Add government contract award features to df.

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

    raw = _load_contracts(ticker)
    qdf = _load_quarterly(ticker)

    if df.empty:
        return df.assign(**zeros)

    bar_dates = df.index.normalize()
    if bar_dates.tz is not None:
        bar_dates = bar_dates.tz_localize(None)

    ttm_usd = np.zeros(len(df), dtype=np.float64)
    count_30d = np.zeros(len(df), dtype=np.float64)
    qoq_growth = np.full(len(df), np.nan, dtype=np.float64)

    if not raw.empty:
        dates_ord = raw["action_date"].values.astype("datetime64[D]").astype(np.int64)
        amounts = raw["amount"].fillna(0.0).values.astype(np.float64)

        for i, bar_d in enumerate(bar_dates):
            cutoff = np.datetime64(bar_d, "D").astype(np.int64)
            mask = dates_ord < cutoff
            if not mask.any():
                continue
            valid_dates = dates_ord[mask]
            valid_amounts = amounts[mask]
            # TTM = trailing 365 calendar days
            ttm_mask = valid_dates >= (cutoff - 365)
            ttm_usd[i] = valid_amounts[ttm_mask].sum()
            # 30-day count
            c30_mask = valid_dates >= (cutoff - 30)
            count_30d[i] = c30_mask.sum()

    if not qdf.empty:
        # Assign QoQ growth per bar: find which fiscal quarter each bar falls in
        # then look up the previous quarter's growth rate
        for i, bar_d in enumerate(bar_dates):
            yr = bar_d.year
            q = (bar_d.month - 1) // 3 + 1
            q_idx = yr * 4 + q
            # Use previous quarter's growth (as-of safe: current quarter not closed yet)
            prev_rows = qdf[qdf["q_idx"] < q_idx]
            if not prev_rows.empty:
                latest_prev = prev_rows.iloc[-1]
                qoq_growth[i] = latest_prev["qoq_growth"] if pd.notna(latest_prev["qoq_growth"]) else np.nan

    qoq_filled = pd.Series(qoq_growth, index=df.index).fillna(0.0)

    return df.assign(
        gt_contracts_ttm_usd=ttm_usd,
        gt_contracts_award_count_30d=count_30d,
        gt_contracts_qoq_growth=qoq_filled.values,
    )
