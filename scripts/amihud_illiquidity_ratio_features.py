"""
amihud_illiquidity_ratio_features.py
=====================================
Amihud Illiquidity Ratio features for the XGBoost pipeline.
Reference: Amihud, Y. (2002) "Illiquidity and stock returns: cross-section and
time-series effects", Journal of Financial Markets 5(1), 31-56.
License: MIT (own implementation; reference paper is academic/public-domain).

The Amihud Illiquidity Ratio measures the average price impact per dollar of
trading volume: ILLIQ_t = |R_t| / DVOL_t, where R_t = daily return and
DVOL_t = daily dollar volume (close * volume). Higher values indicate a less
liquid stock where trades have greater price impact.

Data source: yfinance_daily_OHLCV (close + volume columns already present in
the v9/v10 feature stack — no additional API keys required).

NO-LOOKAHEAD AUDIT
------------------
All same-bar quantities (close, volume, return) are computed first, then the
entire raw series is shifted by 1 bar before any rolling statistics are
applied, so that bar-t features encode only information available through
end-of-day t-1.

Concretely:
  1. daily_ret      = close.pct_change()              # same-bar return |R_t|
  2. dvol           = close * volume                  # same-bar dollar volume
  3. amihud_raw     = |daily_ret| / dvol.clip(1)      # same-bar ILLIQ_t ratio
  4. amihud_lagged  = amihud_raw.shift(1)             # ** NO-LOOKAHEAD SHIFT **
     — bar-t features now reflect bar t-1 ILLIQ only.
  5. All rolling stats (mean, std, median over windows 5/20/21/63) are
     computed over amihud_lagged; value at index t reflects bars [t-w, t-1].

Result: all five output columns at row t are determined entirely by data from
bars up to and including t-1. No same-bar or future data is used.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

AMIHUD_FEATURE_NAMES: list[str] = [
    "amihud_illiq",        # 20d rolling mean of daily Amihud ratio (lagged)
    "amihud_illiq_z21",    # z-score of amihud_illiq over 21-bar rolling window
    "amihud_illiq_trend",  # sign(5d_avg - 20d_avg) of daily Amihud: +1 / 0 / -1
    "amihud_illiq_ma5",    # 5d rolling mean of daily Amihud ratio (lagged)
    "amihud_illiq_spike",  # 1 if amihud_illiq > 2x its 63d rolling median
]


def compute_amihud_illiquidity_ratio_features(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
    window: int = 20,
) -> pd.DataFrame:
    """Append Amihud Illiquidity Ratio features to df.

    Parameters
    ----------
    df : pd.DataFrame
        Daily feature DataFrame already built by the v9/v10 stack.
        Must contain columns: close (or Close/adj_close) and volume (or Volume).
    ticker : str, optional
        Ticker symbol; used only for log messages.
    window : int
        Rolling window (trading days) for the main Amihud average (default 20).

    Returns
    -------
    pd.DataFrame
        df with five new columns appended (see AMIHUD_FEATURE_NAMES).
        Missing / insufficient data is zero-filled; existing columns are NOT
        overwritten (idempotent guard).
    """
    label = ticker or "?"

    if all(c in df.columns for c in AMIHUD_FEATURE_NAMES):
        logger.debug("[amihud] columns already present for %s — skipping", label)
        return df

    close_col = _find_col(df, ["close", "Close", "adj_close", "Adj Close"])
    vol_col   = _find_col(df, ["volume", "Volume"])

    if close_col is None or vol_col is None:
        logger.warning(
            "[amihud] required columns (close/volume) not found for %s — zeroing", label
        )
        return _zero_fill(df)

    close  = df[close_col].astype(float)
    volume = df[vol_col].astype(float).clip(lower=1)

    # ---- Same-bar Amihud ratio (shifted before use — see NO-LOOKAHEAD AUDIT) ----
    daily_ret  = close.pct_change()
    dvol       = (close * volume).clip(lower=1)
    amihud_raw = daily_ret.abs() / dvol

    # ---- NO-LOOKAHEAD: shift by 1 so bar-t features use only bar t-1 data ----
    amihud_lagged = amihud_raw.shift(1)

    # ---- 20d rolling mean (canonical Amihud illiquidity measure) ----
    amihud_illiq = amihud_lagged.rolling(
        window=window, min_periods=max(5, window // 4)
    ).mean()

    # ---- 5d rolling mean ----
    amihud_illiq_ma5 = amihud_lagged.rolling(window=5, min_periods=2).mean()

    # ---- Z-score over 21 bars ----
    roll_mean = amihud_illiq.rolling(window=21, min_periods=5).mean()
    roll_std  = amihud_illiq.rolling(window=21, min_periods=5).std().replace(0, np.nan)
    amihud_illiq_z21 = ((amihud_illiq - roll_mean) / roll_std).fillna(0.0)

    # ---- Trend: sign of (5d_avg - 20d_avg) of the raw lagged series ----
    avg_5  = amihud_lagged.rolling(window=5,  min_periods=2).mean()
    avg_20 = amihud_lagged.rolling(window=20, min_periods=5).mean()
    amihud_illiq_trend = np.sign(avg_5 - avg_20).fillna(0.0)

    # ---- Spike flag: 1 if amihud_illiq > 2x its 63d rolling median ----
    median_63 = amihud_illiq.rolling(window=63, min_periods=10).median()
    amihud_illiq_spike = (amihud_illiq > 2.0 * median_63).astype(float).fillna(0.0)

    df = df.copy()
    df["amihud_illiq"]       = amihud_illiq.fillna(0.0)
    df["amihud_illiq_z21"]   = amihud_illiq_z21
    df["amihud_illiq_trend"] = amihud_illiq_trend
    df["amihud_illiq_ma5"]   = amihud_illiq_ma5.fillna(0.0)
    df["amihud_illiq_spike"] = amihud_illiq_spike

    n_nonzero = (df["amihud_illiq"] != 0).sum()
    logger.info(
        "[amihud] %s: added 5 features; %d/%d non-zero rows",
        label, n_nonzero, len(df),
    )
    return df


def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _zero_fill(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in AMIHUD_FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0.0
    return df
