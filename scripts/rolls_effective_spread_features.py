"""
rolls_effective_spread_features.py — Roll (1984 JFE) effective-spread estimate
from Alpaca 1-min intraday bars.

No-Lookahead Audit
------------------
  ALL three output columns are shift(1)-safe:
  - rolls_spread_eod:  computed from intraday 1-min returns on day d,
    then assigned to the NEXT calendar row via pd.merge_asof with
    direction="backward" + a strict date offset: the daily-df date t
    receives only data from days < t (strict less-than merge with shift).
    Implementation: build a series indexed by trading_date, then
    df["rolls_spread_eod"] = df.index.normalize().map(rolls_series).shift(1).
  - rolls_spread_z21:  rolling 21-day z-score computed on shifted values.
  - rolls_spread_rel:  daily spread / prior-close (close already a lagged column).
  No current-bar data is ever consumed at prediction time.

Algorithm (Roll 1984 JFE)
--------------------------
  Per trading day d:
    1. Collect 1-min close-price series: [p_0, p_1, ..., p_T]
    2. Compute price changes: Δp_t = p_t − p_{t-1}  (length T)
    3. Serial covariance: γ = Cov(Δp_t, Δp_{t-1})   (requires T ≥ 3)
    4. Roll spread: s_d = 2 * sqrt(max(0, -γ))

  License: MIT (own implementation). Reference: Roll, R. (1984). "A Simple
  Implicit Measure of the Effective Bid-Ask Spread in an Efficient Market."
  Journal of Finance, 39(4), 1127-1139.

Features added (3)
------------------
  rolls_spread_eod   — daily Roll spread estimate (price units; 0 when
                       serial cov is non-negative, i.e., no friction signal)
  rolls_spread_z21   — 21-day z-score of rolls_spread_eod
  rolls_spread_rel   — rolls_spread_eod / prior-close (relative/proportional)
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROLLS_EFFECTIVE_SPREAD_FEATURE_NAMES: list[str] = [
    "rolls_spread_eod",
    "rolls_spread_z21",
    "rolls_spread_rel",
]

_ET_TZ = "America/New_York"

_CACHE_ALPACA = (
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools/s&p500-ticker-mastery/cache/alpaca_features"
)
_CACHE_CLAUDES = (
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/claudes test/data/timeframes/S&P500 5 Year Historical Data"
    "/Minutes TimeFrames/1Min_merged"
)


def _load_1min(ticker: str) -> pd.DataFrame:
    """Load 1-min bars from Alpaca cache or claudes-test fallback."""
    for root, suffix in (
        (_CACHE_ALPACA, f"{ticker}_1min.parquet"),
        (_CACHE_CLAUDES, f"{ticker}.parquet"),
    ):
        path = os.path.join(root, suffix)
        if os.path.exists(path):
            try:
                raw = pd.read_parquet(path)
                if "timestamp" in raw.columns:
                    raw = raw.set_index("timestamp")
                raw = raw.sort_index()
                if raw.index.tz is None:
                    raw.index = raw.index.tz_localize("UTC")
                raw.index = raw.index.tz_convert(_ET_TZ)
                return raw
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"rolls_effective_spread_features: failed to load {path!r}: {exc}"
                )
                continue
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


def _roll_spread_one_day(day_bars: pd.DataFrame) -> float:
    """Return Roll effective spread for one trading day.

    Returns 0.0 if serial covariance is non-negative (no friction signal),
    or NaN if insufficient data.
    """
    close = day_bars["close"].astype(float).values
    if len(close) < 3:
        return np.nan
    dp = np.diff(close)  # length n-1
    if len(dp) < 2:
        return np.nan
    # Serial covariance: Cov(dp[1:], dp[:-1])
    dp_t = dp[1:]
    dp_lag = dp[:-1]
    if len(dp_t) < 2:
        return np.nan
    gamma = float(np.cov(dp_t, dp_lag)[0, 1])
    if not np.isfinite(gamma):
        return np.nan
    # Roll (1984): spread = 2 * sqrt(-gamma) when gamma < 0
    return float(2.0 * np.sqrt(-gamma) if gamma < 0.0 else 0.0)


def compute_rolls_effective_spread_features(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
) -> pd.DataFrame:
    """Add Roll effective spread features to a daily-indexed DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Daily feature DataFrame with DatetimeIndex (tz-aware or naive).
        Must have at least one row. The existing 'close' column is used for
        rolls_spread_rel (falls back to 1.0 if absent).
    ticker : str, optional
        Stock symbol for loading intraday bars.

    Returns
    -------
    pd.DataFrame
        Same df with three new columns appended (no rows removed).
    """
    intraday = _load_1min(ticker) if ticker else pd.DataFrame()
    use_intraday = len(intraday) > 0 and "close" in intraday.columns

    if use_intraday:
        # Group 1-min bars by ET trading date
        dates = intraday.index.normalize().date
        daily_rolls: dict[object, float] = {}
        for d, group in intraday.groupby(dates):
            daily_rolls[d] = _roll_spread_one_day(group)

        # Build a date-keyed Series (Python date objects)
        rolls_series = pd.Series(daily_rolls, dtype=float)
        rolls_series.index = pd.to_datetime(rolls_series.index)

        # Align to df index (tz-normalize to date for matching)
        df_dates = pd.to_datetime(df.index).normalize()
        if df_dates.tz is not None:
            df_dates = df_dates.tz_localize(None)
        rolls_aligned = rolls_series.reindex(df_dates.values).values
    else:
        # Fallback: all-NaN (no intraday data available)
        rolls_aligned = np.full(len(df), np.nan)

    # --- SHIFT(1): assign prior-day spread to current row ---
    rolls_eod_raw = pd.Series(rolls_aligned, index=df.index, dtype=float)
    df["rolls_spread_eod"] = rolls_eod_raw.shift(1)

    # rolls_spread_z21: 21-day rolling z-score on the shifted series
    m21 = df["rolls_spread_eod"].rolling(21, min_periods=10).mean()
    s21 = df["rolls_spread_eod"].rolling(21, min_periods=10).std(ddof=1)
    df["rolls_spread_z21"] = ((df["rolls_spread_eod"] - m21) / s21.clip(lower=1e-10)).clip(-5, 5)

    # rolls_spread_rel: spread / prior-close (close is already a prior-bar column)
    prior_close = df["close"].shift(1) if "close" in df.columns else pd.Series(1.0, index=df.index)
    prior_close = prior_close.replace(0, np.nan)
    df["rolls_spread_rel"] = (df["rolls_spread_eod"] / prior_close).clip(0, 0.1)

    # Fill remaining NaNs with 0 (spread = 0 ≡ no friction signal measured)
    for col in ROLLS_EFFECTIVE_SPREAD_FEATURE_NAMES:
        df[col] = df[col].fillna(0.0)

    return df
