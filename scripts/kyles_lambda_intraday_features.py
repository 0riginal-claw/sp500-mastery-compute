"""
kyles_lambda_intraday_features.py
==================================
Kyle's Lambda (intraday approximation) features for the XGBoost pipeline.
Reference: Kyle (1985) "Continuous Auctions and Insider Trading", Econometrica.
License: MIT (own implementation; reference paper is academic/public-domain).

Kyle's Lambda (λ) quantifies market impact: the price change per unit of signed
order flow.  A higher λ indicates a less liquid, more information-sensitive
market.  Estimated here from daily OHLCV bars using Bulk Volume Classification
(BVC) as the signed-order-flow proxy — consistent with the vpin_50bucket module's
approach of approximating alpaca_1min_bars via BVC (López de Prado/O'Hara 2012).

NO-LOOKAHEAD AUDIT
------------------
Every step that references a same-bar quantity is preceded by .shift(1) before
being used in the regression, ensuring that bar-t features encode only
information available through end-of-day t-1.

Concretely:
  1. delta_price_lagged  = close.shift(1) - close.shift(2)   # yesterday's ΔP
  2. signed_vol_lagged   = (volume * sign(close - open)).shift(1)  # yesterday's Q
  3. lambda_series is computed from rolling regressions over {lagged} series;
     the lambda value at index t reflects bars [t-window, t-1].
  4. Rolling stats (mean, std) used for z-score are themselves computed over
     the already-lagged lambda_series — no additional shift needed.
  5. The trend feature uses rolling_5d vs rolling_20d of lambda_series, computed
     the same way.

Result: all three output columns at row t are determined entirely by data from
bars up to and including t-1.  No same-bar or future data is used.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Feature names exported for use in backtest_xgb_v10.py
KYLES_LAMBDA_FEATURE_NAMES: list[str] = [
    "kyles_lambda",         # rolling-20d median of daily lambda estimates (lagged)
    "kyles_lambda_z21",     # z-score of kyles_lambda over a 21-bar rolling window
    "kyles_lambda_trend",   # sign of (5d_avg - 20d_avg) of lambda: +1 / 0 / -1
]


def compute_kyles_lambda_intraday_features(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
    window: int = 20,
) -> pd.DataFrame:
    """Append Kyle's Lambda (intraday BVC approximation) features to df.

    Parameters
    ----------
    df : pd.DataFrame
        Daily feature DataFrame already built by the v9/v10 stack.
        Must contain columns: close, open, volume (or their lowercase variants).
    ticker : str, optional
        Ticker symbol; used only for log messages.
    window : int
        Rolling window (trading days) for the lambda regression and smoothing.

    Returns
    -------
    pd.DataFrame
        df with three new columns appended (see KYLES_LAMBDA_FEATURE_NAMES).
        Missing / insufficient data is zero-filled; existing columns are NOT
        overwritten (idempotent guard).
    """
    label = ticker or "?"

    # Idempotent: if already computed, skip
    if all(c in df.columns for c in KYLES_LAMBDA_FEATURE_NAMES):
        logger.debug("[kyles_lambda] columns already present for %s — skipping", label)
        return df

    # Locate price and volume columns (handle both title-case and lower-case)
    close_col = _find_col(df, ["close", "Close", "adj_close", "Adj Close"])
    open_col  = _find_col(df, ["open", "Open"])
    vol_col   = _find_col(df, ["volume", "Volume"])

    if close_col is None or open_col is None or vol_col is None:
        logger.warning(
            "[kyles_lambda] required columns (close/open/volume) not found for %s — zeroing",
            label,
        )
        return _zero_fill(df)

    close  = df[close_col].astype(float)
    open_  = df[open_col].astype(float)
    volume = df[vol_col].astype(float).clip(lower=0)

    # ---- BVC signed-volume proxy (same bar) ----
    # sign(close - open): +1 = net buying pressure, -1 = net selling pressure
    direction = np.sign(close - open_).replace(0, np.nan).ffill().fillna(1)
    signed_vol = volume * direction  # units: shares with sign

    # ---- NO-LOOKAHEAD: shift both price-change and signed-vol by 1 ----
    # delta_price at t  → lagged so value at index t uses bar t-1 close delta
    delta_price_lagged = close.diff().shift(1)     # ΔP_{t-1}
    signed_vol_lagged  = signed_vol.shift(1)        # Q_{t-1}

    # ---- Daily Kyle's Lambda via rolling OLS ----
    # λ_t = Cov(ΔP, Q) / Var(Q)  over [t-window, t-1]
    # Computed entirely from lagged series — no same-bar data.
    lambda_series = _rolling_kyle_lambda(
        delta_price=delta_price_lagged,
        signed_vol=signed_vol_lagged,
        window=window,
    )

    # ---- Smoothed lambda (20d rolling median of already-lagged lambda) ----
    kyles_lambda = lambda_series.rolling(window=window, min_periods=max(5, window // 4)).median()

    # ---- Z-score over 21 bars ----
    roll_mean = kyles_lambda.rolling(window=21, min_periods=5).mean()
    roll_std  = kyles_lambda.rolling(window=21, min_periods=5).std().replace(0, np.nan)
    kyles_lambda_z21 = ((kyles_lambda - roll_mean) / roll_std).fillna(0.0)

    # ---- Trend: sign of (5d_avg - 20d_avg) ----
    avg_5  = lambda_series.rolling(window=5,  min_periods=2).mean()
    avg_20 = lambda_series.rolling(window=20, min_periods=5).mean()
    kyles_lambda_trend = np.sign(avg_5 - avg_20).fillna(0.0)

    # ---- Assign to df ----
    # All series are derived from lagged data — no additional shift required.
    df = df.copy()
    df["kyles_lambda"]       = kyles_lambda.fillna(0.0)
    df["kyles_lambda_z21"]   = kyles_lambda_z21
    df["kyles_lambda_trend"] = kyles_lambda_trend

    n_nonzero = (df["kyles_lambda"] != 0).sum()
    logger.info(
        "[kyles_lambda] %s: added 3 features; %d/%d non-zero rows",
        label, n_nonzero, len(df),
    )
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Return the first candidate column name present in df, or None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _rolling_kyle_lambda(
    delta_price: pd.Series,
    signed_vol: pd.Series,
    window: int,
) -> pd.Series:
    """Compute rolling Kyle's Lambda via the OLS formula: Cov(ΔP,Q)/Var(Q).

    Returns a Series of the same length as inputs; NaN where insufficient data.
    """
    min_periods = max(5, window // 4)
    result = pd.Series(np.nan, index=delta_price.index)

    dp = delta_price.values.astype(float)
    sv = signed_vol.values.astype(float)
    n  = len(dp)

    for i in range(window - 1, n):
        lo = i - window + 1
        dp_w = dp[lo : i + 1]
        sv_w = sv[lo : i + 1]

        # Drop NaN pairs
        mask = ~(np.isnan(dp_w) | np.isnan(sv_w))
        if mask.sum() < min_periods:
            continue

        dp_w = dp_w[mask]
        sv_w = sv_w[mask]

        var_q = np.var(sv_w, ddof=1)
        if var_q < 1e-12:
            result.iat[i] = 0.0
            continue

        cov_dp_q = np.cov(dp_w, sv_w, ddof=1)[0, 1]
        result.iat[i] = cov_dp_q / var_q

    return result


def _zero_fill(df: pd.DataFrame) -> pd.DataFrame:
    """Fill all Kyle's Lambda feature columns with 0.0 (graceful degradation)."""
    df = df.copy()
    for col in KYLES_LAMBDA_FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0.0
    return df
