"""neuralforecast_features_features.py
Decomposition features inspired by the NeuralForecast library
(github:Nixtla/neuralforecast, Apache-2.0).

Computes 5 features based on NeuralForecast's NBEATS/NHITS decomposition
philosophy: trend basis, seasonality Fourier components, and residual
volatility. Implemented in pure numpy/pandas — neuralforecast need not be
installed in the trading venv.

NO-LOOKAHEAD AUDIT (2026-05-17):
  All raw inputs (close) are shifted by 1 bar before any computation.
  At prediction bar T every feature value references data from bars T-1 and
  earlier — no same-bar lookahead.

  nf_trend_slope_21d  : OLS slope over close[T-22..T-2] (21 prior bars).
  nf_trend_slope_63d  : OLS slope over close[T-64..T-2] (63 prior bars).
  nf_fourier_sin_annual: sin(2π × dayofyear[T] / 252) — current bar's
                         calendar position, no future info.
  nf_fourier_cos_annual: cos(2π × dayofyear[T] / 252) — same.
  nf_residual_vol_21d : rolling std of OLS-detrended close over prior 21 bars,
                         divided by rolling mean close for scale-invariance.

  Consumer (build_v10_features) does NOT need an additional .shift(1) because
  all price/volume inputs are already lagged inside this module.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

NEURALFORECAST_FEATURE_COUNT = 5
NEURALFORECAST_FEATURE_NAMES: list[str] = [
    "nf_trend_slope_21d",    # normalised 21-bar OLS slope of lag-1 close
    "nf_trend_slope_63d",    # normalised 63-bar OLS slope of lag-1 close
    "nf_fourier_sin_annual", # sin(2π × dayofyear / 252) — annual cycle
    "nf_fourier_cos_annual", # cos(2π × dayofyear / 252) — annual cycle
    "nf_residual_vol_21d",   # detrended rolling-21 vol / rolling-21 mean close
]


def compute_neuralforecast_features_features(
    df: pd.DataFrame,
    ticker: str | None = None,
) -> pd.DataFrame:
    """Add 5 NeuralForecast-inspired decomposition features to *df*.

    All features reference at most bar T-1 price data, shift(1)-safe.

    Args:
        df: DataFrame indexed by date (DatetimeIndex preferred) or integer.
            Must have column ``close``.
        ticker: Unused; kept for API consistency with other feature modules.

    Returns:
        df with NEURALFORECAST_FEATURE_NAMES columns added (zero-filled on error).
    """
    try:
        _check_required_columns(df)
        _add_features(df)
    except Exception as exc:
        logger.warning(
            "[neuralforecast_features] compute failed (%s): %s — zeroing", ticker, exc
        )
        for col in NEURALFORECAST_FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0.0
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_required_columns(df: pd.DataFrame) -> None:
    missing = [c for c in ("close",) if c not in df.columns]
    if missing:
        raise ValueError(f"neuralforecast_features: missing columns {missing}")


def _rolling_ols_slope(series: pd.Series, window: int) -> pd.Series:
    """Return per-row OLS slope over the preceding *window* values, normalised
    by the rolling mean to make the slope scale-invariant (units: fraction per bar).
    """
    x = np.arange(window, dtype=float)
    x -= x.mean()  # centre for numerical stability
    x_ss = (x * x).sum()

    slopes = np.full(len(series), np.nan)
    vals = series.to_numpy(dtype=float)
    for i in range(window - 1, len(vals)):
        y = vals[i - window + 1 : i + 1]
        if np.any(np.isnan(y)):
            continue
        slope = (x * (y - y.mean())).sum() / x_ss
        mean_y = y.mean()
        slopes[i] = slope / mean_y if mean_y != 0 else 0.0
    return pd.Series(slopes, index=series.index)


def _rolling_residual_vol(series: pd.Series, window: int) -> pd.Series:
    """Return per-row std of OLS residuals over *window* bars, normalised by
    the rolling mean (same scale-invariance as slope).
    """
    x = np.arange(window, dtype=float)
    x -= x.mean()
    x_ss = (x * x).sum()

    rvols = np.full(len(series), np.nan)
    vals = series.to_numpy(dtype=float)
    for i in range(window - 1, len(vals)):
        y = vals[i - window + 1 : i + 1]
        if np.any(np.isnan(y)):
            continue
        slope = (x * (y - y.mean())).sum() / x_ss
        intercept = y.mean() - slope * x.mean()
        residuals = y - (slope * x + intercept)
        mean_y = y.mean()
        rvols[i] = residuals.std() / mean_y if mean_y != 0 else 0.0
    return pd.Series(rvols, index=series.index)


def _add_features(df: pd.DataFrame) -> None:
    # Lag close by 1 bar — strict no-lookahead guarantee
    close_l1 = df["close"].shift(1)

    # ---- Trend slopes (NBEATS-style polynomial basis) ----
    df["nf_trend_slope_21d"] = _rolling_ols_slope(close_l1, window=21)
    df["nf_trend_slope_63d"] = _rolling_ols_slope(close_l1, window=63)

    # ---- Fourier seasonality components (annual cycle, 252 trading days) ----
    if isinstance(df.index, pd.DatetimeIndex):
        doy = df.index.dayofyear.astype(float)
    else:
        # Fallback: treat sequential position modulo 252 as proxy for day-of-year
        doy = (np.arange(len(df), dtype=float) % 252) + 1

    two_pi_t = 2.0 * np.pi * doy / 252.0
    df["nf_fourier_sin_annual"] = np.sin(two_pi_t)
    df["nf_fourier_cos_annual"] = np.cos(two_pi_t)

    # ---- Residual volatility (NHITS-style residual block signal) ----
    df["nf_residual_vol_21d"] = _rolling_residual_vol(close_l1, window=21)
