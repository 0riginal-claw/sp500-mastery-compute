"""mlforecast_features_features.py
Lag-transform features inspired by the mlforecast library
(github:Nixtla/mlforecast, Apache-2.0).

Computes 11 rolling/expanding/EWM lag features on EOD OHLCV bars.
Design follows mlforecast's LagTransforms API (RollingMean, RollingStd,
ExpandingMean, ExponentiallyWeightedMean) but implemented in pandas so
mlforecast need not be installed in the trading venv.

NO-LOOKAHEAD AUDIT (2026-05-17):
  All raw inputs (close, high, low, volume) are shifted by 1 bar before
  computing any rolling window. At prediction bar T, every feature value
  references data from bars T-1 and earlier — no same-bar lookahead.
  pct_change() on the shifted close = (close[T-1] - close[T-2]) / close[T-2],
  also strictly backward-looking.
  Rolling / expanding / EWM operations on the shifted series are causal by
  construction (pandas defaults: no min_periods lookahead, no center=True).
  Consumer (build_v10_features) does NOT need an additional .shift(1) because
  all features are already lagged at the module level.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MLFORECAST_FEATURE_COUNT = 11
MLFORECAST_FEATURE_NAMES: list[str] = [
    "mlf_close_roll5_mean",       # 5-bar rolling mean of lag-1 close
    "mlf_close_roll5_std",        # 5-bar rolling std of lag-1 close
    "mlf_close_roll21_mean",      # 21-bar rolling mean of lag-1 close
    "mlf_returns_roll5_mean",     # 5-bar rolling mean of lag-1 daily returns
    "mlf_returns_roll5_std",      # 5-bar rolling std of lag-1 daily returns
    "mlf_volume_roll5_mean",      # 5-bar rolling mean of lag-1 volume
    "mlf_volume_roll5_std",       # 5-bar rolling std of lag-1 volume
    "mlf_close_ewm_alpha02",      # EWM(alpha=0.2) of lag-1 close (≈ mlforecast ExponentiallyWeightedMean)
    "mlf_returns_expanding_mean", # expanding mean of lag-1 returns (≈ mlforecast ExpandingMean)
    "mlf_hl_range_roll5_mean",    # 5-bar rolling mean of lag-1 daily HL range
    "mlf_close_roll21_max_ratio", # 21-bar max of lag-1 close / lag-1 close (distance from 21d high)
]


def compute_mlforecast_features_features(
    df: pd.DataFrame,
    ticker: str | None = None,
) -> pd.DataFrame:
    """Add 11 mlforecast-style lag features to *df* (in-place column additions).

    All features reference at most bar T-1 data, shift(1)-safe.

    Args:
        df: DataFrame indexed by date (or integer), must have columns
            ``close``, ``high``, ``low``, ``volume``.
        ticker: Unused; kept for API consistency with other feature modules.

    Returns:
        df with MLFORECAST_FEATURE_NAMES columns added (zero-filled on error).
    """
    try:
        _check_required_columns(df)
        _add_features(df)
    except Exception as exc:
        logger.warning("[mlforecast_features] compute failed (%s): %s — zeroing", ticker, exc)
        for col in MLFORECAST_FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0.0
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_required_columns(df: pd.DataFrame) -> None:
    missing = [c for c in ("close", "high", "low", "volume") if c not in df.columns]
    if missing:
        raise ValueError(f"mlforecast_features: missing columns {missing}")


def _add_features(df: pd.DataFrame) -> None:
    # Lag all raw inputs by 1 bar — strict no-lookahead guarantee
    close_l1 = df["close"].shift(1)
    high_l1 = df["high"].shift(1)
    low_l1 = df["low"].shift(1)
    volume_l1 = df["volume"].shift(1).replace(0, np.nan)

    returns_l1 = close_l1.pct_change()          # (close[T-1] - close[T-2]) / close[T-2]
    hl_range_l1 = high_l1 - low_l1              # daily HL range, strictly lagged

    # ---- Rolling mean / std (RollingMean / RollingStd in mlforecast) ----
    df["mlf_close_roll5_mean"] = close_l1.rolling(5, min_periods=2).mean()
    df["mlf_close_roll5_std"] = close_l1.rolling(5, min_periods=2).std()
    df["mlf_close_roll21_mean"] = close_l1.rolling(21, min_periods=5).mean()

    df["mlf_returns_roll5_mean"] = returns_l1.rolling(5, min_periods=2).mean()
    df["mlf_returns_roll5_std"] = returns_l1.rolling(5, min_periods=2).std()

    df["mlf_volume_roll5_mean"] = volume_l1.rolling(5, min_periods=2).mean()
    df["mlf_volume_roll5_std"] = volume_l1.rolling(5, min_periods=2).std()

    # ---- EWM (ExponentiallyWeightedMean in mlforecast) ----
    df["mlf_close_ewm_alpha02"] = close_l1.ewm(alpha=0.2, adjust=False).mean()

    # ---- Expanding mean (ExpandingMean in mlforecast) ----
    df["mlf_returns_expanding_mean"] = returns_l1.expanding(min_periods=5).mean()

    # ---- HL range rolling mean ----
    df["mlf_hl_range_roll5_mean"] = hl_range_l1.rolling(5, min_periods=2).mean()

    # ---- 21-bar rolling max ratio (distance from recent high) ----
    roll21_max = close_l1.rolling(21, min_periods=5).max()
    df["mlf_close_roll21_max_ratio"] = (roll21_max / close_l1.replace(0, np.nan)).fillna(1.0)
