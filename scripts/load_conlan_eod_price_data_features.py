"""
load_conlan_eod_price_data_features.py — EOD-style features inspired by
chrisconlan/algorithmic-trading-with-python (MIT License).

Source:    github:chrisconlan/algorithmic-trading-with-python/data/eod
License:   MIT — clean, no Commons Clause, no copyleft.
Requires paid API: NO — pure OHLCV calculation, no external calls at runtime.

NO-LOOKAHEAD AUDIT (2026-05-18)
---------------------------------
All 6 features are derived exclusively from past OHLCV bars already present in
the input DataFrame.  The full indicator series is computed over the backward-
looking rolling window on bar t, then shifted forward by .shift(1) before being
written into the output DataFrame.  The model therefore sees only information
confirmed at bar t-1 — no current-bar data leaks into any feature.

  - rolling(252).max()/mean()/std(): lookback over prior bars → safe.
  - rolling(20/50).mean(): backward-looking moving averages → safe.
  - ewm(span=200): exponential decay over history → safe.
  - .shift(1) applied to ALL 6 output columns (explicit guard in compute fn).
  - No external data fetched at runtime; no intraday feeds; no paid API.

Integration cost: LOW — 6 vectorised pandas passes on OHLCV, ~10 ms/ticker.
Expected lift: ~0% AUC improvement (Wave CEP1 — data-source parity exercise).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature registry
# ---------------------------------------------------------------------------

CONLAN_EOD_FEATURE_NAMES: list[str] = [
    "conlan_eod_pct_below_52w_high",   # (52w_high - close) / 52w_high
    "conlan_eod_mom_6m",               # 6-month normalised momentum (126d)
    "conlan_eod_vol_trend_ratio",      # 20d avg volume / 50d avg volume
    "conlan_eod_close_above_200ma",    # binary: close > 200-day SMA (1.0/0.0)
    "conlan_eod_atr_pct",              # ATR(14) / close — normalised range
    "conlan_eod_dollar_vol_z",         # z-score of dollar volume (63d window)
]

CONLAN_EOD_FEATURE_COUNT: int = len(CONLAN_EOD_FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_shift(s: pd.Series) -> pd.Series:
    return s.shift(1)


def _atr14(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Wilder ATR(14) — vectorised, no TA-Lib."""
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_load_conlan_eod_price_data_features(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
) -> pd.DataFrame:
    """Add 6 Conlan-EOD-inspired features to *df* in-place and return it.

    Input contract:
      - df indexed by timestamp (daily bars).
      - df must contain columns: close, high, low, volume.
      - Columns open/adj_close used if present but not required.

    Output: df with 6 new `conlan_eod_*` columns appended.
    All output columns are .shift(1)-safe (represent bar t-1 values).
    Missing inputs are handled gracefully — zero-filled.
    """
    required = {"close", "high", "low", "volume"}
    missing = required - set(df.columns)
    if missing:
        logger.warning(
            "[conlan_eod] ticker=%s missing cols %s — zeroing all features", ticker, missing
        )
        for col in CONLAN_EOD_FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0.0
        return df

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    # 1. pct_below_52w_high — how far below 52-week high
    high_52w = close.rolling(252, min_periods=20).max()
    pct_below = (high_52w - close) / high_52w.replace(0, np.nan)

    # 2. 6-month momentum (126 trading days), normalised by stdev
    ret_126 = close.pct_change(126)
    std_126 = ret_126.rolling(63, min_periods=10).std().replace(0, np.nan)
    mom_6m = ret_126 / std_126

    # 3. volume trend ratio — 20d / 50d average volume
    vol_ma20 = volume.rolling(20, min_periods=5).mean()
    vol_ma50 = volume.rolling(50, min_periods=10).mean().replace(0, np.nan)
    vol_trend_ratio = vol_ma20 / vol_ma50

    # 4. close above 200-day SMA — binary
    sma200 = close.ewm(span=200, adjust=False, min_periods=40).mean()
    close_above_200 = (close > sma200).astype(float)

    # 5. ATR(14) as fraction of close — normalised daily range proxy
    atr14 = _atr14(high, low, close)
    atr_pct = atr14 / close.replace(0, np.nan)

    # 6. dollar-volume z-score over 63-day window
    dollar_vol = close * volume
    dv_mean = dollar_vol.rolling(63, min_periods=10).mean()
    dv_std = dollar_vol.rolling(63, min_periods=10).std().replace(0, np.nan)
    dollar_vol_z = (dollar_vol - dv_mean) / dv_std

    # Apply .shift(1) to every output series (no-lookahead guard)
    outputs = {
        "conlan_eod_pct_below_52w_high": _safe_shift(pct_below),
        "conlan_eod_mom_6m":             _safe_shift(mom_6m),
        "conlan_eod_vol_trend_ratio":    _safe_shift(vol_trend_ratio),
        "conlan_eod_close_above_200ma":  _safe_shift(close_above_200),
        "conlan_eod_atr_pct":            _safe_shift(atr_pct),
        "conlan_eod_dollar_vol_z":       _safe_shift(dollar_vol_z),
    }

    for col, series in outputs.items():
        df[col] = series.fillna(0.0)

    logger.debug(
        "[conlan_eod] ticker=%s added %d features", ticker, CONLAN_EOD_FEATURE_COUNT
    )
    return df
