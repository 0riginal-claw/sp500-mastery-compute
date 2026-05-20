"""
Jesse indicator features wrapper.
Source: https://github.com/jesse-ai/jesse (MIT)
Exposes: add_jesse_features(df, **kwargs) -> pd.DataFrame

Adds jesse_chop (Choppiness Index) and jesse_atr columns.
Jesse candles format: [timestamp, open, close, high, low, volume]

Falls back to pure numpy when jesse_rust (Rust ext) is unavailable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _df_to_jesse_candles(df: pd.DataFrame) -> np.ndarray:
    """Convert OHLCV DataFrame to Jesse numpy candle array (N, 6)."""
    ts = np.arange(len(df), dtype=np.float64)  # synthetic timestamps
    candles = np.column_stack([
        ts,
        df["open"].values.astype(np.float64),
        df["close"].values.astype(np.float64),
        df["high"].values.astype(np.float64),
        df["low"].values.astype(np.float64),
        df["volume"].values.astype(np.float64),
    ])
    return candles


def _atr_numpy(candles: np.ndarray, period: int = 14) -> np.ndarray:
    high = candles[:, 3]
    low = candles[:, 4]
    prev_close = np.concatenate([[candles[0, 2]], candles[:-1, 2]])
    tr = np.maximum(high - low, np.maximum(abs(high - prev_close), abs(low - prev_close)))
    result = np.full(len(tr), np.nan)
    result[period - 1] = tr[:period].mean()
    alpha = 1.0 / period
    for i in range(period, len(tr)):
        result[i] = alpha * tr[i] + (1 - alpha) * result[i - 1]
    return result


def _chop_numpy(candles: np.ndarray, period: int = 14, scalar: float = 100.0) -> np.ndarray:
    high = candles[:, 3]
    low = candles[:, 4]
    prev_close = np.concatenate([[candles[0, 2]], candles[:-1, 2]])
    tr = np.maximum(high - low, np.maximum(abs(high - prev_close), abs(low - prev_close)))
    result = np.full(len(candles), np.nan)
    for i in range(period - 1, len(candles)):
        window = candles[i - period + 1: i + 1]
        atr_sum = tr[i - period + 1: i + 1].sum()
        hl_range = window[:, 3].max() - window[:, 4].min()
        if hl_range > 0:
            result[i] = scalar * np.log10(atr_sum / hl_range) / np.log10(period)
    return result


def add_jesse_features(
    df: pd.DataFrame,
    atr_period: int = 14,
    chop_period: int = 14,
    chop_scalar: float = 100.0,
    **kwargs,
) -> pd.DataFrame:
    """
    Add Jesse-derived indicators to df.

    Columns added:
      jesse_atr   — Average True Range (Wilder smoothing)
      jesse_chop  — Choppiness Index [0-100]; >61.8 = choppy, <38.2 = trending

    Args:
        df: OHLCV DataFrame with columns [open, high, low, close, volume]
        atr_period: ATR lookback (default 14)
        chop_period: Choppiness Index lookback (default 14)
        chop_scalar: Choppiness scalar (default 100)
    """
    df = df.copy()
    candles = _df_to_jesse_candles(df)

    try:
        from jesse.indicators import atr as jesse_atr_fn
        from jesse.indicators import chop as jesse_chop_fn
        df["jesse_atr"] = jesse_atr_fn(candles, period=atr_period, sequential=True)
        df["jesse_chop"] = jesse_chop_fn(
            candles, period=chop_period, scalar=chop_scalar, sequential=True
        )
    except ImportError:
        # jesse_rust not installed — use numpy fallback
        df["jesse_atr"] = _atr_numpy(candles, period=atr_period)
        df["jesse_chop"] = _chop_numpy(candles, period=chop_period, scalar=chop_scalar)

    return df
