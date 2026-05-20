# Source: /Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/claudes test/research/archive/cycle054_multi_tf_vwap_sweep_2026-05-04/refine_per_ticker_v6.py
"""
Wrapper for cycle054 multi-timeframe VWAP features (attach_mtf_vwap_to_cache).

Cycle054 tested cross-timeframe VWAP alignment as additional gates on top of
the cycle052 5-min intraday VWAP. Three gates were tested:
  S40-above_vwap_5m_and_15m    : 5-min VWAP AND 15-min VWAP both above
  S40-above_vwap_5m_and_60m    : 5-min VWAP AND 60-min VWAP both above
  S40-above_vwap_5m_and_15m_and_60m : all three aligned

Daily adaptation:
  5-min VWAP  → 5-day rolling VWAP  (short-term price anchor)
  15-min VWAP → 3-day rolling VWAP  (3 × 5-min bars per 15-min)
  60-min VWAP → 12-day rolling VWAP (12 × 5-min bars per 60-min)

Features emitted (all .shift(1)-safe):
  c054_above_vwap_3d       : bool — close > 3-day rolling VWAP (15-min proxy)
  c054_above_vwap_5d       : bool — close > 5-day rolling VWAP (5-min proxy)
  c054_above_vwap_12d      : bool — close > 12-day rolling VWAP (60-min proxy)
  c054_vwap_5d_and_3d      : bool — above both 5d and 3d VWAP (S40-5m+15m proxy)
  c054_vwap_5d_and_12d     : bool — above both 5d and 12d VWAP (S40-5m+60m proxy)
  c054_vwap_5d_3d_and_12d  : bool — above all three VWAPs (S40-5m+15m+60m proxy)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CYCLE054_FEATURE_NAMES: list[str] = [
    "c054_above_vwap_3d",
    "c054_above_vwap_5d",
    "c054_above_vwap_12d",
    "c054_vwap_5d_and_3d",
    "c054_vwap_5d_and_12d",
    "c054_vwap_5d_3d_and_12d",
]


def _zero_fill(df: pd.DataFrame) -> pd.DataFrame:
    for col in CYCLE054_FEATURE_NAMES:
        if col not in df.columns:
            df[col] = False
    return df


def _rolling_vwap(close: pd.Series, high: pd.Series, low: pd.Series,
                  volume: pd.Series, win: int) -> pd.Series:
    """Rolling n-day VWAP: cumsum(typical_price * volume) / cumsum(volume)."""
    typ = (high + low + close) / 3.0
    num = (typ * volume).rolling(win, min_periods=max(1, win // 2)).sum()
    den = volume.rolling(win, min_periods=max(1, win // 2)).sum().replace(0, np.nan)
    return (num / den).fillna(close)


def add_cycle054_features(df: pd.DataFrame, ticker: str = None) -> pd.DataFrame:
    """Append cycle054 multi-TF VWAP features to daily OHLCV df. Idempotent.

    Requires 'close'. 'high', 'low', 'volume' improve accuracy.
    All output is .shift(1)-safe.
    """
    if df is None or len(df) == 0:
        return df
    if all(c in df.columns for c in CYCLE054_FEATURE_NAMES):
        return df
    if "close" not in df.columns:
        return _zero_fill(df)

    close = pd.to_numeric(df["close"], errors="coerce").astype(float)
    high = pd.to_numeric(df["high"], errors="coerce").astype(float) if "high" in df.columns else close
    low = pd.to_numeric(df["low"], errors="coerce").astype(float) if "low" in df.columns else close
    volume = pd.to_numeric(df["volume"], errors="coerce").astype(float) if "volume" in df.columns else pd.Series(1.0, index=df.index)

    vwap_3d = _rolling_vwap(close, high, low, volume, 3)
    vwap_5d = _rolling_vwap(close, high, low, volume, 5)
    vwap_12d = _rolling_vwap(close, high, low, volume, 12)

    above_3d = close > vwap_3d
    above_5d = close > vwap_5d
    above_12d = close > vwap_12d

    vwap_5d_and_3d = above_5d & above_3d
    vwap_5d_and_12d = above_5d & above_12d
    vwap_all_three = above_5d & above_3d & above_12d

    if "c054_above_vwap_3d" not in df.columns:
        df["c054_above_vwap_3d"] = above_3d.shift(1).fillna(False).astype(bool).values
    if "c054_above_vwap_5d" not in df.columns:
        df["c054_above_vwap_5d"] = above_5d.shift(1).fillna(False).astype(bool).values
    if "c054_above_vwap_12d" not in df.columns:
        df["c054_above_vwap_12d"] = above_12d.shift(1).fillna(False).astype(bool).values
    if "c054_vwap_5d_and_3d" not in df.columns:
        df["c054_vwap_5d_and_3d"] = vwap_5d_and_3d.shift(1).fillna(False).astype(bool).values
    if "c054_vwap_5d_and_12d" not in df.columns:
        df["c054_vwap_5d_and_12d"] = vwap_5d_and_12d.shift(1).fillna(False).astype(bool).values
    if "c054_vwap_5d_3d_and_12d" not in df.columns:
        df["c054_vwap_5d_3d_and_12d"] = vwap_all_three.shift(1).fillna(False).astype(bool).values

    return df
