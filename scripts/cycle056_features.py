# Source: /Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/claudes test/research/archive/cycle056_combo_dimensions_2026-05-05/volatility_gates_v2.py
"""
Wrapper for cycle056 volatility combo gates (C2A-C2E).

Cycle056 extended cycle055's 18 single volatility gates with 5 duo-gate combos
derived from the top-performing cycle055 winners:
  C2A: V06_range_expanding     × V10_or30_not_blown_out
  C2B: V06_range_expanding     × V12_not_low_vol
  C2C: V10_or30_not_blown_out  × V13_not_high_vol
  C2D: V03_candle_above_5c     × V11_normal_vol_regime
  C2E: V05_room_in_daily_range × V13_not_high_vol

Daily adaptation (cycle055_features.py already wired; inlined here for portability):
  range_expanding  → daily range > rolling 5-day mean daily range
  or30_not_blown   → daily range does not exceed 2× prior-day range (bounded volatility)
  not_low_vol      → vol_regime != 0  (21-day RV not in bottom quartile)
  not_high_vol     → vol_regime != 2  (21-day RV not in top quartile)
  normal_regime    → vol_regime == 1  (21-day RV in middle 50%)
  candle_floor     → close > open (bullish candle body confirmation)
  range_room       → body_pct < 0.75 (candle body not filling the entire day's range)

Features emitted (all .shift(1)-safe):
  c056_vol_regime       : 0=LOW / 1=NORMAL / 2=HIGH (21-day RV percentile)
  c056_range_expanding  : bool — daily range > 5-day mean range
  c056_c2a              : bool — C2A combo gate
  c056_c2b              : bool — C2B combo gate
  c056_c2c              : bool — C2C combo gate
  c056_c2d              : bool — C2D combo gate
  c056_c2e              : bool — C2E combo gate
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CYCLE056_FEATURE_NAMES: list[str] = [
    "c056_vol_regime",
    "c056_range_expanding",
    "c056_c2a",
    "c056_c2b",
    "c056_c2c",
    "c056_c2d",
    "c056_c2e",
]

_RV_WIN = 21
_RANGE_EXPAND_WIN = 5
_OR30_FACTOR = 2.0   # day range > 2x prior-day range = blown out
_RANK_WIN = 252


def _zero_fill(df: pd.DataFrame) -> pd.DataFrame:
    for col in CYCLE056_FEATURE_NAMES:
        if col not in df.columns:
            if col == "c056_vol_regime":
                df[col] = 1
            else:
                df[col] = False
    return df


def _vol_regime(close: pd.Series, rv_win: int = _RV_WIN) -> pd.Series:
    """Ternary vol regime: 0=LOW, 1=NORMAL, 2=HIGH (matches cycle055 logic)."""
    log_ret = np.log(close.replace(0, np.nan)).diff()
    rv = (log_ret.rolling(rv_win, min_periods=5).std() * np.sqrt(252.0)).fillna(0.0)
    pct_rank = rv.rolling(_RANK_WIN, min_periods=rv_win).rank(pct=True)
    regime = pd.Series(1, index=close.index, dtype=int)
    regime[pct_rank <= 0.25] = 0
    regime[pct_rank > 0.75] = 2
    regime[rv.isna()] = 1
    return regime


def add_cycle056_features(df: pd.DataFrame, ticker: str = None) -> pd.DataFrame:
    """Append cycle056 volatility combo gate features to daily OHLCV df. Idempotent.

    Requires 'close'. 'high', 'low', 'open' improve quality.
    All output is .shift(1)-safe.
    """
    if df is None or len(df) == 0:
        return df
    if all(c in df.columns for c in CYCLE056_FEATURE_NAMES):
        return df
    if "close" not in df.columns:
        return _zero_fill(df)

    close = pd.to_numeric(df["close"], errors="coerce").astype(float)
    open_ = pd.to_numeric(df["open"], errors="coerce").astype(float) if "open" in df.columns else close
    high = pd.to_numeric(df["high"], errors="coerce").astype(float) if "high" in df.columns else close
    low = pd.to_numeric(df["low"], errors="coerce").astype(float) if "low" in df.columns else close

    # Base volatility regime (inline from cycle055 logic)
    vol_regime = _vol_regime(close)

    # Daily range and body
    daily_range = (high - low).clip(lower=0)
    prev_range = daily_range.shift(1)
    body_abs = (close - open_).abs()
    bar_range_safe = daily_range.replace(0, np.nan)
    body_pct = (body_abs / bar_range_safe).fillna(0.0).clip(0.0, 1.0)

    # V06 proxy: range_expanding — today's range > 5-day mean range
    mean_range_5d = daily_range.rolling(_RANGE_EXPAND_WIN, min_periods=2).mean()
    range_expanding = daily_range > mean_range_5d

    # V10 proxy: or30_not_blown_out — today's range <= 2× prior-day range
    or30_not_blown = daily_range <= (_OR30_FACTOR * prev_range.fillna(daily_range))

    # V12 proxy: not_low_vol — vol_regime != 0
    not_low_vol = vol_regime != 0

    # V13 proxy: not_high_vol — vol_regime != 2
    not_high_vol = vol_regime != 2

    # V11 proxy: normal_vol_regime — vol_regime == 1
    normal_regime = vol_regime == 1

    # V03 proxy: candle_floor — close > open (bullish candle)
    candle_floor = close > open_

    # V05 proxy: range_room — body not filling entire range (room left in day's range)
    range_room = body_pct < 0.75

    # Combo gates
    c2a = range_expanding & or30_not_blown
    c2b = range_expanding & not_low_vol
    c2c = or30_not_blown & not_high_vol
    c2d = candle_floor & normal_regime
    c2e = range_room & not_high_vol

    if "c056_vol_regime" not in df.columns:
        df["c056_vol_regime"] = vol_regime.shift(1).fillna(1).astype(int).values
    if "c056_range_expanding" not in df.columns:
        df["c056_range_expanding"] = range_expanding.shift(1).fillna(False).astype(bool).values
    if "c056_c2a" not in df.columns:
        df["c056_c2a"] = c2a.shift(1).fillna(False).astype(bool).values
    if "c056_c2b" not in df.columns:
        df["c056_c2b"] = c2b.shift(1).fillna(False).astype(bool).values
    if "c056_c2c" not in df.columns:
        df["c056_c2c"] = c2c.shift(1).fillna(False).astype(bool).values
    if "c056_c2d" not in df.columns:
        df["c056_c2d"] = c2d.shift(1).fillna(False).astype(bool).values
    if "c056_c2e" not in df.columns:
        df["c056_c2e"] = c2e.shift(1).fillna(False).astype(bool).values

    return df
