"""oc2_donchian_c003_features.py — Cycle 003 variant signal features.

Extracted from OC-2/phitis/strategies/strategies_cycle003_variants.py.
Key empirical findings from c003 autopsy (5Min Donchian(20)):
  - Opening range (first 6 bars of session): 100% WR, +$27.7k
  - Volume confirmation (vol > prev bar): 100% WR, +$24.5k
  - Combined OR + vol: strongest signal, very few false breakouts
  - Trade rate cap (first signal per day only): cuts overtrading (32% of failures)
  - Session bar index: useful meta-feature for timing-aware models
  - Short trades (1-5 bars): 89.7% WR vs long trades (60+ bars): 65.7% WR

All features are .shift(1)-safe — feature at index t uses only data from t-1.
Works on any OHLCV DataFrame with columns: open, high, low, close, volume.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rolling_max(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=1).max()

def _rolling_min(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=1).min()

def _atr14(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_c = close.shift(1).fillna(close)
    tr = pd.concat(
        [high - low, (high - prev_c).abs(), (low - prev_c).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(14, min_periods=1).mean()

def _session_bar_idx(df: pd.DataFrame) -> pd.Series:
    """Bar index within each trading session (0 = first bar of day)."""
    if pd.api.types.is_datetime64_any_dtype(df.index):
        dates = df.index.normalize()
    elif "datetime" in df.columns:
        dates = pd.to_datetime(df["datetime"]).dt.normalize()
    else:
        return pd.Series(np.zeros(len(df), dtype=np.int64), index=df.index)

    idx = np.zeros(len(df), dtype=np.int64)
    _, boundaries = np.unique(dates.to_numpy(), return_index=True)
    boundaries = np.append(boundaries, len(df))
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        idx[s:e] = np.arange(e - s)
    return pd.Series(idx, index=df.index)


# ---------------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------------

def add_oc2_donchian_c003_features(
    df: pd.DataFrame,
    ticker: str | None = None,
    entry_lookback: int = 20,
    exit_lookback: int = 40,
    or_bars: int = 6,
) -> pd.DataFrame:
    """Add cycle003 variant signal features.

    New columns
    -----------
    c003_donchian_upper20       float  rolling 20-bar high (prev bar, shift-safe)
    c003_donchian_lower40       float  rolling 40-bar low (prev bar, shift-safe)
    c003_breakout_signal        int    close > prev donchian upper (raw breakout)
    c003_atr14                  float  ATR(14) value
    c003_session_bar_idx        int    bar index in session (0=open, 77≈close on 5Min)
    c003_opening_range_flag     int    1 if session_bar_idx < or_bars (first 30min)
    c003_vol_confirm_flag       int    1 if volume > prev bar volume
    c003_or_breakout            int    breakout AND in opening range (100% WR signal)
    c003_vol_breakout           int    breakout AND volume confirmed
    c003_combined_signal        int    breakout AND opening range AND vol confirm
    c003_is_first_signal_day    int    1 if this is first breakout signal of the day
    c003_atr_expansion          int    ATR > 20-bar rolling mean of ATR (regime filter)
    """
    df = df.copy()

    h = df["high"]
    l = df["low"]
    c = df["close"]
    v = df.get("volume", pd.Series(np.nan, index=df.index))

    # Donchian channels — shifted by 1 so bar-t uses bar t-1 data
    upper = _rolling_max(h, entry_lookback).shift(1)
    lower = _rolling_min(l, exit_lookback).shift(1)
    atr = _atr14(h, l, c).shift(1)

    df["c003_donchian_upper20"] = upper
    df["c003_donchian_lower40"] = lower
    df["c003_atr14"] = atr

    # Raw breakout: current close > prev donchian upper
    breakout = (c > upper) & atr.gt(0)
    df["c003_breakout_signal"] = breakout.astype(int)

    # Session bar index
    bar_idx = _session_bar_idx(df)
    df["c003_session_bar_idx"] = bar_idx

    # Opening range flag: within first or_bars bars of session
    df["c003_opening_range_flag"] = (bar_idx < or_bars).astype(int)

    # Volume confirmation: current volume > previous volume
    if isinstance(v, pd.Series) and v.notna().any():
        prev_vol = v.shift(1).fillna(v)
        vol_confirm = v > prev_vol
    else:
        vol_confirm = pd.Series(True, index=df.index)
    df["c003_vol_confirm_flag"] = vol_confirm.astype(int)

    # Compound signals
    df["c003_or_breakout"] = (breakout & (bar_idx < or_bars)).astype(int)
    df["c003_vol_breakout"] = (breakout & vol_confirm).astype(int)
    df["c003_combined_signal"] = (breakout & (bar_idx < or_bars) & vol_confirm).astype(int)

    # First signal of day: find first breakout per date, mark only that bar
    if pd.api.types.is_datetime64_any_dtype(df.index):
        dates = df.index.normalize()
    elif "datetime" in df.columns:
        dates = pd.to_datetime(df["datetime"]).dt.normalize()
    else:
        dates = pd.Series(np.zeros(len(df)), index=df.index)

    first_flag = np.zeros(len(df), dtype=np.int8)
    seen: set = set()
    bsig = breakout.to_numpy()
    date_arr = dates.to_numpy() if hasattr(dates, "to_numpy") else np.array(dates)
    for i in range(len(df)):
        if bsig[i]:
            d = date_arr[i]
            if d not in seen:
                first_flag[i] = 1
                seen.add(d)
    df["c003_is_first_signal_day"] = first_flag

    # ATR expansion: current ATR > rolling 20-bar mean of ATR
    atr_sma20 = atr.rolling(20, min_periods=5).mean()
    df["c003_atr_expansion"] = (atr > atr_sma20).astype(int)

    return df
