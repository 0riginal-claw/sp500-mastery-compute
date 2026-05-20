"""Photis strategy family: Breakout
Extracted from Rayner Teo 'stock_strategy' catalog (571 videos).
Representative titles: 'Breakout Trading Explained', 'Breakout Trading Secrets',
'Break and Retest Trading Strategy', 'Do You Make This Breakout Trading Mistake?',
'5 Things To Look For Before You Place A Trade (Breakout Trading Strategy)'.
Core logic: Donchian channel breakout with volume confirmation and ATR expansion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def add_photis_strategy_breakout_features(
    df: pd.DataFrame, ticker: str | None = None
) -> pd.DataFrame:
    """Add 6 breakout signal columns derived from Rayner catalog.

    Columns added (all .shift(1)-safe):
        photis_breakout_donchian_upper20  : 20-bar rolling high (channel top)
        photis_breakout_donchian_lower20  : 20-bar rolling low (channel bottom)
        photis_breakout_range_pct         : (upper - lower) / lower — channel width as %
        photis_breakout_signal            : 1 if close crossed above prev donchian_upper20
        photis_breakout_vol_confirm       : 1 if volume > 1.5 * 20-bar avg volume
        photis_breakout_atr_expansion     : bar_range / ATR14 — volatility expansion flag
    """
    out = df.copy()
    c = out["close"]
    h = out["high"]
    lo = out["low"]
    vol = out["volume"] if "volume" in out.columns else pd.Series(np.nan, index=out.index)

    upper20 = h.rolling(20).max()
    lower20 = lo.rolling(20).min()
    atr14 = _atr(h, lo, c, 14)
    vol_avg20 = vol.rolling(20).mean()
    bar_range = h - lo

    out["photis_breakout_donchian_upper20"] = upper20.shift(1)
    out["photis_breakout_donchian_lower20"] = lower20.shift(1)
    out["photis_breakout_range_pct"] = ((upper20 - lower20) / (lower20 + 1e-9)).shift(1)
    out["photis_breakout_signal"] = (c.shift(1) > upper20.shift(2)).astype(float)
    out["photis_breakout_vol_confirm"] = (vol.shift(1) > 1.5 * vol_avg20.shift(1)).astype(float)
    out["photis_breakout_atr_expansion"] = (bar_range.shift(1) / (atr14.shift(1) + 1e-9))

    return out
