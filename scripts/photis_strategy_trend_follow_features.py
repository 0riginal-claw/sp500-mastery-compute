"""Photis strategy family: Trend Following
Extracted from Rayner Teo 'stock_strategy' catalog (571 videos).
Representative titles: 'A Simple Trend Following Strategy', '3 Toughest Markets for Trend Followers',
'3 Little Known Ways to Trade With the Trend', 'A Forex Trading Strategy To Profit In Bull & Bear Markets'.
Core logic: identify trend direction via EMAs, measure pullback opportunity, trail with MA.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def add_photis_strategy_trend_follow_features(
    df: pd.DataFrame, ticker: str | None = None
) -> pd.DataFrame:
    """Add 6 trend-following signal columns derived from Rayner catalog.

    Columns added (all .shift(1)-safe — assigned after shift):
        photis_trend_follow_ema50_slope    : normalized slope of 50-EMA over 5 bars
        photis_trend_follow_above_ema200   : 1 if close > 200-EMA else 0
        photis_trend_follow_trend_strength : (ema50 - ema200) / ATR14, trend momentum
        photis_trend_follow_higher_highs   : 1 if close > close 10 bars ago else 0
        photis_trend_follow_pullback_to_ema: 1 if |close - ema50| < 1.0 * ATR14
        photis_trend_follow_trail_signal   : 1 if close > ema50 AND ema50 > ema200 (full trend)
    """
    out = df.copy()
    c = out["close"]
    h = out["high"]
    lo = out["low"]

    ema50 = _ema(c, 50)
    ema200 = _ema(c, 200)
    atr14 = _atr(h, lo, c, 14)

    ema50_slope = (ema50 - ema50.shift(5)) / (atr14 * 5 + 1e-9)

    out["photis_trend_follow_ema50_slope"] = ema50_slope.shift(1)
    out["photis_trend_follow_above_ema200"] = (c.shift(1) > ema200.shift(1)).astype(float)
    out["photis_trend_follow_trend_strength"] = ((ema50 - ema200) / (atr14 + 1e-9)).shift(1)
    out["photis_trend_follow_higher_highs"] = (c.shift(1) > c.shift(11)).astype(float)
    out["photis_trend_follow_pullback_to_ema"] = (
        (c.shift(1) - ema50.shift(1)).abs() < atr14.shift(1)
    ).astype(float)
    out["photis_trend_follow_trail_signal"] = (
        (c.shift(1) > ema50.shift(1)) & (ema50.shift(1) > ema200.shift(1))
    ).astype(float)

    return out
