"""Photis strategy family: Price Action / Candlestick Patterns
Extracted from Rayner Teo 'stock_strategy' catalog (571 videos).
Representative titles: '11 Price Action Trading Strategies', '3 POWERFUL Doji Candlestick Patterns',
'3 Best Price Action Strategies', 'A pinbar trading strategy that works',
'Candlestick Patterns For Beginners'.
Core logic: bar-level pattern detection — pin bar, doji, engulfing, inside bar.
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


def add_photis_strategy_price_action_features(
    df: pd.DataFrame, ticker: str | None = None
) -> pd.DataFrame:
    """Add 6 price-action pattern columns derived from Rayner catalog.

    Columns added (all .shift(1)-safe):
        photis_price_action_bar_range_norm  : (high - low) / ATR14 — normalized bar size
        photis_price_action_close_strength  : (close - low) / (high - low) — close position in bar
        photis_price_action_pin_bar         : 1 if wick >= 2/3 of total bar range (either end)
        photis_price_action_engulfing_bull  : 1 if bullish engulfing vs previous bar
        photis_price_action_doji            : 1 if body <= 15% of bar range
        photis_price_action_inside_bar      : 1 if bar is fully inside previous bar's range
    """
    out = df.copy()
    o = out["open"]
    c = out["close"]
    h = out["high"]
    lo = out["low"]

    atr14 = _atr(h, lo, c, 14)
    bar_range = (h - lo).clip(lower=1e-9)
    body = (c - o).abs()
    upper_wick = h - c.where(c > o, o)
    lower_wick = c.where(c < o, o) - lo

    pin_bar = (
        (upper_wick >= 2 / 3 * bar_range) | (lower_wick >= 2 / 3 * bar_range)
    )
    engulfing_bull = (
        (c > o) &
        (c > h.shift(1)) &
        (o < lo.shift(1))
    )
    doji = body < 0.15 * bar_range
    inside_bar = (h < h.shift(1)) & (lo > lo.shift(1))

    out["photis_price_action_bar_range_norm"] = (bar_range / (atr14 + 1e-9)).shift(1)
    out["photis_price_action_close_strength"] = ((c - lo) / bar_range).shift(1)
    out["photis_price_action_pin_bar"] = pin_bar.shift(1).astype(float)
    out["photis_price_action_engulfing_bull"] = engulfing_bull.shift(1).astype(float)
    out["photis_price_action_doji"] = doji.shift(1).astype(float)
    out["photis_price_action_inside_bar"] = inside_bar.shift(1).astype(float)

    return out
