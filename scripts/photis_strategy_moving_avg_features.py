"""Photis strategy family: Moving Average Crossovers
Extracted from Rayner Teo 'stock_strategy' catalog (571 videos).
Representative titles: 'A Moving Average Trading Strategy That Works',
'Golden Cross Explained', 'Do You Make These Moving Average Mistakes?',
'5 AMAZING Trend Indicators'.
Core logic: EMA crossover signals, MA stack alignment, price-to-MA distance.
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


def add_photis_strategy_moving_avg_features(
    df: pd.DataFrame, ticker: str | None = None
) -> pd.DataFrame:
    """Add 6 moving-average signal columns derived from Rayner catalog.

    Columns added (all .shift(1)-safe):
        photis_moving_avg_golden_cross     : 1 if EMA20 crossed above EMA50 (golden cross)
        photis_moving_avg_death_cross      : 1 if EMA20 crossed below EMA50 (death cross)
        photis_moving_avg_stack_bullish    : 1 if close > EMA20 > EMA50 > EMA200
        photis_moving_avg_price_vs_ema20   : (close - EMA20) / ATR14 — normalized distance
        photis_moving_avg_ema_separation   : (EMA20 - EMA50) / close — MA gap ratio
        photis_moving_avg_compression      : 1 if |EMA20 - EMA50| < 0.5 * ATR14 (squeeze)
    """
    out = df.copy()
    c = out["close"]
    h = out["high"]
    lo = out["low"]

    ema20 = _ema(c, 20)
    ema50 = _ema(c, 50)
    ema200 = _ema(c, 200)
    atr14 = _atr(h, lo, c, 14)

    golden_cross = (ema20 > ema50) & (ema20.shift(1) <= ema50.shift(1))
    death_cross = (ema20 < ema50) & (ema20.shift(1) >= ema50.shift(1))
    stack_bull = (c > ema20) & (ema20 > ema50) & (ema50 > ema200)
    compression = (ema20 - ema50).abs() < 0.5 * atr14

    out["photis_moving_avg_golden_cross"] = golden_cross.shift(1).astype(float)
    out["photis_moving_avg_death_cross"] = death_cross.shift(1).astype(float)
    out["photis_moving_avg_stack_bullish"] = stack_bull.shift(1).astype(float)
    out["photis_moving_avg_price_vs_ema20"] = ((c - ema20) / (atr14 + 1e-9)).shift(1)
    out["photis_moving_avg_ema_separation"] = ((ema20 - ema50) / (c + 1e-9)).shift(1)
    out["photis_moving_avg_compression"] = compression.shift(1).astype(float)

    return out
