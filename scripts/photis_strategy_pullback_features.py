"""Photis strategy family: Pullback / Mean-Reversion
Extracted from Rayner Teo 'stock_strategy' catalog (571 videos).
Representative titles: '5 Things To Look For Before You Place A Trade (Pullback Trading)',
'Do You Make This Pullback Trading Mistake?', 'Bollinger Band Pullback System (+2834%)',
'Forex Trading Secrets: How To Buy Low And Sell High'.
Core logic: RSI-based pullback detection in an established trend; bounce from EMA.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def add_photis_strategy_pullback_features(
    df: pd.DataFrame, ticker: str | None = None
) -> pd.DataFrame:
    """Add 5 pullback signal columns derived from Rayner catalog.

    Columns added (all .shift(1)-safe):
        photis_pullback_rsi14         : 14-period RSI
        photis_pullback_rsi_zone      : 1 if RSI < 45 while above EMA50 (pullback in uptrend)
        photis_pullback_retrace_atr   : (EMA50 - close) / ATR14 — normalized retrace depth
        photis_pullback_bounce_signal : 1 if close crossed back above EMA20 from below
        photis_pullback_higher_low    : 1 if close > prev_low AND close > EMA50
    """
    out = df.copy()
    c = out["close"]
    h = out["high"]
    lo = out["low"]

    ema20 = _ema(c, 20)
    ema50 = _ema(c, 50)
    rsi14 = _rsi(c, 14)
    atr14 = _atr(h, lo, c, 14)

    above_ema50 = (c > ema50).astype(float)
    rsi_pullback = (rsi14 < 45) & (above_ema50 == 1)

    below_ema20_prev = c.shift(2) < ema20.shift(2)
    above_ema20_now = c.shift(1) > ema20.shift(1)
    bounce = (below_ema20_prev & above_ema20_now)

    out["photis_pullback_rsi14"] = rsi14.shift(1)
    out["photis_pullback_rsi_zone"] = rsi_pullback.shift(1).astype(float)
    out["photis_pullback_retrace_atr"] = ((ema50 - c) / (atr14 + 1e-9)).shift(1)
    out["photis_pullback_bounce_signal"] = bounce.shift(1).astype(float)
    out["photis_pullback_higher_low"] = (
        (c.shift(1) > lo.shift(2)) & (c.shift(1) > ema50.shift(1))
    ).astype(float)

    return out
