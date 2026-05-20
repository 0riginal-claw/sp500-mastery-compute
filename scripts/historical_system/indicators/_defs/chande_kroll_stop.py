"""Chande Kroll Stop — long/short stops using high/low ± ATR multiples."""
from __future__ import annotations
import pandas as pd
from historical_system.indicators.base import Indicator, register
from historical_system.indicators import kernels as K

@register
class ChandeKrollStop(Indicator):
    name = "chande_kroll_stop"
    outputs = ("ck_long_stop", "ck_short_stop")
    params = {"p": 10, "x": 1.0, "q": 9}
    deps = ("high", "low", "close")
    def compute(self, df, p=10, x=1.0, q=9):
        a = K.atr(df["high"], df["low"], df["close"], p)
        high_stop = K.rolling_max(df["high"], p) - x * a
        low_stop = K.rolling_min(df["low"], p) + x * a
        long_stop = pd.Series(high_stop).rolling(q, min_periods=q).max().to_numpy()
        short_stop = pd.Series(low_stop).rolling(q, min_periods=q).min().to_numpy()
        return {"ck_long_stop": long_stop, "ck_short_stop": short_stop}
