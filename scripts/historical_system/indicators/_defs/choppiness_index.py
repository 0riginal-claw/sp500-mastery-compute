"""Choppiness Index — 100 * log10(sum(TR)/range) / log10(n)."""
from __future__ import annotations
import numpy as np, pandas as pd
from historical_system.indicators.base import Indicator, register
from historical_system.indicators import kernels as K

@register
class ChoppinessIndex(Indicator):
    name = "choppiness_index"
    outputs = ("chop",)
    params = {"length": 14}
    deps = ("high", "low", "close")
    def compute(self, df, length=14):
        h = df["high"].to_numpy(); l = df["low"].to_numpy(); c = df["close"].to_numpy()
        tr = K.true_range(h, l, c)
        sum_tr = pd.Series(tr).rolling(length, min_periods=length).sum()
        hh = pd.Series(h).rolling(length, min_periods=length).max()
        ll = pd.Series(l).rolling(length, min_periods=length).min()
        rng = (hh - ll).replace(0, np.nan)
        return {"chop": (100.0 * np.log10(sum_tr / rng) / np.log10(length)).to_numpy()}
