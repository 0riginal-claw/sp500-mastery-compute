"""Bollinger %B — (price - lower) / (upper - lower)."""
from __future__ import annotations
import numpy as np
from historical_system.indicators.base import Indicator, register
from historical_system.indicators import kernels as K

@register
class BollingerPercentB(Indicator):
    name = "bollinger_percent_b"
    outputs = ("bb_percent_b",)
    params = {"length": 20, "mult": 2.0, "source": "close"}
    deps = ("close",)
    def compute(self, df, length=20, mult=2.0, source="close"):
        basis = K.sma(df[source], length); sd = K.rolling_std(df[source], length)
        up = basis + mult*sd; lo = basis - mult*sd
        rng = up - lo
        return {"bb_percent_b": (df[source].to_numpy() - lo) / np.where(rng == 0, np.nan, rng)}
