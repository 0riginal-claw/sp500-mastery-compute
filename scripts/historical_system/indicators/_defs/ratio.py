"""Ratio — close / lookback_close. Simple relative-to-self indicator."""
from __future__ import annotations
import numpy as np
from historical_system.indicators.base import Indicator, register

@register
class Ratio(Indicator):
    name = "ratio"
    outputs = ("ratio",)
    params = {"lookback": 20, "source": "close"}
    deps = ("close",)
    def compute(self, df, lookback=20, source="close"):
        s = df[source]
        base = s.shift(lookback)
        return {"ratio": (s / base.replace(0, np.nan)).to_numpy()}
