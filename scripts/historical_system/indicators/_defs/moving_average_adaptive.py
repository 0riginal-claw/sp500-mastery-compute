"""KAMA (Kaufman Adaptive MA) — efficiency-ratio-weighted EMA."""
from __future__ import annotations
import numpy as np, pandas as pd
from historical_system.indicators.base import Indicator, register

@register
class KAMA(Indicator):
    name = "ma_adaptive"
    outputs = ("kama",)
    params = {"length": 10, "fast": 2, "slow": 30, "source": "close"}
    deps = ("close",)
    def compute(self, df, length=10, fast=2, slow=30, source="close"):
        c = df[source].to_numpy(dtype=np.float64)
        n = len(c)
        out = np.full(n, np.nan, dtype=np.float64)
        if n <= length:
            return {"kama": out}
        change = np.abs(c[length:] - c[:-length])
        volatility = np.convolve(np.abs(np.diff(c, prepend=c[0])), np.ones(length), mode="valid")[1:]
        volatility = np.where(volatility == 0, 1e-12, volatility)
        er = change / volatility
        sc = (er * (2.0/(fast+1) - 2.0/(slow+1)) + 2.0/(slow+1)) ** 2
        out[length] = c[length]
        for i in range(length + 1, n):
            out[i] = out[i-1] + sc[i-length-1] * (c[i] - out[i-1])
        return {"kama": out}
