"""On Balance Volume — cumulative signed volume by close direction."""
from __future__ import annotations
import numpy as np
from historical_system.indicators.base import Indicator, register

@register
class OBV(Indicator):
    name = "obv"
    outputs = ("obv",)
    params = {}
    deps = ("close", "volume")
    def compute(self, df):
        c = df["close"].to_numpy(dtype=np.float64); v = df["volume"].to_numpy(dtype=np.float64)
        d = np.diff(c, prepend=c[0])
        sign = np.where(d > 0, 1.0, np.where(d < 0, -1.0, 0.0))
        return {"obv": (sign * v).cumsum()}
