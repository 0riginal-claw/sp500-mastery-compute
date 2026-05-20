"""Relative Vigor Index — (close-open) / (high-low), SMA-smoothed + signal."""
from __future__ import annotations
import numpy as np
from historical_system.indicators.base import Indicator, register
from historical_system.indicators import kernels as K

@register
class RVI(Indicator):
    name = "relative_vigor_index"
    outputs = ("rvi", "rvi_signal")
    params = {"length": 10}
    deps = ("open", "high", "low", "close")
    def compute(self, df, length=10):
        num = (df["close"] - df["open"]).to_numpy()
        den = (df["high"] - df["low"]).to_numpy()
        num_sm = K.sma(num, length); den_sm = K.sma(den, length)
        rvi = np.where(den_sm == 0, np.nan, num_sm / den_sm)
        sig = K.sma(rvi, 4)
        return {"rvi": rvi, "rvi_signal": sig}
