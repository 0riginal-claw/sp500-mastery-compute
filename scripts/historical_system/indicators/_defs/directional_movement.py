"""Directional Movement — +DI, -DI (without the ADX smoothing)."""
from __future__ import annotations
import numpy as np
from historical_system.indicators.base import Indicator, register
from historical_system.indicators import kernels as K

@register
class DirectionalMovement(Indicator):
    name = "directional_movement"
    outputs = ("plus_di", "minus_di")
    params = {"length": 14}
    deps = ("high", "low", "close")
    def compute(self, df, length=14):
        h = df["high"].to_numpy(); l = df["low"].to_numpy(); c = df["close"].to_numpy()
        p, m = K.directional_movement(h, l)
        tr = K.true_range(h, l, c)
        atr = K.rma(tr, length)
        atr_safe = np.where(atr == 0, np.nan, atr)
        return {"plus_di": 100.0 * K.rma(p, length) / atr_safe,
                "minus_di": 100.0 * K.rma(m, length) / atr_safe}
