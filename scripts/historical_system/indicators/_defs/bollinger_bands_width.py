"""Bollinger Bands Width — (upper - lower) / basis."""
from __future__ import annotations
import numpy as np
from historical_system.indicators.base import Indicator, register
from historical_system.indicators import kernels as K

@register
class BollingerBandsWidth(Indicator):
    name = "bollinger_bands_width"
    outputs = ("bb_width",)
    params = {"length": 20, "mult": 2.0, "source": "close"}
    deps = ("close",)
    def compute(self, df, length=20, mult=2.0, source="close"):
        basis = K.sma(df[source], length); sd = K.rolling_std(df[source], length)
        return {"bb_width": (2.0 * mult * sd) / np.where(basis == 0, np.nan, basis)}
