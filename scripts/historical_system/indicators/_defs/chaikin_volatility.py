"""Chaikin Volatility — rate of change of EMA(high - low)."""
from __future__ import annotations
import numpy as np
from historical_system.indicators.base import Indicator, register
from historical_system.indicators import kernels as K

@register
class ChaikinVolatility(Indicator):
    name = "chaikin_volatility"
    outputs = ("chaikin_vol",)
    params = {"ema_len": 10, "roc_len": 10}
    deps = ("high", "low")
    def compute(self, df, ema_len=10, roc_len=10):
        e = K.ema(df["high"] - df["low"], ema_len)
        e_shift = np.roll(e, roc_len); e_shift[:roc_len] = np.nan
        return {"chaikin_vol": 100.0 * (e - e_shift) / np.where(e_shift == 0, np.nan, e_shift)}
