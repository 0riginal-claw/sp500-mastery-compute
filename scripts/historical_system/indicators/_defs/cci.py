"""Commodity Channel Index — (TP - SMA(TP)) / (0.015 * mean_abs_dev)."""
from __future__ import annotations
import numpy as np, pandas as pd
from historical_system.indicators.base import Indicator, register
from historical_system.indicators import kernels as K

@register
class CCI(Indicator):
    name = "cci"
    outputs = ("cci",)
    params = {"length": 20}
    deps = ("high", "low", "close")
    def compute(self, df, length=20):
        tp = K.typical_price(df)
        ma = K.sma(tp, length)
        mad = pd.Series(tp).rolling(length, min_periods=length).apply(
            lambda w: np.mean(np.abs(w - w.mean())), raw=True
        ).to_numpy()
        mad = np.where(mad == 0, np.nan, mad)
        return {"cci": (tp - ma) / (0.015 * mad)}
