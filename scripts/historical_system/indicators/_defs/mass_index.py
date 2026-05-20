"""Mass Index — sum(EMA9(H-L)/EMA9(EMA9(H-L))) over n bars."""
from __future__ import annotations
import pandas as pd, numpy as np
from historical_system.indicators.base import Indicator, register
from historical_system.indicators import kernels as K

@register
class MassIndex(Indicator):
    name = "mass_index"
    outputs = ("mass_index",)
    params = {"length": 25, "ema_len": 9}
    deps = ("high", "low")
    def compute(self, df, length=25, ema_len=9):
        rng = (df["high"] - df["low"]).to_numpy()
        e1 = K.ema(rng, ema_len)
        e2 = K.ema(e1, ema_len)
        ratio = np.where(e2 == 0, np.nan, e1 / e2)
        return {"mass_index": pd.Series(ratio).rolling(length, min_periods=length).sum().to_numpy()}
