"""Session VWAP — cumulative pv / cumulative v, reset at each session boundary."""
from __future__ import annotations
import numpy as np, pandas as pd
from historical_system.indicators.base import Indicator, register

@register
class VWAP(Indicator):
    name = "vwap"
    outputs = ("vwap",)
    params = {}
    deps = ("high", "low", "close", "volume")
    def compute(self, df):
        tp = (df["high"] + df["low"] + df["close"]) / 3.0
        pv = tp * df["volume"]
        day = pd.to_datetime(df.index).date
        cum_pv = pd.Series(pv.values, index=df.index).groupby(day).cumsum()
        cum_v = pd.Series(df["volume"].values, index=df.index).groupby(day).cumsum().replace(0, np.nan)
        return {"vwap": (cum_pv / cum_v).to_numpy()}
