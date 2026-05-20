"""Money Flow Index — RSI of volume-weighted typical price."""
from __future__ import annotations
import numpy as np, pandas as pd
from historical_system.indicators.base import Indicator, register

@register
class MoneyFlowIndex(Indicator):
    name = "money_flow_index"
    outputs = ("mfi",)
    params = {"length": 14}
    deps = ("high", "low", "close", "volume")
    def compute(self, df, length=14):
        tp = (df["high"] + df["low"] + df["close"]) / 3.0
        mf = tp * df["volume"]
        d = tp.diff()
        pos = mf.where(d > 0, 0.0)
        neg = mf.where(d < 0, 0.0)
        pos_sum = pos.rolling(length, min_periods=length).sum()
        neg_sum = neg.rolling(length, min_periods=length).sum().replace(0, np.nan)
        ratio = pos_sum / neg_sum
        return {"mfi": (100.0 - 100.0 / (1.0 + ratio)).to_numpy()}
