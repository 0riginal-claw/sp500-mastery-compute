"""Chaikin Money Flow — rolling-sum(MFV) / rolling-sum(volume)."""
from __future__ import annotations
import numpy as np, pandas as pd
from historical_system.indicators.base import Indicator, register

@register
class ChaikinMoneyFlow(Indicator):
    name = "chaikin_money_flow"
    outputs = ("cmf",)
    params = {"length": 20}
    deps = ("high", "low", "close", "volume")
    def compute(self, df, length=20):
        h = df["high"]; l = df["low"]; c = df["close"]; v = df["volume"]
        rng = (h - l).replace(0, np.nan)
        mfv = ((c - l) - (h - c)) / rng * v
        num = mfv.rolling(length, min_periods=length).sum()
        den = v.rolling(length, min_periods=length).sum().replace(0, np.nan)
        return {"cmf": (num / den).to_numpy()}
