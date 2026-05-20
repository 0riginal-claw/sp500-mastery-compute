"""Exponential Moving Average — TradingView seed (EMA[0]=x[0])."""
from __future__ import annotations
from historical_system.indicators.base import Indicator, register
from historical_system.indicators import kernels as K

@register
class EMA(Indicator):
    name = "ema"
    outputs = ("ema",)
    params = {"length": 9, "source": "close"}
    deps = ("close",)
    def compute(self, df, length=9, source="close"):
        return {"ema": K.ema(df[source], length)}
