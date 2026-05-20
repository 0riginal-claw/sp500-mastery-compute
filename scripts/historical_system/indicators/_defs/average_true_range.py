"""Average True Range — Wilder smoothing by default."""
from historical_system.indicators.base import Indicator, register
from historical_system.indicators import kernels as K

@register
class ATR(Indicator):
    name = "atr"
    outputs = ("atr",)
    params = {"length": 14, "method": "rma"}
    deps = ("high", "low", "close")
    def compute(self, df, length=14, method="rma"):
        return {"atr": K.atr(df["high"], df["low"], df["close"], length, method)}
