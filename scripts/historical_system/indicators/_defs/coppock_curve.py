"""Coppock Curve — WMA(10) of (ROC(14) + ROC(11))."""
from __future__ import annotations
from historical_system.indicators.base import Indicator, register
from historical_system.indicators import kernels as K

@register
class CoppockCurve(Indicator):
    name = "coppock_curve"
    outputs = ("coppock",)
    params = {"long_roc": 14, "short_roc": 11, "wma_len": 10, "source": "close"}
    deps = ("close",)
    def compute(self, df, long_roc=14, short_roc=11, wma_len=10, source="close"):
        c = df[source]
        r = 100.0 * (K.pct_change(c, long_roc) + K.pct_change(c, short_roc))
        return {"coppock": K.wma(r, wma_len)}
