"""Chop Zone — 0..8 discrete state based on EMA slope angle."""
from __future__ import annotations
import numpy as np
from historical_system.indicators.base import Indicator, register
from historical_system.indicators import kernels as K

@register
class ChopZone(Indicator):
    name = "chop_zone"
    outputs = ("chop_zone",)
    params = {"length": 30, "source": "close"}
    deps = ("close",)
    def compute(self, df, length=30, source="close"):
        e = K.ema(df[source], length)
        slope = np.degrees(np.arctan(np.diff(e, prepend=e[0])))
        # Bucket angle into 0..8 (coarser bins as angle grows)
        buckets = np.digitize(slope, [-60, -30, -15, -5, 5, 15, 30, 60]).astype(np.float64)
        return {"chop_zone": buckets}
