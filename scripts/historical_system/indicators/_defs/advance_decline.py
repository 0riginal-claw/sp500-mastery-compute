"""Advance/Decline — single-symbol direction count (+1 up, -1 down)
cumulative. For true breadth use a cross-symbol variant.
"""
from __future__ import annotations
import numpy as np
from historical_system.indicators.base import Indicator, register

@register
class AdvanceDecline(Indicator):
    name = "advance_decline"
    outputs = ("ad_diff", "ad_line")
    params = {}
    deps = ("close",)
    cross_symbol = True  # flagged for the future cross-symbol runner
    def compute(self, df):
        d = df["close"].diff().to_numpy()
        sign = np.where(d > 0, 1.0, np.where(d < 0, -1.0, 0.0))
        return {"ad_diff": sign, "ad_line": np.cumsum(sign)}
