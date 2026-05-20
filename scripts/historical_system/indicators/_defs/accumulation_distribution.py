"""Accumulation/Distribution Line — cumulative money-flow volume."""
from __future__ import annotations
import numpy as np
from historical_system.indicators.base import Indicator, register

@register
class AccumulationDistribution(Indicator):
    name = "accumulation_distribution"
    outputs = ("ad_line",)
    params = {}
    deps = ("high", "low", "close", "volume")
    def compute(self, df):
        h = df["high"].to_numpy(); l = df["low"].to_numpy(); c = df["close"].to_numpy(); v = df["volume"].to_numpy()
        rng = h - l
        mfm = np.where(rng == 0, 0.0, ((c - l) - (h - c)) / rng)
        return {"ad_line": (mfm * v).cumsum()}
