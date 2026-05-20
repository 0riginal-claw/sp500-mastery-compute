"""Chaikin Oscillator — EMA3(AD) - EMA10(AD)."""
from __future__ import annotations
import numpy as np
from historical_system.indicators.base import Indicator, register
from historical_system.indicators import kernels as K

@register
class ChaikinOscillator(Indicator):
    name = "chaikin_oscillator"
    outputs = ("chaikin_osc",)
    params = {"fast": 3, "slow": 10}
    deps = ("high", "low", "close", "volume")
    def compute(self, df, fast=3, slow=10):
        h = df["high"].to_numpy(); l = df["low"].to_numpy(); c = df["close"].to_numpy(); v = df["volume"].to_numpy()
        rng = h - l
        mfm = np.where(rng == 0, 0.0, ((c - l) - (h - c)) / rng)
        ad = (mfm * v).cumsum()
        return {"chaikin_osc": K.ema(ad, fast) - K.ema(ad, slow)}
