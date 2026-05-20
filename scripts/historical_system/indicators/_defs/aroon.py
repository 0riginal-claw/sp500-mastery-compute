"""Aroon — 100 * (n - bars since highest high)/n and similar for low.

Window is (n+1) bars. ``bars_since_extreme`` is measured from the current
bar backward: if the extreme is on the current bar, bars_since = 0 and
Aroon = 100; if the extreme is ``n`` bars ago, bars_since = n and Aroon = 0.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from historical_system.indicators.base import Indicator, register

@register
class Aroon(Indicator):
    name = "aroon"
    outputs = ("aroon_up", "aroon_down", "aroon_osc")
    params = {"length": 14}
    deps = ("high", "low")
    def compute(self, df, length=14):
        win = length + 1
        def _bars_since(arr, argfn):
            # Rolling window of (length+1) bars. ``argfn`` returns the index
            # (0..length) of the extreme within the window. The most recent
            # bar is at position ``length``, so ``bars_since = length - idx``.
            s = pd.Series(arr)
            return s.rolling(win, min_periods=win).apply(
                lambda w: length - int(argfn(w.to_numpy())), raw=False
            ).to_numpy()
        up_bs = _bars_since(df["high"].to_numpy(), np.argmax)
        down_bs = _bars_since(df["low"].to_numpy(), np.argmin)
        au = 100.0 * (length - up_bs) / length
        ad = 100.0 * (length - down_bs) / length
        return {"aroon_up": au, "aroon_down": ad, "aroon_osc": au - ad}
