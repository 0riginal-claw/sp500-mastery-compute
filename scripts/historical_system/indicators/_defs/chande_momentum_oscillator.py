"""Chande Momentum Oscillator — 100 * (sumUp - sumDown) / (sumUp + sumDown)."""
from __future__ import annotations
import numpy as np, pandas as pd
from historical_system.indicators.base import Indicator, register

@register
class CMO(Indicator):
    name = "cmo"
    outputs = ("cmo",)
    params = {"length": 9, "source": "close"}
    deps = ("close",)
    def compute(self, df, length=9, source="close"):
        c = df[source].to_numpy(dtype=np.float64)
        d = np.diff(c, prepend=c[0])
        up = pd.Series(np.where(d > 0, d, 0.0)).rolling(length, min_periods=length).sum()
        dn = pd.Series(np.where(d < 0, -d, 0.0)).rolling(length, min_periods=length).sum()
        tot = up + dn
        out = np.where(tot == 0, 0.0, 100.0 * (up - dn) / tot.replace(0, np.nan))
        return {"cmo": out}
