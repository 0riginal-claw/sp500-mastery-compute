"""Correlation - Log — rolling Pearson of log returns."""
from __future__ import annotations
import numpy as np, pandas as pd
from historical_system.indicators.base import Indicator, register

@register
class CorrelationLog(Indicator):
    name = "correlation_log"
    outputs = ("corr_log",)
    params = {"length": 20, "source": "close", "ref": "close"}
    deps = ("close",)
    def compute(self, df, length=20, source="close", ref="close"):
        a = np.log(df[source] / df[source].shift(1))
        b = np.log(df[ref] / df[ref].shift(1))
        return {"corr_log": pd.Series(a).rolling(length, min_periods=length).corr(pd.Series(b)).to_numpy()}
