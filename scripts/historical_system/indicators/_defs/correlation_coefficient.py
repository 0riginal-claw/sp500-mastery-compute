"""Correlation Coefficient — rolling Pearson of a source vs a reference series.

By default correlates the close with a reference passed via ``df['_ref']``.
For same-symbol use (price vs SMA), set ``ref=some_column``.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from historical_system.indicators.base import Indicator, register

@register
class CorrelationCoefficient(Indicator):
    name = "correlation_coefficient"
    outputs = ("corr",)
    params = {"length": 20, "source": "close", "ref": "close"}
    deps = ("close",)
    def compute(self, df, length=20, source="close", ref="close"):
        a = pd.Series(df[source])
        b = pd.Series(df[ref])
        return {"corr": a.rolling(length, min_periods=length).corr(b).to_numpy()}
