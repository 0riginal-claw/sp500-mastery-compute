"""Historical Volatility — annualized stdev of log returns, percent."""
from __future__ import annotations
import numpy as np, pandas as pd
from historical_system.indicators.base import Indicator, register

@register
class HistoricalVolatility(Indicator):
    name = "historical_volatility"
    outputs = ("hv",)
    params = {"length": 10, "bars_per_year": 252, "source": "close"}
    deps = ("close",)
    def compute(self, df, length=10, bars_per_year=252, source="close"):
        r = np.log(df[source] / df[source].shift(1))
        return {"hv": (100.0 * r.rolling(length, min_periods=length).std(ddof=0) * np.sqrt(bars_per_year)).to_numpy()}
