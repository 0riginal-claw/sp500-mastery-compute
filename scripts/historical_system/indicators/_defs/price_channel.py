"""Price Channel — like Donchian but shifted by 1 (excludes current bar)."""
from __future__ import annotations
import pandas as pd
from historical_system.indicators.base import Indicator, register

@register
class PriceChannel(Indicator):
    name = "price_channel"
    outputs = ("price_channel_upper", "price_channel_lower")
    params = {"length": 20}
    deps = ("high", "low")
    def compute(self, df, length=20):
        up = pd.Series(df["high"]).rolling(length, min_periods=length).max().shift(1).to_numpy()
        lo = pd.Series(df["low"]).rolling(length, min_periods=length).min().shift(1).to_numpy()
        return {"price_channel_upper": up, "price_channel_lower": lo}
