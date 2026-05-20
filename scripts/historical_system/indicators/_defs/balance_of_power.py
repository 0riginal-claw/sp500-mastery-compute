"""Balance of Power — (close - open) / (high - low)."""
from __future__ import annotations
import numpy as np
from historical_system.indicators.base import Indicator, register

@register
class BalanceOfPower(Indicator):
    name = "balance_of_power"
    outputs = ("bop",)
    params = {}
    deps = ("open", "high", "low", "close")
    def compute(self, df):
        rng = (df["high"] - df["low"]).replace(0, np.nan)
        return {"bop": ((df["close"] - df["open"]) / rng).to_numpy()}
