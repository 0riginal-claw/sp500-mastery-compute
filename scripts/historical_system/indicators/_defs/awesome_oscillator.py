"""Awesome Oscillator — SMA(median_price,5) - SMA(median_price,34)."""
from historical_system.indicators.base import Indicator, register
from historical_system.indicators import kernels as K

@register
class AwesomeOscillator(Indicator):
    name = "awesome_oscillator"
    outputs = ("ao",)
    params = {"fast": 5, "slow": 34}
    deps = ("high", "low")
    def compute(self, df, fast=5, slow=34):
        mp = K.median_price(df)
        return {"ao": K.sma(mp, fast) - K.sma(mp, slow)}
