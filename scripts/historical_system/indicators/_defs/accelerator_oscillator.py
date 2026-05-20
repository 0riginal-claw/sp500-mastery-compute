"""Accelerator Oscillator — AO - SMA(AO, 5)."""
from historical_system.indicators.base import Indicator, register
from historical_system.indicators import kernels as K

@register
class AcceleratorOscillator(Indicator):
    name = "accelerator_oscillator"
    outputs = ("ac",)
    params = {"fast": 5, "slow": 34, "smooth": 5}
    deps = ("high", "low")
    def compute(self, df, fast=5, slow=34, smooth=5):
        mp = K.median_price(df)
        ao = K.sma(mp, fast) - K.sma(mp, slow)
        return {"ac": ao - K.sma(ao, smooth)}
