"""ALMA — Gaussian-weighted window (offset=0.85, sigma=6 default)."""
from historical_system.indicators.base import Indicator, register
from historical_system.indicators import kernels as K

@register
class ALMA(Indicator):
    name = "alma"
    outputs = ("alma",)
    params = {"length": 9, "source": "close", "offset": 0.85, "sigma": 6.0}
    deps = ("close",)
    def compute(self, df, length=9, source="close", offset=0.85, sigma=6.0):
        return {"alma": K.alma(df[source], length, offset, sigma)}
