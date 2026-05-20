"""Bollinger Bands — SMA ± k * stdev."""
from historical_system.indicators.base import Indicator, register
from historical_system.indicators import kernels as K

@register
class BollingerBands(Indicator):
    name = "bollinger_bands"
    outputs = ("bb_upper", "bb_basis", "bb_lower")
    params = {"length": 20, "mult": 2.0, "source": "close"}
    deps = ("close",)
    def compute(self, df, length=20, mult=2.0, source="close"):
        basis = K.sma(df[source], length)
        sd = K.rolling_std(df[source], length)
        return {"bb_upper": basis + mult * sd,
                "bb_basis": basis,
                "bb_lower": basis - mult * sd}
