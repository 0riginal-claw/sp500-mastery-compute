"""Average Price — (O+H+L+C)/4."""
from historical_system.indicators.base import Indicator, register
from historical_system.indicators import kernels as K

@register
class AveragePrice(Indicator):
    name = "average_price"
    outputs = ("average_price",)
    params = {}
    deps = ("open", "high", "low", "close")
    def compute(self, df):
        return {"average_price": K.average_price(df)}
