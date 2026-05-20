"""Pivot Points Standard — floor-trader pivots based on previous session H/L/C.

Resets daily. Returns pivot + S1..S3 + R1..R3. First session's values are NaN.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from historical_system.indicators.base import Indicator, register

@register
class PivotPointsStandard(Indicator):
    name = "pivot_points_standard"
    outputs = ("pp", "s1", "s2", "s3", "r1", "r2", "r3")
    params = {}
    deps = ("high", "low", "close")
    def compute(self, df):
        day = pd.to_datetime(df.index).date
        grp = pd.DataFrame({"h": df["high"], "l": df["low"], "c": df["close"], "day": day})
        agg = grp.groupby("day").agg(H=("h","max"), L=("l","min"), C=("c","last"))
        agg = agg.shift(1)
        pp = (agg["H"] + agg["L"] + agg["C"]) / 3.0
        r1 = 2 * pp - agg["L"]; s1 = 2 * pp - agg["H"]
        r2 = pp + (agg["H"] - agg["L"]); s2 = pp - (agg["H"] - agg["L"])
        r3 = agg["H"] + 2 * (pp - agg["L"]); s3 = agg["L"] - 2 * (agg["H"] - pp)
        # Map back to per-bar via day join
        look = pd.DataFrame({"day": day}, index=df.index)
        joined = look.join(agg.assign(pp=pp, r1=r1, s1=s1, r2=r2, s2=s2, r3=r3, s3=s3), on="day")
        return {"pp": joined["pp"].to_numpy(), "s1": joined["s1"].to_numpy(),
                "s2": joined["s2"].to_numpy(), "s3": joined["s3"].to_numpy(),
                "r1": joined["r1"].to_numpy(), "r2": joined["r2"].to_numpy(),
                "r3": joined["r3"].to_numpy()}
