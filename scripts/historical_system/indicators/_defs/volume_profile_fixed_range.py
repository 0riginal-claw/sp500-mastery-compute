"""Volume Profile (Fixed Range) — histogram of volume by price bucket over a window.

Returns a non-timeseries dict: (bin_edges, bin_volumes, poc_price, va_high, va_low).
Because this is non-timeseries, it's marked non_timeseries=True and won't be joined
into the main df. Strategies using VP inspect its return directly.
"""
from __future__ import annotations
import numpy as np
from historical_system.indicators.base import Indicator, register

@register
class VolumeProfileFixedRange(Indicator):
    name = "volume_profile_fixed_range"
    outputs = ("vp_edges", "vp_volumes", "vp_poc", "vp_va_high", "vp_va_low")
    params = {"bins": 24, "va_pct": 0.70}
    deps = ("close", "volume")
    non_timeseries = True
    def compute(self, df, bins=24, va_pct=0.70):
        p = df["close"].to_numpy(); v = df["volume"].to_numpy(dtype=np.float64)
        if len(p) == 0 or v.sum() == 0:
            return {"vp_edges": np.array([]), "vp_volumes": np.array([]),
                    "vp_poc": np.nan, "vp_va_high": np.nan, "vp_va_low": np.nan}
        edges = np.linspace(p.min(), p.max(), bins + 1)
        idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
        vols = np.zeros(bins)
        for i, vi in zip(idx, v): vols[i] += vi
        poc = (edges[vols.argmax()] + edges[vols.argmax() + 1]) / 2.0
        target = vols.sum() * va_pct
        order = np.argsort(-vols)
        chosen = set()
        running = 0.0
        for j in order:
            chosen.add(j); running += vols[j]
            if running >= target: break
        va_high = edges[max(chosen) + 1]
        va_low = edges[min(chosen)]
        return {"vp_edges": edges, "vp_volumes": vols,
                "vp_poc": poc, "vp_va_high": va_high, "vp_va_low": va_low}
