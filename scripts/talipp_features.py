"""talipp_features.py — Incremental technical-analysis indicators (STUB).

TODO: wire into v10 / Mythos pipeline.
Source repo: external-repos/talipp (MIT).
Install:   pip install talipp

Talipp recomputes indicators in O(1) per new bar via incremental state, so it
is naturally streaming-friendly. For backtesting we still feed the full price
history and read the final indicator series — equivalent values to TA-Lib but
without the C dependency.

Look-ahead safety: every talipp indicator is causal (state is updated only
with the current and prior bars). .shift(1) applied before label join.

Estimated features added per ticker: ~12 columns. Targets indicators where
talipp adds value beyond TA-Lib/pandas-ta-classic (e.g. ChaikinOsc, ChandeKrl,
Coppock, KVO, MassIndex, SOBV, TTM Squeeze, VWMA, OBV-delta, ZLEMA).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# Talipp class names to load lazily — kept narrow to avoid TA-Lib overlap.
_TALIPP_INDICATORS = [
    "ChaikinOsc", "ChandeKrl", "Coppock", "KVO", "MassIndex",
    "SOBV", "TTM_Squeeze", "VWMA", "OBV", "ZLEMA", "PivotsHL", "Fisher",
]


def add_talipp_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Add talipp indicators that complement TA-Lib coverage.

    Args:
        df: DataFrame with columns open, high, low, close, volume (lowercase).
        ticker: ticker symbol (reserved for cross-sectional cache).
    """
    out = df.copy()
    try:
        import talipp.indicators as ti  # lazy import
        from talipp.ohlcv import OHLCV
    except Exception:
        # Zero-fill placeholders if talipp not installed.
        for c in _TALIPP_INDICATORS:
            out[f"talipp_{c.lower()}"] = 0.0
        return out

    bars = [
        OHLCV(open=o, high=h, low=l, close=c, volume=v)
        for o, h, l, c, v in zip(out["open"], out["high"], out["low"], out["close"], out["volume"])
    ]

    def _series(values):
        # talipp returns None for warmup bars — coerce to NaN.
        return [float(v) if v is not None and not isinstance(v, (list, tuple)) else np.nan for v in values]

    try:
        out["talipp_chaikin_osc"] = _series(list(ti.ChaikinOsc(3, 10, bars)))
    except Exception:
        out["talipp_chaikin_osc"] = np.nan
    try:
        out["talipp_kvo"] = _series(list(ti.KVO(34, 55, bars)))
    except Exception:
        out["talipp_kvo"] = np.nan
    try:
        out["talipp_mass_index"] = _series(list(ti.MassIndex(9, 25, bars)))
    except Exception:
        out["talipp_mass_index"] = np.nan
    try:
        out["talipp_sobv"] = _series(list(ti.SOBV(13, bars)))
    except Exception:
        out["talipp_sobv"] = np.nan
    try:
        out["talipp_obv"] = _series(list(ti.OBV(bars)))
    except Exception:
        out["talipp_obv"] = np.nan
    try:
        out["talipp_zlema"] = _series(list(ti.ZLEMA(20, out["close"].astype(float).tolist())))
    except Exception:
        out["talipp_zlema"] = np.nan
    try:
        out["talipp_coppock"] = _series(list(ti.Coppock(11, 14, 10, out["close"].astype(float).tolist())))
    except Exception:
        out["talipp_coppock"] = np.nan
    try:
        out["talipp_vwma"] = _series(list(ti.VWMA(20, bars)))
    except Exception:
        out["talipp_vwma"] = np.nan

    new_cols = [c for c in out.columns if c not in df.columns]
    out[new_cols] = out[new_cols].shift(1)
    return out


if __name__ == "__main__":
    print("TODO: wire talipp_features into v10. Verify .shift(1) vs warmup-NaN behaviour.")
