"""pandas_ta_classic_features.py — TA indicators not in TA-Lib (STUB).

TODO: wire into v10 / Mythos pipeline.
Source repo: https://github.com/xgboosted/pandas-ta-classic (MIT, 334 stars, 2026-05-16).
Install:   pip install pandas-ta-classic

Look-ahead safety: every pandas-ta-classic indicator is causal by
construction (rolling over past bars). .shift(1) applied before label join.

Estimated features added per ticker: ~30-40 columns of indicators NOT
covered by the existing TA-Lib integration (Aberration, AMAT, Bias, Chop,
KVO, KST, PVO, PVT, QStick, RVGI, SMI, STC, TSI, Vortex, Squeeze, RVI,
NVI/PVI, Disparity Index, BOP, CG, ER, FISHER, INERTIA, PSL, RSX, etc.).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# Indicators NOT in TA-Lib — restrict the adapter to these for true uniqueness.
_PT_CLASSIC_UNIQUE = [
    "aberration", "amat", "bias", "chop", "kvo", "kst", "pvo", "pvt",
    "qstick", "rvgi", "smi", "stc", "tsi", "vortex", "squeeze",
    "nvi", "pvi", "ebsw", "fisher", "inertia", "psl", "rsx",
    "true_range", "vhf", "willr", "cg", "er", "bop",
]


def add_pandas_ta_classic_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Add pandas-ta-classic indicators NOT already in v10's TA-Lib coverage.

    Args:
        df: DataFrame with columns open, high, low, close, volume (lowercase).
        ticker: ticker symbol (reserved for cross-sectional cache).
    """
    import pandas_ta_classic as ptac  # lazy import

    out = df.copy()
    # Strategy lets pandas-ta apply many indicators in one call.
    strategy = ptac.Strategy(
        name="v10_unique",
        ta=[{"kind": k} for k in _PT_CLASSIC_UNIQUE],
    )
    # ptac registers a DataFrame accessor via core; use df.ta.strategy(...)
    out.ta.strategy(strategy)
    # .shift(1) on every newly added column to guarantee causal merge.
    new_cols = [c for c in out.columns if c not in df.columns]
    out[new_cols] = out[new_cols].shift(1)
    return out


if __name__ == "__main__":
    print("TODO: wire pandas_ta_classic_features into v10. Diff against TA-Lib for overlap.")
