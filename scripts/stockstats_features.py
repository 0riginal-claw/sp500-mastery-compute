"""stockstats_features.py — Stockstats TA indicators wrapper (STUB).

TODO: wire into v10 / Mythos pipeline.
Source repo: external-repos/stockstats (BSD-3-Clause).
Install:   pip install stockstats

Look-ahead safety: stockstats indicators are causal rolling-window
computations over past bars. .shift(1) applied to every emitted column
before label join (enforced below).

Estimated features added per ticker: ~30 columns covering moving averages
(SMA/EMA/SMMA/TEMA/KAMA/VWMA), momentum (StochRSI/PPO/KDJ/CMO/KST/Coppock/AO),
trend (Supertrend/Aroon/Ichimoku/DMI+/ADX/TRIX/WT), volatility (CCI/WR/CHOP/
KER/Z-Score/MAD/PGO), and volume (VR/MFI/PVO). Targets indicators NOT
already covered by existing TA-Lib + pandas-ta-classic wrappers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# Indicators chosen to avoid overlap with existing TA-Lib / pandas-ta-classic
# coverage already wired in v10. Names follow stockstats accessor syntax.
_STOCKSTATS_UNIQUE = [
    "kama", "tema", "smma_5", "supertrend", "supertrend_ub", "supertrend_lb",
    "aroon", "ichimoku", "trix", "wt1", "wt2",
    "stochrsi", "kdjk", "kdjd", "kdjj", "cmo", "kst", "coppock", "ao",
    "cci", "wr_14", "chop", "ker", "mad", "pgo",
    "vr", "pvo", "mfi_14",
]


def add_stockstats_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Add stockstats indicators not already covered by existing wrappers.

    Args:
        df: DataFrame with columns open, high, low, close, volume (lowercase).
        ticker: ticker symbol (reserved for cross-sectional cache).
    """
    try:
        from stockstats import wrap as _ss_wrap  # lazy import
    except Exception:
        # Library unavailable — emit zero-fill placeholder columns so the
        # consumer daemon does not crash. Replace with real values once
        # `pip install stockstats` runs in the prod venv.
        out = df.copy()
        for c in _STOCKSTATS_UNIQUE:
            out[f"stockstats_{c}"] = 0.0
        return out

    out = df.copy()
    sdf = _ss_wrap(out.copy())
    for ind in _STOCKSTATS_UNIQUE:
        try:
            col = sdf[ind]
            out[f"stockstats_{ind}"] = col.astype(float).values
        except Exception:
            out[f"stockstats_{ind}"] = np.nan
    new_cols = [c for c in out.columns if c not in df.columns]
    out[new_cols] = out[new_cols].shift(1)
    return out


if __name__ == "__main__":
    print("TODO: wire stockstats_features into v10. Diff against TA-Lib / pandas-ta for overlap.")
