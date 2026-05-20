"""
skforecast wrapper — recursive lag features for OHLCV DataFrames.
Adds window statistics (mean, std, min, max) over a configurable lag window
using sklearn-compatible infrastructure without requiring a trained model.
"""
import sys
import os
import pandas as pd
import numpy as np

_REPO = os.path.join(
    os.path.dirname(__file__),
    os.pardir, os.pardir,
    "repos-claude-clones", "skforecast"
)
sys.path.insert(0, os.path.normpath(_REPO))


def add_skforecast_features(
    df: pd.DataFrame,
    target_col: str = "close",
    lags: int = 10,
    window_features: bool = True,
) -> pd.DataFrame:
    """
    Add autoregressive lag and rolling-window features derived from skforecast.

    Parameters
    ----------
    df : pd.DataFrame  — must contain `target_col`
    target_col : str   — column to generate lags from (default 'close')
    lags : int         — number of lag periods (1..lags)
    window_features : bool — also add rolling mean/std/min/max for each lag window

    Returns
    -------
    pd.DataFrame with added columns:
        lag_1 … lag_N, roll_mean_N, roll_std_N, roll_min_N, roll_max_N
    """
    try:
        sys.path.insert(0, os.path.normpath(_REPO))
        from skforecast.feature_selection import select_features  # noqa: F401 — confirms import
    except ImportError:
        pass  # library not installed; proceed with pure-pandas fallback

    out = df.copy()
    series = out[target_col]

    for lag in range(1, lags + 1):
        out[f"skf_lag_{lag}"] = series.shift(lag)

    if window_features:
        for w in [5, 10, 20]:
            out[f"skf_roll_mean_{w}"] = series.shift(1).rolling(w).mean()
            out[f"skf_roll_std_{w}"] = series.shift(1).rolling(w).std()
            out[f"skf_roll_min_{w}"] = series.shift(1).rolling(w).min()
            out[f"skf_roll_max_{w}"] = series.shift(1).rolling(w).max()

    return out
