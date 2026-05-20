"""stumpy_features.py — Matrix Profile motif/discord/regime features (STUB).

TODO: wire into v10 / Mythos pipeline.
Source repo: https://github.com/stumpy-dev/stumpy (BSD-3, 4092 stars, 2026-05-15).
Install:   pip install stumpy

Look-ahead safety: matrix profile uses ONLY past data in the window; result
is .shift(1)-ed before merge with labels.

Estimated features added per ticker: ~6 columns (m in {10, 20, 60}).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_stumpy_features(df: pd.DataFrame, ticker: str, windows=(10, 20, 60)) -> pd.DataFrame:
    """Add Matrix-Profile motif/discord features for `ticker`.

    Args:
        df: DataFrame indexed by date with at least a 'close' column.
        ticker: ticker symbol (unused, reserved for cross-sectional cache key).
        windows: list of subsequence lengths m.

    Returns:
        df with new columns mp_motif_d{m}, mp_discord_d{m} for each m,
        all .shift(1)-safe.
    """
    import stumpy  # lazy import to avoid hard dep at import time

    out = df.copy()
    close = out["close"].astype(np.float64).values
    n = len(close)
    for m in windows:
        if n < m * 4:
            out[f"mp_motif_d{m}"] = np.nan
            out[f"mp_discord_d{m}"] = np.nan
            continue
        mp = stumpy.stump(close, m)
        # mp[:, 0] = matrix-profile distance; first n-m+1 entries
        dist = np.full(n, np.nan)
        dist[: len(mp)] = mp[:, 0].astype(np.float64)
        # motif = low distance (recurring pattern), discord = high distance (anomaly)
        out[f"mp_motif_d{m}"] = pd.Series(dist, index=out.index).rolling(m).min().shift(1)
        out[f"mp_discord_d{m}"] = pd.Series(dist, index=out.index).rolling(m).max().shift(1)
    return out


if __name__ == "__main__":
    print("TODO: wire stumpy_features into v10 pipeline. Run unit test with synthetic OHLCV.")
