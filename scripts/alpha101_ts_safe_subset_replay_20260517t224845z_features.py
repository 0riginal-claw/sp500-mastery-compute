"""
alpha101_ts_safe_subset_replay_20260517t224845z_features.py

One-feature replay of Alpha#6 from "101 Formulaic Alphas" (Kakushadze 2015).
Source inspiration: github:STHSF/alpha101 (MIT, no paid API, no human review).

--- NO-LOOKAHEAD AUDIT ---
Column references:
  open   → shifted 1 bar (open.shift(1)) before any computation
  volume → shifted 1 bar (volume.shift(1)) before any computation
Rolling correlation window (10 bars): operates on already-shifted series —
  no same-bar quantity enters the computation.
Z-score window (21 bars): rolling mean/std of the already-shifted correlation
  series — still no same-bar data.
Conclusion: SAFE — no future information can leak into any row.
---------------------------

Feature emitted (1 total):
  a101_ts_wq6_z21  — 21-bar rolling z-score of Alpha#6
                     Alpha#6 raw: -1 * correlation(open_lag1, volume_lag1, 10)
                     Neutral fill: 0.0 on NaN (first ~30 bars warm-up).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

ALPHA101_TS_SAFE_FEATURE_COUNT: int = 1
ALPHA101_TS_SAFE_FEATURE_NAMES: list[str] = ["a101_ts_wq6_z21"]


def compute_alpha101_ts_safe_subset_replay_20260517t224845z_features(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
) -> pd.DataFrame:
    """Compute Alpha#6 z-score from STHSF/alpha101 (MIT).

    Adds column 'a101_ts_wq6_z21' to df in-place and returns df.
    Gracefully zero-fills if required columns are missing.
    """
    col = "a101_ts_wq6_z21"

    if col in df.columns:
        return df

    required = {"open", "volume"}
    if not required.issubset(df.columns):
        df[col] = 0.0
        return df

    open_lag = df["open"].shift(1)
    vol_lag = df["volume"].shift(1)

    alpha6_raw = (
        -1.0
        * open_lag.rolling(10, min_periods=5).corr(vol_lag)
    )

    roll_mean = alpha6_raw.rolling(21, min_periods=10).mean()
    roll_std = alpha6_raw.rolling(21, min_periods=10).std(ddof=0)

    z = (alpha6_raw - roll_mean) / (roll_std + 1e-8)
    z = z.clip(-5.0, 5.0)

    df[col] = z.fillna(0.0)
    return df
