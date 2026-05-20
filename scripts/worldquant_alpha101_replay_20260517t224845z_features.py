"""
worldquant_alpha101_replay_20260517t224845z_features.py
=======================================================
WorldQuant Alpha-101 replay feature (single alpha from Zura Kakushadze's
"101 Formulaic Alphas" paper, via github:lvlh2/alpha101, MIT-licensed).

NO-LOOKAHEAD AUDIT
------------------
All raw OHLCV inputs are shifted by 1 bar before any computation begins
(see `_shift1` assignment below). Every rolling window therefore looks back
over *prior completed bars only* — no current-bar price or volume is ever
referenced in the output column.

Derived quantities:
  raw_alpha101  = (close_t-1 - open_t-1) / (high_t-1 - low_t-1 + 1e-6)
                  This is Alpha#101 from the WorldQuant 101 paper:
                  ((CLOSE - OPEN) / ((HIGH - LOW) + .001))
                  Pure intra-bar candle direction / body-to-shadow ratio.
                  Shift-1 applied before division — shift-1 safe by construction.

  wq101_replay_alpha101_z21: 21-bar rolling z-score of raw_alpha101.
                  Window is entirely over already-shifted (t-1) values,
                  so the z-score output at bar t references only bars ≤ t-1.

No paid API required; no ticker-level network call; pure pandas/numpy.
License: MIT (WorldQuant Alpha-101 reference implementation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Canonical name for the one output column produced by this module.
WQ101_REPLAY_FEATURE_NAMES: list[str] = ["wq101_replay_alpha101_z21"]
WQ101_REPLAY_FEATURE_COUNT: int = len(WQ101_REPLAY_FEATURE_NAMES)


def compute_worldquant_alpha101_replay_20260517t224845z_features(
    df: pd.DataFrame,
    ticker: str | None = None,
) -> pd.DataFrame:
    """Compute WorldQuant Alpha#101 replay feature and append to *df*.

    Alpha#101 from Kakushadze (2016): ((CLOSE - OPEN) / ((HIGH - LOW) + 0.001))
    Measures intra-session directional conviction relative to the candle range.
    Values near +1 → strong bullish close; near -1 → strong bearish close.

    The z-scored variant (`wq101_replay_alpha101_z21`) normalises over a 21-bar
    lookback so the signal is zero-mean and unit-variance within recent history,
    which is more useful as an XGBoost feature than the raw ratio.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame indexed by timestamp (DatetimeIndex). Must contain
        columns ``close``, ``open``, ``high``, ``low`` (case-insensitive not
        assumed — lowercase expected from v9 pipeline).
    ticker : str, optional
        Not used for computation; accepted for interface compatibility.

    Returns
    -------
    pd.DataFrame
        *df* with column ``wq101_replay_alpha101_z21`` appended in-place.
        If the required price columns are absent, the column is zero-filled
        and a warning is issued.
    """
    col = "wq101_replay_alpha101_z21"

    # Idempotency guard — don't overwrite if already computed upstream.
    if col in df.columns:
        return df

    required = {"close", "open", "high", "low"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        import warnings
        warnings.warn(
            f"[wq101_replay] Missing OHLC columns {missing_cols}; zero-filling {col}",
            RuntimeWarning,
            stacklevel=2,
        )
        df[col] = 0.0
        return df

    # Shift-1: use prior-bar OHLC only — guarantees no lookahead.
    _c = df["close"].shift(1)
    _o = df["open"].shift(1)
    _h = df["high"].shift(1)
    _l = df["low"].shift(1)

    raw = (_c - _o) / ((_h - _l).abs() + 1e-6)
    raw = raw.clip(-3.0, 3.0)  # winsorise at ±3 to suppress outliers

    # 21-bar z-score (all over prior-shifted values — shift-1 safe).
    roll = raw.rolling(21, min_periods=10)
    z21 = (raw - roll.mean()) / (roll.std().replace(0, np.nan))
    z21 = z21.fillna(0.0)

    df[col] = z21
    return df
