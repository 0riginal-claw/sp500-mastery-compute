# Source: /Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/claudes test/research/archive/cycle044_regime_split_validator_2026-05-03/volume_intelligence.py
"""
Wrapper for cycle044 Volume Intelligence Layer (VIL) features + regime classifier.

Sources:
  volume_intelligence.py  — 17-feature VIL engine (rel_vol, range_pct, body_pct, etc.)
  regime_classifier.py    — bull/chop/bear regime from 60-day basket return

Adapted from intraday 5-min bars to daily OHLCV. Intraday-specific features
(vwap_position per session, fake_breakout within session, vol_trend_5m/15m/1h)
are not meaningful on daily bars and are omitted.

Features emitted (all .shift(1)-safe):
  c044_rel_vol          : volume / rolling 20-day median (rel_vol_5min analogue)
  c044_vol_spike        : bool — rel_vol > 2.0 (vol_spike analogue)
  c044_range_pct        : (high - low) / close — intraday range proxy
  c044_body_pct         : abs(close - open) / (high - low) — candle body fraction
  c044_absorption       : bool — high vol + narrow body (rel_vol > 1.5 AND body_pct < 0.25)
  c044_vol_confirms_dir : ternary — sign(close-prev_close) × sign(rel_vol-1)
  c044_own_regime_60d   : 0=bear / 1=chop / 2=bull — 60-day own-ticker return regime
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CYCLE044_FEATURE_NAMES: list[str] = [
    "c044_rel_vol",
    "c044_vol_spike",
    "c044_range_pct",
    "c044_body_pct",
    "c044_absorption",
    "c044_vol_confirms_dir",
    "c044_own_regime_60d",
]

_REL_VOL_WIN = 20
_REGIME_WIN = 60
_BULL_THRESH = 0.10
_BEAR_THRESH = -0.10


def _zero_fill(df: pd.DataFrame) -> pd.DataFrame:
    for col in CYCLE044_FEATURE_NAMES:
        if col not in df.columns:
            if col in ("c044_vol_spike", "c044_absorption"):
                df[col] = False
            elif col == "c044_own_regime_60d":
                df[col] = 1
            else:
                df[col] = 0.0
    return df


def add_cycle044_features(df: pd.DataFrame, ticker: str = None) -> pd.DataFrame:
    """Append cycle044 VIL daily features to daily OHLCV df. Idempotent.

    Requires 'close'. 'high', 'low', 'open', 'volume' improve quality.
    All output is .shift(1)-safe — each column uses only prior-bar data.
    """
    if df is None or len(df) == 0:
        return df
    if all(c in df.columns for c in CYCLE044_FEATURE_NAMES):
        return df
    if "close" not in df.columns:
        return _zero_fill(df)

    close = pd.to_numeric(df["close"], errors="coerce").astype(float)
    open_ = pd.to_numeric(df["open"], errors="coerce").astype(float) if "open" in df.columns else close
    high = pd.to_numeric(df["high"], errors="coerce").astype(float) if "high" in df.columns else close
    low = pd.to_numeric(df["low"], errors="coerce").astype(float) if "low" in df.columns else close
    volume = pd.to_numeric(df["volume"], errors="coerce").astype(float) if "volume" in df.columns else pd.Series(1.0, index=df.index)

    # Relative volume: bar volume / rolling 20-day median
    vol_median = volume.rolling(_REL_VOL_WIN, min_periods=5).median().replace(0, np.nan)
    rel_vol = (volume / vol_median).fillna(1.0)

    # Vol spike: rel_vol > 2.0
    vol_spike = rel_vol > 2.0

    # Range pct: (high - low) / close
    range_pct = ((high - low) / close.replace(0, np.nan)).fillna(0.0).clip(0.0, 0.5)

    # Body pct: abs(close - open) / (high - low)
    bar_range = (high - low).replace(0, np.nan)
    body_pct = ((close - open_).abs() / bar_range).fillna(0.0).clip(0.0, 1.0)

    # Absorption: high vol + narrow body
    absorption = (rel_vol > 1.5) & (body_pct < 0.25)

    # Vol-confirms-direction: sign(close_change) * sign(rel_vol - 1)
    dir_sign = np.sign(close.diff()).fillna(0)
    vol_sign = np.sign(rel_vol - 1.0).fillna(0)
    vol_confirms_dir = (dir_sign * vol_sign).fillna(0.0)

    # Own-ticker 60-day regime (simplified basket proxy: uses ticker's own return)
    ret_60d = close.pct_change(_REGIME_WIN)
    own_regime = pd.Series(1, index=close.index, dtype=int)  # default chop
    own_regime[ret_60d >= _BULL_THRESH] = 2
    own_regime[ret_60d <= _BEAR_THRESH] = 0

    if "c044_rel_vol" not in df.columns:
        df["c044_rel_vol"] = rel_vol.shift(1).fillna(1.0).values
    if "c044_vol_spike" not in df.columns:
        df["c044_vol_spike"] = vol_spike.shift(1).fillna(False).astype(bool).values
    if "c044_range_pct" not in df.columns:
        df["c044_range_pct"] = range_pct.shift(1).fillna(0.0).values
    if "c044_body_pct" not in df.columns:
        df["c044_body_pct"] = body_pct.shift(1).fillna(0.0).values
    if "c044_absorption" not in df.columns:
        df["c044_absorption"] = absorption.shift(1).fillna(False).astype(bool).values
    if "c044_vol_confirms_dir" not in df.columns:
        df["c044_vol_confirms_dir"] = vol_confirms_dir.shift(1).fillna(0.0).values
    if "c044_own_regime_60d" not in df.columns:
        df["c044_own_regime_60d"] = own_regime.shift(1).fillna(1).astype(int).values

    return df
