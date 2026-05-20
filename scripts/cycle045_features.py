# Source: /Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/claudes test/research/archive/cycle045_vil_stack_integration_2026-05-03/vil_bar_features.py
"""
Wrapper for cycle045 VILFeatureEngine streaming features.

Cycle045 introduced a pure-Python streaming VIL engine (VILFeatureEngine) that
maintains per-ticker session state across 5-min bars and emits three features
per bar: rel_vol, above_vwap, fake_bo_today.

Adapted from streaming intraday-bar engine to daily OHLCV batch computation.
Session-reset logic becomes rolling windows; intraday VWAP becomes rolling
n-day VWAP (weighted avg of typical price × volume over rolling window).

Features emitted (all .shift(1)-safe):
  c045_rel_vol_20d      : volume / rolling 20-day mean (rel_vol analogue)
  c045_rolling_vwap_20d : rolling 20-day VWAP — (typ_px * vol).rolling(20).sum() / vol.rolling(20).sum()
  c045_above_rolling_vwap : bool — prior close > prior rolling VWAP
  c045_fake_bo_day      : bool — prior day's high broke prior 10-day high by >0.1% then closed below it
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CYCLE045_FEATURE_NAMES: list[str] = [
    "c045_rel_vol_20d",
    "c045_rolling_vwap_20d",
    "c045_above_rolling_vwap",
    "c045_fake_bo_day",
]

_REL_VOL_WIN = 20
_VWAP_WIN = 20
_FAKE_BO_WIN = 10
_BO_THRESH = 0.001   # >0.1% break of prior N-day high counts as breakout
_REENTRY_THRESH = 0.0  # if close back below prior high, counts as fake


def _zero_fill(df: pd.DataFrame) -> pd.DataFrame:
    for col in CYCLE045_FEATURE_NAMES:
        if col not in df.columns:
            if col in ("c045_above_rolling_vwap", "c045_fake_bo_day"):
                df[col] = False
            else:
                df[col] = 0.0
    return df


def add_cycle045_features(df: pd.DataFrame, ticker: str = None) -> pd.DataFrame:
    """Append cycle045 streaming-VIL features to daily OHLCV df. Idempotent.

    Requires 'close'. 'high', 'low', 'volume' improve quality.
    All output is .shift(1)-safe.
    """
    if df is None or len(df) == 0:
        return df
    if all(c in df.columns for c in CYCLE045_FEATURE_NAMES):
        return df
    if "close" not in df.columns:
        return _zero_fill(df)

    close = pd.to_numeric(df["close"], errors="coerce").astype(float)
    high = pd.to_numeric(df["high"], errors="coerce").astype(float) if "high" in df.columns else close
    low = pd.to_numeric(df["low"], errors="coerce").astype(float) if "low" in df.columns else close
    volume = pd.to_numeric(df["volume"], errors="coerce").astype(float) if "volume" in df.columns else pd.Series(1.0, index=df.index)

    # Relative volume: volume / rolling 20-day mean
    vol_mean = volume.rolling(_REL_VOL_WIN, min_periods=5).mean().replace(0, np.nan)
    rel_vol_20d = (volume / vol_mean).fillna(1.0)

    # Rolling 20-day VWAP: cumulative-sum style via rolling window
    typ = (high + low + close) / 3.0
    vwap_num = (typ * volume).rolling(_VWAP_WIN, min_periods=5).sum()
    vwap_den = volume.rolling(_VWAP_WIN, min_periods=5).sum().replace(0, np.nan)
    rolling_vwap = (vwap_num / vwap_den).fillna(close)

    above_rolling_vwap = close > rolling_vwap

    # Fake-breakout day: today's high > prior 10-day rolling high by >0.1%
    # AND today's close is back below that prior rolling high
    prior_10d_high = high.shift(1).rolling(_FAKE_BO_WIN, min_periods=3).max()
    broke_out = high > prior_10d_high * (1.0 + _BO_THRESH)
    closed_back = close <= prior_10d_high
    fake_bo_day = broke_out & closed_back

    if "c045_rel_vol_20d" not in df.columns:
        df["c045_rel_vol_20d"] = rel_vol_20d.shift(1).fillna(1.0).values
    if "c045_rolling_vwap_20d" not in df.columns:
        df["c045_rolling_vwap_20d"] = rolling_vwap.shift(1).ffill().bfill().values
    if "c045_above_rolling_vwap" not in df.columns:
        df["c045_above_rolling_vwap"] = above_rolling_vwap.shift(1).fillna(False).astype(bool).values
    if "c045_fake_bo_day" not in df.columns:
        # fake_bo_day uses prior-bar data already (prior_10d_high uses high.shift(1))
        # shift(1) makes it: "was yesterday a fake breakout day?"
        df["c045_fake_bo_day"] = fake_bo_day.shift(1).fillna(False).astype(bool).values

    return df
