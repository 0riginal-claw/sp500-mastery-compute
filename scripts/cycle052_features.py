# Source: /Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/claudes test/research/archive/cycle052_vwap_refinement_2026-05-04/refine_vwap.py
"""
Wrapper for cycle052 VWAP Intelligence features (attach_vwap_v3).

Cycle052 extended the existing VWAP suite with 10 new VWAP-derived per-bar
features computed on intraday session data. This wrapper ports the full
attach_vwap_v3() logic to daily OHLCV by replacing the intraday session
cumulative VWAP with a rolling 20-day VWAP (weighted average over 20 daily bars).

Features emitted (all .shift(1)-safe):
  c052_vwap_20d         : rolling 20-day VWAP level
  c052_vwap_slope_pct   : pct change of VWAP over prior 6 days (vwap_slope_pct analogue)
  c052_vwap_dist_pct    : signed pct distance from close to VWAP ((close-vwap)/close*100)
  c052_vwap_above       : bool — close > rolling VWAP
  c052_vwap_extended_up : bool — dist_pct > 0.40% (extended above VWAP)
  c052_vwap_extended_dn : bool — dist_pct < -0.40% (extended below VWAP)
  c052_vwap_reclaim     : bool — prior close was below VWAP, current close above VWAP
  c052_vwap_reject      : bool — prior close was above VWAP, current close below VWAP
  c052_vwap_support_hold: bool — low touched within 0.1% of VWAP AND close above VWAP
  c052_vwap_resist_reject: bool — high touched within 0.1% of VWAP (from below) AND close below
  c052_vwap_chop_count  : int — number of VWAP crossings in prior 12 days
  c052_vwap_retest_below: bool — prior close above VWAP, current low dipped below, current close back above
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CYCLE052_FEATURE_NAMES: list[str] = [
    "c052_vwap_20d",
    "c052_vwap_slope_pct",
    "c052_vwap_dist_pct",
    "c052_vwap_above",
    "c052_vwap_extended_up",
    "c052_vwap_extended_dn",
    "c052_vwap_reclaim",
    "c052_vwap_reject",
    "c052_vwap_support_hold",
    "c052_vwap_resist_reject",
    "c052_vwap_chop_count",
    "c052_vwap_retest_below",
]

_VWAP_WIN = 20
_SLOPE_LOOKBACK = 6
_EXTENDED_PCT = 0.40      # >0.4% from VWAP = extended
_SUPPORT_TOUCH_PCT = 0.10  # within 0.1% of VWAP = touch
_CHOP_LOOKBACK = 12


def _zero_fill(df: pd.DataFrame) -> pd.DataFrame:
    bool_cols = {
        "c052_vwap_above", "c052_vwap_extended_up", "c052_vwap_extended_dn",
        "c052_vwap_reclaim", "c052_vwap_reject", "c052_vwap_support_hold",
        "c052_vwap_resist_reject", "c052_vwap_retest_below",
    }
    for col in CYCLE052_FEATURE_NAMES:
        if col not in df.columns:
            df[col] = False if col in bool_cols else 0.0
    return df


def add_cycle052_features(df: pd.DataFrame, ticker: str = None) -> pd.DataFrame:
    """Append cycle052 VWAP intelligence features to daily OHLCV df. Idempotent.

    Requires 'close'. 'high', 'low', 'volume' required for full feature set.
    All output is .shift(1)-safe — each column uses only prior-bar data.
    """
    if df is None or len(df) == 0:
        return df
    if all(c in df.columns for c in CYCLE052_FEATURE_NAMES):
        return df
    if "close" not in df.columns:
        return _zero_fill(df)

    close = pd.to_numeric(df["close"], errors="coerce").astype(float)
    high = pd.to_numeric(df["high"], errors="coerce").astype(float) if "high" in df.columns else close
    low = pd.to_numeric(df["low"], errors="coerce").astype(float) if "low" in df.columns else close
    volume = pd.to_numeric(df["volume"], errors="coerce").astype(float) if "volume" in df.columns else pd.Series(1.0, index=df.index)
    n = len(close)

    # Rolling 20-day VWAP
    typ = (high + low + close) / 3.0
    vwap_num = (typ * volume).rolling(_VWAP_WIN, min_periods=5).sum()
    vwap_den = volume.rolling(_VWAP_WIN, min_periods=5).sum().replace(0, np.nan)
    vwap = (vwap_num / vwap_den).fillna(close)

    # VWAP slope: pct change over SLOPE_LOOKBACK bars
    with np.errstate(divide="ignore", invalid="ignore"):
        slope_pct = np.where(
            (vwap.shift(_SLOPE_LOOKBACK) > 0) & np.isfinite(vwap.shift(_SLOPE_LOOKBACK)),
            (vwap - vwap.shift(_SLOPE_LOOKBACK)) / vwap.shift(_SLOPE_LOOKBACK) * 100,
            np.nan,
        )
    slope_pct = pd.Series(slope_pct, index=close.index).fillna(0.0)

    # Signed pct distance from close to VWAP
    with np.errstate(divide="ignore", invalid="ignore"):
        dist_pct = np.where(
            close > 0, (close - vwap) / close * 100, np.nan
        )
    dist_pct = pd.Series(dist_pct, index=close.index).fillna(0.0)

    above = close > vwap
    below = close < vwap

    # Extended
    extended_up = dist_pct > _EXTENDED_PCT
    extended_dn = dist_pct < -_EXTENDED_PCT

    # Reclaim: was below (any time in prior 20 bars) AND now above
    was_below = below.shift(1).fillna(False)
    reclaim = was_below & above

    # Reject: was above AND now below
    was_above = above.shift(1).fillna(False)
    reject = was_above & below

    # Support hold: low touched within 0.1% of VWAP AND close above VWAP
    with np.errstate(divide="ignore", invalid="ignore"):
        near_vwap_low = np.where(
            vwap > 0, np.abs(low - vwap) / vwap * 100, np.inf
        )
    near_vwap_low = pd.Series(near_vwap_low, index=close.index)
    support_hold = (near_vwap_low < _SUPPORT_TOUCH_PCT) & above

    # Resist reject: high touched near VWAP from below AND closed below
    with np.errstate(divide="ignore", invalid="ignore"):
        near_vwap_high = np.where(
            vwap > 0, np.abs(high - vwap) / vwap * 100, np.inf
        )
    near_vwap_high = pd.Series(near_vwap_high, index=close.index)
    resist_reject = (near_vwap_high < _SUPPORT_TOUCH_PCT) & below & was_below

    # VWAP chop count: crossings in last CHOP_LOOKBACK days
    cross = (above.astype(int) != above.shift(1).fillna(above).astype(int)).astype(int)
    chop_count = cross.rolling(_CHOP_LOOKBACK, min_periods=1).sum().fillna(0).astype(int)

    # Retest from below: prior above VWAP, current low < VWAP, close back >= VWAP
    retest_below = was_above & (low < vwap) & (close >= vwap)

    # Assign with shift(1) for no-lookahead
    if "c052_vwap_20d" not in df.columns:
        df["c052_vwap_20d"] = vwap.shift(1).ffill().bfill().values
    if "c052_vwap_slope_pct" not in df.columns:
        df["c052_vwap_slope_pct"] = slope_pct.shift(1).fillna(0.0).values
    if "c052_vwap_dist_pct" not in df.columns:
        df["c052_vwap_dist_pct"] = dist_pct.shift(1).fillna(0.0).values
    if "c052_vwap_above" not in df.columns:
        df["c052_vwap_above"] = above.shift(1).fillna(False).astype(bool).values
    if "c052_vwap_extended_up" not in df.columns:
        df["c052_vwap_extended_up"] = extended_up.shift(1).fillna(False).astype(bool).values
    if "c052_vwap_extended_dn" not in df.columns:
        df["c052_vwap_extended_dn"] = extended_dn.shift(1).fillna(False).astype(bool).values
    if "c052_vwap_reclaim" not in df.columns:
        # reclaim already references prior bar (was_below = below.shift(1))
        # shift(1) makes it: "did yesterday experience a reclaim event?"
        df["c052_vwap_reclaim"] = reclaim.shift(1).fillna(False).astype(bool).values
    if "c052_vwap_reject" not in df.columns:
        df["c052_vwap_reject"] = reject.shift(1).fillna(False).astype(bool).values
    if "c052_vwap_support_hold" not in df.columns:
        df["c052_vwap_support_hold"] = support_hold.shift(1).fillna(False).astype(bool).values
    if "c052_vwap_resist_reject" not in df.columns:
        df["c052_vwap_resist_reject"] = resist_reject.shift(1).fillna(False).astype(bool).values
    if "c052_vwap_chop_count" not in df.columns:
        df["c052_vwap_chop_count"] = chop_count.shift(1).fillna(0).astype(int).values
    if "c052_vwap_retest_below" not in df.columns:
        df["c052_vwap_retest_below"] = retest_below.shift(1).fillna(False).astype(bool).values

    return df
