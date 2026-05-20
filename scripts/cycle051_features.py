"""
cycle051_features.py — Wrapper for cycle051 MULTI-TIMEFRAME SR features (Wave Cycle, 2026-05-17).

v10's existing `multi_timeframe_features.py` exposes h1_*, h4_*, m5_*, m15_*
metrics (RSI / EMA / range stats) at 5/15/60/240-min timeframes. The cycle051
engine adds a complementary set: SR (support/resistance) levels at higher
timeframes — specifically the classic-pivot family (PP, R1, S1) and the
distance-to-pivot-percentage features.

This wrapper exposes 5 features computed PURELY from daily OHLCV (no intraday
cache needed for the daily-pivot family):
  - sr_1day_pp                  : classic floor-trader pivot point
                                  = (prior_high + prior_low + prior_close) / 3
  - sr_1day_r1                  : R1 = 2*pp - prior_low
  - sr_1day_s1                  : S1 = 2*pp - prior_high
  - sr_dist_1day_pp_pct         : abs(close - pp) / close * 100
  - sr_above_1day_pp            : binary (close > pp)

All five are .shift(1)-safe by construction (pp/r1/s1 are derived from
prior_high/prior_low/prior_close — i.e. yesterday's session, available before
today's open). The 15-min / 60-min swing-pivot features from cycle051 require
intraday 1-min bars + a per-day cache layout — those are NOT included here
because they would require an entire intraday resampling layer just for 2-3
extra features. The daily-pivot family captures the dominant SR signal.

The cycle051 source at
`research/archive/cycle051_multi_tf_features_2026-05-03/multi_tf_features.py`
remains unmodified — this wrapper re-implements the daily pivot logic
(compute_classic_pivot) verbatim against v10's daily DataFrame.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CYCLE051_FEATURE_NAMES: list[str] = [
    "sr_1day_pp",
    "sr_1day_r1",
    "sr_1day_s1",
    "sr_dist_1day_pp_pct",
    "sr_above_1day_pp",
]


def _zero_fill(df: pd.DataFrame) -> pd.DataFrame:
    for col in CYCLE051_FEATURE_NAMES:
        if col not in df.columns:
            if col == "sr_above_1day_pp":
                df[col] = 0
            else:
                df[col] = 0.0
    return df


def add_cycle051_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Append 5 daily-pivot SR features to df (idempotent, .shift(1)-safe).

    Requires high, low, close columns. Without high/low the wrapper zero-fills.
    """
    if df is None or len(df) == 0:
        return df
    if all(c in df.columns for c in CYCLE051_FEATURE_NAMES):
        return df
    if not all(c in df.columns for c in ("high", "low", "close")):
        return _zero_fill(df)

    high = pd.to_numeric(df["high"], errors="coerce").astype(float)
    low = pd.to_numeric(df["low"], errors="coerce").astype(float)
    close = pd.to_numeric(df["close"], errors="coerce").astype(float)

    # Prior session HLC
    ph = high.shift(1)
    pl = low.shift(1)
    pc = close.shift(1)

    pp = (ph + pl + pc) / 3.0
    r1 = 2.0 * pp - pl
    s1 = 2.0 * pp - ph

    dist_pct = (np.abs(close - pp) / close.replace(0, np.nan) * 100.0).fillna(0.0)
    above_pp = (close > pp).astype("int8")
    # mask leading NaN rows to 0 (no prior session yet)
    valid = ph.notna()
    pp = pp.where(valid, 0.0).fillna(0.0)
    r1 = r1.where(valid, 0.0).fillna(0.0)
    s1 = s1.where(valid, 0.0).fillna(0.0)
    dist_pct = dist_pct.where(valid, 0.0).fillna(0.0)
    above_pp = above_pp.where(valid, 0).fillna(0).astype(int)

    if "sr_1day_pp" not in df.columns:
        df["sr_1day_pp"] = pp.astype(float).values
    if "sr_1day_r1" not in df.columns:
        df["sr_1day_r1"] = r1.astype(float).values
    if "sr_1day_s1" not in df.columns:
        df["sr_1day_s1"] = s1.astype(float).values
    if "sr_dist_1day_pp_pct" not in df.columns:
        df["sr_dist_1day_pp_pct"] = dist_pct.astype(float).clip(0.0, 100.0).values
    if "sr_above_1day_pp" not in df.columns:
        df["sr_above_1day_pp"] = above_pp.astype(int).values
    return df


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    tk = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    idx = pd.date_range(end=pd.Timestamp.utcnow().date(), periods=80, freq="B")
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(0.05, 1.2, len(idx)))
    high = close + np.abs(rng.normal(0, 0.8, len(idx)))
    low = close - np.abs(rng.normal(0, 0.8, len(idx)))
    demo = pd.DataFrame({"high": high, "low": low, "close": close}, index=idx)
    out = add_cycle051_features(demo, tk)
    print(f"In cols: 3  Out cols: {out.shape[1]}")
    print(out[CYCLE051_FEATURE_NAMES].tail(5).to_string())
