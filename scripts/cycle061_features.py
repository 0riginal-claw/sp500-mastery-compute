"""
cycle061_features.py — Wrapper for cycle061 TIME-OF-DAY features (Wave Cycle, 2026-05-17).

v10's existing time_of_day_features (Wave A) provides 1 column:
    time_of_day_bucket   (int 0-4 from paper-trade signal lookup or fallback).

This wrapper extends with FOUR daily-aggregate features derived from cycle061's
intraday engine + gate library. Whereas the cycle061 engine processes per-bar
5-min RTH bars, this wrapper precomputes ONE row per trading day by aggregating:

  - tod_OR_break_up_rate_5d        : 5-day rolling mean of {within-day did_break_or_up}
  - tod_OR_break_down_rate_5d      : 5-day rolling mean of {within-day did_break_or_down}
  - tod_morning_volume_share       : (OPEN+POST_OPEN volume) / day total volume, 5-day mean
  - tod_power_hour_volume_share    : POWER_HOUR (15:30-15:55 ET) volume / day total, 5-day mean

All four are .shift(1)-safe (the bar at date D consumes ONLY the rolling stats
computed from bars dated <= D-1). Implementation reads 1-min bars via the same
loader pattern that v10's vpin_features / tick_imbalance_features / etc use; if
no 1-min cache is found, zero-fills.

The cycle061 engine source files at:
    research/active/cycle061_time_of_day/engine/tod_features.py
    research/active/cycle061_time_of_day/gates/tod_gates.py
remain unchanged — this wrapper re-implements the *daily-aggregate* slice using
the same ET-tz window definitions (09:30-10:30 OPEN, 10:30-11:30 POST_OPEN,
15:30-15:55 POWER_HOUR).
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CYCLE061_FEATURE_NAMES: list[str] = [
    "tod_OR_break_up_rate_5d",
    "tod_OR_break_down_rate_5d",
    "tod_morning_volume_share",
    "tod_power_hour_volume_share",
]

_ET_TZ = "America/New_York"
_OR_MINUTES = 15  # opening range window
_SMOOTH_WIN = 5  # 5d rolling aggregation

_CACHE_ALPACA = (
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools/s&p500-ticker-mastery/cache/alpaca_features"
)
_CACHE_CLAUDES = (
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/claudes test/data/timeframes/S&P500 5 Year Historical Data"
    "/Minutes TimeFrames/1Min_merged"
)


def _load_1min(ticker: str) -> pd.DataFrame:
    """Load 1-min bars from alpaca cache, fallback to claudes-test, then empty."""
    for root, suffix in (
        (_CACHE_ALPACA, f"{ticker}_1min.parquet"),
        (_CACHE_CLAUDES, f"{ticker}.parquet"),
    ):
        path = os.path.join(root, suffix)
        if not os.path.exists(path):
            continue
        try:
            raw = pd.read_parquet(path)
            if "timestamp" in raw.columns:
                raw = raw.set_index("timestamp")
            raw = raw.sort_index()
            if raw.index.tz is None:
                raw.index = raw.index.tz_localize("UTC")
            raw.index = raw.index.tz_convert(_ET_TZ)
            return raw
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"cycle061_features: load failed {path!r}: {exc}")
            continue
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


def _zero_fill(df: pd.DataFrame) -> pd.DataFrame:
    for col in CYCLE061_FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0.0
    return df


def _daily_aggregates(bars: pd.DataFrame) -> pd.DataFrame:
    """Build per-day stats from RTH 1-min bars. Index = trading date (date)."""
    if bars.empty:
        return pd.DataFrame()

    et = bars.copy()
    # ET hour*60 + minute
    et["hm"] = et.index.hour * 60 + et.index.minute
    open_min = 9 * 60 + 30
    close_min = 16 * 60
    power_open = 15 * 60 + 30
    power_close = 15 * 60 + 55
    morning_end = 11 * 60 + 30  # OPEN + POST_OPEN end
    or_end = open_min + _OR_MINUTES

    et = et[(et["hm"] >= open_min) & (et["hm"] < close_min)]
    if et.empty:
        return pd.DataFrame()
    et["date"] = et.index.date

    per_day: list[dict] = []
    for d, g in et.groupby("date", sort=True):
        if len(g) < 6:
            continue
        g = g.sort_index()
        # opening range high/low locked at or_end
        or_mask = g["hm"] < or_end
        if not or_mask.any():
            continue
        or_high = float(g.loc[or_mask, "high"].max())
        or_low = float(g.loc[or_mask, "low"].min())
        post_or = g[g["hm"] >= or_end]
        did_break_up = bool((post_or["close"].astype(float) > or_high).any())
        did_break_down = bool((post_or["close"].astype(float) < or_low).any())

        tot_v = float(g["volume"].astype(float).sum())
        if tot_v <= 0:
            morn_share = 0.0
            ph_share = 0.0
        else:
            morn_v = float(g.loc[g["hm"] < morning_end, "volume"].astype(float).sum())
            ph_v = float(
                g.loc[(g["hm"] >= power_open) & (g["hm"] < power_close), "volume"]
                .astype(float)
                .sum()
            )
            morn_share = morn_v / tot_v
            ph_share = ph_v / tot_v

        per_day.append(
            {
                "date": pd.Timestamp(d),
                "br_up": float(did_break_up),
                "br_dn": float(did_break_down),
                "morn_share": morn_share,
                "ph_share": ph_share,
            }
        )

    if not per_day:
        return pd.DataFrame()

    out = pd.DataFrame(per_day).set_index("date").sort_index()
    # 5d rolling means
    out["tod_OR_break_up_rate_5d"] = (
        out["br_up"].rolling(_SMOOTH_WIN, min_periods=1).mean()
    )
    out["tod_OR_break_down_rate_5d"] = (
        out["br_dn"].rolling(_SMOOTH_WIN, min_periods=1).mean()
    )
    out["tod_morning_volume_share"] = (
        out["morn_share"].rolling(_SMOOTH_WIN, min_periods=1).mean()
    )
    out["tod_power_hour_volume_share"] = (
        out["ph_share"].rolling(_SMOOTH_WIN, min_periods=1).mean()
    )
    return out[CYCLE061_FEATURE_NAMES]


def add_cycle061_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Append 4 cycle061 daily-aggregated TOD features to df.

    Idempotent + .shift(1)-safe (consumes ONLY prior-bar daily aggregates via
    merge_asof direction='backward' allow_exact_matches=False).
    Zero-fills gracefully when no 1-min cache is present.
    """
    if df is None or len(df) == 0:
        return df
    if all(c in df.columns for c in CYCLE061_FEATURE_NAMES):
        return df

    bars = _load_1min(ticker)
    if bars.empty:
        return _zero_fill(df)

    agg = _daily_aggregates(bars)
    if agg.empty:
        return _zero_fill(df)

    if isinstance(df.index, pd.DatetimeIndex):
        bar_dates = df.index
    elif "date" in df.columns:
        bar_dates = pd.DatetimeIndex(pd.to_datetime(df["date"]))
    else:
        return _zero_fill(df)
    if bar_dates.tz is not None:
        bar_dates = bar_dates.tz_convert(None)

    bar_df = pd.DataFrame(
        {"bar_date": pd.to_datetime(bar_dates.normalize()).astype("datetime64[ns]")}
    ).reset_index(drop=True)
    bar_df["__pos"] = range(len(bar_df))
    bar_sorted = bar_df.sort_values("bar_date").reset_index(drop=True)

    right = agg.reset_index().rename(columns={"date": "bar_date"})
    right["bar_date"] = pd.to_datetime(right["bar_date"]).astype("datetime64[ns]")
    right = right.sort_values("bar_date").reset_index(drop=True)

    merged = pd.merge_asof(
        bar_sorted,
        right,
        on="bar_date",
        direction="backward",
        allow_exact_matches=False,
    )
    merged = merged.sort_values("__pos").reset_index(drop=True)

    for col in CYCLE061_FEATURE_NAMES:
        if col not in df.columns:
            df[col] = merged[col].fillna(0.0).astype(float).values
    return df


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    tk = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    idx = pd.date_range(end=pd.Timestamp.utcnow().date(), periods=80, freq="B")
    demo = pd.DataFrame({"close": np.linspace(100, 110, len(idx))}, index=idx)
    out = add_cycle061_features(demo, tk)
    print(f"In cols: 1  Out cols: {out.shape[1]}")
    print(out[CYCLE061_FEATURE_NAMES].tail(3).to_string())
