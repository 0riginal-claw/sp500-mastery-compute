"""
tick_imbalance_features.py — Wave M-1 #6: Lee-Ready 1991 tick imbalance.

Approach
--------
Per minute, classify buy- vs sell-initiated volume via the tick rule:
  - uptick   (close > prev_close)   → buy
  - downtick (close < prev_close)   → sell
  - zero-tick                       → inherit previous sign (carry-forward)

Per-day:
  tick_imb_eod              = (buy_vol − sell_vol) / total_vol over RTH
  tick_imb_first_hour       = same ratio restricted to 09:30–10:30 ET
  tick_imb_last_hour        = same ratio restricted to 15:00–16:00 ET
  tick_imb_5d_avg           = trailing 5d mean of tick_imb_eod
  tick_imb_first_vs_last_hour_diff = first_hour − last_hour

All outputs .shift(1)-safe: features for row t use only day t−1 intraday.

Features added (5)
------------------
  tick_imb_eod
  tick_imb_first_hour
  tick_imb_last_hour
  tick_imb_5d_avg
  tick_imb_first_vs_last_hour_diff

License : MIT (own impl). Ref: Lee & Ready, "Inferring Trade Direction from
Intraday Data", Journal of Finance 46(2) 1991.
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TICK_IMBALANCE_FEATURE_NAMES: list[str] = [
    "tick_imb_eod",
    "tick_imb_first_hour",
    "tick_imb_last_hour",
    "tick_imb_5d_avg",
    "tick_imb_first_vs_last_hour_diff",
]

_ET_TZ = "America/New_York"
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
    """Shared 1-min loader; same paths as vpin_features._load_1min."""
    for root, suffix in (
        (_CACHE_ALPACA, f"{ticker}_1min.parquet"),
        (_CACHE_CLAUDES, f"{ticker}.parquet"),
    ):
        path = os.path.join(root, suffix)
        if os.path.exists(path):
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
                warnings.warn(f"tick_imbalance: failed to load {path!r}: {exc}")
                continue
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


def _classify_lee_ready(close: np.ndarray) -> np.ndarray:
    """Return +1/-1 sign per bar via tick rule with carry-forward on zero-tick."""
    if len(close) == 0:
        return np.array([], dtype=float)
    sign = np.zeros(len(close), dtype=float)
    last = 1.0  # neutral-positive prior; first bar defaults to +1
    for i in range(len(close)):
        if i == 0:
            sign[i] = last
            continue
        diff = close[i] - close[i - 1]
        if diff > 0:
            last = 1.0
        elif diff < 0:
            last = -1.0
        # else: carry-forward last
        sign[i] = last
    return sign


def _imb_ratio(buy_vol: float, sell_vol: float) -> float:
    tot = buy_vol + sell_vol
    return float((buy_vol - sell_vol) / tot) if tot > 0 else 0.0


def add_tick_imbalance_features(
    df_daily: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    """Append 5 tick-imbalance features to df_daily.

    .shift(1)-safe. Zero-fills if 1-min cache missing.
    """
    df = df_daily.copy()
    for c in TICK_IMBALANCE_FEATURE_NAMES:
        if c not in df.columns:
            df[c] = 0.0

    bars = _load_1min(ticker)
    if bars.empty or "close" not in bars.columns or "volume" not in bars.columns:
        logger.warning(
            "[tick_imbalance] no 1-min bars for %s — zero-filling", ticker
        )
        return df

    try:
        rth = bars.between_time("09:30", "15:59")
    except Exception:  # noqa: BLE001
        rth = bars
    if rth.empty:
        return df

    rows: list[tuple] = []
    by_day = rth.groupby(rth.index.normalize().date)
    for d, day_bars in by_day:
        if len(day_bars) < 30:
            continue
        close = day_bars["close"].astype(float).values
        vol = day_bars["volume"].astype(float).clip(lower=0).values
        sign = _classify_lee_ready(close)
        buy_vol = float(((sign > 0) * vol).sum())
        sell_vol = float(((sign < 0) * vol).sum())
        imb_eod = _imb_ratio(buy_vol, sell_vol)

        # First / last hour slices
        try:
            first_h = day_bars.between_time("09:30", "10:30")
            last_h = day_bars.between_time("15:00", "16:00")
        except Exception:  # noqa: BLE001
            first_h = day_bars.iloc[: min(60, len(day_bars))]
            last_h = day_bars.iloc[-min(60, len(day_bars)) :]

        def _slice_imb(slc: pd.DataFrame) -> float:
            if len(slc) < 5:
                return 0.0
            cl = slc["close"].astype(float).values
            v = slc["volume"].astype(float).clip(lower=0).values
            s = _classify_lee_ready(cl)
            return _imb_ratio(
                float(((s > 0) * v).sum()), float(((s < 0) * v).sum())
            )

        imb_first = _slice_imb(first_h)
        imb_last = _slice_imb(last_h)
        rows.append((pd.Timestamp(d), imb_eod, imb_first, imb_last))
    if not rows:
        return df

    tdf = pd.DataFrame(
        rows, columns=["date", "tick_imb_eod", "tick_imb_first_hour", "tick_imb_last_hour"]
    ).set_index("date").sort_index()
    tdf["tick_imb_5d_avg"] = tdf["tick_imb_eod"].rolling(5, min_periods=1).mean()
    tdf["tick_imb_first_vs_last_hour_diff"] = (
        tdf["tick_imb_first_hour"] - tdf["tick_imb_last_hour"]
    )
    feats = tdf[TICK_IMBALANCE_FEATURE_NAMES].shift(1).fillna(0.0)

    if isinstance(df.index, pd.DatetimeIndex):
        join_idx = pd.DatetimeIndex(df.index.normalize())
    elif "date" in df.columns:
        join_idx = pd.DatetimeIndex(pd.to_datetime(df["date"]).dt.normalize())
    else:
        logger.warning(
            "[tick_imbalance] cannot align to df index (no date col) — zeroing"
        )
        return df
    for col in TICK_IMBALANCE_FEATURE_NAMES:
        df[col] = feats[col].reindex(join_idx).values
        df[col] = df[col].fillna(0.0).astype(float)
    logger.info(
        "[tick_imbalance] %s: tick_imb_eod_mean=%.4f over %d days",
        ticker, float(feats["tick_imb_eod"].mean()), len(feats),
    )
    return df
