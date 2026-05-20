"""
vpin_features.py — Wave M-1 #1: VPIN (Easley-Lopez de Prado-O'Hara 2012 RFS)
using TRUE intraday 1-min bars (vs. the daily-OHLCV BVC approximation in
vpin_50bucket_features.py).

Approach
--------
1. Load 1-min RTH bars for ticker via _load_1min().
2. Per minute: classify volume into buy/sell via Bulk Volume Classification
   (BVC, López de Prado): P_buy = Φ((Δp / σ_Δp)); buy_vol = P_buy·V,
   sell_vol = (1 − P_buy)·V.
3. Aggregate into 50 equal-volume buckets per day; per-bucket
   imbalance = |V_B − V_S| / V_bucket. VPIN_eod = mean of last-50-bucket
   imbalances at the END of session.
4. Roll up to per-DATE features and assign to df_daily indexed by that date.
5. Apply .shift(1) — feature at row t uses ONLY prior-day intraday bars.

Features added (5)
------------------
  vpin_eod          — end-of-day 50-bucket VPIN [0, 1]
  vpin_max_today    — intraday max of rolling VPIN
  vpin_zscore_60d   — 60-day z-score of vpin_eod
  vpin_above_p95    — 1 if vpin_eod above 252d-rolling 95th percentile
  vpin_buy_frac_eod — end-of-day buy-volume fraction

License : MIT (own impl). Ref: Easley/López de Prado/O'Hara, RFS 25(5) 2012.
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)

VPIN_FEATURE_NAMES: list[str] = [
    "vpin_eod",
    "vpin_max_today",
    "vpin_zscore_60d",
    "vpin_above_p95",
    "vpin_buy_frac_eod",
]

_N_BUCKETS = 50
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
    """Load 1-min bars; try alpaca_features cache then claudes-test fallback.

    Returns an ET-tz-aware DataFrame with columns at least [open, high, low,
    close, volume]; or empty DataFrame if neither path exists.
    """
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
                warnings.warn(f"vpin_features: failed to load {path!r}: {exc}")
                continue
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


def _vpin_one_day(day_bars: pd.DataFrame) -> tuple[float, float, float]:
    """Compute (vpin_eod, vpin_max, buy_frac_eod) for a single trading day."""
    if len(day_bars) < 30:
        return (np.nan, np.nan, np.nan)
    close = day_bars["close"].astype(float).values
    volume = day_bars["volume"].astype(float).clip(lower=0).values
    if volume.sum() <= 0:
        return (np.nan, np.nan, np.nan)
    dp = np.diff(close, prepend=close[0])
    sigma = float(np.nanstd(dp))
    if sigma <= 0 or not np.isfinite(sigma):
        return (np.nan, np.nan, np.nan)
    p_buy = norm.cdf(dp / sigma)
    buy_vol = p_buy * volume
    sell_vol = (1.0 - p_buy) * volume

    total_vol = float(volume.sum())
    v_per_bucket = total_vol / _N_BUCKETS
    if v_per_bucket <= 0:
        return (np.nan, np.nan, np.nan)

    # Assign each minute's volume into volume-buckets sequentially.
    cum_vol = np.cumsum(volume)
    bucket_idx = np.minimum(
        (cum_vol / v_per_bucket).astype(int), _N_BUCKETS - 1
    )
    bucket_buy = np.bincount(bucket_idx, weights=buy_vol, minlength=_N_BUCKETS)
    bucket_sell = np.bincount(bucket_idx, weights=sell_vol, minlength=_N_BUCKETS)
    bucket_tot = bucket_buy + bucket_sell
    with np.errstate(divide="ignore", invalid="ignore"):
        bucket_imb = np.where(
            bucket_tot > 0,
            np.abs(bucket_buy - bucket_sell) / bucket_tot,
            np.nan,
        )
    valid = bucket_imb[np.isfinite(bucket_imb)]
    if len(valid) == 0:
        return (np.nan, np.nan, np.nan)

    vpin_eod = float(np.nanmean(valid))
    # rolling vpin per-bucket window=50 truncated by available bins
    # use expanding max over cumulative means of valid as a max-vpin-today proxy
    cummeans = np.array(
        [np.nanmean(valid[: i + 1]) for i in range(len(valid))]
    )
    vpin_max = float(np.nanmax(cummeans))
    buy_frac_eod = float(buy_vol.sum() / total_vol)
    return (vpin_eod, vpin_max, buy_frac_eod)


def add_vpin_features(
    df_daily: pd.DataFrame,
    ticker: str,
    n_buckets: int = _N_BUCKETS,
) -> pd.DataFrame:
    """Append VPIN intraday features (5 cols) to df_daily.

    All outputs are .shift(1)-safe: row t uses ONLY bars from day t−1.
    Zero-fills gracefully when 1-min cache is missing for ticker.
    """
    df = df_daily.copy()
    for c in VPIN_FEATURE_NAMES:
        if c not in df.columns:
            df[c] = 0.0

    bars = _load_1min(ticker)
    if bars.empty or "close" not in bars.columns or "volume" not in bars.columns:
        logger.warning(
            "[vpin_features] no 1-min bars for %s — zero-filling", ticker
        )
        return df

    # Restrict to RTH 09:30–15:59 ET
    try:
        rth = bars.between_time("09:30", "15:59")
    except Exception:  # noqa: BLE001
        rth = bars

    by_day = rth.groupby(rth.index.normalize().date)
    rows: list[tuple] = []
    for d, day_bars in by_day:
        vpin_eod, vpin_max, buy_frac = _vpin_one_day(day_bars)
        rows.append((pd.Timestamp(d), vpin_eod, vpin_max, buy_frac))
    if not rows:
        return df

    vpin_df = pd.DataFrame(
        rows, columns=["date", "vpin_eod_raw", "vpin_max_raw", "vpin_buy_frac_raw"]
    ).set_index("date").sort_index()

    # 60d z-score + 252d p95 indicator
    eod = vpin_df["vpin_eod_raw"]
    roll60 = eod.rolling(60, min_periods=10)
    z60 = ((eod - roll60.mean()) / roll60.std().replace(0, np.nan)).fillna(0.0)
    roll252_p95 = eod.rolling(252, min_periods=20).quantile(0.95)
    above_p95 = (eod > roll252_p95).astype(float).fillna(0.0)

    # Build per-day Series, shift(1), reindex to df_daily index
    feats = pd.DataFrame(
        {
            "vpin_eod": eod,
            "vpin_max_today": vpin_df["vpin_max_raw"],
            "vpin_zscore_60d": z60,
            "vpin_above_p95": above_p95,
            "vpin_buy_frac_eod": vpin_df["vpin_buy_frac_raw"],
        }
    ).shift(1).fillna(0.0)

    # Merge by date.  df.index may be DatetimeIndex (daily) or RangeIndex with
    # a 'date' column; handle both.
    if isinstance(df.index, pd.DatetimeIndex):
        join_idx = pd.DatetimeIndex(df.index.normalize())
        for col in VPIN_FEATURE_NAMES:
            df[col] = feats[col].reindex(join_idx).values
    elif "date" in df.columns:
        join_idx = pd.DatetimeIndex(pd.to_datetime(df["date"]).dt.normalize())
        for col in VPIN_FEATURE_NAMES:
            df[col] = feats[col].reindex(join_idx).values
    else:
        logger.warning(
            "[vpin_features] cannot align to df index (no date col) — zeroing"
        )
    for c in VPIN_FEATURE_NAMES:
        df[c] = df[c].fillna(0.0).astype(float)
    logger.info(
        "[vpin_features] %s: vpin_eod_mean=%.4f over %d days",
        ticker, float(feats["vpin_eod"].mean()), len(feats),
    )
    return df
