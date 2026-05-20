"""
volume_profile_features.py — Wave M-1 #11: Volume Profile / TPO / POC.

Approach
--------
Per day, bin the price range into N=30 buckets, compute volume-weighted
distribution. Point-of-Control (POC) = bin with max volume.  Value Area (VA)
= contiguous bins around POC containing 70% of total volume.  Close relative
to POC/VA-high/VA-low encodes mean-reversion vs trend-day character.

Features added (6)
------------------
  vp_poc_price                 — POC price (mid of POC bin) as fraction of session range
  vp_close_minus_poc_atr       — (close - POC) / ATR proxy (range/14-bar mean)
  vp_va_high                   — VA high price as fraction of session range
  vp_va_low                    — VA low price as fraction of session range
  vp_close_inside_va_indicator — 1 if close within [VA_low, VA_high]
  vp_profile_shape             — skewness of volume distribution

All outputs .shift(1)-safe — features for row t use only day t-1 intraday.

License : MIT (own impl). Ref: Steidlmayer Market Profile.
"""

from __future__ import annotations

import logging
import os
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VOLUME_PROFILE_FEATURE_NAMES: list[str] = [
    "vp_poc_price",
    "vp_close_minus_poc_atr",
    "vp_va_high",
    "vp_va_low",
    "vp_close_inside_va_indicator",
    "vp_profile_shape",
]

_N_BINS = 30
_VA_PCT = 0.70
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
    """Load 1-min bars; same paths as vpin_features._load_1min."""
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
                warnings.warn(f"volume_profile: failed to load {path!r}: {exc}")
                continue
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


def _profile_one_day(day_bars: pd.DataFrame, n_bins: int = _N_BINS) -> tuple:
    """Return (poc_frac, close_minus_poc_norm, va_hi_frac, va_lo_frac,
    inside_va, skewness, day_close, day_range).
    """
    nan_row = (np.nan,) * 8
    if len(day_bars) < 30:
        return nan_row
    high = day_bars["high"].astype(float).values
    low = day_bars["low"].astype(float).values
    close_arr = day_bars["close"].astype(float).values
    vol = day_bars["volume"].astype(float).clip(lower=0).values

    day_high = float(np.nanmax(high))
    day_low = float(np.nanmin(low))
    day_range = day_high - day_low
    if not np.isfinite(day_range) or day_range <= 0 or vol.sum() <= 0:
        return nan_row

    bin_edges = np.linspace(day_low, day_high, n_bins + 1)
    # Use typical price (HLC/3) per bar to assign to a bin.
    typical = (high + low + close_arr) / 3.0
    bin_idx = np.clip(
        np.digitize(typical, bin_edges) - 1, 0, n_bins - 1
    )
    bin_vol = np.bincount(bin_idx, weights=vol, minlength=n_bins)

    # POC = bin with max volume; bin midprice
    poc_bin = int(np.argmax(bin_vol))
    poc_mid = (bin_edges[poc_bin] + bin_edges[poc_bin + 1]) / 2.0
    poc_frac = (poc_mid - day_low) / day_range

    # Value Area: grow outward from POC until covering VA_PCT of volume
    total_vol = float(bin_vol.sum())
    target = _VA_PCT * total_vol
    lo, hi = poc_bin, poc_bin
    cum = float(bin_vol[poc_bin])
    while cum < target and (lo > 0 or hi < n_bins - 1):
        left_vol = float(bin_vol[lo - 1]) if lo > 0 else -1.0
        right_vol = float(bin_vol[hi + 1]) if hi < n_bins - 1 else -1.0
        if right_vol >= left_vol and hi < n_bins - 1:
            hi += 1
            cum += float(bin_vol[hi])
        elif lo > 0:
            lo -= 1
            cum += float(bin_vol[lo])
        else:
            break
    va_lo_price = bin_edges[lo]
    va_hi_price = bin_edges[hi + 1]
    va_lo_frac = (va_lo_price - day_low) / day_range
    va_hi_frac = (va_hi_price - day_low) / day_range

    day_close = float(close_arr[-1])
    inside_va = 1.0 if (va_lo_price <= day_close <= va_hi_price) else 0.0
    close_minus_poc_norm = (day_close - poc_mid) / day_range  # normalised by range

    # Skewness of volume distribution across bins (Pearson moment skew)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    w = bin_vol / total_vol if total_vol > 0 else np.zeros_like(bin_vol)
    mu = float((bin_centers * w).sum())
    var = float(((bin_centers - mu) ** 2 * w).sum())
    sd = float(np.sqrt(var)) if var > 0 else 0.0
    if sd > 0:
        skew = float((((bin_centers - mu) ** 3) * w).sum() / (sd ** 3))
    else:
        skew = 0.0

    return (
        float(poc_frac),
        float(close_minus_poc_norm),
        float(va_hi_frac),
        float(va_lo_frac),
        float(inside_va),
        float(skew),
        float(day_close),
        float(day_range),
    )


def add_volume_profile_features(
    df_daily: pd.DataFrame,
    ticker: str,
    n_bins: int = _N_BINS,
) -> pd.DataFrame:
    """Append 6 volume-profile features to df_daily. Zero-fills if cache missing."""
    df = df_daily.copy()
    for c in VOLUME_PROFILE_FEATURE_NAMES:
        if c not in df.columns:
            df[c] = 0.0

    bars = _load_1min(ticker)
    if bars.empty or "close" not in bars.columns or "volume" not in bars.columns:
        logger.warning(
            "[volume_profile] no 1-min bars for %s — zero-filling", ticker
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
        out = _profile_one_day(day_bars, n_bins=n_bins)
        rows.append((pd.Timestamp(d),) + out)
    if not rows:
        return df

    cols = [
        "date",
        "vp_poc_price",
        "vp_close_minus_poc_norm",
        "vp_va_high",
        "vp_va_low",
        "vp_close_inside_va_indicator",
        "vp_profile_shape",
        "_day_close",
        "_day_range",
    ]
    vdf = pd.DataFrame(rows, columns=cols).set_index("date").sort_index()

    # ATR proxy: 14-day rolling mean of intraday range
    atr_proxy = vdf["_day_range"].rolling(14, min_periods=3).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        close_minus_poc_atr = (
            vdf["vp_close_minus_poc_norm"] * vdf["_day_range"]
        ) / atr_proxy.replace(0, np.nan)

    feats = pd.DataFrame(
        {
            "vp_poc_price": vdf["vp_poc_price"],
            "vp_close_minus_poc_atr": close_minus_poc_atr,
            "vp_va_high": vdf["vp_va_high"],
            "vp_va_low": vdf["vp_va_low"],
            "vp_close_inside_va_indicator": vdf["vp_close_inside_va_indicator"],
            "vp_profile_shape": vdf["vp_profile_shape"],
        }
    ).shift(1).fillna(0.0)

    if isinstance(df.index, pd.DatetimeIndex):
        join_idx = pd.DatetimeIndex(df.index.normalize())
    elif "date" in df.columns:
        join_idx = pd.DatetimeIndex(pd.to_datetime(df["date"]).dt.normalize())
    else:
        logger.warning(
            "[volume_profile] cannot align to df index (no date col) — zeroing"
        )
        return df
    for col in VOLUME_PROFILE_FEATURE_NAMES:
        df[col] = feats[col].reindex(join_idx).values
        df[col] = df[col].fillna(0.0).astype(float)
    logger.info(
        "[volume_profile] %s: poc_frac_mean=%.4f over %d days",
        ticker, float(feats["vp_poc_price"].mean()), len(feats),
    )
    return df
