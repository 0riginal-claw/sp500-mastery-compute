"""
synapse_market_context_features.py
Ticker-agnostic SPY/QQQ market-context and cross-sectional EDGAR/gov features
extracted from the Synapse v3 replay parquet.

# Source: /Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/claudes test/archive_dead/Synapse_2026-05-05/signals/output/v3.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path

_V3_PATH = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/claudes test/archive_dead/Synapse_2026-05-05/signals/output/v3.parquet"
)

# Only columns that are ticker-agnostic (SPY/QQQ/market-level)
_MKT_COLS = [
    "spy_ret_p3", "spy_rsi_p3", "spy_vol_ratio_p3", "spy_rel_ma20_p3",
    "qqq_ret_p3", "qqq_rsi_p3", "qqq_vol_ratio_p3", "qqq_rel_ma20_p3",
    "sp500_1d_trend_mtf", "sp500_5d_trend_mtf",
    "qqq_1d_trend_mtf", "qqq_5d_trend_mtf",
    "mkt_filing_count_v3", "mkt_risk_tone_v3", "mkt_mda_sentiment_v3",
    "mkt_forward_looking_v3", "mkt_pos_ratio_v3",
    "mkt_gov_net_flow_5d_v3", "mkt_gov_trade_count_5d_v3", "mkt_gov_party_bias_5d_v3",
]

_PREFIX = "syn_mc_"
_CACHE: dict = {}


def _load_mkt_daily() -> pd.DataFrame:
    if "daily" in _CACHE:
        return _CACHE["daily"]
    raw = pd.read_parquet(_V3_PATH, engine="pyarrow", columns=["timestamp"] + _MKT_COLS)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    # Strip tz so we can compare with tz-naive DatetimeIndex
    if raw["timestamp"].dt.tz is not None:
        raw["timestamp"] = raw["timestamp"].dt.tz_localize(None)
    raw["_date"] = raw["timestamp"].dt.normalize()
    # End-of-day value per session date (last 5-min bar of day)
    daily = raw.groupby("_date")[_MKT_COLS].last()
    daily.index = pd.DatetimeIndex(daily.index)
    _CACHE["daily"] = daily
    return daily


def add_synapse_market_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge ticker-agnostic SPY/QQQ/market-context features from Synapse v3 parquet.
    All values are .shift(1) (previous day's market context at signal time).

    Args:
        df: DataFrame with DatetimeIndex (daily or higher freq), OHLCV at minimum.
    Returns:
        df with 20 new columns prefixed 'syn_mc_'.
    """
    out_cols = [_PREFIX + c for c in _MKT_COLS]
    # Idempotent: skip if already present
    if all(c in df.columns for c in out_cols):
        return df

    if not _V3_PATH.exists():
        for c in out_cols:
            df[c] = np.nan
        return df

    daily = _load_mkt_daily()

    # Normalize the input df index to date-only for merge
    idx = df.index
    if hasattr(idx, "tz") and idx.tz is not None:
        idx_norm = idx.tz_localize(None).normalize()
    else:
        idx_norm = pd.DatetimeIndex(idx).normalize()

    # Shift-1 on daily: each date gets PREVIOUS day's market context
    daily_shifted = daily.shift(1)
    daily_shifted.columns = out_cols

    merged = daily_shifted.reindex(idx_norm)
    merged.index = df.index

    for c in out_cols:
        df[c] = merged[c].values

    return df
