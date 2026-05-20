"""
synapse_gov_enhanced_features.py
Cross-sectional (market-level) congressional-trading and lobbying features
from Synapse's enhanced gov-trades signal export. Ticker-agnostic — all values
represent aggregate market activity, not a specific company's insider trades.

# Source: /Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/claudes test/archive_dead/Synapse_2026-05-05/signals/gov_trades/p2_enhanced_signals.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

_CSV_PATH = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/claudes test/archive_dead/Synapse_2026-05-05/signals/gov_trades"
    "/p2_enhanced_signals.csv"
)

_SIGNAL_COLS = [
    "avg_excess_return_20d", "avg_excess_return_5d",
    "buy_sell_ratio_20d", "buy_sell_ratio_5d",
    "chamber_divergence_5d",
    "dem_bias_20d", "dem_bias_5d",
    "dem_rep_divergence_20d", "dem_rep_divergence_5d",
    "gottheimer_10d_count", "gottheimer_10d_dir",
    "house_net_20d", "house_net_5d",
    "khanna_5d_avg_excess", "khanna_5d_count", "khanna_5d_dir",
    "large_trade_avg_excess_20d", "large_trade_avg_excess_5d",
    "large_trade_dir_20d", "large_trade_dir_5d",
    "large_trade_volume_20d", "large_trade_volume_5d",
    "lobbying_flag", "lobbying_registrants_90d",
    "lobbying_spend_30d", "lobbying_spend_90d",
    "net_direction_20d", "net_direction_5d",
    "pelosi_10d_avg_excess", "pelosi_10d_count", "pelosi_10d_dir",
    "repub_bias_20d", "repub_bias_5d",
    "senate_net_20d", "senate_net_5d",
    "top3_5d_signal",
    "total_trades_20d", "total_trades_5d",
    "volume_weighted_sentiment_5d",
]

_PREFIX = "syn_gov_"
_CACHE: dict = {}


def _load_gov_daily() -> pd.DataFrame:
    if "gov" in _CACHE:
        return _CACHE["gov"]
    raw = pd.read_csv(_CSV_PATH, parse_dates=["date"])
    raw = raw.sort_values("date").set_index("date")
    raw.index = pd.DatetimeIndex(raw.index).normalize()
    # Forward-fill sparse data (avg ~8.5 day gap between observations)
    # Resample to calendar-day resolution then ffill
    daily = raw[_SIGNAL_COLS].resample("D").last().ffill()
    # Shift-1: yesterday's congressional-trade state at signal time
    daily = daily.shift(1)
    daily.columns = [_PREFIX + c for c in _SIGNAL_COLS]
    _CACHE["gov"] = daily
    return daily


def add_synapse_gov_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge cross-sectional congressional-trading features.
    All 39 columns represent aggregate market-level signals (not per-ticker insider trades).
    Values are .shift(1) (previous day's observation, forward-filled from sparse events).

    Coverage: 2023-08-28 to 2026-04-08. Returns NaN for dates outside coverage.

    Args:
        df: DataFrame with DatetimeIndex (any freq), OHLCV at minimum.
    Returns:
        df with 39 new columns prefixed 'syn_gov_'.
    """
    out_cols = [_PREFIX + c for c in _SIGNAL_COLS]
    if all(c in df.columns for c in out_cols):
        return df

    if not _CSV_PATH.exists():
        for c in out_cols:
            df[c] = np.nan
        return df

    daily = _load_gov_daily()

    idx = df.index
    if hasattr(idx, "tz") and idx.tz is not None:
        idx_norm = idx.tz_localize(None).normalize()
    else:
        idx_norm = pd.DatetimeIndex(idx).normalize()

    merged = daily.reindex(idx_norm)
    merged.index = df.index

    for c in out_cols:
        df[c] = merged[c].values

    return df
