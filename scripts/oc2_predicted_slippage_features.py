"""oc2_predicted_slippage_features.py — Predicted slippage features.

Inspired by OC-2/strategy_intelligence_system/c003_slippage_analysis.py.
Slippage is a first-class concern for 5-Min intraday strategies. High predicted
slippage should either reduce position size or suppress marginal signals.

Slippage drivers on 5-Min bars:
  - Volatility (ATR / price): wider spreads during volatile bars
  - Volume: lower volume = thinner book = more slippage
  - Time of day: opening and closing bars have higher slippage (wider spreads)
  - Bar range: high-range bars often have more price impact
  - Relative volume: bar volume vs session average (thin = slippage spike)

Model: predicted_slippage_bps = f(ATR_pct, vol_ratio, time_tier, range_pct)
  Tier assignments based on OC-2 c003 findings:
    - Opening (bars 0-3): +2 bps extra slippage (opening auction spread)
    - Closing (bars 73-77): +2 bps extra
    - Lunch (bars 24-54): +1 bps extra (thin volume)
    - Normal: 0 extra

All features are .shift(1)-safe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Session bar helper
# ---------------------------------------------------------------------------

def _session_bar_idx(df: pd.DataFrame) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(df.index):
        dates = df.index.normalize().to_numpy()
    elif "datetime" in df.columns:
        dates = pd.to_datetime(df["datetime"]).dt.normalize().to_numpy()
    else:
        return pd.Series(np.zeros(len(df), dtype=np.int64), index=df.index)

    idx = np.zeros(len(df), dtype=np.int64)
    _, boundaries = np.unique(dates, return_index=True)
    boundaries = np.append(boundaries, len(df))
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        idx[s:e] = np.arange(e - s)
    return pd.Series(idx, index=df.index)


# ---------------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------------

def add_oc2_predicted_slippage_features(
    df: pd.DataFrame,
    ticker: str | None = None,
    base_slippage_bps: float = 1.0,
    opening_extra_bps: float = 2.0,
    closing_extra_bps: float = 2.0,
    lunch_extra_bps: float = 1.0,
    opening_bars: int = 4,
    closing_bars_from_end: int = 4,
    lunch_start_bar: int = 24,
    lunch_end_bar: int = 54,
    total_session_bars: int = 78,
) -> pd.DataFrame:
    """Add predicted slippage and execution quality features.

    New columns (all shift(1)-safe)
    -----------
    slip_atr_pct                float  ATR(14) as fraction of close (volatility spread proxy)
    slip_bar_range_pct          float  (high-low)/close — intrabar spread proxy
    slip_vol_ratio_sma20        float  volume / 20-bar rolling avg volume (liquidity)
    slip_vol_below_avg          int    volume < 0.5 * SMA20 (thin market = high slippage)
    slip_time_tier              int    0=normal, 1=lunch, 2=opening, 3=closing
    slip_opening_flag           int    1 if within first opening_bars of session
    slip_closing_flag           int    1 if within last closing_bars_from_end of session
    slip_lunch_flag             int    1 if in lunch zone (bars 24-54 on 5Min)
    slip_predicted_bps          float  predicted slippage in basis points
    slip_tier_label             str    categorical: low / medium / high / very_high
    slip_low_slippage           int    predicted_bps < 1.5 (best execution window)
    slip_high_slippage          int    predicted_bps > 3.0 (avoid or size-down)
    slip_edge_after_slippage    float  signal quality proxy: ATR_pct - slip_predicted_bps/10000
    slip_relative_impact_score  float  slip_bps / ATR_pct_bps (slippage as % of typical move)
    """
    df = df.copy()

    h = df["high"]
    l = df["low"]
    c = df["close"]
    has_vol = "volume" in df.columns and df["volume"].notna().any()
    v = df["volume"].shift(1) if has_vol else pd.Series(1.0, index=df.index)

    # ATR-based spread proxy
    prev_c = c.shift(1).fillna(c)
    tr = pd.concat(
        [h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1
    ).max(axis=1)
    atr14 = tr.rolling(14, min_periods=1).mean().shift(1)
    atr_pct = (atr14 / c.replace(0, np.nan)).fillna(0)
    df["slip_atr_pct"] = atr_pct

    # Bar range
    bar_range = (h.shift(1) - l.shift(1)) / c.replace(0, np.nan)
    df["slip_bar_range_pct"] = bar_range.fillna(0)

    # Volume ratio
    vol_sma20 = v.rolling(20, min_periods=5).mean()
    vol_ratio = (v / vol_sma20.replace(0, np.nan)).fillna(1.0)
    df["slip_vol_ratio_sma20"] = vol_ratio
    df["slip_vol_below_avg"] = (vol_ratio < 0.5).astype(int)

    # Session bar position
    bar_idx = _session_bar_idx(df).shift(1).fillna(0).astype(int)

    opening = bar_idx < opening_bars
    closing = bar_idx >= (total_session_bars - closing_bars_from_end)
    lunch = (bar_idx >= lunch_start_bar) & (bar_idx <= lunch_end_bar)

    df["slip_opening_flag"] = opening.astype(int)
    df["slip_closing_flag"] = closing.astype(int)
    df["slip_lunch_flag"] = lunch.astype(int)

    # Time tier: 0=normal, 1=lunch, 2=opening, 3=closing
    time_tier = np.zeros(len(df), dtype=np.int8)
    time_tier[lunch.to_numpy()] = 1
    time_tier[opening.to_numpy()] = 2
    time_tier[closing.to_numpy()] = 3
    df["slip_time_tier"] = time_tier

    # Extra slippage from time of day
    time_extra = np.zeros(len(df), dtype=np.float64)
    time_extra[lunch.to_numpy()] = lunch_extra_bps
    time_extra[opening.to_numpy()] = opening_extra_bps
    time_extra[closing.to_numpy()] = closing_extra_bps

    # Predicted slippage:
    #   base_bps + volatility_premium + time_premium + thin_market_premium
    #   volatility_premium: scale with atr_pct relative to typical 5-Min ATR of ~0.1%
    vol_premium = np.clip((atr_pct / 0.001 - 1.0) * 0.5, 0, 3.0)
    # thin market premium: extra bps when volume < 0.5x avg
    thin_premium = np.where(vol_ratio < 0.5, 2.0, np.where(vol_ratio < 0.8, 0.5, 0.0))

    predicted_bps = base_slippage_bps + vol_premium + time_extra + thin_premium
    df["slip_predicted_bps"] = predicted_bps

    # Tier label
    tiers = np.where(
        predicted_bps < 1.5, "low",
        np.where(predicted_bps < 2.5, "medium",
                 np.where(predicted_bps < 4.0, "high", "very_high"))
    )
    df["slip_tier_label"] = tiers
    df["slip_low_slippage"] = (predicted_bps < 1.5).astype(int)
    df["slip_high_slippage"] = (predicted_bps > 3.0).astype(int)

    # Edge after slippage
    atr_pct_bps = atr_pct * 10000  # convert to bps
    df["slip_edge_after_slippage"] = (atr_pct - predicted_bps / 10000).clip(-0.01, 0.1)
    df["slip_relative_impact_score"] = (
        predicted_bps / atr_pct_bps.replace(0, np.nan)
    ).fillna(0.5).clip(0, 2.0)

    return df
