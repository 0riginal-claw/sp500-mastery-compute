"""oc2_donchian_mtf_features.py — Multi-timeframe Donchian + filter stack features.

Extracted from OC-2/phitis/strategies/strategies_internet_research.py.
Key strategies re-implemented as ML feature values:
  IR1: StackedFilterDonchianBreakout (vol + ATR + no-lunch + breakout)
  IR3: ThreeTFConfluenceBreakout (5Min/15Min/1Hour trend alignment via approximation)
  IR4: VolumeConfirmedBreakout (volume > 1.5x SMA20)
  IR5: ADXTrendFilterBreakout (ADX(14) > 25 trend gate)
  IR6: CMFAccumulationBreakout (CMF(21) > 0 accumulation)
  IR8: MDDPreventionComposite (ADX + CMF + vol)

Research findings:
  - 3-TF alignment: 58% WR aligned vs 39% non-aligned
  - Volume > 1.5x SMA(20): high-confidence threshold (Bookmap)
  - ADX > 20: cuts false breakouts 30-50%
  - CMF(21) > 0: confirms institutional accumulation
  - Stacking 3-5 filters: WR 80-85%+

All features are .shift(1)-safe — works on 5Min OHLCV bars.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rolling_max(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=1).max()

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def _atr14(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_c = close.shift(1).fillna(close)
    tr = pd.concat(
        [high - low, (high - prev_c).abs(), (low - prev_c).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(14, min_periods=1).mean()

def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    n = len(high)
    h = high.to_numpy(dtype=np.float64)
    l = low.to_numpy(dtype=np.float64)
    c = close.to_numpy(dtype=np.float64)

    h_diff = np.diff(h, prepend=h[0])
    l_diff_neg = -np.diff(l, prepend=l[0])

    plus_dm = np.where((h_diff > l_diff_neg) & (h_diff > 0), h_diff, 0.0)
    minus_dm = np.where((l_diff_neg > h_diff) & (l_diff_neg > 0), l_diff_neg, 0.0)

    prev_c = np.roll(c, 1); prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))

    alpha = 1.0 / period
    decay = 1.0 - alpha

    tr_s = np.zeros(n); pdm_s = np.zeros(n); mdm_s = np.zeros(n)
    tr_s[0] = tr[0]; pdm_s[0] = plus_dm[0]; mdm_s[0] = minus_dm[0]
    for i in range(1, n):
        tr_s[i] = tr_s[i-1] * decay + tr[i] * alpha
        pdm_s[i] = pdm_s[i-1] * decay + plus_dm[i] * alpha
        mdm_s[i] = mdm_s[i-1] * decay + minus_dm[i] * alpha

    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = np.where(tr_s > 0, 100 * pdm_s / tr_s, 0.0)
        minus_di = np.where(tr_s > 0, 100 * mdm_s / tr_s, 0.0)
        denom = plus_di + minus_di
        dx = np.where(denom > 0, 100 * np.abs(plus_di - minus_di) / denom, 0.0)

    adx_arr = np.zeros(n); adx_arr[0] = dx[0]
    for i in range(1, n):
        adx_arr[i] = adx_arr[i-1] * decay + dx[i] * alpha

    return pd.Series(adx_arr, index=high.index)

def _cmf(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 21) -> pd.Series:
    hl = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / hl
    mfv = mfm * volume
    mfv_sum = mfv.rolling(period, min_periods=1).sum()
    vol_sum = volume.rolling(period, min_periods=1).sum()
    return (mfv_sum / vol_sum.replace(0, np.nan)).fillna(0.0)

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

def add_oc2_donchian_mtf_features(
    df: pd.DataFrame,
    ticker: str | None = None,
    entry_lookback: int = 20,
    vol_mult: float = 1.5,
    adx_threshold_strong: float = 25.0,
    adx_threshold_base: float = 20.0,
    cmf_period: int = 21,
) -> pd.DataFrame:
    """Add multi-timeframe Donchian and advanced filter stack features.

    New columns
    -----------
    mtf_donchian_upper20        float  rolling 20-bar high of high (shift-safe)
    mtf_breakout_signal         int    close > prev donchian upper (raw)
    mtf_ema63_above             int    close > EMA(63)  [15Min trend proxy for 5Min bars]
    mtf_sma600_above            int    close > SMA(600) [1Hour trend proxy for 5Min bars]
    mtf_three_tf_aligned        int    breakout AND ema63 AND sma600 aligned (IR3)
    mtf_vol_surge_flag          int    volume > vol_mult * SMA(20) volume (IR4)
    mtf_atr_expanding           int    ATR(14) > 20-bar rolling mean of ATR
    mtf_no_lunch_flag           int    session bar NOT in 24-54 (no false-breakout lunch zone)
    mtf_stacked_four_filter     int    breakout + vol + ATR expansion + no-lunch (IR1)
    mtf_adx14                   float  ADX(14) value
    mtf_adx_gt25                int    ADX > 25 (strong trend filter, IR5)
    mtf_adx_gt20                int    ADX > 20 (base trend filter)
    mtf_cmf21                   float  Chaikin Money Flow (21-period) value (IR6)
    mtf_cmf_positive            int    CMF > 0 (accumulation confirmation)
    mtf_mdd_composite           int    ADX > 20 AND CMF > 0 AND vol > SMA (IR8)
    mtf_n_filters_passing       int    count of [vol, atr, no_lunch, adx>20, cmf>0] active
    """
    df = df.copy()

    h = df["high"]
    l = df["low"]
    c = df["close"]
    has_vol = "volume" in df.columns and df["volume"].notna().any()
    v = df["volume"] if has_vol else pd.Series(1.0, index=df.index)

    # Donchian upper — shifted 1
    upper = _rolling_max(h, entry_lookback).shift(1)
    df["mtf_donchian_upper20"] = upper

    atr = _atr14(h, l, c).shift(1)
    atr_sma20 = atr.rolling(20, min_periods=5).mean()

    breakout = (c > upper).fillna(False).astype(bool)
    df["mtf_breakout_signal"] = breakout.shift(1).fillna(False).astype(int)

    # HTF trend proxies from 5Min bar series (per ThreeTFConfluenceBreakout)
    ema63 = _ema(c, 63).shift(1)
    sma600 = c.rolling(600, min_periods=100).mean().shift(1)
    df["mtf_ema63_above"] = (c > ema63).fillna(False).astype(int)
    df["mtf_sma600_above"] = (c > sma600).fillna(False).astype(int)
    df["mtf_three_tf_aligned"] = (
        breakout.shift(1).fillna(False) & (c > ema63).fillna(False) & (c > sma600).fillna(False)
    ).astype(int)

    # Volume surge (IR4)
    vol_sma20 = v.rolling(20, min_periods=5).mean().shift(1)
    vol_surge = (v.shift(1) > vol_mult * vol_sma20).fillna(False).astype(bool)
    df["mtf_vol_surge_flag"] = vol_surge.astype(int)

    # ATR expansion
    atr_expanding = (atr > atr_sma20).fillna(False).astype(bool)
    df["mtf_atr_expanding"] = atr_expanding.astype(int)

    # No-lunch filter (bars 24-54 on 5Min = 11:30-13:30 ET)
    bar_idx = _session_bar_idx(df)
    no_lunch = ~((bar_idx >= 24) & (bar_idx <= 54))
    df["mtf_no_lunch_flag"] = no_lunch.astype(int)

    # Stacked 4-filter (IR1): breakout + vol + ATR + no-lunch
    df["mtf_stacked_four_filter"] = (
        breakout.shift(1).fillna(False) & vol_surge & atr_expanding & no_lunch
    ).astype(int)

    # ADX
    adx = _adx(h, l, c, 14).shift(1)
    df["mtf_adx14"] = adx
    df["mtf_adx_gt25"] = (adx > adx_threshold_strong).fillna(False).astype(int)
    df["mtf_adx_gt20"] = (adx > adx_threshold_base).fillna(False).astype(int)

    # CMF
    cmf = _cmf(h, l, c, v, cmf_period).shift(1)
    df["mtf_cmf21"] = cmf
    df["mtf_cmf_positive"] = (cmf > 0).fillna(False).astype(int)

    # MDD composite (IR8): ADX > 20 AND CMF > 0 AND vol > SMA
    df["mtf_mdd_composite"] = (
        breakout.shift(1).fillna(False)
        & (adx > adx_threshold_base).fillna(False)
        & (cmf > 0).fillna(False)
        & vol_surge
    ).astype(int)

    # n_filters_passing
    df["mtf_n_filters_passing"] = (
        vol_surge.astype(int)
        + atr_expanding.astype(int)
        + no_lunch.astype(int)
        + (adx > adx_threshold_base).fillna(False).astype(int)
        + (cmf > 0).fillna(False).astype(int)
    )

    return df
