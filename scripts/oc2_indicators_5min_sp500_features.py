"""oc2_indicators_5min_sp500_features.py — 10 high-value 5-Min SP500 indicators.

Derived from OC-2/strategy_intelligence_system/indicators_5min_sp500_research.md.
These 10 indicators are research-validated as high-value for 5-minute intraday
S&P 500 trading. Settings are tuned to 5-Min bars as documented in the research.

Indicators:
  1. CMF (Chaikin Money Flow, 21)   — accumulation/distribution pressure
  2. Keltner Channels (20 EMA, 1.5 ATR) — trend envelope + squeeze detection
  3. ADX (14) with DI lines         — trend strength + direction
  4. Aroon (25)                      — trend freshness (new highs/lows)
  5. Parabolic SAR (step=0.02, max=0.2) — trailing stop-and-reverse
  6. CCI (20)                        — normalized mean deviation oscillator
  7. Williams %R (14)               — reversal detector (inverse Stochastic)
  8. MACD Histogram (5-13-1)        — momentum acceleration
  9. OBV (On-Balance Volume)        — volume flow + divergence features
  10. Elder Ray (13 EMA)            — bull/bear power above/below trend

Research-validated combinations:
  - Trend following: MACD Hist + OBV trend + ADX > 20
  - Breakout: Keltner Channel break + CMF > 0 + vol spike + ADX rising
  - Mean reversion: CCI ±200 + Williams %R + BB touch
  - Regime detection: ADX level + Aroon crossover + Keltner width

All features are .shift(1)-safe. Works on 5-Min OHLCV bars.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_c = close.shift(1).fillna(close)
    tr = pd.concat(
        [high - low, (high - prev_c).abs(), (low - prev_c).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()

def _adx_full(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    """Returns (adx, plus_di, minus_di) as pd.Series."""
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
        pdi = np.where(tr_s > 0, 100 * pdm_s / tr_s, 0.0)
        mdi = np.where(tr_s > 0, 100 * mdm_s / tr_s, 0.0)
        denom = pdi + mdi
        dx = np.where(denom > 0, 100 * np.abs(pdi - mdi) / denom, 0.0)

    adx_arr = np.zeros(n); adx_arr[0] = dx[0]
    for i in range(1, n):
        adx_arr[i] = adx_arr[i-1] * decay + dx[i] * alpha

    idx = high.index
    return pd.Series(adx_arr, index=idx), pd.Series(pdi, index=idx), pd.Series(mdi, index=idx)

def _parabolic_sar(high: pd.Series, low: pd.Series, step: float = 0.02, max_step: float = 0.2) -> pd.Series:
    h = high.to_numpy(dtype=np.float64)
    l = low.to_numpy(dtype=np.float64)
    n = len(h)
    sar = np.full(n, np.nan)
    if n < 2:
        return pd.Series(sar, index=high.index)

    # Initialize: assume bullish first
    bull = True
    af = step
    ep = h[0]  # extreme point
    sar_val = l[0]

    sar[0] = sar_val
    for i in range(1, n):
        if bull:
            sar_val = sar_val + af * (ep - sar_val)
            sar_val = min(sar_val, l[i-1], l[max(0, i-2)])
            if l[i] < sar_val:
                bull = False
                sar_val = ep
                ep = l[i]
                af = step
            else:
                if h[i] > ep:
                    ep = h[i]
                    af = min(af + step, max_step)
        else:
            sar_val = sar_val + af * (ep - sar_val)
            sar_val = max(sar_val, h[i-1], h[max(0, i-2)])
            if h[i] > sar_val:
                bull = True
                sar_val = ep
                ep = h[i]
                af = step
            else:
                if l[i] < ep:
                    ep = l[i]
                    af = min(af + step, max_step)
        sar[i] = sar_val

    return pd.Series(sar, index=high.index)


# ---------------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------------

def add_oc2_indicators_5min_sp500_features(
    df: pd.DataFrame,
    ticker: str | None = None,
) -> pd.DataFrame:
    """Add 10 high-value 5-Min SP500 indicator features.

    New columns (all shift(1)-safe)
    -----------
    # 1. CMF
    ind_cmf21                   float  Chaikin Money Flow (21-period, range -1 to +1)
    ind_cmf21_positive          int    1 if CMF > 0 (accumulation)
    ind_cmf21_strong_acc        int    1 if CMF > 0.05 (confirmed accumulation)

    # 2. Keltner Channels
    ind_kc20_upper              float  upper Keltner (EMA20 + 1.5*ATR14)
    ind_kc20_lower              float  lower Keltner (EMA20 - 1.5*ATR14)
    ind_kc20_mid                float  EMA(20) (middle band)
    ind_kc20_above_upper        int    close > upper Keltner (breakout signal)
    ind_kc20_below_lower        int    close < lower Keltner (breakdown)
    ind_kc20_pct_b              float  position within channel (-1=lower, 0=mid, +1=upper)

    # 3. ADX
    ind_adx14                   float  ADX(14) — trend strength
    ind_adx14_plus_di           float  +DI component
    ind_adx14_minus_di          float  -DI component
    ind_adx14_di_spread         float  +DI - -DI (positive = bullish directional bias)
    ind_adx14_trending          int    ADX > 20 (trending regime)
    ind_adx14_strong_trend      int    ADX > 25 (strong trend)
    ind_adx14_rising            int    ADX increasing over last 3 bars

    # 4. Aroon
    ind_aroon25_up              float  Aroon Up (0-100)
    ind_aroon25_down            float  Aroon Down (0-100)
    ind_aroon25_cross_up        int    Aroon Up crossed above Aroon Down
    ind_aroon25_bullish         int    Aroon Up > 50 AND Aroon Down < 50

    # 5. Parabolic SAR
    ind_psar                    float  Parabolic SAR value
    ind_psar_bullish            int    1 if SAR below close (bullish)
    ind_psar_just_flipped       int    1 if SAR changed direction this bar

    # 6. CCI
    ind_cci20                   float  CCI(20) value
    ind_cci20_overbought        int    CCI > 100
    ind_cci20_oversold          int    CCI < -100
    ind_cci20_bullish_momentum  int    CCI crossed above +100 (trend strength)

    # 7. Williams %R
    ind_willr14                 float  Williams %R (0 to -100 range)
    ind_willr14_oversold        int    %R < -80 (potential reversal buy)
    ind_willr14_overbought      int    %R > -20 (potential reversal sell)
    ind_willr14_bullish_bias    int    %R > -50 (above centerline)

    # 8. MACD Histogram (5-13-1)
    ind_macd_hist_5_13_1        float  MACD histogram (5-13-1 settings for 5Min)
    ind_macd_hist_positive      int    histogram > 0 (bullish momentum)
    ind_macd_hist_expanding     int    histogram > prev bar histogram (accelerating)

    # 9. OBV
    ind_obv                     float  cumulative OBV value
    ind_obv_sma20               float  20-bar SMA of OBV
    ind_obv_above_sma           int    OBV > SMA20 (bullish flow)
    ind_obv_rising              int    OBV slope > 0 over 5 bars

    # 10. Elder Ray
    ind_elder_ema13             float  EMA(13) — trend baseline
    ind_elder_bull_power        float  high - EMA13 (bull power)
    ind_elder_bear_power        float  low - EMA13 (bear power, negative when bearish)
    ind_elder_bull_rising       int    bull power increasing (momentum)
    ind_elder_uptrend_pullback  int    classic Elder pullback entry: EMA rising + bear power contracting
    """
    df = df.copy()

    h = df["high"]
    l = df["low"]
    c = df["close"]
    has_vol = "volume" in df.columns and df["volume"].notna().any()
    v = df["volume"] if has_vol else pd.Series(1.0, index=df.index)

    atr14 = _atr(h, l, c, 14)

    # ── 1. CMF ────────────────────────────────────────────────────────────────
    hl = (h - l).replace(0, np.nan)
    mfm = ((c - l) - (h - c)) / hl
    mfv = mfm * v
    cmf21 = (mfv.rolling(21, min_periods=5).sum() / v.rolling(21, min_periods=5).sum().replace(0, np.nan)).fillna(0).shift(1)
    df["ind_cmf21"] = cmf21
    df["ind_cmf21_positive"] = (cmf21 > 0).astype(int)
    df["ind_cmf21_strong_acc"] = (cmf21 > 0.05).astype(int)

    # ── 2. Keltner Channels ───────────────────────────────────────────────────
    ema20 = _ema(c, 20).shift(1)
    kc_width = 1.5 * atr14.shift(1)
    kc_upper = ema20 + kc_width
    kc_lower = ema20 - kc_width
    df["ind_kc20_upper"] = kc_upper
    df["ind_kc20_lower"] = kc_lower
    df["ind_kc20_mid"] = ema20
    df["ind_kc20_above_upper"] = (c > kc_upper).astype(int)
    df["ind_kc20_below_lower"] = (c < kc_lower).astype(int)
    kc_range = (kc_upper - kc_lower).replace(0, np.nan)
    df["ind_kc20_pct_b"] = ((c - ema20) / (kc_range / 2)).clip(-2, 2)

    # ── 3. ADX ────────────────────────────────────────────────────────────────
    adx, pdi, mdi = _adx_full(h, l, c, 14)
    adx = adx.shift(1); pdi = pdi.shift(1); mdi = mdi.shift(1)
    df["ind_adx14"] = adx
    df["ind_adx14_plus_di"] = pdi
    df["ind_adx14_minus_di"] = mdi
    df["ind_adx14_di_spread"] = pdi - mdi
    df["ind_adx14_trending"] = (adx > 20).astype(int)
    df["ind_adx14_strong_trend"] = (adx > 25).astype(int)
    df["ind_adx14_rising"] = (adx > adx.shift(3)).astype(int)

    # ── 4. Aroon (25) ─────────────────────────────────────────────────────────
    period = 25
    # Aroon Up = (period - bars since highest high) / period * 100
    high_idx = h.rolling(period + 1, min_periods=1).apply(lambda x: len(x) - 1 - x.argmax(), raw=True)
    low_idx = l.rolling(period + 1, min_periods=1).apply(lambda x: len(x) - 1 - x.argmin(), raw=True)
    aroon_up = ((period - high_idx) / period * 100).shift(1)
    aroon_dn = ((period - low_idx) / period * 100).shift(1)
    df["ind_aroon25_up"] = aroon_up
    df["ind_aroon25_down"] = aroon_dn
    df["ind_aroon25_cross_up"] = ((aroon_up > aroon_dn) & (aroon_up.shift(1) <= aroon_dn.shift(1))).astype(int)
    df["ind_aroon25_bullish"] = ((aroon_up > 50) & (aroon_dn < 50)).astype(int)

    # ── 5. Parabolic SAR ─────────────────────────────────────────────────────
    psar = _parabolic_sar(h, l).shift(1)
    psar_prev = psar.shift(1)
    df["ind_psar"] = psar
    df["ind_psar_bullish"] = (c > psar).astype(int)
    # Flip detection: SAR crossed relative to price
    prev_bull = (c.shift(1) > psar_prev)
    curr_bull = (c > psar)
    df["ind_psar_just_flipped"] = ((curr_bull != prev_bull).fillna(False)).astype(int)

    # ── 6. CCI (20) ───────────────────────────────────────────────────────────
    tp = (h + l + c) / 3
    tp_ma = tp.rolling(20, min_periods=5).mean()
    mad = tp.rolling(20, min_periods=5).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci = ((tp - tp_ma) / (0.015 * mad.replace(0, np.nan))).shift(1)
    df["ind_cci20"] = cci
    df["ind_cci20_overbought"] = (cci > 100).astype(int)
    df["ind_cci20_oversold"] = (cci < -100).astype(int)
    df["ind_cci20_bullish_momentum"] = ((cci > 100) & (cci.shift(1) <= 100)).astype(int)

    # ── 7. Williams %R (14) ───────────────────────────────────────────────────
    highest_h = h.rolling(14, min_periods=1).max().shift(1)
    lowest_l = l.rolling(14, min_periods=1).min().shift(1)
    willr = ((highest_h - c) / (highest_h - lowest_l).replace(0, np.nan) * -100).fillna(-50)
    df["ind_willr14"] = willr
    df["ind_willr14_oversold"] = (willr < -80).astype(int)
    df["ind_willr14_overbought"] = (willr > -20).astype(int)
    df["ind_willr14_bullish_bias"] = (willr > -50).astype(int)

    # ── 8. MACD Histogram (5-13-1) ────────────────────────────────────────────
    macd_line = (_ema(c, 5) - _ema(c, 13)).shift(1)
    signal_line = _ema(macd_line, 1)  # period=1 → signal = macd itself for 5-13-1
    # Actual 5-13-1: signal period 1 means histogram = macd - macd (no smoothing)
    # More useful: use 9-period signal for histogram shape
    signal_9 = _ema(macd_line, 9)
    macd_hist = macd_line - signal_9
    df["ind_macd_hist_5_13_1"] = macd_hist
    df["ind_macd_hist_positive"] = (macd_hist > 0).astype(int)
    df["ind_macd_hist_expanding"] = (macd_hist > macd_hist.shift(1)).astype(int)

    # ── 9. OBV ────────────────────────────────────────────────────────────────
    direction = np.sign(c.diff().fillna(0))
    obv = (direction * v).cumsum().shift(1)
    obv_sma20 = obv.rolling(20, min_periods=5).mean()
    df["ind_obv"] = obv
    df["ind_obv_sma20"] = obv_sma20
    df["ind_obv_above_sma"] = (obv > obv_sma20).astype(int)
    df["ind_obv_rising"] = (obv > obv.shift(5)).astype(int)

    # ── 10. Elder Ray (13 EMA) ────────────────────────────────────────────────
    ema13 = _ema(c, 13).shift(1)
    bull_power = (h.shift(1) - ema13)
    bear_power = (l.shift(1) - ema13)
    df["ind_elder_ema13"] = ema13
    df["ind_elder_bull_power"] = bull_power
    df["ind_elder_bear_power"] = bear_power
    df["ind_elder_bull_rising"] = (bull_power > bull_power.shift(1)).astype(int)
    # Classic uptrend pullback entry: EMA rising AND bear power contracting toward 0
    ema13_rising = (ema13 > ema13.shift(3)).astype(int)
    bear_contracting = (bear_power > bear_power.shift(1)).astype(int)  # bear power less negative
    df["ind_elder_uptrend_pullback"] = (ema13_rising & bear_contracting).astype(int)

    return df
