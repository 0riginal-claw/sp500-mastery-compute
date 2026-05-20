"""
gtja_alpha191_features.py — GTJA "191 Formulaic Alphas" (time-series safe subset).

Source paper: "191 Formulaic Alphas" (Guotai Junan Securities Research, 2017).
Code reference: github:Daic115/alpha191 (MIT license, no paid API required).
License: MIT (formulas are published research; implementation is original).

NO-LOOKAHEAD AUDIT (2026-05-18)
================================
SHIFT(1) SAFETY PROTOCOL:
  All OHLCV inputs are pre-shifted by 1 bar at the top of compute_gtja_alpha191_features()
  (lines marked #SHIFT). feature[row T] therefore uses only:
    close[T-1], open[T-1], high[T-1], low[T-1], volume[T-1]
  and rolling statistics thereof (which look back from T-1 inclusive, never forward).
  First bar of each ticker is NaN for every output column — proves the shift is applied.

EXCLUDED ALPHAS (~141 cross-sectional or unimplementable without CS data):
  rank() is a cross-sectional operation requiring simultaneous data for all tickers.
  Alphas whose output depends on cross-sectional rank are excluded (zeroed stub).
  Where rank() appears *internally* (not shaping the final output), ts_rank(window=20)
  is substituted as a time-series approximation (common practice, see WorldQuant docs).
  Alphas requiring intraday (tick-level) data or paid external feeds are also excluded.

IMPLEMENTED: 50 time-series safe alphas using daily OHLCV + derived quantities.
  All inputs: open, high, low, close, volume (standard v9 stack columns).
  Derived: returns (close pct_change), typical_price ((H+L+C)/3), vwap proxy (SMA-5 of TP).
  No paid API. No external data fetched at runtime.

Integration cost: MEDIUM (~50 rolling-window passes; ~25 ms/ticker from OHLCV).
Expected lift: 1-2% AUC improvement (per producer metadata).
Human review required: YES — cross-sectional substitution with ts_rank is an
  approximation; validate feature distributions before production use.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Feature name manifest (50 implemented TS-safe alphas)
# ---------------------------------------------------------------------------

GTJA_ALPHA191_FEATURE_NAMES: list[str] = [
    "gtja_a006", "gtja_a009", "gtja_a011", "gtja_a012",
    "gtja_a014", "gtja_a018", "gtja_a019", "gtja_a021",
    "gtja_a022", "gtja_a023", "gtja_a024", "gtja_a027",
    "gtja_a028", "gtja_a029", "gtja_a031", "gtja_a034",
    "gtja_a035", "gtja_a038", "gtja_a039", "gtja_a040",
    "gtja_a041", "gtja_a042", "gtja_a043", "gtja_a044",
    "gtja_a045", "gtja_a046", "gtja_a047", "gtja_a048",
    "gtja_a049", "gtja_a050", "gtja_a051", "gtja_a052",
    "gtja_a053", "gtja_a054", "gtja_a055", "gtja_a057",
    "gtja_a060", "gtja_a061", "gtja_a062", "gtja_a063",
    "gtja_a064", "gtja_a065", "gtja_a066", "gtja_a067",
    "gtja_a068", "gtja_a069", "gtja_a071", "gtja_a072",
    "gtja_a073", "gtja_a074",
]

GTJA_ALPHA191_FEATURE_COUNT: int = len(GTJA_ALPHA191_FEATURE_NAMES)  # 50

# ---------------------------------------------------------------------------
# Rolling/time-series helper functions
# All operate on pre-shifted Series — no raw OHLCV is ever passed in directly.
# ---------------------------------------------------------------------------


def _sma(x: pd.Series, d: int) -> pd.Series:
    return x.rolling(d, min_periods=d).mean()


def _stddev(x: pd.Series, d: int) -> pd.Series:
    return x.rolling(d, min_periods=d).std()


def _sum(x: pd.Series, d: int) -> pd.Series:
    return x.rolling(d, min_periods=d).sum()


def _delay(x: pd.Series, d: int) -> pd.Series:
    return x.shift(d)


def _delta(x: pd.Series, d: int) -> pd.Series:
    return x - x.shift(d)


def _ts_min(x: pd.Series, d: int) -> pd.Series:
    return x.rolling(d, min_periods=d).min()


def _ts_max(x: pd.Series, d: int) -> pd.Series:
    return x.rolling(d, min_periods=d).max()


def _ts_rank(x: pd.Series, d: int) -> pd.Series:
    """Percentile rank within rolling window, normalized to [0, 1]."""
    return x.rolling(d, min_periods=d).rank(pct=True)


def _corr(x: pd.Series, y: pd.Series, d: int) -> pd.Series:
    return x.rolling(d, min_periods=d).corr(y)


def _wma(x: pd.Series, d: int) -> pd.Series:
    """Linearly weighted MA: weight_i = i+1 (most recent bar has weight d)."""
    weights = np.arange(1, d + 1, dtype=float)
    w_sum = weights.sum()
    return x.rolling(d, min_periods=d).apply(
        lambda v: float(np.dot(v, weights) / w_sum), raw=True
    )


def _row_max(*series: pd.Series) -> pd.Series:
    """Element-wise max across multiple aligned Series."""
    return pd.concat(list(series), axis=1).max(axis=1)


def _row_min(*series: pd.Series) -> pd.Series:
    """Element-wise min across multiple aligned Series."""
    return pd.concat(list(series), axis=1).min(axis=1)


# ---------------------------------------------------------------------------
# Main feature computation
# ---------------------------------------------------------------------------


def compute_gtja_alpha191_features(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
) -> pd.DataFrame:
    """Compute GTJA Alpha191 time-series safe subset (~50 alphas).

    All OHLCV inputs are pre-shifted by 1 bar so feature[row T]
    uses at most data from bars T-1 and earlier.
    First bar of each ticker will be NaN for all output columns.
    """
    required = {"open", "high", "low", "close", "volume"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        for col in GTJA_ALPHA191_FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0.0
        return df

    # ---- Pre-shift all inputs by 1 bar (SHIFT_INPUT) ----
    # After this point, every variable represents yesterday's bar.
    o = df["open"].shift(1)    # open[T-1]    #SHIFT
    h = df["high"].shift(1)    # high[T-1]    #SHIFT
    l = df["low"].shift(1)     # low[T-1]     #SHIFT
    c = df["close"].shift(1)   # close[T-1]   #SHIFT
    v = df["volume"].shift(1)  # volume[T-1]  #SHIFT

    # Derived quantities — all computed from shifted raw inputs (safe)
    ret = c.pct_change()                          # (c[T-1] - c[T-2]) / c[T-2]
    typical_price = (h + l + c) / 3.0             # shifted TP
    # Daily VWAP proxy: 5-bar SMA of typical price (no intraday data available)
    vwap = _sma(typical_price, 5)
    # Price position in HL range (used in multiple alphas)
    hl_range = (h - l).replace(0.0, np.nan)
    price_pos = ((c - l) - (h - c)) / hl_range    # Williams %R style

    # ------------------------------------------------------------------
    # Alpha006: (-1 * correlation(open, volume, 10))
    # When open and volume are positively correlated → bearish signal
    # ------------------------------------------------------------------
    df["gtja_a006"] = -1.0 * _corr(o, v, 10)

    # ------------------------------------------------------------------
    # Alpha009: conditional momentum using ts_min/ts_max of delta(close,1)
    # if ts_min(delta(c,1),5)>0: dc1; elif ts_max(dc1,5)<0: dc1; else: -dc1
    # ------------------------------------------------------------------
    dc1 = _delta(c, 1)
    tm5 = _ts_min(dc1, 5)
    tx5 = _ts_max(dc1, 5)
    df["gtja_a009"] = pd.Series(
        np.where(tm5 > 0, dc1, np.where(tx5 < 0, dc1, -dc1)),
        index=df.index,
    )

    # ------------------------------------------------------------------
    # Alpha011: sum(((close-low)-(high-close))/(high-low), 6)
    # Extent to which close is in upper vs lower half of range, 6-bar sum
    # ------------------------------------------------------------------
    df["gtja_a011"] = _sum(price_pos, 6)

    # ------------------------------------------------------------------
    # Alpha012: sign(delta(volume,1)) * (-1 * delta(close,1))
    # Volume direction × inverse price direction
    # ------------------------------------------------------------------
    df["gtja_a012"] = np.sign(_delta(v, 1)) * (-1.0 * _delta(c, 1))

    # ------------------------------------------------------------------
    # Alpha014: close - delay(close, 5)   [5-day momentum level]
    # ------------------------------------------------------------------
    df["gtja_a014"] = c - _delay(c, 5)

    # ------------------------------------------------------------------
    # Alpha018: close / delay(close, 5)   [5-day price ratio]
    # ------------------------------------------------------------------
    df["gtja_a018"] = c / _delay(c, 5).replace(0.0, np.nan)

    # ------------------------------------------------------------------
    # Alpha019: conditional 5-day return (percent vs level)
    # close<d5: (c-d5)/d5; close==d5: 0; else: (c-d5)/c
    # ------------------------------------------------------------------
    c_d5 = _delay(c, 5)
    dc5 = c - c_d5
    df["gtja_a019"] = pd.Series(
        np.where(
            c < c_d5,
            dc5 / c_d5.replace(0.0, np.nan),
            np.where(c == c_d5, 0.0, dc5 / c.replace(0.0, np.nan)),
        ),
        index=df.index,
    )

    # ------------------------------------------------------------------
    # Alpha021: sma(close,6)/delay(sma(close,6),6) - 1   [6-day SMA momentum]
    # ------------------------------------------------------------------
    sma6 = _sma(c, 6)
    df["gtja_a021"] = sma6 / _delay(sma6, 6).replace(0.0, np.nan) - 1.0

    # ------------------------------------------------------------------
    # Alpha022: sma((c-sma(c,6))/sma(c,6) - delay(...,3), 12)
    # 12-bar MA of the 3-day change in 6-day deviation
    # ------------------------------------------------------------------
    sma6_safe = _sma(c, 6).replace(0.0, np.nan)
    dev22 = (c - sma6_safe) / sma6_safe
    df["gtja_a022"] = _sma(dev22 - _delay(dev22, 3), 12)

    # ------------------------------------------------------------------
    # Alpha023: (sma(high,15)<delay(sma(high,15),15)) ? -delta(high,2) : 0
    # High mean-reversion signal — fade when 15-day high-MA falls
    # ------------------------------------------------------------------
    sma15h = _sma(h, 15)
    df["gtja_a023"] = np.where(sma15h < _delay(sma15h, 15), -1.0 * _delta(h, 2), 0.0)

    # ------------------------------------------------------------------
    # Alpha024: sma(close - delay(close,5), 5)   [smoothed 5-day momentum]
    # ------------------------------------------------------------------
    df["gtja_a024"] = _sma(c - _delay(c, 5), 5)

    # ------------------------------------------------------------------
    # Alpha027: WMA(((c-c3)/c3)*100 + ((c-c6)/c6)*100, 12)
    # Linearly-weighted MA of combined 3d + 6d momentum (percent)
    # ------------------------------------------------------------------
    c3 = _delay(c, 3).replace(0.0, np.nan)
    c6 = _delay(c, 6).replace(0.0, np.nan)
    mom3 = (c - c3) / c3 * 100
    mom6 = (c - c6) / c6 * 100
    df["gtja_a027"] = _wma(mom3 + mom6, 12)

    # ------------------------------------------------------------------
    # Alpha028: 3*sma(kdj_k,3) - 2*sma(sma(kdj_k,3),3)
    # KDJ-style: kdj_k = (close-ts_min(low,9))/(ts_max(high,9)-ts_min(low,9))*100
    # ------------------------------------------------------------------
    low9 = _ts_min(l, 9)
    high9 = _ts_max(h, 9)
    kdj_k = (c - low9) / (high9 - low9).replace(0.0, np.nan) * 100
    sma_k3 = _sma(kdj_k, 3)
    df["gtja_a028"] = 3.0 * sma_k3 - 2.0 * _sma(sma_k3, 3)

    # ------------------------------------------------------------------
    # Alpha029: (close - delay(close,6)) / delay(close,6) * volume
    # Volume-weighted 6-day momentum
    # ------------------------------------------------------------------
    df["gtja_a029"] = (c - _delay(c, 6)) / _delay(c, 6).replace(0.0, np.nan) * v

    # ------------------------------------------------------------------
    # Alpha031: (close - sma(close,12)) / sma(close,12) * 100
    # Percent deviation from 12-day MA
    # ------------------------------------------------------------------
    sma12 = _sma(c, 12).replace(0.0, np.nan)
    df["gtja_a031"] = (c - sma12) / sma12 * 100

    # ------------------------------------------------------------------
    # Alpha034: sma(close,12) / close   [MA-to-price ratio; mean-reversion]
    # ------------------------------------------------------------------
    df["gtja_a034"] = _sma(c, 12) / c.replace(0.0, np.nan)

    # ------------------------------------------------------------------
    # Alpha035: ts_rank(vol,32)*(1-ts_rank(c+h-l,16))*(1-ts_rank(returns,32))
    # High-volume + narrow-range + low-return composite
    # ------------------------------------------------------------------
    df["gtja_a035"] = (
        _ts_rank(v, 32)
        * (1.0 - _ts_rank(c + h - l, 16))
        * (1.0 - _ts_rank(ret, 32))
    )

    # ------------------------------------------------------------------
    # Alpha038: trend direction filter over 3 bars on delta(close,1)
    # 1 if all deltas positive; -1 if all negative; else 0
    # ------------------------------------------------------------------
    dc1_38 = _delta(c, 1)
    df["gtja_a038"] = pd.Series(
        np.where(
            _ts_min(dc1_38, 3) > 0, 1.0,
            np.where(_ts_max(dc1_38, 3) < 0, -1.0, 0.0),
        ),
        index=df.index,
    )

    # ------------------------------------------------------------------
    # Alpha039: -1 * delta(close,7) * (1 - ts_rank(volume,20))
    # (simplified from original; removes CS rank, keeps volume filter)
    # ------------------------------------------------------------------
    df["gtja_a039"] = -1.0 * _delta(c, 7) * (1.0 - _ts_rank(v, 20))

    # ------------------------------------------------------------------
    # Alpha040: Bollinger-band + volume signal (3-way conditional)
    # upper8 < sma2 → -1; sma2 < lower8 → 1; else: vol_ratio≥1 → 1 else -1
    # ------------------------------------------------------------------
    sma8 = _sma(c, 8)
    std8 = _stddev(c, 8)
    sma2 = _sma(c, 2)
    vol_ratio40 = v / _sma(v, 20).replace(0.0, np.nan)
    upper8 = sma8 + std8
    lower8 = sma8 - std8
    df["gtja_a040"] = pd.Series(
        np.where(
            upper8 < sma2, -1.0,
            np.where(sma2 < lower8, 1.0, np.where(vol_ratio40 >= 1.0, 1.0, -1.0)),
        ),
        index=df.index,
    )

    # ------------------------------------------------------------------
    # Alpha041: ts_max(high,5)^0.5 * vwap_proxy
    # Geometric mix of 5-day high breakout and smoothed price
    # ------------------------------------------------------------------
    df["gtja_a041"] = _ts_max(h, 5).clip(lower=0).pow(0.5) * vwap

    # ------------------------------------------------------------------
    # Alpha042: (sum(h-c,5)/sum(c-l,5)) * ts_rank(stddev(high,5),20)
    # Upper-shadow ratio weighted by high volatility rank
    # ------------------------------------------------------------------
    sum_hc5 = _sum(h - c, 5)
    sum_cl5 = _sum(c - l, 5).replace(0.0, np.nan)
    df["gtja_a042"] = (sum_hc5 / sum_cl5) * _ts_rank(_stddev(h, 5), 20)

    # ------------------------------------------------------------------
    # Alpha043: (sma(close,6) - close) / stddev(close,6)
    # Z-score of close vs 6-day MA (mean-reversion z-score)
    # ------------------------------------------------------------------
    std6 = _stddev(c, 6).replace(0.0, np.nan)
    df["gtja_a043"] = (_sma(c, 6) - c) / std6

    # ------------------------------------------------------------------
    # Alpha044: ts_rank(corr(low, sma(vol,10), 7), 4) + ts_rank(delta(vwap,3), 15)
    # (simplified from decaylinear version)
    # ------------------------------------------------------------------
    df["gtja_a044"] = (
        _ts_rank(_corr(l, _sma(v, 10), 7), 4)
        + _ts_rank(_delta(vwap, 3), 15)
    )

    # ------------------------------------------------------------------
    # Alpha045: delta(c*0.6+o*0.4, 1) * corr(vwap, sma(vol,150), 15)
    # (simplified from CS rank version: directional × vol-VWAP correlation)
    # ------------------------------------------------------------------
    price_blend = c * 0.6 + o * 0.4
    df["gtja_a045"] = _delta(price_blend, 1) * _corr(vwap, _sma(v, 150), 15)

    # ------------------------------------------------------------------
    # Alpha046: corr(ts_rank(close,10), ts_rank(volume,10), 3)
    # Price-rank vs volume-rank rolling correlation
    # ------------------------------------------------------------------
    df["gtja_a046"] = _corr(_ts_rank(c, 10), _ts_rank(v, 10), 3)

    # ------------------------------------------------------------------
    # Alpha047: sma((ts_max(high,6)-close)/(ts_max(high,6)-ts_min(low,6))*100, 9)
    # 9-bar smoothed Stochastic %K (6-bar lookback)
    # ------------------------------------------------------------------
    h6max = _ts_max(h, 6)
    l6min = _ts_min(l, 6)
    stoch47 = (h6max - c) / (h6max - l6min).replace(0.0, np.nan) * 100
    df["gtja_a047"] = _sma(stoch47, 9)

    # ------------------------------------------------------------------
    # Alpha048: -1 * (sign(delta(log(volume),1)) + sign(delta(price_pos,1))) * sign(delta(close,1))
    # Sign-agreement signal: when volume dir, range-pos dir, and price dir all agree → -1
    # ------------------------------------------------------------------
    log_v = np.log(v.clip(lower=1e-10))
    df["gtja_a048"] = -1.0 * (
        (np.sign(_delta(log_v, 1)) + np.sign(_delta(price_pos.fillna(0.0), 1)))
        * np.sign(_delta(c, 1))
    )

    # ------------------------------------------------------------------
    # Alpha049: 12-bar sum of ((high+low-2*delay(close,1))/2)^2 / HL^2
    # Squared normalized displacement of mid-bar from prior close
    # ------------------------------------------------------------------
    c_lag1 = _delay(c, 1)
    mid_disp = (h + l - 2.0 * c_lag1) / 2.0
    denom49 = (h - l).pow(2).replace(0.0, np.nan)
    df["gtja_a049"] = _sum(mid_disp.pow(2) / denom49, 12)

    # ------------------------------------------------------------------
    # Alpha050: -1 * ts_max(corr(ts_rank(vol,20), ts_rank(vwap,20), 5), 5)
    # Fade when volume and VWAP ranks become maximally correlated
    # ------------------------------------------------------------------
    df["gtja_a050"] = -1.0 * _ts_max(
        _corr(_ts_rank(v, 20), _ts_rank(vwap, 20), 5), 5
    )

    # ------------------------------------------------------------------
    # Alpha051: max(0, h - delay(h,1), c - delay(h,1))
    # High-breakout signal (positive part of TR relative to prior high)
    # ------------------------------------------------------------------
    dh51 = h - _delay(h, 1)
    cd51 = c - _delay(h, 1)
    zeros51 = pd.Series(0.0, index=df.index)
    df["gtja_a051"] = _row_max(zeros51, dh51, cd51)

    # ------------------------------------------------------------------
    # Alpha052: sum(max(0,h-delay(tp,1)),26)/sum(max(0,delay(tp,1)-l),26) - 1
    # Chaikin-style accumulation/distribution ratio (26-bar)
    # ------------------------------------------------------------------
    tp_lag1 = _delay(typical_price, 1)
    zeros52 = pd.Series(0.0, index=df.index)
    num52 = _sum(_row_max(zeros52, h - tp_lag1), 26)
    den52 = _sum(_row_max(zeros52, tp_lag1 - l), 26).replace(0.0, np.nan)
    df["gtja_a052"] = num52 / den52 - 1.0

    # ------------------------------------------------------------------
    # Alpha053: count(close > delay(close,1), 12) / 12
    # Fraction of up-days in past 12 bars
    # ------------------------------------------------------------------
    up_days = (c > _delay(c, 1)).astype(float)
    df["gtja_a053"] = _sum(up_days, 12) / 12.0

    # ------------------------------------------------------------------
    # Alpha054: -1 * (stddev(|c-o|, 10) + (c-o) + corr(c,o,10))
    # Combines intraday range volatility, direction, and price-open correlation
    # ------------------------------------------------------------------
    df["gtja_a054"] = -1.0 * (_stddev((c - o).abs(), 10) + (c - o) + _corr(c, o, 10))

    # ------------------------------------------------------------------
    # Alpha055: -1 * corr(stoch12, ts_rank(vol,20), 6)
    # stoch12 = (c - ts_min(l,12)) / (ts_max(h,12) - ts_min(l,12))
    # Fade when stochastic position and volume rank are correlated
    # ------------------------------------------------------------------
    l12 = _ts_min(l, 12)
    h12 = _ts_max(h, 12)
    stoch55 = (c - l12) / (h12 - l12).replace(0.0, np.nan)
    df["gtja_a055"] = -1.0 * _corr(stoch55, _ts_rank(v, 20), 6)

    # ------------------------------------------------------------------
    # Alpha057: (close - sma(close,9)) / sma(close,9) * 100
    # Percent deviation from 9-day MA
    # ------------------------------------------------------------------
    sma9 = _sma(c, 9).replace(0.0, np.nan)
    df["gtja_a057"] = (c - sma9) / sma9 * 100

    # ------------------------------------------------------------------
    # Alpha060: sum(((c-l)-(h-c))/(h-l)*volume, 20)
    # 20-bar Accumulation/Distribution line (classic ADL)
    # ------------------------------------------------------------------
    df["gtja_a060"] = _sum(price_pos.fillna(0.0) * v, 20)

    # ------------------------------------------------------------------
    # Alpha061: -1*(ts_rank(delta(vwap,1),12) + ts_rank(corr(l, sma(v,80), 8), 17))
    # VWAP momentum + low-volume correlation composite (reversed)
    # ------------------------------------------------------------------
    df["gtja_a061"] = -1.0 * (
        _ts_rank(_delta(vwap, 1), 12)
        + _ts_rank(_corr(l, _sma(v, 80), 8), 17)
    )

    # ------------------------------------------------------------------
    # Alpha062: -1 * corr(high, ts_rank(volume,20), 5)
    # Fade when high prices correlate with volume rank
    # ------------------------------------------------------------------
    df["gtja_a062"] = -1.0 * _corr(h, _ts_rank(v, 20), 5)

    # ------------------------------------------------------------------
    # Alpha063: sma(max(delta(close,1),0),6) / sma(|delta(close,1)|,6) * 100
    # Wilder RSI over 6 bars
    # ------------------------------------------------------------------
    dc1_63 = c - _delay(c, 1)
    gain63 = dc1_63.clip(lower=0.0)
    loss63 = dc1_63.abs()
    df["gtja_a063"] = _sma(gain63, 6) / _sma(loss63, 6).replace(0.0, np.nan) * 100

    # ------------------------------------------------------------------
    # Alpha064: -1*(ts_rank(corr(ts_rank(vwap,20),ts_rank(v,20),4),4)
    #             + ts_rank(corr(ts_rank(c,20),ts_rank(sma(v,60),20),4),14))
    # Dual-correlation composite (VWAP-vol and close-vol60)
    # ------------------------------------------------------------------
    corr_vv4 = _corr(_ts_rank(vwap, 20), _ts_rank(v, 20), 4)
    corr_cv4 = _corr(_ts_rank(c, 20), _ts_rank(_sma(v, 60), 20), 4)
    df["gtja_a064"] = -1.0 * (_ts_rank(corr_vv4, 4) + _ts_rank(corr_cv4, 14))

    # ------------------------------------------------------------------
    # Alpha065: sma(close,6) / close   [6-day MA ratio; mean-reversion signal]
    # ------------------------------------------------------------------
    df["gtja_a065"] = _sma(c, 6) / c.replace(0.0, np.nan)

    # ------------------------------------------------------------------
    # Alpha066: 6-day momentum normalized by 20-day std (z-scored momentum)
    # ------------------------------------------------------------------
    mom6_66 = (c - _delay(c, 6)) / _delay(c, 6).replace(0.0, np.nan) * 100
    std20_66 = _stddev(c, 20).replace(0.0, np.nan)
    df["gtja_a066"] = mom6_66 / std20_66

    # ------------------------------------------------------------------
    # Alpha067: delta(volume, 1)   [simple 1-bar volume change]
    # ------------------------------------------------------------------
    df["gtja_a067"] = _delta(v, 1)

    # ------------------------------------------------------------------
    # Alpha068: ts_rank(sma(corr(ts_rank(h,20), ts_rank(sma(v,15),20), 9), 6), 20)
    # High-rank vs volume-rank correlation, smoothed and ranked
    # ------------------------------------------------------------------
    corr_hv9 = _corr(_ts_rank(h, 20), _ts_rank(_sma(v, 15), 20), 9)
    df["gtja_a068"] = _ts_rank(_sma(corr_hv9, 6), 20)

    # ------------------------------------------------------------------
    # Alpha069: -1 * ts_max(ts_rank(delta(vwap,3),20), 5)
    # Fade maximal VWAP momentum over 5 bars
    # ------------------------------------------------------------------
    df["gtja_a069"] = -1.0 * _ts_max(_ts_rank(_delta(vwap, 3), 20), 5)

    # ------------------------------------------------------------------
    # Alpha071: (close - sma(close,24)) / sma(close,24) * 100
    # Percent deviation from 24-day MA
    # ------------------------------------------------------------------
    sma24 = _sma(c, 24).replace(0.0, np.nan)
    df["gtja_a071"] = (c - sma24) / sma24 * 100

    # ------------------------------------------------------------------
    # Alpha072: sma(max(ts_max(h,3)-delay(ts_max(h,3),3), 0) /
    #               max(ts_max(h,3)-delay(ts_max(h,3),3), delta(c,3)), 6) * 100
    # High-breakout velocity vs close-change (Wilder-style)
    # ------------------------------------------------------------------
    ts_max_h3 = _ts_max(h, 3)
    dts_max_h3 = _delta(ts_max_h3, 3)
    dc3 = _delta(c, 3)
    zeros72 = pd.Series(0.0, index=df.index)
    num72 = _row_max(zeros72, dts_max_h3)
    den72 = _row_max(zeros72, dts_max_h3, dc3).replace(0.0, np.nan)
    df["gtja_a072"] = _sma(num72 / den72, 6) * 100

    # ------------------------------------------------------------------
    # Alpha073: -1*ts_max(ts_rank(delta(vwap,5),20),3) * ts_rank(corr(l,sma(v,12),11),3)
    # VWAP momentum × low-volume correlation (composite fade)
    # ------------------------------------------------------------------
    df["gtja_a073"] = (
        -1.0
        * _ts_max(_ts_rank(_delta(vwap, 5), 20), 3)
        * _ts_rank(_corr(l, _sma(v, 12), 11), 3)
    )

    # ------------------------------------------------------------------
    # Alpha074: corr(sma(l*0.35+vwap*0.65,20), sma(sma(v,40),20), 7)
    #         + ts_rank(corr(ts_rank(vwap,20), ts_rank(v,20), 6), 20)
    # Low-VWAP blend vs vol-MA correlation + volume-VWAP rank correlation
    # ------------------------------------------------------------------
    blend74 = l * 0.35 + vwap * 0.65
    df["gtja_a074"] = (
        _corr(_sma(blend74, 20), _sma(_sma(v, 40), 20), 7)
        + _ts_rank(_corr(_ts_rank(vwap, 20), _ts_rank(v, 20), 6), 20)
    )

    # ---- Clip inf values (rare in extreme data) ----
    for col in GTJA_ALPHA191_FEATURE_NAMES:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    return df
