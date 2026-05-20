"""
add_jesse_features_features.py — 54 Jesse-inspired technical indicator features.

Data source: github:jesse-ai/jesse (MIT License, Copyright 2020 Jesse.Trade).
License: MIT — clean, no Commons Clause, no copyleft.
Requires paid API: NO — pure OHLCV calculation, no external calls.

No-Lookahead Audit
------------------
Every raw indicator is first computed over the full OHLCV series (standard TA
practice), then the ENTIRE indicator column is shifted forward by 1 bar
(.shift(1)) before being written into the output DataFrame.  This guarantees
that on any given row t the model sees only information that was available at
close of bar t-1.

Columns that depend only on lagged price (e.g. rolling max of shifted close)
are equivalent to computing on prior bars and therefore also safe.

Implementation: vectorized pandas/numpy only — no jesse package dependency.
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JESSE_FEATURE_COUNT: int = 54

JESSE_FEATURE_NAMES: list[str] = [
    # Moving Averages (7)
    "jesse_sma_10", "jesse_sma_20", "jesse_sma_50",
    "jesse_ema_9", "jesse_ema_21",
    "jesse_wma_10", "jesse_vwma_10",
    # Ichimoku (5)
    "jesse_ichi_tenkan", "jesse_ichi_kijun",
    "jesse_ichi_cloud_diff", "jesse_ichi_cloud_bull",
    "jesse_ichi_chikou_diff",
    # Donchian Channel (3)
    "jesse_dc_upper", "jesse_dc_lower", "jesse_dc_mid",
    # MACD (3)
    "jesse_macd", "jesse_macd_signal", "jesse_macd_hist",
    # Momentum Oscillators (10)
    "jesse_rsi_14", "jesse_rsi_7",
    "jesse_stoch_k", "jesse_stoch_d",
    "jesse_cci_20", "jesse_willr_14",
    "jesse_ultimate_osc",
    "jesse_roc_10", "jesse_mom_10",
    "jesse_ao",
    # Trend (7)
    "jesse_adx_14", "jesse_di_plus", "jesse_di_minus",
    "jesse_aroon_up", "jesse_aroon_down", "jesse_aroon_osc",
    "jesse_supertrend_dir",
    # Volatility (7)
    "jesse_atr_14", "jesse_natr_14",
    "jesse_bb_upper", "jesse_bb_lower", "jesse_bb_width", "jesse_bb_pct",
    "jesse_kc_width",
    # Volume / Microstructure (12)
    "jesse_mfi_14", "jesse_cmf_20",
    "jesse_obv_z21",
    "jesse_vwap_dev",
    "jesse_ad_z21",
    "jesse_force_idx_13",
    "jesse_eom_14",
    "jesse_pvt_z21",
    "jesse_nvi_z21", "jesse_pvi_z21",
    "jesse_vwap_range_pct",
    "jesse_vol_ratio_10_50",
]

assert len(JESSE_FEATURE_NAMES) == JESSE_FEATURE_COUNT, (
    f"Feature name count mismatch: {len(JESSE_FEATURE_NAMES)} vs {JESSE_FEATURE_COUNT}"
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def _wma(series: pd.Series, window: int) -> pd.Series:
    weights = np.arange(1, window + 1, dtype=float)
    return series.rolling(window).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tr = _true_range(high, low, close)
    return tr.ewm(span=period, adjust=False, min_periods=period).mean()


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _stoch(high: pd.Series, low: pd.Series, close: pd.Series,
           k_period: int = 14, d_period: int = 3) -> tuple[pd.Series, pd.Series]:
    lowest = low.rolling(k_period, min_periods=k_period).min()
    highest = high.rolling(k_period, min_periods=k_period).max()
    denom = (highest - lowest).replace(0, np.nan)
    k = 100 * (close - lowest) / denom
    d = k.rolling(d_period).mean()
    return k, d


def _zscore(series: pd.Series, window: int) -> pd.Series:
    mu = series.rolling(window, min_periods=window).mean()
    sigma = series.rolling(window, min_periods=window).std().replace(0, np.nan)
    return (series - mu) / sigma


# ---------------------------------------------------------------------------
# Main compute function
# ---------------------------------------------------------------------------

def compute_add_jesse_features_features(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
) -> pd.DataFrame:
    """Compute 54 Jesse-inspired technical features and append to df.

    All output columns are shifted 1 bar (.shift(1)) before assignment —
    strict no-lookahead compliance. See module docstring for audit details.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: open, high, low, close, volume (case-insensitive
        normalised to lower-case internally). Index should be a DatetimeIndex.
    ticker : str, optional
        Ignored — provided for API uniformity with other helper modules.

    Returns
    -------
    pd.DataFrame
        Input df with 54 new jesse_* columns appended.
    """
    out = df.copy()

    # ---- Column normalisation ------------------------------------------
    col_map = {c.lower(): c for c in out.columns}
    def _col(name: str) -> pd.Series:
        return out[col_map[name]].astype(float)

    try:
        o = _col("open")
        h = _col("high")
        l = _col("low")
        c = _col("close")
        v = _col("volume")
    except KeyError as exc:
        logger.warning("[jesse] missing OHLCV column: %s — zero-filling all 54 features", exc)
        for col in JESSE_FEATURE_NAMES:
            out[col] = 0.0
        return out

    n = len(out)

    # ====================================================================
    # 1. Moving Averages
    # ====================================================================
    sma10 = _sma(c, 10)
    sma20 = _sma(c, 20)
    sma50 = _sma(c, 50)
    ema9 = _ema(c, 9)
    ema21 = _ema(c, 21)
    wma10 = _wma(c, 10)
    denom_vwma = v.rolling(10, min_periods=10).sum().replace(0, np.nan)
    vwma10 = (c * v).rolling(10, min_periods=10).sum() / denom_vwma

    # ====================================================================
    # 2. Ichimoku (standard 9/26/52 periods)
    # ====================================================================
    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    cloud_diff = senkou_a - senkou_b
    cloud_bull = (cloud_diff > 0).astype(float)
    chikou_diff = (c - c.shift(26)) / c.shift(26).replace(0, np.nan)

    # ====================================================================
    # 3. Donchian Channel (20)
    # ====================================================================
    dc_upper = h.rolling(20).max()
    dc_lower = l.rolling(20).min()
    dc_mid = (dc_upper + dc_lower) / 2

    # ====================================================================
    # 4. MACD (12/26/9)
    # ====================================================================
    macd_line = _ema(c, 12) - _ema(c, 26)
    macd_signal = _ema(macd_line, 9)
    macd_hist = macd_line - macd_signal

    # ====================================================================
    # 5. Momentum Oscillators
    # ====================================================================
    rsi14 = _rsi(c, 14)
    rsi7 = _rsi(c, 7)
    stoch_k, stoch_d = _stoch(h, l, c, 14, 3)

    # CCI (20)
    tp = (h + l + c) / 3
    sma_tp = _sma(tp, 20)
    mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci20 = (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))

    # Williams %R (14)
    highest14 = h.rolling(14).max()
    lowest14 = l.rolling(14).min()
    willr14 = -100 * (highest14 - c) / (highest14 - lowest14).replace(0, np.nan)

    # Ultimate Oscillator
    bp = c - pd.concat([l, c.shift(1)], axis=1).min(axis=1)
    tr_uo = pd.concat([h, c.shift(1)], axis=1).max(axis=1) - pd.concat([l, c.shift(1)], axis=1).min(axis=1)
    avg7 = bp.rolling(7).sum() / tr_uo.rolling(7).sum().replace(0, np.nan)
    avg14 = bp.rolling(14).sum() / tr_uo.rolling(14).sum().replace(0, np.nan)
    avg28 = bp.rolling(28).sum() / tr_uo.rolling(28).sum().replace(0, np.nan)
    uo = 100 * ((4 * avg7 + 2 * avg14 + avg28) / 7)

    # ROC(10), Momentum(10)
    roc10 = ((c - c.shift(10)) / c.shift(10).replace(0, np.nan)) * 100
    mom10 = c - c.shift(10)

    # Awesome Oscillator (34/5 midpoint SMA)
    mp = (h + l) / 2
    ao = _sma(mp, 5) - _sma(mp, 34)

    # ====================================================================
    # 6. Trend — ADX / DI / Aroon / Supertrend
    # ====================================================================
    atr14 = _atr(h, l, c, 14)
    tr = _true_range(h, l, c)

    prev_h = h.shift(1)
    prev_l = l.shift(1)
    dm_plus_raw = np.where((h - prev_h) > (prev_l - l), (h - prev_h).clip(lower=0), 0.0)
    dm_minus_raw = np.where((prev_l - l) > (h - prev_h), (prev_l - l).clip(lower=0), 0.0)
    dm_plus = pd.Series(dm_plus_raw, index=c.index)
    dm_minus = pd.Series(dm_minus_raw, index=c.index)

    atr14_s = atr14.replace(0, np.nan)
    di_plus = 100 * dm_plus.ewm(span=14, adjust=False).mean() / atr14_s
    di_minus = 100 * dm_minus.ewm(span=14, adjust=False).mean() / atr14_s
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    adx14 = dx.ewm(span=14, adjust=False).mean()

    # Aroon (25)
    aroon_up = h.rolling(26).apply(lambda x: x.argmax() / 25 * 100, raw=True)
    aroon_down = l.rolling(26).apply(lambda x: x.argmin() / 25 * 100, raw=True)
    aroon_osc = aroon_up - aroon_down

    # Supertrend (ATR mult=3, period=10) — direction only
    atr10 = _atr(h, l, c, 10)
    mult = 3.0
    basic_upper = (h + l) / 2 + mult * atr10
    basic_lower = (h + l) / 2 - mult * atr10

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    supertrend_dir = pd.Series(1.0, index=c.index)

    for i in range(1, n):
        fu_prev = final_upper.iloc[i - 1]
        fl_prev = final_lower.iloc[i - 1]
        c_prev = c.iloc[i - 1]
        bu = basic_upper.iloc[i]
        bl = basic_lower.iloc[i]
        final_upper.iloc[i] = bu if (bu < fu_prev or c_prev > fu_prev) else fu_prev
        final_lower.iloc[i] = bl if (bl > fl_prev or c_prev < fl_prev) else fl_prev
        if supertrend_dir.iloc[i - 1] == -1 and c.iloc[i] > final_upper.iloc[i]:
            supertrend_dir.iloc[i] = 1.0
        elif supertrend_dir.iloc[i - 1] == 1 and c.iloc[i] < final_lower.iloc[i]:
            supertrend_dir.iloc[i] = -1.0
        else:
            supertrend_dir.iloc[i] = supertrend_dir.iloc[i - 1]

    # ====================================================================
    # 7. Volatility
    # ====================================================================
    natr14 = (atr14 / c.replace(0, np.nan)) * 100

    bb_mid = _sma(c, 20)
    bb_std = c.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower_s = bb_mid - 2 * bb_std
    bb_width = (bb_upper - bb_lower_s) / bb_mid.replace(0, np.nan)
    bb_pct = (c - bb_lower_s) / (bb_upper - bb_lower_s).replace(0, np.nan)

    # Keltner Channel width (EMA20 ± 2×ATR10)
    kc_mid = _ema(c, 20)
    kc_upper_k = kc_mid + 2 * atr10
    kc_lower_k = kc_mid - 2 * atr10
    kc_width = (kc_upper_k - kc_lower_k) / kc_mid.replace(0, np.nan)

    # ====================================================================
    # 8. Volume / Microstructure
    # ====================================================================
    # MFI (14)
    mf_raw = tp * v
    pos_mf = mf_raw.where(tp > tp.shift(1), 0.0)
    neg_mf = mf_raw.where(tp < tp.shift(1), 0.0)
    mfr = pos_mf.rolling(14).sum() / neg_mf.rolling(14).sum().replace(0, np.nan)
    mfi14 = 100 - (100 / (1 + mfr))

    # CMF (20)
    clv = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
    cmf20 = (clv * v).rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)

    # OBV z-score (21)
    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    obv_z21 = _zscore(obv, 21)

    # VWAP deviation (rolling 20-bar approximation)
    vwap_20 = (c * v).rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)
    vwap_dev = (c - vwap_20) / vwap_20.replace(0, np.nan)

    # A/D Line z-score
    ad = (clv * v).fillna(0).cumsum()
    ad_z21 = _zscore(ad, 21)

    # Force Index EMA-13
    force_raw = c.diff() * v
    force_idx13 = _ema(force_raw, 13)
    force_idx13_z = _zscore(force_idx13, 21)

    # Ease of Movement (14-bar SMA)
    hl_avg = (h + l) / 2
    eom_raw = (hl_avg.diff() / (v / (h - l).replace(0, np.nan)).replace(0, np.nan))
    eom14 = _sma(eom_raw, 14)
    eom14_z = _zscore(eom14, 21)

    # PVT z-score
    pvt = (c.pct_change() * v).fillna(0).cumsum()
    pvt_z21 = _zscore(pvt, 21)

    # NVI / PVI
    ret = c.pct_change()
    nvi = pd.Series(1000.0, index=c.index)
    pvi = pd.Series(1000.0, index=c.index)
    for i in range(1, n):
        if v.iloc[i] < v.iloc[i - 1]:
            nvi.iloc[i] = nvi.iloc[i - 1] * (1 + ret.iloc[i])
        else:
            nvi.iloc[i] = nvi.iloc[i - 1]
        if v.iloc[i] > v.iloc[i - 1]:
            pvi.iloc[i] = pvi.iloc[i - 1] * (1 + ret.iloc[i])
        else:
            pvi.iloc[i] = pvi.iloc[i - 1]
    nvi_z21 = _zscore(nvi, 21)
    pvi_z21 = _zscore(pvi, 21)

    # VWAP range position (where today's close sits in 20-bar VWAP ± 2σ band)
    vwap_std = c.rolling(20).std()
    vwap_band = (2 * vwap_std).replace(0, np.nan)
    vwap_range_pct = (c - vwap_20) / vwap_band

    # Volume ratio 10/50
    vol_ratio = v.rolling(10).mean() / v.rolling(50).mean().replace(0, np.nan)

    # ====================================================================
    # Apply strict .shift(1) no-lookahead before writing to output
    # ====================================================================
    assignments = {
        # Moving averages
        "jesse_sma_10": sma10,
        "jesse_sma_20": sma20,
        "jesse_sma_50": sma50,
        "jesse_ema_9": ema9,
        "jesse_ema_21": ema21,
        "jesse_wma_10": wma10,
        "jesse_vwma_10": vwma10,
        # Ichimoku
        "jesse_ichi_tenkan": tenkan,
        "jesse_ichi_kijun": kijun,
        "jesse_ichi_cloud_diff": cloud_diff,
        "jesse_ichi_cloud_bull": cloud_bull,
        "jesse_ichi_chikou_diff": chikou_diff,
        # Donchian
        "jesse_dc_upper": dc_upper,
        "jesse_dc_lower": dc_lower,
        "jesse_dc_mid": dc_mid,
        # MACD
        "jesse_macd": macd_line,
        "jesse_macd_signal": macd_signal,
        "jesse_macd_hist": macd_hist,
        # Momentum oscillators
        "jesse_rsi_14": rsi14,
        "jesse_rsi_7": rsi7,
        "jesse_stoch_k": stoch_k,
        "jesse_stoch_d": stoch_d,
        "jesse_cci_20": cci20,
        "jesse_willr_14": willr14,
        "jesse_ultimate_osc": uo,
        "jesse_roc_10": roc10,
        "jesse_mom_10": mom10,
        "jesse_ao": ao,
        # Trend
        "jesse_adx_14": adx14,
        "jesse_di_plus": di_plus,
        "jesse_di_minus": di_minus,
        "jesse_aroon_up": aroon_up,
        "jesse_aroon_down": aroon_down,
        "jesse_aroon_osc": aroon_osc,
        "jesse_supertrend_dir": supertrend_dir,
        # Volatility
        "jesse_atr_14": atr14,
        "jesse_natr_14": natr14,
        "jesse_bb_upper": bb_upper,
        "jesse_bb_lower": bb_lower_s,
        "jesse_bb_width": bb_width,
        "jesse_bb_pct": bb_pct,
        "jesse_kc_width": kc_width,
        # Volume
        "jesse_mfi_14": mfi14,
        "jesse_cmf_20": cmf20,
        "jesse_obv_z21": obv_z21,
        "jesse_vwap_dev": vwap_dev,
        "jesse_ad_z21": ad_z21,
        "jesse_force_idx_13": force_idx13_z,
        "jesse_eom_14": eom14_z,
        "jesse_pvt_z21": pvt_z21,
        "jesse_nvi_z21": nvi_z21,
        "jesse_pvi_z21": pvi_z21,
        "jesse_vwap_range_pct": vwap_range_pct,
        "jesse_vol_ratio_10_50": vol_ratio,
    }

    for col_name, raw_series in assignments.items():
        # .shift(1) applied unconditionally — no-lookahead guarantee
        out[col_name] = raw_series.shift(1).values

    new_cols = [c for c in JESSE_FEATURE_NAMES if c in out.columns]
    logger.debug("[jesse] wrote %d/%d features for ticker=%s", len(new_cols), JESSE_FEATURE_COUNT, ticker)
    return out
