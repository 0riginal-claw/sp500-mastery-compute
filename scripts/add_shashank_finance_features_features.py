"""
add_shashank_finance_features_features.py — 58 features inspired by shashankvemuri/Finance.

Source: github:shashankvemuri/Finance (MIT License)
License: MIT — clean, no Commons Clause, no copyleft.
Requires paid API: NO — pure OHLCV calculation, no external calls.

NO-LOOKAHEAD AUDIT
------------------
Every raw indicator series is computed over the full OHLCV history using
vectorised pandas/numpy rolling/EWM operations.  After computation, the
ENTIRE indicator column is shifted forward by 1 bar (.shift(1)) before being
written into the output DataFrame.  On any given row t the model therefore
sees only information available at the close of bar t-1.

Rolling windows (e.g. .rolling(20).mean()) on the original OHLCV series
represent lookback over prior bars; after .shift(1) they become equivalent
to computing on bars 0..t-2 relative to the current row t — fully safe.

Boolean flags derived from the same shifted indicators inherit the same
no-lookahead property.  No column references any value from the current bar.

Implementation: vectorised pandas/numpy only — no TA-library dependency.
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

SHF_FEATURE_COUNT: int = 58

SHF_FEATURE_NAMES: list[str] = [
    # Moving Average Ratios (8)
    "shf_close_sma5_ratio", "shf_close_sma10_ratio",
    "shf_close_sma20_ratio", "shf_close_sma50_ratio",
    "shf_sma5_sma20_ratio", "shf_sma20_sma50_ratio",
    "shf_sma50_sma200_ratio", "shf_ema8_ema21_ratio",
    # RSI Variants (4)
    "shf_rsi_9", "shf_rsi_21",
    "shf_rsi_divergence_14", "shf_rsi_regime",
    # MACD Variants (5)
    "shf_macd_8_17_9", "shf_macd_signal_8_17_9", "shf_macd_hist_8_17_9",
    "shf_macd_pct_price", "shf_macd_momentum",
    # Stochastic (3)
    "shf_stoch_k_14_3", "shf_stoch_d_14_3", "shf_stoch_regime",
    # Bollinger Bands (5)
    "shf_bb_upper_pct", "shf_bb_lower_pct", "shf_bb_width_pct",
    "shf_bb_position", "shf_bb_squeeze_flag",
    # Volatility (5)
    "shf_hist_vol_5", "shf_hist_vol_10", "shf_hist_vol_21",
    "shf_vol_ratio_5_21", "shf_vol_of_vol_21",
    # Volume (6)
    "shf_volume_sma_ratio_5", "shf_volume_sma_ratio_20",
    "shf_obv_sma_ratio_10", "shf_volume_momentum_5",
    "shf_force_index_1", "shf_eom_14",
    # Trend (5)
    "shf_adx_10", "shf_aroon_diff_14", "shf_cci_14", "shf_cci_40",
    "shf_psar_direction",
    # Price Action (6)
    "shf_doji_flag", "shf_hammer_flag", "shf_engulfing_bull_flag",
    "shf_gap_up_flag", "shf_high_52w_pct", "shf_low_52w_pct",
    # Mean Reversion (4)
    "shf_zscore_close_10", "shf_zscore_close_21", "shf_zscore_close_63",
    "shf_distance_from_52w_high",
    # Support / Resistance (3)
    "shf_pivot_high_5", "shf_pivot_low_5", "shf_hh_hl_flag",
    # Additional Oscillators (4)
    "shf_willr_14", "shf_stoch_rsi_14", "shf_mfi_9", "shf_cmf_14",
]

assert len(SHF_FEATURE_NAMES) == SHF_FEATURE_COUNT, (
    f"Feature name count mismatch: {len(SHF_FEATURE_NAMES)} vs {SHF_FEATURE_COUNT}"
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def _sma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=window).mean()


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _zscore(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window, min_periods=window).mean()
    sigma = s.rolling(window, min_periods=window).std().replace(0, np.nan)
    return (s - mu) / sigma


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    dm_plus_raw = high - high.shift(1)
    dm_minus_raw = low.shift(1) - low
    dm_plus = dm_plus_raw.where((dm_plus_raw > 0) & (dm_plus_raw > dm_minus_raw), 0.0)
    dm_minus = dm_minus_raw.where((dm_minus_raw > 0) & (dm_minus_raw > dm_plus_raw), 0.0)
    alpha = 1.0 / period
    atr_w = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    dip = 100 * dm_plus.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / atr_w.replace(0, np.nan)
    dim = 100 * dm_minus.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / atr_w.replace(0, np.nan)
    dx = 100 * (dip - dim).abs() / (dip + dim).replace(0, np.nan)
    return dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()


def _cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tp = (high + low + close) / 3
    tp_sma = tp.rolling(period, min_periods=period).mean()
    mean_dev = tp.rolling(period, min_periods=period).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )
    return (tp - tp_sma) / (0.015 * mean_dev.replace(0, np.nan))


def _mfi(high: pd.Series, low: pd.Series, close: pd.Series,
         volume: pd.Series, period: int = 9) -> pd.Series:
    tp = (high + low + close) / 3
    raw_mf = tp * volume
    tp_chg = tp.diff()
    pos_mf = raw_mf.where(tp_chg > 0, 0.0)
    neg_mf = raw_mf.where(tp_chg < 0, 0.0)
    mfr = pos_mf.rolling(period).sum() / neg_mf.rolling(period).sum().replace(0, np.nan)
    return 100 - 100 / (1 + mfr)


def _cmf(high: pd.Series, low: pd.Series, close: pd.Series,
         volume: pd.Series, period: int = 14) -> pd.Series:
    hl = (high - low).replace(0, np.nan)
    mf_mult = ((close - low) - (high - close)) / hl
    mf_vol = mf_mult * volume
    return mf_vol.rolling(period).sum() / volume.rolling(period).sum().replace(0, np.nan)


def _aroon_diff(high: pd.Series, low: pd.Series, period: int = 14) -> pd.Series:
    win = period + 1
    aroon_up = high.rolling(win).apply(
        lambda x: float(np.argmax(x)) / period * 100, raw=True
    )
    aroon_dn = low.rolling(win).apply(
        lambda x: float(np.argmin(x)) / period * 100, raw=True
    )
    return aroon_up - aroon_dn


# ---------------------------------------------------------------------------
# Main compute function
# ---------------------------------------------------------------------------

def compute_add_shashank_finance_features(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
) -> pd.DataFrame:
    """Compute 58 shashankvemuri/Finance-inspired features and append to df.

    All output columns are shifted 1 bar (.shift(1)) before assignment —
    strict no-lookahead compliance.  See module docstring for audit details.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: high, low, close, volume (open optional; falls back to close).
        Index should be a DatetimeIndex.
    ticker : str, optional
        Ignored — provided for API uniformity.

    Returns
    -------
    pd.DataFrame
        Input df with 58 new shf_* columns appended.
    """
    out = df.copy()

    col_map = {c.lower(): c for c in out.columns}

    def _col(name: str) -> pd.Series:
        return out[col_map[name]].astype(float)

    try:
        h = _col("high")
        l = _col("low")
        c = _col("close")
        v = _col("volume")
    except KeyError as e:
        logger.warning("[shf] Missing required column %s — zero-filling all features", e)
        for col in SHF_FEATURE_NAMES:
            out[col] = 0.0
        return out

    try:
        o = _col("open")
    except KeyError:
        o = c.copy()

    # ---- Moving Average Ratios ------------------------------------------
    sma5 = _sma(c, 5)
    sma10 = _sma(c, 10)
    sma20 = _sma(c, 20)
    sma50 = _sma(c, 50)
    sma200 = _sma(c, 200)
    ema8 = _ema(c, 8)
    ema21 = _ema(c, 21)

    close_sma5_ratio = c / sma5.replace(0, np.nan) - 1
    close_sma10_ratio = c / sma10.replace(0, np.nan) - 1
    close_sma20_ratio = c / sma20.replace(0, np.nan) - 1
    close_sma50_ratio = c / sma50.replace(0, np.nan) - 1
    sma5_sma20_ratio = sma5 / sma20.replace(0, np.nan) - 1
    sma20_sma50_ratio = sma20 / sma50.replace(0, np.nan) - 1
    sma50_sma200_ratio = sma50 / sma200.replace(0, np.nan) - 1
    ema8_ema21_ratio = ema8 / ema21.replace(0, np.nan) - 1

    # ---- RSI Variants ---------------------------------------------------
    rsi9 = _rsi(c, 9)
    rsi14 = _rsi(c, 14)
    rsi21 = _rsi(c, 21)
    rsi_divergence_14 = rsi14 - rsi14.shift(5)
    rsi_regime = pd.cut(rsi14, bins=[-np.inf, 30, 70, np.inf], labels=[0, 1, 2]).astype(float)

    # ---- MACD Variants --------------------------------------------------
    ema8_c = _ema(c, 8)
    ema17_c = _ema(c, 17)
    macd_8_17 = ema8_c - ema17_c
    macd_signal = _ema(macd_8_17, 9)
    macd_hist = macd_8_17 - macd_signal
    macd_pct_price = macd_8_17 / c.replace(0, np.nan)
    macd_momentum = macd_hist.diff(1)

    # ---- Stochastic -----------------------------------------------------
    lowest14 = l.rolling(14, min_periods=14).min()
    highest14 = h.rolling(14, min_periods=14).max()
    stoch_k = 100 * (c - lowest14) / (highest14 - lowest14).replace(0, np.nan)
    stoch_d = stoch_k.rolling(3).mean()
    stoch_regime = pd.cut(stoch_k, bins=[-np.inf, 20, 80, np.inf], labels=[0, 1, 2]).astype(float)

    # ---- Bollinger Bands ------------------------------------------------
    bb_mid = _sma(c, 20)
    bb_std = c.rolling(20, min_periods=20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = bb_upper - bb_lower
    bb_upper_pct = (c - bb_upper) / c.replace(0, np.nan)
    bb_lower_pct = (c - bb_lower) / c.replace(0, np.nan)
    bb_width_pct = bb_width / bb_mid.replace(0, np.nan)
    bb_position = (c - bb_lower) / bb_width.replace(0, np.nan)
    bb_squeeze_flag = (bb_width_pct < 0.1).astype(float)

    # ---- Volatility -----------------------------------------------------
    log_ret = np.log(c / c.shift(1))
    hist_vol_5 = log_ret.rolling(5, min_periods=5).std()
    hist_vol_10 = log_ret.rolling(10, min_periods=10).std()
    hist_vol_21 = log_ret.rolling(21, min_periods=21).std()
    vol_ratio_5_21 = hist_vol_5 / hist_vol_21.replace(0, np.nan)
    vol_of_vol_21 = hist_vol_5.rolling(21, min_periods=10).std()

    # ---- Volume ---------------------------------------------------------
    vol_sma5 = _sma(v, 5)
    vol_sma20 = _sma(v, 20)
    volume_sma_ratio_5 = v / vol_sma5.replace(0, np.nan)
    volume_sma_ratio_20 = v / vol_sma20.replace(0, np.nan)

    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    obv_sma10 = _sma(obv, 10)
    obv_sma_ratio_10 = obv / obv_sma10.replace(0, np.nan)

    volume_momentum_5 = v / v.shift(5).replace(0, np.nan)
    force_index_1 = c.diff() * v

    midpoint_move = (h + l) / 2 - (h.shift(1) + l.shift(1)) / 2
    hl_range = (h - l).replace(0, np.nan)
    box_ratio = (v / 1e8) / hl_range
    eom_raw = midpoint_move / box_ratio.replace(0, np.nan)
    eom_14 = eom_raw.rolling(14, min_periods=5).mean()

    # ---- Trend ----------------------------------------------------------
    adx_10 = _adx(h, l, c, 10)
    aroon_diff_14 = _aroon_diff(h, l, 14)
    cci_14 = _cci(h, l, c, 14)
    cci_40 = _cci(h, l, c, 40)
    dc_mid_10 = (h.rolling(10, min_periods=5).max() + l.rolling(10, min_periods=5).min()) / 2
    psar_direction = np.sign(c - dc_mid_10).fillna(0)

    # ---- Price Action ---------------------------------------------------
    body = (c - o).abs()
    full_range = (h - l).replace(0, np.nan)
    doji_flag = (body / full_range < 0.1).astype(float)

    lower_shadow = (o.clip(upper=c) - l).clip(lower=0)
    hammer_flag = ((lower_shadow / body.replace(0, np.nan)) > 2).astype(float)

    bull_bar = c > o
    bear_prev = c.shift(1) < o.shift(1)
    engulf_open = o < c.shift(1)
    engulf_close = c > o.shift(1)
    engulfing_bull_flag = (bull_bar & bear_prev & engulf_open & engulf_close).astype(float)

    gap_up_flag = (o > c.shift(1) * 1.005).astype(float)

    w252_high = h.rolling(252, min_periods=20).max()
    w252_low = l.rolling(252, min_periods=20).min()
    w252_range = (w252_high - w252_low).replace(0, np.nan)
    high_52w_pct = (c - w252_low) / w252_range
    low_52w_pct = (w252_high - c) / w252_range

    # ---- Mean Reversion -------------------------------------------------
    zscore_close_10 = _zscore(c, 10)
    zscore_close_21 = _zscore(c, 21)
    zscore_close_63 = _zscore(c, 63)
    distance_from_52w_high = (w252_high - c) / c.replace(0, np.nan)

    # ---- Support / Resistance -------------------------------------------
    pivot_high_5 = c / h.rolling(5, min_periods=5).max().replace(0, np.nan) - 1
    pivot_low_5 = c / l.rolling(5, min_periods=5).min().replace(0, np.nan) - 1
    hh_hl_flag = (
        (c > c.shift(5)) & (l > l.shift(5))
    ).astype(float)

    # ---- Additional Oscillators -----------------------------------------
    willr_14 = -100 * (h.rolling(14).max() - c) / (h.rolling(14).max() - l.rolling(14).min()).replace(0, np.nan)

    rsi14_for_srsi = rsi14
    srsi_min = rsi14_for_srsi.rolling(14).min()
    srsi_max = rsi14_for_srsi.rolling(14).max()
    stoch_rsi_14 = (rsi14_for_srsi - srsi_min) / (srsi_max - srsi_min).replace(0, np.nan)

    mfi_9 = _mfi(h, l, c, v, 9)
    cmf_14 = _cmf(h, l, c, v, 14)

    # ---- Assemble with .shift(1) — strict no-lookahead -----------------
    feature_series = [
        # MA ratios (8)
        close_sma5_ratio, close_sma10_ratio, close_sma20_ratio, close_sma50_ratio,
        sma5_sma20_ratio, sma20_sma50_ratio, sma50_sma200_ratio, ema8_ema21_ratio,
        # RSI (4)
        rsi9, rsi21, rsi_divergence_14, rsi_regime,
        # MACD (5)
        macd_8_17, macd_signal, macd_hist, macd_pct_price, macd_momentum,
        # Stochastic (3)
        stoch_k, stoch_d, stoch_regime,
        # Bollinger Bands (5)
        bb_upper_pct, bb_lower_pct, bb_width_pct, bb_position, bb_squeeze_flag,
        # Volatility (5)
        hist_vol_5, hist_vol_10, hist_vol_21, vol_ratio_5_21, vol_of_vol_21,
        # Volume (6)
        volume_sma_ratio_5, volume_sma_ratio_20, obv_sma_ratio_10,
        volume_momentum_5, force_index_1, eom_14,
        # Trend (5)
        adx_10, aroon_diff_14, cci_14, cci_40, psar_direction,
        # Price Action (6)
        doji_flag, hammer_flag, engulfing_bull_flag, gap_up_flag,
        high_52w_pct, low_52w_pct,
        # Mean Reversion (4)
        zscore_close_10, zscore_close_21, zscore_close_63, distance_from_52w_high,
        # Support/Resistance (3)
        pivot_high_5, pivot_low_5, hh_hl_flag,
        # Additional Oscillators (4)
        willr_14, stoch_rsi_14, mfi_9, cmf_14,
    ]

    assert len(feature_series) == SHF_FEATURE_COUNT, (
        f"Series count {len(feature_series)} != SHF_FEATURE_COUNT {SHF_FEATURE_COUNT}"
    )

    for name, series in zip(SHF_FEATURE_NAMES, feature_series):
        out[name] = series.shift(1)

    return out


# ---------------------------------------------------------------------------
# v10 naming aliases (required by backtest_xgb_v10 Helper GT import block)
# ---------------------------------------------------------------------------

SHASHANK_FEATURE_COUNT: int = SHF_FEATURE_COUNT
SHASHANK_FEATURE_NAMES: list[str] = SHF_FEATURE_NAMES


def compute_add_shashank_finance_features_features(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
) -> pd.DataFrame:
    """Alias for compute_add_shashank_finance_features — v10 naming convention."""
    return compute_add_shashank_finance_features(df, ticker=ticker)
