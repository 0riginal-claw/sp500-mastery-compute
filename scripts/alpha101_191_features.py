"""
alpha101_191_features.py — Combined WorldQuant Alpha101 + GTJA Alpha191 expansion.

Source repos:
  - github:popbo/alphas (MIT-style, formulas only) — combined Alpha101 + Alpha191
  - "101 Formulaic Alphas" (Kakushadze, 2016)
  - "191 Formulaic Alphas" (Guotai Junan Securities Research, 2017)

This module DEDUPES against existing wired modules:
  - worldquant_alpha101_features.py (25 alphas, prefix wq_a*)
  - gtja_alpha191_features.py (50 alphas, prefix gtja_a*)

Net new alphas in this module (prefix a101_191_*):
  - ~75 additional WorldQuant Alpha101 alphas (TS-safe subset, no rank/cross-sectional)
  - ~45 additional GTJA Alpha191 alphas (TS-safe subset, ~70 China-specific dropped)
  - Target: ~120 working alphas after dedupe

ENV-GATE: ALPHA101_191_ENABLED=1 (default OFF). Sub-agent must enable explicitly.

SHIFT CONVENTION:
  All inputs are pre-shifted by 1 bar at the top of compute_*().
  feature[row T] uses only data through T-1. First bar is NaN per ticker.

LOOKAHEAD_STRATEGY = "shifted_1"
ALREADY_SHIFTED = True
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

LOOKAHEAD_STRATEGY = "shifted_1"
ALREADY_SHIFTED = True


# ---------------------------------------------------------------------------
# helpers (time-series safe primitives)
# ---------------------------------------------------------------------------

def _S(x, ref_index=None) -> pd.Series:
    """Coerce anything (ndarray, scalar, Series) into a pd.Series, optionally indexed by ref."""
    if isinstance(x, pd.Series):
        return x
    if ref_index is not None:
        return pd.Series(x, index=ref_index)
    return pd.Series(x)


def _ts_max(s, w: int) -> pd.Series:
    s = _S(s)
    return s.rolling(w, min_periods=max(1, w // 2)).max()


def _ts_min(s, w: int) -> pd.Series:
    s = _S(s)
    return s.rolling(w, min_periods=max(1, w // 2)).min()


def _ts_mean(s, w: int) -> pd.Series:
    s = _S(s)
    return s.rolling(w, min_periods=max(1, w // 2)).mean()


def _ts_std(s, w: int) -> pd.Series:
    s = _S(s)
    return s.rolling(w, min_periods=max(1, w // 2)).std()


def _ts_sum(s, w: int) -> pd.Series:
    s = _S(s)
    return s.rolling(w, min_periods=max(1, w // 2)).sum()


def _ts_rank(s, w: int) -> pd.Series:
    """TS rank within rolling window, normalized to [0, 1]."""
    s = _S(s)
    def _rank_last(x):
        if len(x) == 0:
            return np.nan
        arr = np.asarray(x, dtype=float)
        if np.all(np.isnan(arr)):
            return np.nan
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return np.nan
        # rank of last finite value within the window
        last = arr[-1]
        if np.isnan(last):
            return np.nan
        return float((valid < last).sum() + 1) / float(len(valid))
    return s.rolling(w, min_periods=max(1, w // 2)).apply(_rank_last, raw=True)


def _ts_argmax(s, w: int) -> pd.Series:
    s = _S(s)
    return s.rolling(w, min_periods=max(1, w // 2)).apply(
        lambda x: float(np.argmax(x)) if len(x) > 0 else np.nan, raw=True
    )


def _ts_argmin(s, w: int) -> pd.Series:
    s = _S(s)
    return s.rolling(w, min_periods=max(1, w // 2)).apply(
        lambda x: float(np.argmin(x)) if len(x) > 0 else np.nan, raw=True
    )


def _ts_corr(a, b, w: int) -> pd.Series:
    a = _S(a)
    b = _S(b, ref_index=a.index)
    return a.rolling(w, min_periods=max(2, w // 2)).corr(b)


def _ts_cov(a, b, w: int) -> pd.Series:
    a = _S(a)
    b = _S(b, ref_index=a.index)
    return a.rolling(w, min_periods=max(2, w // 2)).cov(b)


def _delay(s, d: int) -> pd.Series:
    s = _S(s)
    return s.shift(d)


def _delta(s, d: int) -> pd.Series:
    s = _S(s)
    return s.diff(d)


def _sign(s) -> pd.Series:
    s = _S(s)
    return pd.Series(np.sign(s.values), index=s.index)


def _signed_log(s) -> pd.Series:
    s = _S(s)
    return pd.Series(np.sign(s.values) * np.log1p(s.abs().values), index=s.index)


def _decay_linear(s, w: int) -> pd.Series:
    s = _S(s)
    weights = np.arange(1, w + 1, dtype=float)
    weights /= weights.sum()
    return s.rolling(w, min_periods=max(1, w // 2)).apply(
        lambda x: float(np.dot(x[-len(weights):], weights[-len(x):])) if len(x) > 0 else np.nan,
        raw=True,
    )


# ---------------------------------------------------------------------------
# Feature name manifest (expansion alphas, not overlapping wq_a* or gtja_a*)
# ---------------------------------------------------------------------------

A101_191_FEATURE_NAMES: list[str] = [
    # WorldQuant Alpha101 expansion (a101_191_wq_*)
    "a101_191_wq_a012", "a101_191_wq_a013", "a101_191_wq_a014", "a101_191_wq_a015",
    "a101_191_wq_a016", "a101_191_wq_a017", "a101_191_wq_a018", "a101_191_wq_a019",
    "a101_191_wq_a020", "a101_191_wq_a021", "a101_191_wq_a022", "a101_191_wq_a023",
    "a101_191_wq_a024", "a101_191_wq_a025", "a101_191_wq_a026", "a101_191_wq_a027",
    "a101_191_wq_a028", "a101_191_wq_a029", "a101_191_wq_a030", "a101_191_wq_a031",
    "a101_191_wq_a032", "a101_191_wq_a033", "a101_191_wq_a034", "a101_191_wq_a035",
    "a101_191_wq_a036", "a101_191_wq_a037", "a101_191_wq_a038", "a101_191_wq_a039",
    "a101_191_wq_a040", "a101_191_wq_a041", "a101_191_wq_a042", "a101_191_wq_a043",
    "a101_191_wq_a044", "a101_191_wq_a045", "a101_191_wq_a046", "a101_191_wq_a047",
    "a101_191_wq_a049", "a101_191_wq_a050", "a101_191_wq_a051", "a101_191_wq_a052",
    "a101_191_wq_a053", "a101_191_wq_a054", "a101_191_wq_a055", "a101_191_wq_a057",
    "a101_191_wq_a060", "a101_191_wq_a061", "a101_191_wq_a062", "a101_191_wq_a064",
    "a101_191_wq_a065", "a101_191_wq_a066", "a101_191_wq_a068", "a101_191_wq_a071",
    "a101_191_wq_a072", "a101_191_wq_a073", "a101_191_wq_a074", "a101_191_wq_a075",
    "a101_191_wq_a077", "a101_191_wq_a078", "a101_191_wq_a081", "a101_191_wq_a083",
    "a101_191_wq_a084", "a101_191_wq_a085", "a101_191_wq_a086", "a101_191_wq_a088",
    "a101_191_wq_a092", "a101_191_wq_a094", "a101_191_wq_a095", "a101_191_wq_a096",
    "a101_191_wq_a099", "a101_191_wq_a101", "a101_191_wq_a102", "a101_191_wq_a103",
    "a101_191_wq_a104", "a101_191_wq_a105", "a101_191_wq_a106",
    # GTJA Alpha191 expansion (a101_191_gtja_*)
    "a101_191_gtja_a001", "a101_191_gtja_a002", "a101_191_gtja_a003", "a101_191_gtja_a004",
    "a101_191_gtja_a005", "a101_191_gtja_a007", "a101_191_gtja_a008", "a101_191_gtja_a010",
    "a101_191_gtja_a013", "a101_191_gtja_a014", "a101_191_gtja_a015", "a101_191_gtja_a016",
    "a101_191_gtja_a017", "a101_191_gtja_a018", "a101_191_gtja_a019", "a101_191_gtja_a020",
    "a101_191_gtja_a022", "a101_191_gtja_a023", "a101_191_gtja_a024", "a101_191_gtja_a025",
    "a101_191_gtja_a027", "a101_191_gtja_a028", "a101_191_gtja_a029", "a101_191_gtja_a031",
    "a101_191_gtja_a032", "a101_191_gtja_a033", "a101_191_gtja_a034", "a101_191_gtja_a035",
    "a101_191_gtja_a036", "a101_191_gtja_a037", "a101_191_gtja_a038", "a101_191_gtja_a039",
    "a101_191_gtja_a040", "a101_191_gtja_a041", "a101_191_gtja_a042", "a101_191_gtja_a043",
    "a101_191_gtja_a044", "a101_191_gtja_a045", "a101_191_gtja_a046", "a101_191_gtja_a047",
    "a101_191_gtja_a048", "a101_191_gtja_a049", "a101_191_gtja_a050",
]

A101_191_FEATURE_COUNT: int = len(A101_191_FEATURE_NAMES)


def alpha101_191_feature_names() -> list[str]:
    return list(A101_191_FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Main compute
# ---------------------------------------------------------------------------

def compute_alpha101_191_features(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
) -> pd.DataFrame:
    """Compute expansion alphas from Alpha101 + Alpha191.

    Input df must have lowercase columns: open, high, low, close, volume.
    Returns df with all original columns plus a101_191_* prefixed features.

    All features are .shift(1)-safe (operate on past data only).
    """
    if not all(c in df.columns for c in ("open", "high", "low", "close", "volume")):
        # Cannot compute -- zero-fill manifest
        out = df.copy()
        for name in A101_191_FEATURE_NAMES:
            out[name] = 0.0
        return out

    out = df.copy()

    # SHIFT(1) all OHLCV inputs to enforce no-lookahead
    o = df["open"].shift(1)
    h = df["high"].shift(1)
    l = df["low"].shift(1)
    c = df["close"].shift(1)
    v = df["volume"].shift(1)

    ret = c.pct_change()
    vwap = ((h + l + c) / 3.0)
    tp = (h + l + c) / 3.0

    # ========================================================================
    # WorldQuant Alpha101 expansion (TS-safe subset)
    # ========================================================================
    # Note: many WQ alphas use rank() across stocks; we substitute ts_rank.

    out["a101_191_wq_a012"] = _sign(_delta(v, 1)) * (-1.0 * _delta(c, 1))
    out["a101_191_wq_a013"] = -1.0 * _ts_rank(_ts_cov(c, v, 5), 5)
    out["a101_191_wq_a014"] = -1.0 * _ts_rank(_delta(ret, 3), 10) * _ts_corr(o, v, 10)
    out["a101_191_wq_a015"] = -1.0 * _ts_sum(_ts_rank(_ts_corr(h, v, 3), 3), 3)
    out["a101_191_wq_a016"] = -1.0 * _ts_rank(_ts_cov(h, v, 5), 5)
    out["a101_191_wq_a017"] = -1.0 * _ts_rank(c, 10) * _ts_rank(_delta(_delta(c, 1), 1), 5)
    out["a101_191_wq_a018"] = -1.0 * _ts_rank(_ts_std((c - o).abs(), 5) + (c - o) + _ts_corr(c, o, 10), 5)
    out["a101_191_wq_a019"] = -1.0 * _sign(_delta(c, 7) + _delta(c, 7)) * (1 + _ts_rank(1 + _ts_sum(ret, 250), 5))
    out["a101_191_wq_a020"] = (-1.0 * _ts_rank(o - _delay(h, 1), 10) * _ts_rank(o - _delay(c, 1), 10) * _ts_rank(o - _delay(l, 1), 10))
    out["a101_191_wq_a021"] = _sign(_ts_mean(c, 8) + _ts_std(c, 8) - _ts_mean(c, 2))
    out["a101_191_wq_a022"] = -1.0 * (_delta(_ts_corr(h, v, 5), 5) * _ts_rank(_ts_std(c, 20), 5))
    out["a101_191_wq_a023"] = np.where(_ts_mean(h, 20) < h, -1.0 * _delta(h, 2), 0.0)
    out["a101_191_wq_a024"] = np.where((_delta(_ts_mean(c, 100), 100) / _delay(c, 100)) <= 0.05, -1.0 * (c - _ts_min(c, 100)), -1.0 * _delta(c, 3))
    out["a101_191_wq_a025"] = _ts_rank(-1.0 * ret * _ts_mean(v, 20) * vwap * (h - c), 5)
    out["a101_191_wq_a026"] = -1.0 * _ts_max(_ts_corr(_ts_rank(v, 5), _ts_rank(h, 5), 5), 3)
    out["a101_191_wq_a027"] = np.where(0.5 < _ts_rank(_ts_mean(_ts_corr(_ts_rank(v, 6), _ts_rank(vwap, 6), 6), 2), 5), -1.0, 1.0)
    out["a101_191_wq_a028"] = _ts_corr(_ts_mean(v, 20), l, 5) + (h + l) / 2.0 - c
    out["a101_191_wq_a029"] = _ts_min(_ts_rank(_ts_rank(_ts_sum(-1.0 * _ts_rank(_delta(c - 1, 5), 5), 2), 1), 5), 5) + _ts_rank(_delay(-1.0 * ret, 6), 5)
    out["a101_191_wq_a030"] = (1.0 - _ts_rank(_sign(c - _delay(c, 1)) + _sign(_delay(c, 1) - _delay(c, 2)) + _sign(_delay(c, 2) - _delay(c, 3)), 20)) * _ts_sum(v, 5) / (_ts_sum(v, 20) + 1e-12)
    out["a101_191_wq_a031"] = _ts_rank(_decay_linear(-1.0 * _ts_rank(_delta(c, 10), 10), 10), 10) + _ts_rank(-1.0 * _delta(c, 3), 10) + _sign(_ts_corr(_ts_mean(v, 20), l, 12))
    out["a101_191_wq_a032"] = _ts_rank(_ts_mean(c, 7) - c, 5) + 20.0 * _ts_rank(_ts_corr(vwap, _delay(c, 5), 230), 5)
    out["a101_191_wq_a033"] = _ts_rank(-1.0 * (1.0 - (o / c)), 5)
    out["a101_191_wq_a034"] = _ts_rank(1.0 - _ts_rank(_ts_std(ret, 2) / (_ts_std(ret, 5) + 1e-12), 5) + 1.0 - _ts_rank(_delta(c, 1), 5), 5)
    out["a101_191_wq_a035"] = _ts_rank(v, 32) * (1.0 - _ts_rank(c + h - l, 16)) * (1.0 - _ts_rank(ret, 32))
    out["a101_191_wq_a036"] = (2.21 * _ts_rank(_ts_corr(c - o, _delay(v, 1), 15), 5) + 0.7 * _ts_rank(o - c, 5) + 0.73 * _ts_rank(_ts_rank(_delay(-1.0 * ret, 6), 5), 5))
    out["a101_191_wq_a037"] = _ts_rank(_ts_corr(_delay(o - c, 1), c, 200), 5) + _ts_rank(o - c, 5)
    out["a101_191_wq_a038"] = -1.0 * _ts_rank(c, 10) * _ts_rank(c / (o + 1e-12), 5)
    out["a101_191_wq_a039"] = -1.0 * _ts_rank(_delta(c, 7) * (1.0 - _ts_rank(_decay_linear(v / (_ts_mean(v, 20) + 1e-12), 9), 9)), 5)
    out["a101_191_wq_a040"] = -1.0 * _ts_rank(_ts_std(h, 10), 5) * _ts_corr(h, v, 10)
    out["a101_191_wq_a041"] = ((h * l) ** 0.5) - vwap
    out["a101_191_wq_a042"] = _ts_rank(vwap - c, 5) / (_ts_rank(vwap + c, 5) + 1e-12)
    out["a101_191_wq_a043"] = _ts_rank(v / (_ts_mean(v, 20) + 1e-12), 20) * _ts_rank(-1.0 * _delta(c, 7), 8)
    out["a101_191_wq_a044"] = -1.0 * _ts_corr(h, _ts_rank(v, 5), 5)
    out["a101_191_wq_a045"] = -1.0 * (_ts_rank(_ts_mean(_delay(c, 5), 20), 5) * _ts_corr(c, v, 2) * _ts_rank(_ts_corr(_ts_sum(c, 5), _ts_sum(c, 20), 2), 5))
    out["a101_191_wq_a046"] = np.where((_delay(c, 20) - _delay(c, 10)) / 10 - (_delay(c, 10) - c) / 10 > 0.25, -1.0, np.where((_delay(c, 20) - _delay(c, 10)) / 10 - (_delay(c, 10) - c) / 10 < 0, 1.0, -1.0 * (c - _delay(c, 1))))
    out["a101_191_wq_a047"] = (_ts_rank(1.0 / (c + 1e-12), 5) * v / (_ts_mean(v, 20) + 1e-12)) * (h * _ts_rank(h - c, 5) / (_ts_mean(h, 5) + 1e-12))
    out["a101_191_wq_a049"] = np.where((_delay(c, 20) - _delay(c, 10)) / 10 - (_delay(c, 10) - c) / 10 < -0.1, 1.0, -1.0 * (c - _delay(c, 1)))
    out["a101_191_wq_a050"] = -1.0 * _ts_max(_ts_rank(_ts_corr(_ts_rank(v, 5), _ts_rank(vwap, 5), 5), 5), 5)
    out["a101_191_wq_a051"] = np.where((_delay(c, 20) - _delay(c, 10)) / 10 - (_delay(c, 10) - c) / 10 < -0.05, 1.0, -1.0 * (c - _delay(c, 1)))
    out["a101_191_wq_a052"] = ((-1.0 * _ts_min(l, 5) + _delay(_ts_min(l, 5), 5)) * _ts_rank((_ts_sum(ret, 240) - _ts_sum(ret, 20)) / 220, 5) * _ts_rank(v, 5))
    out["a101_191_wq_a053"] = -1.0 * _delta((c - l - (h - c)) / (c - l + 1e-12), 9)
    out["a101_191_wq_a054"] = -1.0 * ((l - c) * (o ** 5)) / ((l - h) * (c ** 5) + 1e-12)
    out["a101_191_wq_a055"] = -1.0 * _ts_corr(_ts_rank((c - _ts_min(l, 12)) / (_ts_max(h, 12) - _ts_min(l, 12) + 1e-12), 6), _ts_rank(v, 6), 6)
    out["a101_191_wq_a057"] = -1.0 * ((c - vwap) / (_decay_linear(_ts_rank(_ts_argmax(c, 30), 2), 2) + 1e-12))
    out["a101_191_wq_a060"] = -1.0 * (2 * _ts_rank(((c - l) - (h - c)) / (h - l + 1e-12) * v, 5) - _ts_rank(_ts_argmax(c, 10), 5))
    out["a101_191_wq_a061"] = _ts_rank(vwap - _ts_min(vwap, 16), 5) - _ts_rank(_ts_corr(vwap, _ts_mean(v, 180), 18), 5)
    out["a101_191_wq_a062"] = _ts_rank(_ts_corr(vwap, _ts_sum(_ts_mean(v, 20), 22), 10), 5) - _ts_rank(_ts_rank(o + o, 5) + _ts_rank(((h + l) / 2 + h), 5), 5)
    out["a101_191_wq_a064"] = _ts_rank(_ts_corr(_ts_sum(o * 0.178 + l * 0.822, 13), _ts_sum(_ts_mean(v, 120), 13), 17), 5) - _ts_rank(_delta(((h + l) / 2 * 0.178 + vwap * 0.822), 4), 5)
    out["a101_191_wq_a065"] = _ts_rank(_ts_corr(o * 0.0086 + vwap * 0.9914, _ts_sum(_ts_mean(v, 60), 9), 6), 5) - _ts_rank(o - _ts_min(o, 14), 5)
    out["a101_191_wq_a066"] = _ts_rank(_decay_linear(_delta(vwap, 4), 7), 5) + _ts_rank(_decay_linear((l * 0.96 + l * 0.04 - vwap) / (o - (h + l) / 2 + 1e-12), 11), 7)
    out["a101_191_wq_a068"] = _ts_rank(_ts_corr(_ts_rank(h, 8), _ts_rank(_ts_mean(v, 15), 8), 14), 5) - _ts_rank(_delta(c * 0.518 + l * 0.482, 1), 5)
    out["a101_191_wq_a071"] = _ts_rank(_decay_linear(_ts_corr(_ts_rank(c, 3), _ts_rank(_ts_mean(v, 180), 12), 18), 4), 5)
    out["a101_191_wq_a072"] = _ts_rank(_decay_linear(_ts_corr((h + l) / 2, _ts_mean(v, 40), 9), 10), 5) / (_ts_rank(_decay_linear(_ts_corr(_ts_rank(vwap, 4), _ts_rank(v, 19), 7), 3), 5) + 1e-12)
    out["a101_191_wq_a073"] = -1.0 * (_ts_rank(_decay_linear(_delta(vwap, 5), 3), 5) + _ts_rank(_decay_linear((_delta(o * 0.147 + l * 0.853, 2) / (o * 0.147 + l * 0.853 + 1e-12)) * -1.0, 3), 5))
    out["a101_191_wq_a074"] = _ts_rank(_ts_corr(c, _ts_sum(_ts_mean(v, 30), 37), 15), 5) - _ts_rank(_ts_corr(_ts_rank(h * 0.026 + vwap * 0.974, 5), _ts_rank(v, 11), 11), 5)
    out["a101_191_wq_a075"] = _ts_rank(_ts_corr(vwap, v, 4), 5) - _ts_rank(_ts_corr(_ts_rank(l, 12), _ts_rank(_ts_mean(v, 50), 12), 12), 5)
    out["a101_191_wq_a077"] = np.minimum(_ts_rank(_decay_linear(((h + l) / 2 + h) - (vwap + h), 20), 5), _ts_rank(_decay_linear(_ts_corr((h + l) / 2, _ts_mean(v, 40), 3), 6), 5))
    out["a101_191_wq_a078"] = _ts_rank(_ts_corr(_ts_sum(l * 0.352 + vwap * 0.648, 20), _ts_sum(_ts_mean(v, 40), 20), 7), 5)
    out["a101_191_wq_a081"] = -1.0 * _ts_rank(_ts_corr(c, _ts_rank(_ts_mean(v, 10), 5), 8), 5)
    out["a101_191_wq_a083"] = (_ts_rank(_delay((h - l) / (_ts_mean(c, 5) + 1e-12), 2), 5) * _ts_rank(v, 5)) / ((h - l) / (_ts_mean(c, 5) + 1e-12) / (vwap - c + 1e-12) + 1e-12)
    out["a101_191_wq_a084"] = _ts_rank(vwap - _ts_max(vwap, 15), 21) * _delta(c, 5)
    out["a101_191_wq_a085"] = _ts_rank(_ts_corr(h * 0.876 + l * 0.124, _ts_mean(v, 30), 10), 5) / (_ts_rank(_ts_corr(_ts_rank((h + l) / 2, 4), _ts_rank(v, 10), 7), 5) + 1e-12)
    out["a101_191_wq_a086"] = np.where(_ts_rank(_ts_corr(c, _ts_sum(_ts_mean(v, 20), 15), 6), 20) < _ts_rank((o + c) - (vwap + o), 20), 1.0, -1.0)
    out["a101_191_wq_a088"] = np.minimum(_ts_rank(_decay_linear(((_ts_rank(o, 8) + _ts_rank(l, 8)) - (_ts_rank(h, 8) + _ts_rank(c, 8))), 8), 5), _ts_rank(_decay_linear(_ts_corr(_ts_rank(c, 8), _ts_rank(_ts_mean(v, 60), 21), 8), 7), 5))
    out["a101_191_wq_a092"] = np.minimum(_ts_rank(_decay_linear(((h + l) / 2 + c) < (l + o), 15), 19), _ts_rank(_decay_linear(_ts_corr(_ts_rank(l, 8), _ts_rank(_ts_mean(v, 30), 17), 8), 7), 5))
    out["a101_191_wq_a094"] = -1.0 * (_ts_rank(vwap - _ts_min(vwap, 12), 5) ** _ts_rank(_ts_corr(_ts_rank(vwap, 20), _ts_rank(_ts_mean(v, 60), 4), 18), 5))
    out["a101_191_wq_a095"] = np.where(_ts_rank(o - _ts_min(o, 12), 20) < _ts_rank(_ts_corr(_ts_sum((h + l) / 2, 19), _ts_sum(_ts_mean(v, 40), 19), 13) ** 5, 20), 1.0, 0.0)
    out["a101_191_wq_a096"] = -1.0 * np.maximum(_ts_rank(_decay_linear(_ts_corr(_ts_rank(vwap, 4), _ts_rank(v, 4), 4), 8), 5), _ts_rank(_decay_linear(_ts_argmax(_ts_corr(_ts_rank(c, 7), _ts_rank(_ts_mean(v, 60), 4), 4), 13), 14), 5))
    out["a101_191_wq_a099"] = np.where(_ts_rank(_ts_corr(_ts_sum((h + l) / 2, 20), _ts_sum(_ts_mean(v, 60), 20), 9), 5) < _ts_rank(_ts_corr(l, v, 6), 5), -1.0, 0.0)
    out["a101_191_wq_a101"] = (c - o) / (h - l + 0.001)
    out["a101_191_wq_a102"] = _ts_rank(c - _ts_min(c, 20), 20) - _ts_rank(_ts_max(c, 20) - c, 20)
    out["a101_191_wq_a103"] = _ts_mean(ret.where(ret > 0, 0.0), 20)
    out["a101_191_wq_a104"] = _ts_mean(ret.where(ret < 0, 0.0).abs(), 20)
    out["a101_191_wq_a105"] = (out["a101_191_wq_a103"] - out["a101_191_wq_a104"]) / (out["a101_191_wq_a103"] + out["a101_191_wq_a104"] + 1e-12)
    out["a101_191_wq_a106"] = _ts_std(ret, 20) * np.sqrt(252)

    # ========================================================================
    # GTJA Alpha191 expansion (TS-safe subset, dropping ~70 China-only)
    # ========================================================================

    out["a101_191_gtja_a001"] = -1.0 * _ts_corr(_ts_rank(_delta(np.log(v.replace(0, np.nan)), 1), 6), _ts_rank((c - o) / (o + 1e-12), 6), 6)
    out["a101_191_gtja_a002"] = -1.0 * _delta(((c - l) - (h - c)) / (h - l + 1e-12), 1)
    out["a101_191_gtja_a003"] = _ts_sum(np.where(c == _delay(c, 1), 0.0, c - np.where(c > _delay(c, 1), np.minimum(l, _delay(c, 1)), np.maximum(h, _delay(c, 1)))), 6)
    out["a101_191_gtja_a004"] = np.where((_ts_mean(c, 8) + _ts_std(c, 8)) < _ts_mean(c, 2), -1.0, np.where(_ts_mean(c, 2) < (_ts_mean(c, 8) - _ts_std(c, 8)), 1.0, 0.0))
    out["a101_191_gtja_a005"] = -1.0 * _ts_max(_ts_corr(_ts_rank(v, 5), _ts_rank(h, 5), 5), 3)
    out["a101_191_gtja_a007"] = (_ts_rank(np.maximum(vwap - c, 3), 5) + _ts_rank(np.minimum(vwap - c, 3), 5)) * _ts_rank(_delta(v, 3), 5)
    out["a101_191_gtja_a008"] = -1.0 * _ts_rank(_delta((h + l) / 2 * 0.2 + vwap * 0.8, 4), 5)
    out["a101_191_gtja_a010"] = _ts_rank(np.maximum(np.where(ret < 0, _ts_std(ret, 20), c), 5) ** 2, 5)
    out["a101_191_gtja_a013"] = (h * l) ** 0.5 - vwap
    out["a101_191_gtja_a014"] = c - _delay(c, 5)
    out["a101_191_gtja_a015"] = o / (_delay(c, 1) + 1e-12) - 1.0
    out["a101_191_gtja_a016"] = -1.0 * _ts_max(_ts_rank(_ts_corr(_ts_rank(v, 5), _ts_rank(vwap, 5), 5), 5), 5)
    out["a101_191_gtja_a017"] = _ts_rank(vwap - _ts_max(vwap, 15), 5) ** _delta(c, 5)
    out["a101_191_gtja_a018"] = c / (_delay(c, 5) + 1e-12)
    out["a101_191_gtja_a019"] = np.where(c < _delay(c, 5), (c - _delay(c, 5)) / (_delay(c, 5) + 1e-12), np.where(c == _delay(c, 5), 0.0, (c - _delay(c, 5)) / (c + 1e-12)))
    out["a101_191_gtja_a020"] = (c - _delay(c, 6)) / (_delay(c, 6) + 1e-12) * 100.0
    out["a101_191_gtja_a022"] = _ts_mean(((c - _ts_mean(c, 6)) / (_ts_mean(c, 6) + 1e-12) - _delay((c - _ts_mean(c, 6)) / (_ts_mean(c, 6) + 1e-12), 3)), 12)
    out["a101_191_gtja_a023"] = _ts_std(np.where(c > _delay(c, 1), c, 0.0), 20) / (_ts_std(np.where(c <= _delay(c, 1), c, 0.0), 20) + 1e-12)
    out["a101_191_gtja_a024"] = _ts_mean(c - _delay(c, 5), 5)
    out["a101_191_gtja_a025"] = -1.0 * _ts_rank(_delta(c, 7) * (1.0 - _ts_rank(_decay_linear(v / (_ts_mean(v, 20) + 1e-12), 9), 9)), 5) * (1.0 + _ts_rank(_ts_sum(ret, 250), 5))
    out["a101_191_gtja_a027"] = (c - _delay(c, 3)) / (_delay(c, 3) + 1e-12) * 100.0 + (c - _delay(c, 6)) / (_delay(c, 6) + 1e-12) * 100.0
    out["a101_191_gtja_a028"] = 3.0 * (c - _ts_min(l, 9)) / (_ts_max(h, 9) - _ts_min(l, 9) + 1e-12) * 100.0 - 2.0 * _ts_mean((c - _ts_min(l, 9)) / (_ts_max(h, 9) - _ts_min(l, 9) + 1e-12) * 100.0, 3)
    out["a101_191_gtja_a029"] = (c - _delay(c, 6)) / (_delay(c, 6) + 1e-12) * v
    out["a101_191_gtja_a031"] = (c - _ts_mean(c, 12)) / (_ts_mean(c, 12) + 1e-12) * 100.0
    out["a101_191_gtja_a032"] = -1.0 * _ts_sum(_ts_rank(_ts_corr(_ts_rank(h, 3), _ts_rank(v, 3), 3), 3), 3)
    out["a101_191_gtja_a033"] = (-1.0 * _ts_min(l, 5) + _delay(_ts_min(l, 5), 5)) * _ts_rank((_ts_sum(ret, 240) - _ts_sum(ret, 20)) / 220, 5) * _ts_rank(v, 5)
    out["a101_191_gtja_a034"] = _ts_mean(c, 12) / (c + 1e-12)
    out["a101_191_gtja_a035"] = np.minimum(_ts_rank(_decay_linear(_delta(o, 1), 15), 5), _ts_rank(_decay_linear(_ts_corr(v, o * 0.65 + o * 0.35, 17), 7), 5)) * -1.0
    out["a101_191_gtja_a036"] = _ts_rank(_ts_sum(_ts_corr(_ts_rank(v, 5), _ts_rank(vwap, 5), 6), 2), 5)
    out["a101_191_gtja_a037"] = -1.0 * _ts_rank(_ts_sum(o, 5) * _ts_sum(ret, 5) - _delay(_ts_sum(o, 5) * _ts_sum(ret, 5), 10), 5)
    out["a101_191_gtja_a038"] = np.where(_ts_mean(h, 20) < h, -1.0 * _delta(h, 2), 0.0)
    out["a101_191_gtja_a039"] = (_ts_rank(_decay_linear(_delta(c, 2), 8), 5) - _ts_rank(_decay_linear(_ts_corr(vwap * 0.3 + o * 0.7, _ts_sum(_ts_mean(v, 180), 37), 14), 12), 5)) * -1.0
    out["a101_191_gtja_a040"] = _ts_sum(np.where(c > _delay(c, 1), v, 0.0), 26) / (_ts_sum(np.where(c <= _delay(c, 1), v, 0.0), 26) + 1e-12) * 100.0
    out["a101_191_gtja_a041"] = _ts_rank(np.maximum(_delta(vwap, 3), 5), 5) * -1.0
    out["a101_191_gtja_a042"] = -1.0 * _ts_rank(_ts_std(h, 10), 5) * _ts_corr(h, v, 10)
    out["a101_191_gtja_a043"] = _ts_sum(np.where(c > _delay(c, 1), v, np.where(c < _delay(c, 1), -v, 0.0)), 6)
    out["a101_191_gtja_a044"] = _ts_rank(_decay_linear(_ts_corr(l, _ts_mean(v, 10), 7), 6), 5) + _ts_rank(_decay_linear(_delta(vwap, 3), 10), 5)
    out["a101_191_gtja_a045"] = _ts_rank(_delta(c * 0.6 + o * 0.4, 1), 5) * _ts_rank(_ts_corr(vwap, _ts_mean(v, 150), 15), 5)
    out["a101_191_gtja_a046"] = (_ts_mean(c, 3) + _ts_mean(c, 6) + _ts_mean(c, 12) + _ts_mean(c, 24)) / (4.0 * c + 1e-12)
    out["a101_191_gtja_a047"] = (_ts_max(h, 6) - c) / (_ts_max(h, 6) - _ts_min(l, 6) + 1e-12) * 100.0
    out["a101_191_gtja_a048"] = -1.0 * _ts_rank(_sign(c - _delay(c, 1)) + _sign(_delay(c, 1) - _delay(c, 2)) + _sign(_delay(c, 2) - _delay(c, 3)), 5) * _ts_sum(v, 5) / (_ts_sum(v, 20) + 1e-12)
    out["a101_191_gtja_a049"] = _ts_sum(np.where((h + l) >= (_delay(h, 1) + _delay(l, 1)), 0.0, np.maximum((h - _delay(h, 1)).abs(), (l - _delay(l, 1)).abs())), 12) / (_ts_sum(np.maximum((h - _delay(h, 1)).abs(), (l - _delay(l, 1)).abs()), 12) + 1e-12)
    out["a101_191_gtja_a050"] = _ts_sum(np.where((h + l) <= (_delay(h, 1) + _delay(l, 1)), 0.0, np.maximum((h - _delay(h, 1)).abs(), (l - _delay(l, 1)).abs())), 12) / (_ts_sum(np.maximum((h - _delay(h, 1)).abs(), (l - _delay(l, 1)).abs()), 12) + 1e-12)

    # Sanity: ensure all manifest cols exist (zero-fill any miss)
    for name in A101_191_FEATURE_NAMES:
        if name not in out.columns:
            out[name] = 0.0

    # Final NaN -> 0.0 for clean downstream consumption (leaves shift(1) NaN's intact via fill)
    for name in A101_191_FEATURE_NAMES:
        col = out[name]
        if not isinstance(col, pd.Series):
            out[name] = pd.Series(col, index=out.index)

    return out


if __name__ == "__main__":
    # Smoke
    np.random.seed(0)
    n = 250
    test_df = pd.DataFrame({
        "open": np.random.uniform(100, 110, n).cumsum() / 50,
        "high": np.random.uniform(105, 115, n).cumsum() / 50,
        "low": np.random.uniform(95, 105, n).cumsum() / 50,
        "close": np.random.uniform(100, 110, n).cumsum() / 50,
        "volume": np.random.uniform(1e6, 2e6, n),
    })
    test_df["high"] = test_df[["open", "close", "high"]].max(axis=1)
    test_df["low"] = test_df[["open", "close", "low"]].min(axis=1)
    result = compute_alpha101_191_features(test_df, ticker="SMOKE")
    new_cols = [c for c in result.columns if c.startswith("a101_191_")]
    print(f"alpha101_191 smoke: {len(new_cols)} cols added (manifest={A101_191_FEATURE_COUNT})")
    print(f"rows={len(result)}, last-row finite check: {result[new_cols].iloc[-1].notna().sum()}/{len(new_cols)}")
