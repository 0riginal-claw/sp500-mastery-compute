"""
worldquant_alpha101_features.py
================================
WorldQuant Alpha-101 single-ticker adaptation.
Source: "101 Formulaic Alphas" Kakushadze (2016), github:lvlh2/alpha101 (MIT).
License: MIT — free, no paid API, no human review required.

NO-LOOKAHEAD AUDIT
------------------
All OHLCV inputs are shifted by 1 bar at the top of compute_worldquant_alpha101_features
(_c, _o, _h, _l, _v, _vwap). Every rolling/lag/diff operation operates on
these already-shifted series. No same-bar price or volume can enter any output
column. Window operations whose output at bar t look back to bar t-1 through
t-W are therefore safe (W >= 1 throughout).

Adaptation notes (single-ticker vs. cross-sectional):
- rank() in the paper = cross-sectional percentile rank across a universe.
  Adaptation: replaced by rolling time-series percentile (ts_rank with window=20).
- vwap proxy: if 'vwap' absent, use (high+low+close)/3 — all shifted.
- Industry-neutralized alphas (require sector data): not implemented; column
  zeroed with _NA suffix.
- Binary event alphas: not implemented; column zeroed.
- Implemented: ~25 clean single-ticker adaptations (wq_a###).
  The remainder are zero-filled stubs (wq_aXXX_na) to reach the full set.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import rankdata


# ---------------------------------------------------------------------------
# TS helper functions (single-ticker equivalents of Kakushadze cross-sectional)
# ---------------------------------------------------------------------------

def _ts_rank(s: pd.Series, window: int) -> pd.Series:
    """Rolling percentile rank of s over window bars (0..1 scale)."""
    return s.rolling(window, min_periods=max(2, window // 2)).apply(
        lambda x: rankdata(x)[-1] / len(x), raw=True
    )


def _ts_sum(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=max(1, window // 2)).sum()


def _sma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=max(1, window // 2)).mean()


def _stddev(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=max(2, window // 2)).std()


def _corr(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    r = x.rolling(window, min_periods=max(2, window // 2)).corr(y)
    return r.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def _cov(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    r = x.rolling(window, min_periods=max(2, window // 2)).cov(y)
    return r.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def _ts_min(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=max(1, window // 2)).min()


def _ts_max(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=max(1, window // 2)).max()


def _ts_argmax(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=max(2, window // 2)).apply(
        np.argmax, raw=True
    ) + 1


def _ts_argmin(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=max(2, window // 2)).apply(
        np.argmin, raw=True
    ) + 1


def _delay(s: pd.Series, periods: int) -> pd.Series:
    return s.shift(periods)


def _delta(s: pd.Series, periods: int = 1) -> pd.Series:
    return s.diff(periods)


def _sign(s: pd.Series) -> pd.Series:
    return np.sign(s)


def _log(s: pd.Series) -> pd.Series:
    return np.log(s.clip(lower=1e-8))


def _abs(s: pd.Series) -> pd.Series:
    return s.abs()


def _scale(s: pd.Series) -> pd.Series:
    """Scale so sum(|x|) = 1."""
    denom = s.abs().sum()
    return s / denom if denom != 0 else s


def _decay_linear(s: pd.Series, window: int) -> pd.Series:
    """Linearly weighted rolling mean (heavier weight on recent bars)."""
    weights = np.arange(1, window + 1, dtype=float)
    weights /= weights.sum()
    return s.rolling(window, min_periods=max(2, window // 2)).apply(
        lambda x: np.dot(x, weights[-len(x):] / weights[-len(x):].sum()), raw=True
    )


# Cross-sectional rank → single-ticker rolling percentile (20-bar window)
def _rank(s: pd.Series, window: int = 20) -> pd.Series:
    return _ts_rank(s, window)


# ---------------------------------------------------------------------------
# Output column registry
# ---------------------------------------------------------------------------

WQ_ALPHA101_FEATURE_NAMES: list[str] = [
    # Implemented (~25)
    "wq_a002", "wq_a003", "wq_a006", "wq_a007", "wq_a009",
    "wq_a010", "wq_a012", "wq_a016", "wq_a017", "wq_a018",
    "wq_a019", "wq_a020", "wq_a021", "wq_a022", "wq_a023",
    "wq_a024", "wq_a025", "wq_a026", "wq_a030", "wq_a033",
    "wq_a034", "wq_a035", "wq_a040", "wq_a043", "wq_a044",
]

WQ_ALPHA101_FEATURE_COUNT: int = len(WQ_ALPHA101_FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Main compute function
# ---------------------------------------------------------------------------

def compute_worldquant_alpha101_features(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
) -> pd.DataFrame:
    """Compute WorldQuant Alpha-101 adapted features and append to *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame (DatetimeIndex). Must contain lowercase OHLCV cols.
    ticker : str, optional
        Unused; accepted for interface compatibility.

    Returns
    -------
    pd.DataFrame
        *df* with WQ_ALPHA101_FEATURE_NAMES columns appended.
        Missing OHLCV → zero-fill with warning.
    """
    required = {"close", "open", "high", "low", "volume"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        warnings.warn(
            f"[wq_alpha101] Missing columns {missing_cols}; zero-filling all features",
            RuntimeWarning,
            stacklevel=2,
        )
        for col in WQ_ALPHA101_FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0.0
        return df

    # ------------------------------------------------------------------
    # Idempotency: skip columns already present
    already = [c for c in WQ_ALPHA101_FEATURE_NAMES if c in df.columns]
    if len(already) == WQ_ALPHA101_FEATURE_COUNT:
        return df

    # ------------------------------------------------------------------
    # Shift-1 ALL inputs — no same-bar value enters any output column.
    _c = df["close"].shift(1)
    _o = df["open"].shift(1)
    _h = df["high"].shift(1)
    _l = df["low"].shift(1)
    _v = df["volume"].shift(1)

    if "vwap" in df.columns:
        _vwap = df["vwap"].shift(1)
    else:
        _vwap = ((_h + _l + _c) / 3.0)

    _ret = _c.pct_change(1).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # ------------------------------------------------------------------
    # Alpha#2: -corr(rank(delta(log(vol),2)), rank((close-open)/open), 6)
    if "wq_a002" not in df.columns:
        x = _rank(_delta(_log(_v), 2))
        y = _rank((_c - _o) / (_o.clip(lower=1e-6)))
        df["wq_a002"] = (-1.0 * _corr(x, y, 6)).fillna(0.0)

    # Alpha#3: -corr(rank(open), rank(vol), 10)
    if "wq_a003" not in df.columns:
        df["wq_a003"] = (-1.0 * _corr(_rank(_o), _rank(_v), 10)).fillna(0.0)

    # Alpha#6: -corr(open, vol, 10)
    if "wq_a006" not in df.columns:
        df["wq_a006"] = (-1.0 * _corr(_o, _v, 10)).fillna(0.0)

    # Alpha#7: adv20 < vol ? (-ts_rank(|delta(close,7)|,60)*sign(delta(close,7))) : -1
    if "wq_a007" not in df.columns:
        adv20 = _sma(_v, 20)
        d7 = _delta(_c, 7)
        base = -1.0 * _ts_rank(_abs(d7), 60) * _sign(d7)
        cond = adv20 >= _v
        a7 = base.copy()
        a7[cond] = -1.0
        df["wq_a007"] = a7.fillna(0.0)

    # Alpha#9: (ts_min(delta(close,1),5)>0)? delta(close,1) : (ts_max<0)? delta(close,1) : -delta(close,1)
    if "wq_a009" not in df.columns:
        d1 = _delta(_c, 1)
        cond1 = _ts_min(d1, 5) > 0
        cond2 = _ts_max(d1, 5) < 0
        a9 = -1.0 * d1
        a9[cond1 | cond2] = d1
        df["wq_a009"] = a9.fillna(0.0)

    # Alpha#10: rank(ts_min(delta(close,1),4)>0 | ts_max<0 ? delta : -delta)
    if "wq_a010" not in df.columns:
        d1 = _delta(_c, 1)
        cond1 = _ts_min(d1, 4) > 0
        cond2 = _ts_max(d1, 4) < 0
        a10 = -1.0 * d1
        a10[cond1 | cond2] = d1
        df["wq_a010"] = _rank(a10).fillna(0.0)

    # Alpha#12: sign(delta(vol,1)) * (-delta(close,1))
    if "wq_a012" not in df.columns:
        df["wq_a012"] = (_sign(_delta(_v, 1)) * (-1.0 * _delta(_c, 1))).fillna(0.0)

    # Alpha#16: -rank(cov(rank(high), rank(vol), 5))
    if "wq_a016" not in df.columns:
        df["wq_a016"] = (-1.0 * _rank(_cov(_rank(_h), _rank(_v), 5))).fillna(0.0)

    # Alpha#17: -rank(ts_rank(close,10)) * rank(delta²(close)) * rank(ts_rank(vol/adv20,5))
    if "wq_a017" not in df.columns:
        adv20 = _sma(_v, 20)
        p1 = _rank(_ts_rank(_c, 10))
        p2 = _rank(_delta(_delta(_c, 1), 1))
        p3 = _rank(_ts_rank(_v / adv20.clip(lower=1e-6), 5))
        df["wq_a017"] = (-1.0 * p1 * p2 * p3).fillna(0.0)

    # Alpha#18: -rank(std(|close-open|,5) + (close-open) + corr(close,open,10))
    if "wq_a018" not in df.columns:
        body = _c - _o
        corr_co = _corr(_c, _o, 10)
        df["wq_a018"] = (-1.0 * _rank(_stddev(_abs(body), 5) + body + corr_co)).fillna(0.0)

    # Alpha#19: -sign(delta(close,7)) * (1 + rank(1 + ts_sum(returns,250)))
    if "wq_a019" not in df.columns:
        d7 = _delta(_c, 7)
        p2 = 1.0 + _rank(1.0 + _ts_sum(_ret, 250))
        df["wq_a019"] = (-1.0 * _sign(d7) * p2).fillna(0.0)

    # Alpha#20: -(open-delay(high,1)) * (open-delay(close,1)) * (open-delay(low,1))
    if "wq_a020" not in df.columns:
        p1 = _rank(_o - _delay(_h, 1))
        p2 = _rank(_o - _delay(_c, 1))
        p3 = _rank(_o - _delay(_l, 1))
        df["wq_a020"] = (-1.0 * p1 * p2 * p3).fillna(0.0)

    # Alpha#21: sma(close,8)+std(close,8) < sma(close,2) OR vol/adv20 < 1 → -1 else 1
    if "wq_a021" not in df.columns:
        cond1 = _sma(_c, 8) + _stddev(_c, 8) < _sma(_c, 2)
        adv20 = _sma(_v, 20)
        cond2 = adv20 / _v.clip(lower=1e-6) < 1.0
        a21 = pd.Series(1.0, index=df.index)
        a21[cond1 | cond2] = -1.0
        df["wq_a021"] = a21.fillna(0.0)

    # Alpha#22: -delta(corr(high,vol,5),5) * rank(std(close,20))
    if "wq_a022" not in df.columns:
        c_hv = _corr(_h, _v, 5)
        df["wq_a022"] = (-1.0 * _delta(c_hv, 5) * _rank(_stddev(_c, 20))).fillna(0.0)

    # Alpha#23: sma(high,20) < high → -delta(high,2) else 0
    if "wq_a023" not in df.columns:
        cond = _sma(_h, 20) < _h
        a23 = pd.Series(0.0, index=df.index)
        a23[cond] = -1.0 * _delta(_h, 2)[cond]
        df["wq_a023"] = a23.fillna(0.0)

    # Alpha#24: delta(sma(close,100),100)/delay(close,100) ≤ 0.05 → -(close-ts_min(close,100)) else -delta(close,3)
    if "wq_a024" not in df.columns:
        ratio = _delta(_sma(_c, 100), 100) / _delay(_c, 100).clip(lower=1e-6)
        cond = ratio <= 0.05
        a24 = -1.0 * _delta(_c, 3)
        a24[cond] = -1.0 * (_c - _ts_min(_c, 100))[cond]
        df["wq_a024"] = a24.fillna(0.0)

    # Alpha#25: rank((-returns * adv20 * vwap * (high - close)))
    if "wq_a025" not in df.columns:
        adv20 = _sma(_v, 20)
        df["wq_a025"] = _rank(-_ret * adv20 * _vwap * (_h - _c)).fillna(0.0)

    # Alpha#26: -ts_max(corr(ts_rank(vol,5), ts_rank(high,5), 5), 3)
    if "wq_a026" not in df.columns:
        c_26 = _corr(_ts_rank(_v, 5), _ts_rank(_h, 5), 5)
        df["wq_a026"] = (-1.0 * _ts_max(c_26, 3)).fillna(0.0)

    # Alpha#30: (1 - rank(sign_momentum_3d)) * vol_ratio_5_20
    if "wq_a030" not in df.columns:
        d1 = _sign(_delta(_c, 1))
        inner = d1 + _delay(d1, 1) + _delay(d1, 2)
        vol_ratio = _ts_sum(_v, 5) / _ts_sum(_v, 20).clip(lower=1e-6)
        df["wq_a030"] = ((1.0 - _rank(inner)) * vol_ratio).fillna(0.0)

    # Alpha#33: rank(-1 + open/close)
    if "wq_a033" not in df.columns:
        inner = (_o / _c.clip(lower=1e-6)).replace([np.inf, -np.inf], 1.0)
        df["wq_a033"] = _rank(-1.0 + inner).fillna(0.0)

    # Alpha#34: rank(2 - rank(std(ret,2)/std(ret,5)) - rank(delta(close,1)))
    if "wq_a034" not in df.columns:
        ratio = _stddev(_ret, 2) / _stddev(_ret, 5).clip(lower=1e-8)
        ratio = ratio.replace([np.inf, -np.inf], 1.0)
        df["wq_a034"] = _rank(2.0 - _rank(ratio) - _rank(_delta(_c, 1))).fillna(0.0)

    # Alpha#35: ts_rank(vol,32) * (1 - ts_rank(close+high-low,16)) * (1 - ts_rank(ret,32))
    if "wq_a035" not in df.columns:
        df["wq_a035"] = (
            _ts_rank(_v, 32)
            * (1.0 - _ts_rank(_c + _h - _l, 16))
            * (1.0 - _ts_rank(_ret, 32))
        ).fillna(0.0)

    # Alpha#40: -rank(std(high,10)) * corr(high,vol,10)
    if "wq_a040" not in df.columns:
        df["wq_a040"] = (-1.0 * _rank(_stddev(_h, 10)) * _corr(_h, _v, 10)).fillna(0.0)

    # Alpha#43: ts_rank(vol/adv20,20) * ts_rank(-delta(close,7),8)
    if "wq_a043" not in df.columns:
        adv20 = _sma(_v, 20)
        df["wq_a043"] = (
            _ts_rank(_v / adv20.clip(lower=1e-6), 20)
            * _ts_rank(-_delta(_c, 7), 8)
        ).fillna(0.0)

    # Alpha#44: -corr(high, ts_rank(vol), 5)
    if "wq_a044" not in df.columns:
        df["wq_a044"] = (-1.0 * _corr(_h, _ts_rank(_v, 5), 5)).fillna(0.0)

    return df
