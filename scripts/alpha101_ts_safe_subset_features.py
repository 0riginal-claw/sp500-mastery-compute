"""
alpha101_ts_safe_subset_features.py — 15 time-series-safe Alpha101 factors.

NO-LOOKAHEAD AUDIT
==================
All base OHLCV inputs are pre-shifted by 1 bar at the top of
compute_alpha101_ts_safe_subset_features() before any calculation:
  c = df['close'].shift(1)
  o = df['open'].shift(1)
  h = df['high'].shift(1)
  l = df['low'].shift(1)
  v = df['volume'].shift(1)
  r = c.pct_change()  # already on shifted close, no same-bar return

Every derived quantity (rolling stats, correlations, rank transforms)
operates on these shifted series, so bar-t features reference at most
bar-(t-1) data.  No cross-sectional rank needed — ts_rank() replaces it.

Source inspiration: github:STHSF/alpha101 (MIT license, no paid API).
Implementation: pure pandas + numpy; no external packages required.
shift(1)-safe: YES (pre-shift pattern above, confirmed 2026-05-18).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ALPHA101_TS_SAFE_FEATURE_NAMES: list[str] = [
    "a101ts_alpha001",   # return z-score reversal
    "a101ts_alpha003",   # open-volume negative correlation
    "a101ts_alpha006",   # open vs smoothed-volume correlation
    "a101ts_alpha008",   # open×return rolling sum change rank
    "a101ts_alpha012",   # volume-direction × price reversal
    "a101ts_alpha016",   # high-volume ts_rank cross-correlation
    "a101ts_alpha019",   # 7d trend × return rank persistence
    "a101ts_alpha020",   # open-gap triple-rank product
    "a101ts_alpha023",   # high delta conditional reversal
    "a101ts_alpha033",   # open/close ratio reversal rank
    "a101ts_alpha034",   # relative stddev reversal rank
    "a101ts_alpha035",   # volume × HL+close rank product
    "a101ts_alpha040",   # high stddev × high-vol correlation
    "a101ts_alpha041",   # geometric mean deviation (HL vs close)
    "a101ts_alpha051",   # open vs HL range z-score
]
ALPHA101_TS_SAFE_FEATURE_COUNT: int = len(ALPHA101_TS_SAFE_FEATURE_NAMES)


def _ts_rank(s: pd.Series, window: int) -> pd.Series:
    """Percentile rank of the last value within a rolling window (0→1)."""
    def _rank_last(x: np.ndarray) -> float:
        if len(x) == 0:
            return 0.5
        sorted_x = np.sort(x)
        pos = np.searchsorted(sorted_x, x[-1], side="left")
        denom = len(x) - 1
        return float(pos / denom) if denom > 0 else 0.5

    min_p = max(1, window // 2)
    return s.rolling(window, min_periods=min_p).apply(_rank_last, raw=True)


def compute_alpha101_ts_safe_subset_features(
    df: pd.DataFrame,
    ticker: str | None = None,
) -> pd.DataFrame:
    """Add 15 STHSF/Alpha101-inspired ts-safe features to df.

    All inputs pre-shifted by 1 bar; returns new columns in-place on a copy.
    """
    required = {"close", "open", "high", "low", "volume"}
    missing = required - set(df.columns)
    if missing:
        for col in ALPHA101_TS_SAFE_FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0.0
        return df

    df = df.copy()

    # Pre-shift all base OHLCV (no same-bar look-ahead)
    c = df["close"].shift(1)
    o = df["open"].shift(1)
    h = df["high"].shift(1)
    l = df["low"].shift(1)
    v = df["volume"].shift(1)
    r = c.pct_change()  # return on already-shifted close

    try:
        # 1. alpha001 — return z-score (20d) mean-reversion signal
        df["a101ts_alpha001"] = (
            (r - r.rolling(20, min_periods=5).mean())
            / (r.rolling(20, min_periods=5).std() + 1e-8)
        )
    except Exception:
        df["a101ts_alpha001"] = 0.0

    try:
        # 2. alpha003 — -corr(open, volume, 10)
        df["a101ts_alpha003"] = -1.0 * o.rolling(10, min_periods=5).corr(v)
    except Exception:
        df["a101ts_alpha003"] = 0.0

    try:
        # 3. alpha006 — -corr(open, smoothed_volume, 10)
        v_smooth = v.rolling(10, min_periods=5).mean()
        df["a101ts_alpha006"] = -1.0 * o.rolling(10, min_periods=5).corr(v_smooth)
    except Exception:
        df["a101ts_alpha006"] = 0.0

    try:
        # 4. alpha008 — -ts_rank(sum5(open×return) - delay10(sum5(open×return)), 5)
        or_prod = o * r
        sum_or = or_prod.rolling(5, min_periods=3).sum()
        df["a101ts_alpha008"] = -1.0 * _ts_rank(sum_or - sum_or.shift(10), 5)
    except Exception:
        df["a101ts_alpha008"] = 0.0

    try:
        # 5. alpha012 — sign(delta_vol) × (-delta_close)
        df["a101ts_alpha012"] = np.sign(v.diff(1)) * (-1.0 * c.diff(1))
    except Exception:
        df["a101ts_alpha012"] = 0.0

    try:
        # 6. alpha016 — -ts_rank(corr(ts_rank(high,5), ts_rank(vol,5), 5), 5)
        rh5 = _ts_rank(h, 5)
        rv5 = _ts_rank(v, 5)
        corr_hv = rh5.rolling(5, min_periods=3).corr(rv5)
        df["a101ts_alpha016"] = -1.0 * _ts_rank(corr_hv, 5)
    except Exception:
        df["a101ts_alpha016"] = 0.0

    try:
        # 7. alpha019 — 7d momentum × ts_rank(returns, 252) factor
        mom7 = (c - c.shift(7)) / (c.shift(7).abs() + 1e-8)
        df["a101ts_alpha019"] = mom7 * _ts_rank(r, min(252, len(r)))
    except Exception:
        df["a101ts_alpha019"] = 0.0

    try:
        # 8. alpha020 — -ts_rank(o-h)*ts_rank(o-c1)*ts_rank(o-l)
        p1 = _ts_rank(o - h, 5)
        p2 = _ts_rank(o - c.shift(1), 5)
        p3 = _ts_rank(o - l, 5)
        df["a101ts_alpha020"] = -1.0 * p1 * p2 * p3
    except Exception:
        df["a101ts_alpha020"] = 0.0

    try:
        # 9. alpha023 — 0 if delta(high,2)>0, else -delta(high,2)
        dh2 = h.diff(2)
        df["a101ts_alpha023"] = np.where(dh2 > 0, 0.0, -dh2)
    except Exception:
        df["a101ts_alpha023"] = 0.0

    try:
        # 10. alpha033 — ts_rank(-(1 - open/close), 20)
        ratio = -(1.0 - o / (c + 1e-8))
        df["a101ts_alpha033"] = _ts_rank(ratio, 20)
    except Exception:
        df["a101ts_alpha033"] = 0.0

    try:
        # 11. alpha034 — 1 - ts_rank(std2 / std5, 20)
        std2 = r.rolling(2, min_periods=1).std()
        std5 = r.rolling(5, min_periods=2).std()
        rel_std = std2 / (std5 + 1e-8)
        df["a101ts_alpha034"] = 1.0 - _ts_rank(rel_std, 20)
    except Exception:
        df["a101ts_alpha034"] = 0.0

    try:
        # 12. alpha035 — ts_rank(vol,32)*(1-ts_rank(h+c-l,16))*(1-ts_rank(r,32))
        fv = _ts_rank(v, 32)
        fhlc = 1.0 - _ts_rank(h + c - l, 16)
        fr = 1.0 - _ts_rank(r, 32)
        df["a101ts_alpha035"] = fv * fhlc * fr
    except Exception:
        df["a101ts_alpha035"] = 0.0

    try:
        # 13. alpha040 — -ts_rank(std(high,10),10) * corr(high,vol,10)
        h_std = h.rolling(10, min_periods=5).std()
        h_vol_corr = h.rolling(10, min_periods=5).corr(v)
        df["a101ts_alpha040"] = -1.0 * _ts_rank(h_std, 10) * h_vol_corr
    except Exception:
        df["a101ts_alpha040"] = 0.0

    try:
        # 14. alpha041 — sqrt(high*low) - close (geometric mean vs close)
        gm = (h * l).clip(lower=0.0).pow(0.5)
        df["a101ts_alpha041"] = gm - c
    except Exception:
        df["a101ts_alpha041"] = 0.0

    try:
        # 15. alpha051 — z-score of open's position in HL range (20d)
        hl_range = (h - l).replace(0, np.nan)
        occ = (o - l) / hl_range  # 0=low end, 1=high end
        df["a101ts_alpha051"] = (
            (occ - occ.rolling(20, min_periods=5).mean())
            / (occ.rolling(20, min_periods=5).std() + 1e-8)
        )
    except Exception:
        df["a101ts_alpha051"] = 0.0

    return df
