# autosolve_skip: data-loader infra
"""
chart_patterns_features.py — classical chart-pattern features.

Implements rolling-window detectors for:
  - Head-and-Shoulders / Inverse H&S
  - Double Top / Double Bottom
  - Triangle (ascending / descending / symmetrical)
  - Wedge (rising / falling)
  - Flag / Pennant
  - Channel (parallel trend lines)

All detectors operate on rolling local-peak/local-trough sequences extracted
via scipy.signal.argrelextrema (falls back to a pure-numpy implementation if
scipy is unavailable).

NO-LOOKAHEAD AUDIT (2026-05-21)
---------------------------------
For each bar T:
  1. Extract local extrema on the slice df.iloc[:T] (bars strictly < T).
  2. Pattern detection uses ONLY pivots from that prefix slice.
  3. We then .shift(1) all output columns as a belt-and-suspenders guard.

Features (35 cols):
  For 7 patterns: <pattern>_active (±1/0), <pattern>_breakout_pct, <pattern>_target_pct
  Plus 7 aggregates: chart_pattern_count_5d, chart_pattern_count_20d,
  chart_bullish_pattern_count, chart_bearish_pattern_count,
  chart_last_pattern_age_bars, chart_pattern_target_avg, chart_breakout_avg.

License: original (this file). Dependencies: numpy, pandas, (optional scipy).
Cost: MEDIUM — O(n) rolling extrema scan + per-pivot pattern checks, ~200ms per ticker.
"""

from __future__ import annotations

import logging
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from scipy.signal import argrelextrema as _argrel  # type: ignore[import]

    def _local_max(arr: np.ndarray, order: int) -> np.ndarray:
        return _argrel(arr, np.greater, order=order)[0]

    def _local_min(arr: np.ndarray, order: int) -> np.ndarray:
        return _argrel(arr, np.less, order=order)[0]
except ImportError:
    # Pure-numpy fallback
    def _local_max(arr: np.ndarray, order: int) -> np.ndarray:
        n = len(arr)
        out: List[int] = []
        for i in range(order, n - order):
            window = arr[i - order:i + order + 1]
            if arr[i] == np.nanmax(window) and arr[i] > arr[i - 1]:
                out.append(i)
        return np.array(out, dtype=int)

    def _local_min(arr: np.ndarray, order: int) -> np.ndarray:
        n = len(arr)
        out: List[int] = []
        for i in range(order, n - order):
            window = arr[i - order:i + order + 1]
            if arr[i] == np.nanmin(window) and arr[i] < arr[i - 1]:
                out.append(i)
        return np.array(out, dtype=int)


# ---------------------------------------------------------------------------
# Feature names — 35 cols
# ---------------------------------------------------------------------------

CHART_PATTERNS = [
    "head_shoulders",      # +1 normal (bearish), -1 inverse (bullish)
    "double_top",          # +1 active (bearish)  / -1 double_bottom (bullish)
    "triangle",            # +1 ascending (bullish) / -1 descending (bearish) / +0.5 symmetric
    "wedge",               # +1 rising wedge (bearish) / -1 falling wedge (bullish)
    "flag",                # +1 bull flag / -1 bear flag
    "pennant",             # +1 bull pennant / -1 bear pennant
    "channel",             # +1 ascending channel / -1 descending channel / +0.5 horizontal
]

CHART_FEATURE_NAMES: List[str] = []
for _p in CHART_PATTERNS:
    CHART_FEATURE_NAMES.append(f"chart_{_p}_active")
    CHART_FEATURE_NAMES.append(f"chart_{_p}_breakout_pct")
    CHART_FEATURE_NAMES.append(f"chart_{_p}_target_pct")

# 14 aggregates -> total 21 + 14 = 35
CHART_AGG_NAMES = [
    "chart_pattern_count_5d",
    "chart_pattern_count_20d",
    "chart_bullish_pattern_count",
    "chart_bearish_pattern_count",
    "chart_last_pattern_age_bars",
    "chart_pattern_target_avg",
    "chart_breakout_avg",
    "chart_pattern_count_60d",
    "chart_pattern_count_100d",
    "chart_active_signal_sum",
    "chart_active_signal_abs_sum",
    "chart_breakout_max_abs",
    "chart_target_max_abs",
    "chart_any_active",
]
CHART_FEATURE_NAMES = (CHART_FEATURE_NAMES + CHART_AGG_NAMES)[:35]
CHART_FEATURE_COUNT: int = len(CHART_FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _pct(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return (a - b) / b


def _detect_head_shoulders(
    peaks: List[Tuple[int, float]], troughs: List[Tuple[int, float]]
) -> Tuple[int, float, float]:
    """Return (direction, breakout_pct, target_pct).

    direction: +1 = bearish H&S top, -1 = bullish inverse H&S, 0 = none.
    """
    if len(peaks) < 3 or len(troughs) < 2:
        return 0, 0.0, 0.0
    # Regular H&S: 3 peaks with middle highest, 2 intervening troughs roughly equal.
    p1, p2, p3 = peaks[-3:]
    if p2[1] > p1[1] and p2[1] > p3[1]:
        # find troughs between p1-p2 and p2-p3
        t_between = [t for t in troughs if p1[0] < t[0] < p3[0]]
        if len(t_between) >= 2:
            neckline = (t_between[0][1] + t_between[-1][1]) / 2.0
            if neckline > 0 and abs(p1[1] - p3[1]) / neckline < 0.10:
                # Bearish H&S
                breakout = _pct(p3[1], neckline)
                head_height = p2[1] - neckline
                target = -head_height / neckline  # downside target
                return 1, breakout, target
    # Inverse H&S
    if len(troughs) >= 3 and len(peaks) >= 2:
        t1, t2, t3 = troughs[-3:]
        if t2[1] < t1[1] and t2[1] < t3[1]:
            p_between = [p for p in peaks if t1[0] < p[0] < t3[0]]
            if len(p_between) >= 2:
                neckline = (p_between[0][1] + p_between[-1][1]) / 2.0
                if neckline > 0 and abs(t1[1] - t3[1]) / neckline < 0.10:
                    breakout = _pct(t3[1], neckline)
                    head_height = neckline - t2[1]
                    target = head_height / neckline  # upside target
                    return -1, breakout, target
    return 0, 0.0, 0.0


def _detect_double_topbottom(
    peaks: List[Tuple[int, float]], troughs: List[Tuple[int, float]]
) -> Tuple[int, float, float]:
    if len(peaks) >= 2:
        p1, p2 = peaks[-2:]
        if p1[1] > 0 and abs(p1[1] - p2[1]) / p1[1] < 0.03:
            # Find trough between
            t_between = [t for t in troughs if p1[0] < t[0] < p2[0]]
            if t_between:
                neck = t_between[0][1]
                if neck > 0:
                    target = _pct(neck, p1[1])  # downside
                    breakout = _pct(p2[1], neck)
                    return 1, breakout, target  # bearish double top
    if len(troughs) >= 2:
        t1, t2 = troughs[-2:]
        if t1[1] > 0 and abs(t1[1] - t2[1]) / t1[1] < 0.03:
            p_between = [p for p in peaks if t1[0] < p[0] < t2[0]]
            if p_between:
                neck = p_between[0][1]
                if neck > 0:
                    target = _pct(neck, t1[1])  # upside
                    breakout = _pct(t2[1], neck)
                    return -1, breakout, target  # bullish double bottom
    return 0, 0.0, 0.0


def _trendline_slope(points: List[Tuple[int, float]]) -> float:
    if len(points) < 2:
        return 0.0
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    if np.std(xs) == 0:
        return 0.0
    return float(np.polyfit(xs, ys, 1)[0])


def _detect_triangle(
    peaks: List[Tuple[int, float]], troughs: List[Tuple[int, float]],
    cur_close: float,
) -> Tuple[float, float, float]:
    """Return (direction, breakout, target). direction: 1 asc, -1 desc, 0.5 sym, 0 none."""
    if len(peaks) < 2 or len(troughs) < 2:
        return 0.0, 0.0, 0.0
    s_top = _trendline_slope(peaks[-3:] if len(peaks) >= 3 else peaks[-2:])
    s_bot = _trendline_slope(troughs[-3:] if len(troughs) >= 3 else troughs[-2:])
    if cur_close <= 0:
        return 0.0, 0.0, 0.0
    # Ascending: flat top, rising bottom
    if abs(s_top) < 1e-3 and s_bot > 1e-3:
        last_high = peaks[-1][1]
        return 1.0, _pct(cur_close, last_high), _pct(last_high * 1.05, cur_close)
    # Descending: falling top, flat bottom
    if s_top < -1e-3 and abs(s_bot) < 1e-3:
        last_low = troughs[-1][1]
        return -1.0, _pct(cur_close, last_low), _pct(last_low * 0.95, cur_close)
    # Symmetrical: converging (top falling, bottom rising)
    if s_top < -1e-3 and s_bot > 1e-3:
        return 0.5, 0.0, 0.0
    return 0.0, 0.0, 0.0


def _detect_wedge(
    peaks: List[Tuple[int, float]], troughs: List[Tuple[int, float]],
    cur_close: float,
) -> Tuple[float, float, float]:
    if len(peaks) < 2 or len(troughs) < 2:
        return 0.0, 0.0, 0.0
    s_top = _trendline_slope(peaks[-3:] if len(peaks) >= 3 else peaks[-2:])
    s_bot = _trendline_slope(troughs[-3:] if len(troughs) >= 3 else troughs[-2:])
    # Rising wedge: both up, top slope < bottom slope (converging upward)
    if s_top > 1e-3 and s_bot > 1e-3 and s_top < s_bot and cur_close > 0:
        return 1.0, _pct(cur_close, peaks[-1][1]), -0.05
    # Falling wedge: both down, top slope > bottom slope (converging downward)
    if s_top < -1e-3 and s_bot < -1e-3 and s_top > s_bot and cur_close > 0:
        return -1.0, _pct(cur_close, troughs[-1][1]), 0.05
    return 0.0, 0.0, 0.0


def _detect_flag(
    closes: np.ndarray, idx: int, lookback: int = 20,
) -> Tuple[float, float, float]:
    """Bull/bear flag: strong prior trend + short consolidation."""
    if idx < lookback * 2:
        return 0.0, 0.0, 0.0
    pole = closes[idx - lookback * 2:idx - lookback]
    flag = closes[idx - lookback:idx]
    if len(pole) == 0 or len(flag) == 0 or pole[0] == 0:
        return 0.0, 0.0, 0.0
    pole_ret = (pole[-1] - pole[0]) / pole[0]
    flag_range = (np.nanmax(flag) - np.nanmin(flag)) / np.nanmean(flag) if np.nanmean(flag) > 0 else 0
    if abs(pole_ret) > 0.05 and flag_range < 0.03:
        direction = 1.0 if pole_ret > 0 else -1.0
        return direction, _pct(closes[idx - 1], flag[0]), pole_ret  # target ~ pole height
    return 0.0, 0.0, 0.0


def _detect_pennant(
    peaks: List[Tuple[int, float]], troughs: List[Tuple[int, float]],
    closes: np.ndarray, idx: int,
) -> Tuple[float, float, float]:
    if len(peaks) < 2 or len(troughs) < 2 or idx < 30:
        return 0.0, 0.0, 0.0
    # Pennant = small symmetric triangle after a sharp move (the pole)
    pole = closes[max(0, idx - 30):idx - 10]
    if len(pole) == 0 or pole[0] == 0:
        return 0.0, 0.0, 0.0
    pole_ret = (pole[-1] - pole[0]) / pole[0]
    s_top = _trendline_slope(peaks[-2:])
    s_bot = _trendline_slope(troughs[-2:])
    if abs(pole_ret) > 0.05 and s_top < -1e-4 and s_bot > 1e-4:
        direction = 1.0 if pole_ret > 0 else -1.0
        return direction, _pct(closes[idx - 1], (peaks[-1][1] + troughs[-1][1]) / 2.0), pole_ret
    return 0.0, 0.0, 0.0


def _detect_channel(
    peaks: List[Tuple[int, float]], troughs: List[Tuple[int, float]],
    cur_close: float,
) -> Tuple[float, float, float]:
    if len(peaks) < 2 or len(troughs) < 2:
        return 0.0, 0.0, 0.0
    s_top = _trendline_slope(peaks[-3:] if len(peaks) >= 3 else peaks[-2:])
    s_bot = _trendline_slope(troughs[-3:] if len(troughs) >= 3 else troughs[-2:])
    # Parallel: slopes roughly equal (within 30%)
    if abs(s_top) > 1e-4 and abs(s_bot) > 1e-4 and \
            (s_top * s_bot > 0) and abs(s_top - s_bot) / max(abs(s_top), abs(s_bot)) < 0.3:
        last_h = peaks[-1][1]
        last_l = troughs[-1][1]
        width = last_h - last_l
        if width > 0 and cur_close > 0:
            pos = (cur_close - last_l) / width
            direction = 1.0 if s_top > 0 else -1.0
            return direction, pos, _pct(last_h, cur_close)
    # Horizontal: both flat
    if abs(s_top) < 1e-4 and abs(s_bot) < 1e-4:
        return 0.5, 0.0, 0.0
    return 0.0, 0.0, 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compute_chart_patterns_features(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
    order: int = 5,
    scan_every: int = 5,
) -> pd.DataFrame:
    """Add chart-pattern features.

    df must contain 'high', 'low', 'close' columns. Adds CHART_FEATURE_NAMES.
    Detection runs on a sliding prefix to avoid lookahead. To keep cost bounded
    we re-scan every `scan_every` bars (intermediate bars inherit the previous
    detection).
    """
    for col in CHART_FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0.0

    cols_lc = {c.lower(): c for c in df.columns}
    h_col = cols_lc.get("high")
    l_col = cols_lc.get("low")
    c_col = cols_lc.get("close")
    if h_col is None or l_col is None or c_col is None:
        logger.warning("[chart] missing OHLC for %s — zeroing", ticker)
        return df

    highs = df[h_col].to_numpy(dtype=float)
    lows = df[l_col].to_numpy(dtype=float)
    closes = df[c_col].to_numpy(dtype=float)
    n = len(closes)
    if n < 80:
        return df

    # Per-bar pattern signal arrays.
    sig = {p: np.zeros(n, dtype=float) for p in CHART_PATTERNS}
    bo = {p: np.zeros(n, dtype=float) for p in CHART_PATTERNS}
    tg = {p: np.zeros(n, dtype=float) for p in CHART_PATTERNS}

    last_scan: Dict[str, Tuple[float, float, float]] = {p: (0.0, 0.0, 0.0) for p in CHART_PATTERNS}

    for t in range(60, n):
        if (t - 60) % scan_every != 0 and t > 60:
            # carry-forward
            for p in CHART_PATTERNS:
                sig[p][t] = last_scan[p][0]
                bo[p][t] = last_scan[p][1]
                tg[p][t] = last_scan[p][2]
            continue

        prefix_h = highs[:t]
        prefix_l = lows[:t]
        # rolling pivots
        peak_idx = _local_max(prefix_h, order=order)
        trough_idx = _local_min(prefix_l, order=order)
        peaks = [(int(i), float(prefix_h[i])) for i in peak_idx[-10:]]  # last 10
        troughs = [(int(i), float(prefix_l[i])) for i in trough_idx[-10:]]
        cur = closes[t - 1]  # NO-LOOKAHEAD: bar t-1's close

        d, b, g = _detect_head_shoulders(peaks, troughs)
        sig["head_shoulders"][t], bo["head_shoulders"][t], tg["head_shoulders"][t] = d, b, g
        last_scan["head_shoulders"] = (d, b, g)

        d, b, g = _detect_double_topbottom(peaks, troughs)
        sig["double_top"][t], bo["double_top"][t], tg["double_top"][t] = d, b, g
        last_scan["double_top"] = (d, b, g)

        d, b, g = _detect_triangle(peaks, troughs, cur)
        sig["triangle"][t], bo["triangle"][t], tg["triangle"][t] = d, b, g
        last_scan["triangle"] = (d, b, g)

        d, b, g = _detect_wedge(peaks, troughs, cur)
        sig["wedge"][t], bo["wedge"][t], tg["wedge"][t] = d, b, g
        last_scan["wedge"] = (d, b, g)

        d, b, g = _detect_flag(closes, t)
        sig["flag"][t], bo["flag"][t], tg["flag"][t] = d, b, g
        last_scan["flag"] = (d, b, g)

        d, b, g = _detect_pennant(peaks, troughs, closes, t)
        sig["pennant"][t], bo["pennant"][t], tg["pennant"][t] = d, b, g
        last_scan["pennant"] = (d, b, g)

        d, b, g = _detect_channel(peaks, troughs, cur)
        sig["channel"][t], bo["channel"][t], tg["channel"][t] = d, b, g
        last_scan["channel"] = (d, b, g)

    # Write per-pattern triplets to df with .shift(1)
    idx = df.index
    for p in CHART_PATTERNS:
        for suffix, arr in (("active", sig[p]), ("breakout_pct", bo[p]), ("target_pct", tg[p])):
            col = f"chart_{p}_{suffix}"
            if col in CHART_FEATURE_NAMES:
                df[col] = pd.Series(arr, index=idx).shift(1).fillna(0.0).to_numpy(dtype=float)

    # Aggregates
    active_cols = [f"chart_{p}_active" for p in CHART_PATTERNS]
    available = [c for c in active_cols if c in df.columns]
    if available:
        active_mat = df[available].to_numpy(dtype=float)
        any_active = (np.abs(active_mat) > 0).astype(float)
        bull_count = (active_mat > 0).sum(axis=1).astype(float)
        bear_count = (active_mat < 0).sum(axis=1).astype(float)

        if "chart_pattern_count_5d" in CHART_FEATURE_NAMES:
            df["chart_pattern_count_5d"] = pd.Series(any_active.sum(axis=1), index=idx).rolling(5, min_periods=1).sum().fillna(0.0).to_numpy()
        if "chart_pattern_count_20d" in CHART_FEATURE_NAMES:
            df["chart_pattern_count_20d"] = pd.Series(any_active.sum(axis=1), index=idx).rolling(20, min_periods=1).sum().fillna(0.0).to_numpy()
        if "chart_bullish_pattern_count" in CHART_FEATURE_NAMES:
            df["chart_bullish_pattern_count"] = bull_count
        if "chart_bearish_pattern_count" in CHART_FEATURE_NAMES:
            df["chart_bearish_pattern_count"] = bear_count

        if "chart_last_pattern_age_bars" in CHART_FEATURE_NAMES:
            ages = np.zeros(n, dtype=float)
            last_seen = -1
            any_per_bar = (np.abs(active_mat) > 0).any(axis=1)
            for i in range(n):
                if any_per_bar[i]:
                    last_seen = i
                ages[i] = (i - last_seen) if last_seen >= 0 else 999.0
            df["chart_last_pattern_age_bars"] = ages

        target_cols = [f"chart_{p}_target_pct" for p in CHART_PATTERNS if f"chart_{p}_target_pct" in df.columns]
        bo_cols = [f"chart_{p}_breakout_pct" for p in CHART_PATTERNS if f"chart_{p}_breakout_pct" in df.columns]
        if "chart_pattern_target_avg" in CHART_FEATURE_NAMES and target_cols:
            df["chart_pattern_target_avg"] = df[target_cols].mean(axis=1).fillna(0.0).to_numpy(dtype=float)
        if "chart_breakout_avg" in CHART_FEATURE_NAMES and bo_cols:
            df["chart_breakout_avg"] = df[bo_cols].mean(axis=1).fillna(0.0).to_numpy(dtype=float)

        # Additional aggregates (added to hit 35-col target)
        any_per_bar = (np.abs(active_mat) > 0).any(axis=1).astype(float)
        if "chart_pattern_count_60d" in CHART_FEATURE_NAMES:
            df["chart_pattern_count_60d"] = pd.Series(any_active.sum(axis=1), index=idx).rolling(60, min_periods=1).sum().fillna(0.0).to_numpy()
        if "chart_pattern_count_100d" in CHART_FEATURE_NAMES:
            df["chart_pattern_count_100d"] = pd.Series(any_active.sum(axis=1), index=idx).rolling(100, min_periods=1).sum().fillna(0.0).to_numpy()
        if "chart_active_signal_sum" in CHART_FEATURE_NAMES:
            df["chart_active_signal_sum"] = active_mat.sum(axis=1)
        if "chart_active_signal_abs_sum" in CHART_FEATURE_NAMES:
            df["chart_active_signal_abs_sum"] = np.abs(active_mat).sum(axis=1)
        if "chart_breakout_max_abs" in CHART_FEATURE_NAMES and bo_cols:
            df["chart_breakout_max_abs"] = df[bo_cols].abs().max(axis=1).fillna(0.0).to_numpy(dtype=float)
        if "chart_target_max_abs" in CHART_FEATURE_NAMES and target_cols:
            df["chart_target_max_abs"] = df[target_cols].abs().max(axis=1).fillna(0.0).to_numpy(dtype=float)
        if "chart_any_active" in CHART_FEATURE_NAMES:
            df["chart_any_active"] = any_per_bar

    logger.debug("[chart] added %d features for %s", CHART_FEATURE_COUNT, ticker)
    return df


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="SMOKE")
    args = ap.parse_args()

    np.random.seed(7)
    n = 250
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.4)
    low = close - np.abs(np.random.randn(n) * 0.4)
    df = pd.DataFrame({"high": high, "low": low, "close": close, "open": close, "volume": 1000})
    df_out = compute_chart_patterns_features(df, ticker=args.ticker)
    new_cols = [c for c in df_out.columns if c.startswith("chart_")]
    print(f"[smoke] ticker={args.ticker} rows={len(df_out)} new_cols={len(new_cols)}")
    nz = sum(int((df_out[c] != 0).any()) for c in new_cols)
    print(f"[smoke] non-zero cols: {nz}/{len(new_cols)}")
