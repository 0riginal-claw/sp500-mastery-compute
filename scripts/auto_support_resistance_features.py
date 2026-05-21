# autosolve_skip: data-loader infra
"""
auto_support_resistance_features.py — automatic S/R zone detection.

Detects pivot points via rolling local-max/min, then clusters nearby pivots
into S/R zones via simple price-proximity binning (DBSCAN-like with epsilon =
0.5% of price). Produces 10 features summarizing current bar's relationship
to the nearest support/resistance bands.

NO-LOOKAHEAD AUDIT (2026-05-21)
---------------------------------
For each bar T, pivots are detected on a prefix slice df.iloc[:T] (bars
strictly < T). All output columns are .shift(1) before joining as a
belt-and-suspenders guard.

Features (10 cols):
  - auto_sr_above_dist           : % distance to nearest resistance above (cur close)
  - auto_sr_below_dist           : % distance to nearest support below
  - auto_sr_strength_above       : touch count of nearest resistance band
  - auto_sr_strength_below       : touch count of nearest support band
  - auto_sr_n_levels             : total distinct S/R zones identified
  - auto_sr_above_age_bars       : bars since most recent resistance touch
  - auto_sr_below_age_bars       : bars since most recent support touch
  - auto_sr_range_pct            : (above - below) / cur_close
  - auto_sr_position_in_range    : (cur - below) / (above - below), 0..1
  - auto_sr_breakout_score       : signed by closeness; +1 broke up, -1 broke down

Dependencies: numpy, pandas. No external libs.
Cost: LOW — ~50ms per ticker.
"""

from __future__ import annotations

import logging
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

AUTO_SR_FEATURE_NAMES: List[str] = [
    "auto_sr_above_dist",
    "auto_sr_below_dist",
    "auto_sr_strength_above",
    "auto_sr_strength_below",
    "auto_sr_n_levels",
    "auto_sr_above_age_bars",
    "auto_sr_below_age_bars",
    "auto_sr_range_pct",
    "auto_sr_position_in_range",
    "auto_sr_breakout_score",
]
AUTO_SR_FEATURE_COUNT: int = len(AUTO_SR_FEATURE_NAMES)


def _rolling_pivot_highs(highs: np.ndarray, window: int) -> List[Tuple[int, float]]:
    n = len(highs)
    out: List[Tuple[int, float]] = []
    if n < 2 * window + 1:
        return out
    for i in range(window, n - window):
        w = highs[i - window:i + window + 1]
        if highs[i] == np.nanmax(w) and highs[i] > highs[i - 1]:
            out.append((i, float(highs[i])))
    return out


def _rolling_pivot_lows(lows: np.ndarray, window: int) -> List[Tuple[int, float]]:
    n = len(lows)
    out: List[Tuple[int, float]] = []
    if n < 2 * window + 1:
        return out
    for i in range(window, n - window):
        w = lows[i - window:i + window + 1]
        if lows[i] == np.nanmin(w) and lows[i] < lows[i - 1]:
            out.append((i, float(lows[i])))
    return out


def _cluster_levels(
    pivots: List[Tuple[int, float]],
    cur_price: float,
    epsilon: float = 0.005,
) -> List[Tuple[float, int, int]]:
    """Cluster pivot prices into bands. Returns (band_price, touch_count, last_touch_idx)."""
    if not pivots:
        return []
    # sort by price
    sorted_p = sorted(pivots, key=lambda x: x[1])
    bands: List[List[Tuple[int, float]]] = [[sorted_p[0]]]
    for px in sorted_p[1:]:
        last_band_avg = np.mean([p[1] for p in bands[-1]])
        if last_band_avg > 0 and abs(px[1] - last_band_avg) / last_band_avg < epsilon:
            bands[-1].append(px)
        else:
            bands.append([px])
    # Merge into (avg_price, count, last_idx)
    out: List[Tuple[float, int, int]] = []
    for b in bands:
        avg = float(np.mean([p[1] for p in b]))
        cnt = len(b)
        last_idx = max(p[0] for p in b)
        out.append((avg, cnt, last_idx))
    return out


def compute_auto_support_resistance_features(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
    scan_every: int = 5,
) -> pd.DataFrame:
    for col in AUTO_SR_FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0.0

    cols_lc = {c.lower(): c for c in df.columns}
    h_col = cols_lc.get("high")
    l_col = cols_lc.get("low")
    c_col = cols_lc.get("close")
    if h_col is None or l_col is None or c_col is None:
        logger.warning("[auto_sr] missing OHLC for %s — zeroing", ticker)
        return df

    highs = df[h_col].to_numpy(dtype=float)
    lows = df[l_col].to_numpy(dtype=float)
    closes = df[c_col].to_numpy(dtype=float)
    n = len(closes)
    if n < 60:
        return df

    out = {name: np.zeros(n, dtype=float) for name in AUTO_SR_FEATURE_NAMES}
    out["auto_sr_above_age_bars"][:] = 999.0
    out["auto_sr_below_age_bars"][:] = 999.0
    out["auto_sr_above_dist"][:] = 0.0
    out["auto_sr_below_dist"][:] = 0.0
    out["auto_sr_position_in_range"][:] = 0.5

    last_vals = {k: 0.0 for k in AUTO_SR_FEATURE_NAMES}

    for t in range(40, n):
        if (t - 40) % scan_every != 0 and t > 40:
            for k in AUTO_SR_FEATURE_NAMES:
                out[k][t] = last_vals[k]
            continue
        prefix_h = highs[:t]
        prefix_l = lows[:t]
        cur = closes[t - 1] if t > 0 else closes[t]
        if cur <= 0:
            for k in AUTO_SR_FEATURE_NAMES:
                out[k][t] = last_vals[k]
            continue
        pivot_hi_windows: List[Tuple[int, float]] = []
        for w in (5, 10, 20):
            pivot_hi_windows.extend(_rolling_pivot_highs(prefix_h, w))
        pivot_lo_windows: List[Tuple[int, float]] = []
        for w in (5, 10, 20):
            pivot_lo_windows.extend(_rolling_pivot_lows(prefix_l, w))

        # dedup by idx
        pivot_hi = list({(i, p): None for (i, p) in pivot_hi_windows}.keys())
        pivot_lo = list({(i, p): None for (i, p) in pivot_lo_windows}.keys())

        resistance_bands = _cluster_levels(pivot_hi, cur, epsilon=0.005)
        support_bands = _cluster_levels(pivot_lo, cur, epsilon=0.005)

        above = [b for b in resistance_bands if b[0] > cur]
        below = [b for b in support_bands if b[0] < cur]

        if above:
            nearest_above = min(above, key=lambda b: b[0] - cur)
            out["auto_sr_above_dist"][t] = (nearest_above[0] - cur) / cur
            out["auto_sr_strength_above"][t] = float(nearest_above[1])
            out["auto_sr_above_age_bars"][t] = float(max(0, t - 1 - nearest_above[2]))
        if below:
            nearest_below = max(below, key=lambda b: b[0])
            out["auto_sr_below_dist"][t] = (cur - nearest_below[0]) / cur
            out["auto_sr_strength_below"][t] = float(nearest_below[1])
            out["auto_sr_below_age_bars"][t] = float(max(0, t - 1 - nearest_below[2]))

        n_levels = len(resistance_bands) + len(support_bands)
        out["auto_sr_n_levels"][t] = float(n_levels)

        if above and below:
            ap_, bp_ = nearest_above[0], nearest_below[0]
            rng = ap_ - bp_
            if rng > 0:
                out["auto_sr_range_pct"][t] = rng / cur
                out["auto_sr_position_in_range"][t] = (cur - bp_) / rng

        # Breakout: cur close pierces nearest resistance from below (+1) or
        # nearest support from above (-1).
        if t >= 2:
            prev = closes[t - 2]
            if above and prev <= nearest_above[0] < cur:
                out["auto_sr_breakout_score"][t] = 1.0
            elif below and prev >= nearest_below[0] > cur:
                out["auto_sr_breakout_score"][t] = -1.0

        for k in AUTO_SR_FEATURE_NAMES:
            last_vals[k] = out[k][t]

    # .shift(1) write-back
    idx = df.index
    for name in AUTO_SR_FEATURE_NAMES:
        df[name] = pd.Series(out[name], index=idx).shift(1).fillna(0.0).to_numpy(dtype=float)

    logger.debug("[auto_sr] added %d features for %s", AUTO_SR_FEATURE_COUNT, ticker)
    return df


if __name__ == "__main__":  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="SMOKE")
    args = ap.parse_args()
    np.random.seed(13)
    n = 250
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.6)
    high = close + np.abs(np.random.randn(n) * 0.4)
    low = close - np.abs(np.random.randn(n) * 0.4)
    df = pd.DataFrame({"high": high, "low": low, "close": close, "open": close, "volume": 1000})
    df_out = compute_auto_support_resistance_features(df, ticker=args.ticker)
    new_cols = [c for c in df_out.columns if c.startswith("auto_sr_")]
    print(f"[smoke] ticker={args.ticker} rows={len(df_out)} new_cols={len(new_cols)}")
    nz = sum(int((df_out[c] != 0).any()) for c in new_cols)
    print(f"[smoke] non-zero cols: {nz}/{len(new_cols)}")
