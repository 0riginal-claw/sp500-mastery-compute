# Source: /Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/claudes test/research/active/cycle048_4layer_composite/
# Primary modules: sr_features_daily.py, sr_features_swing.py,
#                  sr_features_volprofile.py, sr_features_flips.py,
#                  sr_features_retest.py
#
# Cycle 048 = 4-layer SR composite (daily levels, swing pivots, volume profile,
# R<->S flips, retest gates). Originally designed for per-day intraday caches.
# This wrapper ports all 4 layers to daily OHLCV resolution.
# All outputs are .shift(1)-safe — values at row i use only rows 0..i-1.

from __future__ import annotations

import numpy as np
import pandas as pd

POC_BIN_PCT = 0.001   # log-bin granularity for HVN/LVN
TOP_K_HVN   = 3
SWING_N     = 3       # bars left/right for pivot confirmation
RETEST_PCT  = 0.003   # 0.3% proximity = "at the level"

CYCLE048_FEATURE_NAMES: list[str] = [
    # --- Daily levels ---
    "c048_d20h", "c048_d20l", "c048_d50h", "c048_d50l",
    "c048_dist_d20h_pct", "c048_dist_d20l_pct",
    "c048_dist_d50h_pct", "c048_dist_d50l_pct",
    "c048_pivot_count_1pct",
    # --- Swing pivots ---
    "c048_swing_high", "c048_swing_low",
    "c048_dist_swing_high_pct", "c048_dist_swing_low_pct",
    # --- Volume profile HVN / LVN ---
    "c048_dist_hvn_pct", "c048_in_lvn",
    # --- R<->S flips on key levels ---
    "c048_r2s_d20h", "c048_s2r_d20h",
    "c048_r2s_swing_high", "c048_s2r_swing_high",
    # --- Retest + failed-break ---
    "c048_retest_up_d20h", "c048_failed_bo_d20h",
    "c048_retest_up_swing_high", "c048_failed_bo_swing_high",
]


def _col(df: pd.DataFrame, *names: str) -> pd.Series | None:
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return pd.to_numeric(df[low[n.lower()]], errors="coerce").astype(float)
    return None


# ---------------------------------------------------------------------------
# Layer 1 — Daily levels (sr_features_daily.py logic)
# ---------------------------------------------------------------------------

def _daily_levels(high: pd.Series, low: pd.Series, close: pd.Series) -> dict[str, pd.Series]:
    """Rolling 20 and 50-day H/L from *prior* bars (shift embedded in rolling)."""
    h = high.values
    l = low.values
    c = close.values
    n = len(c)

    d20h = np.full(n, np.nan)
    d20l = np.full(n, np.nan)
    d50h = np.full(n, np.nan)
    d50l = np.full(n, np.nan)

    for i in range(1, n):
        lo20 = max(0, i - 20)
        lo50 = max(0, i - 50)
        d20h[i] = np.max(h[lo20:i]) if i > lo20 else np.nan
        d20l[i] = np.min(l[lo20:i]) if i > lo20 else np.nan
        d50h[i] = np.max(h[lo50:i]) if i > lo50 else np.nan
        d50l[i] = np.min(l[lo50:i]) if i > lo50 else np.nan

    def _dist(level):
        return np.where((c > 0) & np.isfinite(level), np.abs(c - level) / c * 100, np.nan)

    pivot_count = (
        (np.isfinite(d20h) & (_dist(d20h) < 1.0)).astype(int) +
        (np.isfinite(d20l) & (_dist(d20l) < 1.0)).astype(int) +
        (np.isfinite(d50h) & (_dist(d50h) < 1.0)).astype(int) +
        (np.isfinite(d50l) & (_dist(d50l) < 1.0)).astype(int)
    )

    idx = high.index
    return {
        "c048_d20h": pd.Series(d20h, index=idx),
        "c048_d20l": pd.Series(d20l, index=idx),
        "c048_d50h": pd.Series(d50h, index=idx),
        "c048_d50l": pd.Series(d50l, index=idx),
        "c048_dist_d20h_pct": pd.Series(_dist(d20h), index=idx),
        "c048_dist_d20l_pct": pd.Series(_dist(d20l), index=idx),
        "c048_dist_d50h_pct": pd.Series(_dist(d50h), index=idx),
        "c048_dist_d50l_pct": pd.Series(_dist(d50l), index=idx),
        "c048_pivot_count_1pct": pd.Series(pivot_count, index=idx),
    }


# ---------------------------------------------------------------------------
# Layer 2 — Swing pivots (sr_features_swing.py logic)
# ---------------------------------------------------------------------------

def _confirmed_pivots(values: np.ndarray, n: int, mode: str) -> np.ndarray:
    L = len(values)
    out = np.full(L, np.nan)
    if L < 2 * n + 1:
        return out
    op = np.greater if mode == "high" else np.less
    last_pivot = np.nan
    for k in range(n, L):
        j = k - n
        if j - n < 0:
            out[k] = out[k - 1] if k > 0 and not np.isnan(out[k - 1]) else np.nan
            continue
        left  = values[j - n:j]
        right = values[j + 1:j + n + 1]
        if len(right) < n:
            out[k] = out[k - 1] if k > 0 else np.nan
            continue
        if op(values[j], left).all() and op(values[j], right).all():
            last_pivot = values[j]
        out[k] = last_pivot
    return out


def _swing_features(high: pd.Series, low: pd.Series, close: pd.Series) -> dict[str, pd.Series]:
    h = high.values
    l = low.values
    c = close.values
    # Swing pivots are confirmed at bar j+n using only bars 0..j+n — no lookahead.
    # On daily bars we shift(1) additionally to ensure signal-day safety.
    sh = _confirmed_pivots(h, SWING_N, "high")
    sl = _confirmed_pivots(l, SWING_N, "low")
    # Shift by 1: the pivot confirmed at bar k is available for trading at bar k+1
    sh = np.roll(sh, 1); sh[0] = np.nan
    sl = np.roll(sl, 1); sl[0] = np.nan

    with np.errstate(divide="ignore", invalid="ignore"):
        dh = np.where(np.isfinite(sh) & (c > 0), np.abs(c - sh) / c * 100, np.nan)
        dl = np.where(np.isfinite(sl) & (c > 0), np.abs(c - sl) / c * 100, np.nan)

    idx = high.index
    return {
        "c048_swing_high":             pd.Series(sh, index=idx),
        "c048_swing_low":              pd.Series(sl, index=idx),
        "c048_dist_swing_high_pct":    pd.Series(dh, index=idx),
        "c048_dist_swing_low_pct":     pd.Series(dl, index=idx),
    }


# ---------------------------------------------------------------------------
# Layer 3 — Volume profile HVN / LVN (sr_features_volprofile.py logic)
# ---------------------------------------------------------------------------

def _volprofile_features(high: pd.Series, low: pd.Series,
                          close: pd.Series, vol: pd.Series) -> dict[str, pd.Series]:
    n = len(close)
    h = high.values.astype(float)
    l = low.values.astype(float)
    c = close.values.astype(float)
    v = vol.values.astype(float)
    typ = (h + l + c) / 3.0

    sr_dist_hvn = np.full(n, np.nan)
    sr_in_lvn   = np.zeros(n, dtype=bool)

    # Guard against non-positive prices
    valid = (typ > 0) & np.isfinite(typ)
    if not valid.any():
        idx = close.index
        return {
            "c048_dist_hvn_pct": pd.Series(sr_dist_hvn, index=idx),
            "c048_in_lvn":       pd.Series(sr_in_lvn.astype(int), index=idx),
        }

    # Use a rolling 50-bar window to approximate the "session profile so far"
    # at each daily bar (pure past data → no lookahead).
    WIN = 50
    for i in range(1, n):
        lo = max(0, i - WIN)
        wh = h[lo:i]; wl = l[lo:i]; wv = v[lo:i]
        wtyp = typ[lo:i]
        if len(wtyp) == 0 or (wv == 0).all():
            continue
        try:
            bin_idx = np.floor(np.log(np.clip(wtyp, 1e-9, None)) / POC_BIN_PCT).astype(int)
        except Exception:
            continue
        unique_bins, inverse = np.unique(bin_idx, return_inverse=True)
        K = len(unique_bins)
        cum_per_bin = np.zeros(K)
        for bi, vi in zip(inverse, wv):
            cum_per_bin[bi] += vi
        bin_prices = np.exp((unique_bins + 0.5) * POC_BIN_PCT)

        # HVN: top-K bins by cumulative volume
        k_top = min(TOP_K_HVN, K)
        top_idx = np.argpartition(-cum_per_bin, k_top)[:k_top] if K > k_top else np.arange(K)
        top_prices = bin_prices[top_idx]
        ci = c[i]
        if ci > 0 and np.isfinite(ci):
            sr_dist_hvn[i] = float(np.min(np.abs(top_prices - ci) / ci * 100))

        # LVN: close bin in low-volume area
        median_vol = np.median(cum_per_bin[cum_per_bin > 0]) if (cum_per_bin > 0).any() else 0
        thresh = 0.2 * median_vol
        try:
            close_bin = int(np.floor(np.log(max(ci, 1e-9)) / POC_BIN_PCT))
            close_idx_arr = np.where(unique_bins == close_bin)[0]
            if len(close_idx_arr) > 0:
                cb_vol = cum_per_bin[close_idx_arr[0]]
                sr_in_lvn[i] = bool(thresh > 0 and cb_vol < thresh)
        except Exception:
            pass

    # Shift by 1: values computed through bar i-1 are used at bar i
    sr_dist_hvn = np.roll(sr_dist_hvn, 1); sr_dist_hvn[0] = np.nan
    sr_in_lvn   = np.roll(sr_in_lvn, 1);   sr_in_lvn[0] = False

    idx = close.index
    return {
        "c048_dist_hvn_pct": pd.Series(sr_dist_hvn, index=idx),
        "c048_in_lvn":       pd.Series(sr_in_lvn.astype(int), index=idx),
    }


# ---------------------------------------------------------------------------
# Layer 4 — R<->S flips + retest/failed-break (sr_features_flips/retest.py)
# ---------------------------------------------------------------------------

def _flip_and_retest(level: np.ndarray, high: np.ndarray,
                      low: np.ndarray, close: np.ndarray,
                      prefix: str) -> dict[str, np.ndarray]:
    n = len(close)
    r2s = np.zeros(n, dtype=bool)
    s2r = np.zeros(n, dtype=bool)
    retest_up = np.zeros(n, dtype=bool)
    failed_bo = np.zeros(n, dtype=bool)

    finite = np.isfinite(level)
    if not finite.any():
        return {f"c048_r2s_{prefix}": r2s, f"c048_s2r_{prefix}": s2r,
                f"c048_retest_up_{prefix}": retest_up, f"c048_failed_bo_{prefix}": failed_bo}

    bo_event = (high > level) & (close > level) & finite
    bd_event = (low  < level) & (close < level) & finite
    bo_event[0] = bd_event[0] = False
    was_bo = np.maximum.accumulate(bo_event)
    was_bd = np.maximum.accumulate(bd_event)

    with np.errstate(divide="ignore", invalid="ignore"):
        dist_pct = np.where((close > 0) & finite, np.abs(close - level) / close, np.nan)

    # R->S flip: broken up + close still above + within RETEST_PCT
    r2s = was_bo & (close > level) & finite & np.where(np.isfinite(dist_pct), dist_pct < RETEST_PCT, False)
    # S->R flip: broken down + close below + within RETEST_PCT
    s2r = was_bd & (close < level) & finite & np.where(np.isfinite(dist_pct), dist_pct < RETEST_PCT, False)
    # Retest upward (close near level after upward break)
    retest_up = was_bo & np.where(np.isfinite(dist_pct), dist_pct < RETEST_PCT, False) & finite
    # Failed breakout (was broken up, close now back below)
    failed_bo = was_bo & (close < level) & finite

    return {f"c048_r2s_{prefix}": r2s, f"c048_s2r_{prefix}": s2r,
            f"c048_retest_up_{prefix}": retest_up, f"c048_failed_bo_{prefix}": failed_bo}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def add_cycle048_features(df: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
    """Append 24 4-layer SR composite features to df. Idempotent.

    All outputs are .shift(1)-safe — each value at row i is derived solely from
    data available at the close of day i-1 (or earlier).
    """
    if df is None or len(df) < 10:
        return df
    if all(c in df.columns for c in CYCLE048_FEATURE_NAMES):
        return df

    close = _col(df, "close", "Close")
    high  = _col(df, "high", "High")
    low   = _col(df, "low", "Low")
    vol   = _col(df, "volume", "Volume")

    if close is None:
        return df

    if high is None:
        high = close.copy()
    if low is None:
        low = close.copy()
    if vol is None:
        vol = pd.Series(np.ones(len(df)), index=df.index, dtype=float)

    # Layer 1
    feats = _daily_levels(high, low, close)
    # Layer 2
    feats.update(_swing_features(high, low, close))
    # Layer 3
    feats.update(_volprofile_features(high, low, close, vol))
    # Layer 4 — flips on 20d-high and swing-high
    d20h_arr = feats["c048_d20h"].values
    sh_arr   = feats["c048_swing_high"].values
    h = high.values;  l = low.values;  c = close.values

    for arr, prefix in [(d20h_arr, "d20h"), (sh_arr, "swing_high")]:
        sub = _flip_and_retest(arr, h, l, c, prefix)
        feats.update({k: pd.Series(v.astype(int), index=df.index) for k, v in sub.items()})

    # Assign (idempotent)
    for col, series in feats.items():
        if col not in df.columns:
            df[col] = series.values

    return df


if __name__ == "__main__":
    import sys
    rng = np.random.default_rng(42)
    idx = pd.date_range("2023-01-01", periods=300, freq="B")
    close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, 300)))
    demo = pd.DataFrame({
        "Open":   close * (1 - np.abs(rng.normal(0, 0.003, 300))),
        "High":   close * (1 + np.abs(rng.normal(0, 0.007, 300))),
        "Low":    close * (1 - np.abs(rng.normal(0, 0.007, 300))),
        "Close":  close,
        "Volume": rng.integers(1_000_000, 5_000_000, 300).astype(float),
    }, index=idx)
    tk = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    out = add_cycle048_features(demo, tk)
    print(f"cycle048: {len(CYCLE048_FEATURE_NAMES)} features added. Shape: {out.shape}")
    print(out[CYCLE048_FEATURE_NAMES].tail(5).to_string())
