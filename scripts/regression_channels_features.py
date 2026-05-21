# autosolve_skip: data-loader infra
"""
regression_channels_features.py — linear regression channel features.

For each window N in {20, 50, 100}, computes a rolling linear regression
over log(close) and the residual standard deviation. Produces:
  - reg_channel_slope_N    : slope of the fit (per-bar log return units)
  - reg_channel_width_N    : 2-sigma channel width / mid
  - reg_channel_pos_pct_N  : current close position within ±2σ envelope (0..1)
  - reg_channel_breakout_N : +1 if close > upper, -1 if < lower, else 0

NO-LOOKAHEAD AUDIT (2026-05-21)
---------------------------------
For bar T, the fit is computed on bars [T-N..T-1] (strictly past). The
position-within-channel is computed using bar T-1's close. We then
.shift(1) before joining so bar T's input comes from a fit ending at T-2.

Features: 4 cols × 3 windows = 12.

Dependencies: numpy, pandas (no scipy needed — pure numpy).
Cost: LOW — single vectorized rolling polyfit, ~30ms per ticker.
"""

from __future__ import annotations

import logging
from typing import Optional, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REG_WINDOWS = (20, 50, 100)

REGRESSION_FEATURE_NAMES: List[str] = []
for _w in REG_WINDOWS:
    REGRESSION_FEATURE_NAMES.extend([
        f"reg_channel_slope_{_w}",
        f"reg_channel_width_{_w}",
        f"reg_channel_pos_pct_{_w}",
        f"reg_channel_breakout_{_w}",
    ])
REGRESSION_FEATURE_COUNT: int = len(REGRESSION_FEATURE_NAMES)


def _rolling_regression(y: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (slope, intercept, residual_std) arrays of length len(y).

    For each index i >= window-1, fits log-price ~ slope*x + intercept on
    y[i-window+1:i+1] and computes residual stdev.
    """
    n = len(y)
    slope = np.full(n, np.nan)
    intercept = np.full(n, np.nan)
    resid = np.full(n, np.nan)
    if n < window:
        return slope, intercept, resid
    xs = np.arange(window, dtype=float)
    x_mean = xs.mean()
    xs_dev = xs - x_mean
    denom = (xs_dev ** 2).sum()
    if denom == 0:
        return slope, intercept, resid
    for i in range(window - 1, n):
        seg = y[i - window + 1:i + 1]
        if np.any(~np.isfinite(seg)):
            continue
        y_mean = seg.mean()
        s = (xs_dev * (seg - y_mean)).sum() / denom
        b = y_mean - s * x_mean
        slope[i] = s
        intercept[i] = b
        fit = s * xs + b
        r = seg - fit
        resid[i] = float(np.sqrt((r * r).mean()))
    return slope, intercept, resid


def compute_regression_channels_features(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
) -> pd.DataFrame:
    """Add 12 regression-channel feature columns."""
    for col in REGRESSION_FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0.0

    cols_lc = {c.lower(): c for c in df.columns}
    c_col = cols_lc.get("close")
    if c_col is None:
        logger.warning("[reg_channel] missing 'close' for %s — zeroing", ticker)
        return df

    closes = df[c_col].to_numpy(dtype=float)
    n = len(closes)
    if n < max(REG_WINDOWS) + 5:
        return df

    # log price for return-additive regression
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.log(np.where(closes > 0, closes, np.nan))

    idx = df.index
    for win in REG_WINDOWS:
        slope, intercept, resid_std = _rolling_regression(log_p, win)
        # midline value at the END bar of each fit window
        end_x = win - 1
        midline = slope * end_x + intercept
        upper = midline + 2.0 * resid_std
        lower = midline - 2.0 * resid_std
        # position pct (0..1) of current log_p in channel
        with np.errstate(invalid="ignore", divide="ignore"):
            width = upper - lower
            pos = np.where(width > 0, (log_p - lower) / width, 0.5)
            pos = np.clip(pos, -0.5, 1.5)  # allow slight over/undershoot
            breakout = np.where(log_p > upper, 1.0, np.where(log_p < lower, -1.0, 0.0))

        # Channel width fraction: 4*sigma over midline. Use exp(width)-1 to scale.
        with np.errstate(invalid="ignore"):
            width_frac = np.where(np.isfinite(resid_std), 4.0 * resid_std, 0.0)

        slope_s = pd.Series(slope, index=idx).shift(1).fillna(0.0)
        width_s = pd.Series(width_frac, index=idx).shift(1).fillna(0.0)
        pos_s = pd.Series(pos, index=idx).shift(1).fillna(0.5)
        bo_s = pd.Series(breakout, index=idx).shift(1).fillna(0.0)

        df[f"reg_channel_slope_{win}"] = slope_s.to_numpy(dtype=float)
        df[f"reg_channel_width_{win}"] = width_s.to_numpy(dtype=float)
        df[f"reg_channel_pos_pct_{win}"] = pos_s.to_numpy(dtype=float)
        df[f"reg_channel_breakout_{win}"] = bo_s.to_numpy(dtype=float)

    logger.debug("[reg_channel] added %d features for %s", REGRESSION_FEATURE_COUNT, ticker)
    return df


if __name__ == "__main__":  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="SMOKE")
    args = ap.parse_args()
    np.random.seed(11)
    n = 300
    close = 100.0 * np.exp(np.cumsum(np.random.randn(n) * 0.01))
    df = pd.DataFrame({"close": close, "high": close, "low": close, "open": close, "volume": 1000})
    df_out = compute_regression_channels_features(df, ticker=args.ticker)
    new_cols = [c for c in df_out.columns if c.startswith("reg_channel_")]
    print(f"[smoke] ticker={args.ticker} rows={len(df_out)} new_cols={len(new_cols)}")
    nz = sum(int((df_out[c] != 0).any()) for c in new_cols)
    print(f"[smoke] non-zero cols: {nz}/{len(new_cols)}")
