"""pdpbox_features.py — partial-dependence-derived meta-features via SauceCat/PDPbox (STUB).

TODO: wire into v10 / Mythos pipeline.
Source repo: https://github.com/SauceCat/PDPbox (MIT, ~800 stars).
Clone path: AI-Tools/repos-claude-clones/PDPbox
Install:   pip install pdpbox

Look-ahead safety: PDP curves are computed on a rolling-past window of
(X, y). Output features summarize the CURVE (slope at current X position,
monotonicity score, max marginal effect) — all anchored to past-fit data
+ current bar's X value. .shift(1) applied to all outputs.

Estimated features added per ticker: ~4 columns
(pdp_top_feat_slope, pdp_top_feat_monotonicity, pdp_max_marginal_effect,
pdp_curve_curvature).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _bin_pdp(x: np.ndarray, y: np.ndarray, bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Cheap PDP proxy: bin x → mean(y) per bin (the "averaged response")."""
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 30:
        return np.array([]), np.array([])
    edges = np.quantile(x[mask], np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return np.array([]), np.array([])
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_means = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = mask & (x >= lo) & (x <= hi)
        bin_means.append(float(np.nanmean(y[m])) if m.sum() else np.nan)
    return centers, np.array(bin_means)


def _slope_at_current(centers: np.ndarray, response: np.ndarray, x_cur: float) -> float:
    if len(centers) < 2:
        return np.nan
    idx = int(np.argmin(np.abs(centers - x_cur)))
    lo = max(0, idx - 1)
    hi = min(len(centers) - 1, idx + 1)
    if hi == lo:
        return np.nan
    return float((response[hi] - response[lo]) / max(1e-9, centers[hi] - centers[lo]))


def _monotonicity_score(response: np.ndarray) -> float:
    """+1 if strictly increasing, -1 if strictly decreasing, 0 if mixed."""
    r = response[~np.isnan(response)]
    if len(r) < 3:
        return np.nan
    diffs = np.diff(r)
    if np.all(diffs >= 0):
        return 1.0
    if np.all(diffs <= 0):
        return -1.0
    pos = (diffs > 0).sum()
    neg = (diffs < 0).sum()
    return float((pos - neg) / max(1, len(diffs)))


def add_pdpbox_features(
    df: pd.DataFrame,
    ticker: str,
    feature_col: str | None = None,
    label_col: str = "label_5d_up",
    window: int = 250,
    bins: int = 10,
    update_every: int = 20,
) -> pd.DataFrame:
    """Add rolling partial-dependence meta-features for one feature.

    Args:
        df: DataFrame with the feature + label.
        ticker: ticker symbol.
        feature_col: feature to summarize via PDP. If None, pick the highest-
            variance non-OHLCV numeric column.
        label_col: binary label.
        window: rolling window for the PDP fit.
        bins: number of PDP bins.
        update_every: cadence of PDP recomputation.

    Causal: past-only window + .shift(1) on all outputs.
    """
    out = df.copy()
    if label_col not in out.columns:
        return out
    if feature_col is None:
        skip = {"open", "high", "low", "close", "volume", label_col}
        numeric = out.select_dtypes(include=[np.number]).columns.tolist()
        cands = [c for c in numeric if c not in skip and not c.startswith("label_")]
        if not cands:
            return out
        feature_col = max(cands, key=lambda c: float(np.nan_to_num(out[c].var(), nan=0.0)))

    slopes = np.full(len(out), np.nan)
    monos = np.full(len(out), np.nan)
    max_eff = np.full(len(out), np.nan)
    curvature = np.full(len(out), np.nan)

    cur_centers, cur_response = np.array([]), np.array([])
    for i in range(window, len(out)):
        if (i - window) % update_every == 0:
            x_win = out[feature_col].iloc[i - window: i].values
            y_win = out[label_col].iloc[i - window: i].values
            cur_centers, cur_response = _bin_pdp(x_win, y_win, bins=bins)
        if len(cur_centers) < 2:
            continue
        x_cur = float(out[feature_col].iloc[i])
        slopes[i] = _slope_at_current(cur_centers, cur_response, x_cur)
        monos[i] = _monotonicity_score(cur_response)
        valid = cur_response[~np.isnan(cur_response)]
        if len(valid):
            max_eff[i] = float(np.nanmax(valid) - np.nanmin(valid))
        # rough 2nd-difference curvature
        d2 = np.diff(cur_response, n=2)
        d2 = d2[~np.isnan(d2)]
        if len(d2):
            curvature[i] = float(np.nanmean(np.abs(d2)))

    out["pdp_slope_at_x"] = pd.Series(slopes, index=out.index).shift(1)
    out["pdp_monotonicity"] = pd.Series(monos, index=out.index).shift(1)
    out["pdp_max_marginal_effect"] = pd.Series(max_eff, index=out.index).shift(1)
    out["pdp_curvature"] = pd.Series(curvature, index=out.index).shift(1)
    return out


if __name__ == "__main__":
    print("TODO: wire pdpbox_features into v10 stacking meta-input.")
