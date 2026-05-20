"""eli5_features.py — permutation-importance-derived meta-features via TeamHG-Memex/eli5 (STUB).

TODO: wire into v10 / Mythos pipeline.
Source repo: https://github.com/TeamHG-Memex/eli5 (MIT, 2.8k stars).
Clone path: AI-Tools/repos-claude-clones/eli5
Install:   pip install eli5

Look-ahead safety: per-bar feature importance is computed using ONLY a
trailing window of (features, label). Since model + permutation runs on
strictly past data and outputs are .shift(1), no leakage. The resulting
feature is "how much do features X1..Xk currently matter for the rolling
model?" — a regime indicator.

Estimated features added per ticker: ~3 columns
(eli5_top_feat_pi, eli5_feat_importance_gini, eli5_top_feat_change_rate).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _gini_concentration(weights: np.ndarray) -> float:
    """Gini coefficient of an importance distribution (0=flat, 1=concentrated)."""
    w = np.abs(weights)
    w = w[~np.isnan(w)]
    if len(w) < 2:
        return np.nan
    w_sorted = np.sort(w)
    n = len(w_sorted)
    cumw = np.cumsum(w_sorted)
    if cumw[-1] == 0:
        return np.nan
    return float((n + 1 - 2 * (cumw.sum() / cumw[-1])) / n)


def add_eli5_features(
    df: pd.DataFrame,
    ticker: str,
    feature_cols: list[str] | None = None,
    label_col: str = "label_5d_up",
    window: int = 250,
    update_every: int = 20,
) -> pd.DataFrame:
    """Add rolling permutation-importance meta-features.

    Args:
        df: DataFrame with feature columns + label column.
        ticker: ticker symbol.
        feature_cols: which columns to evaluate. If None, top-10 numeric.
        label_col: binary label column.
        window: trailing window for the rolling model fit.
        update_every: refit/refute cadence (cheap proxy — we don't refit every bar).

    Notes:
        - This stub computes a cheap correlation-based proxy for PI:
          per feature, |spearman(feat_t-w..t-1, label_t-w..t-1)|. The eli5-based
          replacement at integration time will fit a sklearn estimator and call
          eli5.sklearn.PermutationImportance.
        - Causal: only past data used; .shift(1) applied.
    """
    out = df.copy()
    if label_col not in out.columns:
        return out
    if feature_cols is None:
        skip = {"open", "high", "low", "close", "volume", label_col}
        numeric = out.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric if c not in skip and not c.startswith("label_")][:10]
    if not feature_cols:
        return out

    top_pi = np.full(len(out), np.nan)
    gini = np.full(len(out), np.nan)
    last_top_feat_idx = -1
    change_count = 0
    change_rate = np.full(len(out), np.nan)
    for i in range(window, len(out)):
        if (i - window) % update_every != 0 and i > window:
            top_pi[i] = top_pi[i - 1]
            gini[i] = gini[i - 1]
            change_rate[i] = change_rate[i - 1]
            continue
        y = out[label_col].iloc[i - window: i].values
        if pd.isna(y).all() or np.unique(y[~pd.isna(y)]).size < 2:
            continue
        imps = []
        for c in feature_cols:
            x = out[c].iloc[i - window: i].values
            mask = ~(pd.isna(x) | pd.isna(y))
            if mask.sum() < 30:
                imps.append(np.nan)
                continue
            try:
                imps.append(abs(float(pd.Series(x[mask]).corr(pd.Series(y[mask]), method="spearman"))))
            except Exception:
                imps.append(np.nan)
        imps = np.array(imps)
        if np.isnan(imps).all():
            continue
        top_pi[i] = float(np.nanmax(imps))
        gini[i] = _gini_concentration(imps)
        top_idx = int(np.nanargmax(imps))
        if last_top_feat_idx >= 0 and top_idx != last_top_feat_idx:
            change_count += 1
        last_top_feat_idx = top_idx
        change_rate[i] = change_count / max(1, (i - window + 1) / update_every)
    out["eli5_top_pi"] = pd.Series(top_pi, index=out.index).shift(1)
    out["eli5_pi_gini"] = pd.Series(gini, index=out.index).shift(1)
    out["eli5_top_feat_change_rate"] = pd.Series(change_rate, index=out.index).shift(1)
    return out


if __name__ == "__main__":
    print("TODO: wire eli5_features into v10 stacking meta-input.")
