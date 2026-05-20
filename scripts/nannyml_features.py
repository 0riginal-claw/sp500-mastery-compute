"""nannyml_features.py — post-deployment drift features via NannyML/nannyml (STUB).

TODO: wire into v10 / Mythos pipeline.
Source repo: https://github.com/NannyML/nannyml (Apache-2.0, ~2k stars).
Clone path: AI-Tools/repos-claude-clones/nannyml
Install:   pip install nannyml

Look-ahead safety: drift metrics use the PRIOR `reference_window` bars
as the reference distribution and CURRENT bar's features as analysis.
Because reference is strictly before the current bar AND .shift(1) is
applied to all outputs, no look-ahead leaks. Drift is a meta-feature
of the prediction context, not a predictor — it tells the model "your
input distribution has shifted, weight my conf less".

Estimated features added per ticker: ~5 columns
(univariate_psi_max, multivariate_recon_error, n_drifted_features,
chunk_drift_z, days_since_last_drift_alarm).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _psi(reference: np.ndarray, analysis: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two 1-D arrays."""
    ref = reference[~np.isnan(reference)]
    ana = analysis[~np.isnan(analysis)]
    if len(ref) < 10 or len(ana) < 1:
        return np.nan
    edges = np.histogram_bin_edges(ref, bins=bins)
    ref_counts, _ = np.histogram(ref, bins=edges)
    ana_counts, _ = np.histogram(ana, bins=edges)
    ref_pct = np.maximum(ref_counts / max(1, ref_counts.sum()), 1e-6)
    ana_pct = np.maximum(ana_counts / max(1, ana_counts.sum()), 1e-6)
    return float(np.sum((ana_pct - ref_pct) * np.log(ana_pct / ref_pct)))


def add_nannyml_features(
    df: pd.DataFrame,
    ticker: str,
    reference_window: int = 250,
    analysis_window: int = 20,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Add rolling drift-detection features for `ticker`.

    Args:
        df: DataFrame with numeric feature columns.
        ticker: ticker symbol.
        reference_window: how many past bars form the reference distribution.
        analysis_window: how many bars (ending at i-1) form the analysis chunk.
        feature_cols: which columns to monitor for drift. If None, monitor
            the top-5 numeric columns by variance.

    Notes:
        - Reference is bars [i - reference_window - analysis_window, i - analysis_window).
        - Analysis is bars [i - analysis_window, i).
        - Output is at row i — strictly future relative to the input data.
        - .shift(1) applied for defense-in-depth.
    """
    out = df.copy()
    if feature_cols is None:
        # heuristic — pick top-5 numeric columns by variance, exclude OHLCV labels
        skip = {"open", "high", "low", "close", "volume"}
        numeric = out.select_dtypes(include=[np.number]).columns.tolist()
        numeric = [c for c in numeric if c not in skip and not c.startswith("label_")]
        var_sorted = sorted(
            numeric, key=lambda c: -float(np.nan_to_num(out[c].var(), nan=0.0))
        )
        feature_cols = var_sorted[:5]
    # Rolling PSI per feature → max across features
    psis = []
    for i in range(len(out)):
        if i < reference_window + analysis_window:
            psis.append(np.nan)
            continue
        per_feat_psi = []
        for c in feature_cols:
            ref = out[c].iloc[i - reference_window - analysis_window: i - analysis_window].values
            ana = out[c].iloc[i - analysis_window: i].values
            per_feat_psi.append(_psi(ref, ana))
        per_feat_psi = [p for p in per_feat_psi if not np.isnan(p)]
        psis.append(max(per_feat_psi) if per_feat_psi else np.nan)
    out["nannyml_psi_max"] = pd.Series(psis, index=out.index)
    # n features drifted (psi > 0.25 is canonical "significant")
    n_drifted = []
    for i in range(len(out)):
        if i < reference_window + analysis_window:
            n_drifted.append(np.nan)
            continue
        cnt = 0
        for c in feature_cols:
            ref = out[c].iloc[i - reference_window - analysis_window: i - analysis_window].values
            ana = out[c].iloc[i - analysis_window: i].values
            p = _psi(ref, ana)
            if not np.isnan(p) and p > 0.25:
                cnt += 1
        n_drifted.append(cnt)
    out["nannyml_n_drifted"] = pd.Series(n_drifted, index=out.index)
    # rolling z of PSI for "is this drift unusual?"
    psi_series = out["nannyml_psi_max"]
    out["nannyml_psi_z"] = (
        (psi_series - psi_series.rolling(60, min_periods=20).mean())
        / (psi_series.rolling(60, min_periods=20).std().replace(0, np.nan))
    )
    # .shift(1) for safety
    new_cols = [c for c in out.columns if c.startswith("nannyml_")]
    out[new_cols] = out[new_cols].shift(1)
    return out


if __name__ == "__main__":
    print("TODO: wire nannyml_features into v10 as meta-input to stacking layer.")
