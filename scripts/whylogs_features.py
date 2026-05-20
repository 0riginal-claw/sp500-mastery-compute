"""whylogs_features.py — data-profile-drift features via whylabs/whylogs (STUB).

TODO: wire into v10 / Mythos pipeline.
Source repo: https://github.com/whylabs/whylogs (Apache-2.0, ~2.5k stars).
Clone path: AI-Tools/repos-claude-clones/whylogs
Install:   pip install whylogs

Look-ahead safety: per-bar profile summarizes ONLY past `chunk_window`
bars. Comparison is against a reference profile of bars STRICTLY further
in the past. All outputs .shift(1) for defense-in-depth.

Estimated features added per ticker: ~4 columns
(whylogs_kl_div, whylogs_jensen_shannon, whylogs_n_features_drifted,
whylogs_close_distribution_shift).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    p = np.maximum(p / max(p.sum(), 1e-9), 1e-9)
    q = np.maximum(q / max(q.sum(), 1e-9), 1e-9)
    m = 0.5 * (p + q)
    return float(0.5 * (p * np.log(p / m)).sum() + 0.5 * (q * np.log(q / m)).sum())


def _kl_div(p: np.ndarray, q: np.ndarray) -> float:
    p = np.maximum(p / max(p.sum(), 1e-9), 1e-9)
    q = np.maximum(q / max(q.sum(), 1e-9), 1e-9)
    return float((p * np.log(p / q)).sum())


def _histogram_pair(ref: np.ndarray, ana: np.ndarray, bins: int = 20) -> tuple[np.ndarray, np.ndarray]:
    ref = ref[~np.isnan(ref)]
    ana = ana[~np.isnan(ana)]
    if len(ref) < 10 or len(ana) < 5:
        return np.array([]), np.array([])
    edges = np.histogram_bin_edges(np.concatenate([ref, ana]), bins=bins)
    ref_h, _ = np.histogram(ref, bins=edges)
    ana_h, _ = np.histogram(ana, bins=edges)
    return ref_h.astype(float), ana_h.astype(float)


def add_whylogs_features(
    df: pd.DataFrame,
    ticker: str,
    reference_window: int = 250,
    analysis_window: int = 20,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Add rolling whylogs-style profile-drift features.

    Args:
        df: DataFrame with numeric feature columns.
        ticker: ticker symbol.
        reference_window: bars in reference profile.
        analysis_window: bars in analysis profile (most recent).
        feature_cols: columns to monitor. If None: close + top-3 var.

    Notes:
        - For each bar i, ref = [i-ref-ana, i-ana), ana = [i-ana, i).
        - JS-div, KL-div per feature; aggregate stats exposed.
    """
    out = df.copy()
    if feature_cols is None:
        cands = ["close"]
        skip = {"open", "high", "low", "close", "volume"}
        numeric = out.select_dtypes(include=[np.number]).columns.tolist()
        non_ohlcv = [c for c in numeric if c not in skip and not c.startswith("label_")]
        non_ohlcv.sort(key=lambda c: -float(np.nan_to_num(out[c].var(), nan=0.0)))
        feature_cols = (cands + non_ohlcv[:3])[:4]

    kl_max = np.full(len(out), np.nan)
    js_max = np.full(len(out), np.nan)
    n_drifted = np.full(len(out), np.nan)
    close_shift = np.full(len(out), np.nan)
    for i in range(len(out)):
        if i < reference_window + analysis_window:
            continue
        kls, jss = [], []
        for c in feature_cols:
            ref = out[c].iloc[i - reference_window - analysis_window: i - analysis_window].values
            ana = out[c].iloc[i - analysis_window: i].values
            rh, ah = _histogram_pair(ref, ana)
            if len(rh) == 0:
                continue
            kls.append(_kl_div(ah, rh))
            jss.append(_jensen_shannon(rh, ah))
            if c == "close":
                close_shift[i] = jss[-1]
        if kls:
            kl_max[i] = float(np.nanmax(kls))
        if jss:
            js_max[i] = float(np.nanmax(jss))
            n_drifted[i] = int(sum(j > 0.1 for j in jss))
    out["whylogs_kl_max"] = pd.Series(kl_max, index=out.index).shift(1)
    out["whylogs_js_max"] = pd.Series(js_max, index=out.index).shift(1)
    out["whylogs_n_drifted"] = pd.Series(n_drifted, index=out.index).shift(1)
    out["whylogs_close_js"] = pd.Series(close_shift, index=out.index).shift(1)
    return out


if __name__ == "__main__":
    print("TODO: wire whylogs_features into v10 as meta-input + monitoring.")
