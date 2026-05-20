"""dcurves_features.py — Decision Curve Analysis features via MSKCC-Epi-Bio/dcurves (STUB).

TODO: wire into v10 / Mythos pipeline.
Source repo: https://github.com/MSKCC-Epi-Bio/dcurves (Apache-2.0).
Clone path: AI-Tools/repos-claude-clones/dcurves
Install:   pip install dcurves

Look-ahead safety: rolling-window prob calibration; per-bar metric uses
ONLY past N bars of (predicted_prob, outcome). Outputs .shift(1) applied
before merge with labels. Net-benefit is a calibration / utility metric
of the existing model, NOT a predictor — it is exposed as a meta-feature
that downstream stacking models can consume.

Estimated features added per ticker: ~6 columns
(net_benefit at thresholds 0.05/0.10/0.20/0.30 + treat_all_NB + best_NB_threshold).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _net_benefit(probs: np.ndarray, outcomes: np.ndarray, threshold: float) -> float:
    """Compute net benefit at a given threshold (Vickers & Elkin 2006)."""
    if len(probs) == 0:
        return np.nan
    flagged = probs >= threshold
    n = len(probs)
    if n == 0 or threshold >= 1.0:
        return np.nan
    tp = int(((outcomes == 1) & flagged).sum())
    fp = int(((outcomes == 0) & flagged).sum())
    # Vickers net benefit formula
    return (tp / n) - (fp / n) * (threshold / max(1e-9, 1 - threshold))


def add_dcurves_features(
    df: pd.DataFrame,
    ticker: str,
    prob_col: str = "v10_prob",
    outcome_col: str = "label_5d_up",
    window: int = 60,
    thresholds: tuple = (0.05, 0.10, 0.20, 0.30),
) -> pd.DataFrame:
    """Add rolling Decision Curve Analysis features.

    Args:
        df: DataFrame containing model probabilities + realized outcomes.
        ticker: ticker symbol.
        prob_col: column with predicted probabilities (e.g. 'v10_prob').
        outcome_col: column with realized binary outcomes (label).
        window: rolling window over which to compute net-benefit.
        thresholds: probability thresholds at which to evaluate net-benefit.

    Notes:
        - All output columns are .shift(1)-safe by construction (rolling-past).
        - Falls back to internal _net_benefit if `dcurves` package unavailable.
    """
    out = df.copy()
    if prob_col not in out.columns or outcome_col not in out.columns:
        # gracefully no-op so we don't break pipeline before model probs exist
        return out
    probs = out[prob_col].astype(float).values
    outs = out[outcome_col].astype(float).values
    for t in thresholds:
        col = f"dcurves_nb_t{int(t*100):02d}_w{window}"
        rolling_nb = np.full(len(out), np.nan, dtype=float)
        for i in range(window, len(out)):
            sl_probs = probs[i - window:i]
            sl_outs = outs[i - window:i]
            mask = ~(np.isnan(sl_probs) | np.isnan(sl_outs))
            if mask.sum() >= 10:
                rolling_nb[i] = _net_benefit(sl_probs[mask], sl_outs[mask], t)
        out[col] = pd.Series(rolling_nb, index=out.index).shift(1)
    # "treat all" baseline
    out[f"dcurves_treat_all_nb_w{window}"] = (
        out[outcome_col].rolling(window).mean().shift(1)
    )
    return out


if __name__ == "__main__":
    print("TODO: wire dcurves_features into v10 stacking layer.")
