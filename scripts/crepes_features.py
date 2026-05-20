"""crepes_features.py — conformal-prediction interval features via henrikbostrom/crepes (STUB).

TODO: wire into v10 / Mythos pipeline.
Source repo: https://github.com/henrikbostrom/crepes (BSD-3, ~400 stars).
Clone path: AI-Tools/repos-claude-clones/crepes
Install:   pip install crepes

Look-ahead safety: conformal calibration scores are computed on a PAST
calibration window, then applied to the current bar's prediction. Outputs:
prediction interval width + p-value of current observation under the
calibrated distribution. All outputs .shift(1) for defense-in-depth.

Estimated features added per ticker: ~3 columns
(crepes_interval_width_p80, crepes_interval_width_p95, crepes_pvalue).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_crepes_features(
    df: pd.DataFrame,
    ticker: str,
    pred_col: str = "v10_prob",
    outcome_col: str = "label_5d_up",
    cal_window: int = 250,
    quantiles: tuple = (0.80, 0.95),
) -> pd.DataFrame:
    """Add rolling conformal prediction interval / p-value features.

    Args:
        df: DataFrame with prediction + outcome columns.
        ticker: ticker symbol.
        pred_col: model prediction column (binary prob or regression pred).
        outcome_col: realized outcome column.
        cal_window: trailing calibration window length.
        quantiles: which interval coverages to expose.

    Notes:
        - Non-conformity score = |pred - outcome|.
        - Interval width at coverage q = empirical quantile of past
          non-conformity scores. Strictly past-only.
        - .shift(1) applied to all outputs.
    """
    out = df.copy()
    if pred_col not in out.columns or outcome_col not in out.columns:
        return out
    nc = (out[pred_col] - out[outcome_col]).abs()
    for q in quantiles:
        col = f"crepes_interval_width_p{int(q*100):02d}"
        out[col] = nc.rolling(cal_window, min_periods=50).quantile(q).shift(1)
    # rolling p-value of the current non-conformity score under past distribution
    nc_now = nc
    # rank of current vs past cal_window samples → p-value
    def _rank_pct(window_vals: np.ndarray) -> float:
        if np.isnan(window_vals[-1]) or len(window_vals) < 50:
            return np.nan
        past = window_vals[:-1]
        past = past[~np.isnan(past)]
        if len(past) < 30:
            return np.nan
        return float((past >= window_vals[-1]).mean())
    out["crepes_pvalue"] = (
        nc_now.rolling(cal_window, min_periods=50)
        .apply(_rank_pct, raw=True)
        .shift(1)
    )
    return out


if __name__ == "__main__":
    print("TODO: wire crepes_features into v10 risk-sizing layer.")
