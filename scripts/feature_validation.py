"""Alphalens-based feature validation gate.

Top-10 ship #9 (2026-05-22): force-multiplier validator that kills low-IC
features BEFORE they enter the production manifest. NOT a feature source.

Public API:
    validate_feature(feature_series, returns_series, quantiles=5) -> dict
        keys: ic, ic_pval, quantile_returns, turnover, n_obs, decision

    validate_feature_batch(features_df, returns_series, min_ic=0.02) -> pd.DataFrame
        Run validate_feature on each column; return rank-sorted summary.

    write_ic_scores(scores_df, path='feature_ic_scores.parquet') -> Path

Decision rule:
    |IC| >= min_ic (default 0.02) -> KEEP
    else                          -> REJECT

Activation:
    ALPHALENS_VALIDATION=1   (default OFF — don't break existing pipeline)

Workflow (recommended, post-sweep):
    1. Compute feature columns for ticker universe.
    2. Compute forward returns_series (e.g. 1-day fwd close-to-close).
    3. For each candidate feature, call validate_feature.
    4. Drop rejects; promote keeps to feature_manifest.json.
    5. Write feature_ic_scores.parquet for diff vs prior run.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

# Env-gate. Default OFF.
ENABLED = os.environ.get("ALPHALENS_VALIDATION", "0") == "1"

# Defaults — overridable per call.
DEFAULT_QUANTILES = 5
DEFAULT_MIN_IC = 0.02


def _spearman_ic(feature: pd.Series, returns: pd.Series) -> tuple[float, float]:
    """Spearman rank correlation + 2-sided p-value.

    Aligns indices via inner-join, drops NaN pairs.
    Returns (ic, pval). On <30 obs returns (nan, nan).
    """
    df = pd.concat([feature, returns], axis=1, join="inner").dropna()
    if len(df) < 30:
        return float("nan"), float("nan")
    try:
        from scipy.stats import spearmanr  # lazy
        rho, pval = spearmanr(df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy())
        return float(rho), float(pval)
    except Exception as e:
        LOG.debug("spearman fallback (no scipy?): %s", e)
        # Pandas fallback — no pval.
        rho = df.iloc[:, 0].corr(df.iloc[:, 1], method="spearman")
        return float(rho), float("nan")


def _quantile_returns(
    feature: pd.Series, returns: pd.Series, quantiles: int,
) -> dict[str, float]:
    """Quantile spread analysis. Returns mean return by quantile + top-bottom spread."""
    df = pd.concat([feature, returns], axis=1, join="inner").dropna()
    df.columns = ["f", "r"]
    if len(df) < quantiles * 10:
        return {"top_quantile": float("nan"), "bottom_quantile": float("nan"), "spread": float("nan")}
    try:
        df["q"] = pd.qcut(df["f"].rank(method="first"), quantiles, labels=False, duplicates="drop")
    except Exception as e:
        LOG.debug("qcut failed: %s", e)
        return {"top_quantile": float("nan"), "bottom_quantile": float("nan"), "spread": float("nan")}
    grp = df.groupby("q")["r"].mean()
    if len(grp) < 2:
        return {"top_quantile": float("nan"), "bottom_quantile": float("nan"), "spread": float("nan")}
    top = float(grp.iloc[-1])
    bot = float(grp.iloc[0])
    return {"top_quantile": top, "bottom_quantile": bot, "spread": top - bot}


def _turnover(feature: pd.Series, quantiles: int) -> float:
    """Average per-period quantile-bin churn. 0 = no churn, 1 = full re-rank each step."""
    if not isinstance(feature.index, pd.DatetimeIndex) and not feature.index.is_monotonic_increasing:
        return float("nan")
    if len(feature) < quantiles * 10:
        return float("nan")
    try:
        binned = pd.qcut(
            feature.rank(method="first"), quantiles, labels=False, duplicates="drop",
        )
        if binned.isna().all():
            return float("nan")
        changes = (binned.diff().fillna(0) != 0).sum()
        return float(changes / max(len(binned) - 1, 1))
    except Exception as e:
        LOG.debug("turnover calc failed: %s", e)
        return float("nan")


def validate_feature(
    feature_series: pd.Series,
    returns_series: pd.Series,
    quantiles: int = DEFAULT_QUANTILES,
    min_ic: float = DEFAULT_MIN_IC,
) -> dict[str, Any]:
    """Validate one feature against forward returns.

    Args:
        feature_series: indexed by date (or [date,ticker] MultiIndex)
        returns_series: forward returns, same index spec
        quantiles:      number of quantile buckets (default 5)
        min_ic:         |IC| threshold for KEEP decision (default 0.02)

    Returns:
        dict with keys:
            ic                : Spearman rank IC (signed)
            ic_pval           : 2-sided p-value (nan if scipy unavailable)
            quantile_returns  : {top_quantile, bottom_quantile, spread}
            turnover          : avg per-period quantile churn (0..1)
            n_obs             : aligned non-NaN sample size
            decision          : 'KEEP' | 'REJECT'
            min_ic            : threshold used
    """
    n_obs = int(pd.concat([feature_series, returns_series], axis=1, join="inner").dropna().shape[0])
    ic, pval = _spearman_ic(feature_series, returns_series)
    qret = _quantile_returns(feature_series, returns_series, quantiles)
    turn = _turnover(feature_series, quantiles)
    keep = (not np.isnan(ic)) and (abs(ic) >= min_ic)
    decision = "KEEP" if keep else "REJECT"
    return {
        "ic": ic,
        "ic_pval": pval,
        "quantile_returns": qret,
        "turnover": turn,
        "n_obs": n_obs,
        "decision": decision,
        "min_ic": min_ic,
    }


def validate_feature_batch(
    features_df: pd.DataFrame,
    returns_series: pd.Series,
    quantiles: int = DEFAULT_QUANTILES,
    min_ic: float = DEFAULT_MIN_IC,
) -> pd.DataFrame:
    """Run validate_feature across every column. Returns rank-sorted summary."""
    if not ENABLED:
        LOG.debug("alphalens validation disabled (ALPHALENS_VALIDATION!=1)")
        return pd.DataFrame()
    rows = []
    for col in features_df.columns:
        res = validate_feature(features_df[col], returns_series, quantiles, min_ic)
        row = {
            "feature": col,
            "ic": res["ic"],
            "ic_pval": res["ic_pval"],
            "abs_ic": abs(res["ic"]) if not np.isnan(res["ic"]) else float("nan"),
            "spread": res["quantile_returns"]["spread"],
            "turnover": res["turnover"],
            "n_obs": res["n_obs"],
            "decision": res["decision"],
        }
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("abs_ic", ascending=False, na_position="last")
    df.attrs["min_ic"] = min_ic
    return df


def write_ic_scores(
    scores_df: pd.DataFrame,
    path: str | Path = "feature_ic_scores.parquet",
) -> Path:
    """Persist scores to parquet for diff-vs-prior-run analysis."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        scores_df.to_parquet(p, index=False)
    except Exception as e:
        LOG.warning("parquet write failed (%s), falling back to csv", e)
        p = p.with_suffix(".csv")
        scores_df.to_csv(p, index=False)
    return p


def alphalens_full_tearsheet(
    feature_series: pd.Series,
    prices_df: pd.DataFrame,
    quantiles: int = DEFAULT_QUANTILES,
) -> Any:
    """Optional deep-dive via alphalens-reloaded (slow, full plots).

    Use sparingly. validate_feature() is the fast gate; this is for promoted
    features where a full IC decay + Sharpe report is wanted.
    """
    if not ENABLED:
        return None
    try:
        import alphalens as al  # noqa: WPS433
        factor_data = al.utils.get_clean_factor_and_forward_returns(
            factor=feature_series,
            prices=prices_df,
            quantiles=quantiles,
        )
        return factor_data
    except Exception as e:
        LOG.warning("alphalens tearsheet failed: %s", e)
        return None


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    # Quick sanity: fake feature + fake returns
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-01", periods=200, freq="B")
    feat = pd.Series(np.sin(np.arange(200) * 0.1), index=idx, name="sine")
    rets = pd.Series(feat.values * 0.01 + rng.normal(0, 0.005, 200), index=idx, name="r")
    print("enabled:", ENABLED)
    res = validate_feature(feat, rets)
    print("smoke result:", res)
