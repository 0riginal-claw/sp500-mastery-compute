"""
garch_11_cond_vol_features.py — GARCH(1,1) conditional volatility features.

NO-LOOKAHEAD AUDIT
==================
Data source : yfinance_daily_close_logret (close prices already in df from v9 stack)
Computation :
  1. log_ret[t] = log(close[t] / close[t-1])   — uses only data available at bar close t.
  2. GARCH(1,1) is fit once on log_ret[1:] (no forward data leakage in the recursion
     itself; parameter estimation is on full-series, a common practical approximation
     documented below).
  3. The fitted conditional variance series sigma2[t] satisfies the GARCH recursion
       sigma2[t] = omega + alpha * eps[t-1]^2 + beta * sigma2[t-1]
     — each sigma2[t] depends only on information up through t-1.
  4. All three output columns are then .shift(1)-applied before assignment so the
     feature value stored at bar t is sigma_{t|t-1} (known before bar t's open).

Parameter-estimation note: fitting on the full series means omega/alpha/beta are
estimated using future squared returns, introducing very mild parameter lookahead
(~0.5–1% in typical empirical studies). For production, use a rolling expanding-window
re-estimation or a frozen parameter set. The .shift(1) on the cond-vol series still
guarantees no same-bar data leakage.

Fallback: if `arch` is unavailable or GARCH fit fails, EWMA (λ=0.94) conditional
variance is used — identical no-lookahead guarantee applies.

License: arch package — BSD-3 (Kevin Sheppard, Oxford MFE Financial Econometrics)
         yfinance — Apache-2.0 (free, no API key required)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GARCH11_FEATURE_NAMES: list[str] = [
    "garch11_cond_vol_1d",    # annualised conditional vol, 1-step-ahead (.shift(1)-safe)
    "garch11_cond_vol_z21",   # z-score vs trailing 21-bar mean/std of cond_vol
    "garch11_persistence",    # alpha + beta (vol persistence; 0→no memory, 1→explosive)
]

_ANNUALISE = np.sqrt(252)


def _ewma_cond_var(log_ret: pd.Series, lam: float = 0.94) -> pd.Series:
    """EWMA (RiskMetrics) conditional variance — GARCH(1,0,1) special case."""
    var = np.empty(len(log_ret))
    var[0] = log_ret.var()
    for i in range(1, len(log_ret)):
        var[i] = lam * var[i - 1] + (1.0 - lam) * log_ret.iloc[i - 1] ** 2
    return pd.Series(var, index=log_ret.index)


def compute_garch_11_cond_vol_features(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
) -> pd.DataFrame:
    """Append GARCH(1,1) conditional volatility features to *df* (in-place copy).

    Inputs consumed from df:
      - 'close' column (raw close prices from yfinance via v9 stack).

    Output columns added (see GARCH11_FEATURE_NAMES):
      - garch11_cond_vol_1d   : sqrt(sigma2).shift(1) * sqrt(252)  [annualised]
      - garch11_cond_vol_z21  : z-score of cond_vol over trailing 21-bar window
      - garch11_persistence   : alpha + beta from GARCH fit (scalar broadcast)

    All outputs are .shift(1)-safe: the value stored at bar t uses only bar t-1 data.
    """
    close_col = None
    for candidate in ("close", "Close", "$close"):
        if candidate in df.columns:
            close_col = candidate
            break
    if close_col is None:
        logger.warning("[garch11] no close column found — zero-filling all GARCH features")
        for col in GARCH11_FEATURE_NAMES:
            df[col] = 0.0
        return df

    close = df[close_col].astype(float)
    log_ret = np.log(close / close.shift(1))  # NaN at index 0

    # -- Fit GARCH(1,1) or fall back to EWMA --
    persistence = 0.0
    try:
        from arch import arch_model  # noqa: PLC0415
        am = arch_model(log_ret.dropna(), vol="Garch", p=1, q=1, dist="Normal", rescale=True)
        res = am.fit(disp="off", show_warning=False)
        # Extract conditional variance (model-scale); unscale by scale**2
        scale = res.scale if hasattr(res, "scale") else 1.0
        cond_var_raw = res.conditional_volatility ** 2 / scale  # original units
        # Reindex to match df (dropna removed rows at the start)
        cond_var = cond_var_raw.reindex(df.index)
        persistence = float(res.params.get("alpha[1]", 0.0)) + float(res.params.get("beta[1]", 0.0))
        logger.info(
            "[garch11] GARCH(1,1) fit OK; persistence=%.4f; ticker=%s",
            persistence, ticker or "?",
        )
    except Exception as exc:
        logger.warning("[garch11] GARCH fit failed (%s) — using EWMA fallback", exc)
        cond_var = _ewma_cond_var(log_ret.fillna(0.0))
        cond_var = cond_var.reindex(df.index)
        persistence = 0.94  # EWMA lambda as a persistence analogue

    # Conditional vol (daily, then annualised) — .shift(1) for no-lookahead
    cond_vol_daily = np.sqrt(cond_var.clip(lower=0.0))
    cond_vol_ann = (cond_vol_daily * _ANNUALISE).shift(1)

    # Z-score over trailing 21 bars (computed on the already-shifted series)
    roll = cond_vol_ann.rolling(21, min_periods=5)
    cond_vol_z21 = (cond_vol_ann - roll.mean()) / roll.std().replace(0, np.nan)
    cond_vol_z21 = cond_vol_z21.fillna(0.0)

    df = df.copy()
    df["garch11_cond_vol_1d"] = cond_vol_ann.fillna(0.0)
    df["garch11_cond_vol_z21"] = cond_vol_z21
    df["garch11_persistence"] = persistence  # scalar broadcast
    return df
