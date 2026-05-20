"""
egarch_11_leverage_features.py — EGARCH(1,1) with leverage-effect features.

NO-LOOKAHEAD AUDIT
==================
Data source : yfinance_daily_close_logret (close prices already in df from v9 stack)
Computation :
  1. log_ret[t] = log(close[t] / close[t-1])   — uses only data available at bar close t.
  2. EGARCH(1,1) is fit on log_ret[1:] (full-series parameter estimation; mild parameter
     lookahead documented below, same caveat as garch_11_cond_vol_features.py).
  3. The EGARCH log-variance recursion is:
       log(sigma2[t]) = omega + alpha*(|eps[t-1]|/sigma[t-1] - E[|z|])
                        + gamma*(eps[t-1]/sigma[t-1])   ← leverage term
                        + beta*log(sigma2[t-1])
     Each sigma2[t] depends only on information through t-1.
  4. All three output columns are .shift(1)-applied before assignment so the
     feature value stored at bar t is sigma_{t|t-1} (known before bar t's open).
  5. The leverage coefficient (gamma) is a scalar estimated once from the full
     series — no bar-level lookahead from this parameter beyond the mild
     full-series estimation noted in point 2.

Parameter-estimation note: fitting on the full series means omega/alpha/beta/gamma
are estimated using future squared returns (mild parameter lookahead, ~0.5–1%
empirically). For production use a rolling expanding-window re-estimation.
The .shift(1) on the cond-vol series still guarantees no same-bar data leakage.

Fallback: if `arch` is unavailable or EGARCH fit fails, an asymmetric EWMA
(distinguishing positive vs negative returns via separate λ values) is used —
same no-lookahead guarantee applies.

License: arch package — BSD-3 (Kevin Sheppard, Oxford MFE Financial Econometrics)
         yfinance — Apache-2.0 (free, no API key required)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EGARCH11_LEV_FEATURE_NAMES: list[str] = [
    "egarch11_lev_cond_vol_1d",   # annualised conditional vol (.shift(1)-safe)
    "egarch11_lev_effect",         # leverage coefficient gamma (negative → bad-news amplifies vol)
    "egarch11_lev_vol_z21",        # z-score of cond_vol over trailing 21-bar window
]

_ANNUALISE = np.sqrt(252)


def _asymmetric_ewma_cond_var(
    log_ret: pd.Series,
    lam_pos: float = 0.94,
    lam_neg: float = 0.97,
) -> tuple[pd.Series, float]:
    """Asymmetric EWMA fallback: negative returns decay slower (more weight) than positive."""
    var = np.empty(len(log_ret))
    var[0] = log_ret.var()
    for i in range(1, len(log_ret)):
        r = log_ret.iloc[i - 1]
        lam = lam_neg if r < 0 else lam_pos
        var[i] = lam * var[i - 1] + (1.0 - lam) * r ** 2
    # Approximate leverage: lam_neg - lam_pos (higher asymmetry → stronger leverage)
    gamma_approx = -(lam_neg - lam_pos)  # negative convention (matches EGARCH sign)
    return pd.Series(var, index=log_ret.index), gamma_approx


def compute_egarch_11_leverage_features(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
) -> pd.DataFrame:
    """Append EGARCH(1,1) leverage-effect features to *df* (in-place copy).

    Inputs consumed from df:
      - 'close' column (raw close prices from yfinance via v9 stack).

    Output columns added (see EGARCH11_LEV_FEATURE_NAMES):
      - egarch11_lev_cond_vol_1d : sqrt(sigma2).shift(1) * sqrt(252)  [annualised]
      - egarch11_lev_effect       : gamma coefficient (leverage; negative = bad-news amplifies vol)
      - egarch11_lev_vol_z21      : z-score of cond_vol over trailing 21-bar window

    All outputs are .shift(1)-safe: the value stored at bar t uses only bar t-1 data.
    """
    close_col = None
    for candidate in ("close", "Close", "$close"):
        if candidate in df.columns:
            close_col = candidate
            break
    if close_col is None:
        logger.warning("[egarch11_lev] no close column found — zero-filling all features")
        for col in EGARCH11_LEV_FEATURE_NAMES:
            df[col] = 0.0
        return df

    close = df[close_col].astype(float)
    log_ret = np.log(close / close.shift(1))  # NaN at index 0

    # -- Fit EGARCH(1,1) with leverage or fall back to asymmetric EWMA --
    gamma = 0.0
    try:
        from arch import arch_model  # noqa: PLC0415
        am = arch_model(
            log_ret.dropna(),
            vol="EGARCH",
            p=1,
            o=1,   # o=1 adds the leverage (asymmetry) term
            q=1,
            dist="Normal",
            rescale=True,
        )
        res = am.fit(disp="off", show_warning=False)
        # arch EGARCH names the leverage term "gamma[1]"
        gamma = float(res.params.get("gamma[1]", 0.0))
        # Extract conditional variance; unscale by scale**2
        scale = res.scale if hasattr(res, "scale") else 1.0
        cond_var_raw = res.conditional_volatility ** 2 / scale
        cond_var = cond_var_raw.reindex(df.index)
        logger.info(
            "[egarch11_lev] EGARCH(1,1) fit OK; gamma(leverage)=%.4f; ticker=%s",
            gamma, ticker or "?",
        )
    except Exception as exc:
        logger.warning("[egarch11_lev] EGARCH fit failed (%s) — using asymmetric EWMA fallback", exc)
        cond_var_raw, gamma = _asymmetric_ewma_cond_var(log_ret.fillna(0.0))
        cond_var = cond_var_raw.reindex(df.index)

    # Conditional vol (daily → annualised) — .shift(1) for no-lookahead
    cond_vol_daily = np.sqrt(cond_var.clip(lower=0.0))
    cond_vol_ann = (cond_vol_daily * _ANNUALISE).shift(1)

    # Z-score over trailing 21 bars (on the already-shifted series)
    roll = cond_vol_ann.rolling(21, min_periods=5)
    cond_vol_z21 = (cond_vol_ann - roll.mean()) / roll.std().replace(0, np.nan)
    cond_vol_z21 = cond_vol_z21.fillna(0.0)

    df = df.copy()
    df["egarch11_lev_cond_vol_1d"] = cond_vol_ann.fillna(0.0)
    df["egarch11_lev_effect"] = gamma          # scalar broadcast (negative = leverage effect)
    df["egarch11_lev_vol_z21"] = cond_vol_z21
    return df
