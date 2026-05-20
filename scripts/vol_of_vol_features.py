"""
vol_of_vol_features.py — Vol-of-vol (realized variance of realized variance).

Wave V-1 (LOW-cost, no new deps). Wired 2026-05-17.

# NO-LOOKAHEAD AUDIT
# ------------------
# All rolling outputs are .shift(1)-ed before assignment, so the value at row t
# reflects ONLY past data through bar t-1.  This matches the v10 convention used
# by every other wired feature module.
#
# Theory: Carr-Wu VRP literature shows vol-of-vol is a priced risk factor;
# high vov empirically predicts regime instability and is orthogonal to vol level.
# Implementation: rolling 20-day std-dev of the rolling 20-day realized vol of
# log-returns (and the 60/60 variant), plus a 252-day z-score for normalization.
#
# Pure pandas/numpy — no external dependencies.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VOL_OF_VOL_FEATURE_NAMES: list[str] = [
    "vov_20_20",
    "vov_60_60",
    "vov_zscore_252",
]


def _find_close(df: pd.DataFrame) -> str | None:
    for c in ("close", "Close", "adj_close", "Adj Close"):
        if c in df.columns:
            return c
    return None


def _zero_fill(df: pd.DataFrame) -> pd.DataFrame:
    for col in VOL_OF_VOL_FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0.0
    return df


def add_vol_of_vol_features(
    df: pd.DataFrame,
    ticker: str | None = None,
    inner: int = 20,
    outer: int = 20,
) -> pd.DataFrame:
    """Append 3 vol-of-vol features. Idempotent + graceful zero-fill on failure."""
    if df is None or len(df) == 0:
        return df

    # Idempotency guard
    if all(c in df.columns for c in VOL_OF_VOL_FEATURE_NAMES):
        return df

    close_col = _find_close(df)
    if close_col is None:
        logger.warning("[vov] close column not found for %s — zeroing", ticker)
        return _zero_fill(df)

    try:
        close = df[close_col].astype(float)
        log_ret = np.log(close / close.shift(1))

        # inner-window realized vol (rolling stdev of log-returns)
        rv_inner = log_ret.rolling(inner, min_periods=max(5, inner // 2)).std()
        # outer-window vol-of-vol = stdev of rv_inner
        vov_20 = rv_inner.rolling(outer, min_periods=max(5, outer // 2)).std()

        # 60/60 variant
        rv_60 = log_ret.rolling(60, min_periods=20).std()
        vov_60 = rv_60.rolling(60, min_periods=20).std()

        # 252-day z-score of vov_20_20
        roll_mean = vov_20.rolling(252, min_periods=60).mean()
        roll_std = vov_20.rolling(252, min_periods=60).std().replace(0, np.nan)
        z252 = ((vov_20 - roll_mean) / roll_std)

        # .shift(1) so row t uses only data through t-1
        out = df.copy()
        out["vov_20_20"] = vov_20.shift(1).fillna(0.0).values
        out["vov_60_60"] = vov_60.shift(1).fillna(0.0).values
        out["vov_zscore_252"] = z252.shift(1).fillna(0.0).values

        logger.info(
            "[vov] %s: added 3 cols (non-zero rows vov_20_20=%d)",
            ticker, int((out["vov_20_20"] != 0).sum()),
        )
        return out
    except Exception as exc:
        logger.warning("[vov] %s: computation failed (%s) — zeroing", ticker, exc)
        return _zero_fill(df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=400, freq="B")
    rng = np.random.default_rng(0)
    demo = pd.DataFrame({"close": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))}, index=idx)
    out = add_vol_of_vol_features(demo, "DEMO")
    print(out[VOL_OF_VOL_FEATURE_NAMES].tail(5).to_string())
