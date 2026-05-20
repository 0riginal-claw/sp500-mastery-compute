"""
rv_term_structure_features.py — Realized-vol term-structure slope (5d / 21d / 63d).

Wave V-1 (LOW-cost, no new deps). Wired 2026-05-17.

# NO-LOOKAHEAD AUDIT
# ------------------
# Rolling-window stdevs end at bar t. We .shift(1) the assembled features
# before assignment so the value at row t reflects only bars t-1, t-2, ….
#
# Distinct from VIX term-structure (which is implied vol slope across maturities).
# This module is REALIZED vol slope across timeframes for the *same* ticker.
#
# Features:
#   rv5_over_rv21          — short / mid RV ratio
#   rv5_over_rv63          — short / long RV ratio
#   rv_slope_252z          — 252-day z-score of (rv5 - rv63)
#   rv_backwardation_indicator — int8, 1 when rv5 > rv63 (stress regime)
#
# Pure pandas/numpy.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RV_TERM_FEATURE_NAMES: list[str] = [
    "rv5_over_rv21",
    "rv5_over_rv63",
    "rv_slope_252z",
    "rv_backwardation_indicator",
]


def _find_close(df: pd.DataFrame) -> str | None:
    for c in ("close", "Close", "adj_close", "Adj Close"):
        if c in df.columns:
            return c
    return None


def _zero_fill(df: pd.DataFrame) -> pd.DataFrame:
    for col in RV_TERM_FEATURE_NAMES:
        if col not in df.columns:
            if col == "rv_backwardation_indicator":
                df[col] = np.int8(0)
            elif col in ("rv5_over_rv21", "rv5_over_rv63"):
                df[col] = 1.0
            else:
                df[col] = 0.0
    return df


def add_rv_term_structure_features(
    df: pd.DataFrame,
    ticker: str | None = None,
) -> pd.DataFrame:
    """Append 4 RV term-structure features. Idempotent + graceful."""
    if df is None or len(df) == 0:
        return df

    if all(c in df.columns for c in RV_TERM_FEATURE_NAMES):
        return df

    close_col = _find_close(df)
    if close_col is None:
        logger.warning("[rv_term] %s: close column not found — zeroing", ticker)
        return _zero_fill(df)

    try:
        close = df[close_col].astype(float)
        log_ret = np.log(close / close.shift(1))

        rv5 = log_ret.rolling(5, min_periods=3).std() * np.sqrt(252)
        rv21 = log_ret.rolling(21, min_periods=10).std() * np.sqrt(252)
        rv63 = log_ret.rolling(63, min_periods=30).std() * np.sqrt(252)

        safe21 = rv21.replace(0, np.nan)
        safe63 = rv63.replace(0, np.nan)
        ratio_5_21 = (rv5 / safe21).fillna(1.0)
        ratio_5_63 = (rv5 / safe63).fillna(1.0)

        slope = rv5 - rv63
        roll_mean = slope.rolling(252, min_periods=60).mean()
        roll_std = slope.rolling(252, min_periods=60).std().replace(0, np.nan)
        slope_z = ((slope - roll_mean) / roll_std).fillna(0.0)

        backwardation = (rv5 > rv63).astype(np.int8)

        out = df.copy()
        out["rv5_over_rv21"] = ratio_5_21.shift(1).fillna(1.0).values
        out["rv5_over_rv63"] = ratio_5_63.shift(1).fillna(1.0).values
        out["rv_slope_252z"] = slope_z.shift(1).fillna(0.0).values
        out["rv_backwardation_indicator"] = (
            backwardation.shift(1).fillna(0).astype(np.int8).values
        )

        logger.info(
            "[rv_term] %s: added 4 cols (backwardation rows=%d)",
            ticker, int(out["rv_backwardation_indicator"].sum()),
        )
        return out
    except Exception as exc:
        logger.warning("[rv_term] %s: computation failed (%s) — zeroing", ticker, exc)
        return _zero_fill(df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=300, freq="B")
    rng = np.random.default_rng(0)
    demo = pd.DataFrame({"close": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))}, index=idx)
    out = add_rv_term_structure_features(demo, "DEMO")
    print(out[RV_TERM_FEATURE_NAMES].tail(5).to_string())
