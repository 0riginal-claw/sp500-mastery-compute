"""
vol_target_sizing_features.py — vol-target / forecast-vol position-sizing signal.

Wave V-1 (LOW-cost, no new deps). Wired 2026-05-17.

# NO-LOOKAHEAD AUDIT
# ------------------
# Consumes garch11_cond_vol_1d which is already .shift(1)-safe inside its own
# module (the GARCH conditional variance at row t reflects the one-step-ahead
# forecast made AT t-1).  We additionally .shift(1) before assignment so even
# if the upstream module is ever changed, this layer remains safe.
#
# If garch11_cond_vol_1d is missing or zero everywhere, we fall back to a pure
# realized-vol forecast (21-day annualized stdev) so the feature still has
# signal without the GARCH dep.
#
# Pure pandas/numpy.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VOL_TARGET_FEATURE_NAMES: list[str] = [
    "vol_target_ratio",
    "vol_target_clipped_5x",
]

# Default annualized target vol (15 % is industry-standard for vol-targeting)
_TARGET_ANNUAL_VOL = 0.15
# Floor on forecast vol to avoid divide-by-tiny (5 bps daily ≈ 0.8 % annualized)
_VOL_FLOOR = 1e-4


def _find_close(df: pd.DataFrame) -> str | None:
    for c in ("close", "Close", "adj_close", "Adj Close"):
        if c in df.columns:
            return c
    return None


def _zero_fill(df: pd.DataFrame) -> pd.DataFrame:
    for col in VOL_TARGET_FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 1.0  # 1.0 = neutral sizing (don't scale up or down)
    return df


def add_vol_target_sizing_features(
    df: pd.DataFrame,
    ticker: str | None = None,
    target_annual_vol: float = _TARGET_ANNUAL_VOL,
) -> pd.DataFrame:
    """Append 2 vol-target sizing features. Idempotent + graceful."""
    if df is None or len(df) == 0:
        return df

    if all(c in df.columns for c in VOL_TARGET_FEATURE_NAMES):
        return df

    try:
        # Prefer upstream GARCH conditional vol forecast (already in vol-points,
        # i.e. annualized in the garch_11 module's convention).
        forecast_vol = None
        if "garch11_cond_vol_1d" in df.columns:
            garch_series = df["garch11_cond_vol_1d"].astype(float)
            if (garch_series.abs() > 1e-12).any():
                # garch_11 outputs are already in annualized vol-points but
                # divided by 100 — see garch_11_cond_vol_features.py.  Use as-is.
                forecast_vol = garch_series.copy()

        if forecast_vol is None:
            # Fallback: 21-day RV annualized
            close_col = _find_close(df)
            if close_col is None:
                logger.warning("[vol_target] %s: no close/garch — zeroing", ticker)
                return _zero_fill(df)
            close = df[close_col].astype(float)
            log_ret = np.log(close / close.shift(1))
            forecast_vol = log_ret.rolling(21, min_periods=10).std() * np.sqrt(252)
            forecast_vol = forecast_vol.fillna(method="ffill").fillna(target_annual_vol)

        # Compute target / forecast ratio with floor protection
        safe_vol = forecast_vol.clip(lower=_VOL_FLOOR)
        ratio = target_annual_vol / safe_vol
        # Clip to 5x (industry-standard leverage cap)
        clipped = ratio.clip(upper=5.0, lower=0.0)

        out = df.copy()
        out["vol_target_ratio"] = ratio.shift(1).fillna(1.0).values
        out["vol_target_clipped_5x"] = clipped.shift(1).fillna(1.0).values

        logger.info(
            "[vol_target] %s: added 2 cols (mean ratio=%.3f)",
            ticker, float(out["vol_target_ratio"].mean()),
        )
        return out
    except Exception as exc:
        logger.warning("[vol_target] %s: computation failed (%s) — neutral fill", ticker, exc)
        return _zero_fill(df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=200, freq="B")
    rng = np.random.default_rng(0)
    demo = pd.DataFrame({"close": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))}, index=idx)
    out = add_vol_target_sizing_features(demo, "DEMO")
    print(out[VOL_TARGET_FEATURE_NAMES].tail(5).to_string())
