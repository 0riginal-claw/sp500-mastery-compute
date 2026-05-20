"""
vol_breakout_nr_features.py — NR4 / NR7 / WR7 vol-cluster patterns (Crabel 1990).

Wave V-1 (LOW-cost, no new deps). Wired 2026-05-17.

# NO-LOOKAHEAD AUDIT
# ------------------
# All five features rely on (high - low) range over a rolling window ending at
# bar t.  We compute on the full series then .shift(1) before assignment so the
# value at row t depends ONLY on bars t-1, t-2, … (never bar t).
#
# Patterns:
#   NR4 = today's range is the narrowest of the last 4 days (Crabel breakout)
#   NR7 = narrowest of 7
#   WR7 = widest of 7
#   days_since_nr7 = bars since the most recent NR7 (capped at 252)
#   range_pct_of_atr20 = today's range / 20-day ATR (regime-normalized)
#
# Pure pandas/numpy.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VOL_BREAKOUT_FEATURE_NAMES: list[str] = [
    "nr4_indicator",
    "nr7_indicator",
    "wr7_indicator",
    "days_since_nr7",
    "range_pct_of_atr20",
]


def _find(df: pd.DataFrame, options: tuple[str, ...]) -> str | None:
    for c in options:
        if c in df.columns:
            return c
    return None


def _zero_fill(df: pd.DataFrame) -> pd.DataFrame:
    for col in VOL_BREAKOUT_FEATURE_NAMES:
        if col not in df.columns:
            if col == "days_since_nr7":
                df[col] = 252  # max
            elif col == "range_pct_of_atr20":
                df[col] = 1.0
            else:
                df[col] = np.int8(0)
    return df


def add_vol_breakout_nr_features(
    df: pd.DataFrame,
    ticker: str | None = None,
) -> pd.DataFrame:
    """Append 5 NR4/NR7/WR7 + days-since features. Idempotent + graceful."""
    if df is None or len(df) == 0:
        return df

    if all(c in df.columns for c in VOL_BREAKOUT_FEATURE_NAMES):
        return df

    high_col = _find(df, ("high", "High"))
    low_col = _find(df, ("low", "Low"))
    if high_col is None or low_col is None:
        logger.warning("[vol_breakout] %s: high/low columns missing — zeroing", ticker)
        return _zero_fill(df)

    try:
        high = df[high_col].astype(float)
        low = df[low_col].astype(float)
        rng = (high - low).clip(lower=0)

        rng4_min = rng.rolling(4, min_periods=4).min()
        rng7_min = rng.rolling(7, min_periods=7).min()
        rng7_max = rng.rolling(7, min_periods=7).max()

        nr4 = (rng == rng4_min).astype(np.int8)
        nr7 = (rng == rng7_min).astype(np.int8)
        wr7 = (rng == rng7_max).astype(np.int8)

        # days_since_nr7: counter resetting to 0 on each NR7, else +1
        days_since = pd.Series(np.zeros(len(df), dtype=np.int32), index=df.index)
        counter = 252
        nr7_vals = nr7.values
        for i in range(len(df)):
            if nr7_vals[i] == 1:
                counter = 0
            else:
                counter = min(counter + 1, 252)
            days_since.iat[i] = counter

        # range as % of 20-day ATR (close diff proxy)
        close_col = _find(df, ("close", "Close"))
        if close_col is None:
            atr20 = rng.rolling(20, min_periods=10).mean()
        else:
            close = df[close_col].astype(float)
            prev_close = close.shift(1)
            true_range = pd.concat([
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ], axis=1).max(axis=1)
            atr20 = true_range.rolling(20, min_periods=10).mean()
        atr20_safe = atr20.replace(0, np.nan)
        range_pct = (rng / atr20_safe).fillna(1.0)

        out = df.copy()
        out["nr4_indicator"] = nr4.shift(1).fillna(0).astype(np.int8).values
        out["nr7_indicator"] = nr7.shift(1).fillna(0).astype(np.int8).values
        out["wr7_indicator"] = wr7.shift(1).fillna(0).astype(np.int8).values
        out["days_since_nr7"] = days_since.shift(1).fillna(252).astype(np.int32).values
        out["range_pct_of_atr20"] = range_pct.shift(1).fillna(1.0).values

        logger.info(
            "[vol_breakout] %s: added 5 cols (nr7 hits=%d)",
            ticker, int(out["nr7_indicator"].sum()),
        )
        return out
    except Exception as exc:
        logger.warning("[vol_breakout] %s: computation failed (%s) — zeroing", ticker, exc)
        return _zero_fill(df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=100, freq="B")
    rng_arr = np.random.default_rng(0).normal(0, 0.01, len(idx))
    close = 100 * np.exp(np.cumsum(rng_arr))
    demo = pd.DataFrame({
        "close": close,
        "high": close * 1.01,
        "low": close * 0.99,
    }, index=idx)
    out = add_vol_breakout_nr_features(demo, "DEMO")
    print(out[VOL_BREAKOUT_FEATURE_NAMES].tail(8).to_string())
