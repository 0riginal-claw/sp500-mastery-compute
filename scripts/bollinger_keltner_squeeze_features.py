"""
bollinger_keltner_squeeze_features.py — TTM Squeeze (Carter 2008).

Wave V-1 (LOW-cost, no new deps). Wired 2026-05-17.

# NO-LOOKAHEAD AUDIT
# ------------------
# Bollinger Bands and Keltner Channels use rolling means/stdevs over a 20-bar
# window ENDING at bar t.  We compute on the full series then .shift(1) before
# assignment so row t uses only data through t-1.
#
# Squeeze ON  := (BB upper < KC upper) AND (BB lower > KC lower)
#   i.e. Bollinger band sits entirely INSIDE the Keltner channel = vol contraction.
# Squeeze release : first bar that flips from squeeze_on -> off.
# squeeze_momentum_proxy : 12-bar linear-regression slope on (close - midpoint).
#
# Pure pandas/numpy.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SQUEEZE_FEATURE_NAMES: list[str] = [
    "squeeze_on_indicator",
    "days_in_squeeze",
    "squeeze_release_indicator",
    "squeeze_momentum_proxy",
]


def _find(df: pd.DataFrame, options: tuple[str, ...]) -> str | None:
    for c in options:
        if c in df.columns:
            return c
    return None


def _zero_fill(df: pd.DataFrame) -> pd.DataFrame:
    for col in SQUEEZE_FEATURE_NAMES:
        if col not in df.columns:
            if col in ("squeeze_on_indicator", "squeeze_release_indicator"):
                df[col] = np.int8(0)
            elif col == "days_in_squeeze":
                df[col] = 0
            else:
                df[col] = 0.0
    return df


def add_bollinger_keltner_squeeze_features(
    df: pd.DataFrame,
    ticker: str | None = None,
    bb_window: int = 20,
    bb_sigma: float = 2.0,
    kc_window: int = 20,
    kc_mult: float = 1.5,
) -> pd.DataFrame:
    """Append 4 TTM-squeeze features. Idempotent + graceful."""
    if df is None or len(df) == 0:
        return df

    if all(c in df.columns for c in SQUEEZE_FEATURE_NAMES):
        return df

    close_col = _find(df, ("close", "Close", "adj_close", "Adj Close"))
    high_col = _find(df, ("high", "High"))
    low_col = _find(df, ("low", "Low"))
    if close_col is None or high_col is None or low_col is None:
        logger.warning("[squeeze] %s: missing HLC — zeroing", ticker)
        return _zero_fill(df)

    try:
        close = df[close_col].astype(float)
        high = df[high_col].astype(float)
        low = df[low_col].astype(float)

        # --- Bollinger ---
        mid = close.rolling(bb_window, min_periods=bb_window // 2).mean()
        std = close.rolling(bb_window, min_periods=bb_window // 2).std()
        bb_up = mid + bb_sigma * std
        bb_lo = mid - bb_sigma * std

        # --- Keltner (typical ATR formulation) ---
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(kc_window, min_periods=kc_window // 2).mean()
        kc_up = mid + kc_mult * atr
        kc_lo = mid - kc_mult * atr

        # Squeeze: Bollinger inside Keltner
        squeeze = ((bb_up < kc_up) & (bb_lo > kc_lo)).astype(np.int8)

        # days_in_squeeze running counter
        days_in = pd.Series(np.zeros(len(df), dtype=np.int32), index=df.index)
        counter = 0
        vals = squeeze.values
        for i in range(len(df)):
            counter = counter + 1 if vals[i] == 1 else 0
            days_in.iat[i] = counter

        # release: 1 on first bar squeeze turns off (i.e. prev=1, curr=0)
        release = ((squeeze.shift(1) == 1) & (squeeze == 0)).astype(np.int8)

        # momentum proxy: 12-bar slope of (close - mid)
        delta = (close - mid)
        # Simple rolling linear-regression slope via cov / var
        x = pd.Series(np.arange(len(df)), index=df.index, dtype=float)
        roll_window = 12
        cov = delta.rolling(roll_window).cov(x)
        var = x.rolling(roll_window).var().replace(0, np.nan)
        slope = (cov / var).fillna(0.0)

        out = df.copy()
        out["squeeze_on_indicator"] = squeeze.shift(1).fillna(0).astype(np.int8).values
        out["days_in_squeeze"] = days_in.shift(1).fillna(0).astype(np.int32).values
        out["squeeze_release_indicator"] = release.shift(1).fillna(0).astype(np.int8).values
        out["squeeze_momentum_proxy"] = slope.shift(1).fillna(0.0).values

        logger.info(
            "[squeeze] %s: added 4 cols (squeeze rows=%d)",
            ticker, int(out["squeeze_on_indicator"].sum()),
        )
        return out
    except Exception as exc:
        logger.warning("[squeeze] %s: computation failed (%s) — zeroing", ticker, exc)
        return _zero_fill(df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=200, freq="B")
    rng_arr = np.random.default_rng(0).normal(0, 0.01, len(idx))
    close = 100 * np.exp(np.cumsum(rng_arr))
    demo = pd.DataFrame({
        "close": close, "high": close * 1.012, "low": close * 0.988,
    }, index=idx)
    out = add_bollinger_keltner_squeeze_features(demo, "DEMO")
    print(out[SQUEEZE_FEATURE_NAMES].tail(8).to_string())
