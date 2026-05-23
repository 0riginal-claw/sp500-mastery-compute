"""
qlib_alpha360_features.py — Pure-pandas port of Qlib Alpha360 feature set.

Source: microsoft/qlib (MIT License, https://github.com/microsoft/qlib)
Origin: qlib/contrib/data/loader.py  class Alpha360DL.get_feature_config()
Ported: native pandas/numpy — no qlib import required.

Alpha360 = 60 lagged values * 6 fields (open, high, low, close, vwap, volume)
         each normalized by the reference price/volume at bar t-1.
         Total = 360 features.

Naming convention: alpha360_<FIELD><LAG> where FIELD ∈ {OPEN, HIGH, LOW, CLOSE, VWAP, VOL}
and LAG ∈ {0..59}. Lag 0 = bar t-1 (shifted), Lag 59 = bar t-60.

SHIFT CONVENTION (Patch 3 aligned with sibling modules):
  Applies a final .shift(1) so at bar t, alpha360_*[t] uses data through t-1.
  LOOKAHEAD_STRATEGY = "shifted_1"
  ALREADY_SHIFTED = True

ENV-GATE: QLIB_ALPHA360_ENABLED=1 (default OFF). Sub-agent must enable explicitly.

Input df must have lowercase columns: open, high, low, close, volume.
VWAP is optional; if missing it falls back to (high+low+close)/3.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

LOOKAHEAD_STRATEGY = "shifted_1"
ALREADY_SHIFTED = True

# Default Alpha360 spec: 60-bar lookback × 6 fields
ALPHA360_WINDOW = 60
ALPHA360_FIELDS = ["OPEN", "HIGH", "LOW", "CLOSE", "VWAP", "VOL"]


def alpha360_feature_names() -> list[str]:
    names = []
    for field in ALPHA360_FIELDS:
        for lag in range(ALPHA360_WINDOW):
            names.append(f"alpha360_{field}{lag}")
    return names


ALPHA360_FEATURE_NAMES: list[str] = alpha360_feature_names()
ALPHA360_FEATURE_COUNT: int = len(ALPHA360_FEATURE_NAMES)  # 360


def add_alpha360_features(
    df: pd.DataFrame,
    window: int = ALPHA360_WINDOW,
) -> pd.DataFrame:
    """Compute Alpha360 features.

    Returns new DataFrame with original columns + alpha360_* features.
    All features are .shift(1)-safe.
    """
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        # Zero-fill manifest if inputs missing
        out = df.copy()
        for name in ALPHA360_FEATURE_NAMES:
            out[name] = 0.0
        return out

    out = df.copy()

    # Reference (denominator) is the immediately-previous bar's close / volume
    # Then lag-k normalized = (field[t-1-k] / ref[t-1]).
    o = df["open"]
    h = df["high"]
    l = df["low"]
    c = df["close"]
    v = df["volume"]
    vwap = df["vwap"] if "vwap" in df.columns else (h + l + c) / 3.0

    ref_price = c.shift(1).replace(0, np.nan)
    ref_vol = v.shift(1).replace(0, np.nan)

    fields = {
        "OPEN": (o, ref_price),
        "HIGH": (h, ref_price),
        "LOW": (l, ref_price),
        "CLOSE": (c, ref_price),
        "VWAP": (vwap, ref_price),
        "VOL": (v, ref_vol),
    }

    new_cols = {}
    for field_name, (series, ref) in fields.items():
        # SHIFT(1) the series to keep no-lookahead
        s = series.shift(1)
        for lag in range(window):
            col_name = f"alpha360_{field_name}{lag}"
            new_cols[col_name] = s.shift(lag) / ref

    # Concat all 360 columns at once to avoid fragmentation
    alpha_df = pd.DataFrame(new_cols, index=out.index)
    out = pd.concat([out, alpha_df], axis=1)

    return out


def compute_qlib_alpha360_features(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
) -> pd.DataFrame:
    """Pipeline-API alias for backtest_xgb_v10 chain."""
    return add_alpha360_features(df)


if __name__ == "__main__":
    np.random.seed(0)
    n = 200
    test_df = pd.DataFrame({
        "open": np.random.uniform(100, 110, n).cumsum() / 50,
        "high": np.random.uniform(105, 115, n).cumsum() / 50,
        "low": np.random.uniform(95, 105, n).cumsum() / 50,
        "close": np.random.uniform(100, 110, n).cumsum() / 50,
        "volume": np.random.uniform(1e6, 2e6, n),
    })
    test_df["high"] = test_df[["open", "close", "high"]].max(axis=1)
    test_df["low"] = test_df[["open", "close", "low"]].min(axis=1)
    result = compute_qlib_alpha360_features(test_df, ticker="SMOKE")
    new_cols = [c for c in result.columns if c.startswith("alpha360_")]
    print(f"qlib_alpha360 smoke: {len(new_cols)} cols added (manifest={ALPHA360_FEATURE_COUNT})")
    last = result[new_cols].iloc[-1]
    print(f"rows={len(result)}, last-row finite check: {last.notna().sum()}/{len(new_cols)}")
    first = result[new_cols].iloc[0]
    print(f"first-row NaN check (shift sanity): {first.isna().sum()}/{len(new_cols)} should be all NaN")
