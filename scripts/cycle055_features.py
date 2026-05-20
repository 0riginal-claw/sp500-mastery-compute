"""
cycle055_features.py — Wrapper for cycle055 VOLATILITY GATES (Wave Cycle, 2026-05-17).

The original cycle055 engine
(`research/archive/cycle055_volatility_gates_2026-05-05/volatility_gates.py`) is
a BAR-LEVEL gate library (18 binary masks: V01..V18). Each gate consumes an
intraday cache dict and returns a boolean array of which bars are eligible for
entry.

For DAILY ML feature consumption, this wrapper exposes 5 features representing
the AGGREGATE volatility-regime characterization the gates encode, computed
from daily OHLCV that v10 already loads. All .shift(1)-safe.

Features:
  - vg_atr_pct_14         : ATR(14)/Close — proxy for the "atr_room_for_target"
                            family (V01-V02). High = enough range to reach a
                            target ≥ 0.5% pct move.
  - vg_range_5d_pct       : 5-day mean daily range / Close — proxy for
                            range-expansion gates (V05-V06).
  - vg_vol_regime         : ternary 0/1/2 (LOW / NORMAL / HIGH) computed from
                            21-day realized vol percentile (the V11-V13 family
                            encodes this regime).
  - vg_in_normal_regime   : binary indicator vol_regime==1 (V11 pass-rate proxy).
  - vg_rvol_floor_ok      : binary {realized_vol_21d > floor(1e-4)} (V14 proxy).

The bar-level gate library is left untouched. This wrapper exposes only the
information a DAILY model can use without intraday cache access. If you want
true bar-level gate pass-rates (e.g., "fraction of yesterday's bars that passed
V07-or30_breakout"), wire the cycle055 cache_day dict separately and aggregate
per-day in this module — but that requires a much heavier 1-min pipeline and
isn't justified for a daily classifier.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CYCLE055_FEATURE_NAMES: list[str] = [
    "vg_atr_pct_14",
    "vg_range_5d_pct",
    "vg_vol_regime",
    "vg_in_normal_regime",
    "vg_rvol_floor_ok",
]

_ATR_WIN = 14
_RANGE_WIN = 5
_RV_WIN = 21
_RVOL_FLOOR = 1e-4


def _zero_fill(df: pd.DataFrame) -> pd.DataFrame:
    for col in CYCLE055_FEATURE_NAMES:
        if col not in df.columns:
            if col == "vg_vol_regime":
                df[col] = 1  # default NORMAL
            elif col in ("vg_in_normal_regime", "vg_rvol_floor_ok"):
                df[col] = 0
            else:
                df[col] = 0.0
    return df


def _classify_regime(rv: pd.Series) -> pd.Series:
    """Ternary regime: 0 = LOW (rv pct <= 0.25), 1 = NORMAL (0.25..0.75),
    2 = HIGH (> 0.75). Uses 252-day rolling rank."""
    pct = rv.rolling(252, min_periods=21).rank(pct=True)
    out = pd.Series(1, index=rv.index, dtype="int8")
    out[pct <= 0.25] = 0
    out[pct > 0.75] = 2
    out[rv.isna()] = 1
    return out


def add_cycle055_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Append 5 daily volatility-gate proxy features to df. Idempotent.

    Requires df to have 'close' and ideally 'high', 'low' columns. Without
    high/low, falls back to estimating range from |daily return|. Output is
    .shift(1)-safe — all rolling stats are shifted by 1 bar before assignment.
    """
    if df is None or len(df) == 0:
        return df
    if all(c in df.columns for c in CYCLE055_FEATURE_NAMES):
        return df
    if "close" not in df.columns:
        return _zero_fill(df)

    close = pd.to_numeric(df["close"], errors="coerce").astype(float)

    # ---- ATR(14) / Close ----
    if "high" in df.columns and "low" in df.columns:
        high = pd.to_numeric(df["high"], errors="coerce").astype(float)
        low = pd.to_numeric(df["low"], errors="coerce").astype(float)
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(_ATR_WIN, min_periods=2).mean()
        atr_pct = (atr / close.replace(0, np.nan)).fillna(0.0).clip(0.0, 0.5)
        daily_range = (high - low) / close.replace(0, np.nan)
    else:
        daily_ret = close.pct_change().abs()
        atr_pct = daily_ret.rolling(_ATR_WIN, min_periods=2).mean().fillna(0.0).clip(0.0, 0.5)
        daily_range = daily_ret

    range_5d = daily_range.rolling(_RANGE_WIN, min_periods=1).mean().fillna(0.0).clip(0.0, 0.5)

    # ---- Realized vol (21d) ----
    log_ret = np.log(close.replace(0, np.nan)).diff()
    rv = (log_ret.rolling(_RV_WIN, min_periods=5).std() * np.sqrt(252.0)).fillna(0.0)
    regime = _classify_regime(rv)

    # ---- Shift by 1 for no-lookahead ----
    if "vg_atr_pct_14" not in df.columns:
        df["vg_atr_pct_14"] = atr_pct.shift(1).fillna(0.0).values
    if "vg_range_5d_pct" not in df.columns:
        df["vg_range_5d_pct"] = range_5d.shift(1).fillna(0.0).values
    if "vg_vol_regime" not in df.columns:
        df["vg_vol_regime"] = regime.shift(1).fillna(1).astype(int).values
    if "vg_in_normal_regime" not in df.columns:
        df["vg_in_normal_regime"] = (
            (regime.shift(1).fillna(1) == 1).astype(int).values
        )
    if "vg_rvol_floor_ok" not in df.columns:
        df["vg_rvol_floor_ok"] = (rv.shift(1).fillna(0.0) > _RVOL_FLOOR).astype(int).values

    return df


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    tk = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    idx = pd.date_range(end=pd.Timestamp.utcnow().date(), periods=300, freq="B")
    rng = np.random.default_rng(42)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0001, 0.012, len(idx))))
    demo = pd.DataFrame(
        {
            "close": close,
            "high": close * (1 + np.abs(rng.normal(0, 0.005, len(idx)))),
            "low": close * (1 - np.abs(rng.normal(0, 0.005, len(idx)))),
        },
        index=idx,
    )
    out = add_cycle055_features(demo, tk)
    print(f"In cols: 3  Out cols: {out.shape[1]}")
    print(out[CYCLE055_FEATURE_NAMES].tail(5).to_string())
    print(f"\nvg_vol_regime distribution: {out['vg_vol_regime'].value_counts().to_dict()}")
