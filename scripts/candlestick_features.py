"""
candlestick_features.py — TA-Lib CDL* candlestick pattern features for v10/Mythos.

Covers BUCKET 31 + BUCKET 32 of the 2026-05-18 research mission. TA-Lib exposes
61 CDL functions (CDL2CROWS through CDLXSIDEGAP3METHODS). Each returns an int8
array per bar: +100 = bullish detection, -100 = bearish detection, 0 = no
pattern. We emit per-bar features as the raw int + a rolling-N flag count.

Output columns per ticker per bar:
    cdl_<name>          (raw int -100/0/+100)
    cdl_<name>_pos5     (count of +100 in last 5 bars)
    cdl_<name>_neg5     (count of -100 in last 5 bars)

Plus aggregate columns:
    cdl_bullish_count   (count of +100 patterns triggered THIS bar)
    cdl_bearish_count   (count of -100 patterns triggered THIS bar)
    cdl_net_signal      (bullish_count - bearish_count)
    cdl_bullish_density_5  (rolling 5-bar mean of bullish_count)
    cdl_bearish_density_5  (rolling 5-bar mean of bearish_count)

Usage:
    from candlestick_features import add_candlestick_features
    df = add_candlestick_features(df)   # df has columns: open, high, low, close

Idempotent: if columns already exist they are overwritten.
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

try:
    import talib
    _HAVE_TALIB = True
except ImportError:
    _HAVE_TALIB = False
    log.warning("talib not installed — candlestick features will return empty cols")


def _cdl_names() -> List[str]:
    """Return sorted list of TA-Lib CDL function names."""
    if not _HAVE_TALIB:
        return []
    return sorted(f for f in dir(talib) if f.startswith("CDL"))


def add_candlestick_features(
    df: pd.DataFrame,
    rolling_window: int = 5,
    include_rolling: bool = True,
) -> pd.DataFrame:
    """Add TA-Lib candlestick pattern features in place.

    Args:
        df: DataFrame with columns 'open', 'high', 'low', 'close' (any case).
            Must be sorted by time ascending.
        rolling_window: window for the rolling pos/neg counts (default 5).
        include_rolling: whether to emit per-pattern rolling counts (61 patterns
            × 2 dirs = 122 extra cols). If you want to keep the matrix narrow,
            set False and only the raw + aggregate cols are added.

    Returns:
        Same df with new columns appended. Original df is mutated.
    """
    if not _HAVE_TALIB:
        log.warning("add_candlestick_features: talib missing — no-op")
        return df

    cols_lower = {c.lower(): c for c in df.columns}
    try:
        o = df[cols_lower["open"]].astype(float).values
        h = df[cols_lower["high"]].astype(float).values
        l = df[cols_lower["low"]].astype(float).values
        c = df[cols_lower["close"]].astype(float).values
    except KeyError as e:
        log.error(f"add_candlestick_features: missing required column {e}")
        return df

    names = _cdl_names()
    bullish_count = np.zeros(len(df), dtype=np.int16)
    bearish_count = np.zeros(len(df), dtype=np.int16)

    for name in names:
        fn = getattr(talib, name)
        try:
            res = fn(o, h, l, c).astype(np.int16)  # -100, 0, +100
        except Exception as e:
            log.debug(f"{name} failed: {e}")
            continue
        col_raw = f"cdl_{name.lower()}"
        df[col_raw] = res
        bullish_count += (res > 0).astype(np.int16)
        bearish_count += (res < 0).astype(np.int16)
        if include_rolling:
            pos_series = pd.Series((res > 0).astype(np.int8), index=df.index)
            neg_series = pd.Series((res < 0).astype(np.int8), index=df.index)
            df[f"{col_raw}_pos{rolling_window}"] = pos_series.rolling(
                rolling_window, min_periods=1
            ).sum().values
            df[f"{col_raw}_neg{rolling_window}"] = neg_series.rolling(
                rolling_window, min_periods=1
            ).sum().values

    df["cdl_bullish_count"] = bullish_count
    df["cdl_bearish_count"] = bearish_count
    df["cdl_net_signal"] = bullish_count - bearish_count
    df["cdl_bullish_density_5"] = (
        pd.Series(bullish_count, index=df.index).rolling(5, min_periods=1).mean().values
    )
    df["cdl_bearish_density_5"] = (
        pd.Series(bearish_count, index=df.index).rolling(5, min_periods=1).mean().values
    )

    return df


def feature_columns(include_rolling: bool = True) -> List[str]:
    """Return the list of column names this module emits — used by feature
    matrix builders that need the schema in advance."""
    if not _HAVE_TALIB:
        return []
    cols: List[str] = []
    for n in _cdl_names():
        base = f"cdl_{n.lower()}"
        cols.append(base)
        if include_rolling:
            cols.append(f"{base}_pos5")
            cols.append(f"{base}_neg5")
    cols.extend([
        "cdl_bullish_count",
        "cdl_bearish_count",
        "cdl_net_signal",
        "cdl_bullish_density_5",
        "cdl_bearish_density_5",
    ])
    return cols


def feature_count(include_rolling: bool = True) -> int:
    return len(feature_columns(include_rolling))


if __name__ == "__main__":
    # Smoke test
    import sys
    logging.basicConfig(level=logging.INFO)
    if not _HAVE_TALIB:
        sys.exit("talib not installed — exit 1")
    print(f"TA-Lib version: {talib.__version__}")
    print(f"CDL function count: {len(_cdl_names())}")
    print(f"Feature columns with rolling: {feature_count(True)}")
    print(f"Feature columns without rolling: {feature_count(False)}")

    # build a tiny df
    n = 100
    rng = np.random.default_rng(42)
    base = 100.0 + np.cumsum(rng.normal(0, 1, n))
    df = pd.DataFrame({
        "open":  base + rng.normal(0, 0.5, n),
        "high":  base + np.abs(rng.normal(0.5, 0.5, n)),
        "low":   base - np.abs(rng.normal(0.5, 0.5, n)),
        "close": base + rng.normal(0, 0.5, n),
    })
    # enforce h>=max(o,c) and l<=min(o,c)
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)

    out = add_candlestick_features(df.copy())
    print(f"Output shape: {out.shape} (input was {df.shape})")
    print(f"New cols added: {out.shape[1] - df.shape[1]}")
    print(f"Bullish-count sum: {int(out['cdl_bullish_count'].sum())}")
    print(f"Bearish-count sum: {int(out['cdl_bearish_count'].sum())}")
    print(f"Sample net_signal values: {out['cdl_net_signal'].tail(5).tolist()}")
