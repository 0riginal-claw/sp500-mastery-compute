"""
Wrapper: zvt (zvtvz/zvt) — MA/EMA/MACD technical factors from pure-pandas algorithm.py.
No DB or zvt domain objects needed — algorithm functions operate directly on pd.Series.
"""
import sys
import os
from pathlib import Path

import pandas as pd

ZVT_SRC = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools/repos-claude-clones/zvt/src"
)
if str(ZVT_SRC) not in sys.path:
    sys.path.insert(0, str(ZVT_SRC))

try:
    from zvt.factors.algorithm import ma as _ma, ema as _ema, macd as _macd
    _ZVT_AVAILABLE = True
except ImportError:
    _ZVT_AVAILABLE = False


def add_zvt_features(
    df: pd.DataFrame,
    close_col: str = "close",
    ma_windows: list = None,
    ema_windows: list = None,
    macd_slow: int = 26,
    macd_fast: int = 12,
    macd_signal: int = 9,
) -> pd.DataFrame:
    """
    Add zvt MA/EMA/MACD features to OHLCV dataframe.

    Adds columns:
      zvt_ma_{w}      — rolling MA for each window in ma_windows
      zvt_ema_{w}     — EMA for each window in ema_windows
      zvt_macd        — MACD line (fast EMA - slow EMA)
      zvt_macd_signal — signal line (EMA of MACD)
      zvt_macd_hist   — histogram (macd - signal)

    Falls back to pandas-native computation if zvt import fails.
    """
    if ma_windows is None:
        ma_windows = [5, 10, 20]
    if ema_windows is None:
        ema_windows = [12, 26]

    df = df.copy()
    s = df[close_col]

    if _ZVT_AVAILABLE:
        for w in ma_windows:
            df[f"zvt_ma_{w}"] = _ma(s, window=w)
        for w in ema_windows:
            df[f"zvt_ema_{w}"] = _ema(s, window=w)
        macd_df = _macd(s, slow=macd_slow, fast=macd_fast, n=macd_signal, return_type="df")
        df["zvt_macd"] = macd_df.get("macd", pd.Series(dtype=float))
        df["zvt_macd_signal"] = macd_df.get("signal", pd.Series(dtype=float))
        df["zvt_macd_hist"] = macd_df.get("hist", pd.Series(dtype=float))
    else:
        # pandas fallback — same logic as zvt algorithm.py
        for w in ma_windows:
            df[f"zvt_ma_{w}"] = s.rolling(window=w, min_periods=w).mean()
        for w in ema_windows:
            df[f"zvt_ema_{w}"] = s.ewm(span=w, adjust=False, min_periods=w).mean()
        ema_fast = s.ewm(span=macd_fast, adjust=False, min_periods=macd_fast).mean()
        ema_slow = s.ewm(span=macd_slow, adjust=False, min_periods=macd_slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=macd_signal, adjust=False, min_periods=macd_signal).mean()
        df["zvt_macd"] = macd_line
        df["zvt_macd_signal"] = signal_line
        df["zvt_macd_hist"] = macd_line - signal_line

    return df


if __name__ == "__main__":
    # smoke test
    import numpy as np
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    rng = np.random.default_rng(42)
    test_df = pd.DataFrame(
        {"open": 100, "high": 105, "low": 95, "close": 100 + rng.standard_normal(100).cumsum(), "volume": 1e6},
        index=dates,
    )
    out = add_zvt_features(test_df)
    print(out[["close", "zvt_ma_5", "zvt_ema_12", "zvt_macd", "zvt_macd_hist"]].tail(5))
    print("zvt_features smoke test PASS" if "zvt_macd" in out.columns else "FAIL")
