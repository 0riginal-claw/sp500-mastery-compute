"""py_market_profile_features.py — Market profile / volume profile features (STUB).

TODO: wire into v10 / Mythos pipeline.
Source repo: repos-claude-clones/py-market-profile (BSD).
Install:   pip install marketprofile

Computes the Market Profile (volume distribution across price buckets) for
each rolling window. Distinct from the existing `volume_profile_features.py`
in that this library exposes the Time-Price-Opportunity (TPO) style market
profile rather than a simple volume-by-price histogram — useful for value
area / point-of-control / initial-balance features.

Look-ahead safety: each profile is built from PAST bars only. .shift(1)
applied before label join.

Estimated features added per ticker: ~6 columns
(poc_price, value_area_high, value_area_low, value_area_width_pct,
 close_position_in_va, initial_balance_range_pct).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


_FEATURES = [
    "poc_price",
    "value_area_high",
    "value_area_low",
    "value_area_width_pct",
    "close_position_in_va",
    "initial_balance_range_pct",
]

_LOOKBACK = 20  # bars per profile (≈ 1 trading month for daily)


def add_py_market_profile_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Add TPO / Market-Profile summary features per bar.

    Args:
        df: DataFrame with columns open, high, low, close, volume (lowercase)
            and a DatetimeIndex or 'date' column.
        ticker: ticker symbol (reserved for cross-sectional cache).
    """
    out = df.copy()

    try:
        from market_profile import MarketProfile  # lazy import
    except Exception:
        for c in _FEATURES:
            out[f"pymp_{c}"] = 0.0
        new_cols = [c for c in out.columns if c not in df.columns]
        out[new_cols] = out[new_cols].shift(1)
        return out

    # The MarketProfile constructor expects an OHLCV DataFrame with a
    # DatetimeIndex. We use a rolling _LOOKBACK window and slice via the
    # library's accessor. To stay cheap we only evaluate once per
    # _LOOKBACK-stride and forward-fill — sufficient signal for daily bars.
    n = len(out)
    cols = {c: np.full(n, np.nan, dtype=float) for c in _FEATURES}

    if "date" in out.columns and not isinstance(out.index, pd.DatetimeIndex):
        try:
            out_dt = out.set_index(pd.DatetimeIndex(out["date"]))
        except Exception:
            out_dt = out
    else:
        out_dt = out

    stride = max(1, _LOOKBACK // 4)
    for end in range(_LOOKBACK, n, stride):
        window = out_dt.iloc[end - _LOOKBACK:end]
        try:
            mp = MarketProfile(window)
            mps = mp[window.index.min():window.index.max()]
            poc = float(getattr(mps, "poc_price", np.nan))
            va_low, va_high = mps.value_area
            va_low = float(va_low); va_high = float(va_high)
            ib_low, ib_high = mps.initial_balance()
            close = float(window["close"].iloc[-1])
            va_width = (va_high - va_low) / close if close else np.nan
            close_pos = (close - va_low) / (va_high - va_low) if va_high > va_low else 0.5
            ib_range = (float(ib_high) - float(ib_low)) / close if close else np.nan
            for i in range(end, min(end + stride, n)):
                cols["poc_price"][i] = poc
                cols["value_area_high"][i] = va_high
                cols["value_area_low"][i] = va_low
                cols["value_area_width_pct"][i] = va_width
                cols["close_position_in_va"][i] = close_pos
                cols["initial_balance_range_pct"][i] = ib_range
        except Exception:
            continue

    for c in _FEATURES:
        out[f"pymp_{c}"] = cols[c]
    new_cols = [c for c in out.columns if c not in df.columns]
    out[new_cols] = out[new_cols].shift(1)
    return out


if __name__ == "__main__":
    print("TODO: wire py_market_profile_features into v10. Validate non-overlap with volume_profile_features.")
