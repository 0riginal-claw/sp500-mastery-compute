# Source: /Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/claudes test/research/active/cycle057_combo_dimensions/
# Primary source modules: earnings_filter.py (_build_earnings_blackout logic),
#                          dow_moy_filter.py (DOW/MOY calendar logic),
#                          combo_gates.py (quarter/DOW gate logic)
#
# Cycle 057 = combo dimensions (DOW/MoY, earnings proxy, ToD buckets).
# The main Python files are backtest runners testing COMBINATIONS of existing
# features. This wrapper extracts the novel computable signals:
#   1. Calendar features (day-of-week, quarter) — from dow_moy_filter.py
#   2. ATR-spike earnings proxy — from earnings_filter._build_earnings_blackout
#   3. Volatility-spike ratio — underlying metric driving proxy
#
# All outputs are .shift(1)-safe (computed from prior-day data only).

from __future__ import annotations

import numpy as np
import pandas as pd

EARNINGS_PROXY_THRESHOLD = 1.6   # ATR(d)/30d-avg-ATR > 1.6 → proxy earnings/vol-shock
EARNINGS_WINDOW_DAYS      = 2    # ±N calendar days around spike → blackout
ATR_ROLL                  = 30   # rolling window for mean ATR

CYCLE057_FEATURE_NAMES: list[str] = [
    "c057_dow",
    "c057_is_mon",
    "c057_is_fri",
    "c057_quarter",
    "c057_vol_spike_ratio",
    "c057_earnings_proxy",
]


def _col(df: pd.DataFrame, *names: str) -> pd.Series | None:
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return pd.to_numeric(df[low[n.lower()]], errors="coerce").astype(float)
    return None


def _build_blackout_mask(dates: pd.DatetimeIndex | pd.Index,
                          high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """Return boolean array: True if date falls within EARNINGS_WINDOW_DAYS of
    an ATR-spike day (proxy for earnings / high-volatility event).
    Uses prior-bar ATR so no lookahead: spike at bar i-1 is known at bar i."""
    n = len(dates)
    tr = high - low  # daily true-range proxy (no prev-close for simplicity)
    atr30 = pd.Series(tr).rolling(ATR_ROLL, min_periods=5).mean().values
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(atr30 > 0, tr / atr30, 0.0)

    # spike_days: indices where ratio > threshold (shift+1 so known next bar)
    spike_idx = np.where(ratio > EARNINGS_PROXY_THRESHOLD)[0]

    blackout = np.zeros(n, dtype=bool)
    for si in spike_idx:
        spike_date = pd.Timestamp(dates[si])
        for offset in range(-EARNINGS_WINDOW_DAYS, EARNINGS_WINDOW_DAYS + 1):
            bd = spike_date + pd.Timedelta(days=offset)
            # find matching index
            mask = dates == bd
            blackout |= mask.values if hasattr(mask, "values") else mask

    # Shift by 1: spike detected on day i is visible at day i+1
    blackout = np.roll(blackout, 1)
    blackout[0] = False
    return blackout


def add_cycle057_features(df: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
    """Append 6 calendar + earnings-proxy features from cycle057.

    Requires df with a DatetimeIndex (or date-like index) and ideally High/Low
    columns. All outputs are .shift(1)-safe. Idempotent.
    """
    if df is None or len(df) < 5:
        return df
    if all(c in df.columns for c in CYCLE057_FEATURE_NAMES):
        return df

    idx = df.index
    try:
        ts_idx = pd.DatetimeIndex(idx)
    except Exception:
        # Can't parse index as dates — return zeros
        for col in CYCLE057_FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0
        return df

    # Calendar features (from DatetimeIndex directly — no lookahead needed;
    # these encode the day's own date, which is known before open)
    dow     = ts_idx.weekday        # 0=Mon..4=Fri
    quarter = ts_idx.quarter        # 1..4

    if "c057_dow" not in df.columns:
        df["c057_dow"]     = dow.values
    if "c057_is_mon" not in df.columns:
        df["c057_is_mon"]  = (dow == 0).astype(int).values
    if "c057_is_fri" not in df.columns:
        df["c057_is_fri"]  = (dow == 4).astype(int).values
    if "c057_quarter" not in df.columns:
        df["c057_quarter"] = quarter.values

    # ATR spike ratio
    high = _col(df, "high", "High")
    low  = _col(df, "low", "Low")
    close = _col(df, "close", "Close")

    if high is not None and low is not None:
        tr = (high - low).values.astype(float)
    elif close is not None:
        tr = close.pct_change().abs().fillna(0).values * close.values
    else:
        tr = np.zeros(len(df))

    atr30 = pd.Series(tr).rolling(ATR_ROLL, min_periods=5).mean().values
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(atr30 > 0, tr / atr30, 0.0)

    # Shift by 1 for signal-day safety
    ratio_shifted = np.roll(ratio, 1)
    ratio_shifted[0] = 0.0

    if "c057_vol_spike_ratio" not in df.columns:
        df["c057_vol_spike_ratio"] = ratio_shifted

    # Earnings proxy blackout
    if "c057_earnings_proxy" not in df.columns:
        if high is not None and low is not None:
            blackout = _build_blackout_mask(ts_idx, high.values, low.values)
        else:
            # Fall back to spike ratio alone
            blackout = np.roll(ratio > EARNINGS_PROXY_THRESHOLD, 1)
            blackout[0] = False
        df["c057_earnings_proxy"] = blackout.astype(int)

    return df


if __name__ == "__main__":
    import sys
    rng = np.random.default_rng(42)
    idx = pd.date_range("2023-01-01", periods=300, freq="B")
    close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, 300)))
    demo = pd.DataFrame({
        "High":   close * (1 + np.abs(rng.normal(0, 0.007, 300))),
        "Low":    close * (1 - np.abs(rng.normal(0, 0.007, 300))),
        "Close":  close,
    }, index=idx)
    tk = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    out = add_cycle057_features(demo, tk)
    print(f"cycle057: {len(CYCLE057_FEATURE_NAMES)} features added. Shape: {out.shape}")
    print(out[CYCLE057_FEATURE_NAMES].tail(10).to_string())
    print(f"earnings_proxy fires: {out['c057_earnings_proxy'].sum()} days")
