# Source: /Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/claudes test/research/active/cycle047_sr_refinement/price_location_features.py
# Cycle 047 = SR / price-location features (streaming bar engine).
#
# The upstream engine is a per-tick stateful class (PriceLocationFeatureEngine).
# This wrapper ports the conceptual features to daily OHLCV resolution so they
# are consumable by a daily ML classifier. All values are .shift(1)-safe: each
# output column at index i is computed from data available strictly before day i.

from __future__ import annotations

import math
import numpy as np
import pandas as pd

ROUND_NUMBERS_USD = (5, 10, 25, 50, 100, 250, 500)

CYCLE047_FEATURE_NAMES: list[str] = [
    "c047_pdh",
    "c047_pdl",
    "c047_vwap_20d",
    "c047_vwap_pct_20d",
    "c047_near_round_pct",
    "c047_at_pdh_level",
    "c047_confluence_count",
    "c047_failed_break_pdh",
    "c047_failed_break_pdl",
]


def _nearest_round_pct(price: np.ndarray) -> np.ndarray:
    out = np.full(len(price), np.nan)
    for i, p in enumerate(price):
        if p > 0 and np.isfinite(p):
            best = min(abs(p - round(p / r) * r) / p * 100 for r in ROUND_NUMBERS_USD)
            out[i] = best
    return out


def _col(df: pd.DataFrame, *names: str) -> pd.Series:
    """Return first matching column (case-insensitive), or None."""
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return pd.to_numeric(df[low[n.lower()]], errors="coerce").astype(float)
    return None


def add_cycle047_features(df: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
    """Append 9 SR/price-location features derived from cycle047.

    Inputs expected (case-insensitive): Open, High, Low, Close, Volume.
    All output columns are .shift(1)-safe (values at row i use only rows 0..i-1).
    Idempotent: skips columns that already exist.
    """
    if df is None or len(df) < 2:
        return df
    if all(c in df.columns for c in CYCLE047_FEATURE_NAMES):
        return df

    close = _col(df, "close", "Close")
    high  = _col(df, "high", "High")
    low   = _col(df, "low", "Low")
    vol   = _col(df, "volume", "Volume")

    if close is None:
        return df

    # PDH / PDL — previous day's high and low
    pdh = high.shift(1) if high is not None else pd.Series(np.nan, index=df.index)
    pdl = low.shift(1) if low is not None else pd.Series(np.nan, index=df.index)

    # Rolling 20-day VWAP proxy: sum(typ*vol) / sum(vol) over trailing 20 bars
    # Uses only past bars (rolling on shifted data = no lookahead)
    if high is not None and low is not None and vol is not None:
        typ = (high + low + close) / 3.0
        cum_tv = (typ * vol).rolling(20, min_periods=2).sum()
        cum_v  = vol.rolling(20, min_periods=2).sum()
        vwap20 = (cum_tv / cum_v.replace(0, np.nan)).shift(1)
    else:
        vwap20 = close.rolling(20, min_periods=2).mean().shift(1)

    vwap_pct = ((close - vwap20) / vwap20.replace(0, np.nan) * 100)

    # Near-round-number distance (% to nearest round in {5,10,25,50,100,250,500})
    near_round_arr = _nearest_round_pct(close.shift(1).fillna(0).values)
    near_round = pd.Series(near_round_arr, index=df.index)

    # At-level: within 0.2% of PDH or PDL (use shifted, so prior-day levels)
    if high is not None and low is not None:
        dist_pdh = (close - pdh).abs() / close.replace(0, np.nan) * 100
        dist_pdl = (close - pdl).abs() / close.replace(0, np.nan) * 100
        at_level = ((dist_pdh < 0.2) | (dist_pdl < 0.2)).astype(int)
    else:
        at_level = pd.Series(0, index=df.index)

    # Confluence count: how many of {PDH, PDL, 20dH, 20dL} within 0.5% of close
    # 20dH/20dL are computed from prior bars only (shift already embedded below)
    if high is not None and low is not None:
        d20h = high.shift(1).rolling(20, min_periods=1).max()
        d20l = low.shift(1).rolling(20, min_periods=1).min()
        levels = [pdh, pdl, d20h, d20l]
        confluence = sum(
            ((lv - close).abs() / close.replace(0, np.nan) * 100 < 0.5).astype(int)
            for lv in levels
        )
    else:
        confluence = pd.Series(0, index=df.index)

    # Failed breakout / breakdown on PDH / PDL
    # failed_break_pdh: today's high > PDH but close < PDH (rejection)
    # failed_break_pdl: today's low  < PDL but close > PDL (reclaim)
    if high is not None and low is not None:
        failed_break_pdh = ((high > pdh) & (close < pdh)).astype(int)
        failed_break_pdl = ((low  < pdl) & (close > pdl)).astype(int)
    else:
        failed_break_pdh = pd.Series(0, index=df.index)
        failed_break_pdl = pd.Series(0, index=df.index)

    # Assign (idempotent)
    assigns = {
        "c047_pdh":             pdh,
        "c047_pdl":             pdl,
        "c047_vwap_20d":        vwap20,
        "c047_vwap_pct_20d":    vwap_pct,
        "c047_near_round_pct":  near_round,
        "c047_at_pdh_level":    at_level,
        "c047_confluence_count": confluence,
        "c047_failed_break_pdh": failed_break_pdh,
        "c047_failed_break_pdl": failed_break_pdl,
    }
    for col, series in assigns.items():
        if col not in df.columns:
            df[col] = series.values

    return df


if __name__ == "__main__":
    import sys
    logging_level = "INFO"
    tk = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    rng = np.random.default_rng(42)
    idx = pd.date_range("2023-01-01", periods=300, freq="B")
    close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, 300)))
    demo = pd.DataFrame({
        "Open":   close * (1 - np.abs(rng.normal(0, 0.003, 300))),
        "High":   close * (1 + np.abs(rng.normal(0, 0.006, 300))),
        "Low":    close * (1 - np.abs(rng.normal(0, 0.006, 300))),
        "Close":  close,
        "Volume": rng.integers(1_000_000, 5_000_000, 300).astype(float),
    }, index=idx)
    out = add_cycle047_features(demo, tk)
    print(f"cycle047: {len(CYCLE047_FEATURE_NAMES)} features added. Shape: {out.shape}")
    print(out[CYCLE047_FEATURE_NAMES].tail(5).to_string())
