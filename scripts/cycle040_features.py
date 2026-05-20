# Source: /Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/claudes test/research/archive/cycle040_enrichment_G_dimensions_2026-05-03/enrich_v2.py
"""
Wrapper for cycle040 enrichment G-dimension features.

Cycle040 introduced 8 new enrichment families:
  G1: Multi-TF trend (1Day+4Hr+15MIN) — requires intraday bars (omitted)
  G2: Volume z-score >1.5σ            — EXTRACTABLE from daily OHLCV
  G3: Low-ATR-tertile day             — EXTRACTABLE from daily OHLCV
  G4: Bullish entry bar (close>open)  — EXTRACTABLE from daily OHLCV
  G5: GovBuy + 5D mom stack           — requires external govtrades DB (omitted)
  G6: 30MIN RSI < 30                  — requires intraday bars (omitted)
  G7: SHORT-side bear high-break      — signal variant, not a feature column
  G8: Volume-of-day rank > 70%        — adapted to rolling daily percentile

Features emitted (all .shift(1)-safe, daily OHLCV input):
  c040_vol_zscore    : volume z-score vs 20-bar rolling mean/std (G2 analogue)
  c040_atr_tertile   : 0=low / 1=mid / 2=high ATR vs rolling 252-day distribution (G3)
  c040_bullish_bar   : bool — prior bar was bullish (close > open) (G4)
  c040_vol_rank_pct  : volume percentile rank vs rolling 252-day window (G8 analogue)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CYCLE040_FEATURE_NAMES: list[str] = [
    "c040_vol_zscore",
    "c040_atr_tertile",
    "c040_bullish_bar",
    "c040_vol_rank_pct",
]

_VOL_Z_WIN = 20
_ATR_WIN = 14
_RANK_WIN = 252
_TERTILE_WIN = 252


def _zero_fill(df: pd.DataFrame) -> pd.DataFrame:
    for col in CYCLE040_FEATURE_NAMES:
        if col not in df.columns:
            if col == "c040_bullish_bar":
                df[col] = False
            elif col == "c040_atr_tertile":
                df[col] = 1
            else:
                df[col] = 0.0
    return df


def add_cycle040_features(df: pd.DataFrame, ticker: str = None) -> pd.DataFrame:
    """Append cycle040 G-dimension features to daily OHLCV df. Idempotent.

    Requires df to have 'close'. 'high', 'low', 'open', 'volume' recommended.
    All output is .shift(1)-safe — each column uses only prior-bar data.
    """
    if df is None or len(df) == 0:
        return df
    if all(c in df.columns for c in CYCLE040_FEATURE_NAMES):
        return df
    if "close" not in df.columns:
        return _zero_fill(df)

    close = pd.to_numeric(df["close"], errors="coerce").astype(float)
    open_ = pd.to_numeric(df["open"], errors="coerce").astype(float) if "open" in df.columns else close
    high = pd.to_numeric(df["high"], errors="coerce").astype(float) if "high" in df.columns else close
    low = pd.to_numeric(df["low"], errors="coerce").astype(float) if "low" in df.columns else close
    volume = pd.to_numeric(df["volume"], errors="coerce").astype(float) if "volume" in df.columns else pd.Series(1.0, index=df.index)

    # G2: Volume z-score (rolling 20-bar mean+std)
    vol_mean = volume.rolling(_VOL_Z_WIN, min_periods=5).mean()
    vol_std = volume.rolling(_VOL_Z_WIN, min_periods=5).std().replace(0, np.nan)
    vol_zscore = ((volume - vol_mean) / vol_std).fillna(0.0)

    # G3: ATR tertile (rolling 252-day ATR distribution, ternary 0/1/2)
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(_ATR_WIN, min_periods=1).mean()
    atr_rank = atr.rolling(_TERTILE_WIN, min_periods=30).rank(pct=True)
    atr_tertile = pd.cut(
        atr_rank,
        bins=[-np.inf, 1 / 3, 2 / 3, np.inf],
        labels=[0, 1, 2],
    ).astype(float).fillna(1.0)

    # G4: Bullish bar (prior bar close > open)
    bullish_bar = close > open_

    # G8: Volume rank percentile (rolling 252-day window)
    vol_rank_pct = volume.rolling(_RANK_WIN, min_periods=30).rank(pct=True).fillna(0.5)

    if "c040_vol_zscore" not in df.columns:
        df["c040_vol_zscore"] = vol_zscore.shift(1).fillna(0.0).values
    if "c040_atr_tertile" not in df.columns:
        df["c040_atr_tertile"] = atr_tertile.shift(1).fillna(1.0).astype(int).values
    if "c040_bullish_bar" not in df.columns:
        # already references prior bar when shifted
        df["c040_bullish_bar"] = bullish_bar.shift(1).fillna(False).astype(bool).values
    if "c040_vol_rank_pct" not in df.columns:
        df["c040_vol_rank_pct"] = vol_rank_pct.shift(1).fillna(0.5).values

    return df
