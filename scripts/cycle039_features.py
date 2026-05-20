# Source: /Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/claudes test/research/archive/cycle039_enrichment_F_dimensions_2026-05-01/enrich_ticker.py
"""
Wrapper for cycle039 enrichment F-dimension features (OHLCV-extractable subset).

Cycle039 tested 6 enrichment families on top of LowBounce baseline:
  F1: EDGAR no-trade window    — requires external edgar DB (omitted; see cycle059)
  F2: GovBuy momentum          — requires external govtrades DB (omitted; see cycle059)
  F3: GovSell avoidance        — requires external govtrades DB (omitted; see cycle059)
  F4: Sector relative strength — requires basket price data (omitted)
  F5: 5D momentum gate         — EXTRACTABLE from OHLCV
  F6: Time-of-day window       — intraday parameter, not a daily feature

Features emitted (all .shift(1)-safe, daily OHLCV input):
  c039_ret_5d     : 5-day price return (prior bar)
  c039_ret_1d     : 1-day price return (prior bar)
  c039_ema20      : 20-period EMA of close (prior bar)
  c039_ema20_bull : bool — prior close > prior EMA20 (the cycle039 "1d_bull" proxy)
  c039_mom_pos_5d : bool — prior 5d return > 0 (F5 gate in boolean form)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CYCLE039_FEATURE_NAMES: list[str] = [
    "c039_ret_5d",
    "c039_ret_1d",
    "c039_ema20",
    "c039_ema20_bull",
    "c039_mom_pos_5d",
]


def _zero_fill(df: pd.DataFrame) -> pd.DataFrame:
    for col in CYCLE039_FEATURE_NAMES:
        if col not in df.columns:
            if col in ("c039_ema20_bull", "c039_mom_pos_5d"):
                df[col] = False
            else:
                df[col] = 0.0
    return df


def add_cycle039_features(df: pd.DataFrame, ticker: str = None) -> pd.DataFrame:
    """Append cycle039 extractable momentum/trend features to daily OHLCV df. Idempotent.

    Requires df to have 'close'. All output is .shift(1)-safe.
    External-data-dependent families (F1/F2/F3/F4) are not emitted here.
    """
    if df is None or len(df) == 0:
        return df
    if all(c in df.columns for c in CYCLE039_FEATURE_NAMES):
        return df
    if "close" not in df.columns:
        return _zero_fill(df)

    close = pd.to_numeric(df["close"], errors="coerce").astype(float)

    ret_5d = close.pct_change(5)
    ret_1d = close.pct_change(1)
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema20_bull = close > ema20
    mom_pos_5d = ret_5d > 0

    if "c039_ret_5d" not in df.columns:
        df["c039_ret_5d"] = ret_5d.shift(1).fillna(0.0).values
    if "c039_ret_1d" not in df.columns:
        df["c039_ret_1d"] = ret_1d.shift(1).fillna(0.0).values
    if "c039_ema20" not in df.columns:
        df["c039_ema20"] = ema20.shift(1).ffill().bfill().values
    if "c039_ema20_bull" not in df.columns:
        df["c039_ema20_bull"] = ema20_bull.shift(1).fillna(False).astype(bool).values
    if "c039_mom_pos_5d" not in df.columns:
        df["c039_mom_pos_5d"] = mom_pos_5d.shift(1).fillna(False).astype(bool).values

    return df
