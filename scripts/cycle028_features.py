# Source: /Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/claudes test/research/archive/cycle028_aapl_intraday_research_2026-04-29/engine/walkforward.py
"""
Wrapper for cycle028 AAPL intraday indicator features.

Extracts add_indicator_features() logic adapted to daily OHLCV.
gov_features (gov_net_flow_30d, cluster_distinct, etc.) and edgar_features
require external DBs — omitted here. Wire cycle059 for gov/edgar columns.

Features (all .shift(1)-safe, daily OHLCV input):
  c028_rsi_2        : 2-period RSI of close (prior bar value used at signal time)
  c028_ema_20       : 20-period EMA of close (prior bar)
  c028_atr_14       : 14-period ATR (prior bar)
  c028_day_bias_mtf : sign(prior_close - prior_open) — which way prior day closed
  c028_volume_z     : volume z-score vs rolling 100-bar mean/std (prior bar)
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CYCLE028_FEATURE_NAMES: list[str] = [
    "c028_rsi_2",
    "c028_ema_20",
    "c028_atr_14",
    "c028_day_bias_mtf",
    "c028_volume_z",
]

_RSI_WIN = 2
_EMA_WIN = 20
_ATR_WIN = 14
_VOL_Z_WIN = 100


def _zero_fill(df: pd.DataFrame) -> pd.DataFrame:
    for col in CYCLE028_FEATURE_NAMES:
        if col not in df.columns:
            if col == "c028_day_bias_mtf":
                df[col] = 0
            elif col == "c028_rsi_2":
                df[col] = 50.0
            else:
                df[col] = 0.0
    return df


def add_cycle028_features(df: pd.DataFrame, ticker: str = None) -> pd.DataFrame:
    """Append cycle028 indicator features to daily OHLCV df. Idempotent.

    Requires df to have 'close'. 'high', 'low', 'open', 'volume' improve quality.
    All output columns are .shift(1)-safe — each uses only prior-bar data.
    """
    if df is None or len(df) == 0:
        return df
    if all(c in df.columns for c in CYCLE028_FEATURE_NAMES):
        return df
    if "close" not in df.columns:
        return _zero_fill(df)

    close = pd.to_numeric(df["close"], errors="coerce").astype(float)
    open_ = pd.to_numeric(df["open"], errors="coerce").astype(float) if "open" in df.columns else close
    high = pd.to_numeric(df["high"], errors="coerce").astype(float) if "high" in df.columns else close
    low = pd.to_numeric(df["low"], errors="coerce").astype(float) if "low" in df.columns else close
    volume = pd.to_numeric(df["volume"], errors="coerce").astype(float) if "volume" in df.columns else pd.Series(1.0, index=df.index)

    # RSI-2 (on prior bar via shift(1) at assignment)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(_RSI_WIN, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(_RSI_WIN, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_2 = (100 - (100 / (1 + rs))).fillna(50.0)

    # EMA-20 (on prior bar via shift(1) at assignment)
    ema_20 = close.ewm(span=_EMA_WIN, adjust=False).mean()

    # ATR-14 (uses prior close internally, then shift(1) at assignment)
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr_14 = tr.rolling(_ATR_WIN, min_periods=1).mean()

    # Day-bias: sign(prior_close - prior_open) — already references prior bar
    day_bias_mtf = np.sign(close.shift(1) - open_.shift(1)).fillna(0.0)

    # Volume z-score (100-bar rolling, prior bar via shift(1) at assignment)
    vol_mean = volume.rolling(_VOL_Z_WIN, min_periods=20).mean()
    vol_std = volume.rolling(_VOL_Z_WIN, min_periods=20).std().replace(0, np.nan)
    volume_z = ((volume - vol_mean) / vol_std).fillna(0.0)

    if "c028_rsi_2" not in df.columns:
        df["c028_rsi_2"] = rsi_2.shift(1).fillna(50.0).values
    if "c028_ema_20" not in df.columns:
        df["c028_ema_20"] = ema_20.shift(1).ffill().bfill().values
    if "c028_atr_14" not in df.columns:
        df["c028_atr_14"] = atr_14.shift(1).fillna(0.0).values
    if "c028_day_bias_mtf" not in df.columns:
        # already prior-bar based (close.shift(1) - open_.shift(1)); no extra shift
        df["c028_day_bias_mtf"] = day_bias_mtf.astype(float).values
    if "c028_volume_z" not in df.columns:
        df["c028_volume_z"] = volume_z.shift(1).fillna(0.0).values

    return df
