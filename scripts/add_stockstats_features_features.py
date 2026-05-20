"""
add_stockstats_features_features.py — stockstats technical indicator features.

Source: github:jealous/stockstats (BSD 3-Clause, no paid API required).
Features added (28): see STOCKSTATS_FEATURE_NAMES below.

NO-LOOKAHEAD AUDIT (2026-05-18)
---------------------------------
All inputs are EOD OHLCV bars that are fully known at market close.
stockstats computes indicators using only historical (past and current) bars —
no future bars enter any calculation. The StockDataFrame internally operates
on the same-bar close/open/high/low/volume values.

Because every indicator at bar T is derived solely from bars ≤ T (all rolling
windows are backward-looking), and because v10's label (fwd_ret_21d) is the
NEXT bar's forward return, we apply .shift(1) to every feature series before
attaching it to the DataFrame.  This ensures that bar T's model input comes
from the indicator as of bar T-1 (the last completed bar), making it safe
for live trading where the current bar is not yet closed.

Summary:
  - stockstats StockDataFrame: only uses historical bars internally (safe).
  - .shift(1) applied to ALL 28 output columns before joining (explicit guard).
  - No external data sources, no intraday feeds, no paid API.

License: BSD 3-Clause (jealous/stockstats). Pure-pandas/numpy dependency only.
Integration cost: LOW — single in-process StockDataFrame construction, ~150ms.
Expected lift: ~1.0% CV AUC improvement per feature spec.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature names — 28 columns
# ---------------------------------------------------------------------------

STOCKSTATS_FEATURE_NAMES: list[str] = [
    # MACD family (3)
    "ss_macd",
    "ss_macds",
    "ss_macdh",
    # RSI (1)
    "ss_rsi_14",
    # Bollinger Bands (3)
    "ss_boll",
    "ss_boll_ub",
    "ss_boll_lb",
    # CCI (1)
    "ss_cci",
    # Williams %R (1)
    "ss_wr_14",
    # KDJ / Stochastic (3)
    "ss_kdjk",
    "ss_kdjd",
    "ss_kdjj",
    # ATR (1)
    "ss_atr_14",
    # DMA — difference of moving averages (1)
    "ss_dma",
    # Volume ratio (1)
    "ss_vr",
    # CR momentum indicator (1)
    "ss_cr",
    # Simple moving averages (4)
    "ss_close_5_sma",
    "ss_close_10_sma",
    "ss_close_20_sma",
    "ss_close_50_sma",
    # Exponential moving averages (2)
    "ss_close_5_ema",
    "ss_close_20_ema",
    # Rolling volatility / stddev (2)
    "ss_close_5_mstd",
    "ss_close_10_mstd",
    # Money Flow Index (1)
    "ss_mfi",
    # TRIX oscillator (1)
    "ss_trix",
    # Rate of Change (1)
    "ss_close_10_roc",
    # Volume SMA (1)
    "ss_volume_5_sma",
]

STOCKSTATS_FEATURE_COUNT: int = len(STOCKSTATS_FEATURE_NAMES)
assert STOCKSTATS_FEATURE_COUNT == 28, f"Expected 28 features, got {STOCKSTATS_FEATURE_COUNT}"

# Internal stockstats keys → output column name mapping
_SS_KEY_MAP: list[tuple[str, str]] = [
    ("macd",          "ss_macd"),
    ("macds",         "ss_macds"),
    ("macdh",         "ss_macdh"),
    ("rsi_14",        "ss_rsi_14"),
    ("boll",          "ss_boll"),
    ("boll_ub",       "ss_boll_ub"),
    ("boll_lb",       "ss_boll_lb"),
    ("cci",           "ss_cci"),
    ("wr_14",         "ss_wr_14"),
    ("kdjk",          "ss_kdjk"),
    ("kdjd",          "ss_kdjd"),
    ("kdjj",          "ss_kdjj"),
    ("atr_14",        "ss_atr_14"),
    ("dma",           "ss_dma"),
    ("vr",            "ss_vr"),
    ("cr",            "ss_cr"),
    ("close_5_sma",   "ss_close_5_sma"),
    ("close_10_sma",  "ss_close_10_sma"),
    ("close_20_sma",  "ss_close_20_sma"),
    ("close_50_sma",  "ss_close_50_sma"),
    ("close_5_ema",   "ss_close_5_ema"),
    ("close_20_ema",  "ss_close_20_ema"),
    ("close_5_mstd",  "ss_close_5_mstd"),
    ("close_10_mstd", "ss_close_10_mstd"),
    ("mfi",           "ss_mfi"),
    ("trix",          "ss_trix"),
    ("close_10_roc",  "ss_close_10_roc"),
    ("volume_5_sma",  "ss_volume_5_sma"),
]

assert len(_SS_KEY_MAP) == STOCKSTATS_FEATURE_COUNT, (
    f"Key map length {len(_SS_KEY_MAP)} != STOCKSTATS_FEATURE_COUNT {STOCKSTATS_FEATURE_COUNT}"
)


def compute_add_stockstats_features(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
) -> pd.DataFrame:
    """Compute 28 stockstats technical indicators and append to df.

    All output columns are .shift(1)-safe: the raw indicator series are
    shifted one bar before assignment so that bar-T features reflect only
    information available at bar-(T-1) close.

    Args:
        df: DataFrame with at least [open, high, low, close, volume] columns,
            indexed by timestamp (DatetimeIndex or integer).
        ticker: Optional ticker symbol for logging.

    Returns:
        df with 28 new ``ss_*`` columns appended.
    """
    try:
        from stockstats import StockDataFrame  # lazy import — optional dependency
    except ImportError as exc:
        logger.warning(
            "[stockstats] stockstats not installed (%s): %s — zeroing 28 cols", ticker, exc
        )
        for col in STOCKSTATS_FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0.0
        return df

    required = {"open", "high", "low", "close", "volume"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        logger.warning(
            "[stockstats] missing input cols %s for %s — zeroing 28 cols", missing_cols, ticker
        )
        for col in STOCKSTATS_FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0.0
        return df

    if len(df) < 52:
        logger.warning(
            "[stockstats] too few rows (%d < 52) for %s — zeroing 28 cols", len(df), ticker
        )
        for col in STOCKSTATS_FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0.0
        return df

    try:
        # Build StockDataFrame from OHLCV columns only to avoid contaminating it
        # with existing feature columns that stockstats might misparse.
        ohlcv = df[["open", "high", "low", "close", "volume"]].copy()
        stock = StockDataFrame.retype(ohlcv)

        for ss_key, out_col in _SS_KEY_MAP:
            if out_col in df.columns:
                continue  # idempotent guard
            try:
                raw: pd.Series = stock[ss_key]
                # .shift(1): bar-T feature = indicator computed through bar-(T-1)
                df[out_col] = raw.shift(1).reindex(df.index).astype(float)
            except Exception as inner_exc:
                logger.debug(
                    "[stockstats] indicator '%s' failed for %s: %s — zeroing col",
                    ss_key, ticker, inner_exc,
                )
                df[out_col] = 0.0

    except Exception as exc:
        logger.warning(
            "[stockstats] StockDataFrame construction failed for %s: %s — zeroing 28 cols",
            ticker, exc,
        )
        for col in STOCKSTATS_FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0.0

    return df


# Alias used by backtest_xgb_v10.py wiring convention
compute_add_stockstats_features_features = compute_add_stockstats_features
