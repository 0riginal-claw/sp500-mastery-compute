"""
add_talipp_features_features.py — talipp technical indicator features.

Source: github:nardew/talipp (MIT License, no paid API required).
Features added (8): see TALIPP_FEATURE_NAMES below.

NO-LOOKAHEAD AUDIT (2026-05-18)
---------------------------------
All inputs are EOD close prices that are fully known at market close.
Each talipp indicator at bar T is derived solely from close prices at bars ≤ T
(all rolling windows are backward-looking). The forward-return label (fwd_ret_21d)
is the NEXT bar's outcome, so bar T's indicator value is safe to use as a model
input without lookahead.

Because the current bar's close price IS available at the time we compute features
(EOD pipeline), we apply .shift(1) to every output series before attaching it to
the DataFrame. This makes bar T's model input come from the indicator value at T-1
(the last FULLY confirmed bar), which is appropriate for live trading where the
current bar may not yet be closed.

Summary:
  - talipp indicators: backward-looking rolling windows on close only (safe).
  - .shift(1) applied to ALL 8 output columns before joining (explicit guard).
  - No external data sources, no intraday feeds, no paid API.

License: MIT (nardew/talipp). Dependencies: numpy only (no TA-Lib required).
Integration cost: LOW — 8 indicator passes on close series, ~50ms per ticker.
Expected lift: ~0.5% CV AUC improvement per feature spec.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature names — 8 columns
# ---------------------------------------------------------------------------

TALIPP_FEATURE_NAMES: list[str] = [
    "talipp_tema_10",   # Triple EMA (10-period)
    "talipp_dema_10",   # Double EMA (10-period)
    "talipp_hma_14",    # Hull MA (14-period)
    "talipp_trix_10",   # TRIX 1-period ROC of triple-smoothed EMA (10-period)
    "talipp_dpo_20",    # Detrended Price Oscillator (20-period)
    "talipp_roc_10",    # Rate of Change (10-period)
    "talipp_zlema_10",  # Zero-Lag EMA (10-period)
    "talipp_wma_10",    # Weighted MA (10-period)
]

TALIPP_FEATURE_COUNT: int = len(TALIPP_FEATURE_NAMES)


def _indicator_to_series(values: list, index: pd.Index) -> pd.Series:
    """Convert a talipp output list (may contain None) to a float64 Series."""
    arr = np.array([np.nan if v is None else float(v) for v in values], dtype=np.float64)
    return pd.Series(arr, index=index)


def compute_add_talipp_features(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
) -> pd.DataFrame:
    """Compute 8 talipp TA indicators and attach them to df.

    Args:
        df: DataFrame indexed by timestamp with at minimum a 'close' column.
        ticker: Stock symbol (used for logging only).

    Returns:
        df with 8 new talipp_* columns appended.  All columns are shifted 1 bar
        (no-lookahead) and NaN-filled with 0.
    """
    from talipp.indicators import TEMA, DEMA, HMA, TRIX, DPO, ROC, ZLEMA, WMA

    if "close" not in df.columns:
        logger.warning("[talipp] 'close' column missing for %s — zeroing all features", ticker)
        for col in TALIPP_FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0.0
        return df

    closes: list[float] = df["close"].tolist()
    idx = df.index
    n = len(closes)

    if n < 22:
        logger.warning("[talipp] Not enough bars (%d) for %s — zeroing", n, ticker)
        for col in TALIPP_FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0.0
        return df

    try:
        raw: dict[str, pd.Series] = {
            "talipp_tema_10":  _indicator_to_series(TEMA(10, input_values=closes), idx),
            "talipp_dema_10":  _indicator_to_series(DEMA(10, input_values=closes), idx),
            "talipp_hma_14":   _indicator_to_series(HMA(14, input_values=closes), idx),
            "talipp_trix_10":  _indicator_to_series(TRIX(10, input_values=closes), idx),
            "talipp_dpo_20":   _indicator_to_series(DPO(20, input_values=closes), idx),
            "talipp_roc_10":   _indicator_to_series(ROC(10, input_values=closes), idx),
            "talipp_zlema_10": _indicator_to_series(ZLEMA(10, input_values=closes), idx),
            "talipp_wma_10":   _indicator_to_series(WMA(10, input_values=closes), idx),
        }
    except Exception as exc:
        logger.warning("[talipp] indicator computation failed for %s: %s — zeroing", ticker, exc)
        for col in TALIPP_FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0.0
        return df

    for col, series in raw.items():
        # .shift(1) ensures bar T gets indicator value from bar T-1 (no-lookahead guard)
        df[col] = series.shift(1).fillna(0.0)

    logger.debug("[talipp] added %d features for %s", TALIPP_FEATURE_COUNT, ticker)
    return df
