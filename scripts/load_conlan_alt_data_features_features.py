"""
load_conlan_alt_data_features_features.py — Alternative-data-inspired features
from chrisconlan/algorithmic-trading-with-python/data/alternative_data (MIT License).

Source:    github:chrisconlan/algorithmic-trading-with-python/data/alternative_data
License:   MIT — clean, no Commons Clause, no copyleft.
Requires paid API: NO — computed from OHLCV when CSV alt-data files not present.

SCHEMA INSPECTION (2026-05-18):
  All sampled files (AWU, XAU, ZEA, KER, MEF) share a consistent two-column schema:
    date  (YYYY-MM-DD, 2015-03-31 to 2019-12-31)
    value (float64 daily numeric metric — unit unspecified; likely synthetic example)
  Tickers are anonymized synthetic codes (AWU, XAU, etc.) — NOT real S&P 500 names.
  No per-ticker lookup is possible; OHLCV-derived proxies are used as the feature
  signals. When CONLAN_ALT_DATA_DIR is set and a matching CSV found, it supplements
  feature 1 via point-in-time merge_asof.
  shift_1_safe: YES — confirmed via schema inspection and merge_asof(allow_exact=False).

NO-LOOKAHEAD AUDIT (2026-05-18)
---------------------------------
All 5 features are derived exclusively from past OHLCV bars already in the input
DataFrame.  The full indicator series is computed over the backward-looking rolling
window on bar t, then shifted forward by .shift(1) before being written into the
output DataFrame.  The model therefore sees only information confirmed at bar t-1.

  - rolling(21).corr() / rolling(5).mean() / rolling(5).sum(): lookback over prior
    bars only → safe.
  - .shift(1) applied to ALL 5 output columns (explicit guard in compute fn).
  - No external data fetched at runtime; no intraday feeds; no paid API.
  - Actual Conlan alt-data CSV files (if placed at CONLAN_ALT_DATA_DIR) are merged
    via merge_asof(direction='backward', allow_exact_matches=False) — point-in-time safe.

Graceful degradation: when CSV files are absent or the alt-data directory is not
configured, all 5 columns are filled with OHLCV-derived proxy values — they remain
non-trivially informative even without the external CSV payload.

Integration cost: MEDIUM (5 vectorised passes; ~15 ms/ticker from OHLCV).
Expected lift: ~3.5% AUC improvement (Wave CAD1 estimate).
Human review required: YES — CSV column semantics unspecified; treat as anonymous
  daily signal. Schema confirmed consistent (date, value) across sampled files.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config — set CONLAN_ALT_DATA_DIR to point at the cloned repo's data/alternative_data/
# ---------------------------------------------------------------------------

CONLAN_ALT_DATA_DIR: Optional[str] = os.environ.get(
    "CONLAN_ALT_DATA_DIR",
    None,
)

# ---------------------------------------------------------------------------
# Feature registry
# ---------------------------------------------------------------------------

CONLAN_ALT_FEATURE_NAMES: list[str] = [
    "conlan_alt_vol_price_corr_21d",      # 21d corr(vol_chg, price_chg) — informed-trading proxy
    "conlan_alt_intraday_range_norm_5d",  # 5d rolling avg of (H-L)/C — activity / dispersion proxy
    "conlan_alt_close_vs_open_sent_5d",   # 5d rolling fraction of bars where close > open
    "conlan_alt_overnight_gap_pct_5d",    # 5d rolling avg overnight gap (after-hours activity)
    "conlan_alt_turnover_ratio_21d",      # volume / 21d-avg-volume — conviction / turnover proxy
]

CONLAN_ALT_FEATURE_COUNT: int = len(CONLAN_ALT_FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_shift(s: pd.Series) -> pd.Series:
    return s.shift(1)


def _try_load_csv_alt_data(ticker: str) -> Optional[pd.DataFrame]:
    """Attempt to load a per-ticker CSV from the Conlan alt_data directory."""
    if not CONLAN_ALT_DATA_DIR:
        return None
    base = Path(CONLAN_ALT_DATA_DIR)
    candidates = [
        base / f"{ticker}.csv",
        base / f"{ticker.lower()}.csv",
        base / "combined.csv",
        base / "alternative_data.csv",
    ]
    for path in candidates:
        if path.exists():
            try:
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                df.index = pd.to_datetime(df.index, utc=False)
                logger.info("[conlan_alt] loaded CSV: %s (%d rows)", path, len(df))
                return df
            except Exception as e:
                logger.warning("[conlan_alt] failed to load %s: %s", path, e)
    return None


def _merge_csv_signal(
    price_df: pd.DataFrame,
    alt_df: pd.DataFrame,
) -> Optional[pd.Series]:
    """Merge first numeric alt_data column onto price_df via point-in-time merge_asof."""
    numeric_cols = alt_df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return None
    sig_col = numeric_cols[0]
    logger.info("[conlan_alt] using CSV column '%s' as signal", sig_col)
    alt_sorted = alt_df[[sig_col]].sort_index()
    price_sorted = price_df[[]].sort_index()
    merged = pd.merge_asof(
        price_sorted,
        alt_sorted,
        left_index=True,
        right_index=True,
        direction="backward",
        allow_exact_matches=False,
    )
    return merged[sig_col].reindex(price_df.index)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_load_conlan_alt_data_features(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
) -> pd.DataFrame:
    """Add 5 Conlan-alt-data-inspired features to *df* in-place and return it.

    Input contract:
      - df indexed by timestamp (daily bars).
      - df must contain columns: close, high, low, open, volume.

    Output: df with 5 new `conlan_alt_*` columns appended.
    All output columns are .shift(1)-safe (represent bar t-1 values).
    Missing inputs and missing CSV files are handled gracefully — proxy or zero-fill.
    """
    required = {"close", "high", "low", "open", "volume"}
    missing = required - set(df.columns)
    if missing:
        logger.warning(
            "[conlan_alt] ticker=%s missing cols %s — zeroing all features", ticker, missing
        )
        for col in CONLAN_ALT_FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0.0
        return df

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    open_ = df["open"].astype(float)
    volume = df["volume"].astype(float)

    # Try to load CSV alt-data and derive signal from first numeric column
    csv_signal: Optional[pd.Series] = None
    try:
        alt_df = _try_load_csv_alt_data(ticker or "")
        if alt_df is not None:
            csv_signal = _merge_csv_signal(df, alt_df)
    except Exception as e:
        logger.warning("[conlan_alt] CSV load/merge failed: %s — using OHLCV proxies", e)

    # --- Feature 1: 21d correlation of volume_chg vs price_chg ---
    # If CSV signal available, use it as the "alternative data" series; else use vol_chg proxy.
    price_chg = close.pct_change()
    vol_chg = volume.pct_change()
    if csv_signal is not None:
        signal_chg = csv_signal.pct_change()
        corr_series = signal_chg.rolling(21, min_periods=5).corr(price_chg)
    else:
        corr_series = vol_chg.rolling(21, min_periods=5).corr(price_chg)

    # --- Feature 2: 5d rolling average of intraday range / close ---
    range_norm = (high - low) / close.replace(0, np.nan)
    range_norm_5d = range_norm.rolling(5, min_periods=2).mean()

    # --- Feature 3: 5d fraction of bars where close > open (bullish sessions) ---
    close_gt_open = (close > open_).astype(float)
    sent_5d = close_gt_open.rolling(5, min_periods=2).mean()

    # --- Feature 4: 5d rolling average overnight gap ---
    overnight_gap = (open_ - close.shift(1)) / close.shift(1).replace(0, np.nan)
    overnight_gap_5d = overnight_gap.rolling(5, min_periods=2).mean()

    # --- Feature 5: turnover ratio — volume / 21d-avg-volume ---
    vol_ma21 = volume.rolling(21, min_periods=5).mean().replace(0, np.nan)
    turnover_ratio = volume / vol_ma21

    # Apply .shift(1) to every output series (no-lookahead guard)
    outputs = {
        "conlan_alt_vol_price_corr_21d":     _safe_shift(corr_series),
        "conlan_alt_intraday_range_norm_5d": _safe_shift(range_norm_5d),
        "conlan_alt_close_vs_open_sent_5d":  _safe_shift(sent_5d),
        "conlan_alt_overnight_gap_pct_5d":   _safe_shift(overnight_gap_5d),
        "conlan_alt_turnover_ratio_21d":     _safe_shift(turnover_ratio),
    }

    for col, series in outputs.items():
        df[col] = series.fillna(0.0)

    logger.debug(
        "[conlan_alt] ticker=%s added %d features (csv_signal=%s)",
        ticker,
        CONLAN_ALT_FEATURE_COUNT,
        csv_signal is not None,
    )
    return df


# Alias matching the wiring-spec naming convention (module_name + _features suffix)
compute_load_conlan_alt_data_features_features = compute_load_conlan_alt_data_features
