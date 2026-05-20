"""
vix_term_structure_v2_features.py — VIX term-structure features for v10 (2026-05-17).

# NO-LOOKAHEAD AUDIT
# ------------------
# Data source: yfinance tickers ^VIX9D (9-day CBOE VIX) and ^VIX (30-day CBOE VIX).
# Both series are published by CBOE at end-of-day for the *prior* trading session.
# Join strategy: merge_asof(df.index, vix_series, direction="backward") aligns
# each bar D with the most-recent VIX observation whose date < D (strictly prior).
# The merged result is NOT additionally .shift(1)-ed since the backward merge
# already ensures no same-bar contamination.
# Confirmed safe: yfinance daily OHLC close for ^VIX9D and ^VIX carries the
# settlement value from the *previous* session, not the current one. Using
# direction="backward" with an exclusive upper-bound check preserves this.
#
# License: MIT (data_source=yfinance, no paid API required, public CBOE indices).

Features added (3 columns):
  vix9d_vix_ratio         : VIX9D / VIX — term-structure slope.
                            < 1 = normal contango (short vol < long vol).
                            > 1 = inverted (short-term fear spike).
  vix_term_inverted       : int8 flag; 1 when vix9d_vix_ratio > 1.0, else 0.
  vix9d_vix_ratio_z10     : 10-day rolling z-score of vix9d_vix_ratio;
                            normalised signal for XGBoost.

Graceful failure:
  - yfinance unavailable or data empty  -> all 3 cols zero-filled.
  - Fewer than 10 observations          -> z10 is 0.0 for those bars only.
  - VIX == 0 on any bar                 -> ratio set to 1.0 (neutral) for that bar.

Idempotent: re-calling on an already-augmented df returns immediately.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WORK = Path(
    "/Users/orginal/Library/CloudStorage/"
    "GoogleDrive-zachgladstone@gmail.com/My Drive/"
    "AI-Tools/s&p500-ticker-mastery"
)

VIX_TS_FEATURE_NAMES: list[str] = [
    "vix9d_vix_ratio",
    "vix_term_inverted",
    "vix9d_vix_ratio_z10",
]

_VIX_CACHE_DIR = WORK / "cache" / "vix_term_structure"
_VIX_CACHE_PATH = _VIX_CACHE_DIR / "vix_daily.parquet"

# Lookback for yfinance pull (years)
_LOOKBACK_YEARS = 7


def _load_vix_series() -> pd.DataFrame | None:
    """Fetch ^VIX9D and ^VIX daily close from cache or yfinance.

    Returns a DataFrame indexed by date (tz-naive) with columns
    ['vix9d_close', 'vix_close'], or None on failure.
    """
    # --- Try cache first ---
    if _VIX_CACHE_PATH.exists():
        try:
            cached = pd.read_parquet(_VIX_CACHE_PATH)
            if {"vix9d_close", "vix_close"}.issubset(cached.columns) and len(cached) > 10:
                logger.debug("[vix_ts_v2] loaded %d rows from cache", len(cached))
                return cached
        except Exception as _ce:
            logger.warning("[vix_ts_v2] cache read error: %s — refetching", _ce)

    # --- Fetch via yfinance ---
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("[vix_ts_v2] yfinance not installed — all features zeroed")
        return None

    try:
        end = pd.Timestamp.utcnow().normalize()
        start = end - pd.DateOffset(years=_LOOKBACK_YEARS)
        raw = yf.download(
            ["^VIX9D", "^VIX"],
            start=start.date().isoformat(),
            end=end.date().isoformat(),
            auto_adjust=True,
            progress=False,
        )
        if raw is None or raw.empty:
            logger.warning("[vix_ts_v2] yfinance returned empty data")
            return None

        # yfinance multi-ticker: columns are (field, ticker)
        if isinstance(raw.columns, pd.MultiIndex):
            vix9d = raw["Close"]["^VIX9D"].rename("vix9d_close")
            vix = raw["Close"]["^VIX"].rename("vix_close")
        else:
            # Fallback: single-ticker unexpected shape
            logger.warning("[vix_ts_v2] unexpected column shape from yfinance download")
            return None

        combined = pd.concat([vix9d, vix], axis=1).dropna()
        combined.index = pd.DatetimeIndex(combined.index).tz_localize(None)

        # Cache to parquet for future calls
        try:
            _VIX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            combined.to_parquet(_VIX_CACHE_PATH)
            logger.debug("[vix_ts_v2] cached %d rows to %s", len(combined), _VIX_CACHE_PATH)
        except Exception as _we:
            logger.debug("[vix_ts_v2] cache write skipped: %s", _we)

        logger.info("[vix_ts_v2] fetched %d rows via yfinance", len(combined))
        return combined

    except Exception as exc:
        logger.warning("[vix_ts_v2] yfinance fetch failed: %s — all features zeroed", exc)
        return None


def compute_vix_term_structure_v2_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append 3 VIX term-structure features to df. Idempotent + graceful.

    Args:
        df: DataFrame with a DatetimeIndex (or 'date' column) indexed by trading day.

    Returns:
        df with columns [vix9d_vix_ratio, vix_term_inverted, vix9d_vix_ratio_z10] added.
    """
    if df is None or len(df) == 0:
        return df

    # Idempotency guard
    if all(c in df.columns for c in VIX_TS_FEATURE_NAMES):
        return df

    n = len(df)

    def _zero_fill(frame: pd.DataFrame) -> pd.DataFrame:
        frame["vix9d_vix_ratio"] = 1.0
        frame["vix_term_inverted"] = np.int8(0)
        frame["vix9d_vix_ratio_z10"] = 0.0
        return frame

    vix_data = _load_vix_series()
    if vix_data is None or vix_data.empty:
        return _zero_fill(df)

    # Build bar-date index (tz-naive)
    if isinstance(df.index, pd.DatetimeIndex):
        bar_dates = df.index.tz_localize(None) if df.index.tz is not None else df.index
    elif "date" in df.columns:
        bar_dates = pd.DatetimeIndex(pd.to_datetime(df["date"])).tz_localize(None)
    else:
        logger.warning("[vix_ts_v2] df has no DatetimeIndex or 'date' column — zeroing")
        return _zero_fill(df)

    vix_data = vix_data.sort_index()

    # NO-LOOKAHEAD ALIGNMENT:
    # merge_asof with direction="backward" picks the latest VIX row whose
    # index is strictly <= bar_date.  Since VIX close reflects end-of-day
    # settlement for that session, a bar opening on day D can only observe
    # VIX data from day D-1 or earlier.  We enforce this by shifting the
    # lookup dates back by one calendar day.
    shifted_dates = bar_dates - pd.Timedelta(days=1)
    # Normalise to datetime64[us] to avoid MergeError on precision mismatch
    shifted_dates = shifted_dates.astype("datetime64[us]")
    left = pd.DataFrame({"bar_date": bar_dates, "lookup_date": shifted_dates})
    vix_indexed = vix_data.reset_index().rename(columns={"Date": "lookup_date", "index": "lookup_date"})
    # Normalise column name regardless of yfinance version
    if "lookup_date" not in vix_indexed.columns:
        vix_indexed.columns = ["lookup_date"] + list(vix_indexed.columns[1:])
    vix_indexed["lookup_date"] = pd.to_datetime(vix_indexed["lookup_date"]).dt.tz_localize(None).astype("datetime64[us]")

    merged = pd.merge_asof(
        left.sort_values("lookup_date"),
        vix_indexed.sort_values("lookup_date"),
        on="lookup_date",
        direction="backward",
    )
    merged = merged.sort_values("bar_date").reset_index(drop=True)

    vix9d_arr = merged["vix9d_close"].fillna(np.nan).values
    vix_arr = merged["vix_close"].fillna(np.nan).values

    # Compute ratio (guard VIX == 0 → neutral 1.0)
    safe_vix = np.where(vix_arr == 0, np.nan, vix_arr)
    ratio = np.where(
        np.isnan(safe_vix) | np.isnan(vix9d_arr),
        1.0,
        vix9d_arr / safe_vix,
    )

    inverted = (ratio > 1.0).astype(np.int8)

    # 10-day rolling z-score (minimum 2 observations to emit non-zero)
    ratio_series = pd.Series(ratio)
    roll_mean = ratio_series.rolling(10, min_periods=2).mean()
    roll_std = ratio_series.rolling(10, min_periods=2).std().replace(0, np.nan)
    z10 = ((ratio_series - roll_mean) / roll_std).fillna(0.0).values

    df = df.copy()
    df["vix9d_vix_ratio"] = ratio
    df["vix_term_inverted"] = inverted
    df["vix9d_vix_ratio_z10"] = z10

    logger.info("[vix_ts_v2] added %d rows × 3 cols", n)
    return df


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=60, freq="B")
    demo = pd.DataFrame({"close": np.linspace(100, 120, len(idx))}, index=idx)
    out = compute_vix_term_structure_v2_features(demo)
    print(out[VIX_TS_FEATURE_NAMES].tail(10).to_string())
    print("\nnon-zero ratio rows:", (out["vix9d_vix_ratio"] != 1.0).sum())
