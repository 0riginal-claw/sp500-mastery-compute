"""
vix_term_structure_features.py — VIX term-structure features (v1) for v10 (2026-05-18).

# NO-LOOKAHEAD AUDIT
# ------------------
# Data source: yfinance tickers ^VIX9D (9-day CBOE VIX) and ^VIX (30-day CBOE VIX).
# Both series are published by CBOE at end-of-day for the *prior* trading session.
# Join strategy: merge_asof(df.index, vix_series, direction="backward") with each
# bar D aligned to the most-recent VIX observation whose date is strictly < D
# (enforced by subtracting 1 calendar day from bar_date before the merge).
# The merged values therefore reflect the prior trading session's settlement —
# no same-bar contamination is possible.
#
# All three output columns are computed solely from the merged prior-bar values
# and rolling windows over those values; no same-bar OHLCV or forward quantities
# are referenced at any point.
#
# License: MIT (data_source=yfinance, no paid API required, public CBOE indices).

Features added (3 columns) — complementary to vix_term_structure_v2_features:
  vix_ts_spread           : VIX - VIX9D (level spread; positive = normal contango
                            where longer-dated vol > short-dated vol).
  vix_ts_spread_z21       : 21-day rolling z-score of vix_ts_spread.
                            Normalised signal for XGBoost; 0.0 for early bars.
  vix_ts_contango_streak  : Signed consecutive-day regime streak.
                            Positive = N days in contango (VIX9D < VIX).
                            Negative = N days in backwardation (VIX9D >= VIX).

Graceful failure:
  - yfinance unavailable or data empty  -> all 3 cols zero-filled.
  - Fewer than 2 observations           -> z21 is 0.0 for those bars.
  - VIX9D / VIX both 0 on a bar        -> spread set to 0.0 (neutral) for that bar.

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

VIX_TERM_FEATURE_NAMES: list[str] = [
    "vix_ts_spread",
    "vix_ts_spread_z21",
    "vix_ts_contango_streak",
]

_VIX_CACHE_DIR = WORK / "cache" / "vix_term_structure"
_VIX_CACHE_PATH = _VIX_CACHE_DIR / "vix_daily.parquet"

_LOOKBACK_YEARS = 7


def _load_vix_series() -> pd.DataFrame | None:
    """Fetch ^VIX9D and ^VIX daily close from cache or yfinance.

    Returns DataFrame indexed by date (tz-naive) with columns
    ['vix9d_close', 'vix_close'], or None on failure.
    Reuses the same cache path as vix_term_structure_v2_features.
    """
    if _VIX_CACHE_PATH.exists():
        try:
            cached = pd.read_parquet(_VIX_CACHE_PATH)
            if {"vix9d_close", "vix_close"}.issubset(cached.columns) and len(cached) > 10:
                logger.debug("[vix_ts_v1] loaded %d rows from cache", len(cached))
                return cached
        except Exception as _ce:
            logger.warning("[vix_ts_v1] cache read error: %s — refetching", _ce)

    try:
        import yfinance as yf
    except ImportError:
        logger.warning("[vix_ts_v1] yfinance not installed — all features zeroed")
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
            logger.warning("[vix_ts_v1] yfinance returned empty data")
            return None

        if isinstance(raw.columns, pd.MultiIndex):
            vix9d = raw["Close"]["^VIX9D"].rename("vix9d_close")
            vix = raw["Close"]["^VIX"].rename("vix_close")
        else:
            logger.warning("[vix_ts_v1] unexpected column shape from yfinance download")
            return None

        combined = pd.concat([vix9d, vix], axis=1).dropna()
        combined.index = pd.DatetimeIndex(combined.index).tz_localize(None)

        try:
            _VIX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            combined.to_parquet(_VIX_CACHE_PATH)
            logger.debug("[vix_ts_v1] cached %d rows to %s", len(combined), _VIX_CACHE_PATH)
        except Exception as _we:
            logger.debug("[vix_ts_v1] cache write skipped: %s", _we)

        logger.info("[vix_ts_v1] fetched %d rows via yfinance", len(combined))
        return combined

    except Exception as exc:
        logger.warning("[vix_ts_v1] yfinance fetch failed: %s — all features zeroed", exc)
        return None


def compute_vix_term_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append 3 VIX term-structure features to df. Idempotent + graceful.

    Produces features complementary to (non-overlapping with)
    vix_term_structure_v2_features: spread, spread z-score, and regime streak.

    Args:
        df: DataFrame with a DatetimeIndex (or 'date' column) indexed by trading day.

    Returns:
        df with columns [vix_ts_spread, vix_ts_spread_z21, vix_ts_contango_streak] added.
    """
    if df is None or len(df) == 0:
        return df

    if all(c in df.columns for c in VIX_TERM_FEATURE_NAMES):
        return df

    n = len(df)

    def _zero_fill(frame: pd.DataFrame) -> pd.DataFrame:
        frame["vix_ts_spread"] = 0.0
        frame["vix_ts_spread_z21"] = 0.0
        frame["vix_ts_contango_streak"] = 0
        return frame

    vix_data = _load_vix_series()
    if vix_data is None or vix_data.empty:
        return _zero_fill(df)

    if isinstance(df.index, pd.DatetimeIndex):
        bar_dates = df.index.tz_localize(None) if df.index.tz is not None else df.index
    elif "date" in df.columns:
        bar_dates = pd.DatetimeIndex(pd.to_datetime(df["date"])).tz_localize(None)
    else:
        logger.warning("[vix_ts_v1] df has no DatetimeIndex or 'date' column — zeroing")
        return _zero_fill(df)

    vix_data = vix_data.sort_index()

    # NO-LOOKAHEAD: shift lookup date back 1 day so bar D only sees VIX data from < D.
    shifted_dates = bar_dates - pd.Timedelta(days=1)
    shifted_dates = shifted_dates.astype("datetime64[us]")
    left = pd.DataFrame({"bar_date": bar_dates, "lookup_date": shifted_dates})
    vix_indexed = vix_data.reset_index().rename(
        columns={"Date": "lookup_date", "index": "lookup_date"}
    )
    if "lookup_date" not in vix_indexed.columns:
        vix_indexed.columns = ["lookup_date"] + list(vix_indexed.columns[1:])
    vix_indexed["lookup_date"] = (
        pd.to_datetime(vix_indexed["lookup_date"]).dt.tz_localize(None).astype("datetime64[us]")
    )

    merged = pd.merge_asof(
        left.sort_values("lookup_date"),
        vix_indexed.sort_values("lookup_date"),
        on="lookup_date",
        direction="backward",
    )
    merged = merged.sort_values("bar_date").reset_index(drop=True)

    vix9d_arr = merged["vix9d_close"].fillna(np.nan).values
    vix_arr = merged["vix_close"].fillna(np.nan).values

    # Spread: VIX - VIX9D; NaN → 0.0 (neutral)
    spread = np.where(
        np.isnan(vix9d_arr) | np.isnan(vix_arr),
        0.0,
        vix_arr - vix9d_arr,
    )

    # 21-day rolling z-score of the spread
    spread_series = pd.Series(spread)
    roll_mean = spread_series.rolling(21, min_periods=2).mean()
    roll_std = spread_series.rolling(21, min_periods=2).std().replace(0, np.nan)
    spread_z21 = ((spread_series - roll_mean) / roll_std).fillna(0.0).values

    # Contango streak: +N consecutive contango days (VIX9D < VIX → spread > 0),
    # -N consecutive backwardation days (VIX9D >= VIX → spread <= 0).
    contango_flag = (spread > 0).astype(int)  # 1=contango, 0=backwardation
    streak = np.zeros(n, dtype=np.int32)
    if n > 0:
        streak[0] = 1 if contango_flag[0] else -1
        for i in range(1, n):
            if contango_flag[i]:
                streak[i] = streak[i - 1] + 1 if streak[i - 1] > 0 else 1
            else:
                streak[i] = streak[i - 1] - 1 if streak[i - 1] < 0 else -1

    df = df.copy()
    df["vix_ts_spread"] = spread
    df["vix_ts_spread_z21"] = spread_z21
    df["vix_ts_contango_streak"] = streak

    logger.info("[vix_ts_v1] added %d rows × 3 cols", n)
    return df


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=60, freq="B")
    demo = pd.DataFrame({"close": np.linspace(100, 120, len(idx))}, index=idx)
    out = compute_vix_term_structure_features(demo)
    print(out[VIX_TERM_FEATURE_NAMES].tail(10).to_string())
    print("\nnon-zero spread rows:", (out["vix_ts_spread"] != 0.0).sum())
