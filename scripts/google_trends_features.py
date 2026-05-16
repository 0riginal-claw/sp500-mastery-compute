"""
google_trends_features.py
=========================
Adds Google Trends search-volume features for a given ticker onto a
daily price DataFrame used for S&P 500 backtesting.

All features are .shift(1)-safe: they are built from the weekly trends
series forward-filled to daily, then aligned to the bar date so that
today's feature value reflects search interest as of *yesterday's* week.

Features added
--------------
gtrends_score_7d_avg    : 7-day rolling mean of trends score (0-100)
gtrends_score_30d_avg   : 30-day rolling mean of trends score
gtrends_zscore_60d      : (current_score - 60d_mean) / 60d_std; 0 if std==0
gtrends_change_5d       : 5-day % change in score; 0 if prior==0
gtrends_pct_rank_252d   : percentile rank within trailing 252 business days (0-1)
gtrends_spike_flag      : 1 if score > 80, else 0
gtrends_dropoff_flag    : 1 if score < 10, else 0

Data source
-----------
Google Trends via pytrends (no API key required).
  - Keyword  : f"{ticker} stock"  (e.g. "AAPL stock")
  - Timeframe: 2021-01-01 to 2026-04-30 (matches backtest window)
  - Resolution: weekly (5-year span forces weekly; daily not available)
  - Weekly scores are forward-filled to daily frequency.

Caching
-------
Raw weekly trends are cached to
  cache/gtrends/<TICKER>.parquet
relative to this script's parent directory.  Cache is re-used on
subsequent calls; delete the file to force a refresh.

Rate-limit handling
-------------------
Google Trends imposes aggressive rate limits (~1 req / 2-3 s; stricter
for repeated calls).  The fetcher sleeps 3 s between calls and retries
once on 429.  If a second 429 occurs, it logs a warning and returns the
input DataFrame unchanged — the pipeline will simply have NaN for all
trends columns, which downstream models treat as missing data.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_CACHE_DIR = _PROJECT_ROOT / "cache" / "gtrends"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_TRENDS_START = "2021-01-01"
_TRENDS_END   = "2026-04-30"
_SLEEP_SECS   = 3          # pause between pytrends requests
_SPIKE_THRESH = 80
_DROP_THRESH  = 10


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _fetch_trends(ticker: str) -> pd.Series | None:
    """Fetch weekly Google Trends for '<ticker> stock', return daily-indexed Series.

    Returns None on unrecoverable failure (rate-limit / network error).
    """
    try:
        from pytrends.request import TrendReq  # lazy import
    except ImportError:
        log.warning("pytrends not installed; skipping Google Trends features.")
        return None

    keyword = f"{ticker} stock"
    timeframe = f"{_TRENDS_START} {_TRENDS_END}"

    def _pull() -> pd.DataFrame | None:
        try:
            # Avoid passing retries/backoff_factor — urllib3 2.x renamed
            # 'method_whitelist' to 'allowed_methods'; pytrends 4.x passes
            # the old name internally when those kwargs are supplied.
            pt = TrendReq(hl="en-US", tz=0, timeout=(10, 25))
            pt.build_payload([keyword], timeframe=timeframe)
            time.sleep(_SLEEP_SECS)
            df = pt.interest_over_time()
            return df
        except Exception as exc:
            return exc

    result = _pull()

    # On 429 / ResponseError, retry once after a longer sleep
    if isinstance(result, Exception):
        err_str = str(result)
        if "429" in err_str or "response" in err_str.lower():
            log.warning("Google Trends 429 for %s — sleeping 30 s then retrying once.", ticker)
            time.sleep(30)
            result = _pull()

    if isinstance(result, Exception):
        log.warning("Google Trends fetch failed for %s (%s); returning df unchanged.", ticker, result)
        return None

    if result is None or result.empty:
        log.warning("Google Trends returned empty data for %s.", ticker)
        return None

    # Drop the 'isPartial' column if present
    if "isPartial" in result.columns:
        result = result.drop(columns=["isPartial"])

    # Keep only our keyword column
    if keyword not in result.columns:
        # Sometimes pytrends renames; take first numeric column
        numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            log.warning("No numeric column in trends data for %s.", ticker)
            return None
        weekly = result[numeric_cols[0]].rename(keyword)
    else:
        weekly = result[keyword]

    # weekly index is tz-aware (UTC) or naive depending on pytrends version — normalise to tz-naive
    if weekly.index.tz is not None:
        weekly.index = weekly.index.tz_localize(None)

    weekly.index = pd.to_datetime(weekly.index).normalize()

    # Resample / forward-fill to calendar daily, then we'll align to business days later
    daily = (
        weekly
        .reindex(pd.date_range(weekly.index.min(), _TRENDS_END, freq="D"))
        .ffill()
    )
    return daily


def _load_cached(ticker: str) -> pd.Series | None:
    cache_path = _CACHE_DIR / f"{ticker}.parquet"
    if cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
            s = df.iloc[:, 0]
            s.index = pd.to_datetime(s.index).normalize()
            if s.index.tz is not None:
                s.index = s.index.tz_localize(None)
            log.debug("Loaded cached trends for %s from %s", ticker, cache_path)
            return s
        except Exception as exc:
            log.warning("Cache read failed for %s (%s); re-fetching.", ticker, exc)
    return None


def _save_cache(ticker: str, series: pd.Series) -> None:
    cache_path = _CACHE_DIR / f"{ticker}.parquet"
    try:
        series.to_frame(name="score").to_parquet(cache_path)
        log.debug("Cached trends for %s to %s", ticker, cache_path)
    except Exception as exc:
        log.warning("Failed to cache trends for %s (%s).", ticker, exc)


def _get_trends_series(ticker: str) -> pd.Series | None:
    """Return tz-naive daily trends series for ticker (cached or fetched)."""
    cached = _load_cached(ticker)
    if cached is not None:
        return cached

    series = _fetch_trends(ticker)
    if series is not None:
        _save_cache(ticker, series)
    return series


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_google_trends_features(daily_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Add Google Trends search-volume features for *ticker* to *daily_df*.

    Parameters
    ----------
    daily_df : pd.DataFrame
        Daily OHLCV-style DataFrame with a DatetimeIndex (tz-aware or naive).
        Must have at least one row.
    ticker : str
        Ticker symbol (e.g. "AAPL").

    Returns
    -------
    pd.DataFrame
        Input DataFrame with up to 7 new columns appended.  If trends data
        cannot be fetched, the original DataFrame is returned unchanged
        (all new columns would be NaN, which downstream models handle as
        missing).

    Notes
    -----
    All columns are .shift(1)-safe: the trends score for bar date T reflects
    the weekly Google search interest that was *already published* before T.
    We achieve this by forward-filling the weekly (Sunday-dated) series to
    daily and then aligning on tz-naive dates.
    """
    df = daily_df.copy()

    # Normalise index to tz-naive for alignment
    idx = df.index
    if hasattr(idx, "tz") and idx.tz is not None:
        idx_naive = idx.tz_localize(None)
    else:
        idx_naive = pd.to_datetime(idx)
    idx_naive = idx_naive.normalize()

    # ------------------------------------------------------------------
    # Load trends data
    # ------------------------------------------------------------------
    trends = _get_trends_series(ticker)

    feature_cols = [
        "gtrends_score_7d_avg",
        "gtrends_score_30d_avg",
        "gtrends_zscore_60d",
        "gtrends_change_5d",
        "gtrends_pct_rank_252d",
        "gtrends_spike_flag",
        "gtrends_dropoff_flag",
    ]

    if trends is None:
        log.warning("No trends data for %s; inserting NaN columns.", ticker)
        for col in feature_cols:
            df[col] = np.nan
        return df

    # Align: reindex trends onto our bar dates (forward-fill gaps)
    score = trends.reindex(idx_naive).ffill().values.astype(float)
    score_s = pd.Series(score, index=df.index)

    # ------------------------------------------------------------------
    # Feature engineering (all rolling operations on score_s)
    # ------------------------------------------------------------------

    # Rolling averages
    df["gtrends_score_7d_avg"]  = score_s.rolling(7,  min_periods=1).mean()
    df["gtrends_score_30d_avg"] = score_s.rolling(30, min_periods=1).mean()

    # Z-score vs 60-day rolling window
    roll60_mean = score_s.rolling(60, min_periods=10).mean()
    roll60_std  = score_s.rolling(60, min_periods=10).std()
    zscore = (score_s - roll60_mean) / roll60_std
    df["gtrends_zscore_60d"] = zscore.fillna(0.0)

    # 5-day % change; avoid div-by-zero
    prior5 = score_s.shift(5)
    pct_chg = np.where(prior5 != 0, (score_s - prior5) / prior5 * 100, 0.0)
    df["gtrends_change_5d"] = pd.Series(pct_chg, index=df.index).fillna(0.0)

    # Percentile rank in trailing 252 business days
    df["gtrends_pct_rank_252d"] = (
        score_s
        .rolling(252, min_periods=20)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    )

    # Spike / drop-off flags
    df["gtrends_spike_flag"]   = (score_s > _SPIKE_THRESH).astype(int)
    df["gtrends_dropoff_flag"] = (score_s < _DROP_THRESH).astype(int)

    return df


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
    )

    print("=" * 60)
    print("Google Trends Features — smoke test")
    print("=" * 60)

    dates = pd.date_range("2024-01-01", "2024-12-31", freq="B", tz="UTC")
    base_df = pd.DataFrame({"close": 100.0}, index=dates)

    for tk in ["AAPL", "NVDA", "TSLA"]:
        out = add_google_trends_features(base_df.copy(), tk)
        new_cols = [c for c in out.columns if c not in base_df.columns]
        print(f"\n{tk}: +{len(new_cols)} trends features")
        for c in new_cols[:5]:
            if pd.api.types.is_numeric_dtype(out[c]):
                non_zero_pct = (out[c].notna() & (out[c] != 0)).mean() * 100
                print(f"  {c}: {non_zero_pct:.0f}% non-zero")

    print("\nSmoke test complete.")
