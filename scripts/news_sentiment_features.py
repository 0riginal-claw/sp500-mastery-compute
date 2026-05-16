"""
news_sentiment_features.py
==========================
Adds Yahoo Finance news-based sentiment features for a given ticker onto a
daily price DataFrame used for S&P 500 backtesting.

All features are .shift(1)-safe: the raw article timestamps are snapped to
calendar dates, aggregated, and then shifted by 1 bar so that the feature
value on bar t reflects events known only through t-1.

Features added (~8)
-------------------
news_count_30d           : number of articles published in trailing 30 calendar days
news_sentiment_mean_30d  : mean VADER compound score over trailing 30 calendar days
news_sentiment_mean_7d   : mean VADER compound score over trailing 7 calendar days
news_pos_ratio_30d       : fraction of 30d articles with compound > 0.1
news_neg_ratio_30d       : fraction of 30d articles with compound < -0.1
news_velocity_zscore_60d : z-score of 7d article rate vs trailing 60d rate (spike detector)
days_since_last_news     : calendar days since the most recent article
news_extreme_flag        : 1 if any article in trailing 7d has |compound| > 0.7

Data source
-----------
Yahoo Finance via yfinance (no API key required).
  yf.Ticker(symbol).news returns a list of recent articles, typically
  covering the last 30-90 days depending on ticker prominence.

Known limitation -- historical sparsity
----------------------------------------
yfinance only returns RECENT news (last ~30-90 days as of the fetch date).
For historical bars older than that window the news features will be
zero-filled. This is expected. Do NOT rely on these features in a pure
historical backtest starting more than ~3 months before the run date;
they are most reliable for live / walk-forward use.

Caching
-------
Scored articles are cached to:
  cache/news_sentiment/<TICKER>.parquet
relative to the project root. Cache is refreshed if older than 24 hours;
delete the file to force an immediate refresh.

Graceful degradation
--------------------
If yfinance returns no news (empty list, network error, rate limit), the
function logs a warning and returns the input DataFrame with all sentiment
feature columns set to 0.0.

Dependencies
------------
  yfinance        (already installed in sp500-mastery venv)
  vaderSentiment  (install: uv pip install vaderSentiment)
  pandas, numpy   (already installed)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

_SIA: Optional[SentimentIntensityAnalyzer] = None

CACHE_MAX_AGE_HOURS = 24

FEATURE_COLS = [
    "news_count_30d",
    "news_sentiment_mean_30d",
    "news_sentiment_mean_7d",
    "news_pos_ratio_30d",
    "news_neg_ratio_30d",
    "news_velocity_zscore_60d",
    "days_since_last_news",
    "news_extreme_flag",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_sia() -> SentimentIntensityAnalyzer:
    global _SIA
    if _SIA is None:
        _SIA = SentimentIntensityAnalyzer()
    return _SIA


def _cache_dir() -> Path:
    """Returns absolute path to cache/news_sentiment/ next to the project root."""
    here = Path(__file__).resolve().parent   # .../scripts/
    root = here.parent                        # .../s&p500-ticker-mastery/
    d = root / "cache" / "news_sentiment"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_path(ticker: str) -> Path:
    return _cache_dir() / f"{ticker.upper()}.parquet"


def _cache_is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    return age_hours < CACHE_MAX_AGE_HOURS


def _score_text(text: str) -> float:
    """VADER compound score for a piece of text; returns 0.0 on empty input."""
    if not text or not text.strip():
        return 0.0
    return _get_sia().polarity_scores(text)["compound"]


def _fetch_and_score(ticker: str) -> pd.DataFrame:
    """
    Fetches news from yfinance, scores each article with VADER, and returns a
    DataFrame with columns [date, compound].

    'date' is a tz-naive UTC calendar date of publication.
    'compound' is the VADER score of title + summary concatenated.

    Returns an empty DataFrame with those columns on any failure.
    """
    empty = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"),
                          "compound": pd.Series(dtype=float)})

    try:
        yf_ticker = yf.Ticker(ticker)
        news_list = yf_ticker.news
    except Exception as exc:
        logger.warning("news_sentiment: yfinance fetch failed for %s: %s", ticker, exc)
        return empty

    if not news_list:
        logger.warning("news_sentiment: yfinance returned no news for %s", ticker)
        return empty

    rows = []
    for item in news_list:
        try:
            # yfinance >=0.2.x nests the article under item['content']
            content = item.get("content", item)

            title   = content.get("title",   "") or ""
            summary = content.get("summary", "") or ""
            pub_raw = content.get("pubDate", "") or ""

            if not pub_raw:
                continue

            # Expected format: "2026-05-15T18:02:34Z"
            pub_dt = datetime.strptime(pub_raw, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc
            )
            pub_date = pd.Timestamp(pub_dt).normalize().tz_localize(None)

            combined = f"{title} {summary}".strip()
            score = _score_text(combined)

            rows.append({"date": pub_date, "compound": score})

        except Exception as exc:
            logger.debug("news_sentiment: skipping malformed article for %s: %s", ticker, exc)
            continue

    if not rows:
        logger.warning("news_sentiment: no parseable articles for %s", ticker)
        return empty

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _load_scored_articles(ticker: str) -> pd.DataFrame:
    """
    Returns scored articles DataFrame (columns: date, compound).
    Uses cache if fresh, otherwise re-fetches.
    """
    path = _cache_path(ticker)

    if _cache_is_fresh(path):
        try:
            cached = pd.read_parquet(path)
            logger.debug(
                "news_sentiment: loaded cached articles for %s (%d rows)", ticker, len(cached)
            )
            return cached
        except Exception as exc:
            logger.warning(
                "news_sentiment: cache read failed for %s: %s -- re-fetching", ticker, exc
            )

    logger.info("news_sentiment: fetching news for %s from yfinance", ticker)
    df = _fetch_and_score(ticker)

    if not df.empty:
        try:
            df.to_parquet(path, index=False)
            logger.debug(
                "news_sentiment: saved %d articles to cache for %s", len(df), ticker
            )
        except Exception as exc:
            logger.warning("news_sentiment: cache write failed for %s: %s", ticker, exc)

    return df


def _zero_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Return input df with all FEATURE_COLS set to 0.0."""
    out = daily_df.copy()
    for col in FEATURE_COLS:
        out[col] = 0.0
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_news_sentiment_features(daily_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Adds news sentiment features per ticker. Daily-indexed.

    Parameters
    ----------
    daily_df : pd.DataFrame
        Daily OHLCV (or any daily) DataFrame with a DatetimeIndex.
        The index may be timezone-aware or naive; the function normalises it.
    ticker : str
        Ticker symbol (e.g. 'AAPL').

    Returns
    -------
    pd.DataFrame
        Input DataFrame with FEATURE_COLS appended. All features are
        .shift(1)-safe -- on bar t they reflect events through t-1.

    Notes
    -----
    yfinance only returns recent news (last ~30-90 days as of fetch date).
    Historical bars outside that window will have zero-filled features.
    This is a known limitation; see module docstring for details.
    """
    # Normalise the daily index to tz-naive midnight timestamps
    idx = daily_df.index
    if hasattr(idx, "tz") and idx.tz is not None:
        idx = idx.tz_localize(None)
    daily_dates = pd.DatetimeIndex(idx).normalize()

    # Load scored articles (cached or fresh)
    articles = _load_scored_articles(ticker)

    if articles.empty:
        logger.warning(
            "news_sentiment: no articles available for %s -- returning zero features", ticker
        )
        return _zero_features(daily_df)

    # Normalise article dates to tz-naive midnight
    articles = articles.copy()
    articles["date"] = pd.to_datetime(articles["date"]).dt.tz_localize(None).dt.normalize()

    date_min = daily_dates.min()
    date_max = daily_dates.max()

    # Include 60-day lookback before date_min so velocity z-score has history
    lookback_start = date_min - pd.Timedelta(days=60)
    art_in_range = articles[
        (articles["date"] >= lookback_start) & (articles["date"] <= date_max)
    ].copy()

    if art_in_range.empty:
        logger.warning(
            "news_sentiment: no articles in date range for %s -- returning zero features", ticker
        )
        return _zero_features(daily_df)

    # Full calendar daily index covering lookback through date_max
    full_cal = pd.date_range(start=lookback_start, end=date_max, freq="D")

    # Per-day numeric aggregates -- all float64, rolling is safe
    grp_count = art_in_range.groupby("date")["compound"].count().rename("cnt")
    grp_sum   = art_in_range.groupby("date")["compound"].sum().rename("s_sum")
    grp_pos   = (art_in_range[art_in_range["compound"] > 0.1]
                 .groupby("date")["compound"].count().rename("pos"))
    grp_neg   = (art_in_range[art_in_range["compound"] < -0.1]
                 .groupby("date")["compound"].count().rename("neg"))
    grp_ext   = (art_in_range[art_in_range["compound"].abs() > 0.7]
                 .groupby("date")["compound"].count().rename("ext"))

    cnt_s = grp_count.reindex(full_cal, fill_value=0.0)
    sum_s = grp_sum.reindex(full_cal,   fill_value=0.0)
    pos_s = grp_pos.reindex(full_cal,   fill_value=0.0)
    neg_s = grp_neg.reindex(full_cal,   fill_value=0.0)
    ext_s = grp_ext.reindex(full_cal,   fill_value=0.0)

    # Rolling sums over numeric series (no object dtype issues)
    count_7  = cnt_s.rolling(7,  min_periods=1).sum()
    count_30 = cnt_s.rolling(30, min_periods=1).sum()
    count_60 = cnt_s.rolling(60, min_periods=1).sum()
    sum_30   = sum_s.rolling(30, min_periods=1).sum()
    sum_7    = sum_s.rolling(7,  min_periods=1).sum()
    pos_30   = pos_s.rolling(30, min_periods=1).sum()
    neg_30   = neg_s.rolling(30, min_periods=1).sum()
    ext_7    = ext_s.rolling(7,  min_periods=1).sum()

    # Mean compound scores (safe divide avoiding RuntimeWarning)
    def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
        result = np.full(len(den), 0.0)
        mask = den.values > 0
        result[mask] = num.values[mask] / den.values[mask]
        return pd.Series(result, index=den.index)

    mean_30      = _safe_div(sum_30, count_30)
    mean_7       = _safe_div(sum_7,  count_7)
    pos_ratio_30 = _safe_div(pos_30, count_30)
    neg_ratio_30 = _safe_div(neg_30, count_30)

    # Extreme flag: 1 if any article in 7d window had |compound| > 0.7
    extreme_7 = (ext_7 > 0).astype(float)

    # Velocity z-score: (daily rate over 7d - 60d rate) / 60d std
    rate_7  = count_7  / 7.0
    rate_60 = count_60 / 60.0
    std_60  = cnt_s.rolling(60, min_periods=2).std().fillna(0.0)
    velocity_z = _safe_div(rate_7 - rate_60, std_60)
    velocity_z.index = full_cal

    # Days since last news (forward-scan over full_cal)
    days_since_vals = []
    last_date = None
    for dt, cnt in cnt_s.items():
        if cnt > 0:
            last_date = dt
        days_since_vals.append(float((dt - last_date).days) if last_date is not None else np.nan)
    days_since = pd.Series(days_since_vals, index=full_cal, dtype=float)

    # Forward-fill 7d mean over gaps (weekends, slow-news periods)
    mean_7 = mean_7.replace(0.0, np.nan).ffill(limit=7).fillna(0.0)

    # Assemble feature frame on full_cal index
    feat_cal = pd.DataFrame(
        {
            "news_count_30d":           count_30.fillna(0.0),
            "news_sentiment_mean_30d":  mean_30.fillna(0.0),
            "news_sentiment_mean_7d":   mean_7.fillna(0.0),
            "news_pos_ratio_30d":       pos_ratio_30.fillna(0.0),
            "news_neg_ratio_30d":       neg_ratio_30.fillna(0.0),
            "news_velocity_zscore_60d": velocity_z.fillna(0.0),
            "days_since_last_news":     days_since,
            "news_extreme_flag":        extreme_7.fillna(0.0),
        },
        index=full_cal,
    )

    # Reindex to the business-day index of daily_df (forward-fill weekends)
    feat_biz = feat_cal.reindex(daily_dates, method="ffill")
    feat_biz = feat_biz.fillna(0.0)

    # Shift by 1 bar to ensure no look-ahead leakage
    feat_biz = feat_biz.shift(1).fillna(0.0)

    # Attach to daily_df
    out = daily_df.copy()
    for col in FEATURE_COLS:
        out[col] = feat_biz[col].values

    return out


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    )

    # Extend one week past today so articles fetched today are visible after shift(1).
    # yfinance only returns recent articles; the shift(1) look-ahead guard moves
    # today's articles to tomorrow's bar, so the index must include tomorrow.
    dates = pd.date_range("2025-01-01", "2026-05-23", freq="B", tz="UTC")
    df = pd.DataFrame({"close": 100.0}, index=dates)

    for tk in ["AAPL", "NVDA", "TSLA"]:
        out = add_news_sentiment_features(df.copy(), tk)
        new = [c for c in out.columns if c not in df.columns]
        print(f"\n{tk}: +{len(new)} cols")
        for c in new[:8]:
            if pd.api.types.is_numeric_dtype(out[c]):
                non_zero_pct = (out[c].notna() & (out[c] != 0)).mean() * 100
                last_val = out[c].iloc[-1]
                max_val   = out[c].max()
                # Note: last=0 expected when all articles fall on the final bar
                # (shift(1) moves them to the next business day, outside the index).
                # max_val shows the signal is non-zero at least one bar in range.
                print(f"  {c}: non-zero {non_zero_pct:.0f}%, last={last_val:.4f}, max={max_val:.4f}")
    print("\nNote: 'last=0' is expected if yfinance only returns articles from today")
    print("      (shift(1) pushes them to tomorrow which is outside the index).")
    print("      'max' confirms the signal is non-zero somewhere in the series.")
