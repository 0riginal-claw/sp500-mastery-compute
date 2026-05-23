"""
earnings_transcript_features.py
===============================
Adds earnings-call-transcript sentiment features to a daily price DataFrame
for S&P 500 backtesting, using defeatbeta-api.

Source: GitHub TOP-10 #7 — defeatbeta-api
  pip install defeatbeta-api

Cadence
-------
Earnings call transcripts are quarterly. Sentiment per transcript is
computed once and forward-filled for ~90 calendar days onto the daily panel.
A `.shift(1)` ensures bar t reflects only info known at t-1.

Features added (4 columns, prefixed `earn_`)
--------------------------------------------
earn_transcript_sentiment   : VADER compound score of full transcript text
                              (FinBERT fallback when wired separately, #6)
earn_transcript_pos_ratio   : fraction of sentences with compound > 0.1
earn_transcript_neg_ratio   : fraction of sentences with compound < -0.1
earn_days_since_transcript  : calendar days since the most recent transcript

Env gate
--------
EARNINGS_TRANSCRIPT_ENABLED=1 to activate. Default OFF (returns df unchanged
with all `earn_*` columns set to 0.0).

Graceful degradation
--------------------
If defeatbeta-api fails (network, ticker missing, empty transcripts), the
function logs a warning and returns the input df with all `earn_*` columns
filled to 0.0.

Dependencies
------------
  defeatbeta-api  (pip install defeatbeta-api)
  vaderSentiment  (already installed for news_sentiment_features.py)
  pandas, numpy   (already in sp500-mastery venv)
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ENV_FLAG = "EARNINGS_TRANSCRIPT_ENABLED"
TRANSCRIPT_HOLD_DAYS = 90  # forward-fill horizon

FEATURE_COLS: List[str] = [
    "earn_transcript_sentiment",
    "earn_transcript_pos_ratio",
    "earn_transcript_neg_ratio",
    "earn_days_since_transcript",
]

_SIA = None  # vaderSentiment singleton


def _get_sia():
    global _SIA
    if _SIA is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        _SIA = SentimentIntensityAnalyzer()
    return _SIA


def _zero_fill(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in FEATURE_COLS:
        if c not in out.columns:
            out[c] = 0.0
    return out


def _score_text(text: str) -> Tuple[float, float, float]:
    """Returns (compound, pos_ratio, neg_ratio) for a transcript string."""
    if not text or not isinstance(text, str):
        return 0.0, 0.0, 0.0
    sia = _get_sia()
    # Split on sentence-ish boundaries; cheap + portable
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    if not sentences:
        return 0.0, 0.0, 0.0
    scores = [sia.polarity_scores(s)["compound"] for s in sentences]
    arr = np.asarray(scores)
    compound = float(arr.mean())
    pos = float((arr > 0.1).mean())
    neg = float((arr < -0.1).mean())
    return compound, pos, neg


def _fetch_transcripts(ticker: str, max_quarters: int = 24) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
      date (Timestamp), sentiment (float), pos_ratio (float), neg_ratio (float)
    One row per quarterly earnings call (most recent `max_quarters`). Empty if
    no transcripts found.

    defeatbeta-api two-step pattern:
      t.earning_call_transcripts() -> Transcripts wrapper
      .get_transcripts_list() -> DataFrame [symbol, fiscal_year, fiscal_quarter, report_date]
      .get_transcript(year, quarter) -> DataFrame [paragraph_number, speaker, content]
    """
    from defeatbeta_api.data.ticker import Ticker

    t = Ticker(ticker)
    wrapper = t.earning_call_transcripts()
    try:
        idx = wrapper.get_transcripts_list()
    except Exception as e:
        logger.warning("defeatbeta get_transcripts_list failed for %s: %s", ticker, e)
        return pd.DataFrame(columns=["date", "sentiment", "pos_ratio", "neg_ratio"])

    if idx is None or not isinstance(idx, pd.DataFrame) or idx.empty:
        return pd.DataFrame(columns=["date", "sentiment", "pos_ratio", "neg_ratio"])

    # Keep the most recent N transcripts
    idx_sorted = idx.sort_values("report_date").tail(max_quarters)

    rows = []
    for _, meta in idx_sorted.iterrows():
        try:
            year = int(meta["fiscal_year"])
            quarter = int(meta["fiscal_quarter"])
            d = pd.to_datetime(meta["report_date"]).normalize()
            try:
                tx_df = wrapper.get_transcript(year, quarter)
            except Exception as e:
                logger.debug("get_transcript(%d,%d) failed for %s: %s", year, quarter, ticker, e)
                continue
            if not isinstance(tx_df, pd.DataFrame) or tx_df.empty or "content" not in tx_df.columns:
                continue
            full_text = " ".join(str(x) for x in tx_df["content"].dropna().tolist())
            if not full_text.strip():
                continue
            compound, pos, neg = _score_text(full_text)
            rows.append(
                {"date": d, "sentiment": compound, "pos_ratio": pos, "neg_ratio": neg}
            )
        except Exception as e:
            logger.debug("skip transcript meta row: %s", e)
            continue

    if not rows:
        return pd.DataFrame(columns=["date", "sentiment", "pos_ratio", "neg_ratio"])
    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return out


def add_earnings_transcript_features(
    df: pd.DataFrame,
    ticker: str,
    date_col: str = "Date",
) -> pd.DataFrame:
    """
    Merge earnings-transcript sentiment features onto a daily-bar DataFrame.

    Forward-fills each transcript's sentiment for `TRANSCRIPT_HOLD_DAYS`
    calendar days. After the hold window expires, sentiment columns decay
    to 0.0 and `earn_days_since_transcript` keeps incrementing until the
    next transcript.

    Returns the input df with `earn_*` columns appended, shifted by 1 bar
    to enforce no-lookahead.
    """
    if os.getenv(ENV_FLAG, "0") != "1":
        return _zero_fill(df)

    if df is None or len(df) == 0:
        return df

    try:
        tdf = _fetch_transcripts(ticker)
    except Exception as e:
        logger.warning("defeatbeta-api transcript fetch failed for %s: %s", ticker, e)
        return _zero_fill(df)

    if tdf.empty:
        logger.info("No transcripts returned for %s", ticker)
        return _zero_fill(df)

    # Build daily series
    if isinstance(df.index, pd.DatetimeIndex):
        daily_idx = df.index
    else:
        if date_col not in df.columns:
            logger.warning("date_col '%s' not in df; zero-filling", date_col)
            return _zero_fill(df)
        daily_idx = pd.DatetimeIndex(pd.to_datetime(df[date_col]))

    panel = pd.DataFrame(
        0.0,
        index=daily_idx.sort_values(),
        columns=["earn_transcript_sentiment", "earn_transcript_pos_ratio", "earn_transcript_neg_ratio"],
    )
    days_since = pd.Series(np.nan, index=panel.index, dtype=float)

    # Apply each transcript's values forward for TRANSCRIPT_HOLD_DAYS
    for _, row in tdf.iterrows():
        d0 = row["date"]
        d1 = d0 + pd.Timedelta(days=TRANSCRIPT_HOLD_DAYS)
        mask = (panel.index >= d0) & (panel.index <= d1)
        panel.loc[mask, "earn_transcript_sentiment"] = row["sentiment"]
        panel.loc[mask, "earn_transcript_pos_ratio"] = row["pos_ratio"]
        panel.loc[mask, "earn_transcript_neg_ratio"] = row["neg_ratio"]

    # days_since_transcript: nearest past transcript distance (NaN if none)
    transcript_dates = pd.DatetimeIndex(tdf["date"].values)
    for i, d in enumerate(panel.index):
        past = transcript_dates[transcript_dates <= d]
        if len(past) > 0:
            days_since.iat[i] = (d - past.max()).days
    panel["earn_days_since_transcript"] = days_since.fillna(9999.0)

    # Shift by 1 bar
    panel = panel.shift(1).fillna(0.0)
    # Re-align to original df ordering
    panel = panel.reindex(daily_idx).fillna(0.0)

    out = df.copy()
    for c in FEATURE_COLS:
        out[c] = panel[c].values

    return out


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    tkr = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    os.environ[ENV_FLAG] = "1"
    dates = pd.date_range("2023-01-01", periods=600, freq="B")
    sample = pd.DataFrame({"Date": dates, "Close": np.linspace(150, 200, 600)})
    out = add_earnings_transcript_features(sample, tkr, date_col="Date")
    earn_cols = [c for c in out.columns if c.startswith("earn_")]
    print(f"ticker={tkr} rows={len(out)} earn_cols={len(earn_cols)}")
    print("col list:", earn_cols)
    print("nonzero head:")
    nz = out[out["earn_transcript_sentiment"] != 0]
    print(nz[earn_cols].head(3) if len(nz) else "(all zero — transcript may be unavailable)")
