# model_reason: paper-trade feature expansion — FinBERT ProsusAI sentiment (Top-10 #6)
# autosolve_skip: ship FinBERT sentiment, success=news headline -> 3-class sentiment feature wired
"""
news_sentiment_finbert_features.py
===================================
FinBERT (ProsusAI/finbert) news sentiment features for S&P 500 backtests.

Distinct from `news_sentiment_features.py` (which uses VADER lexicon scoring).
This module uses the transformer-based FinBERT model fine-tuned for financial
news, producing 3-class probability outputs (positive / negative / neutral)
per headline.

All features are .shift(1)-safe: raw article timestamps are snapped to
calendar dates, aggregated, and shifted by 1 bar so feature value on bar t
reflects events known only through t-1.

Features added (6)
-------------------
finbert_pos_prob_mean_5d     : mean positive-class probability (trailing 5d)
finbert_neg_prob_mean_5d     : mean negative-class probability (trailing 5d)
finbert_neu_prob_mean_5d     : mean neutral-class probability (trailing 5d)
finbert_headline_count_5d    : number of scored headlines (trailing 5d)
finbert_weighted_sentiment   : pos - neg (latest day, recency-weighted EWMA)
finbert_rolling_5d_sentiment : rolling 5d mean of (pos - neg)

Data sources (in order)
-----------------------
1. Alpaca News API (v1beta1) -- requires APCA_API_KEY_ID / APCA_API_SECRET_KEY
   loaded from ~/.config/auto_signup/alpaca.env if present.
2. Yahoo Finance news (yfinance) -- fallback when Alpaca unavailable.

Caching
-------
Scored articles cached to:
  data/news_sentiment/<TICKER>/<YEAR>.parquet
relative to project root. Avoid re-running FinBERT on same headlines.

Activation
----------
Env-gated `NEWS_SENTIMENT_ENABLED=1` (default OFF). When OFF,
`add_finbert_sentiment_features()` is a no-op returning zero-filled columns.

Model
-----
ProsusAI/finbert (~440MB) -- HuggingFace download on first call to
~/.cache/huggingface/hub/. CPU inference, ~50ms/headline on M-series Mac.
Hard cap: 100 headlines per fetch to keep latency bounded.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "finbert_pos_prob_mean_5d",
    "finbert_neg_prob_mean_5d",
    "finbert_neu_prob_mean_5d",
    "finbert_headline_count_5d",
    "finbert_weighted_sentiment",
    "finbert_rolling_5d_sentiment",
]
MAX_HEADLINES_PER_RUN = 100  # cap FinBERT inference cost
CACHE_FRESH_HOURS = 24
EWMA_HALFLIFE_DAYS = 3.0
ALPACA_CRED_PATH = Path.home() / ".config" / "auto_signup" / "alpaca.env"
ALPACA_NEWS_URL = "https://data.alpaca.markets/v1beta1/news"


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _cache_dir(ticker: str) -> Path:
    p = _project_root() / "data" / "news_sentiment" / ticker.upper()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _cache_path(ticker: str, year: int) -> Path:
    return _cache_dir(ticker) / f"{year}.parquet"


def _cache_is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    age_h = (time.time() - path.stat().st_mtime) / 3600.0
    return age_h < CACHE_FRESH_HOURS


# ----------------------------------------------------------------------------
# FinBERT model loader (lazy, singleton)
# ----------------------------------------------------------------------------
_MODEL = None
_TOKENIZER = None


def _load_finbert():
    global _MODEL, _TOKENIZER
    if _MODEL is not None:
        return _TOKENIZER, _MODEL
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch  # noqa: F401

    logger.info("Loading FinBERT (ProsusAI/finbert) — first call downloads ~440MB")
    _TOKENIZER = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    # use_safetensors=True avoids the torch>=2.6 requirement introduced by
    # transformers >=4.50 for CVE-2025-32434 (we have torch 2.2.2).
    # ProsusAI/finbert publishes a safetensors variant (commit 0574315).
    try:
        _MODEL = AutoModelForSequenceClassification.from_pretrained(
            "ProsusAI/finbert", use_safetensors=True
        )
    except Exception as e:
        logger.warning(f"safetensors load failed ({e}); falling back to default")
        _MODEL = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    _MODEL.eval()
    return _TOKENIZER, _MODEL


def _score_headlines(headlines: list[str]) -> pd.DataFrame:
    """Run FinBERT on list of headlines. Returns DataFrame with pos/neg/neu probs."""
    import torch

    if not headlines:
        return pd.DataFrame(columns=["text", "pos", "neg", "neu"])
    headlines = headlines[:MAX_HEADLINES_PER_RUN]
    tok, model = _load_finbert()
    with torch.no_grad():
        enc = tok(headlines, padding=True, truncation=True, max_length=128,
                  return_tensors="pt")
        out = model(**enc)
        # Use .tolist() then np.asarray to dodge torch<->numpy ABI conflict
        # (torch 2.2.2 was compiled against numpy 1.x; env has numpy 2.x —
        # tensor.numpy() raises "Numpy is not available" but .tolist() is safe).
        probs_list = torch.softmax(out.logits, dim=1).tolist()
    probs = np.asarray(probs_list, dtype=np.float64)
    # FinBERT label order: 0=positive, 1=negative, 2=neutral
    df = pd.DataFrame({
        "text": headlines,
        "pos": probs[:, 0],
        "neg": probs[:, 1],
        "neu": probs[:, 2],
    })
    return df


# ----------------------------------------------------------------------------
# Alpaca news loader
# ----------------------------------------------------------------------------
def _load_alpaca_creds() -> tuple[Optional[str], Optional[str]]:
    key = os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY")
    secret = (os.environ.get("APCA_API_SECRET_KEY")
              or os.environ.get("ALPACA_API_SECRET"))
    if (not key or not secret) and ALPACA_CRED_PATH.exists():
        for line in ALPACA_CRED_PATH.read_text().splitlines():
            line = line.strip()
            if line.startswith("export ") and "=" in line:
                k, v = line[7:].split("=", 1)
                v = v.strip().strip('"').strip("'")
                os.environ.setdefault(k, v)
        key = os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY")
        secret = (os.environ.get("APCA_API_SECRET_KEY")
                  or os.environ.get("ALPACA_API_SECRET"))
    return key, secret


def _fetch_alpaca_news(ticker: str, days: int = 7) -> pd.DataFrame:
    key, secret = _load_alpaca_creds()
    if not key or not secret:
        logger.info("Alpaca creds unavailable — skipping Alpaca news fetch")
        return pd.DataFrame(columns=["date", "headline"])
    try:
        import requests
    except ImportError:
        logger.warning("requests not installed — skipping Alpaca news")
        return pd.DataFrame(columns=["date", "headline"])
    start = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    params = {"symbols": ticker.upper(), "start": start, "limit": 50,
              "sort": "desc"}
    try:
        r = requests.get(ALPACA_NEWS_URL, headers=headers, params=params, timeout=15)
        r.raise_for_status()
        data = r.json().get("news", [])
    except Exception as e:
        logger.warning(f"Alpaca news fetch failed for {ticker}: {e}")
        return pd.DataFrame(columns=["date", "headline"])
    rows = []
    for item in data:
        ts = item.get("created_at") or item.get("updated_at")
        headline = (item.get("headline") or "").strip()
        if not ts or not headline:
            continue
        try:
            d = pd.Timestamp(ts).tz_convert("UTC").normalize().tz_localize(None)
        except Exception:
            continue
        rows.append({"date": d, "headline": headline})
    return pd.DataFrame(rows)


def _fetch_yahoo_news(ticker: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        return pd.DataFrame(columns=["date", "headline"])
    try:
        news = yf.Ticker(ticker).news or []
    except Exception as e:
        logger.warning(f"Yahoo news fetch failed for {ticker}: {e}")
        return pd.DataFrame(columns=["date", "headline"])
    rows = []
    for item in news:
        # yfinance returns either flat keys or nested under 'content'
        content = item.get("content") if isinstance(item, dict) else None
        if isinstance(content, dict):
            headline = (content.get("title") or "").strip()
            ts = content.get("pubDate") or content.get("displayTime")
        else:
            headline = (item.get("title") or "").strip()
            ts_unix = item.get("providerPublishTime")
            ts = (datetime.fromtimestamp(ts_unix, tz=timezone.utc).isoformat()
                  if ts_unix else None)
        if not ts or not headline:
            continue
        try:
            d = pd.Timestamp(ts).tz_convert("UTC").normalize().tz_localize(None)
        except Exception:
            try:
                d = pd.Timestamp(ts).normalize()
            except Exception:
                continue
        rows.append({"date": d, "headline": headline})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Cache + scoring pipeline
# ----------------------------------------------------------------------------
def _fetch_and_score(ticker: str, days: int = 7) -> pd.DataFrame:
    df = _fetch_alpaca_news(ticker, days=days)
    if df.empty:
        df = _fetch_yahoo_news(ticker)
    if df.empty:
        return pd.DataFrame(columns=["date", "headline", "pos", "neg", "neu"])
    df = df.drop_duplicates(subset=["headline"]).reset_index(drop=True)
    headlines = df["headline"].tolist()
    scored = _score_headlines(headlines)
    df = df.iloc[:len(scored)].copy()
    df[["pos", "neg", "neu"]] = scored[["pos", "neg", "neu"]].values
    return df


def _load_or_fetch(ticker: str, days: int = 7) -> pd.DataFrame:
    year = datetime.now(timezone.utc).year
    cpath = _cache_path(ticker, year)
    if _cache_is_fresh(cpath):
        try:
            return pd.read_parquet(cpath)
        except Exception as e:
            logger.warning(f"Cache read failed {cpath}: {e}")
    df = _fetch_and_score(ticker, days=days)
    if not df.empty:
        try:
            if cpath.exists():
                old = pd.read_parquet(cpath)
                df = (pd.concat([old, df])
                      .drop_duplicates(subset=["headline"])
                      .reset_index(drop=True))
            df.to_parquet(cpath, index=False)
        except Exception as e:
            logger.warning(f"Cache write failed {cpath}: {e}")
    return df


# ----------------------------------------------------------------------------
# Feature aggregation
# ----------------------------------------------------------------------------
def _zero_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    out = daily_df.copy()
    for col in FEATURE_COLUMNS:
        out[col] = 0.0
    return out


def add_finbert_sentiment_features(daily_df: pd.DataFrame,
                                   ticker: str) -> pd.DataFrame:
    """Add 6 FinBERT sentiment features to a daily-price DataFrame.

    Env-gated: requires NEWS_SENTIMENT_ENABLED=1. Otherwise returns input with
    zero-filled feature columns.
    """
    if os.environ.get("NEWS_SENTIMENT_ENABLED", "0") != "1":
        logger.info("NEWS_SENTIMENT_ENABLED!=1 — returning zero-filled features")
        return _zero_features(daily_df)
    if daily_df is None or daily_df.empty:
        return _zero_features(daily_df if daily_df is not None else pd.DataFrame())

    scored = _load_or_fetch(ticker)
    if scored.empty:
        return _zero_features(daily_df)

    # Aggregate to daily
    g = scored.groupby("date")[["pos", "neg", "neu"]].mean()
    counts = scored.groupby("date").size().rename("count")
    daily = g.join(counts).sort_index()
    daily["net"] = daily["pos"] - daily["neg"]
    # EWMA weighted sentiment
    daily["weighted"] = (
        daily["net"].ewm(halflife=EWMA_HALFLIFE_DAYS, adjust=False).mean()
    )
    daily["rolling_5d"] = daily["net"].rolling(5, min_periods=1).mean()

    # Reindex onto daily_df dates (assume daily_df has DatetimeIndex)
    out = daily_df.copy()
    idx = pd.to_datetime(out.index).normalize() if not isinstance(
        out.index, pd.DatetimeIndex) else out.index.normalize()
    daily_reindexed = daily.reindex(idx).ffill().fillna(0.0)

    out["finbert_pos_prob_mean_5d"] = (
        daily_reindexed["pos"].rolling(5, min_periods=1).mean().values
    )
    out["finbert_neg_prob_mean_5d"] = (
        daily_reindexed["neg"].rolling(5, min_periods=1).mean().values
    )
    out["finbert_neu_prob_mean_5d"] = (
        daily_reindexed["neu"].rolling(5, min_periods=1).mean().values
    )
    out["finbert_headline_count_5d"] = (
        daily_reindexed["count"].fillna(0).rolling(5, min_periods=1).sum().values
    )
    out["finbert_weighted_sentiment"] = daily_reindexed["weighted"].values
    out["finbert_rolling_5d_sentiment"] = daily_reindexed["rolling_5d"].values

    # .shift(1)-safety
    for col in FEATURE_COLUMNS:
        out[col] = out[col].shift(1).fillna(0.0)

    return out


__all__ = [
    "add_finbert_sentiment_features",
    "FEATURE_COLUMNS",
]
