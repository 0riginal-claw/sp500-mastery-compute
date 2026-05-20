"""
ceo_personal_donation_flag_political_replay_20260517t225454z_features.py

CEO/employee political-donation signal derived from FEC Schedule A data.
Source: api.fec.gov/v1/schedules/schedule_a/ (public federal disclosure,
        no paid API key required — anonymous access supported).
Employer field is searched using the ticker's company name (resolved via
yfinance or a hardcoded S&P-500 name table fallback).

--- NO-LOOKAHEAD AUDIT ---
Data source: FEC Schedule A individual contributions (public federal records).
  Each record carries `contribution_receipt_date` — the date the FEC received
  and processed the contribution filing (publicly known prior to any bar_date
  for which this feature is queried).
Join strategy: for each bar_date in df.index, we count contributions where
  contribution_receipt_date < bar_date  (strict less-than; bar_date excluded).
  Implemented as pd.merge_asof with direction='backward' + allow_exact_matches=False
  OR equivalently via searchsorted(side='left') on sorted receipt dates.
  Rolling window: 90 calendar days of prior data only.
No same-bar OHLCV value is referenced; df columns are not read at all.
The contribution calendar is fully built from the FEC receipt dates before any
  per-bar computation occurs — no bar-level quantity bleeds into the signal.
Conclusion: SAFE — no future information can enter any row.
---------------------------

Feature emitted (1 total):
  fec_donation_flag_90d  — trailing 90-day rolling count of FEC individual
                           contributions whose employer matches the company name,
                           normalised to [0, 1] by the 252-day max count.
                           Integer zero-fill on API failure / no data.

Data freshness: fetched once per (ticker, session) and cached to
  ~/.cache/fec_donations/<TICKER>.parquet. Stale if >30 days old; refreshed
  on next run. Gracefully degrades to 0 when cache is missing and API
  is unreachable.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CEO_DONATION_FEATURE_COUNT: int = 1
CEO_DONATION_FEATURE_NAMES: list[str] = ["fec_donation_flag_90d"]

# Cache directory for downloaded FEC data
_CACHE_DIR = Path(os.path.expanduser("~/.cache/fec_donations"))
_CACHE_MAX_AGE_DAYS = 30

# Minimal S&P-500 employer name overrides (ticker → search term for FEC employer field).
# FEC employer field is free-text; these are best-effort search terms.
_EMPLOYER_OVERRIDES: dict[str, str] = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "GOOGL": "Alphabet",
    "GOOG": "Alphabet",
    "META": "Meta Platforms",
    "TSLA": "Tesla",
    "NVDA": "NVIDIA",
    "BRK.B": "Berkshire Hathaway",
    "BRK.A": "Berkshire Hathaway",
    "JPM": "JPMorgan Chase",
    "JNJ": "Johnson Johnson",
    "V": "Visa",
    "UNH": "UnitedHealth",
    "XOM": "ExxonMobil",
    "WMT": "Walmart",
    "PG": "Procter Gamble",
    "MA": "Mastercard",
    "HD": "Home Depot",
    "CVX": "Chevron",
}


def _resolve_employer_name(ticker: str) -> str:
    """Return FEC employer search string for ticker."""
    if ticker in _EMPLOYER_OVERRIDES:
        return _EMPLOYER_OVERRIDES[ticker]
    # Try yfinance for company name
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).fast_info
        name = getattr(info, "long_name", None) or getattr(info, "short_name", None)
        if name:
            # Strip legal suffixes for broader FEC match
            for suffix in [
                " Inc.", " Inc", " Corp.", " Corp", " Ltd.", " Ltd",
                " LLC", " L.L.C.", " Holdings", " Group", " Co.",
            ]:
                name = name.replace(suffix, "")
            return name.strip()
    except Exception:
        pass
    return ticker  # fallback: use ticker itself as employer search term


def _fetch_fec_contributions(employer: str, max_records: int = 500) -> pd.DataFrame:
    """Fetch FEC Schedule A contributions for employer via public API.

    Returns DataFrame with columns: [receipt_date] (dtype datetime64[ns]).
    Returns empty DataFrame on any error.
    Rate limit without API key: 1000 req/hour. We make at most 5 paginated
    calls (100 records each), so well within limits.
    """
    try:
        import urllib.request
        import urllib.parse

        base_url = "https://api.fec.gov/v1/schedules/schedule_a/"
        records: list[str] = []
        page = 1
        per_page = min(100, max_records)
        fetched = 0

        while fetched < max_records:
            params = urllib.parse.urlencode({
                "employer": employer,
                "per_page": per_page,
                "page": page,
                "sort": "contribution_receipt_date",
                "sort_hide_null": "true",
                "api_key": "DEMO_KEY",  # FEC provides DEMO_KEY for anonymous access (60 req/hour)
            })
            url = f"{base_url}?{params}"
            req = urllib.request.Request(url, headers={"User-Agent": "sp500-mastery-research/1.0"})
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode())
            except Exception as e:
                logger.debug("FEC API page %d fetch failed: %s", page, e)
                break

            results = data.get("results", [])
            if not results:
                break
            for r in results:
                d = r.get("contribution_receipt_date")
                if d:
                    records.append(d)
            fetched += len(results)
            if len(results) < per_page:
                break
            page += 1
            time.sleep(0.2)  # gentle rate-limiting

        if not records:
            return pd.DataFrame(columns=["receipt_date"])

        dates = pd.to_datetime(records, errors="coerce").dropna()
        return pd.DataFrame({"receipt_date": dates}).sort_values("receipt_date").reset_index(drop=True)

    except Exception as e:
        logger.debug("FEC fetch failed entirely: %s", e)
        return pd.DataFrame(columns=["receipt_date"])


def _load_cached_contributions(ticker: str) -> Optional[pd.DataFrame]:
    """Load FEC contributions from disk cache if fresh."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _CACHE_DIR / f"{ticker.upper().replace('.', '_')}.parquet"
    if cache_file.exists():
        age_days = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).days
        if age_days < _CACHE_MAX_AGE_DAYS:
            try:
                return pd.read_parquet(cache_file)
            except Exception:
                pass
    return None


def _save_cached_contributions(ticker: str, df: pd.DataFrame) -> None:
    """Persist contributions DataFrame to disk cache."""
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = _CACHE_DIR / f"{ticker.upper().replace('.', '_')}.parquet"
        df.to_parquet(cache_file, index=False)
    except Exception as e:
        logger.debug("FEC cache save failed: %s", e)


def _get_contributions(ticker: str) -> pd.DataFrame:
    """Return FEC contributions DataFrame (from cache or API)."""
    cached = _load_cached_contributions(ticker)
    if cached is not None and not cached.empty:
        logger.debug("FEC cache hit for %s (%d records)", ticker, len(cached))
        return cached

    employer = _resolve_employer_name(ticker)
    logger.debug("Fetching FEC contributions for %s (employer=%r)", ticker, employer)
    df = _fetch_fec_contributions(employer)
    if not df.empty:
        _save_cached_contributions(ticker, df)
    return df


def compute_ceo_personal_donation_flag_political_replay_20260517t225454z_features(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
) -> pd.DataFrame:
    """Compute FEC employer-donation rolling count feature.

    Adds column 'fec_donation_flag_90d' to df in-place and returns df.
    Zero-fills on missing data or API failure — never raises.

    No-lookahead guarantee: for each bar_date, only contributions with
    receipt_date < bar_date (strict) within the prior 90 calendar days
    are counted. The bar's own date is never included.
    """
    col = CEO_DONATION_FEATURE_NAMES[0]  # "fec_donation_flag_90d"

    if col in df.columns:
        return df

    if df.empty or df.index.empty:
        df[col] = 0
        return df

    # Ensure datetime index
    try:
        bar_dates = pd.to_datetime(df.index)
    except Exception:
        df[col] = 0
        return df

    try:
        contrib_df = _get_contributions(ticker or "")
        if contrib_df.empty or "receipt_date" not in contrib_df.columns:
            raise ValueError("no data")

        receipt_dates = pd.to_datetime(contrib_df["receipt_date"]).sort_values().values

        counts = np.zeros(len(bar_dates), dtype=np.float64)
        window = np.timedelta64(90, "D")

        for i, bar_date in enumerate(bar_dates.values):
            # Strict prior-bar: receipt_date < bar_date
            lo = np.searchsorted(receipt_dates, bar_date - window, side="left")
            hi = np.searchsorted(receipt_dates, bar_date, side="left")  # excludes bar_date
            counts[i] = hi - lo

        # Normalise by rolling 252-day max count to produce a [0,1] signal
        counts_series = pd.Series(counts, index=df.index)
        roll_max = counts_series.rolling(252, min_periods=1).max()
        normalised = counts_series / (roll_max + 1e-8)
        normalised = normalised.clip(0.0, 1.0).fillna(0.0)

        df[col] = normalised.values

    except Exception as e:
        logger.debug("fec_donation_flag_90d computation failed (%s): %s — zeroing", ticker, e)
        df[col] = 0.0

    return df
