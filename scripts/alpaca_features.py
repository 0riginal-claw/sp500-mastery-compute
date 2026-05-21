"""
alpaca_features.py
==================
Alpaca-derived (and companion yfinance) features for the XGBoost pipeline.

All features are .shift(1)-safe: every feature value on bar t reflects only
information available through end-of-day t-1.

No Alpaca API key required: credentials are not present in the Keychain.
All data sourced from:
  1. yfinance (earnings dates, splits, dividends, asset metadata)
  2. Locally cached news_sentiment parquets (existing cache)

Features added (up to 13)
--------------------------
Earnings proximity
  days_until_earnings      : trading days until next known earnings event
  is_earnings_week         : 1 if earnings within next 5 trading days
  earnings_surprise_last   : EPS surprise (%) from most recent reported earnings
  days_since_last_earnings : calendar days since last reported earnings

Ex-dividend / dividend
  ex_div_proximity         : 1/(1 + calendar days to next ex-div), 0 if none known
  days_since_last_exdiv    : calendar days since last ex-dividend date
  div_yield_trailing       : trailing-12-month dividend yield (div/price, annualised)
  dividend_growth_yoy      : % change in last dividend vs 1yr-ago dividend

Corporate actions (splits)
  days_since_last_split    : calendar days since most recent split; capped at 730
  is_post_split_60d        : 1 if a split occurred within the past 60 calendar days

Asset metadata (static per ticker, no API key)
  log_market_cap           : log10(marketCap); 0 if unavailable
  short_interest_pct       : short interest as % of float; 0 if unavailable
  sector_encoded           : ordinal encoding of GICS sector (0-11)

Caching
-------
Each data type is cached to cache/alpaca_features/<TICKER>_<type>.parquet
under the project WORK directory. Cache refreshes if older than 24 hours.

Graceful degradation
--------------------
Any fetch failure silently zero-fills that feature group. The pipeline
must not raise on network errors or missing data.
"""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
WORK = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools/s&p500-ticker-mastery"
)
CACHE_DIR = WORK / "cache" / "alpaca_features"
CACHE_MAX_AGE_HOURS = 24

# ---------------------------------------------------------------------------
# GICS sector ordinal map (static)
# ---------------------------------------------------------------------------
SECTOR_MAP: dict[str, int] = {
    "Technology": 0,
    "Health Care": 1,
    "Financials": 2,
    "Consumer Discretionary": 3,
    "Consumer Cyclical": 3,  # yfinance alias
    "Communication Services": 4,
    "Industrials": 5,
    "Consumer Staples": 6,
    "Energy": 7,
    "Utilities": 8,
    "Real Estate": 9,
    "Materials": 10,
    "Basic Materials": 10,  # yfinance alias
}


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, dtype: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{ticker.upper()}_{dtype}.parquet"


def _cache_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    age_h = (time.time() - path.stat().st_mtime) / 3600.0
    return age_h < CACHE_MAX_AGE_HOURS


def _save_cache(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path, index=True)
    except Exception as exc:
        logger.warning("alpaca_features: cache write failed %s: %s", path.name, exc)


def _load_cache(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        logger.warning("alpaca_features: cache read failed %s: %s", path.name, exc)
        return None


# ---------------------------------------------------------------------------
# yfinance fetch helpers (each returns a small DataFrame or None on error)
# ---------------------------------------------------------------------------
def _fetch_earnings(ticker: str) -> Optional[pd.DataFrame]:
    """Fetch earnings_dates from yfinance. Returns DataFrame indexed by date."""
    cp = _cache_path(ticker, "earnings")
    if _cache_fresh(cp):
        return _load_cache(cp)
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        ed = tk.earnings_dates
        if ed is None or len(ed) == 0:
            return None
        # Normalize index to UTC date
        ed = ed.copy()
        ed.index = pd.to_datetime(ed.index, utc=True).normalize()
        ed.index.name = "date"
        _save_cache(ed, cp)
        return ed
    except Exception as exc:
        logger.warning("alpaca_features: earnings fetch failed %s: %s", ticker, exc)
        return None


def _fetch_splits(ticker: str) -> Optional[pd.Series]:
    """Fetch split history. Returns Series indexed by UTC timestamp."""
    cp = _cache_path(ticker, "splits")
    if _cache_fresh(cp):
        df = _load_cache(cp)
        return df["ratio"] if df is not None else None
    try:
        import yfinance as yf
        splits = yf.Ticker(ticker).splits
        if splits is None or len(splits) == 0:
            return None
        splits = splits.copy()
        splits.index = pd.to_datetime(splits.index, utc=True)
        df = splits.rename("ratio").to_frame()
        _save_cache(df, cp)
        return df["ratio"]
    except Exception as exc:
        logger.warning("alpaca_features: splits fetch failed %s: %s", ticker, exc)
        return None


def _fetch_dividends(ticker: str) -> Optional[pd.Series]:
    """Fetch dividend history. Returns Series indexed by UTC timestamp."""
    cp = _cache_path(ticker, "dividends")
    if _cache_fresh(cp):
        df = _load_cache(cp)
        return df["amount"] if df is not None else None
    try:
        import yfinance as yf
        divs = yf.Ticker(ticker).dividends
        if divs is None or len(divs) == 0:
            return None
        divs = divs.copy()
        divs.index = pd.to_datetime(divs.index, utc=True)
        df = divs.rename("amount").to_frame()
        _save_cache(df, cp)
        return df["amount"]
    except Exception as exc:
        logger.warning("alpaca_features: dividends fetch failed %s: %s", ticker, exc)
        return None


def _fetch_info(ticker: str) -> dict:
    """Fetch ticker info dict from yfinance. Returns {} on error."""
    cp = _cache_path(ticker, "info")
    if _cache_fresh(cp):
        df = _load_cache(cp)
        if df is not None and len(df) > 0:
            return df.iloc[0].to_dict()
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
        wanted = {
            k: info.get(k) for k in [
                "marketCap", "floatShares", "sector", "shortPercentOfFloat",
                "dividendYield",
            ]
        }
        df = pd.DataFrame([wanted])
        _save_cache(df, cp)
        return wanted
    except Exception as exc:
        logger.warning("alpaca_features: info fetch failed %s: %s", ticker, exc)
        return {}


def _fetch_calendar(ticker: str) -> dict:
    """Fetch calendar dict from yfinance (ex-div date, next earnings). {} on error."""
    cp = _cache_path(ticker, "calendar")
    if _cache_fresh(cp):
        df = _load_cache(cp)
        if df is not None and len(df) > 0:
            return df.iloc[0].to_dict()
    try:
        import yfinance as yf
        cal = yf.Ticker(ticker).calendar or {}
        # Flatten: only keep string-serialisable keys
        out: dict = {}
        for k, v in cal.items():
            if isinstance(v, list) and len(v) > 0:
                out[k] = str(v[0])
            elif v is not None:
                out[k] = str(v)
        df = pd.DataFrame([out])
        _save_cache(df, cp)
        return out
    except Exception as exc:
        logger.warning("alpaca_features: calendar fetch failed %s: %s", ticker, exc)
        return {}


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------
def _earnings_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Add 4 earnings-proximity features."""
    ZERO_COLS = [
        "days_until_earnings",
        "is_earnings_week",
        "earnings_surprise_last",
        "days_since_last_earnings",
    ]
    ed = _fetch_earnings(ticker)
    if ed is None:
        for c in ZERO_COLS:
            df[c] = 0.0
        return df

    # Separate future (no Reported EPS) from past events
    reported_mask = ed["Reported EPS"].notna()
    past_dates_raw = ed[reported_mask].index  # UTC dates as reported in yfinance
    future_dates_raw = ed[~reported_mask].index  # UTC dates (known schedule)

    # POINT-IN-TIME GUARD (no-lookahead audit 2026-05-21, Patch 2):
    # yfinance returns a CURRENT snapshot, not a point-in-time view -- so dates
    # may be retroactively added/edited. To stay conservative, shift each
    # earnings_date by +1 BDay before exposing it as a feature. Rationale:
    # an announcement is only KNOWN to the market AFTER its release; treating
    # earnings_date itself as "known on day-of" risks treating same-bar
    # information that wasn't fully available at the prior bar close. The
    # +1 BDay shift gives us a defensible "only known post-announcement"
    # contract. (Output features are also .shift(1) at end of function for
    # additional belt-and-suspenders alignment with sibling modules.)
    # earnings_date shifted +1 BDay: only known post-announcement
    _ONE_BDAY = pd.tseries.offsets.BDay(1)
    past_dates = pd.DatetimeIndex([d + _ONE_BDAY for d in past_dates_raw])
    future_dates = pd.DatetimeIndex([d + _ONE_BDAY for d in future_dates_raw])
    # Map shifted-date -> raw-date so we can still look up Surprise(%) on ed.
    _shifted_to_raw_past = {
        (d + _ONE_BDAY): d for d in past_dates_raw
    }

    # Ensure df.index is UTC-aware
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")

    days_until = []
    is_week = []
    days_since = []
    surprise_vals = []

    for dt in idx:
        dt_date = dt.date()

        # Days until next known earnings
        future_dt_dates = [d.date() for d in future_dates if d.date() > dt_date]
        if future_dt_dates:
            next_earn = min(future_dt_dates)
            days_until.append((next_earn - dt_date).days)
        else:
            days_until.append(np.nan)

        # Past earnings (past_dates is the +1 BDay-shifted post-announcement view)
        past_dt_dates = sorted([d.date() for d in past_dates if d.date() <= dt_date])
        if past_dt_dates:
            last_earn = past_dt_dates[-1]
            days_since.append((dt_date - last_earn).days)
            # Surprise for most recent earnings: dereference back to RAW
            # (un-shifted) date so we can index `ed` (which is yfinance-native).
            last_idx_shifted = [d for d in past_dates if d.date() == last_earn]
            if last_idx_shifted:
                _raw = _shifted_to_raw_past.get(last_idx_shifted[0])
                if _raw is not None and _raw in ed.index:
                    surp = ed.loc[_raw, "Surprise(%)"]
                    surprise_vals.append(float(surp) if pd.notna(surp) else 0.0)
                else:
                    surprise_vals.append(0.0)
            else:
                surprise_vals.append(0.0)
        else:
            days_since.append(np.nan)
            surprise_vals.append(0.0)

        # is_earnings_week: next earnings within 5 calendar days
        if days_until and not np.isnan(days_until[-1]):
            is_week.append(1.0 if days_until[-1] <= 5 else 0.0)
        else:
            is_week.append(0.0)

    df["days_until_earnings"] = days_until
    df["is_earnings_week"] = is_week
    df["days_since_last_earnings"] = days_since
    df["earnings_surprise_last"] = surprise_vals

    # Shift by 1 bar (no-lookahead)
    for c in ZERO_COLS:
        df[c] = df[c].shift(1)

    return df


def _dividend_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Add 4 dividend/ex-div proximity features."""
    ZERO_COLS = [
        "ex_div_proximity",
        "days_since_last_exdiv",
        "div_yield_trailing",
        "dividend_growth_yoy",
    ]
    divs = _fetch_dividends(ticker)
    cal = _fetch_calendar(ticker)

    if divs is None:
        for c in ZERO_COLS:
            df[c] = 0.0
        return df

    # Normalize dividend index to dates
    div_dates = sorted(pd.to_datetime(divs.index, utc=True).normalize().to_series().dt.date.tolist())
    div_amounts = {
        pd.to_datetime(d, utc=True).normalize().date(): float(v)
        for d, v in zip(divs.index, divs.values)
    }

    # Next ex-div date from calendar (may be in the future)
    next_exdiv = None
    exdiv_str = cal.get("Ex-Dividend Date")
    if exdiv_str:
        try:
            next_exdiv = pd.to_datetime(exdiv_str, utc=True).date()
        except Exception:
            pass

    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")

    prox_vals = []
    days_since_vals = []
    yield_vals = []
    growth_vals = []

    # Close price for yield calculation
    close = df["close"].values

    for i, dt in enumerate(idx):
        dt_date = dt.date()
        price = float(close[i]) if close[i] > 0 else np.nan

        # Days since last ex-div (use dividend dates as proxy for ex-div dates)
        past_divs = [d for d in div_dates if d < dt_date]
        if past_divs:
            last_div_date = past_divs[-1]
            days_since = (dt_date - last_div_date).days
            days_since_vals.append(float(days_since))
        else:
            days_since_vals.append(np.nan)

        # Ex-div proximity: 1/(1 + days_to_next_exdiv), 0 if no future exdiv known
        future_divs = [d for d in div_dates if d > dt_date]
        next_div_date = future_divs[0] if future_divs else next_exdiv
        if next_div_date and next_div_date > dt_date:
            days_to = (next_div_date - dt_date).days
            prox_vals.append(1.0 / (1.0 + days_to))
        else:
            prox_vals.append(0.0)

        # Trailing 12-month dividend yield
        cutoff = pd.Timestamp(dt_date) - pd.DateOffset(years=1)
        cutoff_date = cutoff.date()
        ttm_divs = [v for d, v in div_amounts.items() if cutoff_date <= d < dt_date]
        if ttm_divs and price and not np.isnan(price):
            yield_vals.append(sum(ttm_divs) / price)
        else:
            yield_vals.append(0.0)

        # Dividend growth YoY: last div vs div ~1yr ago
        if len(past_divs) >= 2:
            last_amt = div_amounts.get(past_divs[-1], 0.0)
            # Find div from roughly 1 year before last div
            ref_cutoff = past_divs[-1] - pd.Timedelta(days=365).to_pytimedelta()
            earlier = [d for d in div_dates if d <= ref_cutoff]
            if earlier:
                earlier_amt = div_amounts.get(earlier[-1], 0.0)
                if earlier_amt > 0:
                    growth_vals.append((last_amt - earlier_amt) / earlier_amt)
                else:
                    growth_vals.append(0.0)
            else:
                growth_vals.append(0.0)
        else:
            growth_vals.append(0.0)

    df["ex_div_proximity"] = prox_vals
    df["days_since_last_exdiv"] = days_since_vals
    df["div_yield_trailing"] = yield_vals
    df["dividend_growth_yoy"] = growth_vals

    # Shift 1 bar — no lookahead
    for c in ZERO_COLS:
        df[c] = df[c].shift(1)

    return df


def _split_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Add 2 corporate action (split) features."""
    ZERO_COLS = ["days_since_last_split", "is_post_split_60d"]
    splits = _fetch_splits(ticker)
    if splits is None:
        for c in ZERO_COLS:
            df[c] = 0.0
        return df

    split_idx = pd.to_datetime(splits.index, utc=True).normalize()
    split_dates = sorted(split_idx.date.tolist())

    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")

    days_since_vals = []
    post60_vals = []

    for dt in idx:
        dt_date = dt.date()
        past_splits = [d for d in split_dates if d <= dt_date]
        if past_splits:
            last_split = past_splits[-1]
            days_since = (dt_date - last_split).days
            days_since_vals.append(min(float(days_since), 730.0))
            post60_vals.append(1.0 if days_since <= 60 else 0.0)
        else:
            days_since_vals.append(730.0)  # cap: no known split
            post60_vals.append(0.0)

    df["days_since_last_split"] = days_since_vals
    df["is_post_split_60d"] = post60_vals

    # Shift 1 bar
    for c in ZERO_COLS:
        df[c] = df[c].shift(1)

    return df


def _metadata_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Add 3 static asset metadata features."""
    info = _fetch_info(ticker)

    mkt_cap = info.get("marketCap") or 0
    float_shares = info.get("floatShares") or 0
    short_pct = info.get("shortPercentOfFloat") or 0.0
    sector = info.get("sector") or ""

    df["log_market_cap"] = math.log10(mkt_cap) if mkt_cap > 0 else 0.0
    df["short_interest_pct"] = float(short_pct) if short_pct else 0.0
    df["sector_encoded"] = float(SECTOR_MAP.get(sector, 12))

    # These are static per ticker — no shift needed (no lookahead risk)
    return df


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------
def add_alpaca_features(daily_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Add Alpaca/yfinance-derived features to a daily OHLCV DataFrame.

    Parameters
    ----------
    daily_df : pd.DataFrame
        Daily price data with at least a 'close' column. DatetimeIndex
        (UTC-aware or naive). Must be sorted ascending.
    ticker : str
        Stock ticker symbol (e.g. 'AAPL').

    Returns
    -------
    pd.DataFrame
        Input DataFrame with up to 13 new feature columns appended.
        All new columns are .shift(1)-safe (no lookahead).

    Features
    --------
    Earnings (4): days_until_earnings, is_earnings_week,
                  earnings_surprise_last, days_since_last_earnings
    Dividends (4): ex_div_proximity, days_since_last_exdiv,
                   div_yield_trailing, dividend_growth_yoy
    Splits (2):   days_since_last_split, is_post_split_60d
    Metadata (3): log_market_cap, short_interest_pct, sector_encoded
    """
    df = daily_df.copy()
    ticker = ticker.upper()

    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("daily_df must have a DatetimeIndex")

    # Ensure we have a 'close' column (needed for yield calculation)
    if "close" not in df.columns:
        df["close"] = 100.0

    try:
        df = _earnings_features(df, ticker)
    except Exception as exc:
        logger.warning("alpaca_features: earnings group failed %s: %s", ticker, exc)
        for c in ["days_until_earnings", "is_earnings_week",
                  "earnings_surprise_last", "days_since_last_earnings"]:
            df[c] = 0.0

    try:
        df = _dividend_features(df, ticker)
    except Exception as exc:
        logger.warning("alpaca_features: dividend group failed %s: %s", ticker, exc)
        for c in ["ex_div_proximity", "days_since_last_exdiv",
                  "div_yield_trailing", "dividend_growth_yoy"]:
            df[c] = 0.0

    try:
        df = _split_features(df, ticker)
    except Exception as exc:
        logger.warning("alpaca_features: split group failed %s: %s", ticker, exc)
        for c in ["days_since_last_split", "is_post_split_60d"]:
            df[c] = 0.0

    try:
        df = _metadata_features(df, ticker)
    except Exception as exc:
        logger.warning("alpaca_features: metadata group failed %s: %s", ticker, exc)
        for c in ["log_market_cap", "short_interest_pct", "sector_encoded"]:
            df[c] = 0.0

    return df
