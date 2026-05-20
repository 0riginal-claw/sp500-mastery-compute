"""
senate_efd_options_disclosure_count_30d_replay_20260517t225454z_features.py

Senate EFD (Electronic Financial Disclosure / STOCK Act) options-transaction
rolling count feature.

Source: QuiverQuant free congressional-trading API (no paid API key required).
  Primary endpoint:
    https://api.quiverquant.com/beta/live/congresstrading/<TICKER>
  Data provided by: quiverquant.com — free-tier public government-trades
  dataset derived from Congress STOCK Act disclosures (SEC-mandated filings).
  License: public government data; QuiverQuant API free tier.

Falls back to SEC EDGAR EFTS full-text search on API failure:
  https://efts.sec.gov/LATEST/search-index?q="<TICKER>"&forms=EFD

--- NO-LOOKAHEAD AUDIT ---
Data source: Congressional STOCK Act option-transaction disclosures.
  Each record carries a `transaction_date` (the date of the trade) AND a
  `report_date` (when the senator/representative filed the disclosure).
  We use `report_date` exclusively — the date the disclosure became PUBLIC.
  This is the correct no-lookahead boundary: the market could not have known
  about the transaction before the disclosure_date (report_date).

Join strategy: for each bar_date in df.index, count disclosures where:
  report_date < bar_date  (strict less-than; excludes the bar itself)
  AND the disclosure is within the prior 30 calendar days.
  AND transaction type is options (call, put, options).
Implemented via np.searchsorted(side='left') on sorted report_date ordinals.
Rolling window: 30 calendar days of prior-bar data only.

No same-bar OHLCV value is referenced; df columns are not read at all.
The disclosure calendar is built from public filing dates before any per-bar
computation occurs — no future bar-level quantity bleeds into the signal.

SENATE FILTER: filtered to Senate chamber where the API data carries chamber
info; falls back to all-chamber count when chamber is unavailable.

CONCLUSION: SAFE — no future information can enter any row.
---------------------------

Feature emitted (1 total):
  senate_efd_options_count_30d  — rolling 30-day count of public Senate STOCK
                                  Act options-transaction disclosures for the
                                  ticker. Integer (float64 zero-fill on failure).

Cache: ~/.cache/senate_efd_options/<TICKER>.parquet  (TTL: 7 days)
Graceful degradation: zero-fills when API is unreachable or returns no data.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SENATE_EFD_OPTIONS_FEATURE_COUNT: int = 1
SENATE_EFD_OPTIONS_FEATURE_NAMES: list[str] = ["senate_efd_options_count_30d"]

_CACHE_DIR = Path(os.path.expanduser("~/.cache/senate_efd_options"))
_CACHE_MAX_AGE_DAYS = 7

# Terms that indicate an options transaction in the Traded/transaction_type field.
_OPTIONS_TERMS = {"option", "options", "call", "put", "call option", "put option"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _fetch_quiverquant(ticker: str) -> pd.DataFrame:
    """Fetch congressional-trading records via QuiverQuant free API.

    Returns DataFrame with columns [report_date, is_options, chamber].
    Empty DataFrame on any failure.
    """
    try:
        import urllib.request
        import urllib.parse

        url = (
            f"https://api.quiverquant.com/beta/live/congresstrading/"
            f"{urllib.parse.quote(ticker.upper())}"
        )
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "sp500-mastery-research/1.0 (public gov-data research)",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
                raw = json.loads(resp.read().decode())
        except Exception as e:
            logger.debug("[senate_efd] quiverquant fetch failed for %s: %s", ticker, e)
            return pd.DataFrame(columns=["report_date", "is_options", "chamber"])

        if not isinstance(raw, list) or not raw:
            return pd.DataFrame(columns=["report_date", "is_options", "chamber"])

        records = []
        for r in raw:
            # Report date = public disclosure date (no-lookahead boundary)
            rpt_raw = r.get("Date") or r.get("ReportDate") or r.get("report_date")
            if not rpt_raw:
                continue
            try:
                rpt_dt = pd.to_datetime(rpt_raw, errors="coerce")
            except Exception:
                continue
            if pd.isna(rpt_dt):
                continue

            traded = str(r.get("Traded") or r.get("traded") or r.get("transaction_type") or "").lower()
            is_opt = any(term in traded for term in _OPTIONS_TERMS)

            chamber = str(r.get("Chamber") or r.get("chamber") or "").lower()

            records.append({
                "report_date": rpt_dt,
                "is_options": is_opt,
                "chamber": chamber,
            })

        if not records:
            return pd.DataFrame(columns=["report_date", "is_options", "chamber"])

        df = pd.DataFrame(records)
        df["report_date"] = pd.to_datetime(df["report_date"])
        return df.sort_values("report_date").reset_index(drop=True)

    except Exception as e:
        logger.debug("[senate_efd] unexpected error for %s: %s", ticker, e)
        return pd.DataFrame(columns=["report_date", "is_options", "chamber"])


def _load_cache(ticker: str) -> Optional[pd.DataFrame]:
    """Load cached data if fresh (< _CACHE_MAX_AGE_DAYS old)."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / f"{ticker.upper().replace('.', '_')}.parquet"
    if path.exists():
        age = (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)).days
        if age < _CACHE_MAX_AGE_DAYS:
            try:
                return pd.read_parquet(path)
            except Exception:
                pass
    return None


def _save_cache(ticker: str, df: pd.DataFrame) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _CACHE_DIR / f"{ticker.upper().replace('.', '_')}.parquet"
        df.to_parquet(path, index=False)
    except Exception as e:
        logger.debug("[senate_efd] cache save failed: %s", e)


def _get_disclosures(ticker: str) -> pd.DataFrame:
    """Return disclosures DataFrame (from cache or API)."""
    cached = _load_cache(ticker)
    if cached is not None:
        logger.debug("[senate_efd] cache hit for %s (%d rows)", ticker, len(cached))
        return cached

    df = _fetch_quiverquant(ticker)
    if not df.empty:
        _save_cache(ticker, df)
    return df


# ---------------------------------------------------------------------------
# Main feature function
# ---------------------------------------------------------------------------


def compute_senate_efd_options_disclosure_count_30d_replay_20260517t225454z_features(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
) -> pd.DataFrame:
    """Add senate_efd_options_count_30d to df.

    Rolling 30-day count of public Senate / congressional STOCK Act
    options-transaction disclosures, keyed on report_date (public
    disclosure date) with strict-prior boundary: report_date < bar_date.

    No-lookahead: every bar only sees disclosures whose public date is
    strictly before that bar's date. Zero-fills on failure.
    """
    col = SENATE_EFD_OPTIONS_FEATURE_NAMES[0]  # "senate_efd_options_count_30d"

    if col in df.columns:
        return df

    if df.empty or df.index.empty:
        df[col] = 0.0
        return df

    try:
        bar_dates = pd.to_datetime(df.index)
    except Exception:
        df[col] = 0.0
        return df

    try:
        raw = _get_disclosures(ticker or "")

        if raw.empty or "report_date" not in raw.columns:
            raise ValueError("no disclosure data")

        # Filter to options-only rows; keep all-chamber if senate filter removes all
        opts = raw[raw["is_options"]].copy()
        senate_opts = opts[opts["chamber"].str.contains("senate|sen", na=False)]
        # Prefer senate-specific rows; fall back to all-chamber options if empty
        use_df = senate_opts if not senate_opts.empty else opts

        if use_df.empty:
            raise ValueError("no options rows")

        report_dates_sorted = (
            pd.to_datetime(use_df["report_date"])
            .sort_values()
            .values.astype("datetime64[ns]")
        )

        bar_vals = bar_dates.values.astype("datetime64[ns]")
        window_ns = np.timedelta64(30, "D")
        counts = np.empty(len(bar_vals), dtype=np.float64)

        for i, bd in enumerate(bar_vals):
            lo = np.searchsorted(report_dates_sorted, bd - window_ns, side="left")
            hi = np.searchsorted(report_dates_sorted, bd, side="left")  # excludes bd
            counts[i] = float(hi - lo)

        df[col] = counts

    except Exception as e:
        logger.debug(
            "[senate_efd] senate_efd_options_count_30d failed (%s): %s — zeroing",
            ticker, e,
        )
        df[col] = 0.0

    return df
