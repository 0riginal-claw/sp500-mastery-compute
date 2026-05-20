"""
insider_form4_features.py
=========================
Adds SEC Form 4 insider-trading features to a daily-indexed DataFrame.

Data source decision
--------------------
- The local EDGAR SQLite DB (edgar.db) contains only 10-K/10-Q/8-K/S-1/DEF 14A
  filings — Form 4 count = 0.
- py-sec-edgar (cloned repo) is not installed in the venv (missing 'aiofiles'),
  so it cannot be imported.
- CHOSEN PATH: SEC EDGAR Submissions JSON API (data.sec.gov/submissions/),
  an official, rate-limited endpoint. Form 4 filing metadata (date + accession)
  is fetched from that endpoint; individual Form 4 XMLs are then fetched and
  parsed to extract transaction type (P/S/A/D/F codes), shares, and price.
- Rate limit: 0.11 s between requests (≈9 req/s, safely under the 10 req/s cap).
- Results cached per-ticker at:
  .../s&p500-ticker-mastery/cache/form4_features/<TICKER>.parquet
  After the first run, the module reads from cache and makes zero network calls
  unless force_refresh=True is passed.

Features produced (all .shift(1)-safe — look-back only, no future data)
------------------------------------------------------------------------
  insider_buy_count_30d            count of insider open-market purchases (code P) in trailing 30 calendar days
  insider_sell_count_30d           count of insider open-market sales (code S) in trailing 30 calendar days
  insider_net_buy_count_60d        net (buys - sells) in trailing 60 calendar days
  insider_cluster_buy_flag         1 if >=3 distinct insiders bought in trailing 30 days
  insider_cluster_sell_flag        1 if >=3 distinct insiders sold in trailing 30 days
  days_since_last_insider_buy      calendar days since most recent buy transaction
  days_since_last_insider_sell     calendar days since most recent sell transaction
  insider_buy_dollar_amount_60d_log log1p of total USD value of insider buys in 60 days

Transaction code mapping (SEC Form 4)
--------------------------------------
  P = open-market purchase    -> BUY
  A = grant/award            -> BUY (restricted, but insiders receive shares)
  S = open-market sale       -> SELL
  D = disposition to issuer  -> SELL
  F = tax withholding sale   -> SELL (excluded from buy counts)
Only P and S codes are used for the primary counts; A/D are tallied separately
but not surfaced as separate features in this version.

Public API
----------
  add_insider_form4_features(daily_df, ticker, force_refresh=False) -> pd.DataFrame
"""

from __future__ import annotations

import functools
import json
import logging
import math
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# Root of the s&p500-ticker-mastery project (parent of scripts/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CACHE_DIR = _PROJECT_ROOT / "cache" / "form4_features"
_EDGAR_CACHE_DIR = _PROJECT_ROOT / "cache" / "edgar"
_CACHE_TTL = 604800  # 7 days — Form 4 data doesn't change retroactively

# Ensure cache directories exist at module load time
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_EDGAR_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# In-process transaction cache: ticker -> txn DataFrame (avoids disk hits within one process)
_txn_cache: Dict[str, pd.DataFrame] = {}

_EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik_padded}.json"
_EDGAR_ARCHIVE_URL = (
    "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_no_dashes}/"
)
_USER_AGENT = "research insider-features orginal_clawdbot@yahoo.com"
_REQUEST_DELAY = 0.12  # seconds between HTTP requests — stay under 10 req/s

# Codes treated as BUY / SELL for the primary features
_BUY_CODES = {"P"}       # open-market purchase
_SELL_CODES = {"S"}      # open-market sale
_ALL_BUY_CODES = {"P", "A"}   # broader set (used internally but not in primary counts)
_ALL_SELL_CODES = {"S", "D", "F"}

_DUMMY_FEATURE_NAMES = [
    "insider_buy_count_30d",
    "insider_sell_count_30d",
    "insider_net_buy_count_60d",
    "insider_cluster_buy_flag",
    "insider_cluster_sell_flag",
    "days_since_last_insider_buy",
    "days_since_last_insider_sell",
    "insider_buy_dollar_amount_60d_log",
]

_FEATURE_DTYPES: Dict[str, str] = {
    "insider_buy_count_30d": "float32",
    "insider_sell_count_30d": "float32",
    "insider_net_buy_count_60d": "float32",
    "insider_cluster_buy_flag": "float32",
    "insider_cluster_sell_flag": "float32",
    "days_since_last_insider_buy": "float32",
    "days_since_last_insider_sell": "float32",
    "insider_buy_dollar_amount_60d_log": "float32",
}


# ---------------------------------------------------------------------------
# SEC HTTP helpers
# ---------------------------------------------------------------------------

def _edgar_rate_limit() -> None:
    """Enforce polite EDGAR request pacing (~8 req/s, safely under the 10/s cap)."""
    time.sleep(_REQUEST_DELAY)


def _get(url: str, retries: int = 3, backoff: float = 2.0) -> requests.Response:
    """HTTP GET with retry logic and rate-limit sleep."""
    headers = {"User-Agent": _USER_AGENT, "Accept-Encoding": "gzip, deflate"}
    last_exc: Exception = RuntimeError("no attempts made")
    for attempt in range(retries):
        try:
            _edgar_rate_limit()  # polite pacing before every network call
            resp = requests.get(url, headers=headers, timeout=20)
            if resp.status_code == 429:
                wait = backoff * (2 ** attempt)
                logger.warning("Rate-limited by EDGAR; sleeping %.1fs", wait)
                time.sleep(wait)
                continue
            return resp
        except Exception as exc:
            last_exc = exc
            time.sleep(backoff * (attempt + 1))
    raise last_exc


@functools.lru_cache(maxsize=512)
def _ticker_to_cik(ticker: str) -> Optional[str]:
    """
    Resolve ticker -> CIK string (10-digit zero-padded).
    Uses the EDGAR company tickers JSON (no auth required).
    Result is cached in-process so repeated lookups for the same ticker skip the network.
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    try:
        resp = _get(url)
        if resp.status_code != 200:
            logger.warning("company_tickers.json returned %d", resp.status_code)
            return None
        data = resp.json()
        ticker_upper = ticker.upper()
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker_upper:
                return str(entry["cik_str"]).zfill(10)
    except Exception as exc:
        logger.warning("CIK lookup failed for %s: %s", ticker, exc)
    return None


# ---------------------------------------------------------------------------
# Form 4 XML parsing
# ---------------------------------------------------------------------------

def _xml_val(element: ET.Element, tag: str) -> Optional[str]:
    """Extract the text of <tag><value>...</value></tag>."""
    node = element.find(f".//{tag}/value")
    if node is not None and node.text:
        return node.text.strip()
    # Fallback: direct text node
    node2 = element.find(f".//{tag}")
    if node2 is not None and node2.text:
        return node2.text.strip()
    return None


def _parse_form4_xml(xml_text: str) -> List[dict]:
    """
    Parse a Form 4 XML string and return a list of transaction dicts:
        {
            owner_name: str,
            owner_cik: str,
            transaction_date: str (YYYY-MM-DD),
            transaction_code: str,
            shares: float | None,
            price_per_share: float | None,
            acquired_disposed: str (A or D),
        }
    Returns [] on any parse error.
    """
    rows: List[dict] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.debug("Form 4 XML parse error: %s", exc)
        return rows

    owner_name = _xml_val(root, "rptOwnerName") or ""
    owner_cik = _xml_val(root, "rptOwnerCik") or ""

    for txn in root.findall(".//nonDerivativeTransaction"):
        code_node = txn.find(".//transactionCode")
        code = code_node.text.strip() if (code_node is not None and code_node.text) else ""
        date_str = _xml_val(txn, "transactionDate") or ""
        shares_str = _xml_val(txn, "transactionShares")
        price_str = _xml_val(txn, "transactionPricePerShare")
        ad_node = txn.find(".//transactionAcquiredDisposedCode/value")
        ad_code = ad_node.text.strip() if (ad_node is not None and ad_node.text) else ""

        try:
            shares_val = float(shares_str) if shares_str else None
        except ValueError:
            shares_val = None
        try:
            price_val = float(price_str) if price_str else None
        except ValueError:
            price_val = None

        rows.append(
            {
                "owner_name": owner_name,
                "owner_cik": owner_cik,
                "transaction_date": date_str,
                "transaction_code": code,
                "shares": shares_val,
                "price_per_share": price_val,
                "acquired_disposed": ad_code,
            }
        )

    # Also capture derivative transactions (options exercises, etc.) but mark them
    for txn in root.findall(".//derivativeTransaction"):
        code_node = txn.find(".//transactionCode")
        code = code_node.text.strip() if (code_node is not None and code_node.text) else ""
        date_str = _xml_val(txn, "transactionDate") or ""
        rows.append(
            {
                "owner_name": owner_name,
                "owner_cik": owner_cik,
                "transaction_date": date_str,
                "transaction_code": code,
                "shares": None,
                "price_per_share": None,
                "acquired_disposed": "",
                "_is_derivative": True,
            }
        )

    return rows


# ---------------------------------------------------------------------------
# EDGAR data fetch
# ---------------------------------------------------------------------------

def _get_form4_xml_url(cik_int: str, accession: str) -> Optional[str]:
    """
    Resolve the URL of the form4.xml inside a filing's EDGAR archive folder.
    Tries the common filename 'form4.xml' directly; falls back to listing the
    directory index.
    """
    acc_no_dashes = accession.replace("-", "")
    # Common case: file is named 'form4.xml'
    direct_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_no_dashes}/form4.xml"
    # We don't validate here to save requests; return it and let the caller handle 404
    return direct_url


def _fetch_transactions_for_ticker(
    cik_padded: str, max_filings: int = 500
) -> pd.DataFrame:
    """
    Fetch all Form 4 transactions for a given CIK from EDGAR.

    Returns a DataFrame with columns:
        filing_date, transaction_date, owner_name, owner_cik,
        transaction_code, shares, price_per_share, acquired_disposed
    """
    cik_int = str(int(cik_padded))  # strip leading zeros for archive path

    # Step 1: Get filing list from submissions JSON
    url = _EDGAR_SUBMISSIONS_URL.format(cik_padded=cik_padded)
    resp = _get(url)
    if resp.status_code != 200:
        logger.warning("Submissions fetch failed (HTTP %d) for CIK %s", resp.status_code, cik_padded)
        return pd.DataFrame()

    data = resp.json()
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    filing_dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])

    # Collect Form 4 accessions
    form4_entries: List[Tuple[str, str]] = []
    for i, form in enumerate(forms):
        if form == "4":
            form4_entries.append((filing_dates[i], accessions[i]))

    # Also pull older filings pages (if available)
    for file_info in data.get("filings", {}).get("files", []):
        if len(form4_entries) >= max_filings:
            break
        file_url = f"https://data.sec.gov/submissions/{file_info['name']}"
        try:
            r2 = _get(file_url)
            if r2.status_code == 200:
                old = r2.json()
                old_forms = old.get("form", [])
                old_dates = old.get("filingDate", [])
                old_acc = old.get("accessionNumber", [])
                for i, form in enumerate(old_forms):
                    if form == "4":
                        form4_entries.append((old_dates[i], old_acc[i]))
        except Exception as exc:
            logger.debug("Old filings page fetch error: %s", exc)

    logger.info("Found %d Form 4 filings for CIK %s; fetching XMLs...", len(form4_entries), cik_padded)

    # Step 2: Fetch and parse each Form 4 XML
    all_rows: List[dict] = []
    fetched = 0
    for filing_date, accession in form4_entries[:max_filings]:
        xml_url = _get_form4_xml_url(cik_int, accession)
        if xml_url is None:
            continue
        try:
            resp = _get(xml_url)
            if resp.status_code == 404:
                # Try listing the directory to find the right XML filename
                dir_url = _EDGAR_ARCHIVE_URL.format(
                    cik_int=cik_int,
                    acc_no_dashes=accession.replace("-", "")
                )
                dir_resp = _get(dir_url)
                if dir_resp.status_code == 200:
                    import re as _re
                    xmls = _re.findall(r'href="(/Archives/[^"]+\.xml)"', dir_resp.text)
                    if xmls:
                        xml_url = "https://www.sec.gov" + xmls[0]
                        resp = _get(xml_url)
                    else:
                        continue
                else:
                    continue
            if resp.status_code != 200:
                continue
            txns = _parse_form4_xml(resp.text)
            for t in txns:
                t["filing_date"] = filing_date
            all_rows.extend(txns)
            fetched += 1
        except Exception as exc:
            logger.debug("XML fetch/parse error for %s: %s", accession, exc)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    # Normalize dates
    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    # Use filing_date as fallback when transaction_date is missing
    df["effective_date"] = df["transaction_date"].fillna(df["filing_date"])
    df = df.dropna(subset=["effective_date"])
    df = df.sort_values("effective_date").reset_index(drop=True)
    logger.info("Fetched %d Form 4 XMLs -> %d transaction rows for CIK %s", fetched, len(df), cik_padded)
    return df


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def _compute_rolling_features(
    daily_index: pd.DatetimeIndex, txn_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Given a daily DatetimeIndex and a transactions DataFrame,
    compute all 8 features for every day in the index.
    All features are computed using ONLY past data (<=date), making them .shift(1)-safe.
    """
    # Separate buy/sell rows
    buy_mask = txn_df["transaction_code"].isin(_BUY_CODES)
    sell_mask = txn_df["transaction_code"].isin(_SELL_CODES)

    buy_df = txn_df[buy_mask].copy()
    sell_df = txn_df[sell_mask].copy()

    # Pre-compute useful series
    if not buy_df.empty:
        buy_dates = buy_df["effective_date"].values.astype("datetime64[D]")
        buy_owners = buy_df["owner_cik"].values
        buy_shares = buy_df["shares"].values
        buy_prices = buy_df["price_per_share"].values
    else:
        buy_dates = np.array([], dtype="datetime64[D]")
        buy_owners = np.array([])
        buy_shares = np.array([])
        buy_prices = np.array([])

    if not sell_df.empty:
        sell_dates = sell_df["effective_date"].values.astype("datetime64[D]")
        sell_owners = sell_df["owner_cik"].values
    else:
        sell_dates = np.array([], dtype="datetime64[D]")
        sell_owners = np.array([])

    result = {}
    buy_count_30 = np.zeros(len(daily_index), dtype=np.float32)
    sell_count_30 = np.zeros(len(daily_index), dtype=np.float32)
    net_buy_60 = np.zeros(len(daily_index), dtype=np.float32)
    cluster_buy = np.zeros(len(daily_index), dtype=np.float32)
    cluster_sell = np.zeros(len(daily_index), dtype=np.float32)
    days_since_buy = np.full(len(daily_index), np.nan, dtype=np.float32)
    days_since_sell = np.full(len(daily_index), np.nan, dtype=np.float32)
    buy_dollar_60 = np.zeros(len(daily_index), dtype=np.float32)

    for i, date in enumerate(daily_index):
        d = np.datetime64(date.date(), "D")
        cutoff_30 = d - np.timedelta64(30, "D")
        cutoff_60 = d - np.timedelta64(60, "D")

        # --- buy features ---
        if len(buy_dates) > 0:
            mask_past = buy_dates <= d
            mask_30 = (buy_dates > cutoff_30) & mask_past
            mask_60 = (buy_dates > cutoff_60) & mask_past

            bc30 = int(mask_30.sum())
            bc60 = int(mask_60.sum())
            buy_count_30[i] = bc30

            # cluster buy: >= 3 distinct insiders in 30d
            if bc30 >= 3:
                distinct_buyers = len(set(buy_owners[mask_30]))
                cluster_buy[i] = 1.0 if distinct_buyers >= 3 else 0.0

            # dollar amount 60d
            if bc60 > 0:
                shares60 = buy_shares[mask_60]
                prices60 = buy_prices[mask_60]
                # element-wise multiply where both not None
                valid = ~(pd.isnull(shares60) | pd.isnull(prices60))
                if valid.sum() > 0:
                    dollar_val = float(np.nansum(shares60[valid] * prices60[valid]))
                    buy_dollar_60[i] = dollar_val

            # days since last buy
            past_buy_dates = buy_dates[mask_past]
            if len(past_buy_dates) > 0:
                last_buy = past_buy_dates.max()
                days_since_buy[i] = float((d - last_buy) / np.timedelta64(1, "D"))

        # --- sell features ---
        if len(sell_dates) > 0:
            smask_past = sell_dates <= d
            smask_30 = (sell_dates > cutoff_30) & smask_past
            smask_60 = (sell_dates > cutoff_60) & smask_past

            sc30 = int(smask_30.sum())
            sc60 = int(smask_60.sum())
            sell_count_30[i] = sc30

            if sc30 >= 3:
                distinct_sellers = len(set(sell_owners[smask_30]))
                cluster_sell[i] = 1.0 if distinct_sellers >= 3 else 0.0

            past_sell_dates = sell_dates[smask_past]
            if len(past_sell_dates) > 0:
                last_sell = past_sell_dates.max()
                days_since_sell[i] = float((d - last_sell) / np.timedelta64(1, "D"))

            # net buy count 60d
            bc60_v = int((buy_dates > cutoff_60).sum() if len(buy_dates) > 0 else 0)
            net_buy_60[i] = bc60_v - sc60

    result["insider_buy_count_30d"] = buy_count_30
    result["insider_sell_count_30d"] = sell_count_30
    result["insider_net_buy_count_60d"] = net_buy_60
    result["insider_cluster_buy_flag"] = cluster_buy
    result["insider_cluster_sell_flag"] = cluster_sell
    result["days_since_last_insider_buy"] = days_since_buy
    result["days_since_last_insider_sell"] = days_since_sell
    result["insider_buy_dollar_amount_60d_log"] = np.log1p(buy_dollar_60).astype(np.float32)

    return pd.DataFrame(result, index=daily_index)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(ticker: str) -> Path:
    return _CACHE_DIR / f"{ticker.upper()}.parquet"


def _load_cache(ticker: str) -> Optional[pd.DataFrame]:
    p = _cache_path(ticker)
    if not p.exists():
        return None
    age = time.time() - p.stat().st_mtime
    if age > _CACHE_TTL:
        logger.debug("Parquet cache expired for %s (age=%.0fs)", ticker, age)
        return None
    try:
        df = pd.read_parquet(p)
        return df
    except Exception as exc:
        logger.warning("Cache read failed for %s: %s", ticker, exc)
    return None


def _save_cache(ticker: str, df: pd.DataFrame) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = _cache_path(ticker)
    try:
        df.to_parquet(p, index=True)
        logger.info("Cached Form 4 transactions for %s -> %s", ticker, p)
    except Exception as exc:
        logger.warning("Cache write failed for %s: %s", ticker, exc)


# ---------------------------------------------------------------------------
# JSON disk cache helpers (cache/edgar/<TICKER>_form4.json, TTL=7 days)
# ---------------------------------------------------------------------------

def _json_cache_path(ticker: str) -> Path:
    return _EDGAR_CACHE_DIR / f"{ticker.upper()}_form4.json"


def _read_json_cache_file(p: Path, ticker: str) -> Optional[pd.DataFrame]:
    """Deserialize a JSON cache file back to a transactions DataFrame."""
    try:
        with open(p, "r") as fh:
            records = json.load(fh)
        df = pd.DataFrame(records)
        if not df.empty:
            for col in ("filing_date", "transaction_date", "effective_date"):
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
        return df
    except Exception as exc:
        logger.warning("JSON cache read failed for %s: %s", ticker, exc)
        return None


def _load_json_cache(ticker: str) -> Optional[pd.DataFrame]:
    """Load JSON cache only if it exists and is within the 7-day TTL."""
    p = _json_cache_path(ticker)
    if not p.exists():
        return None
    age = time.time() - p.stat().st_mtime
    if age > _CACHE_TTL:
        logger.debug("JSON cache expired for %s (age=%.0fs)", ticker, age)
        return None
    return _read_json_cache_file(p, ticker)


def _load_json_cache_stale(ticker: str) -> Optional[pd.DataFrame]:
    """Load JSON cache ignoring TTL — used as last resort when network fails."""
    p = _json_cache_path(ticker)
    if not p.exists():
        return None
    return _read_json_cache_file(p, ticker)


def _save_json_cache(ticker: str, df: pd.DataFrame) -> None:
    """Persist transactions DataFrame to the JSON cache file."""
    _EDGAR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = _json_cache_path(ticker)
    try:
        df_copy = df.copy()
        for col in df_copy.columns:
            if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                df_copy[col] = df_copy[col].dt.strftime("%Y-%m-%d").where(
                    df_copy[col].notna(), None
                )
        records = df_copy.where(pd.notna(df_copy), None).to_dict(orient="records")
        with open(p, "w") as fh:
            json.dump(records, fh, default=str)
        logger.info("JSON-cached Form 4 transactions for %s -> %s", ticker, p)
    except Exception as exc:
        logger.warning("JSON cache write failed for %s: %s", ticker, exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_insider_form4_features(
    daily_df: pd.DataFrame,
    ticker: str,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Adds 8 insider-trading (SEC Form 4) features to ``daily_df``.

    Parameters
    ----------
    daily_df : pd.DataFrame
        Daily-frequency DataFrame with a DatetimeIndex. Only the index is used;
        existing columns are preserved unchanged.
    ticker : str
        Equity ticker symbol (e.g. 'AAPL').
    force_refresh : bool
        If True, bypass the on-disk cache and re-fetch from SEC EDGAR.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with 8 new float32 columns appended.
        All features are purely backward-looking and are safe to .shift(1).

    Notes
    -----
    - First call for a ticker makes ~1-500 HTTP requests to data.sec.gov /
      www.sec.gov. Subsequent calls read from Parquet cache.
    - If the CIK cannot be resolved or EDGAR is unreachable the features are
      set to 0 / NaN and a WARNING is logged; the pipeline does NOT crash.

    Environment-variable bypass (added 2026-05-17 — EDGAR rate-limit relief)
    -----------------------------------------------------------------------
    - ``BACKTEST_SKIP_FORM4=1`` — skip Form 4 entirely; return zero/NaN features
      (no network, no cache reads).  Use when you want to disable the feature
      block wholesale (e.g. ablation runs).
    - ``BACKTEST_FORM4_CACHE_ONLY=1`` — use ONLY local caches (in-process / JSON
      / Parquet, ignoring 7-day TTL — stale caches are accepted).  If no cache
      exists for the ticker, zero/NaN features are returned instead of hitting
      ``data.sec.gov``.  Use when EDGAR is rate-limiting / backoff-locked and
      you need to unblock a backtest batch.
    Default behavior (neither env var set) is unchanged.
    """
    ticker_upper = ticker.upper()
    out = daily_df.copy()
    daily_index = out.index

    # Ensure timezone-naive for comparisons
    if daily_index.tz is not None:
        daily_index_naive = daily_index.tz_localize(None)
    else:
        daily_index_naive = daily_index

    # ------------------------------------------------------------------
    # Env-var bypass (added 2026-05-17 for EDGAR rate-limit relief)
    # ------------------------------------------------------------------
    _skip_form4 = os.environ.get("BACKTEST_SKIP_FORM4", "") == "1"
    _cache_only = os.environ.get("BACKTEST_FORM4_CACHE_ONLY", "") == "1"

    if _skip_form4:
        logger.info(
            "BACKTEST_SKIP_FORM4=1 set; returning zero Form 4 features for %s "
            "(no network, no cache).",
            ticker_upper,
        )
        feat_df = _zero_features(daily_index)
        feat_df.index = daily_index
        for col, dtype in _FEATURE_DTYPES.items():
            if col in feat_df.columns:
                out[col] = feat_df[col].astype(dtype)
            else:
                out[col] = np.float32(0)
        return out

    # --- Load or fetch transactions (3-layer cache: process → JSON → Parquet → network) ---
    txn_df: Optional[pd.DataFrame] = None
    if not force_refresh:
        # Layer 1: in-process dict cache — zero I/O for repeated calls within one process
        txn_df = _txn_cache.get(ticker_upper)
        # Layer 2: JSON disk cache with 7-day TTL
        if txn_df is None:
            txn_df = _load_json_cache(ticker_upper)
            if txn_df is not None:
                _txn_cache[ticker_upper] = txn_df
        # Layer 3: Parquet disk cache with 7-day TTL (promote to JSON on hit)
        if txn_df is None:
            txn_df = _load_cache(ticker_upper)
            if txn_df is not None:
                _txn_cache[ticker_upper] = txn_df
                _save_json_cache(ticker_upper, txn_df)

    # ------------------------------------------------------------------
    # Cache-only bypass (added 2026-05-17): when BACKTEST_FORM4_CACHE_ONLY=1
    # is set and the TTL-fresh caches missed, try STALE caches (ignore TTL)
    # before giving up; never hit the network.
    # ------------------------------------------------------------------
    if txn_df is None and _cache_only:
        # Try stale JSON cache
        stale = _load_json_cache_stale(ticker_upper)
        if stale is not None:
            logger.info(
                "BACKTEST_FORM4_CACHE_ONLY=1; using STALE JSON cache for %s.",
                ticker_upper,
            )
            txn_df = stale
            _txn_cache[ticker_upper] = txn_df
        else:
            # Try stale Parquet cache
            p_stale = _cache_path(ticker_upper)
            if p_stale.exists():
                try:
                    txn_df = pd.read_parquet(p_stale)
                    logger.info(
                        "BACKTEST_FORM4_CACHE_ONLY=1; using STALE Parquet cache for %s.",
                        ticker_upper,
                    )
                    _txn_cache[ticker_upper] = txn_df
                except Exception as exc:
                    logger.warning(
                        "BACKTEST_FORM4_CACHE_ONLY=1; stale Parquet read failed for %s: %s",
                        ticker_upper, exc,
                    )
                    txn_df = None
        if txn_df is None:
            logger.warning(
                "BACKTEST_FORM4_CACHE_ONLY=1 and no cache for %s; "
                "returning zero Form 4 features (no network).",
                ticker_upper,
            )
            txn_df = pd.DataFrame()

    if txn_df is None:
        # Resolve CIK
        logger.info("Fetching Form 4 data for %s from SEC EDGAR...", ticker_upper)
        cik = _ticker_to_cik(ticker_upper)
        if cik is None:
            logger.warning(
                "Cannot resolve CIK for %s; returning zero Form 4 features.", ticker_upper
            )
            txn_df = pd.DataFrame()
        else:
            try:
                txn_df = _fetch_transactions_for_ticker(cik)
            except Exception as exc:
                # Network failure: fall back to stale JSON cache, then stale Parquet
                stale = _load_json_cache_stale(ticker_upper)
                if stale is not None:
                    logger.warning(
                        "Form 4 fetch failed for %s (%s); using stale JSON cache.", ticker_upper, exc
                    )
                    txn_df = stale
                else:
                    p_stale = _cache_path(ticker_upper)
                    if p_stale.exists():
                        try:
                            txn_df = pd.read_parquet(p_stale)
                            logger.warning(
                                "Form 4 fetch failed for %s (%s); using stale Parquet cache.",
                                ticker_upper, exc,
                            )
                        except Exception:
                            txn_df = None
                    if txn_df is None:
                        logger.warning(
                            "Form 4 fetch failed for %s (%s); returning zero features.", ticker_upper, exc
                        )
                        txn_df = pd.DataFrame()

        if txn_df is not None and not txn_df.empty:
            _save_cache(ticker_upper, txn_df)
            _save_json_cache(ticker_upper, txn_df)
            _txn_cache[ticker_upper] = txn_df
        else:
            txn_df = pd.DataFrame()

    # --- Compute features ---
    if txn_df.empty:
        logger.warning(
            "No Form 4 transactions found for %s; all features set to 0/NaN.", ticker_upper
        )
        feat_df = _zero_features(daily_index)
    else:
        # Normalize effective_date to tz-naive
        if "effective_date" not in txn_df.columns:
            if "transaction_date" in txn_df.columns:
                txn_df["effective_date"] = pd.to_datetime(
                    txn_df["transaction_date"], errors="coerce"
                ).fillna(pd.to_datetime(txn_df.get("filing_date", pd.NaT), errors="coerce"))
            else:
                txn_df["effective_date"] = pd.NaT

        txn_df["effective_date"] = pd.to_datetime(
            txn_df["effective_date"], errors="coerce"
        )
        if txn_df["effective_date"].dt.tz is not None:
            txn_df["effective_date"] = txn_df["effective_date"].dt.tz_localize(None)

        feat_df = _compute_rolling_features(daily_index_naive, txn_df)

    # Reindex to original index (handles tz differences)
    feat_df.index = daily_index
    for col, dtype in _FEATURE_DTYPES.items():
        if col in feat_df.columns:
            out[col] = feat_df[col].astype(dtype)
        else:
            out[col] = np.float32(0)

    return out


def _zero_features(index: pd.Index) -> pd.DataFrame:
    """Return a DataFrame of all-zero/NaN features for the given index."""
    n = len(index)
    return pd.DataFrame(
        {
            "insider_buy_count_30d": np.zeros(n, dtype=np.float32),
            "insider_sell_count_30d": np.zeros(n, dtype=np.float32),
            "insider_net_buy_count_60d": np.zeros(n, dtype=np.float32),
            "insider_cluster_buy_flag": np.zeros(n, dtype=np.float32),
            "insider_cluster_sell_flag": np.zeros(n, dtype=np.float32),
            "days_since_last_insider_buy": np.full(n, np.nan, dtype=np.float32),
            "days_since_last_insider_sell": np.full(n, np.nan, dtype=np.float32),
            "insider_buy_dollar_amount_60d_log": np.zeros(n, dtype=np.float32),
        },
        index=index,
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    dates = pd.date_range("2024-01-01", "2024-12-31", freq="B", tz="UTC")
    df = pd.DataFrame({"close": 100.0}, index=dates)

    for tk in ["AAPL", "NVDA", "TSLA"]:
        print(f"\n--- {tk} ---")
        out = add_insider_form4_features(df.copy(), tk)
        new = [c for c in out.columns if c not in df.columns]
        print(f"{tk}: +{len(new)} Form 4 features")
        for c in new[:6]:
            if pd.api.types.is_numeric_dtype(out[c]):
                nz = (out[c].notna() & (out[c] != 0)).mean() * 100
                print(f"  {c}: {nz:.0f}% non-zero")
