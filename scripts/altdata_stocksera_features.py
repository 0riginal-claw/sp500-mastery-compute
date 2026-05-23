"""
altdata_stocksera_features.py — Stocksera + ApeWisdom alt-data features (TOP-10 #5).

Shipped 2026-05-22 (Wave-altdata). Two backends:
  - ApeWisdom (free, no API key) — primary source for WSB / Reddit mentions.
  - Stocksera (key-gated, off by default) — adds dark-pool, off-exchange short
    volume, news sentiment, short interest. Stocksera's public free endpoint at
    `stocksera.pythonanywhere.com/api/...` is currently "Coming Soon" (probed
    2026-05-22) — module is fully optional and falls back to zero-fill.

Dedup vs existing modules (HARD constraint per ship brief):
  - govtrades_features / alt_data_features already cover congress trades +
    lobbying (55k rows). This module does NOT touch congress.
  - This module covers only: WSB velocity, Reddit mentions, dark-pool DIX,
    off-exchange short volume %, news sentiment — orthogonal to govtrades.

All features are .shift(1)-safe: bar at date D only sees rows whose
event_date < D. ApeWisdom snapshot is daily (no intraday); we treat the daily
"mentions" as available at next-bar open (i.e. shift(1)).

Cache:
  - Per-ticker daily snapshot → `data/altdata/stocksera/<TICKER>.parquet`
  - 24h TTL; re-fetch from network only if cache stale or missing.

Env-gates:
  - STOCKSERA_ENABLED=1 (default 0) — enables Stocksera HTTP calls (needs
    STOCKSERA_API_KEY). When 0, only ApeWisdom is queried.
  - STOCKSERA_API_KEY=<key>
  - APEWISDOM_ENABLED=1 (default 1) — disables network entirely when 0.

Features added (8 cols):
  - wsb_mentions_1d                : count — yesterday's r/wallstreetbets mentions
  - wsb_mentions_z_30d             : float — z-score of wsb_mentions vs trailing 30d
  - reddit_velocity_1d             : float — (mentions - mentions_24h_ago) / max(mentions_24h_ago, 1)
  - reddit_upvotes_1d              : count — yesterday's total upvotes (ApeWisdom)
  - reddit_rank                    : int   — yesterday's rank in WSB top-N (0 if not ranked)
  - darkpool_dix_1d                : float — Stocksera dark-pool index (0 if disabled)
  - offexch_short_pct_1d           : float — Stocksera off-exchange short volume % (0 if disabled)
  - news_sentiment_1d              : float — Stocksera news sentiment score in [-1,+1] (0 if disabled)

Graceful failure: any network/cache error → zero-fill. NEVER raises on missing data.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = _ROOT / "data" / "altdata" / "stocksera"
CACHE_TTL_SEC = 24 * 3600  # 24h

ALTDATA_FEATURE_NAMES: list[str] = [
    "wsb_mentions_1d",
    "wsb_mentions_z_30d",
    "reddit_velocity_1d",
    "reddit_upvotes_1d",
    "reddit_rank",
    "darkpool_dix_1d",
    "offexch_short_pct_1d",
    "news_sentiment_1d",
]

APEWISDOM_URL = "https://apewisdom.io/api/v1.0/filter/wallstreetbets/page/{page}"
STOCKSERA_BASE = "https://stocksera.pythonanywhere.com/api"  # public free endpoint


def _enabled_apewisdom() -> bool:
    return os.environ.get("APEWISDOM_ENABLED", "1") == "1"


def _enabled_stocksera() -> bool:
    return os.environ.get("STOCKSERA_ENABLED", "0") == "1"


def _stocksera_key() -> str:
    return os.environ.get("STOCKSERA_API_KEY", "")


def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker.upper()}.parquet"


def _cache_fresh(p: Path) -> bool:
    if not p.exists():
        return False
    age = time.time() - p.stat().st_mtime
    return age < CACHE_TTL_SEC


def _fetch_apewisdom_snapshot(max_pages: int = 5) -> dict[str, dict]:
    """Fetch full WSB ranking (ticker -> {mentions, upvotes, rank, ...}).
    ApeWisdom is free, no key required. ~1300 tickers across ~9 pages.
    """
    import requests  # local import — module-level optional

    out: dict[str, dict] = {}
    for page in range(1, max_pages + 1):
        try:
            r = requests.get(APEWISDOM_URL.format(page=page), timeout=10)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            logger.warning("[altdata] apewisdom page %d failed: %s", page, e)
            break
        def _ii(v):
            try:
                return int(v) if v is not None else 0
            except (TypeError, ValueError):
                return 0

        for row in data.get("results", []):
            tk = (row.get("ticker") or "").upper()
            if not tk:
                continue
            mentions = _ii(row.get("mentions"))
            mentions_prev = _ii(row.get("mentions_24h_ago"))
            out[tk] = {
                "mentions": mentions,
                "upvotes": _ii(row.get("upvotes")),
                "rank": _ii(row.get("rank")),
                "mentions_prev": mentions_prev,
                "velocity": (mentions - mentions_prev) / max(mentions_prev, 1),
            }
        if data.get("current_page", 0) >= data.get("pages", 0):
            break
    return out


def _fetch_stocksera(ticker: str) -> dict[str, float]:
    """Fetch dark-pool, off-exchange short vol, news sentiment for a single ticker.
    Returns dict of feature -> value. Falls back to zero on any error.
    Stocksera API key required (env STOCKSERA_API_KEY). Endpoint may also be
    offline ("Coming Soon" status as of 2026-05-22).
    """
    if not _enabled_stocksera():
        return {}
    import requests

    headers = {}
    key = _stocksera_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"

    out: dict[str, float] = {}
    # Dark pool: latest DIX (single number 0..1)
    try:
        r = requests.get(f"{STOCKSERA_BASE}/dark_pool/?ticker={ticker}", headers=headers, timeout=10)
        if r.ok and r.headers.get("content-type", "").startswith("application/json"):
            j = r.json()
            if isinstance(j, list) and j:
                out["darkpool_dix_1d"] = float(j[0].get("dix", 0.0) or 0.0)
            elif isinstance(j, dict):
                out["darkpool_dix_1d"] = float(j.get("dix", 0.0) or 0.0)
    except Exception as e:
        logger.debug("[altdata] stocksera dark_pool %s failed: %s", ticker, e)

    # Off-exchange short volume %
    try:
        r = requests.get(f"{STOCKSERA_BASE}/short_volume/?ticker={ticker}", headers=headers, timeout=10)
        if r.ok and r.headers.get("content-type", "").startswith("application/json"):
            j = r.json()
            if isinstance(j, list) and j:
                row = j[0]
                tot = float(row.get("total_volume", 0) or 0)
                sh = float(row.get("short_volume", 0) or 0)
                out["offexch_short_pct_1d"] = (sh / tot) if tot > 0 else 0.0
    except Exception as e:
        logger.debug("[altdata] stocksera short_volume %s failed: %s", ticker, e)

    # News sentiment (latest aggregate)
    try:
        r = requests.get(f"{STOCKSERA_BASE}/news_sentiment/?ticker={ticker}", headers=headers, timeout=10)
        if r.ok and r.headers.get("content-type", "").startswith("application/json"):
            j = r.json()
            if isinstance(j, list) and j:
                vals = [float(x.get("sentiment", 0) or 0) for x in j[:20]]
                if vals:
                    out["news_sentiment_1d"] = float(np.mean(vals))
    except Exception as e:
        logger.debug("[altdata] stocksera news_sentiment %s failed: %s", ticker, e)

    return out


_apewisdom_snapshot_cache: Optional[dict[str, dict]] = None
_apewisdom_snapshot_ts: float = 0.0


def _get_apewisdom_snapshot() -> dict[str, dict]:
    """Module-level memoized ApeWisdom snapshot (1h TTL)."""
    global _apewisdom_snapshot_cache, _apewisdom_snapshot_ts
    if _apewisdom_snapshot_cache is not None and (time.time() - _apewisdom_snapshot_ts) < 3600:
        return _apewisdom_snapshot_cache
    if not _enabled_apewisdom():
        _apewisdom_snapshot_cache = {}
        _apewisdom_snapshot_ts = time.time()
        return _apewisdom_snapshot_cache
    snap = _fetch_apewisdom_snapshot()
    _apewisdom_snapshot_cache = snap
    _apewisdom_snapshot_ts = time.time()
    return snap


def fetch_ticker_snapshot(ticker: str, use_cache: bool = True) -> dict[str, float]:
    """Fetch a single ticker's daily snapshot, caching to parquet.
    Returns dict feature_name -> value (single-day point estimate).
    """
    ticker = ticker.upper()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cp = _cache_path(ticker)

    # Cache hit?
    if use_cache and _cache_fresh(cp):
        try:
            df = pd.read_parquet(cp)
            if not df.empty:
                row = df.iloc[-1]
                return {k: float(row[k]) for k in ALTDATA_FEATURE_NAMES if k in df.columns}
        except Exception as e:
            logger.debug("[altdata] cache read failed %s: %s", ticker, e)

    out: dict[str, float] = {k: 0.0 for k in ALTDATA_FEATURE_NAMES}

    # ApeWisdom (single snapshot for all tickers; lookup by ticker)
    snap = _get_apewisdom_snapshot()
    rec = snap.get(ticker, {})
    out["wsb_mentions_1d"] = float(rec.get("mentions", 0))
    out["reddit_upvotes_1d"] = float(rec.get("upvotes", 0))
    out["reddit_rank"] = float(rec.get("rank", 0))
    out["reddit_velocity_1d"] = float(rec.get("velocity", 0.0))

    # Stocksera (key-gated)
    out.update(_fetch_stocksera(ticker))

    # wsb_mentions_z_30d needs historical context — append today's row to cache,
    # then compute z over the trailing 30d window from the cache itself.
    today = datetime.now(timezone.utc).date()
    new_row = {"date": today, **out}
    try:
        if cp.exists():
            hist = pd.read_parquet(cp)
            hist["date"] = pd.to_datetime(hist["date"]).dt.date
            # drop today if present (overwrite)
            hist = hist[hist["date"] != today]
            hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
        else:
            hist = pd.DataFrame([new_row])
        hist = hist.sort_values("date").reset_index(drop=True)
        # z over trailing 30 (exclude current row to avoid lookahead)
        prev = hist.iloc[:-1].tail(30)
        if len(prev) >= 5:
            mu = prev["wsb_mentions_1d"].mean()
            sd = prev["wsb_mentions_1d"].std(ddof=0)
            out["wsb_mentions_z_30d"] = float((out["wsb_mentions_1d"] - mu) / sd) if sd > 0 else 0.0
        hist.loc[hist.index[-1], "wsb_mentions_z_30d"] = out["wsb_mentions_z_30d"]
        hist.to_parquet(cp, index=False)
    except Exception as e:
        logger.warning("[altdata] cache write failed %s: %s", ticker, e)

    return out


def _zero_fill(df: pd.DataFrame) -> pd.DataFrame:
    for col in ALTDATA_FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0.0
    return df


def add_altdata_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Append 8 alt-data features to df. Idempotent. .shift(1)-safe.

    Strategy: alt-data is a *current* snapshot (ApeWisdom only exposes the last
    24h). To stay PIT-safe, we treat the snapshot as known at next-bar open —
    i.e. each bar at date D gets the snapshot value cached at date D-1. For
    historical bars (D < today-1) we have no snapshot — those bars zero-fill.
    This is consistent with the "events only enter the model on the next bar"
    convention used by govtrades_features.py.
    """
    if df is None or len(df) == 0:
        return df
    if all(c in df.columns for c in ALTDATA_FEATURE_NAMES):
        return df

    # Only fetch live if module is enabled
    if not (_enabled_apewisdom() or _enabled_stocksera()):
        return _zero_fill(df)

    try:
        snap = fetch_ticker_snapshot(ticker)
    except Exception as e:
        logger.warning("[altdata] snapshot fetch failed %s: %s", ticker, e)
        return _zero_fill(df)

    # Determine bar dates
    if isinstance(df.index, pd.DatetimeIndex):
        bar_dates = df.index
    elif "date" in df.columns:
        bar_dates = pd.DatetimeIndex(pd.to_datetime(df["date"]))
    else:
        return _zero_fill(df)
    if bar_dates.tz is not None:
        bar_dates = bar_dates.tz_convert(None)

    # Latest snapshot represents "yesterday's WSB activity, known today".
    # For the most-recent bar in df we apply the snapshot value. Earlier bars
    # use the cached parquet history (if any) — looked up by date.
    cp = _cache_path(ticker)
    hist_by_date: dict = {}
    if cp.exists():
        try:
            hist = pd.read_parquet(cp)
            hist["date"] = pd.to_datetime(hist["date"]).dt.date
            for _, row in hist.iterrows():
                hist_by_date[row["date"]] = {k: float(row[k]) for k in ALTDATA_FEATURE_NAMES if k in row}
        except Exception:
            pass

    # Build column arrays — for each bar date D, lookup snapshot from D-1
    # (next-bar-open convention). If no cache hit, zero.
    n = len(bar_dates)
    cols: dict[str, np.ndarray] = {k: np.zeros(n, dtype=np.float64) for k in ALTDATA_FEATURE_NAMES}
    today = datetime.now(timezone.utc).date()
    for i, ts in enumerate(bar_dates):
        d_prev = (ts.normalize().to_pydatetime().date() - timedelta(days=1))
        rec = hist_by_date.get(d_prev)
        if rec is None and i == n - 1 and ts.date() >= today - timedelta(days=1):
            # Most recent bar — use the freshly-fetched snapshot
            rec = snap
        if rec:
            for k in ALTDATA_FEATURE_NAMES:
                cols[k][i] = float(rec.get(k, 0.0))

    for k in ALTDATA_FEATURE_NAMES:
        if k not in df.columns:
            df[k] = cols[k]
    return df


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    tk = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    print(f"[smoke] ticker={tk} days={days}")
    snap = fetch_ticker_snapshot(tk)
    print(f"[smoke] snapshot: {json.dumps(snap, indent=2)}")

    idx = pd.date_range(end=pd.Timestamp.utcnow().date(), periods=days, freq="B")
    demo = pd.DataFrame({"close": np.linspace(100, 110, len(idx))}, index=idx)
    out = add_altdata_features(demo, tk)
    print(f"[smoke] input cols: 1, output cols: {out.shape[1]}")
    print(out[ALTDATA_FEATURE_NAMES].tail(5).to_string())
    print(
        "[smoke] non-zero feature counts:",
        {k: int((out[k] != 0).sum()) for k in ALTDATA_FEATURE_NAMES},
    )
