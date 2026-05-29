"""
News knowledge — durable Alpaca News article cache, REST + WebSocket persisted.

Cache architecture (built 2026-05-28):
  - Hot SQLite:  /Volumes/ZG-2TB/zg/news_cache/alpaca_news.db   (WAL, 2TB tier)
  - Drive mirror: /My Drive/AI-Tools/data/news_cache/           (durable snapshots)
  - REST backfill fetcher: /Volumes/ZG-2TB/zg/news_cache/historical_news_fetcher.py
  - WS persister daemon:   /Volumes/ZG-2TB/zg/news_cache/ws_news_persister.py
                           launchd plist com.zg.alpaca_news_ws

Schema:
  articles(id PK, headline, summary, content, author, url, source,
           published_utc, updated_utc, symbols_json, sentiment_score,
           sentiment_label, ingested_utc, ingest_source)
  article_symbols(article_id, symbol)  -- join table for fast per-ticker lookup
  backfill_cursor(symbol, window_start_utc, window_end_utc, ...)
  daemon_heartbeat(daemon_name, pid, last_event_utc, ...)

Top funcs:
  get_news(ticker, start=None, end=None)  -- DataFrame of articles for a ticker
  get_latest(ticker, n=10)                -- most recent N for a ticker
  get_article(article_id)                 -- single record by Alpaca id
  search(query, limit=50)                 -- LIKE-search on headline+summary
  coverage()                              -- rows, tickers, date range, daemon status
  daemon_status()                         -- live WS daemon heartbeat
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import pandas as _pd
    _HAS_PD = True
except ImportError:  # graceful degradation
    _HAS_PD = False
    _pd = None  # type: ignore

# Path resolution — 2TB hot tier preferred; Drive mirror as fallback
_DEFAULT_DB_2TB = "/Volumes/ZG-2TB/zg/news_cache/alpaca_news.db"
_DRIVE_MIRROR   = "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/data/news_cache/alpaca_news.db"


def _db_path() -> str:
    p = os.environ.get("ALPACA_NEWS_DB")
    if p and Path(p).exists():
        return p
    if Path(_DEFAULT_DB_2TB).exists():
        return _DEFAULT_DB_2TB
    if Path(_DRIVE_MIRROR).exists():
        return _DRIVE_MIRROR
    # Even if missing, return the 2TB default — the caller will get a clearer
    # error from sqlite3.connect than from this helper.
    return _DEFAULT_DB_2TB


def _connect(readonly: bool = True) -> sqlite3.Connection:
    p = _db_path()
    if readonly:
        # URI mode for read-only safety
        conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=30)
    else:
        conn = sqlite3.connect(p, timeout=60)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=60000")
    conn.row_factory = sqlite3.Row
    return conn


def _rows_to_df(rows: List[sqlite3.Row]) -> Any:
    if _HAS_PD:
        return _pd.DataFrame([dict(r) for r in rows])
    return [dict(r) for r in rows]


def get_news(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    *,
    limit: Optional[int] = None,
    include_content: bool = False,
) -> Any:
    """Return all news articles where the ticker appears in `symbols`.

    Args:
      ticker: symbol (case-insensitive)
      start:  ISO date string lower bound on published_utc (inclusive)
      end:    ISO date string upper bound on published_utc (inclusive)
      limit:  cap rows; default None = all
      include_content: include full HTML body column (large)

    Returns DataFrame (if pandas installed) or list of dicts.
    """
    ticker = ticker.upper()
    cols = "a.id, a.headline, a.summary, a.author, a.url, a.source, a.published_utc, a.updated_utc, a.symbols_json, a.sentiment_score, a.sentiment_label, a.ingest_source"
    if include_content:
        cols += ", a.content"
    sql = (
        f"SELECT {cols} "
        "FROM article_symbols s JOIN articles a ON a.id = s.article_id "
        "WHERE s.symbol = ?"
    )
    params: list[Any] = [ticker]
    if start:
        sql += " AND a.published_utc >= ?"
        params.append(start)
    if end:
        sql += " AND a.published_utc <= ?"
        params.append(end)
    sql += " ORDER BY a.published_utc DESC"
    if limit:
        sql += f" LIMIT {int(limit)}"
    with _connect(readonly=True) as conn:
        rows = list(conn.execute(sql, params).fetchall())
    return _rows_to_df(rows)


def get_latest(ticker: str, n: int = 10, *, include_content: bool = False) -> Any:
    """Return the most recent N articles for a ticker."""
    return get_news(ticker, limit=n, include_content=include_content)


def get_article(article_id: int) -> Optional[Dict[str, Any]]:
    """Return a single article record by Alpaca article id."""
    with _connect(readonly=True) as conn:
        row = conn.execute(
            "SELECT * FROM articles WHERE id = ?", (int(article_id),)
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    try:
        d["symbols"] = json.loads(d.pop("symbols_json", "[]"))
    except Exception:
        d["symbols"] = []
    return d


def search(query: str, limit: int = 50, *, since: Optional[str] = None) -> Any:
    """LIKE-search on headline + summary. Slow on large DB — for ad-hoc use only."""
    q = f"%{query}%"
    sql = (
        "SELECT id, headline, summary, source, url, published_utc, symbols_json "
        "FROM articles WHERE (headline LIKE ? OR summary LIKE ?)"
    )
    params: list[Any] = [q, q]
    if since:
        sql += " AND published_utc >= ?"
        params.append(since)
    sql += f" ORDER BY published_utc DESC LIMIT {int(limit)}"
    with _connect(readonly=True) as conn:
        rows = list(conn.execute(sql, params).fetchall())
    return _rows_to_df(rows)


@lru_cache(maxsize=1)
def coverage() -> Dict[str, Any]:
    """Cache stats: row counts, ticker count, date range, daemon status."""
    out: Dict[str, Any] = {
        "db_path": _db_path(),
        "schema": "v1 (2026-05-28)",
        "articles": 0,
        "article_symbol_rows": 0,
        "distinct_tickers": 0,
        "date_range": [None, None],
        "by_source": {},
        "by_ingest_source": {},
        "backfill_status": {},
        "daemon": None,
        "as_of": datetime.now(timezone.utc).isoformat(),
    }
    try:
        with _connect(readonly=True) as conn:
            r = conn.execute("SELECT COUNT(*) FROM articles").fetchone()
            out["articles"] = r[0] if r else 0
            r = conn.execute("SELECT COUNT(*) FROM article_symbols").fetchone()
            out["article_symbol_rows"] = r[0] if r else 0
            r = conn.execute("SELECT COUNT(DISTINCT symbol) FROM article_symbols").fetchone()
            out["distinct_tickers"] = r[0] if r else 0
            r = conn.execute("SELECT MIN(published_utc), MAX(published_utc) FROM articles").fetchone()
            out["date_range"] = [r[0], r[1]] if r else [None, None]
            for src, c in conn.execute("SELECT source, COUNT(*) FROM articles GROUP BY source ORDER BY 2 DESC LIMIT 20"):
                out["by_source"][src or "_unknown"] = c
            for src, c in conn.execute("SELECT ingest_source, COUNT(*) FROM articles GROUP BY ingest_source"):
                out["by_ingest_source"][src or "_unknown"] = c
            for st, c in conn.execute("SELECT status, COUNT(*) FROM backfill_cursor GROUP BY status"):
                out["backfill_status"][st or "_unknown"] = c
            r = conn.execute(
                "SELECT daemon_name, pid, last_event_utc, last_heartbeat, "
                "events_since_start, started_utc, status "
                "FROM daemon_heartbeat WHERE daemon_name = 'alpaca_news_ws'"
            ).fetchone()
            if r:
                out["daemon"] = dict(r)
    except sqlite3.OperationalError as e:
        out["error"] = str(e)
    return out


def daemon_status() -> Dict[str, Any]:
    """Just the WS daemon heartbeat row."""
    c = coverage()
    return c.get("daemon") or {"status": "no_heartbeat_row"}


def _clear_cache() -> None:
    coverage.cache_clear()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=None)
    ap.add_argument("--latest", type=int, default=5)
    args = ap.parse_args()
    print(json.dumps(coverage(), indent=2, default=str))
    if args.ticker:
        rows = get_latest(args.ticker, n=args.latest)
        if _HAS_PD:
            print(rows[["published_utc", "source", "headline"]].to_string(index=False))
        else:
            for r in rows:
                print(r.get("published_utc"), "|", r.get("source"), "|", (r.get("headline") or "")[:80])
