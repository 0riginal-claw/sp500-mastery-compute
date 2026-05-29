"""
Edgar data access — thin wrapper over edgar_cache_loader.

The actual SQLite loader is at `scripts/edgar_cache_loader.py` (sibling of this module).
This wrapper re-exports the public API and adds coverage metadata for discoverability.

Lab-side canonical DB:  /My Drive/claudes test/data/edgar/data/edgar.db
Source-of-truth DB:     /My Drive/Ph0tis/Edgar/data/index/edgar.db
Local 2TB working copy: /Users/orginal/.zg/edgar_state/edgar.db (symlink to /Volumes/ZG-2TB/zg/...)

Top funcs (re-exported from edgar_cache_loader when importable):
  get_filings(ticker, start=None, end=None, form=None)
  get_form4(ticker, start=None, end=None) — insider transactions (pd.DataFrame)
  get_form4_insider_cluster(ticker, end, lookback_d=5) — cluster metrics
  get_8k(ticker)      — 8-K material events
  get_10k(ticker)     — annual reports
  get_10q(ticker)     — quarterly reports
  table_counts()      — row counts per form type

Plus knowledge metadata (always available):
  coverage()  — date range, ticker count, form completeness
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

_COVERAGE = {
    "tickers": 502,
    "rows_total": 58502,
    "date_range_start": "2020-01-02",
    "date_range_end": "2026-05-28",
    "forms_complete": ["10-K", "10-Q", "8-K", "DEF 14A", "S-1", "4"],
    "forms_partial_backfill_pending": ["Form 3", "Form 5", "SC 13G", "SC 13D"],
    "form4_transactions_table": "form4_transactions (one row per insider txn)",
    "as_of_session": "2026-05-28",
}


def coverage() -> Dict[str, Any]:
    """Edgar data coverage: tickers, date range, form types."""
    return dict(_COVERAGE)


# Lazy re-export of the underlying loader. If the import fails (e.g. Drive FUSE blind to
# the file at import time), expose stubs that raise with a clear message.
def _underlying():
    try:
        import edgar_cache_loader as _e  # type: ignore
        return _e
    except ImportError:
        return None


def get_filings(ticker: str, start: Optional[str] = None,
                end: Optional[str] = None, form: Optional[str] = None,
                form_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get filings for a ticker. Forwards to edgar_cache_loader.get_filings if available.

    The outward-facing kwarg ``form=`` is preserved for backward compatibility and is
    aliased to ``form_type=`` (the underlying ``EdgarCache.get_filings`` signature).
    Callers may pass either ``form=`` or ``form_type=``; if both are given,
    ``form_type=`` wins.
    """
    # Alias: prefer explicit form_type, fall back to form for back-compat.
    _form_type = form_type if form_type is not None else form
    u = _underlying()
    if u and hasattr(u, "get_filings"):
        return u.get_filings(ticker, start=start, end=end, form_type=_form_type)
    if u and hasattr(u, "EdgarCache"):
        # Underlying API exposes a classmethod on EdgarCache.
        cache = u.EdgarCache
        return cache.get_filings(ticker, form_type=_form_type, start=start, end=end)
    raise NotImplementedError(
        "edgar_cache_loader not importable from this context. "
        "Ensure scripts/ is on PYTHONPATH or call from a sibling script."
    )


# ----------------------------------------------------------------------
# Form 4 insider-transaction helpers (read form4_transactions table)
# ----------------------------------------------------------------------

_EDGAR_DB_CANDIDATES = (
    # Local 2TB working copy is the freshest (where the backfill writes).
    "/Volumes/ZG-2TB/zg/edgar_state/index/edgar.db",
    "/Users/orginal/.zg/edgar_state/index/edgar.db",
    # Lab-side canonical mirror.
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/"
    "My Drive/claudes test/data/edgar/data/edgar.db",
    # Drive source-of-truth.
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/"
    "My Drive/Ph0tis/Edgar/data/index/edgar.db",
)


def _edgar_db_path() -> Optional[str]:
    for p in _EDGAR_DB_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def _connect_edgar_ro() -> sqlite3.Connection:
    path = _edgar_db_path()
    if path is None:
        raise FileNotFoundError(
            f"edgar.db not found at any of: {_EDGAR_DB_CANDIDATES}"
        )
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10.0)


def get_form4(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Any:
    """Insider Form 4 transactions for a ticker, as a pandas DataFrame.

    Columns:
      filing_accession_no, issuer_ticker, issuer_cik, insider_cik,
      insider_name, insider_relationship (JSON list), officer_title,
      txn_date, security_title, code, shares, price_per_share, value,
      direction, shares_owned_after, is_derivative, filed_at.

    If pandas isn't available, returns List[Dict] instead.
    """
    if not ticker:
        try:
            import pandas as pd
            return pd.DataFrame()
        except ImportError:
            return []

    sql = (
        "SELECT filing_accession_no, issuer_ticker, issuer_cik, insider_cik, "
        "insider_name, insider_relationship, officer_title, txn_date, "
        "security_title, code, shares, price_per_share, value, direction, "
        "shares_owned_after, is_derivative, filed_at, source_format "
        "FROM form4_transactions WHERE issuer_ticker = ?"
    )
    params: List = [ticker.upper()]
    if start:
        sql += " AND txn_date >= ?"
        params.append(start)
    if end:
        sql += " AND txn_date <= ?"
        params.append(end)
    sql += " ORDER BY txn_date, txn_id"

    try:
        with _connect_edgar_ro() as con:
            cur = con.execute(sql, params)
            cols = [c[0] for c in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    except Exception:
        rows = []

    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        if not df.empty:
            df["txn_date"] = pd.to_datetime(df["txn_date"], errors="coerce")
            df["filed_at"] = pd.to_datetime(df["filed_at"], errors="coerce")
            df["is_derivative"] = df["is_derivative"].astype(bool)
        return df
    except ImportError:
        return rows


def get_form4_insider_cluster(
    ticker: str,
    end: str,
    lookback_d: int = 5,
) -> Dict[str, Any]:
    """Cluster-buy metrics for the alt-data overlay (e.g. InsiderClusterBuy_5d_2plus).

    Looks at non-derivative Form 4 P (purchase) transactions in the window
    [end - lookback_d, end]. Returns:

        {
            "ticker": str,
            "window_start": "YYYY-MM-DD",
            "window_end": "YYYY-MM-DD",
            "n_buyers": int,             # distinct insider_cik with code='P'
            "n_buy_filings": int,        # distinct filings with any P
            "total_shares_bought": float,
            "total_value_bought": float,
            "directors_buying": int,
            "officers_buying": int,
            "buyers": [{"insider_cik", "insider_name", "shares", "value"}, ...]
        }

    A `cluster_buy_2plus` flag is True iff n_buyers >= 2 (at least two distinct
    insiders purchased shares in the window).
    """
    try:
        end_dt = datetime.strptime(end[:10], "%Y-%m-%d")
    except Exception:
        return {"ticker": ticker, "n_buyers": 0, "cluster_buy_2plus": False,
                "error": f"bad_end_date: {end}"}
    start_dt = end_dt - timedelta(days=int(lookback_d))
    start = start_dt.strftime("%Y-%m-%d")
    end = end_dt.strftime("%Y-%m-%d")

    sql = (
        "SELECT insider_cik, insider_name, insider_relationship, shares, value "
        "FROM form4_transactions "
        "WHERE issuer_ticker = ? AND code = 'P' AND is_derivative = 0 "
        "AND direction = 'A' AND txn_date >= ? AND txn_date <= ?"
    )
    try:
        with _connect_edgar_ro() as con:
            rows = con.execute(sql, (ticker.upper(), start, end)).fetchall()
    except Exception as e:
        return {
            "ticker": ticker.upper(), "window_start": start, "window_end": end,
            "n_buyers": 0, "cluster_buy_2plus": False, "error": str(e),
        }

    buyers_agg: Dict[str, Dict[str, Any]] = {}
    n_director = 0
    n_officer = 0
    for cik, name, rels_json, shares, value in rows:
        rels = []
        try:
            rels = json.loads(rels_json or "[]")
        except Exception:
            pass
        b = buyers_agg.setdefault(cik, {
            "insider_cik": cik,
            "insider_name": name,
            "shares": 0.0,
            "value": 0.0,
            "relationships": rels,
        })
        b["shares"] += float(shares or 0)
        b["value"] += float(value or 0)
        if "director" in rels:
            n_director += 1
        if "officer" in rels:
            n_officer += 1

    n_buyers = len(buyers_agg)
    return {
        "ticker": ticker.upper(),
        "window_start": start,
        "window_end": end,
        "lookback_d": int(lookback_d),
        "n_buyers": n_buyers,
        "n_buy_filings": len(rows),
        "total_shares_bought": sum(b["shares"] for b in buyers_agg.values()),
        "total_value_bought": sum(b["value"] for b in buyers_agg.values()),
        "directors_buying": n_director,
        "officers_buying": n_officer,
        "cluster_buy_2plus": n_buyers >= 2,
        "buyers": list(buyers_agg.values()),
    }


def get_8k(ticker: str) -> List[Dict[str, Any]]:
    """8-K material events for a ticker."""
    return get_filings(ticker, form="8-K")


def get_10k(ticker: str) -> List[Dict[str, Any]]:
    """10-K annual reports for a ticker."""
    return get_filings(ticker, form="10-K")


def get_10q(ticker: str) -> List[Dict[str, Any]]:
    """10-Q quarterly reports for a ticker."""
    return get_filings(ticker, form="10-Q")


def table_counts() -> Dict[str, int]:
    """Row counts per form type. Forwards to underlying loader or returns static estimate."""
    u = _underlying()
    if u and hasattr(u, "table_counts"):
        return u.table_counts()
    return {"_note": "underlying loader not importable", "total_rows_estimate": _COVERAGE["rows_total"]}


def _clear_cache():
    pass


if __name__ == "__main__":
    import json
    print(json.dumps(coverage(), indent=2))
