"""
Edgar data access — thin wrapper over edgar_cache_loader.

The actual SQLite loader is at `scripts/edgar_cache_loader.py` (sibling of this module).
This wrapper re-exports the public API and adds coverage metadata for discoverability.

Lab-side canonical DB:  /My Drive/claudes test/data/edgar/data/edgar.db
Source-of-truth DB:     /My Drive/Ph0tis/Edgar/data/index/edgar.db
Local 2TB working copy: /Users/orginal/.zg/edgar_state/edgar.db (symlink to /Volumes/ZG-2TB/zg/...)

Top funcs (re-exported from edgar_cache_loader when importable):
  get_filings(ticker, start=None, end=None, form=None)
  get_form4(ticker)   — insider transactions
  get_8k(ticker)      — 8-K material events
  get_10k(ticker)     — annual reports
  get_10q(ticker)     — quarterly reports
  table_counts()      — row counts per form type

Plus knowledge metadata (always available):
  coverage()  — date range, ticker count, form completeness
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

_COVERAGE = {
    "tickers": 502,
    "rows_total": 58502,
    "date_range_start": "2020-01-02",
    "date_range_end": "2026-05-28",
    "forms_complete": ["10-K", "10-Q", "8-K", "DEF 14A", "S-1"],
    "forms_partial_backfill_pending": ["Form 3", "Form 4", "Form 5", "SC 13G", "SC 13D"],
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
                end: Optional[str] = None, form: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get filings for a ticker. Forwards to edgar_cache_loader.get_filings if available."""
    u = _underlying()
    if u and hasattr(u, "get_filings"):
        return u.get_filings(ticker, start=start, end=end, form=form)
    if u and hasattr(u, "EdgarCache"):
        # Older API may expose a class
        cache = u.EdgarCache()
        return cache.get_filings(ticker, start=start, end=end, form=form)
    raise NotImplementedError(
        "edgar_cache_loader not importable from this context. "
        "Ensure scripts/ is on PYTHONPATH or call from a sibling script."
    )


def get_form4(ticker: str) -> List[Dict[str, Any]]:
    """Insider Form 4 transactions for a ticker."""
    return get_filings(ticker, form="4")


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
