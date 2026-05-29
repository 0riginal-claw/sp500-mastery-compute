"""
GovTrades data access — thin wrapper over govtrades_cache_loader.

The actual SQLite loader is at `scripts/govtrades_cache_loader.py` (sibling of this module).
This wrapper re-exports the public API and adds coverage metadata for discoverability.

Lab-side canonical DB:  /My Drive/Ph0tis/Gov-Trades/data/govtrades.db
Local 2TB working copy: /Volumes/ZG-2TB/zg/govtrades/data/govtrades.db
Data source:            QuiverQuant Tier: Hobbyist ($30/mo)

Top funcs (re-exported from govtrades_cache_loader when importable):
  get_congress_trades(ticker, start=None, end=None)
  get_lobbying(ticker, start=None, end=None)
  get_contracts(ticker, start=None, end=None)        — gov_contracts_awards
  get_contracts_quarterly(ticker)                    — gov_contracts_quarterly
  get_offexchange(ticker, start=None, end=None)      — NEW THIS SESSION (Dark Pool Index)
  get_flights(ticker, start=None, end=None)          — NEW THIS SESSION (corporate jet)
  get_politicians()                                  — all politicians (bio_guide_id, party, etc.)
  get_corporate_donors(ticker)                       — corporate PAC donations

Plus knowledge metadata (always available):
  coverage()  — table list, row counts (latest known), tier info
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

_COVERAGE = {
    "tier": "QuiverQuant Hobbyist ($30/mo)",
    "endpoints_unlocked": [
        "live/congresstrading", "live/housetrading", "live/senatetrading",
        "live/congress/politicians", "live/lobbying",
        "live/govcontracts", "live/govcontractsall",
        "live/offexchange (NEW)", "historical/offexchange (NEW)",
        "live/flights (NEW)",
        "bulk/congresstrading", "bulk/congress/politicians", "bulk/corporatedonors",
        "historical/congresstrading/{tk}", "historical/lobbying/{tk}", "historical/govcontracts/{tk}",
    ],
    "endpoints_requires_trader_tier": [
        "live/insiders", "live/wallstreetbets", "live/politicalbeta",
        "live/allpatents", "live/quivernews", "live/topshareholders",
        "historical/executivecompensation", "historical/twitter",
        "historical/spacs", "historical/flights",
    ],
    "tables": [
        "politicians", "congress_trades", "lobbying",
        "gov_contracts_quarterly", "gov_contracts_awards",
        "corporate_donors",
        "offexchange",  # NEW
        "flights",      # NEW
        "fetch_log",
    ],
    "new_this_session": ["offexchange (Dark Pool Index live + historical)", "flights (corporate jet live)"],
    "as_of_session": "2026-05-28",
}


def coverage() -> Dict[str, Any]:
    """GovTrades data coverage: endpoints, tables, tier info, what's new this session."""
    return dict(_COVERAGE)


def _underlying():
    try:
        import govtrades_cache_loader as _g  # type: ignore
        return _g
    except ImportError:
        return None


def _dispatch(method_name: str, *args, **kwargs):
    u = _underlying()
    if u is None:
        raise NotImplementedError(
            "govtrades_cache_loader not importable from this context. "
            "Ensure scripts/ is on PYTHONPATH or call from a sibling script."
        )
    if hasattr(u, method_name):
        return getattr(u, method_name)(*args, **kwargs)
    # Class-based API
    cache_cls = getattr(u, "GovTradesCache", None)
    if cache_cls and hasattr(cache_cls, method_name):
        return getattr(cache_cls, method_name)(*args, **kwargs)
    raise NotImplementedError(f"govtrades_cache_loader has no {method_name}")


def get_congress_trades(ticker: str, start: Optional[str] = None,
                        end: Optional[str] = None) -> List[Dict[str, Any]]:
    """Congress trades for a ticker."""
    return _dispatch("get_congress_trades", ticker, start=start, end=end)


def get_lobbying(ticker: str, start: Optional[str] = None,
                 end: Optional[str] = None) -> List[Dict[str, Any]]:
    """Lobbying records for a ticker."""
    return _dispatch("get_lobbying", ticker, start=start, end=end)


def get_contracts(ticker: str, start: Optional[str] = None,
                  end: Optional[str] = None) -> List[Dict[str, Any]]:
    """Gov contract awards for a ticker."""
    return _dispatch("get_contracts", ticker, start=start, end=end)


def get_contracts_quarterly(ticker: str) -> List[Dict[str, Any]]:
    """Quarterly aggregate gov contract amounts for a ticker."""
    return _dispatch("get_contracts_quarterly", ticker)


def get_offexchange(ticker: str, start: Optional[str] = None,
                    end: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    NEW THIS SESSION: Off-exchange / dark-pool short volume.
    Columns: ticker, date, otc_short, otc_total, dpi (Dark Pool Index = otc_short/otc_total).
    """
    return _dispatch("get_offexchange", ticker, start=start, end=end)


def get_flights(ticker: str, start: Optional[str] = None,
                end: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    NEW THIS SESSION: Corporate jet flight tracking (sparse signal, M&A leading indicator).
    Columns: ticker, date, departure_city, arrival_city.
    """
    return _dispatch("get_flights", ticker, start=start, end=end)


def get_politicians() -> List[Dict[str, Any]]:
    """All politicians with bio_guide_id, party, chamber, etc."""
    return _dispatch("get_politicians")


def get_corporate_donors(ticker: str) -> List[Dict[str, Any]]:
    """Corporate PAC donations to politicians, keyed by ticker."""
    return _dispatch("get_corporate_donors", ticker)


def _clear_cache():
    pass


if __name__ == "__main__":
    import json
    print(json.dumps(coverage(), indent=2))
