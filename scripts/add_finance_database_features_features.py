"""
add_finance_database_features_features.py — FinanceDatabase metadata features.

Source: github:JerBouma/FinanceDatabase (MIT license, no paid API required).
Features added (4): fdb_sector, fdb_industry, fdb_market_cap, fdb_exchange.

NO-LOOKAHEAD AUDIT
------------------
All four features are static ticker-level metadata from FinanceDatabase:
  sector, industry, market_cap, exchange

They describe fixed company attributes that do not change bar-to-bar (and
rarely change at all). There is NO time-varying quantity involved — no
price, volume, return, or signal that could leak future information.
Because they are constant over the entire DataFrame, .shift(1) is not
needed and would produce identical values (except NaN on the first row,
which is dropped by v10's dropna guard anyway). The no-lookahead property
is satisfied by construction: static metadata carries zero predictive
information about the direction of future price moves beyond what the model
can infer from the ticker identity itself.

Integration cost: LOW — single in-process pandas lookup, ~250ms first call,
cached thereafter via module-level dict.

Expected lift: ~1.5% improvement in CV AUC per feature-wiring spec.
"""

from __future__ import annotations

from typing import Optional
import logging

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column names exported for module_feature_counts tracking
# ---------------------------------------------------------------------------

FDB_FEATURE_NAMES: list[str] = [
    "fdb_sector",
    "fdb_industry",
    "fdb_market_cap",
    "fdb_exchange",
]

FDB_FEATURE_COUNT: int = len(FDB_FEATURE_NAMES)

# ---------------------------------------------------------------------------
# Static encoding maps (deterministic — same across all processes/sessions)
# ---------------------------------------------------------------------------

_SECTOR_MAP: dict[str, int] = {
    "Communication Services": 1,
    "Consumer Discretionary": 2,
    "Consumer Staples": 3,
    "Energy": 4,
    "Financials": 5,
    "Health Care": 6,
    "Industrials": 7,
    "Information Technology": 8,
    "Materials": 9,
    "Real Estate": 10,
    "Utilities": 11,
}

_MARKET_CAP_MAP: dict[str, int] = {
    "Nano Cap": 1,
    "Micro Cap": 2,
    "Small Cap": 3,
    "Mid Cap": 4,
    "Large Cap": 5,
    "Mega Cap": 6,
}

# Module-level cache: ticker → (sector_int, industry_int, mktcap_int, exchange_int)
_TICKER_CACHE: dict[str, tuple[int, int, int, int]] = {}
_FDB_LOADED: bool = False
_FDB_DF: Optional[pd.DataFrame] = None
# industry → int and exchange → int derived at load time
_INDUSTRY_MAP: dict[str, int] = {}
_EXCHANGE_MAP: dict[str, int] = {}


def _load_fdb() -> None:
    """Load FinanceDatabase equities once and build encoding maps."""
    global _FDB_LOADED, _FDB_DF, _INDUSTRY_MAP, _EXCHANGE_MAP
    if _FDB_LOADED:
        return
    try:
        import financedatabase as fd
        eq = fd.Equities()
        _FDB_DF = eq.select()
        # Build industry and exchange encodings from full dataset
        industries = sorted(_FDB_DF["industry"].dropna().unique())
        _INDUSTRY_MAP = {v: i + 1 for i, v in enumerate(industries)}
        exchanges = sorted(_FDB_DF["exchange"].dropna().unique())
        _EXCHANGE_MAP = {v: i + 1 for i, v in enumerate(exchanges)}
        logger.info("[fdb] Loaded %d equities; %d industries, %d exchanges",
                    len(_FDB_DF), len(_INDUSTRY_MAP), len(_EXCHANGE_MAP))
    except Exception as e:
        logger.warning("[fdb] financedatabase load failed: %s — will zero-fill", e)
        _FDB_DF = None
    _FDB_LOADED = True


def _lookup_ticker(ticker: str) -> tuple[int, int, int, int]:
    """Return (sector_int, industry_int, market_cap_int, exchange_int) for ticker."""
    if ticker in _TICKER_CACHE:
        return _TICKER_CACHE[ticker]

    _load_fdb()
    result = (0, 0, 0, 0)
    if _FDB_DF is not None and ticker in _FDB_DF.index:
        row = _FDB_DF.loc[ticker]
        # When multiple rows share the same symbol, take the first
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        sec_int = _SECTOR_MAP.get(str(row.get("sector", "") or ""), 0)
        ind_int = _INDUSTRY_MAP.get(str(row.get("industry", "") or ""), 0)
        mcap_int = _MARKET_CAP_MAP.get(str(row.get("market_cap", "") or ""), 0)
        exch_int = _EXCHANGE_MAP.get(str(row.get("exchange", "") or ""), 0)
        result = (sec_int, ind_int, mcap_int, exch_int)

    _TICKER_CACHE[ticker] = result
    return result


def compute_add_finance_database_features_features(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
) -> pd.DataFrame:
    """Append 4 FinanceDatabase metadata columns to df.

    All columns are constant (static metadata) across every bar for the
    given ticker. Zero-filled when ticker is unknown or financedatabase
    is unavailable.

    Args:
        df: Feature DataFrame indexed by date (ts).
        ticker: Stock symbol used for the lookup.

    Returns:
        df with four new columns appended:
          fdb_sector      — int 0-11 (0 = unknown)
          fdb_industry    — int 0-N  (0 = unknown)
          fdb_market_cap  — int 0-6  (0 = unknown, 6 = Mega Cap)
          fdb_exchange    — int 0-M  (0 = unknown)
    """
    sec_int, ind_int, mcap_int, exch_int = (0, 0, 0, 0)
    if ticker:
        try:
            sec_int, ind_int, mcap_int, exch_int = _lookup_ticker(ticker)
        except Exception as e:
            logger.warning("[fdb] lookup failed for %s: %s — zero-filling", ticker, e)

    df = df.copy()
    df["fdb_sector"] = sec_int
    df["fdb_industry"] = ind_int
    df["fdb_market_cap"] = mcap_int
    df["fdb_exchange"] = exch_int
    return df
