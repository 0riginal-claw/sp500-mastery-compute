"""OpenBB SDK unified data loader.

Top-10 ship #8 (2026-05-22): consolidate ad-hoc FMP/Tiingo/Polygon/Finnhub
adapters behind a single OpenBB facade. Lower bug surface, single import,
provider fallback chain managed by OpenBB.

Public API:
    load_equity_history(ticker, start, end, provider=None) -> pd.DataFrame
    load_equity_fundamentals(ticker, statement='income', provider=None) -> pd.DataFrame
    load_equity_news(ticker, limit=20, provider=None) -> pd.DataFrame
    load_economy_indicator(symbol, start, end, provider=None) -> pd.DataFrame
    list_available_providers(category='equity') -> list[str]
    health_check() -> dict

Activation:
    OPENBB_LOADER_ENABLED=1  (default OFF — additive only, no forced refactor)

Provider fallback chain (auto-picked per call when provider=None):
    equity.price.historical: yfinance -> fmp -> tiingo -> polygon -> intrinio
    equity.fundamental.*:    fmp -> intrinio -> sec
    equity.news:             benzinga -> fmp -> tiingo -> intrinio
    economy.indicator:       fred -> econdb -> oecd

Safety:
- No money/trading. Read-only data fetch.
- Graceful degradation: returns empty DataFrame on full chain failure.
- Logs full chain attempts at DEBUG.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

LOG = logging.getLogger(__name__)

# Env-gate. Default OFF — callers MUST opt-in.
ENABLED = os.environ.get("OPENBB_LOADER_ENABLED", "0") == "1"

# Provider fallback chains. Order = preferred -> fallback.
_FALLBACK_CHAINS: dict[str, list[str]] = {
    "equity.price.historical": ["yfinance", "fmp", "tiingo", "polygon", "intrinio"],
    "equity.fundamental.income": ["fmp", "intrinio", "sec"],
    "equity.fundamental.balance": ["fmp", "intrinio", "sec"],
    "equity.fundamental.cash": ["fmp", "intrinio", "sec"],
    "equity.news": ["benzinga", "fmp", "tiingo", "intrinio"],
    "economy.indicator": ["fred", "econdb", "oecd"],
}


def _obb():
    """Lazy import. OpenBB cold-start is ~10s — defer until first call."""
    try:
        from openbb import obb  # noqa: WPS433
        return obb
    except Exception as e:  # pragma: no cover
        LOG.warning("openbb import failed: %s", e)
        return None


def _to_df(result: Any) -> pd.DataFrame:
    """Coerce OpenBB result to DataFrame. Handles OBBject + raw list/dict."""
    if result is None:
        return pd.DataFrame()
    # OBBject has .to_df()
    if hasattr(result, "to_df"):
        try:
            return result.to_df()
        except Exception:
            pass
    if hasattr(result, "to_dataframe"):
        try:
            return result.to_dataframe()
        except Exception:
            pass
    if isinstance(result, pd.DataFrame):
        return result
    if isinstance(result, (list, tuple)):
        try:
            return pd.DataFrame(result)
        except Exception:
            return pd.DataFrame()
    if isinstance(result, dict):
        try:
            return pd.DataFrame([result])
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _try_chain(endpoint_fn, providers: list[str], **kwargs) -> pd.DataFrame:
    """Try each provider in order. Return first non-empty DF or empty."""
    last_err: str = ""
    for prov in providers:
        try:
            res = endpoint_fn(provider=prov, **kwargs)
            df = _to_df(res)
            if not df.empty:
                LOG.debug("openbb_loader hit on provider=%s len=%d", prov, len(df))
                df.attrs["openbb_provider"] = prov
                return df
            last_err = f"empty df from {prov}"
        except Exception as e:
            last_err = f"{prov}: {type(e).__name__}: {str(e)[:80]}"
            LOG.debug("openbb_loader miss on provider=%s err=%s", prov, last_err)
            continue
    LOG.info("openbb_loader full chain failed (%s): %s", providers, last_err)
    return pd.DataFrame()


def load_equity_history(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    provider: str | None = None,
) -> pd.DataFrame:
    """Fetch OHLCV history.

    Args:
        ticker: equity ticker (e.g. "AAPL")
        start: ISO date 'YYYY-MM-DD'; defaults to 1 year ago
        end:   ISO date 'YYYY-MM-DD'; defaults to today
        interval: '1d' default; '1h', '5m' supported by some providers
        provider: explicit provider; None -> walk fallback chain

    Returns:
        DataFrame indexed by date with open/high/low/close/volume cols.
        Empty DF on full failure.
    """
    if not ENABLED:
        LOG.debug("openbb_loader disabled (OPENBB_LOADER_ENABLED!=1)")
        return pd.DataFrame()
    obb = _obb()
    if obb is None:
        return pd.DataFrame()
    if end is None:
        end = datetime.utcnow().strftime("%Y-%m-%d")
    if start is None:
        start = (datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%d")
    fn = obb.equity.price.historical
    providers = [provider] if provider else _FALLBACK_CHAINS["equity.price.historical"]
    return _try_chain(
        fn, providers, symbol=ticker, start_date=start, end_date=end, interval=interval,
    )


def load_equity_fundamentals(
    ticker: str,
    statement: str = "income",
    period: str = "annual",
    limit: int = 5,
    provider: str | None = None,
) -> pd.DataFrame:
    """Fetch fundamental statement.

    Args:
        ticker:    equity ticker
        statement: 'income' | 'balance' | 'cash'
        period:    'annual' | 'quarter'
        limit:     number of periods to return

    Returns:
        DataFrame of statement rows. Empty on failure.
    """
    if not ENABLED:
        return pd.DataFrame()
    obb = _obb()
    if obb is None:
        return pd.DataFrame()
    chain_key = f"equity.fundamental.{statement}"
    if chain_key not in _FALLBACK_CHAINS:
        LOG.warning("unknown statement=%s", statement)
        return pd.DataFrame()
    fn_map = {
        "income": obb.equity.fundamental.income,
        "balance": obb.equity.fundamental.balance,
        "cash": obb.equity.fundamental.cash,
    }
    fn = fn_map[statement]
    providers = [provider] if provider else _FALLBACK_CHAINS[chain_key]
    return _try_chain(fn, providers, symbol=ticker, period=period, limit=limit)


def load_equity_news(
    ticker: str,
    limit: int = 20,
    provider: str | None = None,
) -> pd.DataFrame:
    """Fetch news items for a ticker. Empty DF on failure."""
    if not ENABLED:
        return pd.DataFrame()
    obb = _obb()
    if obb is None:
        return pd.DataFrame()
    fn = obb.news.company
    providers = [provider] if provider else _FALLBACK_CHAINS["equity.news"]
    return _try_chain(fn, providers, symbol=ticker, limit=limit)


def load_economy_indicator(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    provider: str | None = None,
) -> pd.DataFrame:
    """Fetch macro indicator series (FRED-style symbol)."""
    if not ENABLED:
        return pd.DataFrame()
    obb = _obb()
    if obb is None:
        return pd.DataFrame()
    if end is None:
        end = datetime.utcnow().strftime("%Y-%m-%d")
    if start is None:
        start = (datetime.utcnow() - timedelta(days=365 * 3)).strftime("%Y-%m-%d")
    fn = obb.economy.fred_series
    providers = [provider] if provider else _FALLBACK_CHAINS["economy.indicator"]
    return _try_chain(fn, providers, symbol=symbol, start_date=start, end_date=end)


def list_available_providers(category: str = "equity") -> list[str]:
    """Return providers covering a category."""
    obb = _obb()
    if obb is None:
        return []
    try:
        coverage = obb.coverage.providers
        if isinstance(coverage, dict):
            return sorted(coverage.keys())
        return list(coverage)
    except Exception as e:
        LOG.debug("list_available_providers failed: %s", e)
        return []


def health_check() -> dict:
    """Quick sanity check. Used by smoke + monitoring."""
    obb = _obb()
    return {
        "enabled": ENABLED,
        "openbb_importable": obb is not None,
        "fallback_chains": list(_FALLBACK_CHAINS.keys()),
    }


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    print("health:", health_check())
    if ENABLED:
        print("AAPL history sample:", load_equity_history("AAPL").head())
    else:
        print("set OPENBB_LOADER_ENABLED=1 to enable")
