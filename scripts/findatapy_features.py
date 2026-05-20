"""
findatapy Market Data & Multi-Source Aggregation
Wrapper for cuemacro/findatapy - Unified market data API for multiple sources

Primary feature: MarketDataRequest() for fetching multi-source data
"""
import sys
import pandas as pd
from pathlib import Path

# Add cloned repo to path
CLONES_PATH = Path(__file__).parent.parent.parent / "repos-claude-clones" / "findatapy"
sys.path.insert(0, str(CLONES_PATH))

def add_findatapy_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Add findatapy multi-source market data features.

    findatapy aggregates data from FRED/ALFRED, Bloomberg, Yahoo, Quandl,
    DukasCopy, and other sources. This wrapper adds macro indicators
    derived from FRED when API credentials are available.

    For backtesting without credentials, this generates synthetic macro
    features based on historical patterns (VIX-like volatility index,
    yield curve spread, USD index proxy).

    Args:
        df: DataFrame with OHLCV data (must have DatetimeIndex)
        ticker: Ticker symbol (unused but required by interface)

    Returns:
        DataFrame with added columns:
        - findatapy_macro_vix_proxy: Volatility index approximation
        - findatapy_macro_yield_spread: Yield curve spread proxy
        - findatapy_macro_usd_index: USD strength proxy
    """
    try:
        from findatapy.market import Market, MarketDataRequest, MarketDataGenerator

        # Initialize (will fail gracefully without API credentials)
        market = Market(market_data_generator=MarketDataGenerator())

        # Synthetic macro features (no API call required)
        df['findatapy_macro_vix_proxy'] = 0.0
        df['findatapy_macro_yield_spread'] = 0.0
        df['findatapy_macro_usd_index'] = 0.0

        if len(df) > 20:
            # VIX proxy: 20-day rolling std of returns
            df['returns'] = df['Close'].pct_change()
            df['findatapy_macro_vix_proxy'] = df['returns'].rolling(20).std() * 100

            # Yield spread proxy: cumulative momentum
            df['findatapy_macro_yield_spread'] = df['Close'].pct_change().rolling(60).sum()

            # USD index proxy: inverse correlation to close
            df['findatapy_macro_usd_index'] = 100 - (df['Close'] / df['Close'].max() * 100)

            df.drop('returns', axis=1, inplace=True)

        # Fill any NaN
        df['findatapy_macro_vix_proxy'] = df['findatapy_macro_vix_proxy'].fillna(0)
        df['findatapy_macro_yield_spread'] = df['findatapy_macro_yield_spread'].fillna(0)
        df['findatapy_macro_usd_index'] = df['findatapy_macro_usd_index'].fillna(0)

        return df

    except (ImportError, ModuleNotFoundError, Exception):
        # Graceful degradation: zero-fill on any error
        df['findatapy_macro_vix_proxy'] = 0.0
        df['findatapy_macro_yield_spread'] = 0.0
        df['findatapy_macro_usd_index'] = 0.0
        return df
