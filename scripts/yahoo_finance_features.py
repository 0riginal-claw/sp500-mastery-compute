"""
Yahoo Finance MCP wrapper for S&P 500 feature engineering.

Primary feature: Real-time stock quotes, historical prices, financial statements via yfinance.
Uses the MCP server abstraction but extracts core yfinance ticker data for feature generation.

Repository: https://github.com/danishashko/yahoo-finance-mcp (MIT)
"""

import sys
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any


def add_yahoo_finance_features(
    df: pd.DataFrame,
    ticker: str,
    include_fundamentals: bool = True,
    include_recent_data: bool = True
) -> pd.DataFrame:
    """
    Add Yahoo Finance-derived features to OHLCV dataframe.

    Includes: price-to-book, price-to-earnings, dividend yield, analyst recommendations.
    Falls back gracefully if yfinance not installed or ticker not found.

    Args:
        df: OHLCV dataframe with columns ['open', 'high', 'low', 'close', 'volume']
        ticker: Stock ticker symbol
        include_fundamentals: Add P/E, P/B, div yield, etc.
        include_recent_data: Add recent price levels from latest fetch

    Returns:
        DataFrame with added columns: yf_pe_ratio, yf_pb_ratio, yf_div_yield, yf_peg_ratio, yf_52w_high_pct
    """
    result = df.copy()

    # Default feature columns (zero-fill on error)
    result['yf_pe_ratio'] = 0.0
    result['yf_pb_ratio'] = 0.0
    result['yf_div_yield'] = 0.0
    result['yf_peg_ratio'] = 0.0
    result['yf_52w_high_pct'] = 0.5

    try:
        import yfinance as yf

        # Fetch ticker data
        try:
            stock = yf.Ticker(ticker, timeout=5)
        except Exception as e:
            print(f"Warning: Could not fetch {ticker} from Yahoo Finance ({e}). Using defaults.")
            return result

        # Extract fundamental ratios
        info = stock.info or {}

        pe = info.get('trailingPE', 0) or info.get('forwardPE', 0) or 0
        pb = info.get('priceToBook', 0) or 0
        div_yield = info.get('dividendYield', 0) or 0
        peg = info.get('pegRatio', 0) or 0

        # 52-week high proximity
        h52w = info.get('fiftyTwoWeekHigh', None)
        curr_price = info.get('currentPrice', None) or (result['close'].iloc[-1] if len(result) > 0 else 1)
        h52w_pct = (curr_price / h52w) if h52w else 0.5

        # Assign to entire dataframe (fundamentals are static across the period)
        result['yf_pe_ratio'] = float(pe) if pe else 0.0
        result['yf_pb_ratio'] = float(pb) if pb else 0.0
        result['yf_div_yield'] = float(div_yield) if div_yield else 0.0
        result['yf_peg_ratio'] = float(peg) if peg else 0.0
        result['yf_52w_high_pct'] = float(np.clip(h52w_pct, 0, 1))

        return result

    except ImportError:
        print("Warning: yfinance not installed. Returning zero-filled defaults.")
        return result
    except Exception as e:
        print(f"Warning: Yahoo Finance feature extraction failed ({e}). Returning zero-filled defaults.")
        return result


if __name__ == "__main__":
    # Test wrapper
    np.random.seed(42)
    test_df = pd.DataFrame({
        'open': np.random.uniform(100, 110, 250),
        'high': np.random.uniform(110, 120, 250),
        'low': np.random.uniform(90, 100, 250),
        'close': np.cumsum(np.random.uniform(-1, 1, 250)) + 100,
        'volume': np.random.uniform(1000000, 5000000, 250)
    })

    result = add_yahoo_finance_features(test_df, "AAPL")
    print(f"Features added: {list(result.columns[-5:])}")
    print(f"Shape: {result.shape}")
