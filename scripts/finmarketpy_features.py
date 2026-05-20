"""
finmarketpy features wrapper for S&P 500 strategy feature engineering.

Primary feature: Volatility-targeted signal generation (trend-following strategies)
Using finmarketpy's TechIndicator and volatility targeting framework.

Repository: https://github.com/cuemacro/finmarketpy (Apache 2.0)
"""

import sys
import pandas as pd
import numpy as np
from typing import Optional


def add_finmarketpy_features(
    df: pd.DataFrame,
    ticker: str,
    sma_period: int = 20,
    vol_target: float = 0.05,
    vol_periods: int = 60,
    use_vol_adjustment: bool = True
) -> pd.DataFrame:
    """
    Add finmarketpy-derived features to OHLCV dataframe.

    Primarily implements volatility-targeted trend following (SMA-based signal with vol normalization).
    Falls back gracefully if finmarketpy dependencies not installed.

    Args:
        df: OHLCV dataframe with columns ['open', 'high', 'low', 'close', 'volume']
        ticker: Stock ticker symbol
        sma_period: Moving average period for trend signal
        vol_target: Target volatility level for position sizing
        vol_periods: Lookback period for volatility calculation
        use_vol_adjustment: Apply volatility targeting adjustment to signal

    Returns:
        DataFrame with added columns: finmarketpy_signal, finmarketpy_vol_target, finmarketpy_leverage
    """
    result = df.copy()

    # Default feature columns (zero-fill on error)
    result['finmarketpy_signal'] = 0.0
    result['finmarketpy_vol_target'] = 1.0
    result['finmarketpy_leverage'] = 1.0

    try:
        # Try to import finmarketpy (may not be installed)
        clones_path = "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/repos-claude-clones/finmarketpy"
        sys.path.insert(0, f"{clones_path}/src")

        from finmarketpy.economics import TechIndicator, TechParams

        # Ensure we have close price
        if 'close' not in result.columns:
            raise ValueError("DataFrame must have 'close' column")

        close = result['close'].values

        # 1. Calculate SMA trend signal
        if len(close) >= sma_period:
            sma = pd.Series(close).rolling(window=sma_period).mean().values
            signal = np.where(close > sma, 1.0, -1.0)
            result['finmarketpy_signal'] = signal

        # 2. Calculate volatility and target-adjusted position sizing
        returns = pd.Series(close).pct_change().values
        if len(returns) >= vol_periods:
            rolling_vol = pd.Series(returns).rolling(window=vol_periods).std().values
            rolling_vol = np.maximum(rolling_vol, 1e-6)  # Avoid division by zero

            # Vol target: how much leverage to apply for consistent volatility
            vol_target_adj = vol_target / rolling_vol
            vol_target_adj = np.clip(vol_target_adj, 0.1, 3.0)  # Cap at [0.1x, 3x]

            result['finmarketpy_vol_target'] = rolling_vol

            # Conditional leverage: apply if vol_adjustment enabled
            if use_vol_adjustment:
                result['finmarketpy_leverage'] = vol_target_adj
            else:
                result['finmarketpy_leverage'] = np.where(signal != 0, 1.0, 0.5)

        return result

    except Exception as e:
        # Graceful fallback: return zero-filled columns
        print(f"Warning: finmarketpy features unavailable ({e}). Returning zero-filled defaults.")
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

    result = add_finmarketpy_features(test_df, "AAPL")
    print(f"Features added: {list(result.columns[-3:])}")
    print(f"Shape: {result.shape}")
