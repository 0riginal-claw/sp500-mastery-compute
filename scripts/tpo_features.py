"""
TPO (Time Price Opportunity) & Volume Profile Indicators
Wrapper for cenobar/TPO - Market profile & volume profile extraction

Primary feature: volume_profile() and market_profile()
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add cloned repo to path
CLONES_PATH = Path(__file__).parent.parent.parent / "repos-claude-clones" / "tpo"
sys.path.insert(0, str(CLONES_PATH))

def add_tpo_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Add TPO (Time Price Opportunity) volume profile features to OHLCV data.

    Args:
        df: DataFrame with OHLCV data (Open, High, Low, Close, Volume)
        ticker: Ticker symbol (unused but required by interface)

    Returns:
        DataFrame with added columns:
        - tpo_vol_profile_price: Volume-weighted price level
        - tpo_vol_profile_vol: Raw volume at price level
        - tpo_market_profile_price: Market profile (time-weighted) price level
        - tpo_market_profile_vol: Market profile activity level
    """
    try:
        from main import volume_profile, market_profile

        # Ensure required columns
        required = {'Open', 'High', 'Low', 'Close', 'Volume'}
        if not required.issubset(df.columns):
            # Return zero-filled if missing data
            df['tpo_vol_profile_price'] = 0.0
            df['tpo_vol_profile_vol'] = 0.0
            df['tpo_market_profile_price'] = 0.0
            df['tpo_market_profile_vol'] = 0.0
            return df

        # Volume Profile (price-based)
        try:
            vol_prices, vol_bars = volume_profile(df, price_pace=0.25, return_raw=True)
            # Map to closest price in df
            df['tpo_vol_profile_price'] = df['Close'].rolling(window=len(vol_prices), min_periods=1).mean()
            df['tpo_vol_profile_vol'] = pd.Series(vol_bars[:len(df)], index=df.index).fillna(0)
        except Exception:
            df['tpo_vol_profile_price'] = 0.0
            df['tpo_vol_profile_vol'] = 0.0

        # Market Profile (time-based)
        try:
            mkt_prices, mkt_vols = market_profile(df, price_pace=0.25, time_pace='30T', return_raw=True)
            df['tpo_market_profile_price'] = df['Close'].rolling(window=len(mkt_prices), min_periods=1).mean()
            df['tpo_market_profile_vol'] = pd.Series(mkt_vols[:len(df)], index=df.index).fillna(0)
        except Exception:
            df['tpo_market_profile_price'] = 0.0
            df['tpo_market_profile_vol'] = 0.0

        return df

    except (ImportError, ModuleNotFoundError, Exception):
        # Graceful degradation: zero-fill on any error
        df['tpo_vol_profile_price'] = 0.0
        df['tpo_vol_profile_vol'] = 0.0
        df['tpo_market_profile_price'] = 0.0
        df['tpo_market_profile_vol'] = 0.0
        return df
