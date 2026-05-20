import sys
import pandas as pd
import numpy as np
from pathlib import Path

repo_path = Path("/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/repos-claude-clones/statsforecast")
sys.path.insert(0, str(repo_path))

def add_statsforecast_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Add statistical forecasting features (ARIMA-style differencing, autocorrelation hints)."""
    try:
        df = df.copy()

        if 'close' not in df.columns:
            return df

        # First difference (ARIMA I(1) differencing)
        df['close_diff'] = df['close'].diff().fillna(0)

        # Seasonal differences (20-day = ~1 month)
        df['close_diff_20'] = df['close'].diff(20).fillna(0)

        # ACF-like features: correlation with lagged values
        for lag in [1, 5, 10, 20]:
            shifted = df['close'].shift(lag)
            corr = df['close'].corr(shifted)
            df[f'close_autocorr_{lag}'] = corr

        # Variance ratio (GARCH-style volatility)
        rolling_std = df['close'].rolling(window=20, min_periods=1).std()
        df['volatility_20'] = rolling_std

        return df.fillna(0)
    except Exception as e:
        print(f"statsforecast error: {e}")
        return df
