import sys
import pandas as pd
import numpy as np
from pathlib import Path

repo_path = Path("/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/repos-claude-clones/mlforecast")
sys.path.insert(0, str(repo_path))

def add_mlforecast_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Add ML forecast lagged features for time series."""
    try:
        df = df.copy()

        if 'close' not in df.columns:
            return df

        # Create lagged returns feature set common in ML forecasting
        for lag in [1, 2, 5, 10, 20]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            if 'volume' in df.columns:
                df[f'volume_lag_{lag}'] = df['volume'].shift(lag)

        # Rolling mean features
        for window in [5, 10, 20]:
            df[f'close_ma_{window}'] = df['close'].rolling(window=window, min_periods=1).mean()
            df[f'close_std_{window}'] = df['close'].rolling(window=window, min_periods=1).std()

        return df.fillna(0)
    except Exception as e:
        print(f"mlforecast error: {e}")
        return df
