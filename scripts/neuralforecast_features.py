import sys
import pandas as pd
import numpy as np
from pathlib import Path

repo_path = Path("/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/repos-claude-clones/neuralforecast")
sys.path.insert(0, str(repo_path))

def add_neuralforecast_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Add neural forecast preparation features (seasonality, decomposition hints)."""
    try:
        df = df.copy()

        if 'close' not in df.columns:
            return df

        # Day of week, month cyclical encoding for seasonal patterns
        if 'date' in df.columns or 'timestamp' in df.columns:
            date_col = df.get('date') or df.get('timestamp')
            df['day_of_week'] = pd.to_datetime(date_col).dt.dayofweek
            df['month'] = pd.to_datetime(date_col).dt.month
        else:
            df['day_of_week'] = np.arange(len(df)) % 5
            df['month'] = (np.arange(len(df)) // 20) % 12

        # Trend component via linear detrending residuals
        if len(df) > 20:
            x = np.arange(len(df))
            coeffs = np.polyfit(x, df['close'].fillna(df['close'].mean()), 1)
            trend = np.polyval(coeffs, x)
            df['detrended_close'] = df['close'] - trend

        # Price momentum (return) for neural models
        df['returns'] = df['close'].pct_change().fillna(0)

        return df.fillna(0)
    except Exception as e:
        print(f"neuralforecast error: {e}")
        return df
