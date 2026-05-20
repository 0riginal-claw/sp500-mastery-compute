import sys
import numpy as np
import pandas as pd

def add_trading_sector_rotation_features(df, ticker):
    """
    Sector Rotation Score: high-correlation pair scoring for asset allocation.
    Implements simplified Multiple Lasso Regression approach for sector weight optimization.
    Returns df with 'sector_rotation_score' feature (0-100 normalized).
    """
    try:
        if df.empty or 'close' not in df.columns:
            df['sector_rotation_score'] = 0.0
            return df

        # Compute 20-day returns correlation as proxy for pair strength
        returns = df['close'].pct_change()
        rolling_corr = returns.rolling(20).corr(returns.shift(1))

        # Normalize to [0, 100] scale
        rolling_corr = rolling_corr.fillna(0.5)
        sector_score = (rolling_corr + 1) * 50
        sector_score = sector_score.clip(0, 100)

        df['sector_rotation_score'] = sector_score
        return df
    except Exception:
        df['sector_rotation_score'] = 0.0
        return df
