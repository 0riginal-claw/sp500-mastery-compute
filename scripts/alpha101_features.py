import sys
import pandas as pd
import numpy as np
from pathlib import Path

repo_path = Path("/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/repos-claude-clones/alpha101")
sys.path.insert(0, str(repo_path))

def add_alpha101_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Add alpha 101 factor-inspired features (price-volume correlations, rank-based signals)."""
    try:
        df = df.copy()

        required = ['close', 'open', 'high', 'low', 'volume']
        if not all(col in df.columns for col in required):
            # Fill missing with simple approximations
            if 'open' not in df.columns:
                df['open'] = df.get('close', 0)
            if 'high' not in df.columns:
                df['high'] = df.get('close', 0)
            if 'low' not in df.columns:
                df['low'] = df.get('close', 0)
            if 'volume' not in df.columns:
                df['volume'] = 0

        # Alpha#3 simplified: correlation(rank(open), rank(volume), 10)
        open_rank = df['open'].rank()
        vol_rank = df['volume'].rank()
        df['alpha3_corr'] = open_rank.rolling(10, min_periods=1).corr(vol_rank)

        # Alpha#6: correlation(open, volume, 10)
        df['alpha6_corr'] = df['open'].rolling(10, min_periods=1).corr(df['volume'].rolling(10, min_periods=1).mean())

        # Price-volume mean reversion signal (Alpha#8-like)
        df['hl_spread'] = (df['high'] - df['low']) / (df['close'] + 1e-8)
        df['hl_spread_sma'] = df['hl_spread'].rolling(5, min_periods=1).mean()

        # Return momentum (Alpha#1-inspired)
        df['returns'] = df['close'].pct_change().fillna(0)
        df['return_std_20'] = df['returns'].rolling(20, min_periods=1).std()

        return df.fillna(0)
    except Exception as e:
        print(f"alpha101 error: {e}")
        return df
