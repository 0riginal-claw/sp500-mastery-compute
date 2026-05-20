"""alpha_features_core_features.py — wrapper for github:GiovanniPioDelvecchio/alpha_features_core (license: MIT)
Imports the cloned repo, exposes a single feature function for v10 pipeline.
Lookahead-safe: all returned columns are .shift(1) at consumer time.
"""
import sys, os
import pandas as pd
import numpy as np

_REPO_PATH = "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/repos-claude-clones/alpha_features_core"
if _REPO_PATH not in sys.path:
    sys.path.insert(0, _REPO_PATH)

def add_alpha_features_core_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Add alpha features from GiovanniPioDelvecchio/alpha_features_core to df.
    Returns df with new cols prefixed 'afc_' (alpha feature core subset, first 30 alphas).
    Expects df columns: ticker, date, open, high, low, close, volume, amount, past_return.
    """
    try:
        from alpha_features_core.alpha191 import Alphas191

        # Filter to single ticker, ensure sorted by date
        ticker_df = df[df['ticker'] == ticker].copy().sort_values('date').reset_index(drop=True)

        if len(ticker_df) < 2:
            # Insufficient data for alpha calculation
            df['afc_alpha_count'] = 0
            return df

        # Compute subset of alphas (first 30 to avoid massive feature explosion)
        try:
            result = Alphas191(ticker_df).calculate_all_alphas(return_long=False, nums=list(range(1, 31)))

            # result is wide format: dates × [alpha001, alpha002, ..., alpha030]
            # Rename to afc_alpha001, etc.
            result_renamed = result.rename(columns={f'alpha{i:03d}': f'afc_alpha{i:03d}' for i in range(1, 31)})

            # Merge back to original df on date
            df_merged = df.merge(result_renamed, left_on='date', right_index=True, how='left')

            # Fill NaN with 0 for missing alphas
            afc_cols = [col for col in df_merged.columns if col.startswith('afc_')]
            df_merged[afc_cols] = df_merged[afc_cols].fillna(0.0)

            return df_merged
        except Exception as e:
            # Fallback if calculation fails
            import logging
            logging.getLogger(__name__).warning(f"alpha_features_core calculation failed for {ticker}: {e}")
            df['afc_zero'] = 0.0
            return df
    except ImportError as e:
        import logging
        logging.getLogger(__name__).warning(f"alpha_features_core not importable: {e}")
        df['afc_zero'] = 0.0
        return df
