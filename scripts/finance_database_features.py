"""finance_database_features.py — wrapper for github:JerBouma/FinanceDatabase (license: MIT)
Imports the cloned repo, exposes a single feature function for v10 pipeline.
Lookahead-safe: metadata is ticker-level and does not change intra-day.
"""
import sys, os
import pandas as pd
import numpy as np

_REPO_PATH = "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/repos-claude-clones/FinanceDatabase"
if _REPO_PATH not in sys.path:
    sys.path.insert(0, _REPO_PATH)

def add_finance_database_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Add metadata features from JerBouma/FinanceDatabase to df.
    Returns df with new cols: fdb_sector, fdb_industry, fdb_market_cap, fdb_exchange.
    Ticker-level metadata fetched once per unique ticker and broadcast to all rows.
    """
    try:
        import financedatabase as fd

        # Load equities database (cached after first load)
        equities = fd.Equities()

        # Look up the ticker in the database
        # FinanceDatabase uses 'symbol' as the index
        try:
            ticker_data = equities.select(index=ticker)

            if ticker_data.empty:
                # Ticker not found, zero-fill
                df['fdb_sector'] = 'unknown'
                df['fdb_industry'] = 'unknown'
                df['fdb_market_cap'] = 'unknown'
                df['fdb_exchange'] = 'unknown'
                return df

            # Take the first match (in case of duplicates across exchanges)
            row = ticker_data.iloc[0]

            # Extract metadata
            sector = str(row.get('sector', 'unknown')).lower()
            industry = str(row.get('industry', 'unknown')).lower()
            market_cap = str(row.get('market_cap', 'unknown')).lower()
            exchange = str(row.get('exchange', 'unknown')).lower()

            # Broadcast to all rows in df
            df['fdb_sector'] = sector
            df['fdb_industry'] = industry
            df['fdb_market_cap'] = market_cap
            df['fdb_exchange'] = exchange

            return df
        except Exception as e:
            # Fallback if lookup fails
            import logging
            logging.getLogger(__name__).warning(f"FinanceDatabase lookup failed for {ticker}: {e}")
            df['fdb_sector'] = 'unknown'
            df['fdb_industry'] = 'unknown'
            df['fdb_market_cap'] = 'unknown'
            df['fdb_exchange'] = 'unknown'
            return df
    except ImportError as e:
        import logging
        logging.getLogger(__name__).warning(f"FinanceDatabase not importable: {e}")
        df['fdb_sector'] = 'unknown'
        df['fdb_industry'] = 'unknown'
        df['fdb_market_cap'] = 'unknown'
        df['fdb_exchange'] = 'unknown'
        return df
