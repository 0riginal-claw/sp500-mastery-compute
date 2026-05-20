import sys
import numpy as np
import pandas as pd

def add_ma_energy_indicator_features(df, ticker):
    """
    MA Energy Indicator: multi-timeframe momentum normalized by volatility.
    Computes 10-day, 20-day, 50-day MA energy + volatility normalization.
    Returns df with 'ma_energy' feature (-1 to +1 scale).
    """
    try:
        if df.empty or len(df) < 50:
            df['ma_energy'] = 0.0
            return df

        close = df['close']

        # Compute MAs
        ma10 = close.rolling(10).mean()
        ma20 = close.rolling(20).mean()
        ma50 = close.rolling(50).mean()

        # MA differences as energy proxy
        energy_short = (ma10 - ma20) / (ma20 + 1e-9)
        energy_long = (ma20 - ma50) / (ma50 + 1e-9)

        # Combine: short + long momentum
        raw_energy = (energy_short * 0.6) + (energy_long * 0.4)

        # Normalize by rolling volatility
        volatility = close.pct_change().rolling(20).std()
        volatility = volatility.fillna(volatility.mean() or 0.01)

        ma_energy = raw_energy / (volatility + 0.01)
        df['ma_energy'] = ma_energy.fillna(0.0).clip(-1, 1)

        return df
    except Exception:
        df['ma_energy'] = 0.0
        return df
