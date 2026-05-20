import sys
import numpy as np
import pandas as pd

def add_macd_oscillator_features(df, ticker):
    """
    MACD Oscillator: momentum via short/long MA crossover + signal line.
    Computes MACD (12-26 EMA), signal (9-day EMA of MACD), histogram.
    Returns df with 'macd', 'macd_signal', 'macd_histogram' features.
    """
    try:
        if df.empty or len(df) < 26:
            df['macd'] = 0.0
            df['macd_signal'] = 0.0
            df['macd_histogram'] = 0.0
            return df

        close = df['close']

        # Exponential moving averages
        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()

        # MACD line
        macd_line = exp12 - exp26

        # Signal line (9-day EMA of MACD)
        signal_line = macd_line.ewm(span=9, adjust=False).mean()

        # Histogram
        histogram = macd_line - signal_line

        df['macd'] = macd_line.fillna(0.0)
        df['macd_signal'] = signal_line.fillna(0.0)
        df['macd_histogram'] = histogram.fillna(0.0)

        return df
    except Exception:
        df['macd'] = 0.0
        df['macd_signal'] = 0.0
        df['macd_histogram'] = 0.0
        return df
