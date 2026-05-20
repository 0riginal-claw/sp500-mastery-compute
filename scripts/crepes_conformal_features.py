import sys
import numpy as np
import pandas as pd

def add_crepes_conformal_features(df, ticker):
    """
    Crepes Conformal Prediction: wraps price predictor to generate prediction intervals.
    Simulates non-conformity score based on recent prediction error variance.
    Returns df with 'conf_lower_bound', 'conf_upper_bound', 'conf_width' features.
    NOTE: No actual crepes library call; placeholder for integration.
    """
    try:
        if df.empty or len(df) < 20:
            df['conf_lower_bound'] = 0.0
            df['conf_upper_bound'] = 0.0
            df['conf_width'] = 0.0
            return df

        close = df['close']

        # Proxy: rolling MA as predictor, std as interval width
        ma5 = close.rolling(5).mean()
        ma5_error = abs(close - ma5).rolling(10).std()
        ma5_error = ma5_error.fillna(ma5_error.mean() or 0.01)

        # Confidence level = 95%
        z = 1.96
        df['conf_lower_bound'] = (ma5 - z * ma5_error).fillna(0.0)
        df['conf_upper_bound'] = (ma5 + z * ma5_error).fillna(0.0)
        df['conf_width'] = (2 * z * ma5_error).fillna(0.0)

        return df
    except Exception:
        df['conf_lower_bound'] = 0.0
        df['conf_upper_bound'] = 0.0
        df['conf_width'] = 0.0
        return df
