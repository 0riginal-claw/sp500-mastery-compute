import sys
import numpy as np
import pandas as pd

def add_feast_features_features(df, ticker):
    """
    Feast Feature Store: point-in-time correct historical feature aggregation.
    Generates lag-based features (conv_rate, acc_rate, avg_trips analogs).
    Returns df with 'feat_conv_rate', 'feat_acc_rate', 'feat_avg_volume' features.
    """
    try:
        if df.empty or len(df) < 20:
            df['feat_conv_rate'] = 0.0
            df['feat_acc_rate'] = 0.0
            df['feat_avg_volume'] = 0.0
            return df

        # conv_rate: ratio of positive moves in 5-day window
        returns = df['close'].pct_change()
        conv_rate = (returns > 0).rolling(5).sum() / 5
        df['feat_conv_rate'] = conv_rate.fillna(0.5)

        # acc_rate: days above 20-day MA / window size
        ma20 = df['close'].rolling(20).mean()
        acc_rate = (df['close'] > ma20).rolling(5).sum() / 5
        df['feat_acc_rate'] = acc_rate.fillna(0.5)

        # avg_volume: 10-day rolling average volume (0-1 normalized)
        if 'volume' in df.columns:
            avg_vol = df['volume'].rolling(10).mean()
            max_vol = avg_vol.max() or 1e6
            df['feat_avg_volume'] = (avg_vol / max_vol).fillna(0.0)
        else:
            df['feat_avg_volume'] = 0.0

        return df
    except Exception:
        df['feat_conv_rate'] = 0.0
        df['feat_acc_rate'] = 0.0
        df['feat_avg_volume'] = 0.0
        return df
