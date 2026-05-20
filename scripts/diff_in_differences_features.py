import sys
import numpy as np
import pandas as pd

def add_diff_in_differences_features(df, ticker):
    """
    Difference-in-Differences: causal inference via treatment/control comparison.
    Implements simplified DiD: splits on high-vol vs low-vol regime, compares pre/post price.
    Returns df with 'did_treatment', 'did_outcome', 'did_att_proxy' features.
    NOTE: Simplified; real DiD requires panel data + control group.
    """
    try:
        if df.empty or len(df) < 40:
            df['did_treatment'] = 0
            df['did_outcome'] = 0.0
            df['did_att_proxy'] = 0.0
            return df

        close = df['close']
        volatility = close.pct_change().rolling(20).std()

        # Treatment: above-median volatility (True=1, False=0)
        vol_median = volatility.median()
        treated = (volatility > vol_median).astype(int)

        # Split at midpoint: pre (0) vs post (1)
        midpoint = len(df) // 2
        period = np.zeros(len(df))
        period[midpoint:] = 1

        # Outcome: 10-day forward return
        forward_ret = close.shift(-10).pct_change()
        forward_ret = forward_ret.fillna(0.0)

        df['did_treatment'] = treated
        df['did_outcome'] = forward_ret

        # Simple ATT proxy: mean diff in treated post vs treated pre
        treated_post = forward_ret[(treated == 1) & (period == 1)]
        treated_pre = forward_ret[(treated == 1) & (period == 0)]

        if len(treated_post) > 0 and len(treated_pre) > 0:
            att = treated_post.mean() - treated_pre.mean()
        else:
            att = 0.0

        df['did_att_proxy'] = att

        return df
    except Exception:
        df['did_treatment'] = 0
        df['did_outcome'] = 0.0
        df['did_att_proxy'] = 0.0
        return df
