import sys
import numpy as np
import pandas as pd

def add_volume_profiles_features(df, ticker):
    """
    Volume Profiles: identifies significant price levels via transaction volume.
    Bins price into deciles, sums volume, finds POC (Point of Control).
    Returns df with 'vol_poc_distance' (distance to POC) and 'vol_profile_strength' features.
    """
    try:
        if df.empty or 'volume' not in df.columns or len(df) < 20:
            df['vol_poc_distance'] = 0.0
            df['vol_profile_strength'] = 0.0
            return df

        close = df['close']
        volume = df['volume']

        # Rolling 20-bar price/volume profile
        window = 20
        poc_distances = []
        profile_strengths = []

        for i in range(len(df)):
            if i < window:
                poc_distances.append(0.0)
                profile_strengths.append(0.0)
                continue

            # Bin window prices into 10 deciles, sum volume
            window_prices = close.iloc[i-window:i]
            window_vols = volume.iloc[i-window:i]

            price_min, price_max = window_prices.min(), window_prices.max()
            if price_max == price_min:
                poc_distances.append(0.0)
                profile_strengths.append(0.0)
                continue

            bins = np.linspace(price_min, price_max, 11)
            bin_idx = np.digitize(window_prices, bins) - 1
            bin_vols = np.zeros(10)
            for j, idx in enumerate(bin_idx):
                if 0 <= idx < 10:
                    bin_vols[idx] += window_vols.iloc[j]

            # POC = price level with max volume
            poc_bin = np.argmax(bin_vols)
            poc_price = (bins[poc_bin] + bins[poc_bin + 1]) / 2
            current_price = close.iloc[i]

            poc_dist = abs(current_price - poc_price) / (price_max - price_min + 1e-9)
            poc_distances.append(poc_dist)
            profile_strengths.append(bin_vols[poc_bin] / (window_vols.sum() + 1e-9))

        df['vol_poc_distance'] = pd.Series(poc_distances, index=df.index).fillna(0.0)
        df['vol_profile_strength'] = pd.Series(profile_strengths, index=df.index).fillna(0.0)

        return df
    except Exception:
        df['vol_poc_distance'] = 0.0
        df['vol_profile_strength'] = 0.0
        return df
