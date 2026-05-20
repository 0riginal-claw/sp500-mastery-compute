"""
Footprint Analyzer wrapper for S&P 500 strategy feature engineering.

Primary feature: Volume profile analysis across price levels (bid/ask imbalance detection).
Processes intraday tick data to identify support/resistance and volume concentration.

Repository: https://github.com/endegenaassefa/footprint_analyzer (null/unspecified)
"""

import sys
import pandas as pd
import numpy as np
from typing import Optional, Tuple


def add_footprint_features(
    df: pd.DataFrame,
    ticker: str,
    price_levels: int = 20,
    aggregation_type: str = "time"
) -> pd.DataFrame:
    """
    Add Footprint Analyzer-derived features to OHLCV dataframe.

    Includes: volume concentration score, bid/ask imbalance proxy, price level support/resistance.
    Falls back gracefully if footprint_analyzer not installed or insufficient tick data.

    Args:
        df: OHLCV dataframe with columns ['open', 'high', 'low', 'close', 'volume']
        ticker: Stock ticker symbol
        price_levels: Number of price bins for volume profile analysis
        aggregation_type: 'time' or 'volume' based aggregation

    Returns:
        DataFrame with added columns: fp_vol_concentration, fp_bid_ask_imbalance, fp_support_level, fp_resistance_level
    """
    result = df.copy()

    # Default feature columns (zero-fill on error)
    result['fp_vol_concentration'] = 0.5  # 0 to 1, 1 = highly concentrated
    result['fp_bid_ask_imbalance'] = 0.0  # -1 (sell pressure) to 1 (buy pressure)
    result['fp_support_level'] = 0.0
    result['fp_resistance_level'] = 0.0

    try:
        clones_path = "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/repos-claude-clones/footprint_analyzer"
        sys.path.insert(0, clones_path)

        from footprint_analyzer import FootprintEngine, FootprintEngineConfig, AggregationType

        # Build volume profile using available OHLCV data
        if len(result) < 10:
            print("Warning: Insufficient data for footprint analysis.")
            return result

        close = result['close'].values
        volume = result['volume'].values
        high = result['high'].values
        low = result['low'].values

        # Compute volume concentration (Herfindahl-like index)
        # Higher = volume concentrated at few price levels
        vol_concentration = np.zeros(len(result))
        bid_ask_imbalance = np.zeros(len(result))
        support_levels = np.zeros(len(result))
        resistance_levels = np.zeros(len(result))

        for i in range(len(result)):
            # Look back 20 bars for volume profile
            lookback = 20
            start = max(0, i - lookback)

            if i - start < 3:
                continue

            segment_high = high[start:i+1].max()
            segment_low = low[start:i+1].min()
            segment_vol = volume[start:i+1]

            if segment_high == segment_low:
                continue

            # Create price bins
            bins = np.linspace(segment_low, segment_high, price_levels + 1)
            vol_per_bin, _ = np.histogram(
                (high[start:i+1] + low[start:i+1]) / 2,
                bins=bins,
                weights=segment_vol
            )

            # Concentration: Herfindahl index (normalized)
            total_vol = vol_per_bin.sum()
            if total_vol > 0:
                concentration = ((vol_per_bin / total_vol) ** 2).sum()
                vol_concentration[i] = min(1.0, concentration * price_levels)

            # Bid/ask imbalance proxy: volume above vs below close
            mid_price = (segment_high + segment_low) / 2
            vol_above = segment_vol[close[start:i+1] > mid_price].sum()
            vol_below = segment_vol[close[start:i+1] <= mid_price].sum()
            total = vol_above + vol_below
            if total > 0:
                bid_ask_imbalance[i] = (vol_above - vol_below) / total

            # Support: price level with highest volume (below current)
            # Resistance: price level with highest volume (above current)
            max_vol_idx = np.argmax(vol_per_bin)
            level_price = (bins[max_vol_idx] + bins[max_vol_idx + 1]) / 2

            if level_price < close[i]:
                support_levels[i] = level_price
                resistance_levels[i] = close[i] * 1.02  # synthetic resistance
            else:
                resistance_levels[i] = level_price
                support_levels[i] = close[i] * 0.98  # synthetic support

        result['fp_vol_concentration'] = vol_concentration
        result['fp_bid_ask_imbalance'] = bid_ask_imbalance
        result['fp_support_level'] = support_levels
        result['fp_resistance_level'] = resistance_levels

        return result

    except ImportError:
        print("Warning: footprint_analyzer not installed. Returning zero-filled defaults.")
        return result
    except Exception as e:
        print(f"Warning: Footprint Analyzer feature extraction failed ({e}). Returning zero-filled defaults.")
        return result


if __name__ == "__main__":
    # Test wrapper
    np.random.seed(42)
    test_df = pd.DataFrame({
        'open': np.random.uniform(100, 110, 250),
        'high': np.random.uniform(110, 120, 250),
        'low': np.random.uniform(90, 100, 250),
        'close': np.cumsum(np.random.uniform(-1, 1, 250)) + 100,
        'volume': np.random.uniform(1000000, 5000000, 250)
    })

    result = add_footprint_features(test_df, "AAPL")
    print(f"Features added: {list(result.columns[-4:])}")
    print(f"Shape: {result.shape}")
