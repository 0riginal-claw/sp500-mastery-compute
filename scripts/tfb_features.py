"""
TFB feature wrapper — decisionintelligence/TFB (PVLDB 2024).
Time series benchmark with Characteristics Extractor (trend, seasonality,
stationarity, shifting, transition, correlation).
Uses TFB's CharacteristicsExtractor as signal — no checkpoint needed.
Status: ready_for_consumer (characteristics extractor is pure analysis code).
"""
import os
import sys
import numpy as np
import pandas as pd

CLONE_DIR = os.path.join(
    os.path.expanduser("~"),
    "Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive",
    "AI-Tools/repos-claude-clones/TFB",
)


def _fallback_characteristics(series: np.ndarray) -> dict:
    """Pure-numpy fallback when TFB not importable."""
    n = len(series)
    t = np.arange(n)
    slope = np.polyfit(t, series, 1)[0] if n > 1 else 0.0
    demeaned = series - series.mean()
    acf1 = (
        float(np.corrcoef(demeaned[:-1], demeaned[1:])[0, 1])
        if n > 2
        else 0.0
    )
    variance = float(np.var(series)) if n > 1 else 0.0
    return {"trend": float(slope), "acf1": acf1, "variance": variance}


def add_tfb_features(df: pd.DataFrame, lookback: int = 64) -> pd.DataFrame:
    """
    Add TFB time-series characteristics as features + a linear prediction proxy.

    Features (rolling `lookback` window over close):
        - tfb_trend: linear trend slope (normalized)
        - tfb_acf1: lag-1 autocorrelation (seasonality proxy)
        - tfb_variance: rolling variance
        - tfb_stationarity: ADF-like ratio (std ratio short/long)
        - tfb_predicted_ret_t1: slope-extrapolated 1-step return

    Args:
        df: DataFrame with [close] column (optionally open/high/low/volume).
        lookback: rolling window length.

    Returns:
        df with added tfb_* columns.
    """
    df = df.copy()
    close = df["close"].values.astype(np.float64)
    n = len(close)

    trend_arr = np.full(n, np.nan)
    acf1_arr = np.full(n, np.nan)
    var_arr = np.full(n, np.nan)
    stat_arr = np.full(n, np.nan)
    pred_ret_arr = np.full(n, np.nan)

    try:
        if CLONE_DIR not in sys.path:
            sys.path.insert(0, CLONE_DIR)
        from characteristics_extractor.Characteristics_Extractor import (  # type: ignore
            CharacteristicsExtractor,
        )
        extractor = CharacteristicsExtractor()
        use_extractor = True
    except ImportError:
        use_extractor = False

    for i in range(lookback, n):
        window = close[i - lookback : i]
        w_norm = (window - window.mean()) / (window.std() + 1e-10)
        t = np.arange(lookback)

        if use_extractor:
            try:
                chars = extractor.extract(window)
                trend_arr[i] = chars.get("trend", 0.0)
                acf1_arr[i] = chars.get("seasonality", 0.0)
                var_arr[i] = chars.get("variance", float(np.var(window)))
                stat_arr[i] = chars.get("stationarity", 0.0)
            except Exception:
                ch = _fallback_characteristics(w_norm)
                trend_arr[i] = ch["trend"]
                acf1_arr[i] = ch["acf1"]
                var_arr[i] = ch["variance"]
                stat_arr[i] = 0.0
        else:
            ch = _fallback_characteristics(w_norm)
            trend_arr[i] = ch["trend"]
            acf1_arr[i] = ch["acf1"]
            var_arr[i] = ch["variance"]
            # stationarity: ratio of short-term vs long-term std
            half = lookback // 2
            stat_arr[i] = float(window[half:].std() / (window[:half].std() + 1e-10))

        # Extrapolated 1-step price → return
        slope, intercept = np.polyfit(t, window, 1)
        pred_price = slope * lookback + intercept
        pred_ret_arr[i] = pred_price / (close[i - 1] + 1e-10) - 1.0

    df["tfb_trend"] = trend_arr
    df["tfb_acf1"] = acf1_arr
    df["tfb_variance"] = var_arr
    df["tfb_stationarity"] = stat_arr
    df["tfb_predicted_ret_t1"] = pred_ret_arr

    return df
