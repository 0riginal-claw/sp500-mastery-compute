"""
Stock_Analysis_For_Quant feature wrapper — LastAncientOne/Stock_Analysis_For_Quant.
Extracts classic technical indicator ensemble (drawn from repo's 150+ indicator notebooks)
as predictive features for 1-step-ahead return. Pure pandas/numpy — no heavy deps.
Status: ready_for_consumer (no checkpoint needed).
"""
import numpy as np
import pandas as pd


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - 100 / (1 + rs)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hi, lo, cl = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [hi - lo, (hi - cl.shift()).abs(), (lo - cl.shift()).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def add_stock_analysis_for_quant_features(
    df: pd.DataFrame, lookback: int = 64
) -> pd.DataFrame:
    """
    Add technical indicator ensemble features + linear-regression predicted return.

    Features computed (all from LastAncientOne indicator collection):
        - RSI(14), RSI(7)
        - MACD line and signal
        - Bollinger Band %B (20,2)
        - ATR(14) normalized
        - EMA crossover ratio (12/26)
        - Stochastic %K (14)
        - Williams %R (14)
        - Rate of Change (10)
        - OBV momentum (10-bar)
        - Linear-regression slope predicted_ret_t1 proxy

    Args:
        df: DataFrame with [open, high, low, close, volume] columns.
        lookback: rolling window for linear-regression prediction.

    Returns:
        df with added `saq_predicted_ret_t1` column and indicator columns.
    """
    df = df.copy()
    close = df["close"]

    df["saq_rsi_14"] = _rsi(close, 14)
    df["saq_rsi_7"] = _rsi(close, 7)

    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd = ema12 - ema26
    df["saq_macd"] = macd
    df["saq_macd_signal"] = _ema(macd, 9)
    df["saq_macd_hist"] = macd - df["saq_macd_signal"]
    df["saq_ema_cross_ratio"] = ema12 / (ema26 + 1e-10) - 1.0

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["saq_bb_pct"] = (close - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-10)

    if "high" in df.columns and "low" in df.columns:
        df["saq_atr14_norm"] = _atr(df, 14) / (close + 1e-10)

        lo14 = df["low"].rolling(14).min()
        hi14 = df["high"].rolling(14).max()
        df["saq_stoch_k"] = 100 * (close - lo14) / (hi14 - lo14 + 1e-10)
        df["saq_williams_r"] = -100 * (hi14 - close) / (hi14 - lo14 + 1e-10)

    df["saq_roc10"] = close.pct_change(10)

    if "volume" in df.columns:
        obv = (np.sign(close.diff()) * df["volume"]).fillna(0).cumsum()
        df["saq_obv_mom10"] = obv.pct_change(10)

    # Linear-regression slope over lookback window → proxy for next-bar return
    log_ret = np.log(close / close.shift(1))

    def _lr_slope(x):
        n = len(x)
        if n < 2 or np.isnan(x).any():
            return np.nan
        t = np.arange(n)
        slope = np.polyfit(t, x, 1)[0]
        return slope

    df["saq_predicted_ret_t1"] = (
        log_ret.rolling(lookback).apply(_lr_slope, raw=True)
    )

    return df
