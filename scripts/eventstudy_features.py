"""eventstudy_features.py — event-study abnormal-return features via zrxbeijing/EventStudy (STUB).

TODO: wire into v10 / Mythos pipeline.
Source repo: https://github.com/zrxbeijing/EventStudy (MIT).
Clone path: AI-Tools/repos-claude-clones/EventStudy

Look-ahead safety: market-model parameters are estimated on PAST window
[t-estimation_period-window_distance, t-window_distance) — strictly past.
Abnormal return at event date t computed using past-fit alpha/beta. All
outputs .shift(1) on top.

Estimated features added per ticker: ~6 columns
(es_car_5d, es_car_10d, es_ar_t0, es_bmp_test_stat, es_abn_vol_z,
days_since_last_significant_event).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ols_alpha_beta(stock_ret: np.ndarray, mkt_ret: np.ndarray) -> tuple[float, float]:
    """OLS fit r_stock = alpha + beta * r_mkt; returns (alpha, beta)."""
    mask = ~(np.isnan(stock_ret) | np.isnan(mkt_ret))
    if mask.sum() < 30:
        return np.nan, np.nan
    x = mkt_ret[mask]
    y = stock_ret[mask]
    x_mean = x.mean()
    y_mean = y.mean()
    cov = float(((x - x_mean) * (y - y_mean)).sum())
    var_x = float(((x - x_mean) ** 2).sum())
    if var_x == 0:
        return np.nan, np.nan
    beta = cov / var_x
    alpha = y_mean - beta * x_mean
    return alpha, beta


def add_eventstudy_features(
    df: pd.DataFrame,
    ticker: str,
    market_close_col: str = "spy_close",
    event_flag_cols: tuple = ("earnings_event", "feat_gap_up", "fomc_day", "form4_cluster"),
    estimation_period: int = 200,
    window_distance: int = 20,
    car_horizons: tuple = (5, 10),
) -> pd.DataFrame:
    """Add event-study CAR / abnormal-return features.

    Args:
        df: DataFrame with 'close' + market benchmark close + event flag columns.
        ticker: ticker symbol.
        market_close_col: column with market benchmark close (e.g. 'spy_close').
        event_flag_cols: columns whose nonzero/True values mark event dates.
        estimation_period: bars used for alpha/beta OLS.
        window_distance: gap between estimation end and event date (avoid leakage).
        car_horizons: rolling CAR windows.

    Notes:
        - If market_close_col absent, falls back to using the stock's own
          rolling-mean as market proxy (CARs become "abnormal vs own avg").
        - Always uses ONLY past data; .shift(1) on outputs.
    """
    out = df.copy()
    stock_ret = out["close"].pct_change()
    if market_close_col in out.columns:
        mkt_ret = out[market_close_col].pct_change()
    else:
        mkt_ret = pd.Series(np.nan, index=out.index)
        # Fallback: stock-vs-own-mean abnormal return
        roll_mean_ret = stock_ret.rolling(20).mean().shift(1)
    abnormal_ret = pd.Series(np.full(len(out), np.nan), index=out.index)
    alpha_series = pd.Series(np.full(len(out), np.nan), index=out.index)
    beta_series = pd.Series(np.full(len(out), np.nan), index=out.index)
    n = len(out)
    for i in range(estimation_period + window_distance, n):
        est_end = i - window_distance
        est_start = est_end - estimation_period
        if est_start < 0:
            continue
        s = stock_ret.iloc[est_start:est_end].values
        if market_close_col in out.columns:
            m = mkt_ret.iloc[est_start:est_end].values
            alpha, beta = _ols_alpha_beta(s, m)
            if np.isnan(beta):
                continue
            predicted = alpha + beta * (mkt_ret.iloc[i] if not pd.isna(mkt_ret.iloc[i]) else 0.0)
        else:
            alpha = float(np.nanmean(s))
            beta = 0.0
            predicted = alpha
        alpha_series.iloc[i] = alpha
        beta_series.iloc[i] = beta
        actual = stock_ret.iloc[i]
        abnormal_ret.iloc[i] = actual - predicted
    out["es_ar_t0"] = abnormal_ret.shift(1)
    out["es_alpha"] = alpha_series.shift(1)
    out["es_beta"] = beta_series.shift(1)
    for h in car_horizons:
        # CAR = rolling sum of past h abnormal returns
        out[f"es_car_{h}d"] = abnormal_ret.rolling(h).sum().shift(1)
    # BMP-style standardized AR (using estimation-period sigma)
    est_sigma = stock_ret.rolling(estimation_period).std().shift(window_distance + 1)
    out["es_bmp_z"] = (abnormal_ret / est_sigma.replace(0, np.nan)).shift(1)
    # Abnormal volume z (separate from AR)
    if "volume" in out.columns:
        vol_mean = out["volume"].rolling(estimation_period).mean().shift(window_distance + 1)
        vol_std = out["volume"].rolling(estimation_period).std().shift(window_distance + 1)
        out["es_abn_vol_z"] = ((out["volume"] - vol_mean) / vol_std.replace(0, np.nan)).shift(1)
    # Days since last significant event (|BMP-z| > 2)
    event_flag = (out["es_bmp_z"].abs() > 2.0).astype(int).fillna(0)
    out["es_days_since_significant_event"] = (
        (~event_flag.astype(bool)).astype(int).groupby((event_flag == 1).cumsum()).cumsum().shift(1)
    )
    return out


if __name__ == "__main__":
    print("TODO: wire eventstudy_features into v10 cross-sectional / regime layer.")
