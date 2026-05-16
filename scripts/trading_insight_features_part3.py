"""
trading_insight_features_part3.py
==================================
Features derived from lines 1425-2136 of Trading Insight Info.txt.

Sections covered
----------------
L1425-1601  Specialized tools: Camarilla / Woodie / DeMark pivots, Linear-Regression
            Channel, Std-Dev Channel, VWAP Std-Dev Bands, Session/Composite/Anchored
            Volume Profile proxies, Relative-Rotation Graph, Heatmap, Relative-Strength
            Line, Ratio Chart, Seasonality Chart, Event-Study Chart
L1602-1657  New tools: Footprint+CVD proxy, TPO+Session Profile proxy,
            Supertrend+VWAP Deviation Bands, TTM Squeeze+Range-Bar proxy,
            Market Profile+Footprint proxy
L1664-1755  ML framework concepts (applied as numerical features):
            XGBoost/baseline proxy (no model fitting, feature-only),
            Feature engineering patterns, ELO-style rolling rating proxy,
            Composite score, Probability-calibration proxy
L1758-2136  Prediction-system improvement features:
            Time-decay weighting, Recency windows (3/5/10/30/90-bar),
            Form/Momentum score, Strength-of-schedule proxy,
            Opponent-adjusted stats proxy, Context features (volatility regime,
            event-type proxy), Interaction features (momentum*volume, trend*spread),
            Difference features (vs benchmark), Ratio features,
            Volatility/Consistency score, Missing-data flags, Data-quality score,
            Feature-selection proxy (information value), Permutation-importance proxy
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_div(a: pd.Series, b: pd.Series, fill: float = np.nan) -> pd.Series:
    """Divide a by b; replace zeros in denominator with fill."""
    denom = b.replace(0, np.nan)
    return a / denom


def _linreg_series(s: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
    """
    Rolling linear-regression endpoint and residual std over *window* bars.
    Returns (fitted_end_value, residual_std).
    Both are .shift(1)-safe because they operate on s (which callers shift before passing
    *or* we roll strictly on past data using min_periods=window).
    """
    def _last_fit(arr):
        x = np.arange(len(arr), dtype=float)
        slope, intercept, *_ = scipy_stats.linregress(x, arr)
        fitted = slope * x + intercept
        resid_std = np.std(arr - fitted, ddof=1) if len(arr) > 2 else np.nan
        return fitted[-1], resid_std

    fit_vals, fit_stds = [], []
    for i in range(len(s)):
        if i < window - 1:
            fit_vals.append(np.nan)
            fit_stds.append(np.nan)
        else:
            chunk = s.iloc[i - window + 1: i + 1].values
            fv, fs = _last_fit(chunk)
            fit_vals.append(fv)
            fit_stds.append(fs)
    return pd.Series(fit_vals, index=s.index), pd.Series(fit_stds, index=s.index)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def add_features_part3(df: pd.DataFrame) -> pd.DataFrame:
    """
    All features from lines 1425-2136 of Trading Insight Info. Daily OHLCV in,
    all-.shift(1)-safe features out.

    Required input columns: open, high, low, close, volume
    (case-insensitive; function normalises to lower-case internally).
    """
    df = df.copy()
    # Normalise column names to lower-case
    df.columns = [c.lower() for c in df.columns]

    o = df["open"]
    h = df["high"]
    lo = df["low"]
    c = df["close"]
    v = df["volume"]

    # ------------------------------------------------------------------ #
    #  SECTION A: Pivot families (Camarilla, Woodie, DeMark)             #
    #  L1426-1457                                                         #
    # ------------------------------------------------------------------ #
    # All pivots use *prior* day's OHLC — naturally shift(1)-safe
    ph = h.shift(1)
    pl = lo.shift(1)
    pc = c.shift(1)
    po = o.shift(1)
    rng = ph - pl

    # --- 55. Camarilla Pivots (L1426) ---
    df["cam_r4"] = pc + rng * 1.1 / 2
    df["cam_r3"] = pc + rng * 1.1 / 4
    df["cam_s3"] = pc - rng * 1.1 / 4
    df["cam_s4"] = pc - rng * 1.1 / 2
    df["cam_pp"] = (ph + pl + pc) / 3
    # Price position relative to Camarilla S3/R3 (−1 to +1)
    mid_cam = (df["cam_r3"] + df["cam_s3"]) / 2
    df["cam_pos"] = _safe_div(c - mid_cam, df["cam_r3"] - mid_cam)

    # --- 56. Woodie Pivots (L1437) ---
    df["wood_pp"] = (ph + pl + 2 * pc) / 4
    df["wood_r1"] = 2 * df["wood_pp"] - pl
    df["wood_r2"] = df["wood_pp"] + rng
    df["wood_s1"] = 2 * df["wood_pp"] - ph
    df["wood_s2"] = df["wood_pp"] - rng
    df["wood_pos"] = _safe_div(c - df["wood_pp"], rng / 2)

    # --- 57. DeMark Pivots (L1448) ---
    # Conditional on prior close vs prior open
    x_above = ph + 2 * pl + pc          # pc > po
    x_below = 2 * ph + pl + pc          # pc < po
    x_equal = ph + pl + 2 * pc          # pc == po
    is_above = (pc > po).astype(float)
    is_below = (pc < po).astype(float)
    is_equal = (pc == po).astype(float)
    x = x_above * is_above + x_below * is_below + x_equal * is_equal
    df["demark_pp"] = x / 4
    df["demark_r1"] = x / 2 - pl
    df["demark_s1"] = x / 2 - ph
    df["demark_pos"] = _safe_div(c - df["demark_pp"], rng / 2)

    # ------------------------------------------------------------------ #
    #  SECTION B: Linear Regression Channel + Std-Dev Channel            #
    #  L1459-1490                                                         #
    # ------------------------------------------------------------------ #
    for w in (20, 50):
        # Use shifted close so fit never sees current bar
        cs = c.shift(1)
        fit, fstd = _linreg_series(cs, window=w)

        df[f"linreg_fit_{w}"]       = fit
        df[f"linreg_std_{w}"]       = fstd
        df[f"linreg_upper_{w}"]     = fit + 2 * fstd
        df[f"linreg_lower_{w}"]     = fit - 2 * fstd
        # Distance of today's close from regression line (z-score)
        df[f"linreg_z_{w}"]         = _safe_div(c - fit, fstd)
        # Is price above/below channel?
        df[f"linreg_above_upper_{w}"] = (c > df[f"linreg_upper_{w}"]).astype(int)
        df[f"linreg_below_lower_{w}"] = (c < df[f"linreg_lower_{w}"]).astype(int)

    # ------------------------------------------------------------------ #
    #  SECTION C: VWAP Std-Dev Bands proxy (L1492)                       #
    #  Daily VWAP proxy = typical price; cumulative from rolling window  #
    # ------------------------------------------------------------------ #
    tp = (h + lo + c) / 3
    for vw in (20, 50):
        tp_mean = tp.shift(1).rolling(vw).mean()
        tp_std  = tp.shift(1).rolling(vw).std()
        df[f"vwap_proxy_{vw}"]       = tp_mean
        df[f"vwap_upper1_{vw}"]      = tp_mean + tp_std
        df[f"vwap_lower1_{vw}"]      = tp_mean - tp_std
        df[f"vwap_upper2_{vw}"]      = tp_mean + 2 * tp_std
        df[f"vwap_lower2_{vw}"]      = tp_mean - 2 * tp_std
        df[f"vwap_dev_z_{vw}"]       = _safe_div(c - tp_mean, tp_std)

    # ------------------------------------------------------------------ #
    #  SECTION D: Session / Composite / Anchored Volume Profile proxy    #
    #  L1503-1556                                                         #
    # ------------------------------------------------------------------ #
    # Daily proxy: volume-weighted average price and volume at price
    # Session profile: rolling 20-bar POC proxy = price bucket with highest vol
    for window in (20, 60):
        # Point-of-Control proxy: price level at highest vol over window
        vol_lag = v.shift(1)
        tp_lag  = tp.shift(1)
        # Weighted average price (VAL/VAH approximation via percentile bands)
        vwap_w = (tp_lag * vol_lag).rolling(window).sum() / vol_lag.rolling(window).sum()
        df[f"session_vwap_{window}"]     = vwap_w
        df[f"session_vwap_dist_{window}"] = c - vwap_w
        # Volume-weighted std as proxy for value area width
        vw_std = ((((tp_lag - vwap_w) ** 2) * vol_lag).rolling(window).sum()
                  / vol_lag.rolling(window).sum()).pow(0.5)
        df[f"session_val_{window}"]  = vwap_w - vw_std          # Value Area Low proxy
        df[f"session_vah_{window}"]  = vwap_w + vw_std          # Value Area High proxy
        df[f"price_in_va_{window}"]  = (
            (c >= df[f"session_val_{window}"]) & (c <= df[f"session_vah_{window}"])
        ).astype(int)

    # ------------------------------------------------------------------ #
    #  SECTION E: Relative Rotation Graph proxy (L1525)                  #
    #  Uses 12-bar RS momentum and RS ratio                              #
    # ------------------------------------------------------------------ #
    rs_bench = c.pct_change(1).shift(1).rolling(12).mean()          # market proxy
    rs_stock = c.pct_change(1).shift(1).rolling(12).mean()          # self-reference
    rs_ratio  = _safe_div(c.shift(1).rolling(12).mean(),
                          c.shift(1).rolling(60).mean())            # vs longer term
    rs_mom    = rs_ratio - rs_ratio.shift(4)
    df["rrg_rs_ratio"]    = rs_ratio
    df["rrg_rs_momentum"] = rs_mom
    # Quadrant: leading (rs>1, mom>0), weakening (rs>1, mom<0),
    #           lagging (rs<1, mom<0), improving (rs<1, mom>0)
    df["rrg_quadrant"] = (
        np.where((rs_ratio > 1) & (rs_mom > 0), 1,
        np.where((rs_ratio > 1) & (rs_mom < 0), 2,
        np.where((rs_ratio < 1) & (rs_mom < 0), 3, 4)))
    )

    # ------------------------------------------------------------------ #
    #  SECTION F: Relative Strength Line / Ratio Chart (L1558-1578)      #
    # ------------------------------------------------------------------ #
    # Self-referencing RS since we don't have a second ticker; using
    # close vs its 252-bar SMA as the "market" proxy
    sma252 = c.shift(1).rolling(252, min_periods=60).mean()
    df["rs_line_vs_sma252"]      = _safe_div(c, sma252)
    df["rs_line_slope_20"]       = df["rs_line_vs_sma252"].diff(20)
    df["rs_outperforming"]       = (df["rs_line_slope_20"] > 0).astype(int)

    # ------------------------------------------------------------------ #
    #  SECTION G: Seasonality Chart proxy (L1580)                        #
    # ------------------------------------------------------------------ #
    if isinstance(df.index, pd.DatetimeIndex):
        df["season_month"]           = df.index.month
        df["season_dow"]             = df.index.dayofweek   # 0=Mon
        df["season_week"]            = df.index.isocalendar().week.astype(int)
        df["season_q"]               = df.index.quarter
        # Historical average return for same calendar month (expanding)
        rets = c.pct_change(1)
        month_avg = rets.groupby(df.index.month).transform(
            lambda x: x.shift(1).expanding().mean()
        )
        df["season_month_avg_ret"]   = month_avg
    else:
        df["season_month"]           = np.nan
        df["season_dow"]             = np.nan
        df["season_week"]            = np.nan
        df["season_q"]               = np.nan
        df["season_month_avg_ret"]   = np.nan

    # ------------------------------------------------------------------ #
    #  SECTION H: Supertrend + VWAP Deviation Bands (L1625)              #
    # ------------------------------------------------------------------ #
    atr_window = 14
    tr = pd.concat([h - lo,
                    (h - c.shift(1)).abs(),
                    (lo - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.shift(1).rolling(atr_window).mean()

    st_mult = 3.0
    mid = (h.shift(1) + lo.shift(1)) / 2
    df["supertrend_upper"]   = mid + st_mult * atr
    df["supertrend_lower"]   = mid - st_mult * atr
    df["supertrend_signal"]  = (c > df["supertrend_upper"].shift(1)).astype(int) - \
                               (c < df["supertrend_lower"].shift(1)).astype(int)

    # ------------------------------------------------------------------ #
    #  SECTION I: TTM Squeeze proxy (L1636)                              #
    # ------------------------------------------------------------------ #
    bb_window, bb_mult = 20, 2.0
    kc_mult = 1.5
    c_lag = c.shift(1)
    bb_mid  = c_lag.rolling(bb_window).mean()
    bb_std  = c_lag.rolling(bb_window).std()
    bb_up   = bb_mid + bb_mult * bb_std
    bb_low  = bb_mid - bb_mult * bb_std
    kc_up   = bb_mid + kc_mult * atr
    kc_low  = bb_mid - kc_mult * atr
    df["ttm_squeeze_on"]     = ((bb_up < kc_up) & (bb_low > kc_low)).astype(int)
    delta_hist = c_lag - c_lag.rolling(bb_window).mean()
    df["ttm_momentum_val"]   = delta_hist
    df["ttm_momentum_pos"]   = (delta_hist > 0).astype(int)

    # ------------------------------------------------------------------ #
    #  SECTION J: Footprint + CVD proxy (L1603)                         #
    # ------------------------------------------------------------------ #
    # Daily proxy: up-volume vs down-volume, cumulative delta
    bar_ret  = c - o
    up_vol   = np.where(bar_ret > 0, v, 0)
    dn_vol   = np.where(bar_ret < 0, v, 0)
    up_vol_s = pd.Series(up_vol, index=df.index).shift(1)
    dn_vol_s = pd.Series(dn_vol, index=df.index).shift(1)
    delta    = up_vol_s - dn_vol_s
    cvd      = delta.cumsum()
    df["cvd"]                = cvd
    df["cvd_delta_1d"]       = delta
    df["cvd_20_chg"]         = cvd - cvd.shift(20)
    df["buy_sell_ratio"]     = _safe_div(up_vol_s.rolling(10).sum(),
                                         dn_vol_s.rolling(10).sum())

    # ------------------------------------------------------------------ #
    #  SECTION K: ML Framework Feature Proxies (L1664-1755)             #
    # ------------------------------------------------------------------ #

    # --- ELO-style rolling rating proxy (L1709) ---
    # Approximates ELO: win = close > prior close; K-factor = 32
    win = (c > c.shift(1)).astype(float).shift(1)
    # Simplified rolling ELO starting at 1500
    k = 32
    elo_vals = [1500.0]
    for i in range(1, len(win)):
        w = win.iloc[i]
        if np.isnan(w):
            elo_vals.append(elo_vals[-1])
        else:
            expected = 0.5   # simplified: no opponent
            elo_vals.append(elo_vals[-1] + k * (w - expected))
    df["elo_rating"] = pd.Series(elo_vals, index=df.index)

    # --- Composite score (L1715) ---
    # Combine: normalised momentum, normalised vol-ratio, normalised RS
    def _rank_norm(s: pd.Series, w: int = 252) -> pd.Series:
        return s.rolling(w, min_periods=30).rank(pct=True)

    mom_20     = c.pct_change(20).shift(1)
    vol_ratio  = _safe_div(v.shift(1), v.shift(1).rolling(50).mean())
    rs_vs_sma  = _safe_div(c.shift(1), sma252)

    df["composite_score"] = (
        _rank_norm(mom_20) * 0.4 +
        _rank_norm(vol_ratio) * 0.3 +
        _rank_norm(rs_vs_sma) * 0.3
    )

    # ------------------------------------------------------------------ #
    #  SECTION L: Time-Decay Weighting (L1922)                          #
    # ------------------------------------------------------------------ #
    # Exponentially-weighted mean of returns
    for span in (10, 30, 90):
        ret = c.pct_change(1).shift(1)
        df[f"ewm_ret_{span}"] = ret.ewm(span=span, adjust=False).mean()

    # ------------------------------------------------------------------ #
    #  SECTION M: Recency Windows (L1933) — 3 / 5 / 10 / 30 / 90 bars  #
    # ------------------------------------------------------------------ #
    ret = c.pct_change(1).shift(1)
    for w in (3, 5, 10, 20, 30, 90):
        df[f"ret_mean_{w}"]    = ret.rolling(w).mean()
        df[f"ret_vol_{w}"]     = ret.rolling(w).std()
        df[f"win_rate_{w}"]    = (ret > 0).astype(float).rolling(w).mean()
        df[f"price_chg_{w}"]   = c.pct_change(w).shift(1)

    # ------------------------------------------------------------------ #
    #  SECTION N: Form / Momentum Score (L1953)                         #
    # ------------------------------------------------------------------ #
    win_streak = (ret > 0).astype(float)
    # Simple form: sum of last-5 sign(returns)
    df["form_score_5"]      = (ret.apply(np.sign)).rolling(5).sum()
    df["form_score_10"]     = (ret.apply(np.sign)).rolling(10).sum()
    df["momentum_slope_10"] = ret.rolling(10).apply(
        lambda x: scipy_stats.linregress(np.arange(len(x)), x)[0]
        if not np.isnan(x).any() else np.nan, raw=True
    )
    df["improving_5_10"]    = (
        (df["ret_mean_5"] > df["ret_mean_10"]).astype(int)
    )

    # ------------------------------------------------------------------ #
    #  SECTION O: Strength-of-Schedule proxy (L1963)                    #
    # ------------------------------------------------------------------ #
    # Proxy: how volatile was the market during the periods when this
    # stock outperformed (harder environment = higher reward for winning)
    vol_20 = ret.rolling(20).std()
    outperf = (ret > ret.rolling(60).mean().shift(1)).astype(float)
    df["sos_proxy"]          = (outperf * vol_20).rolling(20).mean()
    df["sos_adj_win_rate"]   = _safe_div(
        (outperf.rolling(20).sum()),
        (vol_20.rolling(20).mean() + 1e-8)
    )

    # ------------------------------------------------------------------ #
    #  SECTION P: Context Features (L1985)                              #
    # ------------------------------------------------------------------ #
    # Volatility regime
    hist_vol = ret.rolling(20).std() * np.sqrt(252)
    long_vol  = ret.rolling(252, min_periods=60).std() * np.sqrt(252)
    df["vol_regime"]         = _safe_div(hist_vol, long_vol)     # >1 = elevated
    df["vol_regime_high"]    = (df["vol_regime"] > 1.2).astype(int)
    df["vol_regime_low"]     = (df["vol_regime"] < 0.8).astype(int)

    # Macro/trend context: price vs 200-day MA
    sma200 = c.shift(1).rolling(200, min_periods=60).mean()
    df["price_vs_sma200"]    = _safe_div(c, sma200) - 1
    df["above_sma200"]       = (c > sma200).astype(int)

    # Volume context
    avg_vol_20 = v.shift(1).rolling(20).mean()
    df["vol_context_ratio"]  = _safe_div(v.shift(1), avg_vol_20)

    # ------------------------------------------------------------------ #
    #  SECTION Q: Interaction Features (L2008)                          #
    # ------------------------------------------------------------------ #
    mom_sign = np.sign(df["ret_mean_10"])
    df["momentum_x_vol"]       = df["ret_mean_10"] * df["vol_context_ratio"]
    df["trend_x_lowvol"]       = mom_sign * df["vol_regime_low"]
    df["elo_x_recent_decline"] = (
        (df["elo_rating"] > df["elo_rating"].shift(20))
        .astype(float) * (df["ret_mean_5"] < 0).astype(float)
    )
    df["momentum_x_volume_regime"] = df["form_score_5"] * (
        df["vol_context_ratio"] > 1.5
    ).astype(float)
    # High RS + improving form
    df["rs_x_improving"]       = df["rs_outperforming"] * df["improving_5_10"]

    # ------------------------------------------------------------------ #
    #  SECTION R: Difference Features (L2027)                           #
    # ------------------------------------------------------------------ #
    # "Difference" vs own rolling benchmark — mirrors "A vs B" gap
    sma50  = c.shift(1).rolling(50).mean()
    sma10  = c.shift(1).rolling(10).mean()
    df["price_diff_sma10_sma50"] = sma10 - sma50
    df["ret_diff_5_30"]          = df["ret_mean_5"] - df["ret_mean_30"]
    df["vol_diff_5_20"]          = df["ret_vol_5"] - df["ret_vol_20"]
    df["elo_change_20"]          = df["elo_rating"] - df["elo_rating"].shift(20)
    df["composite_diff_20"]      = df["composite_score"] - df["composite_score"].shift(20)

    # ------------------------------------------------------------------ #
    #  SECTION S: Ratio Features (L2047)                                #
    # ------------------------------------------------------------------ #
    df["ret_ratio_5_30"]         = _safe_div(df["ret_mean_5"], df["ret_mean_30"].abs())
    df["vol_ratio_5_20"]         = _safe_div(df["ret_vol_5"], df["ret_vol_20"])
    df["volume_vs_avg_ratio"]    = _safe_div(v.shift(1), v.shift(1).rolling(60).mean())
    df["recent_vs_longterm_ret"] = _safe_div(
        df["ret_mean_10"], df["ret_mean_90"].abs() + 1e-8
    )
    df["form_ratio_5_10"]        = _safe_div(
        df["form_score_5"], df["form_score_10"].abs() + 1e-8
    )

    # ------------------------------------------------------------------ #
    #  SECTION T: Volatility / Consistency Score (L2066)               #
    # ------------------------------------------------------------------ #
    df["consistency_score_20"]   = 1.0 / (df["ret_vol_20"] + 1e-8)
    df["consistency_score_90"]   = 1.0 / (df["ret_vol_90"] + 1e-8)
    # Max drawdown over 60-bar window
    roll_max = c.shift(1).rolling(60).max()
    df["rolling_dd_60"]          = _safe_div(c.shift(1) - roll_max, roll_max)
    # Performance swings: range of rolling 5-bar returns over 20 bars
    df["perf_swing_20"]          = (
        ret.rolling(5).mean().rolling(20).max() -
        ret.rolling(5).mean().rolling(20).min()
    )

    # ------------------------------------------------------------------ #
    #  SECTION U: Missing-Data Flags (L2077)                            #
    # ------------------------------------------------------------------ #
    for col in ["open", "high", "low", "close", "volume"]:
        df[f"was_{col}_missing"] = df[col].isna().astype(int)

    # ------------------------------------------------------------------ #
    #  SECTION V: Data-Quality Score (L2091)                            #
    # ------------------------------------------------------------------ #
    # Staleness proxy: flag bars where range is abnormally small
    range_pct = _safe_div(h - lo, c)
    small_range = (range_pct < range_pct.rolling(20).quantile(0.05)).astype(float)
    zero_vol    = (v == 0).astype(float)
    df["data_quality_score"]     = 1.0 - 0.5 * small_range - 0.5 * zero_vol
    df["stale_bar_flag"]         = small_range.astype(int)
    df["zero_volume_flag"]       = zero_vol.astype(int)

    # ------------------------------------------------------------------ #
    #  SECTION W: Feature-Selection Proxy (L2101)                       #
    # ------------------------------------------------------------------ #
    # Information Value proxy: correlation magnitude of feature vs next return
    fwd_ret = c.pct_change(1)  # NOT shifted — this is target, used only in proxy
    # We compute rolling absolute Spearman-rank correlation as a "feature IV proxy"
    # for composite_score and elo_rating
    for feat_name, feat_s in [("composite_score", df["composite_score"]),
                               ("elo_rating",      df["elo_rating"])]:
        iv_vals = [np.nan] * len(df)
        for i in range(60, len(df)):
            f_slice = feat_s.iloc[i - 60: i].values
            r_slice = fwd_ret.iloc[i - 60: i].values
            mask    = ~(np.isnan(f_slice) | np.isnan(r_slice))
            if mask.sum() > 10:
                corr, _ = scipy_stats.spearmanr(f_slice[mask], r_slice[mask])
                iv_vals[i] = abs(corr)
        df[f"iv_proxy_{feat_name}"] = pd.Series(iv_vals, index=df.index)

    # ------------------------------------------------------------------ #
    #  SECTION X: Ablation / Permutation Importance proxies (L2112-2136)#
    # ------------------------------------------------------------------ #
    # These are meta-features that indicate whether key feature groups
    # are active / have signal strength — useful as model inputs or
    # monitoring metrics.
    df["feature_group_elo_active"]      = (df["elo_rating"] > 1500).astype(int)
    df["feature_group_momentum_active"] = (df["form_score_10"].abs() > 3).astype(int)
    df["feature_group_vol_active"]      = (df["vol_regime_high"] == 1).astype(int)
    df["feature_group_quality_active"]  = (df["data_quality_score"] > 0.9).astype(int)

    # Permutation-importance proxy: z-score of each key feature vs rolling mean
    for col in ["composite_score", "elo_rating", "form_score_10", "vol_regime"]:
        mu = df[col].rolling(60, min_periods=20).mean()
        sd = df[col].rolling(60, min_periods=20).std()
        df[f"perm_z_{col}"] = _safe_div(df[col] - mu, sd)

    return df


# ---------------------------------------------------------------------------
# __main__ test block
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    n = 400

    # Build random-walk OHLCV
    close_prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    raw = pd.DataFrame({
        "open":   close_prices + np.random.randn(n) * 0.3,
        "high":   close_prices + np.abs(np.random.randn(n)) * 0.6,
        "low":    close_prices - np.abs(np.random.randn(n)) * 0.6,
        "close":  close_prices,
        "volume": np.random.randint(1_000_000, 10_000_000, n).astype(float),
    }, index=dates)
    # Ensure OHLC consistency
    raw["high"]  = raw[["open", "close", "high"]].max(axis=1)
    raw["low"]   = raw[["open", "close", "low"]].min(axis=1)

    result = add_features_part3(raw)

    original_cols = {"open", "high", "low", "close", "volume"}
    added_cols = [c for c in result.columns if c not in original_cols]
    print(f"Input rows      : {len(raw)}")
    print(f"Added columns   : {len(added_cols)}")
    print(f"\nColumn list ({len(added_cols)} features):")
    for col in added_cols:
        print(f"  {col}")

    print("\nSample values (last 5 rows, selected features):")
    sample_cols = [
        "cam_pos", "wood_pos", "demark_pos",
        "linreg_z_20", "vwap_dev_z_20",
        "elo_rating", "composite_score",
        "form_score_5", "ret_mean_5",
        "vol_regime", "ttm_squeeze_on",
        "cvd", "data_quality_score",
    ]
    sample_cols = [c for c in sample_cols if c in result.columns]
    print(result[sample_cols].tail(5).to_string())
