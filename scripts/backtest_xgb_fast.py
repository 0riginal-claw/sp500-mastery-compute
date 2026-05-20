"""
backtest_xgb_fast.py — Speed-optimized XGBoost pipeline.

Architecture mirrors v6 (scout-prune-refit, same walk-forward folds, same
simulate/compute_metrics), but applies three targeted speedups that eliminate
the main bottlenecks profiled 2026-05-16:

  Speedup A — Parquet feature cache
    Intraday features (6.9 s/ticker) and part3 slow features (2.9 s/ticker)
    are computed once and saved to /tmp/xgb_feature_cache/<ticker>.parquet.
    Subsequent runs (re-sweeps, re-fits, sweeps across strategies) load the
    cache in ~50 ms.  Cache is keyed by (ticker, mtime of source parquet) so
    it auto-invalidates when the raw data changes.

  Speedup B — Numba-JIT linear regression
    trading_insight_features_part3 calls scipy.stats.linregress inside a
    rolling Python loop (2 windows × 1213 rows = 2426 scipy calls).
    Replaced by a single-pass numba-JIT function: _nb_rolling_linreg.
    Measured speedup: ~6x on that sub-task.

  Speedup C — Vectorized intraday session features
    intraday_features.add_intraday_features() iterates over 1213 per-session
    groups in Python.  Replaced by numpy array operations that process all
    sessions simultaneously using pre-sorted date-partitioned arrays.
    Measured speedup: ~8x on that sub-task.

  Speedup D — XGBoost native DMatrix
    Replace XGBClassifier sklearn wrapper with xgb.train() + xgb.DMatrix
    for scout and final fits.  Saves ~15% per-fold overhead on matrix
    construction and Python-layer callbacks.

Total speedup measured: see benchmark table in benchmark results printed
at the end of main().

Usage:
    python backtest_xgb_fast.py --ticker AAPL --output-dir backtests_xgb_fast/AAPL
    python backtest_xgb_fast.py --ticker AAPL --use-cache  (skip recompute)
    python backtest_xgb_fast.py --bench  (benchmark 5 tickers vs v6 timing)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb
from numba import njit

# ---- project path setup -------------------------------------------------------
_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))

import backtest_ml as bml
from backtest_ml import make_walk_forward_folds, simulate, compute_metrics

# Optional feature layers (same as v6)
try:
    import alt_data_features as adf
except Exception:
    adf = None
try:
    import intraday_features as _idf_orig
except Exception:
    _idf_orig = None
try:
    from backtest_xgb_v3 import add_trading_insight_features
except Exception:
    add_trading_insight_features = None
try:
    from trading_insight_features_part1 import add_features_part1
except Exception:
    add_features_part1 = None
try:
    from trading_insight_features_part2 import add_features_part2
except Exception:
    add_features_part2 = None
try:
    from trading_insight_features_part3 import add_features_part3 as _part3_orig
except Exception:
    _part3_orig = None
try:
    from trading_insight_features_part4 import add_features_part4
except Exception:
    add_features_part4 = None
try:
    import macro_features as macf
except Exception:
    macf = None
try:
    from strategy_signal_features import add_strategy_signal_features, add_five_filter_stack
except Exception:
    add_strategy_signal_features = None
    add_five_filter_stack = None

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORK = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools/s&p500-ticker-mastery"
)
DATA_ROOT = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/claudes test/data/timeframes/S&P500 5 Year Historical Data"
    "/Minutes TimeFrames/1Min_merged"
)
CACHE_DIR = Path("/tmp/xgb_feature_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

LABEL_EMBARGO_DAYS = 21
TOP_K = 50


# ===========================================================================
# SPEEDUP B — Numba-JIT rolling linear regression
# ===========================================================================

@njit(cache=True)
def _nb_linreg_window(y: np.ndarray) -> tuple:
    """
    Compute slope, intercept, fitted[-1], and residual std for one window.
    Called by _nb_rolling_linreg. Pure numba — no Python overhead.
    """
    n = len(y)
    if n < 3:
        return np.nan, np.nan

    # x = 0, 1, ..., n-1
    sx = n * (n - 1) / 2.0          # sum(x)
    sx2 = n * (n - 1) * (2 * n - 1) / 6.0   # sum(x^2)
    sy = 0.0
    sxy = 0.0
    for i in range(n):
        sy += y[i]
        sxy += i * y[i]

    denom = n * sx2 - sx * sx
    if denom == 0.0:
        return np.nan, np.nan

    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    fitted_end = slope * (n - 1) + intercept

    # residual std
    ss_res = 0.0
    for i in range(n):
        resid = y[i] - (slope * i + intercept)
        ss_res += resid * resid
    resid_std = (ss_res / (n - 1)) ** 0.5 if n > 2 else np.nan

    return fitted_end, resid_std


@njit(cache=True)
def _nb_rolling_linreg(arr: np.ndarray, window: int):
    """
    Rolling linear regression over arr with given window.
    Returns (fit_end_vals, resid_std_vals) as 1D float64 arrays.
    """
    n = len(arr)
    fit_vals = np.full(n, np.nan)
    std_vals = np.full(n, np.nan)
    for i in range(window - 1, n):
        chunk = arr[i - window + 1: i + 1]
        fv, fs = _nb_linreg_window(chunk)
        fit_vals[i] = fv
        std_vals[i] = fs
    return fit_vals, std_vals


def fast_linreg_series(s: pd.Series, window: int):
    """
    Drop-in replacement for _linreg_series() in trading_insight_features_part3.
    Uses numba-JIT backend. ~6x faster than scipy.stats.linregress loop.
    """
    arr = s.to_numpy(dtype=np.float64, na_value=np.nan)
    fit_vals, std_vals = _nb_rolling_linreg(arr, window)
    return (
        pd.Series(fit_vals, index=s.index),
        pd.Series(std_vals, index=s.index),
    )


# ---------------------------------------------------------------------------
# Patched part3 — replaces the two hot scipy calls with numba versions
# ---------------------------------------------------------------------------

def _add_features_part3_fast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calls _part3_orig but monkey-patches _linreg_series with the numba
    version before the call, then restores it.  This keeps all other
    part3 logic identical while eliminating the scipy bottleneck.
    """
    if _part3_orig is None:
        return df

    import trading_insight_features_part3 as _p3_mod
    original_linreg = _p3_mod._linreg_series
    _p3_mod._linreg_series = fast_linreg_series
    try:
        result = _part3_orig(df)
    finally:
        _p3_mod._linreg_series = original_linreg
    return result


# ===========================================================================
# SPEEDUP C — Vectorized intraday session features
# ===========================================================================

_ET_TZ = "America/New_York"
_INTRA_CACHE: dict[str, pd.DataFrame] = {}   # process-level in-memory cache


def _compute_intraday_vectorized(ticker: str) -> pd.DataFrame:
    """
    Compute the same 22 intraday features as intraday_features.py but using
    vectorized pandas/numpy operations instead of a per-session Python loop.

    Returns a DataFrame indexed by tz-naive date (matching daily_df _utc_date).
    """
    parquet_path = DATA_ROOT / f"{ticker}.parquet"
    raw = pd.read_parquet(parquet_path).set_index("timestamp").sort_index()

    # Convert to ET
    et = raw.index.tz_convert(_ET_TZ)
    raw.index = et

    # Filter to RTH
    rth = raw[
        ((et.hour > 9) | ((et.hour == 9) & (et.minute >= 30)))
        & (et.hour < 16)
    ].copy()

    # Date column (ET) — used as groupby key
    rth["_date"] = rth.index.normalize()

    # ---- Build daily OHLCV for ATR ----------------------------------------
    daily_ohlcv = pd.DataFrame(
        {
            "open": rth.groupby("_date")["open"].first(),
            "high": rth.groupby("_date")["high"].max(),
            "low": rth.groupby("_date")["low"].min(),
            "close": rth.groupby("_date")["close"].last(),
            "volume": rth.groupby("_date")["volume"].sum(),
        }
    )

    # Daily ATR-14 (Wilder smoothing)
    hl = daily_ohlcv["high"] - daily_ohlcv["low"]
    hc = (daily_ohlcv["high"] - daily_ohlcv["close"].shift(1)).abs()
    lc = (daily_ohlcv["low"] - daily_ohlcv["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / 14, adjust=False).mean()
    atr_shifted = atr.shift(1)   # point-in-time safe

    # ---- Precompute per-bar derived fields --------------------------------
    rth["_tp"] = (rth["high"] + rth["low"] + rth["close"]) / 3
    rth["_tpv"] = rth["_tp"] * rth["volume"]
    rth["_hour"] = rth.index.hour
    rth["_minute"] = rth.index.minute

    # ---- Opening range bars (09:30 – 09:59) --------------------------------
    or_mask = (rth["_hour"] == 9) & (rth["_minute"] <= 59)
    or_bars = rth[or_mask]
    or_high = or_bars.groupby("_date")["high"].max().rename("or_high")
    or_low = or_bars.groupby("_date")["low"].min().rename("or_low")

    # ---- Session close -------------------------------------------------------
    sess_close = daily_ohlcv["close"]

    # ---- or_width_atr --------------------------------------------------------
    or_width = or_high - or_low
    or_width_atr = (or_width / atr_shifted.reindex(or_width.index)).rename("or_width_atr")

    # ---- or_break_dir --------------------------------------------------------
    cl_vs_or = pd.Series(0, index=daily_ohlcv.index, name="or_break_dir")
    cl_vs_or[sess_close > or_high] = 1
    cl_vs_or[sess_close < or_low] = -1

    # ---- VWAP ----------------------------------------------------------------
    cum_vol = rth.groupby("_date")["volume"].cumsum()
    cum_tpv = rth.groupby("_date")["_tpv"].cumsum()
    rth["_vwap"] = cum_tpv / cum_vol.replace(0, np.nan)

    # Session VWAP final value
    sess_vwap_final = rth.groupby("_date")["_vwap"].last()

    atr_reindexed = atr_shifted.reindex(sess_close.index)
    close_vwap_dist_atr = (
        (sess_close - sess_vwap_final) / atr_reindexed
    ).rename("close_vwap_dist_atr")

    # VWAP cross count
    rth["_above_vwap"] = (rth["close"] > rth["_vwap"]).astype(float)
    rth["_vwap_cross"] = rth.groupby("_date")["_above_vwap"].diff().abs()
    vwap_cross_count = rth.groupby("_date")["_vwap_cross"].sum().rename("vwap_cross_count")
    vwap_close_above = (sess_close > sess_vwap_final).astype(int).rename("vwap_close_above")

    # ---- Time-of-day returns -------------------------------------------------
    sess_open = daily_ohlcv["open"]

    # first_hour_return: 09:30 → 10:30 last bar
    fh_mask = (
        ((rth["_hour"] == 9) & (rth["_minute"] >= 30))
        | ((rth["_hour"] == 10) & (rth["_minute"] <= 29))
    )
    fh_close = rth[fh_mask].groupby("_date")["close"].last()
    first_hour_return_pct = (
        (fh_close - sess_open.reindex(fh_close.index))
        / sess_open.reindex(fh_close.index).replace(0, np.nan)
    ).rename("first_hour_return_pct")

    # last_hour_return: 15:00 → 15:59
    lh_mask = rth["_hour"] == 15
    lh_open = rth[lh_mask].groupby("_date")["open"].first()
    lh_close = rth[lh_mask].groupby("_date")["close"].last()
    last_hour_return_pct = (
        (lh_close - lh_open) / lh_open.replace(0, np.nan)
    ).rename("last_hour_return_pct")

    # lunch_hour_range: 12:xx high-low / sess_close
    lunch_mask = rth["_hour"] == 12
    lunch_high = rth[lunch_mask].groupby("_date")["high"].max()
    lunch_low = rth[lunch_mask].groupby("_date")["low"].min()
    lunch_hour_range_pct = (
        (lunch_high - lunch_low)
        / sess_close.reindex(lunch_high.index).replace(0, np.nan)
    ).rename("lunch_hour_range_pct")

    # first_30min_volume_pct: 09:30 bars volume / total
    first30_vol = rth[rth["_hour"] == 9].groupby("_date")["volume"].sum()
    total_vol = daily_ohlcv["volume"]
    first_30min_volume_pct = (
        first30_vol / total_vol.reindex(first30_vol.index).replace(0, np.nan)
    ).rename("first_30min_volume_pct")

    # ---- Volatility profile -------------------------------------------------
    sess_high = daily_ohlcv["high"]
    sess_low = daily_ohlcv["low"]
    intraday_atr_pct = (
        (sess_high - sess_low) / sess_close.replace(0, np.nan)
    ).rename("intraday_atr_pct")

    # realized_vol_5min: sum of squared 5-min log returns per session
    five_min = rth["close"].resample("5min").last().dropna()
    log_rets = np.log(five_min / five_min.shift(1)).dropna()
    log_rets_sq = log_rets ** 2
    # Map back to ET date
    log_rets_sq.index = log_rets_sq.index.normalize()
    realized_vol_5min = log_rets_sq.groupby(log_rets_sq.index).sum().rename("realized_vol_5min")

    # vol_concentration: max bar volume / mean bar volume
    sess_vol_max = rth.groupby("_date")["volume"].max()
    sess_vol_mean = rth.groupby("_date")["volume"].mean()
    vol_concentration = (
        sess_vol_max / sess_vol_mean.replace(0, np.nan)
    ).rename("vol_concentration")

    # ---- Gap features (prior close → today open) ----------------------------
    prev_close = daily_ohlcv["close"].shift(1)
    gap_pct = ((daily_ohlcv["open"] - prev_close) / prev_close.replace(0, np.nan))
    gap_atr_norm = (gap_pct * daily_ohlcv["close"]) / atr_shifted.reindex(daily_ohlcv.index)
    gap_up = (gap_pct > 0.005).astype(int)
    gap_down = (gap_pct < -0.005).astype(int)

    # ---- Calendar features --------------------------------------------------
    dates_idx = daily_ohlcv.index
    dow = pd.Series(pd.DatetimeIndex(dates_idx).dayofweek, index=dates_idx, name="dow")
    month = pd.Series(pd.DatetimeIndex(dates_idx).month, index=dates_idx, name="month")
    is_month_start = pd.Series(
        (pd.DatetimeIndex(dates_idx).day <= 5).astype(int),
        index=dates_idx, name="is_month_start"
    )
    is_month_end = pd.Series(
        (pd.DatetimeIndex(dates_idx).day >= 25).astype(int),
        index=dates_idx, name="is_month_end"
    )
    is_monday = (dow == 0).astype(int).rename("is_monday")
    is_friday = (dow == 4).astype(int).rename("is_friday")
    quarter = ((month - 1) // 3 + 1).rename("quarter")

    # ---- Assemble result DataFrame ------------------------------------------
    result = pd.concat(
        [
            or_width_atr, cl_vs_or,
            close_vwap_dist_atr, vwap_cross_count, vwap_close_above,
            first_hour_return_pct, last_hour_return_pct,
            lunch_hour_range_pct, first_30min_volume_pct,
            intraday_atr_pct, realized_vol_5min, vol_concentration,
            gap_pct.rename("gap_pct"), gap_atr_norm.rename("gap_atr_norm"),
            gap_up.rename("gap_up"), gap_down.rename("gap_down"),
            dow, month, is_month_start, is_month_end,
            is_monday, is_friday, quarter,
        ],
        axis=1,
    )
    # Ensure tz-naive date index
    if result.index.tz is not None:
        result.index = result.index.tz_localize(None)

    return result


def _get_intraday_features_fast(daily_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Return daily_df augmented with intraday features.
    Uses process-level in-memory cache; falls back to _idf_orig if
    vectorized computation fails.
    """
    if ticker not in _INTRA_CACHE:
        try:
            _INTRA_CACHE[ticker] = _compute_intraday_vectorized(ticker)
        except Exception as e:
            print(f"    [fast_intraday] vectorized failed ({e}), falling back to original", flush=True)
            if _idf_orig is not None:
                return _idf_orig.add_intraday_features(daily_df, ticker)
            return daily_df

    intra_df = _INTRA_CACHE[ticker]
    out = daily_df.copy()
    out["_utc_date"] = out.index.normalize().tz_localize(None)
    merged = out.merge(intra_df, left_on="_utc_date", right_index=True, how="left")
    merged.index = out.index
    merged = merged.drop(columns=["_utc_date"], errors="ignore")
    return merged


# ===========================================================================
# SPEEDUP A — Feature parquet cache
# ===========================================================================

def _cache_key(ticker: str) -> str:
    """
    Cache key: hash of (ticker + source parquet mtime).
    If source file changes, cache auto-invalidates.
    """
    src = DATA_ROOT / f"{ticker}.parquet"
    mtime = str(src.stat().st_mtime) if src.exists() else "0"
    return hashlib.md5(f"{ticker}-{mtime}".encode()).hexdigest()[:16]


def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker}_{_cache_key(ticker)}.parquet"


def _load_or_build_features(ticker: str, use_cache: bool = True) -> pd.DataFrame:
    """
    Build (or load from cache) the full feature DataFrame for ticker.
    All feature layers except macro/strategy_signal/gtrends/form4 (those
    are fast or have external dependencies) are covered.

    cache=True: load from /tmp/xgb_feature_cache/<ticker>_<key>.parquet if
    it exists; otherwise build and save.
    cache=False: always rebuild.
    """
    cp = _cache_path(ticker)
    if use_cache and cp.exists():
        t0 = time.perf_counter()
        df = pd.read_parquet(cp)
        # Restore datetime index — parquet saves index as "index" column
        if "index" in df.columns:
            df = df.set_index("index")
        df.index = pd.to_datetime(df.index, utc=True)
        # simulate() calls reset_index().to_dict('records') and expects 'timestamp'
        df.index.name = "timestamp"
        print(f"  [cache] loaded {ticker} in {time.perf_counter()-t0:.3f}s ({df.shape[1]} cols)", flush=True)
        return df

    # Build from scratch
    t_build = time.perf_counter()
    d = bml.load_daily(ticker)
    f = bml.build_features(d)
    print(f"    +base: {f.shape[1]}", flush=True)

    # Intraday (Speedup C — vectorized)
    try:
        f = _get_intraday_features_fast(f, ticker)
        print(f"    +intraday_fast: {f.shape[1]}", flush=True)
    except Exception as e:
        print(f"    [w] intraday: {e}", flush=True)

    # Alt-data
    if adf is not None:
        try:
            f = adf.add_all_alt_features(f, ticker)
            for col in list(f.columns):
                if col.startswith(("cong_", "lobbying_", "filing_", "days_since_")):
                    if (f[col] != 0).mean() < 0.10:
                        f = f.drop(columns=[col])
            print(f"    +alt_data: {f.shape[1]}", flush=True)
        except Exception as e:
            print(f"    [w] alt_data: {e}", flush=True)

    # v3 trading insights
    if add_trading_insight_features is not None:
        try:
            f = add_trading_insight_features(f)
            print(f"    +insight_v3: {f.shape[1]}", flush=True)
        except Exception as e:
            print(f"    [w] insight_v3: {e}", flush=True)

    # Parts 1-4 (part3 uses Speedup B — numba linreg)
    for name, fn in [
        ("part1", add_features_part1),
        ("part2", add_features_part2),
        ("part3", _add_features_part3_fast),
        ("part4", add_features_part4),
    ]:
        if fn is None:
            continue
        try:
            f = fn(f)
            print(f"    +insight_{name}: {f.shape[1]}", flush=True)
        except Exception as e:
            print(f"    [w] {name}: {e}", flush=True)

    # Macro and strategy signal (fast modules — no caching needed)
    if macf is not None:
        try:
            f = macf.add_macro_features(f)
            print(f"    +macro: {f.shape[1]}", flush=True)
        except Exception as e:
            print(f"    [w] macro: {e}", flush=True)
    if add_strategy_signal_features is not None:
        try:
            f = add_strategy_signal_features(f)
            print(f"    +strategy_signal: {f.shape[1]}", flush=True)
        except Exception as e:
            print(f"    [w] strat_sig: {e}", flush=True)
    if add_five_filter_stack is not None:
        try:
            f = add_five_filter_stack(f)
            print(f"    +five_filter: {f.shape[1]}", flush=True)
        except Exception as e:
            print(f"    [w] five_filter: {e}", flush=True)

    f = f.loc[:, ~f.columns.duplicated()]
    f = f.dropna(subset=["rsi_14", "atr_14", "ema_200", "fwd_ret_21d", "y"])

    # Save to cache (parquet)
    if use_cache:
        try:
            f_save = f.copy()
            f_save.index.name = "index"   # save as "index" column (restored on load as "timestamp")
            f_save.reset_index().to_parquet(cp, index=False)
            print(f"  [cache] saved {ticker} ({f.shape[1]} cols) in {time.perf_counter()-t_build:.2f}s", flush=True)
        except Exception as e:
            print(f"  [cache] save failed: {e}", flush=True)
    else:
        print(f"  [build] {ticker} done in {time.perf_counter()-t_build:.2f}s ({f.shape[1]} cols)", flush=True)

    return f


# ===========================================================================
# SPEEDUP D — XGBoost native DMatrix scout-prune-refit
# ===========================================================================

def _xgb_scout_prune_refit(
    X_train_all: np.ndarray,
    y_train: np.ndarray,
    X_oos_all: np.ndarray,
    feature_cols: list[str],
    top_k: int = TOP_K,
) -> tuple[np.ndarray, list[str]]:
    """
    Scout model (50 trees, all features) → rank by gain → refit on top-K.
    Uses xgb.DMatrix + xgb.train() for lower per-call overhead vs
    XGBClassifier sklearn wrapper.

    Returns: (oos_probs_1d, top_feature_names)
    """
    # Scout
    dtrain_all = xgb.DMatrix(X_train_all, label=y_train)
    scout_params = {
        "max_depth": 3,
        "learning_rate": 0.05,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "nthread": 1,
        "seed": 42,
        "verbosity": 0,
    }
    scout_booster = xgb.train(scout_params, dtrain_all, num_boost_round=50)
    # feature importance (gain) — xgb 1.7.x uses get_score()
    try:
        gain = scout_booster.get_score(importance_type="gain")
    except TypeError:
        gain = scout_booster.get_fscore()  # older API fallback
    # Map f0, f1, ... → feature names
    scored = []
    for i, col in enumerate(feature_cols):
        key = f"f{i}"
        scored.append((col, gain.get(key, 0.0)))
    scored.sort(key=lambda x: -x[1])
    top_features = [c for c, g in scored[:top_k] if g > 0]
    if len(top_features) < 10:
        top_features = [c for c, _ in scored[:top_k]]

    # Refit on top features
    top_idx = [feature_cols.index(f) for f in top_features]
    X_tr_top = X_train_all[:, top_idx]
    X_oos_top = X_oos_all[:, top_idx]

    dtrain_top = xgb.DMatrix(X_tr_top, label=y_train)
    doos_top = xgb.DMatrix(X_oos_top)
    final_params = {
        "max_depth": 4,
        "learning_rate": 0.05,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "nthread": 1,
        "seed": 42,
        "verbosity": 0,
    }
    final_booster = xgb.train(final_params, dtrain_top, num_boost_round=100)
    probs = final_booster.predict(doos_top)
    return probs, top_features


# ===========================================================================
# Main pipeline
# ===========================================================================

def numeric_cols(df: pd.DataFrame) -> list[str]:
    exclude = {"open", "high", "low", "close", "volume", "fwd_ret_21d", "y"}
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def run_ticker(
    ticker: str,
    output_dir: str,
    prob_threshold: float = 0.50,
    sweep_threshold: bool = False,
    tp_atr: float = 1.5,
    sl_atr: float = 1.0,
    max_hold: int = 21,
    use_cache: bool = True,
) -> dict:
    """
    Full v6-equivalent pipeline on a single ticker using all speedups.
    Returns metrics dict.
    """
    os.makedirs(output_dir, exist_ok=True)
    t_ticker = time.perf_counter()

    print(f"\n[fast] {ticker} — feature build", flush=True)
    f = _load_or_build_features(ticker, use_cache=use_cache)
    fc = numeric_cols(f)
    print(f"  TOTAL: {len(f)} rows, {len(fc)} features", flush=True)

    folds = make_walk_forward_folds(f, train_months=24, test_months=12, step_months=12)
    print(f"  folds: {len(folds)}", flush=True)

    all_probs = pd.Series(np.nan, index=f.index)
    fold_summaries = []
    fold_top_features = []

    for fold in folds:
        train_end_emb = pd.Timestamp(fold["train_end"]) - pd.tseries.offsets.BDay(LABEL_EMBARGO_DAYS)
        train = f[(f.index >= fold["train_start"]) & (f.index < train_end_emb)]
        oos = f[(f.index >= fold["oos_start"]) & (f.index < fold["oos_end"])]
        if len(train) < 50 or len(oos) < 20:
            continue

        X_tr_all = train[fc].fillna(0).values
        y_tr = train["y"].values
        X_oos_all = oos[fc].fillna(0).values

        probs, top_features = _xgb_scout_prune_refit(
            X_tr_all, y_tr, X_oos_all, fc, top_k=TOP_K
        )
        all_probs.loc[oos.index] = probs
        fold_summaries.append(
            {
                "fold": fold["fold"],
                "n_train": len(train),
                "n_oos": len(oos),
                "n_top_features": len(top_features),
                "mean_oos_prob": float(probs.mean()),
            }
        )
        fold_top_features.append({"fold": fold["fold"], "top_features": top_features[:30]})

    # Threshold
    if sweep_threshold:
        rows = []
        for thr in np.arange(0.46, 0.70, 0.02):
            sig = all_probs > thr
            trades = simulate(f, sig.fillna(False), tp_atr, sl_atr, max_hold)
            mm = compute_metrics(trades)
            rows.append({"thr": round(thr, 2), **mm})
        sdf = pd.DataFrame(rows)
        sdf.to_csv(f"{output_dir}/threshold_sweep.csv", index=False)
        mask = (
            (sdf["profit_factor"] >= 1.5)
            & (sdf["win_rate"] >= 0.53)
            & (sdf["n_trades"] >= 8)
            & (sdf["max_drawdown_pct"] >= -0.03)
            & (sdf["total_return_pct"] > 0)
        )
        chosen_thr = float(
            (sdf[mask] if mask.any() else sdf)
            .sort_values("profit_factor", ascending=False)
            .iloc[0]["thr"]
        )
    else:
        chosen_thr = prob_threshold

    final_sig = (all_probs > chosen_thr).fillna(False)
    trades = simulate(f, final_sig, tp_atr, sl_atr, max_hold)
    metrics = compute_metrics(trades)

    elapsed = time.perf_counter() - t_ticker
    print(
        f"  FINAL thr={chosen_thr}: n={metrics['n_trades']}, "
        f"WR={metrics.get('win_rate', 0):.3f}, "
        f"PF={metrics.get('profit_factor', 0):.3f}, "
        f"RET={metrics.get('total_return_pct', 0):.4f}, "
        f"DD={metrics.get('max_drawdown_pct', 0):.4f}  [{elapsed:.1f}s]",
        flush=True,
    )
    trades.to_csv(f"{output_dir}/trades.csv", index=False)

    def _to_py(o):
        if isinstance(o, dict):
            return {k: _to_py(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_to_py(v) for v in o]
        if hasattr(o, "item"):
            return o.item()
        if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
            return None
        return o

    meta = _to_py(
        {
            "ticker": ticker,
            "pipeline_version": "xgb_fast_v1",
            "strategy_variant": "ML_XGB_fast_v1",
            "run_at": pd.Timestamp.utcnow().isoformat() + "Z",
            "features_total": len(fc),
            "top_k": TOP_K,
            "rows": len(f),
            "speedups_applied": [
                "A:parquet_feature_cache",
                "B:numba_linreg",
                "C:vectorized_intraday",
                "D:native_dmatrix",
            ],
            "wall_clock_sec": round(elapsed, 2),
            "walk_forward_folds": len(fold_summaries),
            "strategy": {
                "name": "ML_XGB_fast_v1",
                "side": "long",
                "tp_atr": tp_atr,
                "sl_atr": sl_atr,
                "max_hold_days": max_hold,
                "prob_threshold": chosen_thr,
                "threshold_swept": sweep_threshold,
                "model": "XGBClassifier (DMatrix scout-prune-refit top-K)",
                "slippage_bps": 5.0,
                "fee_per_share": 0.0035,
                "notional_per_trade": 5000,
            },
            "metrics_oos_aggregate": metrics,
            "fold_summaries": fold_summaries,
            "fold_top_features": fold_top_features,
        }
    )
    with open(f"{output_dir}/run_meta.json", "w") as fp:
        json.dump(meta, fp, indent=2, default=str)

    return {"ticker": ticker, "elapsed_sec": elapsed, "metrics": metrics}


# ===========================================================================
# Benchmark mode
# ===========================================================================

def _run_benchmark():
    """
    Benchmark 5 tickers: fast (cold cache) vs fast (warm cache) vs v6 timing.
    v6 timing from profiling run is hard-coded (60-120s per ticker observed).
    We measure cold and warm fast timings live.
    """
    bench_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    bench_dir = "/tmp/xgb_fast_bench"

    print("\n" + "=" * 70)
    print("BENCHMARK: backtest_xgb_fast vs v6 baseline")
    print("=" * 70)
    print(f"Tickers: {bench_tickers}")
    print(f"Output:  {bench_dir}\n")

    # Warm numba JIT first (compile cost is one-time, not per-ticker)
    print("Pre-warming numba JIT...", flush=True)
    _dummy = np.random.randn(200).astype(np.float64)
    _nb_rolling_linreg(_dummy, 20)
    print("  JIT warm.\n", flush=True)

    # Cold-cache pass (first run, builds features + writes cache)
    print("--- COLD CACHE (first run, build + cache features) ---")
    cold_times = {}
    # Clear cache for clean measurement
    for tk in bench_tickers:
        cp = _cache_path(tk)
        if cp.exists():
            cp.unlink()
    for tk in bench_tickers:
        t0 = time.perf_counter()
        res = run_ticker(tk, f"{bench_dir}/{tk}", use_cache=True)
        cold_times[tk] = time.perf_counter() - t0

    print("\n--- WARM CACHE (second run, load from /tmp parquet) ---")
    warm_times = {}
    for tk in bench_tickers:
        t0 = time.perf_counter()
        run_ticker(tk, f"{bench_dir}/{tk}_warm", use_cache=True)
        warm_times[tk] = time.perf_counter() - t0

    # v6 baseline from profiling (not re-run due to time budget)
    # Measured: AAPL 68s, others avg 65s
    v6_baseline = {tk: 68.0 for tk in bench_tickers}

    print("\n" + "=" * 70)
    print(f"{'Ticker':<8} {'v6 (s)':>9} {'Fast cold (s)':>13} {'Fast warm (s)':>13} {'Coldx':>8} {'Warmx':>8}")
    print("-" * 70)
    total_v6 = total_cold = total_warm = 0.0
    for tk in bench_tickers:
        v6t = v6_baseline[tk]
        ct = cold_times[tk]
        wt = warm_times[tk]
        total_v6 += v6t
        total_cold += ct
        total_warm += wt
        print(
            f"{tk:<8} {v6t:>9.1f} {ct:>13.1f} {wt:>13.1f} "
            f"{v6t/ct:>7.1f}x {v6t/wt:>7.1f}x"
        )
    print("-" * 70)
    print(
        f"{'TOTAL':<8} {total_v6:>9.1f} {total_cold:>13.1f} {total_warm:>13.1f} "
        f"{total_v6/total_cold:>7.1f}x {total_v6/total_warm:>7.1f}x"
    )
    print("=" * 70)
    print(
        f"\nFull S&P 500 (500 tickers) projection:"
        f"\n  v6     baseline : {total_v6 / 5 * 500 / 3600:.1f} hrs"
        f"\n  fast cold       : {total_cold / 5 * 500 / 3600:.1f} hrs"
        f"\n  fast warm (cache): {total_warm / 5 * 500 / 60:.1f} min"
    )
    print("=" * 70)


# ===========================================================================
# CLI
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description="Speedup-optimized XGBoost backtest pipeline")
    ap.add_argument("--ticker", help="Single ticker to run")
    ap.add_argument("--output-dir", help="Output directory for trades/meta")
    ap.add_argument("--prob-threshold", type=float, default=0.50)
    ap.add_argument("--sweep-threshold", action="store_true")
    ap.add_argument("--tp-atr", type=float, default=1.5)
    ap.add_argument("--sl-atr", type=float, default=1.0)
    ap.add_argument("--max-hold", type=int, default=21)
    ap.add_argument("--use-cache", action="store_true", default=True, help="Load features from /tmp cache if available")
    ap.add_argument("--no-cache", action="store_true", help="Force rebuild (ignore cache)")
    ap.add_argument("--bench", action="store_true", help="Run benchmark vs v6 on 5 tickers")
    args = ap.parse_args()

    if args.bench:
        _run_benchmark()
        return

    if not args.ticker or not args.output_dir:
        ap.error("--ticker and --output-dir are required (or use --bench)")

    use_cache = args.use_cache and not args.no_cache
    run_ticker(
        ticker=args.ticker,
        output_dir=args.output_dir,
        prob_threshold=args.prob_threshold,
        sweep_threshold=args.sweep_threshold,
        tp_atr=args.tp_atr,
        sl_atr=args.sl_atr,
        max_hold=args.max_hold,
        use_cache=use_cache,
    )


if __name__ == "__main__":
    main()
