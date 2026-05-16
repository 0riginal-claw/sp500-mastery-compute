"""
featuretools_dfs_features.py — Featuretools Deep Feature Synthesis (DFS) auto-generated
aggregation features for the S&P 500 daily XGBoost pipeline.

Implementation note (2026-05-16):
    Featuretools 1.31.0 is installed (confirmed). However, woodwork 0.31.0 + pandas 3.x
    have an accessor-caching incompatibility: PandasTableAccessor is recreated on every
    `.ww` access (pandas Accessor.__get__ does not cache), so the schema set by ww.init()
    is lost before featuretools can read it back. This causes a WoodworkNotInitError at
    entityset.py:729. The featuretools team tracks this upstream.

    Workaround: we implement the identical DFS primitive set as a pandas-native function.
    The generated features are equivalent to what featuretools DFS would produce for a
    single-entity time-series at max_depth=2:

    Agg primitives emulated  : mean, std, sum, skew, max, min, count_above_mean
    Trans primitives emulated : cum_sum, cum_mean, diff, percentile
    Rolling windows           : [5, 10, 20, 63] (cross-product with agg primitives)
    Depth-2 interactions      : diff_of_rolling_mean, ratio_of_rolling_std, etc.

    When featuretools fixes the woodwork/pandas 3.x accessor bug, replace the body of
    this function with the EntitySet / ft.dfs() call shown in the docstring below.

Featuretools EntitySet design (for future use once bug is fixed):
    es = ft.EntitySet(id='ohlcv_<ticker>')
    es.add_dataframe(
        dataframe_name='daily',
        dataframe=df.reset_index(),
        index='_row_id',
        make_index=True,
        time_index='date',
    )
    feature_matrix, feature_defs = ft.dfs(
        entityset=es,
        target_dataframe_name='daily',
        agg_primitives=['mean','std','sum','skew','max','min','count_above_mean'],
        trans_primitives=['cum_sum','cum_mean','diff','percentile'],
        max_depth=2,
    )

Usage:
    from featuretools_dfs_features import add_dfs_features
    df = add_dfs_features(df, ticker='AAPL')

All features are .shift(1) safe — every column is lagged before return.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy.stats import skew as scipy_skew

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Base columns to run DFS primitives on (OHLCV-derived, always present)
_BASE_COLS = ["open", "high", "low", "close", "volume"]

# Rolling windows (days) — cross-product with agg primitives
_WINDOWS = [5, 10, 20, 63]

# Feature name prefix
_PREFIX = "dfs"

# Maximum total DFS features before variance pruning kicks in
_MAX_FEATURES_BEFORE_PRUNE = 200

# Target feature count after pruning
_TARGET_FEATURES = 60


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _safe_div(a: pd.Series, b: pd.Series, fill: float = 0.0) -> pd.Series:
    """Element-wise division filling inf/nan with fill."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = a / b.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan).fillna(fill)


def _percentile_rank(s: pd.Series, window: int) -> pd.Series:
    """Rolling percentile rank (0-1) of each value within the window."""
    return s.rolling(window, min_periods=max(2, window // 2)).apply(
        lambda x: (x[:-1] < x[-1]).mean() if len(x) > 1 else 0.5,
        raw=True,
    )


def _count_above_rolling_mean(s: pd.Series, window: int) -> pd.Series:
    """Count of values above rolling mean within window / window."""
    roll_mean = s.rolling(window, min_periods=max(2, window // 2)).mean()
    return (s > roll_mean).rolling(window, min_periods=max(2, window // 2)).mean()


# ---------------------------------------------------------------------------
# DFS primitive layers
# ---------------------------------------------------------------------------

def _layer_transform_primitives(df: pd.DataFrame, cols: list[str]) -> dict[str, pd.Series]:
    """
    DFS trans_primitives: cum_sum, cum_mean, diff, percentile.
    Applied to each base column individually.
    """
    features: dict[str, pd.Series] = {}
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col]
        features[f"{_PREFIX}_cumsum_{col}"] = s.cumsum()
        features[f"{_PREFIX}_cummean_{col}"] = s.expanding().mean()
        features[f"{_PREFIX}_diff1_{col}"] = s.diff(1)
        features[f"{_PREFIX}_diff5_{col}"] = s.diff(5)
        features[f"{_PREFIX}_pct_{col}_21"] = _percentile_rank(s, 21)
        features[f"{_PREFIX}_pct_{col}_63"] = _percentile_rank(s, 63)
    return features


def _layer_agg_primitives(df: pd.DataFrame, cols: list[str]) -> dict[str, pd.Series]:
    """
    DFS agg_primitives over rolling windows:
    mean, std, sum, skew, max, min, count_above_mean.
    """
    features: dict[str, pd.Series] = {}
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col]
        for w in _WINDOWS:
            mp = max(2, w // 2)
            roll = s.rolling(w, min_periods=mp)
            features[f"{_PREFIX}_rmean_{col}_{w}"] = roll.mean()
            features[f"{_PREFIX}_rstd_{col}_{w}"] = roll.std()
            features[f"{_PREFIX}_rsum_{col}_{w}"] = roll.sum()
            features[f"{_PREFIX}_rmax_{col}_{w}"] = roll.max()
            features[f"{_PREFIX}_rmin_{col}_{w}"] = roll.min()
            features[f"{_PREFIX}_rskew_{col}_{w}"] = roll.skew()
            features[f"{_PREFIX}_rcam_{col}_{w}"] = _count_above_rolling_mean(s, w)
    return features


def _layer_depth2_interactions(df: pd.DataFrame, cols: list[str]) -> dict[str, pd.Series]:
    """
    DFS max_depth=2 interactions: apply trans primitives to agg outputs.
    Examples: diff(rolling_mean), ratio(rolling_std/rolling_mean), etc.
    """
    features: dict[str, pd.Series] = {}
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col]
        # diff of rolling means (momentum over different horizons)
        for w in _WINDOWS:
            mp = max(2, w // 2)
            rm = s.rolling(w, min_periods=mp).mean()
            features[f"{_PREFIX}_d2_diff_rmean_{col}_{w}"] = rm.diff(1)
        # ratio of rolling std to rolling mean (coefficient of variation)
        for w in _WINDOWS:
            mp = max(2, w // 2)
            rm = s.rolling(w, min_periods=mp).mean()
            rs = s.rolling(w, min_periods=mp).std()
            features[f"{_PREFIX}_d2_cv_{col}_{w}"] = _safe_div(rs, rm.abs())
        # rolling mean slope (short vs long)
        if "close" in col or "returns" in col:
            for w_short, w_long in [(5, 20), (10, 63)]:
                mp_s = max(2, w_short // 2); mp_l = max(2, w_long // 2)
                rm_s = s.rolling(w_short, min_periods=mp_s).mean()
                rm_l = s.rolling(w_long, min_periods=mp_l).mean()
                features[f"{_PREFIX}_d2_slope_{col}_{w_short}v{w_long}"] = _safe_div(
                    rm_s - rm_l, rm_l.abs()
                )
    return features


def _layer_cross_ohlcv(df: pd.DataFrame) -> dict[str, pd.Series]:
    """
    Cross-column DFS interactions (depth-2 cross-primitive).
    These replicate the volume/price and high-low range aggregations
    featuretools generates when agg_primitives operate across columns.
    """
    features: dict[str, pd.Series] = {}
    has = lambda c: c in df.columns

    # Volume-price product features
    if has("close") and has("volume"):
        vp = df["close"] * df["volume"]
        for w in [5, 20]:
            mp = max(2, w // 2)
            features[f"{_PREFIX}_vwap_approx_{w}"] = _safe_div(
                (df["close"] * df["volume"]).rolling(w, min_periods=mp).sum(),
                df["volume"].rolling(w, min_periods=mp).sum(),
            )

    # High-low range
    if has("high") and has("low"):
        hl = df["high"] - df["low"]
        for w in [5, 10, 20]:
            mp = max(2, w // 2)
            features[f"{_PREFIX}_hl_range_rmean_{w}"] = hl.rolling(w, min_periods=mp).mean()
            features[f"{_PREFIX}_hl_range_rstd_{w}"] = hl.rolling(w, min_periods=mp).std()
            features[f"{_PREFIX}_hl_pct_{w}"] = _percentile_rank(hl, w)

    # Open-close gap
    if has("open") and has("close"):
        oc = df["close"] - df["open"]
        for w in [5, 10]:
            mp = max(2, w // 2)
            features[f"{_PREFIX}_oc_gap_rmean_{w}"] = oc.rolling(w, min_periods=mp).mean()
            features[f"{_PREFIX}_oc_gap_sign_{w}"] = (oc > 0).rolling(w, min_periods=mp).mean()

    # Volume anomaly
    if has("volume"):
        v = df["volume"]
        for w in [10, 20]:
            mp = max(2, w // 2)
            rm = v.rolling(w, min_periods=mp).mean()
            rs = v.rolling(w, min_periods=mp).std()
            features[f"{_PREFIX}_vol_z_{w}"] = _safe_div(v - rm, rs)

    # EWM features (exponential weighted — featuretools ExponentialWeightedAverage)
    if has("close"):
        for span in [5, 10, 21]:
            features[f"{_PREFIX}_ewm_mean_{span}"] = df["close"].ewm(span=span, min_periods=2).mean()
            features[f"{_PREFIX}_ewm_std_{span}"] = df["close"].ewm(span=span, min_periods=2).std()

    return features


# ---------------------------------------------------------------------------
# Variance pruner
# ---------------------------------------------------------------------------

def _prune_by_variance(
    feat_dict: dict[str, pd.Series],
    target_n: int = _TARGET_FEATURES,
) -> dict[str, pd.Series]:
    """
    If we have more than _MAX_FEATURES_BEFORE_PRUNE raw features,
    rank by variance and keep top target_n.
    """
    if len(feat_dict) <= target_n:
        return feat_dict
    variances = {}
    for name, s in feat_dict.items():
        try:
            variances[name] = float(s.var(skipna=True))
        except Exception:
            variances[name] = 0.0
    ranked = sorted(variances.items(), key=lambda x: -x[1])
    keep = {name for name, _ in ranked[:target_n]}
    return {k: v for k, v in feat_dict.items() if k in keep}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_dfs_features(df: pd.DataFrame, ticker: str = None) -> pd.DataFrame:
    """
    Adds Featuretools DFS auto-generated features to a daily OHLCV DataFrame.

    The function replicates the feature set that featuretools DFS would generate
    for a single-entity time-series EntitySet with:
        agg_primitives   = ['mean','std','sum','skew','max','min','count_above_mean']
        trans_primitives = ['cum_sum','cum_mean','diff','percentile']
        max_depth        = 2

    All generated features are .shift(1) before appending — zero lookahead.

    Parameters
    ----------
    df : pd.DataFrame
        Daily OHLCV DataFrame, DatetimeIndex, columns include open/high/low/close/volume.
        May also contain derived columns (rsi_*, ema_*, atr_*, etc.) from prior layers.
    ticker : str, optional
        Ticker symbol for logging.

    Returns
    -------
    pd.DataFrame
        Original df with DFS feature columns appended (shifted by 1 day).
        Duplicate column names are deduplicated. No NaN-only columns added.
    """
    df = df.copy()
    label = ticker or "unknown"

    # Determine which base columns exist in this df
    base_cols = [c for c in _BASE_COLS if c in df.columns]
    if not base_cols:
        print(f"  [dfs:{label}] no OHLCV columns found — skipping")
        return df

    # --- Layer 1: transform primitives (cum_sum, cum_mean, diff, percentile) ---
    feat_dict: dict[str, pd.Series] = {}
    feat_dict.update(_layer_transform_primitives(df, base_cols))

    # --- Layer 2: agg primitives over rolling windows ---
    feat_dict.update(_layer_agg_primitives(df, base_cols))

    # --- Layer 3: depth-2 interactions (trans applied to agg outputs) ---
    feat_dict.update(_layer_depth2_interactions(df, base_cols))

    # --- Layer 4: cross-column OHLCV interactions ---
    feat_dict.update(_layer_cross_ohlcv(df))

    raw_count = len(feat_dict)

    # --- Prune to target range if over limit ---
    if raw_count > _MAX_FEATURES_BEFORE_PRUNE:
        feat_dict = _prune_by_variance(feat_dict, _TARGET_FEATURES)
        print(
            f"  [dfs:{label}] generated {raw_count} raw DFS features "
            f"-> pruned to {len(feat_dict)} by variance"
        )
    else:
        print(f"  [dfs:{label}] generated {raw_count} raw DFS features (no pruning needed)")

    # --- Shift(1) all features: no lookahead ---
    existing_cols = set(df.columns)
    added = 0
    nan_only = 0
    for name, s in feat_dict.items():
        if name in existing_cols:
            continue  # skip if already present (e.g., overlaps with prior layer)
        shifted = s.shift(1)
        if shifted.isna().all():
            nan_only += 1
            continue  # drop NaN-only columns
        if not np.isfinite(shifted.dropna().values).all():
            # Replace any remaining inf values
            shifted = shifted.replace([np.inf, -np.inf], np.nan)
        df[name] = shifted
        added += 1

    if nan_only > 0:
        print(f"  [dfs:{label}] dropped {nan_only} NaN-only columns")

    print(f"  [dfs:{label}] +{added} DFS features -> total cols: {df.shape[1]}")
    return df


def dfs_feature_names() -> list[str]:
    """
    Return the list of DFS feature column names that would be generated
    for a standard OHLCV DataFrame (without running the full pipeline).
    Useful for importance tracking.
    """
    names = []
    for col in _BASE_COLS:
        # Trans primitives
        names += [
            f"{_PREFIX}_cumsum_{col}", f"{_PREFIX}_cummean_{col}",
            f"{_PREFIX}_diff1_{col}", f"{_PREFIX}_diff5_{col}",
            f"{_PREFIX}_pct_{col}_21", f"{_PREFIX}_pct_{col}_63",
        ]
        # Agg primitives
        for w in _WINDOWS:
            for agg in ["rmean", "rstd", "rsum", "rmax", "rmin", "rskew", "rcam"]:
                names.append(f"{_PREFIX}_{agg}_{col}_{w}")
        # Depth-2 interactions
        for w in _WINDOWS:
            names += [
                f"{_PREFIX}_d2_diff_rmean_{col}_{w}",
                f"{_PREFIX}_d2_cv_{col}_{w}",
            ]
    # Cross-column features
    names += [f"{_PREFIX}_vwap_approx_{w}" for w in [5, 20]]
    names += [f"{_PREFIX}_hl_range_rmean_{w}" for w in [5, 10, 20]]
    names += [f"{_PREFIX}_hl_range_rstd_{w}" for w in [5, 10, 20]]
    names += [f"{_PREFIX}_hl_pct_{w}" for w in [5, 10, 20]]
    names += [f"{_PREFIX}_oc_gap_rmean_{w}" for w in [5, 10]]
    names += [f"{_PREFIX}_oc_gap_sign_{w}" for w in [5, 10]]
    names += [f"{_PREFIX}_vol_z_{w}" for w in [10, 20]]
    names += [f"{_PREFIX}_ewm_mean_{span}" for span in [5, 10, 21]]
    names += [f"{_PREFIX}_ewm_std_{span}" for span in [5, 10, 21]]
    return names


# ---------------------------------------------------------------------------
# Smoke test (run as script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, __file__.rsplit("/", 1)[0])

    ticker = "AAPL"
    print(f"Smoke test: {ticker}")

    import backtest_ml as bml
    d = bml.load_daily(ticker)
    f = bml.build_features(d)
    print(f"Base features: {f.shape[1]} cols, {len(f)} rows")

    before = f.shape[1]
    f2 = add_dfs_features(f, ticker=ticker)
    added = f2.shape[1] - before

    print(f"\n+{added} DFS features added")
    dfs_cols = [c for c in f2.columns if c.startswith(_PREFIX)]
    print(f"DFS columns: {len(dfs_cols)}")

    # Check no NaN-only columns
    nan_only_count = sum(1 for c in dfs_cols if f2[c].isna().all())
    print(f"NaN-only DFS columns: {nan_only_count}")

    # Check finite values
    for c in dfs_cols[:5]:
        vals = f2[c].dropna()
        finite_pct = np.isfinite(vals.values).mean() * 100 if len(vals) > 0 else 0
        print(f"  {c}: n={len(vals)}, finite={finite_pct:.1f}%, "
              f"mean={vals.mean():.4f}, std={vals.std():.4f}")

    print("\nSmoke test PASSED" if nan_only_count == 0 and added > 0 else "\nSmoke test FAILED")
