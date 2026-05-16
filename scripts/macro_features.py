"""
macro_features.py
=================
Daily macro context features for the S&P 500 ML pipeline.

All features are ticker-independent and derived solely from freely available
market data via yfinance (no API key required).

POINT-IN-TIME SAFETY
--------------------
Every feature column is .shift(1) so the value visible at bar t uses only
data through t-1.  This matches the convention in cross_sectional_features.py
and intraday_features.py.

CACHING
-------
Raw OHLCV data from yfinance is cached to:
    AI-Tools/s&p500-ticker-mastery/cache/macro_data.parquet

On subsequent calls the cache is loaded if it is <24 hours old.  Pass
force_refresh=True to add_macro_features() to bypass the staleness check.

FALLBACK
--------
If yfinance fails for any symbol the module logs a warning and skips features
derived from that symbol.  If yfinance is entirely unreachable and SPY data is
needed, the module tries to fall back to the 1Min parquets already on disk.

USAGE
-----
    from macro_features import add_macro_features

    daily_df = add_macro_features(daily_df)          # adds ~25-30 columns
    daily_df = add_macro_features(daily_df, force_refresh=True)  # skip cache
"""

from __future__ import annotations

import logging
import os
import time
import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = (
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools/s&p500-ticker-mastery"
)
CACHE_DIR = os.path.join(_PROJECT_ROOT, "cache")
CACHE_PATH = os.path.join(CACHE_DIR, "macro_data.parquet")

# 1Min fallback root (used only when yfinance is entirely unreachable for SPY)
_1MIN_ROOT = (
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/claudes test/data/timeframes"
    "/S&P500 5 Year Historical Data/Minutes TimeFrames/1Min_merged"
)

# ---------------------------------------------------------------------------
# Symbols
# ---------------------------------------------------------------------------
_MACRO_SYMBOLS: Dict[str, str] = {
    # Core macro
    "^VIX":       "vix",
    "^TNX":       "t10y",
    "^TYX":       "t30y",
    "DX-Y.NYB":   "dxy",
    "GC=F":       "gold",
    "CL=F":       "oil",
    "BTC-USD":    "btc",
    # Indices
    "^GSPC":      "spy",      # proxy for S&P 500 index
    "^IXIC":      "nasdaq",
    # Sector ETFs
    "XLF":        "xlf",
    "XLK":        "xlk",
    "XLE":        "xle",
    "XLV":        "xlv",
    "XLI":        "xli",
    "XLY":        "xly",
    "XLP":        "xlp",
    "XLU":        "xlu",
    "XLB":        "xlb",
    "XLRE":       "xlre",
}

_SECTOR_PREFIXES = ["xlf", "xlk", "xle", "xlv", "xli", "xly", "xlp", "xlu", "xlb", "xlre"]

# In-process singleton to avoid re-fetching on repeated calls in the same process
_MACRO_RAW_SINGLETON: Optional[Dict[str, pd.Series]] = None

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_is_fresh() -> bool:
    """Return True if the on-disk cache exists and is <24 h old."""
    if not os.path.exists(CACHE_PATH):
        return False
    age_s = time.time() - os.path.getmtime(CACHE_PATH)
    return age_s < 86_400  # 24 hours


def _load_cache() -> Optional[Dict[str, pd.Series]]:
    """Load the parquet cache and return a dict of {prefix: close_series}."""
    try:
        df = pd.read_parquet(CACHE_PATH)
        result: Dict[str, pd.Series] = {}
        for col in df.columns:
            result[col] = df[col].dropna()
        logger.info("macro_features: loaded %d series from cache %s", len(result), CACHE_PATH)
        return result
    except Exception as exc:
        logger.warning("macro_features: cache read failed (%s), will re-fetch", exc)
        return None


def _save_cache(data: Dict[str, pd.Series]) -> None:
    """Persist the close series dict to parquet."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        df = pd.DataFrame(data)
        df.to_parquet(CACHE_PATH)
        logger.info("macro_features: saved cache to %s", CACHE_PATH)
    except Exception as exc:
        logger.warning("macro_features: cache write failed (%s)", exc)


# ---------------------------------------------------------------------------
# yfinance fetch
# ---------------------------------------------------------------------------

def _fetch_yfinance(start: str = "2020-01-01") -> Dict[str, pd.Series]:
    """
    Download daily close prices for all macro symbols via yfinance.

    Returns a dict mapping prefix (e.g. 'vix') -> daily close pd.Series
    with a UTC DatetimeIndex.  Symbols that fail are skipped with a warning.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.error("macro_features: yfinance not installed — run: uv pip install yfinance")
        return {}

    symbols = list(_MACRO_SYMBOLS.keys())
    result: Dict[str, pd.Series] = {}

    # Batch download — faster than one-by-one
    try:
        raw = yf.download(
            symbols,
            start=start,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        # yfinance returns MultiIndex columns (field, ticker) in batch mode
        if isinstance(raw.columns, pd.MultiIndex):
            close_df = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw.xs("Close", level=0, axis=1)
        else:
            close_df = raw[["Close"]] if "Close" in raw.columns else raw

        # Normalise to UTC
        if close_df.index.tz is None:
            close_df.index = close_df.index.tz_localize("UTC")
        else:
            close_df.index = close_df.index.tz_convert("UTC")

        for ticker, prefix in _MACRO_SYMBOLS.items():
            if ticker in close_df.columns:
                s = close_df[ticker].dropna()
                if len(s) > 0:
                    result[prefix] = s
                    logger.debug("macro_features: fetched %s -> %s (%d rows)", ticker, prefix, len(s))
                else:
                    logger.warning("macro_features: %s returned empty series, skipping", ticker)
            else:
                logger.warning("macro_features: %s not in batch response, skipping", ticker)

    except Exception as exc:
        logger.warning("macro_features: batch download failed (%s), falling back to per-symbol", exc)
        # Per-symbol fallback
        for ticker, prefix in _MACRO_SYMBOLS.items():
            try:
                tk = yf.Ticker(ticker)
                hist = tk.history(start=start, auto_adjust=True)
                if hist.empty:
                    logger.warning("macro_features: %s returned empty, skipping", ticker)
                    continue
                s = hist["Close"].dropna()
                if s.index.tz is None:
                    s.index = s.index.tz_localize("UTC")
                else:
                    s.index = s.index.tz_convert("UTC")
                result[prefix] = s
            except Exception as sym_exc:
                logger.warning("macro_features: failed to fetch %s (%s), skipping", ticker, sym_exc)

    return result


# ---------------------------------------------------------------------------
# 1Min parquet fallback for SPY (only if yfinance entirely fails)
# ---------------------------------------------------------------------------

def _spy_from_1min() -> Optional[pd.Series]:
    """
    Build a daily SPY close proxy from the on-disk 1Min parquets.
    Used only when yfinance cannot reach the network at all.
    """
    spy_path = os.path.join(_1MIN_ROOT, "SPY.parquet")
    if not os.path.exists(spy_path):
        logger.warning("macro_features: SPY 1Min parquet not found at %s", spy_path)
        return None
    try:
        df = pd.read_parquet(spy_path, columns=["close"])
        if df.index.tz is None:
            df.index = df.index.tz_localize("America/New_York")
        df.index = df.index.tz_convert("UTC")
        # Resample to daily close (last bar of each session)
        daily = df["close"].resample("1D").last().dropna()
        logger.info("macro_features: SPY fallback from 1Min parquet (%d rows)", len(daily))
        return daily
    except Exception as exc:
        logger.warning("macro_features: SPY 1Min fallback failed (%s)", exc)
        return None


# ---------------------------------------------------------------------------
# Module-level data loader (called once, cached in-process)
# ---------------------------------------------------------------------------

def _load_macro_data(force_refresh: bool = False) -> Dict[str, pd.Series]:
    """
    Return the macro raw close price dict.  Resolution order:
      1. In-process singleton (fastest)
      2. On-disk parquet cache if <24 h old (fast)
      3. yfinance live fetch + write cache
      4. Partial result if some symbols failed
    """
    global _MACRO_RAW_SINGLETON

    if not force_refresh and _MACRO_RAW_SINGLETON is not None:
        return _MACRO_RAW_SINGLETON

    if not force_refresh and _cache_is_fresh():
        data = _load_cache()
        if data:
            _MACRO_RAW_SINGLETON = data
            return data

    logger.info("macro_features: fetching macro data from yfinance ...")
    data = _fetch_yfinance()

    # If SPY missing entirely try 1Min fallback
    if "spy" not in data:
        spy_series = _spy_from_1min()
        if spy_series is not None:
            data["spy"] = spy_series

    if data:
        _save_cache(data)

    _MACRO_RAW_SINGLETON = data
    return data


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------

def _reindex_to(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """
    Forward-fill a macro series onto the target daily index.
    Handles weekends / holidays where macro data may have gaps.
    """
    if series.index.tz is None:
        series = series.copy()
        series.index = series.index.tz_localize("UTC")
    # Normalise both to date-only UTC midnight for alignment
    series = series.copy()
    series.index = series.index.normalize()
    target = index.normalize()
    combined = series.reindex(target.union(series.index)).sort_index()
    combined = combined.ffill()
    return combined.reindex(target)


def _rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window, min_periods=window // 2).mean()
    sigma = s.rolling(window, min_periods=window // 2).std()
    return (s - mu) / sigma.replace(0, np.nan)


def _pct_change_n(s: pd.Series, n: int) -> pd.Series:
    return s.pct_change(n)


def _rolling_vol(s: pd.Series, window: int) -> pd.Series:
    """Annualised rolling volatility of log returns."""
    lr = np.log(s / s.shift(1))
    return lr.rolling(window, min_periods=window // 2).std() * np.sqrt(252)


def _atr_pct(s: pd.Series, window: int = 14) -> pd.Series:
    """
    Simplified ATR% using close-to-close (we only have close from yfinance batch).
    Uses rolling std of log returns scaled to the close.
    """
    lr = np.log(s / s.shift(1))
    atr_approx = lr.rolling(window, min_periods=window // 2).std() * s
    return (atr_approx / s).replace([np.inf, -np.inf], np.nan)


def _above_sma(s: pd.Series, window: int = 50) -> pd.Series:
    sma = s.rolling(window, min_periods=window // 2).mean()
    return (s > sma).astype(float)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_macro_features(
    daily_df: pd.DataFrame,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Add daily macro context features to *daily_df* (ticker-independent).

    Parameters
    ----------
    daily_df : pd.DataFrame
        Daily price DataFrame with a UTC-aware DatetimeIndex.
    force_refresh : bool
        If True, bypass both the in-process singleton and the on-disk cache
        and re-fetch from yfinance.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with ~25-30 new columns appended.
        All columns are .shift(1) so bar t uses only data through t-1.
    """
    out = daily_df.copy()
    idx = out.index  # UTC daily DatetimeIndex

    # Ensure UTC
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
        out.index = idx

    raw = _load_macro_data(force_refresh=force_refresh)

    if not raw:
        logger.error("macro_features: no macro data available — returning df unchanged")
        return out

    def _get(prefix: str) -> Optional[pd.Series]:
        """Return a macro close series reindexed to idx, or None if unavailable."""
        if prefix not in raw:
            return None
        return _reindex_to(raw[prefix], idx)

    # ------------------------------------------------------------------
    # VIX features
    # ------------------------------------------------------------------
    vix = _get("vix")
    if vix is not None:
        out["vix_close"]          = vix.shift(1)
        out["vix_close_zscore_60d"] = _rolling_zscore(vix, 60).shift(1)
        out["vix_above_25"]       = (vix > 25).astype(float).shift(1)
        out["vix_above_30"]       = (vix > 30).astype(float).shift(1)
        # VIX spike: current VIX vs 5-day trailing min
        vix_min5 = vix.rolling(5, min_periods=2).min()
        out["vix_spike_5d"]       = ((vix - vix_min5) / vix_min5.replace(0, np.nan)).shift(1)
    else:
        logger.warning("macro_features: VIX unavailable, skipping vix_* features")

    # ------------------------------------------------------------------
    # Yield features
    # ------------------------------------------------------------------
    t10y = _get("t10y")
    t30y = _get("t30y")

    if t10y is not None:
        out["t10y_close"]       = t10y.shift(1)
        out["t10y_change_5d"]   = _pct_change_n(t10y, 5).shift(1)
    else:
        logger.warning("macro_features: ^TNX unavailable, skipping t10y_* features")

    if t10y is not None and t30y is not None:
        spread = t30y - t10y
        out["yield_curve_30y_10y_spread"] = spread.shift(1)
        out["yield_inverted_flag"]        = (spread < 0).astype(float).shift(1)
    else:
        logger.warning("macro_features: yield curve features skipped (missing TNX or TYX)")

    # ------------------------------------------------------------------
    # Dollar / commodity features
    # ------------------------------------------------------------------
    dxy = _get("dxy")
    if dxy is not None:
        out["dxy_close_change_5d"] = _pct_change_n(dxy, 5).shift(1)
    else:
        logger.warning("macro_features: DXY unavailable, skipping dxy_* features")

    gold = _get("gold")
    if gold is not None:
        out["gold_change_5d"] = _pct_change_n(gold, 5).shift(1)
    else:
        logger.warning("macro_features: Gold unavailable, skipping gold_* features")

    oil = _get("oil")
    if oil is not None:
        out["oil_change_5d"] = _pct_change_n(oil, 5).shift(1)
        out["oil_vol_30d"]   = _rolling_vol(oil, 30).shift(1)
    else:
        logger.warning("macro_features: Oil unavailable, skipping oil_* features")

    # ------------------------------------------------------------------
    # Crypto (risk-on proxy)
    # ------------------------------------------------------------------
    btc = _get("btc")
    if btc is not None:
        out["btc_return_5d"] = _pct_change_n(btc, 5).shift(1)
        out["btc_vol_30d"]   = _rolling_vol(btc, 30).shift(1)
    else:
        logger.warning("macro_features: BTC unavailable, skipping btc_* features")

    # ------------------------------------------------------------------
    # S&P 500 / index features
    # ------------------------------------------------------------------
    spy_idx = _get("spy")
    nasdaq  = _get("nasdaq")

    if spy_idx is not None:
        out["spy_close"]       = spy_idx.shift(1)
        out["spy_return_5d"]   = _pct_change_n(spy_idx, 5).shift(1)
        out["spy_return_21d"]  = _pct_change_n(spy_idx, 21).shift(1)
        out["spy_atr_pct_14"]  = _atr_pct(spy_idx, 14).shift(1)
    else:
        logger.warning("macro_features: SPY/GSPC unavailable, skipping spy_* features")

    if spy_idx is not None and nasdaq is not None:
        ratio = nasdaq / spy_idx.replace(0, np.nan)
        out["nasdaq_spy_ratio_change"] = _pct_change_n(ratio, 5).shift(1)
    else:
        logger.warning("macro_features: NASDAQ/SPY ratio skipped (missing series)")

    # ------------------------------------------------------------------
    # Sector ETF features
    # ------------------------------------------------------------------
    for prefix in _SECTOR_PREFIXES:
        sec = _get(prefix)
        if sec is None:
            logger.warning("macro_features: sector %s unavailable, skipping", prefix)
            continue
        out[f"{prefix}_return_5d"]    = _pct_change_n(sec, 5).shift(1)
        out[f"{prefix}_above_sma_50"] = _above_sma(sec, 50).shift(1)

    logger.info(
        "macro_features: added %d new columns",
        len([c for c in out.columns if c not in daily_df.columns]),
    )
    return out


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    import pandas as pd

    dates = pd.date_range("2024-01-01", "2024-12-31", freq="B", tz="UTC")
    df = pd.DataFrame({"close": 100.0}, index=dates)

    out = add_macro_features(df.copy())

    new = [c for c in out.columns if c not in df.columns]
    print(f"\n+{len(new)} macro features")

    for c in new[:15]:
        if pd.api.types.is_numeric_dtype(out[c]):
            non_zero_pct = (out[c].notna() & (out[c] != 0)).mean() * 100
            last_val = out[c].iloc[-1]
            last_str = f"{last_val:.4f}" if pd.notna(last_val) else "NaN"
            print(f"  {c}: {non_zero_pct:.0f}% non-zero, last={last_str}")

    print("\nAll feature names:")
    for c in new:
        print(f"  {c}")
