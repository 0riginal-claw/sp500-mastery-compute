"""
cross_sectional_features.py
===========================
Cross-sectional and macro-relative features for the S&P 500 ML pipeline.

Each ticker's local features (RSI, MACD, ATR ...) live in backtest_xgb.py /
generate_mastery_file_daily.py.  This module adds the context a ticker cannot
compute alone:

    A. SPY-relative momentum & beta / correlation
    B. VIX-proxy regime (cross-sectional realized vol of the 502-ticker universe)
    C. Sector context (mean sector return, relative return, intra-sector percentile)
    D. Cross-sectional rank features (return, volume, volatility across all 502)
    E. Calendar cyclical encoding (DOW and month sin/cos)

SURVIVORSHIP-BIAS CAVEAT
------------------------
Universe aggregates are computed from the 502 tickers present in the data
snapshot (2021-04-21 to 2026-04-21).  This is a *static* list of survivors --
companies that were present throughout the entire window.  VIX proxy, sector
returns, and cross-sectional ranks will therefore be computed on a slightly
survivorship-biased universe.  This does NOT affect per-ticker point-in-time
safety (no future leakage within any single ticker's own history), but it does
mean the universe factors may slightly over-represent stable, large-cap
companies relative to the true live S&P 500 composition at each historical date.

POINT-IN-TIME SAFETY
--------------------
All features that reference a ticker's own series use .shift(1) so bar t's
feature uses data through t-1.  Universe aggregates (SPY proxy, sector returns,
VIX proxy, cross-sectional ranks) are computed from the previous day's close
prices -- they are point-in-time safe when joined at bar t.

CACHING
-------
precompute_universe_aggregates() is expensive (~60-120 s for 502 parquets on
first run).  It caches to:
    AI-Tools/s&p500-ticker-mastery/cache/universe_agg.parquet  (marker file)
    AI-Tools/s&p500-ticker-mastery/cache/universe_agg_*.parquet  (data files)
On subsequent calls (same process or new process) the cache is loaded instead.
Pass force_recompute=True to override.

Usage
-----
    from cross_sectional_features import (
        precompute_universe_aggregates,
        add_cross_sectional_features,
    )

    agg  = precompute_universe_aggregates()            # once per process
    df_e = add_cross_sectional_features(daily_df, 'AAPL', agg)
"""

from __future__ import annotations

import glob
import json
import logging
import math
import os
import shutil
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
DATA_ROOT = (
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/claudes test/data/timeframes"
    "/S&P500 5 Year Historical Data/Minutes TimeFrames/1Min_merged"
)
CACHE_DIR = (
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools/s&p500-ticker-mastery/cache"
)
CACHE_PATH = os.path.join(CACHE_DIR, "universe_agg.parquet")
MANIFEST_PATH = os.path.join(CACHE_DIR, "universe_agg_manifest.json")

# /tmp daily cache -- 502 small per-ticker daily parquets (~50 KB each, 25 MB total)
# Written once; subsequent recomputes skip Drive I/O entirely.
TMP_DAILY_CACHE = "/tmp/sp500_daily_cache"

# In-process singleton -- avoids re-reading the cache on repeated calls
_UNIVERSE_AGG_SINGLETON: Optional[Dict] = None

# ---------------------------------------------------------------------------
# Sector map -- GICS sectors for the ~100 most-liquid S&P 500 names.
# Tickers not in this map default to "OTHER".
# ---------------------------------------------------------------------------
SECTOR_MAP: Dict[str, str] = {
    # Information Technology
    "AAPL": "TECH", "MSFT": "TECH", "NVDA": "TECH", "AVGO": "TECH",
    "ORCL": "TECH", "CRM": "TECH", "ACN": "TECH", "CSCO": "TECH",
    "IBM": "TECH", "ADBE": "TECH", "QCOM": "TECH", "TXN": "TECH",
    "AMAT": "TECH", "AMD": "TECH", "INTC": "TECH", "MU": "TECH",
    "LRCX": "TECH", "KLAC": "TECH", "SNPS": "TECH", "CDNS": "TECH",
    "HPQ": "TECH", "KEYS": "TECH", "FTNT": "TECH", "PANW": "TECH",
    "CRWD": "TECH", "NOW": "TECH", "ADSK": "TECH", "ANSS": "TECH",
    "IT": "TECH", "EPAM": "TECH", "ZBRA": "TECH", "AKAM": "TECH",
    "FFIV": "TECH", "JNPR": "TECH", "GLW": "TECH", "STX": "TECH",
    "WDC": "TECH", "HPE": "TECH", "NTAP": "TECH", "CDW": "TECH",
    "GEN": "TECH", "CTSH": "TECH",
    # Communication Services
    "GOOG": "COMMS", "GOOGL": "COMMS", "META": "COMMS", "NFLX": "COMMS",
    "DIS": "COMMS", "CMCSA": "COMMS", "CHTR": "COMMS", "T": "COMMS",
    "VZ": "COMMS", "TMUS": "COMMS", "ATVI": "COMMS", "EA": "COMMS",
    "OMC": "COMMS", "IPG": "COMMS", "LYV": "COMMS", "WBD": "COMMS",
    "PARA": "COMMS", "FOXA": "COMMS", "FOX": "COMMS",
    # Consumer Discretionary
    "AMZN": "CONS_DISC", "TSLA": "CONS_DISC", "HD": "CONS_DISC",
    "MCD": "CONS_DISC", "NKE": "CONS_DISC", "SBUX": "CONS_DISC",
    "LOW": "CONS_DISC", "TJX": "CONS_DISC", "BKNG": "CONS_DISC",
    "MAR": "CONS_DISC", "HLT": "CONS_DISC", "YUM": "CONS_DISC",
    "ABNB": "CONS_DISC", "GM": "CONS_DISC", "F": "CONS_DISC",
    "ORLY": "CONS_DISC", "AZO": "CONS_DISC", "ROST": "CONS_DISC",
    "DHI": "CONS_DISC", "LEN": "CONS_DISC", "NVR": "CONS_DISC",
    "PHM": "CONS_DISC", "EBAY": "CONS_DISC", "ETSY": "CONS_DISC",
    "POOL": "CONS_DISC", "WYNN": "CONS_DISC", "MGM": "CONS_DISC",
    "CZR": "CONS_DISC", "CCL": "CONS_DISC", "RCL": "CONS_DISC",
    "HAS": "CONS_DISC", "MHK": "CONS_DISC",
    # Consumer Staples
    "PG": "CONS_STAP", "KO": "CONS_STAP", "PEP": "CONS_STAP",
    "WMT": "CONS_STAP", "COST": "CONS_STAP", "PM": "CONS_STAP",
    "MO": "CONS_STAP", "MDLZ": "CONS_STAP", "CL": "CONS_STAP",
    "KMB": "CONS_STAP", "GIS": "CONS_STAP", "K": "CONS_STAP",
    "HRL": "CONS_STAP", "SJM": "CONS_STAP", "CAG": "CONS_STAP",
    "CPB": "CONS_STAP", "CHD": "CONS_STAP", "CLX": "CONS_STAP",
    "EL": "CONS_STAP", "KVUE": "CONS_STAP",
    # Energy
    "XOM": "ENERGY", "CVX": "ENERGY", "COP": "ENERGY", "EOG": "ENERGY",
    "SLB": "ENERGY", "MPC": "ENERGY", "PSX": "ENERGY", "VLO": "ENERGY",
    "OXY": "ENERGY", "PXD": "ENERGY", "HES": "ENERGY", "DVN": "ENERGY",
    "BKR": "ENERGY", "HAL": "ENERGY", "FANG": "ENERGY", "APA": "ENERGY",
    "MRO": "ENERGY", "CTRA": "ENERGY",
    # Financials
    "JPM": "FIN", "BAC": "FIN", "WFC": "FIN", "GS": "FIN",
    "MS": "FIN", "BLK": "FIN", "C": "FIN", "AXP": "FIN",
    "SPGI": "FIN", "MCO": "FIN", "USB": "FIN", "TFC": "FIN",
    "PNC": "FIN", "COF": "FIN", "SCHW": "FIN", "ICE": "FIN",
    "CME": "FIN", "CB": "FIN", "MET": "FIN", "PRU": "FIN",
    "AIG": "FIN", "AFL": "FIN", "ALL": "FIN", "PGR": "FIN",
    "TRV": "FIN", "HIG": "FIN", "L": "FIN", "LNC": "FIN",
    "FITB": "FIN", "RF": "FIN", "HBAN": "FIN", "MTB": "FIN",
    "CFG": "FIN", "KEY": "FIN", "SIVB": "FIN", "CMA": "FIN",
    "ZION": "FIN", "WRB": "FIN", "AIZ": "FIN", "CINF": "FIN",
    "GL": "FIN", "BEN": "FIN", "IVZ": "FIN", "AJG": "FIN",
    "MMC": "FIN", "AON": "FIN", "WTW": "FIN", "ACGL": "FIN",
    "RJF": "FIN", "NDAQ": "FIN", "CBOE": "FIN",
    # Health Care
    "JNJ": "HEALTH", "LLY": "HEALTH", "UNH": "HEALTH", "ABT": "HEALTH",
    "PFE": "HEALTH", "MRK": "HEALTH", "ABBV": "HEALTH", "BMY": "HEALTH",
    "AMGN": "HEALTH", "GILD": "HEALTH", "MDT": "HEALTH", "SYK": "HEALTH",
    "BSX": "HEALTH", "EW": "HEALTH", "ZBH": "HEALTH", "BDX": "HEALTH",
    "ISRG": "HEALTH", "RMD": "HEALTH", "HOLX": "HEALTH", "PODD": "HEALTH",
    "DXCM": "HEALTH", "IDXX": "HEALTH", "IQV": "HEALTH", "CRL": "HEALTH",
    "A": "HEALTH", "MTD": "HEALTH", "WAT": "HEALTH", "TMO": "HEALTH",
    "DHR": "HEALTH", "BAX": "HEALTH", "MCK": "HEALTH", "CVS": "HEALTH",
    "CI": "HEALTH", "HUM": "HEALTH", "CNC": "HEALTH", "MOH": "HEALTH",
    "ANTM": "HEALTH", "AMED": "HEALTH", "HCA": "HEALTH", "THC": "HEALTH",
    "VRTX": "HEALTH", "REGN": "HEALTH", "BIIB": "HEALTH", "ALNY": "HEALTH",
    "MRNA": "HEALTH", "INCY": "HEALTH",
    # Industrials
    "CAT": "INDUS", "DE": "INDUS", "HON": "INDUS", "UPS": "INDUS",
    "RTX": "INDUS", "LMT": "INDUS", "GE": "INDUS", "BA": "INDUS",
    "MMM": "INDUS", "EMR": "INDUS", "ETN": "INDUS", "ITW": "INDUS",
    "GD": "INDUS", "NOC": "INDUS", "LHX": "INDUS", "TDG": "INDUS",
    "HWM": "INDUS", "PWR": "INDUS", "CARR": "INDUS", "OTIS": "INDUS",
    "VRSK": "INDUS", "CBRE": "INDUS", "FDX": "INDUS", "NSC": "INDUS",
    "UNP": "INDUS", "CSX": "INDUS", "DAL": "INDUS", "UAL": "INDUS",
    "AAL": "INDUS", "LUV": "INDUS", "EXPD": "INDUS", "XYL": "INDUS",
    "AME": "INDUS", "ROP": "INDUS", "HUBB": "INDUS", "IR": "INDUS",
    "PH": "INDUS", "ROK": "INDUS", "DOV": "INDUS", "FTV": "INDUS",
    "GNRC": "INDUS", "LDOS": "INDUS", "SAIC": "INDUS", "J": "INDUS",
    "AOS": "INDUS", "SWK": "INDUS", "MAS": "INDUS",
    # Materials
    "LIN": "MATER", "APD": "MATER", "SHW": "MATER", "ECL": "MATER",
    "PPG": "MATER", "NEM": "MATER", "FCX": "MATER", "NUE": "MATER",
    "STLD": "MATER", "PKG": "MATER", "IP": "MATER", "WRK": "MATER",
    "CF": "MATER", "MOS": "MATER", "FMC": "MATER", "CE": "MATER",
    "DD": "MATER", "DOW": "MATER", "LYB": "MATER", "ALB": "MATER",
    "BALL": "MATER", "AVY": "MATER",
    # Real Estate
    "AMT": "REIT", "PLD": "REIT", "CCI": "REIT", "EQIX": "REIT",
    "PSA": "REIT", "DLR": "REIT", "O": "REIT", "WY": "REIT",
    "SPG": "REIT", "VICI": "REIT", "AVB": "REIT", "EQR": "REIT",
    "MAA": "REIT", "UDR": "REIT", "ESS": "REIT", "CPT": "REIT",
    "EXR": "REIT", "CUBE": "REIT", "IRM": "REIT", "CSGP": "REIT",
    "HST": "REIT", "REG": "REIT", "FRT": "REIT", "KIM": "REIT",
    "NNN": "REIT", "MPW": "REIT",
    # Utilities
    "NEE": "UTIL", "DUK": "UTIL", "SO": "UTIL", "D": "UTIL",
    "AEP": "UTIL", "EXC": "UTIL", "SRE": "UTIL", "XEL": "UTIL",
    "ED": "UTIL", "ETR": "UTIL", "EIX": "UTIL", "PPL": "UTIL",
    "FE": "UTIL", "CMS": "UTIL", "NI": "UTIL", "AEE": "UTIL",
    "LNT": "UTIL", "EVRG": "UTIL", "WEC": "UTIL", "AWK": "UTIL",
    "AES": "UTIL", "PEG": "UTIL", "ES": "UTIL", "CNP": "UTIL",
    "DTE": "UTIL", "OGE": "UTIL",
}
DEFAULT_SECTOR = "OTHER"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_daily_close_volume(ticker: str) -> Optional[pd.DataFrame]:
    """Load a single ticker parquet and resample to daily close + volume.

    Mirrors the load_daily() logic in backtest_xgb.py:
      - UTC timestamps -> America/New_York conversion
      - RTH filter: 09:30-15:59 NY
      - Resample to calendar day with left-closed, left-labelled 1D bins
    Returns a DataFrame with DatetimeIndex (UTC midnight) and columns
    ['close', 'volume'].  Returns None on any error.
    """
    path = os.path.join(DATA_ROOT, f"{ticker}.parquet")
    try:
        raw = pd.read_parquet(path, columns=["timestamp", "close", "volume"])
    except Exception as exc:
        logger.debug("Could not load %s: %s", ticker, exc)
        return None

    raw = raw.set_index("timestamp").sort_index()
    et = raw.index.tz_convert("America/New_York")
    rth_mask = (
        ((et.hour > 9) | ((et.hour == 9) & (et.minute >= 30))) & (et.hour < 16)
    )
    rth = raw[rth_mask]
    if rth.empty:
        return None

    daily = (
        rth.resample("1D", closed="left", label="left")
        .agg({"close": "last", "volume": "sum"})
        .dropna(subset=["close"])
    )
    if daily.index.tz is None:
        daily.index = daily.index.tz_localize("UTC")
    return daily


# ---------------------------------------------------------------------------
# /tmp daily cache -- mirrors alt_data_features.py's local-copy pattern
# ---------------------------------------------------------------------------

def _build_tmp_daily_cache(tickers: list[str]) -> None:
    """Resample each ticker's 1-min parquet to daily OHLCV and write to /tmp.

    This converts ~500 MB of Drive parquets into ~25 MB of local daily files.
    Reads each file once through Drive; all subsequent recomputes hit /tmp only.
    Skips tickers already written.
    """
    os.makedirs(TMP_DAILY_CACHE, exist_ok=True)
    missing = [tk for tk in tickers if not os.path.exists(os.path.join(TMP_DAILY_CACHE, f"{tk}.parquet"))]
    if not missing:
        logger.info("All %d tickers already in /tmp daily cache.", len(tickers))
        return

    logger.info("Building /tmp daily cache for %d tickers (Drive read, one-time)...", len(missing))
    t0 = time.time()
    written = 0
    for i, tk in enumerate(missing):
        if i % 50 == 0 and i > 0:
            logger.info("  /tmp cache: %d/%d tickers written (%.0fs elapsed)...", i, len(missing), time.time() - t0)
        df = _load_daily_close_volume(tk)
        if df is not None and not df.empty:
            out_path = os.path.join(TMP_DAILY_CACHE, f"{tk}.parquet")
            df.to_parquet(out_path)
            written += 1
    logger.info("/tmp daily cache built: %d/%d tickers in %.1fs.", written, len(missing), time.time() - t0)


def _load_tmp_daily(ticker: str) -> Optional[pd.DataFrame]:
    """Load a ticker's daily close+volume from /tmp cache (fast, local I/O)."""
    path = os.path.join(TMP_DAILY_CACHE, f"{ticker}.parquet")
    if not os.path.exists(path):
        return None
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        logger.debug("Could not load /tmp daily for %s: %s", ticker, exc)
        return None


# ---------------------------------------------------------------------------
# New multi-file cache helpers
# ---------------------------------------------------------------------------

def _save_manifest_cache(agg: Dict) -> None:
    """Write the v6 multi-file cache layout alongside the legacy files.

    New files (fast targeted loads for v6):
        cache/spy_proxy.parquet        -- spy_return + spy_price (narrow)
        cache/vix_proxy.parquet        -- vix_proxy + vix_regime + vix_pct5d
        cache/sector_returns.parquet   -- sector_returns DataFrame
        cache/xs_ranks.parquet         -- xs_rank_21d_return + xs_rank_volume_21d + xs_rank_volatility_21d
                                          (stacked: each wide DF stored as a group in the same file
                                           but we use a simpler approach: separate parquets)
        cache/universe_agg_manifest.json -- paths + build timestamp
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    ts = pd.Timestamp.now().isoformat()

    # spy_proxy.parquet
    spy_path = os.path.join(CACHE_DIR, "spy_proxy.parquet")
    pd.concat([agg["spy"].rename("spy_return"), agg["spy_price"].rename("spy_price")], axis=1).to_parquet(spy_path)

    # vix_proxy.parquet
    vix_path = os.path.join(CACHE_DIR, "vix_proxy.parquet")
    pd.concat([
        agg["vix_proxy"].rename("vix_proxy"),
        agg["vix_regime"].astype("int8").rename("vix_regime"),
        agg["vix_proxy_pct_change_5d"].rename("vix_proxy_pct_change_5d"),
    ], axis=1).to_parquet(vix_path)

    # sector_returns.parquet
    sect_path = os.path.join(CACHE_DIR, "sector_returns.parquet")
    agg["sector_returns"].to_parquet(sect_path)

    # xs_ranks.parquet -- concatenate the three wide rank panels with a MultiIndex column
    xs_path = os.path.join(CACHE_DIR, "xs_ranks.parquet")
    agg["xs_rank_21d_return"].to_parquet(os.path.join(CACHE_DIR, "xs_rank_21d_return.parquet"))
    agg["xs_rank_volume_21d"].to_parquet(os.path.join(CACHE_DIR, "xs_rank_volume_21d.parquet"))
    agg["xs_rank_volatility_21d"].to_parquet(os.path.join(CACHE_DIR, "xs_rank_volatility_21d.parquet"))
    # Also write a combined file for convenience
    xs_combined = pd.concat(
        [agg["xs_rank_21d_return"], agg["xs_rank_volume_21d"], agg["xs_rank_volatility_21d"]],
        axis=1,
        keys=["xs_rank_21d_return", "xs_rank_volume_21d", "xs_rank_volatility_21d"],
    )
    xs_combined.to_parquet(xs_path)

    manifest = {
        "built_at": ts,
        "files": {
            "spy_proxy": spy_path,
            "vix_proxy": vix_path,
            "sector_returns": sect_path,
            "xs_ranks": xs_path,
            "xs_rank_21d_return": os.path.join(CACHE_DIR, "xs_rank_21d_return.parquet"),
            "xs_rank_volume_21d": os.path.join(CACHE_DIR, "xs_rank_volume_21d.parquet"),
            "xs_rank_volatility_21d": os.path.join(CACHE_DIR, "xs_rank_volatility_21d.parquet"),
            "log_ret_panel": os.path.join(CACHE_DIR, "universe_agg_log_ret_panel.parquet"),
            "sector_rank_5d_df": os.path.join(CACHE_DIR, "universe_agg_sector_rank_5d_df.parquet"),
            "ticker_sectors": os.path.join(CACHE_DIR, "universe_agg_ticker_sectors.parquet"),
        },
    }
    with open(MANIFEST_PATH, "w") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info("Manifest written to %s", MANIFEST_PATH)


def _load_manifest_cache() -> Dict:
    """Load universe aggregates using the manifest (fast path -- pure parquet reads)."""
    with open(MANIFEST_PATH) as fh:
        manifest = json.load(fh)
    files = manifest["files"]

    spy_df = pd.read_parquet(files["spy_proxy"])
    vix_df = pd.read_parquet(files["vix_proxy"])
    sect_df = pd.read_parquet(files["sector_returns"])

    agg: Dict = {
        "spy": spy_df["spy_return"],
        "spy_price": spy_df["spy_price"],
        "vix_proxy": vix_df["vix_proxy"],
        "vix_regime": vix_df["vix_regime"].astype("int8"),
        "vix_proxy_pct_change_5d": vix_df["vix_proxy_pct_change_5d"],
        "sector_returns": sect_df,
        "xs_rank_21d_return": pd.read_parquet(files["xs_rank_21d_return"]),
        "xs_rank_volume_21d": pd.read_parquet(files["xs_rank_volume_21d"]),
        "xs_rank_volatility_21d": pd.read_parquet(files["xs_rank_volatility_21d"]),
        "log_ret_panel": pd.read_parquet(files["log_ret_panel"]),
        "sector_rank_5d_df": pd.read_parquet(files["sector_rank_5d_df"]),
        "ticker_sectors": pd.read_parquet(files["ticker_sectors"])["sector"].to_dict(),
    }
    logger.info("Loaded universe aggregates from manifest (built %s).", manifest.get("built_at", "?"))
    return agg


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def precompute_universe_aggregates(
    start: str = "2021-04-21",
    end: str = "2026-04-21",
    force_recompute: bool = False,
) -> Dict:
    """One-time per-process computation of universe-level aggregates.

    Aggregates computed
    -------------------
    spy                  : pd.Series  -- daily log returns (equal-weighted proxy; no SPY.parquet)
    spy_price            : pd.Series  -- cumulative price index (base 100)
    vix_proxy            : pd.Series  -- median 21d annualised realised vol across universe
    vix_regime           : pd.Series  -- categorical 0/1/2 (low/mid/high vol regime)
    vix_proxy_pct_change_5d : pd.Series  -- 5-day % change in vix_proxy
    sector_returns       : pd.DataFrame -- columns=sectors, values=5d mean log return (shifted)
    sector_rank_5d_df    : pd.DataFrame -- columns=tickers, values=intra-sector 5d return rank [0,1]
    xs_rank_21d_return   : pd.DataFrame -- columns=tickers, percentile rank of 21d return
    xs_rank_volume_21d   : pd.DataFrame -- columns=tickers, percentile rank of 21d avg volume
    xs_rank_volatility_21d : pd.DataFrame -- columns=tickers, percentile rank of 21d realised vol
    log_ret_panel        : pd.DataFrame -- raw daily log returns (date x ticker)
    ticker_sectors       : dict         -- ticker -> sector string

    Caches to disk at CACHE_PATH.  On subsequent calls (same process or new
    process) the cache is loaded instead of recomputed.

    SURVIVORSHIP-BIAS NOTE: The universe is the static list of 502 tickers in
    the data snapshot.  Historical S&P 500 membership is not replicated.
    """
    global _UNIVERSE_AGG_SINGLETON

    if _UNIVERSE_AGG_SINGLETON is not None and not force_recompute:
        return _UNIVERSE_AGG_SINGLETON

    os.makedirs(CACHE_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Fast path 1: manifest cache (v6 layout -- individual named parquets)
    # ------------------------------------------------------------------
    if os.path.exists(MANIFEST_PATH) and not force_recompute:
        try:
            logger.info("Loading universe aggregates from manifest: %s", MANIFEST_PATH)
            t_load = time.time()
            agg = _load_manifest_cache()
            logger.info("Manifest load complete in %.1fs.", time.time() - t_load)
            _UNIVERSE_AGG_SINGLETON = agg
            return agg
        except Exception as exc:
            logger.warning("Manifest load failed (%s); falling back to legacy cache.", exc)

    # ------------------------------------------------------------------
    # Fast path 2: legacy cache (universe_agg.parquet marker)
    # ------------------------------------------------------------------
    if os.path.exists(CACHE_PATH) and not force_recompute:
        logger.info("Loading universe aggregates from legacy cache: %s", CACHE_PATH)
        t_load = time.time()
        agg = _load_cache(CACHE_PATH)
        logger.info("Legacy cache load complete in %.1fs.", time.time() - t_load)
        # Write the manifest so next call is even faster
        try:
            _save_manifest_cache(agg)
        except Exception as exc:
            logger.warning("Could not write manifest after legacy load: %s", exc)
        _UNIVERSE_AGG_SINGLETON = agg
        return agg

    # ------------------------------------------------------------------
    # Cold-start: build from scratch using /tmp daily cache
    # ------------------------------------------------------------------
    logger.info("Computing universe aggregates from scratch...")
    t0 = time.time()

    # Discover all ticker parquets
    files = sorted(glob.glob(os.path.join(DATA_ROOT, "*.parquet")))
    tickers = [os.path.basename(f).replace(".parquet", "") for f in files]
    logger.info("Found %d ticker parquets.", len(tickers))

    # Step 1: populate /tmp daily cache (Drive read, one-time; ~10 min cold, 0s warm)
    _build_tmp_daily_cache(tickers)

    # ------------------------------------------------------------------
    # Load all daily close + volume from /tmp (fast local I/O)
    # ------------------------------------------------------------------
    close_list: list[pd.Series] = []
    volume_list: list[pd.Series] = []

    for i, tk in enumerate(tickers):
        if i % 50 == 0:
            logger.info("  Loading ticker %d/%d from /tmp ...", i, len(tickers))
        # Prefer /tmp cache; fall back to Drive read if missing
        df = _load_tmp_daily(tk)
        if df is None:
            df = _load_daily_close_volume(tk)
        if df is not None and not df.empty:
            close_list.append(df["close"].rename(tk))
            volume_list.append(df["volume"].rename(tk))

    close_panel = pd.concat(close_list, axis=1).sort_index()
    volume_panel = pd.concat(volume_list, axis=1).sort_index()

    # Restrict to requested date window
    close_panel = close_panel.loc[start:end]
    volume_panel = volume_panel.loc[start:end]

    # ------------------------------------------------------------------
    # A. SPY proxy -- equal-weighted log returns (SPY.parquet not present)
    # ------------------------------------------------------------------
    log_ret = np.log(close_panel / close_panel.shift(1))  # date x ticker (raw, unshifted)

    ew_price = close_panel.mean(axis=1)
    spy_return = np.log(ew_price / ew_price.shift(1)).rename("spy_return")
    # Normalised cumulative price index (base 100 at start)
    spy_price = (1 + spy_return.fillna(0)).cumprod() * 100
    spy_price.name = "spy_price"

    # ------------------------------------------------------------------
    # B. VIX proxy -- median 21d realised vol per day
    # log_ret is raw (unshifted); apply shift(1) so day-t's vix_proxy
    # only uses returns through day t-1. Annualise by sqrt(252).
    # ------------------------------------------------------------------
    log_ret_shifted = log_ret.shift(1)
    rv21 = log_ret_shifted.rolling(21, min_periods=10).std() * math.sqrt(252)
    vix_proxy = rv21.median(axis=1).rename("vix_proxy")

    # VIX regime: rolling 252-day percentile thresholds (30th / 70th pctl)
    vix_pctl_30 = vix_proxy.rolling(252, min_periods=63).quantile(0.30)
    vix_pctl_70 = vix_proxy.rolling(252, min_periods=63).quantile(0.70)
    vix_regime = pd.Series(1, index=vix_proxy.index, name="vix_regime", dtype="int8")
    vix_regime[vix_proxy < vix_pctl_30] = 0
    vix_regime[vix_proxy > vix_pctl_70] = 2

    vix_proxy_pct5d = vix_proxy.pct_change(5).rename("vix_proxy_pct_change_5d")

    # ------------------------------------------------------------------
    # C. Sector returns and intra-sector ranks
    # ret5d_shifted: 5d cumulative log return with shift(1) for PIT safety
    # ------------------------------------------------------------------
    ret5d_shifted = log_ret.rolling(5).sum().shift(1)   # date x ticker, PIT-safe

    ticker_sectors = {tk: SECTOR_MAP.get(tk, DEFAULT_SECTOR) for tk in close_panel.columns}
    all_sectors = sorted(set(ticker_sectors.values()))

    # Per-sector mean 5d return (sector_returns DataFrame)
    sector_ret_dict: Dict[str, pd.Series] = {}
    for sector in all_sectors:
        members = [tk for tk, s in ticker_sectors.items() if s == sector]
        if members:
            sector_ret_dict[sector] = ret5d_shifted[members].mean(axis=1)
    sector_return_df = pd.DataFrame(sector_ret_dict)

    # Intra-sector percentile rank of 5d return -- precomputed for all tickers
    # so add_cross_sectional_features does NOT recompute the full panel each call
    sector_rank_frames: list[pd.DataFrame] = []
    for sector in all_sectors:
        members = [tk for tk, s in ticker_sectors.items() if s == sector]
        if len(members) > 1:
            slc = ret5d_shifted[members]
            ranks = slc.rank(axis=1, pct=True)
            sector_rank_frames.append(ranks)
        elif len(members) == 1:
            # Only one ticker in sector -- rank is trivially 1.0
            solo = members[0]
            sector_rank_frames.append(
                pd.DataFrame({solo: 1.0}, index=ret5d_shifted.index)
            )
    sector_rank_5d_df = pd.concat(sector_rank_frames, axis=1)

    # ------------------------------------------------------------------
    # D. Cross-sectional rank features (percentile rank across all tickers)
    # All use shift(1) for PIT safety.
    # ------------------------------------------------------------------
    ret21_shifted = log_ret.rolling(21).sum().shift(1)
    xs_rank_21d_return = ret21_shifted.rank(axis=1, pct=True)

    vol21_shifted = volume_panel.rolling(21).mean().shift(1)
    xs_rank_volume_21d = vol21_shifted.rank(axis=1, pct=True)

    xs_rank_volatility_21d = rv21.rank(axis=1, pct=True)

    # ------------------------------------------------------------------
    # Assemble and cache
    # ------------------------------------------------------------------
    agg = {
        "spy": spy_return,
        "spy_price": spy_price,
        "vix_proxy": vix_proxy,
        "vix_regime": vix_regime,
        "vix_proxy_pct_change_5d": vix_proxy_pct5d,
        "sector_returns": sector_return_df,
        "sector_rank_5d_df": sector_rank_5d_df,
        "xs_rank_21d_return": xs_rank_21d_return,
        "xs_rank_volume_21d": xs_rank_volume_21d,
        "xs_rank_volatility_21d": xs_rank_volatility_21d,
        "log_ret_panel": log_ret,
        "ticker_sectors": ticker_sectors,
    }

    logger.info("Caching universe aggregates (legacy + manifest)...")
    _save_cache(agg, CACHE_PATH)
    _save_manifest_cache(agg)

    elapsed = time.time() - t0
    logger.info("Universe aggregates ready in %.1f s.", elapsed)

    _UNIVERSE_AGG_SINGLETON = agg
    return agg


# ---------------------------------------------------------------------------
# Cache helpers -- each large DataFrame/Series stored as a separate parquet
# ---------------------------------------------------------------------------

def _save_cache(agg: Dict, base_path: str) -> None:
    """Persist universe aggregates.  Each large object stored as a separate
    .parquet file alongside the marker file at base_path."""
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    stem = base_path.replace(".parquet", "")

    def _to_df(obj, name):
        return obj.to_frame(name) if isinstance(obj, pd.Series) else obj

    # Scalar Series -- stored together as a single narrow file
    series_keys = [
        "spy", "spy_price", "vix_proxy", "vix_regime", "vix_proxy_pct_change_5d",
    ]
    scalar_df = pd.concat(
        [_to_df(agg[k], k) for k in series_keys if k in agg], axis=1
    )
    scalar_df.to_parquet(f"{stem}_scalars.parquet")

    # Wide DataFrames -- one file each
    wide_keys = [
        "sector_returns", "sector_rank_5d_df",
        "xs_rank_21d_return", "xs_rank_volume_21d",
        "xs_rank_volatility_21d", "log_ret_panel",
    ]
    for k in wide_keys:
        if k in agg:
            agg[k].to_parquet(f"{stem}_{k}.parquet")

    # ticker_sectors dict
    pd.Series(agg["ticker_sectors"]).to_frame("sector").to_parquet(
        f"{stem}_ticker_sectors.parquet"
    )

    # Marker file (acts as existence check)
    pd.DataFrame({"ts": [pd.Timestamp.now()]}).to_parquet(base_path)
    logger.info("Cache saved (%d files).", len(series_keys) + len(wide_keys) + 2)


def _load_cache(base_path: str) -> Dict:
    """Load universe aggregates from cached parquet files."""
    stem = base_path.replace(".parquet", "")

    scalar_df = pd.read_parquet(f"{stem}_scalars.parquet")
    agg: Dict = {
        "spy": scalar_df["spy"],
        "spy_price": scalar_df["spy_price"],
        "vix_proxy": scalar_df["vix_proxy"],
        "vix_regime": scalar_df["vix_regime"].astype("int8"),
        "vix_proxy_pct_change_5d": scalar_df["vix_proxy_pct_change_5d"],
    }

    wide_keys = [
        "sector_returns", "sector_rank_5d_df",
        "xs_rank_21d_return", "xs_rank_volume_21d",
        "xs_rank_volatility_21d", "log_ret_panel",
    ]
    for k in wide_keys:
        agg[k] = pd.read_parquet(f"{stem}_{k}.parquet")

    ts_df = pd.read_parquet(f"{stem}_ticker_sectors.parquet")
    agg["ticker_sectors"] = ts_df["sector"].to_dict()
    return agg


# ---------------------------------------------------------------------------
# Per-ticker feature enrichment
# ---------------------------------------------------------------------------

def add_cross_sectional_features(
    daily_df: pd.DataFrame,
    ticker: str,
    universe_agg: Optional[Dict] = None,
) -> pd.DataFrame:
    """Join all cross-sectional features onto a per-ticker daily DataFrame.

    Parameters
    ----------
    daily_df : pd.DataFrame
        Daily OHLCV (or feature) DataFrame for a single ticker.
        Must have a DatetimeIndex.
    ticker   : str
        Ticker symbol (case-sensitive, must match parquet filenames).
    universe_agg : dict or None
        Output of precompute_universe_aggregates().  If None, it will be
        called -- expensive on first call, so prefer passing a shared instance.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with 17 new cross-sectional feature columns appended.
        Original columns are not modified.  Rows with no universe data for a
        given date will have NaN in the new columns.

    POINT-IN-TIME SAFETY
    --------------------
    All features are point-in-time safe at bar t:
    - SPY-relative returns use rolling windows then shift(1) on the raw series.
    - corr_spy_60d / beta_spy_60d apply shift(1) on the raw log-return series
      before computing the 60-bar rolling window.
    - Universe aggregates (VIX proxy, sector returns, xs-ranks) were computed
      with shift(1) during precompute_universe_aggregates() and are joined as-is.
    - Calendar features are deterministic and require no shift.
    """
    if universe_agg is None:
        universe_agg = precompute_universe_aggregates()

    df = daily_df.copy()

    # Normalise index to UTC DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("daily_df must have a DatetimeIndex.")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    elif str(df.index.tz) != "UTC":
        df.index = df.index.tz_convert("UTC")

    dates = df.index

    # ------------------------------------------------------------------
    # Resolve per-ticker log-return series from the panel (already loaded)
    # ------------------------------------------------------------------
    log_ret_panel: pd.DataFrame = universe_agg["log_ret_panel"]
    spy_return: pd.Series = universe_agg["spy"]

    if ticker in log_ret_panel.columns:
        tk_log_ret = log_ret_panel[ticker]
    else:
        # Ticker not in universe panel -- derive from daily_df if possible
        if "close" in df.columns:
            _c = df["close"].ffill()
            tk_log_ret = np.log(_c / _c.shift(1))
        else:
            tk_log_ret = pd.Series(np.nan, index=dates, dtype=float)

    spy_aligned = spy_return.reindex(tk_log_ret.index)

    # ------------------------------------------------------------------
    # A. SPY-relative features
    # ------------------------------------------------------------------
    # Cumulative log returns, shifted by 1 for PIT safety
    tk_ret5 = tk_log_ret.rolling(5).sum().shift(1)
    tk_ret21 = tk_log_ret.rolling(21).sum().shift(1)
    spy_ret5 = spy_aligned.rolling(5).sum().shift(1)
    spy_ret21 = spy_aligned.rolling(21).sum().shift(1)

    spy_rel_5d = (tk_ret5 - spy_ret5).rename("spy_relative_return_5d")
    spy_rel_21d = (tk_ret21 - spy_ret21).rename("spy_relative_return_21d")

    # Rolling 60d correlation with SPY proxy.
    # Both series are raw log returns; shift(1) once so the window at t
    # uses only data through t-1.
    def _rolling_corr(a: pd.Series, b: pd.Series, window: int = 60) -> pd.Series:
        a_s = a.shift(1)
        b_s = b.shift(1)
        return a_s.rolling(window, min_periods=20).corr(b_s)

    def _rolling_beta(a: pd.Series, b: pd.Series, window: int = 60) -> pd.Series:
        """Beta = Cov(a,b) / Var(b) -- rolling window."""
        a_s = a.shift(1)
        b_s = b.shift(1)
        cov = a_s.rolling(window, min_periods=20).cov(b_s)
        var_b = b_s.rolling(window, min_periods=20).var()
        return (cov / var_b.replace(0, np.nan)).rename("beta_spy_60d")

    corr_spy = _rolling_corr(tk_log_ret, spy_aligned, 60).rename("corr_spy_60d")
    beta_spy = _rolling_beta(tk_log_ret, spy_aligned, 60)

    # ------------------------------------------------------------------
    # B. VIX proxy regime features  (precomputed, join directly)
    # ------------------------------------------------------------------
    vix_proxy: pd.Series = universe_agg["vix_proxy"]
    vix_regime: pd.Series = universe_agg["vix_regime"]
    vix_pct5d: pd.Series = universe_agg["vix_proxy_pct_change_5d"]

    # ------------------------------------------------------------------
    # C. Sector context  (all precomputed, O(1) column lookup)
    # ------------------------------------------------------------------
    sector_return_df: pd.DataFrame = universe_agg["sector_returns"]
    sector_rank_5d_df: pd.DataFrame = universe_agg["sector_rank_5d_df"]
    ticker_sectors: Dict[str, str] = universe_agg["ticker_sectors"]
    sector = ticker_sectors.get(ticker, DEFAULT_SECTOR)

    # Mean 5d return of the ticker's sector
    if sector in sector_return_df.columns:
        sector_ret5d = sector_return_df[sector].rename("sector_return_5d")
    else:
        sector_ret5d = pd.Series(
            np.nan, index=sector_return_df.index, dtype=float, name="sector_return_5d"
        )

    # Relative return: ticker's own 5d return minus sector mean
    sector_rel_5d = (
        tk_ret5.reindex(sector_return_df.index) - sector_ret5d
    ).rename("sector_relative_return_5d")

    # Intra-sector percentile rank -- already precomputed
    if ticker in sector_rank_5d_df.columns:
        sector_rank_5d = sector_rank_5d_df[ticker].rename("sector_rank_5d_return")
    else:
        sector_rank_5d = pd.Series(
            np.nan, index=log_ret_panel.index, dtype=float, name="sector_rank_5d_return"
        )

    # ------------------------------------------------------------------
    # D. Cross-sectional rank features  (all precomputed, O(1) lookup)
    # ------------------------------------------------------------------
    def _xs_col(df_wide: pd.DataFrame, col_name: str) -> pd.Series:
        if ticker in df_wide.columns:
            return df_wide[ticker].rename(col_name)
        return pd.Series(np.nan, index=log_ret_panel.index, dtype=float, name=col_name)

    xs_rank_21d_ret_tk = _xs_col(universe_agg["xs_rank_21d_return"], "xs_rank_21d_return")
    xs_rank_vol_tk = _xs_col(universe_agg["xs_rank_volume_21d"], "xs_rank_volume_21d")
    xs_rank_volat_tk = _xs_col(universe_agg["xs_rank_volatility_21d"], "xs_rank_volatility_21d")

    # ------------------------------------------------------------------
    # E. Calendar cyclical encoding  (deterministic, no shift needed)
    # ------------------------------------------------------------------
    dow = dates.dayofweek.astype(float)   # 0=Mon ... 4=Fri
    month = dates.month.astype(float)

    df["dow_sin"] = np.sin(2 * math.pi * dow / 5.0)
    df["dow_cos"] = np.cos(2 * math.pi * dow / 5.0)
    df["month_sin"] = np.sin(2 * math.pi * (month - 1) / 12.0)
    df["month_cos"] = np.cos(2 * math.pi * (month - 1) / 12.0)

    # ------------------------------------------------------------------
    # Join all universe-derived features onto df by reindexing to df.index
    # ------------------------------------------------------------------
    def _join(series: pd.Series, col_name: str) -> None:
        df[col_name] = series.reindex(df.index)

    _join(spy_rel_5d,           "spy_relative_return_5d")
    _join(spy_rel_21d,          "spy_relative_return_21d")
    _join(corr_spy,             "corr_spy_60d")
    _join(beta_spy,             "beta_spy_60d")
    _join(vix_proxy,            "vix_proxy")
    _join(vix_regime,           "vix_regime")
    _join(vix_pct5d,            "vix_proxy_pct_change_5d")
    _join(sector_ret5d,         "sector_return_5d")
    _join(sector_rel_5d,        "sector_relative_return_5d")
    _join(sector_rank_5d,       "sector_rank_5d_return")
    _join(xs_rank_21d_ret_tk,   "xs_rank_21d_return")
    _join(xs_rank_vol_tk,       "xs_rank_volume_21d")
    _join(xs_rank_volat_tk,     "xs_rank_volatility_21d")

    return df


# ---------------------------------------------------------------------------
# Smoke test (run directly: python cross_sectional_features.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time as _time

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    t0 = _time.time()
    agg = precompute_universe_aggregates()
    print(f"\nprecompute took {_time.time()-t0:.1f}s; cache keys: {list(agg.keys())}")

    test_dates = pd.date_range("2024-06-01", "2024-12-31", freq="B", tz="UTC")
    df_base = pd.DataFrame({"close": 100.0}, index=test_dates)

    for tk in ["AAPL", "NVDA", "XOM", "BEN"]:
        out = add_cross_sectional_features(df_base.copy(), tk, agg)
        new_cols = [c for c in out.columns if c not in df_base.columns]
        print(f"\n=== {tk} ({len(new_cols)} new cols) ===")
        print(out[new_cols].iloc[20:24].to_string())
        for c in new_cols:
            if out[c].dtype != object:
                print(f"  {c}: non-zero {(out[c] != 0).mean() * 100:.0f}%")
