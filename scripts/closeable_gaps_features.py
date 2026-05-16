"""
closeable_gaps_features.py
==========================
Closes 3 of 6 still-missing data gaps in the v8 XGBoost pipeline using
FREE, no-key-required sources:

  1. XBRL fundamentals      -> yfinance Ticker.info snapshot
     (trailingPE, forwardPE, priceToBook, ebitdaMargins, profitMargins,
      returnOnEquity, debtToEquity, pegRatio)
  2. FINRA short-volume     -> daily CSV at https://cdn.finra.org/equity/regsho/daily/
     (short_volume_ratio, plus 5d/20d MAs, z-score, MA crossover)
  3. Analyst consensus      -> yfinance Ticker.info + analyst_price_targets
     (targetMean/High/Low pct from price, recommendationMean,
      numberOfAnalystOpinions)

NOT closed (paid-only or infeasible in 60-min budget):
  - Real options flow (IV, put/call, gamma)   -- Polygon options $29/mo or CBOE
  - Real L2 footprint (order book, BAV)       -- Polygon L2 / IEX DEEP $0 but L2 limited
  - Satellite / supply-chain (geospatial)     -- RS Metrics, Orbital Insight (enterprise)

Design notes
------------
* yfinance info is a *snapshot* (no historical series). The fundamentals and
  analyst values are broadcast across all rows of the input DataFrame and
  then `.shift(1)` is applied so today's bar uses yesterday's value. This is
  technically backward-leaky for very old rows but matches the convention of
  other v7 modules (e.g. google_trends 5y window) and is acceptable for a
  v8 prototype. Production hardening would require a point-in-time
  fundamentals vendor.
* FINRA daily short volume *is* genuinely time-series, fetched per trading
  day and aligned to the bar index, then `.shift(1)` for safety.
* All features end in `_snap` for snapshot fundamentals/analyst fields and
  `_sv` for short-volume so they're easy to identify in importance reports.
* Cache: per-ticker parquet at cache/closeable_gaps/<TICKER>_finra_sv.parquet
  and cache/closeable_gaps/<TICKER>_yfinfo.json. Delete to force refresh.
"""
from __future__ import annotations

import io
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

# Cache layout matches other modules (gtrends/form4)
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_CACHE_DIR = _PROJECT_ROOT / "cache" / "closeable_gaps"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_FINRA_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{ymd}.txt"
_HEADERS = {"User-Agent": "sp500-mastery-research/0.1 (research@example.org)"}
_HTTP_TIMEOUT = 15
_FINRA_SLEEP = 0.05   # 50ms between requests to be polite to FINRA CDN


# =====================================================================
# 1. yfinance fundamentals + analyst snapshot
# =====================================================================
def _fetch_yfinance_info(ticker: str) -> dict:
    cache = _CACHE_DIR / f"{ticker}_yfinfo.json"
    if cache.exists():
        try:
            with open(cache) as fp:
                return json.load(fp)
        except Exception:
            pass
    try:
        import yfinance as yf
    except ImportError:
        log.warning("yfinance not installed; closeable_gaps will be NaN")
        return {}
    try:
        t = yf.Ticker(ticker)
        info = dict(t.info) if t.info else {}
        # Compact to JSON-serializable primitives only
        serial = {}
        for k, v in info.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                serial[k] = v
        with open(cache, "w") as fp:
            json.dump(serial, fp)
        return serial
    except Exception as e:
        log.warning("yfinance info(%s) failed: %s", ticker, e)
        return {}


def add_fundamentals_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Add 8 fundamentals features from yfinance Ticker.info snapshot.

    Features (all .shift(1) safe broadcast snapshot):
      fund_trailing_pe_snap, fund_forward_pe_snap, fund_price_to_book_snap,
      fund_peg_ratio_snap, fund_profit_margin_snap, fund_ebitda_margin_snap,
      fund_return_on_equity_snap, fund_debt_to_equity_snap
    """
    info = _fetch_yfinance_info(ticker)
    mapping = {
        "fund_trailing_pe_snap":        "trailingPE",
        "fund_forward_pe_snap":         "forwardPE",
        "fund_price_to_book_snap":      "priceToBook",
        "fund_peg_ratio_snap":          "pegRatio",
        "fund_profit_margin_snap":      "profitMargins",
        "fund_ebitda_margin_snap":      "ebitdaMargins",
        "fund_return_on_equity_snap":   "returnOnEquity",
        "fund_debt_to_equity_snap":     "debtToEquity",
    }
    out = df.copy()
    for feat, key in mapping.items():
        v = info.get(key)
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            out[feat] = np.nan
        else:
            try:
                out[feat] = float(v)
            except Exception:
                out[feat] = np.nan
    # Shift by 1 for label-leak safety (even though snapshot is invariant)
    for feat in mapping:
        out[feat] = out[feat].shift(1)
    return out


def add_analyst_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Add 5 analyst-consensus features from yfinance snapshot.

    Features (snapshot, .shift(1)-safe):
      analyst_n_opinions_snap, analyst_rec_mean_snap,
      analyst_target_mean_pct_snap, analyst_target_high_pct_snap,
      analyst_target_low_pct_snap

    *_pct features are (target / current_price) - 1, computed per-bar
    against the bar's close (so they evolve as price moves even though
    the underlying target is a snapshot).
    """
    info = _fetch_yfinance_info(ticker)
    n_opinions = info.get("numberOfAnalystOpinions")
    rec_mean = info.get("recommendationMean")
    tgt_mean = info.get("targetMeanPrice")
    tgt_high = info.get("targetHighPrice")
    tgt_low = info.get("targetLowPrice")

    out = df.copy()
    out["analyst_n_opinions_snap"] = float(n_opinions) if n_opinions else np.nan
    out["analyst_rec_mean_snap"] = float(rec_mean) if rec_mean else np.nan

    # Use today's close to compute target-pct (target stays constant; price moves)
    close = out["close"] if "close" in out.columns else None
    for feat, tgt in [
        ("analyst_target_mean_pct_snap", tgt_mean),
        ("analyst_target_high_pct_snap", tgt_high),
        ("analyst_target_low_pct_snap", tgt_low),
    ]:
        if tgt is None or close is None:
            out[feat] = np.nan
        else:
            try:
                out[feat] = (float(tgt) / close) - 1.0
            except Exception:
                out[feat] = np.nan

    # Shift for safety
    for feat in [
        "analyst_n_opinions_snap", "analyst_rec_mean_snap",
        "analyst_target_mean_pct_snap", "analyst_target_high_pct_snap",
        "analyst_target_low_pct_snap",
    ]:
        out[feat] = out[feat].shift(1)
    return out


# =====================================================================
# 2. FINRA short-volume daily
# =====================================================================
def _fetch_finra_sv_range(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Fetch daily FINRA short-volume ratio (short_vol / total_vol) for ticker."""
    cache = _CACHE_DIR / f"{ticker}_finra_sv.parquet"
    if cache.exists():
        try:
            cached = pd.read_parquet(cache)
            # If covers requested range, return slice
            if cached.index.min() <= start and cached.index.max() >= end:
                return cached["short_vol_ratio"].loc[start:end]
        except Exception:
            pass

    # Build list of business days
    dates = pd.date_range(start, end, freq="B")
    records = []
    for dt in dates:
        ymd = dt.strftime("%Y%m%d")
        url = _FINRA_URL.format(ymd=ymd)
        try:
            r = requests.get(url, headers=_HEADERS, timeout=_HTTP_TIMEOUT)
            if r.status_code != 200:
                continue
            df = pd.read_csv(io.StringIO(r.text), sep="|")
            row = df[df["Symbol"] == ticker]
            if len(row) == 0:
                continue
            r0 = row.iloc[0]
            sv = float(r0["ShortVolume"])
            tv = float(r0["TotalVolume"])
            ratio = sv / tv if tv > 0 else np.nan
            records.append({"date": dt, "short_vol_ratio": ratio,
                            "short_volume": sv, "total_volume": tv})
        except Exception as e:
            log.debug("FINRA %s err %s: %s", ymd, ticker, e)
        time.sleep(_FINRA_SLEEP)

    if not records:
        return pd.Series(dtype=float)

    out = pd.DataFrame(records).set_index("date").sort_index()
    try:
        out.to_parquet(cache)
    except Exception as e:
        log.debug("FINRA cache write failed: %s", e)
    return out["short_vol_ratio"]


def add_short_interest_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Add 5 FINRA short-volume features (genuinely time-series, .shift(1)-safe).

    Features:
      short_vol_ratio_sv      : daily short_vol / total_vol (lagged 1 day)
      short_vol_ratio_ma5_sv  : 5-day MA
      short_vol_ratio_ma20_sv : 20-day MA
      short_vol_ratio_z60_sv  : z-score vs trailing 60-day mean/std
      short_vol_ratio_cross_sv: 1 if MA5 > MA20 else 0
    """
    out = df.copy()
    if len(out) == 0:
        return out

    # Determine fetch range (use df index tz-naive for FINRA daily files)
    idx_naive = out.index.tz_convert(None) if getattr(out.index, "tz", None) else out.index
    start = pd.Timestamp(idx_naive.min()).normalize()
    end = pd.Timestamp(idx_naive.max()).normalize()

    sv = _fetch_finra_sv_range(ticker, start, end)
    if sv.empty:
        for feat in ["short_vol_ratio_sv", "short_vol_ratio_ma5_sv",
                     "short_vol_ratio_ma20_sv", "short_vol_ratio_z60_sv",
                     "short_vol_ratio_cross_sv"]:
            out[feat] = np.nan
        return out

    # Align to df index (tz-naive on both sides for the join, then assign back)
    sv.index = pd.to_datetime(sv.index)
    sv_aligned = sv.reindex(idx_naive).ffill()

    # Compute derived series
    ma5 = sv_aligned.rolling(5, min_periods=2).mean()
    ma20 = sv_aligned.rolling(20, min_periods=5).mean()
    mean60 = sv_aligned.rolling(60, min_periods=15).mean()
    std60 = sv_aligned.rolling(60, min_periods=15).std()
    z60 = (sv_aligned - mean60) / std60.replace(0, np.nan)
    cross = (ma5 > ma20).astype(float)

    # Assign back to df index (preserve original tz-aware index)
    out["short_vol_ratio_sv"] = sv_aligned.values
    out["short_vol_ratio_ma5_sv"] = ma5.values
    out["short_vol_ratio_ma20_sv"] = ma20.values
    out["short_vol_ratio_z60_sv"] = z60.values
    out["short_vol_ratio_cross_sv"] = cross.values

    # .shift(1) safety on all derived
    for feat in ["short_vol_ratio_sv", "short_vol_ratio_ma5_sv",
                 "short_vol_ratio_ma20_sv", "short_vol_ratio_z60_sv",
                 "short_vol_ratio_cross_sv"]:
        out[feat] = out[feat].shift(1)
    return out


# =====================================================================
# Master combined helper
# =====================================================================
def add_closeable_gap_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Add all 3 closeable-gap feature groups (fundamentals + short + analyst).

    Total: 18 features (8 fundamentals + 5 short + 5 analyst).
    """
    out = df
    try:
        out = add_fundamentals_features(out, ticker)
    except Exception as e:
        log.warning("add_fundamentals_features(%s) failed: %s", ticker, e)
    try:
        out = add_short_interest_features(out, ticker)
    except Exception as e:
        log.warning("add_short_interest_features(%s) failed: %s", ticker, e)
    try:
        out = add_analyst_features(out, ticker)
    except Exception as e:
        log.warning("add_analyst_features(%s) failed: %s", ticker, e)
    return out


def closeable_gap_feature_names() -> list:
    """Return canonical list of 18 feature names produced by this module."""
    return [
        # fundamentals (8)
        "fund_trailing_pe_snap", "fund_forward_pe_snap", "fund_price_to_book_snap",
        "fund_peg_ratio_snap", "fund_profit_margin_snap", "fund_ebitda_margin_snap",
        "fund_return_on_equity_snap", "fund_debt_to_equity_snap",
        # short interest (5)
        "short_vol_ratio_sv", "short_vol_ratio_ma5_sv", "short_vol_ratio_ma20_sv",
        "short_vol_ratio_z60_sv", "short_vol_ratio_cross_sv",
        # analyst (5)
        "analyst_n_opinions_snap", "analyst_rec_mean_snap",
        "analyst_target_mean_pct_snap", "analyst_target_high_pct_snap",
        "analyst_target_low_pct_snap",
    ]


if __name__ == "__main__":
    # Smoke test entry point
    import sys
    tk = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    sys.path.insert(0, str(_SCRIPT_DIR))
    import backtest_ml as bml
    d = bml.load_daily(tk)
    f = bml.build_features(d)
    before = f.shape[1]
    f = add_closeable_gap_features(f, tk)
    print(f"{tk}: {before} -> {f.shape[1]} cols (+{f.shape[1]-before})")
    feats = closeable_gap_feature_names()
    print("Sample tail:")
    print(f[feats].tail(3))
