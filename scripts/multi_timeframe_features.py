"""
multi_timeframe_features.py
============================
Aggregates 1Min bars to 5Min / 15Min / 60Min / 240Min timeframes and attaches
quality multi-timeframe features to a daily-indexed DataFrame.

API
---
    add_multi_timeframe_features(daily_df, ticker) -> pd.DataFrame

All features are .shift(1) so bar t uses ONLY the prior session's intraday data
(point-in-time safe).  Session window: 09:30–16:00 ET per calendar day.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ONE_MIN_ROOT = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/claudes test/data/timeframes"
    "/S&P500 5 Year Historical Data/Minutes TimeFrames/1Min_merged"
)

ET_TZ = "America/New_York"
SESSION_OPEN_HOUR = 9
SESSION_OPEN_MIN = 30
SESSION_CLOSE_HOUR = 16  # exclusive upper bound (16:00 bar is after-hours)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level memoization: ticker -> (timestamp-indexed, ET-localized) 1Min DF
# ---------------------------------------------------------------------------

_CACHE: Dict[str, pd.DataFrame] = {}


def _load_1min(ticker: str) -> pd.DataFrame:
    """Read 1Min parquet once per process, cache it, return ET-indexed DF."""
    if ticker in _CACHE:
        return _CACHE[ticker]

    path = ONE_MIN_ROOT / f"{ticker}.parquet"
    if not path.exists():
        logger.warning("1Min parquet not found for %s at %s", ticker, path)
        _CACHE[ticker] = pd.DataFrame()
        return _CACHE[ticker]

    df = pd.read_parquet(path)
    df = df.set_index("timestamp").sort_index()
    # Convert UTC -> ET
    df.index = df.index.tz_convert(ET_TZ)
    _CACHE[ticker] = df
    return df


def _session_only(df_et: pd.DataFrame) -> pd.DataFrame:
    """Keep only regular-session minutes (09:30 inclusive – 16:00 exclusive ET)."""
    t = df_et.index
    in_session = (
        (t.hour > SESSION_OPEN_HOUR)
        | ((t.hour == SESSION_OPEN_HOUR) & (t.minute >= SESSION_OPEN_MIN))
    ) & (t.hour < SESSION_CLOSE_HOUR)
    return df_et.loc[in_session]


# ---------------------------------------------------------------------------
# Pure-pandas technical helpers (no external TA library required)
# ---------------------------------------------------------------------------

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def _macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    macd_line = _ema(close, fast) - _ema(close, slow)
    sig_line = _ema(macd_line, signal)
    return macd_line - sig_line


# ---------------------------------------------------------------------------
# Per-timeframe feature builders — operate on CLOSED resampled bars
# ---------------------------------------------------------------------------

_RESAMPLE_KWARGS = dict(closed="left", label="left")


def _resample_ohlcv(df_1min: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample 1Min OHLCV to `rule` using closed='left', label='left'."""
    return df_1min.resample(rule, **_RESAMPLE_KWARGS).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna(subset=["close"])


def _build_h1_features(df_1min_session: pd.DataFrame) -> pd.DataFrame:
    """1-Hour bar features: RSI-14, close vs EMA-20 / ATR, MACD-hist sign."""
    h1 = _resample_ohlcv(df_1min_session, "1h")
    if h1.empty:
        return pd.DataFrame()

    rsi = _rsi(h1["close"], 14).rename("h1_rsi_14")
    atr = _atr(h1["high"], h1["low"], h1["close"], 14)
    ema20 = _ema(h1["close"], 20)
    close_vs_ema20_atr = ((h1["close"] - ema20) / atr.replace(0.0, np.nan)).rename(
        "h1_close_vs_ema20_atr"
    )
    macd_hist_sign = (_macd_hist(h1["close"]) > 0).astype(float).rename("h1_macd_hist_sign")

    out = pd.concat([rsi, close_vs_ema20_atr, macd_hist_sign], axis=1)
    return out


def _build_h4_features(df_1min_session: pd.DataFrame) -> pd.DataFrame:
    """4-Hour bar features: RSI-14, above EMA-50 binary, consecutive up bars."""
    h4 = _resample_ohlcv(df_1min_session, "4h")
    if h4.empty:
        return pd.DataFrame()

    rsi = _rsi(h4["close"], 14).rename("h4_rsi_14")
    ema50 = _ema(h4["close"], 50)
    above_ema50 = (h4["close"] > ema50).astype(float).rename("h4_above_ema_50")

    # Consecutive positive bars (close > open)
    is_up = (h4["close"] > h4["open"]).astype(int)
    groups = (is_up != is_up.shift(1)).cumsum()
    consec = is_up.groupby(groups).cumsum() * is_up
    consec = consec.rename("h4_consecutive_up_bars")

    out = pd.concat([rsi, above_ema50, consec], axis=1)
    return out


def _build_m5_features(df_1min_session: pd.DataFrame) -> pd.DataFrame:
    """5-Min intraday volatility profile features."""
    m5 = _resample_ohlcv(df_1min_session, "5min")
    if m5.empty:
        return pd.DataFrame()

    log_ret = np.log(m5["close"] / m5["close"].shift(1))
    rolling_ret = log_ret.rolling(5).sum()
    high_vol_flag = (log_ret.abs() > 0.005).astype(float)

    rows = {}
    for d, grp_idx in m5.groupby(m5.index.date).groups.items():
        key = pd.Timestamp(d, tz="UTC")
        lr = log_ret.reindex(grp_idx)
        rr = rolling_ret.reindex(grp_idx)
        hv = high_vol_flag.reindex(grp_idx)
        rows[key] = {
            "m5_realized_vol_session": float((lr ** 2).sum()),
            "m5_max_5bar_gain_pct": float(rr.max()) if not rr.isna().all() else np.nan,
            "m5_max_5bar_loss_pct": float(rr.min()) if not rr.isna().all() else np.nan,
            "m5_high_vol_minute_count": float(hv.sum()),
        }

    out = pd.DataFrame(rows).T.rename_axis(None)
    return out


def _build_m15_features(df_1min_session: pd.DataFrame, df_1min_full: pd.DataFrame) -> pd.DataFrame:
    """15-Min features: ATR%, close at session low binary, open-to-close / ATR."""
    m15_sess = _resample_ohlcv(df_1min_session, "15min")
    if m15_sess.empty:
        return pd.DataFrame()

    atr = _atr(m15_sess["high"], m15_sess["low"], m15_sess["close"], 14)

    # Is the last 15-min bar of session (15:45 ET) near the daily low?
    # "Near" = within 0.2% of session low close
    def last_bar_near_low(grp: pd.DataFrame) -> float:
        if grp.empty:
            return np.nan
        session_low = grp["low"].min()
        last_close = grp["close"].iloc[-1]
        if session_low == 0 or pd.isna(session_low):
            return np.nan
        return float((last_close - session_low) / session_low < 0.002)

    # Open-to-close range / ATR: use daily first open and last close
    def open_to_close_atr(grp: pd.DataFrame) -> float:
        if grp.empty or len(grp) < 2:
            return np.nan
        daily_range = abs(grp["close"].iloc[-1] - grp["open"].iloc[0])
        mean_atr = atr.reindex(grp.index).mean()
        if pd.isna(mean_atr) or mean_atr == 0:
            return np.nan
        return float(daily_range / mean_atr)

    # ATR pct per bar
    atr_pct_bar = atr / m15_sess["close"].replace(0.0, np.nan)

    # Aggregate all three features per calendar date using groupby(date)
    rows = {}
    for d, grp in m15_sess.groupby(m15_sess.index.date):
        key = pd.Timestamp(d, tz="UTC")
        atr_pct_grp = atr_pct_bar.reindex(grp.index)
        rows[key] = {
            "m15_atr_pct": atr_pct_grp.mean(),
            "m15_close_at_15min_low": last_bar_near_low(grp),
            "m15_session_open_to_close_diff_atr": open_to_close_atr(grp),
        }

    out = pd.DataFrame(rows).T.rename_axis(None)
    return out


# ---------------------------------------------------------------------------
# Daily aggregation: collapse intrabar features to one value per calendar day
# ---------------------------------------------------------------------------

def _last_value_per_day(tf_df: pd.DataFrame) -> pd.DataFrame:
    """For bar-level TF data: take the LAST observation within each ET calendar day."""
    return tf_df.groupby(tf_df.index.date).last()


def _to_daily_index(daily_val: pd.Series | pd.DataFrame) -> pd.DataFrame:
    """Convert any index (date objects, tz-aware Timestamps) to UTC-midnight DatetimeIndex."""
    if isinstance(daily_val, pd.Series):
        daily_val = daily_val.to_frame()
    idx = daily_val.index
    if not isinstance(idx, pd.DatetimeIndex):
        # Plain Python date objects -> UTC midnight
        daily_val = daily_val.copy()
        daily_val.index = pd.to_datetime([str(d) for d in idx]).tz_localize("UTC")
    elif idx.tz is None:
        daily_val = daily_val.copy()
        daily_val.index = idx.tz_localize("UTC")
    elif str(idx.tz) != "UTC":
        # tz-aware but not UTC: extract date portion -> UTC midnight
        daily_val = daily_val.copy()
        daily_val.index = pd.to_datetime([ts.date() for ts in idx]).tz_localize("UTC")
    # else: already UTC, leave as-is
    return daily_val


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_multi_timeframe_features(
    daily_df: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    """
    Resample 1Min bars to 5/15/60/240 min, compute features, attach to daily_df.

    Parameters
    ----------
    daily_df : pd.DataFrame
        Daily-indexed DataFrame (DatetimeIndex, any timezone).
    ticker : str
        Ticker symbol; matches filename under ONE_MIN_ROOT.

    Returns
    -------
    pd.DataFrame
        daily_df with additional MTF feature columns (all .shift(1) applied).
    """
    df_1min = _load_1min(ticker)
    if df_1min.empty:
        logger.warning("No 1Min data for %s — returning daily_df unchanged.", ticker)
        return daily_df

    # ------------------------------------------------------------------
    # Session-filtered 1Min data
    # ------------------------------------------------------------------
    df_sess = _session_only(df_1min)

    # ------------------------------------------------------------------
    # Build features at each timeframe
    # ------------------------------------------------------------------
    h1_bars = _build_h1_features(df_sess)
    h4_bars = _build_h4_features(df_sess)
    m5_daily = _build_m5_features(df_sess)      # already daily
    m15_daily = _build_m15_features(df_sess, df_1min)  # already daily

    # ------------------------------------------------------------------
    # Collapse H1 / H4 to one row per calendar day (last bar of day)
    # ------------------------------------------------------------------
    feature_frames = []

    if not h1_bars.empty:
        h1_daily = _last_value_per_day(h1_bars)
        feature_frames.append(_to_daily_index(h1_daily))

    if not h4_bars.empty:
        h4_daily = _last_value_per_day(h4_bars)
        feature_frames.append(_to_daily_index(h4_daily))

    if not m5_daily.empty:
        feature_frames.append(_to_daily_index(m5_daily))

    if not m15_daily.empty:
        feature_frames.append(_to_daily_index(m15_daily))

    if not feature_frames:
        return daily_df

    # ------------------------------------------------------------------
    # Merge all daily feature frames
    # ------------------------------------------------------------------
    mtf = feature_frames[0]
    for f in feature_frames[1:]:
        mtf = mtf.join(f, how="outer")

    # ------------------------------------------------------------------
    # Cross-TF derived features
    # ------------------------------------------------------------------
    if "h1_rsi_14" in mtf.columns and "h4_rsi_14" in mtf.columns:
        mtf["h1_vs_h4_rsi_diff"] = mtf["h1_rsi_14"] - mtf["h4_rsi_14"]

    # All three TFs bullish: daily close > H1 EMA20, H4 above EMA50
    # Proxy: daily RSI > 50 on both H1 and H4 (no daily RSI in this module)
    if "h1_rsi_14" in mtf.columns and "h4_rsi_14" in mtf.columns and "h4_above_ema_50" in mtf.columns:
        mtf["daily_above_h1_above_h4"] = (
            (mtf["h1_rsi_14"] > 50) & (mtf["h4_rsi_14"] > 50) & (mtf["h4_above_ema_50"] > 0)
        ).astype(float)

    # ------------------------------------------------------------------
    # .shift(1): feature at bar t uses prior session's data
    # ------------------------------------------------------------------
    mtf = mtf.sort_index().shift(1)

    # ------------------------------------------------------------------
    # Align to daily_df index
    # Both sides now use UTC-midnight DatetimeIndex; just reindex.
    # ------------------------------------------------------------------
    daily_df = daily_df.copy()

    # Normalize daily_df index to UTC midnight for matching
    if not isinstance(daily_df.index, pd.DatetimeIndex):
        daily_df.index = pd.to_datetime(daily_df.index)
    if daily_df.index.tz is None:
        utc_idx = daily_df.index.tz_localize("UTC").normalize()
    else:
        utc_idx = daily_df.index.tz_convert("UTC").normalize()

    # mtf index is already UTC midnight (set by _to_daily_index)
    mtf_aligned = mtf.reindex(utc_idx)
    mtf_aligned.index = daily_df.index

    # Attach new columns (avoid overwriting existing columns)
    new_cols = [c for c in mtf_aligned.columns if c not in daily_df.columns]
    daily_df[new_cols] = mtf_aligned[new_cols]

    return daily_df


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(levelname)s %(message)s")

    dates = pd.date_range("2024-06-01", "2024-08-15", freq="B", tz="UTC")
    df_base = pd.DataFrame({"close": 100.0}, index=dates)

    for tk in ["AAPL"]:
        out = add_multi_timeframe_features(df_base.copy(), tk)
        new = [c for c in out.columns if c not in df_base.columns]
        print(f"\n{tk}: +{len(new)} MTF cols")
        for c in new:
            if pd.api.types.is_numeric_dtype(out[c]):
                non_zero_pct = (out[c].notna() & (out[c] != 0)).mean() * 100
                print(f"  {c}: non-zero {non_zero_pct:.0f}%")
        print("\nSample rows (last 5):")
        print(out[new].tail(5).to_string())
