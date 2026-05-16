"""
intraday_features.py — Intraday-derived features for the S&P 500 daily ML pipeline.

All features are computed from 1-min RTH bars BEFORE the daily resample.
They are attached to the daily bar indexed at that session's date in UTC.

POINT-IN-TIME SAFE: for bar at date t, every computation uses only data with
ET timestamp <= 16:00:00 on day t.  No forward references.

Usage (after build_features in backtest_xgb.py):
    from intraday_features import add_intraday_features
    daily_df = add_intraday_features(daily_df, ticker)
"""

from __future__ import annotations

import os
import warnings
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DATA_ROOT_DEFAULT = (
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/claudes test/data/timeframes/S&P500 5 Year Historical Data"
    "/Minutes TimeFrames/1Min_merged"
)
# Allow CI / GH Actions to override via env var.  When BACKTEST_DATA_SOURCE is
# set to "yfinance_daily" the intraday root will not exist; _load_1min() handles
# that gracefully (returns empty DataFrame) so all callers skip intraday features.
DATA_ROOT = os.environ.get("BACKTEST_DATA_ROOT", _DATA_ROOT_DEFAULT)

ET_TZ = "America/New_York"

# ---------------------------------------------------------------------------
# FOMC meeting weeks lookup  2021-2026
# Each tuple is (year, month, approx_week_of_month) — we mark the full ISO
# week containing that meeting as an FOMC week.
# Source: Federal Reserve historical + projected schedule.
# ---------------------------------------------------------------------------
_FOMC_DATES: list[str] = [
    # 2021
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16",
    "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
    "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-17",
    # 2026
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-11-04", "2026-12-16",
]

# Build a set of ISO (year, week) tuples that are FOMC weeks.
_FOMC_WEEKS: set[tuple[int, int]] = set()
for _d in _FOMC_DATES:
    _ts = pd.Timestamp(_d)
    _FOMC_WEEKS.add((_ts.isocalendar().year, _ts.isocalendar().week))


# ---------------------------------------------------------------------------
# Internal 1-min loader — memoised per ticker
# ---------------------------------------------------------------------------
_CACHE: dict[str, pd.DataFrame] = {}


def _load_1min(ticker: str) -> pd.DataFrame:
    """Load and cache the 1-min parquet for ticker.

    Returns a DataFrame indexed by ET-timezone-aware timestamps, or an empty
    DataFrame if the parquet file does not exist (e.g. when running on GH Actions
    with BACKTEST_DATA_SOURCE=yfinance_daily).
    """
    if ticker in _CACHE:
        return _CACHE[ticker]

    path = os.path.join(DATA_ROOT, f"{ticker}.parquet")
    if not os.path.exists(path):
        warnings.warn(
            f"intraday_features: 1-min parquet not found at {path!r}; "
            "intraday features will be skipped (DATA_SOURCE may be yfinance_daily)."
        )
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        _CACHE[ticker] = empty
        return empty

    raw = pd.read_parquet(path).set_index("timestamp").sort_index()
    # Convert index to ET
    raw.index = raw.index.tz_convert(ET_TZ)
    _CACHE[ticker] = raw
    return raw


# ---------------------------------------------------------------------------
# ATR helper (14-day, rolling on DAILY bars)
# ---------------------------------------------------------------------------

def _daily_atr(daily_rth: pd.DataFrame, period: int = 14) -> pd.Series:
    """True-range ATR on daily OHLC, returned as a Series indexed by date."""
    h = daily_rth["high"]
    l = daily_rth["low"]
    pc = daily_rth["close"].shift(1)
    tr = pd.concat(
        [h - l, (h - pc).abs(), (l - pc).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


# ---------------------------------------------------------------------------
# Per-session feature computation
# ---------------------------------------------------------------------------

def _session_features(et_bars: pd.DataFrame, atr_val: float) -> dict:
    """Compute intraday features for a single RTH session.

    Parameters
    ----------
    et_bars : 1-min bars for this session already filtered to RTH
              (09:30–15:59 ET, inclusive), index is ET-aware timestamps.
    atr_val : 14-day ATR value for this session (from daily bars, prior close).

    Returns dict of scalar feature values.
    """
    feat: dict = {}

    if et_bars.empty or len(et_bars) < 2:
        return feat

    # ---- Opening-range (9:30–9:59) ----------------------------------------
    or_mask = (et_bars.index.hour == 9) & (et_bars.index.minute <= 59)
    or_bars = et_bars[or_mask]
    if not or_bars.empty:
        or_high = or_bars["high"].max()
        or_low = or_bars["low"].min()
        feat["or_high"] = or_high
        feat["or_low"] = or_low
        or_width = or_high - or_low
        feat["or_width_atr"] = or_width / atr_val if atr_val > 0 else np.nan

        session_close = et_bars["close"].iloc[-1]
        if session_close > or_high:
            feat["or_break_dir"] = 1
        elif session_close < or_low:
            feat["or_break_dir"] = -1
        else:
            feat["or_break_dir"] = 0
    else:
        feat.update({"or_high": np.nan, "or_low": np.nan,
                     "or_width_atr": np.nan, "or_break_dir": 0})

    # ---- VWAP ----------------------------------------------------------------
    # Compute cumulative VWAP from session open; native vwap column if present
    # but we recompute to guarantee it's cumulative from RTH open.
    tp = (et_bars["high"] + et_bars["low"] + et_bars["close"]) / 3
    cum_vol = et_bars["volume"].cumsum()
    session_vwap = (tp * et_bars["volume"]).cumsum() / cum_vol.replace(0, np.nan)
    session_vwap_final = session_vwap.iloc[-1]

    session_close = et_bars["close"].iloc[-1]
    feat["close_vwap_dist_atr"] = (
        (session_close - session_vwap_final) / atr_val if atr_val > 0 else np.nan
    )

    # VWAP crosses: count bars where price switches side of VWAP
    above_vwap = et_bars["close"] > session_vwap
    feat["vwap_cross_count"] = int(above_vwap.diff().abs().sum())
    feat["vwap_close_above"] = int(session_close > session_vwap_final)

    # ---- Time-of-day returns -------------------------------------------------
    # first_hour_return_pct: 9:30 open → 10:30 last bar close
    open_930 = et_bars["open"].iloc[0]
    first_hour_mask = (
        (et_bars.index.hour == 9) & (et_bars.index.minute >= 30)
    ) | (
        (et_bars.index.hour == 10) & (et_bars.index.minute <= 29)
    )
    first_hour_bars = et_bars[first_hour_mask]
    if not first_hour_bars.empty and open_930 != 0:
        feat["first_hour_return_pct"] = (
            first_hour_bars["close"].iloc[-1] - open_930
        ) / open_930
    else:
        feat["first_hour_return_pct"] = np.nan

    # last_hour_return_pct: 15:00 open → 15:59 close
    last_hour_mask = et_bars.index.hour == 15
    last_hour_bars = et_bars[last_hour_mask]
    if not last_hour_bars.empty:
        lh_open = last_hour_bars["open"].iloc[0]
        lh_close = last_hour_bars["close"].iloc[-1]
        feat["last_hour_return_pct"] = (
            (lh_close - lh_open) / lh_open if lh_open != 0 else np.nan
        )
    else:
        feat["last_hour_return_pct"] = np.nan

    # lunch_hour_range_pct: 12:00–12:59 high-low as % of close
    lunch_mask = et_bars.index.hour == 12
    lunch_bars = et_bars[lunch_mask]
    if not lunch_bars.empty and session_close != 0:
        feat["lunch_hour_range_pct"] = (
            lunch_bars["high"].max() - lunch_bars["low"].min()
        ) / session_close
    else:
        feat["lunch_hour_range_pct"] = np.nan

    # first_30min_volume_pct
    first_30_mask = (et_bars.index.hour == 9) & (et_bars.index.minute >= 30)
    first_30_vol = et_bars[first_30_mask]["volume"].sum()
    total_vol = et_bars["volume"].sum()
    feat["first_30min_volume_pct"] = (
        first_30_vol / total_vol if total_vol > 0 else np.nan
    )

    # ---- Volatility profile --------------------------------------------------
    feat["intraday_atr_pct"] = (
        (et_bars["high"].max() - et_bars["low"].min()) / session_close
        if session_close != 0
        else np.nan
    )

    # realized_vol_5min: sum of squared 5-min log returns
    five_min = et_bars["close"].resample("5min").last().dropna()
    log_rets_5 = np.log(five_min / five_min.shift(1)).dropna()
    feat["realized_vol_5min"] = float((log_rets_5 ** 2).sum())

    # vol_concentration: max single-bar volume / mean bar volume
    mean_vol = et_bars["volume"].mean()
    feat["vol_concentration"] = (
        et_bars["volume"].max() / mean_vol if mean_vol > 0 else np.nan
    )

    return feat


# ---------------------------------------------------------------------------
# Gap features helper (operates on the daily frame)
# ---------------------------------------------------------------------------

def _gap_features(daily_min: pd.DataFrame) -> pd.DataFrame:
    """Compute gap features using 9:30 open and 15:59 close from 1-min data.

    daily_min is the full 1-min ET-indexed DataFrame for the ticker.
    Returns a DataFrame indexed by ET date with gap columns.
    """
    # 9:30 bar open for each day (first bar of RTH)
    rth = daily_min[
        ((daily_min.index.hour > 9) | ((daily_min.index.hour == 9) & (daily_min.index.minute >= 30)))
        & (daily_min.index.hour < 16)
    ]

    # group by date
    dates = rth.index.normalize()

    # today's open = first bar's open per day
    today_open = rth["open"].groupby(dates).first()

    # yesterday's close = last bar's close per day (15:59)
    yest_close_raw = rth["close"].groupby(dates).last()

    gap_df = pd.DataFrame({"today_open": today_open, "yest_close": yest_close_raw})
    gap_df["prev_close"] = gap_df["yest_close"].shift(1)
    gap_df["prev_open"] = gap_df["today_open"].shift(1)  # yesterday's open

    # overnight_gap_pct
    gap_df["overnight_gap_pct"] = (
        (gap_df["today_open"] - gap_df["prev_close"]) / gap_df["prev_close"]
    )

    # gap_filled: did today's price fill yesterday's gap?
    # Gap up (today_open > prev_close): filled if today's low <= prev_close
    # Gap down (today_open < prev_close): filled if today's high >= prev_close
    today_low = rth["low"].groupby(dates).min()
    today_high = rth["high"].groupby(dates).max()
    gap_df["today_low"] = today_low
    gap_df["today_high"] = today_high

    gap_up = gap_df["today_open"] > gap_df["prev_close"]
    gap_down = gap_df["today_open"] < gap_df["prev_close"]
    filled = (
        (gap_up & (gap_df["today_low"] <= gap_df["prev_close"])) |
        (gap_down & (gap_df["today_high"] >= gap_df["prev_close"]))
    ).astype(int)
    gap_df["gap_filled"] = filled

    # consecutive_gap_dir
    gap_sign = np.sign(gap_df["overnight_gap_pct"].fillna(0))
    consec = []
    streak = 0
    prev_sign = 0
    for s in gap_sign:
        if s == prev_sign and s != 0:
            streak += 1
        elif s != 0:
            streak = 1
        else:
            streak = 0
        consec.append(streak * int(prev_sign if s == 0 else s))
        prev_sign = s if s != 0 else prev_sign
    gap_df["consecutive_gap_dir"] = consec

    return gap_df[["overnight_gap_pct", "gap_filled", "consecutive_gap_dir"]]


# ---------------------------------------------------------------------------
# Calendar features helper
# ---------------------------------------------------------------------------

def _calendar_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Compute calendar features for a DatetimeIndex of ET session dates (tz-naive).

    The input dates are the actual ET calendar dates of each trading session
    (e.g. 2024-06-03 for the June 3rd session), derived from 1-min data.
    """
    rows = []
    # Normalize to midnight (dates are already normalized, but be safe)
    et_dates = dates.normalize()

    # Build a set of all trading dates for end-of-month / end-of-quarter checks
    all_trading = sorted(set(et_dates.date))
    # map date -> index in sorted list
    date_rank = {d: i for i, d in enumerate(all_trading)}

    # EOM: last 3 trading days of each calendar month
    eom_dates: set = set()
    from itertools import groupby as _groupby
    by_ym = {}
    for d in all_trading:
        key = (d.year, d.month)
        by_ym.setdefault(key, []).append(d)
    for ym, days in by_ym.items():
        for d in sorted(days)[-3:]:
            eom_dates.add(d)

    # EOQ: last 3 trading days of each calendar quarter (Mar, Jun, Sep, Dec)
    eoq_dates: set = set()
    by_yq = {}
    for d in all_trading:
        q = (d.month - 1) // 3
        key = (d.year, q)
        by_yq.setdefault(key, []).append(d)
    for yq, days in by_yq.items():
        for d in sorted(days)[-3:]:
            eoq_dates.add(d)

    # OPEX: third Friday of each month (monthly options expiration)
    def _is_opex_week(d: pd.Timestamp) -> int:
        """Return 1 if d falls in the same ISO week as the third Friday of d's month."""
        # Find third Friday
        first_day = pd.Timestamp(d.year, d.month, 1)
        # weekday(): Monday=0, Friday=4
        days_to_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day + pd.Timedelta(days=days_to_friday)
        third_friday = first_friday + pd.Timedelta(weeks=2)
        return int(d.isocalendar().week == third_friday.isocalendar().week
                   and d.year == third_friday.year)

    for ts in et_dates:
        d = ts.date()
        dow = ts.dayofweek  # 0=Monday
        iso = ts.isocalendar()
        fomc = int((iso.year, iso.week) in _FOMC_WEEKS)
        rows.append({
            "dow": dow,
            "is_monday": int(dow == 0),
            "is_friday": int(dow == 4),
            "is_opex_week": _is_opex_week(ts),
            "is_eom": int(d in eom_dates),
            "is_eoq": int(d in eoq_dates),
            "is_fomc_week": fomc,
        })

    return pd.DataFrame(rows, index=et_dates)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_intraday_features(
    daily_df: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    """Reads the 1-min parquet for ticker, computes per-session intraday features,
    joins to the existing daily_df (whose index is daily timestamps).
    Returns daily_df with new columns added.

    NaN-fill with 0 except dow which is always present.

    Parameters
    ----------
    daily_df : DataFrame with a UTC-aware DatetimeIndex (one row per trading day).
    ticker   : Ticker symbol string matching the parquet filename.

    Returns
    -------
    daily_df with intraday feature columns appended.

    POINT-IN-TIME SAFETY
    --------------------
    Every per-session feature uses only 1-min bars with ET timestamp in
    [09:30, 16:00) on that session date.  Gap features use the previous
    session's 15:59 close and today's 09:30 open — both are available at
    market close of today.  No bar from a future session is ever accessed.
    """
    # ---- Load 1-min data -----------------------------------------------
    try:
        min1 = _load_1min(ticker)
    except FileNotFoundError:
        warnings.warn(f"intraday_features: no 1-min parquet for {ticker}, skipping.")
        return daily_df

    # Empty frame means the parquet file wasn't found (e.g. yfinance_daily mode).
    if min1.empty:
        return daily_df

    # ---- Daily ATR for normalization ------------------------------------
    # We need daily OHLC to compute 14-day ATR.  Build it from RTH bars.
    rth_all = min1[
        ((min1.index.hour > 9) | ((min1.index.hour == 9) & (min1.index.minute >= 30)))
        & (min1.index.hour < 16)
    ]
    dates_et = rth_all.index.normalize()
    daily_ohlcv = pd.DataFrame({
        "open":  rth_all["open"].groupby(dates_et).first(),
        "high":  rth_all["high"].groupby(dates_et).max(),
        "low":   rth_all["low"].groupby(dates_et).min(),
        "close": rth_all["close"].groupby(dates_et).last(),
        "volume": rth_all["volume"].groupby(dates_et).sum(),
    })
    daily_ohlcv.index = pd.to_datetime(daily_ohlcv.index)
    atr_series = _daily_atr(daily_ohlcv)  # indexed by ET date, no shift needed here
    # Shift ATR by 1 so bar t uses ATR computed through day t-1 (point-in-time safe)
    atr_series_shifted = atr_series.shift(1)

    # ---- Per-session intraday features ----------------------------------
    session_records: list[dict] = []

    for date, group in rth_all.groupby(rth_all.index.normalize()):
        atr_val = atr_series_shifted.get(date, np.nan)
        if pd.isna(atr_val) or atr_val <= 0:
            atr_val = (daily_ohlcv.loc[date, "high"] - daily_ohlcv.loc[date, "low"]
                       if date in daily_ohlcv.index else 1.0)
        feat = _session_features(group, atr_val)
        feat["_date"] = date
        session_records.append(feat)

    intra_df = pd.DataFrame(session_records).set_index("_date")
    intra_df.index = pd.to_datetime(intra_df.index)

    # Drop raw or_high / or_low from features (internal use only, not features)
    intra_df = intra_df.drop(columns=["or_high", "or_low"], errors="ignore")

    # ---- Gap features ---------------------------------------------------
    gap_df = _gap_features(min1)
    gap_df.index = pd.to_datetime(gap_df.index)
    intra_df = intra_df.join(gap_df, how="left")

    # ---- Build the join key ---------------------------------------------
    # load_daily() uses resample('1D', closed='left', label='left') in UTC,
    # so the daily index is midnight UTC on the session's calendar date
    # (e.g. 2024-06-03 00:00 UTC == the June 3 trading session).
    # The 1-min parquet ET dates (from .index.normalize() in ET) match
    # exactly the UTC calendar date of those midnight UTC timestamps.
    # Therefore: join key = UTC date of the daily_df index, NOT the ET
    # conversion of that midnight timestamp (which backs up one calendar day).

    daily_df = daily_df.copy()
    # Save the original UTC-aware DatetimeIndex so we can restore it after merge
    original_index = daily_df.index
    daily_df["_utc_date"] = original_index.normalize().tz_localize(None)

    # Ensure intra_df and cal_df indices are tz-naive date-only
    intra_df.index = (
        intra_df.index.tz_localize(None) if intra_df.index.tz is not None else intra_df.index
    )
    # intra_df index is ET dates from 1-min data; these equal the UTC calendar
    # date of the session (both are the same calendar date — 2024-06-03).
    # So the join on _utc_date vs intra_df.index is correct.

    # ---- Calendar features ----------------------------------------------
    # Calendar should use the actual ET session dates (from 1-min data),
    # not the daily_df UTC midnight dates, so we build it from intra_df.index.
    cal_df = _calendar_features(pd.DatetimeIndex(intra_df.index))
    cal_df.index = (
        cal_df.index.tz_localize(None) if cal_df.index.tz is not None else cal_df.index
    )

    # ---- Join intraday + calendar features to daily_df ------------------
    daily_df = daily_df.merge(
        intra_df, left_on="_utc_date", right_index=True, how="left"
    )
    daily_df = daily_df.merge(
        cal_df, left_on="_utc_date", right_index=True, how="left"
    )
    daily_df = daily_df.drop(columns=["_utc_date"])

    # Restore the original UTC-aware DatetimeIndex (merge resets to RangeIndex)
    daily_df.index = original_index

    # ---- NaN fill -------------------------------------------------------
    # dow: derive from 1-min session dates where missing (should be rare)
    if "dow" in daily_df.columns:
        missing_dow = daily_df["dow"].isna()
        if missing_dow.any():
            # Use UTC calendar date (= session date) for dow
            daily_df.loc[missing_dow, "dow"] = (
                daily_df.index[missing_dow].normalize().dayofweek
            )

    # All other new intraday columns: fill NaN with 0
    original_cols = {"open", "high", "low", "close", "volume"}
    new_cols = [c for c in daily_df.columns if c not in original_cols and c != "dow"]
    daily_df[new_cols] = daily_df[new_cols].fillna(0)

    return daily_df


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd

    test_dates = pd.date_range("2024-06-01", "2024-06-28", freq="B", tz="UTC")
    df = pd.DataFrame({"close": 100.0}, index=test_dates)
    for tk in ["AAPL", "NVDA", "XOM"]:
        out = add_intraday_features(df.copy(), tk)
        new_cols = [c for c in out.columns if c not in df.columns]
        print(f"\n=== {tk} ({len(new_cols)} new cols) ===")
        print(out[new_cols].iloc[5:8].to_string())
        print("non-zero pct per col:")
        for c in new_cols:
            if out[c].dtype != object:
                print(f"  {c}: {(out[c] != 0).mean() * 100:.0f}%")
