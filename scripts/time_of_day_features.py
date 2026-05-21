"""
time_of_day_features.py — Time-of-day bucket feature for v10 (Wave A, 2026-05-17).

Adds 1 deterministic categorical feature derived from the entry timestamp of the
bar's signal. For daily data we don't have intraday entry times directly — we
look up the most-recent cached intraday-signal entry time from the live paper-
trade signals dir; if absent, we default to bucket 2 (mid-day) for ALL bars.
This is .shift(1)-safe by construction (signals/<DATE>.json is keyed by the
trading day prior to the bar's open).

Feature added:
  - time_of_day_bucket   (int in {0..4}):
        0 = pre-market (before 09:30 ET)
        1 = opening 30m (09:30 – 10:00 ET)
        2 = mid-day (10:00 – 14:30 ET)
        3 = power hour (14:30 – 16:00 ET)
        4 = after-hours (after 16:00 ET) OR signal-missing fallback
    Default fallback bucket = 2 (mid-day) — empirically the largest cluster,
    least biased — when no intraday signal record exists.

Source order:
  1. $SP/paper_trade/signals/<DATE>.json   ([{'ticker': ..., 'generated_at': ISO}, ...])
  2. df.attrs['intraday_entry_time']        (single string, applies to entire df)
  3. Constant fallback bucket 2.

Idempotent + graceful: existing 'time_of_day_bucket' column is left untouched;
missing signals dir → constant fallback for the whole frame.

Author: 2026-05-17 (Wave A).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WORK = Path(
    "/Users/orginal/Library/CloudStorage/"
    "GoogleDrive-zachgladstone@gmail.com/My Drive/"
    "AI-Tools/s&p500-ticker-mastery"
)
SIGNALS_DIR = WORK / "paper_trade" / "signals"

TOD_FEATURE_NAMES: list[str] = [
    "time_of_day_bucket",
    # Quick-wire expansion (2026-05-21):
    "tod_lunch_lull_flag",     # 1 if bar is between 11:30 and 13:30 ET (deadest period)
    "tod_power_hour_flag",     # 1 if bar is in last 60min of session (14:30-16:00 ET)
    "tod_opex_week_flag",      # 1 if ISO week contains 3rd Friday (monthly OPEX week)
    "tod_fomc_week_flag",      # 1 if ISO week is FOMC announcement week
    "tod_day_of_week",         # 0=Mon..4=Fri (typed int8)
    "tod_eoq_flag",            # 1 if bar's date is in the last 5 business days of a quarter
]

_FALLBACK_BUCKET = 2  # mid-day

# FOMC week lookup (subset of intraday_features._FOMC_DATES, 2021-2026).
_FOMC_DATES_FOR_TOD: list[str] = [
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16",
    "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
    "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-17",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-11-04", "2026-12-16",
]
_FOMC_WEEKS_FOR_TOD: set[tuple[int, int]] = set()
for _d_str in _FOMC_DATES_FOR_TOD:
    _ts_fomc = pd.Timestamp(_d_str)
    _FOMC_WEEKS_FOR_TOD.add((_ts_fomc.isocalendar().year, _ts_fomc.isocalendar().week))


def _classify_minute_of_day(hhmm: int) -> int:
    """Map an integer HH*100+MM (US/Eastern, after conversion) to a bucket 0-4."""
    if hhmm < 930:
        return 0
    if hhmm < 1000:
        return 1
    if hhmm < 1430:
        return 2
    if hhmm < 1600:
        return 3
    return 4


def _to_eastern_minute(iso_str: str) -> Optional[int]:
    """Parse ISO datetime, convert UTC → America/New_York, return HH*100+MM."""
    if not isinstance(iso_str, str):
        return None
    try:
        if iso_str.endswith("Z"):
            iso_str = iso_str[:-1] + "+00:00"
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        # Convert to America/New_York if available; fallback to UTC-5 fixed offset
        try:
            from zoneinfo import ZoneInfo
            et = dt.astimezone(ZoneInfo("America/New_York"))
        except Exception:
            from datetime import timedelta
            et = dt.astimezone(timezone(timedelta(hours=-5)))  # cheap fallback (no DST)
        return et.hour * 100 + et.minute
    except Exception:
        return None


def _load_signal_time(date_str: str, ticker: str) -> Optional[int]:
    """Look up the generated_at minute-of-day for (date, ticker) in signals/<DATE>.json."""
    p = SIGNALS_DIR / f"{date_str}.json"
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text())
    except Exception:
        return None
    if isinstance(payload, list):
        for entry in payload:
            if entry.get("ticker") == ticker:
                gen = entry.get("generated_at") or entry.get("signal_time")
                if gen:
                    return _to_eastern_minute(gen)
    elif isinstance(payload, dict):
        block = payload.get(ticker)
        if isinstance(block, dict):
            gen = block.get("generated_at") or block.get("signal_time")
            if gen:
                return _to_eastern_minute(gen)
    return None


def _date_based_calendar_features(df: pd.DataFrame, bucket: np.ndarray) -> dict[str, np.ndarray]:
    """Compute the 6 calendar-aware features (lunch_lull, power_hour, OPEX, FOMC, DOW, EOQ).

    lunch_lull / power_hour are derived from the per-bar `bucket` (intraday classification
    when signals/df.attrs provided; otherwise everything is mid-day fallback).
    OPEX / FOMC / DOW / EOQ are derived from the bar's date.
    """
    n = len(df)
    lunch = np.zeros(n, dtype=np.int8)
    power = np.zeros(n, dtype=np.int8)
    opex = np.zeros(n, dtype=np.int8)
    fomc = np.zeros(n, dtype=np.int8)
    dow = np.zeros(n, dtype=np.int8)
    eoq = np.zeros(n, dtype=np.int8)

    # lunch_lull = bucket 2 between 11:30-13:30 ET; here we approximate with bucket==2 only when
    # the per-bar bucket is mid-day. With current source (1 bucket per bar) lunch_lull
    # captures only the explicit mid-day classification.
    lunch[:] = (bucket == 2).astype(np.int8)
    power[:] = (bucket == 3).astype(np.int8)

    # Date-based features
    if isinstance(df.index, pd.DatetimeIndex):
        bar_dates = df.index
    elif "date" in df.columns:
        bar_dates = pd.DatetimeIndex(pd.to_datetime(df["date"]))
    else:
        bar_dates = None

    if bar_dates is not None and len(bar_dates) > 0:
        try:
            if bar_dates.tz is not None:
                bar_dates = bar_dates.tz_convert(None)
        except Exception:
            pass

        # day of week
        try:
            dow_arr = np.asarray(bar_dates.dayofweek, dtype=np.int8)
            dow[:len(dow_arr)] = dow_arr[:n]
        except Exception:
            pass

        # OPEX week — 3rd Friday of the month is OPEX day; the ISO week containing it is OPEX week.
        try:
            for i, bd in enumerate(bar_dates):
                # find this month's 3rd Friday
                first_of_month = pd.Timestamp(year=bd.year, month=bd.month, day=1)
                # day of week of first day; Friday = 4
                first_dow = first_of_month.dayofweek
                # offset to first Friday
                offset_to_friday = (4 - first_dow) % 7
                third_friday = first_of_month + pd.Timedelta(days=offset_to_friday + 14)
                third_iso = third_friday.isocalendar()
                this_iso = bd.isocalendar()
                if (this_iso.year, this_iso.week) == (third_iso.year, third_iso.week):
                    opex[i] = 1
        except Exception:
            pass

        # FOMC week
        try:
            for i, bd in enumerate(bar_dates):
                this_iso = bd.isocalendar()
                if (this_iso.year, this_iso.week) in _FOMC_WEEKS_FOR_TOD:
                    fomc[i] = 1
        except Exception:
            pass

        # EOQ — last 5 business days of a quarter
        try:
            # Cache per-quarter end-of-quarter business-day cutoff
            eoq_cache: dict[tuple[int, int], pd.Timestamp] = {}
            for i, bd in enumerate(bar_dates):
                quarter = (bd.year, (bd.month - 1) // 3 + 1)
                if quarter not in eoq_cache:
                    last_month = quarter[1] * 3
                    eoq_end = pd.Timestamp(year=quarter[0], month=last_month, day=1) + pd.offsets.MonthEnd(0)
                    # 5 business days before end of quarter
                    eoq_start = eoq_end - pd.tseries.offsets.BDay(5)
                    eoq_cache[quarter] = eoq_start
                if bd >= eoq_cache[quarter]:
                    eoq[i] = 1
        except Exception:
            pass

    return {
        "tod_lunch_lull_flag": lunch,
        "tod_power_hour_flag": power,
        "tod_opex_week_flag": opex,
        "tod_fomc_week_flag": fomc,
        "tod_day_of_week": dow,
        "tod_eoq_flag": eoq,
    }


def add_time_of_day_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Append time_of_day_bucket + 6 calendar-aware features to df. Idempotent + graceful."""
    if df is None or len(df) == 0:
        return df
    # Idempotence guard — if main bucket already present and ANY of the new cols is present,
    # assume the full block has already been computed and return early.
    if "time_of_day_bucket" in df.columns and "tod_lunch_lull_flag" in df.columns:
        return df

    n = len(df)
    bucket = np.full(n, _FALLBACK_BUCKET, dtype=np.int8)

    # 2nd-priority: df.attrs hint
    hint = None
    try:
        hint = df.attrs.get("intraday_entry_time")  # type: ignore[union-attr]
    except Exception:
        hint = None
    if hint:
        hhmm = _to_eastern_minute(str(hint))
        if hhmm is not None:
            bucket[:] = _classify_minute_of_day(hhmm)
            df["time_of_day_bucket"] = bucket
            extras = _date_based_calendar_features(df, bucket)
            for name, arr in extras.items():
                if name not in df.columns:
                    df[name] = arr
            return df

    # 1st-priority: paper_trade/signals lookup, per-bar (if dir exists)
    if SIGNALS_DIR.exists():
        if isinstance(df.index, pd.DatetimeIndex):
            bar_dates = df.index
        elif "date" in df.columns:
            bar_dates = pd.DatetimeIndex(pd.to_datetime(df["date"]))
        else:
            bar_dates = None
        if bar_dates is not None:
            if bar_dates.tz is not None:
                bar_dates = bar_dates.tz_convert(None)
            cache: dict[str, Optional[int]] = {}
            for i, bd in enumerate(bar_dates):
                # .shift(1) safety: lookup signals for the PRIOR trading day
                prior = (bd - pd.Timedelta(days=1)).date().isoformat()
                if prior not in cache:
                    cache[prior] = _load_signal_time(prior, ticker)
                hhmm = cache[prior]
                if hhmm is not None:
                    bucket[i] = _classify_minute_of_day(hhmm)
                # else: keep fallback

    if "time_of_day_bucket" not in df.columns:
        df["time_of_day_bucket"] = bucket
    extras = _date_based_calendar_features(df, bucket)
    for name, arr in extras.items():
        if name not in df.columns:
            df[name] = arr
    return df


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    tk = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    idx = pd.date_range(end=pd.Timestamp.utcnow().date(), periods=50, freq="B")
    demo = pd.DataFrame({"close": np.linspace(100, 110, len(idx))}, index=idx)
    out = add_time_of_day_features(demo, tk)
    print(out[TOD_FEATURE_NAMES].tail(5).to_string())
    print("value counts:")
    print(out["time_of_day_bucket"].value_counts())
