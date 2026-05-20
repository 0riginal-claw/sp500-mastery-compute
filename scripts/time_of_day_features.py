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

TOD_FEATURE_NAMES: list[str] = ["time_of_day_bucket"]

_FALLBACK_BUCKET = 2  # mid-day


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


def add_time_of_day_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Append 'time_of_day_bucket' (int 0-4) to df. Idempotent + graceful."""
    if df is None or len(df) == 0:
        return df
    if "time_of_day_bucket" in df.columns:
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
                # (signals/<DATE>.json's entry timestamp couldn't have informed bar D
                # unless DATE < D).
                prior = (bd - pd.Timedelta(days=1)).date().isoformat()
                if prior not in cache:
                    cache[prior] = _load_signal_time(prior, ticker)
                hhmm = cache[prior]
                if hhmm is not None:
                    bucket[i] = _classify_minute_of_day(hhmm)
                # else: keep fallback

    df["time_of_day_bucket"] = bucket
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
