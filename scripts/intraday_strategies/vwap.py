"""VWAP mean-reversion — fade when close < VWAP - 0.5sigma (long-only).

Window: 10:00-15:00 ET (avoid open noise + close-flatten clash).

target = VWAP (revert to mean), stop = entry - 1*ATR14 (fallback).
prob = clamp(distance_in_sigmas, 0, 1).
"""
from __future__ import annotations

from datetime import datetime, time
from typing import Optional

import pandas as pd
import pytz

from . import (
    ATR_PERIOD,
    RTH_OPEN,
    SL_ATR_MULT,
    TRAIL_ARM_ATR_MULT,
    compute_atr14,
)

ET = pytz.timezone("America/New_York")
_ENTRY_START = time(10, 0)
_ENTRY_END = time(15, 0)
_STD_ROLL = 30
_SIGMA_THRESH = 0.5
_MIN_BARS = 30


def _empty(ticker: str, reason: str, meta: Optional[dict] = None) -> dict:
    return dict(
        strategy_id="vwap_mean_revert",
        ticker=ticker,
        signal=0,
        prob=0.0,
        entry=None,
        target=None,
        stop=None,
        trailing_stop_arm_at=None,
        reason=reason,
        meta=meta or {},
    )


def score(
    bars_1min: pd.DataFrame,
    *,
    ticker: str,
    params: Optional[dict] = None,
    now_et: Optional[datetime] = None,
) -> dict:
    if bars_1min is None or len(bars_1min) < _MIN_BARS:
        return _empty(ticker, "insufficient_bars")

    df = bars_1min.copy()
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(ET)
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert(ET)

    current_date = df["timestamp"].iloc[-1].date()
    session_open_ts = ET.localize(datetime.combine(current_date, RTH_OPEN))
    session = df[df["timestamp"] >= session_open_ts].reset_index(drop=True)
    if len(session) < _MIN_BARS:
        return _empty(ticker, "insufficient_bars")

    _now = now_et if now_et is not None else df["timestamp"].iloc[-1].to_pydatetime()
    if getattr(_now, "tzinfo", None) is None:
        _now = ET.localize(_now)
    ct = _now.time()
    if not (_ENTRY_START <= ct < _ENTRY_END):
        return _empty(ticker, f"outside_window_{_ENTRY_START}_{_ENTRY_END}")

    cum_vol = session["volume"].cumsum()
    vwap_series = (
        (session["close"] * session["volume"]).cumsum()
        / cum_vol.replace(0, float("nan"))
    )
    vwap = float(vwap_series.iloc[-1])
    sigma = float(
        (session["close"] - vwap_series).rolling(_STD_ROLL, min_periods=1)
        .std()
        .iloc[-1]
    )
    if sigma == 0 or (sigma != sigma):  # zero or NaN
        return _empty(ticker, "zero_sigma")

    cur_close = float(session["close"].iloc[-1])
    dist_sigmas = (vwap - cur_close) / sigma
    atr = compute_atr14(df, ATR_PERIOD)

    meta = dict(
        vwap=round(vwap, 4),
        sigma=round(sigma, 4),
        distance_sigmas=round(dist_sigmas, 3),
        atr14=round(atr, 4),
    )

    if cur_close < vwap - _SIGMA_THRESH * sigma:
        prob = min(1.0, float(max(0.0, dist_sigmas)))
        entry = cur_close
        return dict(
            strategy_id="vwap_mean_revert",
            ticker=ticker,
            signal=1,
            prob=round(prob, 4),
            entry=round(entry, 4),
            target=round(vwap, 4),
            stop=round(entry - SL_ATR_MULT * atr, 4),
            trailing_stop_arm_at=round(entry + TRAIL_ARM_ATR_MULT * atr, 4),
            reason=(
                f"close {_SIGMA_THRESH}sigma below VWAP: "
                f"distance={dist_sigmas:.3f}sigma"
            ),
            meta=meta,
        )

    return _empty(
        ticker,
        f"close not below VWAP-{_SIGMA_THRESH}sigma (distance={dist_sigmas:.3f}sigma)",
        meta,
    )
