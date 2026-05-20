"""ORB-15min — opening range breakout (long-only).

Signal=1 when current close breaks above the 15-min opening range high AND the
current 1-min bar volume is >= 1.5x the rolling-20-bar average. No entries
after 15:30 ET (engine flattens any remainder at 15:55 ET).

prob = clamp((close - OR_high) / OR_range, 0, 1)  -- distance past breakout
TP   = entry + 2 * ATR14
SL   = entry - 1 * ATR14
trail-arm = entry + 1 * ATR14
"""
from __future__ import annotations

from datetime import datetime, time
from typing import Optional

import pandas as pd
import pytz

from . import (
    ATR_PERIOD,
    OR_MINUTES,
    RTH_OPEN,
    SL_ATR_MULT,
    TP_ATR_MULT,
    TRAIL_ARM_ATR_MULT,
    compute_atr14,
)

ET = pytz.timezone("America/New_York")
_CUTOFF = time(15, 30)
_VOL_SPIKE_MULT = 1.5
_VOL_ROLL = 20
_MIN_BARS = OR_MINUTES + 1


def _empty(ticker: str, reason: str, meta: Optional[dict] = None) -> dict:
    return dict(
        strategy_id="orb_15min",
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
    if _now.time() >= _CUTOFF:
        return _empty(ticker, f"past_cutoff_{_CUTOFF}")

    or_bars = session.iloc[:OR_MINUTES]
    or_high = float(or_bars["high"].max())
    or_low = float(or_bars["low"].min())
    or_range = or_high - or_low
    if or_range <= 0:
        return _empty(ticker, "zero_or_range")

    cur_close = float(session["close"].iloc[-1])
    cur_vol = float(session["volume"].iloc[-1])
    vol_mean = float(
        session["volume"].rolling(_VOL_ROLL, min_periods=1).mean().iloc[-1]
    )
    vol_ratio = cur_vol / vol_mean if vol_mean > 0 else 0.0
    atr = compute_atr14(df, ATR_PERIOD)

    meta = dict(
        OR_high=round(or_high, 4),
        OR_low=round(or_low, 4),
        OR_range=round(or_range, 4),
        vol_spike_ratio=round(vol_ratio, 3),
        atr14=round(atr, 4),
    )

    if cur_close > or_high and vol_ratio >= _VOL_SPIKE_MULT:
        prob = min(1.0, (cur_close - or_high) / or_range)
        entry = cur_close
        return dict(
            strategy_id="orb_15min",
            ticker=ticker,
            signal=1,
            prob=round(prob, 4),
            entry=round(entry, 4),
            target=round(entry + TP_ATR_MULT * atr, 4),
            stop=round(entry - SL_ATR_MULT * atr, 4),
            trailing_stop_arm_at=round(entry + TRAIL_ARM_ATR_MULT * atr, 4),
            reason=(
                f"broke OR-high by "
                f"{round((cur_close - or_high) / or_range, 3):.3f}*OR_range "
                f"with vol_spike {vol_ratio:.2f}x"
            ),
            meta=meta,
        )

    parts = []
    if cur_close <= or_high:
        parts.append(f"close {cur_close:.4f} <= OR_high {or_high:.4f}")
    if vol_ratio < _VOL_SPIKE_MULT:
        parts.append(f"vol_spike {vol_ratio:.2f}x < {_VOL_SPIKE_MULT}x")
    return _empty(ticker, "; ".join(parts) or "no_breakout", meta)
