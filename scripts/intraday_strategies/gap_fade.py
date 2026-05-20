"""Gap-fade-open — fade overnight gap >2% in first 30min (long-only).

Long-only stack rejects gap-up shorts (signal=0 with reason='gap_up_no_short').
Gap-down >= -2%: signal=1 LONG, target = prev_close (full mean-revert).

prev_close source priority:
    1. params['prev_close'] (caller-supplied)
    2. Last bar in df with timestamp < today's 09:30 ET (pre-market or prior session)
    3. Else: signal=0 with reason='no_prev_close_available'
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
_ENTRY_END = time(10, 0)
_GAP_THRESH = 0.02
_PROB_MAX_GAP = 0.05


def _empty(ticker: str, reason: str, meta: Optional[dict] = None) -> dict:
    return dict(
        strategy_id="gap_fade_open",
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
    if bars_1min is None or len(bars_1min) < 1:
        return _empty(ticker, "insufficient_bars")

    df = bars_1min.copy()
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(ET)
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert(ET)

    _now = now_et if now_et is not None else df["timestamp"].iloc[-1].to_pydatetime()
    if getattr(_now, "tzinfo", None) is None:
        _now = ET.localize(_now)
    ct = _now.time()
    if not (RTH_OPEN <= ct < _ENTRY_END):
        return _empty(ticker, f"outside_gap_fade_window_{RTH_OPEN}_{_ENTRY_END}")

    current_date = df["timestamp"].iloc[-1].date()
    session_open_ts = ET.localize(datetime.combine(current_date, RTH_OPEN))
    today_bars = df[df["timestamp"] >= session_open_ts]
    if len(today_bars) == 0:
        return _empty(ticker, "no_rth_bars")
    today_open = float(today_bars["open"].iloc[0])

    _params = params or {}
    prev_close = _params.get("prev_close")
    if prev_close is None:
        pre_open = df[df["timestamp"] < session_open_ts]
        if len(pre_open) == 0:
            return _empty(ticker, "no_prev_close_available")
        prev_close = float(pre_open["close"].iloc[-1])
    prev_close = float(prev_close)
    if prev_close <= 0:
        return _empty(ticker, "invalid_prev_close")

    gap_pct = (today_open - prev_close) / prev_close
    abs_gap = abs(gap_pct)
    atr = compute_atr14(df, ATR_PERIOD)
    cur_close = float(df["close"].iloc[-1])

    meta = dict(
        gap_pct=round(gap_pct, 5),
        prev_close=round(prev_close, 4),
        today_open=round(today_open, 4),
        atr14=round(atr, 4),
    )

    if abs_gap < _GAP_THRESH:
        return _empty(
            ticker, f"gap_too_small: {abs_gap:.3%} < {_GAP_THRESH:.0%}", meta
        )

    prob = min(1.0, abs_gap / _PROB_MAX_GAP)

    if gap_pct > 0:
        # Long-only stack — gap-up requires shorting, which we disable.
        return _empty(
            ticker, f"gap_up_no_short: gap={gap_pct:.3%} (long-only)", meta
        )

    entry = cur_close
    return dict(
        strategy_id="gap_fade_open",
        ticker=ticker,
        signal=1,
        prob=round(prob, 4),
        entry=round(entry, 4),
        target=round(prev_close, 4),
        stop=round(entry - SL_ATR_MULT * atr, 4),
        trailing_stop_arm_at=round(entry + TRAIL_ARM_ATR_MULT * atr, 4),
        reason=f"fade gap_down {gap_pct:.3%} back to prev_close {prev_close:.4f}",
        meta=meta,
    )
