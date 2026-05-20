"""Momentum-15min — 3 consecutive green 15-min bars + volume spike (long-only).

Resamples 1-min bars to completed 15-min bars (rejects partial trailing bar).
Signal=1 when last 3 completed 15-min bars are green AND last bar volume >=
1.5x rolling-20 avg. No entries after 15:00 ET (need room for TP to fire
before flatten).

prob = clamp(cum_return_of_3_bars / 0.01, 0, 1)  -- 1% in 45min = prob 1.0
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
    TP_ATR_MULT,
    TRAIL_ARM_ATR_MULT,
    compute_atr14,
)

ET = pytz.timezone("America/New_York")
_CUTOFF = time(15, 0)
_VOL_ROLL = 20
_VOL_SPIKE_MULT = 1.5
_MIN_15MIN = 3
_PROB_MAX_RETURN = 0.01


def _empty(ticker: str, reason: str, meta: Optional[dict] = None) -> dict:
    return dict(
        strategy_id="momentum_15min",
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
    if bars_1min is None or len(bars_1min) < 20:
        return _empty(ticker, "insufficient_bars")

    df = bars_1min.copy()
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(ET)
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert(ET)

    _now = now_et if now_et is not None else df["timestamp"].iloc[-1].to_pydatetime()
    if getattr(_now, "tzinfo", None) is None:
        _now = ET.localize(_now)
    if _now.time() >= _CUTOFF:
        return _empty(ticker, f"past_cutoff_{_CUTOFF}")

    current_date = df["timestamp"].iloc[-1].date()
    session_open_ts = ET.localize(datetime.combine(current_date, RTH_OPEN))
    session = (
        df[df["timestamp"] >= session_open_ts]
        .copy()
        .set_index("timestamp")
        .sort_index()
    )
    bars_15 = (
        session.resample("15min")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .dropna(subset=["open"])
    )
    # Reject any 15-min bar whose window extends past _now (partial bar)
    bars_15 = bars_15[bars_15.index + pd.Timedelta(minutes=15) <= pd.Timestamp(_now)]
    if len(bars_15) < _MIN_15MIN:
        return _empty(ticker, "insufficient_bars")

    last3 = bars_15.iloc[-3:]
    bar_returns = tuple(float(r["close"] / r["open"] - 1) for _, r in last3.iterrows())
    all_green = all(r > 0 for r in bar_returns)

    vol_mean = float(
        bars_15["volume"].rolling(_VOL_ROLL, min_periods=1).mean().iloc[-1]
    )
    last_vol = float(bars_15["volume"].iloc[-1])
    vol_ratio = last_vol / vol_mean if vol_mean > 0 else 0.0
    has_spike = vol_ratio >= _VOL_SPIKE_MULT

    atr = compute_atr14(df, ATR_PERIOD)

    meta = dict(
        bar_returns=tuple(round(r, 5) for r in bar_returns),
        vol_spike_ratio=round(vol_ratio, 3),
        atr14=round(atr, 4),
        num_15min_bars=len(bars_15),
    )

    if all_green and has_spike:
        cum_ret = float(bars_15["close"].iloc[-1] / bars_15["open"].iloc[-3] - 1)
        prob = min(1.0, max(0.0, cum_ret / _PROB_MAX_RETURN))
        entry = float(df["close"].iloc[-1])
        return dict(
            strategy_id="momentum_15min",
            ticker=ticker,
            signal=1,
            prob=round(prob, 4),
            entry=round(entry, 4),
            target=round(entry + TP_ATR_MULT * atr, 4),
            stop=round(entry - SL_ATR_MULT * atr, 4),
            trailing_stop_arm_at=round(entry + TRAIL_ARM_ATR_MULT * atr, 4),
            reason=(
                f"3 green 15-min bars "
                f"(returns={[round(r, 4) for r in bar_returns]}) "
                f"+ vol_spike {vol_ratio:.2f}x"
            ),
            meta=meta,
        )

    parts = []
    if not all_green:
        parts.append(f"not_3_green bars={[round(r, 4) for r in bar_returns]}")
    if not has_spike:
        parts.append(f"vol_spike {vol_ratio:.2f}x < {_VOL_SPIKE_MULT}x")
    return _empty(ticker, "; ".join(parts) or "no_momentum", meta)
