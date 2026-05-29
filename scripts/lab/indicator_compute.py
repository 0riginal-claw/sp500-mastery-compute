"""indicator_compute.py — Pure-NumPy indicator implementations.

Each indicator returns either a value series or (signal, info) where signal is in {-1, 0, +1}
(short, flat, long). All indicators expect a Bars dict with numpy arrays: open, high, low, close,
volume — all same length, oldest first.

Settings follow the Phase 0/1 hardening plan:
- Wilder's smoothing (com=p-1) where TA convention specifies; EWM(span) noted otherwise.
- Cost model: 5 bps per side baseline, VIX-multiplied. Applied in the eval driver, not here.
- No look-ahead: all indicators are causal (shift-1 when using same-bar close to trigger).

INDICATOR_AXIS dict at bottom maps each indicator name → one of 6 informational axes
(see lab.knowledge.indicators.informational_axes()). Use axis_for(name) to look up.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

ArrayDict = dict[str, np.ndarray]


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    """Standard EMA, span = n. alpha = 2/(n+1)."""
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(x, dtype=np.float64)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def _wilder(x: np.ndarray, period: int) -> np.ndarray:
    """Wilder's smoothing: equivalent to EMA with span = 2p - 1 (alpha = 1/p)."""
    alpha = 1.0 / period
    out = np.empty_like(x, dtype=np.float64)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def _sma(x: np.ndarray, period: int) -> np.ndarray:
    """Rolling simple mean; first period-1 are NaN."""
    out = np.full_like(x, np.nan, dtype=np.float64)
    if len(x) < period:
        return out
    csum = np.cumsum(np.insert(x, 0, 0.0))
    out[period - 1:] = (csum[period:] - csum[:-period]) / period
    return out


def _rolling_std(x: np.ndarray, period: int, ddof: int = 0) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=np.float64)
    for i in range(period - 1, len(x)):
        out[i] = np.std(x[i - period + 1:i + 1], ddof=ddof)
    return out


def _rolling_max(x: np.ndarray, period: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=np.float64)
    for i in range(period - 1, len(x)):
        out[i] = np.max(x[i - period + 1:i + 1])
    return out


def _rolling_min(x: np.ndarray, period: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=np.float64)
    for i in range(period - 1, len(x)):
        out[i] = np.min(x[i - period + 1:i + 1])
    return out


def _true_range(bars: ArrayDict) -> np.ndarray:
    h, l, c = bars["high"], bars["low"], bars["close"]
    prev_c = np.concatenate([[c[0]], c[:-1]])
    return np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])


def _atr(bars: ArrayDict, period: int = 14) -> np.ndarray:
    return _wilder(_true_range(bars), period)


# ---------------------------------------------------------------------------
# Indicators (returns float series, NaN until enough data)
# ---------------------------------------------------------------------------


def macd(bars: ArrayDict, fast: int = 12, slow: int = 26, signal: int = 9) -> dict[str, np.ndarray]:
    c = bars["close"]
    macd_line = _ema(c, fast) - _ema(c, slow)
    sig = _ema(macd_line, signal)
    hist = macd_line - sig
    return {"macd": macd_line, "signal": sig, "hist": hist}


def bollinger(bars: ArrayDict, period: int = 20, nstd: float = 2.0) -> dict[str, np.ndarray]:
    c = bars["close"]
    mid = _sma(c, period)
    std = _rolling_std(c, period, ddof=0)
    upper = mid + nstd * std
    lower = mid - nstd * std
    pctb = np.full_like(c, np.nan, dtype=np.float64)
    denom = upper - lower
    mask = denom > 0
    pctb[mask] = (c[mask] - lower[mask]) / denom[mask]
    return {"upper": upper, "mid": mid, "lower": lower, "pctb": pctb}


def keltner(bars: ArrayDict, ema_period: int = 20, atr_period: int = 14, mult: float = 1.5) -> dict[str, np.ndarray]:
    c = bars["close"]
    mid = _ema(c, ema_period)
    atr = _atr(bars, atr_period)
    return {"upper": mid + mult * atr, "mid": mid, "lower": mid - mult * atr}


def obv(bars: ArrayDict) -> np.ndarray:
    c, v = bars["close"], bars["volume"]
    sign = np.sign(np.diff(c, prepend=c[0]))
    return np.cumsum(sign * v).astype(np.float64)


def stochastic(bars: ArrayDict, k: int = 14, d: int = 3, smooth_k: int = 3) -> dict[str, np.ndarray]:
    h, l, c = bars["high"], bars["low"], bars["close"]
    hh = _rolling_max(h, k)
    ll = _rolling_min(l, k)
    raw_k = np.full_like(c, np.nan, dtype=np.float64)
    denom = hh - ll
    mask = denom > 0
    raw_k[mask] = 100.0 * (c[mask] - ll[mask]) / denom[mask]
    k_sm = _sma(raw_k, smooth_k)
    d_sm = _sma(k_sm, d)
    return {"k": k_sm, "d": d_sm}


def williams_r(bars: ArrayDict, period: int = 14) -> np.ndarray:
    h, l, c = bars["high"], bars["low"], bars["close"]
    hh = _rolling_max(h, period)
    ll = _rolling_min(l, period)
    out = np.full_like(c, np.nan, dtype=np.float64)
    denom = hh - ll
    mask = denom > 0
    out[mask] = -100.0 * (hh[mask] - c[mask]) / denom[mask]
    return out


def cci(bars: ArrayDict, period: int = 20) -> np.ndarray:
    h, l, c = bars["high"], bars["low"], bars["close"]
    tp = (h + l + c) / 3.0
    sma_tp = _sma(tp, period)
    out = np.full_like(c, np.nan, dtype=np.float64)
    for i in range(period - 1, len(tp)):
        win = tp[i - period + 1:i + 1]
        mad = np.mean(np.abs(win - sma_tp[i]))
        if mad > 0:
            out[i] = (tp[i] - sma_tp[i]) / (0.015 * mad)
    return out


def mfi(bars: ArrayDict, period: int = 14) -> np.ndarray:
    h, l, c, v = bars["high"], bars["low"], bars["close"], bars["volume"]
    tp = (h + l + c) / 3.0
    raw_mf = tp * v
    pos = np.where(np.diff(tp, prepend=tp[0]) > 0, raw_mf, 0.0)
    neg = np.where(np.diff(tp, prepend=tp[0]) < 0, raw_mf, 0.0)
    pos_sum = _sma(pos, period) * period
    neg_sum = _sma(neg, period) * period
    out = np.full_like(c, np.nan, dtype=np.float64)
    mask = neg_sum > 0
    mr = np.where(mask, pos_sum / np.where(mask, neg_sum, 1.0), np.inf)
    out = 100.0 - 100.0 / (1.0 + mr)
    out[neg_sum == 0] = 100.0
    out[:period] = np.nan
    return out


def fisher_transform(bars: ArrayDict, period: int = 10) -> np.ndarray:
    h, l = bars["high"], bars["low"]
    median = (h + l) / 2.0
    hh = _rolling_max(median, period)
    ll = _rolling_min(median, period)
    out = np.full_like(median, np.nan, dtype=np.float64)
    x = np.zeros_like(median)
    fish = np.zeros_like(median)
    for i in range(period, len(median)):
        denom = hh[i] - ll[i]
        if denom <= 0 or np.isnan(denom):
            continue
        v = 0.33 * 2.0 * ((median[i] - ll[i]) / denom - 0.5) + 0.67 * x[i - 1]
        v = max(min(v, 0.999), -0.999)
        x[i] = v
        fish[i] = 0.5 * np.log((1 + v) / (1 - v)) + 0.5 * fish[i - 1]
        out[i] = fish[i]
    return out


def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])
    up = np.where(delta > 0, delta, 0.0)
    dn = np.where(delta < 0, -delta, 0.0)
    avg_up = _wilder(up, period)
    avg_dn = _wilder(dn, period)
    out = np.full_like(close, np.nan, dtype=np.float64)
    mask = avg_dn > 0
    rs = np.where(mask, avg_up / np.where(mask, avg_dn, 1.0), np.inf)
    out = 100.0 - 100.0 / (1.0 + rs)
    out[avg_dn == 0] = 100.0
    out[:period] = np.nan
    return out


def connors_rsi(bars: ArrayDict, rsi_period: int = 3, streak_period: int = 2, rank_period: int = 100) -> np.ndarray:
    """Connors RSI = (RSI(close, rsi_period) + RSI(streak, streak_period) + PctRank(roc, rank_period)) / 3."""
    c = bars["close"]
    r1 = _rsi(c, rsi_period)
    # Streak: signed run-length of up/down days
    streak = np.zeros_like(c, dtype=np.float64)
    diff = np.diff(c, prepend=c[0])
    for i in range(1, len(c)):
        if diff[i] > 0:
            streak[i] = streak[i - 1] + 1 if streak[i - 1] >= 0 else 1
        elif diff[i] < 0:
            streak[i] = streak[i - 1] - 1 if streak[i - 1] <= 0 else -1
        else:
            streak[i] = 0
    r2 = _rsi(streak, streak_period)
    # PctRank of 1-bar ROC over rank_period
    roc = np.full_like(c, np.nan, dtype=np.float64)
    roc[1:] = (c[1:] - c[:-1]) / c[:-1]
    rank = np.full_like(c, np.nan, dtype=np.float64)
    for i in range(rank_period, len(c)):
        win = roc[i - rank_period + 1:i + 1]
        rank[i] = 100.0 * (np.sum(win < roc[i]) + 0.5 * np.sum(win == roc[i])) / rank_period
    crsi = (r1 + r2 + rank) / 3.0
    return crsi


def supertrend(bars: ArrayDict, atr_period: int = 10, mult: float = 3.0) -> dict[str, np.ndarray]:
    h, l, c = bars["high"], bars["low"], bars["close"]
    hl2 = (h + l) / 2.0
    atr = _atr(bars, atr_period)
    upper_basic = hl2 + mult * atr
    lower_basic = hl2 - mult * atr
    upper = upper_basic.copy()
    lower = lower_basic.copy()
    trend = np.ones_like(c, dtype=np.int8)  # 1 = uptrend, -1 = downtrend
    st = np.copy(c)
    for i in range(1, len(c)):
        upper[i] = upper_basic[i] if (upper_basic[i] < upper[i - 1] or c[i - 1] > upper[i - 1]) else upper[i - 1]
        lower[i] = lower_basic[i] if (lower_basic[i] > lower[i - 1] or c[i - 1] < lower[i - 1]) else lower[i - 1]
        if st[i - 1] == upper[i - 1]:
            st[i] = upper[i] if c[i] <= upper[i] else lower[i]
            trend[i] = -1 if c[i] <= upper[i] else 1
        else:
            st[i] = lower[i] if c[i] >= lower[i] else upper[i]
            trend[i] = 1 if c[i] >= lower[i] else -1
    return {"st": st, "trend": trend, "upper": upper, "lower": lower}


# ---------------------------------------------------------------------------
# Public helpers — wrappers around the building blocks so callers don't reach
# into _underscore names.
# ---------------------------------------------------------------------------


def sma(bars_or_arr: ArrayDict | np.ndarray, period: int = 20, key: str = "close") -> np.ndarray:
    """Simple moving average. Accepts ArrayDict (uses key) or raw 1-D array."""
    x = bars_or_arr[key] if isinstance(bars_or_arr, dict) else bars_or_arr
    return _sma(np.asarray(x, dtype=np.float64), period)


def ema(bars_or_arr: ArrayDict | np.ndarray, period: int = 20, key: str = "close") -> np.ndarray:
    """Exponential moving average, span = period (alpha = 2/(p+1))."""
    x = bars_or_arr[key] if isinstance(bars_or_arr, dict) else bars_or_arr
    out = _ema(np.asarray(x, dtype=np.float64), period)
    # Mask warm-up so callers can distinguish unseeded values
    if period > 1:
        out = out.copy()
        out[:period - 1] = np.nan
    return out


def ema_pair(bars: ArrayDict, fast: int = 9, slow: int = 21) -> dict[str, np.ndarray]:
    """The trend-cluster winner per knowledge.indicators redundancy notes."""
    c = bars["close"]
    return {
        "fast": ema(c, fast),
        "slow": ema(c, slow),
        "diff": ema(c, fast) - ema(c, slow),
    }


def dema(bars_or_arr: ArrayDict | np.ndarray, period: int = 20, key: str = "close") -> np.ndarray:
    """Double EMA = 2*EMA(p) - EMA(EMA(p), p). Faster than EMA(p)."""
    x = bars_or_arr[key] if isinstance(bars_or_arr, dict) else bars_or_arr
    e1 = _ema(np.asarray(x, dtype=np.float64), period)
    e2 = _ema(e1, period)
    out = 2.0 * e1 - e2
    out[:2 * period - 2] = np.nan
    return out


def tema(bars_or_arr: ArrayDict | np.ndarray, period: int = 20, key: str = "close") -> np.ndarray:
    """Triple EMA = 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))."""
    x = bars_or_arr[key] if isinstance(bars_or_arr, dict) else bars_or_arr
    e1 = _ema(np.asarray(x, dtype=np.float64), period)
    e2 = _ema(e1, period)
    e3 = _ema(e2, period)
    out = 3.0 * e1 - 3.0 * e2 + e3
    out[:3 * period - 3] = np.nan
    return out


def rsi(bars_or_arr: ArrayDict | np.ndarray, period: int = 14, key: str = "close") -> np.ndarray:
    """Wilder's RSI public wrapper."""
    x = bars_or_arr[key] if isinstance(bars_or_arr, dict) else bars_or_arr
    return _rsi(np.asarray(x, dtype=np.float64), period)


def atr(bars: ArrayDict, period: int = 14) -> np.ndarray:
    """Average True Range (Wilder)."""
    out = _atr(bars, period)
    out = out.copy()
    out[:period] = np.nan
    return out


def true_range(bars: ArrayDict) -> np.ndarray:
    return _true_range(bars)


# ---------------------------------------------------------------------------
# TREND axis indicators
# ---------------------------------------------------------------------------


def adx(bars: ArrayDict, period: int = 14) -> dict[str, np.ndarray]:
    """Wilder's ADX with +DI / -DI. Used as regime gate (ADX > 25 = trending)."""
    h, l, c = bars["high"], bars["low"], bars["close"]
    up_move = np.diff(h, prepend=h[0])
    dn_move = -np.diff(l, prepend=l[0])
    plus_dm = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
    tr = _true_range(bars)
    atr_v = _wilder(tr, period)
    plus_di = np.full_like(c, np.nan, dtype=np.float64)
    minus_di = np.full_like(c, np.nan, dtype=np.float64)
    plus_dm_s = _wilder(plus_dm, period)
    minus_dm_s = _wilder(minus_dm, period)
    mask = atr_v > 0
    plus_di[mask] = 100.0 * plus_dm_s[mask] / atr_v[mask]
    minus_di[mask] = 100.0 * minus_dm_s[mask] / atr_v[mask]
    dx = np.full_like(c, np.nan, dtype=np.float64)
    s = plus_di + minus_di
    mask2 = s > 0
    dx[mask2] = 100.0 * np.abs(plus_di[mask2] - minus_di[mask2]) / s[mask2]
    # ADX = wilder smoothing of DX (handle NaN with zero-fill before smoothing)
    dx_filled = np.where(np.isnan(dx), 0.0, dx)
    adx_v = _wilder(dx_filled, period)
    adx_v[:2 * period] = np.nan
    plus_di[:period] = np.nan
    minus_di[:period] = np.nan
    return {"adx": adx_v, "plus_di": plus_di, "minus_di": minus_di}


def aroon(bars: ArrayDict, period: int = 25) -> dict[str, np.ndarray]:
    """Aroon Up / Aroon Down / Aroon Oscillator. Period 25 default, 14-20 for intraday."""
    h, l = bars["high"], bars["low"]
    n = len(h)
    up = np.full(n, np.nan, dtype=np.float64)
    dn = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        win_h = h[i - period:i + 1]
        win_l = l[i - period:i + 1]
        # bars since extreme (0 = current bar is the extreme)
        argmax = period - int(np.argmax(win_h))  # bars since high
        argmin = period - int(np.argmin(win_l))
        up[i] = 100.0 * (period - argmax) / period
        dn[i] = 100.0 * (period - argmin) / period
    return {"up": up, "down": dn, "osc": up - dn}


def parabolic_sar(bars: ArrayDict, step: float = 0.02, max_step: float = 0.20) -> dict[str, np.ndarray]:
    """Wilder's Parabolic SAR. Returns sar series + trend (+1/-1)."""
    h, l = bars["high"], bars["low"]
    n = len(h)
    sar = np.full(n, np.nan, dtype=np.float64)
    trend = np.zeros(n, dtype=np.int8)
    if n < 2:
        return {"sar": sar, "trend": trend}
    # Initial trend guess from first 2 bars
    t = 1 if h[1] >= h[0] else -1
    ep = h[1] if t == 1 else l[1]
    af = step
    sar[1] = l[0] if t == 1 else h[0]
    trend[1] = t
    for i in range(2, n):
        sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
        if t == 1:
            sar[i] = min(sar[i], l[i - 1], l[i - 2])
            if l[i] < sar[i]:
                # Flip
                t = -1
                sar[i] = ep
                ep = l[i]
                af = step
            else:
                if h[i] > ep:
                    ep = h[i]
                    af = min(af + step, max_step)
        else:
            sar[i] = max(sar[i], h[i - 1], h[i - 2])
            if h[i] > sar[i]:
                t = 1
                sar[i] = ep
                ep = h[i]
                af = step
            else:
                if l[i] < ep:
                    ep = l[i]
                    af = min(af + step, max_step)
        trend[i] = t
    return {"sar": sar, "trend": trend}


def ichimoku(bars: ArrayDict, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52) -> dict[str, np.ndarray]:
    """Ichimoku Kinko Hyo. Note: senkou lines are projected forward 26 bars but we keep them
    aligned to the bar they're CALCULATED at — caller applies shift if displaying as cloud.
    No look-ahead: all values use only past data through index i."""
    h, l, c = bars["high"], bars["low"], bars["close"]
    tenkan_v = (_rolling_max(h, tenkan) + _rolling_min(l, tenkan)) / 2.0
    kijun_v = (_rolling_max(h, kijun) + _rolling_min(l, kijun)) / 2.0
    senkou_a = (tenkan_v + kijun_v) / 2.0  # plotted 26 bars ahead by convention
    senkou_b_v = (_rolling_max(h, senkou_b) + _rolling_min(l, senkou_b)) / 2.0
    # chikou = close shifted back 26 (we return raw close; caller does shift)
    return {
        "tenkan": tenkan_v,
        "kijun": kijun_v,
        "senkou_a": senkou_a,
        "senkou_b": senkou_b_v,
        "chikou": c.astype(np.float64),
    }


def vortex(bars: ArrayDict, period: int = 14) -> dict[str, np.ndarray]:
    """Vortex Indicator. VI+ > VI- = uptrend."""
    h, l, c = bars["high"], bars["low"], bars["close"]
    prev_l = np.concatenate([[l[0]], l[:-1]])
    prev_h = np.concatenate([[h[0]], h[:-1]])
    vmp = np.abs(h - prev_l)
    vmm = np.abs(l - prev_h)
    tr = _true_range(bars)
    sum_vmp = _sma(vmp, period) * period
    sum_vmm = _sma(vmm, period) * period
    sum_tr = _sma(tr, period) * period
    vi_p = np.full_like(c, np.nan, dtype=np.float64)
    vi_m = np.full_like(c, np.nan, dtype=np.float64)
    mask = sum_tr > 0
    vi_p[mask] = sum_vmp[mask] / sum_tr[mask]
    vi_m[mask] = sum_vmm[mask] / sum_tr[mask]
    return {"vi_plus": vi_p, "vi_minus": vi_m}


def trix(bars: ArrayDict, period: int = 15, signal: int = 9) -> dict[str, np.ndarray]:
    """TRIX = 1-bar ROC of triple-smoothed EMA(close)."""
    c = bars["close"]
    e1 = _ema(c, period)
    e2 = _ema(e1, period)
    e3 = _ema(e2, period)
    prev = np.concatenate([[e3[0]], e3[:-1]])
    out = np.full_like(c, np.nan, dtype=np.float64)
    mask = prev != 0
    out[mask] = 100.0 * (e3[mask] - prev[mask]) / prev[mask]
    out[:3 * period] = np.nan
    sig = _ema(np.where(np.isnan(out), 0.0, out), signal)
    sig[:3 * period + signal] = np.nan
    return {"trix": out, "signal": sig}


def macd_histogram(bars: ArrayDict, fast: int = 5, slow: int = 13, signal: int = 1) -> np.ndarray:
    """MACD Histogram only (Raschke 5/13/1 default)."""
    m = macd(bars, fast, slow, signal)
    return m["hist"]


# ---------------------------------------------------------------------------
# MOMENTUM_OSCILLATOR axis
# ---------------------------------------------------------------------------


def ultimate_oscillator(bars: ArrayDict, p1: int = 7, p2: int = 14, p3: int = 28) -> np.ndarray:
    """Williams' Ultimate Oscillator (7/14/28)."""
    h, l, c = bars["high"], bars["low"], bars["close"]
    prev_c = np.concatenate([[c[0]], c[:-1]])
    bp = c - np.minimum(l, prev_c)
    tr = _true_range(bars)
    sum_bp_1 = _sma(bp, p1) * p1
    sum_tr_1 = _sma(tr, p1) * p1
    sum_bp_2 = _sma(bp, p2) * p2
    sum_tr_2 = _sma(tr, p2) * p2
    sum_bp_3 = _sma(bp, p3) * p3
    sum_tr_3 = _sma(tr, p3) * p3
    out = np.full_like(c, np.nan, dtype=np.float64)
    mask = (sum_tr_1 > 0) & (sum_tr_2 > 0) & (sum_tr_3 > 0)
    a1 = np.divide(sum_bp_1, sum_tr_1, out=np.zeros_like(c, dtype=np.float64), where=sum_tr_1 > 0)
    a2 = np.divide(sum_bp_2, sum_tr_2, out=np.zeros_like(c, dtype=np.float64), where=sum_tr_2 > 0)
    a3 = np.divide(sum_bp_3, sum_tr_3, out=np.zeros_like(c, dtype=np.float64), where=sum_tr_3 > 0)
    out[mask] = 100.0 * (4 * a1[mask] + 2 * a2[mask] + a3[mask]) / 7.0
    return out


def awesome_oscillator(bars: ArrayDict, fast: int = 5, slow: int = 34) -> np.ndarray:
    """AO = SMA(midprice, 5) - SMA(midprice, 34) where midprice = (H+L)/2."""
    h, l = bars["high"], bars["low"]
    mid = (h + l) / 2.0
    return _sma(mid, fast) - _sma(mid, slow)


def dpo(bars: ArrayDict, period: int = 20) -> np.ndarray:
    """Detrended Price Oscillator. NO LOOK-AHEAD VARIANT: DPO[i] = close[i - (n/2+1)] - SMA(close,n)[i].

    Classical DPO uses a forward-shifted SMA which IS look-ahead. We instead align
    the SMA at the current bar and lag the close, which preserves the detrending
    intent without leaking future bars.
    """
    c = bars["close"]
    n = period
    shift = n // 2 + 1
    sma_v = _sma(c, n)
    out = np.full_like(c, np.nan, dtype=np.float64)
    out[shift:] = c[:-shift] - sma_v[shift:]
    return out


def ppo(bars: ArrayDict, fast: int = 12, slow: int = 26, signal: int = 9) -> dict[str, np.ndarray]:
    """Percentage Price Oscillator = 100*(EMA_fast - EMA_slow) / EMA_slow."""
    c = bars["close"]
    ef = _ema(c, fast)
    es = _ema(c, slow)
    out = np.full_like(c, np.nan, dtype=np.float64)
    mask = es != 0
    out[mask] = 100.0 * (ef[mask] - es[mask]) / es[mask]
    sig = _ema(out, signal)
    return {"ppo": out, "signal": sig, "hist": out - sig}


def roc(bars_or_arr: ArrayDict | np.ndarray, period: int = 10, key: str = "close") -> np.ndarray:
    """Rate of Change = 100 * (x[i] - x[i-n]) / x[i-n]."""
    x = bars_or_arr[key] if isinstance(bars_or_arr, dict) else bars_or_arr
    x = np.asarray(x, dtype=np.float64)
    out = np.full_like(x, np.nan, dtype=np.float64)
    if len(x) <= period:
        return out
    prev = x[:-period]
    out[period:] = np.where(prev != 0, 100.0 * (x[period:] - prev) / prev, np.nan)
    return out


def momentum(bars_or_arr: ArrayDict | np.ndarray, period: int = 10, key: str = "close") -> np.ndarray:
    """Raw momentum = x[i] - x[i-n]."""
    x = bars_or_arr[key] if isinstance(bars_or_arr, dict) else bars_or_arr
    x = np.asarray(x, dtype=np.float64)
    out = np.full_like(x, np.nan, dtype=np.float64)
    if len(x) <= period:
        return out
    out[period:] = x[period:] - x[:-period]
    return out


def stochastic_rsi(bars: ArrayDict, rsi_period: int = 14, stoch_period: int = 14,
                   k: int = 3, d: int = 3) -> dict[str, np.ndarray]:
    """Stochastic RSI = stochastic applied to RSI series."""
    r = _rsi(bars["close"], rsi_period)
    r_filled = np.where(np.isnan(r), 50.0, r)
    hh = _rolling_max(r_filled, stoch_period)
    ll = _rolling_min(r_filled, stoch_period)
    raw = np.full_like(r, np.nan, dtype=np.float64)
    denom = hh - ll
    mask = denom > 0
    raw[mask] = 100.0 * (r_filled[mask] - ll[mask]) / denom[mask]
    raw[:rsi_period + stoch_period - 1] = np.nan
    k_v = _sma(raw, k)
    d_v = _sma(k_v, d)
    return {"k": k_v, "d": d_v, "raw": raw}


def kst(bars: ArrayDict, r1: int = 10, r2: int = 15, r3: int = 20, r4: int = 30,
        s1: int = 10, s2: int = 10, s3: int = 10, s4: int = 15,
        signal: int = 9) -> dict[str, np.ndarray]:
    """Pring's Know Sure Thing — weighted sum of 4 smoothed ROCs."""
    c = bars["close"]
    roc1 = _sma(roc(c, r1), s1)
    roc2 = _sma(roc(c, r2), s2)
    roc3 = _sma(roc(c, r3), s3)
    roc4 = _sma(roc(c, r4), s4)
    kst_v = roc1 + 2 * roc2 + 3 * roc3 + 4 * roc4
    sig = _sma(kst_v, signal)
    return {"kst": kst_v, "signal": sig}


def tsi(bars: ArrayDict, long_p: int = 25, short_p: int = 13, signal: int = 7) -> dict[str, np.ndarray]:
    """True Strength Index = 100 * EMA(EMA(mom, long), short) / EMA(EMA(|mom|, long), short)."""
    c = bars["close"]
    mom = np.diff(c, prepend=c[0])
    abs_mom = np.abs(mom)
    num = _ema(_ema(mom, long_p), short_p)
    den = _ema(_ema(abs_mom, long_p), short_p)
    out = np.full_like(c, np.nan, dtype=np.float64)
    mask = den > 0
    out[mask] = 100.0 * num[mask] / den[mask]
    sig = _ema(out, signal)
    return {"tsi": out, "signal": sig}


# ---------------------------------------------------------------------------
# VOLATILITY_BAND / STRUCTURE_GEOMETRY axes
# ---------------------------------------------------------------------------


def donchian(bars: ArrayDict, period: int = 20) -> dict[str, np.ndarray]:
    """Donchian Channel — single highest-WR indicator in the catalog (0.547 AAPL / 0.753 cohort).

    upper = rolling_max(high, n); lower = rolling_min(low, n); mid = (upper+lower)/2.
    'up_breakout' = close > prior upper; 'dn_breakout' = close < prior lower.
    """
    h, l, c = bars["high"], bars["low"], bars["close"]
    upper = _rolling_max(h, period)
    lower = _rolling_min(l, period)
    mid = (upper + lower) / 2.0
    # prior-bar bands for breakout test (no look-ahead)
    prior_upper = np.concatenate([[np.nan], upper[:-1]])
    prior_lower = np.concatenate([[np.nan], lower[:-1]])
    up_brk = (c > prior_upper).astype(np.int8)
    dn_brk = (c < prior_lower).astype(np.int8)
    return {"upper": upper, "lower": lower, "mid": mid,
            "up_breakout": up_brk, "dn_breakout": dn_brk}


def heikin_ashi(bars: ArrayDict) -> dict[str, np.ndarray]:
    """Heikin-Ashi smoothed OHLC."""
    o, h, l, c = bars["open"], bars["high"], bars["low"], bars["close"]
    n = len(c)
    ha_c = (o + h + l + c) / 4.0
    ha_o = np.empty(n, dtype=np.float64)
    ha_o[0] = (o[0] + c[0]) / 2.0
    for i in range(1, n):
        ha_o[i] = (ha_o[i - 1] + ha_c[i - 1]) / 2.0
    ha_h = np.maximum.reduce([h, ha_o, ha_c])
    ha_l = np.minimum.reduce([l, ha_o, ha_c])
    return {"open": ha_o, "high": ha_h, "low": ha_l, "close": ha_c}


def ttm_squeeze(bars: ArrayDict, bb_period: int = 20, bb_std: float = 2.0,
                kc_ema: int = 20, kc_atr: int = 14, kc_mult: float = 1.5) -> dict[str, np.ndarray]:
    """TTM Squeeze: BB inside KC = squeeze ON; momentum direction = linreg slope of (close - mid).

    Returns:
      squeeze_on: 1 when BB upper < KC upper AND BB lower > KC lower (compression)
      momentum: simplified momentum proxy = close - midprice of last 20 bars
    """
    bb = bollinger(bars, bb_period, bb_std)
    kc = keltner(bars, kc_ema, kc_atr, kc_mult)
    squeeze_on = ((bb["upper"] < kc["upper"]) & (bb["lower"] > kc["lower"])).astype(np.int8)
    # Lazzy momentum proxy — diff from rolling midpoint of close
    c = bars["close"]
    hh = _rolling_max(c, bb_period)
    ll = _rolling_min(c, bb_period)
    midprice = (hh + ll) / 2.0
    momentum_v = c - midprice
    return {"squeeze_on": squeeze_on, "momentum": momentum_v}


def zigzag(bars: ArrayDict, threshold_pct: float = 3.0) -> np.ndarray:
    """ZigZag — marks pivot points where price reverses by >= threshold_pct from last pivot.

    Returns array of pivot prices (NaN at non-pivot bars). Causal: pivot is confirmed only
    when the threshold reversal completes, so the marked bar is the past pivot bar — but
    the confirmation timestamp lags. We mark the pivot bar's index with its high/low.
    """
    h, l = bars["high"], bars["low"]
    n = len(h)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return out
    pivot_idx = 0
    pivot_price = h[0]
    direction = 0  # 0 = unknown, 1 = up, -1 = down
    threshold = threshold_pct / 100.0
    for i in range(1, n):
        if direction >= 0:
            # Watching for new high or down reversal
            if h[i] > pivot_price:
                pivot_idx = i
                pivot_price = h[i]
                direction = 1
            elif (pivot_price - l[i]) / pivot_price >= threshold:
                out[pivot_idx] = pivot_price
                pivot_idx = i
                pivot_price = l[i]
                direction = -1
        else:
            if l[i] < pivot_price:
                pivot_idx = i
                pivot_price = l[i]
                direction = -1
            elif (h[i] - pivot_price) / pivot_price >= threshold:
                out[pivot_idx] = pivot_price
                pivot_idx = i
                pivot_price = h[i]
                direction = 1
    return out


def pivot_points_classic(bars: ArrayDict) -> dict[str, np.ndarray]:
    """Classic Pivot Points — computed from PRIOR bar's H/L/C to avoid look-ahead.

    Useful for daily pivots on intraday bars: caller should pre-aggregate to daily H/L/C
    and broadcast back. Here we compute as if each bar is its own session (prior-bar based).
    """
    h, l, c = bars["high"], bars["low"], bars["close"]
    prev_h = np.concatenate([[np.nan], h[:-1]])
    prev_l = np.concatenate([[np.nan], l[:-1]])
    prev_c = np.concatenate([[np.nan], c[:-1]])
    p = (prev_h + prev_l + prev_c) / 3.0
    r1 = 2 * p - prev_l
    s1 = 2 * p - prev_h
    r2 = p + (prev_h - prev_l)
    s2 = p - (prev_h - prev_l)
    r3 = prev_h + 2 * (p - prev_l)
    s3 = prev_l - 2 * (prev_h - p)
    return {"pivot": p, "r1": r1, "s1": s1, "r2": r2, "s2": s2, "r3": r3, "s3": s3}


def pivot_points_fib(bars: ArrayDict) -> dict[str, np.ndarray]:
    """Fibonacci Pivot Points — same prior-bar basis with Fib ratios."""
    h, l, c = bars["high"], bars["low"], bars["close"]
    prev_h = np.concatenate([[np.nan], h[:-1]])
    prev_l = np.concatenate([[np.nan], l[:-1]])
    prev_c = np.concatenate([[np.nan], c[:-1]])
    p = (prev_h + prev_l + prev_c) / 3.0
    rng = prev_h - prev_l
    return {
        "pivot": p,
        "r1": p + 0.382 * rng,
        "s1": p - 0.382 * rng,
        "r2": p + 0.618 * rng,
        "s2": p - 0.618 * rng,
        "r3": p + 1.000 * rng,
        "s3": p - 1.000 * rng,
    }


def opening_range(bars: ArrayDict, bars_in_range: int = 6,
                  session_starts: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Opening Range — first N bars of each session. Default 6 bars = 30 min on 5min S&P.

    If session_starts is None, treats the whole input as one session (useful for one-day arrays).
    Otherwise expects a boolean array marking each session-start bar.
    Output: or_high, or_low, in_range (1 during the OR window, 0 after).
    No look-ahead: values become known only AFTER the OR window completes.
    """
    h, l = bars["high"], bars["low"]
    n = len(h)
    or_h = np.full(n, np.nan, dtype=np.float64)
    or_l = np.full(n, np.nan, dtype=np.float64)
    in_range = np.zeros(n, dtype=np.int8)
    if session_starts is None:
        starts = np.zeros(n, dtype=bool)
        starts[0] = True
    else:
        starts = np.asarray(session_starts, dtype=bool)
    cur_h = np.nan
    cur_l = np.nan
    bars_into_session = 0
    for i in range(n):
        if starts[i]:
            cur_h = h[i]
            cur_l = l[i]
            bars_into_session = 1
        else:
            bars_into_session += 1
        if bars_into_session <= bars_in_range:
            cur_h = max(cur_h, h[i]) if not np.isnan(cur_h) else h[i]
            cur_l = min(cur_l, l[i]) if not np.isnan(cur_l) else l[i]
            in_range[i] = 1
            # OR values become known only at the END of the window; mark NaN within
        else:
            or_h[i] = cur_h
            or_l[i] = cur_l
    return {"or_high": or_h, "or_low": or_l, "in_range": in_range}


def choppiness_index(bars: ArrayDict, period: int = 14) -> np.ndarray:
    """Choppiness Index (Bill Dreiss). 0-100; >61.8 = consolidation; <38.2 = trend."""
    tr = _true_range(bars)
    sum_tr = _sma(tr, period) * period
    hh = _rolling_max(bars["high"], period)
    ll = _rolling_min(bars["low"], period)
    rng = hh - ll
    out = np.full_like(bars["close"], np.nan, dtype=np.float64)
    mask = (rng > 0) & (sum_tr > 0)
    out[mask] = 100.0 * np.log10(sum_tr[mask] / rng[mask]) / np.log10(period)
    return out


def chop_idx(bars: ArrayDict, period: int = 14) -> np.ndarray:
    """Alias for choppiness_index (Mission 12 ChopIdx naming)."""
    return choppiness_index(bars, period)


def zscore(bars_or_arr: ArrayDict | np.ndarray, period: int = 20, key: str = "close") -> np.ndarray:
    """Rolling z-score = (x - mean) / std."""
    x = bars_or_arr[key] if isinstance(bars_or_arr, dict) else bars_or_arr
    x = np.asarray(x, dtype=np.float64)
    mu = _sma(x, period)
    sd = _rolling_std(x, period, ddof=0)
    out = np.full_like(x, np.nan, dtype=np.float64)
    mask = sd > 0
    out[mask] = (x[mask] - mu[mask]) / sd[mask]
    return out


# ---------------------------------------------------------------------------
# VOLUME_CONVICTION axis
# ---------------------------------------------------------------------------


def vwap(bars: ArrayDict, session_starts: np.ndarray | None = None) -> np.ndarray:
    """Volume-Weighted Average Price. Resets per session. Default single session.

    No look-ahead — cumulative through bar i only.
    """
    h, l, c, v = bars["high"], bars["low"], bars["close"], bars["volume"]
    n = len(c)
    tp = (h + l + c) / 3.0
    out = np.full(n, np.nan, dtype=np.float64)
    if session_starts is None:
        starts = np.zeros(n, dtype=bool)
        starts[0] = True
    else:
        starts = np.asarray(session_starts, dtype=bool)
    cum_pv = 0.0
    cum_v = 0.0
    for i in range(n):
        if starts[i]:
            cum_pv = 0.0
            cum_v = 0.0
        cum_pv += tp[i] * v[i]
        cum_v += v[i]
        if cum_v > 0:
            out[i] = cum_pv / cum_v
    return out


def volume_expansion(bars: ArrayDict, period: int = 20, mult: float = 1.5) -> dict[str, np.ndarray]:
    """Volume Expansion: v > mult * SMA(v, period). 0.568 WR per ir4. CONFIRMATION ONLY."""
    v = bars["volume"]
    avg = _sma(v, period)
    out = np.full_like(v, np.nan, dtype=np.float64)
    mask = avg > 0
    out[mask] = v[mask] / avg[mask]
    is_expanding = (out >= mult).astype(np.int8)
    return {"ratio": out, "is_expanding": is_expanding, "avg": avg}


def cmf(bars: ArrayDict, period: int = 21) -> np.ndarray:
    """Chaikin Money Flow. Period 21 default, 10-14 scalp."""
    h, l, c, v = bars["high"], bars["low"], bars["close"], bars["volume"]
    rng = h - l
    mfm = np.zeros_like(c, dtype=np.float64)
    mask = rng > 0
    mfm[mask] = ((c[mask] - l[mask]) - (h[mask] - c[mask])) / rng[mask]
    mfv = mfm * v
    sum_mfv = _sma(mfv, period) * period
    sum_v = _sma(v, period) * period
    out = np.full_like(c, np.nan, dtype=np.float64)
    m2 = sum_v > 0
    out[m2] = sum_mfv[m2] / sum_v[m2]
    return out


def ad_line(bars: ArrayDict) -> np.ndarray:
    """Accumulation/Distribution Line. Cumulative Chaikin money flow volume."""
    h, l, c, v = bars["high"], bars["low"], bars["close"], bars["volume"]
    rng = h - l
    mfm = np.zeros_like(c, dtype=np.float64)
    mask = rng > 0
    mfm[mask] = ((c[mask] - l[mask]) - (h[mask] - c[mask])) / rng[mask]
    return np.cumsum(mfm * v)


def force_index(bars: ArrayDict, period: int = 13) -> dict[str, np.ndarray]:
    """Elder's Force Index. raw = (close - prev_close) * volume; smoothed via EMA."""
    c, v = bars["close"], bars["volume"]
    prev_c = np.concatenate([[c[0]], c[:-1]])
    raw = (c - prev_c) * v
    smoothed = _ema(raw, period)
    return {"raw": raw, "smoothed": smoothed}


def elder_ray(bars: ArrayDict, ema_p: int = 13) -> dict[str, np.ndarray]:
    """Elder Ray: bull_power = high - EMA(close); bear_power = low - EMA(close)."""
    h, l, c = bars["high"], bars["low"], bars["close"]
    e = _ema(c, ema_p)
    return {"bull_power": h - e, "bear_power": l - e, "ema": e}


def volume_profile(bars: ArrayDict, bins: int = 24, window: int | None = None) -> dict[str, Any]:
    """Volume Profile for a window (default: whole input).

    Returns POC (price of highest-volume bin), VAH/VAL (70% value area edges).
    For session-aware use, caller should slice bars first.
    """
    h, l, c, v = bars["high"], bars["low"], bars["close"], bars["volume"]
    n = len(c)
    if window is None or window > n:
        window = n
    start = n - window
    tp = (h[start:] + l[start:] + c[start:]) / 3.0
    vs = v[start:]
    if len(tp) == 0:
        return {"poc": np.nan, "vah": np.nan, "val": np.nan, "bins": None, "hist": None}
    lo, hi = float(np.min(tp)), float(np.max(tp))
    if hi <= lo:
        return {"poc": lo, "vah": lo, "val": lo, "bins": np.array([lo]), "hist": np.array([np.sum(vs)])}
    edges = np.linspace(lo, hi, bins + 1)
    idx = np.clip(np.searchsorted(edges, tp, side="right") - 1, 0, bins - 1)
    hist = np.zeros(bins, dtype=np.float64)
    for i, vi in zip(idx, vs):
        hist[i] += vi
    poc_bin = int(np.argmax(hist))
    poc = (edges[poc_bin] + edges[poc_bin + 1]) / 2.0
    # Value area: expand outward from POC until 70% of total volume captured
    total = hist.sum()
    target = 0.70 * total
    captured = hist[poc_bin]
    lo_b = hi_b = poc_bin
    while captured < target and (lo_b > 0 or hi_b < bins - 1):
        next_lo = hist[lo_b - 1] if lo_b > 0 else -1
        next_hi = hist[hi_b + 1] if hi_b < bins - 1 else -1
        if next_hi >= next_lo:
            hi_b += 1
            captured += hist[hi_b]
        else:
            lo_b -= 1
            captured += hist[lo_b]
    vah = edges[hi_b + 1]
    val = edges[lo_b]
    return {"poc": poc, "vah": vah, "val": val, "bins": edges, "hist": hist}


# ---------------------------------------------------------------------------
# MEAN_REVERSION axis (extras beyond Connors RSI)
# ---------------------------------------------------------------------------


def bb_percent_b(bars: ArrayDict, period: int = 20, nstd: float = 2.0) -> np.ndarray:
    """BB %B as standalone — explicit mean-reversion axis function."""
    return bollinger(bars, period, nstd)["pctb"]


def bb_width(bars: ArrayDict, period: int = 20, nstd: float = 2.0) -> np.ndarray:
    """BB bandwidth = (upper - lower) / mid. Squeeze when low; expansion when high."""
    bb = bollinger(bars, period, nstd)
    out = np.full_like(bars["close"], np.nan, dtype=np.float64)
    mask = bb["mid"] > 0
    out[mask] = (bb["upper"][mask] - bb["lower"][mask]) / bb["mid"][mask]
    return out


def keltner_width(bars: ArrayDict, ema_period: int = 20, atr_period: int = 14, mult: float = 1.5) -> np.ndarray:
    kc = keltner(bars, ema_period, atr_period, mult)
    return (kc["upper"] - kc["lower"]) / kc["mid"]


# ---------------------------------------------------------------------------
# Signal generators (entry/exit rules → +1/0/-1 position per bar, NO look-ahead)
# All rules shift the signal forward 1 bar before returns are computed by the runner.
# ---------------------------------------------------------------------------


def sig_macd_zero(bars: ArrayDict, fast: int = 12, slow: int = 26, signal: int = 9) -> np.ndarray:
    """+1 when MACD>0 AND MACD>signal, -1 when MACD<0 AND MACD<signal, 0 otherwise."""
    m = macd(bars, fast, slow, signal)
    sig = np.zeros(len(bars["close"]), dtype=np.int8)
    sig[(m["macd"] > 0) & (m["macd"] > m["signal"])] = 1
    sig[(m["macd"] < 0) & (m["macd"] < m["signal"])] = -1
    return sig


def sig_bb_mean_rev(bars: ArrayDict, period: int = 20, nstd: float = 2.0) -> np.ndarray:
    """Mean-reversion: +1 when close < lower, -1 when close > upper, exit at mid touch.

    Stateful: hold position until close crosses mid.
    """
    bb = bollinger(bars, period, nstd)
    c = bars["close"]
    pos = np.zeros(len(c), dtype=np.int8)
    state = 0
    for i in range(period, len(c)):
        if state == 0:
            if c[i] < bb["lower"][i]:
                state = 1
            elif c[i] > bb["upper"][i]:
                state = -1
        elif state == 1 and c[i] >= bb["mid"][i]:
            state = 0
        elif state == -1 and c[i] <= bb["mid"][i]:
            state = 0
        pos[i] = state
    return pos


def sig_bb_pctb(bars: ArrayDict, period: int = 20, nstd: float = 2.0, lo: float = 0.05, hi: float = 0.95) -> np.ndarray:
    bb = bollinger(bars, period, nstd)
    pb = bb["pctb"]
    sig = np.zeros(len(bars["close"]), dtype=np.int8)
    sig[pb < lo] = 1
    sig[pb > hi] = -1
    return sig


def sig_keltner_breakout(bars: ArrayDict, ema_period: int = 20, atr_period: int = 14, mult: float = 1.5) -> np.ndarray:
    kc = keltner(bars, ema_period, atr_period, mult)
    c = bars["close"]
    sig = np.zeros(len(c), dtype=np.int8)
    sig[c > kc["upper"]] = 1
    sig[c < kc["lower"]] = -1
    return sig


def sig_obv_trend(bars: ArrayDict, ema_span: int = 20) -> np.ndarray:
    o = obv(bars)
    o_ema = _ema(o, ema_span)
    sig = np.where(o > o_ema, 1, -1).astype(np.int8)
    return sig


def sig_stoch(bars: ArrayDict, k: int = 14, d: int = 3, sm: int = 3, lo: float = 20, hi: float = 80) -> np.ndarray:
    s = stochastic(bars, k, d, sm)
    sig = np.zeros(len(bars["close"]), dtype=np.int8)
    sig[(s["k"] < lo) & (s["k"] > s["d"])] = 1  # oversold, %K turning up
    sig[(s["k"] > hi) & (s["k"] < s["d"])] = -1
    return sig


def sig_williams_r(bars: ArrayDict, period: int = 14, lo: float = -80, hi: float = -20) -> np.ndarray:
    w = williams_r(bars, period)
    sig = np.zeros(len(bars["close"]), dtype=np.int8)
    sig[w < lo] = 1
    sig[w > hi] = -1
    return sig


def sig_cci(bars: ArrayDict, period: int = 20, lo: float = -100, hi: float = 100) -> np.ndarray:
    c = cci(bars, period)
    sig = np.zeros(len(bars["close"]), dtype=np.int8)
    sig[c < lo] = 1
    sig[c > hi] = -1
    return sig


def sig_mfi(bars: ArrayDict, period: int = 14, lo: float = 20, hi: float = 80) -> np.ndarray:
    m = mfi(bars, period)
    sig = np.zeros(len(bars["close"]), dtype=np.int8)
    sig[m < lo] = 1
    sig[m > hi] = -1
    return sig


def sig_fisher(bars: ArrayDict, period: int = 10, lo: float = -1.5, hi: float = 1.5) -> np.ndarray:
    f = fisher_transform(bars, period)
    sig = np.zeros(len(bars["close"]), dtype=np.int8)
    sig[f < lo] = 1
    sig[f > hi] = -1
    return sig


def sig_connors_rsi(bars: ArrayDict, rsi_p: int = 3, streak_p: int = 2, rank_p: int = 100,
                    lo: float = 10, hi: float = 90) -> np.ndarray:
    cr = connors_rsi(bars, rsi_p, streak_p, rank_p)
    sig = np.zeros(len(bars["close"]), dtype=np.int8)
    sig[cr < lo] = 1
    sig[cr > hi] = -1
    return sig


def sig_supertrend(bars: ArrayDict, atr_period: int = 10, mult: float = 3.0) -> np.ndarray:
    st = supertrend(bars, atr_period, mult)
    return st["trend"].astype(np.int8)


# ---------------------------------------------------------------------------
# PATH B — Thin wrappers for catalog param variants + additional well-known
# TA indicators (KAMA, Hull, ALMA, T3, ZLEMA, FRAMA, NATR, Ulcer, etc.).
# Added 2026-05-29 to make every catalog name callable by exact name.
# ---------------------------------------------------------------------------


# ── Catalog param-variant aliases (Donchian / RSI / EMA / SMA / VWAP / etc.) ──


def donchian_20_up(bars: ArrayDict) -> np.ndarray:
    """Donchian Channel (20) UP — series of 20-bar rolling high. axis=structure_geometry."""
    return donchian(bars, period=20)["upper"]


def donchian_20_dn(bars: ArrayDict) -> np.ndarray:
    """Donchian Channel (20) DN — series of 20-bar rolling low. axis=structure_geometry."""
    return donchian(bars, period=20)["lower"]


def donchian_40_up(bars: ArrayDict) -> np.ndarray:
    """Donchian Channel (40) UP — slower but cleaner breakout band (optimized_cycle001)."""
    return donchian(bars, period=40)["upper"]


def donchian_40_dn(bars: ArrayDict) -> np.ndarray:
    """Donchian Channel (40) DN."""
    return donchian(bars, period=40)["lower"]


def rsi_2(bars: ArrayDict) -> np.ndarray:
    """RSI(2) — ultra-short Connors-style mean-reversion entry trigger."""
    return rsi(bars, period=2)


def rsi_14(bars: ArrayDict) -> np.ndarray:
    """RSI(14) — Wilder default."""
    return rsi(bars, period=14)


def rsi_scalp(bars: ArrayDict) -> np.ndarray:
    """RSI(7) — scalping default."""
    return rsi(bars, period=7)


def ema_9(bars: ArrayDict) -> np.ndarray:
    """EMA(9) — fast bias filter."""
    return ema(bars, period=9)


def ema_20(bars: ArrayDict) -> np.ndarray:
    """EMA(20) — trend filter default."""
    return ema(bars, period=20)


def ema_21(bars: ArrayDict) -> np.ndarray:
    """EMA(21) — pair partner with EMA(9)."""
    return ema(bars, period=21)


def ema_50(bars: ArrayDict) -> np.ndarray:
    """EMA(50) — medium-term bias."""
    return ema(bars, period=50)


def ema_200(bars: ArrayDict) -> np.ndarray:
    """EMA(200) — long-term bias."""
    return ema(bars, period=200)


def sma_20(bars: ArrayDict) -> np.ndarray:
    """SMA(20) — trend filter default."""
    return sma(bars, period=20)


def sma_50(bars: ArrayDict) -> np.ndarray:
    return sma(bars, period=50)


def sma_200(bars: ArrayDict) -> np.ndarray:
    return sma(bars, period=200)


def atr_14(bars: ArrayDict) -> np.ndarray:
    """ATR(14) — Wilder default for exit/sizing."""
    return atr(bars, period=14)


def atr_scalp(bars: ArrayDict) -> np.ndarray:
    """ATR(7) — scalp variant."""
    return atr(bars, period=7)


# ── MACD param variants from catalog ──


def macd_hist_raschke(bars: ArrayDict) -> np.ndarray:
    """MACD Histogram — Raschke params (5,13,1). axis=trend."""
    return macd_histogram(bars, fast=5, slow=13, signal=1)


def macd_hist_scalp(bars: ArrayDict) -> np.ndarray:
    """MACD Histogram — scalp params (3,10,1)."""
    return macd_histogram(bars, fast=3, slow=10, signal=1)


def macd_hist_smoother(bars: ArrayDict) -> np.ndarray:
    """MACD Histogram — smoother params (8,17,9)."""
    return macd_histogram(bars, fast=8, slow=17, signal=9)


def macd_12_26_9(bars: ArrayDict) -> dict[str, np.ndarray]:
    """MACD(12,26,9) classic."""
    return macd(bars, fast=12, slow=26, signal=9)


# ── Stochastic / Williams %R / CCI / MFI variants ──


def williams_r_14(bars: ArrayDict) -> np.ndarray:
    """Williams %R(14)."""
    return williams_r(bars, period=14)


def williams_r_scalp(bars: ArrayDict) -> np.ndarray:
    """Williams %R(10) — scalp variant with -90/-10 thresholds."""
    return williams_r(bars, period=10)


def stochastic_14_3_3(bars: ArrayDict) -> dict[str, np.ndarray]:
    return stochastic(bars, k=14, d=3, smooth_k=3)


def cci_20(bars: ArrayDict) -> np.ndarray:
    return cci(bars, period=20)


def mfi_14(bars: ArrayDict) -> np.ndarray:
    return mfi(bars, period=14)


def roc_10(bars: ArrayDict) -> np.ndarray:
    return roc(bars, period=10)


# ── ADX / Aroon / Parabolic SAR variants ──


def adx_14(bars: ArrayDict) -> dict[str, np.ndarray]:
    return adx(bars, period=14)


def adx_scalp(bars: ArrayDict) -> dict[str, np.ndarray]:
    """ADX(7-10) scalp variant — use period=8."""
    return adx(bars, period=8)


def aroon_25(bars: ArrayDict) -> dict[str, np.ndarray]:
    return aroon(bars, period=25)


def aroon_scalp(bars: ArrayDict) -> dict[str, np.ndarray]:
    """Aroon intraday (14-20) — use period=17."""
    return aroon(bars, period=17)


def parabolic_sar_scalp(bars: ArrayDict) -> dict[str, np.ndarray]:
    """Parabolic SAR scalp variant (step=0.03, max=0.30)."""
    return parabolic_sar(bars, step=0.03, max_step=0.30)


# ── Connors RSI / Fisher / Bollinger variants ──


def connors_rsi_3(bars: ArrayDict) -> np.ndarray:
    return connors_rsi(bars, rsi_period=3, streak_period=2, rank_period=100)


def fisher_transform_10(bars: ArrayDict) -> np.ndarray:
    return fisher_transform(bars, period=10)


def bollinger_20_2(bars: ArrayDict) -> dict[str, np.ndarray]:
    return bollinger(bars, period=20, nstd=2.0)


def bollinger_pctb(bars: ArrayDict) -> np.ndarray:
    return bb_percent_b(bars, period=20, nstd=2.0)


def keltner_20_15(bars: ArrayDict) -> dict[str, np.ndarray]:
    """Keltner Channels (20, 1.5 ATR)."""
    return keltner(bars, ema_period=20, atr_period=14, mult=1.5)


def keltner_fast(bars: ArrayDict) -> dict[str, np.ndarray]:
    """Keltner Channels fast variant (10 EMA / 1.0 ATR)."""
    return keltner(bars, ema_period=10, atr_period=10, mult=1.0)


def chopping_index_14(bars: ArrayDict) -> np.ndarray:
    return choppiness_index(bars, period=14)


def cmf_21(bars: ArrayDict) -> np.ndarray:
    return cmf(bars, period=21)


def cmf_scalp(bars: ArrayDict) -> np.ndarray:
    """CMF scalp variant period=12."""
    return cmf(bars, period=12)


def force_index_13(bars: ArrayDict) -> dict[str, np.ndarray]:
    return force_index(bars, period=13)


def elder_ray_13(bars: ArrayDict) -> dict[str, np.ndarray]:
    return elder_ray(bars, ema_p=13)


def elder_ray_scalp(bars: ArrayDict) -> dict[str, np.ndarray]:
    """Elder Ray scalp variant (EMA 9)."""
    return elder_ray(bars, ema_p=9)


def supertrend_10_3(bars: ArrayDict) -> dict[str, np.ndarray]:
    return supertrend(bars, atr_period=10, mult=3.0)


def opening_range_6(bars: ArrayDict) -> dict[str, np.ndarray]:
    return opening_range(bars, bars_in_range=6)


def pivot_points_standard(bars: ArrayDict) -> dict[str, np.ndarray]:
    """Pivot Points (Standard) — same as classic."""
    return pivot_points_classic(bars)


def zscore_20(bars: ArrayDict) -> np.ndarray:
    return zscore(bars, period=20)


# ── New TA indicators not previously implemented ──


def kama(bars_or_arr: ArrayDict | np.ndarray, period: int = 10,
         fast: int = 2, slow: int = 30, key: str = "close") -> np.ndarray:
    """Kaufman Adaptive Moving Average. Adapts speed by efficiency ratio.

    ER = |C - C[period]| / sum(|C - C[-1]|) over period.
    SC = (ER * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    KAMA_t = KAMA_{t-1} + SC * (C_t - KAMA_{t-1}).
    axis=trend.
    """
    x = bars_or_arr[key] if isinstance(bars_or_arr, dict) else bars_or_arr
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float64)
    if n <= period:
        return out
    change = np.abs(x[period:] - x[:-period])
    volatility = np.zeros(n - period, dtype=np.float64)
    diffs = np.abs(np.diff(x))
    for i in range(period, n):
        volatility[i - period] = np.sum(diffs[i - period:i])
    er = np.where(volatility > 0, change / volatility, 0.0)
    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    out[period] = x[period]
    for i in range(period + 1, n):
        out[i] = out[i - 1] + sc[i - period] * (x[i] - out[i - 1])
    return out


def hull_ma(bars_or_arr: ArrayDict | np.ndarray, period: int = 20,
            key: str = "close") -> np.ndarray:
    """Hull Moving Average. WMA(2*WMA(n/2) - WMA(n), sqrt(n)). axis=trend."""
    x = bars_or_arr[key] if isinstance(bars_or_arr, dict) else bars_or_arr
    x = np.asarray(x, dtype=np.float64)
    n = len(x)

    def _wma(arr: np.ndarray, p: int) -> np.ndarray:
        w = np.arange(1, p + 1, dtype=np.float64)
        ws = w.sum()
        out = np.full_like(arr, np.nan, dtype=np.float64)
        for i in range(p - 1, len(arr)):
            out[i] = np.dot(arr[i - p + 1:i + 1], w) / ws
        return out

    half = max(2, period // 2)
    sq = max(2, int(np.sqrt(period)))
    wma_half = _wma(x, half)
    wma_full = _wma(x, period)
    raw = 2.0 * wma_half - wma_full
    # WMA over the diff series; NaN-safe slice
    out = np.full(n, np.nan, dtype=np.float64)
    valid_from = period - 1
    if valid_from + sq <= n:
        seg = raw[valid_from:]
        out[valid_from + sq - 1:] = _wma(seg, sq)[sq - 1:]
    return out


def hma(bars_or_arr: ArrayDict | np.ndarray, period: int = 20,
        key: str = "close") -> np.ndarray:
    """HMA alias for Hull MA."""
    return hull_ma(bars_or_arr, period=period, key=key)


def alma(bars_or_arr: ArrayDict | np.ndarray, period: int = 9,
         offset: float = 0.85, sigma: float = 6.0, key: str = "close") -> np.ndarray:
    """Arnaud Legoux Moving Average. Gaussian-weighted with sigma+offset. axis=trend."""
    x = bars_or_arr[key] if isinstance(bars_or_arr, dict) else bars_or_arr
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float64)
    m = offset * (period - 1)
    s = period / sigma
    w = np.exp(-((np.arange(period) - m) ** 2) / (2 * s * s))
    ws = w.sum()
    for i in range(period - 1, n):
        out[i] = np.dot(x[i - period + 1:i + 1], w) / ws
    return out


def t3(bars_or_arr: ArrayDict | np.ndarray, period: int = 5,
       vfactor: float = 0.7, key: str = "close") -> np.ndarray:
    """T3 (Tillson) — 6-pass EMA with volume-factor blend. axis=trend."""
    x = bars_or_arr[key] if isinstance(bars_or_arr, dict) else bars_or_arr
    x = np.asarray(x, dtype=np.float64)
    e1 = _ema(x, period)
    e2 = _ema(e1, period)
    e3 = _ema(e2, period)
    e4 = _ema(e3, period)
    e5 = _ema(e4, period)
    e6 = _ema(e5, period)
    v = vfactor
    c1 = -v ** 3
    c2 = 3 * v ** 2 + 3 * v ** 3
    c3 = -6 * v ** 2 - 3 * v - 3 * v ** 3
    c4 = 1 + 3 * v + v ** 3 + 3 * v ** 2
    return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3


def zlema(bars_or_arr: ArrayDict | np.ndarray, period: int = 20,
          key: str = "close") -> np.ndarray:
    """Zero-Lag EMA: EMA(2*x - x[lag], period) where lag = (period-1)/2. axis=trend."""
    x = bars_or_arr[key] if isinstance(bars_or_arr, dict) else bars_or_arr
    x = np.asarray(x, dtype=np.float64)
    lag = (period - 1) // 2
    if lag <= 0:
        return _ema(x, period)
    shifted = np.concatenate([np.full(lag, x[0]), x[:-lag]])
    de = 2.0 * x - shifted
    return _ema(de, period)


def smma(bars_or_arr: ArrayDict | np.ndarray, period: int = 20,
         key: str = "close") -> np.ndarray:
    """Smoothed Moving Average — equivalent to Wilder smoothing. axis=trend."""
    x = bars_or_arr[key] if isinstance(bars_or_arr, dict) else bars_or_arr
    x = np.asarray(x, dtype=np.float64)
    return _wilder(x, period)


def wma(bars_or_arr: ArrayDict | np.ndarray, period: int = 20,
        key: str = "close") -> np.ndarray:
    """Weighted Moving Average (linear weights). axis=trend."""
    x = bars_or_arr[key] if isinstance(bars_or_arr, dict) else bars_or_arr
    x = np.asarray(x, dtype=np.float64)
    w = np.arange(1, period + 1, dtype=np.float64)
    ws = w.sum()
    out = np.full_like(x, np.nan, dtype=np.float64)
    for i in range(period - 1, len(x)):
        out[i] = np.dot(x[i - period + 1:i + 1], w) / ws
    return out


def natr(bars: ArrayDict, period: int = 14) -> np.ndarray:
    """Normalized ATR: 100 * ATR / close. axis=volatility_band."""
    a = _atr(bars, period)
    c = bars["close"]
    out = np.full_like(c, np.nan, dtype=np.float64)
    mask = c > 0
    out[mask] = 100.0 * a[mask] / c[mask]
    return out


def ulcer_index(bars: ArrayDict, period: int = 14) -> np.ndarray:
    """Ulcer Index — RMS of percentage drawdown from rolling high. axis=volatility_band."""
    c = bars["close"]
    n = len(c)
    out = np.full(n, np.nan, dtype=np.float64)
    rmax = _rolling_max(c, period)
    dd = np.where(rmax > 0, 100.0 * (c - rmax) / rmax, 0.0)
    dd2 = dd * dd
    # rolling mean of dd2
    for i in range(period - 1, n):
        out[i] = np.sqrt(np.mean(dd2[i - period + 1:i + 1]))
    return out


def chaikin_volatility(bars: ArrayDict, ema_period: int = 10, roc_period: int = 10) -> np.ndarray:
    """Chaikin Volatility — ROC of EMA of (high - low). axis=volatility_band."""
    h, l = bars["high"], bars["low"]
    rng = h - l
    em = _ema(rng, ema_period)
    out = np.full_like(em, np.nan, dtype=np.float64)
    n = len(em)
    if n > roc_period:
        prev = em[:-roc_period]
        out[roc_period:] = np.where(prev != 0, 100.0 * (em[roc_period:] - prev) / prev, np.nan)
    return out


def rvi(bars: ArrayDict, period: int = 10) -> dict[str, np.ndarray]:
    """Relative Vigor Index. axis=momentum_oscillator.

    num = SMA(close - open, period)
    den = SMA(high - low, period)
    rvi = num/den ; signal = SMA(rvi, 4)
    """
    o, h, l, c = bars["open"], bars["high"], bars["low"], bars["close"]
    num = _sma(c - o, period)
    den = _sma(h - l, period)
    out = np.full_like(c, np.nan, dtype=np.float64)
    mask = (den != 0) & np.isfinite(den)
    out[mask] = num[mask] / den[mask]
    sig = _sma(out, 4)
    return {"rvi": out, "signal": sig}


def coppock_curve(bars: ArrayDict, roc1: int = 14, roc2: int = 11, wma_p: int = 10) -> np.ndarray:
    """Coppock Curve = WMA(ROC(close, roc1) + ROC(close, roc2), wma_p). axis=momentum_oscillator."""
    c = bars["close"]
    r1 = np.full_like(c, np.nan, dtype=np.float64)
    r2 = np.full_like(c, np.nan, dtype=np.float64)
    if len(c) > roc1:
        r1[roc1:] = 100.0 * (c[roc1:] - c[:-roc1]) / np.where(c[:-roc1] != 0, c[:-roc1], np.nan)
    if len(c) > roc2:
        r2[roc2:] = 100.0 * (c[roc2:] - c[:-roc2]) / np.where(c[:-roc2] != 0, c[:-roc2], np.nan)
    summed = r1 + r2
    w = np.arange(1, wma_p + 1, dtype=np.float64)
    ws = w.sum()
    out = np.full_like(c, np.nan, dtype=np.float64)
    for i in range(wma_p - 1, len(c)):
        seg = summed[i - wma_p + 1:i + 1]
        if np.all(np.isfinite(seg)):
            out[i] = np.dot(seg, w) / ws
    return out


def chande_momentum(bars: ArrayDict, period: int = 14) -> np.ndarray:
    """Chande Momentum Oscillator. axis=momentum_oscillator."""
    c = bars["close"]
    diff = np.diff(c, prepend=c[0])
    up = np.where(diff > 0, diff, 0.0)
    dn = np.where(diff < 0, -diff, 0.0)
    n = len(c)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        su = np.sum(up[i - period + 1:i + 1])
        sd = np.sum(dn[i - period + 1:i + 1])
        if (su + sd) > 0:
            out[i] = 100.0 * (su - sd) / (su + sd)
    return out


def detrended_oscillator(bars: ArrayDict, period: int = 14) -> np.ndarray:
    """Detrended Price Oscillator alias (catalog name 'Detrended Oscillator')."""
    return dpo(bars, period=period)


def klinger_oscillator(bars: ArrayDict, fast: int = 34, slow: int = 55, signal: int = 13) -> dict[str, np.ndarray]:
    """Klinger Volume Oscillator. axis=volume_conviction.

    Trend t = sign((H+L+C)_t - (H+L+C)_{t-1}); dm_t = H_t - L_t (TR-like);
    cm starts at dm and resets/sums with trend changes; VF = V * trend * |2*dm/cm - 1| * 100.
    Simplified causal numpy version.
    """
    h, l, c, v = bars["high"], bars["low"], bars["close"], bars["volume"]
    hlc = h + l + c
    prev_hlc = np.concatenate([[hlc[0]], hlc[:-1]])
    trend = np.sign(hlc - prev_hlc)
    dm = h - l
    cm = np.zeros_like(dm)
    prev_trend = 0.0
    for i in range(len(dm)):
        if i == 0:
            cm[i] = dm[i]
        elif trend[i] == prev_trend:
            cm[i] = cm[i - 1] + dm[i]
        else:
            cm[i] = dm[i - 1] + dm[i] if i > 0 else dm[i]
        prev_trend = trend[i]
    safe_cm = np.where(cm > 0, cm, 1.0)
    vf = v * trend * np.abs(2.0 * dm / safe_cm - 1.0) * 100.0
    kvo = _ema(vf, fast) - _ema(vf, slow)
    sig = _ema(kvo, signal)
    return {"kvo": kvo, "signal": sig}


def eom(bars: ArrayDict, period: int = 14, divisor: float = 10000.0) -> np.ndarray:
    """Ease of Movement (Arms). axis=volume_conviction."""
    h, l, v = bars["high"], bars["low"], bars["volume"]
    mid = (h + l) / 2.0
    prev_mid = np.concatenate([[mid[0]], mid[:-1]])
    box = np.where((h - l) > 0, (v / divisor) / (h - l), 0.0)
    raw = (mid - prev_mid) / np.where(box > 0, box, np.nan)
    return _sma(raw, period)


def pvt(bars: ArrayDict) -> np.ndarray:
    """Price Volume Trend (cumulative). axis=volume_conviction."""
    c, v = bars["close"], bars["volume"]
    prev_c = np.concatenate([[c[0]], c[:-1]])
    pct = np.where(prev_c != 0, (c - prev_c) / prev_c, 0.0)
    return np.cumsum(pct * v)


def vwma(bars: ArrayDict, period: int = 20) -> np.ndarray:
    """Volume-Weighted Moving Average. axis=volume_conviction."""
    c, v = bars["close"], bars["volume"]
    pv = c * v
    num = _sma(pv, period)
    den = _sma(v, period)
    out = np.full_like(c, np.nan, dtype=np.float64)
    mask = (den != 0) & np.isfinite(den)
    out[mask] = num[mask] / den[mask]
    return out


def vroc(bars: ArrayDict, period: int = 14) -> np.ndarray:
    """Volume Rate of Change. axis=volume_conviction."""
    v = bars["volume"]
    out = np.full_like(v, np.nan, dtype=np.float64)
    if len(v) > period:
        prev = v[:-period]
        out[period:] = np.where(prev > 0, 100.0 * (v[period:] - prev) / prev, np.nan)
    return out


def correlation_coefficient(bars: ArrayDict, ref_close: np.ndarray | None = None,
                            period: int = 60) -> np.ndarray:
    """Rolling Pearson correlation of close vs ref_close (auto-correlation if ref None).

    axis=structure_geometry. For pair selection / xsym wrapper.
    """
    c = bars["close"].astype(np.float64)
    if ref_close is None:
        # Autocorrelation against lagged self as a degenerate fallback.
        ref = np.concatenate([[c[0]], c[:-1]])
    else:
        ref = np.asarray(ref_close, dtype=np.float64)
        if len(ref) != len(c):
            raise ValueError(f"ref_close length {len(ref)} != close length {len(c)}")
    out = np.full_like(c, np.nan, dtype=np.float64)
    for i in range(period - 1, len(c)):
        a = c[i - period + 1:i + 1]
        b = ref[i - period + 1:i + 1]
        sa = a.std()
        sb = b.std()
        if sa > 0 and sb > 0:
            out[i] = np.corrcoef(a, b)[0, 1]
    return out


def candlestick_pattern_score(bars: ArrayDict) -> np.ndarray:
    """Composite candlestick-pattern score in {-1, 0, +1}: +1 hammer/bullish-engulf,
    -1 shooting-star/bearish-engulf, 0 otherwise. axis=structure_geometry.
    Simplified per-bar implementation suitable as a confirmation token.
    """
    o, h, l, c = bars["open"], bars["high"], bars["low"], bars["close"]
    n = len(c)
    out = np.zeros(n, dtype=np.float64)
    body = np.abs(c - o)
    full = (h - l).clip(min=1e-12)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    # Hammer: small body, long lower shadow (>=2*body), upper shadow <= body
    hammer = (lower >= 2.0 * body) & (upper <= body) & (body > 0)
    # Shooting star: inverse
    shooter = (upper >= 2.0 * body) & (lower <= body) & (body > 0)
    out[hammer] = 1.0
    out[shooter] = -1.0
    # Engulfing (need previous bar)
    for i in range(1, n):
        if (c[i] > o[i]) and (c[i - 1] < o[i - 1]) \
                and (c[i] >= o[i - 1]) and (o[i] <= c[i - 1]):
            out[i] = 1.0
        elif (c[i] < o[i]) and (c[i - 1] > o[i - 1]) \
                and (o[i] >= c[i - 1]) and (c[i] <= o[i - 1]):
            out[i] = -1.0
    return out


def historical_volatility(bars: ArrayDict, period: int = 20,
                           bars_per_year: int = 252) -> np.ndarray:
    """Annualized historical volatility (log returns std). axis=volatility_band."""
    c = bars["close"]
    log_ret = np.diff(np.log(np.where(c > 0, c, np.nan)), prepend=0.0)
    out = np.full_like(c, np.nan, dtype=np.float64)
    for i in range(period - 1, len(c)):
        seg = log_ret[i - period + 1:i + 1]
        if np.all(np.isfinite(seg)):
            out[i] = seg.std(ddof=0) * np.sqrt(bars_per_year)
    return out


def true_strength_index_classic(bars: ArrayDict, long_p: int = 25, short_p: int = 13) -> np.ndarray:
    """TSI value (without signal) — convenience scalar alias of tsi()."""
    return tsi(bars, long_p=long_p, short_p=short_p)["tsi"]


def williams_alligator(bars: ArrayDict) -> dict[str, np.ndarray]:
    """Williams Alligator (jaw=SMMA13 shifted 8, teeth=SMMA8 shifted 5, lips=SMMA5 shifted 3).
    Returns same-length series; the shift is documented but applied as a forward lag (NaN-pad).
    axis=trend.
    """
    c = bars["close"]
    jaw_raw = _wilder(c, 13)
    teeth_raw = _wilder(c, 8)
    lips_raw = _wilder(c, 5)

    def _lag(arr: np.ndarray, k: int) -> np.ndarray:
        if k <= 0:
            return arr.copy()
        out = np.concatenate([np.full(k, np.nan), arr[:-k]])
        return out

    return {"jaw": _lag(jaw_raw, 8), "teeth": _lag(teeth_raw, 5), "lips": _lag(lips_raw, 3)}


def gator_oscillator(bars: ArrayDict) -> dict[str, np.ndarray]:
    """Gator = |jaw - teeth| (upper), -|teeth - lips| (lower)."""
    a = williams_alligator(bars)
    upper = np.abs(a["jaw"] - a["teeth"])
    lower = -np.abs(a["teeth"] - a["lips"])
    return {"upper": upper, "lower": lower}


def disparity_index(bars: ArrayDict, period: int = 14) -> np.ndarray:
    """100 * (close - SMA(close,p)) / SMA(close,p). axis=mean_reversion."""
    c = bars["close"]
    s = _sma(c, period)
    out = np.full_like(c, np.nan, dtype=np.float64)
    mask = s > 0
    out[mask] = 100.0 * (c[mask] - s[mask]) / s[mask]
    return out


def percent_price_oscillator(bars: ArrayDict, fast: int = 12, slow: int = 26,
                              signal: int = 9) -> dict[str, np.ndarray]:
    """Alias of PPO."""
    return ppo(bars, fast=fast, slow=slow, signal=signal)


def schaff_trend_cycle(bars: ArrayDict, fast: int = 23, slow: int = 50,
                       cycle: int = 10) -> np.ndarray:
    """Schaff Trend Cycle — double-Stochastic of MACD. axis=trend."""
    c = bars["close"]
    macd_line = _ema(c, fast) - _ema(c, slow)
    n = len(c)
    out1 = np.full(n, np.nan, dtype=np.float64)
    for i in range(cycle - 1, n):
        seg = macd_line[i - cycle + 1:i + 1]
        lo, hi = np.nanmin(seg), np.nanmax(seg)
        if hi > lo:
            out1[i] = 100.0 * (macd_line[i] - lo) / (hi - lo)
    # Smooth out1, then second stochastic
    f1 = _ema(np.nan_to_num(out1, nan=50.0), 3)
    out2 = np.full(n, np.nan, dtype=np.float64)
    for i in range(cycle - 1, n):
        seg = f1[i - cycle + 1:i + 1]
        lo, hi = np.nanmin(seg), np.nanmax(seg)
        if hi > lo:
            out2[i] = 100.0 * (f1[i] - lo) / (hi - lo)
    return _ema(np.nan_to_num(out2, nan=50.0), 3)


def mass_index(bars: ArrayDict, ema_p: int = 9, sum_p: int = 25) -> np.ndarray:
    """Mass Index — reversal detector by EMA(H-L)/EMA(EMA(H-L)) summed. axis=volatility_band."""
    h, l = bars["high"], bars["low"]
    rng = h - l
    e1 = _ema(rng, ema_p)
    e2 = _ema(e1, ema_p)
    ratio = np.where(e2 > 0, e1 / e2, np.nan)
    out = np.full_like(rng, np.nan, dtype=np.float64)
    for i in range(sum_p - 1, len(rng)):
        seg = ratio[i - sum_p + 1:i + 1]
        if np.all(np.isfinite(seg)):
            out[i] = seg.sum()
    return out


def stc(bars: ArrayDict) -> np.ndarray:
    """Alias of Schaff Trend Cycle (default params)."""
    return schaff_trend_cycle(bars)


def klinger(bars: ArrayDict) -> dict[str, np.ndarray]:
    """Alias of klinger_oscillator (default params)."""
    return klinger_oscillator(bars)


def williams_alligator_jaw(bars: ArrayDict) -> np.ndarray:
    """Jaw line of Williams Alligator."""
    return williams_alligator(bars)["jaw"]


def williams_alligator_teeth(bars: ArrayDict) -> np.ndarray:
    return williams_alligator(bars)["teeth"]


def williams_alligator_lips(bars: ArrayDict) -> np.ndarray:
    return williams_alligator(bars)["lips"]


# ---------------------------------------------------------------------------
# Registry — name → (callable, default params, param-grid for stability)
# Banned pairs per Phase 1 redundancy matrix are NOT enforced here; runner respects ban list.
# ---------------------------------------------------------------------------


SignalFn = Callable[..., np.ndarray]

REGISTRY: dict[str, dict[str, Any]] = {
    "MACD_12_26_9": {
        "fn": sig_macd_zero,
        "default": {"fast": 12, "slow": 26, "signal": 9},
        "grid": [
            {"fast": 11, "slow": 24, "signal": 8},
            {"fast": 12, "slow": 26, "signal": 9},
            {"fast": 13, "slow": 28, "signal": 10},
        ],
    },
    "BB_20_2": {
        "fn": sig_bb_mean_rev,
        "default": {"period": 20, "nstd": 2.0},
        "grid": [
            {"period": 18, "nstd": 1.8},
            {"period": 20, "nstd": 2.0},
            {"period": 22, "nstd": 2.2},
        ],
    },
    "BB_pctB": {
        "fn": sig_bb_pctb,
        "default": {"period": 20, "nstd": 2.0, "lo": 0.05, "hi": 0.95},
        "grid": [
            {"period": 18, "nstd": 2.0, "lo": 0.05, "hi": 0.95},
            {"period": 20, "nstd": 2.0, "lo": 0.05, "hi": 0.95},
            {"period": 22, "nstd": 2.0, "lo": 0.05, "hi": 0.95},
        ],
    },
    "Keltner_20_1.5": {
        "fn": sig_keltner_breakout,
        "default": {"ema_period": 20, "atr_period": 14, "mult": 1.5},
        "grid": [
            {"ema_period": 18, "atr_period": 14, "mult": 1.4},
            {"ema_period": 20, "atr_period": 14, "mult": 1.5},
            {"ema_period": 22, "atr_period": 14, "mult": 1.6},
        ],
    },
    "OBV": {
        "fn": sig_obv_trend,
        "default": {"ema_span": 20},
        "grid": [{"ema_span": 18}, {"ema_span": 20}, {"ema_span": 22}],
    },
    "Stoch_14_3_3": {
        "fn": sig_stoch,
        "default": {"k": 14, "d": 3, "sm": 3, "lo": 20, "hi": 80},
        "grid": [
            {"k": 12, "d": 3, "sm": 3, "lo": 20, "hi": 80},
            {"k": 14, "d": 3, "sm": 3, "lo": 20, "hi": 80},
            {"k": 16, "d": 3, "sm": 3, "lo": 20, "hi": 80},
        ],
    },
    "Williams_R_14": {
        "fn": sig_williams_r,
        "default": {"period": 14, "lo": -80, "hi": -20},
        "grid": [
            {"period": 12, "lo": -80, "hi": -20},
            {"period": 14, "lo": -80, "hi": -20},
            {"period": 16, "lo": -80, "hi": -20},
        ],
    },
    "CCI_20": {
        "fn": sig_cci,
        "default": {"period": 20, "lo": -100, "hi": 100},
        "grid": [
            {"period": 18, "lo": -100, "hi": 100},
            {"period": 20, "lo": -100, "hi": 100},
            {"period": 22, "lo": -100, "hi": 100},
        ],
    },
    "MFI_14": {
        "fn": sig_mfi,
        "default": {"period": 14, "lo": 20, "hi": 80},
        "grid": [
            {"period": 12, "lo": 20, "hi": 80},
            {"period": 14, "lo": 20, "hi": 80},
            {"period": 16, "lo": 20, "hi": 80},
        ],
    },
    "Fisher_Transform_10": {
        "fn": sig_fisher,
        "default": {"period": 10, "lo": -1.5, "hi": 1.5},
        "grid": [
            {"period": 9, "lo": -1.5, "hi": 1.5},
            {"period": 10, "lo": -1.5, "hi": 1.5},
            {"period": 11, "lo": -1.5, "hi": 1.5},
        ],
    },
    "Connors_RSI_3": {
        "fn": sig_connors_rsi,
        "default": {"rsi_p": 3, "streak_p": 2, "rank_p": 100, "lo": 10, "hi": 90},
        "grid": [
            {"rsi_p": 3, "streak_p": 2, "rank_p": 90, "lo": 10, "hi": 90},
            {"rsi_p": 3, "streak_p": 2, "rank_p": 100, "lo": 10, "hi": 90},
            {"rsi_p": 3, "streak_p": 2, "rank_p": 110, "lo": 10, "hi": 90},
        ],
    },
    "Supertrend_10_3": {
        "fn": sig_supertrend,
        "default": {"atr_period": 10, "mult": 3.0},
        "grid": [
            {"atr_period": 9, "mult": 2.7},
            {"atr_period": 10, "mult": 3.0},
            {"atr_period": 11, "mult": 3.3},
        ],
    },
}


# Mathematical redundancy ban list per Phase 1
REDUNDANT_PAIRS = [
    ("Williams_R_14", "Stoch_14_3_3"),  # Williams %R = StochK*-1 - 100
    ("MACD_12_26_9", "EMA_cross"),  # MACD zero-cross ≡ EMA(12,26) cross
]


# ---------------------------------------------------------------------------
# INDICATOR_AXIS — maps every public indicator function to its informational
# axis. Reference: lab.knowledge.indicators.informational_axes().
# Six axes: trend / momentum_oscillator / volatility_band / volume_conviction
# / mean_reversion / structure_geometry.
# ---------------------------------------------------------------------------

INDICATOR_AXIS: dict[str, str] = {
    # ── trend ──
    "ema": "trend",
    "ema_pair": "trend",
    "sma": "trend",
    "dema": "trend",
    "tema": "trend",
    "macd": "trend",
    "macd_histogram": "trend",
    "supertrend": "trend",
    "adx": "trend",
    "aroon": "trend",
    "parabolic_sar": "trend",
    "ichimoku": "trend",
    "vortex": "trend",
    "trix": "trend",
    # ── momentum_oscillator ──
    "rsi": "momentum_oscillator",
    "stochastic": "momentum_oscillator",
    "stochastic_rsi": "momentum_oscillator",
    "williams_r": "momentum_oscillator",
    "cci": "momentum_oscillator",
    "mfi": "momentum_oscillator",
    "fisher_transform": "momentum_oscillator",
    "ultimate_oscillator": "momentum_oscillator",
    "awesome_oscillator": "momentum_oscillator",
    "dpo": "momentum_oscillator",
    "ppo": "momentum_oscillator",
    "roc": "momentum_oscillator",
    "momentum": "momentum_oscillator",
    "kst": "momentum_oscillator",
    "tsi": "momentum_oscillator",
    # ── volatility_band ──
    "bollinger": "volatility_band",
    "bb_width": "volatility_band",
    "keltner": "volatility_band",
    "keltner_width": "volatility_band",
    "donchian": "volatility_band",
    "atr": "volatility_band",
    "true_range": "volatility_band",
    "ttm_squeeze": "volatility_band",
    "choppiness_index": "volatility_band",
    "chop_idx": "volatility_band",
    "heikin_ashi": "volatility_band",
    # ── volume_conviction ──
    "obv": "volume_conviction",
    "vwap": "volume_conviction",
    "volume_expansion": "volume_conviction",
    "cmf": "volume_conviction",
    "ad_line": "volume_conviction",
    "force_index": "volume_conviction",
    "elder_ray": "volume_conviction",
    "volume_profile": "volume_conviction",
    # ── mean_reversion ──
    "bb_percent_b": "mean_reversion",
    "connors_rsi": "mean_reversion",
    "zscore": "mean_reversion",
    # ── structure_geometry ──
    "zigzag": "structure_geometry",
    "pivot_points_classic": "structure_geometry",
    "pivot_points_fib": "structure_geometry",
    "opening_range": "structure_geometry",
    # ── PATH B (2026-05-29): catalog name aliases + new indicators ──
    # trend aliases / new
    "ema_9": "trend",
    "ema_20": "trend",
    "ema_21": "trend",
    "ema_50": "trend",
    "ema_200": "trend",
    "sma_20": "trend",
    "sma_50": "trend",
    "sma_200": "trend",
    "macd_hist_raschke": "trend",
    "macd_hist_scalp": "trend",
    "macd_hist_smoother": "trend",
    "macd_12_26_9": "trend",
    "adx_14": "trend",
    "adx_scalp": "trend",
    "aroon_25": "trend",
    "aroon_scalp": "trend",
    "parabolic_sar_scalp": "trend",
    "supertrend_10_3": "trend",
    "kama": "trend",
    "hull_ma": "trend",
    "hma": "trend",
    "alma": "trend",
    "t3": "trend",
    "zlema": "trend",
    "smma": "trend",
    "wma": "trend",
    "williams_alligator": "trend",
    "williams_alligator_jaw": "trend",
    "williams_alligator_teeth": "trend",
    "williams_alligator_lips": "trend",
    "gator_oscillator": "trend",
    "schaff_trend_cycle": "trend",
    "stc": "trend",
    # momentum_oscillator aliases / new
    "rsi_2": "momentum_oscillator",
    "rsi_14": "momentum_oscillator",
    "rsi_scalp": "momentum_oscillator",
    "williams_r_14": "momentum_oscillator",
    "williams_r_scalp": "momentum_oscillator",
    "stochastic_14_3_3": "momentum_oscillator",
    "cci_20": "momentum_oscillator",
    "mfi_14": "momentum_oscillator",
    "roc_10": "momentum_oscillator",
    "fisher_transform_10": "momentum_oscillator",
    "rvi": "momentum_oscillator",
    "coppock_curve": "momentum_oscillator",
    "chande_momentum": "momentum_oscillator",
    "detrended_oscillator": "momentum_oscillator",
    "true_strength_index_classic": "momentum_oscillator",
    "percent_price_oscillator": "momentum_oscillator",
    # volatility_band aliases / new
    "atr_14": "volatility_band",
    "atr_scalp": "volatility_band",
    "bollinger_20_2": "volatility_band",
    "keltner_20_15": "volatility_band",
    "keltner_fast": "volatility_band",
    "chopping_index_14": "volatility_band",
    "natr": "volatility_band",
    "ulcer_index": "volatility_band",
    "chaikin_volatility": "volatility_band",
    "historical_volatility": "volatility_band",
    "mass_index": "volatility_band",
    # volume_conviction aliases / new
    "cmf_21": "volume_conviction",
    "cmf_scalp": "volume_conviction",
    "force_index_13": "volume_conviction",
    "elder_ray_13": "volume_conviction",
    "elder_ray_scalp": "volume_conviction",
    "klinger_oscillator": "volume_conviction",
    "klinger": "volume_conviction",
    "eom": "volume_conviction",
    "pvt": "volume_conviction",
    "vwma": "volume_conviction",
    "vroc": "volume_conviction",
    # mean_reversion aliases / new
    "bollinger_pctb": "mean_reversion",
    "connors_rsi_3": "mean_reversion",
    "zscore_20": "mean_reversion",
    "disparity_index": "mean_reversion",
    # structure_geometry aliases / new
    "donchian_20_up": "structure_geometry",
    "donchian_20_dn": "structure_geometry",
    "donchian_40_up": "structure_geometry",
    "donchian_40_dn": "structure_geometry",
    "opening_range_6": "structure_geometry",
    "pivot_points_standard": "structure_geometry",
    "correlation_coefficient": "structure_geometry",
    "candlestick_pattern_score": "structure_geometry",
}


def axis_for(indicator_name: str) -> str:
    """Return the informational axis ('trend', 'momentum_oscillator', etc.) for an indicator.

    Falls back to 'unknown' for unregistered names so callers don't crash.
    """
    return INDICATOR_AXIS.get(indicator_name, "unknown")


def indicators_by_axis() -> dict[str, list[str]]:
    """Inverse map: axis → list of indicator names registered to it."""
    out: dict[str, list[str]] = {}
    for name, ax in INDICATOR_AXIS.items():
        out.setdefault(ax, []).append(name)
    return {k: sorted(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Smoke-test entrypoint
# ---------------------------------------------------------------------------


def _smoke_synthetic(n: int = 5000) -> ArrayDict:
    rng = np.random.default_rng(0)
    close = 100.0 + np.cumsum(rng.normal(0, 0.1, n))
    high = close + np.abs(rng.normal(0, 0.05, n))
    low = close - np.abs(rng.normal(0, 0.05, n))
    open_ = close + rng.normal(0, 0.05, n)
    vol = rng.lognormal(10, 0.3, n)
    return {"open": open_, "high": high, "low": low, "close": close, "volume": vol}


def _try_load_real_bars():
    """Try to load real 5min AAPL bars; return None on failure."""
    candidates = [
        "/Volumes/ZG-2TB/zg/gabriel_store/Minutes TimeFrames/5Min/AAPL/2026-04.parquet",
        "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/s&p500-ticker-mastery/data/Minutes TimeFrames/5Min/AAPL/2026-04.parquet",
    ]
    for p in candidates:
        try:
            import pandas as pd
            df = pd.read_parquet(p)
            cols = {c.lower(): c for c in df.columns}
            return {
                "open": df[cols["open"]].to_numpy(dtype=np.float64),
                "high": df[cols["high"]].to_numpy(dtype=np.float64),
                "low": df[cols["low"]].to_numpy(dtype=np.float64),
                "close": df[cols["close"]].to_numpy(dtype=np.float64),
                "volume": df[cols["volume"]].to_numpy(dtype=np.float64),
            }
        except (ImportError, FileNotFoundError, KeyError):
            continue
    return None


_SUMMARY_FUNCS = {"volume_profile"}  # legitimately returns non-per-bar arrays


def _smoke_one(name: str, fn, bars: ArrayDict, params: dict | None = None) -> tuple[bool, str]:
    """Run one indicator function, check shape + warm-up NaN policy."""
    try:
        out = fn(bars, **(params or {}))
        n = len(bars["close"])
        if isinstance(out, dict):
            if name not in _SUMMARY_FUNCS:
                for k, v in out.items():
                    arr = np.asarray(v) if v is not None else None
                    if arr is None or arr.ndim == 0:
                        continue  # scalar
                    if len(arr) != n:
                        return False, f"shape mismatch: {k}={len(arr)} vs n={n}"
            # Pick a representative key for range
            first_arr = next((np.asarray(v) for v in out.values()
                              if v is not None and hasattr(v, "__len__") and len(np.asarray(v)) > 0), None)
            if first_arr is None:
                return True, "ok(scalar-only dict)"
            finite = first_arr[np.isfinite(first_arr)]
            rng = f"[{finite.min():.3g}, {finite.max():.3g}]" if len(finite) else "[empty]"
        else:
            arr = np.asarray(out)
            if len(arr) != n:
                return False, f"shape mismatch: {len(arr)} vs n={n}"
            finite = arr[np.isfinite(arr)]
            rng = f"[{finite.min():.3g}, {finite.max():.3g}]" if len(finite) else "[empty]"
        return True, rng
    except Exception as e:  # noqa: BLE001
        return False, f"EXC: {type(e).__name__}: {e}"


_SMOKE_SUITE: list[tuple[str, Callable, dict | None]] = [
    # trend
    ("ema", ema, {"period": 21}),
    ("ema_pair", ema_pair, {"fast": 9, "slow": 21}),
    ("sma", sma, {"period": 20}),
    ("dema", dema, {"period": 20}),
    ("tema", tema, {"period": 20}),
    ("macd", macd, None),
    ("macd_histogram", macd_histogram, {"fast": 5, "slow": 13, "signal": 1}),
    ("supertrend", supertrend, None),
    ("adx", adx, {"period": 14}),
    ("aroon", aroon, {"period": 25}),
    ("parabolic_sar", parabolic_sar, None),
    ("ichimoku", ichimoku, None),
    ("vortex", vortex, {"period": 14}),
    ("trix", trix, {"period": 15, "signal": 9}),
    # momentum_oscillator
    ("rsi", rsi, {"period": 14}),
    ("stochastic", stochastic, None),
    ("stochastic_rsi", stochastic_rsi, None),
    ("williams_r", williams_r, {"period": 14}),
    ("cci", cci, {"period": 20}),
    ("mfi", mfi, {"period": 14}),
    ("fisher_transform", fisher_transform, {"period": 10}),
    ("ultimate_oscillator", ultimate_oscillator, None),
    ("awesome_oscillator", awesome_oscillator, None),
    ("dpo", dpo, {"period": 20}),
    ("ppo", ppo, None),
    ("roc", roc, {"period": 10}),
    ("momentum", momentum, {"period": 10}),
    ("kst", kst, None),
    ("tsi", tsi, None),
    # volatility_band
    ("bollinger", bollinger, None),
    ("bb_width", bb_width, None),
    ("keltner", keltner, None),
    ("keltner_width", keltner_width, None),
    ("donchian", donchian, {"period": 20}),
    ("atr", atr, {"period": 14}),
    ("true_range", true_range, None),
    ("ttm_squeeze", ttm_squeeze, None),
    ("choppiness_index", choppiness_index, {"period": 14}),
    ("chop_idx", chop_idx, {"period": 14}),
    ("heikin_ashi", heikin_ashi, None),
    # volume_conviction
    ("obv", obv, None),
    ("vwap", vwap, None),
    ("volume_expansion", volume_expansion, None),
    ("cmf", cmf, {"period": 21}),
    ("ad_line", ad_line, None),
    ("force_index", force_index, {"period": 13}),
    ("elder_ray", elder_ray, {"ema_p": 13}),
    ("volume_profile", volume_profile, {"bins": 24}),
    # mean_reversion
    ("bb_percent_b", bb_percent_b, None),
    ("connors_rsi", connors_rsi, None),
    ("zscore", zscore, {"period": 20}),
    # structure_geometry
    ("zigzag", zigzag, {"threshold_pct": 3.0}),
    ("pivot_points_classic", pivot_points_classic, None),
    ("pivot_points_fib", pivot_points_fib, None),
    ("opening_range", opening_range, {"bars_in_range": 6}),
    # ── PATH B additions (2026-05-29) ──
    ("donchian_20_up", donchian_20_up, None),
    ("donchian_40_up", donchian_40_up, None),
    ("rsi_2", rsi_2, None),
    ("ema_9", ema_9, None),
    ("ema_20", ema_20, None),
    ("macd_hist_raschke", macd_hist_raschke, None),
    ("macd_hist_scalp", macd_hist_scalp, None),
    ("williams_r_scalp", williams_r_scalp, None),
    ("adx_scalp", adx_scalp, None),
    ("aroon_scalp", aroon_scalp, None),
    ("kama", kama, None),
    ("hull_ma", hull_ma, None),
    ("alma", alma, None),
    ("t3", t3, None),
    ("zlema", zlema, None),
    ("smma", smma, None),
    ("wma", wma, None),
    ("natr", natr, None),
    ("ulcer_index", ulcer_index, None),
    ("chaikin_volatility", chaikin_volatility, None),
    ("rvi", rvi, None),
    ("coppock_curve", coppock_curve, None),
    ("chande_momentum", chande_momentum, None),
    ("klinger_oscillator", klinger_oscillator, None),
    ("eom", eom, None),
    ("pvt", pvt, None),
    ("vwma", vwma, None),
    ("vroc", vroc, None),
    ("disparity_index", disparity_index, None),
    ("historical_volatility", historical_volatility, None),
    ("mass_index", mass_index, None),
    ("schaff_trend_cycle", schaff_trend_cycle, None),
    ("williams_alligator", williams_alligator, None),
    ("gator_oscillator", gator_oscillator, None),
    ("candlestick_pattern_score", candlestick_pattern_score, None),
    ("correlation_coefficient", correlation_coefficient, None),
]


if __name__ == "__main__":
    bars = _try_load_real_bars()
    src = "real(AAPL 5min)"
    if bars is None or len(bars["close"]) < 200:
        bars = _smoke_synthetic(5000)
        src = "synthetic(n=5000)"
    n = len(bars["close"])
    print(f"# Indicator smoke test — bars source: {src}, n={n}")
    print(f"# Total indicators registered to INDICATOR_AXIS: {len(INDICATOR_AXIS)}")
    per_axis: dict[str, int] = {}
    for ax in INDICATOR_AXIS.values():
        per_axis[ax] = per_axis.get(ax, 0) + 1
    for ax in sorted(per_axis):
        print(f"#   axis={ax:22s} count={per_axis[ax]}")
    print()
    print(f"{'STATUS':<8}{'AXIS':<24}{'NAME':<24}{'NOTES'}")
    ok = fail = 0
    for name, fn, params in _SMOKE_SUITE:
        passed, note = _smoke_one(name, fn, bars, params)
        ax = axis_for(name)
        status = "OK" if passed else "FAIL"
        if passed:
            ok += 1
        else:
            fail += 1
        print(f"{status:<8}{ax:<24}{name:<24}{note}")
    print()
    print(f"# smoke summary: {ok} ok / {fail} fail / {len(_SMOKE_SUITE)} total")
    # Also run signal-rule wrappers on synthetic for non-zero coverage report
    print()
    print("# Signal-rule wrappers (non-zero bar count):")
    for sig_name, cfg in REGISTRY.items():
        try:
            sig = cfg["fn"](bars, **cfg["default"])
            nz = int(np.sum(sig != 0))
            print(f"  {sig_name:25s} non-zero bars={nz}/{n}")
        except Exception as e:  # noqa: BLE001
            print(f"  {sig_name:25s} EXC: {type(e).__name__}: {e}")
