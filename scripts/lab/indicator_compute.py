"""indicator_compute.py — Pure-NumPy indicator implementations.

Each indicator returns either a value series or (signal, info) where signal is in {-1, 0, +1}
(short, flat, long). All indicators expect a Bars dict with numpy arrays: open, high, low, close,
volume — all same length, oldest first.

Settings follow the Phase 0/1 hardening plan:
- Wilder's smoothing (com=p-1) where TA convention specifies; EWM(span) noted otherwise.
- Cost model: 5 bps per side baseline, VIX-multiplied. Applied in the eval driver, not here.
- No look-ahead: all indicators are causal (shift-1 when using same-bar close to trigger).
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


if __name__ == "__main__":
    # Sanity: synthetic OHLC
    rng = np.random.default_rng(0)
    n = 5000
    close = 100.0 + np.cumsum(rng.normal(0, 0.1, n))
    high = close + np.abs(rng.normal(0, 0.05, n))
    low = close - np.abs(rng.normal(0, 0.05, n))
    open_ = close + rng.normal(0, 0.05, n)
    vol = rng.lognormal(10, 0.3, n)
    bars = {"open": open_, "high": high, "low": low, "close": close, "volume": vol}
    for name, cfg in REGISTRY.items():
        sig = cfg["fn"](bars, **cfg["default"])
        nz = int(np.sum(sig != 0))
        print(f"  {name:25s} non-zero bars={nz}/{n}")
