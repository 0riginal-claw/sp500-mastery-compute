"""signal_rules.py — Strategy-rule signal generators (PATH B, 2026-05-29).

A signal rule is a small composite function that takes already-computed indicator
series (passed as a dict) and returns a per-bar boolean signal in {-1, 0, +1}
(short/flat/long). These are the "catalog as a strategy rule" entries that don't
fit cleanly as a single indicator function.

Each rule:
- Takes a `bars` dict (open, high, low, close, volume — numpy arrays, oldest first)
- Returns np.ndarray of int8 in {-1, 0, +1}, same length as bars
- Strictly causal (uses only data up to and including the current bar)
- Documents the inputs and threshold in its docstring

Per CLAUDE.md methodology: these rules are CONFIRMATION layers / TRIGGERS — never
standalone signals. They live alongside a regime gate + bias + exit in a strategy
hypothesis dict.

RULE_REGISTRY maps each rule_name → (callable, axis, role).
SIGNAL_RULE_AXIS maps each rule_name → axis (for indicators_by_axis lookups).
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from . import indicator_compute as ic

ArrayDict = dict[str, np.ndarray]


# ---------------------------------------------------------------------------
# Helper: causal cross detection
# ---------------------------------------------------------------------------


def _cross_up(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """True where a crosses above b on this bar (a[t] > b[t] AND a[t-1] <= b[t-1])."""
    if len(a) == 0:
        return np.zeros(0, dtype=bool)
    prev_a = np.concatenate([[a[0]], a[:-1]])
    prev_b = np.concatenate([[b[0]], b[:-1]])
    return (a > b) & (prev_a <= prev_b)


def _cross_dn(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0:
        return np.zeros(0, dtype=bool)
    prev_a = np.concatenate([[a[0]], a[:-1]])
    prev_b = np.concatenate([[b[0]], b[:-1]])
    return (a < b) & (prev_a >= prev_b)


# ---------------------------------------------------------------------------
# Strategy-rule signals
# ---------------------------------------------------------------------------


def vwap_touch_reject(bars: ArrayDict, tolerance_pct: float = 0.1) -> np.ndarray:
    """Long when price touches VWAP from above and rejects up; short on opposite.

    Touch: bar's low within tolerance_pct of VWAP AND close > VWAP (long-reject).
    Inverse for short.
    axis=volume_conviction.
    """
    v = ic.vwap(bars)
    c, l, h = bars["close"], bars["low"], bars["high"]
    n = len(c)
    out = np.zeros(n, dtype=np.int8)
    tol = tolerance_pct / 100.0
    touched_below = (l <= v * (1.0 + tol)) & (l >= v * (1.0 - tol))
    touched_above = (h >= v * (1.0 - tol)) & (h <= v * (1.0 + tol))
    out[touched_below & (c > v)] = 1
    out[touched_above & (c < v)] = -1
    return out


def ema_stack_bull(bars: ArrayDict) -> np.ndarray:
    """+1 when EMA(9) > EMA(21) > EMA(50) (bullish stack); -1 inverse; 0 mixed.

    axis=trend (used as bias filter).
    """
    e9 = ic.ema(bars, period=9)
    e21 = ic.ema(bars, period=21)
    e50 = ic.ema(bars, period=50)
    out = np.zeros(len(e9), dtype=np.int8)
    out[(e9 > e21) & (e21 > e50)] = 1
    out[(e9 < e21) & (e21 < e50)] = -1
    return out


def donchian_breakout_with_volume(bars: ArrayDict, period: int = 20,
                                    vol_mult: float = 1.5) -> np.ndarray:
    """+1 on new Donchian-`period` UP breakout AND volume >= vol_mult * SMA(v,period).

    Inverse for DN breakout. axis=structure_geometry (entry trigger w/ confirm).
    """
    d = ic.donchian(bars, period=period)
    c, v = bars["close"], bars["volume"]
    avg_v = ic._sma(v, period)
    vol_ok = v >= (vol_mult * avg_v)
    out = np.zeros(len(c), dtype=np.int8)
    # New high vs previous bar's upper
    prev_upper = np.concatenate([[d["upper"][0]], d["upper"][:-1]])
    prev_lower = np.concatenate([[d["lower"][0]], d["lower"][:-1]])
    out[(c > prev_upper) & vol_ok] = 1
    out[(c < prev_lower) & vol_ok] = -1
    return out


def rsi_oversold_in_uptrend(bars: ArrayDict, rsi_p: int = 14, rsi_lo: float = 30.0,
                              ema_p: int = 200) -> np.ndarray:
    """+1 when RSI < rsi_lo AND close > EMA(ema_p) (pullback in uptrend).

    axis=mean_reversion (timing within bias).
    """
    r = ic.rsi(bars, period=rsi_p)
    e = ic.ema(bars, period=ema_p)
    c = bars["close"]
    out = np.zeros(len(c), dtype=np.int8)
    out[(r < rsi_lo) & (c > e)] = 1
    out[(r > (100.0 - rsi_lo)) & (c < e)] = -1
    return out


def macd_zero_cross_with_adx_gate(bars: ArrayDict, fast: int = 12, slow: int = 26,
                                    signal: int = 9, adx_min: float = 20.0) -> np.ndarray:
    """+1 when MACD line crosses above zero AND ADX > adx_min (trending).

    Inverse for cross below. axis=trend (entry trigger gated by regime).
    """
    m = ic.macd(bars, fast=fast, slow=slow, signal=signal)
    a = ic.adx(bars, period=14)
    line = m["macd"]
    zero = np.zeros_like(line)
    cu = _cross_up(line, zero)
    cd = _cross_dn(line, zero)
    adx_ok = a["adx"] >= adx_min
    out = np.zeros(len(line), dtype=np.int8)
    out[cu & adx_ok] = 1
    out[cd & adx_ok] = -1
    return out


def bb_squeeze_release_bull(bars: ArrayDict, bb_p: int = 20, bb_nstd: float = 2.0,
                              kc_p: int = 20, kc_mult: float = 1.5) -> np.ndarray:
    """TTM Squeeze release: +1 when BB exits inside Keltner AND MACD-hist rising.

    -1 when exits AND hist falling. axis=volatility_band (regime + trigger).
    """
    sq = ic.ttm_squeeze(bars, bb_period=bb_p, bb_std=bb_nstd,
                       kc_ema=kc_p, kc_atr=14, kc_mult=kc_mult)
    in_sq = np.asarray(sq["squeeze_on"]).astype(bool)
    # Released = was in squeeze yesterday, not today
    prev_in = np.concatenate([[in_sq[0]], in_sq[:-1]])
    released = prev_in & (~in_sq)
    hist = ic.macd_histogram(bars, fast=12, slow=26, signal=9)
    prev_hist = np.concatenate([[hist[0]], hist[:-1]])
    rising = hist > prev_hist
    out = np.zeros(len(in_sq), dtype=np.int8)
    out[released & rising] = 1
    out[released & (~rising)] = -1
    return out


def opening_range_breakout(bars: ArrayDict, bars_in_range: int = 6) -> np.ndarray:
    """+1 when close > opening-range high; -1 when close < opening-range low.

    axis=structure_geometry (time-anchored trigger).
    """
    orng = ic.opening_range(bars, bars_in_range=bars_in_range)
    c = bars["close"]
    out = np.zeros(len(c), dtype=np.int8)
    or_high = orng.get("or_high", orng.get("upper"))
    or_low = orng.get("or_low", orng.get("lower"))
    if or_high is not None and or_low is not None:
        out[c > or_high] = 1
        out[c < or_low] = -1
    return out


def supertrend_flip(bars: ArrayDict, atr_period: int = 10, mult: float = 3.0) -> np.ndarray:
    """+1 on supertrend flip-to-bull; -1 on flip-to-bear. axis=trend."""
    st = ic.supertrend(bars, atr_period=atr_period, mult=mult)
    trend = st["trend"].astype(np.int8)
    prev = np.concatenate([[trend[0]], trend[:-1]])
    out = np.zeros(len(trend), dtype=np.int8)
    out[(trend == 1) & (prev != 1)] = 1
    out[(trend == -1) & (prev != -1)] = -1
    return out


def keltner_breakout_with_volume(bars: ArrayDict, ema_p: int = 20, atr_p: int = 14,
                                   mult: float = 1.5, vol_mult: float = 1.5) -> np.ndarray:
    """+1 on Keltner upper-band breakout AND volume expansion. axis=volatility_band."""
    k = ic.keltner(bars, ema_period=ema_p, atr_period=atr_p, mult=mult)
    c, v = bars["close"], bars["volume"]
    avg_v = ic._sma(v, ema_p)
    vol_ok = v >= (vol_mult * avg_v)
    out = np.zeros(len(c), dtype=np.int8)
    out[(c > k["upper"]) & vol_ok] = 1
    out[(c < k["lower"]) & vol_ok] = -1
    return out


def parabolic_sar_flip(bars: ArrayDict, step: float = 0.02, max_step: float = 0.20) -> np.ndarray:
    """+1 on PSAR flip-to-bull; -1 on flip-to-bear. axis=trend."""
    p = ic.parabolic_sar(bars, step=step, max_step=max_step)
    trend = p.get("trend")
    if trend is None:
        # Derive from sar vs close
        sar = p.get("sar")
        c = bars["close"]
        trend = np.where(c > sar, 1, -1).astype(np.int8)
    trend = np.asarray(trend, dtype=np.int8)
    prev = np.concatenate([[trend[0]], trend[:-1]])
    out = np.zeros(len(trend), dtype=np.int8)
    out[(trend == 1) & (prev != 1)] = 1
    out[(trend == -1) & (prev != -1)] = -1
    return out


def cci_extreme_reversal(bars: ArrayDict, period: int = 20,
                          lo: float = -100.0, hi: float = 100.0) -> np.ndarray:
    """+1 when CCI re-enters from below lo (oversold reversal); -1 above hi."""
    c = ic.cci(bars, period=period)
    prev = np.concatenate([[c[0]], c[:-1]])
    out = np.zeros(len(c), dtype=np.int8)
    out[(prev <= lo) & (c > lo)] = 1
    out[(prev >= hi) & (c < hi)] = -1
    return out


def volume_expansion_with_close_above_vwap(bars: ArrayDict,
                                              vol_period: int = 20,
                                              vol_mult: float = 1.5) -> np.ndarray:
    """+1 when volume > vol_mult*SMA AND close > VWAP. axis=volume_conviction."""
    v = bars["volume"]
    c = bars["close"]
    av = ic._sma(v, vol_period)
    vp = ic.vwap(bars)
    out = np.zeros(len(c), dtype=np.int8)
    out[(v >= vol_mult * av) & (c > vp)] = 1
    out[(v >= vol_mult * av) & (c < vp)] = -1
    return out


def obv_trend_break(bars: ArrayDict, ema_span: int = 20) -> np.ndarray:
    """+1 when OBV crosses above its EMA(span); -1 inverse. axis=volume_conviction."""
    o = ic.obv(bars)
    e = ic._ema(o, ema_span)
    cu = _cross_up(o, e)
    cd = _cross_dn(o, e)
    out = np.zeros(len(o), dtype=np.int8)
    out[cu] = 1
    out[cd] = -1
    return out


def stochastic_oversold_reversal(bars: ArrayDict, k: int = 14, d: int = 3,
                                  sm: int = 3, lo: float = 20.0, hi: float = 80.0) -> np.ndarray:
    """+1 when Stoch %K crosses above lo from below; -1 cross below hi from above."""
    s = ic.stochastic(bars, k=k, d=d, smooth_k=sm)
    kv = s["k"]
    prev = np.concatenate([[kv[0]], kv[:-1]])
    out = np.zeros(len(kv), dtype=np.int8)
    out[(prev <= lo) & (kv > lo)] = 1
    out[(prev >= hi) & (kv < hi)] = -1
    return out


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


RuleFn = Callable[..., np.ndarray]


RULE_REGISTRY: dict[str, dict[str, Any]] = {
    "vwap_touch_reject": {
        "fn": vwap_touch_reject, "axis": "volume_conviction", "role": "entry trigger"},
    "ema_stack_bull": {
        "fn": ema_stack_bull, "axis": "trend", "role": "bias filter"},
    "donchian_breakout_with_volume": {
        "fn": donchian_breakout_with_volume, "axis": "structure_geometry", "role": "entry trigger"},
    "rsi_oversold_in_uptrend": {
        "fn": rsi_oversold_in_uptrend, "axis": "mean_reversion", "role": "timing trigger"},
    "macd_zero_cross_with_adx_gate": {
        "fn": macd_zero_cross_with_adx_gate, "axis": "trend", "role": "entry trigger"},
    "bb_squeeze_release_bull": {
        "fn": bb_squeeze_release_bull, "axis": "volatility_band", "role": "entry trigger"},
    "opening_range_breakout": {
        "fn": opening_range_breakout, "axis": "structure_geometry", "role": "entry trigger"},
    "supertrend_flip": {
        "fn": supertrend_flip, "axis": "trend", "role": "entry trigger"},
    "keltner_breakout_with_volume": {
        "fn": keltner_breakout_with_volume, "axis": "volatility_band", "role": "entry trigger"},
    "parabolic_sar_flip": {
        "fn": parabolic_sar_flip, "axis": "trend", "role": "entry trigger"},
    "cci_extreme_reversal": {
        "fn": cci_extreme_reversal, "axis": "momentum_oscillator", "role": "entry trigger"},
    "volume_expansion_with_close_above_vwap": {
        "fn": volume_expansion_with_close_above_vwap,
        "axis": "volume_conviction", "role": "confirmation"},
    "obv_trend_break": {
        "fn": obv_trend_break, "axis": "volume_conviction", "role": "confirmation"},
    "stochastic_oversold_reversal": {
        "fn": stochastic_oversold_reversal, "axis": "momentum_oscillator", "role": "entry trigger"},
}


SIGNAL_RULE_AXIS: dict[str, str] = {k: v["axis"] for k, v in RULE_REGISTRY.items()}


def rule(name: str) -> RuleFn:
    """Look up a signal-rule callable by name."""
    return RULE_REGISTRY[name]["fn"]


def all_rules() -> list[str]:
    return sorted(RULE_REGISTRY)


def axis_for(name: str) -> str:
    return SIGNAL_RULE_AXIS.get(name, "unknown")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    bars = ic._smoke_synthetic(5000)
    n = len(bars["close"])
    print(f"# signal_rules smoke — n={n} bars (synthetic)")
    print(f"# total rules: {len(RULE_REGISTRY)}")
    print()
    print(f"{'STATUS':<8}{'AXIS':<22}{'RULE':<42}{'+1 / -1 / 0'}")
    ok = fail = 0
    for name, cfg in RULE_REGISTRY.items():
        try:
            sig = cfg["fn"](bars)
            sig = np.asarray(sig)
            if len(sig) != n:
                print(f"{'FAIL':<8}{cfg['axis']:<22}{name:<42}shape {len(sig)} != {n}")
                fail += 1
                continue
            n_long = int((sig == 1).sum())
            n_short = int((sig == -1).sum())
            n_flat = int((sig == 0).sum())
            print(f"{'OK':<8}{cfg['axis']:<22}{name:<42}{n_long} / {n_short} / {n_flat}")
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"{'FAIL':<8}{cfg['axis']:<22}{name:<42}EXC {type(e).__name__}: {e}")
            fail += 1
    print()
    print(f"# rules smoke: {ok} ok / {fail} fail / {len(RULE_REGISTRY)} total")
