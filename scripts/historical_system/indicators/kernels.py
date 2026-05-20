"""Shared math kernels used by indicator definitions.

Everything here is vectorized (numpy / pandas) and has no knowledge of
indicator names — it's just math. Indicator files import from here rather
than each re-implementing the wheel.

Conventions
-----------
* Inputs are either numpy arrays or pandas Series. Functions that accept
  either convert internally and return numpy arrays.
* All rolling/ewm windows use ``min_periods=period`` unless the indicator
  needs seed values (e.g., Wilder smoothing, which uses ``min_periods=0``
  to match streaming semantics).
* Population std (ddof=0) is the default because TradingView uses ddof=0.
  Overrides are available where needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------
def sma(x: np.ndarray | pd.Series, n: int) -> np.ndarray:
    """Simple moving average."""
    return pd.Series(x).rolling(n, min_periods=n).mean().to_numpy()


def ema(x: np.ndarray | pd.Series, n: int) -> np.ndarray:
    """Exponential moving average, seeded at the first bar (TradingView style).

    y[t] = a*x[t] + (1-a)*y[t-1], a = 2/(n+1), y[0] = x[0].
    ``pandas.Series.ewm(span=n, adjust=False, min_periods=0)`` matches this.
    """
    s = pd.Series(x)
    return s.ewm(span=n, adjust=False, min_periods=0).mean().to_numpy()


def wma(x: np.ndarray | pd.Series, n: int) -> np.ndarray:
    """Linearly-weighted moving average. Most recent bar weighted highest."""
    x = np.asarray(x, dtype=np.float64)
    weights = np.arange(1, n + 1, dtype=np.float64)
    denom = weights.sum()

    def _w(window):
        return np.dot(window, weights) / denom

    return pd.Series(x).rolling(n, min_periods=n).apply(_w, raw=True).to_numpy()


def hma(x: np.ndarray | pd.Series, n: int) -> np.ndarray:
    """Hull moving average: WMA( 2*WMA(n/2) - WMA(n), sqrt(n) )."""
    half = max(int(n / 2), 1)
    sqrt_n = max(int(np.sqrt(n)), 1)
    raw = 2.0 * wma(x, half) - wma(x, n)
    return wma(raw, sqrt_n)


def dema(x: np.ndarray | pd.Series, n: int) -> np.ndarray:
    """Double EMA: 2*EMA(n) - EMA(EMA(n))."""
    e1 = ema(x, n)
    e2 = ema(e1, n)
    return 2.0 * e1 - e2


def tema(x: np.ndarray | pd.Series, n: int) -> np.ndarray:
    """Triple EMA: 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))."""
    e1 = ema(x, n)
    e2 = ema(e1, n)
    e3 = ema(e2, n)
    return 3.0 * e1 - 3.0 * e2 + e3


def smma(x: np.ndarray | pd.Series, n: int) -> np.ndarray:
    """Smoothed MA / Wilder's smoothing (alpha = 1/n)."""
    return pd.Series(x).ewm(alpha=1.0 / n, adjust=False, min_periods=0).mean().to_numpy()


def rma(x: np.ndarray | pd.Series, n: int) -> np.ndarray:
    """Alias for SMMA — used by RSI, ATR, ADX. Same as pine's ``ta.rma``."""
    return smma(x, n)


def vwma(price: np.ndarray | pd.Series, volume: np.ndarray | pd.Series, n: int) -> np.ndarray:
    """Volume-weighted MA."""
    price = pd.Series(price)
    volume = pd.Series(volume)
    num = (price * volume).rolling(n, min_periods=n).sum()
    den = volume.rolling(n, min_periods=n).sum().replace(0, np.nan)
    return (num / den).to_numpy()


def alma(x: np.ndarray | pd.Series, n: int = 9, offset: float = 0.85, sigma: float = 6.0) -> np.ndarray:
    """Arnaud Legoux MA — Gaussian-weighted window with offset ``m`` and ``s = n/sigma``."""
    x = np.asarray(x, dtype=np.float64)
    m = offset * (n - 1)
    s = n / sigma
    i = np.arange(n, dtype=np.float64)
    w = np.exp(-((i - m) ** 2) / (2.0 * s * s))
    denom = w.sum()

    def _f(window):
        return np.dot(window, w) / denom

    return pd.Series(x).rolling(n, min_periods=n).apply(_f, raw=True).to_numpy()


def lsma(x: np.ndarray | pd.Series, n: int) -> np.ndarray:
    """Least-squares / linear-regression MA — last fitted value of a rolling OLS."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < n:
        return np.full_like(x, np.nan, dtype=np.float64)
    t = np.arange(n, dtype=np.float64)
    t_mean = t.mean()
    t_var = ((t - t_mean) ** 2).sum()

    def _f(window):
        y_mean = window.mean()
        slope = ((t - t_mean) * (window - y_mean)).sum() / t_var
        intercept = y_mean - slope * t_mean
        return intercept + slope * (n - 1)

    return pd.Series(x).rolling(n, min_periods=n).apply(_f, raw=True).to_numpy()


def linreg_slope(x: np.ndarray | pd.Series, n: int) -> np.ndarray:
    """Slope of a rolling OLS on ``x``."""
    x = np.asarray(x, dtype=np.float64)
    t = np.arange(n, dtype=np.float64)
    t_mean = t.mean()
    t_var = ((t - t_mean) ** 2).sum()

    def _f(window):
        y_mean = window.mean()
        return ((t - t_mean) * (window - y_mean)).sum() / t_var

    return pd.Series(x).rolling(n, min_periods=n).apply(_f, raw=True).to_numpy()


def hamming_ma(x: np.ndarray | pd.Series, n: int) -> np.ndarray:
    """Hamming-windowed MA: weights = 0.54 - 0.46*cos(2*pi*i/(n-1))."""
    x = np.asarray(x, dtype=np.float64)
    i = np.arange(n, dtype=np.float64)
    w = 0.54 - 0.46 * np.cos(2.0 * np.pi * i / max(n - 1, 1))
    denom = w.sum()

    def _f(window):
        return np.dot(window, w) / denom

    return pd.Series(x).rolling(n, min_periods=n).apply(_f, raw=True).to_numpy()


# ---------------------------------------------------------------------------
# Dispersion / volatility
# ---------------------------------------------------------------------------
def rolling_std(x: np.ndarray | pd.Series, n: int, ddof: int = 0) -> np.ndarray:
    """Rolling standard deviation. ``ddof=0`` matches TradingView."""
    return pd.Series(x).rolling(n, min_periods=n).std(ddof=ddof).to_numpy()


def rolling_var(x: np.ndarray | pd.Series, n: int, ddof: int = 0) -> np.ndarray:
    return pd.Series(x).rolling(n, min_periods=n).var(ddof=ddof).to_numpy()


def true_range(high: np.ndarray | pd.Series,
               low: np.ndarray | pd.Series,
               close: np.ndarray | pd.Series) -> np.ndarray:
    """Wilder's true range. ``tr[0] = high[0] - low[0]``."""
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)
    n = len(c)
    tr = np.empty(n, dtype=np.float64)
    if n == 0:
        return tr
    tr[0] = h[0] - l[0]
    if n > 1:
        hl = h[1:] - l[1:]
        hc = np.abs(h[1:] - c[:-1])
        lc = np.abs(l[1:] - c[:-1])
        tr[1:] = np.maximum(np.maximum(hl, hc), lc)
    return tr


def atr(high, low, close, n: int = 14, method: str = "rma") -> np.ndarray:
    """Average True Range. ``method='rma'`` (Wilder) matches TradingView."""
    tr = true_range(high, low, close)
    if method == "rma":
        return rma(tr, n)
    if method == "sma":
        return sma(tr, n)
    if method == "ema":
        return ema(tr, n)
    raise ValueError(f"unknown atr method: {method}")


# ---------------------------------------------------------------------------
# Rolling summaries
# ---------------------------------------------------------------------------
def rolling_max(x: np.ndarray | pd.Series, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n, min_periods=n).max().to_numpy()


def rolling_min(x: np.ndarray | pd.Series, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n, min_periods=n).min().to_numpy()


def rolling_sum(x: np.ndarray | pd.Series, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n, min_periods=n).sum().to_numpy()


def rolling_rank(x: np.ndarray | pd.Series, n: int) -> np.ndarray:
    """Percent-rank of last value within the last ``n`` observations (0..100)."""
    s = pd.Series(x)
    return s.rolling(n, min_periods=n).apply(
        lambda w: (w.rank().iloc[-1] - 1) / (len(w) - 1) * 100.0 if len(w) > 1 else 0.0,
        raw=False,
    ).to_numpy()


def pct_change(x: np.ndarray | pd.Series, n: int = 1) -> np.ndarray:
    return pd.Series(x).pct_change(n).to_numpy()


def diff(x: np.ndarray | pd.Series, n: int = 1) -> np.ndarray:
    return pd.Series(x).diff(n).to_numpy()


def crossover(a, b) -> np.ndarray:
    """Boolean array — True where a crosses above b."""
    a = pd.Series(a)
    b = pd.Series(b) if not np.isscalar(b) else pd.Series(np.full(len(a), b))
    prev = (a.shift(1) <= b.shift(1))
    cur = (a > b)
    return (prev & cur).to_numpy()


def crossunder(a, b) -> np.ndarray:
    a = pd.Series(a)
    b = pd.Series(b) if not np.isscalar(b) else pd.Series(np.full(len(a), b))
    prev = (a.shift(1) >= b.shift(1))
    cur = (a < b)
    return (prev & cur).to_numpy()


# ---------------------------------------------------------------------------
# RSI family
# ---------------------------------------------------------------------------
def rsi(close, n: int = 14) -> np.ndarray:
    """Wilder RSI. Matches ``pine ta.rsi(close, n)``."""
    c = np.asarray(close, dtype=np.float64)
    nn = len(c)
    out = np.full(nn, np.nan, dtype=np.float64)
    if nn < 2:
        return out
    deltas = np.diff(c, prepend=c[0])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_g = rma(gains, n)
    avg_l = rma(losses, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(avg_l > 0, avg_g / avg_l, np.inf)
        out = 100.0 - (100.0 / (1.0 + rs))
    out[0] = np.nan
    return out


# ---------------------------------------------------------------------------
# Directional movement (ADX family)
# ---------------------------------------------------------------------------
def directional_movement(high, low) -> tuple[np.ndarray, np.ndarray]:
    """Returns (+DM, -DM) arrays per Wilder."""
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)
    n = len(h)
    plus = np.zeros(n, dtype=np.float64)
    minus = np.zeros(n, dtype=np.float64)
    if n < 2:
        return plus, minus
    up = h[1:] - h[:-1]
    down = l[:-1] - l[1:]
    p = np.where((up > down) & (up > 0), up, 0.0)
    m = np.where((down > up) & (down > 0), down, 0.0)
    plus[1:] = p
    minus[1:] = m
    return plus, minus


# ---------------------------------------------------------------------------
# Price series helpers
# ---------------------------------------------------------------------------
def typical_price(df) -> np.ndarray:
    return ((df["high"] + df["low"] + df["close"]) / 3.0).to_numpy()


def median_price(df) -> np.ndarray:
    return ((df["high"] + df["low"]) / 2.0).to_numpy()


def average_price(df) -> np.ndarray:
    return ((df["open"] + df["high"] + df["low"] + df["close"]) / 4.0).to_numpy()


def hlc3(df) -> np.ndarray:
    return typical_price(df)


def ohlc4(df) -> np.ndarray:
    return average_price(df)


# ---------------------------------------------------------------------------
# Session utilities
# ---------------------------------------------------------------------------
def session_day(index: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray(pd.to_datetime(index).date)


__all__ = [
    "sma", "ema", "wma", "hma", "dema", "tema", "smma", "rma", "vwma",
    "alma", "lsma", "linreg_slope", "hamming_ma",
    "rolling_std", "rolling_var", "rolling_max", "rolling_min", "rolling_sum",
    "rolling_rank", "pct_change", "diff",
    "true_range", "atr",
    "rsi", "directional_movement",
    "crossover", "crossunder",
    "typical_price", "median_price", "average_price", "hlc3", "ohlc4",
    "session_day",
]
