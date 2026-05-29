"""indicator_compute_xsym.py — Cross-symbol indicators (PATH B, 2026-05-29).

Multi-bars-dict interface. Input is `bars_by_symbol: dict[str, pd.DataFrame]` keyed
by symbol, each DataFrame with columns [open, high, low, close, volume] (DatetimeIndex
preferred but not required — we align by positional index against the FIRST symbol's bars).

Every function returns a `pd.Series` (or `np.ndarray` for dict-keyed convenience)
indexed/aligned to the symbol-of-interest's bars.

Coverage symbols expected from Gabriel store
(`version_3 - Gabriel/Gabriel_Alpaca TimeFrames/Day TimeFrames/1Day/<SYMBOL>/`):
  SPY, QQQ, VIX, VXX, XLK, XLF, XLE, XLV, XLU, XLY, XLP, XLI, XLB, XLRE, XLC,
  UUP, HYG, LQD, TLT, DXY

If the underlying parquet store is missing a needed symbol, the indicator raises
an `ImportError` with the missing-symbol name. Callers should catch and skip.

Functions
---------
- vix_multiplied_atr        — ATR scaled by VIX regime
- spy_beta                  — rolling beta vs SPY
- sector_relative_strength  — symbol return minus sector return
- spy_correlation           — rolling correlation vs SPY
- sector_rotation_rank      — sector's rank among sector ETFs
- vix_term_structure        — VIX-front / VIX-back ratio
- hyg_lqd_ratio             — credit-spread proxy
- dxy_delta                 — dollar-strength delta (UUP proxy)

axis-tags: most are 'trend' (regime / context) or 'volume_conviction' (confirmation).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DRIVE_BASE = "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive"

# Per ohlcv_loader: Cache B GOLD primary, yfinance_5yr fallback.
_GABRIEL_1D_ROOT = (
    f"{DRIVE_BASE}/version_3 - Gabriel/Gabriel_Alpaca TimeFrames/Day TimeFrames/1Day"
)
_YF5Y_ROOT = (
    f"{DRIVE_BASE}/AI-Tools/s&p500-ticker-mastery/cache/yfinance_5yr"
)


# ---------------------------------------------------------------------------
# Internal symbol loader (lightweight, no pyarrow dependency)
# ---------------------------------------------------------------------------


def _load_reference_symbol(symbol: str) -> pd.DataFrame:
    """Load a reference symbol's daily OHLCV. Returns DataFrame with [open, high, low, close, volume]
    and a DatetimeIndex named 'timestamp'. Raises ImportError if the symbol can't be found in any
    known store.
    """
    # 1) Try Gabriel partitioned 1Day store
    gd = Path(_GABRIEL_1D_ROOT) / symbol
    if gd.exists():
        files = sorted(gd.glob("*.parquet"))
        if files:
            frames = []
            for f in files:
                try:
                    frames.append(pd.read_parquet(f))
                except (ImportError, OSError) as e:
                    logger.debug("skipping %s: %s", f, e)
            if frames:
                df = pd.concat(frames, ignore_index=True)
                return _normalize(df)

    # 2) Try yfinance_5yr fallback
    yf = Path(_YF5Y_ROOT) / f"{symbol}.parquet"
    if yf.exists():
        try:
            df = pd.read_parquet(yf)
            return _normalize(df)
        except (ImportError, OSError) as e:
            logger.debug("yf5y load failed for %s: %s", symbol, e)

    raise ImportError(
        f"Reference symbol '{symbol}' not found in any store. "
        f"Tried: {gd}, {yf}. Skip the xsym indicator or provide bars_by_symbol explicitly."
    )


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize OHLCV DataFrame to lowercase columns + DatetimeIndex named 'timestamp'."""
    out = df.copy()
    out.columns = [c.lower() for c in out.columns]
    # Determine timestamp column
    ts_col = None
    for cand in ("timestamp", "date", "datetime"):
        if cand in out.columns:
            ts_col = cand
            break
    if ts_col:
        out["timestamp"] = pd.to_datetime(out[ts_col], errors="coerce", utc=True).dt.tz_localize(None)
        out = out.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    else:
        # Already indexed
        out.index = pd.to_datetime(out.index, errors="coerce")
        out = out.sort_index()
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in out.columns]
    return out[keep]


def _ensure_loaded(bars_by_symbol: Optional[dict], symbol: str) -> dict[str, pd.DataFrame]:
    """If `bars_by_symbol` is None or missing `symbol`, load it. Returns the (possibly augmented) dict."""
    if bars_by_symbol is None:
        bars_by_symbol = {}
    if symbol not in bars_by_symbol:
        bars_by_symbol[symbol] = _load_reference_symbol(symbol)
    return bars_by_symbol


def _align(target: pd.DataFrame, ref: pd.DataFrame) -> pd.Series:
    """Reindex ref close series onto target's index using nearest backward (forward-filled)."""
    ref_close = ref["close"]
    return ref_close.reindex(target.index, method="ffill")


def _safe_ret(close: pd.Series) -> pd.Series:
    return close.pct_change()


# ---------------------------------------------------------------------------
# 1. VIX-multiplied ATR
# ---------------------------------------------------------------------------


def vix_multiplied_atr(bars_by_symbol: Optional[dict], symbol: str,
                        vix_symbol: str = "VIX", n: int = 14,
                        baseline_vix: float = 20.0) -> pd.Series:
    """ATR(symbol, n) * VIX_close / baseline_vix.

    Normalizes ATR by current vol regime. VIX of `baseline_vix` (20) leaves ATR unchanged.
    Returns Series aligned to symbol's bars; NaN until ATR warm-up satisfied.
    axis=volatility_band (regime).
    """
    bars_by_symbol = _ensure_loaded(bars_by_symbol, symbol)
    bars_by_symbol = _ensure_loaded(bars_by_symbol, vix_symbol)
    tgt = bars_by_symbol[symbol]
    vix = _align(tgt, bars_by_symbol[vix_symbol])
    h, l, c = tgt["high"], tgt["low"], tgt["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    # Wilder smoothing via ewm with alpha=1/n
    atr_s = tr.ewm(alpha=1.0 / n, adjust=False).mean()
    atr_s.iloc[:n - 1] = np.nan
    return atr_s * vix / baseline_vix


# ---------------------------------------------------------------------------
# 2. Rolling SPY beta
# ---------------------------------------------------------------------------


def spy_beta(bars_by_symbol: Optional[dict], symbol: str,
              spy_symbol: str = "SPY", n: int = 60) -> pd.Series:
    """Rolling `n`-day beta: Cov(ret_sym, ret_spy) / Var(ret_spy). axis=trend (regime context)."""
    bars_by_symbol = _ensure_loaded(bars_by_symbol, symbol)
    bars_by_symbol = _ensure_loaded(bars_by_symbol, spy_symbol)
    tgt = bars_by_symbol[symbol]
    spy = _align(tgt, bars_by_symbol[spy_symbol])
    r_sym = _safe_ret(tgt["close"])
    r_spy = _safe_ret(spy)
    cov = r_sym.rolling(n).cov(r_spy)
    var = r_spy.rolling(n).var()
    return cov / var.replace(0.0, np.nan)


# ---------------------------------------------------------------------------
# 3. Sector relative strength
# ---------------------------------------------------------------------------


def sector_relative_strength(bars_by_symbol: Optional[dict], symbol: str,
                              sector_etf: str, n: int = 20) -> pd.Series:
    """(symbol return over n bars) - (sector return over n bars). Positive = outperforming.

    axis=trend (regime / bias context).
    """
    bars_by_symbol = _ensure_loaded(bars_by_symbol, symbol)
    bars_by_symbol = _ensure_loaded(bars_by_symbol, sector_etf)
    tgt = bars_by_symbol[symbol]
    sec = _align(tgt, bars_by_symbol[sector_etf])
    r_sym = tgt["close"].pct_change(n)
    r_sec = sec.pct_change(n)
    return r_sym - r_sec


# ---------------------------------------------------------------------------
# 4. Rolling SPY correlation
# ---------------------------------------------------------------------------


def spy_correlation(bars_by_symbol: Optional[dict], symbol: str,
                     spy_symbol: str = "SPY", n: int = 20) -> pd.Series:
    """Rolling `n`-day Pearson correlation between symbol returns and SPY returns.
    axis=trend (regime).
    """
    bars_by_symbol = _ensure_loaded(bars_by_symbol, symbol)
    bars_by_symbol = _ensure_loaded(bars_by_symbol, spy_symbol)
    tgt = bars_by_symbol[symbol]
    spy = _align(tgt, bars_by_symbol[spy_symbol])
    r_sym = _safe_ret(tgt["close"])
    r_spy = _safe_ret(spy)
    return r_sym.rolling(n).corr(r_spy)


# ---------------------------------------------------------------------------
# 5. Sector rotation rank
# ---------------------------------------------------------------------------


DEFAULT_SECTOR_ETFS = (
    "XLK", "XLF", "XLE", "XLV", "XLU", "XLY", "XLP", "XLI", "XLB", "XLRE", "XLC",
)


def sector_rotation_rank(bars_by_symbol: Optional[dict],
                          sector_etfs: Iterable[str] = DEFAULT_SECTOR_ETFS,
                          symbol_sector: Optional[str] = None,
                          n: int = 20,
                          target_symbol: Optional[str] = None) -> pd.Series:
    """Per-bar rank of `symbol_sector` among `sector_etfs` by trailing `n`-bar return.

    If `target_symbol` is provided, the result is aligned to that symbol's index;
    otherwise aligned to the FIRST sector ETF's index. Rank 1 = best.

    Returns int-valued Series (1..len(sector_etfs)) of the sector's rank. axis=trend.
    """
    if symbol_sector is None:
        raise ValueError("Must specify symbol_sector (which sector to rank).")
    bars_by_symbol = bars_by_symbol or {}
    sectors = list(sector_etfs)
    if symbol_sector not in sectors:
        sectors = [symbol_sector] + sectors
    for s in sectors:
        bars_by_symbol = _ensure_loaded(bars_by_symbol, s)

    # Build returns DataFrame on common index
    closes = pd.DataFrame({s: bars_by_symbol[s]["close"] for s in sectors})
    closes = closes.sort_index().dropna(how="all")
    rets = closes.pct_change(n)
    # Per-row rank descending (1 = best)
    ranks = rets.rank(axis=1, ascending=False, method="min")
    ser = ranks[symbol_sector]
    if target_symbol is not None:
        if target_symbol not in bars_by_symbol:
            bars_by_symbol = _ensure_loaded(bars_by_symbol, target_symbol)
        ser = ser.reindex(bars_by_symbol[target_symbol].index, method="ffill")
    return ser


# ---------------------------------------------------------------------------
# 6. VIX term structure
# ---------------------------------------------------------------------------


def vix_term_structure(bars_by_symbol: Optional[dict], vix_symbol: str = "VIX",
                        vx_front: str = "VXX",
                        target_symbol: Optional[str] = None) -> pd.Series:
    """Front/back VIX-vol ratio. Common interpretation: ratio >1 = backwardation (stress).

    Implemented as VIX / VXX since true VX1/VX2 futures aren't typically in store.
    axis=volatility_band (regime).
    """
    bars_by_symbol = bars_by_symbol or {}
    bars_by_symbol = _ensure_loaded(bars_by_symbol, vix_symbol)
    bars_by_symbol = _ensure_loaded(bars_by_symbol, vx_front)
    vix = bars_by_symbol[vix_symbol]["close"]
    vxx = bars_by_symbol[vx_front]["close"]
    # Align onto VIX index then divide
    vxx_aligned = vxx.reindex(vix.index, method="ffill")
    ratio = vix / vxx_aligned.replace(0.0, np.nan)
    if target_symbol is not None:
        if target_symbol not in bars_by_symbol:
            bars_by_symbol = _ensure_loaded(bars_by_symbol, target_symbol)
        ratio = ratio.reindex(bars_by_symbol[target_symbol].index, method="ffill")
    return ratio


# ---------------------------------------------------------------------------
# 7. HYG/LQD ratio
# ---------------------------------------------------------------------------


def hyg_lqd_ratio(bars_by_symbol: Optional[dict], hyg_symbol: str = "HYG",
                   lqd_symbol: str = "LQD", n: int = 5,
                   target_symbol: Optional[str] = None) -> pd.Series:
    """HYG/LQD ratio (smoothed via SMA-`n`). Credit-spread proxy: rising = risk-on.

    axis=volume_conviction (risk-on/off confirmation).
    """
    bars_by_symbol = bars_by_symbol or {}
    bars_by_symbol = _ensure_loaded(bars_by_symbol, hyg_symbol)
    bars_by_symbol = _ensure_loaded(bars_by_symbol, lqd_symbol)
    hyg = bars_by_symbol[hyg_symbol]["close"]
    lqd = bars_by_symbol[lqd_symbol]["close"]
    lqd_aligned = lqd.reindex(hyg.index, method="ffill")
    raw = hyg / lqd_aligned.replace(0.0, np.nan)
    smoothed = raw.rolling(n).mean()
    if target_symbol is not None:
        if target_symbol not in bars_by_symbol:
            bars_by_symbol = _ensure_loaded(bars_by_symbol, target_symbol)
        smoothed = smoothed.reindex(bars_by_symbol[target_symbol].index, method="ffill")
    return smoothed


# ---------------------------------------------------------------------------
# 8. DXY delta (dollar strength)
# ---------------------------------------------------------------------------


def dxy_delta(bars_by_symbol: Optional[dict], dxy_symbol: str = "UUP", n: int = 5,
               target_symbol: Optional[str] = None) -> pd.Series:
    """Trailing `n`-bar percent change of DXY (UUP as proxy). axis=trend (regime).

    Positive = USD strengthening (often bearish for risk-on US equities).
    """
    bars_by_symbol = bars_by_symbol or {}
    bars_by_symbol = _ensure_loaded(bars_by_symbol, dxy_symbol)
    dxy = bars_by_symbol[dxy_symbol]["close"]
    delta = dxy.pct_change(n)
    if target_symbol is not None:
        if target_symbol not in bars_by_symbol:
            bars_by_symbol = _ensure_loaded(bars_by_symbol, target_symbol)
        delta = delta.reindex(bars_by_symbol[target_symbol].index, method="ffill")
    return delta


# ---------------------------------------------------------------------------
# Registry (mirrors INDICATOR_AXIS pattern from indicator_compute)
# ---------------------------------------------------------------------------


XSYM_AXIS: dict[str, str] = {
    "vix_multiplied_atr": "volatility_band",
    "spy_beta": "trend",
    "sector_relative_strength": "trend",
    "spy_correlation": "trend",
    "sector_rotation_rank": "trend",
    "vix_term_structure": "volatility_band",
    "hyg_lqd_ratio": "volume_conviction",
    "dxy_delta": "trend",
}


XSYM_REGISTRY: dict[str, dict] = {
    "vix_multiplied_atr": {"fn": vix_multiplied_atr, "needs": ("VIX",)},
    "spy_beta": {"fn": spy_beta, "needs": ("SPY",)},
    "sector_relative_strength": {"fn": sector_relative_strength, "needs": ("XLK",)},  # any sector
    "spy_correlation": {"fn": spy_correlation, "needs": ("SPY",)},
    "sector_rotation_rank": {"fn": sector_rotation_rank, "needs": DEFAULT_SECTOR_ETFS},
    "vix_term_structure": {"fn": vix_term_structure, "needs": ("VIX", "VXX")},
    "hyg_lqd_ratio": {"fn": hyg_lqd_ratio, "needs": ("HYG", "LQD")},
    "dxy_delta": {"fn": dxy_delta, "needs": ("UUP",)},
}


def xsym_axis_for(name: str) -> str:
    return XSYM_AXIS.get(name, "unknown")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def _smoke_synthetic_xsym(n: int = 800) -> dict[str, pd.DataFrame]:
    """Build synthetic AAPL + reference symbols, indexed by a 800-day DatetimeIndex."""
    rng = np.random.default_rng(0)
    idx = pd.date_range(end="2026-04-30", periods=n, freq="B")  # business days

    def _frame(seed: int, start: float = 100.0, vol: float = 1.0) -> pd.DataFrame:
        r = np.random.default_rng(seed)
        close = start + np.cumsum(r.normal(0, 0.1 * vol, n))
        high = close + np.abs(r.normal(0, 0.05 * vol, n))
        low = close - np.abs(r.normal(0, 0.05 * vol, n))
        op = close + r.normal(0, 0.05 * vol, n)
        v = r.lognormal(10, 0.3, n)
        return pd.DataFrame({"open": op, "high": high, "low": low, "close": close, "volume": v}, index=idx)

    syms = ["AAPL", "SPY", "VIX", "VXX", "XLK", "XLF", "XLE", "XLV", "XLU", "XLY",
            "XLP", "XLI", "XLB", "XLRE", "XLC", "UUP", "HYG", "LQD"]
    bars = {}
    for i, s in enumerate(syms):
        # VIX has different scale
        if s == "VIX":
            bars[s] = _frame(i, start=20.0, vol=0.5)
        elif s == "VXX":
            bars[s] = _frame(i, start=25.0, vol=0.7)
        else:
            bars[s] = _frame(i)
    return bars


def _try_load_real_xsym() -> Optional[dict[str, pd.DataFrame]]:
    """Try to load real reference symbols. Returns None on failure."""
    bars = {}
    needed = ["AAPL", "SPY"]
    try:
        for s in needed:
            bars[s] = _load_reference_symbol(s)
        return bars
    except ImportError:
        return None


if __name__ == "__main__":
    import os
    real = None if os.environ.get("XSYM_SYNTHETIC_ONLY") == "1" else _try_load_real_xsym()
    if real:
        bars_by_symbol = real
        src = "real"
        target = "AAPL"
    else:
        bars_by_symbol = _smoke_synthetic_xsym(800)
        src = "synthetic"
        target = "AAPL"
    print(f"# xsym smoke — bars source: {src}, target={target}")
    print(f"# total xsym indicators: {len(XSYM_REGISTRY)}")
    print()
    n_target = len(bars_by_symbol[target])
    print(f"{'STATUS':<8}{'AXIS':<22}{'NAME':<32}{'NOTES'}")

    ok = fail = 0

    def _check(name: str, ser, expected_n: int) -> tuple[bool, str]:
        try:
            arr = np.asarray(ser)
            if len(arr) != expected_n:
                return False, f"shape mismatch: {len(arr)} vs {expected_n}"
            finite = arr[np.isfinite(arr.astype(float, copy=False))]
            if len(finite) == 0:
                return False, "all-NaN result"
            return True, f"finite={len(finite)}/{expected_n} range=[{finite.min():.3g}, {finite.max():.3g}]"
        except Exception as e:  # noqa: BLE001
            return False, f"check exc: {e}"

    invocations = [
        ("vix_multiplied_atr", lambda: vix_multiplied_atr(bars_by_symbol, target)),
        ("spy_beta", lambda: spy_beta(bars_by_symbol, target)),
        ("sector_relative_strength",
            lambda: sector_relative_strength(bars_by_symbol, target, "XLK")),
        ("spy_correlation", lambda: spy_correlation(bars_by_symbol, target)),
        ("sector_rotation_rank",
            lambda: sector_rotation_rank(bars_by_symbol, symbol_sector="XLK",
                                         target_symbol=target)),
        ("vix_term_structure",
            lambda: vix_term_structure(bars_by_symbol, target_symbol=target)),
        ("hyg_lqd_ratio",
            lambda: hyg_lqd_ratio(bars_by_symbol, target_symbol=target)),
        ("dxy_delta",
            lambda: dxy_delta(bars_by_symbol, target_symbol=target)),
    ]
    for name, runner in invocations:
        try:
            res = runner()
            passed, note = _check(name, res, n_target)
            ax = XSYM_AXIS.get(name, "?")
            status = "OK" if passed else "FAIL"
            if passed:
                ok += 1
            else:
                fail += 1
            print(f"{status:<8}{ax:<22}{name:<32}{note}")
        except ImportError as e:
            print(f"{'SKIP':<8}{XSYM_AXIS.get(name,'?'):<22}{name:<32}missing-symbol: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"{'FAIL':<8}{XSYM_AXIS.get(name,'?'):<22}{name:<32}EXC {type(e).__name__}: {e}")
            fail += 1
    print()
    print(f"# xsym smoke: {ok} ok / {fail} fail / {len(invocations)} total")
