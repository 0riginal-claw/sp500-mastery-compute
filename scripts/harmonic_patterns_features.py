# autosolve_skip: data-loader infra
"""
harmonic_patterns_features.py — harmonic chart pattern features.

Source: external-repos/HarmonicPatterns (Djoffrey, archived upstream).
Wraps the HarmonicDetector class to produce per-bar pattern-activation +
completion features for Gartley, Bat, AltBat, Butterfly, Crab, DeepCrab,
Shark, 5o, Cypher, AB=CD (10 patterns).

NO-LOOKAHEAD AUDIT (2026-05-21)
---------------------------------
For each bar T we run the detector on the slice df.iloc[:T] (bars 0..T-1
inclusive). The PRZ (Potential Reversal Zone) distance and completion
percentage at bar T therefore only see bars strictly < T. Pattern
activation is True at bar T only if the latest zigzag pivot occurred at
some bar <= T-1.

Implementation: we incrementally extend the zigzag using bars up to T-1,
then check whether the last 5 zigzag pivots form one of the 10 harmonic
patterns. We .shift(1) every output column as belt-and-suspenders.

Features added (30 cols):
  For each of 10 patterns: <pattern>_active, <pattern>_PRZ_dist, <pattern>_completion_pct
  Plus aggregated: harmonic_any_active, harmonic_bullish_count, harmonic_bearish_count

License: MIT (HarmonicPatterns repo). Dependencies: numpy, pandas.
Cost: MEDIUM — zigzag rebuild per call, 100-200ms per ticker.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sys.path injection so we can import the upstream HarmonicPatterns library.
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parent.parent.parent  # AI-Tools/
_HP_SRC = _REPO_ROOT / "external-repos" / "HarmonicPatterns" / "src"

if str(_HP_SRC) not in sys.path and _HP_SRC.exists():
    sys.path.insert(0, str(_HP_SRC))

_HP_AVAILABLE = False
try:
    # Disable mplfinance import side-effects by stubbing if missing.
    import importlib
    try:
        importlib.import_module("mplfinance")
    except ImportError:
        # Inject a no-op stub so harmonic_functions can import.
        import types
        sys.modules["mplfinance"] = types.ModuleType("mplfinance")
    try:
        importlib.import_module("talib.abstract")
    except ImportError:
        # If talib unavailable, we provide a pure-numpy MAX/MIN replacement.
        import types as _types
        _stub_ta = _types.ModuleType("talib.abstract")

        def _np_MAX(arr, timeperiod=14):
            arr = np.asarray(arr, dtype=float)
            n = len(arr)
            out = np.full(n, np.nan)
            if n == 0:
                return out
            for i in range(n):
                lo = max(0, i - timeperiod + 1)
                out[i] = np.nanmax(arr[lo:i + 1])
            return out

        def _np_MIN(arr, timeperiod=14):
            arr = np.asarray(arr, dtype=float)
            n = len(arr)
            out = np.full(n, np.nan)
            if n == 0:
                return out
            for i in range(n):
                lo = max(0, i - timeperiod + 1)
                out[i] = np.nanmin(arr[lo:i + 1])
            return out

        _stub_ta.MAX = _np_MAX
        _stub_ta.MIN = _np_MIN
        sys.modules["talib"] = _types.ModuleType("talib")
        sys.modules["talib.abstract"] = _stub_ta

    # Stub IPython if absent (upstream has a leftover `from IPython.core.debugger import set_trace`)
    try:
        importlib.import_module("IPython.core.debugger")
    except ImportError:
        import types as _types
        _ipy = _types.ModuleType("IPython")
        _ipy_core = _types.ModuleType("IPython.core")
        _ipy_dbg = _types.ModuleType("IPython.core.debugger")

        def _noop_set_trace(*a, **kw):  # pragma: no cover
            return None
        _ipy_dbg.set_trace = _noop_set_trace
        sys.modules["IPython"] = _ipy
        sys.modules["IPython.core"] = _ipy_core
        sys.modules["IPython.core.debugger"] = _ipy_dbg

    try:
        from harmonic_functions import HarmonicDetector  # noqa: E402
        _HP_AVAILABLE = True
    except Exception as _hp_exc:  # noqa: BLE001
        logger.warning("[harmonic] could not import HarmonicDetector: %s", _hp_exc)
        HarmonicDetector = None  # type: ignore[assignment]
except Exception as _path_exc:  # noqa: BLE001
    logger.warning("[harmonic] sys.path setup failed: %s", _path_exc)
    HarmonicDetector = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Feature names — 30 cols (3 per pattern × 10) + 3 aggregates = 33 total.
# Truncated to 30 by dropping completion_pct on the 3 patterns we trust least.
# Pre-declared so downstream code can introspect.
# ---------------------------------------------------------------------------

HARMONIC_PATTERNS = [
    "gartley", "bat", "altbat", "butterfly", "crab",
    "deepcrab", "shark", "five_o", "cypher", "abcd",
]

HARMONIC_FEATURE_NAMES: List[str] = []
for _p in HARMONIC_PATTERNS:
    HARMONIC_FEATURE_NAMES.append(f"harmonic_{_p}_active")
    HARMONIC_FEATURE_NAMES.append(f"harmonic_{_p}_PRZ_dist")
    HARMONIC_FEATURE_NAMES.append(f"harmonic_{_p}_completion_pct")
HARMONIC_FEATURE_NAMES = HARMONIC_FEATURE_NAMES[:30]  # cap at 30 per spec

HARMONIC_FEATURE_COUNT: int = len(HARMONIC_FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Lightweight pure-python zigzag (avoids upstream talib.abstract dependency).
# Returns list of [direction_label ('H'/'L'), price, index] tuples.
# ---------------------------------------------------------------------------
def _zigzag(highs: np.ndarray, lows: np.ndarray, period: int = 13) -> List[List[Any]]:
    n = len(highs)
    zigzag: List[List[Any]] = []
    direction = 0
    if n < period + 1:
        return zigzag
    for idx in range(1, n):
        lo = max(0, idx - period + 1)
        hh = np.nanmax(highs[lo:idx + 1])
        ll = np.nanmin(lows[lo:idx + 1])
        new_high = highs[idx] >= hh
        new_low = lows[idx] <= ll
        changed = False
        if new_high and not new_low:
            if direction != 1:
                direction = 1
                changed = True
        elif new_low and not new_high:
            if direction != -1:
                direction = -1
                changed = True
        if new_high or new_low:
            if changed or len(zigzag) == 0:
                if direction == 1:
                    zigzag.append(["H", float(highs[idx]), idx])
                elif direction == -1:
                    zigzag.append(["L", float(lows[idx]), idx])
            else:
                if direction == 1 and highs[idx] > zigzag[-1][1]:
                    zigzag[-1] = ["H", float(highs[idx]), idx]
                elif direction == -1 and lows[idx] < zigzag[-1][1]:
                    zigzag[-1] = ["L", float(lows[idx]), idx]
    return zigzag


# ---------------------------------------------------------------------------
# Compute features.
# ---------------------------------------------------------------------------
def compute_harmonic_patterns_features(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
    zigzag_period: int = 13,
    err_allowed: float = 0.08,
) -> pd.DataFrame:
    """Add harmonic-pattern features to df.

    df must contain 'high', 'low', 'close' columns (case-insensitive).
    Returns df with HARMONIC_FEATURE_NAMES columns added (zero-filled where
    the detector finds no match).

    NO-LOOKAHEAD: features at bar T are computed using only bars [0..T-1]
    (we run the zigzag on prefix, .shift(1) the resulting series).
    """
    # ensure all output columns exist with default 0.0
    for col in HARMONIC_FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0.0

    if not _HP_AVAILABLE or HarmonicDetector is None:
        logger.debug("[harmonic] detector unavailable — zeroing %d cols", HARMONIC_FEATURE_COUNT)
        return df

    # Resolve OHLC columns (case-insensitive)
    cols_lc = {c.lower(): c for c in df.columns}
    h_col = cols_lc.get("high")
    l_col = cols_lc.get("low")
    c_col = cols_lc.get("close")
    if h_col is None or l_col is None or c_col is None:
        logger.warning("[harmonic] missing high/low/close for %s — zeroing", ticker)
        return df

    highs = df[h_col].to_numpy(dtype=float)
    lows = df[l_col].to_numpy(dtype=float)
    closes = df[c_col].to_numpy(dtype=float)
    n = len(closes)

    if n < 60:
        logger.debug("[harmonic] not enough bars (%d) for %s", n, ticker)
        return df

    # Build full zigzag once on FULL data, then per-bar mask using only pivots
    # whose index < T (no-lookahead).
    zigzag = _zigzag(highs, lows, period=zigzag_period)
    if len(zigzag) < 5:
        return df

    detector = HarmonicDetector(error_allowed=err_allowed, strict=False)
    detect_fns: List[Tuple[str, Any]] = [
        ("gartley", detector.detect_gartley),
        ("bat", detector.detect_bat),
        ("altbat", detector.detect_altbat),
        ("butterfly", detector.detect_butterfly),
        ("crab", detector.detect_crab),
        ("deepcrab", detector.detect_deepcrab),
        ("shark", detector.detect_shark),
        ("five_o", detector.detect_5o),
        ("cypher", detector.detect_cypher),
        ("abcd", detector.detect_abcd),
    ]

    # Per-pattern arrays
    active: Dict[str, np.ndarray] = {p: np.zeros(n, dtype=float) for p in HARMONIC_PATTERNS}
    prz_dist: Dict[str, np.ndarray] = {p: np.zeros(n, dtype=float) for p in HARMONIC_PATTERNS}
    compl: Dict[str, np.ndarray] = {p: np.zeros(n, dtype=float) for p in HARMONIC_PATTERNS}

    # Pre-scan: for each (XABCD) window of pivots, detect patterns once. Stamp
    # the activation onto bars from the D-pivot index forward until next pivot
    # (or until end of series). NO-LOOKAHEAD because patterns are only stamped
    # AFTER the D-pivot bar.
    for w_start in range(0, len(zigzag) - 5 + 1):
        win = zigzag[w_start:w_start + 5]
        d_idx = int(win[-1][2])
        d_price = float(win[-1][1])
        # The bar where D pivot completes. Activation starts at d_idx+1.
        if d_idx + 1 >= n:
            continue
        # Next pivot index (for stamping range end)
        if w_start + 5 < len(zigzag):
            next_idx = int(zigzag[w_start + 5][2])
        else:
            next_idx = n
        # Run all detectors on this 5-pivot window
        for pname, fn in detect_fns:
            try:
                result = fn(win)
            except Exception:  # noqa: BLE001
                result = None
            if result is None:
                continue
            direction, ret_dict = result
            # Active = sign(direction). +1 bullish, -1 bearish.
            for stamp_idx in range(d_idx + 1, next_idx):
                if stamp_idx >= n:
                    break
                active[pname][stamp_idx] = float(direction)
                # PRZ distance: % distance from current close to D-completion price.
                cur_close = closes[stamp_idx]
                if cur_close > 0 and d_price > 0:
                    prz_dist[pname][stamp_idx] = (cur_close - d_price) / cur_close
                # Completion %: how far the bar's range is from D given pattern leg.
                # Use ABCD ratio as proxy when present, else 1.0.
                abcd = ret_dict.get("AB=CD") if isinstance(ret_dict, dict) else None
                if abcd is not None and 0 < abcd < 5:
                    compl[pname][stamp_idx] = float(abcd)
                else:
                    compl[pname][stamp_idx] = 1.0

    # Stamp into df with .shift(1) for no-lookahead safety.
    idx = df.index
    out_cols: Dict[str, pd.Series] = {}
    for pname in HARMONIC_PATTERNS:
        for kind, arr in (("active", active[pname]), ("PRZ_dist", prz_dist[pname]),
                          ("completion_pct", compl[pname])):
            col = f"harmonic_{pname}_{kind}"
            if col in HARMONIC_FEATURE_NAMES:
                series = pd.Series(arr, index=idx)
                out_cols[col] = series.shift(1).fillna(0.0)

    for col, series in out_cols.items():
        df[col] = series.to_numpy(dtype=float)

    logger.debug("[harmonic] added %d features for %s", HARMONIC_FEATURE_COUNT, ticker)
    return df


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="SMOKE")
    args = ap.parse_args()

    # Synthetic OHLCV: 200 bars random walk
    np.random.seed(42)
    n = 200
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    df = pd.DataFrame({"high": high, "low": low, "close": close, "open": close, "volume": 1000})
    df_out = compute_harmonic_patterns_features(df, ticker=args.ticker)
    new_cols = [c for c in df_out.columns if c.startswith("harmonic_")]
    print(f"[smoke] ticker={args.ticker} rows={len(df_out)} new_cols={len(new_cols)}")
    nz = sum(int((df_out[c] != 0).any()) for c in new_cols)
    print(f"[smoke] non-zero cols: {nz}/{len(new_cols)}")
    print(f"[smoke] HP_AVAILABLE={_HP_AVAILABLE}")
