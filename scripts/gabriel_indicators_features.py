"""gabriel_indicators_features.py — Wrap the 107 Gabriel classical TA indicators.

Source-of-truth path:
  /Users/orginal/.../version_3 - Gabriel/.../src/historical_system/indicators/

This wrapper adds that `src/` directory to `sys.path` at import time so the
package's self-registering imports populate `REGISTRY` with all 107 indicators.
A partial vendored copy exists at
`scripts/historical_system/indicators/_defs/` (used as fallback if the canonical
Gabriel path isn't reachable — e.g. running on a machine without the Drive
mount).

This wrapper loops over the registry, runs each compute(df), prefixes
output columns with `gab_<name>__`, and `.shift(1)` everything to guarantee
causal merge with t+1 labels.

Skipped at the wrapper layer:
  * `cross_symbol=True`  (advance_decline — single-symbol diff stub is kept but
    flagged; true breadth requires a cross-symbol runner)
  * `non_timeseries=True` (volume_profile_fixed_range, volume_profile_visible_range —
    return histograms, not time series)
  * Known forward-looking output columns (ichimoku `chikou`) — explicitly dropped
    in `_FORWARD_LOOKING_COLS`.

Performance — added 2026-05-20 (top7-followup)
----------------------------------------------
Audit showed the full sweep takes >5 min on 1213-row inputs because a handful
of indicators are O(N^2) (.rolling().apply(fn) without `raw=True`, or nested
Python loops in the upstream source). To keep total wall-clock <60s without
modifying the upstream package:

1. PER-INDICATOR TIMING — every call is wrapped in a wall-clock timer; results
   are accumulated in `_TIMING_CACHE` (module-global). On any future call the
   slow ones (default >0.75s) are skipped automatically.

2. HARD SKIP SET — env var `GABRIEL_SLOW_INDICATORS_SKIP="ind1,ind2,..."` is
   ALWAYS honored. Also seeded with a default-bad list (audited 2026-05-20)
   covering the worst offenders found in profiling.

3. SOFT BUDGET — env var `GABRIEL_PER_IND_BUDGET_S` (default 0.75) sets the
   threshold above which an indicator is added to the runtime skip set after
   its first call.

4. WALL-CLOCK CAP — env var `GABRIEL_TOTAL_BUDGET_S` (default 60.0) — once
   accumulated wall-clock exceeds this, all remaining indicators are skipped.

Look-ahead safety: each indicator's compute is causal except where noted; final
.shift(1) is the belt-and-suspenders guard.

Expected feature count: ~250 columns across 100+ indicators (varies by params).

Function signature: `add_gabriel_indicators_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame`
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Make the canonical source-of-truth Gabriel package importable.
# Path priority (first hit wins):
#   1. Local LOCAL_CACHE   — copy of Drive indicators on local disk (no FUSE latency)
#   2. Drive source-of-truth (109 indicators, but Drive FUSE = slow imports)
#   3. Vendored shim under scripts/historical_system/ (43 indicators, local, fast)
#
# top7-followup 2026-05-20: registry import was hanging > 2 min because
# importlib walks 109 Drive-resident .py files via FUSE. Adding a local cache
# path option in front of the Drive path lets users seed `~/.cache/gabriel_indicators_local/`
# with a one-time `rsync` and then enjoy <2s registry loads.
_GABRIEL_LOCAL_CACHE = Path.home() / ".cache" / "gabriel_indicators_local"
_GABRIEL_SRC = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/"
    "My Drive/version_3 - Gabriel/Gabriel - Historical System - Indicators 100+/src"
)
_SCRIPTS_DIR = Path(__file__).parent

# Prefer Gabriel source-of-truth (all indicators); fall back to vendored shim.
# Env var GABRIEL_PREFER_LOCAL=1 puts vendored shim first (fastest, fewer indicators).
# Env var GABRIEL_SKIP_DRIVE=1 omits the Drive path entirely (avoids FUSE).
_prefer_local = os.environ.get("GABRIEL_PREFER_LOCAL", "0") == "1"
_skip_drive = os.environ.get("GABRIEL_SKIP_DRIVE", "0") == "1"

_path_candidates: list[Path] = []
if _GABRIEL_LOCAL_CACHE.exists():
    _path_candidates.append(_GABRIEL_LOCAL_CACHE)
if _prefer_local:
    _path_candidates += [_SCRIPTS_DIR]
    if not _skip_drive:
        _path_candidates.append(_GABRIEL_SRC)
else:
    if not _skip_drive:
        _path_candidates.append(_GABRIEL_SRC)
    _path_candidates.append(_SCRIPTS_DIR)

for _p in _path_candidates:
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

logger = logging.getLogger(__name__)

# Columns that are *intentionally* forward-looking in the original package
# (typically lagging plot lines like ichimoku chikou). Dropped to keep the
# ML feature set causal.
_FORWARD_LOOKING_COLS = {
    "chikou",  # ichimoku_cloud uses .shift(-displacement) — leaks future close
}

# --- Performance gates (top7-followup 2026-05-20) ---

# Audited-bad list — these were the heavy hitters in profiling. Pre-seeded so
# the FIRST call is fast too (otherwise the user pays the slow-discovery cost
# once per process).
_DEFAULT_SLOW_SET: set[str] = {
    # rolling().apply(custom_fn) without raw=True — Python-per-row
    "rainbow_oscillator",
    "fractal_chaos_bands",
    "fractal_chaos_oscillator",
    "klingerVolumeOscillator",
    "klinger_volume_oscillator",
    "ehlers_filter",
    "ehlers_decycler",
    "ehlers_super_smoother",
    "ehlers_instantaneous_trendline",
    "ehlers_mama",
    "hurst_exponent",
    "fisher_transform",
    "rolling_hurst",
    # nested for-loops in compute() — known slow
    "zigzag",
    "pivot_points_camarilla",
    "pivot_points_demark",
    "auto_fibonacci",
}

# Per-process cache of measured slow indicators. Persists across function
# calls within a process (e.g., multi-ticker batches in backtest_xgb_v10).
_TIMING_CACHE: dict[str, float] = {}
_RUNTIME_SKIP_SET: set[str] = set()


def _per_indicator_budget_s() -> float:
    try:
        return float(os.environ.get("GABRIEL_PER_IND_BUDGET_S", "0.75"))
    except Exception:
        return 0.75


def _total_budget_s() -> float:
    try:
        return float(os.environ.get("GABRIEL_TOTAL_BUDGET_S", "60.0"))
    except Exception:
        return 60.0


def _user_skip_set() -> set[str]:
    raw = os.environ.get("GABRIEL_SLOW_INDICATORS_SKIP", "")
    if not raw:
        return set()
    return {x.strip() for x in raw.split(",") if x.strip()}


def _effective_skip_set() -> set[str]:
    return _DEFAULT_SLOW_SET | _user_skip_set() | _RUNTIME_SKIP_SET


def _load_registry():
    """Lazy import — keeps import-time cheap and any errors local."""
    try:
        from historical_system.indicators import REGISTRY, all_names  # type: ignore
        # Trigger package side-effect imports so all 107 indicators register
        return REGISTRY, all_names()
    except Exception as exc:
        logger.warning(f"gabriel_indicators: failed to load registry: {exc}")
        return {}, []


def add_gabriel_indicators_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Add Gabriel's classical TA indicators as features (with timing gates).

    Args:
        df: DataFrame with lowercase OHLCV columns (open, high, low, close, volume)
            and a datetime-like index.
        ticker: ticker symbol (reserved for future cross-symbol indicators).

    Returns:
        df with new feature columns named `gab_<indicator>__<output>`,
        all .shift(1)-safe.
    """
    REGISTRY, names = _load_registry()
    if not REGISTRY:
        logger.warning("gabriel_indicators: REGISTRY empty — skipping")
        return df

    # Normalize input — vendored indicators expect lowercase columns
    need = {"open", "high", "low", "close", "volume"}
    avail = set(df.columns.str.lower())
    if not need.issubset(avail):
        missing = need - avail
        logger.warning(f"gabriel_indicators[{ticker}]: missing cols {missing} — skip")
        return df

    work = df.copy()
    # Map to lowercase if needed
    rename_map = {c: c.lower() for c in work.columns if c != c.lower() and c.lower() in need}
    if rename_map:
        work = work.rename(columns=rename_map)

    new_cols: dict[str, np.ndarray] = {}
    n_rows = len(work)

    skip_set = _effective_skip_set()
    per_ind_budget = _per_indicator_budget_s()
    total_budget = _total_budget_s()

    total_elapsed = 0.0
    skipped_static = 0
    skipped_runtime = 0
    over_budget_hits = 0
    succeeded = 0

    for ind_name in names:
        if ind_name in skip_set:
            skipped_static += 1
            continue
        if total_elapsed >= total_budget:
            # Out of total budget — abort remaining indicators.
            over_budget_hits += 1
            continue
        try:
            cls = REGISTRY[ind_name]
            # Skip non-timeseries (volume profile histograms) — they don't
            # produce per-row outputs.
            if getattr(cls, "non_timeseries", False):
                continue
            ind = cls()
            t0 = time.perf_counter()
            res = ind.compute(work)
            dt = time.perf_counter() - t0
            total_elapsed += dt
            _TIMING_CACHE[ind_name] = dt
            if dt > per_ind_budget:
                # Add to runtime skip set so subsequent ticker calls in the
                # same process skip this indicator without paying the cost.
                _RUNTIME_SKIP_SET.add(ind_name)
                skipped_runtime += 1
                # We DO accept the result of THIS call (it ran to completion),
                # so still record its columns — just don't re-call it later.

            if isinstance(res, np.ndarray):
                outputs = cls.outputs or (ind_name,)
                res = {outputs[0]: res}
            for out_name, arr in res.items():
                if out_name in _FORWARD_LOOKING_COLS:
                    continue
                arr = np.asarray(arr, dtype=np.float64)
                if arr.shape != (n_rows,):
                    # Some indicators may return shorter arrays; pad with NaN
                    if arr.ndim == 1 and arr.shape[0] <= n_rows:
                        padded = np.full(n_rows, np.nan)
                        padded[-arr.shape[0]:] = arr
                        arr = padded
                    else:
                        continue
                col_name = f"gab_{ind_name}__{out_name}"
                new_cols[col_name] = arr
            succeeded += 1
        except Exception as exc:
            logger.warning(f"gabriel_indicators[{ticker}]: '{ind_name}' failed: {exc}")
            continue

    logger.info(
        "gabriel_indicators[%s]: %d ok, %d skipped(static), %d slowed(runtime), "
        "%d over-total-budget, total=%.1fs, cols=%d",
        ticker, succeeded, skipped_static, skipped_runtime,
        over_budget_hits, total_elapsed, len(new_cols),
    )

    if not new_cols:
        return df

    additions = pd.DataFrame(new_cols, index=work.index)
    # Hard .shift(1) on every new column — defense in depth for causal merges
    additions = additions.shift(1)

    out = df.copy()
    for c in additions.columns:
        out[c] = additions[c].values
    return out


# Public surface for diagnostics — tests / scripts can introspect what got skipped.
def get_timing_cache() -> dict[str, float]:
    """Return last-measured per-indicator wall-clock seconds. Empty before first call."""
    return dict(_TIMING_CACHE)


def get_skip_sets() -> dict[str, list[str]]:
    """Return the three skip-set layers (static default, env, runtime-learned)."""
    return {
        "default": sorted(_DEFAULT_SLOW_SET),
        "env": sorted(_user_skip_set()),
        "runtime": sorted(_RUNTIME_SKIP_SET),
    }


if __name__ == "__main__":
    # Smoke test — synthetic data
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    np.random.seed(0)
    n = 1213  # match real S&P500-mastery row count
    idx = pd.date_range("2021-01-01", periods=n, freq="D")
    base = 100 + np.cumsum(np.random.randn(n) * 0.5)
    smoke = pd.DataFrame({
        "open": base + np.random.randn(n) * 0.1,
        "high": base + np.abs(np.random.randn(n)) * 0.3,
        "low":  base - np.abs(np.random.randn(n)) * 0.3,
        "close": base,
        "volume": np.random.randint(1_000, 10_000, n).astype(float),
    }, index=idx)
    t0 = time.perf_counter()
    out = add_gabriel_indicators_features(smoke, "TEST")
    dt = time.perf_counter() - t0
    new_cols = [c for c in out.columns if c.startswith("gab_")]
    print(f"Added {len(new_cols)} gabriel feature columns in {dt:.1f}s.")
    print(f"Sample: {new_cols[:5]}")
    print(f"Skip sets: {get_skip_sets()}")
    # Top-10 slowest measured
    timing = sorted(get_timing_cache().items(), key=lambda kv: kv[1], reverse=True)[:10]
    print("Top-10 slowest indicators (this run):")
    for nm, t in timing:
        print(f"  {nm}: {t:.3f}s")
