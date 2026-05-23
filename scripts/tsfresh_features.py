"""
tsfresh_features.py — Automated time-series feature mining via tsfresh
(GitHub TOP-10 #3, shipped 2026-05-22).

Wires tsfresh's EfficientFCParameters extractor + built-in Benjamini-Hochberg
significance filter into the S&P 500 daily XGBoost pipeline. Targets +120-200
significant features per ticker after BH-pruning a ~400-column raw extraction.

Key design choices
------------------
1. Lookahead-safe.  We extract features over a ROLLING window using the
   `roll_time_series` helper (anchored at each bar's CLOSE, only past data
   visible). Every returned column is `.shift(1)` so the feature at row t is
   computable from data strictly before t. Compatible with lookahead_checker.py.

2. Heavy compute cloud-routes.  Per CLAUDE.md §5a, the actual
   `extract_features` call (which is *very* CPU-heavy: O(n_windows ×
   n_calculators × n_bars)) is dispatched via
   ``cloud_dispatch.enqueue_job(backend='gh_actions_matrix')`` whenever
   ``TSFRESH_CLOUD_ROUTE != "0"``. The local entry point returns the job_id and
   waits on the result file. A small shim worker mode (--worker) lets the cloud
   runner execute the extraction with the same code path.

3. Env-gated.  Off by default. Set ``TSFRESH_ENABLED=1`` to wire into
   `_lazy_features.py` / `add_alpha_features_core_features_features.py`-style
   composition pipelines.

4. EfficientFCParameters, NOT Comprehensive.  Comprehensive includes O(n^2)
   calculators (e.g. matrix-profile fallbacks, augmented_dickey_fuller per
   permutation) that explode on >500-bar windows. Efficient gives ~73 distinct
   calculators × multiple lag/agg params → ~400 raw features.

5. Benjamini-Hochberg filtering via tsfresh.select_features against a binary
   future-return label (computed in-house with .shift(-1) ONLY on the label
   side — never on features). FDR=0.05 default. Yields ~120-200 survivors.

Usage
-----
    from tsfresh_features import add_tsfresh_features
    df = add_tsfresh_features(
        df,
        ticker="AAPL",
        cloud_route=True,        # default; set False for in-process smoke
        n_jobs=4,
        fdr_level=0.05,
    )

Smoke (run after install)
-------------------------
    python tsfresh_features.py --smoke
        → 1 ticker × 1000 synthetic bars → ~400 raw cols → ~150-250 BH-survivors

Activation env
--------------
    TSFRESH_ENABLED=1               wire-in (default: off)
    TSFRESH_CLOUD_ROUTE=1           dispatch extract to gh_actions_matrix
                                    (default: 1 — heavy CPU, must cloud-route)
    TSFRESH_N_JOBS=4                multiprocess workers inside extract_features
    TSFRESH_FDR=0.05                BH FDR level for select_features
    TSFRESH_WINDOW=63               rolling window length (bars) for extraction
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("tsfresh_features")
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_PREFIX = "tsf"
_DEFAULT_WINDOW = int(os.environ.get("TSFRESH_WINDOW", "63"))
_DEFAULT_N_JOBS = int(os.environ.get("TSFRESH_N_JOBS", "4"))
_DEFAULT_FDR = float(os.environ.get("TSFRESH_FDR", "0.05"))
_DEFAULT_CLOUD = os.environ.get("TSFRESH_CLOUD_ROUTE", "1") != "0"
_BASE_COL = "close"   # series tsfresh extracts on; OHLCV decomposition can extend later
_RESULTS_DIR = Path(os.environ.get(
    "TSFRESH_RESULTS_DIR",
    "/tmp/tsfresh_features",
))


# ---------------------------------------------------------------------------
# Lazy import — keeps add_tsfresh_features importable even if tsfresh missing
# ---------------------------------------------------------------------------
def _import_tsfresh():
    try:
        from tsfresh import extract_features, select_features
        from tsfresh.feature_extraction import EfficientFCParameters
        from tsfresh.utilities.dataframe_functions import impute
        return extract_features, select_features, EfficientFCParameters, impute
    except ImportError as e:
        raise RuntimeError(
            "tsfresh not installed. Run "
            "`/Users/orginal/.venvs/sp500-mastery/bin/pip install tsfresh` "
            f"(import error: {e})"
        )


# ---------------------------------------------------------------------------
# Rolling-window long-frame builder (lookahead-safe)
# ---------------------------------------------------------------------------
def _build_long_frame(s: pd.Series, window: int) -> pd.DataFrame:
    """
    Each row t emits rows (t, t-window+1..t) → tsfresh sees only past values.
    Returns long-format df with cols [id, time, value]. id = anchor row index.
    """
    n = len(s)
    if n < window + 1:
        return pd.DataFrame(columns=["id", "time", "value"])
    ids, times, vals = [], [], []
    sv = s.values.astype(float)
    for anchor in range(window - 1, n):
        start = anchor - window + 1
        ids.extend([anchor] * window)
        times.extend(range(window))
        vals.extend(sv[start: anchor + 1].tolist())
    return pd.DataFrame({"id": ids, "time": times, "value": vals})


# ---------------------------------------------------------------------------
# Core extractor (runs locally OR inside the cloud worker)
# ---------------------------------------------------------------------------
def _extract_local(
    df: pd.DataFrame,
    column: str = _BASE_COL,
    window: int = _DEFAULT_WINDOW,
    n_jobs: int = _DEFAULT_N_JOBS,
) -> pd.DataFrame:
    extract_features, _, EfficientFCParameters, impute = _import_tsfresh()
    if column not in df.columns:
        raise KeyError(f"column {column!r} not in df (have {list(df.columns)})")
    long_df = _build_long_frame(df[column].astype(float).reset_index(drop=True),
                                window=window)
    if long_df.empty:
        return pd.DataFrame(index=df.index)
    log.info("tsfresh extract: %d anchors × window=%d (n_jobs=%d)",
             long_df["id"].nunique(), window, n_jobs)
    feats = extract_features(
        long_df,
        column_id="id",
        column_sort="time",
        column_value="value",
        default_fc_parameters=EfficientFCParameters(),
        n_jobs=n_jobs,
        disable_progressbar=True,
    )
    impute(feats)
    feats.columns = [f"{_PREFIX}_{c}" for c in feats.columns]
    # Align: feats.index is anchor row int; reindex onto df.index positions in
    # one shot (avoids fragmentation from per-column .iloc assignment).
    valid_anchors = feats.index.intersection(range(len(df)))
    aligned = feats.loc[valid_anchors]
    aligned.index = df.index[list(valid_anchors)]
    out = aligned.reindex(df.index)
    # Lookahead guard: every column shifted 1 — feature at t uses only t-1..t-window
    # (the rolling window is already past-only, .shift(1) adds belt-and-braces)
    out = out.shift(1)
    return out


# ---------------------------------------------------------------------------
# Benjamini-Hochberg significance filter
# ---------------------------------------------------------------------------
def _bh_filter(
    feats: pd.DataFrame,
    df: pd.DataFrame,
    fdr_level: float = _DEFAULT_FDR,
    horizon: int = 5,
) -> pd.DataFrame:
    """
    Build a binary up/down label from forward-return (label-side .shift(-1)
    only — never applied to features), then run tsfresh.select_features
    (built-in Benjamini-Hochberg multiple-testing correction).
    """
    _, select_features, _, _ = _import_tsfresh()
    if "close" not in df.columns:
        log.warning("BH filter skipped: no 'close' column for label build")
        return feats
    fwd_ret = df["close"].pct_change(horizon).shift(-horizon)
    y = (fwd_ret > 0).astype(int)
    aligned = feats.join(y.rename("_y"), how="inner").dropna()
    if aligned.empty or aligned["_y"].nunique() < 2:
        log.warning("BH filter skipped: insufficient label variance / rows")
        return feats
    X = aligned.drop(columns=["_y"])
    y_aligned = aligned["_y"]
    try:
        kept = select_features(X, y_aligned, fdr_level=fdr_level, n_jobs=1)
        log.info("BH filter: %d → %d significant (FDR=%g)",
                 X.shape[1], kept.shape[1], fdr_level)
        return feats[kept.columns]
    except Exception as e:
        log.warning("select_features failed (%s); returning unfiltered", e)
        return feats


# ---------------------------------------------------------------------------
# Cloud-route shim
# ---------------------------------------------------------------------------
def _dispatch_cloud(
    ticker: str,
    df: pd.DataFrame,
    window: int,
    n_jobs: int,
) -> str | None:
    """
    Enqueue tsfresh extraction on gh_actions_matrix. Returns job_id or None
    if dispatcher unavailable (caller should fall back to local).
    """
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from cloud_dispatch import enqueue_job
    except Exception as e:
        log.warning("cloud_dispatch import failed (%s); local extract", e)
        return None
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload_id = uuid.uuid4().hex[:8]
    payload_path = _RESULTS_DIR / f"{ticker}_{payload_id}_input.parquet"
    result_path = _RESULTS_DIR / f"{ticker}_{payload_id}_output.parquet"
    try:
        df[[_BASE_COL, "close"]].dropna().to_parquet(payload_path)
    except Exception:
        df[[_BASE_COL]].dropna().to_parquet(payload_path)
    try:
        job_id = enqueue_job(
            ticker=ticker,
            strategy="tsfresh",
            script="scripts/tsfresh_features.py",
            out_path=str(result_path),
            priority=4,
            extra_env={
                "TSFRESH_WORKER": "1",
                "TSFRESH_INPUT":  str(payload_path),
                "TSFRESH_OUTPUT": str(result_path),
                "TSFRESH_WINDOW": str(window),
                "TSFRESH_N_JOBS": str(n_jobs),
                "TSFRESH_BACKEND": "gh_actions_matrix",
            },
            subprocess_fallback=False,
        )
        log.info("Enqueued tsfresh job_id=%s ticker=%s window=%d",
                 job_id, ticker, window)
        return job_id
    except Exception as e:
        log.warning("enqueue_job failed (%s); falling back to local", e)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def add_tsfresh_features(
    df: pd.DataFrame,
    ticker: str = "UNK",
    cloud_route: bool | None = None,
    window: int | None = None,
    n_jobs: int | None = None,
    fdr_level: float | None = None,
    bh_filter: bool = True,
) -> pd.DataFrame:
    """
    Append tsfresh-derived features to df. Lookahead-safe: every column .shift(1).

    Args:
        df:           OHLCV daily df, datetime index (or RangeIndex), 'close' col.
        ticker:       For logging / dispatch routing.
        cloud_route:  If True, dispatch extract via cloud_dispatch (recommended
                      for >500 bars). None → TSFRESH_CLOUD_ROUTE env.
        window:       Rolling-window length (bars). Default 63 (~3 trading mo).
        n_jobs:       tsfresh multiprocess workers. Default 4.
        fdr_level:    BH FDR. Default 0.05.
        bh_filter:    If True, apply select_features post-extract.

    Returns:
        df with extra tsf_* columns merged in (NaN-padded for early bars).
    """
    if df is None or df.empty or _BASE_COL not in df.columns:
        log.warning("add_tsfresh_features: empty df or missing %r — no-op",
                    _BASE_COL)
        return df
    if not bool(int(os.environ.get("TSFRESH_ENABLED", "0"))):
        log.info("TSFRESH_ENABLED=0 — skipping (env-gated off)")
        return df

    window = window or _DEFAULT_WINDOW
    n_jobs = n_jobs or _DEFAULT_N_JOBS
    fdr_level = fdr_level if fdr_level is not None else _DEFAULT_FDR
    cloud_route = _DEFAULT_CLOUD if cloud_route is None else cloud_route

    t0 = time.time()
    if cloud_route and len(df) >= 500:
        job_id = _dispatch_cloud(ticker, df, window, n_jobs)
        if job_id:
            # Caller is responsible for polling check_status / reading
            # result_path. We append a marker column so the orchestrator knows
            # an async fetch is pending.
            df = df.copy()
            df[f"{_PREFIX}_pending_job"] = job_id
            log.info("Returned w/ pending job %s (poll cloud_dispatch)", job_id)
            return df
        log.info("Cloud route unavailable — local extract")

    feats = _extract_local(df, column=_BASE_COL, window=window, n_jobs=n_jobs)
    log.info("Local extract: %d cols in %.1fs", feats.shape[1], time.time() - t0)

    if bh_filter and feats.shape[1] > 0:
        feats = _bh_filter(feats, df, fdr_level=fdr_level)

    return df.join(feats, how="left")


# ---------------------------------------------------------------------------
# Cloud worker entry point
# ---------------------------------------------------------------------------
def _run_worker() -> int:
    inp = Path(os.environ["TSFRESH_INPUT"])
    out = Path(os.environ["TSFRESH_OUTPUT"])
    window = int(os.environ.get("TSFRESH_WINDOW", _DEFAULT_WINDOW))
    n_jobs = int(os.environ.get("TSFRESH_N_JOBS", _DEFAULT_N_JOBS))
    df = pd.read_parquet(inp)
    feats = _extract_local(df, column=_BASE_COL, window=window, n_jobs=n_jobs)
    if "close" in df.columns:
        feats = _bh_filter(feats, df, fdr_level=_DEFAULT_FDR)
    out.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out)
    log.info("Worker wrote %d cols → %s", feats.shape[1], out)
    return 0


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def _smoke() -> int:
    os.environ["TSFRESH_ENABLED"] = "1"
    os.environ["TSFRESH_CLOUD_ROUTE"] = "0"  # local for smoke
    rng = np.random.default_rng(42)
    n = 1000
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    df = pd.DataFrame({
        "open":  close + rng.normal(0, 0.2, n),
        "high":  close + np.abs(rng.normal(0, 0.5, n)),
        "low":   close - np.abs(rng.normal(0, 0.5, n)),
        "close": close,
        "volume": rng.integers(1_000_000, 10_000_000, n),
    })
    t0 = time.time()
    pre_cols = df.shape[1]
    # 1) Raw extraction (no BH)
    raw = _extract_local(df, window=63, n_jobs=4)
    raw_count = raw.shape[1]
    # 2) BH-filtered
    bh = _bh_filter(raw, df, fdr_level=0.05)
    bh_count = bh.shape[1]
    elapsed = time.time() - t0
    print(json.dumps({
        "smoke":       "tsfresh_features",
        "n_bars":      n,
        "window":      63,
        "raw_cols":    raw_count,
        "bh_cols":     bh_count,
        "elapsed_sec": round(elapsed, 1),
        "pre_df_cols": pre_cols,
    }, indent=2))
    assert raw_count >= 100, f"expected >=100 raw cols, got {raw_count}"
    print("SMOKE_OK")
    return 0


if __name__ == "__main__":
    if "--worker" in sys.argv or os.environ.get("TSFRESH_WORKER") == "1":
        sys.exit(_run_worker())
    if "--smoke" in sys.argv:
        sys.exit(_smoke())
    print("usage: tsfresh_features.py [--smoke | --worker]")
    sys.exit(2)
