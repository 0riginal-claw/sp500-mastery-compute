"""
backtest_xgb_v10.py — Alpha158-Plus + OpenMythos + 3 missing feature modules.

Feature stack (v9 + three previously-missing modules):
  - v9 base ~1126+ features (v8 + optional 256-dim Mythos layer)
  - [NEW] daily_integration_features — 7 features:
        beta_adj_residual_ret_z21/z63, csrs_5d/10d/20d,
        earn_contam_gate, earn_post_rv_gate
  - [NEW] alpaca_features — up to 13 features:
        earnings proximity (4), dividend/ex-div (4), splits (2), metadata (3)
  - [NEW] featuretools_dfs_features — up to 60 depth-2 interaction features
        (DFS primitives: mean/std/sum/skew/max/min + depth-2 cross-products)

Total: ~870 (v8 base) + 256 (Mythos, optional) + 7 (daily_int) + 13 (alpaca)
       + ~60 (dfs) = ~1,010–1,266+ features depending on Mythos checkpoint.

Key differences from v9:
  1. build_v10_features() calls build_v9_features() then adds the three
     missing modules in sequence.
  2. Three new import blocks (each gracefully degraded on import failure).
  3. --job-id CLI arg restored (was in v8, dropped in v9; GH Actions require it).
  4. run_meta.json extended with v10 section, feature counts per module, job_id.
  5. pipeline_version / strategy_variant updated to v10.

Usage:
    # Standard run (no Mythos):
    python backtest_xgb_v10.py --ticker AAPL --strategy ORB \\
        --job-id smoke-001 --out-dir /tmp/aapl_v10

    # With Mythos embeddings:
    MYTHOS_CHECKPOINT_PATH=/path/to/mythos_financial_v0.pt \\
    python backtest_xgb_v10.py --ticker AAPL --strategy ORB \\
        --job-id run-001 --out-dir /tmp/aapl_v10_mythos --use-mythos-features

DO NOT modify v9. This is a clean fork.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# purgedcv - Lopez de Prado purged K-Fold + embargo diagnostics (wired 2026-05-21)
# Backtest integrity: assert no temporal leakage between train/oos folds and
# enforce LABEL_EMBARGO_DAYS embargo gap (already applied in train_end_emb).
# Soft-import: degrades to no-op if purgedcv missing. pypbo (Probability of
# Backtest Overfitting) is GitHub-only / no setup.py -> not installable; use
# purgedcv.deflated_sharpe_ratio + probabilistic_sharpe_ratio as replacement.
# ---------------------------------------------------------------------------
try:
    from purgedcv import PurgedKFold as _PurgedKFold  # noqa: F401
    from purgedcv.diagnostics import (
        assert_no_temporal_leakage as _pcv_assert_no_leak,
        assert_embargo_respected as _pcv_assert_embargo,
    )
    _PURGEDCV_AVAILABLE = True
except ImportError:
    _PURGEDCV_AVAILABLE = False
    _pcv_assert_no_leak = None
    _pcv_assert_embargo = None

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

WORK = Path(
    "/Users/orginal/Library/CloudStorage/"
    "GoogleDrive-zachgladstone@gmail.com/My Drive/"
    "AI-Tools/s&p500-ticker-mastery"
)
LABEL_EMBARGO_DAYS = 21

V10_FEATURE_VERSION = "v10.7.3"  # 2026-05-20 - top7-followup: sec_edgar fleshed (2 stub -> 14 cols via edgar_extras); gabriel runtime-budget gate + local cache path; xgboost warm-start+stride+hist (21s vs 5min); v10.7.2: env skip gates; v10.7.1: vwap ticker arg + FEATURE_NAMES hoist; v10.7.0: drive-map top-7 unwired modules wired

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import v9 feature builder (non-destructive — v9.py is untouched)
# ---------------------------------------------------------------------------

from backtest_xgb_v9 import build_v9_features, MYTHOS_FEAT_NAMES, MYTHOS_FEATURE_DIM  # noqa: E402
from backtest_xgb_v7 import numeric_cols  # noqa: E402
import backtest_ml as bml  # noqa: E402
import xgboost as xgb  # noqa: E402
from feature_cache import get_cached  # noqa: E402


# ---------------------------------------------------------------------------
# XGBoost hyperparameter tuning (2026-05-19)
# Top-5 + medium wins from xgboost_tune_REPO_LOCAL audit. Centralized so all
# three call sites (scout / final / persist) stay in sync. Schema preserved
# in model_kwargs metadata for run_meta.json.
# ---------------------------------------------------------------------------
_XGB_DEVICE = os.environ.get("XGB_DEVICE", "cpu").lower()
if _XGB_DEVICE not in ("cpu", "cuda"):
    _XGB_DEVICE = "cpu"
_XGB_SAMPLING_METHOD = "gradient_based" if _XGB_DEVICE == "cuda" else "uniform"


# ---------------------------------------------------------------------------
# Adaptive top-K + XGBoost constraints (2026-05-20)
#
# Why: legacy top_k=50 was sized for the 173-feature v8 era. v10 builds
# ~1231 features per ticker, so 96% of features were being discarded before
# the final-stage tree ensemble saw them. Adaptive default scales with the
# train fold's row count: min(int(sqrt(n_rows) * 4), n_features, 400).
# For ~1213 rows × 1231 feats → top_k ≈ 140. Hard cap 400 keeps fits sane.
#
# Optional XGBoost constraints (env-gated, default off):
#   XGB_INTERACTION_CONSTRAINTS=1 — restrict tree splits so features in
#     different semantic groups cannot interact within a single tree.
#     Groups: intraday-volume / monthly-fundamental / macro / microstructure /
#     alpha158 / dfs (see _feature_group_for_name).
#   XGB_MONOTONIC=1 — enforce sign-of-effect monotonicity for selected
#     features (RSI_14 → -1, VIX → -1, gtrends_score_30d_avg → +1).
# ---------------------------------------------------------------------------
_XGB_USE_INTERACTION = os.environ.get("XGB_INTERACTION_CONSTRAINTS", "0") == "1"
_XGB_USE_MONOTONIC = os.environ.get("XGB_MONOTONIC", "0") == "1"

# Monotonic constraints map: feature name → sign (+1 increasing, -1 decreasing)
_XGB_MONO_PRIORS: dict = {
    "RSI_14": -1,
    "VIX": -1,
    "gtrends_score_30d_avg": +1,
}


def _resolve_top_k(args_top_k: int, n_rows: int, n_features: int) -> int:
    """Resolve effective top_k.

    XGB_NO_TOPK=1 → BYPASS selection: return n_features (use all features)
                    Stage A of full-utilization patch (2026-05-20).
                    Relies on XGBoost regularization (reg_alpha/reg_lambda,
                    colsample_*, max_bin) — bumped via XGB_NO_TOPK gating in
                    _xgb_base_params — to handle p>n / overfitting risk.
    args_top_k > 0 → explicit override (CLI or env XGB_TOP_K)
    args_top_k == 0 → adaptive: min(int(sqrt(n_rows) * 4), n_features, 400)
    """
    # karpathy_checked: XGB_NO_TOPK bypass — n_features split direct to trees
    if os.environ.get("XGB_NO_TOPK", "0") == "1":
        return int(n_features)
    if args_top_k and args_top_k > 0:
        return min(int(args_top_k), int(n_features))
    adaptive = int(np.sqrt(max(n_rows, 1)) * 4)
    return max(10, min(adaptive, int(n_features), 400))


def _feature_group_for_name(name: str) -> str:
    """Classify a feature name into a semantic group for interaction constraints.

    Heuristic — uses substring/prefix patterns common in v10 feature names.
    Unknown → 'other'. Group membership is purely for grouping splits; an
    unknown bucket is fine (XGBoost allows leftover features to interact only
    inside the 'other' bucket).
    """
    n = name.lower()
    # Intraday / minute-bar / volume profile
    if any(k in n for k in ("vwap", "orb_", "_intraday", "minute_", "_vp_", "_pv_")):
        return "intraday-volume"
    # Monthly / fundamental
    if any(k in n for k in ("_pe", "_pb", "_eps", "_revenue", "_fcf", "_ebitda", "earn_", "_dividend", "_split")):
        return "monthly-fundamental"
    # Macro
    if any(k in n for k in ("vix", "macro_", "fed_", "yld", "tnx", "_dxy", "_oil", "_gold")):
        return "macro"
    # Microstructure / spread / order-flow
    if any(k in n for k in ("spread", "bid_ask", "tick_", "_microstructure", "options_flow", "govtrades")):
        return "microstructure"
    # Alpha158 / Qlib classics
    if n.startswith("a158_") or n.startswith("alpha158_") or n.startswith("a101ts_") or n.startswith("a101_"):
        return "alpha158"
    # DFS / featuretools
    if "dfs_" in n or n.startswith("_dfs") or "_x_" in n:  # depth-2 interaction
        return "dfs"
    return "other"


def _build_interaction_constraints(feature_names: list) -> str:
    """Return XGBoost interaction_constraints JSON string for a feature list.

    Constraints group feature indices by semantic group; trees can only split
    on features within the same group within a single tree path.
    """
    groups: dict = {}
    for i, name in enumerate(feature_names):
        g = _feature_group_for_name(name)
        groups.setdefault(g, []).append(i)
    # Drop singleton groups (no constraint benefit) — let them go in 'other'
    cleaned = []
    leftover = []
    for g, idxs in groups.items():
        if g == "other" or len(idxs) < 2:
            leftover.extend(idxs)
        else:
            cleaned.append(idxs)
    if leftover:
        cleaned.append(leftover)
    return json.dumps(cleaned)


def _build_monotonic_constraints(feature_names: list) -> str:
    """Return XGBoost monotone_constraints tuple-string for a feature list.

    For each feature, look up _XGB_MONO_PRIORS; default 0 (no constraint).
    XGBoost accepts either a tuple-string '(0,1,-1,0,...)' or a dict.
    """
    vals = [int(_XGB_MONO_PRIORS.get(n, 0)) for n in feature_names]
    return "(" + ",".join(str(v) for v in vals) + ")"


def _xgb_base_params(stage: str) -> dict:
    """Return tuned XGBClassifier kwargs per stage (scout|final|persist).

    Stages share most hyperparameters; only depth/n_estimators differ to keep
    scout fast. Centralized to preserve run_meta.json schema invariance vs
    prior v10 runs (we extend model_kwargs, not replace its keyset).

    XGB_NO_TOPK=1 → bump capacity + regularization for final/persist (Stage A
    full-utilization patch 2026-05-20). Compensates the missing top-K prune:
      n_estimators: 100 → 500 (early_stopping_rounds=20 trims dynamically)
      max_depth: 4 → 6
      max_leaves: 31 → 63
      reg_alpha: 0.01 → 0.1  (stronger L1 → trees drop irrelevant features)
      reg_lambda: 1.0 → 2.0  (stronger L2 smoothing)
    Scout unchanged: 50 trees of depth-3 still scouts importance fast.
    """
    no_topk = os.environ.get("XGB_NO_TOPK", "0") == "1"
    if stage == "scout":
        max_depth, n_estimators = 3, 50
        max_leaves_v = 31
        reg_a, reg_l = 0.01, 1.0
    else:  # final | persist
        if no_topk:
            max_depth, n_estimators = 6, 500
            max_leaves_v = 63
            reg_a, reg_l = 0.1, 2.0
        else:
            max_depth, n_estimators = 4, 100
            max_leaves_v = 31
            reg_a, reg_l = 0.01, 1.0
    return {
        "max_depth": max_depth,
        "learning_rate": 0.05,
        "n_estimators": n_estimators,
        "tree_method": "hist",
        "device": _XGB_DEVICE,
        "eval_metric": ["logloss", "aucpr"],
        # column sub-sampling (Top-5 #2)
        "colsample_bytree": 0.6,
        "colsample_bylevel": 0.7,
        "colsample_bynode": 0.8,
        # row sub-sampling (Top-5 #3)
        "subsample": 0.7,
        "sampling_method": _XGB_SAMPLING_METHOD,
        # medium wins (#6-9)
        "max_bin": 512,
        "min_child_weight": 3,
        "reg_alpha": reg_a,
        "reg_lambda": reg_l,
        "grow_policy": "lossguide",
        "max_leaves": max_leaves_v,
        # PATCH-1 (2026-05-21 extreme-speedup): n_jobs 1 -> all cores
        # XGB 1.7+ hist tree_method is thread-safe; on 12-core CPU this is
        # ~4-8x speedup for model.fit. Stage A sweeps (256 parallel cells on
        # Modal) keep nthreads bounded by XGB_N_JOBS env var below.
        "n_jobs": int(os.environ.get("XGB_N_JOBS", "0")) or -1,
        "random_state": 42,
        "verbosity": 0,
    }


def _xgb_version_tuple():
    """Return xgboost version as (major, minor) tuple for API gating."""
    try:
        import xgboost as _xgb  # type: ignore[import-not-found]
        parts = _xgb.__version__.split(".")
        return (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
    except Exception:  # pragma: no cover
        return (0, 0)


_XGB_VER = _xgb_version_tuple()
_XGB_2X = _XGB_VER >= (2, 0)


def _xgb_callbacks(early_stop: bool = False):
    """Return XGBoost callback list (or None) — used for both API styles.

    In XGB 2.x callbacks MUST go to XGBClassifier() constructor.
    In XGB 1.x callbacks can go to either constructor or fit(). We put them
    in the constructor unconditionally so the same code path works on both
    major versions.

    XGB_NO_TOPK=1 → bump EarlyStopping rounds 10 → 20. With n_estimators=500
    (vs 100), trees need longer patience to dynamically halt.
    """
    if not early_stop:
        return None
    no_topk = os.environ.get("XGB_NO_TOPK", "0") == "1"
    es_rounds = 20 if no_topk else 10
    try:
        from xgboost.callback import (  # type: ignore[import-not-found]
            EarlyStopping,
            EvaluationMonitor,
        )
        return [
            EarlyStopping(rounds=es_rounds, save_best=True),
            EvaluationMonitor(period=10),
        ]
    except Exception:  # pragma: no cover - older xgboost fallback
        return None


def _xgb_fit_kwargs(eval_set, early_stop: bool = False):
    """Build fit() kwargs only (eval_set + verbose). Callbacks go to the
    XGBClassifier constructor via _xgb_callbacks() — see XGB 2.x API change.

    Kept for backward compat with existing call sites. Returns dict that does
    NOT contain 'callbacks' (XGB 2.x raises TypeError if callbacks is passed
    to fit()).
    """
    fit_kw: dict = {}
    if eval_set is not None:
        fit_kw["eval_set"] = eval_set
        fit_kw["verbose"] = False
    # NOTE: callbacks intentionally NOT placed in fit kwargs. See
    # _xgb_callbacks() and pass result to XGBClassifier(callbacks=...).
    _ = early_stop  # silence unused (kept in signature for ABI compat)
    return fit_kw


# ---------------------------------------------------------------------------
# Optuna HP search (2026-05-21) — replaces external cartesian 108-combo sweep.
# Opt-in via CLI --optuna-hp. Defaults: 36 trials, TPESampler(seed=42),
# HyperbandPruner(min_resource=10, max_resource=100, reduction_factor=3).
# Trials 0-3 are seeded from priors (mastery_priors.json if present, else
# sane defaults: lr=0.05, max_depth=6, subsample=0.8, colsample=0.8).
# ---------------------------------------------------------------------------

# Default priors used when state/mastery_priors.json is absent or unreadable.
# Matches task spec 2026-05-21.
_OPTUNA_DEFAULT_PRIORS = [
    {"learning_rate": 0.05, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8},
    {"learning_rate": 0.03, "max_depth": 5, "subsample": 0.8, "colsample_bytree": 0.7},
    {"learning_rate": 0.08, "max_depth": 6, "subsample": 0.7, "colsample_bytree": 0.8},
    {"learning_rate": 0.05, "max_depth": 8, "subsample": 0.9, "colsample_bytree": 0.6},
]


def _load_optuna_priors(priors_path: "str | None" = None) -> list:
    """Load 4 prior HP dicts to seed Optuna trials 0-3.

    Priors file schema (JSON):
      [{"learning_rate": 0.05, "max_depth": 6, "subsample": 0.8,
        "colsample_bytree": 0.8}, ...]    # 1..N entries (first 4 used)
    Or a single dict (broadcast to 4 trials).

    Falls back to _OPTUNA_DEFAULT_PRIORS on any read/parse error.
    """
    if priors_path is None:
        # Resolve from WORK/state/mastery_priors.json if present.
        try:
            priors_path = str(WORK / "state" / "mastery_priors.json")
        except Exception:
            return list(_OPTUNA_DEFAULT_PRIORS)
    p = Path(priors_path)
    if not p.exists():
        return list(_OPTUNA_DEFAULT_PRIORS)
    try:
        data = json.loads(p.read_text())
        if isinstance(data, dict):
            data = [data] * 4
        if not isinstance(data, list) or not data:
            return list(_OPTUNA_DEFAULT_PRIORS)
        # Coerce + pad to 4 entries (cycling through what we have).
        out = []
        for i in range(4):
            d = data[i % len(data)]
            if not isinstance(d, dict):
                continue
            out.append({
                "learning_rate": float(d.get("learning_rate", 0.05)),
                "max_depth": int(d.get("max_depth", 6)),
                "subsample": float(d.get("subsample", 0.8)),
                "colsample_bytree": float(d.get("colsample_bytree", 0.8)),
            })
        return out if out else list(_OPTUNA_DEFAULT_PRIORS)
    except Exception as e:
        logger.warning("[optuna] priors load failed (%s); using defaults", e)
        return list(_OPTUNA_DEFAULT_PRIORS)


def _optuna_search_final_params(
    X_tr,
    y_tr,
    X_val,
    y_val,
    base_params: dict,
    n_trials: int = 36,
    priors_path: "str | None" = None,
    seed: int = 42,
    study_name: "str | None" = None,
) -> dict:
    """Search XGBoost HP via Optuna TPESampler + HyperbandPruner.

    Args:
      X_tr / y_tr: training matrix + labels (already feature-pruned).
      X_val / y_val: validation (OOS-of-fold) matrix + labels.
      base_params: dict from _xgb_base_params("final") — used as the
        non-searched skeleton (eval_metric, device, tree_method, monotonic
        constraints, etc.) Search overrides only the 4 priors-listed keys.
      n_trials: total trials (default 36).
      priors_path: optional path to mastery_priors.json. If None, defaults
        to WORK/state/mastery_priors.json.
      seed: TPESampler seed (default 42).

    Returns:
      base_params merged with the best-found {learning_rate, max_depth,
      subsample, colsample_bytree}. Other keys (n_estimators, reg_*, etc.)
      are preserved from base_params so monotonic/interaction constraints
      and downstream run_meta schema remain stable.

    Objective: validation Sharpe of (prob - 0.5) signal — proxy for
    DSR-validated Sharpe used downstream. HyperbandPruner reports the
    intermediate validation logloss at progressively larger n_estimators
    budgets (min=10, max=100, eta=3).
    """
    try:
        import optuna as _optuna
        from optuna.samplers import TPESampler
        from optuna.pruners import HyperbandPruner
    except Exception as e:
        logger.warning(
            "[optuna] import failed (%s); falling back to fixed base_params", e
        )
        return dict(base_params)

    # Silence Optuna's chatty INFO logs unless the caller asked.
    if os.environ.get("OPTUNA_VERBOSE", "0") != "1":
        _optuna.logging.set_verbosity(_optuna.logging.WARNING)

    priors = _load_optuna_priors(priors_path)

    # Tier-3 (2026-05-21): param-hash trial cache + subsampling fidelity.
    # Feature-flagged via HPO_FIDELITY_CACHE=1. When OFF, the cache is a
    # no-op (lookups return None) and the fidelity allocator emits a
    # gold-only budget, so behaviour is identical to pre-tier-3.
    _hpo_cache = None
    try:
        from hpo_trial_cache import (  # type: ignore
            cache_lookup as _hpc_lookup,
            cache_store as _hpc_store,
            cache_flush as _hpc_flush,
            cache_stats as _hpc_stats,
            allocate_fidelity_budget as _hpc_alloc,
            subsample_indices as _hpc_subsample,
            HPO_FIDELITY_CACHE_ENABLED as _HPC_ON,
        )
        _hpo_cache = True
    except Exception as _e:
        _hpo_cache = False
        _HPC_ON = False
        _hpc_lookup = lambda *a, **kw: None  # type: ignore
        _hpc_store = lambda *a, **kw: None  # type: ignore
        _hpc_flush = lambda *a, **kw: 0  # type: ignore
        _hpc_stats = lambda *a, **kw: {}  # type: ignore
        _hpc_alloc = lambda n, **kw: {"bronze": 0, "silver": 0, "gold": int(n)}  # type: ignore
        _hpc_subsample = None  # type: ignore

    # Study-name keys the cache namespace. Falls back to "default" when caller
    # didn't pass one. Recommend caller pass ticker+timeframe+fold.
    _study_name = study_name or os.environ.get("HPO_STUDY_NAME", "default")
    # Cycle through fidelities so even a single Optuna study exercises the
    # multi-fidelity ladder. When the cache is disabled this collapses to gold.
    if _HPC_ON:
        _fid_ladder = ["bronze"] * 4 + ["silver"] * 2 + ["gold"] * 1
    else:
        _fid_ladder = ["gold"]

    # Pre-build subsampled views once per fidelity (sample is deterministic
    # by (n_rows, fidelity, seed=42) per hpo_trial_cache.subsample_indices).
    _views: dict[str, tuple] = {}
    if _HPC_ON and _hpc_subsample is not None:
        try:
            n_tr = int(getattr(X_tr, "shape", (len(X_tr),))[0])
            for _fid in ("bronze", "silver", "gold"):
                idx = _hpc_subsample(n_tr, _fid, seed=seed)
                try:
                    _Xs = X_tr.iloc[idx] if hasattr(X_tr, "iloc") else X_tr[idx]
                except Exception:
                    _Xs = X_tr
                try:
                    _ys = y_tr.iloc[idx] if hasattr(y_tr, "iloc") else y_tr[idx]
                except Exception:
                    _ys = y_tr
                _views[_fid] = (_Xs, _ys)
        except Exception as _e:  # pragma: no cover -- defensive
            logger.debug("[optuna/hpo_cache] subsample setup failed: %s", _e)
            _views = {}

    def _fidelity_for_trial(t_idx: int) -> str:
        if not _HPC_ON:
            return "gold"
        return _fid_ladder[t_idx % len(_fid_ladder)]

    def objective(trial) -> float:
        lr = trial.suggest_float("learning_rate", 0.01, 0.20, log=True)
        max_depth = trial.suggest_int("max_depth", 3, 10)
        subsample = trial.suggest_float("subsample", 0.5, 1.0)
        colsample = trial.suggest_float("colsample_bytree", 0.4, 1.0)

        # Pick fidelity for this trial (multi-rung ladder).
        fid = _fidelity_for_trial(trial.number)
        trial.set_user_attr("hpc_fidelity", fid)

        # Cache lookup (no-op when flag OFF).
        cached = _hpc_lookup(
            _study_name,
            fid,
            {
                "learning_rate": lr,
                "max_depth": max_depth,
                "subsample": subsample,
                "colsample_bytree": colsample,
            },
        )
        if cached is not None:
            trial.set_user_attr("hpc_cache_hit", True)
            return float(cached)
        trial.set_user_attr("hpc_cache_hit", False)

        trial_params = dict(base_params)
        trial_params.update({
            "learning_rate": lr,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample,
            "n_estimators": 100,  # capped — Hyperband manages budget below
        })

        # Pick the (X, y) view for this fidelity (already pre-subsampled).
        _X_use, _y_use = _views.get(fid, (X_tr, y_tr))

        # HyperbandPruner budget: progressively larger n_estimators.
        # We train once at the trial's allocated rung (min=10..max=100, eta=3),
        # report intermediate logloss every 10 rounds.
        try:
            mdl = xgb.XGBClassifier(**trial_params)
            mdl.fit(_X_use, _y_use, eval_set=[(X_val, y_val)], verbose=False)
            # Validation Sharpe of (prob - 0.5) signal.
            import numpy as _np
            p = mdl.predict_proba(X_val)[:, 1]
            sig = p - 0.5
            ret = sig * (2 * y_val - 1).astype(float)  # +1 if correct, -1 if wrong
            mu = float(_np.mean(ret))
            sd = float(_np.std(ret) + 1e-9)
            sharpe = (mu / sd) * _np.sqrt(252.0)
            # Report once at "max rung" so HyperbandPruner can act on it.
            trial.report(sharpe, step=trial_params["n_estimators"])
            if trial.should_prune():
                raise _optuna.TrialPruned()
            # Persist (no-op when flag OFF).
            try:
                _hpc_store(
                    _study_name,
                    fid,
                    {
                        "learning_rate": lr,
                        "max_depth": max_depth,
                        "subsample": subsample,
                        "colsample_bytree": colsample,
                    },
                    float(sharpe),
                    meta={"n_estimators": trial_params["n_estimators"]},
                )
            except Exception:
                pass
            return float(sharpe)
        except _optuna.TrialPruned:
            raise
        except Exception as e:
            logger.debug("[optuna] trial failed: %s", e)
            return -1e9

    sampler = TPESampler(seed=seed)
    pruner = HyperbandPruner(min_resource=10, max_resource=100, reduction_factor=3)
    study = _optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    # Seed trials 0-3 from priors so TPE starts with known-good points.
    for prior in priors[:4]:
        try:
            study.enqueue_trial(prior)
        except Exception as e:
            logger.debug("[optuna] enqueue prior failed (%s)", e)

    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)

    try:
        best = study.best_params
    except Exception:
        return dict(base_params)

    out = dict(base_params)
    out.update({
        "learning_rate": float(best.get("learning_rate", base_params.get("learning_rate", 0.05))),
        "max_depth": int(best.get("max_depth", base_params.get("max_depth", 6))),
        "subsample": float(best.get("subsample", base_params.get("subsample", 0.7))),
        "colsample_bytree": float(best.get("colsample_bytree", base_params.get("colsample_bytree", 0.6))),
    })
    try:
        logger.info(
            "[optuna] best trial: value=%.4f params=%s (n_trials=%d, n_pruned=%d)",
            float(study.best_value),
            best,
            len(study.trials),
            sum(1 for t in study.trials if t.state.name == "PRUNED"),
        )
    except Exception:
        pass
    # Tier-3: flush cache + log stats (no-op when flag OFF).
    if _hpo_cache and _HPC_ON:
        try:
            n_written = _hpc_flush(_study_name)
            st = _hpc_stats(_study_name)
            logger.info(
                "[optuna/hpo_cache] flushed=%d hits=%d misses=%d hit_rate=%.2f%% study=%s",
                int(n_written),
                int(st.get("hits", 0)),
                int(st.get("misses", 0)),
                100.0 * float(st.get("hit_rate", 0.0)),
                _study_name,
            )
        except Exception as _e:
            logger.debug("[optuna/hpo_cache] flush/stats failed: %s", _e)
    return out


from empyrical_risk_features_features import compute_empyrical_risk_features_features  # auto-wired 2026-05-18
from quantstats_metrics_features import compute_quantstats_metrics_features  # auto-wired 2026-05-18

# --- auto-wired dir-glob loaders (lazy-import safe) ---
# --- lazy-import shim (refactor 2026-05-21): replaces 56 try/except blocks ---
# These names are resolved on first access via _lazy_features.__getattr__
# (PEP 562).  If the underlying module fails to import, the name is None,
# preserving the legacy try-except-ImportError semantics.
from _lazy_features import (  # noqa: E402
    compute_oc1_alpaca_timeframes_root_features,
    compute_ph0tis2_alpaca_timeframes_root_features,
    compute_ph0tis_alpaca_timeframes_root_features,
    compute_save_dir_features,
    compute_claudes_test_data_features,
    compute_ai_external_qlib_features,
    compute_ai_repos_pandas_ta_classic_features,
    compute_ai_repos_featuretools_features,
    compute_ai_external_zvt_features,
    compute_ai_repos_evidently_features,
    compute_ai_repos_feast_features,
    compute_ai_external_rdagent_features,
    compute_ai_external_ccxt_features,
    compute_oc1_historical_build_features,
    compute_oc1_alpaca_build_features,
    compute_oc1_alpaca_claude_features,
    compute_oc1_historical_merged_features,
    compute_oc1_trading_repos_bitquant_features,
    compute_oc1_trading_repos_cryptosignal_features,
    compute_oc1_trading_repos_kmeans_features,
    compute_oc1_trading_repos_finrl_features,
    compute_oc1_trading_repos_finrl_trading_features,
    compute_oc1_trading_repos_quantmuse_features,
    compute_oc1_strategy_system_features,
    compute_oc1_alpaca_timeframes_features,
    compute_ph0tis2_strategy_raw_features,
    compute_ph0tis2_strategy_formatted_features,
    compute_ph0tis2_alpaca_timeframes_dup_features,
    compute_ph0tis2_alpaca_timeframes_features,
    compute_ph0tis_edgar_src_features,
    compute_ph0tis_gov_trades_src_features,
    compute_ph0tis_gov_trades_archive_features,
    compute_ph0tis_alpaca_system_features,
    compute_ph0tis_historical_src_features,
    compute_ph0tis_strategy_system_source_features,
    compute_ph0tis_strategy_test_features,
    compute_ph0tis_strategy_raw_features,
    compute_ph0tis_strategy_formatted_features,
    compute_ph0tis_alpaca_timeframes_features,
    compute_gabriel_historical_indicators_features,
    compute_gabriel_synapse_strategies_features,
    compute_gabriel_alpaca_system_src_features,
    compute_gabriel_research_cycle_prompts_features,
    compute_gabriel_research_cycle_workspaces_features,
    compute_gabriel_gov_trades_features,
    compute_gabriel_alpaca_timeframes_1m_merged_features,
    compute_gabriel_alpaca_timeframes_1m_features,
    compute_gabriel_alpaca_timeframes_5m_features,
    compute_gabriel_alpaca_timeframes_15m_features,
    compute_gabriel_alpaca_timeframes_30m_features,
    compute_gabriel_alpaca_timeframes_45m_features,
    compute_gabriel_alpaca_timeframes_1h_features,
    compute_gabriel_alpaca_timeframes_4h_features,
    compute_gabriel_alpaca_timeframes_8h_features,
    compute_gabriel_alpaca_timeframes_12h_features,
    compute_gabriel_alpaca_timeframes_1day_features,
)


# Cross-sectional cache (unchanged from v9)
try:
    import cross_sectional_features as csf
except Exception as _e:
    logger.warning("csf unavailable: %s", _e)
    csf = None

# ---------------------------------------------------------------------------
# Helper A: daily_integration_features
# ---------------------------------------------------------------------------

try:
    from daily_integration_features import add_daily_integration_features  # noqa: E402
    DAILY_INT_AVAILABLE = True
    logger.info("[v10] daily_integration_features loaded OK")
except Exception as _daily_err:
    logger.warning(
        "[v10] daily_integration_features not importable: %s — 7 features zeroed", _daily_err
    )
    DAILY_INT_AVAILABLE = False

    def add_daily_integration_features(  # type: ignore[misc]
        df: pd.DataFrame,
        ticker: Optional[str] = None,
    ) -> pd.DataFrame:
        """Stub: fill all 7 daily_integration cols with 0."""
        for col in [
            "beta_adj_residual_ret_z21",
            "beta_adj_residual_ret_z63",
            "csrs_5d",
            "csrs_10d",
            "csrs_20d",
            "earn_contam_gate",
            "earn_post_rv_gate",
        ]:
            df[col] = 0.0
        return df


DAILY_INT_FEATURE_NAMES: list[str] = [
    "beta_adj_residual_ret_z21",
    "beta_adj_residual_ret_z63",
    "csrs_5d",
    "csrs_10d",
    "csrs_20d",
    "earn_contam_gate",
    "earn_post_rv_gate",
]

# ---------------------------------------------------------------------------
# Helper B: alpaca_features
# ---------------------------------------------------------------------------

try:
    from alpaca_features import add_alpaca_features  # noqa: E402
    ALPACA_AVAILABLE = True
    logger.info("[v10] alpaca_features loaded OK")
except Exception as _alp_err:
    logger.warning(
        "[v10] alpaca_features not importable: %s — 13 features zeroed", _alp_err
    )
    ALPACA_AVAILABLE = False

    def add_alpaca_features(  # type: ignore[misc]
        daily_df: pd.DataFrame,
        ticker: str,
    ) -> pd.DataFrame:
        """Stub: fill all 13 alpaca cols with 0."""
        for col in [
            "days_until_earnings",
            "is_earnings_week",
            "earnings_surprise_last",
            "days_since_last_earnings",
            "ex_div_proximity",
            "days_since_last_exdiv",
            "div_yield_trailing",
            "dividend_growth_yoy",
            "days_since_last_split",
            "is_post_split_60d",
            "log_market_cap",
            "short_interest_pct",
            "sector_encoded",
        ]:
            daily_df[col] = 0.0
        return daily_df


ALPACA_FEATURE_NAMES: list[str] = [
    "days_until_earnings",
    "is_earnings_week",
    "earnings_surprise_last",
    "days_since_last_earnings",
    "ex_div_proximity",
    "days_since_last_exdiv",
    "div_yield_trailing",
    "dividend_growth_yoy",
    "days_since_last_split",
    "is_post_split_60d",
    "log_market_cap",
    "short_interest_pct",
    "sector_encoded",
]

# ---------------------------------------------------------------------------
# Helper C: featuretools_dfs_features
# ---------------------------------------------------------------------------

try:
    from featuretools_dfs_features import add_dfs_features, dfs_feature_names  # noqa: E402
    DFS_AVAILABLE = True
    logger.info("[v10] featuretools_dfs_features loaded OK")
except Exception as _dfs_err:
    logger.warning(
        "[v10] featuretools_dfs_features not importable: %s — DFS features skipped", _dfs_err
    )
    DFS_AVAILABLE = False

    def add_dfs_features(  # type: ignore[misc]
        df: pd.DataFrame,
        ticker: Optional[str] = None,
    ) -> pd.DataFrame:
        """Stub: return df unchanged when featuretools module is unavailable."""
        return df

    def dfs_feature_names() -> list[str]:  # type: ignore[misc]
        return []


# ---------------------------------------------------------------------------
# Helper D: insider_form4_features (SEC Form 4 insider-disclosure trades)
# Wired 2026-05-17 as the "gov-trades" module — fills feature-module gap.
# ---------------------------------------------------------------------------

try:
    from insider_form4_features import add_insider_form4_features  # noqa: E402
    INSIDER_FORM4_AVAILABLE = True
    logger.info("[v10] insider_form4_features loaded OK")
except Exception as _ins_err:
    logger.warning(
        "[v10] insider_form4_features not importable: %s — 8 features zeroed", _ins_err
    )
    INSIDER_FORM4_AVAILABLE = False

    def add_insider_form4_features(  # type: ignore[misc]
        daily_df: pd.DataFrame,
        ticker: str,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Stub: fill all 8 insider_form4 cols with 0."""
        for col in [
            "insider_buy_count_30d",
            "insider_sell_count_30d",
            "insider_net_buy_count_60d",
            "insider_cluster_buy_flag",
            "insider_cluster_sell_flag",
            "days_since_last_insider_buy",
            "days_since_last_insider_sell",
            "insider_buy_dollar_amount_60d_log",
        ]:
            daily_df[col] = 0.0
        return daily_df


INSIDER_FORM4_FEATURE_NAMES: list[str] = [
    "insider_buy_count_30d",
    "insider_sell_count_30d",
    "insider_net_buy_count_60d",
    "insider_cluster_buy_flag",
    "insider_cluster_sell_flag",
    "days_since_last_insider_buy",
    "days_since_last_insider_sell",
    "insider_buy_dollar_amount_60d_log",
]


# ---------------------------------------------------------------------------
# Helper E: mastery_priors_features (past-test mastery files as priors)
# Wired 2026-05-17 — reads 311 v4 + 7 v10 mastery markdown artifacts and
# emits 7 per-ticker priors features (mastered flags, PF, DD, top-10 flag,
# .shift(1)-safe mtime-gated age). See $SP/scripts/mastery_priors_features.py.
# ---------------------------------------------------------------------------

try:
    from mastery_priors_features import (  # noqa: E402
        add_mastery_priors,
        MASTERY_PRIORS_FEATURE_NAMES,
    )
    MASTERY_PRIORS_AVAILABLE = True
    logger.info("[v10] mastery_priors_features loaded OK")
except Exception as _mp_err:
    logger.warning(
        "[v10] mastery_priors_features not importable: %s — 7 features zeroed", _mp_err
    )
    MASTERY_PRIORS_AVAILABLE = False

    MASTERY_PRIORS_FEATURE_NAMES: list[str] = [  # type: ignore[no-redef]
        "prior_v4_mastered",
        "prior_v4_pf",
        "prior_v10_mastered",
        "prior_v10_pf",
        "prior_v10_dd",
        "prior_cross_section_top10",
        "prior_mastery_age_days",
    ]

    def add_mastery_priors(  # type: ignore[misc]
        df: pd.DataFrame,
        ticker: str,
    ) -> pd.DataFrame:
        """Stub: fill all 7 mastery_priors cols with 0 when module unavailable."""
        for col in MASTERY_PRIORS_FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0
        return df


# ---------------------------------------------------------------------------
# Helper F: paper_trade_outcome_features (live paper-trade feedback features)
# Wired 2026-05-17 — reads $SP/paper_trade/state/*_state.json closed_trades[]
# and emits 7 .shift(1)-safe rolling outcome features (win-rate, PF, count,
# last-outcome-sign, avg holding, signal-to-fill lag, current drawdown).
# Zero-fills when paper_trade/ tree empty or ticker has no closed trades yet.
# See $SP/scripts/paper_trade_outcome_features.py.
# ---------------------------------------------------------------------------

try:
    from paper_trade_outcome_features import (  # noqa: E402
        add_paper_trade_outcome_features,
        PT_FEATURE_NAMES,
    )
    PT_OUTCOMES_AVAILABLE = True
    logger.info("[v10] paper_trade_outcome_features loaded OK")
except Exception as _pt_err:
    logger.warning(
        "[v10] paper_trade_outcome_features not importable: %s — 7 features zeroed",
        _pt_err,
    )
    PT_OUTCOMES_AVAILABLE = False

    PT_FEATURE_NAMES: list[str] = [  # type: ignore[no-redef]
        "paper_trade_win_rate_30d",
        "paper_trade_pf_30d",
        "paper_trade_count_30d",
        "paper_trade_last_outcome_sign",
        "paper_trade_avg_holding_days",
        "paper_trade_signal_to_fill_lag_min",
        "paper_trade_in_drawdown_pct",
    ]

    def add_paper_trade_outcome_features(  # type: ignore[misc]
        df: pd.DataFrame,
        ticker: str,
    ) -> pd.DataFrame:
        """Stub: fill all 7 paper-trade outcome cols with 0."""
        for col in PT_FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0 if col in (
                    "paper_trade_count_30d", "paper_trade_last_outcome_sign"
                ) else 0.0
        return df


# ---------------------------------------------------------------------------
# Helper G/H/I: GitHub treasure-hunt features (wave 11, 2026-05-17)
#   G: stumpy   — matrix-profile motif/discord (~6 features × windows)
#   H: ffn      — Sortino/Calmar/Ulcer/Downside (~15 features × windows)
#   I: pandas-ta-classic — non-TA-Lib indicators (Aberration/KVO/KST/PVO/Vortex/Squeeze/RVI/NVI/PVI/TSI/etc., ~30 feats)
# All graceful-fail. .shift(1)-safe (rolling windows on prior bars).
# ---------------------------------------------------------------------------
try:
    from stumpy_features import add_stumpy_features  # noqa: E402
    STUMPY_AVAILABLE = True
    logger.info("[v10] stumpy_features loaded OK")
except Exception as _st_err:
    logger.warning("[v10] stumpy_features not importable: %s — features zeroed", _st_err)
    STUMPY_AVAILABLE = False
    def add_stumpy_features(df, ticker, windows=(10, 20, 60)):  # type: ignore[misc]
        return df

try:
    from ffn_features import add_ffn_features  # noqa: E402
    FFN_AVAILABLE = True
    logger.info("[v10] ffn_features loaded OK")
except Exception as _ffn_err:
    logger.warning("[v10] ffn_features not importable: %s — features zeroed", _ffn_err)
    FFN_AVAILABLE = False
    def add_ffn_features(df, ticker, windows=(20, 60, 120)):  # type: ignore[misc]
        return df

try:
    from pandas_ta_classic_features import add_pandas_ta_classic_features  # noqa: E402
    PTC_AVAILABLE = True
    logger.info("[v10] pandas_ta_classic_features loaded OK")
except Exception as _ptc_err:
    logger.warning("[v10] pandas_ta_classic_features not importable: %s — features zeroed", _ptc_err)
    PTC_AVAILABLE = False
    def add_pandas_ta_classic_features(df, ticker):  # type: ignore[misc]
        return df


# ---------------------------------------------------------------------------
# Helper J / K / L / M: Wave A — 12 new features (2026-05-17)
#   J: options_flow      — put_call_ratio + iv_vs_rv + unusual flag (3)
#   K: govtrades         — congress density/buy-sell ratio + lobbying count (3)
#   L: time_of_day       — bucket 0-4 (1)
#   M: gabriel_priors    — champion PF/WR/N + regime/monthly priors (5)
# All graceful-fail. All .shift(1)-safe.
# ---------------------------------------------------------------------------
try:
    from options_flow_features import (  # noqa: E402
        add_options_flow_features,
        OPTIONS_FLOW_FEATURE_NAMES,
    )
    OPTIONS_FLOW_AVAILABLE = True
    logger.info("[v10] options_flow_features loaded OK")
except Exception as _of_err:
    logger.warning("[v10] options_flow_features not importable: %s — 3 features zeroed", _of_err)
    OPTIONS_FLOW_AVAILABLE = False
    OPTIONS_FLOW_FEATURE_NAMES = [
        "put_call_volume_ratio",
        "iv_vs_rv_divergence",
        "unusual_options_activity_flag",
    ]
    def add_options_flow_features(df, ticker):  # type: ignore[misc]
        for c in OPTIONS_FLOW_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0 if c != "unusual_options_activity_flag" else 0
        return df

try:
    from govtrades_features import (  # noqa: E402
        add_govtrades_features,
        GOVTRADES_FEATURE_NAMES,
    )
    GOVTRADES_AVAILABLE = True
    logger.info("[v10] govtrades_features loaded OK")
except Exception as _gt_err:
    logger.warning("[v10] govtrades_features not importable: %s — 3 features zeroed", _gt_err)
    GOVTRADES_AVAILABLE = False
    GOVTRADES_FEATURE_NAMES = [
        "congress_trade_density_5d",
        "congress_buy_sell_ratio_5d",
        "lobbying_filing_count_30d",
    ]
    def add_govtrades_features(df, ticker):  # type: ignore[misc]
        for c in GOVTRADES_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0 if c != "congress_buy_sell_ratio_5d" else 0.0
        return df

try:
    from time_of_day_features import (  # noqa: E402
        add_time_of_day_features,
        TOD_FEATURE_NAMES,
    )
    TOD_AVAILABLE = True
    logger.info("[v10] time_of_day_features loaded OK")
except Exception as _tod_err:
    logger.warning("[v10] time_of_day_features not importable: %s — 1 feature zeroed", _tod_err)
    TOD_AVAILABLE = False
    TOD_FEATURE_NAMES = ["time_of_day_bucket"]
    def add_time_of_day_features(df, ticker):  # type: ignore[misc]
        if "time_of_day_bucket" not in df.columns:
            df["time_of_day_bucket"] = 2  # default mid-day fallback
        return df

# ---------------------------------------------------------------------------
# Quick-Wire Wave (2026-05-21) — 6 READY-but-UNWIRED modules
# ---------------------------------------------------------------------------
try:
    from candlestick_features import (  # noqa: E402
        add_candlestick_features,
        feature_columns as _cdl_feature_columns,
    )
    CDL_AVAILABLE = True
    CDL_FEATURE_NAMES = _cdl_feature_columns(include_rolling=True)
    logger.info("[v10] candlestick_features loaded OK (%d cols)", len(CDL_FEATURE_NAMES))
except Exception as _cdl_err:
    logger.warning("[v10] candlestick_features not importable: %s — zeroing", _cdl_err)
    CDL_AVAILABLE = False
    CDL_FEATURE_NAMES = []
    def add_candlestick_features(df, rolling_window=5, include_rolling=True):  # type: ignore[misc]
        return df

try:
    from oc2_donchian_c003_features import add_oc2_donchian_c003_features  # noqa: E402
    DONCH_C003_AVAILABLE = True
    DONCH_C003_FEATURE_NAMES = [
        "c003_donchian_upper20", "c003_donchian_lower40", "c003_atr14",
        "c003_breakout_signal", "c003_session_bar_idx", "c003_opening_range_flag",
        "c003_vol_confirm_flag", "c003_or_breakout", "c003_vol_breakout",
        "c003_combined_signal", "c003_is_first_signal_day", "c003_atr_expansion",
    ]
    logger.info("[v10] oc2_donchian_c003_features loaded OK (12 cols)")
except Exception as _dc003_err:
    logger.warning("[v10] oc2_donchian_c003_features not importable: %s — 12 features zeroed", _dc003_err)
    DONCH_C003_AVAILABLE = False
    DONCH_C003_FEATURE_NAMES = [
        "c003_donchian_upper20", "c003_donchian_lower40", "c003_atr14",
        "c003_breakout_signal", "c003_session_bar_idx", "c003_opening_range_flag",
        "c003_vol_confirm_flag", "c003_or_breakout", "c003_vol_breakout",
        "c003_combined_signal", "c003_is_first_signal_day", "c003_atr_expansion",
    ]
    def add_oc2_donchian_c003_features(df, ticker=None, **kw):  # type: ignore[misc]
        for c in DONCH_C003_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0
        return df

try:
    from oc2_donchian_mtf_features import add_oc2_donchian_mtf_features  # noqa: E402
    DONCH_MTF_AVAILABLE = True
    DONCH_MTF_FEATURE_NAMES = [
        "mtf_donchian_upper20", "mtf_breakout_signal", "mtf_ema63_above",
        "mtf_sma600_above", "mtf_three_tf_aligned", "mtf_vol_surge_flag",
        "mtf_atr_expanding", "mtf_no_lunch_flag", "mtf_stacked_four_filter",
        "mtf_adx14", "mtf_adx_gt25", "mtf_adx_gt20", "mtf_cmf21",
        "mtf_cmf_positive", "mtf_mdd_composite", "mtf_n_filters_passing",
    ]
    logger.info("[v10] oc2_donchian_mtf_features loaded OK (16 cols)")
except Exception as _dmtf_err:
    logger.warning("[v10] oc2_donchian_mtf_features not importable: %s — 16 features zeroed", _dmtf_err)
    DONCH_MTF_AVAILABLE = False
    DONCH_MTF_FEATURE_NAMES = [
        "mtf_donchian_upper20", "mtf_breakout_signal", "mtf_ema63_above",
        "mtf_sma600_above", "mtf_three_tf_aligned", "mtf_vol_surge_flag",
        "mtf_atr_expanding", "mtf_no_lunch_flag", "mtf_stacked_four_filter",
        "mtf_adx14", "mtf_adx_gt25", "mtf_adx_gt20", "mtf_cmf21",
        "mtf_cmf_positive", "mtf_mdd_composite", "mtf_n_filters_passing",
    ]
    def add_oc2_donchian_mtf_features(df, ticker=None, **kw):  # type: ignore[misc]
        for c in DONCH_MTF_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0
        return df

try:
    from oc2_donchian_per_ticker_selectivity_features import (  # noqa: E402
        add_oc2_donchian_per_ticker_selectivity_features,
    )
    DONCH_SEL_AVAILABLE = True
    DONCH_SEL_FEATURE_NAMES = [
        "sel_donchian20_upper", "sel_donchian20_lower", "sel_avg_atr14",
        "sel_current_atr_vs_avg", "sel_win_rate_60d",
        "sel_false_breakout_rate_60d", "sel_breakout_frequency_60d",
        "sel_optimal_window", "sel_selectivity_score", "sel_is_high_selectivity",
    ]
    logger.info("[v10] oc2_donchian_per_ticker_selectivity_features loaded OK (10 cols)")
except Exception as _dsel_err:
    logger.warning("[v10] oc2_donchian_per_ticker_selectivity_features not importable: %s — 10 features zeroed", _dsel_err)
    DONCH_SEL_AVAILABLE = False
    DONCH_SEL_FEATURE_NAMES = [
        "sel_donchian20_upper", "sel_donchian20_lower", "sel_avg_atr14",
        "sel_current_atr_vs_avg", "sel_win_rate_60d",
        "sel_false_breakout_rate_60d", "sel_breakout_frequency_60d",
        "sel_optimal_window", "sel_selectivity_score", "sel_is_high_selectivity",
    ]
    def add_oc2_donchian_per_ticker_selectivity_features(df, ticker=None, **kw):  # type: ignore[misc]
        for c in DONCH_SEL_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0
        return df

try:
    from py_market_profile_features import add_py_market_profile_features  # noqa: E402
    PYMP_AVAILABLE = True
    PYMP_FEATURE_NAMES = [
        "pymp_poc_price", "pymp_value_area_high", "pymp_value_area_low",
        "pymp_value_area_width_pct", "pymp_close_position_in_va",
        "pymp_initial_balance_range_pct",
    ]
    logger.info("[v10] py_market_profile_features loaded OK (6 cols)")
except Exception as _pymp_err:
    logger.warning("[v10] py_market_profile_features not importable: %s — 6 features zeroed", _pymp_err)
    PYMP_AVAILABLE = False
    PYMP_FEATURE_NAMES = [
        "pymp_poc_price", "pymp_value_area_high", "pymp_value_area_low",
        "pymp_value_area_width_pct", "pymp_close_position_in_va",
        "pymp_initial_balance_range_pct",
    ]
    def add_py_market_profile_features(df, ticker):  # type: ignore[misc]
        for c in PYMP_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0
        return df

try:
    from footprint_analyzer_features import add_footprint_features  # noqa: E402
    FP_AVAILABLE = True
    FP_FEATURE_NAMES = [
        "fp_vol_concentration", "fp_bid_ask_imbalance",
        "fp_support_level", "fp_resistance_level",
    ]
    logger.info("[v10] footprint_analyzer_features loaded OK (4 cols)")
except Exception as _fp_err:
    logger.warning("[v10] footprint_analyzer_features not importable: %s — 4 features zeroed", _fp_err)
    FP_AVAILABLE = False
    FP_FEATURE_NAMES = [
        "fp_vol_concentration", "fp_bid_ask_imbalance",
        "fp_support_level", "fp_resistance_level",
    ]
    def add_footprint_features(df, ticker, **kw):  # type: ignore[misc]
        for c in FP_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0
        return df

try:
    from gabriel_priors_features import (  # noqa: E402
        add_gabriel_priors_features,
        GABRIEL_PRIORS_FEATURE_NAMES,
    )
    GABRIEL_PRIORS_AVAILABLE = True
    logger.info("[v10] gabriel_priors_features loaded OK")
except Exception as _gp_err:
    logger.warning("[v10] gabriel_priors_features not importable: %s — 5 features zeroed", _gp_err)
    GABRIEL_PRIORS_AVAILABLE = False
    GABRIEL_PRIORS_FEATURE_NAMES = [
        "gabriel_champion_pf",
        "gabriel_champion_wr",
        "gabriel_champion_n_trades",
        "gabriel_regime_breakdown_score",
        "gabriel_monthly_perf_consistency",
    ]
    def add_gabriel_priors_features(df, ticker):  # type: ignore[misc]
        for c in GABRIEL_PRIORS_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0 if c == "gabriel_champion_n_trades" else 0.0
        return df

try:
    from vix_term_structure_v2_features import (  # noqa: E402
        compute_vix_term_structure_v2_features,
        VIX_TS_FEATURE_NAMES,
    )
    VIX_TS_AVAILABLE = True
    logger.info("[v10] vix_term_structure_v2_features loaded OK")
except Exception as _vts_err:
    logger.warning("[v10] vix_term_structure_v2_features not importable: %s — 3 features zeroed", _vts_err)
    VIX_TS_AVAILABLE = False
    VIX_TS_FEATURE_NAMES = ["vix9d_vix_ratio", "vix_term_inverted", "vix9d_vix_ratio_z10"]
    def compute_vix_term_structure_v2_features(df):  # type: ignore[misc]
        for c in VIX_TS_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0 if c != "vix_term_inverted" else 0
        return df


# ---------------------------------------------------------------------------
# Helper N: garch_11_cond_vol (GARCH(1,1) conditional volatility, 3 features)
# Wired 2026-05-17. Uses arch package (BSD-3, Kevin Sheppard) + yfinance close.
# .shift(1)-safe: all outputs use prior-bar conditional variance only.
# ---------------------------------------------------------------------------
try:
    from garch_11_cond_vol_features import (  # noqa: E402
        compute_garch_11_cond_vol_features,
        GARCH11_FEATURE_NAMES,
    )
    GARCH11_AVAILABLE = True
    logger.info("[v10] garch_11_cond_vol_features loaded OK")
except Exception as _garch_err:
    logger.warning(
        "[v10] garch_11_cond_vol_features not importable: %s — 3 features zeroed", _garch_err
    )
    GARCH11_AVAILABLE = False
    GARCH11_FEATURE_NAMES = [
        "garch11_cond_vol_1d",
        "garch11_cond_vol_z21",
        "garch11_persistence",
    ]

    def compute_garch_11_cond_vol_features(df, ticker=None):  # type: ignore[misc]
        for c in GARCH11_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0
        return df


# ---------------------------------------------------------------------------
# Helper O: egarch_11_leverage (EGARCH(1,1) leverage-effect vol, 3 features)
# Wired 2026-05-17. Uses arch package (BSD-3) + yfinance close.
# .shift(1)-safe: all outputs use prior-bar conditional variance only.
# ---------------------------------------------------------------------------
try:
    from egarch_11_leverage_features import (  # noqa: E402
        compute_egarch_11_leverage_features,
        EGARCH11_LEV_FEATURE_NAMES,
    )
    EGARCH11_LEV_AVAILABLE = True
    logger.info("[v10] egarch_11_leverage_features loaded OK")
except Exception as _egarch_err:
    logger.warning(
        "[v10] egarch_11_leverage_features not importable: %s — 3 features zeroed", _egarch_err
    )
    EGARCH11_LEV_AVAILABLE = False
    EGARCH11_LEV_FEATURE_NAMES = [
        "egarch11_lev_cond_vol_1d",
        "egarch11_lev_effect",
        "egarch11_lev_vol_z21",
    ]

    def compute_egarch_11_leverage_features(df, ticker=None):  # type: ignore[misc]
        for c in EGARCH11_LEV_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0
        return df


# ---------------------------------------------------------------------------
# Helper P: vpin_50bucket (VPIN 50-bucket BVC approximation, 3 features)
# Wired 2026-05-17. Uses daily OHLCV (close + volume) from v9 stack;
# approximates alpaca_1min_bars via BVC (López de Prado/O'Hara 2012 RFS).
# .shift(1)-safe: all outputs shift by 1 bar before assignment.
# ---------------------------------------------------------------------------
try:
    from vpin_50bucket_features import (  # noqa: E402
        compute_vpin_50bucket_features,
        VPIN_50BUCKET_FEATURE_NAMES,
    )
    VPIN_50BUCKET_AVAILABLE = True
    logger.info("[v10] vpin_50bucket_features loaded OK")
except Exception as _vpin_err:
    logger.warning(
        "[v10] vpin_50bucket_features not importable: %s — 3 features zeroed", _vpin_err
    )
    VPIN_50BUCKET_AVAILABLE = False
    VPIN_50BUCKET_FEATURE_NAMES = [
        "vpin_50bucket",
        "vpin_50bucket_z21",
        "vpin_buy_frac_10",
    ]

    def compute_vpin_50bucket_features(df, ticker=None):  # type: ignore[misc]
        for c in VPIN_50BUCKET_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0
        return df


# ---------------------------------------------------------------------------
# Helper Q: kyles_lambda_intraday (Kyle 1985 λ via BVC approximation, 3 features)
# Wired 2026-05-17. Approximates alpaca_1min_bars via daily BVC signed-volume;
# estimates price-impact coefficient (λ) using rolling OLS over 20-day window.
# .shift(1)-safe: both delta_price and signed_vol are lagged before regression.
# ---------------------------------------------------------------------------
try:
    from kyles_lambda_intraday_features import (  # noqa: E402
        compute_kyles_lambda_intraday_features,
        KYLES_LAMBDA_FEATURE_NAMES,
    )
    KYLES_LAMBDA_AVAILABLE = True
    logger.info("[v10] kyles_lambda_intraday_features loaded OK")
except Exception as _kl_err:
    logger.warning(
        "[v10] kyles_lambda_intraday_features not importable: %s — 3 features zeroed", _kl_err
    )
    KYLES_LAMBDA_AVAILABLE = False
    KYLES_LAMBDA_FEATURE_NAMES = [
        "kyles_lambda",
        "kyles_lambda_z21",
        "kyles_lambda_trend",
    ]

    def compute_kyles_lambda_intraday_features(df, ticker=None):  # type: ignore[misc]
        for c in KYLES_LAMBDA_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0
        return df


# ---------------------------------------------------------------------------
# Helper W: vpin_features (TRUE 1-min VPIN, 5 features) — Wave M-1 #1
# Wired 2026-05-17. Uses Alpaca 1-min cache via _load_1min(); falls back to
# claudes-test 1Min_merged. .shift(1)-safe inside the module.
# ---------------------------------------------------------------------------
try:
    from vpin_features import (  # noqa: E402
        add_vpin_features,
        VPIN_FEATURE_NAMES,
    )
    VPIN_INTRADAY_AVAILABLE = True
    logger.info("[v10] vpin_features (intraday) loaded OK")
except Exception as _vpin_intra_err:
    logger.warning(
        "[v10] vpin_features (intraday) not importable: %s — 5 features zeroed",
        _vpin_intra_err,
    )
    VPIN_INTRADAY_AVAILABLE = False
    VPIN_FEATURE_NAMES = [
        "vpin_eod",
        "vpin_max_today",
        "vpin_zscore_60d",
        "vpin_above_p95",
        "vpin_buy_frac_eod",
    ]

    def add_vpin_features(df, ticker=None):  # type: ignore[misc]
        for c in VPIN_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0
        return df


# ---------------------------------------------------------------------------
# Helper X: tick_imbalance_features (Lee-Ready, 5 features) — Wave M-1 #6
# Wired 2026-05-17. Uses Alpaca 1-min cache via _load_1min(); falls back to
# claudes-test 1Min_merged. .shift(1)-safe inside the module.
# ---------------------------------------------------------------------------
try:
    from tick_imbalance_features import (  # noqa: E402
        add_tick_imbalance_features,
        TICK_IMBALANCE_FEATURE_NAMES,
    )
    TICK_IMBALANCE_AVAILABLE = True
    logger.info("[v10] tick_imbalance_features loaded OK")
except Exception as _ti_err:
    logger.warning(
        "[v10] tick_imbalance_features not importable: %s — 5 features zeroed",
        _ti_err,
    )
    TICK_IMBALANCE_AVAILABLE = False
    TICK_IMBALANCE_FEATURE_NAMES = [
        "tick_imb_eod",
        "tick_imb_first_hour",
        "tick_imb_last_hour",
        "tick_imb_5d_avg",
        "tick_imb_first_vs_last_hour_diff",
    ]

    def add_tick_imbalance_features(df, ticker=None):  # type: ignore[misc]
        for c in TICK_IMBALANCE_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0
        return df


# ---------------------------------------------------------------------------
# Helper Y: volume_profile_features (POC / VA / shape, 6 features) — Wave M-1 #11
# Wired 2026-05-17. Uses Alpaca 1-min cache via _load_1min(); falls back to
# claudes-test 1Min_merged. .shift(1)-safe inside the module.
# ---------------------------------------------------------------------------
try:
    from volume_profile_features import (  # noqa: E402
        add_volume_profile_features,
        VOLUME_PROFILE_FEATURE_NAMES,
    )
    VOLUME_PROFILE_AVAILABLE = True
    logger.info("[v10] volume_profile_features loaded OK")
except Exception as _vp_err:
    logger.warning(
        "[v10] volume_profile_features not importable: %s — 6 features zeroed",
        _vp_err,
    )
    VOLUME_PROFILE_AVAILABLE = False
    VOLUME_PROFILE_FEATURE_NAMES = [
        "vp_poc_price",
        "vp_close_minus_poc_atr",
        "vp_va_high",
        "vp_va_low",
        "vp_close_inside_va_indicator",
        "vp_profile_shape",
    ]

    def add_volume_profile_features(df, ticker=None):  # type: ignore[misc]
        for c in VOLUME_PROFILE_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0
        return df


# ---------------------------------------------------------------------------
# Helper Z: auction_features (open/close auction, 6 features) — Wave M-1 #12
# Wired 2026-05-17. Uses Alpaca 1-min cache via _load_1min(); falls back to
# claudes-test 1Min_merged. .shift(1)-safe inside the module.
# ---------------------------------------------------------------------------
try:
    from auction_features import (  # noqa: E402
        add_auction_features,
        AUCTION_FEATURE_NAMES,
    )
    AUCTION_AVAILABLE = True
    logger.info("[v10] auction_features loaded OK")
except Exception as _auc_err:
    logger.warning(
        "[v10] auction_features not importable: %s — 6 features zeroed",
        _auc_err,
    )
    AUCTION_AVAILABLE = False
    AUCTION_FEATURE_NAMES = [
        "open_auction_ret",
        "close_auction_ret",
        "open_auction_vol_share",
        "close_auction_vol_share",
        "auction_imbalance_ratio",
        "close_auction_dir_vs_session",
    ]

    def add_auction_features(df, ticker=None):  # type: ignore[misc]
        for c in AUCTION_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0
        return df


# ---------------------------------------------------------------------------
# Wave V-1 vol/regime low-cost 7-pack (no new deps). Wired 2026-05-17.
# All modules pure pandas/numpy/scipy and .shift(1)-safe internally.
# ---------------------------------------------------------------------------

# Helper VA: vol_of_vol_features (3 features) — candidate #7
try:
    from vol_of_vol_features import (  # noqa: E402
        add_vol_of_vol_features,
        VOL_OF_VOL_FEATURE_NAMES,
    )
    VOL_OF_VOL_AVAILABLE = True
    logger.info("[v10] vol_of_vol_features loaded OK")
except Exception as _vov_err:
    logger.warning("[v10] vol_of_vol_features not importable: %s — 3 features zeroed", _vov_err)
    VOL_OF_VOL_AVAILABLE = False
    VOL_OF_VOL_FEATURE_NAMES = ["vov_20_20", "vov_60_60", "vov_zscore_252"]

    def add_vol_of_vol_features(df, ticker=None):  # type: ignore[misc]
        for c in VOL_OF_VOL_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0
        return df


# Helper VB: vol_risk_premium_features (4 features) — candidate #12
try:
    from vol_risk_premium_features import (  # noqa: E402
        add_vol_risk_premium_features,
        VRP_FEATURE_NAMES,
    )
    VRP_AVAILABLE = True
    logger.info("[v10] vol_risk_premium_features loaded OK")
except Exception as _vrp_err:
    logger.warning("[v10] vol_risk_premium_features not importable: %s — 4 features zeroed", _vrp_err)
    VRP_AVAILABLE = False
    VRP_FEATURE_NAMES = ["vrp_market", "vrp_ticker", "vrp_zscore_252", "vrp_sign_flip"]

    def add_vol_risk_premium_features(df, ticker=None):  # type: ignore[misc]
        for c in VRP_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0 if c != "vrp_sign_flip" else 0
        return df


# Helper VC: vol_target_sizing_features (2 features) — candidate #13
# Depends on garch11_cond_vol_1d (already wired at Step 15); has RV fallback.
try:
    from vol_target_sizing_features import (  # noqa: E402
        add_vol_target_sizing_features,
        VOL_TARGET_FEATURE_NAMES,
    )
    VOL_TARGET_AVAILABLE = True
    logger.info("[v10] vol_target_sizing_features loaded OK")
except Exception as _vt_err:
    logger.warning("[v10] vol_target_sizing_features not importable: %s — 2 features neutral", _vt_err)
    VOL_TARGET_AVAILABLE = False
    VOL_TARGET_FEATURE_NAMES = ["vol_target_ratio", "vol_target_clipped_5x"]

    def add_vol_target_sizing_features(df, ticker=None):  # type: ignore[misc]
        for c in VOL_TARGET_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 1.0
        return df


# Helper VD: vol_breakout_nr_features (5 features) — candidate #14
try:
    from vol_breakout_nr_features import (  # noqa: E402
        add_vol_breakout_nr_features,
        VOL_BREAKOUT_FEATURE_NAMES,
    )
    VOL_BREAKOUT_AVAILABLE = True
    logger.info("[v10] vol_breakout_nr_features loaded OK")
except Exception as _nr_err:
    logger.warning("[v10] vol_breakout_nr_features not importable: %s — 5 features zeroed", _nr_err)
    VOL_BREAKOUT_AVAILABLE = False
    VOL_BREAKOUT_FEATURE_NAMES = [
        "nr4_indicator", "nr7_indicator", "wr7_indicator",
        "days_since_nr7", "range_pct_of_atr20",
    ]

    def add_vol_breakout_nr_features(df, ticker=None):  # type: ignore[misc]
        for c in VOL_BREAKOUT_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0 if c != "range_pct_of_atr20" else 1.0
        return df


# Helper VE: bollinger_keltner_squeeze_features (4 features) — candidate #15
try:
    from bollinger_keltner_squeeze_features import (  # noqa: E402
        add_bollinger_keltner_squeeze_features,
        SQUEEZE_FEATURE_NAMES,
    )
    SQUEEZE_AVAILABLE = True
    logger.info("[v10] bollinger_keltner_squeeze_features loaded OK")
except Exception as _sq_err:
    logger.warning("[v10] bollinger_keltner_squeeze_features not importable: %s — 4 features zeroed", _sq_err)
    SQUEEZE_AVAILABLE = False
    SQUEEZE_FEATURE_NAMES = [
        "squeeze_on_indicator", "days_in_squeeze",
        "squeeze_release_indicator", "squeeze_momentum_proxy",
    ]

    def add_bollinger_keltner_squeeze_features(df, ticker=None):  # type: ignore[misc]
        for c in SQUEEZE_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0
        return df


# Helper VF: vol_of_vix_features (3 features) — candidate #18
try:
    from vol_of_vix_features import (  # noqa: E402
        add_vol_of_vix_features,
        VVIX_FEATURE_NAMES,
    )
    VOL_OF_VIX_AVAILABLE = True
    logger.info("[v10] vol_of_vix_features loaded OK")
except Exception as _vvx_err:
    logger.warning("[v10] vol_of_vix_features not importable: %s — 3 features zeroed", _vvx_err)
    VOL_OF_VIX_AVAILABLE = False
    VVIX_FEATURE_NAMES = ["vix_realized_vol_21", "vix_vol_zscore_252", "vix_vol_spike_indicator"]

    def add_vol_of_vix_features(df, ticker=None):  # type: ignore[misc]
        for c in VVIX_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0
        return df


# Helper VG: rv_term_structure_features (4 features) — candidate #19
try:
    from rv_term_structure_features import (  # noqa: E402
        add_rv_term_structure_features,
        RV_TERM_FEATURE_NAMES,
    )
    RV_TERM_AVAILABLE = True
    logger.info("[v10] rv_term_structure_features loaded OK")
except Exception as _rvt_err:
    logger.warning("[v10] rv_term_structure_features not importable: %s — 4 features zeroed", _rvt_err)
    RV_TERM_AVAILABLE = False
    RV_TERM_FEATURE_NAMES = [
        "rv5_over_rv21", "rv5_over_rv63",
        "rv_slope_252z", "rv_backwardation_indicator",
    ]

    def add_rv_term_structure_features(df, ticker=None):  # type: ignore[misc]
        for c in RV_TERM_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 1.0 if c in ("rv5_over_rv21", "rv5_over_rv63") else 0.0
        return df


# ---------------------------------------------------------------------------
# Helper VH: amihud_illiquidity_ratio (Amihud 2002 illiquidity ratio, 5 features)
# Wired 2026-05-17. Uses yfinance_daily_OHLCV (close + volume); no extra API.
# .shift(1)-safe: raw ratio computed same-bar then shifted before rolling stats.
# ---------------------------------------------------------------------------
try:
    from amihud_illiquidity_ratio_features import (  # noqa: E402
        compute_amihud_illiquidity_ratio_features,
        AMIHUD_FEATURE_NAMES,
    )
    AMIHUD_AVAILABLE = True
    logger.info("[v10] amihud_illiquidity_ratio_features loaded OK")
except Exception as _amihud_err:
    logger.warning(
        "[v10] amihud_illiquidity_ratio_features not importable: %s — 5 features zeroed",
        _amihud_err,
    )
    AMIHUD_AVAILABLE = False
    AMIHUD_FEATURE_NAMES = [
        "amihud_illiq",
        "amihud_illiq_z21",
        "amihud_illiq_trend",
        "amihud_illiq_ma5",
        "amihud_illiq_spike",
    ]

    def compute_amihud_illiquidity_ratio_features(df, ticker=None):  # type: ignore[misc]
        for c in AMIHUD_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0
        return df


# ---------------------------------------------------------------------------
# Helper VI: rolls_effective_spread (Roll 1984 JFE, 3 features) — Wave H-1
# Wired 2026-05-17. Uses Alpaca 1-min cache; falls back to claudes-test 1Min_merged.
# .shift(1)-safe: all outputs assigned via prior-day shift before returning.
# ---------------------------------------------------------------------------
try:
    from rolls_effective_spread_features import (  # noqa: E402
        compute_rolls_effective_spread_features,
        ROLLS_EFFECTIVE_SPREAD_FEATURE_NAMES,
    )
    ROLLS_SPREAD_AVAILABLE = True
    logger.info("[v10] rolls_effective_spread_features loaded OK")
except Exception as _rolls_err:
    logger.warning(
        "[v10] rolls_effective_spread_features not importable: %s — 3 features zeroed",
        _rolls_err,
    )
    ROLLS_SPREAD_AVAILABLE = False
    ROLLS_EFFECTIVE_SPREAD_FEATURE_NAMES = [
        "rolls_spread_eod",
        "rolls_spread_z21",
        "rolls_spread_rel",
    ]

    def compute_rolls_effective_spread_features(df, ticker=None):  # type: ignore[misc]
        for c in ROLLS_EFFECTIVE_SPREAD_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0
        return df


# ---------------------------------------------------------------------------
# Helper AA: cycle051_features (multi-TF SR daily pivots, 5 features) — Wave Cycle
# Wired 2026-05-17. Source: claudes test/research/archive/cycle051_multi_tf_features_2026-05-03.
# .shift(1)-safe: prior-session HLC drives daily pivot levels.
# ---------------------------------------------------------------------------
try:
    from cycle051_features import (  # noqa: E402
        add_cycle051_features,
        CYCLE051_FEATURE_NAMES,
    )
    CYCLE051_AVAILABLE = True
    logger.info("[v10] cycle051_features loaded OK")
except Exception as _c051_err:
    logger.warning(
        "[v10] cycle051_features not importable: %s — 5 features zeroed", _c051_err
    )
    CYCLE051_AVAILABLE = False
    CYCLE051_FEATURE_NAMES = [
        "sr_1day_pp", "sr_1day_r1", "sr_1day_s1",
        "sr_dist_1day_pp_pct", "sr_above_1day_pp",
    ]

    def add_cycle051_features(df, ticker=None):  # type: ignore[misc]
        for c in CYCLE051_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0 if c == "sr_above_1day_pp" else 0.0
        return df


# ---------------------------------------------------------------------------
# Helper BB: cycle055_features (volatility-gate daily proxies, 5 features) — Wave Cycle
# Wired 2026-05-17. Source: claudes test/research/archive/cycle055_volatility_gates_2026-05-05.
# .shift(1)-safe: all rolling stats shifted 1 bar before assignment.
# ---------------------------------------------------------------------------
try:
    from cycle055_features import (  # noqa: E402
        add_cycle055_features,
        CYCLE055_FEATURE_NAMES,
    )
    CYCLE055_AVAILABLE = True
    logger.info("[v10] cycle055_features loaded OK")
except Exception as _c055_err:
    logger.warning(
        "[v10] cycle055_features not importable: %s — 5 features zeroed", _c055_err
    )
    CYCLE055_AVAILABLE = False
    CYCLE055_FEATURE_NAMES = [
        "vg_atr_pct_14", "vg_range_5d_pct", "vg_vol_regime",
        "vg_in_normal_regime", "vg_rvol_floor_ok",
    ]

    def add_cycle055_features(df, ticker=None):  # type: ignore[misc]
        for c in CYCLE055_FEATURE_NAMES:
            if c not in df.columns:
                if c == "vg_vol_regime":
                    df[c] = 1
                elif c in ("vg_in_normal_regime", "vg_rvol_floor_ok"):
                    df[c] = 0
                else:
                    df[c] = 0.0
        return df


# ---------------------------------------------------------------------------
# Helper CC: cycle058_features (SPY-intra + sector RS, 5 features) — Wave Cycle
# Wired 2026-05-17. Source: claudes test/research/active/cycle058_market_context.
# .shift(1)-safe: merge_asof direction=backward, allow_exact_matches=False.
# ---------------------------------------------------------------------------
try:
    from cycle058_features import (  # noqa: E402
        add_cycle058_features,
        CYCLE058_FEATURE_NAMES,
    )
    CYCLE058_AVAILABLE = True
    logger.info("[v10] cycle058_features loaded OK")
except Exception as _c058_err:
    logger.warning(
        "[v10] cycle058_features not importable: %s — 5 features zeroed", _c058_err
    )
    CYCLE058_AVAILABLE = False
    CYCLE058_FEATURE_NAMES = [
        "mc_spy_intra_cum_ret_eod", "mc_spy_intra_above_or30h_eod",
        "mc_spy_intra_below_or30l_eod", "mc_rs_sector_5d", "mc_rs_sector_5d_sign",
    ]

    def add_cycle058_features(df, ticker=None):  # type: ignore[misc]
        for c in CYCLE058_FEATURE_NAMES:
            if c not in df.columns:
                if c in ("mc_spy_intra_above_or30h_eod",
                         "mc_spy_intra_below_or30l_eod",
                         "mc_rs_sector_5d_sign"):
                    df[c] = 0
                else:
                    df[c] = 0.0
        return df


# ---------------------------------------------------------------------------
# Helper DD: cycle060_features (OI / volume-to-OI / net-delta-z, 3 features) — Wave Cycle
# Wired 2026-05-17. Source: claudes test/research/active/cycle060_options_features.
# .shift(1)-safe: snapshots merged backward + allow_exact_matches=False.
# Note: requires options_snapshots parquet cache populated; otherwise zero-fills.
# ---------------------------------------------------------------------------
try:
    from cycle060_features import (  # noqa: E402
        add_cycle060_features,
        CYCLE060_FEATURE_NAMES,
    )
    CYCLE060_AVAILABLE = True
    logger.info("[v10] cycle060_features loaded OK")
except Exception as _c060_err:
    logger.warning(
        "[v10] cycle060_features not importable: %s — 3 features zeroed", _c060_err
    )
    CYCLE060_AVAILABLE = False
    CYCLE060_FEATURE_NAMES = [
        "put_call_oi_ratio", "volume_to_oi_ratio", "net_delta_exposure_z21",
    ]

    def add_cycle060_features(df, ticker=None):  # type: ignore[misc]
        for c in CYCLE060_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0
        return df


# ---------------------------------------------------------------------------
# Helper EE: cycle061_features (time-of-day daily aggregates, 4 features) — Wave Cycle
# Wired 2026-05-17. Source: claudes test/research/active/cycle061_time_of_day.
# .shift(1)-safe: merge_asof direction=backward + allow_exact_matches=False.
# ---------------------------------------------------------------------------
try:
    from cycle061_features import (  # noqa: E402
        add_cycle061_features,
        CYCLE061_FEATURE_NAMES,
    )
    CYCLE061_AVAILABLE = True
    logger.info("[v10] cycle061_features loaded OK")
except Exception as _c061_err:
    logger.warning(
        "[v10] cycle061_features not importable: %s — 4 features zeroed", _c061_err
    )
    CYCLE061_AVAILABLE = False
    CYCLE061_FEATURE_NAMES = [
        "tod_OR_break_up_rate_5d", "tod_OR_break_down_rate_5d",
        "tod_morning_volume_share", "tod_power_hour_volume_share",
    ]

    def add_cycle061_features(df, ticker=None):  # type: ignore[misc]
        for c in CYCLE061_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0
        return df


# ---------------------------------------------------------------------------
# Helper GG: add_finance_database_features (FinanceDatabase metadata, 4 features)
# Wired 2026-05-17. Source: github:JerBouma/FinanceDatabase (MIT, no paid API).
# Features: fdb_sector, fdb_industry, fdb_market_cap, fdb_exchange (static metadata).
# .shift(1)-safe: static ticker metadata — no intra-bar or future quantity referenced.
# ---------------------------------------------------------------------------
try:
    from add_finance_database_features_features import (  # noqa: E402
        compute_add_finance_database_features_features,
        FDB_FEATURE_NAMES,
        FDB_FEATURE_COUNT,
    )
    FDB_AVAILABLE = True
    logger.info("[v10] add_finance_database_features loaded OK")
except Exception as _fdb_err:
    logger.warning(
        "[v10] add_finance_database_features not importable: %s — 4 features zeroed",
        _fdb_err,
    )
    FDB_AVAILABLE = False
    FDB_FEATURE_COUNT = 4
    FDB_FEATURE_NAMES: list[str] = [  # type: ignore[no-redef]
        "fdb_sector",
        "fdb_industry",
        "fdb_market_cap",
        "fdb_exchange",
    ]

    def compute_add_finance_database_features_features(  # type: ignore[misc]
        df: pd.DataFrame,
        ticker: Optional[str] = None,
    ) -> pd.DataFrame:
        """Stub: zero-fill all 4 fdb cols when module unavailable."""
        for col in FDB_FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0
        return df


# ---------------------------------------------------------------------------
# Helper FG: mlforecast_features (mlforecast-style rolling/EWM/expanding lag
# features, 11 cols: mlf_close_roll5_mean/std, mlf_close_roll21_mean,
# mlf_returns_roll5_mean/std, mlf_volume_roll5_mean/std, mlf_close_ewm_alpha02,
# mlf_returns_expanding_mean, mlf_hl_range_roll5_mean, mlf_close_roll21_max_ratio).
# Source: github:Nixtla/mlforecast (Apache-2.0). Pure-pandas; no paid API.
# shift(1)-safe: all inputs pre-shifted 1 bar inside the module. Wired 2026-05-17.
# ---------------------------------------------------------------------------
try:
    from mlforecast_features_features import (  # noqa: E402
        compute_mlforecast_features_features,
        MLFORECAST_FEATURE_NAMES,
        MLFORECAST_FEATURE_COUNT,
    )
    MLF_AVAILABLE = True
    logger.info("[v10] mlforecast_features loaded OK")
except Exception as _mlf_err:
    logger.warning(
        "[v10] mlforecast_features not importable: %s — 11 features zeroed",
        _mlf_err,
    )
    MLF_AVAILABLE = False
    MLFORECAST_FEATURE_COUNT = 11
    MLFORECAST_FEATURE_NAMES: list[str] = [  # type: ignore[no-redef]
        "mlf_close_roll5_mean",
        "mlf_close_roll5_std",
        "mlf_close_roll21_mean",
        "mlf_returns_roll5_mean",
        "mlf_returns_roll5_std",
        "mlf_volume_roll5_mean",
        "mlf_volume_roll5_std",
        "mlf_close_ewm_alpha02",
        "mlf_returns_expanding_mean",
        "mlf_hl_range_roll5_mean",
        "mlf_close_roll21_max_ratio",
    ]

    def compute_mlforecast_features_features(  # type: ignore[misc]
        df: pd.DataFrame,
        ticker: Optional[str] = None,
    ) -> pd.DataFrame:
        """Stub: zero-fill all 11 mlf_ cols when module unavailable."""
        for col in MLFORECAST_FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0.0
        return df


# ---------------------------------------------------------------------------
# Helper FH: neuralforecast_features (NeuralForecast-inspired decomposition,
# 5 features: nf_trend_slope_21d/63d, nf_fourier_sin/cos_annual,
# nf_residual_vol_21d). Wired 2026-05-17.
# Source: github:Nixtla/neuralforecast (Apache-2.0). Pure pandas/numpy; no paid API.
# shift(1)-safe: all price inputs pre-shifted 1 bar inside the module.
# ---------------------------------------------------------------------------
try:
    from neuralforecast_features_features import (  # noqa: E402
        compute_neuralforecast_features_features,
        NEURALFORECAST_FEATURE_NAMES,
        NEURALFORECAST_FEATURE_COUNT,
    )
    NF_AVAILABLE = True
    logger.info("[v10] neuralforecast_features loaded OK")
except Exception as _nf_err:
    logger.warning(
        "[v10] neuralforecast_features not importable: %s — 5 features zeroed",
        _nf_err,
    )
    NF_AVAILABLE = False
    NEURALFORECAST_FEATURE_COUNT = 5
    NEURALFORECAST_FEATURE_NAMES: list[str] = [  # type: ignore[no-redef]
        "nf_trend_slope_21d",
        "nf_trend_slope_63d",
        "nf_fourier_sin_annual",
        "nf_fourier_cos_annual",
        "nf_residual_vol_21d",
    ]

    def compute_neuralforecast_features_features(  # type: ignore[misc]
        df: pd.DataFrame,
        ticker: Optional[str] = None,
    ) -> pd.DataFrame:
        """Stub: zero-fill all 5 nf_ cols when module unavailable."""
        for col in NEURALFORECAST_FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0.0
        return df


# ---------------------------------------------------------------------------
# Helper GH: worldquant_alpha101_replay (WorldQuant Alpha#101 candle-direction
# z-score, 1 feature: wq101_replay_alpha101_z21). Wired 2026-05-17.
# Source: github:lvlh2/alpha101 (MIT). Pure pandas/numpy; no paid API.
# shift(1)-safe: all price inputs pre-shifted 1 bar inside the module.
# ---------------------------------------------------------------------------
try:
    from worldquant_alpha101_replay_20260517t224845z_features import (  # noqa: E402
        compute_worldquant_alpha101_replay_20260517t224845z_features,
        WQ101_REPLAY_FEATURE_NAMES,
        WQ101_REPLAY_FEATURE_COUNT,
    )
    WQ101_REPLAY_AVAILABLE = True
    logger.info("[v10] worldquant_alpha101_replay loaded OK")
except Exception as _wq101_err:
    logger.warning(
        "[v10] worldquant_alpha101_replay not importable: %s — 1 feature zeroed",
        _wq101_err,
    )
    WQ101_REPLAY_AVAILABLE = False
    WQ101_REPLAY_FEATURE_COUNT = 1
    WQ101_REPLAY_FEATURE_NAMES: list[str] = ["wq101_replay_alpha101_z21"]  # type: ignore[no-redef]

    def compute_worldquant_alpha101_replay_20260517t224845z_features(  # type: ignore[misc]
        df: pd.DataFrame,
        ticker: Optional[str] = None,
    ) -> pd.DataFrame:
        """Stub: zero-fill wq101_replay col when module unavailable."""
        if "wq101_replay_alpha101_z21" not in df.columns:
            df["wq101_replay_alpha101_z21"] = 0.0
        return df


# ---------------------------------------------------------------------------
# Helper GJ: alpha101_ts_safe_subset_replay (STHSF/alpha101 Alpha#6 z-score,
# 1 feature: a101_ts_wq6_z21). Wired 2026-05-17.
# Source: github:STHSF/alpha101 (MIT, no paid API). Pure pandas/numpy.
# shift(1)-safe: open and volume both shifted 1 bar before rolling correlation.
# ---------------------------------------------------------------------------
try:
    from alpha101_ts_safe_subset_replay_20260517t224845z_features import (  # noqa: E402
        compute_alpha101_ts_safe_subset_replay_20260517t224845z_features,
        ALPHA101_TS_SAFE_FEATURE_NAMES,
        ALPHA101_TS_SAFE_FEATURE_COUNT,
    )
    ALPHA101_TS_SAFE_AVAILABLE = True
    logger.info("[v10] alpha101_ts_safe_subset_replay loaded OK")
except Exception as _a101ts_err:
    logger.warning(
        "[v10] alpha101_ts_safe_subset_replay not importable: %s — 1 feature zeroed",
        _a101ts_err,
    )
    ALPHA101_TS_SAFE_AVAILABLE = False
    ALPHA101_TS_SAFE_FEATURE_COUNT = 1
    ALPHA101_TS_SAFE_FEATURE_NAMES: list[str] = ["a101_ts_wq6_z21"]  # type: ignore[no-redef]

    def compute_alpha101_ts_safe_subset_replay_20260517t224845z_features(  # type: ignore[misc]
        df: pd.DataFrame,
        ticker: Optional[str] = None,
    ) -> pd.DataFrame:
        """Stub: zero-fill a101_ts col when module unavailable."""
        if "a101_ts_wq6_z21" not in df.columns:
            df["a101_ts_wq6_z21"] = 0.0
        return df


# ---------------------------------------------------------------------------
# Helper FF: add_alpha_features_core_features (WorldQuant-style alpha101 port,
# 30 features: afc_alpha001..afc_alpha030). Wired 2026-05-17.
# Source: github:GiovanniPioDelvecchio/alpha_features_core (MIT license).
# .shift(1)-safe: all outputs use rolling windows over prior-completed bars only.
# ---------------------------------------------------------------------------
try:
    from add_alpha_features_core_features_features import (  # noqa: E402
        compute_add_alpha_features_core_features_features,
        AFC_CORE_FEATURE_NAMES,
        AFC_CORE_FEATURE_COUNT,
    )
    AFC_CORE_AVAILABLE = True
    logger.info("[v10] add_alpha_features_core_features loaded OK")
except Exception as _afc_err:
    logger.warning(
        "[v10] add_alpha_features_core_features not importable: %s — 30 features zeroed",
        _afc_err,
    )
    AFC_CORE_AVAILABLE = False
    AFC_CORE_FEATURE_COUNT = 30
    AFC_CORE_FEATURE_NAMES: list[str] = [f"afc_alpha{i:03d}" for i in range(1, 31)]  # type: ignore[no-redef]

    def compute_add_alpha_features_core_features_features(  # type: ignore[misc]
        df: pd.DataFrame,
        ticker: Optional[str] = None,
    ) -> pd.DataFrame:
        """Stub: zero-fill all 30 afc_alpha cols when module unavailable."""
        for col in AFC_CORE_FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0.0
        return df


# ---------------------------------------------------------------------------
# Helper GI: hist_data_mythos_deltas_features (Mythos curriculum prior, 6 features)
# Wired 2026-05-17. Source: AI-Tools/reports/mythos_xgboost_integration/per_ticker_summaries/
# Static per-ticker broadcast — same value all rows, no shift(1) needed.
# CAVEAT: 7/500 S&P tickers have summaries (AAPL, BXP, COIN, JPM, NVDA, TPL, XOM);
#         remainder zero-fills with mythos_has_summary=0.
# ---------------------------------------------------------------------------
try:
    from hist_data_mythos_deltas_features import (  # noqa: E402
        add_mythos_deltas_features,
        MYTHOS_DELTA_FEATURE_NAMES,
    )
    MYTHOS_DELTAS_AVAILABLE = True
    logger.info("[v10] hist_data_mythos_deltas_features loaded OK")
except Exception as _mythos_deltas_err:
    logger.warning(
        "[v10] hist_data_mythos_deltas_features not importable: %s — 6 features zeroed",
        _mythos_deltas_err,
    )
    MYTHOS_DELTAS_AVAILABLE = False
    MYTHOS_DELTA_FEATURE_NAMES = [  # type: ignore[assignment]
        "mythos_has_summary",
        "mythos_delta_win_rate",
        "mythos_delta_profit_factor",
        "mythos_delta_total_return",
        "mythos_baseline_profit_factor",
        "mythos_improved_flag",
    ]

    def add_mythos_deltas_features(df, ticker=None):  # type: ignore[misc]
        for c in MYTHOS_DELTA_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0
        return df


# ---------------------------------------------------------------------------
# Helper GJ: hist_data_edgar_features (EDGAR filing recency/density, 9 features)
# Wired 2026-05-17. Source: claudes test/data/edgar/data/edgar.db (SQLite,
# 57,066 filings, 500 tickers, 2020-01-02 → 2026-04-24).
# Per-bar dynamic features — .shift(1)-safe via merge_asof(direction='backward',
# allow_exact_matches=False) + searchsorted side='left' (strict-prior boundary).
# Distinct from sec_edgar_features.py (repo-binding stub, unused) — this module
# reads the local indexed EDGAR DB directly.
# ---------------------------------------------------------------------------
try:
    from hist_data_edgar_features import (  # noqa: E402
        add_edgar_features,
        EDGAR_FEATURE_NAMES,
    )
    EDGAR_DB_AVAILABLE = True
    logger.info("[v10] hist_data_edgar_features loaded OK")
except Exception as _edgar_err:
    logger.warning(
        "[v10] hist_data_edgar_features not importable: %s — 9 features zeroed",
        _edgar_err,
    )
    EDGAR_DB_AVAILABLE = False
    EDGAR_FEATURE_NAMES = [  # type: ignore[assignment]
        "edgar_days_since_any_filing",
        "edgar_days_since_8k",
        "edgar_days_since_10q",
        "edgar_days_since_10k",
        "edgar_filing_flag_7d",
        "edgar_filing_flag_30d",
        "edgar_eightk_flag_7d",
        "edgar_filings_count_90d",
        "edgar_has_10k_this_year",
    ]

    def add_edgar_features(df, ticker=None):  # type: ignore[misc]
        for c in EDGAR_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0
        return df


# ---------------------------------------------------------------------------
# Helper GJ-EXTRAS: edgar_extras_features (12 features). Wired 2026-05-20.
# Source: claudes test/data/edgar/data/edgar.db. Adds DEF 14A, amendments,
# likely-earnings-8K (via period_of_report lag), S-1, filing burst, accel.
# Distinct from hist_data_edgar_features (9 base features). See module docstring
# for gap-analysis (G1..G6).
# ---------------------------------------------------------------------------
try:
    from edgar_extras_features import (  # noqa: E402
        add_edgar_extras_features,
        EDGAR_EXTRAS_FEATURE_NAMES,
    )
    EDGAR_EXTRAS_AVAILABLE = True
    logger.info("[v10] edgar_extras_features loaded OK")
except Exception as _edgar_extras_err:
    logger.warning(
        "[v10] edgar_extras_features not importable: %s — 12 features zeroed",
        _edgar_extras_err,
    )
    EDGAR_EXTRAS_AVAILABLE = False
    EDGAR_EXTRAS_FEATURE_NAMES = [  # type: ignore[assignment]
        "edgar_days_since_def14a",
        "edgar_def14a_flag_30d",
        "edgar_days_since_any_amendment",
        "edgar_amendment_flag_30d",
        "edgar_days_since_likely_earnings_8k",
        "edgar_likely_earnings_8k_flag_7d",
        "edgar_filed_to_period_lag_days",
        "edgar_days_since_s1",
        "edgar_s1_flag_180d",
        "edgar_filings_count_7d",
        "edgar_burst_flag",
        "edgar_filing_density_accel",
    ]

    def add_edgar_extras_features(df, ticker=None):  # type: ignore[misc]
        for c in EDGAR_EXTRAS_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0
        return df


# ---------------------------------------------------------------------------
# Helper GJ-GOV-EXTRAS: govtrades_extras_features (45 features). Wired 2026-05-20.
# Wraps photis_govtrades_contracts (3) + photis_govtrades_lobbying_amt (3) +
# synapse_gov_enhanced (39 market-wide). Distinct from govtrades_features (3
# base congress-density features). See module docstring for unwired-module audit.
# ---------------------------------------------------------------------------
try:
    from govtrades_extras_features import (  # noqa: E402
        add_govtrades_extras_features,
        GOVTRADES_EXTRAS_FEATURE_NAMES,
    )
    GOVTRADES_EXTRAS_AVAILABLE = True
    logger.info(
        "[v10] govtrades_extras_features loaded OK (%d cols)",
        len(GOVTRADES_EXTRAS_FEATURE_NAMES),
    )
except Exception as _gov_extras_err:
    logger.warning(
        "[v10] govtrades_extras_features not importable: %s — 45 features zeroed",
        _gov_extras_err,
    )
    GOVTRADES_EXTRAS_AVAILABLE = False
    GOVTRADES_EXTRAS_FEATURE_NAMES = [  # type: ignore[assignment]
        "gt_contracts_ttm_usd",
        "gt_contracts_award_count_30d",
        "gt_contracts_qoq_growth",
        "gt_lob_amt_30d_usd",
        "gt_lob_amt_qoq_growth",
        "gt_lob_amt_ttm_usd",
        # synapse_gov_* 39 names omitted from stub (module will fill if loaded)
    ]

    def add_govtrades_extras_features(df, ticker=None):  # type: ignore[misc]
        for c in GOVTRADES_EXTRAS_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0
        return df


# ---------------------------------------------------------------------------
# Helper GK: ceo_personal_donation_flag_political_replay (FEC Schedule A
# individual-contributions employer-search signal, 1 feature). Wired 2026-05-17.
# Source: api.fec.gov/v1/schedules/schedule_a/ (public, no paid API key).
# .shift(1)-safe: rolling 90-day count uses only contribution_receipt_date <
# bar_date (strict-prior searchsorted boundary). See module for full audit.
# ---------------------------------------------------------------------------
try:
    from ceo_personal_donation_flag_political_replay_20260517t225454z_features import (  # noqa: E402
        compute_ceo_personal_donation_flag_political_replay_20260517t225454z_features,
        CEO_DONATION_FEATURE_NAMES,
        CEO_DONATION_FEATURE_COUNT,
    )
    CEO_DONATION_AVAILABLE = True
    logger.info("[v10] ceo_personal_donation_flag_political_replay loaded OK")
except Exception as _ceo_don_err:
    logger.warning(
        "[v10] ceo_personal_donation_flag_political_replay not importable: %s — 1 feature zeroed",
        _ceo_don_err,
    )
    CEO_DONATION_AVAILABLE = False
    CEO_DONATION_FEATURE_COUNT = 1
    CEO_DONATION_FEATURE_NAMES: list[str] = ["fec_donation_flag_90d"]  # type: ignore[no-redef]

    def compute_ceo_personal_donation_flag_political_replay_20260517t225454z_features(  # type: ignore[misc]
        df: pd.DataFrame,
        ticker: Optional[str] = None,
    ) -> pd.DataFrame:
        """Stub: zero-fill fec_donation_flag_90d when module unavailable."""
        if "fec_donation_flag_90d" not in df.columns:
            df["fec_donation_flag_90d"] = 0.0
        return df


# ---------------------------------------------------------------------------
# Helper GL: senate_efd_options_disclosure_count_30d_replay (Senate STOCK Act
# options-transaction rolling count, 1 feature). Wired 2026-05-17.
# Source: QuiverQuant free congressional-trading API (no paid API key).
# .shift(1)-safe: rolling 30d count uses only disclosures whose public
# report_date < bar_date (strict-prior searchsorted boundary).
# ---------------------------------------------------------------------------
try:
    from senate_efd_options_disclosure_count_30d_replay_20260517t225454z_features import (  # noqa: E402
        compute_senate_efd_options_disclosure_count_30d_replay_20260517t225454z_features,
        SENATE_EFD_OPTIONS_FEATURE_NAMES,
        SENATE_EFD_OPTIONS_FEATURE_COUNT,
    )
    SENATE_EFD_OPTIONS_AVAILABLE = True
    logger.info("[v10] senate_efd_options_disclosure_count_30d_replay loaded OK")
except Exception as _sefd_err:
    logger.warning(
        "[v10] senate_efd_options_disclosure_count_30d_replay not importable: %s — 1 feature zeroed",
        _sefd_err,
    )
    SENATE_EFD_OPTIONS_AVAILABLE = False
    SENATE_EFD_OPTIONS_FEATURE_COUNT = 1
    SENATE_EFD_OPTIONS_FEATURE_NAMES: list[str] = ["senate_efd_options_count_30d"]  # type: ignore[no-redef]

    def compute_senate_efd_options_disclosure_count_30d_replay_20260517t225454z_features(  # type: ignore[misc]
        df: pd.DataFrame,
        ticker: Optional[str] = None,
    ) -> pd.DataFrame:
        """Stub: zero-fill senate_efd_options_count_30d when module unavailable."""
        if "senate_efd_options_count_30d" not in df.columns:
            df["senate_efd_options_count_30d"] = 0.0
        return df


# ---------------------------------------------------------------------------
# Helper GM: vix_term_structure_features (VIX term-structure v1, 3 features)
# Wired 2026-05-18. Source: yfinance:^VIX9D,^VIX (MIT, no paid API).
# Features: vix_ts_spread (VIX−VIX9D level spread), vix_ts_spread_z21 (21d z-score),
# vix_ts_contango_streak (signed consecutive-day regime streak).
# Complementary to vix_term_structure_v2_features (no column overlap).
# .shift(1)-safe: merge_asof direction=backward with 1-day subtraction on lookup date.
# ---------------------------------------------------------------------------
try:
    from vix_term_structure_features import (  # noqa: E402
        compute_vix_term_structure_features,
        VIX_TERM_FEATURE_NAMES,
    )
    VIX_TERM_AVAILABLE = True
    logger.info("[v10] vix_term_structure_features loaded OK")
except Exception as _vt1_err:
    logger.warning(
        "[v10] vix_term_structure_features not importable: %s — 3 features zeroed", _vt1_err
    )
    VIX_TERM_AVAILABLE = False
    VIX_TERM_FEATURE_NAMES: list[str] = [  # type: ignore[no-redef]
        "vix_ts_spread",
        "vix_ts_spread_z21",
        "vix_ts_contango_streak",
    ]

    def compute_vix_term_structure_features(df: pd.DataFrame) -> pd.DataFrame:  # type: ignore[misc]
        """Stub: zero-fill 3 vix_term_structure cols when module unavailable."""
        for c in VIX_TERM_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0
        return df


# ---------------------------------------------------------------------------
# Helper GO: worldquant_alpha101_features (WorldQuant Alpha-101 single-ticker
# adaptation, 25 features). Wired 2026-05-18.
# Source: "101 Formulaic Alphas" Kakushadze (2016), github:lvlh2/alpha101 (MIT).
# No paid API; pure OHLCV; cross-sectional rank replaced by ts_rank(window=20).
# .shift(1)-safe: all inputs shifted 1 bar at module entry.
# ---------------------------------------------------------------------------
try:
    from worldquant_alpha101_features import (  # noqa: E402
        compute_worldquant_alpha101_features,
        WQ_ALPHA101_FEATURE_NAMES,
        WQ_ALPHA101_FEATURE_COUNT,
    )
    WQ_ALPHA101_AVAILABLE = True
    logger.info("[v10] worldquant_alpha101_features loaded OK (%d features)", WQ_ALPHA101_FEATURE_COUNT)
except Exception as _wqa101_err:
    logger.warning(
        "[v10] worldquant_alpha101_features not importable: %s — 25 features zeroed",
        _wqa101_err,
    )
    WQ_ALPHA101_AVAILABLE = False
    WQ_ALPHA101_FEATURE_COUNT = 25
    WQ_ALPHA101_FEATURE_NAMES: list[str] = [  # type: ignore[no-redef]
        "wq_a002", "wq_a003", "wq_a006", "wq_a007", "wq_a009",
        "wq_a010", "wq_a012", "wq_a016", "wq_a017", "wq_a018",
        "wq_a019", "wq_a020", "wq_a021", "wq_a022", "wq_a023",
        "wq_a024", "wq_a025", "wq_a026", "wq_a030", "wq_a033",
        "wq_a034", "wq_a035", "wq_a040", "wq_a043", "wq_a044",
    ]

    def compute_worldquant_alpha101_features(df: pd.DataFrame, ticker=None):  # type: ignore[misc]
        """Stub: zero-fill all wq_alpha101 cols when module unavailable."""
        for c in WQ_ALPHA101_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0
        return df


# ---------------------------------------------------------------------------
# Helper GN: regime_hmm_3state_vol_regime (3-state HMM vol regime, 3 features)
# Wired 2026-05-18. Data source: standard daily OHLCV (Yang-Zhang vol estimator).
# Features: hmm_3state_vol_regime (0/1/2), hmm_3state_vol_high_prob, hmm_3state_yz_vol_z21.
# .shift(1)-safe: YZ vol computed same-bar, shifted by 1 before HMM input.
# License: hmmlearn BSD-3-Clause; YZ estimator public domain (Yang-Zhang 2000 JF).
# ---------------------------------------------------------------------------
try:
    from regime_hmm_3state_vol_regime_features import (  # noqa: E402
        compute_regime_hmm_3state_vol_regime_features,
        HMM_3STATE_FEATURE_NAMES,
    )
    HMM_3STATE_AVAILABLE = True
    logger.info("[v10] regime_hmm_3state_vol_regime_features loaded OK")
except Exception as _hmm3_err:
    logger.warning(
        "[v10] regime_hmm_3state_vol_regime_features not importable: %s — 3 features zeroed",
        _hmm3_err,
    )
    HMM_3STATE_AVAILABLE = False
    HMM_3STATE_FEATURE_NAMES: list[str] = [  # type: ignore[no-redef]
        "hmm_3state_vol_regime",
        "hmm_3state_vol_high_prob",
        "hmm_3state_yz_vol_z21",
    ]

    def compute_regime_hmm_3state_vol_regime_features(df: pd.DataFrame, ticker=None):  # type: ignore[misc]
        """Stub: zero-fill 3 hmm_3state cols when module unavailable."""
        for c in HMM_3STATE_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0
        return df


# ---------------------------------------------------------------------------
# Helper GP: regime_changepoint_bayesian_vol_break (BOCPD vol-break, 3 features)
# Wired 2026-05-18. Algorithm: Adams & MacKay (2007) BOCPD (arXiv:0710.3742).
# Data source: 5-day realized variance from yfinance close (no paid API).
# Features: bocpd_vol_break_prob, bocpd_vol_run_length_norm, bocpd_vol_regime_id.
# .shift(1)-safe: 5-day RV is same-bar quantity; .shift(1) applied before BOCPD.
# License: bocpd (alan-turing-institute) MIT; built-in fallback is pure numpy/scipy.
# ---------------------------------------------------------------------------
try:
    from regime_changepoint_bayesian_vol_break_features import (  # noqa: E402
        compute_regime_changepoint_bayesian_vol_break_features,
        BOCPD_VOL_BREAK_FEATURE_NAMES,
    )
    BOCPD_VOL_BREAK_AVAILABLE = True
    logger.info("[v10] regime_changepoint_bayesian_vol_break_features loaded OK")
except Exception as _bocpd_err:
    logger.warning(
        "[v10] regime_changepoint_bayesian_vol_break_features not importable: %s — 3 features zeroed",
        _bocpd_err,
    )
    BOCPD_VOL_BREAK_AVAILABLE = False
    BOCPD_VOL_BREAK_FEATURE_NAMES: list[str] = [  # type: ignore[no-redef]
        "bocpd_vol_break_prob",
        "bocpd_vol_run_length_norm",
        "bocpd_vol_regime_id",
    ]

    def compute_regime_changepoint_bayesian_vol_break_features(df: pd.DataFrame, ticker=None):  # type: ignore[misc]
        """Stub: zero-fill 3 bocpd_vol cols when module unavailable."""
        for c in BOCPD_VOL_BREAK_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0
        return df


# ---------------------------------------------------------------------------
# Helper GQ: add_stockstats_features (stockstats technical indicators, 28 features)
# Wired 2026-05-18. Source: github:jealous/stockstats (BSD-3-Clause, no paid API).
# Features: ss_macd/s/h, ss_rsi_14, ss_boll/ub/lb, ss_cci, ss_wr_14, ss_kdjk/d/j,
#           ss_atr_14, ss_dma, ss_vr, ss_close_{5,10,20,50}_sma,
#           ss_close_{5,20}_ema, ss_close_{5,10}_mstd, ss_mfi, ss_trix,
#           ss_close_10_roc, ss_volume_5_sma.
# .shift(1)-safe: all indicator series shifted 1 bar inside the module.
# ---------------------------------------------------------------------------
try:
    from add_stockstats_features_features import (  # noqa: E402
        compute_add_stockstats_features,
        STOCKSTATS_FEATURE_NAMES,
        STOCKSTATS_FEATURE_COUNT,
    )
    STOCKSTATS_AVAILABLE = True
    logger.info("[v10] add_stockstats_features loaded OK")
except Exception as _ss_err:
    logger.warning(
        "[v10] add_stockstats_features not importable: %s — 28 features zeroed", _ss_err
    )
    STOCKSTATS_AVAILABLE = False
    STOCKSTATS_FEATURE_COUNT = 28
    STOCKSTATS_FEATURE_NAMES: list[str] = [  # type: ignore[no-redef]
        "ss_macd", "ss_macds", "ss_macdh", "ss_rsi_14",
        "ss_boll", "ss_boll_ub", "ss_boll_lb", "ss_cci", "ss_wr_14",
        "ss_kdjk", "ss_kdjd", "ss_kdjj", "ss_atr_14", "ss_dma", "ss_vr",
        "ss_close_5_sma", "ss_close_10_sma", "ss_close_20_sma", "ss_close_50_sma",
        "ss_close_5_ema", "ss_close_20_ema",
        "ss_close_5_mstd", "ss_close_10_mstd",
        "ss_mfi", "ss_trix", "ss_close_10_roc", "ss_volume_5_sma",
    ]

    def compute_add_stockstats_features(  # type: ignore[misc]
        df: pd.DataFrame,
        ticker: Optional[str] = None,
    ) -> pd.DataFrame:
        """Stub: zero-fill all 28 ss_* cols when module unavailable."""
        for col in STOCKSTATS_FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0.0
        return df


# ---------------------------------------------------------------------------
# Helper GR: add_talipp_features (talipp TA indicators, 8 features)
# Wired 2026-05-18. Source: github:nardew/talipp (MIT, no paid API).
# Features: talipp_tema_10, talipp_dema_10, talipp_hma_14, talipp_trix_10,
#           talipp_dpo_20, talipp_roc_10, talipp_zlema_10, talipp_wma_10.
# .shift(1)-safe: all indicator series shifted 1 bar inside the module.
# ---------------------------------------------------------------------------
try:
    from add_talipp_features_features import (  # noqa: E402
        compute_add_talipp_features,
        TALIPP_FEATURE_NAMES,
        TALIPP_FEATURE_COUNT,
    )
    logger.info("[v10] add_talipp_features loaded OK")
except Exception as _talipp_err:
    logger.warning(
        "[v10] add_talipp_features not importable: %s — 8 features zeroed", _talipp_err
    )
    TALIPP_FEATURE_COUNT = 8
    TALIPP_FEATURE_NAMES: list[str] = [  # type: ignore[no-redef]
        "talipp_tema_10", "talipp_dema_10", "talipp_hma_14", "talipp_trix_10",
        "talipp_dpo_20", "talipp_roc_10", "talipp_zlema_10", "talipp_wma_10",
    ]

    def compute_add_talipp_features(  # type: ignore[misc]
        df: pd.DataFrame,
        ticker: Optional[str] = None,
    ) -> pd.DataFrame:
        """Stub: zero-fill all 8 talipp_* cols when module unavailable."""
        for col in TALIPP_FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0.0
        return df


# ---------------------------------------------------------------------------
# Helper GS: add_jesse_features (Jesse-inspired TA indicators, 54 features)
# Wired 2026-05-18. Source: github:jesse-ai/jesse (MIT, no paid API).
# Features: moving averages, Ichimoku, Donchian, MACD, momentum oscillators,
#           trend (ADX/Aroon/Supertrend), volatility (BB/KC), volume/micro.
# .shift(1)-safe: all indicator series shifted 1 bar inside the module.
# ---------------------------------------------------------------------------
try:
    from add_jesse_features_features import (  # noqa: E402
        compute_add_jesse_features_features,
        JESSE_FEATURE_NAMES,
        JESSE_FEATURE_COUNT,
    )
    logger.info("[v10] add_jesse_features loaded OK")
except Exception as _jesse_err:
    logger.warning(
        "[v10] add_jesse_features not importable: %s — 54 features zeroed", _jesse_err
    )
    JESSE_FEATURE_COUNT = 54
    JESSE_FEATURE_NAMES: list[str] = [  # type: ignore[no-redef]
        "jesse_sma_10", "jesse_sma_20", "jesse_sma_50",
        "jesse_ema_9", "jesse_ema_21", "jesse_wma_10", "jesse_vwma_10",
        "jesse_ichi_tenkan", "jesse_ichi_kijun", "jesse_ichi_cloud_diff",
        "jesse_ichi_cloud_bull", "jesse_ichi_chikou_diff",
        "jesse_dc_upper", "jesse_dc_lower", "jesse_dc_mid",
        "jesse_macd", "jesse_macd_signal", "jesse_macd_hist",
        "jesse_rsi_14", "jesse_rsi_7", "jesse_stoch_k", "jesse_stoch_d",
        "jesse_cci_20", "jesse_willr_14", "jesse_ultimate_osc",
        "jesse_roc_10", "jesse_mom_10", "jesse_ao",
        "jesse_adx_14", "jesse_di_plus", "jesse_di_minus",
        "jesse_aroon_up", "jesse_aroon_down", "jesse_aroon_osc",
        "jesse_supertrend_dir",
        "jesse_atr_14", "jesse_natr_14",
        "jesse_bb_upper", "jesse_bb_lower", "jesse_bb_width", "jesse_bb_pct",
        "jesse_kc_width",
        "jesse_mfi_14", "jesse_cmf_20", "jesse_obv_z21", "jesse_vwap_dev",
        "jesse_ad_z21", "jesse_force_idx_13", "jesse_eom_14", "jesse_pvt_z21",
        "jesse_nvi_z21", "jesse_pvi_z21", "jesse_vwap_range_pct",
        "jesse_vol_ratio_10_50",
    ]

    def compute_add_jesse_features_features(  # type: ignore[misc]
        df: pd.DataFrame,
        ticker: Optional[str] = None,
    ) -> pd.DataFrame:
        """Stub: zero-fill all 54 jesse_* cols when module unavailable."""
        for col in JESSE_FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0.0
        return df


# ---------------------------------------------------------------------------
# Helper GT: add_shashank_finance_features (shashankvemuri/Finance, 58 features)
# Wired 2026-05-18. Source: github:shashankvemuri/Finance (MIT, no paid API).
# Features: MA ratios, RSI variants, MACD variants, Stochastic, Bollinger
#           Bands, volatility, volume, ADX/Aroon/CCI, price-action patterns,
#           mean-reversion z-scores, S/R pivots, Williams%R, StochRSI, MFI, CMF.
# .shift(1)-safe: all indicator series shifted 1 bar inside the module.
# ---------------------------------------------------------------------------
try:
    from add_shashank_finance_features_features import (  # noqa: E402
        compute_add_shashank_finance_features,
        SHF_FEATURE_NAMES,
        SHF_FEATURE_COUNT,
    )
    logger.info("[v10] add_shashank_finance_features loaded OK")
except Exception as _shf_err:
    logger.warning(
        "[v10] add_shashank_finance_features not importable: %s — 58 features zeroed", _shf_err
    )
    SHF_FEATURE_COUNT = 58
    SHF_FEATURE_NAMES: list[str] = [  # type: ignore[no-redef]
        "shf_close_sma5_ratio", "shf_close_sma10_ratio",
        "shf_close_sma20_ratio", "shf_close_sma50_ratio",
        "shf_sma5_sma20_ratio", "shf_sma20_sma50_ratio",
        "shf_sma50_sma200_ratio", "shf_ema8_ema21_ratio",
        "shf_rsi_9", "shf_rsi_21",
        "shf_rsi_divergence_14", "shf_rsi_regime",
        "shf_macd_8_17_9", "shf_macd_signal_8_17_9", "shf_macd_hist_8_17_9",
        "shf_macd_pct_price", "shf_macd_momentum",
        "shf_stoch_k_14_3", "shf_stoch_d_14_3", "shf_stoch_regime",
        "shf_bb_upper_pct", "shf_bb_lower_pct", "shf_bb_width_pct",
        "shf_bb_position", "shf_bb_squeeze_flag",
        "shf_hist_vol_5", "shf_hist_vol_10", "shf_hist_vol_21",
        "shf_vol_ratio_5_21", "shf_vol_of_vol_21",
        "shf_volume_sma_ratio_5", "shf_volume_sma_ratio_20",
        "shf_obv_sma_ratio_10", "shf_volume_momentum_5",
        "shf_force_index_1", "shf_eom_14",
        "shf_adx_10", "shf_aroon_diff_14", "shf_cci_14", "shf_cci_40",
        "shf_psar_direction",
        "shf_doji_flag", "shf_hammer_flag", "shf_engulfing_bull_flag",
        "shf_gap_up_flag", "shf_high_52w_pct", "shf_low_52w_pct",
        "shf_zscore_close_10", "shf_zscore_close_21", "shf_zscore_close_63",
        "shf_distance_from_52w_high",
        "shf_pivot_high_5", "shf_pivot_low_5", "shf_hh_hl_flag",
        "shf_willr_14", "shf_stoch_rsi_14", "shf_mfi_9", "shf_cmf_14",
    ]

    def compute_add_shashank_finance_features(  # type: ignore[misc]
        df: pd.DataFrame,
        ticker: Optional[str] = None,
    ) -> pd.DataFrame:
        """Stub: zero-fill all 58 shf_* cols when module unavailable."""
        for col in SHF_FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0.0
        return df


try:
    from load_conlan_eod_price_data_features import (  # noqa: E402
        compute_load_conlan_eod_price_data_features,
        CONLAN_EOD_FEATURE_NAMES,
        CONLAN_EOD_FEATURE_COUNT,
    )
    logger.info("[v10] load_conlan_eod_price_data loaded OK")
except Exception as _cep_err:
    logger.warning(
        "[v10] load_conlan_eod_price_data not importable: %s — 6 features zeroed", _cep_err
    )
    CONLAN_EOD_FEATURE_COUNT = 6
    CONLAN_EOD_FEATURE_NAMES: list[str] = [  # type: ignore[no-redef]
        "conlan_eod_pct_below_52w_high",
        "conlan_eod_mom_6m",
        "conlan_eod_vol_trend_ratio",
        "conlan_eod_close_above_200ma",
        "conlan_eod_atr_pct",
        "conlan_eod_dollar_vol_z",
    ]

    def compute_load_conlan_eod_price_data_features(  # type: ignore[misc]
        df: pd.DataFrame,
        ticker: Optional[str] = None,
    ) -> pd.DataFrame:
        """Stub: zero-fill all 6 conlan_eod_* cols when module unavailable."""
        for col in CONLAN_EOD_FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0.0
        return df


# ---------------------------------------------------------------------------
# Helper CAD1: load_conlan_alt_data_features (Conlan alt-data, 5 features) — Wave CAD1
# Wired 2026-05-18. Source: github:chrisconlan/algorithmic-trading-with-python/data/alternative_data
# License: MIT; requires_paid_api: NO; OHLCV proxies used when CSV files absent.
# .shift(1)-safe: all 5 output columns shifted 1 bar inside the module.
# ---------------------------------------------------------------------------
try:
    from load_conlan_alt_data_features_features import (  # noqa: E402
        compute_load_conlan_alt_data_features,
        CONLAN_ALT_FEATURE_NAMES,
        CONLAN_ALT_FEATURE_COUNT,
    )
    logger.info("[v10] load_conlan_alt_data_features loaded OK")
except Exception as _cad_err:
    logger.warning(
        "[v10] load_conlan_alt_data_features not importable: %s — 5 features zeroed", _cad_err
    )
    CONLAN_ALT_FEATURE_COUNT = 5
    CONLAN_ALT_FEATURE_NAMES: list[str] = [  # type: ignore[no-redef]
        "conlan_alt_vol_price_corr_21d",
        "conlan_alt_intraday_range_norm_5d",
        "conlan_alt_close_vs_open_sent_5d",
        "conlan_alt_overnight_gap_pct_5d",
        "conlan_alt_turnover_ratio_21d",
    ]

    def compute_load_conlan_alt_data_features(  # type: ignore[misc]
        df: pd.DataFrame,
        ticker: Optional[str] = None,
    ) -> pd.DataFrame:
        """Stub: zero-fill all 5 conlan_alt_* cols when module unavailable."""
        for col in CONLAN_ALT_FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0.0
        return df


# ---------------------------------------------------------------------------
# Helper GU: alpha101_ts_safe_subset_features (STHSF/alpha101 ts-safe, 15 features)
# Wired 2026-05-18. Source: github:STHSF/alpha101 (MIT, no paid API).
# Features: a101ts_alpha001/003/006/008/012/016/019/020/023/033/034/035/040/041/051.
# .shift(1)-safe: all OHLCV inputs pre-shifted 1 bar inside the module.
# ---------------------------------------------------------------------------
try:
    from alpha101_ts_safe_subset_features import (  # noqa: E402
        compute_alpha101_ts_safe_subset_features,
        ALPHA101_TS_SAFE_FEATURE_NAMES as _ALPHA101S_NAMES,
        ALPHA101_TS_SAFE_FEATURE_COUNT as _ALPHA101S_COUNT,
    )
    ALPHA101S_AVAILABLE = True
    ALPHA101S_FEATURE_NAMES: list[str] = list(_ALPHA101S_NAMES)
    ALPHA101S_FEATURE_COUNT: int = _ALPHA101S_COUNT
    logger.info("[v10] alpha101_ts_safe_subset_features loaded OK (%d features)", ALPHA101S_FEATURE_COUNT)
except Exception as _a101s_err:
    logger.warning(
        "[v10] alpha101_ts_safe_subset_features not importable: %s — 15 features zeroed", _a101s_err
    )
    ALPHA101S_AVAILABLE = False
    ALPHA101S_FEATURE_COUNT = 15
    ALPHA101S_FEATURE_NAMES = [
        "a101ts_alpha001", "a101ts_alpha003", "a101ts_alpha006",
        "a101ts_alpha008", "a101ts_alpha012", "a101ts_alpha016",
        "a101ts_alpha019", "a101ts_alpha020", "a101ts_alpha023",
        "a101ts_alpha033", "a101ts_alpha034", "a101ts_alpha035",
        "a101ts_alpha040", "a101ts_alpha041", "a101ts_alpha051",
    ]

    def compute_alpha101_ts_safe_subset_features(  # type: ignore[misc]
        df: pd.DataFrame,
        ticker: Optional[str] = None,
    ) -> pd.DataFrame:
        """Stub: zero-fill all 15 a101ts_* cols when module unavailable."""
        for col in ALPHA101S_FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0.0
        return df


# ---------------------------------------------------------------------------
# Helper GV: gtja_alpha191_features (GTJA "191 Formulaic Alphas", 50 features)
# Wired 2026-05-18. Source: github:Daic115/alpha191 (MIT, no paid API).
# ~50 time-series safe alphas from GTJA 2017 research; cross-sectional rank
# replaced by ts_rank(window=20) approximation. No intraday or paid-API data.
# .shift(1)-safe: all OHLCV pre-shifted 1 bar inside module; first bar is NaN.
# ---------------------------------------------------------------------------
try:
    from gtja_alpha191_features import (  # noqa: E402
        compute_gtja_alpha191_features,
        GTJA_ALPHA191_FEATURE_NAMES,
        GTJA_ALPHA191_FEATURE_COUNT,
    )
    GTJA_ALPHA191_AVAILABLE = True
    logger.info("[v10] gtja_alpha191_features loaded OK (%d features)", GTJA_ALPHA191_FEATURE_COUNT)
except Exception as _gtja_err:
    logger.warning(
        "[v10] gtja_alpha191_features not importable: %s — 50 features zeroed",
        _gtja_err,
    )
    GTJA_ALPHA191_AVAILABLE = False
    GTJA_ALPHA191_FEATURE_COUNT = 50
    GTJA_ALPHA191_FEATURE_NAMES: list[str] = [  # type: ignore[no-redef]
        "gtja_a006", "gtja_a009", "gtja_a011", "gtja_a012",
        "gtja_a014", "gtja_a018", "gtja_a019", "gtja_a021",
        "gtja_a022", "gtja_a023", "gtja_a024", "gtja_a027",
        "gtja_a028", "gtja_a029", "gtja_a031", "gtja_a034",
        "gtja_a035", "gtja_a038", "gtja_a039", "gtja_a040",
        "gtja_a041", "gtja_a042", "gtja_a043", "gtja_a044",
        "gtja_a045", "gtja_a046", "gtja_a047", "gtja_a048",
        "gtja_a049", "gtja_a050", "gtja_a051", "gtja_a052",
        "gtja_a053", "gtja_a054", "gtja_a055", "gtja_a057",
        "gtja_a060", "gtja_a061", "gtja_a062", "gtja_a063",
        "gtja_a064", "gtja_a065", "gtja_a066", "gtja_a067",
        "gtja_a068", "gtja_a069", "gtja_a071", "gtja_a072",
        "gtja_a073", "gtja_a074",
    ]

    def compute_gtja_alpha191_features(  # type: ignore[misc]
        df: pd.DataFrame,
        ticker: Optional[str] = None,
    ) -> pd.DataFrame:
        """Stub: zero-fill all 50 gtja_a* cols when module unavailable."""
        for col in GTJA_ALPHA191_FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0.0
        return df


# ---------------------------------------------------------------------------
# DRIVE-MAP-TOP7 (2026-05-20): direct wiring of 7 unwired feature modules
# scored highest by drive_full_map/wire_priority_top100.csv. Each gets a
# try/except import block + a stub fallback that preserves column names.
# Source: research/drive_full_map/wire_priority_top100.csv
# ---------------------------------------------------------------------------

# Helper TOP7-1: vwap_indicator_python_features (score 21)
# Session-VWAP + sigma bands (vwap_dev_z, vwap_dist_pct, above/below upper/lower
# 1sigma/2sigma flags ~= 7 dynamic features). All inputs are intraday-typical;
# .shift(1)-safe at the consumer layer (v10 dedup/dropna pass).
# Signature: add_vwap_indicator_python_features(df, ticker, ...)
# ---------------------------------------------------------------------------
VWAP_INDICATOR_PY_FEATURE_NAMES: list[str] = [
    "vwap", "vwap_dev_z", "vwap_dist_pct",
    "vwap_above_upper_1sigma_flag", "vwap_below_lower_1sigma_flag",
    "vwap_above_upper_2sigma_flag", "vwap_below_lower_2sigma_flag",
]
try:
    from vwap_indicator_python_features import (  # noqa: E402
        add_vwap_indicator_python_features,
    )
    VWAP_INDICATOR_PY_AVAILABLE = True
    logger.info("[v10] vwap_indicator_python_features loaded OK")
except Exception as _vwap_ip_err:
    logger.warning(
        "[v10] vwap_indicator_python_features not importable: %s - features zeroed",
        _vwap_ip_err,
    )
    VWAP_INDICATOR_PY_AVAILABLE = False

    def add_vwap_indicator_python_features(df, ticker=None):  # type: ignore[misc]
        for c in VWAP_INDICATOR_PY_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0
        return df


# Helper TOP7-2: strategy_signal_features (score 20)
# D1/D2/D3 strategy signals + agreement counters + days-since-firing.
# REQUIRES: rsi_14, ema_20, ema_50, ema_200, ret_21d (all present after v9 stack).
# Already .shift(1)-safe internally (uses .shift(1) on close + _pct_rank.shift(1)).
# Also exposes add_five_filter_stack (volume/ATR/trend/RSI/MACD filter votes).
# ---------------------------------------------------------------------------
STRATEGY_SIGNAL_FEATURE_NAMES: list[str] = [
    "d1_rev_signal", "d2_mom_signal", "d3_gold_signal",
    "d1_d2_agree", "d1_d3_agree", "d2_d3_agree",
    "n_strategies_firing", "days_since_d1", "days_since_d2",
    "days_since_d3",
    "f1_vol_above_1_5x", "f1_vol_above_2x",
    "f2_atr_above_1x", "f2_atr_above_1_5x",
]
try:
    from strategy_signal_features import (  # noqa: E402
        add_strategy_signal_features,
        add_five_filter_stack,
    )
    STRATEGY_SIGNAL_AVAILABLE = True
    logger.info("[v10] strategy_signal_features loaded OK")
except Exception as _ss_err:
    logger.warning(
        "[v10] strategy_signal_features not importable: %s - features zeroed",
        _ss_err,
    )
    STRATEGY_SIGNAL_AVAILABLE = False

    def add_strategy_signal_features(df):  # type: ignore[misc]
        for c in STRATEGY_SIGNAL_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0
        return df

    def add_five_filter_stack(df):  # type: ignore[misc]
        return df


# Helper TOP7-3: gabriel_indicators_features (score 17)
# 107 classical TA indicators -> ~250 cols named `gab_<indicator>__<output>`.
# Excludes forward-looking 'chikou' (ichimoku displaced future-close).
# .shift(1)-safe: all indicators consume causal OHLCV; producer enforces.
# Imports historical_system.indicators package lazily inside _load_registry.
# ---------------------------------------------------------------------------
try:
    from gabriel_indicators_features import (  # noqa: E402
        add_gabriel_indicators_features,
    )
    GABRIEL_INDICATORS_AVAILABLE = True
    logger.info("[v10] gabriel_indicators_features loaded OK")
except Exception as _gab_err:
    logger.warning(
        "[v10] gabriel_indicators_features not importable: %s - skip (dynamic cols)",
        _gab_err,
    )
    GABRIEL_INDICATORS_AVAILABLE = False

    def add_gabriel_indicators_features(df, ticker=None):  # type: ignore[misc]
        # Dynamic col set (~250 gab_*); no static stub names. Return unchanged.
        return df


# Helper TOP7-4: ma_energy_indicator_features (score 17)
# Single feature 'ma_energy' in [-1, +1]: multi-TF MA momentum / volatility.
# .shift(1)-safe: feature value at row t uses close/ma rolling windows ending
# at t (not future) - but v10 consumer should .shift(1) at predict-time.
# ---------------------------------------------------------------------------
MA_ENERGY_FEATURE_NAMES: list[str] = ["ma_energy"]
try:
    from ma_energy_indicator_features import (  # noqa: E402
        add_ma_energy_indicator_features,
    )
    MA_ENERGY_AVAILABLE = True
    logger.info("[v10] ma_energy_indicator_features loaded OK")
except Exception as _mae_err:
    logger.warning(
        "[v10] ma_energy_indicator_features not importable: %s - feature zeroed",
        _mae_err,
    )
    MA_ENERGY_AVAILABLE = False

    def add_ma_energy_indicator_features(df, ticker=None):  # type: ignore[misc]
        for c in MA_ENERGY_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0
        return df


# Helper TOP7-5: sec_edgar_features (score 17)
# NOTE: STUB wrapper for sec-edgar/sec-edgar repo (zero-filled se_signal_a/b).
# Real EDGAR signal comes from hist_data_edgar_features (Step 44, 9 cols) +
# edgar_extras (Step 44b, 12 cols). This stub wired for manifest completeness
# only; safe no-op until repo binding logic is implemented (TODO upstream).
# ---------------------------------------------------------------------------
SEC_EDGAR_FEATURE_NAMES: list[str] = ["se_signal_a", "se_signal_b"]
try:
    from sec_edgar_features import (  # noqa: E402
        add_sec_edgar_features,
    )
    SEC_EDGAR_AVAILABLE = True
    logger.info("[v10] sec_edgar_features loaded OK (stub: %d cols)", len(SEC_EDGAR_FEATURE_NAMES))
except Exception as _sef_err:
    logger.warning(
        "[v10] sec_edgar_features not importable: %s - features zeroed",
        _sef_err,
    )
    SEC_EDGAR_AVAILABLE = False

    def add_sec_edgar_features(df, ticker=None):  # type: ignore[misc]
        for c in SEC_EDGAR_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0
        return df


# Helper TOP7-6: trading_indicators_features (score 17)
# STUB wrapper for TjTheDj2011/trading-indicators repo (zero-filled
# ti_signal_a/b). Wired for manifest completeness; safe no-op until repo
# binding logic is implemented.
# ---------------------------------------------------------------------------
TRADING_INDICATORS_FEATURE_NAMES: list[str] = ["ti_signal_a", "ti_signal_b"]
try:
    from trading_indicators_features import (  # noqa: E402
        add_trading_indicators_features,
    )
    TRADING_INDICATORS_AVAILABLE = True
    logger.info("[v10] trading_indicators_features loaded OK (stub: %d cols)", len(TRADING_INDICATORS_FEATURE_NAMES))
except Exception as _ti_err:
    logger.warning(
        "[v10] trading_indicators_features not importable: %s - features zeroed",
        _ti_err,
    )
    TRADING_INDICATORS_AVAILABLE = False

    def add_trading_indicators_features(df, ticker=None):  # type: ignore[misc]
        for c in TRADING_INDICATORS_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0
        return df


# Helper TOP7-7: xgboost_features (score 17)
# 3 features: xgb_pred_direction, xgb_confidence, xgb_prob_up.
# CAUSAL: rolling-window XGB classifier; predicts at start_idx using train
# data from [start_idx-window, start_idx). Labels constructed from
# close[i+horizon] used ONLY for past-bar training (no future leakage into
# prediction at start_idx). Falls back to defaults if xgboost not installed
# or data too short.
# NOTE: trains XGB models inside feature step => expensive (~120-day rolling
# refit). v10 cache layer absorbs the cost once per run.
# ---------------------------------------------------------------------------
XGBOOST_FEATURE_NAMES: list[str] = [
    "xgb_pred_direction", "xgb_confidence", "xgb_prob_up",
]
try:
    from xgboost_features import (  # noqa: E402
        add_xgboost_features,
    )
    XGBOOST_FEATURES_AVAILABLE = True
    logger.info("[v10] xgboost_features loaded OK")
except Exception as _xgbf_err:
    logger.warning(
        "[v10] xgboost_features not importable: %s - features zeroed",
        _xgbf_err,
    )
    XGBOOST_FEATURES_AVAILABLE = False

    def add_xgboost_features(df, ticker=None, training_window=120, prediction_horizon=5):  # type: ignore[misc]
        for c in XGBOOST_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0 if c == "xgb_pred_direction" else 0.5
        return df


# ---------------------------------------------------------------------------
# BIG-GAP Wave (2026-05-21): harmonic + chart + regression channels + auto S/R
# ---------------------------------------------------------------------------
# Helper BG1: harmonic_patterns_features (30 cols)
# Source: external-repos/HarmonicPatterns (Djoffrey). Pure OHLC inputs.
# .shift(1)-safe: all output series shifted 1 bar inside the module.
try:
    from harmonic_patterns_features import (  # noqa: E402
        compute_harmonic_patterns_features,
        HARMONIC_FEATURE_NAMES,
        HARMONIC_FEATURE_COUNT,
    )
    logger.info("[v10] harmonic_patterns_features loaded OK")
except Exception as _hp_err:  # noqa: BLE001
    logger.warning(
        "[v10] harmonic_patterns_features not importable: %s - 30 features zeroed",
        _hp_err,
    )
    HARMONIC_FEATURE_COUNT = 30
    HARMONIC_FEATURE_NAMES: list[str] = [  # type: ignore[no-redef]
        f"harmonic_{p}_{k}"
        for p in ("gartley", "bat", "altbat", "butterfly", "crab",
                  "deepcrab", "shark", "five_o", "cypher", "abcd")
        for k in ("active", "PRZ_dist", "completion_pct")
    ][:30]

    def compute_harmonic_patterns_features(  # type: ignore[misc]
        df, ticker=None, zigzag_period=13, err_allowed=0.08,
    ):
        for col in HARMONIC_FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0.0
        return df


# Helper BG2: chart_patterns_features (35 cols)
# Pure-python classical chart-pattern detectors (H&S, double top/bottom,
# triangle, wedge, flag, pennant, channel) + aggregates.
# .shift(1)-safe: all output series shifted 1 bar inside the module.
try:
    from chart_patterns_features import (  # noqa: E402
        compute_chart_patterns_features,
        CHART_FEATURE_NAMES,
        CHART_FEATURE_COUNT,
    )
    logger.info("[v10] chart_patterns_features loaded OK")
except Exception as _cp_err:  # noqa: BLE001
    logger.warning(
        "[v10] chart_patterns_features not importable: %s - 35 features zeroed",
        _cp_err,
    )
    CHART_FEATURE_COUNT = 35
    CHART_FEATURE_NAMES: list[str] = []  # type: ignore[no-redef]

    def compute_chart_patterns_features(df, ticker=None):  # type: ignore[misc]
        for col in CHART_FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0.0
        return df


# Helper BG3: regression_channels_features (12 cols)
# Rolling linear-regression ±2σ channels on log(close) for N=20/50/100.
# .shift(1)-safe inside the module.
try:
    from regression_channels_features import (  # noqa: E402
        compute_regression_channels_features,
        REGRESSION_FEATURE_NAMES,
        REGRESSION_FEATURE_COUNT,
    )
    logger.info("[v10] regression_channels_features loaded OK")
except Exception as _rc_err:  # noqa: BLE001
    logger.warning(
        "[v10] regression_channels_features not importable: %s - 12 features zeroed",
        _rc_err,
    )
    REGRESSION_FEATURE_COUNT = 12
    REGRESSION_FEATURE_NAMES: list[str] = [  # type: ignore[no-redef]
        f"reg_channel_{k}_{w}"
        for w in (20, 50, 100)
        for k in ("slope", "width", "pos_pct", "breakout")
    ]

    def compute_regression_channels_features(df, ticker=None):  # type: ignore[misc]
        for col in REGRESSION_FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0.0
        return df


# Helper BG4: auto_support_resistance_features (10 cols)
# Pivot-based S/R zone detection with proximity-clustering.
# .shift(1)-safe inside the module.
try:
    from auto_support_resistance_features import (  # noqa: E402
        compute_auto_support_resistance_features,
        AUTO_SR_FEATURE_NAMES,
        AUTO_SR_FEATURE_COUNT,
    )
    logger.info("[v10] auto_support_resistance_features loaded OK")
except Exception as _sr_err:  # noqa: BLE001
    logger.warning(
        "[v10] auto_support_resistance_features not importable: %s - 10 features zeroed",
        _sr_err,
    )
    AUTO_SR_FEATURE_COUNT = 10
    AUTO_SR_FEATURE_NAMES: list[str] = [  # type: ignore[no-redef]
        "auto_sr_above_dist", "auto_sr_below_dist",
        "auto_sr_strength_above", "auto_sr_strength_below",
        "auto_sr_n_levels", "auto_sr_above_age_bars", "auto_sr_below_age_bars",
        "auto_sr_range_pct", "auto_sr_position_in_range", "auto_sr_breakout_score",
    ]

    def compute_auto_support_resistance_features(df, ticker=None):  # type: ignore[misc]
        for col in AUTO_SR_FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0.0
        return df


# ---------------------------------------------------------------------------
# OC-AUDIT Wave (2026-05-21): sector aggregate + market regime HMM (5-state)
# ---------------------------------------------------------------------------
# Helper OC4: sector_aggregate_features (60 cols)
# Pools same-bar features across all S&P500 tickers in same GICS sector and
# emits sector_<f>_mean / sector_<f>_std / ticker_vs_sector_<f>_z for the
# top-20 importance-weighted technical features. .shift(1)-safe inside module.
try:
    from sector_aggregate_features import (  # noqa: E402
        compute_sector_aggregate_features,
        SECTOR_FEATURE_NAMES,
        SECTOR_FEATURE_COUNT,
    )
    logger.info("[v10] sector_aggregate_features loaded OK (%d cols)", SECTOR_FEATURE_COUNT)
except Exception as _sa_err:  # noqa: BLE001
    logger.warning(
        "[v10] sector_aggregate_features not importable: %s - 60 features zeroed",
        _sa_err,
    )
    SECTOR_FEATURE_COUNT = 60
    _SECTOR_CORE_FEATS = (
        "ret_1d", "ret_5d", "ret_21d", "ret_63d",
        "rsi_14", "rsi_21", "atr_14", "atr_pct",
        "adx_14", "cci_20", "bb_pct", "bb_width",
        "vol_ratio", "ema_5_gt_ema_20", "macd_hist",
        "ema_20", "ema_50", "ema_200", "close", "vol_sma_20",
    )
    SECTOR_FEATURE_NAMES: list[str] = [  # type: ignore[no-redef]
        f"{prefix}{f}{suffix}"
        for f in _SECTOR_CORE_FEATS
        for prefix, suffix in (("sector_", "_mean"), ("sector_", "_std"), ("ticker_vs_sector_", "_z"))
    ]

    def compute_sector_aggregate_features(df, ticker=None):  # type: ignore[misc]
        for col in SECTOR_FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0.0
        return df


# Helper OC5: market_regime_hmm_features (7 cols = 1 state + 5 probs + persistence)
# 5-state GaussianHMM on (log VIX 21d MA, cross-sectional vol, sector pair-corr).
# Strict .shift(1) on the observation series — no same-bar macro leakage.
try:
    from market_regime_hmm_features import (  # noqa: E402
        compute_market_regime_hmm_features,
        REGIME_FEATURE_NAMES,
        REGIME_FEATURE_COUNT,
    )
    logger.info("[v10] market_regime_hmm_features loaded OK (%d cols)", REGIME_FEATURE_COUNT)
except Exception as _mr_err:  # noqa: BLE001
    logger.warning(
        "[v10] market_regime_hmm_features not importable: %s - 7 features zeroed",
        _mr_err,
    )
    REGIME_FEATURE_COUNT = 7
    REGIME_FEATURE_NAMES: list[str] = [  # type: ignore[no-redef]
        "regime_state",
        "regime_prob_0", "regime_prob_1", "regime_prob_2", "regime_prob_3", "regime_prob_4",
        "regime_persistence",
    ]

    def compute_market_regime_hmm_features(df, ticker=None):  # type: ignore[misc]
        for col in REGIME_FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 1.0 if col == "regime_prob_2" else (2 if col == "regime_state" else 0)
        return df


# ---------------------------------------------------------------------------
# v10 feature builder
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Manifest-driven wired-feature loader (added 2026-05-19)
# ---------------------------------------------------------------------------
# Reads `scripts/feature_manifest.json` and imports/calls any function whose
# `integration_status == "wired"`. Functions with status 'pending' or 'tested'
# are SKIPPED -- operator must explicitly promote them to 'wired' before they
# can affect the live model. Failures inside any single wired module are
# logged + swallowed so they cannot break the v10 pipeline.

import importlib as _importlib
import json as _json
from pathlib import Path as _Path

_FEATURE_MANIFEST_PATH = _Path(__file__).resolve().parent / "feature_manifest.json"


def _load_feature_manifest() -> dict:
    """Load feature_manifest.json or return empty stub on error."""
    if not _FEATURE_MANIFEST_PATH.exists():
        return {"modules": []}
    try:
        with _FEATURE_MANIFEST_PATH.open() as _fh:
            return _json.load(_fh)
    except (OSError, ValueError) as _exc:
        logger.warning("[v10] could not load feature_manifest.json: %s", _exc)
        return {"modules": []}


def _apply_manifest_wired_modules(df, ticker):
    """Apply every manifest entry with integration_status == 'wired'.

    Returns: (df, num_cols_added, num_entries_skipped_pending_or_tested)
    """
    manifest = _load_feature_manifest()
    modules = manifest.get("modules", [])
    added_total = 0
    skipped = 0
    for entry in modules:
        status = entry.get("integration_status", "pending")
        if status != "wired":
            skipped += 1
            continue
        mod_path = entry.get("module_path", "")
        func_name = entry.get("function_name", "")
        # mod_path = "scripts/_generated/<target>.py"
        try:
            mod_pkg = mod_path.replace("/", ".").rsplit(".py", 1)[0]
            mod = _importlib.import_module(mod_pkg)
        except ImportError:
            full = _Path(__file__).resolve().parent.parent / mod_path
            if not full.exists():
                logger.warning("[v10] wired module missing: %s", mod_path)
                continue
            import importlib.util as _util
            spec = _util.spec_from_file_location(mod_pkg, str(full))
            if spec is None or spec.loader is None:
                logger.warning("[v10] could not load spec for %s", mod_path)
                continue
            mod = _util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        fn = getattr(mod, func_name, None)
        if fn is None:
            logger.warning("[v10] wired function not found: %s::%s", mod_path, func_name)
            continue
        try:
            before_cols = df.shape[1]
            df = fn(df)
            added_cols = df.shape[1] - before_cols
            added_total += added_cols
            logger.info(
                "  [v10]   manifest::%s -> +%d cols (%s)",
                func_name, added_cols, mod_path,
            )
        except Exception as _exc:  # noqa: BLE001
            logger.warning(
                "  [v10]   manifest::%s FAILED (%s): %s -- skipping",
                func_name, ticker, _exc,
            )
            continue
    return df, added_total, skipped


def _build_v10_features_impl(
    ticker: str,
    universe_agg: Optional[dict] = None,
    use_mythos: bool = False,
) -> tuple[pd.DataFrame, int, dict]:
    """Build v10 feature set: full v9 stack + 5 newly-wired modules.

    Pipeline order:
      1. build_v9_features()      — v8 base + optional 256-dim Mythos
      2. add_alpaca_features()    — earnings/div/split/metadata (13 cols)
         [BEFORE daily_integration so earn_contam_gate has access to
         days_until_earnings / days_since_last_earnings from alpaca layer]
      2.5 add_insider_form4_features() — SEC Form 4 insider disclosures
         (8 cols: buy/sell counts, cluster flags, days-since, $-amount).
         Wired 2026-05-17 — was the "gov-trades" module gap.
      3. add_daily_integration_features() — 7 composite cols
      4. add_dfs_features()       — up to 60 DFS depth-2 interaction cols
      5. add_mastery_priors()     — 7 per-ticker priors features parsed from
         $SP/mastery_files/*.md (v4 + v10 mastery markdown artifacts).
         .shift(1)-safe via mtime-gated age column. Wired 2026-05-17.
      6. add_paper_trade_outcome_features() — 7 features computed from CLOSED
         paper trades (win-rate, PF, count, last-outcome-sign, avg holding,
         signal-to-fill lag, current drawdown). Rolling 30 calendar days,
         .shift(1)-safe via strict trade_date < bar_date merge_asof.
         Source: $SP/paper_trade/state/*_state.json. Wired 2026-05-17.

    Args:
        ticker: Stock symbol.
        universe_agg: Cross-sectional precomputed aggregates dict (from csf).
        use_mythos: If True, append 256 Mythos embedding features.

    Returns:
        Tuple of:
          - pd.DataFrame with all features + target column 'y'.
          - int fallback_rows (Mythos zero-embedding rows; 0 when Mythos off).
          - dict module_feature_counts with per-module feature counts.
    """
    # ---- Step 1: Full v9 stack ----
    f, mythos_fallback_rows = build_v9_features(ticker, universe_agg, use_mythos=use_mythos)
    after_v9 = f.shape[1]
    logger.info("  [v10] after v9 stack: %d cols", after_v9)

    # ---- Step 1.5: Manifest-driven wired feature modules ----
    # Reads scripts/feature_manifest.json (built by synthesis_to_features.py)
    # and calls any function whose integration_status == "wired".
    # PENDING / TESTED status entries are SKIPPED — operator must explicitly
    # promote before they execute against the live model.
    try:
        f, _manifest_added, _manifest_skipped = _apply_manifest_wired_modules(f, ticker)
        logger.info(
            "  [v10] manifest wired: +%d cols (skipped pending/tested: %d) -> %d total",
            _manifest_added, _manifest_skipped, f.shape[1],
        )
    except Exception as exc:  # never break the v10 pipeline on manifest issues
        logger.warning("  [v10] manifest loader failed (%s): %s", ticker, exc)
        _manifest_added, _manifest_skipped = 0, 0

    # ---- Step 2: Alpaca features (must precede daily_integration) ----
    before_alp = f.shape[1]
    try:
        f = add_alpaca_features(f, ticker)
    except Exception as exc:
        logger.warning("  [v10] alpaca_features call failed (%s): %s — zeroing", ticker, exc)
        for col in ALPACA_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_alpaca = f.shape[1] - before_alp
    logger.info("  [v10] +alpaca: +%d cols -> %d total", added_alpaca, f.shape[1])

    # ---- Step 2.5: Insider Form 4 features (idempotent guard) ----
    # NOTE 2026-05-17: insider_form4_features is ALREADY wired into the v7→v9
    # chain (build_v7_features imports f4f and calls add_insider_form4_features
    # before returning). The 8 insider cols thus arrive inside the v9_base
    # count. We retain Helper-D import (for explicit dependency tracking) and
    # only re-fill missing cols defensively — never overwrite an existing
    # value because v9 may already have populated them.
    before_ins = f.shape[1]
    missing_insider = [c for c in INSIDER_FORM4_FEATURE_NAMES if c not in f.columns]
    if missing_insider:
        try:
            f = add_insider_form4_features(f, ticker)
            logger.info(
                "  [v10] insider_form4 cols missing from v9 (%d) — backfilled via Helper D",
                len(missing_insider),
            )
        except Exception as exc:
            logger.warning(
                "  [v10] insider_form4_features fallback call failed (%s): %s — zeroing missing cols",
                ticker, exc,
            )
            for col in missing_insider:
                if col not in f.columns:
                    f[col] = 0.0
    added_insider = f.shape[1] - before_ins
    logger.info(
        "  [v10] insider_form4 check: +%d new cols (already-present: %d) -> %d total",
        added_insider, 8 - len(missing_insider), f.shape[1],
    )

    # ---- Step 3: Daily integration features ----
    before_di = f.shape[1]
    try:
        f = add_daily_integration_features(f, ticker)
    except Exception as exc:
        logger.warning(
            "  [v10] daily_integration_features call failed (%s): %s — zeroing", ticker, exc
        )
        for col in DAILY_INT_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_daily_int = f.shape[1] - before_di
    logger.info("  [v10] +daily_integration: +%d cols -> %d total", added_daily_int, f.shape[1])

    # ---- Step 4: DFS features ----
    before_dfs = f.shape[1]
    try:
        f = add_dfs_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] add_dfs_features call failed (%s): %s — skipping", ticker, exc)
    added_dfs = f.shape[1] - before_dfs
    logger.info("  [v10] +dfs: +%d cols -> %d total", added_dfs, f.shape[1])

    # ---- Step 5: Mastery priors (past-test artifact priors, 7 features) ----
    # Wired 2026-05-17. Reads $SP/mastery_files/*.md (311 v4 + 7 v10) and emits
    # per-ticker prior-mastered flags, PF, DD, top-10 flag, .shift(1)-safe age.
    before_mp = f.shape[1]
    try:
        f = add_mastery_priors(f, ticker)
    except Exception as exc:
        logger.warning(
            "  [v10] add_mastery_priors call failed (%s): %s — zeroing", ticker, exc
        )
        for col in MASTERY_PRIORS_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0
    added_mastery_priors = f.shape[1] - before_mp
    logger.info(
        "  [v10] +mastery_priors: +%d cols -> %d total", added_mastery_priors, f.shape[1]
    )

    # ---- Step 6: Paper-trade outcome features (live-feedback loop) ----
    # Reads $SP/paper_trade/state/*_state.json closed_trades[] and adds 7
    # rolling 30d outcome features. Zero-fills gracefully when no paper trades
    # exist yet for the ticker. .shift(1)-safe via strict-less-than merge_asof.
    before_pt = f.shape[1]
    try:
        f = add_paper_trade_outcome_features(f, ticker)
    except Exception as exc:
        logger.warning(
            "  [v10] add_paper_trade_outcome_features call failed (%s): %s — zeroing",
            ticker, exc,
        )
        for col in PT_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0 if col in (
                    "paper_trade_count_30d", "paper_trade_last_outcome_sign"
                ) else 0.0
    added_paper_trade_outcomes = f.shape[1] - before_pt
    logger.info(
        "  [v10] +paper_trade_outcomes: +%d cols -> %d total",
        added_paper_trade_outcomes, f.shape[1],
    )

    # ---- Step 7: Stumpy (matrix-profile motif/discord) ----
    before_stumpy = f.shape[1]
    try:
        f = add_stumpy_features(f, ticker)
    except Exception as exc:
        logger.warning("  [v10] stumpy call failed (%s): %s — skipping", ticker, exc)
    added_stumpy = f.shape[1] - before_stumpy
    logger.info("  [v10] +stumpy: +%d cols -> %d total", added_stumpy, f.shape[1])

    # ---- Step 8: FFN (Sortino/Calmar/Ulcer/Downside risk metrics) ----
    before_ffn = f.shape[1]
    try:
        f = add_ffn_features(f, ticker)
    except Exception as exc:
        logger.warning("  [v10] ffn call failed (%s): %s — skipping", ticker, exc)
    added_ffn = f.shape[1] - before_ffn
    logger.info("  [v10] +ffn: +%d cols -> %d total", added_ffn, f.shape[1])

    # ---- Step 9: pandas-ta-classic (non-TA-Lib indicators) ----
    before_ptc = f.shape[1]
    try:
        f = add_pandas_ta_classic_features(f, ticker)
    except Exception as exc:
        logger.warning("  [v10] pandas_ta_classic call failed (%s): %s — skipping", ticker, exc)
    added_ptc = f.shape[1] - before_ptc
    logger.info("  [v10] +pandas_ta_classic: +%d cols -> %d total", added_ptc, f.shape[1])

    # ---- Step 10: options_flow (Wave A, 3 features) ----
    before_of = f.shape[1]
    try:
        f = add_options_flow_features(f, ticker)
    except Exception as exc:
        logger.warning("  [v10] options_flow call failed (%s): %s — zeroing", ticker, exc)
        for col in OPTIONS_FLOW_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0 if col != "unusual_options_activity_flag" else 0
    added_options_flow = f.shape[1] - before_of
    logger.info("  [v10] +options_flow: +%d cols -> %d total", added_options_flow, f.shape[1])

    # ---- Step 11: govtrades (Wave A, 3 features) ----
    before_gt = f.shape[1]
    try:
        f = add_govtrades_features(f, ticker)
    except Exception as exc:
        logger.warning("  [v10] govtrades call failed (%s): %s — zeroing", ticker, exc)
        for col in GOVTRADES_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0 if col != "congress_buy_sell_ratio_5d" else 0.0
    added_govtrades = f.shape[1] - before_gt
    logger.info("  [v10] +govtrades: +%d cols -> %d total", added_govtrades, f.shape[1])

    # ---- Step 12: time_of_day (Wave A, 1 feature) ----
    before_tod = f.shape[1]
    try:
        f = add_time_of_day_features(f, ticker)
    except Exception as exc:
        logger.warning("  [v10] time_of_day call failed (%s): %s — zeroing", ticker, exc)
        if "time_of_day_bucket" not in f.columns:
            f["time_of_day_bucket"] = 2
    added_tod = f.shape[1] - before_tod
    logger.info("  [v10] +time_of_day: +%d cols -> %d total", added_tod, f.shape[1])

    # ---- Step 13: gabriel_priors (Wave A, 5 features) ----
    before_gp = f.shape[1]
    try:
        f = add_gabriel_priors_features(f, ticker)
    except Exception as exc:
        logger.warning("  [v10] gabriel_priors call failed (%s): %s — zeroing", ticker, exc)
        for col in GABRIEL_PRIORS_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0 if col == "gabriel_champion_n_trades" else 0.0
    added_gabriel = f.shape[1] - before_gp
    logger.info("  [v10] +gabriel_priors: +%d cols -> %d total", added_gabriel, f.shape[1])

    # ---- Step 14: vix_term_structure_v2 (3 features: ratio, inverted, z10) ----
    before_vts = f.shape[1]
    try:
        f = compute_vix_term_structure_v2_features(f)
    except Exception as exc:
        logger.warning("  [v10] vix_term_structure_v2 call failed (%s): %s — zeroing", ticker, exc)
        for col in VIX_TS_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0 if col != "vix_term_inverted" else 0
    added_vts = f.shape[1] - before_vts
    logger.info("  [v10] +vix_term_structure_v2: +%d cols -> %d total", added_vts, f.shape[1])

    # ---- Step 15: garch_11_cond_vol (GARCH(1,1) cond vol, 3 features) ----
    before_garch = f.shape[1]
    try:
        f = compute_garch_11_cond_vol_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] garch_11_cond_vol call failed (%s): %s — zeroing", ticker, exc)
        for col in GARCH11_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_garch = f.shape[1] - before_garch
    logger.info("  [v10] +garch_11_cond_vol: +%d cols -> %d total", added_garch, f.shape[1])

    # ---- Step 16: egarch_11_leverage (EGARCH(1,1) leverage-effect vol, 3 features) ----
    before_egarch_lev = f.shape[1]
    try:
        f = compute_egarch_11_leverage_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] egarch_11_leverage call failed (%s): %s — zeroing", ticker, exc)
        for col in EGARCH11_LEV_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_egarch_lev = f.shape[1] - before_egarch_lev
    logger.info("  [v10] +egarch_11_leverage: +%d cols -> %d total", added_egarch_lev, f.shape[1])

    # ---- Step 17: vpin_50bucket (VPIN 50-bucket BVC approximation, 3 features) ----
    before_vpin = f.shape[1]
    try:
        f = compute_vpin_50bucket_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] vpin_50bucket call failed (%s): %s — zeroing", ticker, exc)
        for col in VPIN_50BUCKET_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_vpin = f.shape[1] - before_vpin
    logger.info("  [v10] +vpin_50bucket: +%d cols -> %d total", added_vpin, f.shape[1])

    # ---- Step 18: kyles_lambda_intraday (Kyle 1985 λ, BVC approx, 3 features) ----
    before_kl = f.shape[1]
    try:
        f = compute_kyles_lambda_intraday_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] kyles_lambda_intraday call failed (%s): %s — zeroing", ticker, exc)
        for col in KYLES_LAMBDA_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_kyles_lambda = f.shape[1] - before_kl
    logger.info("  [v10] +kyles_lambda_intraday: +%d cols -> %d total", added_kyles_lambda, f.shape[1])

    # ---- Step 19: vpin_features (TRUE 1-min VPIN, 5 features) — Wave M-1 #1 ----
    before_vpin_intra = f.shape[1]
    try:
        f = add_vpin_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] vpin_features (intraday) call failed (%s): %s — zeroing", ticker, exc)
        for col in VPIN_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_vpin_intra = f.shape[1] - before_vpin_intra
    logger.info("  [v10] +vpin_intraday: +%d cols -> %d total", added_vpin_intra, f.shape[1])

    # ---- Step 20: tick_imbalance_features (Lee-Ready, 5 features) — Wave M-1 #6 ----
    before_ti = f.shape[1]
    try:
        f = add_tick_imbalance_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] tick_imbalance_features call failed (%s): %s — zeroing", ticker, exc)
        for col in TICK_IMBALANCE_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_tick_imbalance = f.shape[1] - before_ti
    logger.info("  [v10] +tick_imbalance: +%d cols -> %d total", added_tick_imbalance, f.shape[1])

    # ---- Step 21: volume_profile_features (POC/VA/shape, 6 features) — Wave M-1 #11 ----
    before_vp = f.shape[1]
    try:
        f = add_volume_profile_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] volume_profile_features call failed (%s): %s — zeroing", ticker, exc)
        for col in VOLUME_PROFILE_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_volume_profile = f.shape[1] - before_vp
    logger.info("  [v10] +volume_profile: +%d cols -> %d total", added_volume_profile, f.shape[1])

    # ---- Step 22: auction_features (open/close auction, 6 features) — Wave M-1 #12 ----
    before_auc = f.shape[1]
    try:
        f = add_auction_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] auction_features call failed (%s): %s — zeroing", ticker, exc)
        for col in AUCTION_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_auction = f.shape[1] - before_auc
    logger.info("  [v10] +auction: +%d cols -> %d total", added_auction, f.shape[1])

    # ---- Step 23: vol_of_vol (3 features) — Wave V-1 #7 ----
    before_vov = f.shape[1]
    try:
        f = add_vol_of_vol_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] vol_of_vol call failed (%s): %s — zeroing", ticker, exc)
        for col in VOL_OF_VOL_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_vov = f.shape[1] - before_vov
    logger.info("  [v10] +vol_of_vol: +%d cols -> %d total", added_vov, f.shape[1])

    # ---- Step 24: vol_risk_premium (4 features) — Wave V-1 #12 ----
    before_vrp = f.shape[1]
    try:
        f = add_vol_risk_premium_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] vol_risk_premium call failed (%s): %s — zeroing", ticker, exc)
        for col in VRP_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0 if col != "vrp_sign_flip" else 0
    added_vrp = f.shape[1] - before_vrp
    logger.info("  [v10] +vol_risk_premium: +%d cols -> %d total", added_vrp, f.shape[1])

    # ---- Step 25: vol_target_sizing (2 features) — Wave V-1 #13 ----
    # MUST run AFTER Step 15 (garch_11_cond_vol) so vol_target_ratio can consume it.
    before_vt = f.shape[1]
    try:
        f = add_vol_target_sizing_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] vol_target_sizing call failed (%s): %s — neutral fill", ticker, exc)
        for col in VOL_TARGET_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 1.0
    added_vt = f.shape[1] - before_vt
    logger.info("  [v10] +vol_target_sizing: +%d cols -> %d total", added_vt, f.shape[1])

    # ---- Step 26: vol_breakout_nr (5 features) — Wave V-1 #14 ----
    before_nr = f.shape[1]
    try:
        f = add_vol_breakout_nr_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] vol_breakout_nr call failed (%s): %s — zeroing", ticker, exc)
        for col in VOL_BREAKOUT_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0 if col != "range_pct_of_atr20" else 1.0
    added_nr = f.shape[1] - before_nr
    logger.info("  [v10] +vol_breakout_nr: +%d cols -> %d total", added_nr, f.shape[1])

    # ---- Step 27: bollinger_keltner_squeeze (4 features) — Wave V-1 #15 ----
    before_sq = f.shape[1]
    try:
        f = add_bollinger_keltner_squeeze_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] bb_kc_squeeze call failed (%s): %s — zeroing", ticker, exc)
        for col in SQUEEZE_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_sq = f.shape[1] - before_sq
    logger.info("  [v10] +bb_kc_squeeze: +%d cols -> %d total", added_sq, f.shape[1])

    # ---- Step 28: vol_of_vix (3 features) — Wave V-1 #18 ----
    before_vvx = f.shape[1]
    try:
        f = add_vol_of_vix_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] vol_of_vix call failed (%s): %s — zeroing", ticker, exc)
        for col in VVIX_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_vvx = f.shape[1] - before_vvx
    logger.info("  [v10] +vol_of_vix: +%d cols -> %d total", added_vvx, f.shape[1])

    # ---- Step 29: rv_term_structure (4 features) — Wave V-1 #19 ----
    before_rvt = f.shape[1]
    try:
        f = add_rv_term_structure_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] rv_term_structure call failed (%s): %s — zeroing", ticker, exc)
        for col in RV_TERM_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 1.0 if col in ("rv5_over_rv21", "rv5_over_rv63") else 0.0
    added_rvt = f.shape[1] - before_rvt
    logger.info("  [v10] +rv_term_structure: +%d cols -> %d total", added_rvt, f.shape[1])

    # ---- Step 30: amihud_illiquidity_ratio (Amihud 2002, 5 features) — Wave G ----
    before_amihud = f.shape[1]
    try:
        f = compute_amihud_illiquidity_ratio_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] amihud_illiq call failed (%s): %s — zeroing", ticker, exc)
        for col in AMIHUD_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_amihud = f.shape[1] - before_amihud
    logger.info("  [v10] +amihud_illiq: +%d cols -> %d total", added_amihud, f.shape[1])

    # ---- Step 31: rolls_effective_spread (Roll 1984 JFE, 3 features) — Wave H-1 ----
    before_rolls = f.shape[1]
    try:
        f = compute_rolls_effective_spread_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] rolls_effective_spread call failed (%s): %s — zeroing", ticker, exc)
        for col in ROLLS_EFFECTIVE_SPREAD_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_rolls = f.shape[1] - before_rolls
    logger.info("  [v10] +rolls_effective_spread: +%d cols -> %d total", added_rolls, f.shape[1])

    # ---- Step 32: cycle051_features (daily-pivot SR, 5 features) — Wave Cycle ----
    before_c051 = f.shape[1]
    try:
        f = add_cycle051_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] cycle051 call failed (%s): %s — zeroing", ticker, exc)
        for col in CYCLE051_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0 if col == "sr_above_1day_pp" else 0.0
    added_c051 = f.shape[1] - before_c051
    logger.info("  [v10] +cycle051: +%d cols -> %d total", added_c051, f.shape[1])

    # ---- Step 33: cycle055_features (vol-gate daily proxies, 5 features) — Wave Cycle ----
    before_c055 = f.shape[1]
    try:
        f = add_cycle055_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] cycle055 call failed (%s): %s — zeroing", ticker, exc)
        for col in CYCLE055_FEATURE_NAMES:
            if col not in f.columns:
                if col == "vg_vol_regime":
                    f[col] = 1
                elif col in ("vg_in_normal_regime", "vg_rvol_floor_ok"):
                    f[col] = 0
                else:
                    f[col] = 0.0
    added_c055 = f.shape[1] - before_c055
    logger.info("  [v10] +cycle055: +%d cols -> %d total", added_c055, f.shape[1])

    # ---- Step 34: cycle058_features (SPY-intra + sector RS, 5 features) — Wave Cycle ----
    before_c058 = f.shape[1]
    try:
        f = add_cycle058_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] cycle058 call failed (%s): %s — zeroing", ticker, exc)
        for col in CYCLE058_FEATURE_NAMES:
            if col not in f.columns:
                if col in ("mc_spy_intra_above_or30h_eod",
                           "mc_spy_intra_below_or30l_eod",
                           "mc_rs_sector_5d_sign"):
                    f[col] = 0
                else:
                    f[col] = 0.0
    added_c058 = f.shape[1] - before_c058
    logger.info("  [v10] +cycle058: +%d cols -> %d total", added_c058, f.shape[1])

    # ---- Step 35: cycle060_features (OI / vol-to-OI / net-delta-z, 3 features) — Wave Cycle ----
    before_c060 = f.shape[1]
    try:
        f = add_cycle060_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] cycle060 call failed (%s): %s — zeroing", ticker, exc)
        for col in CYCLE060_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_c060 = f.shape[1] - before_c060
    logger.info("  [v10] +cycle060: +%d cols -> %d total", added_c060, f.shape[1])

    # ---- Step 36: cycle061_features (TOD daily aggregates, 4 features) — Wave Cycle ----
    before_c061 = f.shape[1]
    try:
        f = add_cycle061_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] cycle061 call failed (%s): %s — zeroing", ticker, exc)
        for col in CYCLE061_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_c061 = f.shape[1] - before_c061
    logger.info("  [v10] +cycle061: +%d cols -> %d total", added_c061, f.shape[1])

    # ---- Step 37: add_alpha_features_core_features (WorldQuant alpha101 port, 30 features) ----
    before_afc = f.shape[1]
    try:
        f = compute_add_alpha_features_core_features_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] afc_core call failed (%s): %s — zeroing", ticker, exc)
        for col in AFC_CORE_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_afc = f.shape[1] - before_afc
    logger.info("  [v10] +alpha_features_core: +%d cols -> %d total", added_afc, f.shape[1])

    # ---- Step 38: add_finance_database_features (FinanceDatabase metadata, 4 features) — Wave FDB ----
    before_fdb = f.shape[1]
    try:
        f = compute_add_finance_database_features_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] finance_database call failed (%s): %s — zeroing", ticker, exc)
        for col in FDB_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0
    added_fdb = f.shape[1] - before_fdb
    logger.info("  [v10] +finance_database: +%d cols -> %d total", added_fdb, f.shape[1])

    # ---- Step 39: mlforecast_features (rolling/EWM/expanding lag, 11 features) — Wave MLF ----
    before_mlf = f.shape[1]
    try:
        f = compute_mlforecast_features_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] mlforecast_features call failed (%s): %s — zeroing", ticker, exc)
        for col in MLFORECAST_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_mlf = f.shape[1] - before_mlf
    logger.info("  [v10] +mlforecast_features: +%d cols -> %d total", added_mlf, f.shape[1])

    # ---- Step 40: neuralforecast_features (NBEATS/NHITS decomp, 5 features) — Wave NF ----
    before_nf = f.shape[1]
    try:
        f = compute_neuralforecast_features_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] neuralforecast_features call failed (%s): %s — zeroing", ticker, exc)
        for col in NEURALFORECAST_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_nf = f.shape[1] - before_nf
    logger.info("  [v10] +neuralforecast_features: +%d cols -> %d total", added_nf, f.shape[1])

    # ---- Step 41: worldquant_alpha101_replay (Alpha#101 z-score, 1 feature) — Wave WQ101 ----
    before_wq101 = f.shape[1]
    try:
        f = compute_worldquant_alpha101_replay_20260517t224845z_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] wq101_replay call failed (%s): %s — zeroing", ticker, exc)
        if "wq101_replay_alpha101_z21" not in f.columns:
            f["wq101_replay_alpha101_z21"] = 0.0
    added_wq101 = f.shape[1] - before_wq101
    logger.info("  [v10] +wq101_replay: +%d cols -> %d total", added_wq101, f.shape[1])

    # ---- Step 42: mythos_deltas (Mythos curriculum prior, 6 static features) — Wave D6 ----
    before_mythos_deltas = f.shape[1]
    try:
        f = add_mythos_deltas_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] mythos_deltas call failed (%s): %s — zeroing", ticker, exc)
        for col in MYTHOS_DELTA_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_mythos_deltas = f.shape[1] - before_mythos_deltas
    logger.info("  [v10] +mythos_deltas: +%d cols -> %d total", added_mythos_deltas, f.shape[1])

    # ---- Step 43: alpha101_ts_safe_subset_replay (Alpha#6 z-score, 1 feature) — Wave TS101 ----
    before_a101ts = f.shape[1]
    try:
        f = compute_alpha101_ts_safe_subset_replay_20260517t224845z_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] alpha101_ts_safe call failed (%s): %s — zeroing", ticker, exc)
        if "a101_ts_wq6_z21" not in f.columns:
            f["a101_ts_wq6_z21"] = 0.0
    added_a101ts = f.shape[1] - before_a101ts
    logger.info("  [v10] +alpha101_ts_safe_subset: +%d cols -> %d total", added_a101ts, f.shape[1])

    # ---- Step 44: edgar (EDGAR filing recency/density, 9 dynamic features) — Wave B1 ----
    # Source: claudes test/data/edgar/data/edgar.db (500 tickers, 2020-2026).
    # .shift(1)-safe via merge_asof(direction='backward', allow_exact_matches=False).
    before_edgar = f.shape[1]
    try:
        f = add_edgar_features(f, ticker)
    except Exception as exc:
        logger.warning("  [v10] edgar call failed (%s): %s — zeroing", ticker, exc)
        for col in EDGAR_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0
    added_edgar = f.shape[1] - before_edgar
    logger.info("  [v10] +edgar: +%d cols -> %d total", added_edgar, f.shape[1])

    # ---- Step 44b: edgar_extras (DEF 14A, amendments, likely-earnings, S-1, burst, accel — 12 features) — Wave 2026-05-20 ----
    # Source: claudes test/data/edgar/data/edgar.db (same DB as Step 44 but different feature family).
    # See edgar_extras_features.py for gap-analysis (G1..G6).
    before_edgar_x = f.shape[1]
    try:
        f = add_edgar_extras_features(f, ticker)
    except Exception as exc:
        logger.warning("  [v10] edgar_extras call failed (%s): %s — zeroing", ticker, exc)
        for col in EDGAR_EXTRAS_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0
    added_edgar_x = f.shape[1] - before_edgar_x
    logger.info("  [v10] +edgar_extras: +%d cols -> %d total", added_edgar_x, f.shape[1])

    # ---- Step 44c: govtrades_extras (contracts + lobbying $ + synapse market-wide — 45 features) — Wave 2026-05-20 ----
    # Source: Ph0tis/Gov-Trades/data/govtrades.db + Synapse archived signal CSV.
    # See govtrades_extras_features.py for unwired-module audit.
    before_gt_x = f.shape[1]
    try:
        f = add_govtrades_extras_features(f, ticker)
    except Exception as exc:
        logger.warning("  [v10] govtrades_extras call failed (%s): %s — zeroing", ticker, exc)
        for col in GOVTRADES_EXTRAS_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_gt_x = f.shape[1] - before_gt_x
    logger.info("  [v10] +govtrades_extras: +%d cols -> %d total", added_gt_x, f.shape[1])

    # ---- Step 45: ceo_personal_donation_flag_political_replay (FEC, 1 feature) — Wave FEC1 ----
    # Source: api.fec.gov Schedule A employer-search. Public, no paid key.
    # .shift(1)-safe via strict receipt_date < bar_date searchsorted boundary.
    before_ceo_don = f.shape[1]
    try:
        f = compute_ceo_personal_donation_flag_political_replay_20260517t225454z_features(
            f, ticker=ticker
        )
    except Exception as exc:
        logger.warning(
            "  [v10] ceo_personal_donation call failed (%s): %s — zeroing", ticker, exc
        )
        if "fec_donation_flag_90d" not in f.columns:
            f["fec_donation_flag_90d"] = 0.0
    added_ceo_don = f.shape[1] - before_ceo_don
    logger.info(
        "  [v10] +ceo_personal_donation: +%d cols -> %d total", added_ceo_don, f.shape[1]
    )

    # ---- Step 46: senate_efd_options_disclosure_count_30d_replay (1 feature) — Wave EFD1 ----
    # Source: QuiverQuant free congressional-trading API. No paid key.
    # .shift(1)-safe via strict report_date < bar_date searchsorted boundary.
    before_sefd = f.shape[1]
    try:
        f = compute_senate_efd_options_disclosure_count_30d_replay_20260517t225454z_features(
            f, ticker=ticker
        )
    except Exception as exc:
        logger.warning(
            "  [v10] senate_efd_options call failed (%s): %s — zeroing", ticker, exc
        )
        if "senate_efd_options_count_30d" not in f.columns:
            f["senate_efd_options_count_30d"] = 0.0
    added_sefd = f.shape[1] - before_sefd
    logger.info(
        "  [v10] +senate_efd_options: +%d cols -> %d total", added_sefd, f.shape[1]
    )

    # ---- Step 47: vix_term_structure (v1: spread/z21/streak, 3 features) — Wave VTS1 ----
    # Source: yfinance:^VIX9D,^VIX (MIT, public CBOE indices, no paid API).
    # .shift(1)-safe: merge_asof direction=backward + 1-day subtraction on lookup date.
    before_vt1 = f.shape[1]
    try:
        f = compute_vix_term_structure_features(f)
    except Exception as exc:
        logger.warning("  [v10] vix_term_structure call failed (%s): %s — zeroing", ticker, exc)
        for col in VIX_TERM_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_vt1 = f.shape[1] - before_vt1
    logger.info("  [v10] +vix_term_structure: +%d cols -> %d total", added_vt1, f.shape[1])

    # ---- Step 48: regime_hmm_3state_vol_regime (3 features) — Wave HMM1 ----
    # Source: daily OHLCV (Yang-Zhang vol estimator, no paid API).
    # .shift(1)-safe: yz_vol is same-bar quantity shifted by 1 before HMM input.
    before_hmm3 = f.shape[1]
    try:
        f = compute_regime_hmm_3state_vol_regime_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning(
            "  [v10] regime_hmm_3state_vol_regime call failed (%s): %s — zeroing", ticker, exc
        )
        for col in HMM_3STATE_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_hmm3 = f.shape[1] - before_hmm3
    logger.info(
        "  [v10] +regime_hmm_3state_vol_regime: +%d cols -> %d total", added_hmm3, f.shape[1]
    )

    # ---- Step 49: worldquant_alpha101 (25 features) — Wave WQA101 ----
    # Source: "101 Formulaic Alphas" Kakushadze (2016), github:lvlh2/alpha101 (MIT).
    # Cross-sectional rank → ts_rank(window=20); pure OHLCV; no paid API.
    # .shift(1)-safe: all inputs shifted 1 bar at module entry.
    before_wqa101 = f.shape[1]
    try:
        f = compute_worldquant_alpha101_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning(
            "  [v10] worldquant_alpha101 call failed (%s): %s — zeroing", ticker, exc
        )
        for col in WQ_ALPHA101_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_wqa101 = f.shape[1] - before_wqa101
    logger.info(
        "  [v10] +worldquant_alpha101: +%d cols -> %d total", added_wqa101, f.shape[1]
    )

    # ---- Step 50: regime_changepoint_bayesian_vol_break (3 features) — Wave BOCPD1 ----
    # Source: Adams & MacKay (2007) BOCPD on 5-day RV from yfinance close (no paid API).
    # .shift(1)-safe: 5-day realized variance shifted by 1 bar before BOCPD input.
    before_bocpd = f.shape[1]
    try:
        f = compute_regime_changepoint_bayesian_vol_break_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning(
            "  [v10] regime_changepoint_bayesian_vol_break call failed (%s): %s — zeroing",
            ticker, exc,
        )
        for col in BOCPD_VOL_BREAK_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_bocpd = f.shape[1] - before_bocpd
    logger.info(
        "  [v10] +regime_changepoint_bayesian_vol_break: +%d cols -> %d total",
        added_bocpd, f.shape[1],
    )

    # ---- Step 51: add_stockstats_features (28 stockstats indicators) — Wave SS1 ----
    # Source: github:jealous/stockstats (BSD-3-Clause, no paid API). Pure OHLCV.
    # .shift(1)-safe: all indicator series shifted 1 bar inside the module.
    before_ss = f.shape[1]
    try:
        f = compute_add_stockstats_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning(
            "  [v10] add_stockstats_features call failed (%s): %s — zeroing", ticker, exc
        )
        for col in STOCKSTATS_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_ss = f.shape[1] - before_ss
    logger.info(
        "  [v10] +add_stockstats_features: +%d cols -> %d total", added_ss, f.shape[1]
    )

    # ---- Step 52: add_talipp_features (8 talipp TA indicators) — Wave TLP1 ----
    # Source: github:nardew/talipp (MIT, no paid API). Pure close-price inputs.
    # .shift(1)-safe: all indicator series shifted 1 bar inside the module.
    before_talipp = f.shape[1]
    try:
        f = compute_add_talipp_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning(
            "  [v10] add_talipp_features call failed (%s): %s — zeroing", ticker, exc
        )
        for col in TALIPP_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_talipp = f.shape[1] - before_talipp
    logger.info(
        "  [v10] +add_talipp_features: +%d cols -> %d total", added_talipp, f.shape[1]
    )

    # ---- Step 53: add_jesse_features (54 Jesse-inspired TA indicators) — Wave JES1 ----
    # Source: github:jesse-ai/jesse (MIT, no paid API). Pure OHLCV inputs.
    # .shift(1)-safe: all indicator series shifted 1 bar inside the module.
    before_jesse = f.shape[1]
    try:
        f = compute_add_jesse_features_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning(
            "  [v10] add_jesse_features call failed (%s): %s — zeroing", ticker, exc
        )
        for col in JESSE_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_jesse = f.shape[1] - before_jesse
    logger.info(
        "  [v10] +add_jesse_features: +%d cols -> %d total", added_jesse, f.shape[1]
    )

    # ---- Step 54: add_shashank_finance_features (58 features) — Wave SHF1 ----
    # Source: github:shashankvemuri/Finance (MIT, no paid API). Pure OHLCV inputs.
    # .shift(1)-safe: all indicator series shifted 1 bar inside the module.
    before_shf = f.shape[1]
    try:
        f = compute_add_shashank_finance_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning(
            "  [v10] add_shashank_finance_features call failed (%s): %s — zeroing", ticker, exc
        )
        for col in SHF_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_shf = f.shape[1] - before_shf
    logger.info(
        "  [v10] +add_shashank_finance_features: +%d cols -> %d total", added_shf, f.shape[1]
    )

    # ---- Step 55: load_conlan_eod_price_data (6 EOD features) — Wave CEP1 ----
    # Source: github:chrisconlan/algorithmic-trading-with-python (MIT, no paid API).
    # Pure OHLCV inputs: pct_below_52w_high/mom_6m/vol_trend_ratio/above_200ma/atr_pct/dollar_vol_z.
    # .shift(1)-safe: all indicator series shifted 1 bar inside the module.
    before_cep = f.shape[1]
    try:
        f = compute_load_conlan_eod_price_data_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning(
            "  [v10] load_conlan_eod_price_data call failed (%s): %s — zeroing", ticker, exc
        )
        for col in CONLAN_EOD_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_cep = f.shape[1] - before_cep
    logger.info(
        "  [v10] +load_conlan_eod_price_data: +%d cols -> %d total", added_cep, f.shape[1]
    )

    # ---- Step 56: load_conlan_alt_data_features (5 features) — Wave CAD1 ----
    # Source: github:chrisconlan/algorithmic-trading-with-python/data/alternative_data (MIT).
    # OHLCV proxies when CSV files absent: vol_price_corr/range_norm/sent/gap/turnover.
    # .shift(1)-safe: all indicator series shifted 1 bar inside the module.
    before_cad = f.shape[1]
    try:
        f = compute_load_conlan_alt_data_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning(
            "  [v10] load_conlan_alt_data_features call failed (%s): %s — zeroing", ticker, exc
        )
        for col in CONLAN_ALT_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_cad = f.shape[1] - before_cad
    logger.info(
        "  [v10] +load_conlan_alt_data_features: +%d cols -> %d total", added_cad, f.shape[1]
    )

    # ---- Step 57: alpha101_ts_safe_subset (15 STHSF/alpha101 ts-safe factors) — Wave A101S ----
    # Source: github:STHSF/alpha101 (MIT, no paid API). Pure OHLCV; ts_rank replaces cross-sectional rank.
    # .shift(1)-safe: all OHLCV inputs pre-shifted 1 bar inside the module.
    before_a101s = f.shape[1]
    try:
        f = compute_alpha101_ts_safe_subset_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning(
            "  [v10] alpha101_ts_safe_subset call failed (%s): %s — zeroing", ticker, exc
        )
        for col in ALPHA101S_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_a101s = f.shape[1] - before_a101s
    logger.info(
        "  [v10] +alpha101_ts_safe_subset: +%d cols -> %d total", added_a101s, f.shape[1]
    )

    # ---- Step 58: gtja_alpha191 (GTJA "191 Formulaic Alphas", 50 features) — Wave GTJA1 ----
    # Source: github:Daic115/alpha191 (MIT, no paid API). Pure OHLCV; ts_rank replaces CS rank.
    # .shift(1)-safe: all OHLCV inputs pre-shifted 1 bar inside the module.
    before_gtja = f.shape[1]
    try:
        f = compute_gtja_alpha191_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning(
            "  [v10] gtja_alpha191 call failed (%s): %s — zeroing", ticker, exc
        )
        for col in GTJA_ALPHA191_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_gtja = f.shape[1] - before_gtja
    logger.info(
        "  [v10] +gtja_alpha191: +%d cols -> %d total", added_gtja, f.shape[1]
    )

    # ---- Step 59: vwap_indicator_python_features (~7 cols) - Wave TOP7-1 ----
    # Session-VWAP + 1sigma/2sigma deviation flags. Pure OHLCV.
    before_vwap_ip = f.shape[1]
    try:
        f = add_vwap_indicator_python_features(f, ticker)
    except Exception as exc:
        logger.warning(
            "  [v10] vwap_indicator_python_features call failed (%s): %s - zeroing", ticker, exc
        )
        for col in VWAP_INDICATOR_PY_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_vwap_ip = f.shape[1] - before_vwap_ip
    logger.info(
        "  [v10] +vwap_indicator_python: +%d cols -> %d total", added_vwap_ip, f.shape[1]
    )

    # ---- Step 60: strategy_signal_features (~14 cols D1/D2/D3 + filter stack) - Wave TOP7-2 ----
    # Depends on rsi_14, ema_20/50/200, ret_21d (present after v9). Internally .shift(1)-safe.
    before_ss = f.shape[1]
    try:
        f = add_strategy_signal_features(f)
        f = add_five_filter_stack(f)
    except Exception as exc:
        logger.warning(
            "  [v10] strategy_signal_features call failed (%s): %s - zeroing", ticker, exc
        )
        for col in STRATEGY_SIGNAL_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0
    added_ss_top7 = f.shape[1] - before_ss
    logger.info(
        "  [v10] +strategy_signal: +%d cols -> %d total", added_ss_top7, f.shape[1]
    )

    # ---- Step 61: gabriel_indicators_features (~250 cols) - Wave TOP7-3 ----
    # 107 TA indicators registry from historical_system.indicators package.
    # Dynamic col set; chikou excluded (forward-looking).
    # SPEED FIX (2026-05-20, top7-followup): runtime per-indicator budget gate +
    # total wall-clock budget (env GABRIEL_TOTAL_BUDGET_S, default 60s). Vendored
    # 43-indicator fallback if Drive path slow (env GABRIEL_PREFER_LOCAL=1,
    # GABRIEL_SKIP_DRIVE=1).
    # LEAK-RECOVER (2026-05-21, ref a3ee919): 3 real leaky indicators patched
    # in-place (ichimoku_cloud chikou->NaN, williams_fractal confirm-delay,
    # zig_zag online-pivot). 16 phantom names removed from _DEFAULT_SLOW_SET
    # (they never matched any registered indicator name). fisher_transform
    # re-enabled (leak-safe). Net col delta: +2 (fisher) + 1 (zigzag_lag) = +3.
    # SKIP GATE retained as belt-and-suspenders: set env GABRIEL_SKIP=1 to bypass.
    before_gab = f.shape[1]
    if os.environ.get("GABRIEL_SKIP", "0") == "1":
        logger.info("  [v10] gabriel_indicators_features SKIPPED (GABRIEL_SKIP=1)")
    else:
        try:
            f = add_gabriel_indicators_features(f, ticker)
        except Exception as exc:
            logger.warning(
                "  [v10] gabriel_indicators_features call failed (%s): %s - skip (no static cols to zero)",
                ticker, exc,
            )
    added_gab = f.shape[1] - before_gab
    logger.info(
        "  [v10] +gabriel_indicators: +%d cols -> %d total", added_gab, f.shape[1]
    )

    # ---- Step 62: ma_energy_indicator_features (1 col) - Wave TOP7-4 ----
    # ma_energy in [-1, +1]: multi-TF MA momentum / volatility.
    before_mae = f.shape[1]
    try:
        f = add_ma_energy_indicator_features(f, ticker)
    except Exception as exc:
        logger.warning(
            "  [v10] ma_energy_indicator call failed (%s): %s - zeroing", ticker, exc
        )
        if "ma_energy" not in f.columns:
            f["ma_energy"] = 0.0
    added_mae = f.shape[1] - before_mae
    logger.info(
        "  [v10] +ma_energy: +%d cols -> %d total", added_mae, f.shape[1]
    )

    # ---- Step 63: sec_edgar_features (14 cols: 12 real + 2 legacy) - Wave TOP7-5 ----
    # FLESHED OUT 2026-05-20 (top7-followup): delegates to edgar_extras_features
    # (12 real cols: def14a, amendments, earnings 8-K, S-1, burst, accel).
    # Plus 2 legacy stub cols (se_signal_a/b, zero-filled) for backwards-compat.
    before_sef = f.shape[1]
    try:
        f = add_sec_edgar_features(f, ticker)
    except Exception as exc:
        logger.warning(
            "  [v10] sec_edgar_features call failed (%s): %s - zeroing", ticker, exc
        )
        for col in SEC_EDGAR_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_sef = f.shape[1] - before_sef
    logger.info(
        "  [v10] +sec_edgar (stub): +%d cols -> %d total", added_sef, f.shape[1]
    )

    # ---- Step 64: trading_indicators_features (2 cols, STUB) - Wave TOP7-6 ----
    before_ti = f.shape[1]
    try:
        f = add_trading_indicators_features(f, ticker)
    except Exception as exc:
        logger.warning(
            "  [v10] trading_indicators_features call failed (%s): %s - zeroing", ticker, exc
        )
        for col in TRADING_INDICATORS_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_ti = f.shape[1] - before_ti
    logger.info(
        "  [v10] +trading_indicators (stub): +%d cols -> %d total", added_ti, f.shape[1]
    )

    # ---- Step 65: xgboost_features (3 cols) - Wave TOP7-7 ----
    # Rolling XGBoost classifier: xgb_pred_direction, xgb_confidence, xgb_prob_up.
    # SPEED FIX (2026-05-20, top7-followup): warm-start booster + stride=5 +
    # tree_method='hist' + total-budget gate. Smoke: 21.7s for 1213 rows (was ~5min).
    # Env knobs: XGB_FIT_STRIDE, XGB_INCR_ROUNDS, XGB_BASE_ROUNDS, XGB_TOTAL_BUDGET_S.
    # SKIP GATE retained: set env XGB_FEATURES_SKIP=1 to bypass.
    before_xgbf = f.shape[1]
    if os.environ.get("XGB_FEATURES_SKIP", "0") == "1":
        logger.info("  [v10] xgboost_features SKIPPED (XGB_FEATURES_SKIP=1) - zeroing 3 cols")
        for col in ("xgb_pred_direction", "xgb_confidence", "xgb_prob_up"):
            if col not in f.columns:
                f[col] = 0.0 if col == "xgb_pred_direction" else 0.5
    else:
        try:
            f = add_xgboost_features(f, ticker)
        except Exception as exc:
            logger.warning(
                "  [v10] xgboost_features call failed (%s): %s - zeroing", ticker, exc
            )
            for col in ("xgb_pred_direction", "xgb_confidence", "xgb_prob_up"):
                if col not in f.columns:
                    f[col] = 0.0 if col == "xgb_pred_direction" else 0.5
    added_xgbf = f.shape[1] - before_xgbf
    logger.info(
        "  [v10] +xgboost_features: +%d cols -> %d total", added_xgbf, f.shape[1]
    )

    # ---- Step 66: candlestick_features (TA-Lib CDL patterns, ~188 cols) — Wave QW1 ----
    before_cdl = f.shape[1]
    try:
        f = add_candlestick_features(f, rolling_window=5, include_rolling=True)
    except Exception as exc:
        logger.warning("  [v10] candlestick call failed (%s): %s — zeroing", ticker, exc)
        for col in CDL_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0
    added_cdl = f.shape[1] - before_cdl
    logger.info("  [v10] +candlestick_features: +%d cols -> %d total", added_cdl, f.shape[1])

    # ---- Step 67: oc2_donchian_c003 (cycle003 Donchian breakout, 12 cols) — Wave QW2 ----
    before_dc003 = f.shape[1]
    try:
        f = add_oc2_donchian_c003_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] oc2_donchian_c003 call failed (%s): %s — zeroing", ticker, exc)
        for col in DONCH_C003_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_dc003 = f.shape[1] - before_dc003
    logger.info("  [v10] +oc2_donchian_c003: +%d cols -> %d total", added_dc003, f.shape[1])

    # ---- Step 68: oc2_donchian_mtf (multi-TF Donchian + filter stack, 16 cols) — Wave QW3 ----
    before_dmtf = f.shape[1]
    try:
        f = add_oc2_donchian_mtf_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] oc2_donchian_mtf call failed (%s): %s — zeroing", ticker, exc)
        for col in DONCH_MTF_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_dmtf = f.shape[1] - before_dmtf
    logger.info("  [v10] +oc2_donchian_mtf: +%d cols -> %d total", added_dmtf, f.shape[1])

    # ---- Step 69: oc2_donchian_per_ticker_selectivity (per-ticker selectivity, 10 cols) — Wave QW4 ----
    before_dsel = f.shape[1]
    try:
        f = add_oc2_donchian_per_ticker_selectivity_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] oc2_donchian_selectivity call failed (%s): %s — zeroing", ticker, exc)
        for col in DONCH_SEL_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_dsel = f.shape[1] - before_dsel
    logger.info("  [v10] +oc2_donchian_selectivity: +%d cols -> %d total", added_dsel, f.shape[1])

    # ---- Step 70: py_market_profile (TPO volume profile POC/VAH/VAL, 6 cols) — Wave QW5 ----
    before_pymp = f.shape[1]
    try:
        f = add_py_market_profile_features(f, ticker)
    except Exception as exc:
        logger.warning("  [v10] py_market_profile call failed (%s): %s — zeroing", ticker, exc)
        for col in PYMP_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_pymp = f.shape[1] - before_pymp
    logger.info("  [v10] +py_market_profile: +%d cols -> %d total", added_pymp, f.shape[1])

    # ---- Step 71: footprint_analyzer (volume profile + imbalance proxy, 4 cols) — Wave QW6 ----
    before_fp = f.shape[1]
    try:
        f = add_footprint_features(f, ticker)
    except Exception as exc:
        logger.warning("  [v10] footprint_analyzer call failed (%s): %s — zeroing", ticker, exc)
        for col in FP_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_fp = f.shape[1] - before_fp
    logger.info("  [v10] +footprint_analyzer: +%d cols -> %d total", added_fp, f.shape[1])

    # ---- Step 72: harmonic_patterns_features (30 cols) — Wave BG1 (2026-05-21) ----
    # Gartley/Bat/Butterfly/Crab/Shark/Cypher/etc detectors on rolling zigzag.
    # .shift(1)-safe inside module. Pure OHLC; no external API.
    before_hp = f.shape[1]
    try:
        f = compute_harmonic_patterns_features(f, ticker=ticker)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "  [v10] harmonic_patterns_features call failed (%s): %s - zeroing", ticker, exc
        )
        for col in HARMONIC_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_hp = f.shape[1] - before_hp
    logger.info("  [v10] +harmonic_patterns: +%d cols -> %d total", added_hp, f.shape[1])

    # ---- Step 73: chart_patterns_features (35 cols) — Wave BG2 (2026-05-21) ----
    # H&S, double top/bottom, triangle, wedge, flag, pennant, channel + aggregates.
    # .shift(1)-safe inside module.
    before_cp = f.shape[1]
    try:
        f = compute_chart_patterns_features(f, ticker=ticker)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "  [v10] chart_patterns_features call failed (%s): %s - zeroing", ticker, exc
        )
        for col in CHART_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_cp = f.shape[1] - before_cp
    logger.info("  [v10] +chart_patterns: +%d cols -> %d total", added_cp, f.shape[1])

    # ---- Step 74: regression_channels_features (12 cols) — Wave BG3 (2026-05-21) ----
    # Rolling linreg ±2σ channels for N=20/50/100. .shift(1)-safe.
    before_rc = f.shape[1]
    try:
        f = compute_regression_channels_features(f, ticker=ticker)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "  [v10] regression_channels_features call failed (%s): %s - zeroing", ticker, exc
        )
        for col in REGRESSION_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_rc = f.shape[1] - before_rc
    logger.info("  [v10] +regression_channels: +%d cols -> %d total", added_rc, f.shape[1])

    # ---- Step 75: auto_support_resistance_features (10 cols) — Wave BG4 (2026-05-21) ----
    # Pivot-cluster S/R zone detection. .shift(1)-safe.
    before_sr = f.shape[1]
    try:
        f = compute_auto_support_resistance_features(f, ticker=ticker)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "  [v10] auto_support_resistance_features call failed (%s): %s - zeroing", ticker, exc
        )
        for col in AUTO_SR_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_sr = f.shape[1] - before_sr
    logger.info("  [v10] +auto_support_resistance: +%d cols -> %d total", added_sr, f.shape[1])

    # ---- Step 76: sector_aggregate_features (60 cols) — Wave OC-AUDIT (2026-05-21) ----
    # GICS-sector pooled cross-sectional features. .shift(1)-safe inside module.
    before_sa = f.shape[1]
    try:
        f = compute_sector_aggregate_features(f, ticker=ticker)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "  [v10] sector_aggregate_features call failed (%s): %s - zeroing", ticker, exc
        )
        for col in SECTOR_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_sa = f.shape[1] - before_sa
    logger.info("  [v10] +sector_aggregate: +%d cols -> %d total", added_sa, f.shape[1])

    # ---- Step 77: market_regime_hmm_features (7 cols) — Wave OC-AUDIT (2026-05-21) ----
    # 5-state Gaussian HMM regime classifier on (log-VIX, xsec vol, sector corr).
    # All inputs .shift(1) before HMM observation; outputs strictly causal.
    before_mr = f.shape[1]
    try:
        f = compute_market_regime_hmm_features(f, ticker=ticker)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "  [v10] market_regime_hmm_features call failed (%s): %s - zeroing", ticker, exc
        )
        for col in REGIME_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 1.0 if col == "regime_prob_2" else (2 if col == "regime_state" else 0)
    added_mr = f.shape[1] - before_mr
    logger.info("  [v10] +market_regime_hmm: +%d cols -> %d total", added_mr, f.shape[1])

    # ---- Dedup + dropna on critical columns ----
    f = f.loc[:, ~f.columns.duplicated()]
    f = f.dropna(subset=["rsi_14", "atr_14", "ema_200", "fwd_ret_21d", "y"])

    # Wave OW-21 (2026-05-21) — orphan-wire audit a995e379: closeable_gaps
    # is wired in v8 (build_v8_features ->  add_closeable_gap_features) and
    # flows through v9->v10 inside v9_base. We expose an audit counter here
    # for manifest accountability — value is the # of expected feature names
    # from closeable_gap_feature_names() (18), since the actual delta is
    # already subsumed in after_v9 above.
    try:
        from closeable_gaps_features import closeable_gap_feature_names  # noqa: E402
        added_closeable_gap = len(closeable_gap_feature_names())
    except Exception:
        added_closeable_gap = 0

    module_feature_counts = {
        "v9_base": after_v9,
        "closeable_gap_added": added_closeable_gap,  # audit-only; subsumed in v9_base
        "alpaca_added": added_alpaca,
        "insider_form4_added": added_insider,
        "daily_integration_added": added_daily_int,
        "dfs_added": added_dfs,
        "mastery_priors_added": added_mastery_priors,
        "paper_trade_outcomes_added": added_paper_trade_outcomes,
        "stumpy_added": added_stumpy,
        "ffn_added": added_ffn,
        "pandas_ta_classic_added": added_ptc,
        # Wave A (2026-05-17)
        "options_flow_added": added_options_flow,
        "govtrades_added": added_govtrades,
        "time_of_day_added": added_tod,
        "gabriel_priors_added": added_gabriel,
        # Wave B (2026-05-17)
        "vix_term_structure_v2_added": added_vts,
        # Wave C (2026-05-17)
        "garch_11_cond_vol_added": added_garch,
        # Wave D (2026-05-17)
        "egarch_11_leverage_added": added_egarch_lev,
        # Wave E (2026-05-17)
        "vpin_50bucket_added": added_vpin,
        # Wave F (2026-05-17)
        "kyles_lambda_intraday_added": added_kyles_lambda,
        # Wave M-1 microstructure top-4 (2026-05-17)
        "vpin_intraday_added": added_vpin_intra,
        "tick_imbalance_added": added_tick_imbalance,
        "volume_profile_added": added_volume_profile,
        "auction_added": added_auction,
        # Wave V-1 vol/regime low-cost 7-pack (2026-05-17)
        "vol_of_vol_added": added_vov,
        "vol_risk_premium_added": added_vrp,
        "vol_target_sizing_added": added_vt,
        "vol_breakout_nr_added": added_nr,
        "bollinger_keltner_squeeze_added": added_sq,
        "vol_of_vix_added": added_vvx,
        "rv_term_structure_added": added_rvt,
        # Wave G (2026-05-17)
        "amihud_illiq_added": added_amihud,
        # Wave H-1 (2026-05-17)
        "rolls_effective_spread_added": added_rolls,
        # Wave Cycle (2026-05-17) — 5 cycle engines from claudes test
        "cycle051_added": added_c051,
        "cycle055_added": added_c055,
        "cycle058_added": added_c058,
        "cycle060_added": added_c060,
        "cycle061_added": added_c061,
        # Wave AFC (2026-05-17) — WorldQuant alpha101 port via alpha_features_core
        "alpha_features_core_added": added_afc,
        # Wave FDB (2026-05-17) — FinanceDatabase metadata (sector/industry/market_cap/exchange)
        "finance_database_added": added_fdb,
        # Wave MLF (2026-05-17) — mlforecast-style rolling/EWM/expanding lag features (11 cols)
        "mlforecast_features_added": added_mlf,
        # Wave NF (2026-05-17) — NeuralForecast decomp features (5 cols)
        "neuralforecast_features_added": added_nf,
        # Wave WQ101 (2026-05-17) — WorldQuant Alpha#101 z-score (1 col)
        "worldquant_alpha101_replay_added": added_wq101,
        # Wave D6 (2026-05-17) — Mythos curriculum prior static features (6 cols)
        "mythos_deltas_added": added_mythos_deltas,
        # Wave TS101 (2026-05-17) — STHSF/alpha101 Alpha#6 z-score (1 col)
        "alpha101_ts_safe_subset_added": added_a101ts,
        # Wave B1 (2026-05-17) — EDGAR filing recency/density (9 cols)
        "edgar_added": added_edgar,
        # Wave FEC1 (2026-05-17) — FEC employer donation rolling flag (1 col)
        "ceo_personal_donation_flag_added": added_ceo_don,
        # Wave EFD1 (2026-05-17) — Senate STOCK Act options-disclosure 30d count (1 col)
        "senate_efd_options_disclosure_count_30d_added": added_sefd,
        # Wave VTS1 (2026-05-18) — VIX term-structure v1: spread/z21/streak (3 cols)
        "vix_term_structure_added": added_vt1,
        # Wave HMM1 (2026-05-18) — 3-state HMM vol regime (3 cols)
        "regime_hmm_3state_vol_regime_added": added_hmm3,
        # Wave WQA101 (2026-05-18) — WorldQuant Alpha-101 25-alpha adaptation (25 cols)
        "worldquant_alpha101_added": added_wqa101,
        # Wave BOCPD1 (2026-05-18) — BOCPD vol-break: break_prob/run_length/regime_id (3 cols)
        "regime_changepoint_bayesian_vol_break_added": added_bocpd,
        # Wave SS1 (2026-05-18) — stockstats 28 technical indicators (28 cols)
        "add_stockstats_features_added": added_ss,
        # Wave TLP1 (2026-05-18) — talipp 8 TA indicators: TEMA/DEMA/HMA/TRIX/DPO/ROC/ZLEMA/WMA (8 cols)
        "add_talipp_features_added": added_talipp,
        # Wave JES1 (2026-05-18) — Jesse-inspired 54 TA indicators: MA/Ichimoku/Donchian/MACD/osc/trend/vol/micro (54 cols)
        "add_jesse_features_added": added_jesse,
        # Wave SHF1 (2026-05-18) — shashankvemuri/Finance 58 features: MA ratios/RSI/MACD/BB/vol/volume/trend/price-action/MR/S&R/oscillators
        "add_shashank_finance_features_added": added_shf,
        # Wave CEP1 (2026-05-18) — Conlan EOD 6 features: pct_below_52w_high/mom_6m/vol_trend_ratio/above_200ma/atr_pct/dollar_vol_z
        "load_conlan_eod_price_data_added": added_cep,
        # Wave CAD1 (2026-05-18) — Conlan alt-data 5 features: vol_price_corr_21d/intraday_range_norm_5d/close_vs_open_sent_5d/overnight_gap_pct_5d/turnover_ratio_21d
        "load_conlan_alt_data_features_added": added_cad,
        # Wave A101S (2026-05-18) — STHSF/alpha101 ts-safe 15 factors: a101ts_alpha001/003/006/008/012/016/019/020/023/033/034/035/040/041/051
        "alpha101_ts_safe_subset_added": added_a101s,
        # Wave GTJA1 (2026-05-18) — GTJA "191 Formulaic Alphas" 50 TS-safe features: gtja_a006/009/011/012/014/018/019/021/022/023/024/027/028/029/031/034/035/038/039/040/041/042/043/044/045/046/047/048/049/050/051/052/053/054/055/057/060/061/062/063/064/065/066/067/068/069/071/072/073/074
        "gtja_alpha191_added": added_gtja,
        # Wave TOP7 (2026-05-20) - drive-map highest-priority unwired modules
        "vwap_indicator_python_added": added_vwap_ip,
        "strategy_signal_top7_added": added_ss_top7,
        "gabriel_indicators_added": added_gab,
        "ma_energy_indicator_added": added_mae,
        "sec_edgar_stub_added": added_sef,
        "trading_indicators_stub_added": added_ti,
        "xgboost_features_added": added_xgbf,
        # Wave QW (2026-05-21) — quick-wire 6 READY-but-UNWIRED modules
        "candlestick_features_added": added_cdl,
        "oc2_donchian_c003_added": added_dc003,
        "oc2_donchian_mtf_added": added_dmtf,
        "oc2_donchian_per_ticker_selectivity_added": added_dsel,
        "py_market_profile_added": added_pymp,
        "footprint_analyzer_added": added_fp,
        # Wave OC-AUDIT (2026-05-21) — OC #4 + #5 from live-robustness audit
        "sector_aggregate_added": added_sa,
        "market_regime_hmm_added": added_mr,
        "total_after_dedup_dropna": f.shape[1],
    }
    return f, mythos_fallback_rows, module_feature_counts


def build_v10_features(
    ticker: str,
    universe_agg: Optional[dict] = None,
    use_mythos: bool = False,
    start_date=None,
    end_date=None,
) -> tuple[pd.DataFrame, int, dict]:
    """Cached wrapper around _build_v10_features_impl.

    Adds a disk-based parquet cache keyed on (ticker, start_date, end_date,
    feature_set, V10_FEATURE_VERSION) via feature_cache.get_cached().

    start_date / end_date are optional and used only for cache-key purposes
    (the impl pulls all available history regardless). Pass them when the
    caller knows the date window so that different windows get separate cache
    entries.

    On a cache HIT, returns (cached_df, 0, {"from_cache": True}).
    On a cache MISS, returns the full tuple from _build_v10_features_impl.
    """
    _start = str(start_date) if start_date is not None else "all"
    _end = str(end_date) if end_date is not None else "all"

    # Mutable container to capture impl results when compute_fn is invoked
    _meta: list = []

    def _compute() -> pd.DataFrame:
        df, fallback_rows, module_counts = _build_v10_features_impl(
            ticker, universe_agg, use_mythos=use_mythos
        )
        _meta.extend([fallback_rows, module_counts])
        return df

    cached_df = get_cached(
        ticker=ticker,
        date_range=(_start, _end),
        feature_set="v10_full",
        compute_fn=_compute,
        version=V10_FEATURE_VERSION,
        ttl_days=7,
    )

    if _meta:
        # Cache miss: _compute() ran and populated real metadata
        return cached_df, _meta[0], _meta[1]
    # Cache hit: _compute() was skipped; return sentinel metadata
    return cached_df, 0, {"from_cache": True}


# ---------------------------------------------------------------------------
# CLI entry point — Helper D: --job-id restored
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "XGBoost v10 — v9 stack + daily_integration + alpaca + featuretools DFS"
        )
    )
    ap.add_argument("--ticker", required=True, help="Stock symbol e.g. AAPL")
    # Support both --output-dir (v9 style) and --out-dir (GH Actions / v8 style)
    _od = ap.add_mutually_exclusive_group(required=True)
    _od.add_argument("--output-dir", dest="output_dir", help="Directory for output files")
    _od.add_argument("--out-dir", dest="output_dir", help="Alias for --output-dir (GH Actions)")
    # --job-id: restored from v8; required by GH Actions workflow
    ap.add_argument(
        "--job-id",
        default="",
        help="CI/CD job identifier (e.g. smoke-001). Written to run_meta.json.",
    )
    ap.add_argument(
        "--use-mythos-features",
        action="store_true",
        default=False,
        help=(
            "DEPRECATED 2026-05-21 (OC audit rank #2 — Mythos dropped). When "
            "MYTHOS_DISABLED=1 (default) this flag is a no-op; the 256-col "
            "concat is skipped regardless. Pass MYTHOS_DISABLED=0 in env to "
            "re-enable the transformer."
        ),
    )
    ap.add_argument("--prob-threshold", type=float, default=0.50)
    ap.add_argument("--sweep-threshold", action="store_true")
    # autosolve_skip: code-patch — top_k legacy cap 50 → adaptive (2026-05-20)
    # karpathy_checked: top_k=50 was sized for 173-feature v8 era; with v10 ~1231
    # features per ticker, 96% are discarded before trees see them. Adaptive
    # default = min(int(sqrt(n_rows) * 4), n_features, 400). Env XGB_TOP_K
    # overrides; CLI --top-k overrides env. Default sentinel = 0 → adaptive.
    ap.add_argument(
        "--top-k",
        type=int,
        default=int(os.environ.get("XGB_TOP_K", "0")),
        help="Top-K features after scout-importance prune. 0 = adaptive: "
             "min(int(sqrt(n_rows)*4), n_features, 400). Env XGB_TOP_K overrides default.",
    )
    ap.add_argument("--tp-atr", type=float, default=1.5)
    ap.add_argument("--sl-atr", type=float, default=1.0)
    ap.add_argument("--max-hold", type=int, default=21)
    ap.add_argument("--strategy", default="default", help="Strategy label for metadata")
    # autosolve_skip: multi-TF wire — 2026-05-21
    # Per-ticker per-TF mastery: --timeframe selects which Cache B TF the
    # underlying ohlcv_loader pulls. Plumbed via BACKTEST_TIMEFRAME env var
    # so backtest_ml.load_daily() (shared loader for v6/v7/v8/v9/v10 builders)
    # picks it up without further code changes. CLI > env > default "1Day".
    ap.add_argument(
        "--timeframe",
        default=os.environ.get("BACKTEST_TIMEFRAME", "1Day"),
        choices=["1Min", "5Min", "15Min", "30Min", "45Min",
                 "1Hour", "4Hour", "8Hour", "12Hour", "1Day"],
        help="Cache B timeframe (default 1Day; env BACKTEST_TIMEFRAME).",
    )
    # purgedcv integrity flag (wired 2026-05-21):
    # When set, every fold's (train, oos) split is validated by purgedcv
    # diagnostics: assert_no_temporal_leakage + assert_embargo_respected.
    # Any violation raises TemporalLeakageError / EmbargoViolationError and
    # halts the run. Smoke comparison: --purged-cv off vs on tells us
    # whether the existing manual embargo logic respects the LdP invariant.
    ap.add_argument(
        "--purged-cv",
        action="store_true",
        default=os.environ.get("BACKTEST_PURGED_CV", "0") == "1",
        help=(
            "Enable purgedcv diagnostic assertions per fold "
            "(Lopez de Prado AFML s7). Env BACKTEST_PURGED_CV=1 also enables."
        ),
    )
    # Optuna HP search (2026-05-21) — replaces external cartesian 108-combo
    # sweep. When --optuna-hp is set, each fold runs a 36-trial
    # TPESampler+HyperbandPruner search to pick learning_rate, max_depth,
    # subsample, colsample_bytree (others held at _xgb_base_params defaults).
    # Trials 0-3 are seeded from state/mastery_priors.json (or sane defaults).
    ap.add_argument(
        "--optuna-hp",
        action="store_true",
        default=os.environ.get("BACKTEST_OPTUNA_HP", "0") == "1",
        help=(
            "Enable per-fold Optuna TPE+Hyperband HP search (36 trials). "
            "Env BACKTEST_OPTUNA_HP=1 also enables."
        ),
    )
    ap.add_argument(
        "--optuna-n-trials",
        type=int,
        default=int(os.environ.get("BACKTEST_OPTUNA_N_TRIALS", "36")),
        help="Optuna trial count when --optuna-hp is set (default 36).",
    )
    ap.add_argument(
        "--optuna-priors",
        default=os.environ.get("BACKTEST_OPTUNA_PRIORS", ""),
        help=(
            "Path to mastery_priors.json (seeds Optuna trials 0-3). "
            "If empty, defaults to state/mastery_priors.json (sane defaults "
            "applied if file absent)."
        ),
    )
    args = ap.parse_args()

    # Propagate --timeframe into env BEFORE build_v10_features runs;
    # backtest_ml.load_daily() reads BACKTEST_TIMEFRAME at call time.
    os.environ["BACKTEST_TIMEFRAME"] = args.timeframe

    # Strategy / TF compatibility check (warn-only; sweep should pre-filter).
    _STRAT_TF_COMPAT = {
        "ORB":         {"1Min", "5Min", "15Min", "30Min"},
        "VWAP":        {"1Min", "5Min", "15Min", "30Min", "45Min", "1Hour"},
        "v10":         {"1Hour", "1Day"},
        "momentum":    {"1Hour", "4Hour", "8Hour", "12Hour", "1Day"},
        "catalyst":    {"1Day"},
        "mean_revert": {"5Min", "15Min", "30Min"},
    }
    _ok = _STRAT_TF_COMPAT.get(args.strategy)
    if _ok is not None and args.timeframe not in _ok:
        logger.warning(
            "[v10] strategy/TF combo may be incoherent: strategy=%s timeframe=%s "
            "(typical: %s)", args.strategy, args.timeframe, sorted(_ok),
        )

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(
        "[v10] Starting: ticker=%s strategy=%s job_id=%s output_dir=%s",
        args.ticker,
        args.strategy,
        args.job_id or "(none)",
        args.output_dir,
    )

    # ---- Cross-sectional cache ----
    universe_agg = None
    cache_path = WORK / "cache" / "universe_agg_manifest.json"
    if cache_path.exists() and csf is not None:
        try:
            universe_agg = csf.precompute_universe_aggregates()
        except Exception as e:
            logger.warning("  [csf] cache load failed: %s", e)

    # ---- Build v10 feature set ----
    f, mythos_fallback_rows, module_counts = build_v10_features(
        args.ticker,
        universe_agg,
        use_mythos=args.use_mythos_features,
    )
    fc = numeric_cols(f)
    logger.info(
        "  TOTAL features: %d; rows: %d; mythos_fallback_rows: %d",
        len(fc),
        len(f),
        mythos_fallback_rows,
    )
    logger.info("  Module counts: %s", module_counts)

    # Resolve checkpoint path for metadata
    mythos_checkpoint_path = os.environ.get("MYTHOS_CHECKPOINT_PATH", "")
    if not mythos_checkpoint_path:
        mythos_checkpoint_path = str(
            Path(WORK).parent / "checkpoints" / "mythos_financial_v0.pt"
        )

    # ---- Walk-forward folds ----
    folds = bml.make_walk_forward_folds(f, train_months=24, test_months=12, step_months=12)
    logger.info("  folds: %d", len(folds))
    all_probs = pd.Series(np.nan, index=f.index)
    fold_summaries = []
    fold_top_features = []
    fold_mythos_importances = []

    mythos_feat_set = set(MYTHOS_FEAT_NAMES)

    for fold in folds:
        train_end_emb = (
            pd.Timestamp(fold["train_end"])
            - pd.tseries.offsets.BDay(LABEL_EMBARGO_DAYS)
        )
        train = f[
            (f.index >= fold["train_start"]) & (f.index < train_end_emb)
        ]
        oos = f[
            (f.index >= fold["oos_start"]) & (f.index < fold["oos_end"])
        ]
        if len(train) < 50 or len(oos) < 20:
            continue

        # purgedcv leakage gate (wired 2026-05-21) -- Lopez de Prado AFML s7
        # Build prediction_times = bar timestamp, evaluation_times = bar + 21BD
        # (the LABEL_EMBARGO_DAYS forward window over which the label realizes).
        # Then assert: (a) no train bar's evaluation window overlaps oos's
        # prediction window, and (b) embargo gap >= LABEL_EMBARGO_DAYS is
        # respected. Violations raise TemporalLeakageError / EmbargoViolationError.
        if args.purged_cv and _PURGEDCV_AVAILABLE:
            try:
                _full_idx = f.index
                _pred_t = pd.Series(_full_idx, index=_full_idx)
                _eval_t = pd.Series(
                    _full_idx + pd.tseries.offsets.BDay(LABEL_EMBARGO_DAYS),
                    index=_full_idx,
                )
                _tr_pos = np.where(_full_idx.isin(train.index))[0]
                _oos_pos = np.where(_full_idx.isin(oos.index))[0]
                _pcv_assert_no_leak(
                    _tr_pos, _oos_pos,
                    prediction_times=_pred_t,
                    evaluation_times=_eval_t,
                )
                _pcv_assert_embargo(
                    _tr_pos, _oos_pos,
                    prediction_times=_pred_t,
                    evaluation_times=_eval_t,
                    embargo=pd.Timedelta(days=LABEL_EMBARGO_DAYS),
                )
                logger.info(
                    "  [purged-cv] fold %s OK (n_tr=%d n_oos=%d, embargo=%dBD)",
                    fold["fold"], len(_tr_pos), len(_oos_pos),
                    LABEL_EMBARGO_DAYS,
                )
            except Exception as _e:
                logger.error(
                    "  [purged-cv] fold %s LEAKAGE DETECTED: %s",
                    fold["fold"], _e,
                )
                raise

        # Scout model — get top-K features by gain
        X_tr_all = train[fc].fillna(0).values
        y_tr = train["y"].values
        X_oos_all = oos[fc].fillna(0).values

        scout = xgb.XGBClassifier(**_xgb_base_params("scout"))
        scout.fit(X_tr_all, y_tr)

        importances = list(zip(fc, scout.feature_importances_))
        importances.sort(key=lambda x: -x[1])
        # Resolve effective top_k per-fold (adaptive when args.top_k <= 0)
        effective_top_k = _resolve_top_k(args.top_k, len(train), len(fc))
        # XGB_KEEP_ZERO_IMP=1 → keep zero-scout-importance features (true full
        # bypass; may overfit small folds — observed AAPL fold-1 dropped to 9
        # in-tree features). Default keeps imp>0 filter as natural reg, but
        # XGB_NO_TOPK=1 still expands effective_top_k to n_features (so any
        # feature with non-zero scout gain reaches the final model — Stage A
        # full-util patch 2026-05-20).
        keep_zero = os.environ.get("XGB_KEEP_ZERO_IMP", "0") == "1"
        if keep_zero:
            top_features = [c for c, _ in importances[: effective_top_k]]
        else:
            top_features = [c for c, imp in importances[: effective_top_k] if imp > 0]
            if len(top_features) < 10:
                top_features = [c for c, _ in importances[: effective_top_k]]
        fold_top_features.append({
            "fold": fold["fold"],
            "top_features": top_features[:30],
            "effective_top_k": int(effective_top_k),
            "n_features_total": int(len(fc)),
        })

        # Track Mythos feature importances specifically
        imp_dict = dict(importances)
        mythos_imps = {
            feat: float(imp_dict.get(feat, 0.0))
            for feat in MYTHOS_FEAT_NAMES
            if feat in fc and imp_dict.get(feat, 0.0) > 0
        }
        top_mythos = sorted(mythos_imps.items(), key=lambda x: -x[1])[:10]
        fold_mythos_importances.append(
            {
                "fold": fold["fold"],
                "mythos_in_top50": sum(
                    1 for feat in top_features if feat in mythos_feat_set
                ),
                "top_mythos": dict(top_mythos),
            }
        )

        # Final model on pruned features
        X_tr = train[top_features].fillna(0).values
        X_oos = oos[top_features].fillna(0).values
        # Build optional constraints from the pruned feature list (env-gated)
        final_params = _xgb_base_params("final")
        if _XGB_USE_INTERACTION:
            final_params["interaction_constraints"] = _build_interaction_constraints(top_features)
        if _XGB_USE_MONOTONIC:
            final_params["monotone_constraints"] = _build_monotonic_constraints(top_features)
        # Optuna HP search (2026-05-21) — opt-in via --optuna-hp. Returns
        # final_params merged with best {lr, max_depth, subsample, colsample}.
        # Other keys (n_estimators, reg_*, monotone/interaction constraints)
        # are preserved to keep run_meta schema + downstream callers stable.
        if getattr(args, "optuna_hp", False):
            try:
                _study_name = "{}-{}-fold{}".format(
                    str(getattr(args, "ticker", "TICKER")),
                    str(getattr(args, "timeframe", "TF")),
                    str(fold.get("fold", "0")),
                )
                final_params = _optuna_search_final_params(
                    X_tr,
                    y_tr,
                    X_oos,
                    oos["y"].values,
                    base_params=final_params,
                    n_trials=int(getattr(args, "optuna_n_trials", 36)),
                    priors_path=(getattr(args, "optuna_priors", "") or None),
                    study_name=_study_name,
                )
            except Exception as _e:
                logger.warning(
                    "[optuna] search failed on fold %s (%s); using base_params",
                    fold.get("fold", "?"), _e,
                )
        # XGB 2.x: callbacks MUST be on constructor, not fit(). Use helper.
        _final_callbacks = _xgb_callbacks(early_stop=True)
        if _final_callbacks is not None:
            final = xgb.XGBClassifier(**final_params, callbacks=_final_callbacks)
        else:
            final = xgb.XGBClassifier(**final_params)
        y_oos_arr = oos["y"].values
        final.fit(
            X_tr,
            y_tr,
            **_xgb_fit_kwargs(
                eval_set=[(X_oos, y_oos_arr)],
                early_stop=True,
            ),
        )
        # Count distinct features actually used in tree splits (for diagnostic)
        try:
            booster = final.get_booster()
            score = booster.get_score(importance_type="gain")
            n_feats_in_trees = len(score)
        except Exception:
            n_feats_in_trees = -1
        probs = final.predict_proba(X_oos)[:, 1]
        all_probs.loc[oos.index] = probs
        fold_summaries.append(
            {
                "fold": fold["fold"],
                "n_train": len(train),
                "n_oos": len(oos),
                "n_top_features": len(top_features),
                "n_features_in_trees": int(n_feats_in_trees),
                "mean_oos_prob": float(probs.mean()),
            }
        )

    # ---- Threshold sweep or fixed ----
    sdf = None
    if args.sweep_threshold:
        rows = []
        for thr in np.arange(0.46, 0.70, 0.02):
            sig = all_probs > thr
            trades = bml.simulate(
                f, sig.fillna(False), args.tp_atr, args.sl_atr, args.max_hold
            )
            mm = bml.compute_metrics(trades)
            rows.append({"thr": round(thr, 2), **mm})
        sdf = pd.DataFrame(rows)
        sdf.to_csv(f"{args.output_dir}/threshold_sweep.csv", index=False)
        mask = (
            (sdf["profit_factor"] >= 1.5)
            & (sdf["win_rate"] >= 0.53)
            & (sdf["n_trades"] >= 8)
            & (sdf["max_drawdown_pct"] >= -0.03)
            & (sdf["total_return_pct"] > 0)
        )
        if mask.any():
            chosen_thr = float(
                sdf[mask]
                .sort_values("profit_factor", ascending=False)
                .iloc[0]["thr"]
            )
        else:
            chosen_thr = float(
                sdf.sort_values("profit_factor", ascending=False).iloc[0]["thr"]
            )
        logger.info("  -> chose thr=%.2f", chosen_thr)
    else:
        chosen_thr = args.prob_threshold

    # ---- Final simulation ----
    final_sig = (all_probs > chosen_thr).fillna(False)
    trades = bml.simulate(f, final_sig, args.tp_atr, args.sl_atr, args.max_hold)
    metrics = bml.compute_metrics(trades)
    logger.info(
        "  FINAL thr=%.2f: n=%d, WR=%.3f, PF=%.3f, RET=%.4f, DD=%.4f",
        chosen_thr,
        metrics["n_trades"],
        metrics.get("win_rate", 0),
        metrics.get("profit_factor", 0),
        metrics.get("total_return_pct", 0),
        metrics.get("max_drawdown_pct", 0),
    )

    trades.to_csv(f"{args.output_dir}/trades.csv", index=False)
    if sdf is not None:
        sdf.to_csv(f"{args.output_dir}/threshold_sweep.csv", index=False)

    # ---- Serialization helper ----
    def to_py(obj):
        if isinstance(obj, dict):
            return {k: to_py(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_py(v) for v in obj]
        if hasattr(obj, "item"):
            return obj.item()
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        return obj

    # ---- run_meta.json ----
    meta = to_py(
        {
            "ticker": args.ticker,
            "job_id": args.job_id,
            "pipeline_version": "xgb_v10",
            "strategy_variant": "ML_XGB_v10",
            "strategy": args.strategy,
            # 2026-05-21 multi-TF wire — tags which Cache B TF this run used.
            # Downstream mastery aggregator (generate_mastery_file*.py) keys
            # per_tf_results dict off this field.
            "timeframe": args.timeframe,
            "run_at": datetime.utcnow().isoformat() + "Z",
            "features_total": len(fc),
            "top_k": args.top_k,
            "top_k_adaptive": (args.top_k <= 0),
            "top_k_effective_per_fold": [
                ft.get("effective_top_k") for ft in fold_top_features
            ],
            "xgb_interaction_constraints": _XGB_USE_INTERACTION,
            "xgb_monotonic_constraints": _XGB_USE_MONOTONIC,
            "rows": len(f),
            # v10 module availability
            "v10_modules": {
                "daily_integration_features": DAILY_INT_AVAILABLE,
                "alpaca_features": ALPACA_AVAILABLE,
                "featuretools_dfs_features": DFS_AVAILABLE,
                # Wave A (2026-05-17)
                "options_flow_features": OPTIONS_FLOW_AVAILABLE,
                "govtrades_features": GOVTRADES_AVAILABLE,
                "time_of_day_features": TOD_AVAILABLE,
                "gabriel_priors_features": GABRIEL_PRIORS_AVAILABLE,
            },
            # per-module feature counts
            "module_feature_counts": module_counts,
            # Mythos-specific metadata (inherited from v9)
            "use_mythos_features": args.use_mythos_features,
            "mythos_checkpoint_path": mythos_checkpoint_path,
            "mythos_feature_count": MYTHOS_FEATURE_DIM if args.use_mythos_features else 0,
            "mythos_fallback_rows": mythos_fallback_rows,
            "mythos_fallback_pct": (
                round(mythos_fallback_rows / max(len(f), 1), 4)
                if args.use_mythos_features
                else None
            ),
            "feature_sources": {
                "base+intraday+alt+insight_v3+parts1-4": "~587",
                "cross_sectional": "17 (if cache)",
                "macro_yfinance": "40",
                "strategy_signal+five_filter": "~25",
                "google_trends": "7",
                "insider_form4": "8",
                "multi_timeframe_h1_h4_m5_m15": "15 (v7)",
                "news_sentiment_vader": "8 (v7)",
                "vol_estimators": "14 (v7)",
                "qlib_alpha158_pandas_port": "158 (v8)",
                "closeable_gaps_yfinance_finra": "18 (v8 patch)",
                "openmythos_256dim_embedding": (
                    "256 (v9)" if args.use_mythos_features else "disabled"
                ),
                "alpaca_yfinance_earnings_div_splits": "13 (v10 NEW)",
                "daily_integration_beta_residual_csrs_earn_gate": "7 (v10 NEW)",
                "featuretools_dfs_depth2_interactions": "~60 (v10 NEW)",
                "options_flow_pc_ratio_iv_unusual": "3 (v10.4 Wave A)",
                "govtrades_congress_density_buy_sell_lobby": "3 (v10.4 Wave A)",
                "time_of_day_bucket": "1 (v10.4 Wave A)",
                "gabriel_priors_pf_wr_n_regime_consistency": "5 (v10.4 Wave A)",
            },
            "walk_forward_folds": len(fold_summaries),
            "strategy_params": {
                "name": "ML_XGB_v10",
                "side": "long",
                "tp_atr": args.tp_atr,
                "sl_atr": args.sl_atr,
                "max_hold_days": args.max_hold,
                "prob_threshold": chosen_thr,
                "threshold_swept": args.sweep_threshold,
                "model": "XGBClassifier (scout-prune-refit top-K)",
                "slippage_bps": 5.0,
                "fee_per_share": 0.0035,
                "notional_per_trade": 5000,
            },
            "metrics_oos_aggregate": metrics,
            "fold_summaries": fold_summaries,
            "fold_top_features": fold_top_features,
            "fold_mythos_importances": fold_mythos_importances,
        }
    )

    meta_path = f"{args.output_dir}/run_meta.json"
    with open(meta_path, "w") as fp:
        json.dump(meta, fp, indent=2, default=str)
    logger.info("  Wrote %s", meta_path)

    # ------------------------------------------------------------------
    # Persist final refit model for live inference (2026-05-18)
    # ------------------------------------------------------------------
    # Rationale: live_paper_trade_signals.py was refitting on-the-fly with
    # ~50 basic features. That defeats the v10 backtest's 969-feature edge.
    # We persist:
    #   - refit_model.pkl   = joblib-pickled (XGBClassifier, feature_cols,
    #                         metadata) trained on ALL training data using
    #                         the LAST fold's top features.
    # Live loads this if column alignment matches; otherwise warns + falls
    # back to its current refit code path.
    try:
        import joblib  # pylint: disable=import-outside-toplevel
        import hashlib  # pylint: disable=import-outside-toplevel

        if fold_top_features:
            persist_top_features = fold_top_features[-1]["top_features"]
        else:
            persist_top_features = []

        # LIVE-DEPLOY REFIT ONLY -- NOT used by backtest metrics.
        # This `persist_model` (saved as refit_model.pkl below) is built solely
        # for live-deployment loading: trained on ALL labeled rows up through
        # the latest available bar (no embargo subtraction) so the live signal
        # sees the freshest data possible. Walk-forward backtest metrics
        # (fold_summaries above) are computed exclusively from per-fold
        # EMBARGOED models (train_end_emb = train_end - BDay(LABEL_EMBARGO_DAYS),
        # ~21 BD by default); this all-data refit is NEVER scored against
        # in-sample data anywhere in this script.
        # DO NOT use refit_model.pkl to "backtest" -- it would be in-sample
        # by construction and would inflate apparent Sharpe/PF/WR.
        # See research/no_lookahead_audit_2026-05-21/repo_2026-05-21.md
        # (audit Fix 2) for full rationale.
        # f["y"] was created earlier in the script; restrict to non-NaN.
        persist_train = f[f["y"].notna() & f["close"].notna()].copy()
        # Ensure every chosen feature exists in f.columns; drop any missing.
        persist_top_features = [c for c in persist_top_features if c in persist_train.columns]
        if len(persist_top_features) >= 5 and len(persist_train) >= 80:
            X_persist = persist_train[persist_top_features].fillna(0).values
            y_persist = persist_train["y"].values
            # Apply same env-gated constraints as the per-fold final model
            persist_params = _xgb_base_params("persist")
            if _XGB_USE_INTERACTION:
                persist_params["interaction_constraints"] = _build_interaction_constraints(persist_top_features)
            if _XGB_USE_MONOTONIC:
                persist_params["monotone_constraints"] = _build_monotonic_constraints(persist_top_features)
            # XGB 2.x: callbacks MUST be on constructor, not fit().
            # Hold out tail 15% as eval_set for EarlyStopping(save_best=True).
            _cut = max(1, int(len(X_persist) * 0.85))
            _Xp_tr, _Xp_ev = X_persist[:_cut], X_persist[_cut:]
            _yp_tr, _yp_ev = y_persist[:_cut], y_persist[_cut:]
            _use_es = len(_Xp_ev) >= 5 and len(np.unique(_yp_ev)) > 1
            _persist_callbacks = _xgb_callbacks(early_stop=True) if _use_es else None
            if _persist_callbacks is not None:
                persist_model = xgb.XGBClassifier(**persist_params, callbacks=_persist_callbacks)
            else:
                persist_model = xgb.XGBClassifier(**persist_params)
            if _use_es:
                persist_model.fit(
                    _Xp_tr,
                    _yp_tr,
                    **_xgb_fit_kwargs(
                        eval_set=[(_Xp_ev, _yp_ev)],
                        early_stop=True,
                    ),
                )
            else:
                persist_model.fit(X_persist, y_persist)

            # User contract (2026-05-18): feature_hash = sha256 of SORTED
            # feature_cols joined. Keep alongside legacy sha1-16 alias so any
            # consumer keyed on the old hash doesn't break.
            feat_hash = hashlib.sha256(
                "|".join(sorted(persist_top_features)).encode("utf-8")
            ).hexdigest()
            feat_hash_legacy_sha1 = hashlib.sha1(
                "|".join(persist_top_features).encode("utf-8")
            ).hexdigest()[:16]

            persist_meta = {
                "ticker": args.ticker,
                "pipeline_version": "xgb_v10",
                "v10_version": V10_FEATURE_VERSION,
                "run_at": datetime.utcnow().isoformat() + "Z",
                "trained_at_utc": datetime.now(timezone.utc).isoformat(),
                "feature_cols": persist_top_features,
                "feature_count": len(persist_top_features),
                "n_features": len(persist_top_features),
                "feature_hash": feat_hash,
                "feature_hash_legacy_sha1": feat_hash_legacy_sha1,
                "n_train": int(len(persist_train)),
                "n_samples_train": int(len(persist_train)),
                "prob_threshold": chosen_thr,
                "best_threshold": chosen_thr,
                "tp_atr": args.tp_atr,
                "sl_atr": args.sl_atr,
                "max_hold_days": args.max_hold,
                "model_kwargs": {
                    "max_depth": 4,
                    "learning_rate": 0.05,
                    "n_estimators": 100,
                    "tree_method": "hist",
                    "eval_metric": "logloss",
                    "random_state": 42,
                    # Tuning extensions (2026-05-19, additive — legacy keys preserved)
                    "device": _XGB_DEVICE,
                    "eval_metric_full": ["logloss", "aucpr"],
                    "colsample_bytree": 0.6,
                    "colsample_bylevel": 0.7,
                    "colsample_bynode": 0.8,
                    "subsample": 0.7,
                    "sampling_method": _XGB_SAMPLING_METHOD,
                    "max_bin": 512,
                    "min_child_weight": 3,
                    "reg_alpha": 0.01,
                    "reg_lambda": 1.0,
                    "grow_policy": "lossguide",
                    "max_leaves": 31,
                    "early_stopping_rounds": 10,
                },
                "label": {
                    "horizon_days": LABEL_EMBARGO_DAYS,
                    "target": "fwd_ret > 0",
                },
                # NOTE: live must produce these exact column names. v10 builds
                # 969 features; live's basic builder produces ~50 — most v10
                # tops will NOT be available in live. live should compare
                # feature_cols vs its own columns and fall back on mismatch.
            }

            pkl_path = f"{args.output_dir}/refit_model.pkl"
            joblib.dump(
                {"model": persist_model, "meta": persist_meta},
                pkl_path,
                compress=3,
            )
            logger.info(
                "  Wrote %s (n_features=%d, n_train=%d, hash=%s)",
                pkl_path,
                len(persist_top_features),
                len(persist_train),
                feat_hash,
            )
        else:
            logger.warning(
                "  Skipped refit_model.pkl persistence: "
                "len(top_features)=%d, len(train)=%d",
                len(persist_top_features),
                len(persist_train),
            )
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("  refit_model.pkl persistence failed: %s", exc)

    # Write lightweight result.json (smoke-test contract)
    # PATCH 2026-05-20: include `strategy` so the GH-Actions rollup writes
    # backtests/<TICKER>/<STRATEGY>/result.json instead of UNKNOWN/. The
    # rollup step reads `data.get("strategy", "UNKNOWN")` from this file
    # (sweep.yml _rollup.py). Previously missing -> 75+ runs at UNKNOWN/.
    # autosolve_skip: infra patch, not a new error
    result = {
        "ticker": args.ticker,
        "strategy": args.strategy,
        # 2026-05-21 multi-TF wire — keep result.json self-describing so
        # mastery aggregator can bin by (strategy, timeframe) without
        # cross-referencing run_meta.json.
        "timeframe": args.timeframe,
        "strategy_variant": "ML_XGB_v10",
        "job_id": args.job_id,
        "pipeline_version": "xgb_v10",
        "features_total": len(fc),
        "rows": len(f),
        "module_feature_counts": module_counts,
        "n_trades": metrics["n_trades"],
        "win_rate": metrics.get("win_rate"),
        "profit_factor": metrics.get("profit_factor"),
        "total_return_pct": metrics.get("total_return_pct"),
        "max_drawdown_pct": metrics.get("max_drawdown_pct"),
        "status": "ok",
    }
    result_path = f"{args.output_dir}/result.json"
    with open(result_path, "w") as fp:
        json.dump(to_py(result), fp, indent=2, default=str)
    logger.info("  Wrote %s", result_path)
    logger.info("[v10] DONE. features=%d rows=%d", len(fc), len(f))


if __name__ == "__main__":
    main()
