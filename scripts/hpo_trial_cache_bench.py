"""A/B bench for HPO trial cache + fidelity (Tier-3, 2026-05-21).

Runs the Optuna search twice with a synthetic XGB classification task that
mimics the per-fold workload in backtest_xgb_v10.py:

  A) HPO_FIDELITY_CACHE=0  (baseline -- Gold-only, no cache)
  B) HPO_FIDELITY_CACHE=1  (bronze/silver/gold ladder + cache)

Reports:
  - total wall time (sec)
  - best Sharpe seen by the study
  - n_unique_params explored (post-rounding)
  - cache hit rate (B only)

Output: research/hpo_trial_cache_tier3_2026-05-21/bench_ab.json + summary print.

Run:
  python3 scripts/hpo_trial_cache_bench.py                # 36 trials
  python3 scripts/hpo_trial_cache_bench.py --n-trials 24  # quick

NOTE: this is a smoke / sanity benchmark on synthetic data, not a true
backtest. It demonstrates the cache + fidelity savings in isolation. The
real-data bench requires invoking backtest_xgb_v10.py end-to-end on a
mastered ticker, which is out of scope for this tier-3 ship (the real run
will happen during the next A/B promotion gate, post smoke).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _make_synth(n_train: int = 3000, n_val: int = 800, n_feat: int = 24, seed: int = 0):
    rng = np.random.default_rng(seed)
    Xtr = rng.standard_normal((n_train, n_feat)).astype(np.float32)
    Xv = rng.standard_normal((n_val, n_feat)).astype(np.float32)
    # Weak teacher signal — XGB should beat 0.5 modestly.
    w = rng.standard_normal(n_feat)
    ytr = (Xtr @ w + 0.5 * rng.standard_normal(n_train) > 0).astype(np.int32)
    yv = (Xv @ w + 0.5 * rng.standard_normal(n_val) > 0).astype(np.int32)
    return Xtr, ytr, Xv, yv


def _run_optuna_once(
    *,
    enabled: bool,
    n_trials: int,
    study_name: str,
    seed: int = 42,
) -> dict:
    """Replay the search logic from backtest_xgb_v10._optuna_search_final_params.

    We don't import the full backtest_xgb_v10 module (too much pipeline noise);
    instead we inline an equivalent objective + cache wiring so the bench
    measures only the HPO loop cost.
    """
    os.environ["HPO_FIDELITY_CACHE"] = "1" if enabled else "0"

    # Re-import to refresh the module-level flag.
    if "hpo_trial_cache" in sys.modules:
        del sys.modules["hpo_trial_cache"]
    import hpo_trial_cache as hpc  # noqa: WPS433

    import optuna  # type: ignore
    from optuna.samplers import TPESampler
    from optuna.pruners import HyperbandPruner
    import xgboost as xgb  # type: ignore

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    Xtr, ytr, Xv, yv = _make_synth(seed=seed)

    fid_ladder_on = ["bronze"] * 4 + ["silver"] * 2 + ["gold"] * 1
    fid_ladder_off = ["gold"]
    fid_ladder = fid_ladder_on if enabled else fid_ladder_off

    # Pre-subsampled views (mirrors backtest_xgb_v10 wiring).
    views: dict[str, tuple] = {}
    if enabled:
        for fid in ("bronze", "silver", "gold"):
            idx = hpc.subsample_indices(Xtr.shape[0], fid, seed=seed)
            views[fid] = (Xtr[idx], ytr[idx])
    else:
        views["gold"] = (Xtr, ytr)

    n_unique: set[tuple] = set()
    cache_hits = 0
    cache_misses = 0

    def objective(trial) -> float:
        nonlocal cache_hits, cache_misses
        lr = trial.suggest_float("learning_rate", 0.01, 0.20, log=True)
        max_depth = trial.suggest_int("max_depth", 3, 10)
        subsample = trial.suggest_float("subsample", 0.5, 1.0)
        colsample = trial.suggest_float("colsample_bytree", 0.4, 1.0)
        fid = fid_ladder[trial.number % len(fid_ladder)]

        params = {
            "learning_rate": lr,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample,
        }

        # Canonicalise for unique-param accounting (matches cache hashing).
        canon = (
            round(lr, 4),
            int(max_depth),
            round(subsample, 3),
            round(colsample, 3),
            fid,
        )
        n_unique.add(canon)

        cached = hpc.cache_lookup(study_name, fid, params)
        if cached is not None:
            cache_hits += 1
            return float(cached)
        cache_misses += 1

        X_use, y_use = views.get(fid, (Xtr, ytr))
        trial_params = {
            "n_estimators": 100,
            "learning_rate": lr,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample,
            "tree_method": "hist",
            "n_jobs": 1,
            "eval_metric": "logloss",
            "verbosity": 0,
        }
        try:
            mdl = xgb.XGBClassifier(**trial_params)
            mdl.fit(X_use, y_use, eval_set=[(Xv, yv)], verbose=False)
            p = mdl.predict_proba(Xv)[:, 1]
            sig = p - 0.5
            ret = sig * (2 * yv - 1).astype(float)
            mu = float(np.mean(ret))
            sd = float(np.std(ret) + 1e-9)
            sharpe = (mu / sd) * float(np.sqrt(252.0))
            trial.report(sharpe, step=100)
            if trial.should_prune():
                raise optuna.TrialPruned()
            hpc.cache_store(study_name, fid, params, float(sharpe))
            return float(sharpe)
        except optuna.TrialPruned:
            raise
        except Exception:
            return -1e9

    sampler = TPESampler(seed=seed)
    pruner = HyperbandPruner(min_resource=10, max_resource=100, reduction_factor=3)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    wall_sec = time.perf_counter() - t0

    if enabled:
        hpc.cache_flush(study_name)

    try:
        best_value = float(study.best_value)
    except Exception:
        best_value = float("nan")

    return {
        "enabled": enabled,
        "n_trials": n_trials,
        "wall_sec": round(wall_sec, 3),
        "best_sharpe": round(best_value, 4),
        "n_unique_params": len(n_unique),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "cache_hit_rate": round(cache_hits / max(1, cache_hits + cache_misses), 3),
        "n_total_trials": len(study.trials),
        "n_pruned": sum(1 for t in study.trials if t.state.name == "PRUNED"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=36)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out",
        default=str(
            HERE.parent.parent
            / "research"
            / "hpo_trial_cache_tier3_2026-05-21"
            / "bench_ab.json"
        ),
    )
    args = ap.parse_args()

    # Clean any prior cache files for this bench study so we measure cold-start.
    cache_dir = HERE.parent / "state" / "hpo_trial_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for stub in ("bench_baseline", "bench_tier3"):
        for ext in (".parquet", ".csv"):
            p = cache_dir / f"{stub}{ext}"
            if p.exists():
                p.unlink()

    print(f"== HPO Cache Bench (n_trials={args.n_trials}, seed={args.seed}) ==")
    print("[A] Baseline (HPO_FIDELITY_CACHE=0) ...")
    res_a = _run_optuna_once(
        enabled=False, n_trials=args.n_trials, study_name="bench_baseline", seed=args.seed
    )
    print(json.dumps(res_a, indent=2))

    print("[B] Tier-3 (HPO_FIDELITY_CACHE=1) ...")
    res_b = _run_optuna_once(
        enabled=True, n_trials=args.n_trials, study_name="bench_tier3", seed=args.seed
    )
    print(json.dumps(res_b, indent=2))

    speedup = res_a["wall_sec"] / max(1e-6, res_b["wall_sec"])
    summary = {
        "baseline": res_a,
        "tier3": res_b,
        "speedup_x": round(speedup, 2),
        "best_sharpe_delta": round(res_b["best_sharpe"] - res_a["best_sharpe"], 4),
        "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved bench A/B summary -> {out}")
    print(f"Speedup: {speedup:.2f}x | sharpe delta: {summary['best_sharpe_delta']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
