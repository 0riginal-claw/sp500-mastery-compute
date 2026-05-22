"""Warm-cache replay bench (Tier-3 supplement).

Runs the tier-3 Optuna loop twice with HPO_FIDELITY_CACHE=1 in sequence
against the SAME study name + seed. Run #2 should hit the cache on every
trial -> demonstrates the lookup layer in isolation from fidelity savings.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from hpo_trial_cache_bench import _run_optuna_once  # type: ignore


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
            / "bench_warm.json"
        ),
    )
    args = ap.parse_args()

    cache_dir = HERE.parent / "state" / "hpo_trial_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Wipe any prior warm-bench file so run #1 starts cold.
    for ext in (".parquet", ".csv"):
        p = cache_dir / f"bench_warm{ext}"
        if p.exists():
            p.unlink()

    print(f"== Warm-cache replay bench (n_trials={args.n_trials}) ==")
    print("[1] Cold run (cache empty) ...")
    run1 = _run_optuna_once(
        enabled=True,
        n_trials=args.n_trials,
        study_name="bench_warm",
        seed=args.seed,
    )
    print(json.dumps(run1, indent=2))

    print("[2] Warm run (cache populated, same study + seed) ...")
    run2 = _run_optuna_once(
        enabled=True,
        n_trials=args.n_trials,
        study_name="bench_warm",
        seed=args.seed,
    )
    print(json.dumps(run2, indent=2))

    summary = {
        "cold": run1,
        "warm": run2,
        "warm_speedup_x": round(run1["wall_sec"] / max(1e-6, run2["wall_sec"]), 2),
        "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved warm-bench summary -> {out}")
    print(
        f"Warm speedup: {summary['warm_speedup_x']:.2f}x | "
        f"warm hit_rate: {run2['cache_hit_rate'] * 100:.1f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
