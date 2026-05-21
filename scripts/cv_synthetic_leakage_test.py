# autosolve_skip: pypbo missing handled (replaced with purgedcv.deflated_sharpe_ratio); no new error
"""
cv_synthetic_leakage_test.py — Synthetic-leakage CI gate.

Pattern per López de Prado / shatianming5 Agent_market ade14e0 §E:
Build a target that NO feature can predict (pure noise label, correlated
features). If the CV reports R² (or AUC) materially above the noise floor
on the test fold, the CV is leaking.

Usage:
    python cv_synthetic_leakage_test.py [--n_samples 2000] [--n_features 30]
                                        [--horizon-days 21] [--threshold 0.05]

Exit code: 0 = clean (no leakage detected), 1 = leakage detected, 2 = error.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

try:
    from purgedcv import PurgedKFold, WalkForwardSplit  # noqa: F401
    from purgedcv.diagnostics import (
        assert_no_temporal_leakage,
        assert_embargo_respected,
    )
except ImportError as e:
    print(f"[FATAL] purgedcv not installed: {e}", file=sys.stderr)
    sys.exit(2)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def make_synthetic(n_samples: int, n_features: int, horizon_days: int, seed: int = 42):
    """Generate time-indexed data with random label uncorrelated to features.

    AUC should be ~0.5; R² should be near zero. Any material deviation
    on PurgedKFold splits indicates a CV-side leakage bug.
    """
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2018-01-02", periods=n_samples, freq="B")
    X = pd.DataFrame(
        rng.randn(n_samples, n_features),
        index=idx,
        columns=[f"f{i}" for i in range(n_features)],
    )
    # Pure-noise binary label, independent of X
    y = pd.Series(rng.randint(0, 2, n_samples), index=idx, name="y")
    # Label evaluation_times = prediction_times + horizon (López de Prado)
    pred_times = pd.Series(idx, index=idx)
    eval_times = pd.Series(idx + pd.tseries.offsets.BDay(horizon_days), index=idx)
    return X, y, pred_times, eval_times


def main() -> int:
    ap = argparse.ArgumentParser(description="Synthetic-leakage CV gate")
    ap.add_argument("--n-samples", type=int, default=2000)
    ap.add_argument("--n-features", type=int, default=30)
    ap.add_argument("--horizon-days", type=int, default=21)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--embargo-days", type=int, default=21)
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="Max |AUC - 0.5| considered noise floor")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    X, y, pred_times, eval_times = make_synthetic(
        args.n_samples, args.n_features, args.horizon_days, seed=args.seed
    )

    # 1) Structural assertions: purged + embargo must be respected
    pkf = PurgedKFold(
        n_splits=args.n_splits,
        prediction_times=pred_times,
        evaluation_times=eval_times,
        embargo=pd.Timedelta(days=args.embargo_days),
    )

    aucs = []
    for fold_i, (tr, te) in enumerate(pkf.split(X)):
        # purgedcv invariants - raise on violation
        assert_no_temporal_leakage(
            tr, te,
            prediction_times=pred_times,
            evaluation_times=eval_times,
        )
        assert_embargo_respected(
            tr, te,
            prediction_times=pred_times,
            evaluation_times=eval_times,
            embargo=pd.Timedelta(days=args.embargo_days),
        )
        clf = LogisticRegression(max_iter=200)
        clf.fit(X.values[tr], y.values[tr])
        prob = clf.predict_proba(X.values[te])[:, 1]
        auc = roc_auc_score(y.values[te], prob)
        aucs.append(auc)
        print(f"[fold {fold_i+1}] n_tr={len(tr):5d} n_te={len(te):5d}  AUC={auc:.4f}")

    mean_auc = float(np.mean(aucs))
    deviation = abs(mean_auc - 0.5)
    print(f"\n[synthetic-leakage-gate] mean AUC={mean_auc:.4f}  "
          f"deviation_from_0.5={deviation:.4f}  threshold={args.threshold}")
    if deviation > args.threshold:
        print(f"[FAIL] CV is leaking - features cannot predict pure-noise label, "
              f"but AUC={mean_auc:.4f} (expected ~0.5).", file=sys.stderr)
        return 1
    print("[PASS] No leakage detected - pure-noise label correctly hits noise floor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
