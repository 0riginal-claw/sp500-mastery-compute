#!/usr/bin/env python3
"""
ONNX export of per-ticker XGBoost models for fast inference.

Tier-S #9 (2026-05-21). Converts xgboost-Booster (.pkl/.model/.json) to ONNX
format using `onnxmltools` so prediction can run via ONNX Runtime — empirical
gains: 10-100x throughput vs xgboost.predict() (a1cf916 benchmark), and lets
us drop the xgboost native runtime dep on serving boxes (the .onnx file is
self-contained).

Usage:
    # Single model
    python scripts/onnx_export_xgboost.py \\
        --input mastery_results/AAPL/ORB/model.pkl \\
        --output mastery_results/AAPL/ORB/model.onnx \\
        --n-features 1633

    # Sweep all per-ticker models
    python scripts/onnx_export_xgboost.py --sweep mastery_results/

The export auto-discovers feature count from the booster if not given. The
.onnx file lands next to the original .pkl by default.
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
from pathlib import Path
from typing import Optional


def _load_booster(path: Path):
    """Load an xgboost model from .pkl / .json / .model file."""
    import xgboost as xgb
    suffix = path.suffix.lower()
    if suffix == ".pkl":
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if isinstance(obj, xgb.Booster):
            return obj
        if hasattr(obj, "get_booster"):
            return obj.get_booster()
        if isinstance(obj, dict):
            for k in ("model", "booster", "xgb", "estimator"):
                if k in obj:
                    cand = obj[k]
                    if isinstance(cand, xgb.Booster):
                        return cand
                    if hasattr(cand, "get_booster"):
                        return cand.get_booster()
        raise TypeError(f"Unrecognized pickle payload in {path}: {type(obj)}")
    elif suffix in (".json", ".model", ".ubj"):
        bst = xgb.Booster()
        bst.load_model(str(path))
        return bst
    else:
        raise ValueError(f"Unknown model file extension: {suffix} ({path})")


def _detect_n_features(bst) -> int:
    """Best-effort feature count detection from xgboost Booster."""
    try:
        return bst.num_features()
    except Exception:
        pass
    names = getattr(bst, "feature_names", None)
    if names:
        return len(names)
    raise RuntimeError("Could not auto-detect n_features - pass --n-features explicitly")


def export_one(in_path: Path, out_path: Path, n_features: Optional[int] = None) -> dict:
    """Export a single xgboost model to ONNX. Returns a status dict."""
    try:
        from onnxmltools.convert import convert_xgboost
        # onnxmltools' xgboost converter requires the FloatTensorType from
        # the onnxmltools namespace (not onnxconverter_common) - they are
        # distinct classes despite identical APIs.
        from onnxmltools.convert.common.data_types import FloatTensorType
    except ImportError as exc:
        return {"input": str(in_path), "output": str(out_path),
                "ok": False, "error": f"onnxmltools missing: {exc}"}

    try:
        bst = _load_booster(in_path)
    except Exception as exc:
        return {"input": str(in_path), "output": str(out_path),
                "ok": False, "error": f"load failed: {exc}"}

    if n_features is None:
        try:
            n_features = _detect_n_features(bst)
        except Exception as exc:
            return {"input": str(in_path), "output": str(out_path),
                    "ok": False, "error": str(exc)}

    initial_types = [("input", FloatTensorType([None, n_features]))]
    try:
        onnx_model = convert_xgboost(bst, initial_types=initial_types)
    except Exception as exc:
        return {"input": str(in_path), "output": str(out_path),
                "ok": False, "error": f"convert failed: {exc}"}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fh:
        fh.write(onnx_model.SerializeToString())

    return {"input": str(in_path), "output": str(out_path),
            "ok": True, "n_features": n_features,
            "size_bytes": out_path.stat().st_size}


def sweep(root: Path) -> list:
    """Find every model.pkl under root and export it side-by-side as model.onnx."""
    results = []
    for pkl in glob.glob(str(root / "**" / "model.pkl"), recursive=True):
        in_path = Path(pkl)
        out_path = in_path.with_suffix(".onnx")
        if out_path.exists():
            results.append({"input": str(in_path), "output": str(out_path),
                            "ok": True, "skipped": "already exists"})
            continue
        r = export_one(in_path, out_path)
        results.append(r)
        status = "OK" if r["ok"] else "FAIL"
        print(f"[{status}] {in_path} -> {out_path}"
              + (f"  ({r.get('error','')})" if not r["ok"] else ""))
    return results


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, help="Path to xgboost .pkl/.json/.model")
    ap.add_argument("--output", type=str, help="Path to write .onnx (default: alongside input)")
    ap.add_argument("--n-features", type=int, default=None,
                    help="Feature dimension (auto-detected if omitted)")
    ap.add_argument("--sweep", type=str, default=None,
                    help="Sweep all model.pkl under this root (mastery_results/)")
    args = ap.parse_args(argv)

    if args.sweep:
        rs = sweep(Path(args.sweep))
        ok = sum(1 for r in rs if r["ok"])
        print(f"\nSweep done: {ok}/{len(rs)} ok")
        return 0 if ok == len(rs) else 1

    if not args.input:
        ap.error("Pass --input <path> or --sweep <root>")

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path.with_suffix(".onnx")
    r = export_one(in_path, out_path, n_features=args.n_features)
    print(r)
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
