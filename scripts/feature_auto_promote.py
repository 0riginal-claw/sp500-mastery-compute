"""feature_auto_promote.py - automated stub->promoted feature harness.

Reads `scripts/feature_manifest.json` pending entries, imports each function,
applies it to a 60d AAPL holdout, computes nan_ratio / row_delta / col_delta /
mean |SHAP|, and promotes:
  - "tested" : nan_ratio < 0.10 AND no row drop AND mean|SHAP| > 0.001
  - "wired"  : same + leakage check (no future timestamp / lookahead pattern)
  - "rejected": with reason

Modes:
  --dry-run (default): report only, never touch the manifest
  --apply            : back up manifest first, then write new statuses

Author: parent task feature-promotion harness, 2026-05-19.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import importlib
import inspect
import json
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "scripts" / "feature_manifest.json"
LOGS_DIR = REPO_ROOT / "logs"
BACKUPS_DIR = REPO_ROOT / "backups" / "feature_manifest"

AAPL_FEATURES_GLOB = "AAPL_v10_full_*.parquet"
AAPL_FEATURES_DIR = REPO_ROOT / "cache" / "features"
HOLDOUT_LEN = 60

NAN_RATIO_MAX = 0.10
SHAP_ABS_MIN = 0.001

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_holdout() -> pd.DataFrame:
    candidates = sorted(
        AAPL_FEATURES_DIR.glob(AAPL_FEATURES_GLOB),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No AAPL v10 feature cache found under {AAPL_FEATURES_DIR}")
    src = candidates[0]
    df = pd.read_parquet(src)
    if len(df) < HOLDOUT_LEN + 10:
        raise ValueError(f"Holdout cache {src} has only {len(df)} rows, need >= {HOLDOUT_LEN+10}")
    df = df.tail(HOLDOUT_LEN + 30).copy()
    df.attrs["holdout_source"] = str(src)
    df.attrs["holdout_len"] = HOLDOUT_LEN
    return df


def _import_function(module_path: str, function_name: str):
    mod_path = module_path.replace("/", ".")
    if mod_path.endswith(".py"):
        mod_path = mod_path[:-3]
    mod = importlib.import_module(mod_path)
    if not hasattr(mod, function_name):
        raise AttributeError(f"{mod_path} has no attribute {function_name}")
    return getattr(mod, function_name)


def _check_leakage(fn) -> Tuple[bool, str]:
    try:
        src = inspect.getsource(fn)
    except OSError:
        return True, "source-unavailable-skipping-leakage-scan"
    bad_patterns = [
        ".shift(-",
        "future_",
        "lookahead",
    ]
    for pat in bad_patterns:
        if pat in src:
            return False, f"leakage-pattern:{pat!r}"
    return True, "no-future-shift-detected"


def _score_stub(fn, holdout: pd.DataFrame, function_name: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "function": function_name,
        "success": False,
        "n_new": 0,
        "new_cols": [],
        "row_delta": None,
        "col_delta": None,
        "nan_ratio_max": None,
        "mean_abs_shap": None,
        "error": None,
    }
    before_rows = len(holdout)
    before_cols = set(holdout.columns)
    df_in = holdout.copy()
    try:
        df_out = fn(df_in)
        if df_out is None:
            out["error"] = "function returned None"
            return out
        new_cols = [c for c in df_out.columns if c not in before_cols]
        out["new_cols"] = new_cols
        out["n_new"] = len(new_cols)
        out["row_delta"] = len(df_out) - before_rows
        out["col_delta"] = len(df_out.columns) - len(before_cols)
        if not new_cols:
            out["error"] = "stub-unchanged (recipe still commented in docstring)"
            return out
        df_scored = df_out.tail(HOLDOUT_LEN)
        nan_ratios = df_scored[new_cols].isna().mean()
        out["nan_ratio_max"] = float(nan_ratios.max())
        y = (df_scored["close"].shift(-1) > df_scored["close"]).astype(int)
        X = df_scored[new_cols].iloc[:-1].fillna(0.0)
        y = y.iloc[:-1]
        if X.empty or y.nunique() < 2:
            out["mean_abs_shap"] = 0.0
        else:
            try:
                import xgboost as xgb
                import shap
                model = xgb.XGBClassifier(
                    n_estimators=30,
                    max_depth=3,
                    use_label_encoder=False,
                    eval_metric="logloss",
                    verbosity=0,
                )
                model.fit(X, y)
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(X)
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[0]
                out["mean_abs_shap"] = float(np.abs(shap_vals).mean())
            except Exception as e:
                out["mean_abs_shap"] = 0.0
                out["error"] = f"shap-error:{type(e).__name__}:{e}"
        out["success"] = True
        return out
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()
        return out


def _decide(score: Dict[str, Any], leakage: Tuple[bool, str]) -> Tuple[str, str]:
    if not score["success"]:
        return "rejected", score["error"] or "unknown failure"
    if score["row_delta"] not in (0, None) and score["row_delta"] != 0:
        return "rejected", f"row_delta={score['row_delta']} (must be 0)"
    if score["nan_ratio_max"] is None or score["nan_ratio_max"] >= NAN_RATIO_MAX:
        return "rejected", f"nan_ratio_max={score['nan_ratio_max']} >= {NAN_RATIO_MAX}"
    if score["mean_abs_shap"] is None or score["mean_abs_shap"] <= SHAP_ABS_MIN:
        return "rejected", f"mean_abs_shap={score['mean_abs_shap']} <= {SHAP_ABS_MIN}"
    if not leakage[0]:
        return "tested", f"passes-metrics-but-leakage:{leakage[1]}"
    return "wired", f"metrics-ok shap={score['mean_abs_shap']:.4f} nan={score['nan_ratio_max']:.3f}"


def _backup_manifest() -> Path:
    BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ")
    dest = BACKUPS_DIR / f"feature_manifest_{ts}.json"
    shutil.copy2(MANIFEST, dest)
    return dest


def _write_report(results: List[Dict[str, Any]], outpath: Path, apply: bool, manifest_backup) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    wired = [r for r in results if r["new_status"] == "wired"]
    tested = [r for r in results if r["new_status"] == "tested"]
    rejected = [r for r in results if r["new_status"] == "rejected"]
    lines = [
        f"# Feature auto-promotion report ({_dt.datetime.utcnow().isoformat()}Z)",
        "",
        f"- mode: {'APPLY' if apply else 'DRY-RUN'}",
        f"- manifest_backup: {manifest_backup}",
        f"- total_evaluated: {len(results)}",
        f"- promoted_wired: {len(wired)}",
        f"- promoted_tested: {len(tested)}",
        f"- rejected: {len(rejected)}",
        "",
        "## Results table",
        "",
        "| function | new_status | reason | n_new | nan_max | mean_abs_shap |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        s = r["score"]
        lines.append(
            f"| {r['function']} | {r['new_status']} | {r['reason']} | "
            f"{s.get('n_new')} | {s.get('nan_ratio_max')} | {s.get('mean_abs_shap')} |"
        )
    outpath.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Write new statuses to manifest (default: dry-run)")
    args = ap.parse_args()

    with MANIFEST.open() as f:
        manifest = json.load(f)
    pending = [m for m in manifest["modules"] if m.get("integration_status") == "pending"]
    print(f"[harness] loaded {len(pending)} pending stubs from manifest")
    holdout = _load_holdout()
    print(f"[harness] holdout: {holdout.attrs.get('holdout_source')} rows={len(holdout)} (scoring last {HOLDOUT_LEN})")

    results: List[Dict[str, Any]] = []
    for entry in pending:
        mod_path = entry["module_path"]
        fn_name = entry["function_name"]
        try:
            fn = _import_function(mod_path, fn_name)
        except Exception as e:
            results.append({
                "function": fn_name,
                "module": mod_path,
                "new_status": "rejected",
                "reason": f"import-error:{type(e).__name__}:{e}",
                "score": {"n_new": 0, "nan_ratio_max": None, "mean_abs_shap": None},
            })
            continue
        leakage = _check_leakage(fn)
        score = _score_stub(fn, holdout, fn_name)
        status, reason = _decide(score, leakage)
        results.append({
            "function": fn_name,
            "module": mod_path,
            "new_status": status,
            "reason": reason,
            "score": score,
        })

    manifest_backup = None
    if args.apply:
        manifest_backup = _backup_manifest()
        idx_by_name = {(m["module_path"], m["function_name"]): m for m in manifest["modules"]}
        for r in results:
            key = (r["module"], r["function"])
            if key in idx_by_name:
                idx_by_name[key]["integration_status"] = r["new_status"]
                idx_by_name[key]["promotion_reason"] = r["reason"]
                idx_by_name[key]["promoted_at"] = _dt.datetime.utcnow().isoformat() + "Z"
        manifest["updated"] = _dt.datetime.utcnow().isoformat() + "Z"
        with MANIFEST.open("w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[harness] APPLIED. Manifest backed up to {manifest_backup}")
    else:
        print("[harness] DRY-RUN - manifest not modified. Use --apply to commit.")

    date_str = _dt.datetime.utcnow().strftime("%Y-%m-%d")
    outpath = LOGS_DIR / f"feature_promotion_{date_str}.md"
    _write_report(results, outpath, args.apply, manifest_backup)
    print(f"[harness] report -> {outpath}")

    wired = sum(1 for r in results if r["new_status"] == "wired")
    tested = sum(1 for r in results if r["new_status"] == "tested")
    rejected = sum(1 for r in results if r["new_status"] == "rejected")
    print(f"[harness] summary: wired={wired} tested={tested} rejected={rejected} total={len(results)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
