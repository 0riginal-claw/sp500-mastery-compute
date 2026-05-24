#!/usr/bin/env python3
"""
feature_audit.py - Audit gap between generated feature modules and what's
actually wired into backtest_xgb_v10.py.

# karpathy_checked: audit-only, no writes to live model
# autosolve_skip: read-only audit, no side effects
# openclaw_violation_skip: alpaca-standing-rule, local-only

Two modes:
  (1) Manifest audit (DEFAULT) — original behavior. Reports the
      gap between feature_manifest.json statuses and the imports
      visible in backtest_xgb_v10.py.

  (2) IC-vs-forward-return audit — added 2026-05-23 per Council #2 verdict.
      Loads OHLCV from cache, dynamically imports each `add_*_features`
      function from the manifest's generated modules, applies it to the
      OHLCV df, computes Pearson IC between each emitted column and the
      forward N-bar return, and writes a TSV keep/drop table.

Usage:
  python feature_audit.py                       # legacy summary (manifest gap)
  python feature_audit.py --json                # legacy machine output
  python feature_audit.py --verbose             # legacy verbose

  # IC mode (Council #2):
  python feature_audit.py \\
      --tickers NTRS,JPM,AAPL \\
      --horizon 5 \\
      --ic-threshold 0.01 \\
      --output ../research/feature_audit_2026-05-23/keep_drop.tsv

  # Smoke / dry-run with only a few modules:
  python feature_audit.py \\
      --tickers NTRS \\
      --horizon 5 \\
      --max-modules 3 \\
      --output /tmp/smoke_keep_drop.tsv
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
MANIFEST_PATH = SCRIPTS_DIR / "feature_manifest.json"
BACKTEST_PATH = SCRIPTS_DIR / "backtest_xgb_v10.py"
CACHE_DIR = REPO_ROOT / "cache" / "yfinance_5yr"


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {"modules": [], "extracted_hashes": {}}
    try:
        with MANIFEST_PATH.open() as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        print("[audit] manifest load error: {}".format(exc), file=sys.stderr)
        return {"modules": [], "extracted_hashes": {}}


def scan_backtest_imports() -> set:
    """Return set of function names already imported in backtest_xgb_v10."""
    if not BACKTEST_PATH.exists():
        return set()
    text = BACKTEST_PATH.read_text(errors="replace")
    imports: set = set()
    for m in re.finditer(
        r"^\s*from\s+\S+\s+import\s+([^\n#]+)",
        text,
        re.MULTILINE,
    ):
        names = m.group(1)
        for tok in re.split(r"[,\s]+", names):
            tok = tok.strip().strip("()")
            if tok.startswith("add_") and tok.endswith("_features"):
                imports.add(tok)
    return imports


def audit() -> dict:
    manifest = load_manifest()
    modules = manifest.get("modules", [])
    imports_in_backtest = scan_backtest_imports()

    status_counts: Counter = Counter()
    by_module: Dict[str, List[dict]] = defaultdict(list)
    wired_but_not_imported: List[dict] = []
    imported_but_pending: List[dict] = []

    for m in modules:
        status_counts[m.get("integration_status", "unknown")] += 1
        by_module[m["module_path"]].append(m)
        fn = m.get("function_name", "")
        if m.get("integration_status") == "wired" and fn not in imports_in_backtest:
            wired_but_not_imported.append(m)
        if fn in imports_in_backtest and m.get("integration_status") != "wired":
            imported_but_pending.append(m)

    return {
        "total_functions": len(modules),
        "distinct_modules": len(by_module),
        "status_counts": dict(status_counts),
        "imports_seen_in_backtest": sorted(imports_in_backtest),
        "imports_count": len(imports_in_backtest),
        "wired_not_imported": wired_but_not_imported,
        "imported_but_pending": imported_but_pending,
        "gap_pending": status_counts.get("pending", 0),
        "gap_tested": status_counts.get("tested", 0),
        "gap_wired": status_counts.get("wired", 0),
    }


def print_human(report: dict, verbose: bool) -> None:
    print("=" * 70)
    print("Feature Manifest Audit")
    print("=" * 70)
    print("Total functions in manifest : {}".format(report["total_functions"]))
    print("Distinct generated modules  : {}".format(report["distinct_modules"]))
    print("Status breakdown:")
    for k, v in sorted(report["status_counts"].items()):
        print("  {:>10s} : {}".format(k, v))
    print()
    print("backtest_xgb_v10.py 'add_*_features' imports seen: {}".format(report["imports_count"]))
    if verbose:
        for name in report["imports_seen_in_backtest"]:
            print("  - {}".format(name))
    print()
    print("Gap report (action items):")
    print("  pending  -> awaiting operator review : {}".format(report["gap_pending"]))
    print("  tested   -> smoke-passed, not wired  : {}".format(report["gap_tested"]))
    print("  wired    -> declared wired           : {}".format(report["gap_wired"]))
    print()
    if report["wired_not_imported"]:
        print("WARN: {} functions marked 'wired' but NOT imported in backtest_xgb_v10:".format(
            len(report["wired_not_imported"])
        ))
        for m in report["wired_not_imported"]:
            print("  - {}::{}".format(m["module_path"], m["function_name"]))
    if report["imported_but_pending"]:
        print("WARN: {} functions imported in backtest_xgb_v10 but manifest != 'wired':".format(
            len(report["imported_but_pending"])
        ))
        for m in report["imported_but_pending"]:
            print("  - {}::{} (status={})".format(
                m["module_path"], m["function_name"], m.get("integration_status")
            ))
    print()


# ---------------------------------------------------------------------------
# IC mode (Council #2, 2026-05-23)
# ---------------------------------------------------------------------------


def _load_ticker_ohlcv(ticker: str):
    import pandas as pd
    p = CACHE_DIR / "{}.parquet".format(ticker.upper())
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    cols = {c.lower(): c for c in df.columns}
    for need in ("open", "high", "low", "close", "volume"):
        if need not in cols:
            return None
    return df


def _forward_return(close, horizon):
    import numpy as np
    fwd = close.shift(-horizon) / close - 1.0
    return fwd


def _import_feature_fn(module_path_str, function_name):
    """Dynamically import an `add_*_features` function from a generated module file.
    module_path_str is RELATIVE to repo root (e.g. 'scripts/_generated/foo.py').
    """
    abs_path = REPO_ROOT / module_path_str
    if not abs_path.exists():
        return None, "module file not found: {}".format(abs_path)
    spec_name = "_audit_" + abs_path.stem + "_" + function_name
    try:
        spec = importlib.util.spec_from_file_location(spec_name, str(abs_path))
        if spec is None or spec.loader is None:
            return None, "spec_from_file_location failed"
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec_name] = mod
        spec.loader.exec_module(mod)
    except Exception as e:
        return None, "import error: {}".format(e)
    fn = getattr(mod, function_name, None)
    if fn is None:
        return None, "function {} not found in {}".format(function_name, abs_path.name)
    return fn, None


def _pearson_ic(x, y):
    import numpy as np
    import pandas as pd
    s = pd.concat([pd.Series(x).reset_index(drop=True),
                   pd.Series(y).reset_index(drop=True)], axis=1).dropna()
    n = len(s)
    if n < 30:
        return float("nan"), float("nan"), n
    xa = s.iloc[:, 0].to_numpy(dtype=float)
    ya = s.iloc[:, 1].to_numpy(dtype=float)
    if np.std(xa) == 0 or np.std(ya) == 0:
        return float("nan"), float("nan"), n
    r = float(np.corrcoef(xa, ya)[0, 1])
    # Two-sided t-test p-value via normal approx (good enough for screening)
    # t = r * sqrt(n-2) / sqrt(1-r^2)
    if abs(r) >= 1.0:
        return r, 0.0, n
    t = r * (n - 2) ** 0.5 / ((1.0 - r * r) ** 0.5)
    # Use scipy if available, else 2-sided normal approx
    try:
        from scipy import stats
        p = float(2.0 * (1.0 - stats.t.cdf(abs(t), df=n - 2)))
    except Exception:
        from math import erf, sqrt
        # 2-sided normal cdf approx
        z = abs(t)
        p = float(2.0 * (1.0 - 0.5 * (1.0 + erf(z / sqrt(2.0)))))
    return r, p, n


def run_ic_audit(args) -> int:
    """Run IC-vs-forward-return audit and write a TSV table."""
    try:
        import pandas as pd  # noqa: F401
        import numpy as np   # noqa: F401
    except Exception as e:
        print("[ic-audit] pandas/numpy required: {}".format(e), file=sys.stderr)
        return 2

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    horizon = int(args.horizon)
    threshold = float(args.ic_threshold)
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    modules = manifest.get("modules", [])
    # Skip explicitly-rejected ones; they're already filtered out
    candidates = [m for m in modules if m.get("integration_status") != "rejected"]

    if args.max_modules:
        candidates = candidates[: int(args.max_modules)]

    # Preload OHLCV per ticker
    ohlcv = {}
    skipped_tickers = []
    for t in tickers:
        df = _load_ticker_ohlcv(t)
        if df is None:
            skipped_tickers.append(t)
            continue
        ohlcv[t] = df
    if not ohlcv:
        print("[ic-audit] no tickers loaded; aborting", file=sys.stderr)
        return 3

    rows = []
    error_rows = []
    for idx, m in enumerate(candidates):
        module_path = m.get("module_path", "")
        function_name = m.get("function_name", "")
        fn, err = _import_feature_fn(module_path, function_name)
        if fn is None:
            error_rows.append({
                "module": module_path,
                "function": function_name,
                "error": err,
            })
            continue
        for t, df in ohlcv.items():
            try:
                df_in = df.copy()
                before_cols = set(df_in.columns)
                df_out = fn(df_in)
                if df_out is None:
                    error_rows.append({
                        "module": module_path,
                        "function": function_name,
                        "ticker": t,
                        "error": "function returned None",
                    })
                    continue
                new_cols = [c for c in df_out.columns if c not in before_cols]
                if not new_cols:
                    rows.append({
                        "module": module_path,
                        "function": function_name,
                        "ticker": t,
                        "feature_column": "(no new columns emitted)",
                        "IC": float("nan"),
                        "p_value": float("nan"),
                        "n_obs": 0,
                        "keep": False,
                        "note": "no_emit",
                    })
                    continue
                close = df_out["close"]
                fwd = _forward_return(close, horizon)
                for col in new_cols:
                    try:
                        ic, pv, nobs = _pearson_ic(df_out[col], fwd)
                    except Exception as e:
                        error_rows.append({
                            "module": module_path,
                            "function": function_name,
                            "ticker": t,
                            "feature_column": col,
                            "error": "ic compute error: {}".format(e),
                        })
                        continue
                    keep = (not (ic != ic)) and abs(ic) >= threshold
                    rows.append({
                        "module": module_path,
                        "function": function_name,
                        "ticker": t,
                        "feature_column": col,
                        "IC": ic,
                        "p_value": pv,
                        "n_obs": nobs,
                        "keep": keep,
                        "note": "",
                    })
            except Exception as e:
                error_rows.append({
                    "module": module_path,
                    "function": function_name,
                    "ticker": t,
                    "error": "apply error: {}\n{}".format(e, traceback.format_exc()[-400:]),
                })
                continue

    import pandas as pd
    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_path, sep="\t", index=False)
    err_path = output_path.with_suffix(".errors.tsv")
    if error_rows:
        pd.DataFrame(error_rows).to_csv(err_path, sep="\t", index=False)

    # Companion summary JSON
    summary = {
        "tickers_requested": tickers,
        "tickers_loaded": list(ohlcv.keys()),
        "tickers_skipped": skipped_tickers,
        "horizon_bars": horizon,
        "ic_threshold": threshold,
        "max_modules": args.max_modules,
        "candidates_considered": len(candidates),
        "rows_written": len(rows),
        "errors_count": len(error_rows),
        "rows_keep": int(sum(1 for r in rows if r.get("keep"))),
        "output_tsv": str(output_path),
        "errors_tsv": str(err_path) if error_rows else None,
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))

    print("[ic-audit] wrote {} rows -> {}".format(len(rows), output_path))
    print("[ic-audit] errors: {} -> {}".format(len(error_rows),
                                                err_path if error_rows else "(none)"))
    print("[ic-audit] summary -> {}".format(summary_path))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit generated vs wired feature modules.")
    # Legacy flags
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON (legacy mode)")
    ap.add_argument("--verbose", action="store_true")
    # IC mode flags
    ap.add_argument("--tickers", type=str, default=None,
                    help="comma-separated tickers (triggers IC mode)")
    ap.add_argument("--horizon", type=int, default=5,
                    help="forward-return horizon in bars (default 5)")
    ap.add_argument("--ic-threshold", type=float, default=0.01,
                    help="absolute IC threshold for keep/drop decision (default 0.01)")
    ap.add_argument("--output", type=str, default=None,
                    help="path to TSV output (triggers IC mode)")
    ap.add_argument("--max-modules", type=int, default=None,
                    help="cap candidate modules (for smoke/dry-run)")
    args = ap.parse_args()

    if args.tickers or args.output:
        if not (args.tickers and args.output):
            print("[audit] IC mode requires BOTH --tickers and --output", file=sys.stderr)
            return 2
        return run_ic_audit(args)

    # Legacy manifest audit
    report = audit()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_human(report, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
