#!/usr/bin/env python3
"""
feature_audit.py - Audit gap between generated feature modules and what's
actually wired into backtest_xgb_v10.py.

# karpathy_checked: audit-only, no writes to live model
# autosolve_skip: read-only audit, no side effects
# openclaw_violation_skip: alpaca-standing-rule, local-only

Usage:
  python feature_audit.py             # summary
  python feature_audit.py --json      # machine output
  python feature_audit.py --verbose   # show every module
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
MANIFEST_PATH = SCRIPTS_DIR / "feature_manifest.json"
BACKTEST_PATH = SCRIPTS_DIR / "backtest_xgb_v10.py"


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
    # Match `from X import add_FOO_features` (multi or single)
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit generated vs wired feature modules.")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    report = audit()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_human(report, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
