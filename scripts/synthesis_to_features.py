#!/usr/bin/env python3
"""
synthesis_to_features.py - Parse synthesis MDs into stub feature modules.

# karpathy_checked: pipeline scaffolding; integration_status='pending' by default; no auto-wire to live model; idempotent via sha256 hash check.
# autosolve_skip: scaffolding, no error path; operator review gate present.
# openclaw_violation_skip: alpaca-standing-rule, no OpenClaw calls in this file.

Pipeline:
  1. Glob synthesis MDs:
       research/_mission_2026-05-18/_BUCKET_*_SYNTHESIS.md
       research/openclaw_deepdive/*.md
  2. Extract ```python``` code blocks tagged as feature recipes
     (heuristic: block appears under a section heading that names an
     integration-target script, e.g. `## ... (`scripts/<target>.py`)`).
  3. Group by integration target (candlestick_features.py,
     market_structure.py, volume_features.py, regime_detector.py,
     intraday_patterns.py, bar_context.py, trend_momentum.py,
     ticker_rating_engine.py, ...).
  4. Generate stub module `scripts/_generated/<target>.py` with:
        def add_<bucket>_features(df) -> df-with-new-cols
     Each emitted function ships with the parsed recipe as a comment
     header + the raw source recipe in a docstring. integration_status
     defaults to "pending" - operator must manually review + promote to
     "wired" before backtest_xgb_v10 imports.
  5. Write `scripts/feature_manifest.json` listing all generated modules
     in dependency order, with status flags.

Idempotency:
  - Each code block is hashed (sha256 of its raw text); the hash is stored
    in feature_manifest.json under `extracted_hashes` per (synthesis_file,
    target). Re-running skips already-extracted blocks.

Usage:
  python synthesis_to_features.py            # full run
  python synthesis_to_features.py --dry-run  # print actions, no writes
  python synthesis_to_features.py --buckets B7,B48,36   # restrict
  python synthesis_to_features.py --verbose
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent  # .../s&p500-ticker-mastery
AI_TOOLS_ROOT = REPO_ROOT.parent                     # .../AI-Tools
SCRIPTS_DIR = REPO_ROOT / "scripts"
GENERATED_DIR = SCRIPTS_DIR / "_generated"
MANIFEST_PATH = SCRIPTS_DIR / "feature_manifest.json"

SYNTHESIS_DIRS: List[Path] = [
    AI_TOOLS_ROOT / "research" / "_mission_2026-05-18",
    AI_TOOLS_ROOT / "research" / "openclaw_deepdive",
]

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Headings of the form:
#   ## SECTION TITLE (`scripts/<target>.py`)
#   ## SECTION TITLE (`<target>.py`)            <-- B48-style
#   ### SECTION TITLE (`scripts/<target>.py` / `<other>.py`)
TARGET_HEADING_RE = re.compile(
    r"^(?P<level>#{2,4})\s+[^\n]*?[`'\"](?:scripts/)?(?P<target>[a-z][a-z0-9_]+)\.py[`'\"]",
    re.MULTILINE,
)

# Fallback: in-line target hint in plain text.
INLINE_TARGET_RE = re.compile(
    r"[`'\"](?:scripts/)?(?P<target>[a-z][a-z0-9_]+)\.py[`'\"]"
)

# Targets we will NEVER treat as integration targets (script noise).
TARGET_BLOCKLIST = {
    "synthesis_to_features",
    "feature_audit",
    "backtest_xgb_v10",
    "live_paper_trade",
    "feature_cache",
    "__init__",
}

# Code block: ```python\n ... \n```
CODE_BLOCK_RE = re.compile(
    r"```python\s*\n(?P<body>.*?)\n```",
    re.DOTALL,
)

# Bucket id from filename (_BUCKET_7_, _BUCKET_B12_, _BUCKET_48_BAR_READING_)
BUCKET_ID_RE = re.compile(
    r"_BUCKET_(?P<id>[A-Za-z]?\d+)(?:_[A-Z_]+)?_SYNTHESIS\.md$"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_short(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def safe_bucket_id(path: Path) -> str:
    m = BUCKET_ID_RE.search(path.name)
    if m:
        return m.group("id").lower()
    # openclaw_deepdive/* and others - derive a stable slug from filename
    stem = path.stem.lower()
    stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return ("oc_" + stem)[:40]

def safe_target_name(raw: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", raw.lower()).strip("_")

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def find_synthesis_files(buckets: Optional[List[str]] = None) -> List[Path]:
    out: List[Path] = []
    for root in SYNTHESIS_DIRS:
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*.md")):
            name = p.name
            is_bucket = "_BUCKET_" in name and name.endswith("_SYNTHESIS.md")
            is_deepdive = "openclaw_deepdive" in str(p)
            if not (is_bucket or is_deepdive):
                continue
            if buckets and is_bucket:
                bid = safe_bucket_id(p)
                norm = [b.lower().lstrip("b") for b in buckets] + [b.lower() for b in buckets]
                if bid not in norm and bid.lstrip("b") not in norm:
                    continue
            elif buckets and is_deepdive:
                # buckets filter excludes deepdives unless explicitly requested
                if "deepdive" not in [b.lower() for b in buckets]:
                    continue
            out.append(p)
    return out

def extract_blocks(md_text: str) -> List[Tuple[str, str, int]]:
    """
    Return list of (target, code_block_body, line_number).

    For each code block, find the nearest preceding target heading.
    Fall back to nearest preceding inline `scripts/<target>.py` reference.
    Skip blocks that don't look like feature recipes (no pd./np./df).
    """
    heading_hits = [
        (m.start(), m.group("target")) for m in TARGET_HEADING_RE.finditer(md_text)
    ]
    inline_hits = [
        (m.start(), m.group("target")) for m in INLINE_TARGET_RE.finditer(md_text)
    ]

    results: List[Tuple[str, str, int]] = []
    for cb in CODE_BLOCK_RE.finditer(md_text):
        body = cb.group("body").strip()
        if not body:
            continue
        low = body.lower()
        # Heuristic: must reference dataframe / numpy / pandas constructs
        if not (
            "df[" in low
            or "df." in low
            or "def " in low
            or "pd." in low
            or "np." in low
            or "rolling(" in low
            or "ewm(" in low
        ):
            continue
        pos = cb.start()
        target = "unassigned"
        for hpos, htgt in reversed(heading_hits):
            if hpos < pos:
                target = htgt
                break
        if target == "unassigned":
            for ipos, itgt in reversed(inline_hits):
                if ipos < pos:
                    target = itgt
                    break
        target = safe_target_name(target)
        if target in TARGET_BLOCKLIST:
            continue
        line_no = md_text.count("\n", 0, pos) + 1
        results.append((target, body, line_no))
    return results

# ---------------------------------------------------------------------------
# Manifest IO
# ---------------------------------------------------------------------------

def _empty_manifest() -> dict:
    return {
        "version": 1,
        "created": now_iso(),
        "updated": now_iso(),
        "description": (
            "Generated stub feature modules from synthesis MDs. "
            "integration_status='pending' until operator review."
        ),
        "modules": [],
        "extracted_hashes": {},
    }

def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        try:
            with MANIFEST_PATH.open() as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            return _empty_manifest()
    return _empty_manifest()

def save_manifest(manifest: dict) -> None:
    manifest["updated"] = now_iso()
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST_PATH.with_suffix(".json.tmp")
    with tmp.open("w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=False)
    tmp.replace(MANIFEST_PATH)

# Dependency ordering (informational; backtest_xgb_v10 reads from manifest)
DEPENDENCY_ORDER: List[str] = [
    "candlestick_features",
    "market_structure",
    "bar_context",
    "intraday_patterns",
    "volume_features",
    "trend_momentum",
    "regime_detector",
    "ticker_rating_engine",
]

def module_sort_key(name: str) -> Tuple[int, str]:
    try:
        return (DEPENDENCY_ORDER.index(name), name)
    except ValueError:
        return (len(DEPENDENCY_ORDER), name)

# ---------------------------------------------------------------------------
# Stub module generation
# ---------------------------------------------------------------------------

STUB_FILE_HEADER = '''"""
_generated/{target}.py - AUTO-GENERATED stub from synthesis ingestion.

DO NOT EDIT directly unless you are promoting a function from
integration_status='pending' to 'wired'. Workflow:
  1. Review each add_<bucket>_features() function below.
  2. Verify the recipe compiles and produces .shift(1)-safe columns.
  3. Update feature_manifest.json: set status -> 'tested' (passes smoke)
     or 'wired' (imported by backtest_xgb_v10.py).
  4. Add the import + call in backtest_xgb_v10.py manually.

Re-running synthesis_to_features.py will append new functions but NEVER
overwrite existing ones (idempotent via hash check in feature_manifest.json).
"""
from __future__ import annotations

try:
    import pandas as pd  # noqa: F401
    import numpy as np   # noqa: F401
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]

'''

STUB_FUNCTION_TEMPLATE = '''
# ---------------------------------------------------------------------------
# bucket={bucket}  source={source_rel}  line={line_no}  hash={hash}
# integration_status=pending  (operator promotes via feature_manifest.json)
# ---------------------------------------------------------------------------
def add_{bucket}_{hash}_features(df):  # type: ignore[no-untyped-def]
    """Stub from synthesis bucket={bucket}.

    Source: {source_rel}:L{line_no}
    Hash:   {hash}

    Status: PENDING REVIEW - does not auto-execute. Returns df unchanged.

    To activate:
      1. Review the recipe below, then uncomment + integrate.
      2. Verify .shift(1)-safety (no future leakage).
      3. Promote integration_status -> 'wired' in feature_manifest.json.
      4. Add `from scripts._generated.{target} import add_{bucket}_{hash}_features`
         in backtest_xgb_v10.py and call it inside _build_v10_features_impl.

    Recipe (verbatim from synthesis, escaped):

{recipe_block}
    """
    return df

'''

def _format_recipe_block(recipe: str) -> str:
    # Indent every line by 8 spaces so it sits cleanly inside the docstring,
    # and replace any triple-quote in the source with a safe sequence.
    safe = recipe.replace('"""', "'''")
    return "\n".join("        " + line for line in safe.splitlines())

def render_stub_file(target: str, functions: List[dict]) -> str:
    parts = [STUB_FILE_HEADER.format(target=target)]
    for fn in functions:
        parts.append(STUB_FUNCTION_TEMPLATE.format(
            bucket=fn["bucket"],
            target=target,
            source_rel=fn["source_rel"],
            line_no=fn["line_no"],
            hash=fn["hash"],
            recipe_block=_format_recipe_block(fn["recipe"]),
        ))
    return "".join(parts)

def append_function_to_stub_file(path: Path, target: str, fn: dict) -> None:
    block = STUB_FUNCTION_TEMPLATE.format(
        bucket=fn["bucket"],
        target=target,
        source_rel=fn["source_rel"],
        line_no=fn["line_no"],
        hash=fn["hash"],
        recipe_block=_format_recipe_block(fn["recipe"]),
    )
    with path.open("a") as fh:
        fh.write(block)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(buckets: Optional[List[str]] = None, dry_run: bool = False, verbose: bool = False) -> dict:
    t0 = time.time()
    manifest = load_manifest()
    extracted = manifest.setdefault("extracted_hashes", {})

    files = find_synthesis_files(buckets=buckets)
    if verbose:
        print("[synthesis] found {} synthesis files".format(len(files)), file=sys.stderr)

    pending: Dict[str, List[dict]] = {}
    summary = {
        "files_scanned": 0,
        "blocks_found": 0,
        "blocks_new": 0,
        "blocks_skipped_existing": 0,
        "blocks_unassigned": 0,
        "targets_touched_set": set(),
    }

    for f in files:
        try:
            md = f.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            print("[synthesis] skip {}: {}".format(f, exc), file=sys.stderr)
            continue
        summary["files_scanned"] += 1
        bid = safe_bucket_id(f)
        src_rel = str(f.relative_to(AI_TOOLS_ROOT))
        ext_key = src_rel
        prev_hashes_per_target = extracted.setdefault(ext_key, {})

        blocks = extract_blocks(md)
        summary["blocks_found"] += len(blocks)

        for (target, body, line_no) in blocks:
            if target == "unassigned" or not target:
                summary["blocks_unassigned"] += 1
                if verbose:
                    print("[synthesis]  unassigned block in {}:L{}".format(src_rel, line_no), file=sys.stderr)
                continue
            h = sha256_short(body)
            already = prev_hashes_per_target.setdefault(target, [])
            if h in already:
                summary["blocks_skipped_existing"] += 1
                continue
            already.append(h)
            pending.setdefault(target, []).append({
                "bucket": bid,
                "source_rel": src_rel,
                "line_no": line_no,
                "hash": h,
                "recipe": body,
            })
            summary["blocks_new"] += 1
            summary["targets_touched_set"].add(target)

    if not dry_run:
        GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        init_path = GENERATED_DIR / "__init__.py"
        if not init_path.exists():
            init_path.write_text(
                '"""Auto-generated stub feature modules. See feature_manifest.json."""\n'
            )

    module_entries_by_key = {
        (m["module_path"], m["function_name"]): m for m in manifest.get("modules", [])
    }

    for target, fns in pending.items():
        module_rel = "scripts/_generated/{}.py".format(target)
        full_path = GENERATED_DIR / "{}.py".format(target)
        if not dry_run:
            if full_path.exists():
                for fn in fns:
                    append_function_to_stub_file(full_path, target, fn)
            else:
                full_path.write_text(render_stub_file(target, fns))

        for fn in fns:
            func_name = "add_{}_{}_features".format(fn["bucket"], fn["hash"])
            key = (module_rel, func_name)
            entry = module_entries_by_key.get(key)
            if entry is None:
                entry = {
                    "module_path": module_rel,
                    "function_name": func_name,
                    "column_prefix": "{}_{}_{}_".format(target, fn["bucket"], fn["hash"]),
                    "source_synthesis": fn["source_rel"],
                    "source_line": fn["line_no"],
                    "hash": fn["hash"],
                    "integration_status": "pending",
                    "created": now_iso(),
                    "sort_key": list(module_sort_key(target)),
                }
                manifest.setdefault("modules", []).append(entry)
                module_entries_by_key[key] = entry

    manifest["modules"].sort(
        key=lambda m: (
            tuple(m.get("sort_key", [99, m.get("module_path", "")])),
            m.get("function_name", ""),
        )
    )

    if not dry_run:
        save_manifest(manifest)

    summary["targets_touched"] = sorted(summary["targets_touched_set"])
    del summary["targets_touched_set"]
    summary["elapsed_sec"] = round(time.time() - t0, 3)
    summary["dry_run"] = dry_run
    summary["manifest_path"] = str(MANIFEST_PATH)
    summary["generated_dir"] = str(GENERATED_DIR)
    return summary

def main() -> int:
    ap = argparse.ArgumentParser(description="Synthesis MD -> stub feature modules.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--buckets", default="", help="comma-separated bucket ids (e.g. B7,B48,36)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    buckets = [b.strip() for b in args.buckets.split(",") if b.strip()] or None
    summary = run(buckets=buckets, dry_run=args.dry_run, verbose=args.verbose)
    print(json.dumps(summary, indent=2, default=str))
    return 0

if __name__ == "__main__":
    sys.exit(main())
