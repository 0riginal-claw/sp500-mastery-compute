"""siv_verifier.py — Statistical Inference Validation (SIV) pre-flight gate.

Builds a hash-verified pre-flight gate file at:
  AI-Tools/state/lab_siv/SIV_2026-05-29.signed.json

Council Outsider R3+R4 demand: no SAP promotion to paper trading without the
full SIV checklist physically present + hash-verified.

The 6 R2 artifacts:
  1. SAP factory       — lab/championship_search.py + lab/example_hypotheses.py
  2. Kill-criterion    — reports/champion_kill_criterion_2026-05-29.md
  3. DSMB charter      — independent adjudicator (NOT BUILT)
  4. Blinding          — held-out evaluator (NOT BUILT)
  5. Evidence-base     — edgar.db, govtrades.db, alpaca_news.db, data_inventory/
  6. Holdout separation — physical parquet separation (code-side split only)

Usage:
  python siv_verifier.py            # build + write SIV file
  python siv_verifier.py --check    # build, print, but don't write
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
DRIVE_BASE = Path(
    os.environ.get(
        "DRIVE_BASE",
        "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive",
    )
)
PROJECT = DRIVE_BASE / "AI-Tools" / "s&p500-ticker-mastery"
LAB = PROJECT / "scripts" / "lab"
REPORTS = DRIVE_BASE / "AI-Tools" / "reports"
DATA_INVENTORY = DRIVE_BASE / "AI-Tools" / "research-lab" / "data_inventory"
ZG_LOCAL = Path("/Volumes/ZG-2TB/zg")

SIV_DIR = DRIVE_BASE / "AI-Tools" / "state" / "lab_siv"
SIV_PATH = SIV_DIR / "SIV_2026-05-29.signed.json"
SIV_FALLBACK = ZG_LOCAL / "state" / "lab_siv" / "SIV_2026-05-29.signed.json"


# ─────────────────────────────────────────────────────────────────────────────
# Hashing
# ─────────────────────────────────────────────────────────────────────────────
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    try:
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError as e:
        return f"ERROR:{type(e).__name__}:{e}"


def sha256_dir_manifest(p: Path, max_entries: int = 5000) -> str:
    """Hash a directory by manifesting (sorted relative path, size) pairs.

    Stable across runs as long as file set + sizes don't change. Cheap (no
    file-body reads).
    """
    if not p.exists():
        return "ERROR:NotFound"
    items: list[tuple[str, int]] = []
    try:
        for root, _dirs, files in os.walk(p):
            for fname in files:
                fp = Path(root) / fname
                try:
                    items.append((str(fp.relative_to(p)), fp.stat().st_size))
                except OSError:
                    continue
                if len(items) >= max_entries:
                    break
            if len(items) >= max_entries:
                break
    except OSError as e:
        return f"ERROR:{type(e).__name__}:{e}"
    items.sort()
    h = hashlib.sha256()
    for rel, size in items:
        h.update(f"{rel}\t{size}\n".encode("utf-8"))
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Verifiers — return a uniform record per artifact
# ─────────────────────────────────────────────────────────────────────────────
def _record(status: str, **kwargs) -> dict[str, Any]:
    rec = {"status": status}
    rec.update(kwargs)
    return rec


def verify_sap_factory() -> dict[str, Any]:
    files = [LAB / "championship_search.py", LAB / "example_hypotheses.py"]
    present = [f for f in files if f.exists()]
    missing = [str(f) for f in files if not f.exists()]
    if not missing:
        return _record(
            "PRESENT",
            files=[str(f) for f in present],
            sha256={str(f.name): sha256_file(f) for f in present},
            note="SAP factory (championship search + example hypotheses) physically present.",
        )
    return _record(
        "MISSING",
        files=[str(f) for f in present],
        missing=missing,
        note="SAP factory incomplete.",
    )


def verify_kill_criterion() -> dict[str, Any]:
    candidates = [
        REPORTS / "champion_kill_criterion_2026-05-29.md",
        REPORTS / "kill_criterion_2026-05-29.md",
        DRIVE_BASE / "AI-Tools" / "research-lab" / "champion_kill_criterion_2026-05-29.md",
    ]
    found = [c for c in candidates if c.exists()]
    if found:
        return _record(
            "PRESENT",
            files=[str(f) for f in found],
            sha256={str(f.name): sha256_file(f) for f in found},
            note="Kill-criterion document found.",
        )
    return _record(
        "PENDING",
        files=[],
        searched=[str(c) for c in candidates],
        note=(
            "Kill-criterion document not yet written (task #55 may still be in flight). "
            "SAP promotion gate must remain CLOSED until this exists."
        ),
    )


def verify_dsmb_charter() -> dict[str, Any]:
    return _record(
        "MISSING",
        files=[],
        note=(
            "DSMB charter / independent adjudicator process NOT BUILT. R4 chairman "
            "flagged: not earned the right to skip. Required before any SAP promotion "
            "to paper trading. Next action: draft charter (who adjudicates, when, "
            "what authority to halt a champion)."
        ),
    )


def verify_blinding() -> dict[str, Any]:
    return _record(
        "MISSING",
        files=[],
        note=(
            "Blinding mechanism NOT BUILT. Researcher currently sees all results "
            "before next pre-registration — unblinded. Held-out evaluator process "
            "not implemented. Required: SAPs frozen + handed to a separate evaluator "
            "who runs the holdout and reports a sealed result."
        ),
    )


def verify_evidence_base() -> dict[str, Any]:
    candidates: list[Path] = [
        ZG_LOCAL / "edgar_state" / "index" / "edgar.db",
        DATA_INVENTORY,
        ZG_LOCAL / "govtrades" / "data" / "govtrades.db",
        ZG_LOCAL / "news_cache" / "alpaca_news.db",
    ]
    present: list[Path] = []
    hashes: dict[str, str] = {}
    missing: list[str] = []
    for c in candidates:
        if not c.exists():
            missing.append(str(c))
            continue
        present.append(c)
        if c.is_dir():
            hashes[str(c)] = sha256_dir_manifest(c)
        else:
            hashes[str(c)] = sha256_file(c)
    # Healthy if at least 2 of the 4 are present
    healthy = len(present) >= 2
    status = "PRESENT" if healthy else "MISSING"
    return _record(
        status,
        files=[str(f) for f in present],
        sha256_per_file=hashes,
        missing=missing,
        note=(
            f"{len(present)}/{len(candidates)} mirrors present "
            f"(healthy ≥2). Mirrors: EDGAR/data_inventory/govtrades/alpaca_news."
        ),
    )


def verify_holdout_separation() -> dict[str, Any]:
    cache_dir = PROJECT / "cache" / "yfinance_5yr"
    manifest = cache_dir / "_manifest.json"
    present_train = cache_dir.exists()
    return _record(
        "PARTIAL",
        files=([str(cache_dir)] if present_train else []),
        sha256={"_manifest.json": sha256_file(manifest)} if manifest.exists() else {},
        note=(
            "Code-side date split only — training window (2020-2024) and holdout "
            "window (2025-2026) share the same parquet file per ticker under "
            "cache/yfinance_5yr/<T>.parquet. R2 chairman recommendation: physically "
            "separate the holdout parquets into a different volume the researcher "
            "cannot see during SAP design. Currently NOT separated."
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Builder
# ─────────────────────────────────────────────────────────────────────────────
def build_siv_record(locked_utc: str | None = None,
                     locked_by: str = "user (via /sap/promotion-gate)") -> dict[str, Any]:
    if locked_utc is None:
        locked_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    artifacts = {
        "sap_factory":          verify_sap_factory(),
        "kill_criterion":       verify_kill_criterion(),
        "dsmb_charter":         verify_dsmb_charter(),
        "blinding_mechanism":   verify_blinding(),
        "evidence_base_mirror": verify_evidence_base(),
        "holdout_separation":   verify_holdout_separation(),
    }

    all_present = all(a["status"] == "PRESENT" for a in artifacts.values())
    overall = "PRESENT" if all_present else "INCOMPLETE"

    next_action: list[str] = []
    if artifacts["kill_criterion"]["status"] != "PRESENT":
        next_action.append("Complete kill-criterion document (task #55).")
    if artifacts["dsmb_charter"]["status"] != "PRESENT":
        next_action.append("Build DSMB charter / independent adjudicator process.")
    if artifacts["blinding_mechanism"]["status"] != "PRESENT":
        next_action.append("Implement held-out evaluator (blinding) process.")
    if artifacts["holdout_separation"]["status"] != "PRESENT":
        next_action.append("Physical holdout volume separation (per R2 recommendation).")
    if artifacts["sap_factory"]["status"] != "PRESENT":
        next_action.append("Restore SAP factory files in lab/.")
    if artifacts["evidence_base_mirror"]["status"] != "PRESENT":
        next_action.append("Restore evidence-base mirrors (EDGAR / govtrades / alpaca_news / data_inventory).")

    return {
        "siv_version": "1.0",
        "locked_utc": locked_utc,
        "locked_by": locked_by,
        "council_rounds_addressed": ["R2 chairman", "R3 chairman", "R4 chairman"],
        "artifacts": artifacts,
        "overall_status": overall,
        "gates_enforced": {
            "no_sap_promotion_to_paper_without_full_siv": True,
            "permutation_test_results_required": True,
        },
        "next_action_to_complete": next_action,
    }


# ─────────────────────────────────────────────────────────────────────────────
# IO
# ─────────────────────────────────────────────────────────────────────────────
def write_siv(record: dict[str, Any], path: Path = SIV_PATH,
              fallback: Path = SIV_FALLBACK) -> Path:
    text = json.dumps(record, indent=2, default=str)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(text)
        tmp.replace(path)
        return path
    except OSError as e:
        print(f"[siv] Drive write failed ({e}); falling back to {fallback}", file=sys.stderr)
        fallback.parent.mkdir(parents=True, exist_ok=True)
        tmpf = fallback.with_suffix(".json.tmp")
        tmpf.write_text(text)
        tmpf.replace(fallback)
        return fallback


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="Build + print, do NOT write file.")
    ap.add_argument("--locked-utc", default=None,
                    help="Override lock timestamp (ISO-8601).")
    ap.add_argument("--locked-by", default="user (via /sap/promotion-gate)",
                    help="Who locked the SIV.")
    args = ap.parse_args()

    rec = build_siv_record(locked_utc=args.locked_utc, locked_by=args.locked_by)
    print(json.dumps(rec, indent=2, default=str))

    if not args.check:
        path = write_siv(rec)
        print(f"\n[siv] wrote {path}", file=sys.stderr)
        print(f"[siv] overall_status: {rec['overall_status']}", file=sys.stderr)
    return 0 if rec["overall_status"] == "PRESENT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
