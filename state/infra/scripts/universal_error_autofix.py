#!/usr/bin/env python3
"""universal_error_autofix — read error_pile, classify, dispatch §8 triplet.

Reads:
  ~/.zg/state/error_pile/<utc-date>.jsonl  (canonical, local SSD)
  AI-Tools/state/error_pile/<utc-date>.jsonl (mirror)

For each unresolved error:
  1. classify pattern → known fix or unknown
  2. mark as "pending_triplet" in fixes/<hash>.json
  3. emit a brief at logs/auto_solve_engine/<hash>_<UTC>.md telling the
     orchestrator to spawn 3 §8 helpers (INTERNET + GITHUB + REPO-LOCAL)
  4. when smoke test passes (caller writes fixes/<hash>.resolved), mark done

This script is a one-shot: invoke from a cron-style loop, hook, or daemon.
Designed to be idempotent — multiple invocations on the same hash are safe.

Usage:
  python3 universal_error_autofix.py
    [--date YYYY-MM-DD]    (default: today UTC)
    [--max N]              (default: 50; cap per invocation)
    [--dry-run]            (don't emit briefs, just classify)
    [--smoke-test HASH]    (run smoke test for one resolved hash, mark resolved)

Created 2026-05-20 per universal_error_safety_spec_2026-05-20.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

HOME = Path("/Users/orginal")
DRIVE = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/"
    "My Drive/AI-Tools"
)
LOCAL_PILE = HOME / ".zg" / "state" / "error_pile"
DRIVE_PILE = DRIVE / "state" / "error_pile"
FIXES_DIR = HOME / ".zg" / "state" / "error_pile" / "fixes"
DRIVE_FIXES_DIR = DRIVE / "state" / "error_pile" / "fixes"
BRIEFS_DIR = DRIVE / "logs" / "auto_solve_engine"

# Classification: pattern → (label, suggested fix sketch)
KNOWN_PATTERNS = [
    (
        re.compile(r"ModuleNotFoundError.*'([^']+)'"),
        "python_missing_module",
        "pip install <module> or add to requirements.txt; verify venv active",
    ),
    (
        re.compile(r"ImportError: cannot import name '([^']+)'"),
        "python_import_symbol",
        "check version of source module; symbol may have been renamed/removed",
    ),
    (
        re.compile(r"Permission denied"),
        "perm_denied",
        "chmod +x or sudo if intended; or check macOS TCC grant",
    ),
    (
        re.compile(r"HTTP\s*4\d\d|401 Unauthorized|403 Forbidden|404 Not Found"),
        "http_4xx",
        "verify endpoint, auth token, params; check rate-limit headers",
    ),
    (
        re.compile(r"HTTP\s*5\d\d|500 Internal|502 Bad Gateway|503 Service"),
        "http_5xx",
        "transient — retry with backoff; check service status",
    ),
    (
        re.compile(r"RateLimit"),
        "rate_limit",
        "sleep + exponential backoff; reduce request rate; rotate keys",
    ),
    (
        re.compile(r"ConnectionError|connection refused", re.IGNORECASE),
        "connection",
        "check daemon is running (launchctl list); verify host/port; check VPN/firewall",
    ),
    (
        re.compile(r"TimeoutError|operation timed out", re.IGNORECASE),
        "timeout",
        "increase timeout; check network; verify endpoint responsive",
    ),
    (
        re.compile(r"hook error|PreToolUse[^\n]*error", re.IGNORECASE),
        "hook_error",
        "check hook script syntax; chmod +x; verify path in settings.json",
    ),
    (
        re.compile(r"daemon_exit_nonzero"),
        "daemon_exit_nonzero",
        "check StandardErrorPath log; verify plist EnvironmentVariables + ProgramArguments paths",
    ),
]


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utcdate() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def classify(body: str) -> tuple[str, str]:
    for rx, label, fix in KNOWN_PATTERNS:
        if rx.search(body):
            return label, fix
    return "unknown", "spawn §8 triplet to research"


def read_pile(date: str) -> list[dict]:
    """Read pile entries for a given UTC date (local + drive merged, deduped by hash)."""
    entries: dict[str, dict] = {}
    for piledir in (LOCAL_PILE, DRIVE_PILE):
        fp = piledir / f"{date}.jsonl"
        if not fp.exists():
            continue
        try:
            with open(fp, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    h = d.get("hash") or hashlib.sha256(
                        (d.get("body", "") + d.get("source", "")).encode()
                    ).hexdigest()[:16]
                    d["hash"] = h
                    entries[h] = d
        except OSError:
            continue
    return list(entries.values())


def fix_state_path(h: str) -> Path:
    return FIXES_DIR / f"{h}.json"


def already_processed(h: str) -> bool:
    fp = fix_state_path(h)
    if not fp.exists():
        return False
    try:
        d = json.loads(fp.read_text())
        return d.get("status") in ("pending_triplet", "resolved", "escalated")
    except Exception:
        return False


def mark_pending(h: str, entry: dict, label: str, suggested_fix: str) -> Path:
    state = {
        "hash": h,
        "first_seen": entry.get("ts"),
        "classified_at": utcnow_iso(),
        "label": label,
        "suggested_fix": suggested_fix,
        "status": "pending_triplet",
        "source": entry.get("source"),
        "kind": entry.get("kind"),
        "layer": entry.get("layer"),
        "body_preview": (entry.get("body") or "")[:500],
    }
    body = json.dumps(state, indent=2)
    for d in (FIXES_DIR, DRIVE_FIXES_DIR):
        try:
            d.mkdir(parents=True, exist_ok=True)
            tmp = d / f".{h}.json.tmp.{uuid.uuid4().hex[:8]}"
            tmp.write_text(body)
            os.replace(tmp, d / f"{h}.json")
        except Exception:
            pass
    return FIXES_DIR / f"{h}.json"


def mark_resolved(h: str, smoke_proof: str) -> None:
    for d in (FIXES_DIR, DRIVE_FIXES_DIR):
        fp = d / f"{h}.json"
        if not fp.exists():
            continue
        try:
            state = json.loads(fp.read_text())
            state["resolved_at"] = utcnow_iso()
            state["status"] = "resolved"
            state["smoke_proof"] = smoke_proof
            tmp = d / f".{h}.json.tmp.{uuid.uuid4().hex[:8]}"
            tmp.write_text(json.dumps(state, indent=2))
            os.replace(tmp, fp)
        except Exception:
            pass


def emit_brief(h: str, entry: dict, label: str, suggested_fix: str) -> Path:
    BRIEFS_DIR.mkdir(parents=True, exist_ok=True)
    fp = BRIEFS_DIR / f"{h}_{utcnow_iso().replace(':', '-')}.md"
    body = f"""# Auto-solve brief — hash={h}

# autosolve_skip: this IS the dispatch artifact, not a new error
# model_reason: routing-only (no spawn from this file)

## Classification
- label: `{label}`
- suggested fix: {suggested_fix}
- layer: {entry.get('layer')}
- source: {entry.get('source')}
- kind: {entry.get('kind')}
- severity: {entry.get('severity')}

## Body preview
```
{(entry.get('body') or '')[:1000]}
```

## §8 mandate triplet (HUMAN/ORCHESTRATOR ACTION)

Per `~/.zg/mandates.md §3`, spawn 3 parallel solvers in ONE message:

1. **INTERNET** — `WebSearch` + `WebFetch` for exact error text + fix
2. **GITHUB** — `gh search code/issues/prs` for matching fix pattern
3. **REPO-LOCAL** — `grep -r` `AI-Tools/registry/` + `repos-claude-clones/` for prior pattern

Aggregate, apply lowest-risk fix silently, then write smoke proof to:
  `~/.zg/state/error_pile/fixes/{h}.resolved` (any content; existence = resolved marker)

Re-run `universal_error_autofix.py --smoke-test {h}` to flip status → resolved.

## Auto-skip conditions (don't dispatch triplet if any true)
- error is from `auto_solve_engine/` itself (would loop)
- already resolved in last 24h (same hash)
- requires money / messages / credentials → escalate to user
"""
    fp.write_text(body)
    return fp


def run_smoke_test(h: str) -> str | None:
    """Check for resolved marker; return marker content if present."""
    marker = FIXES_DIR / f"{h}.resolved"
    if marker.exists():
        try:
            return marker.read_text()[:1000]
        except Exception:
            return "marker exists (unreadable)"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=utcdate())
    ap.add_argument("--max", type=int, default=50)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke-test", default="")
    args = ap.parse_args()

    if args.smoke_test:
        proof = run_smoke_test(args.smoke_test)
        if proof is None:
            print(f"no resolved marker for {args.smoke_test}")
            return 1
        mark_resolved(args.smoke_test, proof)
        print(f"resolved {args.smoke_test}")
        return 0

    FIXES_DIR.mkdir(parents=True, exist_ok=True)
    BRIEFS_DIR.mkdir(parents=True, exist_ok=True)

    entries = read_pile(args.date)
    print(f"loaded {len(entries)} pile entries for {args.date}")

    processed = 0
    for entry in entries:
        if processed >= args.max:
            break
        h = entry.get("hash")
        if not h:
            continue
        if already_processed(h):
            continue
        label, fix = classify(entry.get("body", ""))
        if args.dry_run:
            print(f"DRY: {h} {label}: {(entry.get('body') or '')[:80]}")
            processed += 1
            continue
        mark_pending(h, entry, label, fix)
        brief = emit_brief(h, entry, label, fix)
        print(f"queued {h} ({label}) -> {brief.name}")
        processed += 1

    # Also smoke-test all pending: if marker file exists, flip to resolved
    for fp in FIXES_DIR.glob("*.json"):
        try:
            d = json.loads(fp.read_text())
        except Exception:
            continue
        if d.get("status") != "pending_triplet":
            continue
        h = d.get("hash")
        proof = run_smoke_test(h)
        if proof:
            mark_resolved(h, proof)
            print(f"auto-resolved {h} via marker")

    print(f"done. processed={processed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
