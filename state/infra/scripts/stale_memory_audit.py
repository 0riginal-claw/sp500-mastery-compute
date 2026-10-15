#!/usr/bin/env python3
"""
Stale memory auditor — flags large/stale CLAUDE.md and memory/*.md files that
would benefit from /caveman-compress.

Safe by design: produces a JSON manifest + markdown report only. Does NOT
invoke Claude or modify files. Operator runs /caveman-compress <path> next
session for each flagged file (so user sees + approves spend).

Defaults: file >2KB AND age >7 days AND not already *.original.md backup.

Output:
  AI-Tools/logs/stale_memory_audit/<DATE>.json  (machine-readable)
  AI-Tools/logs/stale_memory_audit/<DATE>.md    (human-readable summary)

Designed to run via launchd weekly. Zero cost, zero side effects.
"""
from __future__ import annotations
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

AI_ROOT = Path("/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools")
SCAN_TARGETS = [
    AI_ROOT / "CLAUDE.md",
    AI_ROOT / "ClaudeCode/config/projects",  # memory dirs live under per-project paths
    AI_ROOT / "docs/AGENT_BRIEF_TEMPLATE.md",
    AI_ROOT / "docs/HOT_RELOAD_PATTERNS.md",
]
SIZE_THRESHOLD_BYTES = 2 * 1024  # 2KB
AGE_THRESHOLD_DAYS = 7
LOG_DIR = AI_ROOT / "logs/stale_memory_audit"


def iter_candidates():
    for tgt in SCAN_TARGETS:
        if not tgt.exists():
            continue
        if tgt.is_file():
            yield tgt
        else:
            for p in tgt.rglob("*.md"):
                # Skip backups + non-memory dirs
                if p.name.endswith(".original.md"):
                    continue
                if "/backups/" in str(p):
                    continue
                yield p


def main():
    now = datetime.now(timezone.utc)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    flagged = []
    skipped_recent = 0
    skipped_small = 0
    total = 0
    for p in iter_candidates():
        try:
            st = p.stat()
        except OSError:
            continue
        total += 1
        size = st.st_size
        age_days = (now.timestamp() - st.st_mtime) / 86400
        if size < SIZE_THRESHOLD_BYTES:
            skipped_small += 1
            continue
        if age_days < AGE_THRESHOLD_DAYS:
            skipped_recent += 1
            continue
        flagged.append({
            "path": str(p),
            "size_bytes": size,
            "size_kb": round(size / 1024, 1),
            "age_days": round(age_days, 1),
            "mtime_iso": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
        })

    flagged.sort(key=lambda x: x["size_bytes"], reverse=True)
    date_tag = now.strftime("%Y-%m-%d")
    json_path = LOG_DIR / f"{date_tag}.json"
    md_path = LOG_DIR / f"{date_tag}.md"

    manifest = {
        "scanned_at": now.isoformat(),
        "total_scanned": total,
        "skipped_recent": skipped_recent,
        "skipped_small": skipped_small,
        "flagged_count": len(flagged),
        "thresholds": {
            "size_bytes": SIZE_THRESHOLD_BYTES,
            "age_days": AGE_THRESHOLD_DAYS,
        },
        "flagged": flagged,
    }
    json_path.write_text(json.dumps(manifest, indent=2))

    lines = [
        f"# Stale memory audit — {date_tag}",
        "",
        f"Scanned: {total} · Flagged: {len(flagged)} · Skipped (recent): {skipped_recent} · Skipped (small): {skipped_small}",
        "",
        "## Suggested operator action",
        "Next interactive session, run `/caveman-compress <path>` for any file below.",
        "Caveman compress saves ~50-75% input tokens while preserving substance.",
        "",
        "## Flagged files (largest first)",
        "",
        "| size KB | age days | path |",
        "|--------:|---------:|------|",
    ]
    for f in flagged:
        lines.append(f"| {f['size_kb']:.1f} | {f['age_days']:.1f} | `{f['path']}` |")
    md_path.write_text("\n".join(lines))

    print(f"[stale_memory_audit] flagged={len(flagged)} report={md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
