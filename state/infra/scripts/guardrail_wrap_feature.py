#!/usr/bin/env python3
"""Generic retroactive wrapper: applies the 10-point guardrail-grade checklist
to an existing feature by name.

Usage:
    python guardrail_wrap_feature.py <feature_name> [--plist-label LABEL]

Generates: state heartbeat + 5 hook scripts + doc stub + backup + idempotent
settings.json registration.

For features whose plist already exists this preserves it (Layer 1).
For features needing a brand-new plist, generate it manually via launchd
template (this script doesn't auto-generate plists).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import time
from datetime import datetime
from pathlib import Path

ROOT = Path("/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools")
SETTINGS = ROOT / "ClaudeCode/config/settings.json"


def write_executable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def make_hooks(feature: str, plist_label: str) -> dict[str, str]:
    fdash = feature.replace("_", "-")
    state_dir = f"$ROOT/state/{feature}"
    return {
        f"{fdash}-freshness/check.sh": f"""#!/usr/bin/env bash
# {fdash}-freshness — PreToolUse (Layer 3 of 10)
set -uo pipefail
ROOT="{ROOT}"
HB="$ROOT/state/{feature}/heartbeat.json"
PLIST="$ROOT/home/Library/LaunchAgents/{plist_label}.plist"
STALE_SEC=600
LOG_DIR="$ROOT/logs/auto_solve"
mkdir -p "$LOG_DIR"
[ -f "$HB" ] || {{ launchctl bootstrap "gui/$(id -u)" "$PLIST" 2>/dev/null || true; exit 0; }}
NOW=$(date +%s)
HB_TS=$(python3 -c "import json; print(json.load(open('$HB')).get('ts',0))" 2>/dev/null || echo 0)
AGE=$((NOW - HB_TS))
if [ "$AGE" -gt "$STALE_SEC" ]; then
    echo "[{fdash}-freshness] stale (age=${{AGE}}s) — respawning" >&2
    launchctl bootstrap "gui/$(id -u)" "$PLIST" 2>/dev/null || \
        launchctl kickstart -k "gui/$(id -u)/{plist_label}" 2>/dev/null || true
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) {feature} respawn (age=${{AGE}}s)" \
        >> "$LOG_DIR/{feature}_stale_$(date -u +%Y%m%d).md"
fi
exit 0
""",
        f"{fdash}-bootstrap/bootstrap.sh": f"""#!/usr/bin/env bash
# {fdash}-bootstrap — SessionStart (Layer 4 of 10)
set -uo pipefail
ROOT="{ROOT}"
PLIST="$ROOT/home/Library/LaunchAgents/{plist_label}.plist"
LOG="$ROOT/logs/{feature}_bootstrap.log"
LABEL="{plist_label}"
launchctl list 2>/dev/null | grep -q "$LABEL" && exit 0
[ -f "$PLIST" ] || {{ echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) ERROR: plist missing" >> "$LOG"; exit 0; }}
launchctl bootstrap "gui/$(id -u)" "$PLIST" 2>>"$LOG" || \
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) bootstrap rc=$? (likely already loaded)" >> "$LOG"
exit 0
""",
        f"{fdash}-activity/update.sh": f"""#!/usr/bin/env bash
# {fdash}-activity — PostToolUse (Layer 5 of 10)
set -uo pipefail
ROOT="{ROOT}"
A="$ROOT/state/{feature}/last_session_activity.unix"
mkdir -p "$(dirname "$A")"
date +%s > "$A" 2>/dev/null || true
exit 0
""",
        f"{fdash}-subagent-inject/inject.sh": f"""#!/usr/bin/env bash
# {fdash}-subagent-inject — SubagentStart (Layer 6 of 10)
set -uo pipefail
ROOT="{ROOT}"
HB="$ROOT/state/{feature}/heartbeat.json"
cat <<EOF_BANNER
=== {feature.upper()} INHERITANCE ===
EOF_BANNER
if [ -f "$HB" ]; then
    HB_TS=$(python3 -c "import json; print(json.load(open('$HB')).get('ts',0))" 2>/dev/null || echo 0)
    HB_STATUS=$(python3 -c "import json; print(json.load(open('$HB')).get('status','unknown'))" 2>/dev/null || echo unknown)
    NOW=$(date +%s)
    AGE=$((NOW - HB_TS))
    echo "heartbeat_age_sec=$AGE  status=$HB_STATUS"
else
    echo "heartbeat=missing"
fi
echo "Rules: feature is guardrail-grade; do not bypass its state files."
echo "=== END {feature.upper()} INHERITANCE ==="
exit 0
""",
        f"{fdash}-stop-validate/validate.sh": f"""#!/usr/bin/env bash
# {fdash}-stop-validate — Stop (Layer 7 of 10)
set -uo pipefail
ROOT="{ROOT}"
HB="$ROOT/state/{feature}/heartbeat.json"
V="$ROOT/state/{feature}/violations.jsonl"
STALE_SEC=600
NOW=$(date +%s)
[ -f "$HB" ] || exit 0
HB_TS=$(python3 -c "import json; print(json.load(open('$HB')).get('ts',0))" 2>/dev/null || echo 0)
AGE=$((NOW - HB_TS))
if [ "$AGE" -gt "$STALE_SEC" ]; then
    echo "{{\\"ts\\":$NOW,\\"violation\\":\\"stale_heartbeat\\",\\"age\\":$AGE}}" >> "$V"
    echo "[{fdash}-stop-validate] WARN: heartbeat stale (${{AGE}}s)" >&2
fi
exit 0
""",
    }


def make_doc(feature: str, plist_label: str) -> str:
    upper = feature.upper()
    return f"""# {upper}

Retroactively wrapped to guardrail-grade 2026-05-20.

## Architecture (10-point wrap)

- Layer 1: launchd plist `{plist_label}` (existing)
- Layer 2: heartbeat at `state/{feature}/heartbeat.json`
- Layer 3-7: hooks at `home/.claude/hooks/{feature.replace('_','-')}-{{freshness,bootstrap,activity,subagent-inject,stop-validate}}/`
- Layer 8: settings.json registered idempotently
- Layer 9: this doc
- Layer 10: backup at `backups/{feature}-pre-install-2026-05-20/`

## Smoke test

```bash
launchctl list | grep {plist_label}
python3 -c "import json,time; d=json.load(open('{ROOT}/state/{feature}/heartbeat.json')); print('age_sec=', int(time.time()) - d.get('ts',0))"
ls -la {ROOT}/home/.claude/hooks/{feature.replace('_','-')}-*/*.sh
```

## Revert

```bash
launchctl bootout "gui/$(id -u)/{plist_label}" 2>/dev/null
cp backups/{feature}-pre-install-2026-05-20/{plist_label}.plist home/Library/LaunchAgents/
rm -rf home/.claude/hooks/{feature.replace('_','-')}-{{freshness,bootstrap,activity,subagent-inject,stop-validate}}
# revert settings.json from backups/settings-pre-{feature}-2026-05-20/
```

## Failure modes

| Symptom | Cause | Mitigation |
|---|---|---|
| Heartbeat stale | Daemon crashed/hung | freshness hook auto-respawns |
| Hook not firing | settings.json missing entry | re-run scripts/register_{feature}_hooks.py |
| violations.jsonl growing | Persistent failure | manual investigation |

## Escape hatch

Comment out the freshness hook entry in `ClaudeCode/config/settings.json` if it causes respawn loops.

## Audit history

- 2026-05-20 — Retroactively wrapped per "Guardrail-grade default" mandate.
"""


def register_hooks(feature: str) -> tuple[int, int]:
    fdash = feature.replace("_", "-")
    backup_dir = ROOT / f"backups/settings-pre-{feature}-2026-05-20"
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    shutil.copy2(SETTINGS, backup_dir / f"settings-{ts}.json")

    data = json.loads(SETTINGS.read_text())
    data.setdefault("hooks", {})

    hook_specs = {
        "PreToolUse": (f"{fdash}-freshness/check.sh", {"matcher": ".*"}, 30),
        "SessionStart": (f"{fdash}-bootstrap/bootstrap.sh", {}, 30),
        "PostToolUse": (f"{fdash}-activity/update.sh", {"matcher": ".*"}, 5),
        "SubagentStart": (f"{fdash}-subagent-inject/inject.sh", {}, 10),
        "Stop": (f"{fdash}-stop-validate/validate.sh", {}, 10),
    }

    added = 0
    skipped = 0
    for event, (script_rel, matcher_kv, timeout) in hook_specs.items():
        cmd = str(ROOT / f"home/.claude/hooks/{script_rel}")
        data["hooks"].setdefault(event, [])
        already = False
        for existing in data["hooks"][event]:
            for h in existing.get("hooks", []):
                if h.get("command") == cmd:
                    already = True
                    break
            if already:
                break
        if already:
            skipped += 1
            continue
        entry = {**matcher_kv, "hooks": [{"type": "command", "command": cmd, "timeout": timeout}]}
        data["hooks"][event].append(entry)
        added += 1

    tmp = SETTINGS.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    json.loads(tmp.read_text())
    tmp.replace(SETTINGS)
    return added, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("feature_name")
    ap.add_argument("--plist-label", default=None)
    args = ap.parse_args()

    feature = args.feature_name
    plist_label = args.plist_label or f"com.zg.{feature}"

    # 1. Backup
    backup_dir = ROOT / f"backups/{feature}-pre-install-2026-05-20"
    backup_dir.mkdir(parents=True, exist_ok=True)
    plist_path = ROOT / f"home/Library/LaunchAgents/{plist_label}.plist"
    if plist_path.exists():
        shutil.copy2(plist_path, backup_dir / plist_path.name)
    state_dir = ROOT / "state" / feature
    if state_dir.exists():
        shutil.copytree(state_dir, backup_dir / "state", dirs_exist_ok=True)
    print(f"[1] Backup: {backup_dir}")

    # 2. Heartbeat
    state_dir.mkdir(parents=True, exist_ok=True)
    hb = state_dir / "heartbeat.json"
    if not hb.exists():
        hb.write_text(json.dumps({"ts": int(time.time()), "pid": None, "cycle_id": None, "status": "uninitialized"}))
    activity = state_dir / "last_session_activity.unix"
    if not activity.exists():
        activity.write_text(str(int(time.time())))
    print(f"[2] State: {state_dir}")

    # 3-7. Hooks
    for rel_path, content in make_hooks(feature, plist_label).items():
        full = ROOT / "home/.claude/hooks" / rel_path
        write_executable(full, content)
    print(f"[3-7] 5 hooks created under home/.claude/hooks/{feature.replace('_','-')}-*/")

    # 8. settings.json
    added, skipped = register_hooks(feature)
    print(f"[8] settings.json: added={added} skipped={skipped}")

    # 9. Doc
    doc = ROOT / f"docs/{feature.upper()}.md"
    if not doc.exists():
        doc.write_text(make_doc(feature, plist_label))
        print(f"[9] Doc: {doc}")
    else:
        print(f"[9] Doc exists, skipping: {doc}")

    print(f"\nDONE — feature={feature} plist={plist_label}")
    print(f"Re-run scripts/guardrail_audit_scorer.py to confirm new score.")


if __name__ == "__main__":
    main()
