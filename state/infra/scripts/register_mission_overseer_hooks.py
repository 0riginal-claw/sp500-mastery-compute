#!/usr/bin/env python3
"""Idempotent registration of mission_overseer hooks into settings.json.

- Backs up settings.json to backups/settings-pre-mission_overseer-2026-05-20/
- For each event/hook pair, checks if the command string already exists; appends only if missing.
- Validates JSON round-trip after edit.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path("/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools")
SETTINGS = ROOT / "ClaudeCode/config/settings.json"
BACKUP_DIR = ROOT / "backups/settings-pre-mission_overseer-2026-05-20"

HOOKS_TO_ADD = {
    "PreToolUse": [
        {
            "matcher": ".*",
            "hooks": [
                {
                    "type": "command",
                    "command": str(ROOT / "home/.claude/hooks/mission-overseer-freshness/check.sh"),
                    "timeout": 30,
                }
            ],
        }
    ],
    "SessionStart": [
        {
            "hooks": [
                {
                    "type": "command",
                    "command": str(ROOT / "home/.claude/hooks/mission-overseer-bootstrap/bootstrap.sh"),
                    "timeout": 30,
                }
            ]
        }
    ],
    "PostToolUse": [
        {
            "matcher": ".*",
            "hooks": [
                {
                    "type": "command",
                    "command": str(ROOT / "home/.claude/hooks/mission-overseer-activity/update.sh"),
                    "timeout": 5,
                }
            ],
        }
    ],
    "SubagentStart": [
        {
            "hooks": [
                {
                    "type": "command",
                    "command": str(ROOT / "home/.claude/hooks/mission-overseer-subagent-inject/inject.sh"),
                    "timeout": 10,
                }
            ]
        }
    ],
    "Stop": [
        {
            "hooks": [
                {
                    "type": "command",
                    "command": str(ROOT / "home/.claude/hooks/mission-overseer-stop-validate/validate.sh"),
                    "timeout": 10,
                }
            ]
        }
    ],
}


def main() -> None:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    backup_path = BACKUP_DIR / f"settings-{timestamp}.json"
    shutil.copy2(SETTINGS, backup_path)
    print(f"Backup: {backup_path}")

    data = json.loads(SETTINGS.read_text())
    if "hooks" not in data:
        data["hooks"] = {}

    added = 0
    skipped = 0

    for event, new_entries in HOOKS_TO_ADD.items():
        if event not in data["hooks"]:
            data["hooks"][event] = []

        for new_entry in new_entries:
            new_cmds = {h["command"] for h in new_entry.get("hooks", [])}

            already_present = False
            for existing in data["hooks"][event]:
                existing_cmds = {
                    h.get("command", "") for h in existing.get("hooks", [])
                }
                if new_cmds & existing_cmds:
                    already_present = True
                    break

            if already_present:
                skipped += 1
                print(f"  [{event}] SKIP - already registered: {list(new_cmds)[0]}")
            else:
                data["hooks"][event].append(new_entry)
                added += 1
                print(f"  [{event}] ADD: {list(new_cmds)[0]}")

    tmp = SETTINGS.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    json.loads(tmp.read_text())
    tmp.replace(SETTINGS)

    print(f"\nDone. added={added}  skipped={skipped}")
    print(f"settings.json round-trip validated.")


if __name__ == "__main__":
    main()
