# MEMORY_AUTO_SAVE

Retroactively wrapped to guardrail-grade 2026-05-20.

## Architecture (10-point wrap)

- Layer 1: launchd plist `com.zg.memory_auto_save` (existing)
- Layer 2: heartbeat at `state/memory_auto_save/heartbeat.json`
- Layer 3-7: hooks at `home/.claude/hooks/memory-auto-save-{freshness,bootstrap,activity,subagent-inject,stop-validate}/`
- Layer 8: settings.json registered idempotently
- Layer 9: this doc
- Layer 10: backup at `backups/memory_auto_save-pre-install-2026-05-20/`

## Smoke test

```bash
launchctl list | grep com.zg.memory_auto_save
python3 -c "import json,time; d=json.load(open('/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/state/memory_auto_save/heartbeat.json')); print('age_sec=', int(time.time()) - d.get('ts',0))"
ls -la /Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/home/.claude/hooks/memory-auto-save-*/*.sh
```

## Revert

```bash
launchctl bootout "gui/$(id -u)/com.zg.memory_auto_save" 2>/dev/null
cp backups/memory_auto_save-pre-install-2026-05-20/com.zg.memory_auto_save.plist home/Library/LaunchAgents/
rm -rf home/.claude/hooks/memory-auto-save-{freshness,bootstrap,activity,subagent-inject,stop-validate}
# revert settings.json from backups/settings-pre-memory_auto_save-2026-05-20/
```

## Failure modes

| Symptom | Cause | Mitigation |
|---|---|---|
| Heartbeat stale | Daemon crashed/hung | freshness hook auto-respawns |
| Hook not firing | settings.json missing entry | re-run scripts/register_memory_auto_save_hooks.py |
| violations.jsonl growing | Persistent failure | manual investigation |

## Escape hatch

Comment out the freshness hook entry in `ClaudeCode/config/settings.json` if it causes respawn loops.

## Audit history

- 2026-05-20 — Retroactively wrapped per "Guardrail-grade default" mandate.
