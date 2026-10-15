# AGENT_WATCHDOG

Retroactively wrapped to guardrail-grade 2026-05-20.

## Architecture (10-point wrap)

- Layer 1: launchd plist `com.zg.agent_watchdog` (existing)
- Layer 2: heartbeat at `state/agent_watchdog/heartbeat.json`
- Layer 3-7: hooks at `home/.claude/hooks/agent-watchdog-{freshness,bootstrap,activity,subagent-inject,stop-validate}/`
- Layer 8: settings.json registered idempotently
- Layer 9: this doc
- Layer 10: backup at `backups/agent_watchdog-pre-install-2026-05-20/`

## Smoke test

```bash
launchctl list | grep com.zg.agent_watchdog
python3 -c "import json,time; d=json.load(open('/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/state/agent_watchdog/heartbeat.json')); print('age_sec=', int(time.time()) - d.get('ts',0))"
ls -la /Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/home/.claude/hooks/agent-watchdog-*/*.sh
```

## Revert

```bash
launchctl bootout "gui/$(id -u)/com.zg.agent_watchdog" 2>/dev/null
cp backups/agent_watchdog-pre-install-2026-05-20/com.zg.agent_watchdog.plist home/Library/LaunchAgents/
rm -rf home/.claude/hooks/agent-watchdog-{freshness,bootstrap,activity,subagent-inject,stop-validate}
# revert settings.json from backups/settings-pre-agent_watchdog-2026-05-20/
```

## Failure modes

| Symptom | Cause | Mitigation |
|---|---|---|
| Heartbeat stale | Daemon crashed/hung | freshness hook auto-respawns |
| Hook not firing | settings.json missing entry | re-run scripts/register_agent_watchdog_hooks.py |
| violations.jsonl growing | Persistent failure | manual investigation |

## Escape hatch

Comment out the freshness hook entry in `ClaudeCode/config/settings.json` if it causes respawn loops.

## Audit history

- 2026-05-20 — Retroactively wrapped per "Guardrail-grade default" mandate.
