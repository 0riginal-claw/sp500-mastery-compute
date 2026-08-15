# MISSION_OVERSEER

The mission_overseer daemon coordinates auto-solve sub-agent solver spawns, tracks pending issues, and emits alerts when load thresholds are crossed. Promoted to guardrail-grade 2026-05-20 (was 3/10, retroactively wrapped to ~9/10).

## Architecture (10-point wrap)

```
                        +------------------------------+
                        | scripts/mission_overseer.py  |  daemon entry
                        +--------------+---------------+
                                       |
       +-------------------------------+-----------------------------------+
       |                               |                                   |
       v                               v                                   v
 launchd plist                  state/mission_overseer/             5 Claude hooks
 (Layer 1)                      heartbeat.json (Layer 2)          (Layers 3-7)
 KeepAlive=false                last_session_activity.unix       freshness/bootstrap/
 ThrottleInterval=30            pending_solvers/                  activity/sa-inject/
 com.zg.mission_overseer        alert_history.jsonl                stop-validate
                                noted_issues.json
                                load_history.json
                                violations.jsonl (on miss)
```

## Smoke test

```bash
# 1. plist loaded?
launchctl list | grep com.zg.mission_overseer

# 2. heartbeat fresh?
python3 -c "import json,time; d=json.load(open('/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/state/mission_overseer/heartbeat.json')); print('age_sec=', int(time.time()) - d.get('ts',0))"

# 3. hooks executable?
ls -la "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/home/.claude/hooks/mission-overseer-"*"/"*.sh
```

## Revert procedure

```bash
# 1. unload daemon
launchctl bootout "gui/$(id -u)/com.zg.mission_overseer" 2>/dev/null || true

# 2. restore plist from backup
cp "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/backups/mission_overseer-pre-install-2026-05-20/com.zg.mission_overseer.plist" \
   "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/home/Library/LaunchAgents/"

# 3. remove hook dirs (heartbeat + state preserved for audit)
cd "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools"
for h in freshness bootstrap activity subagent-inject stop-validate; do
    rm -rf "home/.claude/hooks/mission-overseer-$h"
done

# 4. remove hook entries from settings.json
# (manual edit OR re-run guardrail-ify in unregister mode)
```

## Failure modes

| Symptom | Likely cause | Mitigation |
|---|---|---|
| Heartbeat stale | Daemon crashed or hung | freshness hook auto-respawns next tool call |
| Hook not firing | settings.json missing registration | re-run guardrail-ify in idempotent mode |
| Pending solvers stuck | Stale entries from prior session | check `state/mission_overseer/pending_solvers/` and clear by id |
| `launchctl list` shows no entry | bootstrap hook failed | check `logs/mission_overseer_bootstrap.log` |
| violations.jsonl growing | Stop hook detecting stale heartbeat each turn | daemon is dead — manual bootstrap + investigate why KeepAlive=false did not catch |

## Escape hatch

If the freshness hook causes false-positive respawn loops, comment out its entry in `ClaudeCode/config/settings.json` (`hooks.PreToolUse.*` block matching `mission-overseer-freshness`), then restart the daemon manually. Re-enable after diagnosing.

To temporarily disable mission_overseer entirely:

```bash
launchctl bootout "gui/$(id -u)/com.zg.mission_overseer"
# AND comment out the SessionStart bootstrap hook so it doesn't reload
```

## Audit history

- 2026-05-20 — Retroactively wrapped per "Guardrail-grade default" mandate. Initial audit score 3/10 (plist + state dir + 1 backup only). Wrap added: heartbeat init, freshness/bootstrap/activity/subagent-inject/stop-validate hooks, this doc, pre-install backup. New score ~9/10 (settings.json entry pending separate idempotent merge — see Pending below).

## Pending

- [ ] settings.json idempotent merge — append 5 hook entries to PreToolUse/SessionStart/PostToolUse/SubagentStart/Stop arrays. Best done via Python json round-trip with atomic-replace + backup at `backups/settings-pre-mission_overseer-2026-05-20/`. Once merged, re-run `scripts/guardrail_audit_scorer.py` to confirm 10/10.
