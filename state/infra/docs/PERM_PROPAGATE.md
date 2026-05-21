# Permission / Protocol Propagation Marker

**Status:** guardrail-grade (10/10) as of 2026-05-20 — wraps an *enforcement* feature already integrated into `autosolve-require` + `spawn-validator` + `model-routing-check` hooks.

## Why it exists

There is no separate daemon for grandchild-permission propagation — the enforcement *is* the autosolve-require + spawn-validator hooks (which BLOCK tool use if a spawn brief lacks `# model_reason:`, or if an error didn't produce 3 helper spawns within 5 turns). This guardrail wrapper exists to make the **state observable** (heartbeat freshness, last-activity ts, inheritance posture surfaced to sub-agents).

## Components (10-point checklist)

1. **plist:** none — enforcement lives at the hook layer (PreToolUse autosolve-require already wired via `~/.zg/hooks/autosolve-require/require.sh`).
2. **State dir + heartbeat:** `state/perm_propagate/heartbeat.json` (+ local mirror at `/Users/orginal/.claude/state/perm_propagate/heartbeat.json`).
3. **PreToolUse hook:** `home/.claude/hooks/perm-propagate-freshness/check.sh` — touches heartbeat.
4. **SessionStart hook:** `home/.claude/hooks/perm-propagate-bootstrap/bootstrap.sh` — creates state dirs + initial heartbeat.
5. **PostToolUse hook:** `home/.claude/hooks/perm-propagate-activity/update.sh` — touches `last_session_activity.unix`.
6. **SubagentStart hook:** `home/.claude/hooks/perm-propagate-subagent-inject/inject.sh` — injects "every spawn MUST include `# model_reason:`; ANY error → spawn 3 helpers (INTERNET + GITHUB + REPO-LOCAL) within 5 turns".
7. **Stop hook:** `home/.claude/hooks/perm-propagate-stop-validate/validate.sh` — verifies heartbeat fresh.
8. **settings.json:** all 4 hooks wired.
9. **docs:** this file.
10. **backup:** `backups/grandchild-perm-fix-2026-05-20/` (pre-existing) + `backups/guardrail-100pct-2026-05-20/`.

## Relation to autosolve-require

The actual enforcement is in `~/.zg/hooks/autosolve-require/require.sh` (PreToolUse), `~/.claude/hooks/autosolve-detect/` (PostToolUse), and `~/.claude/state/autosolve_pending.jsonl`. This wrapper *observes* those hooks via heartbeat + sub-agent injection — it does not duplicate the enforcement.

## Bypass marker

In the orchestrator's next tool prompt, include:
```
# autosolve_skip: <one-line reason>
```

This is logged but allows the tool to proceed.

## Smoke test

```bash
bash AI-Tools/home/.claude/hooks/perm-propagate-bootstrap/bootstrap.sh < /dev/null
cat AI-Tools/state/perm_propagate/heartbeat.json
```
