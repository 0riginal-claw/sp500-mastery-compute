# Sync Hooks Local — Drive→SSD Mirror

**Status:** guardrail-grade (10/10) as of 2026-05-20.

## Purpose

The Drive canonical path `/Users/orginal/Library/CloudStorage/GoogleDrive-.../My Drive/AI-Tools/home/.claude/hooks/` is on a FUSE filesystem with cold-cache latency that can spike to 100–500 ms per hook script open. The `sync_hooks_local` launchd job mirrors that tree (and a few sibling state dirs) to `/Users/orginal/.zg/` every 5 min so freshly-spawned shells hit warm-cache SSD paths.

## Components (10-point checklist)

1. **plist:** `/Users/orginal/Library/LaunchAgents/com.zg.sync_hooks_local.plist` — `StartInterval=300`, `ThrottleInterval=30`, `KeepAlive` on crash.
2. **State dir + heartbeat:** `AI-Tools/state/sync_hooks_local/heartbeat.json` — written by freshness hook on every PreToolUse. Includes `sync_log_age_sec` (mtime of `/Users/orginal/.zg/.sync.stdout.log`).
3. **PreToolUse hook:** `home/.claude/hooks/sync-hooks-local-freshness/check.sh` — reads `~/.zg/.sync.stdout.log` mtime; if age > 600 s, `launchctl kickstart -k` (or `bootstrap` if not loaded).
4. **SessionStart hook:** `home/.claude/hooks/sync-hooks-local-bootstrap/bootstrap.sh` — idempotent `launchctl bootstrap` if not loaded.
5. **PostToolUse hook:** `home/.claude/hooks/sync-hooks-local-activity/update.sh` — touches `last_session_activity.unix`.
6. **SubagentStart hook:** `home/.claude/hooks/sync-hooks-local-subagent-inject/inject.sh` — tells child agents `~/.zg/` mirror exists for fast cold-cache reads.
7. **Stop hook:** `home/.claude/hooks/sync-hooks-local-stop-validate/validate.sh` — verifies mirror produced log within last 1800 s; logs `sync_hooks_local_validate.log`.
8. **settings.json registration:** entries under `hooks.PreToolUse / PostToolUse / SessionStart / SubagentStart / Stop` matching `.*` so the hooks fire on every tool call (idempotent merge — duplicates are skipped).
9. **docs:** this file.
10. **backup:** `backups/guardrail-100pct-2026-05-20/` — pre-change settings.json snapshot.

## Smoke test

```bash
bash AI-Tools/home/.claude/hooks/sync-hooks-local-freshness/check.sh < /dev/null
cat AI-Tools/state/sync_hooks_local/heartbeat.json   # should have ts within last few seconds
launchctl list | grep com.zg.sync_hooks_local        # should show pid + exit 0
```

## Failure modes

- **FUSE umount** → mirror script exits non-zero; KeepAlive=Crashed flips and respawns.
- **SSD full** → rsync emits error to `.sync.stderr.log`; freshness hook detects stale log age and re-kickstarts.
- **Drive offline** → mirror gracefully skips; resumes next cycle.

## Escape hatch

`launchctl bootout gui/$(id -u) /Users/orginal/Library/LaunchAgents/com.zg.sync_hooks_local.plist`

Restore previous state from `backups/guardrail-100pct-2026-05-20/`.
