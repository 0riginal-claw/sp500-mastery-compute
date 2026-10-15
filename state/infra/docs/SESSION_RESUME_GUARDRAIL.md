# Session Resume Guardrail

**Status:** guardrail-grade (10/10) as of 2026-05-20 (upgraded from 7/10).

## Purpose

Long-form Claude Code sessions get a per-session checkpoint every ~30 s containing host state, daemon list, last-known tool use, and rolling tool-use jsonl. On next session start, the bootstrap hook surfaces resume context (prior session_id, last checkpoint ts, host load, daemons tracked). Capture/inject hooks roll the rolling log.

## Components (10-point checklist)

1. **plist:** `/Users/orginal/Library/LaunchAgents/com.zg.session_resume_checkpointer.plist`.
2. **State dir + heartbeat:** `state/session_resume/heartbeat.json` (+ local mirror at `/Users/orginal/.zg/state/session_resume/`).
3. **PreToolUse hook:** `home/.claude/hooks/session-resume-heartbeat/heartbeat.sh` — atomic touch `last_tool_use.unix`.
4. **SessionStart hook:** `home/.claude/hooks/session-resume-bootstrap/bootstrap.sh` — emits prior-session resume context.
5. **PostToolUse hook:** `home/.claude/hooks/session-resume-capture/capture.sh` — appends rolling tool_use.jsonl (cap 2 MB).
6. **SubagentStart hook:** `home/.claude/hooks/session-resume-subagent-inject/inject.sh` — passes resume context to children.
7. **Stop hook:** `home/.claude/hooks/session-resume-stop-validate/validate.sh` — verifies heartbeat fresh; logs to `session_resume_validate.log`.
8. **settings.json:** all 5 hooks wired.
9. **docs:** this file (replaces the brief 1.3 KB stub).
10. **backup:** `backups/session_resume_guardrail-pre-install-2026-05-20-174903/` (pre-existing) + `backups/guardrail-100pct-2026-05-20/`.

## Smoke test

```bash
bash AI-Tools/home/.claude/hooks/session-resume-heartbeat/heartbeat.sh < /dev/null
cat AI-Tools/state/session_resume/heartbeat.json
launchctl list | grep com.zg.session_resume_checkpointer
```

## Failure modes

- **Drive FUSE flap** → daemon writes to local `/Users/orginal/.zg/state/session_resume/` and mirrors when reachable.
- **tool_use.jsonl > 2 MB** → capture hook truncates to last 1000 lines.

## Escape hatch

`launchctl bootout gui/$(id -u) /Users/orginal/Library/LaunchAgents/com.zg.session_resume_checkpointer.plist`
