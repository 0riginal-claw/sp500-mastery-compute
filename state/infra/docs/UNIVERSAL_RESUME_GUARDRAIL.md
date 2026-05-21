# Universal Resume Guardrail

**Status:** guardrail-grade (10/10) as of 2026-05-20 (upgraded from 4/10).

## Purpose

Every worker class — `claude_main`, `claude_subagents`, `openclaw_main`, `openclaw_subagents`, `ollama` — is checkpointed every 30 s to per-class manifests under `state/universal_resume/<class>/manifest.json` + `diff.json`. On session start, the bootstrap hook surfaces a per-class status card and emits `LOST_<class>.md` reports if any worker class shows missing files vs. its claimed manifest.

## Components (10-point checklist)

1. **plist:** `/Users/orginal/Library/LaunchAgents/com.zg.universal_resume_guardrail.plist` — `KeepAlive=true`, `RunAtLoad=true`, `ThrottleInterval=30`, `UNIVERSAL_RESUME_CYCLE_SEC=30`.
2. **State dir + heartbeat:** `state/universal_resume/heartbeat.json` (local mirror at `/Users/orginal/.zg/state/universal_resume/heartbeat.json`).
3. **PreToolUse hook:** `home/.claude/hooks/universal-resume-freshness/check.sh` — verifies heartbeat < 600 s; respawns via `launchctl kickstart -k` if stale, `bootstrap` if unloaded.
4. **SessionStart hook:** `home/.claude/hooks/universal-resume-bootstrap/bootstrap.sh` (pre-existing) — emits per-class status cards.
5. **PostToolUse hook:** `home/.claude/hooks/universal-resume-activity/update.sh` — touches `last_session_activity.unix`.
6. **SubagentStart hook:** `home/.claude/hooks/universal-resume-subagent-inject/inject.sh` — tells children every worker is universally-resumable.
7. **Stop hook:** `home/.claude/hooks/universal-resume-stop-validate/validate.sh` — verifies heartbeat fresh + at least one class manifest present.
8. **settings.json:** hooks wired under PreToolUse/PostToolUse/SessionStart/SubagentStart/Stop.
9. **docs:** this file.
10. **backup:** `backups/guardrail-100pct-2026-05-20/` — pre-edit settings.json snapshot.

## Smoke test

```bash
bash AI-Tools/home/.claude/hooks/universal-resume-freshness/check.sh < /dev/null
cat AI-Tools/state/universal_resume/heartbeat.json
launchctl list | grep com.zg.universal_resume_guardrail
ls AI-Tools/state/universal_resume/_lost_reports/   # LOST_<class>.md if missing
```

## Failure modes

- **Daemon crash mid-cycle** → freshness hook respawns within 600 s; bootstrap surfaces stale heartbeat in next SessionStart card.
- **One class loses sync** → bootstrap emits `LOST_<class>.md` next session start.
- **Drive offline** → daemon falls back to local SSD state path.

## Escape hatch

`launchctl bootout gui/$(id -u) /Users/orginal/Library/LaunchAgents/com.zg.universal_resume_guardrail.plist`

Restore from `backups/guardrail-100pct-2026-05-20/`.
