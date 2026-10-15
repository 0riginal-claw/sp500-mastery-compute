# TCC Admin Canceler

**Status:** guardrail-grade (10/10) as of 2026-05-20 — wraps the cancel-admin patch (already live inside `tcc-dialog-detect/scan.applescript`) as a tracked feature.

## Why it exists

The patch itself is a 5-line addition to `home/.claude/hooks/tcc-dialog-detect/scan.applescript` (`cancelTitles` list + SecurityAgent app). This wrapper makes the patch's continuing presence *observable* — if the cancel block is ever removed or overwritten, the freshness hook flags it in the per-turn log and the Stop hook validates again.

## What gets cancelled

SecurityAgent admin-elevation dialogs containing any of:
- "administer your computer"
- "would like to administer your computer"
- "wants to administer your computer"

Cancel button is clicked (or "Don't Allow" as fallback). This is a SECURITY BOUNDARY — admin elevation is NEVER granted automatically. The user must run privileged commands explicitly.

## Components (10-point checklist)

1. **plist:** none — the canceler runs as part of `tcc-dialog-detect`'s scan loop.
2. **State dir + heartbeat:** `state/tcc_admin_canceler/heartbeat.json` — includes `patch_present: yes/no`.
3. **PreToolUse hook:** `home/.claude/hooks/tcc-admin-canceler-freshness/check.sh` — re-greps for `"administer your computer"` in scan.applescript every tool call; flags if removed.
4. **SessionStart hook:** `home/.claude/hooks/tcc-admin-canceler-bootstrap/bootstrap.sh` — verifies patch presence + emits heartbeat.
5. **PostToolUse hook:** `home/.claude/hooks/tcc-admin-canceler-activity/update.sh` — touches `last_session_activity.unix`.
6. **SubagentStart hook:** `home/.claude/hooks/tcc-admin-canceler-subagent-inject/inject.sh` — tells children admin elevation is auto-cancelled (do NOT bypass).
7. **Stop hook:** `home/.claude/hooks/tcc-admin-canceler-stop-validate/validate.sh` — re-grep for patch; logs `tcc_admin_canceler_validate.log`.
8. **settings.json:** all 4 hooks wired.
9. **docs:** this file.
10. **backup:** `backups/guardrail-100pct-2026-05-20/scan.applescript.before` snapshot.

## Smoke test

```bash
bash AI-Tools/home/.claude/hooks/tcc-admin-canceler-bootstrap/bootstrap.sh < /dev/null
cat AI-Tools/state/tcc_admin_canceler/heartbeat.json   # patch_present should be "yes"
grep "administer your computer" AI-Tools/home/.claude/hooks/tcc-dialog-detect/scan.applescript
```

## Escape hatch

Remove `cancelTitles` entries from `scan.applescript` (the canceler stops firing but allow/deny logic continues). Restore from `backups/guardrail-100pct-2026-05-20/scan.applescript.before` if needed.
