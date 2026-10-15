#!/usr/bin/env bash
# sync-hooks-local-stop-validate (Stop): verify mirror produced log during turn.
# Created 2026-05-20 by guardrail-100pct remediation.
set +e
LC_ALL=C
cat >/dev/null 2>&1
LOG_FILE="/Users/orginal/.zg/.sync.stdout.log"
VAL_LOG="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/logs/sync_hooks_local_validate.log"
mkdir -p "$(dirname "$VAL_LOG")" 2>/dev/null
if [[ -f "$LOG_FILE" ]]; then
  NOW=$(date +%s)
  MTIME=$(stat -f %m "$LOG_FILE" 2>/dev/null || echo 0)
  AGE=$((NOW - MTIME))
  if [[ "$AGE" -gt 1800 ]]; then
    echo "[$(date +%FT%TZ)] sync-hooks-local-stop-validate: STALE log age=${AGE}s" >> "$VAL_LOG"
  else
    echo "[$(date +%FT%TZ)] sync-hooks-local-stop-validate: OK age=${AGE}s" >> "$VAL_LOG"
  fi
fi
exit 0
