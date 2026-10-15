#!/usr/bin/env bash
# perm-propagate-stop-validate (Stop): confirm marker heartbeat ts was
# touched this turn. Log to perm_propagate_validate.log. Non-blocking.
# Created 2026-05-20 by guardrail-100pct remediation.
set +e
LC_ALL=C
cat >/dev/null 2>&1
HB="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/state/perm_propagate/heartbeat.json"
LOG="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/logs/perm_propagate_validate.log"
mkdir -p "$(dirname "$LOG")" 2>/dev/null
if [[ -s "$HB" ]]; then
  AGE=$(/usr/bin/python3 -c "
import json, time
try:
    print(int(time.time() - json.load(open('$HB')).get('ts', 0)))
except Exception:
    print(-1)
" 2>/dev/null)
  echo "[$(date +%FT%TZ)] perm-propagate-stop-validate: age=${AGE}s" >> "$LOG"
else
  echo "[$(date +%FT%TZ)] perm-propagate-stop-validate: NO HEARTBEAT" >> "$LOG"
fi
exit 0
