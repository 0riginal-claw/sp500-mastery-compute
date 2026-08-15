#!/usr/bin/env bash
# sync-hooks-local-freshness (PreToolUse): verify mirror is fresh, auto-respawn.
# Created 2026-05-20 by guardrail-100pct remediation.
set +e
LC_ALL=C
cat >/dev/null 2>&1
LABEL="com.zg.sync_hooks_local"
PLIST="/Users/orginal/Library/LaunchAgents/$LABEL.plist"
STATE_DIR="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/state/sync_hooks_local"
HB="$STATE_DIR/heartbeat.json"
LOG_FILE="/Users/orginal/.zg/.sync.stdout.log"
LOG="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/logs/sync_hooks_local_freshness.log"
mkdir -p "$STATE_DIR" "$(dirname "$LOG")" 2>/dev/null
AGE=99999
if [[ -f "$LOG_FILE" ]]; then
  NOW=$(date +%s)
  MTIME=$(stat -f %m "$LOG_FILE" 2>/dev/null || echo 0)
  AGE=$((NOW - MTIME))
fi
/usr/bin/python3 -c "
import json, time, os
hb = {'ts': int(time.time()), 'pid': os.getpid(), 'cycle_id': 'freshness-check', 'status': 'checked', 'sync_log_age_sec': $AGE}
tmp = '$HB.tmp'
with open(tmp, 'w') as f:
    json.dump(hb, f)
os.replace(tmp, '$HB')
" 2>/dev/null
if [[ "$AGE" -gt 600 ]]; then
  if [[ -f "$PLIST" ]]; then
    if /bin/launchctl list 2>/dev/null | /usr/bin/grep -q "$LABEL"; then
      /bin/launchctl kickstart -k "gui/$(id -u)/$LABEL" 2>>"$LOG"
    else
      /bin/launchctl bootstrap "gui/$(id -u)" "$PLIST" 2>>"$LOG"
    fi
    echo "[$(date +%FT%TZ)] sync-hooks-local-freshness: STALE (${AGE}s); respawn" >> "$LOG"
  fi
fi
exit 0
