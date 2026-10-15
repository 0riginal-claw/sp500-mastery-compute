#!/usr/bin/env bash
# session-resume-stop-validate (Stop): verify a session checkpoint was
# written during this turn; log warning if not (non-blocking).
# Created 2026-05-20 by six-fail-fix F4b.

set +e
LC_ALL=C

# Drain stdin
cat >/dev/null 2>&1

LOCAL_DIR="/Users/orginal/.zg/state/session_resume"
DRIVE_DIR="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/state/session_resume"
LOG_DIR="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/logs"
mkdir -p "$LOG_DIR" 2>/dev/null

HB="$LOCAL_DIR/heartbeat.json"
[[ ! -s "$HB" ]] && HB="$DRIVE_DIR/heartbeat.json"

if [[ ! -s "$HB" ]]; then
    echo "[$(date +%FT%TZ)] session-resume-stop-validate: NO HEARTBEAT found" >> "$LOG_DIR/session_resume_validate.log"
    exit 0
fi

# Check heartbeat is fresh (<= 5 min)
AGE=$(python3 -c "
import json, time, os
try:
    hb = json.loads(open('$HB').read())
    print(int(time.time() - hb.get('ts', 0)))
except Exception:
    print(-1)
" 2>/dev/null)

if [[ "$AGE" =~ ^-?[0-9]+$ ]] && [[ "$AGE" -gt 300 ]]; then
    echo "[$(date +%FT%TZ)] session-resume-stop-validate: STALE heartbeat age=${AGE}s" >> "$LOG_DIR/session_resume_validate.log"
elif [[ "$AGE" =~ ^-?[0-9]+$ ]] && [[ "$AGE" -ge 0 ]]; then
    echo "[$(date +%FT%TZ)] session-resume-stop-validate: OK age=${AGE}s" >> "$LOG_DIR/session_resume_validate.log"
fi

exit 0
