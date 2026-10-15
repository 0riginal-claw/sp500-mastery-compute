#!/usr/bin/env bash
# tcc-admin-canceler-freshness (PreToolUse): re-verify cancel patch is still
# present in scan.applescript (defends against accidental removal).
# Created 2026-05-20 by guardrail-100pct remediation.
set +e
LC_ALL=C
cat >/dev/null 2>&1
STATE="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/state/tcc_admin_canceler"
SCAN="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/home/.claude/hooks/tcc-dialog-detect/scan.applescript"
LOG="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/logs/tcc_admin_canceler_freshness.log"
mkdir -p "$STATE" "$(dirname "$LOG")" 2>/dev/null
PATCH_PRESENT="no"
if [[ -f "$SCAN" ]] && /usr/bin/grep -q "administer your computer" "$SCAN" 2>/dev/null; then
  PATCH_PRESENT="yes"
fi
/usr/bin/python3 -c "
import json, os, time
hb = {'ts': int(time.time()), 'pid': os.getpid(), 'cycle_id': 'freshness', 'status': 'alive', 'patch_present': '$PATCH_PRESENT'}
p = '$STATE/heartbeat.json'
tmp = p + '.tmp'
with open(tmp, 'w') as f:
    json.dump(hb, f)
os.replace(tmp, p)
" 2>/dev/null
if [[ "$PATCH_PRESENT" == "no" ]]; then
  echo "[$(date +%FT%TZ)] tcc-admin-canceler-freshness: PATCH MISSING from $SCAN" >> "$LOG"
fi
exit 0
