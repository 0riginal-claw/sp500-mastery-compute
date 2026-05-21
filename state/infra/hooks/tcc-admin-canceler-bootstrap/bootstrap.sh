#!/usr/bin/env bash
# tcc-admin-canceler-bootstrap (SessionStart): emit heartbeat indicating
# the cancel-admin patch is live in tcc-dialog-detect/scan.applescript.
# Created 2026-05-20 by guardrail-100pct remediation.
set +e
LC_ALL=C
cat >/dev/null 2>&1
STATE="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/state/tcc_admin_canceler"
SCAN="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/home/.claude/hooks/tcc-dialog-detect/scan.applescript"
mkdir -p "$STATE" 2>/dev/null
# Verify patch present
PATCH_PRESENT="no"
if [[ -f "$SCAN" ]]; then
  /usr/bin/grep -q "administer your computer" "$SCAN" 2>/dev/null && PATCH_PRESENT="yes"
fi
/usr/bin/python3 -c "
import json, os, time
hb = {'ts': int(time.time()), 'pid': os.getpid(), 'cycle_id': 'bootstrap', 'status': 'alive', 'patch_present': '$PATCH_PRESENT', 'host_hook': 'tcc-dialog-detect/scan.applescript', 'cancel_titles': ['administer your computer']}
p = '$STATE/heartbeat.json'
tmp = p + '.tmp'
with open(tmp, 'w') as f:
    json.dump(hb, f)
os.replace(tmp, p)
" 2>/dev/null
exit 0
