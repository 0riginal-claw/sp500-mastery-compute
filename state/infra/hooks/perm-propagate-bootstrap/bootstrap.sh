#!/usr/bin/env bash
# perm-propagate-bootstrap (SessionStart): ensure perm_propagate state dir
# exists locally + on Drive; emit heartbeat so the marker is observable.
# Created 2026-05-20 by guardrail-100pct remediation.
set +e
LC_ALL=C
cat >/dev/null 2>&1
LOCAL="/Users/orginal/.claude/state/perm_propagate"
DRIVE_STATE="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/state/perm_propagate"
mkdir -p "$LOCAL" "$DRIVE_STATE" 2>/dev/null
TS=$(date +%s)
/usr/bin/python3 -c "
import json, os
hb = {'ts': $TS, 'pid': os.getpid(), 'cycle_id': 'bootstrap', 'status': 'alive', 'integration': 'autosolve-require + spawn-validator hooks enforce 3-solver fan-out + #model_reason'}
for d in ('$LOCAL', '$DRIVE_STATE'):
    p = os.path.join(d, 'heartbeat.json')
    tmp = p + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(hb, f)
    os.replace(tmp, p)
" 2>/dev/null
exit 0
