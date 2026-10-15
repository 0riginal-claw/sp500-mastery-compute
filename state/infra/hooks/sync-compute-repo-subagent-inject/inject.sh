#!/usr/bin/env bash
# sync-compute-repo-subagent-inject — SubagentStart (Layer 6 of 10)
set -uo pipefail
ROOT="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools"
HB="$ROOT/state/sync_compute_repo/heartbeat.json"
cat <<EOF_BANNER
=== SYNC_COMPUTE_REPO INHERITANCE ===
EOF_BANNER
if [ -f "$HB" ]; then
    HB_TS=$(python3 -c "import json; print(json.load(open('$HB')).get('ts',0))" 2>/dev/null || echo 0)
    HB_STATUS=$(python3 -c "import json; print(json.load(open('$HB')).get('status','unknown'))" 2>/dev/null || echo unknown)
    NOW=$(date +%s)
    AGE=$((NOW - HB_TS))
    echo "heartbeat_age_sec=$AGE  status=$HB_STATUS"
else
    echo "heartbeat=missing"
fi
echo "Rules: feature is guardrail-grade; do not bypass its state files."
echo "=== END SYNC_COMPUTE_REPO INHERITANCE ==="
exit 0
