#!/usr/bin/env bash
# memory-auto-save-subagent-inject — SubagentStart (Layer 6 of 10)
set -uo pipefail
ROOT="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools"
HB="$ROOT/state/memory_auto_save/heartbeat.json"
cat <<EOF_BANNER
=== MEMORY_AUTO_SAVE INHERITANCE ===
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
echo "=== END MEMORY_AUTO_SAVE INHERITANCE ==="
exit 0
