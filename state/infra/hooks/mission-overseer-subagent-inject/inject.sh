#!/usr/bin/env bash
# mission-overseer-subagent-inject — SubagentStart hook (Layer 6 of 10-point guardrail)
#
# Inject mission_overseer state into every sub-agent's context so the entire
# tree inherits awareness of pending solvers, alerts, and load history.

set -uo pipefail

ROOT="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools"
STATE_DIR="$ROOT/state/mission_overseer"
HB="$STATE_DIR/heartbeat.json"
ALERTS="$STATE_DIR/alert_history.jsonl"
PENDING="$STATE_DIR/pending_solvers"

cat <<'EOF'
=== MISSION-OVERSEER INHERITANCE (workspace solver-coordinator) ===
EOF

if [ -f "$HB" ]; then
    HB_TS=$(python3 -c "import json; d=json.load(open('$HB')); print(d.get('ts',0))" 2>/dev/null || echo 0)
    HB_STATUS=$(python3 -c "import json; d=json.load(open('$HB')); print(d.get('status','unknown'))" 2>/dev/null || echo unknown)
    NOW=$(date +%s)
    AGE=$((NOW - HB_TS))
    echo "heartbeat_age_sec=$AGE  status=$HB_STATUS"
else
    echo "heartbeat=missing"
fi

# Last 2 alerts
if [ -f "$ALERTS" ]; then
    echo "recent_alerts:"
    tail -n 2 "$ALERTS" | sed 's/^/  /'
fi

# Pending solver count
if [ -d "$PENDING" ]; then
    COUNT=$(find "$PENDING" -type f 2>/dev/null | wc -l | tr -d ' ')
    echo "pending_solvers=$COUNT"
fi

cat <<'EOF'
Rules:
- If pending_solvers > 0 and your task is unrelated, do NOT clobber state.
- Solver coordination uses state/mission_overseer/ — never delete entries; mark resolved.
=== END MISSION-OVERSEER INHERITANCE ===
EOF
exit 0
