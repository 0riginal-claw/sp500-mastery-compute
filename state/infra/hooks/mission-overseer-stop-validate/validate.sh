#!/usr/bin/env bash
# mission-overseer-stop-validate — Stop hook (Layer 7 of 10-point guardrail)
#
# At end of every assistant turn, verify mission_overseer produced expected
# output: heartbeat updated within last 10 min OR at least one alert/event
# logged since the prior turn. On miss, log a violation so the NEXT turn's
# context shows it (PostToolUse exit-2 is buggy per memory feedback_auto_signup_architecture.md).

set -uo pipefail

ROOT="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools"
HB="$ROOT/state/mission_overseer/heartbeat.json"
VIOLATIONS="$ROOT/state/mission_overseer/violations.jsonl"
LAST_PROMPT="$HOME/.claude/state/last_user_prompt.unix"
STALE_SEC=600

NOW=$(date +%s)
if [ -f "$HB" ]; then
    HB_TS=$(python3 -c "import json; print(json.load(open('$HB')).get('ts',0))" 2>/dev/null || echo 0)
    AGE=$((NOW - HB_TS))
    if [ "$AGE" -gt "$STALE_SEC" ]; then
        echo "{\"ts\":$NOW,\"violation\":\"stale_heartbeat\",\"age\":$AGE}" >> "$VIOLATIONS"
        echo "[mission-overseer-stop-validate] WARN: heartbeat stale (${AGE}s)" >&2
    fi
fi
exit 0
