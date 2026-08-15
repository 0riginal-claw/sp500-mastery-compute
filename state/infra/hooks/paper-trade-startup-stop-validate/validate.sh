#!/usr/bin/env bash
# paper-trade-startup-stop-validate — Stop (Layer 7 of 10)
set -uo pipefail
ROOT="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools"
HB="$ROOT/state/paper_trade_startup/heartbeat.json"
V="$ROOT/state/paper_trade_startup/violations.jsonl"
STALE_SEC=600
NOW=$(date +%s)
[ -f "$HB" ] || exit 0
HB_TS=$(python3 -c "import json; print(json.load(open('$HB')).get('ts',0))" 2>/dev/null || echo 0)
AGE=$((NOW - HB_TS))
if [ "$AGE" -gt "$STALE_SEC" ]; then
    echo "{\"ts\":$NOW,\"violation\":\"stale_heartbeat\",\"age\":$AGE}" >> "$V"
    echo "[paper-trade-startup-stop-validate] WARN: heartbeat stale (${AGE}s)" >&2
fi
exit 0
