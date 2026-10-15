#!/usr/bin/env bash
# mission-overseer-bootstrap — SessionStart hook (Layer 4 of 10-point guardrail)
#
# Idempotently ensures the mission_overseer daemon is loaded.
# - If already running: exit 0 silently
# - If not: launchctl bootstrap from plist, log result

set -uo pipefail

ROOT="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools"
PLIST="$ROOT/home/Library/LaunchAgents/com.zg.mission_overseer.plist"
LOG="$ROOT/logs/mission_overseer_bootstrap.log"
LABEL="com.zg.mission_overseer"

if launchctl list 2>/dev/null | grep -q "$LABEL"; then
    exit 0
fi

if [ ! -f "$PLIST" ]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) ERROR: plist missing $PLIST" >> "$LOG"
    exit 0
fi

launchctl bootstrap "gui/$(id -u)" "$PLIST" 2>>"$LOG" || \
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) bootstrap rc=$? (likely already loaded)" >> "$LOG"

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) mission_overseer bootstrap attempted" >> "$LOG"
exit 0
