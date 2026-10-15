#!/usr/bin/env bash
# agent-watchdog-bootstrap — SessionStart (Layer 4 of 10)
set -uo pipefail
ROOT="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools"
PLIST="$ROOT/home/Library/LaunchAgents/com.zg.agent_watchdog.plist"
LOG="$ROOT/logs/agent_watchdog_bootstrap.log"
LABEL="com.zg.agent_watchdog"
launchctl list 2>/dev/null | grep -q "$LABEL" && exit 0
[ -f "$PLIST" ] || { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) ERROR: plist missing" >> "$LOG"; exit 0; }
launchctl bootstrap "gui/$(id -u)" "$PLIST" 2>>"$LOG" ||     echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) bootstrap rc=$? (likely already loaded)" >> "$LOG"
exit 0
