#!/usr/bin/env bash
# paper-trade-startup-freshness — PreToolUse (Layer 3 of 10)
set -uo pipefail
ROOT="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools"
HB="$ROOT/state/paper_trade_startup/heartbeat.json"
PLIST="$ROOT/home/Library/LaunchAgents/com.zg.paper_trade_startup.plist"
STALE_SEC=600
LOG_DIR="$ROOT/logs/auto_solve"
mkdir -p "$LOG_DIR"
[ -f "$HB" ] || { launchctl bootstrap "gui/$(id -u)" "$PLIST" 2>/dev/null || true; exit 0; }
NOW=$(date +%s)
HB_TS=$(python3 -c "import json; print(json.load(open('$HB')).get('ts',0))" 2>/dev/null || echo 0)
AGE=$((NOW - HB_TS))
if [ "$AGE" -gt "$STALE_SEC" ]; then
    echo "[paper-trade-startup-freshness] stale (age=${AGE}s) — respawning" >&2
    launchctl bootstrap "gui/$(id -u)" "$PLIST" 2>/dev/null ||         launchctl kickstart -k "gui/$(id -u)/com.zg.paper_trade_startup" 2>/dev/null || true
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) paper_trade_startup respawn (age=${AGE}s)"         >> "$LOG_DIR/paper_trade_startup_stale_$(date -u +%Y%m%d).md"
fi
exit 0
