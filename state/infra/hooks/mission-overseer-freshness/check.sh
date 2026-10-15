#!/usr/bin/env bash
# mission-overseer-freshness — PreToolUse hook (Layer 3 of 10-point guardrail)
#
# Verifies the mission_overseer heartbeat is fresh (<10 min). If stale,
# attempts launchctl bootstrap (idempotent — silent if already loaded).
# Never blocks tool calls — always exits 0. Logs to auto_solve on failure.

set -uo pipefail

ROOT="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools"
HEARTBEAT="$ROOT/state/mission_overseer/heartbeat.json"
PLIST="$ROOT/home/Library/LaunchAgents/com.zg.mission_overseer.plist"
STALE_SEC=600   # 10 minutes
LOG_DIR="$ROOT/logs/auto_solve"
mkdir -p "$LOG_DIR"

if [ ! -f "$HEARTBEAT" ]; then
    echo "[mission-overseer-freshness] heartbeat missing — attempting bootstrap" >&2
    launchctl bootstrap "gui/$(id -u)" "$PLIST" 2>/dev/null || true
    exit 0
fi

NOW=$(date +%s)
HB_TS=$(python3 -c "import json; print(json.load(open('$HEARTBEAT')).get('ts', 0))" 2>/dev/null || echo 0)
AGE=$((NOW - HB_TS))

if [ "$AGE" -gt "$STALE_SEC" ]; then
    echo "[mission-overseer-freshness] stale heartbeat (age=${AGE}s > ${STALE_SEC}s) — respawning" >&2
    launchctl bootstrap "gui/$(id -u)" "$PLIST" 2>/dev/null || \
        launchctl kickstart -k "gui/$(id -u)/com.zg.mission_overseer" 2>/dev/null || true
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) mission_overseer respawn attempted (age=${AGE}s)" \
        >> "$LOG_DIR/mission_overseer_stale_$(date -u +%Y%m%d).md"
fi

exit 0
