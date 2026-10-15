#!/usr/bin/env bash
# sync-compute-repo-freshness — PreToolUse (Layer 3 of 10)
set -uo pipefail
ROOT="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools"
HB="$ROOT/state/sync_compute_repo/heartbeat.json"
PLIST="$ROOT/home/Library/LaunchAgents/com.zg.sync_compute_repo.plist"
STALE_SEC=600
LOG_DIR="$ROOT/logs/auto_solve"
mkdir -p "$LOG_DIR"
[ -f "$HB" ] || { launchctl bootstrap "gui/$(id -u)" "$PLIST" 2>/dev/null || true; exit 0; }
NOW=$(date +%s)
HB_TS=$(python3 -c "import json; print(json.load(open('$HB')).get('ts',0))" 2>/dev/null || echo 0)
AGE=$((NOW - HB_TS))
if [ "$AGE" -gt "$STALE_SEC" ]; then
    echo "[sync-compute-repo-freshness] stale (age=${AGE}s) — respawning" >&2
    launchctl bootstrap "gui/$(id -u)" "$PLIST" 2>/dev/null ||         launchctl kickstart -k "gui/$(id -u)/com.zg.sync_compute_repo" 2>/dev/null || true
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) sync_compute_repo respawn (age=${AGE}s)"         >> "$LOG_DIR/sync_compute_repo_stale_$(date -u +%Y%m%d).md"
fi
exit 0
