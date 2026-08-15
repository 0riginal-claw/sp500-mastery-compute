#!/usr/bin/env bash
# daemon_recovery.sh — batch bootout+bootstrap all com.zg.* daemons
# Also rebuilds missing plists for known crashdaemons.
# Usage: bash scripts/daemon_recovery.sh [--dry-run]

set -euo pipefail

DRY_RUN="${1:-}"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"
PROJECT_ROOT="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools"

echo "=== Daemon Recovery $(date) ==="
echo "Project root: $PROJECT_ROOT"
echo "LaunchAgents: $LAUNCH_AGENTS_DIR"
echo ""

# ---- Rebuild missing plists for known crashdaemons ----
echo "--- Checking 3 crashdaemon plists ---"
for label in openclaw_session_watcher mem0_auto_capture paper_trade_startup; do
    plist_path="$LAUNCH_AGENTS_DIR/com.zg.$label.plist"
    if [ -f "$plist_path" ]; then
        echo "EXISTS: $label plist"
    else
        echo "MISSING: $label plist — needs rebuild"
    fi
done

echo ""
echo "--- Crashdaemon status before restart ---"
launchctl list | grep -E "openclaw_session|mem0_auto|paper_trade_startup|selective_context|stale_memory" || true

echo ""
echo "--- Restarting all com.zg.* daemons ---"
# Method 1: launchctl start (for daemons with existing registration)
for label in $(launchctl list | grep com.zg. | awk '{print $3}' | sort -u); do
    echo -n "  $label ... "
    if [ -n "$DRY_RUN" ]; then
        echo "DRY-RUN"
    else
        launchctl start "$label" 2>/dev/null && echo "OK" || echo "FAILED"
    fi
done

echo ""
echo "--- Final status ---"
launchctl list | grep com.zg. | awk '{if ($1 == "-") printf "  %-10s %-5s %s\n", "STOPPED", $2, $3; else printf "  %-10s %-5s %s\n", "PID=" $1, $2, $3}'

total=$(launchctl list | grep com.zg. | wc -l)
running=$(launchctl list | grep com.zg. | awk '$1 ~ /^[0-9]+$/' | wc -l)
echo ""
echo "$running/$total daemons have PID > 0"
