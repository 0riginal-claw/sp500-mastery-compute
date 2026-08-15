#!/usr/bin/env bash
# agent-watchdog-activity — PostToolUse (Layer 5 of 10)
set -uo pipefail
ROOT="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools"
A="$ROOT/state/agent_watchdog/last_session_activity.unix"
mkdir -p "$(dirname "$A")"
date +%s > "$A" 2>/dev/null || true
exit 0
