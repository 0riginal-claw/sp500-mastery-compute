#!/usr/bin/env bash
# mission-overseer-activity — PostToolUse hook (Layer 5 of 10-point guardrail)
#
# Touch last_session_activity.unix so the daemon knows the user is online.

set -uo pipefail

ROOT="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools"
ACTIVITY_FILE="$ROOT/state/mission_overseer/last_session_activity.unix"

mkdir -p "$(dirname "$ACTIVITY_FILE")"
date +%s > "$ACTIVITY_FILE" 2>/dev/null || true
exit 0
