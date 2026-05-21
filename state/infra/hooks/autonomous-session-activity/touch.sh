#!/usr/bin/env bash
# autonomous-session-activity — PostToolUse signal (Layer 4)
#
# On every tool use, write the current unix timestamp to
# state/autonomous_mode/last_session_activity.unix
#
# The daemon reads this as the "user-online" signal — when fresh (<300s),
# the daemon raises ideation priority (more proactive); when stale, it
# settles into a slower idle cadence.
#
# Layer 4 of 6 in the autonomous-mode guardrail chain. Cheap (~1ms).

set +e
LC_ALL=C

ROOT="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools"
ACT_FILE="$ROOT/state/autonomous_mode/last_session_activity.unix"

mkdir -p "$(dirname "$ACT_FILE")" 2>/dev/null

# Drain stdin
cat >/dev/null 2>&1 || true

# Atomic write
NOW=$(date +%s)
TMP="${ACT_FILE}.tmp.$$"
echo "$NOW" > "$TMP" 2>/dev/null && mv -f "$TMP" "$ACT_FILE" 2>/dev/null

exit 0
