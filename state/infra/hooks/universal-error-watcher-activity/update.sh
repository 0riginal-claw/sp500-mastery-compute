#!/usr/bin/env bash
# PostToolUse activity hook: update last_session_activity.unix so the watcher
# knows the user is online (mirrors the autonomous_mode pattern). Non-blocking.

set +u
DIR_LOCAL="/Users/orginal/.zg/state/universal_error_watcher"
DIR_DRIVE="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/state/universal_error_watcher"
mkdir -p "$DIR_LOCAL" "$DIR_DRIVE" 2>/dev/null
ts=$(date +%s)
for d in "$DIR_LOCAL" "$DIR_DRIVE"; do
  echo "$ts" > "$d/last_session_activity.unix" 2>/dev/null
done
exit 0
