#!/usr/bin/env bash
# PreToolUse freshness hook: verify universal_error_watcher heartbeat <10min old.
# If stale, attempt auto-respawn via launchctl bootstrap.
# Always exits 0 (non-blocking) — this is a self-heal hook, not a gate.

set +u
HB_LOCAL="/Users/orginal/.zg/state/universal_error_watcher/heartbeat.json"
HB_DRIVE="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/state/universal_error_watcher/heartbeat.json"
LABEL="com.zg.universal_error_watcher"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
STALE_SEC=600

now=$(date +%s)
hb=""
for f in "$HB_LOCAL" "$HB_DRIVE"; do
  [[ -f "$f" ]] && hb="$f" && break
done

needs_respawn=0
if [[ -z "$hb" ]]; then
  needs_respawn=1
else
  mtime=$(stat -f %m "$hb" 2>/dev/null || echo 0)
  age=$(( now - mtime ))
  (( age > STALE_SEC )) && needs_respawn=1
fi

if (( needs_respawn )); then
  if [[ -f "$PLIST" ]]; then
    /bin/launchctl bootout "gui/$(id -u)" "$PLIST" >/dev/null 2>&1 || true
    /bin/launchctl bootstrap "gui/$(id -u)" "$PLIST" >/dev/null 2>&1 || \
      /bin/launchctl load "$PLIST" >/dev/null 2>&1 || true
  fi
fi

exit 0
