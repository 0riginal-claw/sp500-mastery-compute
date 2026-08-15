#!/usr/bin/env bash
# sync-hooks-local-bootstrap (SessionStart): ensure mirror job is loaded.
# Created 2026-05-20 by guardrail-100pct remediation.
set +e
LC_ALL=C
cat >/dev/null 2>&1
LABEL="com.zg.sync_hooks_local"
PLIST="/Users/orginal/Library/LaunchAgents/$LABEL.plist"
LOG="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/logs/sync_hooks_local_bootstrap.log"
mkdir -p "$(dirname "$LOG")" 2>/dev/null
if ! /bin/launchctl list 2>/dev/null | /usr/bin/grep -q "$LABEL"; then
  if [[ -f "$PLIST" ]]; then
    /bin/launchctl bootstrap "gui/$(id -u)" "$PLIST" 2>>"$LOG"
    echo "[$(date +%FT%TZ)] sync-hooks-local-bootstrap: loaded $LABEL" >> "$LOG"
  fi
fi
exit 0
