#!/usr/bin/env bash
# Stop hook: validate universal_error_watcher produced expected output
# during this turn (recent heartbeat). Shames next turn on miss (stderr emit).

set +u
HB_LOCAL="/Users/orginal/.zg/state/universal_error_watcher/heartbeat.json"

ok=0
if [[ -f "$HB_LOCAL" ]]; then
  mtime=$(stat -f %m "$HB_LOCAL" 2>/dev/null || echo 0)
  age=$(( $(date +%s) - mtime ))
  if (( age < 120 )); then
    ok=1
  fi
fi

if (( ! ok )); then
  cat >&2 <<'EOF'
universal_error_watcher: no recent heartbeat (>2min). Daemon may be down.
  - Check: launchctl list | grep com.zg.universal_error_watcher
  - Logs:  ~/.zg/state/universal_error_watcher/daemon.stderr.log
  - Bootstrap: launchctl bootstrap gui/$UID ~/Library/LaunchAgents/com.zg.universal_error_watcher.plist
EOF
fi
exit 0
