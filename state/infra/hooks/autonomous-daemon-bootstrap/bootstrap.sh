#!/usr/bin/env bash
# autonomous-daemon-bootstrap — SessionStart guardrail (Layer 3)
#
# At the start of every Claude Code session, ensure the autonomous_mode_daemon
# is loaded into launchd. Idempotent — if already loaded, no-op.
#
# Layer 3 of 6 in the autonomous-mode guardrail chain.

set -uo pipefail
LC_ALL=C

ROOT="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools"
PLIST="$HOME/Library/LaunchAgents/com.zg.autonomous_mode.plist"
LABEL="com.zg.autonomous_mode"
LOG_DIR="$ROOT/logs/auto_solve"
LOG_FILE="$LOG_DIR/autonomous_guardrails.log"

mkdir -p "$LOG_DIR" 2>/dev/null

cat >/dev/null 2>&1 || true

log() {
  printf '[%s] autonomous-daemon-bootstrap: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$LOG_FILE" 2>/dev/null
}

if [ ! -f "$PLIST" ]; then
  log "skip — plist missing at $PLIST"
  exit 0
fi

UID_REAL=$(id -u)

# Already loaded? -> no-op
if launchctl print "gui/$UID_REAL/$LABEL" >/dev/null 2>&1; then
  log "already loaded — no-op"
  exit 0
fi

# Bootstrap
launchctl bootstrap "gui/$UID_REAL" "$PLIST" >/dev/null 2>&1
RC=$?

if [ "$RC" -eq 0 ]; then
  log "bootstrapped daemon at session start"
else
  log "bootstrap FAILED rc=$RC"
fi

exit 0
