#!/usr/bin/env bash
# autonomous-daemon-heartbeat — PreToolUse guardrail (Layer 2)
#
# If the autonomous_mode_daemon's heartbeat is stale (>HEARTBEAT_MAX_AGE_SEC,
# default 300s = 5 min), auto-respawn via launchctl. Non-blocking by default
# (exit 0 so the tool call proceeds). Guardrail-grade — survives:
#   * daemon crash (launchctl KeepAlive missed)
#   * launchd bootstrap missing (after reboot, before any session)
#   * heartbeat file deleted (treated as "never started")
#
# Layer 2 of 6 in the autonomous-mode guardrail chain.
# See docs/AUTONOMOUS_MODE.md "Guardrail-grade enforcement (2026-05-20)".

set -uo pipefail
LC_ALL=C

ROOT="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools"
HB="$ROOT/state/autonomous_mode/heartbeat.json"
PLIST="$HOME/Library/LaunchAgents/com.zg.autonomous_mode.plist"
LABEL="com.zg.autonomous_mode"
LOG_DIR="$ROOT/logs/auto_solve"
LOG_FILE="$LOG_DIR/autonomous_guardrails.log"
COOLDOWN_FILE="$ROOT/state/autonomous_mode/.heartbeat_hook_cooldown"
HEARTBEAT_MAX_AGE_SEC="${HEARTBEAT_MAX_AGE_SEC:-300}"
RESPAWN_COOLDOWN_SEC="${RESPAWN_COOLDOWN_SEC:-60}"

mkdir -p "$LOG_DIR" 2>/dev/null
mkdir -p "$(dirname "$COOLDOWN_FILE")" 2>/dev/null

# Drain stdin so the parent doesn't block.
cat >/dev/null 2>&1 || true

log() {
  printf '[%s] autonomous-daemon-heartbeat: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$LOG_FILE" 2>/dev/null
}

NOW=$(date +%s)

# Cooldown: if we just respawned within COOLDOWN sec, skip (avoid hot-loop).
if [ -f "$COOLDOWN_FILE" ]; then
  LAST=$(cat "$COOLDOWN_FILE" 2>/dev/null || echo 0)
  if [ -n "$LAST" ] && [ "$LAST" -gt 0 ] 2>/dev/null; then
    AGE=$((NOW - LAST))
    if [ "$AGE" -lt "$RESPAWN_COOLDOWN_SEC" ]; then
      exit 0
    fi
  fi
fi

UID_REAL=$(id -u)

needs_respawn=0
reason=""

if [ ! -f "$HB" ]; then
  # No heartbeat file at all — treat as never-started.
  needs_respawn=1
  reason="no_heartbeat_file"
else
  HB_MTIME=$(stat -f %m "$HB" 2>/dev/null || echo 0)
  AGE=$((NOW - HB_MTIME))
  if [ "$AGE" -gt "$HEARTBEAT_MAX_AGE_SEC" ]; then
    needs_respawn=1
    reason="stale_heartbeat_age=${AGE}s"
  fi
fi

# Also respawn if launchctl doesn't know about the service at all.
if [ "$needs_respawn" -eq 0 ]; then
  if ! launchctl print "gui/$UID_REAL/$LABEL" >/dev/null 2>&1; then
    needs_respawn=1
    reason="not_in_launchctl"
  fi
fi

if [ "$needs_respawn" -eq 0 ]; then
  exit 0
fi

# Check plist exists; if not, log + exit (cannot bootstrap without it).
if [ ! -f "$PLIST" ]; then
  log "WARN cannot respawn — plist missing at $PLIST"
  exit 0
fi

# Record cooldown before attempting (so a failed bootstrap doesn't hot-loop).
echo "$NOW" > "$COOLDOWN_FILE" 2>/dev/null

# Best-effort bootout (ignore failures), then bootstrap.
launchctl bootout "gui/$UID_REAL/$LABEL" >/dev/null 2>&1 || true
sleep 0.5
launchctl bootstrap "gui/$UID_REAL" "$PLIST" >/dev/null 2>&1
sleep 0.5

# Verify by post-state, not return code (rc=5 "already loaded" is success-equivalent).
if launchctl print "gui/$UID_REAL/$LABEL" >/dev/null 2>&1; then
  log "respawned daemon (reason=$reason)"
  printf '[autonomous-daemon-heartbeat] daemon respawned (%s)\n' "$reason" >&2
else
  # Fall back: try a kickstart (force the service to launch even if already loaded).
  launchctl kickstart -k "gui/$UID_REAL/$LABEL" >/dev/null 2>&1
  sleep 0.5
  if launchctl print "gui/$UID_REAL/$LABEL" >/dev/null 2>&1; then
    log "respawned daemon via kickstart (reason=$reason)"
    printf '[autonomous-daemon-heartbeat] daemon respawned via kickstart (%s)\n' "$reason" >&2
  else
    log "respawn FAILED — daemon still not in launchctl (reason=$reason)"
    printf '[autonomous-daemon-heartbeat] respawn failed — daemon not loaded (%s)\n' "$reason" >&2
  fi
fi

# Non-blocking — let the tool call proceed.
exit 0
