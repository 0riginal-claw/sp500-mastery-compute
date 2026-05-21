#!/usr/bin/env bash
# gabriel-self-validate — Stop guardrail (Hook 5 of 6)
#
# At end of every assistant turn, check that capability_map.json OR
# reflexions.jsonl was updated since the last user prompt. If neither
# was updated, append an inbox entry forcing the autonomous-mode daemon
# to run _reflect() on its next cycle.
#
# This protects against the silent-failure mode where the self-awareness
# module exists but never actually learns anything new.

set +e
LC_ALL=C

ROOT="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools"
STATE_DIR="$ROOT/state/gabriel_self"
INBOX="$ROOT/state/autonomous_mode/user_inbox.jsonl"
LAST_USER_TS_FILE="$HOME/.claude/state/last_user_prompt.unix"
LOG_DIR="$ROOT/logs/gabriel_self"
LOG_FILE="$LOG_DIR/self_validate.log"

mkdir -p "$STATE_DIR" "$LOG_DIR" "$(dirname "$INBOX")" 2>/dev/null

# Drain stdin
cat >/dev/null 2>&1 || true

log() {
  printf '[%s] gabriel-self-validate: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$LOG_FILE" 2>/dev/null
}

# No last-user-prompt timestamp yet → skip
if [ ! -f "$LAST_USER_TS_FILE" ]; then
  log "skip — no last_user_prompt.unix"
  exit 0
fi

LAST_USER_TS=$(cat "$LAST_USER_TS_FILE" 2>/dev/null || echo 0)
if [ -z "$LAST_USER_TS" ] || [ "$LAST_USER_TS" -le 0 ] 2>/dev/null; then
  exit 0
fi

# Check mtime of capability_map.json and reflexions.jsonl
CAP_MTIME=0
REFLEX_MTIME=0
if [ -f "$STATE_DIR/capability_map.json" ]; then
  CAP_MTIME=$(stat -f %m "$STATE_DIR/capability_map.json" 2>/dev/null || stat -c %Y "$STATE_DIR/capability_map.json" 2>/dev/null || echo 0)
fi
if [ -f "$STATE_DIR/reflexions.jsonl" ]; then
  REFLEX_MTIME=$(stat -f %m "$STATE_DIR/reflexions.jsonl" 2>/dev/null || stat -c %Y "$STATE_DIR/reflexions.jsonl" 2>/dev/null || echo 0)
fi

UPDATED=0
if [ "$CAP_MTIME" -ge "$LAST_USER_TS" ] 2>/dev/null; then
  UPDATED=1
fi
if [ "$REFLEX_MTIME" -ge "$LAST_USER_TS" ] 2>/dev/null; then
  UPDATED=1
fi

if [ "$UPDATED" -eq 1 ]; then
  log "ok — self-awareness state updated this turn (cap=$CAP_MTIME reflex=$REFLEX_MTIME last_user=$LAST_USER_TS)"
  exit 0
fi

# Inject force-reflect inbox entry
NOW_ISO=$(date -u +%Y-%m-%dT%H:%M:%SZ)
ENTRY=$(NOW="$NOW_ISO" LAST_USER_TS="$LAST_USER_TS" python3 -c '
import json, os
print(json.dumps({
    "ts": os.environ.get("NOW"),
    "source": "gabriel-self-validate",
    "kind": "force_reflect",
    "reason": "no capability_map or reflexions update since last user prompt",
    "last_user_prompt_unix": int(os.environ.get("LAST_USER_TS", "0") or 0),
}))
')

echo "$ENTRY" >> "$INBOX" 2>/dev/null
log "force_reflect injected (cap_mtime=$CAP_MTIME reflex_mtime=$REFLEX_MTIME < last_user=$LAST_USER_TS)"

exit 0
