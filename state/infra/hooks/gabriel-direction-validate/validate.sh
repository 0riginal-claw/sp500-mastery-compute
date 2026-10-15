#!/usr/bin/env bash
# gabriel-direction-validate — Stop guardrail (Hook 6 of 6)
#
# At end of every assistant turn, check that goal_tree.json has been
# updated within the last hour. If not, append an inbox entry forcing
# the autonomous-mode daemon to walk the goal tree and generate next
# next-action proposals (plan-without-direction module).

set +e
LC_ALL=C

ROOT="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools"
STATE_DIR="$ROOT/state/gabriel_self"
INBOX="$ROOT/state/autonomous_mode/user_inbox.jsonl"
LOG_DIR="$ROOT/logs/gabriel_self"
LOG_FILE="$LOG_DIR/direction_validate.log"
STALE_SECS=3600   # 1 hour

mkdir -p "$STATE_DIR" "$LOG_DIR" "$(dirname "$INBOX")" 2>/dev/null

# Drain stdin
cat >/dev/null 2>&1 || true

log() {
  printf '[%s] gabriel-direction-validate: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$LOG_FILE" 2>/dev/null
}

GT_FILE="$STATE_DIR/goal_tree.json"
NOW=$(date +%s)

GT_MTIME=0
if [ -f "$GT_FILE" ]; then
  GT_MTIME=$(stat -f %m "$GT_FILE" 2>/dev/null || stat -c %Y "$GT_FILE" 2>/dev/null || echo 0)
fi

if [ "$GT_MTIME" -eq 0 ] 2>/dev/null; then
  log "skip — goal_tree.json missing (bootstrap should create it)"
  exit 0
fi

AGE=$((NOW - GT_MTIME))

if [ "$AGE" -le "$STALE_SECS" ] 2>/dev/null; then
  log "ok — goal_tree.json fresh (age=${AGE}s)"
  exit 0
fi

NOW_ISO=$(date -u +%Y-%m-%dT%H:%M:%SZ)
ENTRY=$(NOW="$NOW_ISO" AGE="$AGE" python3 -c '
import json, os
print(json.dumps({
    "ts": os.environ.get("NOW"),
    "source": "gabriel-direction-validate",
    "kind": "force_plan",
    "reason": "goal_tree.json stale (no direction update in >1h)",
    "stale_seconds": int(os.environ.get("AGE", "0") or 0),
}))
')

echo "$ENTRY" >> "$INBOX" 2>/dev/null
log "force_plan injected (goal_tree stale ${AGE}s)"

exit 0
