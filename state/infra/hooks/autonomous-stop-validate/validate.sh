#!/usr/bin/env bash
# autonomous-stop-validate — Stop guardrail (Layer 6)
#
# At end of every assistant turn, validate that the autonomous daemon has
# produced >= 1 cycle since the last user prompt. If 0 cycles, write a
# user_inbox.jsonl entry to force an ideate cycle.
#
# This protects against the silent-failure mode where the daemon process is
# alive but its loop is wedged (deadlock, semaphore leak, hung subprocess).
#
# Layer 6 of 6 in the autonomous-mode guardrail chain.

set +e
LC_ALL=C

ROOT="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools"
INBOX="$ROOT/state/autonomous_mode/user_inbox.jsonl"
LAST_USER_TS_FILE="$HOME/.claude/state/last_user_prompt.unix"
LOG_DIR="$ROOT/logs/auto_solve"
LOG_FILE="$LOG_DIR/autonomous_guardrails.log"

mkdir -p "$LOG_DIR" 2>/dev/null
mkdir -p "$(dirname "$INBOX")" 2>/dev/null

# Drain stdin
cat >/dev/null 2>&1 || true

log() {
  printf '[%s] autonomous-stop-validate: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$LOG_FILE" 2>/dev/null
}

# Determine the last user prompt timestamp; if unavailable, skip the check
# (we don't want to spam inbox at session start).
if [ ! -f "$LAST_USER_TS_FILE" ]; then
  log "skip — no last_user_prompt.unix yet"
  exit 0
fi

LAST_USER_TS=$(cat "$LAST_USER_TS_FILE" 2>/dev/null || echo 0)
if [ -z "$LAST_USER_TS" ] || [ "$LAST_USER_TS" -le 0 ] 2>/dev/null; then
  exit 0
fi

# Today's audit log
TODAY=$(date -u +%Y-%m-%d)
AUDIT="$ROOT/state/autonomous_mode/audit_${TODAY}.jsonl"

CYCLES_SINCE=0
if [ -f "$AUDIT" ]; then
  # Count audit lines whose ts (ISO8601 or unix) is >= last_user_ts.
  # The audit file uses ISO8601; convert with python for robustness.
  CYCLES_SINCE=$(LAST_USER_TS="$LAST_USER_TS" AUDIT="$AUDIT" python3 -c '
import json, os, sys
from datetime import datetime, timezone
try:
    last = int(os.environ.get("LAST_USER_TS", "0"))
except (ValueError, TypeError):
    last = 0
path = os.environ.get("AUDIT", "")
count = 0
try:
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = row.get("ts") or row.get("timestamp")
            row_unix = None
            if isinstance(ts, (int, float)):
                row_unix = int(ts)
            elif isinstance(ts, str):
                try:
                    # Parse ISO8601 (with or without Z)
                    s = ts.replace("Z", "+00:00")
                    row_unix = int(datetime.fromisoformat(s).timestamp())
                except (ValueError, TypeError):
                    pass
            if row_unix is None:
                continue
            if row_unix >= last and row.get("event") in (None, "cycle_start", "iteration", "decision", "spawn", "react"):
                count += 1
except OSError:
    pass
print(count)
' 2>/dev/null || echo 0)
fi

if [ -z "$CYCLES_SINCE" ]; then
  CYCLES_SINCE=0
fi

if [ "$CYCLES_SINCE" -gt 0 ] 2>/dev/null; then
  log "ok — $CYCLES_SINCE cycle(s) since last user prompt at $LAST_USER_TS"
  exit 0
fi

# 0 cycles — force an ideate by writing an inbox entry.
NOW_ISO=$(date -u +%Y-%m-%dT%H:%M:%SZ)
ENTRY=$(NOW="$NOW_ISO" LAST_USER_TS="$LAST_USER_TS" python3 -c '
import json, os, sys
print(json.dumps({
    "ts": os.environ.get("NOW"),
    "source": "autonomous-stop-validate",
    "kind": "force_ideate",
    "reason": "0 cycles since last_user_prompt (potential daemon wedge)",
    "last_user_prompt_unix": int(os.environ.get("LAST_USER_TS", "0") or 0),
}))
')

# Append atomically (small line, single write is atomic on POSIX <4KB).
echo "$ENTRY" >> "$INBOX" 2>/dev/null

log "force_ideate injected (0 cycles since user prompt)"

exit 0
