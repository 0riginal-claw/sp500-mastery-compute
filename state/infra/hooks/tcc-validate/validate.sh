#!/usr/bin/env bash
# tcc-validate - Stop guardrail (Hook 5 of 5)
#
# At end of every assistant turn, scan logs/tcc_autoallow_audit.jsonl for any
# status=stuck entries since the last user prompt. "Stuck" means the
# Hammerspoon watcher or PostToolUse hook saw a whitelisted TCC dialog but
# couldn't find the Allow button (or the click failed).
#
# Recovery action: append an entry to the autonomous_mode user_inbox so the
# next daemon cycle investigates, and write a dashboard alert flag.
#
# Always exits 0. Pure-signal.

set +e
LC_ALL=C

ROOT="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools"
AUDIT_FILE="$ROOT/logs/tcc_autoallow_audit.jsonl"
INBOX="$ROOT/state/autonomous_mode/user_inbox.jsonl"
DASHBOARD_ALERT="$ROOT/dashboard/tcc_autoallow_alerts.jsonl"
LAST_USER_TS_FILE="$HOME/.claude/state/last_user_prompt.unix"
LOG_DIR="$ROOT/logs/auto_solve"
LOG_FILE="$LOG_DIR/tcc_guardrails.log"

mkdir -p "$LOG_DIR" 2>/dev/null
mkdir -p "$(dirname "$INBOX")" 2>/dev/null
mkdir -p "$(dirname "$DASHBOARD_ALERT")" 2>/dev/null

cat >/dev/null 2>&1 || true

log() {
  printf '[%s] tcc-validate: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$LOG_FILE" 2>/dev/null
}

if [ ! -f "$AUDIT_FILE" ]; then
  exit 0
fi

# Determine last user prompt timestamp (unix seconds)
LAST_USER_TS=0
if [ -f "$LAST_USER_TS_FILE" ]; then
  LAST_USER_TS=$(cat "$LAST_USER_TS_FILE" 2>/dev/null || echo 0)
fi
if [ -z "$LAST_USER_TS" ] || [ "$LAST_USER_TS" -le 0 ] 2>/dev/null; then
  LAST_USER_TS=$(( $(date +%s) - 600 ))  # default look-back 10min
fi

# Count stuck entries since LAST_USER_TS (audit ISO timestamps to unix)
STUCK_INFO=$(LAST_USER_TS="$LAST_USER_TS" AUDIT="$AUDIT_FILE" python3 -c '
import json, os, sys
from datetime import datetime, timezone
try:
    last = int(os.environ.get("LAST_USER_TS", "0"))
except (ValueError, TypeError):
    last = 0
stuck_count = 0
stuck_titles = []
path = os.environ.get("AUDIT", "")
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
            if row.get("status") != "stuck":
                continue
            ts = row.get("ts", "")
            try:
                s = ts.replace("Z", "+00:00") if isinstance(ts, str) else ""
                row_unix = int(datetime.fromisoformat(s).timestamp()) if s else 0
            except (ValueError, TypeError):
                row_unix = 0
            if row_unix >= last:
                stuck_count += 1
                title = row.get("title", "")
                if isinstance(title, str) and title and len(stuck_titles) < 5:
                    stuck_titles.append(title)
except OSError:
    pass
print(json.dumps({"count": stuck_count, "titles": stuck_titles}))
' 2>/dev/null)

if [ -z "$STUCK_INFO" ]; then
  exit 0
fi

STUCK_COUNT=$(echo "$STUCK_INFO" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("count",0))' 2>/dev/null || echo 0)
if [ -z "$STUCK_COUNT" ] || [ "$STUCK_COUNT" = "0" ] 2>/dev/null; then
  exit 0
fi

NOW_ISO=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# Append to inbox for autonomous daemon to investigate
INBOX_ENTRY=$(NOW="$NOW_ISO" STUCK_COUNT="$STUCK_COUNT" STUCK_INFO="$STUCK_INFO" python3 -c '
import json, os
info = json.loads(os.environ.get("STUCK_INFO", "{}"))
print(json.dumps({
    "ts": os.environ.get("NOW"),
    "source": "tcc-validate",
    "kind": "tcc_dialog_stuck",
    "reason": f"{os.environ.get(\"STUCK_COUNT\")} TCC dialog(s) stuck since last user prompt",
    "stuck_count": int(os.environ.get("STUCK_COUNT", "0")),
    "titles": info.get("titles", []),
    "priority": "high",
}))
')

echo "$INBOX_ENTRY" >> "$INBOX" 2>/dev/null

# Dashboard alert
echo "$INBOX_ENTRY" >> "$DASHBOARD_ALERT" 2>/dev/null

log "ALERT: $STUCK_COUNT stuck TCC dialog(s) since last user prompt -> inbox + dashboard"
exit 0
