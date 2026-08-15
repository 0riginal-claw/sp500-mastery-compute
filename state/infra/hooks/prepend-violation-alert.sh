#!/usr/bin/env bash
# prepend-violation-alert — UserPromptSubmit hook
#
# Consumes alerts written by pre-respond-audit.sh on the previous Stop turn.
# If /tmp/cc-violation-alert/<sid>.json exists, read it, emit its contents as
# additionalContext on this UserPromptSubmit (so the orchestrator sees the
# correction BEFORE composing its next response), then delete the file.
#
# Together with auto-solve-violation-detector.sh + periodic-mandate-reminder.sh
# this forms the 3-layer mandate-enforcement stack:
#   1. periodic-mandate-reminder    → re-inject mandates every turn
#   2. pre-respond-audit            → semantic scan on Stop, write alert
#   3. prepend-violation-alert      → inject alert into next user turn
#   4. auto-solve-violation-detector → block Stop on explicit sentinel match
#
# Tunables (env):
#   PREPEND_VIOLATION_ALERT_DISABLE=1 → off
#
# Idempotency: errors → exit 0 silently.

set +e
LC_ALL=C

if [[ "${PREPEND_VIOLATION_ALERT_DISABLE:-0}" == "1" ]]; then exit 0; fi

LOG_DIR="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/logs/prepend_violation_alert"
mkdir -p "$LOG_DIR" 2>/dev/null
LOG_FILE="$LOG_DIR/$(date -u +%Y-%m-%d).log"
TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

ALERT_DIR="/tmp/cc-violation-alert"

PAYLOAD="$(cat 2>/dev/null)"
[[ -z "$PAYLOAD" ]] && exit 0

SID=""
if command -v jq >/dev/null 2>&1; then
  SID=$(printf '%s' "$PAYLOAD" | jq -r '.session_id // empty' 2>/dev/null)
else
  SID=$(printf '%s' "$PAYLOAD" | python3 -c 'import sys,json
try: print(json.load(sys.stdin).get("session_id","") or "")
except Exception: print("")' 2>/dev/null)
fi
[[ -z "$SID" ]] && exit 0

ALERT_FILE="$ALERT_DIR/${SID}.json"
if [[ ! -f "$ALERT_FILE" ]]; then
  exit 0  # no alert pending — silent pass
fi

# Read alert + compose injection text
ALERT_TEXT=$(ALERT_FILE_PATH="$ALERT_FILE" python3 - <<'PYEOF' 2>/dev/null
import json, os, sys
path = os.environ.get("ALERT_FILE_PATH","")
try:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    text = (
        "VIOLATION DETECTED on previous turn (pre-respond-audit).\n"
        f"  pattern:          {d.get('pattern','?')}\n"
        f"  snippet:          {d.get('snippet','')[:300]}\n"
        f"  mandate violated: {d.get('mandate','?')}\n"
        f"  required action:  {d.get('required_action','?')}\n"
        "\n"
        "This injection appears BECAUSE the previous turn narrated a blocker "
        "or punted to the user without spawning solver helpers. Do NOT ack "
        "this message to the user — silently spawn the 3 helpers NOW and "
        "continue the work. Log to AI-Tools/logs/auto_solve/<topic>_<UTC>.md."
    )
    print(text)
except Exception:
    sys.exit(0)
PYEOF
)

if [[ -z "$ALERT_TEXT" ]]; then
  echo "$TS sid=$SID alert file unreadable -> delete + skip" >> "$LOG_FILE" 2>/dev/null
  rm -f "$ALERT_FILE" 2>/dev/null
  exit 0
fi

echo "$TS sid=$SID INJECT alert (${#ALERT_TEXT}c)" >> "$LOG_FILE" 2>/dev/null

# Emit additionalContext, then delete alert file (one-shot)
export ALERT_TEXT
python3 - <<'PYEOF' 2>/dev/null
import json, os
ctx = os.environ.get("ALERT_TEXT","")
if ctx:
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": ctx
        }
    }))
PYEOF
unset ALERT_TEXT

rm -f "$ALERT_FILE" 2>/dev/null
exit 0
