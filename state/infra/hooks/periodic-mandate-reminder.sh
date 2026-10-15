#!/usr/bin/env bash
# periodic-mandate-reminder — UserPromptSubmit hook
#
# Re-injects the 5 active mandates into the model context every Nth user turn.
# This counteracts mid-session "mandate drift" where the orchestrator stops
# applying §3 / §5a / §8 / auto-execute / no-restart / cloud-routing as the
# context window fills with non-CLAUDE.md material.
#
# Mechanism:
#   - Per-session counter at /tmp/cc-mandate-reminder/<sid>.count
#   - Every TURN where (count % REMINDER_EVERY_N == 0), emit a JSON
#     hookSpecificOutput.additionalContext with a compact mandate header.
#   - The reminder is prepended to the model's next-step context, not the user
#     prompt, so the user never sees it.
#
# Tunables (env):
#   MANDATE_REMINDER_EVERY_N=1     → fire every Nth turn (default 1 — every turn,
#                                    updated 2026-05-17 after audit showed every-5
#                                    cadence let mandates drift between reminders)
#   MANDATE_REMINDER_DISABLE=1     → off
#   MANDATE_REMINDER_FIRST_TURN=1  → fire on turn #1 (default ON now — first turn
#                                    is the highest-value injection)
#
# Cost: ~80 tokens × every turn ≈ 0.5% of a 16k-context turn. Worth it (audit:
# mandate violations clustered at turns 3-4 within the 5-turn skip window).
#
# Idempotency: errors → exit 0 silently.

set +e
LC_ALL=C

if [[ "${MANDATE_REMINDER_DISABLE:-0}" == "1" ]]; then exit 0; fi

EVERY_N="${MANDATE_REMINDER_EVERY_N:-1}"
if ! [[ "$EVERY_N" =~ ^[0-9]+$ ]] || (( EVERY_N < 1 )); then EVERY_N=1; fi
# Default first-turn fire to ON (the first reminder is the cheapest insurance)
: "${MANDATE_REMINDER_FIRST_TURN:=1}"

STATE_DIR="/tmp/cc-mandate-reminder"
mkdir -p "$STATE_DIR" 2>/dev/null

LOG_DIR="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/logs/mandate_reminder"
mkdir -p "$LOG_DIR" 2>/dev/null
LOG_FILE="$LOG_DIR/$(date -u +%Y-%m-%d).log"
TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

PAYLOAD="$(cat 2>/dev/null)"
if [[ -z "$PAYLOAD" ]]; then exit 0; fi

SID=""
if command -v jq >/dev/null 2>&1; then
  SID=$(printf '%s' "$PAYLOAD" | jq -r '.session_id // empty' 2>/dev/null)
else
  SID=$(printf '%s' "$PAYLOAD" | python3 -c 'import sys,json
try: print(json.load(sys.stdin).get("session_id","") or "")
except Exception: print("")' 2>/dev/null)
fi
if [[ -z "$SID" ]]; then exit 0; fi

COUNT_FILE="$STATE_DIR/${SID}.count"
COUNT=0
if [[ -f "$COUNT_FILE" ]]; then COUNT=$(cat "$COUNT_FILE" 2>/dev/null || echo 0); fi
COUNT=$(( COUNT + 1 ))
echo "$COUNT" > "$COUNT_FILE"

# Decide whether to fire
FIRE=0
if (( COUNT == 1 )); then
  if [[ "${MANDATE_REMINDER_FIRST_TURN:-0}" == "1" ]]; then FIRE=1; fi
elif (( COUNT % EVERY_N == 0 )); then
  FIRE=1
fi

if (( FIRE == 0 )); then
  echo "$TS sid=$SID turn=$COUNT skip" >> "$LOG_FILE" 2>/dev/null
  exit 0
fi

echo "$TS sid=$SID turn=$COUNT FIRE" >> "$LOG_FILE" 2>/dev/null

# Compact mandate reminder. Kept under ~90 tokens; injected every turn since
# 2026-05-17 (audit found mandate violations clustered between every-5 fires).
# Stop hook now blocks ~100 sentinel/regex patterns including: manual token paste,
# needs first-deploy, skip <X> for now, blocked on <X> requirement, you/user/
# operator must|need|should|can <verb>, paste/visit dashboard, etc.
MANDATES="MANDATES STILL ACTIVE (turn $COUNT, every $EVERY_N turn):
  §3 fan-out         — est/elapsed >5min & >2 slices → spawn N=2-6 helpers (general-purpose). 20-min solo = KILL.
  §5a cloud-routing  — heavy compute → cloud_dispatch.enqueue_job(). NEVER AUTO_CLOUD_DISPATCH=0 unless smoke <60s. Mac load cap=12.
  §8 auto-solve      — ANY error/blocker/manual-step → spawn 3 parallel solvers (INTERNET/GITHUB/REPO-LOCAL), apply silently. NEVER ask user, NEVER punt to user/operator/human, NEVER 'skip X for now'.
  auto-execute       — never 'would you like X' / 'say go' / 'you'll need to paste' / 'user can run' / 'operator should' / 'manual token paste' / 'needs first-deploy' / 'blocked on X requirement'. Stop hook BLOCKS these.
  no-restart         — never suggest 'restart Claude Code'. Use /reload-plugins, sub-agent spawn, or SIGHUP wrapper."

# Emit JSON additionalContext for UserPromptSubmit. Pass MANDATES via env so
# Python doesn't have to shell-quote-safely escape backslashes / quotes / $.
export MANDATES_TEXT="$MANDATES"
python3 - <<'PYEOF' 2>/dev/null
import json, os, sys
ctx = os.environ.get("MANDATES_TEXT", "")
if not ctx:
    sys.exit(0)
print(json.dumps({
    "hookSpecificOutput": {
        "hookEventName": "UserPromptSubmit",
        "additionalContext": ctx
    }
}))
PYEOF
unset MANDATES_TEXT

exit 0
