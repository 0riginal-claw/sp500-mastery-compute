#!/usr/bin/env bash
# recursion-fanout-tracker — §3 enforcement hook (HARDENED 2026-05-19)
#
# Tracks per-session elapsed time and Agent/Task spawn count. Fires on
# PreToolUse for Bash/Edit/Write/MultiEdit/NotebookEdit/Task tools.
#
# HARD-BLOCK ENFORCEMENT (upgraded from warnings):
#   <5 min               → silent pass
#   >=5 min  + spawns<2  → BLOCK (exit 2) — must spawn >=2 grandchildren
#   >=10 min + spawns<4  → BLOCK (exit 2) — must spawn >=4 grandchildren total
#   >=15 min + spawns<6  → BLOCK (exit 2) — must spawn >=6 OR self-terminate
#   >=20 min + spawns<6  → KILL  (exit 2) — protocol violation, terminate
#   spawn threshold met  → silent pass (mandate satisfied for that tier)
#
# ESCAPE HATCH: if the current tool_input prompt/command contains the marker
#   # fanout_skip: <reason>
# the hook passes silently. Use sparingly and document the reason.
#
# Non-blocking failure mode: any internal hook error exits 0 to avoid
# breaking helpers if jq/python missing or paths broken.
#
# Input: JSON stdin from Claude Code with tool_name + tool_input + session_id.

set +e  # never abort on errors — we must not break helpers

TRACKER_DIR="/tmp/cc-recursion-tracker"
mkdir -p "$TRACKER_DIR" 2>/dev/null

# Read full hook payload from stdin
PAYLOAD="$(cat 2>/dev/null)"
if [[ -z "$PAYLOAD" ]]; then exit 0; fi

# Extract fields (jq optional — fall back to python)
SID=""
TOOL=""
TOOL_INPUT_BLOB=""
if command -v jq >/dev/null 2>&1; then
  SID=$(printf '%s' "$PAYLOAD" | jq -r '.session_id // empty' 2>/dev/null)
  TOOL=$(printf '%s' "$PAYLOAD" | jq -r '.tool_name // empty' 2>/dev/null)
  # Flatten tool_input to a searchable blob (prompt, command, content, new_string, etc.)
  TOOL_INPUT_BLOB=$(printf '%s' "$PAYLOAD" | jq -r '.tool_input // {} | [.. | strings] | join(" ")' 2>/dev/null)
else
  SID=$(printf '%s' "$PAYLOAD" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("session_id",""))' 2>/dev/null)
  TOOL=$(printf '%s' "$PAYLOAD" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("tool_name",""))' 2>/dev/null)
  TOOL_INPUT_BLOB=$(printf '%s' "$PAYLOAD" | python3 -c '
import sys, json
def walk(x):
    if isinstance(x, str): yield x
    elif isinstance(x, dict):
        for v in x.values(): yield from walk(v)
    elif isinstance(x, list):
        for v in x: yield from walk(v)
try:
    d = json.load(sys.stdin).get("tool_input", {})
    print(" ".join(walk(d)))
except Exception:
    pass
' 2>/dev/null)
fi

if [[ -z "$SID" ]]; then exit 0; fi

START_FILE="$TRACKER_DIR/$SID.start"
SPAWNS_FILE="$TRACKER_DIR/$SID.spawns"
LAST_WARN_FILE="$TRACKER_DIR/$SID.lastwarn"

# Record start time if missing
NOW=$(date +%s)
if [[ ! -f "$START_FILE" ]]; then
  echo "$NOW" > "$START_FILE"
fi

START=$(cat "$START_FILE" 2>/dev/null || echo "$NOW")
ELAPSED=$(( NOW - START ))
ELAPSED_MIN=$(( ELAPSED / 60 ))

# Count spawn if this is a Task/Agent/MCP_Task call
case "$TOOL" in
  Task|Agent|mcp__plugin_fallback-agent_fallback__Task)
    SPAWNS=0
    if [[ -f "$SPAWNS_FILE" ]]; then SPAWNS=$(cat "$SPAWNS_FILE"); fi
    SPAWNS=$(( SPAWNS + 1 ))
    echo "$SPAWNS" > "$SPAWNS_FILE"
    ;;
esac

SPAWNS=0
if [[ -f "$SPAWNS_FILE" ]]; then SPAWNS=$(cat "$SPAWNS_FILE"); fi

# Decision tree — fast-path silent pass for fresh agents
if (( ELAPSED < 300 )); then
  exit 0  # <5 min, silent
fi

# ESCAPE HATCH: caller explicitly tagged this tool call with a fanout_skip reason.
# Marker must appear in tool_input text (prompt/command/content). Pattern is
# tolerant: matches `# fanout_skip:` with any reason text after.
if printf '%s' "$TOOL_INPUT_BLOB" | grep -q -E '#[[:space:]]*fanout_skip:'; then
  exit 0
fi

# Required spawn count by tier
REQUIRED=0
TIER=""
if (( ELAPSED >= 1200 )); then
  REQUIRED=6
  TIER="KILL"
elif (( ELAPSED >= 900 )); then
  REQUIRED=6
  TIER="T15"
elif (( ELAPSED >= 600 )); then
  REQUIRED=4
  TIER="T10"
elif (( ELAPSED >= 300 )); then
  REQUIRED=2
  TIER="T5"
fi

# Mandate satisfied for current tier — silent pass
if (( SPAWNS >= REQUIRED )); then
  exit 0
fi

# BLOCK with tier-specific stderr message
case "$TIER" in
  KILL)
    cat >&2 <<EOF
PROTOCOL VIOLATION — §3 KILL CONDITION (T+${ELAPSED_MIN}min)

You have ${SPAWNS} sub-agent spawns at ${ELAPSED_MIN} min wall-clock.
The §3 mandate requires >=6 spawns by T+15 and absolutely bans solo work
past T+20.

REQUIRED ACTION NOW (one of):
  (A) Spawn helpers to reach >=6 total via parallel 'Agent' (general-purpose)
      calls in a single message. Become the aggregator.
  (B) Self-terminate: write a partial report covering completed work and exit.
  (C) Add '# fanout_skip: <one-line reason>' to your next tool call IF the
      remaining work is genuinely <2 min and indivisible.

This tool call is BLOCKED. session_id=${SID} elapsed_min=${ELAPSED_MIN} spawns=${SPAWNS} required=${REQUIRED}
EOF
    exit 2
    ;;
  T15)
    cat >&2 <<EOF
BLOCKED §3 (T+${ELAPSED_MIN}min): sub-agent has ${SPAWNS} spawns; need >=6
by T+15 OR self-terminate with partial report. Spawn the remaining helpers
in parallel (single message) or add '# fanout_skip: <reason>' to proceed.
session=${SID} spawns=${SPAWNS} required=6
EOF
    exit 2
    ;;
  T10)
    cat >&2 <<EOF
BLOCKED §3 (T+${ELAPSED_MIN}min): sub-agent at 10min wall-clock has ${SPAWNS}
spawns; must reach >=4 grandchildren on remaining slices. Decompose now and
spawn 2-6 more 'Agent' (general-purpose) helpers in parallel, or add
'# fanout_skip: <reason>' to current tool prompt.
session=${SID} spawns=${SPAWNS} required=4
EOF
    exit 2
    ;;
  T5)
    cat >&2 <<EOF
BLOCKED §3: sub-agent at ${ELAPSED_MIN}min wall-clock must spawn >=2
grandchildren on remaining slices. Decompose now or add
'# fanout_skip: <reason>' to current tool prompt.
session=${SID} spawns=${SPAWNS} required=2
EOF
    exit 2
    ;;
esac

exit 0
