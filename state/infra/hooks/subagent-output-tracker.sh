#!/usr/bin/env bash
# subagent-output-tracker — PostToolUse observer for Task/Agent spawns.
#
# Purpose: log the size of every sub-agent's returned `tool_response` (or
# `tool_output`) so we have empirical data on output-side token cost.
#
# WHY observe-only:
#   PostToolUse hooks run AFTER the tool's response has already been recorded
#   into the parent's context. There is no documented Claude Code hook field
#   that replaces the recorded tool_response in-place. The honest option is
#   to LOG sizes so future work (e.g. a brief instruction telling helpers to
#   self-compress before returning) can be tuned against real data.
#
# Fail-open: ANY error → exit 0 so we never break a sub-agent return.
#
# Log: AI-Tools/logs/hooks/subagent_output_sizes.log
# Format: ISO8601 | tool_name | output_chars | approx_tokens | session_id

set +e

LOG_DIR="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/logs/hooks"
mkdir -p "$LOG_DIR" 2>/dev/null
LOG_FILE="$LOG_DIR/subagent_output_sizes.log"

INPUT=$(cat 2>/dev/null)
[[ -z "$INPUT" ]] && exit 0

# Use jq if available; fallback to python3
TOOL_NAME=""
OUTPUT=""
SESSION_ID=""

if command -v jq >/dev/null 2>&1; then
    TOOL_NAME=$(printf '%s' "$INPUT" | jq -r '.tool_name // empty' 2>/dev/null)
    SESSION_ID=$(printf '%s' "$INPUT" | jq -r '.session_id // empty' 2>/dev/null)
    # tool_response holds Agent/Task return; tool_output is legacy
    OUTPUT=$(printf '%s' "$INPUT" | jq -r '(.tool_response // .tool_output // "") | tostring' 2>/dev/null)
else
    read -r TOOL_NAME OUTPUT SESSION_ID < <(python3 - <<PYEOF 2>/dev/null
import json, sys
try:
    d = json.loads("""$INPUT""")
    tn = d.get("tool_name", "")
    sid = d.get("session_id", "")
    out = d.get("tool_response", d.get("tool_output", ""))
    if not isinstance(out, str):
        out = json.dumps(out)
    print(tn, len(out), sid)
except Exception:
    print("", "", "")
PYEOF
)
fi

# Only track Agent/Task family
case "$TOOL_NAME" in
    Task|Agent|mcp__plugin_fallback-agent_fallback__Task) ;;
    *) exit 0 ;;
esac

OUT_LEN=${#OUTPUT}
[[ $OUT_LEN -eq 0 ]] && exit 0
APPROX_TOKENS=$(( OUT_LEN / 4 ))

TS=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
SID="${SESSION_ID:-unknown}"
printf '%s | tool=%s | output_chars=%d | approx_tokens=%d | session=%s\n' \
    "$TS" "$TOOL_NAME" "$OUT_LEN" "$APPROX_TOKENS" "$SID" >> "$LOG_FILE"

# Flag exceptionally-large outputs to stderr (operator-visible warn, non-blocking)
if [[ $APPROX_TOKENS -gt 4000 ]]; then
    printf 'subagent-output-tracker: WARN sub-agent returned %d tokens (>4k) — instruct helpers to return digests, not transcripts\n' \
        "$APPROX_TOKENS" >&2
fi

exit 0
