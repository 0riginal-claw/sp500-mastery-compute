#!/usr/bin/env bash
# cache-control-injector.sh
# Observability hook for Task/sub-agent spawns.
#
# WHY this exists: Bug #29966 in Claude Code hardcodes enablePromptCaching:false
# for all sub-agent spawns (native Task tool). A PreToolUse hook cannot reach the
# internal boolean — it can only mutate prompt text. This hook is therefore
# observe-only: it logs each spawn so we can quantify wasted uncached tokens.
# When #29966 is fixed upstream, this hook will also verify the fix is active.
#
# Reference: https://github.com/anthropics/claude-code/issues/29966
# Assigned: @ashwin-ant | Status: open as of 2026-05-16

LOG_DIR="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/logs"
LOG_FILE="$LOG_DIR/cache_control_hook.log"

# Ensure log dir exists
mkdir -p "$LOG_DIR"

# Read full hook input from stdin
INPUT=$(cat)

TOOL_NAME=$(printf '%s' "$INPUT" | jq -r '.tool_name // "unknown"' 2>/dev/null)
AGENT_NAME=$(printf '%s' "$INPUT" | jq -r '.tool_input.agent_name // "n/a"' 2>/dev/null)
PROMPT=$(printf '%s' "$INPUT" | jq -r '.tool_input.prompt // ""' 2>/dev/null)
PROMPT_LEN=${#PROMPT}

TIMESTAMP=$(date -u '+%Y-%m-%dT%H:%M:%SZ')

# Cache eligibility estimate (Sonnet threshold = 1024 tokens; rough 4 chars/token)
APPROX_TOKENS=$(( PROMPT_LEN / 4 ))
if (( APPROX_TOKENS >= 4096 )); then
    CACHE_ELIGIBLE="YES(opus/haiku-threshold)"
elif (( APPROX_TOKENS >= 1024 )); then
    CACHE_ELIGIBLE="YES(sonnet-threshold)"
else
    CACHE_ELIGIBLE="NO(below-threshold)"
fi

printf '%s | tool=%s | agent=%s | prompt_chars=%d | approx_tokens=%d | cache_eligible=%s | BUG#29966=ACTIVE\n' \
    "$TIMESTAMP" "$TOOL_NAME" "$AGENT_NAME" "$PROMPT_LEN" "$APPROX_TOKENS" "$CACHE_ELIGIBLE" \
    >> "$LOG_FILE"

# Always exit 0 — non-blocking, never stall sub-agent spawns
exit 0
