#!/usr/bin/env bash
# model-routing-check/check.sh — PreToolUse hook (NON-BLOCKING, warnings only)
#
# Fires on both native Task and mcp__plugin_fallback-agent_fallback__Task spawns.
# Emits warnings to stderr; always exits 0 so spawns are never blocked.
#
# Parameter shapes handled:
#   plugin Task  → .tool_input.model (explicit), .tool_input.prompt
#   native Task  → .tool_input.subagent_type (agent type), .tool_input.prompt
#   defensive    → .tool_input.agentType (alt naming)
#
# Warning triggers:
#   haiku + heavy keyword  → suggest sonnet/opus
#   opus  + light keyword (no heavy) → suggest haiku
#   missing # model_reason: marker → nudge to add rationale
#
# Heavy keywords: synthesize, synthesis, design, architecture, strategy,
#   complex reasoning, deep refactor, orchestrate, multi-step planning
#
# Light keywords: list, copy, inventory, file scan, grep, summarize,
#   format, rename, quick scan, simple summary

set -u
LC_ALL=C

INPUT=$(cat)

if ! command -v jq >/dev/null 2>&1; then
  echo "[model-routing-check] jq not found — skipping model routing validation" >&2
  exit 0
fi

TOOL_NAME=$(printf '%s' "$INPUT" | jq -r '.tool_name // empty')
PROMPT=$(printf '%s' "$INPUT" | jq -r '.tool_input.prompt // empty')

if [[ -z "$PROMPT" ]]; then
  exit 0
fi

PROMPT_LOWER=$(printf '%s' "$PROMPT" | tr '[:upper:]' '[:lower:]')

# Extract model indicator — try explicit model first (plugin Task),
# then subagent_type (native Task), then agentType (defensive alt name)
MODEL=$(printf '%s' "$INPUT" | jq -r '.tool_input.model // empty' | tr '[:upper:]' '[:lower:]')
AGENT_TYPE=$(printf '%s' "$INPUT" | jq -r '.tool_input.subagent_type // .tool_input.agentType // empty' | tr '[:upper:]' '[:lower:]')

LABEL="${MODEL:-${AGENT_TYPE:-unset}}"

# Skip entirely for native Agent with subagent_type=general-purpose
# (no model selection → nothing to route; UI shows widget; spawns are silent-OK)
if [[ "$AGENT_TYPE" == "general-purpose" ]] && [[ -z "$MODEL" ]]; then
  exit 0
fi

# Check for model_reason marker (only when model/agent explicitly chosen)
if ! printf '%s' "$PROMPT" | grep -q '# model_reason:'; then
  echo "[model-routing-check] WARN: no '# model_reason:' marker in prompt (tool=${TOOL_NAME}, model/agent=${LABEL}) — add rationale per CLAUDE.md" >&2
fi

# Keyword mismatch check only applies when an explicit model tier is known
if [[ -z "$MODEL" ]]; then
  exit 0
fi

# Heavy keyword patterns (signals → sonnet+)
HEAVY_KEYWORDS=(
  "synthesize"
  "synthesis"
  "design"
  "architecture"
  "strategy"
  "complex reasoning"
  "deep refactor"
  "orchestrate"
  "multi-step planning"
)

# Light keyword patterns (signals → haiku)
LIGHT_KEYWORDS=(
  "list "
  "copy "
  "inventory"
  "file scan"
  "grep"
  "summarize"
  "format "
  "rename"
  "quick scan"
  "simple summary"
)

find_first_match() {
  local text="$1"
  shift
  local keywords=("$@")
  for kw in "${keywords[@]}"; do
    if [[ "$text" == *"$kw"* ]]; then
      printf '%s' "$kw"
      return 0
    fi
  done
  return 1
}

HEAVY_MATCH=$(find_first_match "$PROMPT_LOWER" "${HEAVY_KEYWORDS[@]}") || HEAVY_MATCH=""
LIGHT_MATCH=$(find_first_match "$PROMPT_LOWER" "${LIGHT_KEYWORDS[@]}") || LIGHT_MATCH=""

# Rule 1: haiku + heavy keyword
if [[ "$MODEL" == *"haiku"* ]] && [[ -n "$HEAVY_MATCH" ]]; then
  echo "[model-routing-check] WARN: model=haiku but prompt contains \"${HEAVY_MATCH}\" — consider sonnet or opus" >&2
fi

# Rule 2: opus + light keyword with no heavy keyword
if [[ "$MODEL" == *"opus"* ]] && [[ -n "$LIGHT_MATCH" ]] && [[ -z "$HEAVY_MATCH" ]]; then
  echo "[model-routing-check] WARN: model=opus but prompt looks mechanical (matched \"${LIGHT_MATCH}\") — consider haiku" >&2
fi

exit 0
