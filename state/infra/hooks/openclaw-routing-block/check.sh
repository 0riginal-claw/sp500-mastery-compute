#!/usr/bin/env bash
# openclaw-routing-block/check.sh
# PreToolUse hook for: Task | Agent | mcp__plugin_fallback-agent_fallback__Task
#
# Enforces the OpenClaw+DeepSeek routing rule from
#   memory/feedback_openclaw_deepseek_routing.md
# All non-Alpaca research/synthesis/audit work MUST go via:
#   $HOME/.../AI-Tools/bin/openclaw-gdrive agent --local --model deepseek/...
#
# BLOCK when:
#   spawn description/prompt contains any of:
#     research | synthesize | audit | survey | literature | bucket
#     topic helper | INTERNET | GITHUB | WebSearch
#   AND prompt does NOT contain any of (Alpaca-trading allowlist):
#     alpaca | live_paper_trade | halt | signal_gen | reconcile
#     snapshot | fills | mythos_features.py
#
# Escape hatches:
#   * Include "alpaca" (or any allowlist token) in the prompt — Alpaca live state work.
#   * Include "# justify_claude: <reason>" header — explicit override (logged).
#
# Exit 2 on block (per Claude Code hook protocol).
# Exit 0 otherwise.

set -u
LC_ALL=C

INPUT=$(cat)
if ! command -v jq >/dev/null 2>&1; then
  echo "openclaw-routing-block: jq missing, allow-open" >&2
  exit 0
fi

TOOL_NAME=$(printf '%s' "$INPUT" | jq -r '.tool_name // empty')
DESC=$(printf '%s'    "$INPUT" | jq -r '.tool_input.description // empty')
PROMPT=$(printf '%s'  "$INPUT" | jq -r '.tool_input.prompt      // empty')
SUBTYPE=$(printf '%s' "$INPUT" | jq -r '.tool_input.subagent_type // empty')
BLOB="${DESC}
${PROMPT}
${SUBTYPE}"

# Only fire on the three spawn tools.
case "$TOOL_NAME" in
  Task|Agent|mcp__plugin_fallback-agent_fallback__Task) ;;
  *) exit 0 ;;
esac

# autosolve_skip: hook patch, bounded build
# Explicit override.
if printf '%s' "$BLOB" | grep -qE '# *justify_claude:'; then
  LOG="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/logs/auto_solve/openclaw_routing_overrides.log"
  { date -u +'%Y-%m-%dT%H:%M:%SZ'; echo "OVERRIDE justify_claude: $(printf '%s' "$BLOB" | grep -oE '# *justify_claude:[^[:cntrl:]]*' | head -1)"; } >> "$LOG" 2>/dev/null
  exit 0
fi

# Auto-escalation override: Claude spawn allowed IF prompt contains
#   # openclaw_failed: <task_id>
# AND failure counter for task_id is >= 3 in
#   ~/.claude/state/openclaw_fail_counter.jsonl
FAIL_MARKER=$(printf '%s' "$BLOB" | grep -oE '# *openclaw_failed:[[:space:]]*[A-Za-z0-9_.:\-]+' | head -1)
if [ -n "$FAIL_MARKER" ]; then
  FAIL_TASK_ID=$(printf '%s' "$FAIL_MARKER" | sed -E 's/.*openclaw_failed:[[:space:]]*//' | tr -d '[:space:]')
  STATE_FILE="${HOME}/.claude/state/openclaw_fail_counter.jsonl"
  LAST=""
  if [ -s "$STATE_FILE" ]; then
    LAST=$(grep -E "\"task_id\":[[:space:]]*\"${FAIL_TASK_ID}\"" "$STATE_FILE" 2>/dev/null | tail -n 1)
  fi
  COUNT=0
  if [ -n "$LAST" ]; then
    COUNT=$(printf '%s' "$LAST" | sed -E 's/.*"count":[[:space:]]*([0-9]+).*/\1/')
  fi
  if [ "${COUNT:-0}" -ge 3 ]; then
    LOG="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/logs/auto_solve/openclaw_routing_overrides.log"
    { date -u +'%Y-%m-%dT%H:%M:%SZ'; echo "OVERRIDE openclaw_failed: task=${FAIL_TASK_ID} count=${COUNT}"; } >> "$LOG" 2>/dev/null
    exit 0
  else
    echo "openclaw-routing-block: openclaw_failed marker present (task=${FAIL_TASK_ID}) but counter=${COUNT} < 3 -- not allowed" >&2
  fi
fi

# Allowlist (Alpaca-trading work stays on Claude).
ALLOW_RE='alpaca|live_paper_trade|halt|signal_gen|reconcile|snapshot|fills|mythos_features\.py'
if printf '%s' "$BLOB" | grep -qEi "$ALLOW_RE"; then
  exit 0
fi

# Auto-fallback: if OC smoke has failed 3+ consecutive times, allow Claude.
# Smoke results logged to ~/.claude/state/oc_smoke_failures.jsonl by the
# session-start smoke runner (one JSON line per smoke attempt: {ts, ok}).
# Counts CONSECUTIVE trailing failures (resets on first ok found from tail).
SMOKE_FILE="${HOME}/.claude/state/oc_smoke_failures.jsonl"
if [ -s "$SMOKE_FILE" ]; then
  # Last 10 lines, walk from newest; count consecutive ok=false until ok=true seen.
  CONSEC_FAILS=$(tail -n 10 "$SMOKE_FILE" 2>/dev/null | awk '
    {
      # parse ok field (boolean, no quotes)
      if (match($0, /"ok":[[:space:]]*(true|false)/, arr)) {
        lines[NR] = arr[1]
      }
    }
    END {
      # walk from end
      c = 0
      for (i = NR; i >= 1; i--) {
        if (lines[i] == "false") c++
        else if (lines[i] == "true") break
      }
      print c
    }
  ')
  if [ "${CONSEC_FAILS:-0}" -ge 3 ]; then
    LOG="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/logs/auto_solve/openclaw_routing_overrides.log"
    { date -u +'%Y-%m-%dT%H:%M:%SZ'; echo "OVERRIDE oc_smoke_failures: consec=${CONSEC_FAILS} (auto-fallback to Claude)"; } >> "$LOG" 2>/dev/null
    exit 0
  fi
fi

# Trigger words: research / synthesis / audit / etc.
TRIGGER_RE='research|synthesi[sz]e|audit|survey|literature|bucket|topic[[:space:]]+helper|INTERNET|GITHUB|WebSearch'
if printf '%s' "$BLOB" | grep -qE "$TRIGGER_RE"; then
  cat >&2 <<'EOF'
BLOCKED by openclaw-routing-block hook.

Non-Alpaca research / synthesis / audit / survey / literature / bucket / WebSearch
work MUST route via OpenClaw + DeepSeek (see memory/feedback_openclaw_deepseek_routing.md).

Use instead:
  "$HOME/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/bin/openclaw-gdrive" \
    agent --local --model deepseek/deepseek-v4-flash --json \
    --message "<task description>"

Escape hatches (only when justified):
  - Include an Alpaca token in the prompt (alpaca, live_paper_trade, halt,
    signal_gen, reconcile, snapshot, fills, mythos_features.py).
  - Add "# justify_claude: <reason>" header to the spawn prompt.
EOF
  exit 2
fi

exit 0
