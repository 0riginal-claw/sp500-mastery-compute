#!/usr/bin/env bash
# karpathy-guidelines-block/check.sh
# PreToolUse hook for: Task | Agent | mcp__plugin_fallback-agent_fallback__Task | Bash
#
# Enforces the karpathy-guidelines pre-flight rule from CLAUDE.md (§karpathy).
# Any ML / quant / backtest / hyperparam / signal-gen / live-paper-trade work
# MUST surface assumptions + verifiable success criteria BEFORE execution.
#
# BLOCK when:
#   spawn prompt OR Bash command contains any of:
#     backtest_xgb | alpha158 | qlib | mythos | train_model |
#     fit_classifier | hyperparam | threshold.sweep |
#     walk.forward | signal_gen | live_paper_trade_signals
#   AND NO marker '# karpathy_checked: <summary>' present
#   AND NOT a smoke/test invocation (smoke|test tokens)
#
# Escape hatches:
#   * Add "# karpathy_checked: <summary>" header to spawn prompt / command.
#   * Include "smoke" or "test" token in prompt/command (smoke runs allowed).
#
# Exit 2 on block (per Claude Code hook protocol).
# Exit 0 otherwise.

set -u
LC_ALL=C

INPUT=$(cat)
if ! command -v jq >/dev/null 2>&1; then
  echo "karpathy-guidelines-block: jq missing, allow-open" >&2
  exit 0
fi

TOOL_NAME=$(printf '%s' "$INPUT" | jq -r '.tool_name // empty')
DESC=$(printf '%s'    "$INPUT" | jq -r '.tool_input.description // empty')
PROMPT=$(printf '%s'  "$INPUT" | jq -r '.tool_input.prompt      // empty')
SUBTYPE=$(printf '%s' "$INPUT" | jq -r '.tool_input.subagent_type // empty')
CMD=$(printf '%s'     "$INPUT" | jq -r '.tool_input.command     // empty')
BLOB="${DESC}
${PROMPT}
${SUBTYPE}
${CMD}"

# Only fire on the four tools.
case "$TOOL_NAME" in
  Task|Agent|mcp__plugin_fallback-agent_fallback__Task|Bash) ;;
  *) exit 0 ;;
esac

# ML / quant trigger words.
TRIGGER_RE='backtest_xgb|alpha158|qlib|mythos|train_model|fit_classifier|hyperparam|threshold\.sweep|walk\.forward|signal_gen|live_paper_trade_signals'

if ! printf '%s' "$BLOB" | grep -qEi "$TRIGGER_RE"; then
  exit 0
fi

# Explicit pre-flight marker.
if printf '%s' "$BLOB" | grep -qE '# *karpathy_checked:'; then
  LOG="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/logs/auto_solve/karpathy_checked.log"
  { date -u +'%Y-%m-%dT%H:%M:%SZ'; echo "PASS karpathy_checked: $(printf '%s' "$BLOB" | grep -oE '# *karpathy_checked:[^[:cntrl:]]*' | head -1)"; } >> "$LOG" 2>/dev/null
  exit 0
fi

# Smoke / test allowance.
if printf '%s' "$BLOB" | grep -qEi '\bsmoke\b|\btest\b'; then
  exit 0
fi

cat >&2 <<'EOF'
BLOCKED by karpathy-guidelines-block hook.

ML / quant / backtest / hyperparam / signal-gen work MUST complete the
karpathy-guidelines pre-flight (surface assumptions + verifiable success
criteria) before execution. See CLAUDE.md §karpathy.

To proceed:
  1. Invoke the `karpathy-guidelines` skill, OR
  2. Add a "# karpathy_checked: <one-line summary of assumptions + success criteria>"
     header near the top of the spawn prompt / Bash command.

Escape hatches (only when truly applicable):
  - Include "smoke" or "test" token (e.g. smoke run, smoke test) — allowed.
  - Add "# karpathy_checked: <reason>" header — explicit pre-flight done.
EOF
exit 2
