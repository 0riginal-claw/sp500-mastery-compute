#!/usr/bin/env bash
# model-reason-block/check.sh
# autosolve_skip: hook patch (bounded build, no error)
# PreToolUse hook for: Task | Agent | mcp__plugin_fallback-agent_fallback__Task
#
# Enforces AGENT_BRIEF_TEMPLATE §5 + CLAUDE.md hard-rule 2026-05-16.
#
# BLOCK when:
#   spawn prompt length > 500 chars
#   AND prompt lacks "# model_reason:" / "# scope_estimate_min:" / "# inline_justification:"
#
# Special path (added 2026-05-19):
#   if "# openclaw_failed:" is present, require model_reason to reference
#   unified_model_router (router-derived model choice). Block otherwise.
#
# Exit 2 on block.

set -u
LC_ALL=C

INPUT=$(cat)
if ! command -v jq >/dev/null 2>&1; then
  echo "model-reason-block: jq missing, allow-open" >&2
  exit 0
fi

TOOL_NAME=$(printf '%s' "$INPUT" | jq -r '.tool_name // empty')
case "$TOOL_NAME" in
  Task|Agent|mcp__plugin_fallback-agent_fallback__Task) ;;
  *) exit 0 ;;
esac

PROMPT=$(printf '%s' "$INPUT" | jq -r '.tool_input.prompt // empty')
LEN=${#PROMPT}

# Auto-escalation path: openclaw_failed marker requires router-derived model.
if printf '%s' "$PROMPT" | grep -qE '# *openclaw_failed:'; then
  if printf '%s' "$PROMPT" | grep -qiE '# *model_reason:.*(unified_model_router|router=|\brouter\b)'; then
    exit 0
  fi
  cat >&2 <<EOF2
BLOCKED by model-reason-block hook (openclaw_failed escalation path).

Prompt contains "# openclaw_failed:" but model_reason is missing
or does not reference unified_model_router.

Required for auto-escalated spawns:
  # openclaw_failed: <task_id>
  # model_reason: router=<model> (chosen via unified_model_router.py)

Run:
  python3 ".../AI-Tools/s&p500-ticker-mastery/scripts/unified_model_router.py" \\
    --complexity <low|medium|high> --task-kind "<kind>"
and embed its stdout in the model_reason line.
EOF2
  exit 2
fi

# Short spawns are exempt (one-shot diagnostics, status checks).
if [ "$LEN" -le 500 ]; then
  exit 0
fi

# Accept any of the three headers as satisfying the rule.
if printf '%s' "$PROMPT" | grep -qE '# *model_reason:|# *scope_estimate_min:|# *inline_justification:'; then
  exit 0
fi

cat >&2 <<EOF3
BLOCKED by model-reason-block hook.

Spawn prompt is ${LEN} chars but contains none of:
  # model_reason: <one-line justification>
  # scope_estimate_min: <integer minutes>
  # inline_justification: <why inline rather than fan-out>

Required by AGENT_BRIEF_TEMPLATE §5 and CLAUDE.md hard-rule (2026-05-16).
Add one (or more) header line near the top of the spawn prompt and retry.
EOF3
exit 2
