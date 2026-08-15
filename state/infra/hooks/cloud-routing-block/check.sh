#!/usr/bin/env bash
# cloud-routing-block/check.sh
# PreToolUse hook for: Bash
#
# Enforces the cloud-routing mandate (CLAUDE.md §5a, feedback_cloud_routing_mandate.md):
# heavy-compute scripts (orb_*/vwap_*/backtest_*/momentum_*/catalyst_*) called
# with --ticker MUST go via cloud_dispatch.enqueue_job(), not local subprocess.
#
# BLOCK when:
#   command matches regex (orb_|vwap_|backtest_|momentum_|catalyst_).*\.py.*--ticker
#   AND command does NOT contain "cloud_dispatch.enqueue_job"
#   AND command does NOT set AUTO_CLOUD_DISPATCH=1
#   AND command does NOT carry a "# justify_local: <reason>" marker
#
# Exit 2 on block.

set -u
LC_ALL=C

INPUT=$(cat)
if ! command -v jq >/dev/null 2>&1; then
  echo "cloud-routing-block: jq missing, allow-open" >&2
  exit 0
fi

TOOL_NAME=$(printf '%s' "$INPUT" | jq -r '.tool_name // empty')
[ "$TOOL_NAME" = "Bash" ] || exit 0

CMD=$(printf '%s' "$INPUT" | jq -r '.tool_input.command // empty')
[ -n "$CMD" ] || exit 0

# Heavy-compute script trigger.
HEAVY_RE='(orb_|vwap_|backtest_|momentum_|catalyst_)[A-Za-z0-9_]*\.py[^\n]*(--ticker|--symbol|-t[[:space:]]+[A-Z])'
if ! printf '%s' "$CMD" | grep -qE "$HEAVY_RE"; then
  exit 0
fi

# Allow: explicitly using the cloud dispatcher inline.
if printf '%s' "$CMD" | grep -qE 'cloud_dispatch\.enqueue_job|enqueue_job\(|--dispatch-mode[[:space:]]+cloud'; then
  exit 0
fi

# Allow: env-flag opt-in to dispatcher.
if printf '%s' "$CMD" | grep -qE '(^|[[:space:]])AUTO_CLOUD_DISPATCH=1'; then
  exit 0
fi

# Allow: explicit override comment.
if printf '%s' "$CMD" | grep -qE '# *justify_local:'; then
  LOG="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/logs/auto_solve/cloud_routing_overrides.log"
  { date -u +'%Y-%m-%dT%H:%M:%SZ'; echo "OVERRIDE: $(printf '%s' "$CMD" | grep -oE '# *justify_local:[^[:cntrl:]]*' | head -1)"; echo "CMD: $CMD"; } >> "$LOG" 2>/dev/null
  exit 0
fi

cat >&2 <<EOF
BLOCKED by cloud-routing-block hook.

Heavy-compute script detected:
  $CMD

Per CLAUDE.md §5a (cloud-routing mandate), this MUST run on Modal/gh_actions:
  - Route via cloud_dispatch.enqueue_job(ticker=..., script=..., ...)
  - OR prefix env: AUTO_CLOUD_DISPATCH=1 python <script> ...
  - OR pass --dispatch-mode cloud

Local override (smoke-test <60s only):
  Append "# justify_local: <reason>" to the command. Override is logged.
EOF
exit 2
