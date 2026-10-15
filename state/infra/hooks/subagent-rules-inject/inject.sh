#!/bin/bash
# SubagentStart hook: inject mandate rules into every sub-agent's context.
#
# Reads from ~/.zg/mandates.md (single source). Falls back to embedded block
# if mandates.md is missing/empty so sub-agents always get something.
#
# Per Claude Code spec, the ONLY hook field that injects text into the
# sub-agent's prompt is `hookSpecificOutput.additionalContext`.
#
# Output: single JSON object on stdout. Stderr logs are advisory.

set -u

LOG_DIR="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/logs/auto_solve"
mkdir -p "$LOG_DIR" 2>/dev/null
LOG_FILE="$LOG_DIR/subagent_rules_inject.log"

# Drain hook payload so parent doesn't block.
INPUT="$(cat 2>/dev/null || true)"

# Resolve mandates path. Try real $HOME first, then Drive-redirected $HOME.
MANDATES_PATH=""
for candidate in \
  "/Users/orginal/.zg/mandates.md" \
  "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/home/.zg/mandates.md" \
  "$HOME/.zg/mandates.md"; do
  if [ -r "$candidate" ] && [ -s "$candidate" ]; then
    MANDATES_PATH="$candidate"
    break
  fi
done

RULES=""
SOURCE="fallback-embedded"
if [ -n "$MANDATES_PATH" ]; then
  RULES="$(cat "$MANDATES_PATH" 2>/dev/null || true)"
  SOURCE="$MANDATES_PATH"
fi

# Fallback embedded block (used only if mandates.md missing).
if [ -z "$RULES" ]; then
  RULES=$(cat <<'EOF'
=== INHERITED MANDATES (fallback — ~/.zg/mandates.md missing) ===
§3 FAN-OUT, §5a CLOUD-ROUTING, §8 AUTO-SOLVE, AUTO-EXECUTE, NEVER RESTART,
TOKEN-SAVERS, GUARDRAIL-GRADE, KARPATHY-PRE-FLIGHT, OPENCLAW-ROUTING,
REPO-INTEL-LAYER, UNIVERSAL-RESUME. See AI-Tools/CLAUDE.md for full text.
EOF
)
fi

# Emit hook JSON via python3 for safe escaping (newlines/quotes/unicode).
RULES="$RULES" python3 -c '
import json, os, sys
rules = os.environ.get("RULES", "")
out = {
    "hookSpecificOutput": {
        "hookEventName": "SubagentStart",
        "additionalContext": rules,
    }
}
sys.stdout.write(json.dumps(out))
'

# Audit log (one line per fire).
{
  printf '[%s] subagent-rules-inject fired (rules=%d bytes, source=%s, input=%d bytes)\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${#RULES}" "$SOURCE" "${#INPUT}"
} >> "$LOG_FILE" 2>/dev/null

exit 0
