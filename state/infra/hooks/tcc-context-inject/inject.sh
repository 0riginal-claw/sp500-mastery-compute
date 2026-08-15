#!/bin/bash
# tcc-context-inject - SubagentStart guardrail (Hook 4 of 5)
#
# Inject the TCC auto-allow posture into every sub-agent's context so children
# know permission dialogs will auto-allow and they shouldn't pause on them.
#
# Output: single JSON object on stdout with hookSpecificOutput.additionalContext.
# Runs in addition to subagent-rules-inject + autonomous-subagent-inject +
# gabriel-context-inject.

set -u

ROOT="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools"
LOG_DIR="$ROOT/logs/auto_solve"
LOG_FILE="$LOG_DIR/tcc_guardrails.log"
AUDIT_FILE="$ROOT/logs/tcc_autoallow_audit.jsonl"
MARKER="$ROOT/state/tcc_autoallow/REFRESH_REQUIRED"

mkdir -p "$LOG_DIR" 2>/dev/null

INPUT="$(cat 2>/dev/null || true)"

HS_STATE="unknown"
if pgrep -f Hammerspoon >/dev/null 2>&1; then
  HS_STATE="alive"
else
  HS_STATE="dead"
fi

MARKER_STATE="fresh"
if [ -f "$MARKER" ]; then
  MARKER_STATE="degraded"
fi

BRIEF=$(cat <<EOF
=== TCC AUTO-ALLOW INHERITANCE (workspace standing posture) ===

This workspace runs a 5-hook TCC-permission auto-allow guardrail chain.
macOS Transparency-Consent-Control (TCC) dialogs ("App X wants to access Y")
auto-click "Allow" via Hammerspoon + PostToolUse AppleScript backstop.

Current state:
  hammerspoon=${HS_STATE}  marker=${MARKER_STATE}

What this means for you:
1. TCC permissions are pre-granted for Python / Terminal / iTerm2 / Claude
   bundle IDs (Accessibility, Files & Folders, Automation services).
2. If a TCC dialog appears DURING your tool run, it will auto-allow within
   1.5s (Hammerspoon poll) or at PostToolUse (osascript backstop).
3. You should NOT stop on permission dialogs - they auto-resolve. If a tool
   call appears to hang on a permission, wait 3s then retry.
4. Safety boundary: sudo / ssh / keychain / Touch ID / Password dialogs are
   NEVER auto-clicked. Those still require user action.

If you observe a TCC dialog stuck >5s:
- Check logs/tcc_autoallow_audit.jsonl for status entries.
- If status=stuck appears, the Allow button couldn't be found - escalate to
  user. If status=skipped_deny, the dialog is safety-sensitive (correct).

State files:
- ${ROOT}/state/tcc_autoallow/ - markers (REFRESH_REQUIRED, REINIT_*).
- ${ROOT}/logs/tcc_autoallow_audit.jsonl - append-only auto-click audit.
- ${ROOT}/logs/auto_solve/tcc_guardrails.log - hook execution log.

For full reference: docs/MACOS_TCC_AUTOALLOW.md.

=== END TCC AUTO-ALLOW INHERITANCE ===
EOF
)

BRIEF="$BRIEF" python3 -c '
import json, os, sys
brief = os.environ.get("BRIEF", "")
out = {
    "hookSpecificOutput": {
        "hookEventName": "SubagentStart",
        "additionalContext": brief,
    }
}
sys.stdout.write(json.dumps(out))
'

{
  printf '[%s] tcc-context-inject fired (brief=%d bytes, hs=%s, marker=%s)\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${#BRIEF}" "$HS_STATE" "$MARKER_STATE"
} >> "$LOG_FILE" 2>/dev/null

exit 0
