#!/bin/bash
# autonomous-subagent-inject — SubagentStart guardrail (Layer 5)
#
# Inject autonomous-mode mission summary into every sub-agent's context so
# children inherit the autonomous-mode posture (idea/decision/spawn loop,
# proof-of-work shape, dampener locations).
#
# Output: single JSON object on stdout with
#   hookSpecificOutput.additionalContext
# This runs in addition to (not instead of) subagent-rules-inject.
#
# Layer 5 of 6.

set -u

LOG_DIR="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/logs/auto_solve"
mkdir -p "$LOG_DIR" 2>/dev/null
LOG_FILE="$LOG_DIR/autonomous_guardrails.log"

# Drain payload so caller doesn't block.
INPUT="$(cat 2>/dev/null || true)"

ROOT="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools"
HB="$ROOT/state/autonomous_mode/heartbeat.json"

# Snapshot daemon state for the brief (best-effort).
DAEMON_STATE="unknown"
CYCLE_ID="-"
INFLIGHT="-"
if [ -f "$HB" ]; then
  DAEMON_STATE=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("state","unknown"))' "$HB" 2>/dev/null || echo "unknown")
  CYCLE_ID=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("cycle_id","-"))' "$HB" 2>/dev/null || echo "-")
  INFLIGHT=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("inflight","-"))' "$HB" 2>/dev/null || echo "-")
fi

BRIEF=$(cat <<EOF
=== AUTONOMOUS-MODE INHERITANCE (workspace standing posture) ===

This workspace runs an always-on autonomous_mode_daemon (PID-managed by launchd,
respawned by Claude Code PreToolUse + SessionStart hooks). The daemon runs an
ideate -> decide -> spawn -> react loop, writes audit_<DATE>.jsonl, and obeys
dampeners (load>12, inflight>N, user-online signal).

Current daemon snapshot:
  state=${DAEMON_STATE}  cycle_id=${CYCLE_ID}  inflight=${INFLIGHT}

If your task touches autonomous mode (daemon code, state files, audit logs,
plans, decisions, inbox), follow these rules:

1. NEVER manually kill the daemon (launchctl bootout) without immediately
   re-bootstrapping or relying on the heartbeat hook to respawn within 5min.
2. State files live at AI-Tools/state/autonomous_mode/ — append-only JSONL
   for audit/decisions/inbox; atomic-replace JSON for heartbeat/config.
3. Plans live at state/autonomous_mode/plans/, blockers at /blockers/,
   spawn briefs at /spawn_briefs/.
4. The daemon reads state/autonomous_mode/last_session_activity.unix as
   "user-online" signal (updated by every PostToolUse). Do not write past
   timestamps there.
5. If you spawn helpers, propagate this brief into their context too
   (the SubagentStart hook does this automatically).

For full reference: docs/AUTONOMOUS_MODE.md.

=== END AUTONOMOUS-MODE INHERITANCE ===
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
  printf '[%s] autonomous-subagent-inject fired (brief=%d bytes, input=%d bytes, state=%s)\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${#BRIEF}" "${#INPUT}" "$DAEMON_STATE"
} >> "$LOG_FILE" 2>/dev/null

exit 0
