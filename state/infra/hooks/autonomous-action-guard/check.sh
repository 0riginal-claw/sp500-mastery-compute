#!/usr/bin/env bash
# autosolve_skip: greenfield safety hook, no error condition
# autonomous-action-guard — PreToolUse hook
# Validates that spawn dispatches originating from autonomous_mode_daemon
# don't carry destructive-action keywords. Logs every action to the audit jsonl.
#
# Trigger: only fires on Bash invocations whose command string contains the
# token "autonomous_mode" OR the brief path under state/autonomous_mode/.
# All other Bash calls pass through untouched.
set -euo pipefail

INPUT="${1:-/dev/stdin}"
PAYLOAD="$(cat "$INPUT" 2>/dev/null || true)"

# Quick exit if not an autonomous-mode dispatch
if ! echo "$PAYLOAD" | grep -qE 'autonomous_mode|state/autonomous_mode/spawn_briefs/' ; then
  exit 0
fi

# Extract command (best-effort json parse, fall back to empty)
CMD="$(echo "$PAYLOAD" | /usr/bin/python3 -c 'import json,sys
try:
  d=json.load(sys.stdin); print(d.get("tool_input",{}).get("command",""))
except Exception:
  print("")
' 2>/dev/null || echo "")"

# Destructive keywords (substring, case-insensitive)
BLOCKLIST=(
  "rm -rf"
  "rm  -rf"
  "force push"
  "force-push"
  "git push --force"
  "git push -f "
  "drop table"
  "sudo rm"
  "kill -9 1"
  "wallet"
  "transfer "
  "wire "
  "send money"
  "send sms"
  "send email"
  "mailto:"
  ".ssh/id_"
  "aws_secret"
  "private_key"
)

LOWER=$(echo "$CMD" | tr '[:upper:]' '[:lower:]')
HIT=""
for kw in "${BLOCKLIST[@]}"; do
  if echo "$LOWER" | grep -qF -- "$kw"; then
    HIT="$kw"; break
  fi
done

AUDIT_DIR="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/state/autonomous_mode"
mkdir -p "$AUDIT_DIR"
DATE=$(date -u +%Y-%m-%d)
TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
AUDIT="$AUDIT_DIR/audit_${DATE}.jsonl"
PREVIEW=$(echo "$CMD" | head -c 200 | tr -d '\n' | tr -d '"')

if [[ -n "$HIT" ]]; then
  echo "{\"timestamp\":\"$TS\",\"event\":\"hook_block\",\"keyword\":\"$HIT\",\"command_preview\":\"$PREVIEW\"}" >> "$AUDIT"
  echo "BLOCKED by autonomous-action-guard: destructive keyword '$HIT' in autonomous-mode dispatch" >&2
  exit 2
fi

echo "{\"timestamp\":\"$TS\",\"event\":\"hook_pass\",\"command_preview\":\"$PREVIEW\"}" >> "$AUDIT"
exit 0
