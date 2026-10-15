#!/usr/bin/env bash
# tcc-dialog-detect - PostToolUse guardrail (Hook 3 of 5)
#
# Defense-in-depth backstop to Hammerspoon. After every tool call,
# scan for any open TCC dialog and auto-click "Allow" via osascript
# if the title matches the whitelist.

set +e
LC_ALL=C

ROOT="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools"
LOG_DIR="$ROOT/logs/auto_solve"
LOG_FILE="$LOG_DIR/tcc_guardrails.log"
AUDIT_FILE="$ROOT/logs/tcc_autoallow_audit.jsonl"
APPLESCRIPT_FILE="$ROOT/home/.claude/hooks/tcc-dialog-detect/scan.applescript"

mkdir -p "$LOG_DIR" 2>/dev/null
mkdir -p "$(dirname "$AUDIT_FILE")" 2>/dev/null

cat >/dev/null 2>&1 || true

log() {
  printf '[%s] tcc-dialog-detect: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$LOG_FILE" 2>/dev/null
}

audit() {
  local status="$1"; shift
  local title="$1"; shift
  local app="$1"
  local now_iso safe_title safe_app
  now_iso=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  safe_title=$(printf '%s' "$title" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().rstrip()))' 2>/dev/null)
  safe_app=$(printf '%s' "$app" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().rstrip()))' 2>/dev/null)
  [ -z "$safe_title" ] && safe_title='""'
  [ -z "$safe_app" ] && safe_app='""'
  printf '{"ts":"%s","source":"posttooluse","status":"%s","title":%s,"app":%s}\n' \
    "$now_iso" "$status" "$safe_title" "$safe_app" >> "$AUDIT_FILE" 2>/dev/null
}

# Run only on macOS
if [ "$(uname)" != "Darwin" ]; then
  exit 0
fi

# Skip if AppleScript file missing
if [ ! -f "$APPLESCRIPT_FILE" ]; then
  log "scan.applescript missing - skipping"
  exit 0
fi

RESULT=$(osascript "$APPLESCRIPT_FILE" 2>/dev/null)

if [ -n "$RESULT" ]; then
  while IFS='|' read -r status title app; do
    [ -z "$status" ] && continue
    audit "$status" "$title" "$app"
    log "dialog status=$status app=$app"
  done <<< "$RESULT"
fi

exit 0
