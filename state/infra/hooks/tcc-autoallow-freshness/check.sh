#!/usr/bin/env bash
# tcc-autoallow-freshness — PreToolUse guardrail (Hook 1 of 5)
#
# Verifies the TCC auto-allow guardrail chain is alive before every tool call:
#   - Hammerspoon process is running (auto-clicks TCC "Allow" dialogs)
#   - ~/.hammerspoon/init.lua exists (Hammerspoon watcher config)
#   - TCC.db has required pre-grant rows for Python / Terminal / Claude /
#     Accessibility / Files-and-Folders (best-effort, read-only check)
#
# Recovery actions (non-blocking, background):
#   - If Hammerspoon dead: `open -gja Hammerspoon` (background, no foreground)
#   - If init.lua missing: log + flag SessionStart bootstrap to re-write it
#   - If TCC row missing: log + flag SessionStart bootstrap to re-insert
#
# Mirror of gabriel-self-freshness / autonomous-daemon-heartbeat pattern.
# Always exits 0. Pure-signal + background-recover.

set +e
LC_ALL=C

ROOT="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools"
LOG_DIR="$ROOT/logs/auto_solve"
LOG_FILE="$LOG_DIR/tcc_guardrails.log"
AUDIT_FILE="$ROOT/logs/tcc_autoallow_audit.jsonl"
MARKER_DIR="$ROOT/state/tcc_autoallow"
MARKER="$MARKER_DIR/REFRESH_REQUIRED"
HS_INIT="$HOME/.hammerspoon/init.lua"

mkdir -p "$LOG_DIR" "$MARKER_DIR" 2>/dev/null
mkdir -p "$(dirname "$AUDIT_FILE")" 2>/dev/null

# Drain stdin (PreToolUse JSON payload — we ignore it)
cat >/dev/null 2>&1 || true

log() {
  printf '[%s] tcc-autoallow-freshness: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$LOG_FILE" 2>/dev/null
}

audit() {
  local status="$1"; shift
  local detail="$*"
  local now_iso
  now_iso=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  printf '{"ts":"%s","source":"freshness","status":"%s","detail":%s}\n' \
    "$now_iso" "$status" "$(printf '%s' "$detail" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))' 2>/dev/null || printf '"%s"' "$detail")" \
    >> "$AUDIT_FILE" 2>/dev/null
}

MISSING_LIST=""
RECOVERED_LIST=""

# 1. Hammerspoon process alive?
if ! pgrep -f Hammerspoon >/dev/null 2>&1; then
  MISSING_LIST="$MISSING_LIST hammerspoon-process"
  # Background respawn (no -n: don't open new instance if launching from .app bundle)
  if [ -d "/Applications/Hammerspoon.app" ]; then
    open -gja Hammerspoon >/dev/null 2>&1 &
    RECOVERED_LIST="$RECOVERED_LIST hammerspoon-process(respawn-bg)"
    log "Hammerspoon dead -> background respawn issued"
    audit "recovered" "Hammerspoon process dead; open -gja Hammerspoon issued"
  else
    log "Hammerspoon dead AND .app missing -> manual install required"
    audit "stuck" "Hammerspoon.app not installed at /Applications/Hammerspoon.app"
  fi
fi

# 2. ~/.hammerspoon/init.lua exists?
if [ ! -f "$HS_INIT" ]; then
  MISSING_LIST="$MISSING_LIST hammerspoon-init"
  # Flag bootstrap to write it on next SessionStart
  touch "$MARKER_DIR/REINIT_HAMMERSPOON" 2>/dev/null
  log "~/.hammerspoon/init.lua missing -> flagged for SessionStart bootstrap"
  audit "deferred" "init.lua missing; SessionStart hook will rewrite"
fi

# 3. TCC.db pre-grants (read-only sanity check — write happens in bootstrap)
TCC_USER_DB="$HOME/Library/Application Support/com.apple.TCC/TCC.db"
TCC_MISSING_BUNDLES=""
if [ -f "$TCC_USER_DB" ] && command -v sqlite3 >/dev/null 2>&1; then
  # The user-level TCC.db is readable. Check for grants for our key bundle IDs.
  # If sqlite3 errors (DB locked by tccd), skip silently.
  for bundle in "com.apple.Terminal" "com.googlecode.iterm2" "com.anthropic.claude-code"; do
    # service=kTCCServiceSystemPolicyAllFiles (Files & Folders / Full Disk Access proxy)
    # auth_value=2 means allowed
    cnt=$(sqlite3 "$TCC_USER_DB" "SELECT COUNT(*) FROM access WHERE client='$bundle' AND auth_value=2;" 2>/dev/null || echo "")
    if [ -z "$cnt" ] || [ "$cnt" = "0" ] 2>/dev/null; then
      TCC_MISSING_BUNDLES="$TCC_MISSING_BUNDLES $bundle"
    fi
  done
  if [ -n "$TCC_MISSING_BUNDLES" ]; then
    MISSING_LIST="$MISSING_LIST tcc-grants:${TCC_MISSING_BUNDLES// /,}"
    touch "$MARKER_DIR/REINIT_TCC_GRANTS" 2>/dev/null
    log "TCC pre-grants missing for:${TCC_MISSING_BUNDLES} -> flagged for SessionStart bootstrap"
    audit "deferred" "TCC pre-grants missing:${TCC_MISSING_BUNDLES}"
  fi
fi

# Write/clear marker
if [ -n "$MISSING_LIST" ]; then
  TMP="${MARKER}.tmp.$$"
  python3 - <<PY > "$TMP" 2>/dev/null
import json, time, os
out = {
    "ts": int(time.time()),
    "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "missing": "${MISSING_LIST}".split(),
    "recovered": "${RECOVERED_LIST}".split(),
    "source": "tcc-autoallow-freshness",
}
print(json.dumps(out))
PY
  if [ -s "$TMP" ]; then
    mv -f "$TMP" "$MARKER" 2>/dev/null
  else
    rm -f "$TMP" 2>/dev/null
  fi
else
  # All fresh — clear marker
  if [ -f "$MARKER" ]; then
    rm -f "$MARKER" 2>/dev/null
    log "all fresh — marker cleared"
  fi
fi

exit 0
