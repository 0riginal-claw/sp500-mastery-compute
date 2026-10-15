#!/usr/bin/env bash
# gabriel-self-freshness — PreToolUse guardrail (Hook 1 of 6)
#
# Check that the gabriel-self module state files exist and are fresh
# (modified <STALE_SECS seconds ago). If any file is missing or stale,
# write a REFRESH_REQUIRED marker so the autonomous-mode daemon's next
# cycle force-runs the corresponding module. Does NOT block the tool
# call (always exits 0); pure-signaling guardrail.
#
# State files watched (under state/gabriel_self/):
#   capability_map.json
#   reflexions.jsonl
#   user_predictor.json
#   curiosity_state.json
#   goal_tree.json
#
# Mirror of autonomous-daemon-heartbeat pattern.

set +e
LC_ALL=C

ROOT="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools"
STATE_DIR="$ROOT/state/gabriel_self"
MARKER="$STATE_DIR/REFRESH_REQUIRED"
LOG_DIR="$ROOT/logs/gabriel_self"
LOG_FILE="$LOG_DIR/freshness.log"
STALE_SECS=600   # 10 minutes

mkdir -p "$STATE_DIR" "$LOG_DIR" 2>/dev/null

# Drain stdin (PreToolUse delivers JSON payload we ignore)
cat >/dev/null 2>&1 || true

log() {
  printf '[%s] gabriel-self-freshness: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$LOG_FILE" 2>/dev/null
}

NOW=$(date +%s)
STALE_LIST=""
MISSING_LIST=""

for f in capability_map.json reflexions.jsonl user_predictor.json curiosity_state.json goal_tree.json; do
  path="$STATE_DIR/$f"
  if [ ! -f "$path" ]; then
    MISSING_LIST="$MISSING_LIST $f"
    continue
  fi
  # macOS stat: -f %m for mtime
  MTIME=$(stat -f %m "$path" 2>/dev/null || stat -c %Y "$path" 2>/dev/null || echo 0)
  if [ -z "$MTIME" ] || [ "$MTIME" -eq 0 ] 2>/dev/null; then
    continue
  fi
  AGE=$((NOW - MTIME))
  if [ "$AGE" -gt "$STALE_SECS" ] 2>/dev/null; then
    STALE_LIST="$STALE_LIST $f($AGE s)"
  fi
done

if [ -n "$MISSING_LIST" ] || [ -n "$STALE_LIST" ]; then
  # Write/update marker atomically
  TMP="${MARKER}.tmp.$$"
  python3 - <<PY > "$TMP" 2>/dev/null
import json, os, time
out = {
    "ts": int(time.time()),
    "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "missing": "${MISSING_LIST}".split(),
    "stale": "${STALE_LIST}".split(),
    "source": "gabriel-self-freshness",
}
print(json.dumps(out))
PY
  if [ -s "$TMP" ]; then
    mv -f "$TMP" "$MARKER" 2>/dev/null
    log "marker written: missing=[$MISSING_LIST] stale=[$STALE_LIST]"
  else
    rm -f "$TMP" 2>/dev/null
  fi
else
  # All fresh — clear any stale marker
  if [ -f "$MARKER" ]; then
    rm -f "$MARKER" 2>/dev/null
    log "all fresh — marker cleared"
  fi
fi

exit 0
