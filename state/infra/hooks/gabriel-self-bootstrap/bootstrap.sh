#!/usr/bin/env bash
# gabriel-self-bootstrap — SessionStart guardrail (Hook 2 of 6)
#
# Ensure state/gabriel_self/ exists and the 5 watched files are initialized
# (touched empty if missing) so the freshness check passes on first run and
# downstream modules don't crash on file-not-found.
#
# Each file is created with minimum valid skeleton if absent. If a file
# already exists with content, it is left untouched.

set +e
LC_ALL=C

ROOT="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools"
STATE_DIR="$ROOT/state/gabriel_self"
LOG_DIR="$ROOT/logs/gabriel_self"
LOG_FILE="$LOG_DIR/bootstrap.log"

mkdir -p "$STATE_DIR" "$LOG_DIR" 2>/dev/null

# Drain stdin
cat >/dev/null 2>&1 || true

log() {
  printf '[%s] gabriel-self-bootstrap: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$LOG_FILE" 2>/dev/null
}

NOW_ISO=$(date -u +%Y-%m-%dT%H:%M:%SZ)
NOW_UNIX=$(date +%s)

init_json() {
  local path="$1"
  local skeleton="$2"
  if [ ! -f "$path" ] || [ ! -s "$path" ]; then
    echo "$skeleton" > "$path" 2>/dev/null
    log "initialized $(basename "$path")"
  fi
}

init_jsonl() {
  local path="$1"
  if [ ! -f "$path" ]; then
    : > "$path" 2>/dev/null
    log "initialized empty $(basename "$path")"
  fi
}

# capability_map.json — describes what Gabriel can do
init_json "$STATE_DIR/capability_map.json" \
  "{\"ts\":\"$NOW_ISO\",\"version\":0,\"capabilities\":[],\"source\":\"gabriel-self-bootstrap\"}"

# reflexions.jsonl — append-only lessons learned
init_jsonl "$STATE_DIR/reflexions.jsonl"

# user_predictor.json — model of user preferences
init_json "$STATE_DIR/user_predictor.json" \
  "{\"ts\":\"$NOW_ISO\",\"version\":0,\"preferences\":{},\"recent_signals\":[],\"source\":\"gabriel-self-bootstrap\"}"

# curiosity_state.json — what Gabriel is currently curious about
init_json "$STATE_DIR/curiosity_state.json" \
  "{\"ts\":\"$NOW_ISO\",\"version\":0,\"open_questions\":[],\"source\":\"gabriel-self-bootstrap\"}"

# goal_tree.json — hierarchical goal/plan tree (plan-without-direction module)
init_json "$STATE_DIR/goal_tree.json" \
  "{\"ts\":\"$NOW_ISO\",\"version\":0,\"root\":{\"id\":\"root\",\"label\":\"workspace\",\"children\":[]},\"source\":\"gabriel-self-bootstrap\"}"

# Touch all 5 so freshness check starts with green status
for f in capability_map.json reflexions.jsonl user_predictor.json curiosity_state.json goal_tree.json; do
  touch "$STATE_DIR/$f" 2>/dev/null
done

log "bootstrap complete"

exit 0
