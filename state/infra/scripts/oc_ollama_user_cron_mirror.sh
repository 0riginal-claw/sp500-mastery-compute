#!/usr/bin/env bash
# oc_ollama_user_cron_mirror.sh
#
# Phase-C of perfect-resume harden (2026-05-20). Independent of the
# universal_resume daemon. Runs in user cron context (NOT launchd) every
# minute and internally iterates 12x with 5s sleep to give effective 5s
# mirror cadence for OC + Ollama state -> local SSD.
#
# WHY a cron-side mirror exists when the daemon already covers it:
#   - Cron survives launchctl bootout / Mac restart cron
#   - Daemon outage (>5s) still leaves a working secondary mirror
#   - Single-purpose script is easier to audit + revive than the full daemon
#
# Source-of-truth: $HOME (OS-level + launcher-redirected Drive home).
# Destination: ~/.zg/state/oc_ollama_cron_mirror/  (local SSD only).
#
# Self-disable: touch  ~/.zg/state/oc_ollama_cron_mirror/.disabled
# Log:          ~/.zg/state/oc_ollama_cron_mirror/cron.log  (auto-rotated @ 1MB)

set -u
LC_ALL=C

OS_HOME="/Users/orginal"
DRIVE_HOME="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/home"
DST="$OS_HOME/.zg/state/oc_ollama_cron_mirror"
LOG="$DST/cron.log"
DISABLED="$DST/.disabled"

mkdir -p "$DST" 2>/dev/null
[[ -f "$DISABLED" ]] && exit 0

# Rotate log at 1MB
if [[ -f "$LOG" ]]; then
  SZ=$(stat -f%z "$LOG" 2>/dev/null || echo 0)
  [[ "$SZ" -gt 1048576 ]] && mv "$LOG" "$LOG.1" 2>/dev/null
fi

mirror_pair() {
  # $1 = src, $2 = relative dst under $DST
  local src="$1" dst="$DST/$2"
  [[ ! -d "$src" ]] && return 0
  mkdir -p "$dst" 2>/dev/null
  # rsync: incremental, atomic, no deletes (additive only — preserves history).
  # --max-size=2M skips huge blobs (model weights, etc.)
  /usr/bin/rsync -a --quiet --max-size=2M \
    --exclude='blobs/' --exclude='models/' --exclude='*.lock' \
    --exclude='gateway.err.log' --exclude='gateway.log' \
    --exclude='id_ed25519' \
    "$src"/ "$dst"/ 2>>"$LOG"
}

run_once() {
  local stamp
  stamp=$(date +%FT%T%z)
  local pairs=(
    "$DRIVE_HOME/.openclaw/sessions|openclaw/sessions"
    "$DRIVE_HOME/.openclaw/agents|openclaw/agents"
    "$DRIVE_HOME/.openclaw/tasks|openclaw/tasks"
    "$DRIVE_HOME/.openclaw/logs|openclaw/logs"
    "$DRIVE_HOME/.openclaw/completions|openclaw/completions"
    "$DRIVE_HOME/.ollama/history|ollama/history"
    "$DRIVE_HOME/.ollama/cache|ollama/cache"
    "$OS_HOME/.openclaw/sessions|openclaw_os/sessions"
    "$OS_HOME/.openclaw/agents|openclaw_os/agents"
    "$OS_HOME/.ollama/history|ollama_os/history"
  )
  for pair in "${pairs[@]}"; do
    src="${pair%%|*}"
    dst="${pair##*|}"
    mirror_pair "$src" "$dst"
  done
  echo "[$stamp] cron mirror cycle done" >> "$LOG"
}

# 12 iterations of 5s each = ~60s total (lines up with cron @ */1 * * * *).
# If invoked manually with --once, run just one cycle.
if [[ "${1:-}" == "--once" ]]; then
  run_once
  exit 0
fi

ITERS=12
SLEEP=5
for ((i=0; i<ITERS; i++)); do
  run_once
  # Skip sleep on last iteration
  [[ $i -lt $((ITERS - 1)) ]] && sleep "$SLEEP"
done

exit 0
