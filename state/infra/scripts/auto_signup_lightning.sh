#!/usr/bin/env bash
# auto_signup_lightning.sh — Install lightning CLI and launch OAuth login
# Safety-gated: requires --confirm-create. lightning is Apache-2.0.
# Generated 2026-05-17.

set -euo pipefail

PROVIDER="lightning_ai"
LOG_DIR="$HOME/AI-Tools/logs/auto_signup"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${PROVIDER}_$(date -u +%Y%m%dT%H%M%SZ).log"

usage() {
  cat <<EOF
Usage: $0 [--confirm-create] [--dry-run]

Installs Lightning AI CLI and launches OAuth login (browser-based).
Lightning has NO programmatic account-create endpoint; the signup happens
in-browser via OAuth (Google/GitHub) or email+password.

  --confirm-create  Required to actually run
  --dry-run         Print without executing
  -h, --help        Show this help
EOF
}

CONFIRM=0
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --confirm-create) CONFIRM=1; shift ;;
    --dry-run)        DRY_RUN=1; shift ;;
    -h|--help)        usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ "$CONFIRM" -ne 1 ]]; then
  echo "SAFETY GATE: refusing to run without --confirm-create" >&2
  exit 1
fi

echo "[$(date -u +%FT%TZ)] $PROVIDER setup begin" | tee -a "$LOG_FILE"

if ! python3 -c "import lightning" >/dev/null 2>&1; then
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "DRY-RUN: would execute: pip install --user lightning" | tee -a "$LOG_FILE"
  else
    pip install --user lightning 2>&1 | tee -a "$LOG_FILE"
  fi
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "DRY-RUN: would execute: lightning login" | tee -a "$LOG_FILE"
  exit 0
fi

echo "Launching 'lightning login' — browser will open for OAuth." | tee -a "$LOG_FILE"
lightning login 2>&1 | tee -a "$LOG_FILE"

echo "[$(date -u +%FT%TZ)] $PROVIDER setup complete" | tee -a "$LOG_FILE"
echo "Smoke test:  lightning list studios"
