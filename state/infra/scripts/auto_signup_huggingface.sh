#!/usr/bin/env bash
# auto_signup_huggingface.sh — Install huggingface_hub CLI and capture token
# Safety-gated: requires --confirm-create. huggingface_hub is Apache-2.0.
# Generated 2026-05-17.
#
# NOTE: HuggingFace does NOT expose a programmatic account-creation endpoint.
# This script INSTALLS the CLI and prompts the human to paste a pre-created
# token (from https://huggingface.co/settings/tokens). The signup itself is
# the manual step documented in CLOUD_SIGNUP_RUNBOOK.md.

set -euo pipefail

PROVIDER="huggingface"
LOG_DIR="$HOME/AI-Tools/logs/auto_signup"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${PROVIDER}_$(date -u +%Y%m%dT%H%M%SZ).log"

usage() {
  cat <<EOF
Usage: $0 [--confirm-create] [--token <TOKEN>] [--dry-run]

Installs huggingface_hub CLI and configures token.

  --confirm-create   Required to actually run
  --token <TOKEN>    Optional: token to save (else interactive prompt)
  --dry-run          Print without executing
  -h, --help         Show this help

Prereq: account exists at https://huggingface.co/join AND a token has been
created at https://huggingface.co/settings/tokens (scope: write).
EOF
}

CONFIRM=0
DRY_RUN=0
TOKEN=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --confirm-create) CONFIRM=1; shift ;;
    --dry-run)        DRY_RUN=1; shift ;;
    --token)          TOKEN="$2"; shift 2 ;;
    -h|--help)        usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ "$CONFIRM" -ne 1 ]]; then
  echo "SAFETY GATE: refusing to run without --confirm-create" >&2
  exit 1
fi

echo "[$(date -u +%FT%TZ)] $PROVIDER setup begin" | tee -a "$LOG_FILE"

if ! python3 -c "import huggingface_hub" >/dev/null 2>&1; then
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "DRY-RUN: would execute: pip install --user huggingface_hub" | tee -a "$LOG_FILE"
  else
    pip install --user huggingface_hub 2>&1 | tee -a "$LOG_FILE"
  fi
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "DRY-RUN: would execute: huggingface-cli login" | tee -a "$LOG_FILE"
  exit 0
fi

if [[ -n "$TOKEN" ]]; then
  mkdir -p "$HOME/.cache/huggingface"
  printf '%s' "$TOKEN" > "$HOME/.cache/huggingface/token"
  chmod 600 "$HOME/.cache/huggingface/token"
  echo "Token saved to ~/.cache/huggingface/token" | tee -a "$LOG_FILE"
else
  echo "Launching huggingface-cli login (paste token from https://huggingface.co/settings/tokens):" | tee -a "$LOG_FILE"
  huggingface-cli login 2>&1 | tee -a "$LOG_FILE"
fi

echo "[$(date -u +%FT%TZ)] $PROVIDER setup complete" | tee -a "$LOG_FILE"
echo "Smoke test:  huggingface-cli whoami"
