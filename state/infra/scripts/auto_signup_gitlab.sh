#!/usr/bin/env bash
# auto_signup_gitlab.sh — Install glab CLI and launch device-flow auth
# Safety-gated: requires --confirm-create. glab is MIT.
# Generated 2026-05-17.
#
# GitLab does NOT expose a programmatic signup endpoint; account creation
# requires the web form. This script installs the CLI and runs `glab auth login`
# which uses the device-code/personal-access-token flow against an EXISTING
# account.

set -euo pipefail

PROVIDER="gitlab"
LOG_DIR="$HOME/AI-Tools/logs/auto_signup"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${PROVIDER}_$(date -u +%Y%m%dT%H%M%SZ).log"

usage() {
  cat <<EOF
Usage: $0 [--confirm-create] [--dry-run]

Configures glab CLI to authenticate with GitLab.com.
Prereq: existing GitLab account (https://gitlab.com/users/sign_up if needed).

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

if ! command -v glab >/dev/null 2>&1; then
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "DRY-RUN: would execute: brew install glab" | tee -a "$LOG_FILE"
  else
    brew install glab 2>&1 | tee -a "$LOG_FILE"
  fi
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "DRY-RUN: would execute: glab auth login --hostname gitlab.com" | tee -a "$LOG_FILE"
  exit 0
fi

echo "Launching glab auth login (device/PAT flow)..." | tee -a "$LOG_FILE"
glab auth login --hostname gitlab.com 2>&1 | tee -a "$LOG_FILE"

echo "[$(date -u +%FT%TZ)] $PROVIDER setup complete" | tee -a "$LOG_FILE"
echo "Smoke test:  glab auth status"
