#!/usr/bin/env bash
# auto_signup_codespaces.sh — Configure gh CLI device-flow for Codespaces access
# Safety-gated: requires --confirm-create. gh CLI is MIT.
# Generated 2026-05-17.
#
# Codespaces requires an existing GitHub account. This script does NOT create
# a GitHub account (which requires phone OTP and would violate ToS to script);
# it ONLY launches `gh auth login` which uses GitHub's device-code flow.

set -euo pipefail

PROVIDER="codespaces"
LOG_DIR="$HOME/AI-Tools/logs/auto_signup"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${PROVIDER}_$(date -u +%Y%m%dT%H%M%SZ).log"

usage() {
  cat <<EOF
Usage: $0 [--confirm-create] [--dry-run]

Configures gh CLI to authenticate with GitHub (enables Codespaces access).
Prereq: existing GitHub account (https://github.com/signup if needed).

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

if ! command -v gh >/dev/null 2>&1; then
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "DRY-RUN: would execute: brew install gh" | tee -a "$LOG_FILE"
  else
    brew install gh 2>&1 | tee -a "$LOG_FILE"
  fi
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "DRY-RUN: would execute: gh auth login --scopes codespace" | tee -a "$LOG_FILE"
  exit 0
fi

echo "Launching gh auth login (device flow) with codespace scope..." | tee -a "$LOG_FILE"
gh auth login --scopes codespace --web 2>&1 | tee -a "$LOG_FILE" || gh auth login --scopes codespace 2>&1 | tee -a "$LOG_FILE"

echo "[$(date -u +%FT%TZ)] $PROVIDER setup complete" | tee -a "$LOG_FILE"
echo "Smoke tests:"
echo "  gh auth status"
echo "  gh codespace list"
