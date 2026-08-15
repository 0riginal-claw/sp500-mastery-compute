#!/usr/bin/env bash
# auto_signup_vercel.sh — Install Vercel CLI and launch device-flow login
# Safety-gated: requires --confirm-create. Vercel CLI is Apache-2.0.
# Generated 2026-05-17 by auto-signup research subagent.
#
# NOTE: Vercel has NO programmatic signup endpoint. The CLI's `vercel login`
# opens a browser window where the user authenticates via email magic-link
# OR GitHub/GitLab/Bitbucket OAuth. This script automates ONLY the install +
# launch; the human still completes the OAuth step manually in the browser.
# This is the closest thing to "auto-signup" the provider supports.

set -euo pipefail

PROVIDER="vercel"
LOG_DIR="$HOME/AI-Tools/logs/auto_signup"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${PROVIDER}_$(date -u +%Y%m%dT%H%M%SZ).log"

usage() {
  cat <<EOF
Usage: $0 [--confirm-create] [--dry-run]

Installs Vercel CLI and launches the device-flow login. Browser will open.
You must complete the OAuth/email step manually in the browser.

  --confirm-create  Required to launch the install + login
  --dry-run         Print what would be done; do not execute
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

echo "[$(date -u +%FT%TZ)] $PROVIDER signup begin" | tee -a "$LOG_FILE"

if ! command -v node >/dev/null 2>&1; then
  echo "ERROR: Node.js not installed. Install Node 18+ first: brew install node" >&2
  exit 3
fi

if ! command -v vercel >/dev/null 2>&1; then
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "DRY-RUN: would execute: npm i -g vercel" | tee -a "$LOG_FILE"
  else
    echo "Installing Vercel CLI (Apache-2.0)..." | tee -a "$LOG_FILE"
    npm i -g vercel 2>&1 | tee -a "$LOG_FILE"
  fi
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "DRY-RUN: would execute: vercel login" | tee -a "$LOG_FILE"
  exit 0
fi

echo "Launching 'vercel login' — browser will open for OAuth/email." | tee -a "$LOG_FILE"
echo "Complete the flow in the browser, then return here." | tee -a "$LOG_FILE"
vercel login 2>&1 | tee -a "$LOG_FILE"

echo "[$(date -u +%FT%TZ)] $PROVIDER signup complete" | tee -a "$LOG_FILE"
echo "Token stored by CLI under ~/.local/share/com.vercel.cli/ (managed; do not edit)."
echo "Smoke test:  vercel whoami"
