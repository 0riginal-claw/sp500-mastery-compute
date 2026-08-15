#!/usr/bin/env bash
# auto_signup_bacalhau.sh — Install Bacalhau CLI; no signup needed for public network
# Safety-gated: requires --confirm-create. License: MIT/Apache-2.0 (Bacalhau itself).
# Generated 2026-05-17 by auto-signup research subagent.

set -euo pipefail

PROVIDER="bacalhau"
TOKEN_ENV_FILE="${TOKEN_ENV_FILE:-$HOME/AI-Tools/secrets/cloud_tokens.env}"
LOG_DIR="$HOME/AI-Tools/logs/auto_signup"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${PROVIDER}_$(date -u +%Y%m%dT%H%M%SZ).log"

usage() {
  cat <<EOF
Usage: $0 [--confirm-create] [--dry-run]

Bacalhau public demo network requires NO account creation. This script
only INSTALLS the CLI binary. Safety gate is enforced anyway for consistency.

  --confirm-create  Required to actually run the install
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
  echo "(Bacalhau install is harmless but the gate is enforced for all auto_signup_* scripts.)" >&2
  exit 1
fi

echo "[$(date -u +%FT%TZ)] $PROVIDER install begin" | tee -a "$LOG_FILE"

if command -v bacalhau >/dev/null 2>&1; then
  echo "bacalhau CLI already installed: $(bacalhau version 2>&1 | head -1)" | tee -a "$LOG_FILE"
  exit 0
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "DRY-RUN: would execute: curl -sL https://get.bacalhau.org/install.sh | bash" | tee -a "$LOG_FILE"
  exit 0
fi

# License-clean: Bacalhau is Apache-2.0 (https://github.com/bacalhau-project/bacalhau/blob/main/LICENSE)
echo "Installing Bacalhau CLI via official install script..." | tee -a "$LOG_FILE"
curl -sL https://get.bacalhau.org/install.sh | bash 2>&1 | tee -a "$LOG_FILE"

echo "Verifying install..." | tee -a "$LOG_FILE"
bacalhau version 2>&1 | tee -a "$LOG_FILE"

echo "[$(date -u +%FT%TZ)] $PROVIDER install complete" | tee -a "$LOG_FILE"
echo "No token needed — public network is open. Test with:"
echo "  bacalhau docker run ubuntu echo hello"
