#!/usr/bin/env bash
# auto_signup_codespaces_cli.sh — CLI-only GitHub Codespaces bootstrap
# License: gh CLI is MIT. Generated 2026-05-17.
#
# Codespaces reuses existing GitHub auth. Requires existing GH account
# (GitHub signup requires phone OTP — out of scope; assume account exists).
# This is the cleanest GREEN path: no new signup at all.

set -euo pipefail

PROVIDER="codespaces"
# Launcher remaps $HOME into Drive. Secrets MUST live on Mac-local /Users/orginal.
MAC_HOME="/Users/orginal"
ENV_DIR="$MAC_HOME/.config/auto_signup"
ENV_FILE="$ENV_DIR/${PROVIDER}.env"
LOG_DIR="$MAC_HOME/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/logs/auto_signup"
CLOUD_USAGE="$MAC_HOME/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/s&p500-ticker-mastery/sweeps/cloud_usage.json"
mkdir -p "$ENV_DIR" "$LOG_DIR"
LOG_FILE="$LOG_DIR/${PROVIDER}_$(date -u +%Y%m%dT%H%M%SZ).log"

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG_FILE"; }

log "$PROVIDER CLI signup begin (dry_run=$DRY_RUN)"

if ! command -v gh >/dev/null 2>&1; then
  log "ERROR: gh CLI not installed. Run: brew install gh"; exit 2
fi

# Reuse existing GITHUB_TOKEN / GH_TOKEN env if present
TOKEN="${GITHUB_TOKEN:-${GH_TOKEN:-}}"
if [[ -z "$TOKEN" ]] && ! gh auth status >/dev/null 2>&1; then
  if [[ "$DRY_RUN" == 1 ]]; then
    log "DRY-RUN: gh auth login --scopes codespace --web"
    exit 0
  fi
  log "No GitHub auth detected. Launching device-flow login (codespace scope)"
  gh auth login --scopes codespace --web 2>&1 | tee -a "$LOG_FILE" || \
    gh auth login --scopes codespace 2>&1 | tee -a "$LOG_FILE"
fi

# Extract token via gh CLI (or use existing env var)
if [[ -z "$TOKEN" ]]; then
  TOKEN=$(gh auth token 2>/dev/null || true)
fi
[[ -z "$TOKEN" ]] && { log "ERROR: could not extract GitHub token after auth"; exit 2; }

if [[ "$DRY_RUN" == 1 ]]; then
  log "DRY-RUN: would write \$ENV_FILE with GITHUB_TOKEN (codespace scope)"
  exit 0
fi

umask 077
{
  echo "GITHUB_TOKEN=$TOKEN"
  echo "CODESPACES_DEFAULT_REPO=0riginal-claw/sp500-mastery-compute"
} > "$ENV_FILE"
chmod 600 "$ENV_FILE"
log "wrote $ENV_FILE (chmod 600)"

# Verify codespace scope
gh auth status 2>&1 | grep -i codespace | tee -a "$LOG_FILE" || \
  log "WARN: codespace scope may be missing. Run: gh auth refresh --scopes codespace"

# Flip enabled=true in cloud_usage.json
python3 - "$CLOUD_USAGE" "$PROVIDER" <<'PY' 2>&1 | tee -a "$LOG_FILE"
import json, sys
p, k = sys.argv[1], sys.argv[2]
d = json.load(open(p))
if k in d and isinstance(d[k], dict): d[k]["enabled"] = True
json.dump(d, open(p, "w"), indent=2)
print(f"flipped {k}.enabled=true in {p}")
PY

log "$PROVIDER setup complete. Smoke: gh codespace list"
