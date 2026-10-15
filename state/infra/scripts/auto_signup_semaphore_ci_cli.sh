#!/usr/bin/env bash
# auto_signup_semaphore_ci_cli.sh — Semi-CLI Semaphore CI bootstrap
# License: sem CLI is Apache-2.0. Generated 2026-05-17.
#
# Semaphore CI requires ONE-TIME web action to:
#   (a) sign up via GitHub OAuth at https://semaphore.io (no email/phone)
#   (b) create org URL (e.g. myorg.semaphoreci.com)
#   (c) generate API token at https://me.semaphoreci.com/account
# After those, this CLI bootstrap handles all subsequent ops. The web step
# is unavoidable per docs.semaphore.io/reference/semaphore-cli (verified
# 2026-05-17): "sem connect <org-url>.semaphoreci.com <API_TOKEN>" requires
# pre-existing org + token.

set -euo pipefail

PROVIDER="semaphore_ci"
MAC_HOME="/Users/orginal"
ENV_DIR="$MAC_HOME/.config/auto_signup"
ENV_FILE="$ENV_DIR/${PROVIDER}.env"
LOG_DIR="$MAC_HOME/AI-Tools/logs/auto_signup"
CLOUD_USAGE="$MAC_HOME/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/s&p500-ticker-mastery/sweeps/cloud_usage.json"
mkdir -p "$ENV_DIR" "$LOG_DIR"
LOG_FILE="$LOG_DIR/${PROVIDER}_$(date -u +%Y%m%dT%H%M%SZ).log"

DRY_RUN=0
SEM_TOKEN="${SEMAPHORE_API_TOKEN:-}"
SEM_ORG="${SEMAPHORE_ORG_URL:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --token) SEM_TOKEN="$2"; shift 2 ;;
    --org) SEM_ORG="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG_FILE"; }

log "$PROVIDER CLI signup begin (dry_run=$DRY_RUN)"

# Install sem CLI via official tap (MIT)
if ! command -v sem >/dev/null 2>&1; then
  if [[ "$DRY_RUN" == 1 ]]; then
    log "DRY-RUN: brew install semaphoreci/tap/sem"
  else
    log "Installing sem via brew tap (Apache-2.0)"
    brew install semaphoreci/tap/sem 2>&1 | tee -a "$LOG_FILE"
  fi
fi

if [[ "$DRY_RUN" == 1 ]]; then
  log "DRY-RUN: sem connect <org>.semaphoreci.com <token>"
  exit 0
fi

# Token + org required — one-time web action prereq
if [[ -z "$SEM_TOKEN" ]] || [[ -z "$SEM_ORG" ]]; then
  cat >&2 <<EOF
ERROR: Need SEMAPHORE_API_TOKEN + SEMAPHORE_ORG_URL.

One-time web action required (no CLI bypass available):
  1. Visit https://semaphore.io and sign up via GitHub OAuth (no email/phone)
  2. Create an org — you'll get a URL like myorg.semaphoreci.com
  3. Visit https://me.semaphoreci.com/account and generate an API token
  4. Re-run:  $0 --org myorg.semaphoreci.com --token <TOKEN>
     OR export SEMAPHORE_API_TOKEN=... SEMAPHORE_ORG_URL=...
EOF
  exit 2
fi

# Connect + persist token
sem connect "$SEM_ORG" "$SEM_TOKEN" 2>&1 | tee -a "$LOG_FILE"

umask 077
{
  echo "SEMAPHORE_API_TOKEN=$SEM_TOKEN"
  echo "SEMAPHORE_ORG_URL=$SEM_ORG"
} > "$ENV_FILE"
chmod 600 "$ENV_FILE"
log "wrote $ENV_FILE (chmod 600)"

# Flip enabled=true in cloud_usage.json
python3 - "$CLOUD_USAGE" "$PROVIDER" <<'PY' 2>&1 | tee -a "$LOG_FILE"
import json, sys
p, k = sys.argv[1], sys.argv[2]
d = json.load(open(p))
if k in d and isinstance(d[k], dict): d[k]["enabled"] = True
json.dump(d, open(p, "w"), indent=2)
print(f"flipped {k}.enabled=true in {p}")
PY

log "$PROVIDER setup complete. Smoke: sem version && sem projects list"
