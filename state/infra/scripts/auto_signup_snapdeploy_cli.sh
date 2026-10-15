#!/usr/bin/env bash
# auto_signup_snapdeploy_cli.sh — Semi-CLI SnapDeploy bootstrap
# License: N/A (SnapDeploy has no public CLI/API docs). Generated 2026-05-17.
#
# SnapDeploy is fully WEB-ONLY per https://snapdeploy.dev/llms-full.md
# (verified 2026-05-17): "Visit snapdeploy.dev, Click Get Started, complete
# email verification". No documented CLI tool, no documented API key flow.
# GitHub OAuth IS supported in the web flow (snapdeploy.dev confirms
# "GitHub OAuth integration"). This script captures the token AFTER manual
# web signup and writes the env file; no CLI bypass available.

set -euo pipefail

PROVIDER="snapdeploy"
MAC_HOME="/Users/orginal"
ENV_DIR="$MAC_HOME/.config/auto_signup"
ENV_FILE="$ENV_DIR/${PROVIDER}.env"
LOG_DIR="$MAC_HOME/AI-Tools/logs/auto_signup"
CLOUD_USAGE="$MAC_HOME/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/s&p500-ticker-mastery/sweeps/cloud_usage.json"
mkdir -p "$ENV_DIR" "$LOG_DIR"
LOG_FILE="$LOG_DIR/${PROVIDER}_$(date -u +%Y%m%dT%H%M%SZ).log"

DRY_RUN=0
SNAP_TOKEN="${SNAPDEPLOY_API_TOKEN:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --token) SNAP_TOKEN="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG_FILE"; }

log "$PROVIDER setup begin (dry_run=$DRY_RUN)"

if [[ "$DRY_RUN" == 1 ]]; then
  log "DRY-RUN: would write \$ENV_FILE + flip enabled=true"
  exit 0
fi

if [[ -z "$SNAP_TOKEN" ]]; then
  cat >&2 <<EOF
ERROR: Need SNAPDEPLOY_API_TOKEN.

Per snapdeploy.dev/llms-full.md, signup is web-only. One-time web action:
  1. Visit https://snapdeploy.dev and click "Get Started"
  2. Sign in with GitHub OAuth (preferred — no email verification needed)
  3. Locate Settings -> API tokens (or contact contact@snapdeploy.dev if not
     surfaced — API may currently be private-beta)
  4. Re-run:  $0 --token <TOKEN>   OR   export SNAPDEPLOY_API_TOKEN=...

NOTE: If API tokens are not yet generally available, set this provider's
'enabled=false' in cloud_usage.json and revisit when SnapDeploy publishes
its public API.
EOF
  exit 2
fi

# Best-effort verify (endpoint inferred; ignore failures since docs are absent)
log "Best-effort token verify (snapdeploy public API undocumented)"
curl -sS -o /dev/null -w "verify HTTP %{http_code}\n" \
  -H "Authorization: Bearer $SNAP_TOKEN" \
  https://api.snapdeploy.dev/v1/me 2>&1 | tee -a "$LOG_FILE" || true

umask 077
echo "SNAPDEPLOY_API_TOKEN=$SNAP_TOKEN" > "$ENV_FILE"
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

log "$PROVIDER setup complete (caveat: API may be private-beta — verify before relying on this adapter)"
