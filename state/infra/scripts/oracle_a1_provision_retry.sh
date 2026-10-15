#!/usr/bin/env bash
# oracle_a1_provision_retry.sh
#
# Capacity-retry loop for Oracle Cloud Always-Free A1 Ampere ARM64 instance.
# A1.Flex shape (VM.Standard.A1.Flex) is heavily capacity-constrained; standard
# practice is to retry every 5-10 min until the home region releases capacity.
#
# Always-Free entitlement (verified 2026-05-20):
#   - 4 OCPU + 24 GB RAM total across all A1 instances in the tenancy (forever)
#   - 200 GB block storage total
#   - 10 TB egress/month
#   - Requires: OCI account with verified payment method (card NOT charged
#     while staying in always-free tier; verification only)
#
# Prerequisites (Phase A — one-time on Mac):
#   1. brew install oci-cli                          # or pip install oci-cli
#   2. oci setup config                              # interactive: tenancy OCID,
#                                                    # user OCID, region, key pair
#   3. Add public key to OCI Console > User Settings > API Keys
#   4. Verify: oci iam region list
#
# Required env vars (export before running, or put in ~/.oci/a1.env):
#   OCI_COMPARTMENT_OCID   - target compartment (root tenancy OCID is fine)
#   OCI_AD                 - availability domain, e.g. "XXxx:US-ASHBURN-AD-1"
#   OCI_SUBNET_OCID        - subnet OCID inside a VCN
#   OCI_IMAGE_OCID         - Canonical Ubuntu 22.04 ARM64 image OCID for region
#   OCI_SSH_PUBKEY_PATH    - default ~/.ssh/id_ed25519.pub
#   OCI_INSTANCE_NAME      - default "sp500-xsec-a1"
#   OCI_OCPUS              - default 4
#   OCI_MEM_GB             - default 24
#   OCI_RETRY_SLEEP_SEC    - default 600 (10 min)
#   OCI_MAX_ATTEMPTS       - default 0 (infinite)
#
# Exit codes: 0 success, 1 launched-but-error, 2 misconfig, 3 max-attempts-hit
#
# Reference: https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm

set -euo pipefail

LOG_DIR="${HOME}/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/logs/auto_solve"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/oracle_a1_provision_$(date -u +%Y%m%dT%H%M%SZ).log"

log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"; }

# Source env overrides if present
[ -f "${HOME}/.oci/a1.env" ] && . "${HOME}/.oci/a1.env"

: "${OCI_COMPARTMENT_OCID:?missing}"
: "${OCI_AD:?missing}"
: "${OCI_SUBNET_OCID:?missing}"
: "${OCI_IMAGE_OCID:?missing}"
OCI_SSH_PUBKEY_PATH="${OCI_SSH_PUBKEY_PATH:-$HOME/.ssh/id_ed25519.pub}"
OCI_INSTANCE_NAME="${OCI_INSTANCE_NAME:-sp500-xsec-a1}"
OCI_OCPUS="${OCI_OCPUS:-4}"
OCI_MEM_GB="${OCI_MEM_GB:-24}"
OCI_RETRY_SLEEP_SEC="${OCI_RETRY_SLEEP_SEC:-600}"
OCI_MAX_ATTEMPTS="${OCI_MAX_ATTEMPTS:-0}"

command -v oci >/dev/null 2>&1 || { log "ERR: oci CLI not installed (brew install oci-cli)"; exit 2; }
[ -f "$OCI_SSH_PUBKEY_PATH" ] || { log "ERR: SSH pubkey not found: $OCI_SSH_PUBKEY_PATH"; exit 2; }

SSH_PUBKEY=$(cat "$OCI_SSH_PUBKEY_PATH")

# Shape config JSON for A1.Flex (flex shape requires explicit OCPU + memory)
SHAPE_CONFIG=$(printf '{"ocpus":%s,"memoryInGBs":%s}' "$OCI_OCPUS" "$OCI_MEM_GB")
METADATA=$(printf '{"ssh_authorized_keys":"%s"}' "$SSH_PUBKEY")

attempt=0
while :; do
  attempt=$((attempt+1))
  log "Attempt #$attempt: launching A1.Flex (${OCI_OCPUS} OCPU / ${OCI_MEM_GB} GB) in AD=$OCI_AD"

  set +e
  RESP=$(oci compute instance launch \
    --availability-domain "$OCI_AD" \
    --compartment-id "$OCI_COMPARTMENT_OCID" \
    --shape "VM.Standard.A1.Flex" \
    --shape-config "$SHAPE_CONFIG" \
    --image-id "$OCI_IMAGE_OCID" \
    --subnet-id "$OCI_SUBNET_OCID" \
    --display-name "$OCI_INSTANCE_NAME" \
    --metadata "$METADATA" \
    --assign-public-ip true \
    --wait-for-state RUNNING \
    --max-wait-seconds 600 \
    2>&1)
  RC=$?
  set -e

  if [ $RC -eq 0 ]; then
    log "SUCCESS: instance running"
    printf '%s\n' "$RESP" | tee -a "$LOG"
    PUB_IP=$(printf '%s' "$RESP" | python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('data',{}).get('public-ip',''))" 2>/dev/null || echo "")
    [ -n "$PUB_IP" ] && log "Public IP: $PUB_IP — ssh ubuntu@$PUB_IP"
    exit 0
  fi

  if printf '%s' "$RESP" | grep -qiE "Out of host capacity|InternalError|TooManyRequests|LimitExceeded"; then
    log "Capacity error (expected for A1.Flex). Sleeping ${OCI_RETRY_SLEEP_SEC}s..."
    log "Snippet: $(printf '%s' "$RESP" | head -c 300)"
  else
    log "Non-capacity error — bailing"
    printf '%s\n' "$RESP" | tee -a "$LOG"
    exit 1
  fi

  if [ "$OCI_MAX_ATTEMPTS" -gt 0 ] && [ "$attempt" -ge "$OCI_MAX_ATTEMPTS" ]; then
    log "Max attempts ($OCI_MAX_ATTEMPTS) reached without capacity"
    exit 3
  fi

  sleep "$OCI_RETRY_SLEEP_SEC"
done
