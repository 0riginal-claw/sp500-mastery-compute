#!/usr/bin/env bash
#
# setup_alpaca_keychain.sh
#
# One-shot helper that stores Alpaca paper-trading API key + secret in the
# macOS login Keychain with an ACL that pre-authorizes the sp500-mastery
# venv python and /usr/bin/security to read them without a GUI prompt.
#
# This avoids the Monday 9:30 ET issue where a cron-launched python
# process triggers an interactive Keychain prompt and stalls.
#
# Usage:
#   ALPACA_API_KEY="..." ALPACA_SECRET_KEY="..." \
#     bash scripts/setup_alpaca_keychain.sh
#
# Credentials MUST come from the environment, not an interactive prompt,
# to avoid leaving the key value in shell history / transcripts.
#
# Verify:
#   security find-generic-password -s alpaca-paper-api-key
#   security find-generic-password -s alpaca-paper-secret-key
#
# Remove (if rotating / undoing):
#   security delete-generic-password -s alpaca-paper-api-key
#   security delete-generic-password -s alpaca-paper-secret-key

set -euo pipefail

API_KEY_SERVICE="alpaca-paper-api-key"
SECRET_SERVICE="alpaca-paper-secret-key"
ACCOUNT="${USER:?USER must be set}"

# Apps pre-authorized via ACL (-T flag). Add more -T entries below if
# additional binaries need silent access.
VENV_PYTHON="/Users/orginal/.venvs/sp500-mastery/bin/python3"
SECURITY_BIN="/usr/bin/security"

# Sanity: required env vars present and non-empty.
: "${ALPACA_API_KEY:?ALPACA_API_KEY must be set in environment (do not paste inline; export first or prefix the command)}"
: "${ALPACA_SECRET_KEY:?ALPACA_SECRET_KEY must be set in environment (do not paste inline; export first or prefix the command)}"

# Sanity: ACL target binaries exist (warn only — venv may not be present
# on every machine, but the keychain entry will still be created).
if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "WARN: $VENV_PYTHON not found or not executable — ACL entry will still be added but won't match anything until the venv is created." >&2
fi
if [[ ! -x "$SECURITY_BIN" ]]; then
    echo "WARN: $SECURITY_BIN not found — unexpected on macOS." >&2
fi

echo "Adding Alpaca paper credentials to login Keychain for account: $ACCOUNT"
echo "  service: $API_KEY_SERVICE"
echo "  service: $SECRET_SERVICE"
echo "  ACL-trusted: $VENV_PYTHON"
echo "  ACL-trusted: $SECURITY_BIN"

# -U updates if already present.
# -T <path> grants the named app silent access (no GUI prompt).
# Multiple -T flags are allowed; only the listed apps are pre-authorized.
# NOTE: -w "<value>" passes the secret value as an argv string; macOS
# scrubs argv for `security` so this does not leak via `ps`, but we
# still avoid echoing the value anywhere in this script.
security add-generic-password \
    -s "$API_KEY_SERVICE" \
    -a "$ACCOUNT" \
    -w "$ALPACA_API_KEY" \
    -T "$VENV_PYTHON" \
    -T "$SECURITY_BIN" \
    -U

security add-generic-password \
    -s "$SECRET_SERVICE" \
    -a "$ACCOUNT" \
    -w "$ALPACA_SECRET_KEY" \
    -T "$VENV_PYTHON" \
    -T "$SECURITY_BIN" \
    -U

# Verify presence (does NOT print secret value — find-generic-password
# without -w only prints metadata).
verify() {
    local svc="$1"
    if security find-generic-password -s "$svc" >/dev/null 2>&1; then
        echo "$svc: OK"
    else
        echo "$svc: FAIL"
        return 1
    fi
}

verify "$API_KEY_SERVICE"
verify "$SECRET_SERVICE"

echo ""
echo "Done. Services added:"
echo "  1) $API_KEY_SERVICE  (account: $ACCOUNT)"
echo "  2) $SECRET_SERVICE   (account: $ACCOUNT)"
echo ""
echo "Python clients can now read these without a GUI prompt via, e.g.:"
echo "  subprocess.check_output(['security', 'find-generic-password',"
echo "                           '-s', '$API_KEY_SERVICE', '-w'])"
