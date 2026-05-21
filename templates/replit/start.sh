#!/usr/bin/env bash
# start.sh — boot a ZG validator inside a Replit container.
#
# Used by .replit's `run = "bash start.sh"`.

set -euo pipefail

NODE_ID="${REPL_OWNER:-replit}-${REPL_SLUG:-zg-node}"
PORT="${PORT:-9933}"
PEERS="${PEERS:-https://seed1.zgc.run,https://seed2.zgc.run}"
VALIDATOR_ADDR="${ZG_VALIDATOR_ADDR:-}"

REPO_BASE="${ZG_REPO_BASE:-https://raw.githubusercontent.com/0riginal-claw/sp500-mastery-compute/main}"

mkdir -p ~/.zg/state/universal_resume ~/.zg/state/zgc/keys scripts

# Pull latest scripts (so this template self-updates on each boot).
for f in zg_chain_node.py merkle_chain.py; do
  if [ ! -f "scripts/${f}" ] || [ -n "${ZG_PULL_LATEST:-}" ]; then
    curl -fsSL "${REPO_BASE}/scripts/${f}" -o "scripts/${f}"
  fi
done

VARGS=()
if [ -n "${VALIDATOR_ADDR}" ]; then
  VARGS+=(--validator-addr "${VALIDATOR_ADDR}")
fi

echo "[replit] booting node ${NODE_ID} on :${PORT} peers=${PEERS}"

exec python3 scripts/zg_chain_node.py serve \
  --port "${PORT}" \
  --node-id "${NODE_ID}" \
  --peers "${PEERS}" \
  --sync-secs 15 \
  --validator-reward 1.0 \
  "${VARGS[@]}"
