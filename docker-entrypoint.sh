#!/usr/bin/env bash
# docker-entrypoint.sh — start the ZG chain node from a single ENV-driven config.
#
# Vars (all optional except NODE_ID):
#   NODE_ID            unique node identifier (default: $HOSTNAME)
#   PORT               RPC port (default 9933)
#   PEERS              comma-separated peer URLs (default: empty — go-it-alone)
#   VALIDATOR_ADDR     ZG address to credit block rewards (default: derived from NODE_ID)
#   VALIDATOR_STAKE    stake amount (default 1.0)
#   SYNC_SECS          peer-poll interval seconds (default 15)
#   VALIDATOR_REWARD   ZGC per accepted block (default 1.0)
#   ZG_HOME            state root (default /data inside container)
#   EXTRA_ARGS         extra CLI flags to forward to zg_chain_node.py
#
# CLI tail (default `serve`):
#   serve         start the daemon (default)
#   status        print local chain head + balance and exit
#   <anything>    forwarded as-is to scripts/zg_chain_node.py

set -euo pipefail

NODE_ID="${NODE_ID:-$(hostname)}"
PORT="${PORT:-9933}"
PEERS="${PEERS:-}"
VALIDATOR_STAKE="${VALIDATOR_STAKE:-1.0}"
SYNC_SECS="${SYNC_SECS:-15}"
VALIDATOR_REWARD="${VALIDATOR_REWARD:-1.0}"
ZG_HOME="${ZG_HOME:-/data}"
export ZG_HOME

mkdir -p "${ZG_HOME}/.zg/state/universal_resume" "${ZG_HOME}/.zg/state/zgc/keys"

SUB="${1:-serve}"
shift || true

case "${SUB}" in
  serve)
    PEER_ARGS=()
    if [[ -n "${PEERS}" ]]; then
      PEER_ARGS=(--peers "${PEERS}")
    fi
    VADDR_ARGS=()
    if [[ -n "${VALIDATOR_ADDR:-}" ]]; then
      VADDR_ARGS=(--validator-addr "${VALIDATOR_ADDR}")
    fi

    echo "[zg-node] starting NODE_ID=${NODE_ID} port=${PORT} peers=${PEERS:-<none>} home=${ZG_HOME}"

    exec python3 /app/scripts/zg_chain_node.py serve \
      --port "${PORT}" \
      --node-id "${NODE_ID}" \
      --sync-secs "${SYNC_SECS}" \
      --validator-reward "${VALIDATOR_REWARD}" \
      --validator-stake "${VALIDATOR_STAKE}" \
      "${PEER_ARGS[@]}" \
      "${VADDR_ARGS[@]}" \
      ${EXTRA_ARGS:-} \
      "$@"
    ;;

  status|balance|transfer|register_validator)
    exec python3 /app/scripts/zg_chain_node.py "${SUB}" "$@"
    ;;

  *)
    # Forward unknown commands directly.
    exec python3 /app/scripts/zg_chain_node.py "${SUB}" "$@"
    ;;
esac
