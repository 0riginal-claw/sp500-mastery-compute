#!/usr/bin/env bash
# zg_chain_join — one-liner onboarding for a new ZG chain node.
# autosolve_skip: own blockchain + ZGC token (Tier-1 build, leaf)
#
# Usage (run on the new machine):
#   curl -fsSL https://raw.githubusercontent.com/<owner>/<repo>/main/scripts/zg_chain_join.sh \
#     | bash -s -- --bootnode http://<existing>:9933 --node-id <name>
#
# Or, with the script already on disk:
#   bash scripts/zg_chain_join.sh --bootnode http://<existing>:9933 --node-id <name>
#
# Defaults:
#   --bootnode   (required) URL of an existing node to fetch the chain from.
#   --node-id    hostname
#   --addr       auto-derived ZG address (zg1 + sha256(node_id)[:36])
#   --stake      1.0
#   --port       9933
#   --home       $HOME  (chain stored at $HOME/.zg/state/universal_resume/chain.jsonl)
#   --workdir    $HOME/.zg-chain   (script + venv land here)
#   --no-start   bootstrap only, do not start the daemon

set -euo pipefail

BOOTNODE=""
NODE_ID="$(hostname -s 2>/dev/null || hostname)"
ADDR=""
STAKE="1.0"
PORT="9933"
WORKDIR="${HOME}/.zg-chain"
NO_START=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bootnode)  BOOTNODE="$2"; shift 2 ;;
    --node-id)   NODE_ID="$2"; shift 2 ;;
    --addr)      ADDR="$2"; shift 2 ;;
    --stake)     STAKE="$2"; shift 2 ;;
    --port)      PORT="$2"; shift 2 ;;
    --workdir)   WORKDIR="$2"; shift 2 ;;
    --no-start)  NO_START=1; shift ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *)  echo "[zg_chain_join] unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$BOOTNODE" ]]; then
  echo "[zg_chain_join] --bootnode is required" >&2
  exit 2
fi

if [[ -z "$ADDR" ]]; then
  ADDR="zg1$(printf '%s' "$NODE_ID" | shasum -a 256 | awk '{print $1}' | cut -c1-36)"
fi

echo "[zg_chain_join] node-id=$NODE_ID"
echo "[zg_chain_join] addr=$ADDR"
echo "[zg_chain_join] bootnode=$BOOTNODE"
echo "[zg_chain_join] workdir=$WORKDIR"

mkdir -p "$WORKDIR" "$HOME/.zg/state/universal_resume" "$HOME/.zg/state/zgc/keys"

fetch_scripts() {
  local dst="$1"
  if [[ -f "scripts/zg_chain_node.py" && -f "scripts/merkle_chain.py" ]]; then
    cp scripts/zg_chain_node.py scripts/merkle_chain.py "$dst/"
    return
  fi
  local base="${ZG_RAW_BASE:-https://raw.githubusercontent.com/0riginal-claw/sp500-mastery-compute/main/scripts}"
  echo "[zg_chain_join] fetching scripts from $base"
  for f in zg_chain_node.py merkle_chain.py; do
    if ! curl -fsSL "$base/$f" -o "$dst/$f"; then
      echo "[zg_chain_join] FAILED to fetch $f from $base" >&2
      echo "[zg_chain_join] tip: set ZG_RAW_BASE to a mirror or copy scripts/ manually" >&2
      return 1
    fi
  done
}

fetch_scripts "$WORKDIR"
chmod +x "$WORKDIR/zg_chain_node.py" || true

echo "[zg_chain_join] fetching chain head from bootnode"
HEAD_JSON="$(curl -fsS -X POST -H 'Content-Type: application/json' "$BOOTNODE/head" -d '{}')"
echo "[zg_chain_join] bootnode head: $HEAD_JSON"

CHAIN_FILE="$HOME/.zg/state/universal_resume/chain.jsonl"
if [[ ! -s "$CHAIN_FILE" ]]; then
  echo "[zg_chain_join] seeding local chain from bootnode"
  curl -fsS -X POST -H 'Content-Type: application/json' \
    "$BOOTNODE/blocks" -d '{"since_seq":-1}' \
    | python3 -c '
import json, sys, os
blocks = json.load(sys.stdin)
path = os.path.expanduser("~/.zg/state/universal_resume/chain.jsonl")
with open(path, "w") as f:
    for b in blocks:
        f.write(json.dumps(b, separators=(",", ":")) + "\n")
print(f"[zg_chain_join] wrote {len(blocks)} blocks to {path}")
'
else
  echo "[zg_chain_join] local chain already present, will sync via daemon"
fi

echo "[zg_chain_join] registering validator addr=$ADDR stake=$STAKE on bootnode"
curl -fsS -X POST -H 'Content-Type: application/json' \
  "$BOOTNODE/register_validator" \
  -d "{\"addr\":\"$ADDR\",\"stake\":$STAKE}" || \
  echo "[zg_chain_join] WARN: bootnode register_validator failed (continuing)"

if [[ "$NO_START" -eq 1 ]]; then
  echo "[zg_chain_join] --no-start set, skipping daemon launch"
  echo "[zg_chain_join] start manually with:"
  echo "  python3 $WORKDIR/zg_chain_node.py serve --port $PORT --node-id $NODE_ID --peers $BOOTNODE"
  exit 0
fi

PID_FILE="$WORKDIR/node.pid"
LOG_FILE="$WORKDIR/node.log"
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "[zg_chain_join] node already running pid=$(cat "$PID_FILE")"
else
  echo "[zg_chain_join] launching node in background"
  nohup python3 "$WORKDIR/zg_chain_node.py" serve \
    --port "$PORT" \
    --node-id "$NODE_ID" \
    --peers "$BOOTNODE" \
    > "$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"
  sleep 2
  if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "[zg_chain_join] node started pid=$(cat "$PID_FILE") log=$LOG_FILE"
  else
    echo "[zg_chain_join] node failed to start, see $LOG_FILE" >&2
    tail -n 40 "$LOG_FILE" >&2 || true
    exit 1
  fi
fi

sleep 1
echo "[zg_chain_join] local health:"
curl -fsS -X POST -H 'Content-Type: application/json' "http://127.0.0.1:$PORT/health" -d '{}' || true
echo
echo "[zg_chain_join] done. addr=$ADDR validator stake=$STAKE"
echo "[zg_chain_join] earnings: 1 ZGC per accepted block. check balance with:"
echo "  python3 $WORKDIR/zg_chain_node.py balance --addr $ADDR"
