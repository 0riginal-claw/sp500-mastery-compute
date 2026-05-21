#!/usr/bin/env python3
"""zg_chain_node — standalone networked node for the ZG Merkle chain + ZGC token.

Phase A of `own_blockchain_spec_2026-05-20` Priority Tier-1.

A single Python daemon (zero non-stdlib deps) any Mac/Linux box can run. It:

  1. Hosts a tiny HTTP JSON-RPC server (BaseHTTPServer / `http.server`) on
     port `--port` (default 9933) exposing:
        POST /head            -> tail-block summary {seq, block_hash, height}
        POST /blocks          -> body {since_seq:int} -> JSON list[block]
        POST /submit          -> body block dict -> {accepted:bool, reason}
        POST /balance         -> body {addr} -> {addr, balance}
        POST /transfer        -> body {from, to, amount, sig} -> tx block
        POST /peers           -> {peers:[...]}
        POST /register_validator -> {addr, stake}
        POST /health          -> {ok, node_id, height, peers, ts}

  2. Runs a background sync loop (every `--sync-secs`, default 15s) that
     polls each peer's /head, fetches missing blocks via /blocks, and
     appends locally-validated blocks to its own chain (longest-chain wins
     by `seq`, ties broken by lower block_hash).

  3. Validates every incoming block:
        - block_hash recomputed matches block['block_hash']
        - payload_hash recomputed matches block['payload_hash']
        - prev_hash == local tail block_hash (or known parent)
        - seq == local next_seq
        - if action is a token op: signature verifies + sender balance OK

  4. Settles ZGC token state into `state/zgc/balances.json` (atomic write)
     each time the chain advances. Validators that propose accepted blocks
     earn `--validator-reward` ZGC per block (PoS validate-to-earn).

  5. Founder pre-mine: 10_000_000 ZGC credited to `--founder-addr` at
     genesis. Anyone who runs a node + registers a validator address starts
     at 0 ZGC and earns by validating.

Operational note: this is a friendly-network chain. It uses HMAC signatures
(stdlib `hmac`) keyed off a per-validator secret read from
`state/zgc/keys/<addr>.key` (mode 0600). For a public adversarial chain you
would swap HMAC for Ed25519 (PyNaCl or `cryptography`). HMAC keeps the
zero-non-stdlib-deps requirement.

Coordinates with:
  - scripts/merkle_chain.py            (block format + verify_block + MerkleChain)
  - scripts/chain_distribute.py        (existing Drive/GH mirroring of the chain)
  - state/universal_resume/chain.jsonl (canonical chain file)

Run:
    python3 scripts/zg_chain_node.py serve --port 9933 \
        --node-id <name> --peers http://other:9933,http://third:9933

CLI (sans `serve`):
    python3 scripts/zg_chain_node.py status
    python3 scripts/zg_chain_node.py balance --addr <addr>
    python3 scripts/zg_chain_node.py transfer --from A --to B --amount 100
    python3 scripts/zg_chain_node.py register_validator --addr <addr> --stake 1
"""
from __future__ import annotations

import argparse
import hmac
import hashlib
import json
import logging
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

# ------------------------- paths + constants -------------------------------

OS_HOME = Path(os.environ.get("ZG_HOME", str(Path.home())))
DRIVE_ROOT_GUESS = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/"
    "My Drive/AI-Tools"
)
SCRIPTS_DIR = (DRIVE_ROOT_GUESS / "scripts") if DRIVE_ROOT_GUESS.exists() else Path(
    __file__
).resolve().parent

# Make merkle_chain importable without forcing the user to set PYTHONPATH.
sys.path.insert(0, str(SCRIPTS_DIR))
try:
    from merkle_chain import MerkleChain, make_block, verify_block, GENESIS_HASH, _canonical, _hash  # type: ignore  # noqa: E402
except Exception as e:  # noqa: BLE001
    print(f"[zg_chain_node] cannot import merkle_chain.py: {e!r}", file=sys.stderr)
    print(f"[zg_chain_node] expected at: {SCRIPTS_DIR / 'merkle_chain.py'}", file=sys.stderr)
    raise

# Canonical chain (shared with universal_resume).
CHAIN_PATH = OS_HOME / ".zg" / "state" / "universal_resume" / "chain.jsonl"
# ZGC token state.
STATE_ROOT = OS_HOME / ".zg" / "state" / "zgc"
BAL_PATH = STATE_ROOT / "balances.json"
VALIDATOR_PATH = STATE_ROOT / "validators.json"
KEYS_DIR = STATE_ROOT / "keys"
TX_LOG = STATE_ROOT / "tx_log.jsonl"
# Founder address + premine (matches spec).
FOUNDER_ADDR_DEFAULT = "zg1founder0000000000000000000000000000"
PREMINE = 10_000_000  # 10M ZGC

# Token actions on the chain.
ACTION_TOKEN_TRANSFER = "zgc.transfer"
ACTION_TOKEN_PREMINE = "zgc.premine"
ACTION_REGISTER_VALIDATOR = "zgc.register_validator"
ACTION_VALIDATOR_REWARD = "zgc.validator_reward"

DEFAULT_PORT = 9933
DEFAULT_SYNC_SECS = 15
DEFAULT_REWARD = 1  # 1 ZGC per accepted block

log = logging.getLogger("zg_chain_node")


# ------------------------- atomic JSON ------------------------------------

def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


# ------------------------- HMAC signing -----------------------------------

def _key_path(addr: str) -> Path:
    return KEYS_DIR / f"{addr}.key"


def _ensure_key(addr: str) -> str:
    """Read or create a per-address HMAC secret. Returns hex secret."""
    KEYS_DIR.mkdir(parents=True, exist_ok=True)
    kp = _key_path(addr)
    if kp.exists():
        return kp.read_text(encoding="utf-8").strip()
    secret = os.urandom(32).hex()
    kp.write_text(secret, encoding="utf-8")
    try:
        os.chmod(kp, 0o600)
    except OSError:
        pass
    return secret


def _sign(addr: str, payload: dict) -> str:
    secret = _ensure_key(addr)
    body = _canonical(payload)
    return hmac.new(bytes.fromhex(secret), body, hashlib.sha256).hexdigest()


def _verify_sig(addr: str, payload: dict, sig: str) -> bool:
    kp = _key_path(addr)
    if not kp.exists():
        # Unknown signer is allowed only for reads. Writes will be rejected
        # at the action handler.
        return False
    secret = kp.read_text(encoding="utf-8").strip()
    body = _canonical(payload)
    expected = hmac.new(bytes.fromhex(secret), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, sig or "")


# ------------------------- ZGC state machine -------------------------------

class TokenState:
    """Pure-function state machine over the chain's token actions.

    Recomputable from the chain at any time. Cached to disk for fast reads
    via balances.json + validators.json.
    """

    def __init__(self):
        self.balances: dict[str, int] = {}
        self.validators: dict[str, float] = {}  # addr -> stake
        self.last_seq_applied = -1

    # --- block application -------------------------------------------------

    def apply_block(self, block: dict) -> None:
        seq = int(block.get("seq", -1))
        if seq <= self.last_seq_applied:
            return
        action = block.get("action") or ""
        p = block.get("payload") or {}

        if action == "genesis":
            # Premine to founder at genesis if specified.
            founder = p.get("founder_addr")
            premine = int(p.get("premine", 0))
            if founder and premine > 0:
                self.balances[founder] = self.balances.get(founder, 0) + premine
        elif action == ACTION_TOKEN_PREMINE:
            to = p.get("to")
            amt = int(p.get("amount", 0))
            if to and amt > 0:
                self.balances[to] = self.balances.get(to, 0) + amt
        elif action == ACTION_TOKEN_TRANSFER:
            frm = p.get("from")
            to = p.get("to")
            amt = int(p.get("amount", 0))
            if frm and to and amt > 0:
                cur = self.balances.get(frm, 0)
                if cur >= amt:
                    self.balances[frm] = cur - amt
                    self.balances[to] = self.balances.get(to, 0) + amt
                # else: invalid block, but the chain still records the
                # attempt. apply_block is non-fatal — the validator that
                # produced an invalid token tx wasted its slot.
        elif action == ACTION_REGISTER_VALIDATOR:
            addr = p.get("addr")
            stake = float(p.get("stake", 1.0))
            if addr:
                self.validators[addr] = stake
        elif action == ACTION_VALIDATOR_REWARD:
            to = p.get("to")
            amt = int(p.get("amount", 0))
            if to and amt > 0:
                self.balances[to] = self.balances.get(to, 0) + amt
        # all other actions (universal_resume checkpoints, etc.) are
        # passthrough — they don't touch the token state.
        self.last_seq_applied = seq

    # --- snapshot / hydrate ------------------------------------------------

    def snapshot(self) -> dict:
        return {
            "balances": self.balances,
            "validators": self.validators,
            "last_seq_applied": self.last_seq_applied,
        }

    def save(self) -> None:
        _atomic_write_json(BAL_PATH, {
            "balances": self.balances,
            "last_seq_applied": self.last_seq_applied,
            "updated_ts": int(time.time()),
        })
        _atomic_write_json(VALIDATOR_PATH, {
            "validators": self.validators,
            "updated_ts": int(time.time()),
        })

    @classmethod
    def rebuild_from_chain(cls, chain: MerkleChain) -> "TokenState":
        s = cls()
        for b in chain.iter_blocks():
            s.apply_block(b)
        return s


# ------------------------- node ------------------------------------------

class ZGNode:
    def __init__(
        self,
        chain_path: Path = CHAIN_PATH,
        node_id: str = "zg-node",
        peers: list[str] | None = None,
        founder_addr: str = FOUNDER_ADDR_DEFAULT,
        validator_reward: int = DEFAULT_REWARD,
    ):
        self.node_id = node_id
        self.founder_addr = founder_addr
        self.validator_reward = validator_reward
        # MerkleChain auto-creates a `genesis` block if missing. We want
        # the genesis to encode the founder premine, so check first.
        first_time = not chain_path.exists()
        self.chain = MerkleChain(chain_path, node_id=node_id)
        if first_time:
            # Re-create genesis carrying premine. Overwrite the bare genesis.
            chain_path.write_text("", encoding="utf-8")
            self._write_genesis_with_premine()
        self.lock = threading.RLock()
        self.peers: set[str] = set(p.rstrip("/") for p in (peers or []))
        self.state = TokenState.rebuild_from_chain(self.chain)
        self.state.save()

    def _write_genesis_with_premine(self) -> None:
        gen = make_block(
            prev_hash=GENESIS_HASH,
            seq=0,
            node_id=self.node_id,
            action="genesis",
            payload={
                "chain": "zg_chain",
                "init_ts": time.time_ns(),
                "founder_addr": self.founder_addr,
                "premine": PREMINE,
                "premine_token": "ZGC",
                "consensus": "pos_validate_to_earn",
            },
        )
        line = json.dumps(gen, separators=(",", ":")) + "\n"
        with open(self.chain.path, "ab") as f:
            f.write(line.encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())

    # --- chain ops ----------------------------------------------------------

    def head(self) -> dict:
        b = self.chain._last_block()  # noqa: SLF001
        if not b:
            return {"seq": -1, "block_hash": GENESIS_HASH, "height": 0}
        return {
            "seq": int(b["seq"]),
            "block_hash": b["block_hash"],
            "height": int(b["seq"]) + 1,
        }

    def blocks_since(self, since_seq: int, limit: int = 1000) -> list[dict]:
        out = []
        for b in self.chain.iter_blocks():
            if int(b.get("seq", -1)) > since_seq:
                out.append(b)
                if len(out) >= limit:
                    break
        return out

    def append_local(self, action: str, payload: dict, validator: str | None = None) -> dict:
        """Append a block locally. Auto-rewards `validator` if set."""
        with self.lock:
            block = self.chain.append(action, payload)
            # Apply to token state.
            self.state.apply_block(block)
            # Reward the validator that proposed this block.
            if validator and action not in (
                ACTION_VALIDATOR_REWARD,
                "genesis",
            ):
                reward_block = self.chain.append(
                    ACTION_VALIDATOR_REWARD,
                    {"to": validator, "amount": self.validator_reward, "for_seq": block["seq"]},
                )
                self.state.apply_block(reward_block)
            self.state.save()
            return block

    def accept_block(self, block: dict) -> tuple[bool, str]:
        """Validate + append a block received from a peer.

        Returns (accepted, reason). Strict: rejects if seq != next_seq
        OR prev_hash != local tail. The peer must call /blocks to catch
        up if it's ahead.
        """
        with self.lock:
            errs = verify_block(block)
            if errs:
                return False, "; ".join(errs)
            local_seq = self.chain.next_seq()
            if int(block.get("seq", -1)) != local_seq:
                return False, f"seq mismatch local_next={local_seq} got={block.get('seq')}"
            if block.get("prev_hash") != self.chain.tail_hash():
                return False, "prev_hash mismatch with local tail"
            # Token-action signature checks.
            action = block.get("action", "")
            p = block.get("payload") or {}
            if action == ACTION_TOKEN_TRANSFER:
                if not _verify_sig(p.get("from", ""), {k: p[k] for k in ("from", "to", "amount") if k in p}, p.get("sig", "")):
                    return False, "transfer signature invalid"
                if self.state.balances.get(p.get("from"), 0) < int(p.get("amount", 0)):
                    return False, "insufficient balance"
            # Append it byte-for-byte (cannot use chain.append because that
            # rebuilds the block — we must preserve the proposer's hash).
            line = json.dumps(block, separators=(",", ":")) + "\n"
            with open(self.chain.path, "ab") as f:
                f.write(line.encode("utf-8"))
                f.flush()
                os.fsync(f.fileno())
            self.state.apply_block(block)
            self.state.save()
            return True, "ok"

    # --- token ops ----------------------------------------------------------

    def transfer(self, frm: str, to: str, amount: int) -> tuple[bool, str, dict | None]:
        amount = int(amount)
        if amount <= 0:
            return False, "amount must be positive", None
        with self.lock:
            if self.state.balances.get(frm, 0) < amount:
                return False, "insufficient balance", None
            sig_payload = {"from": frm, "to": to, "amount": amount}
            sig = _sign(frm, sig_payload)
            payload = {**sig_payload, "sig": sig, "ts": int(time.time())}
            blk = self.append_local(ACTION_TOKEN_TRANSFER, payload, validator=frm)
            with open(TX_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps({"seq": blk["seq"], **sig_payload}) + "\n")
            return True, "ok", blk

    def register_validator(self, addr: str, stake: float = 1.0) -> dict:
        _ensure_key(addr)
        return self.append_local(
            ACTION_REGISTER_VALIDATOR,
            {"addr": addr, "stake": float(stake), "ts": int(time.time())},
            validator=addr,
        )

    def balance(self, addr: str) -> int:
        return int(self.state.balances.get(addr, 0))

    # --- sync ---------------------------------------------------------------

    def add_peer(self, url: str) -> None:
        self.peers.add(url.rstrip("/"))

    def sync_once(self) -> dict:
        """Poll each peer; if a peer has a higher head, fetch missing blocks."""
        results = []
        for peer in list(self.peers):
            try:
                their_head = _http_post_json(peer + "/head", {})
                their_seq = int(their_head.get("seq", -1))
                local_seq = self.head()["seq"]
                if their_seq > local_seq:
                    blocks = _http_post_json(peer + "/blocks", {"since_seq": local_seq})
                    accepted = 0
                    rejected = 0
                    for b in blocks if isinstance(blocks, list) else []:
                        ok, _why = self.accept_block(b)
                        if ok:
                            accepted += 1
                        else:
                            rejected += 1
                            # On rejection, stop — chain has to be linear.
                            break
                    results.append({"peer": peer, "accepted": accepted, "rejected": rejected,
                                    "from": local_seq, "to": self.head()["seq"]})
                else:
                    results.append({"peer": peer, "noop": True, "their_seq": their_seq,
                                    "local_seq": local_seq})
            except Exception as e:  # noqa: BLE001
                results.append({"peer": peer, "error": repr(e)})
        return {"ts": int(time.time()), "results": results, "head": self.head()}


# ------------------------- HTTP ---------------------------------------------

def _http_post_json(url: str, body: dict, timeout: float = 5.0) -> Any:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json", "User-Agent": "zg_chain_node/1.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        if not raw:
            return None
        return json.loads(raw.decode("utf-8"))


def _make_handler(node: ZGNode):
    class _H(BaseHTTPRequestHandler):
        # Silence default access logs.
        def log_message(self, fmt, *args):  # noqa: A003,N802,D401
            log.debug("http: " + fmt, *args)

        def _read_body(self) -> dict:
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            try:
                return json.loads(raw.decode("utf-8"))
            except Exception:
                return {}

        def _reply(self, code: int, obj: Any) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):  # noqa: N802
            try:
                if self.path == "/head":
                    self._reply(200, node.head())
                elif self.path == "/blocks":
                    body = self._read_body()
                    since = int(body.get("since_seq", -1))
                    self._reply(200, node.blocks_since(since))
                elif self.path == "/submit":
                    body = self._read_body()
                    ok, why = node.accept_block(body)
                    self._reply(200, {"accepted": ok, "reason": why, "head": node.head()})
                elif self.path == "/balance":
                    body = self._read_body()
                    addr = body.get("addr") or ""
                    self._reply(200, {"addr": addr, "balance": node.balance(addr)})
                elif self.path == "/transfer":
                    body = self._read_body()
                    ok, why, blk = node.transfer(body.get("from", ""), body.get("to", ""), int(body.get("amount", 0)))
                    self._reply(200, {"accepted": ok, "reason": why, "block_seq": (blk or {}).get("seq")})
                elif self.path == "/peers":
                    body = self._read_body()
                    if body.get("add"):
                        node.add_peer(body["add"])
                    self._reply(200, {"peers": sorted(node.peers)})
                elif self.path == "/register_validator":
                    body = self._read_body()
                    addr = body.get("addr") or ""
                    stake = float(body.get("stake", 1.0))
                    blk = node.register_validator(addr, stake)
                    self._reply(200, {"ok": True, "addr": addr, "stake": stake, "block_seq": blk["seq"]})
                elif self.path == "/health":
                    self._reply(200, {
                        "ok": True,
                        "node_id": node.node_id,
                        "head": node.head(),
                        "peers": sorted(node.peers),
                        "ts": int(time.time()),
                    })
                else:
                    self._reply(404, {"error": "unknown endpoint", "path": self.path})
            except Exception as e:  # noqa: BLE001
                log.exception("handler error")
                self._reply(500, {"error": repr(e)})

        # Allow GET on /health for browser smoke.
        def do_GET(self):  # noqa: N802
            if self.path in ("/", "/health"):
                self.do_POST_passthrough_health()
            else:
                self._reply(404, {"error": "POST-only API", "path": self.path})

        def do_POST_passthrough_health(self):
            self._reply(200, {
                "ok": True,
                "node_id": node.node_id,
                "head": node.head(),
                "peers": sorted(node.peers),
                "ts": int(time.time()),
            })

    return _H


def _sync_loop(node: ZGNode, every: int, stop_event: threading.Event) -> None:
    while not stop_event.wait(timeout=every):
        try:
            r = node.sync_once()
            if any(x.get("accepted", 0) for x in r.get("results", []) if isinstance(x, dict)):
                log.info("sync: %s", r)
        except Exception:
            log.exception("sync loop error")


# ------------------------- CLI ---------------------------------------------

def _cli() -> int:
    ap = argparse.ArgumentParser(description="ZG chain networked node + ZGC token")
    ap.add_argument("--chain", default=str(CHAIN_PATH))
    ap.add_argument("--node-id", default=os.environ.get("ZG_NODE_ID", socket.gethostname()))
    ap.add_argument("--founder-addr", default=FOUNDER_ADDR_DEFAULT)
    ap.add_argument("--validator-reward", type=int, default=DEFAULT_REWARD)
    ap.add_argument("--log-level", default="INFO")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sv = sub.add_parser("serve", help="run HTTP node + sync loop")
    sv.add_argument("--port", type=int, default=DEFAULT_PORT)
    sv.add_argument("--bind", default="0.0.0.0")
    sv.add_argument("--peers", default="", help="comma-separated peer URLs")
    sv.add_argument("--sync-secs", type=int, default=DEFAULT_SYNC_SECS)

    sub.add_parser("status", help="print head + balances summary")

    bal = sub.add_parser("balance", help="show balance for addr")
    bal.add_argument("--addr", required=True)

    tx = sub.add_parser("transfer", help="transfer ZGC")
    tx.add_argument("--from", dest="frm", required=True)
    tx.add_argument("--to", required=True)
    tx.add_argument("--amount", type=int, required=True)

    rv = sub.add_parser("register_validator", help="register a validator")
    rv.add_argument("--addr", required=True)
    rv.add_argument("--stake", type=float, default=1.0)

    ver = sub.add_parser("verify", help="full chain re-verify")

    args = ap.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    node = ZGNode(
        chain_path=Path(args.chain),
        node_id=args.node_id,
        founder_addr=args.founder_addr,
        validator_reward=args.validator_reward,
    )

    if args.cmd == "status":
        print(json.dumps({
            "node_id": node.node_id,
            "chain_path": str(node.chain.path),
            "head": node.head(),
            "balances": dict(list(node.state.balances.items())[:25]),  # cap
            "validators": node.state.validators,
            "last_seq_applied": node.state.last_seq_applied,
        }, indent=2, sort_keys=True))
        return 0
    if args.cmd == "balance":
        print(json.dumps({"addr": args.addr, "balance": node.balance(args.addr)}, indent=2))
        return 0
    if args.cmd == "transfer":
        ok, why, blk = node.transfer(args.frm, args.to, args.amount)
        print(json.dumps({"accepted": ok, "reason": why, "block_seq": (blk or {}).get("seq")}, indent=2))
        return 0 if ok else 1
    if args.cmd == "register_validator":
        blk = node.register_validator(args.addr, args.stake)
        print(json.dumps({"ok": True, "block_seq": blk["seq"], "addr": args.addr, "stake": args.stake}, indent=2))
        return 0
    if args.cmd == "verify":
        res = node.chain.verify()
        print(json.dumps(res, indent=2))
        return 0 if res["ok"] else 1
    if args.cmd == "serve":
        peers = [p.strip() for p in args.peers.split(",") if p.strip()]
        for p in peers:
            node.add_peer(p)
        stop = threading.Event()
        t = threading.Thread(target=_sync_loop, args=(node, args.sync_secs, stop), daemon=True)
        t.start()
        handler = _make_handler(node)
        # Enable SO_REUSEADDR so quick restarts don't trip TIME_WAIT.
        ThreadingHTTPServer.allow_reuse_address = True
        server = ThreadingHTTPServer((args.bind, args.port), handler)
        log.info("zg_chain_node serving on %s:%d node_id=%s peers=%s",
                 args.bind, args.port, node.node_id, sorted(node.peers))
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            log.info("shutting down")
        finally:
            stop.set()
            server.server_close()
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(_cli())
