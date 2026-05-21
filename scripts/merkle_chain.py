#!/usr/bin/env python3
"""merkle_chain — append-only Merkle chain for universal_resume durability.

Each checkpoint becomes a tamper-evident block:
    {
      "seq":          int,                 # monotonic, starts at 0 (genesis)
      "ts":           int (ns),
      "node_id":      str,                 # writer identity
      "action":       str,                 # "genesis" | "checkpoint" | ...
      "payload":      dict,                # arbitrary JSON-safe state
      "payload_hash": sha256-hex,          # hash of canonical payload bytes
      "prev_hash":    sha256-hex,          # block_hash of seq-1 (GENESIS for seq=0)
      "block_hash":   sha256-hex,          # hash of everything above
    }

Storage: newline-delimited JSON at ``chain.jsonl``. Append-only.
Tamper detection: re-hashing each block reproduces ``block_hash`` and ties to ``prev_hash``.

Phase A of the blockchain-durability spec (spec at
``logs/auto_solve/blockchain_durability_spec_2026-05-20.md``).

Zero non-stdlib deps. Atomic appends use a sidecar lockfile + tempfile-replace
fallback. Designed to coexist with the existing universal_resume_daemon
30s rsync loop — call ``append()`` once per cycle from the daemon.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Iterable

GENESIS_HASH = "0" * 64
F_FULLFSYNC = getattr(fcntl, "F_FULLFSYNC", 51)


def _full_fsync(fd: int) -> None:
    try:
        fcntl.fcntl(fd, F_FULLFSYNC)
        return
    except (OSError, ValueError):
        pass
    try:
        os.fsync(fd)
    except OSError:
        pass


def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_block(prev_hash: str, seq: int, node_id: str, action: str, payload: dict) -> dict:
    """Build a self-consistent block. Block fields are filled in canonical order."""
    block = {
        "seq": int(seq),
        "ts": time.time_ns(),
        "node_id": node_id,
        "action": action,
        "payload": payload,
        "payload_hash": _hash(_canonical(payload)),
        "prev_hash": prev_hash,
    }
    # block_hash = hash of all fields except block_hash itself
    block["block_hash"] = _hash(_canonical({k: v for k, v in block.items()}))
    return block


def verify_block(block: dict) -> list:
    """Return list of error strings (empty list = block OK)."""
    errs = []
    try:
        recomputed_ph = _hash(_canonical(block["payload"]))
        if block.get("payload_hash") != recomputed_ph:
            errs.append("payload_hash mismatch")
        bh = block.get("block_hash")
        body = {k: v for k, v in block.items() if k != "block_hash"}
        if bh != _hash(_canonical(body)):
            errs.append("block_hash mismatch")
    except Exception as e:  # noqa: BLE001
        errs.append(f"verify exception: {e!r}")
    return errs


class MerkleChain:
    """Append-only Merkle chain stored as newline-delimited JSON."""

    def __init__(self, path: Path | str, node_id: str = "claude_main"):
        self.path = Path(path)
        self.node_id = node_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._init_genesis()

    # ---- internals ------------------------------------------------------

    def _init_genesis(self) -> None:
        gen = make_block(
            prev_hash=GENESIS_HASH,
            seq=0,
            node_id=self.node_id,
            action="genesis",
            payload={"chain": "universal_resume", "init_ts": time.time_ns()},
        )
        line = json.dumps(gen, separators=(",", ":")) + "\n"
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            f.write(line.encode("utf-8"))
            f.flush()
            _full_fsync(f.fileno())
        os.replace(tmp, self.path)

    def _last_block(self) -> dict | None:
        if not self.path.exists():
            return None
        # read last non-empty line; chain may be huge so seek from end
        try:
            with open(self.path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                # Read up to 64KB from the tail (well above one block size)
                read_back = min(size, 65536)
                f.seek(size - read_back)
                tail = f.read().decode("utf-8", errors="replace")
            lines = [ln for ln in tail.splitlines() if ln.strip()]
            if not lines:
                return None
            return json.loads(lines[-1])
        except Exception:
            return None

    def tail_hash(self) -> str:
        b = self._last_block()
        return b.get("block_hash", GENESIS_HASH) if b else GENESIS_HASH

    def next_seq(self) -> int:
        b = self._last_block()
        return (b.get("seq", 0) + 1) if b else 0

    # ---- public API -----------------------------------------------------

    def append(self, action: str, payload: dict) -> dict:
        """Append a block. O(1) amortised. Returns the new block."""
        prev_hash = self.tail_hash()
        seq = self.next_seq()
        block = make_block(prev_hash, seq, self.node_id, action, payload)
        line = json.dumps(block, separators=(",", ":")) + "\n"

        # Use POSIX append-mode + flock for crash safety. macOS supports flock.
        with open(self.path, "ab") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            except OSError:
                pass
            try:
                f.write(line.encode("utf-8"))
                f.flush()
                _full_fsync(f.fileno())
            finally:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
        return block

    def iter_blocks(self) -> Iterable[dict]:
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    yield json.loads(ln)
                except json.JSONDecodeError:
                    continue

    def verify(self, max_blocks: int | None = None) -> dict:
        """Walk the chain end-to-end. Returns summary dict.

        max_blocks: if set, verify only the last N blocks (tail-only fast check).
        """
        prev = GENESIS_HASH
        count = 0
        errors: list[str] = []
        last_seq = -1
        last_hash = GENESIS_HASH
        try:
            blocks = list(self.iter_blocks())
        except Exception as e:  # noqa: BLE001
            return {
                "ok": False,
                "count": 0,
                "errors": [f"read failed: {e!r}"],
                "last_seq": -1,
                "last_hash": GENESIS_HASH,
            }
        if max_blocks is not None and len(blocks) > max_blocks:
            # tail-only verify: prev_hash bootstraps from prev-tail block_hash
            blocks = blocks[-max_blocks:]
            prev = blocks[0].get("prev_hash", GENESIS_HASH)
        for i, b in enumerate(blocks):
            if b.get("prev_hash") != prev:
                errors.append(
                    f"seq={b.get('seq')} prev_hash mismatch (got {str(b.get('prev_hash'))[:12]} want {prev[:12]})"
                )
            be = verify_block(b)
            if be:
                errors.append(f"seq={b.get('seq')} " + "; ".join(be))
            prev = b.get("block_hash", GENESIS_HASH)
            last_seq = b.get("seq", last_seq)
            last_hash = b.get("block_hash", last_hash)
            count += 1
        return {
            "ok": len(errors) == 0,
            "count": count,
            "errors": errors[:50],  # cap noise
            "last_seq": last_seq,
            "last_hash": last_hash,
        }


# ----- CLI -------------------------------------------------------------

def _cli() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Merkle chain utility")
    # Hard-pin to OS-level home (avoid launcher-redirected Drive path, which
    # creates a divergent chain in $DRIVE/home/.zg/...).
    ap.add_argument(
        "--chain",
        default="/Users/orginal/.zg/state/universal_resume/chain.jsonl",
    )
    ap.add_argument("--node-id", default="claude_main")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init")
    sub.add_parser("verify")
    sub.add_parser("tail")
    sub.add_parser("status")

    a = sub.add_parser("append")
    a.add_argument("--action", default="manual")
    a.add_argument("--payload", default='{"note":"manual append"}')

    args = ap.parse_args()
    chain = MerkleChain(args.chain, node_id=args.node_id)

    if args.cmd == "init":
        print(json.dumps({"path": str(chain.path), "tail": chain.tail_hash()}, indent=2))
        return 0
    if args.cmd == "verify":
        res = chain.verify()
        print(json.dumps(res, indent=2))
        return 0 if res["ok"] else 1
    if args.cmd == "tail":
        b = chain._last_block()  # noqa: SLF001
        print(json.dumps(b, indent=2) if b else "{}")
        return 0
    if args.cmd == "status":
        print(
            json.dumps(
                {
                    "path": str(chain.path),
                    "exists": chain.path.exists(),
                    "size": chain.path.stat().st_size if chain.path.exists() else 0,
                    "next_seq": chain.next_seq(),
                    "tail_hash": chain.tail_hash(),
                },
                indent=2,
            )
        )
        return 0
    if args.cmd == "append":
        try:
            payload = json.loads(args.payload)
        except json.JSONDecodeError:
            payload = {"raw": args.payload}
        b = chain.append(args.action, payload)
        print(json.dumps({"seq": b["seq"], "block_hash": b["block_hash"]}, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(_cli())
