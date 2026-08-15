#!/usr/bin/env python3
"""chain_distribute — N-node distribution + quorum recovery for the Merkle chain.

Nodes (Phase B of the durability spec):
    1. local-ssd: ~/.zg/state/universal_resume/chain.jsonl       (primary writer)
    2. drive:     <DRIVE>/state/universal_resume/chain.jsonl     (rsync mirror)
    3. gh-repo:   ~/repos/sp500-mastery-compute/state/resume_chain.jsonl (git push)
    4. ipfs:      <optional> pinned via ipfs daemon, off by default.

Public CLI:
    chain_distribute.py sync         # mirror SSD -> Drive + Git push (no IPFS)
    chain_distribute.py verify       # verify ALL nodes, report per-node status
    chain_distribute.py quorum       # majority-verify; return highest-seq verified
    chain_distribute.py recover      # quorum-restore corrupted nodes from best

Designed to be safe to call from cron / launchd / daemon — failures non-fatal,
git push best-effort (logged). Locks via local file lock to prevent overlap.

Phase B + C of the blockchain-durability spec (2026-05-20).
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HOME = Path(os.path.expanduser("~"))
DRIVE_ROOT = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/"
    "My Drive/AI-Tools"
)
LOCAL_CHAIN = HOME / ".zg" / "state" / "universal_resume" / "chain.jsonl"
DRIVE_CHAIN = DRIVE_ROOT / "state" / "universal_resume" / "chain.jsonl"

# GH mirror: lives outside Drive (FUSE-incompat for git push hooks)
GH_REPO_DIR = HOME / "repos" / "sp500-mastery-compute"
GH_CHAIN_REL = "state/resume_chain.jsonl"
GH_CHAIN = GH_REPO_DIR / GH_CHAIN_REL
GH_REMOTE = "https://github.com/0riginal-claw/sp500-mastery-compute.git"

# IPFS (optional)
IPFS_PIN_RECORD = DRIVE_ROOT / "state" / "universal_resume" / "ipfs_pins.jsonl"

# Lock to prevent concurrent sync
LOCK_FILE = HOME / ".zg" / "state" / "universal_resume" / "chain_distribute.lock"

# How often to push to GH (in syncs) — default every 1 call (since we throttle externally)
GH_PUSH_EVERY_N = int(os.environ.get("CHAIN_GH_PUSH_EVERY_N", "1"))

SYNC_LOG = DRIVE_ROOT / "logs" / "auto_solve" / "chain_distribute.log"


# Import the chain module from the same directory (works in CLI + daemon imports)
sys.path.insert(0, str(Path(__file__).parent))
from merkle_chain import MerkleChain, GENESIS_HASH  # noqa: E402


def _log(msg: str) -> None:
    SYNC_LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n"
    try:
        with open(SYNC_LOG, "a") as f:
            f.write(line)
    except Exception:
        sys.stderr.write(line)


@contextlib.contextmanager
def _flock():
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    f = open(LOCK_FILE, "w")
    try:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            # held elsewhere — bail out cleanly
            yield False
            return
        yield True
    finally:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        f.close()


def _atomic_copy(src: Path, dst: Path) -> bool:
    if not src.exists() or src.stat().st_size == 0:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    try:
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
        return True
    except Exception as e:  # noqa: BLE001
        _log(f"copy {src} -> {dst} failed: {e!r}")
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        return False


def _git(*args, cwd: Path) -> tuple[int, str, str]:
    env = os.environ.copy()
    # GH_TOKEN expected to be in env (gh auth login already configured)
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    try:
        p = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except Exception as e:  # noqa: BLE001
        return 99, "", str(e)


def ensure_gh_repo() -> bool:
    """Clone the GH repo if missing. Idempotent. Returns True if usable."""
    if (GH_REPO_DIR / ".git").exists():
        return True
    GH_REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
    # Use gh's stored token via x-access-token URL form for non-interactive push
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        # try to fetch from gh cli
        try:
            token = subprocess.check_output(
                ["gh", "auth", "token"], text=True, timeout=10
            ).strip()
        except Exception:
            token = ""
    url = (
        f"https://x-access-token:{token}@github.com/0riginal-claw/sp500-mastery-compute.git"
        if token
        else GH_REMOTE
    )
    rc, out, err = _git("clone", "--depth", "1", url, str(GH_REPO_DIR), cwd=Path.home())
    if rc != 0:
        _log(f"clone failed rc={rc} err={err}")
        return False
    # Hide the token from origin URL on disk: set a non-token URL after clone.
    # The token is only used at clone time; subsequent pushes will re-add it from env.
    _git("remote", "set-url", "origin", GH_REMOTE, cwd=GH_REPO_DIR)
    return True


def sync_to_drive() -> dict:
    ok = _atomic_copy(LOCAL_CHAIN, DRIVE_CHAIN)
    return {"node": "drive", "ok": ok, "size": DRIVE_CHAIN.stat().st_size if DRIVE_CHAIN.exists() else 0}


def sync_to_gh() -> dict:
    if not ensure_gh_repo():
        return {"node": "gh", "ok": False, "reason": "clone_failed"}
    if not _atomic_copy(LOCAL_CHAIN, GH_CHAIN):
        return {"node": "gh", "ok": False, "reason": "copy_failed"}

    # Configure user.email / user.name on first push (idempotent)
    _git("config", "user.email", "noreply@anthropic.com", cwd=GH_REPO_DIR)
    _git("config", "user.name", "universal-resume-daemon", cwd=GH_REPO_DIR)

    rc, _out, err = _git("add", GH_CHAIN_REL, cwd=GH_REPO_DIR)
    if rc != 0:
        return {"node": "gh", "ok": False, "reason": f"add_failed: {err}"}
    # commit may exit non-zero on "nothing to commit" — that's fine
    chain_obj = MerkleChain(LOCAL_CHAIN, node_id="claude_main")
    seq = chain_obj.next_seq() - 1
    rc, _out, err = _git(
        "commit",
        "-m",
        f"chain: seq {seq} ts {int(time.time())}",
        cwd=GH_REPO_DIR,
    )
    nothing_to_commit = "nothing to commit" in (err or "").lower()
    if rc != 0 and not nothing_to_commit:
        _log(f"commit failed rc={rc} err={err}")
        # still try push; non-fatal
    # push with token from env
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        try:
            token = subprocess.check_output(["gh", "auth", "token"], text=True, timeout=10).strip()
        except Exception:
            token = ""
    push_url = (
        f"https://x-access-token:{token}@github.com/0riginal-claw/sp500-mastery-compute.git"
        if token
        else "origin"
    )
    rc, _out, err = _git("push", push_url, "HEAD:main", cwd=GH_REPO_DIR)
    return {
        "node": "gh",
        "ok": rc == 0,
        "seq": seq,
        "push_rc": rc,
        "push_err": (err or "")[:200],
    }


def sync_to_ipfs() -> dict:
    if not shutil.which("ipfs"):
        return {"node": "ipfs", "ok": False, "reason": "no_daemon"}
    if not LOCAL_CHAIN.exists():
        return {"node": "ipfs", "ok": False, "reason": "no_local"}
    try:
        p = subprocess.run(
            ["ipfs", "add", "-Q", "--pin=true", str(LOCAL_CHAIN)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        cid = p.stdout.strip()
        if p.returncode != 0 or not cid:
            return {"node": "ipfs", "ok": False, "reason": p.stderr.strip()[:200]}
        IPFS_PIN_RECORD.parent.mkdir(parents=True, exist_ok=True)
        with open(IPFS_PIN_RECORD, "a") as f:
            f.write(json.dumps({"ts": int(time.time()), "cid": cid}) + "\n")
        return {"node": "ipfs", "ok": True, "cid": cid}
    except Exception as e:  # noqa: BLE001
        return {"node": "ipfs", "ok": False, "reason": repr(e)[:200]}


# ----- multi-node verify / quorum --------------------------------------

NODE_PATHS = {
    "ssd": LOCAL_CHAIN,
    "drive": DRIVE_CHAIN,
    "gh": GH_CHAIN,
}


def verify_all() -> dict:
    """Verify chain integrity on each node. Returns per-node summary."""
    out: dict[str, dict] = {}
    for name, path in NODE_PATHS.items():
        if not path.exists() or path.stat().st_size == 0:
            out[name] = {"present": False, "ok": False, "errors": ["MISSING"], "last_seq": -1}
            continue
        chain = MerkleChain(path, node_id="verify")
        v = chain.verify()
        out[name] = {
            "present": True,
            "ok": v["ok"],
            "last_seq": v["last_seq"],
            "last_hash": v.get("last_hash", GENESIS_HASH),
            "errors": v["errors"][:5],
            "count": v["count"],
            "size_bytes": path.stat().st_size,
        }
    return out


def quorum(verify_results: dict | None = None, threshold: int | None = None) -> dict:
    """Return quorum decision over verified nodes.

    A node is eligible if (present and ok).
    Quorum threshold defaults to ceil(2 * eligible / 3) with minimum 2 when N>=3, else 1.
    Returns: {ok, threshold, eligible, best_node, best_seq, best_hash}.
    """
    v = verify_results or verify_all()
    eligible = [n for n, r in v.items() if r.get("present") and r.get("ok")]
    n_present = sum(1 for r in v.values() if r.get("present"))
    # threshold = at least 2/3 of present nodes, min 1
    if threshold is None:
        threshold = max(1, (2 * n_present + 2) // 3) if n_present > 0 else 1
    # pick the eligible node with highest seq as canonical
    best = None
    best_seq = -1
    best_hash = GENESIS_HASH
    for n in eligible:
        s = v[n].get("last_seq", -1)
        if s > best_seq:
            best, best_seq, best_hash = n, s, v[n].get("last_hash", GENESIS_HASH)
    return {
        "ok": len(eligible) >= threshold,
        "threshold": threshold,
        "eligible": eligible,
        "eligible_count": len(eligible),
        "present_count": n_present,
        "best_node": best,
        "best_seq": best_seq,
        "best_hash": best_hash,
        "per_node": v,
    }


def recover(target_nodes: list[str] | None = None) -> dict:
    """Restore corrupted/missing nodes from the best verified copy.

    SAFE: never deletes; uses atomic copy. Skips if no quorum (manual intervention).
    """
    v = verify_all()
    q = quorum(v)
    if not q["ok"] or q["best_node"] is None:
        return {"ok": False, "reason": "no_quorum", "quorum": q}

    src = NODE_PATHS[q["best_node"]]
    targets = target_nodes or [n for n in NODE_PATHS if n != q["best_node"]]
    results = {}
    for name in targets:
        dst = NODE_PATHS[name]
        before = v.get(name, {})
        if before.get("ok") and before.get("last_seq") == q["best_seq"]:
            results[name] = {"action": "skip", "reason": "already_canonical"}
            continue
        copied = _atomic_copy(src, dst)
        results[name] = {"action": "restore" if copied else "fail", "src": q["best_node"]}
    return {"ok": True, "best": q["best_node"], "best_seq": q["best_seq"], "results": results}


# ----- combined sync -----------------------------------------------------

def sync(do_ipfs: bool = False, do_gh: bool = True) -> dict:
    """Mirror SSD chain to all configured nodes. Best-effort. Returns summary."""
    summary: dict = {"ts": int(time.time()), "results": {}}
    with _flock() as got:
        if not got:
            summary["locked_out"] = True
            return summary
        if not LOCAL_CHAIN.exists():
            # bootstrap genesis
            MerkleChain(LOCAL_CHAIN, node_id="claude_main")
        summary["results"]["drive"] = sync_to_drive()
        if do_gh:
            summary["results"]["gh"] = sync_to_gh()
        if do_ipfs:
            summary["results"]["ipfs"] = sync_to_ipfs()
    _log(f"sync results: {json.dumps(summary['results'])}")
    return summary


# ----- CLI -------------------------------------------------------------

def _cli() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Multi-node chain distribution")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sync")
    s.add_argument("--ipfs", action="store_true")
    s.add_argument("--no-gh", action="store_true")
    sub.add_parser("verify")
    sub.add_parser("quorum")
    sub.add_parser("recover")
    sub.add_parser("status")
    args = ap.parse_args()

    if args.cmd == "sync":
        r = sync(do_ipfs=args.ipfs, do_gh=not args.no_gh)
        print(json.dumps(r, indent=2))
        return 0
    if args.cmd == "verify":
        v = verify_all()
        print(json.dumps(v, indent=2))
        return 0 if all(r.get("ok") for r in v.values() if r.get("present")) else 1
    if args.cmd == "quorum":
        q = quorum()
        # caller-friendly: success when quorum holds
        print(json.dumps(q, indent=2))
        return 0 if q["ok"] else 1
    if args.cmd == "recover":
        r = recover()
        print(json.dumps(r, indent=2))
        return 0 if r["ok"] else 1
    if args.cmd == "status":
        v = verify_all()
        q = quorum(v)
        print(
            json.dumps(
                {
                    "quorum_ok": q["ok"],
                    "best_node": q["best_node"],
                    "best_seq": q["best_seq"],
                    "eligible": q["eligible"],
                    "per_node": {
                        n: {"present": r.get("present"), "ok": r.get("ok"), "last_seq": r.get("last_seq", -1)}
                        for n, r in v.items()
                    },
                },
                indent=2,
            )
        )
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(_cli())
