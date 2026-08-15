#!/usr/bin/env python3
"""universal_resume_daemon - 30s rsync of all 5 agent classes to durable state.

Classes covered:
  1. claude_main         — existing session_resume_checkpointer payload (delegated)
  2. claude_subagents    — /private/tmp/claude-501/<project>/<sid>/tasks/*.output
  3. openclaw_main       — ~/.openclaw/sessions/ + ~/.openclaw/logs/ + ~/.openclaw/workspace/
  4. openclaw_subagents  — nested OC traces under ~/.openclaw/agents/ + ~/.openclaw/tasks/
  5. ollama              — ~/.ollama/ (cache/, models/, config) excluding large blobs

Cycle (every CYCLE_SEC seconds):
  - sweep orphan tmp* files older than 5min in each class dir
  - rsync each source -> local state dir (F_FULLFSYNC-style atomic rename via os.replace)
  - opportunistic mirror to Drive (best-effort; failures non-fatal)
  - compute SHA256 manifest per class, write atomically
  - emit heartbeat
  - write per-class claimed-vs-actual diff (claimed=last manifest; actual=fresh walk)

Created 2026-05-20 by §8 REPO-LOCAL universal resume mission.
"""
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# macOS F_FULLFSYNC = 51 (per /usr/include/sys/fcntl.h). Forces actual flush to
# media, not just kernel write-cache (which fsync(2) only guarantees). Critical
# for crash-safety with APFS + Drive FUSE.
F_FULLFSYNC = getattr(fcntl, "F_FULLFSYNC", 51)

HOME = Path("/Users/orginal")  # OS-level home (passwd NFSHomeDirectory)
DRIVE_ROOT = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/"
    "My Drive/AI-Tools"
)
DRIVE_HOME = DRIVE_ROOT / "home"  # launcher-redirected $HOME (where shell-tools write)
LOCAL_STATE = HOME / ".zg" / "state" / "universal_resume"
DRIVE_STATE = DRIVE_ROOT / "state" / "universal_resume"

CYCLE_SEC = int(os.environ.get("UNIVERSAL_RESUME_CYCLE_SEC", "5"))
SESSION_ID = os.environ.get("CLAUDE_SESSION_ID") or str(uuid.uuid4())[:8]

# Blockchain durability (Phase A+B of blockchain_durability_spec_2026-05-20.md):
# every cycle appends a checkpoint block to a Merkle chain; every CHAIN_SYNC_EVERY
# cycles the chain is mirrored to N=3 nodes (SSD primary + Drive + Git push).
CHAIN_PATH = HOME / ".zg" / "state" / "universal_resume" / "chain.jsonl"
CHAIN_SYNC_EVERY = int(os.environ.get("UNIVERSAL_RESUME_CHAIN_SYNC_EVERY", "10"))
CHAIN_ENABLED = os.environ.get("UNIVERSAL_RESUME_CHAIN", "1") != "0"

# Lazy import so daemon still runs if module is missing (degrades gracefully).
_chain = None
_chain_distribute = None
_chain_cycle_count = 0


def _get_chain():
    global _chain
    if _chain is not None or not CHAIN_ENABLED:
        return _chain
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from merkle_chain import MerkleChain  # noqa: WPS433
        _chain = MerkleChain(CHAIN_PATH, node_id="claude_main")
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"chain import failed: {e!r}\n")
        _chain = None
    return _chain


def _get_chain_distribute():
    global _chain_distribute
    if _chain_distribute is not None or not CHAIN_ENABLED:
        return _chain_distribute
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        import chain_distribute  # noqa: WPS433
        _chain_distribute = chain_distribute
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"chain_distribute import failed: {e!r}\n")
        _chain_distribute = None
    return _chain_distribute

# Per-class source configuration. Each class maps to:
#   sources: list of Paths to mirror (existing only)
#   excludes: rsync-style globs to skip (large blobs etc.)
#   max_bytes_per_file: skip files larger than this (avoid mirroring huge models)
CLASSES = {
    "claude_main": {
        "sources": [
            HOME / ".zg" / "state" / "session_resume",
            DRIVE_HOME / ".zg" / "state" / "session_resume",
        ],
        "excludes": ["daemon.stdout.log", "daemon.stderr.log", "tmp*"],
        "max_bytes_per_file": 5 * 1024 * 1024,
    },
    "claude_subagents": {
        "sources": [Path("/private/tmp/claude-501")],
        "excludes": ["tmp*", "*.tmp", "*.lock"],
        "max_bytes_per_file": 2 * 1024 * 1024,
        "file_glob": "*.output",
    },
    "openclaw_main": {
        "sources": [
            HOME / ".openclaw" / "sessions",
            HOME / ".openclaw" / "workspace",
            HOME / ".openclaw" / "completions",
            HOME / ".openclaw" / "logs",
            DRIVE_HOME / ".openclaw" / "sessions",
            DRIVE_HOME / ".openclaw" / "workspace",
            DRIVE_HOME / ".openclaw" / "completions",
            DRIVE_HOME / ".openclaw" / "logs",
        ],
        "excludes": ["gateway.err.log", "gateway.log", "tmp*", "*.tmp"],
        "max_bytes_per_file": 2 * 1024 * 1024,
    },
    "openclaw_subagents": {
        "sources": [
            HOME / ".openclaw" / "agents",
            HOME / ".openclaw" / "tasks",
            HOME / ".openclaw" / "flows",
            DRIVE_HOME / ".openclaw" / "agents",
            DRIVE_HOME / ".openclaw" / "tasks",
            DRIVE_HOME / ".openclaw" / "flows",
        ],
        "excludes": ["tmp*", "*.tmp", "*.lock"],
        "max_bytes_per_file": 2 * 1024 * 1024,
    },
    "ollama": {
        "sources": [
            HOME / ".ollama" / "history",
            HOME / ".ollama" / "id_ed25519.pub",
            DRIVE_HOME / ".ollama" / "history",
            DRIVE_HOME / ".ollama" / "id_ed25519.pub",
        ],
        "excludes": ["models", "blobs", "manifests", "id_ed25519"],
        "max_bytes_per_file": 512 * 1024,
    },
}


# ----- atomic IO helpers -------------------------------------------------

def _full_fsync(fd: int) -> None:
    """F_FULLFSYNC if available (macOS), else fsync. Non-fatal on failure."""
    try:
        fcntl.fcntl(fd, F_FULLFSYNC)
        return
    except (OSError, ValueError):
        pass
    try:
        os.fsync(fd)
    except OSError:
        pass


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(path.parent), delete=False) as tmp:
        tmp.write(data)
        tmp.flush()
        _full_fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, path)


def sha256_file(path: Path, chunk: int = 65536) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while True:
                b = f.read(chunk)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()
    except Exception:
        return ""


def excluded(name: str, patterns: list) -> bool:
    import fnmatch
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def sweep_orphans(state_dir: Path, max_age_sec: int = 300) -> int:
    if not state_dir.exists():
        return 0
    now = time.time()
    swept = 0
    try:
        for tmp in state_dir.rglob("tmp*"):
            try:
                if now - tmp.stat().st_mtime > max_age_sec:
                    if tmp.is_file():
                        tmp.unlink()
                        swept += 1
            except Exception:
                pass
    except Exception:
        pass
    return swept


# ----- per-class mirror -------------------------------------------------

def walk_source(src: Path, cfg: dict):
    """Yield (relative_path, absolute_path, size) tuples for files to mirror."""
    if not src.exists():
        return
    excludes = cfg.get("excludes", [])
    max_bytes = cfg.get("max_bytes_per_file", 1024 * 1024)
    file_glob = cfg.get("file_glob")

    if src.is_file():
        try:
            size = src.stat().st_size
            if size <= max_bytes and not excluded(src.name, excludes):
                yield (src.name, src, size)
        except Exception:
            pass
        return

    for root, dirs, files in os.walk(src):
        # prune excluded dirs
        dirs[:] = [d for d in dirs if not excluded(d, excludes)]
        for fn in files:
            if excluded(fn, excludes):
                continue
            if file_glob:
                import fnmatch
                if not fnmatch.fnmatch(fn, file_glob):
                    continue
            fp = Path(root) / fn
            try:
                size = fp.stat().st_size
                if size > max_bytes:
                    continue
                rel = fp.relative_to(src)
                yield (str(rel), fp, size)
            except Exception:
                continue


def mirror_class(class_name: str, cfg: dict) -> dict:
    """Mirror one agent class. Return manifest dict."""
    local_dst = LOCAL_STATE / class_name
    local_dst.mkdir(parents=True, exist_ok=True)
    drive_dst = DRIVE_STATE / class_name  # opportunistic

    manifest = {
        "class": class_name,
        "ts": datetime.now(timezone.utc).isoformat(),
        "session_id": SESSION_ID,
        "files": {},
        "skipped_too_large": 0,
        "errors": 0,
        "sources_seen": [],
    }

    for src in cfg.get("sources", []):
        if not src.exists():
            continue
        manifest["sources_seen"].append(str(src))
        for rel, fp, size in walk_source(src, cfg):
            # Namespace by source-basename to avoid collisions across sources
            ns = src.name if src.is_dir() else "_files"
            local_target = local_dst / ns / rel
            try:
                local_target.parent.mkdir(parents=True, exist_ok=True)
                # only copy if missing or size/mtime changed
                need_copy = True
                src_digest = ""
                if local_target.exists():
                    try:
                        ls = local_target.stat()
                        ss = fp.stat()
                        if ls.st_size == ss.st_size and ls.st_mtime >= ss.st_mtime:
                            need_copy = False
                    except Exception:
                        pass
                # SHA256 dedup: even if mtime suggests copy, skip when content matches
                if need_copy and local_target.exists():
                    src_digest = sha256_file(fp)
                    dst_digest = sha256_file(local_target)
                    if src_digest and src_digest == dst_digest:
                        need_copy = False
                if need_copy:
                    # atomic copy via tempfile in same dir, F_FULLFSYNC before rename
                    with tempfile.NamedTemporaryFile(
                        dir=str(local_target.parent), delete=False
                    ) as tmp:
                        with open(fp, "rb") as f:
                            shutil.copyfileobj(f, tmp)
                        _full_fsync(tmp.fileno())
                        tmp_name = tmp.name
                    os.replace(tmp_name, local_target)
                digest = src_digest or sha256_file(local_target)
                manifest["files"][f"{ns}/{rel}"] = {
                    "size": size,
                    "sha256": digest,
                    "src": str(fp),
                }
            except Exception as e:
                manifest["errors"] += 1
                manifest["files"][f"{ns}/{rel}"] = {
                    "size": size,
                    "error": str(e)[:200],
                }

    # write manifest locally
    manifest_path = local_dst / "manifest.json"
    payload = json.dumps(manifest, indent=2).encode()
    try:
        atomic_write(manifest_path, payload)
    except Exception:
        pass

    # opportunistic Drive mirror (best-effort)
    try:
        drive_dst.mkdir(parents=True, exist_ok=True)
        atomic_write(drive_dst / "manifest.json", payload)
    except Exception:
        pass

    return manifest


def claimed_vs_actual_diff(class_name: str, manifest: dict) -> dict:
    """Write a diff of claimed manifest vs actual files on disk in mirror."""
    local_dst = LOCAL_STATE / class_name
    actual_files = set()
    if local_dst.exists():
        for root, dirs, files in os.walk(local_dst):
            for fn in files:
                if fn in ("manifest.json", "diff.json"):
                    continue
                fp = Path(root) / fn
                try:
                    rel = fp.relative_to(local_dst)
                    actual_files.add(str(rel))
                except Exception:
                    pass
    claimed = set(manifest.get("files", {}).keys())
    diff = {
        "ts": manifest["ts"],
        "class": class_name,
        "session_id": SESSION_ID,
        "claimed_count": len(claimed),
        "actual_count": len(actual_files),
        "missing_from_disk": sorted(claimed - actual_files)[:50],
        "orphan_on_disk": sorted(actual_files - claimed)[:50],
    }
    try:
        atomic_write(local_dst / "diff.json", json.dumps(diff, indent=2).encode())
    except Exception:
        pass
    return diff


# ----- heartbeat -------------------------------------------------------

def write_heartbeat(cycle_summary: dict) -> None:
    hb = {
        "ts": int(time.time()),
        "pid": os.getpid(),
        "session_id": SESSION_ID,
        "status": "alive",
        "cycle_id": SESSION_ID,
        "cycle_summary": cycle_summary,
    }
    payload = json.dumps(hb, indent=2).encode()
    for p in (
        LOCAL_STATE / "heartbeat.json",
        DRIVE_STATE / "heartbeat.json",
    ):
        try:
            atomic_write(p, payload)
        except Exception:
            pass


# ----- main loop -------------------------------------------------------

def main():
    LOCAL_STATE.mkdir(parents=True, exist_ok=True)
    try:
        DRIVE_STATE.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    while True:
        cycle_start = time.time()
        cycle_summary = {}
        try:
            sweep_orphans(LOCAL_STATE)
            try:
                sweep_orphans(DRIVE_STATE)
            except Exception:
                pass
            for cls_name, cfg in CLASSES.items():
                try:
                    m = mirror_class(cls_name, cfg)
                    d = claimed_vs_actual_diff(cls_name, m)
                    cycle_summary[cls_name] = {
                        "files": len(m.get("files", {})),
                        "errors": m.get("errors", 0),
                        "missing": len(d.get("missing_from_disk", [])),
                        "orphans": len(d.get("orphan_on_disk", [])),
                    }
                except Exception as e:
                    cycle_summary[cls_name] = {"error": str(e)[:200]}
            cycle_summary["_duration_sec"] = round(time.time() - cycle_start, 2)

            # ----- blockchain durability hook (Phase A+B) -----
            global _chain_cycle_count
            chain = _get_chain()
            if chain is not None:
                try:
                    block = chain.append("checkpoint", {
                        "session_id": SESSION_ID,
                        "ts": int(time.time()),
                        "cycle_summary": cycle_summary,
                    })
                    cycle_summary["_chain_seq"] = block.get("seq")
                except Exception as e:  # noqa: BLE001
                    cycle_summary["_chain_error"] = str(e)[:200]
                _chain_cycle_count += 1
                if _chain_cycle_count % max(1, CHAIN_SYNC_EVERY) == 0:
                    distrib = _get_chain_distribute()
                    if distrib is not None:
                        try:
                            r = distrib.sync(do_ipfs=False, do_gh=True)
                            cycle_summary["_chain_sync"] = {
                                "ts": r.get("ts"),
                                "drive_ok": r.get("results", {}).get("drive", {}).get("ok"),
                                "gh_ok": r.get("results", {}).get("gh", {}).get("ok"),
                            }
                        except Exception as e:  # noqa: BLE001
                            cycle_summary["_chain_sync_error"] = str(e)[:200]

            write_heartbeat(cycle_summary)
        except Exception as e:
            sys.stderr.write(f"universal_resume cycle error: {e}\n")
        sleep_for = max(1, CYCLE_SEC - (time.time() - cycle_start))
        time.sleep(sleep_for)
    return 0


if __name__ == "__main__":
    sys.exit(main())
