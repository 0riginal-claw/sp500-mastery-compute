#!/usr/bin/env python3
"""
drive_sync_batch.py — batch-sync local /tmp staging tree to Drive.

Motivation
----------
Concurrent helpers writing directly to Drive (`/My Drive/AI-Tools/...`)
cause Google Drive sync (Google Drive.app), fileproviderd, mds_stores,
and corespotlightd to spike, driving Mac 1-min load >50 sustained.

Mitigation
----------
Helpers honor `DRIVE_STAGING=/tmp/ai-tools-staging` and write there
(local APFS, no sync, no Spotlight). This script — invoked every 5 min
by launchd com.zg.drive_sync_batch — rsyncs the staging tree into the
real Drive root, batching N small writes into one transfer window
the Drive client can absorb without overload.

Safety
------
* Never permanently deletes anything in Drive. rsync runs WITHOUT
  `--delete` so files that exist only on Drive are untouched.
* Skips if a previous instance is still running (lockfile).
* Logs to `logs/drive_sync_batch_<UTC-DATE>.log`.
* Honors Mac load cap: if 1-min load > ceiling, defers this cycle.
* Honors `DRIVE_SYNC_BATCH_DISABLE=1` env for emergency stop.

Usage
-----
    python3 drive_sync_batch.py            # one cycle, exit
    python3 drive_sync_batch.py --dry-run  # show what would sync
    python3 drive_sync_batch.py --force    # ignore load cap

Layout
------
    /tmp/ai-tools-staging/
        logs/auto_solve/...        -> .../AI-Tools/logs/auto_solve/...
        state/autonomous_mode/...  -> .../AI-Tools/state/autonomous_mode/...
        research/<topic>/...       -> .../AI-Tools/research/<topic>/...
        <any other subtree>        -> .../AI-Tools/<same subtree>

Only subtrees that exist in staging are synced; nothing else is touched.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

STAGING_ROOT = Path(os.environ.get("DRIVE_STAGING", "/tmp/ai-tools-staging"))
DRIVE_ROOT = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools"
)
LOG_DIR = DRIVE_ROOT / "logs"
LOCK_FILE = Path("/tmp/drive_sync_batch.lock")
LOAD_CEILING = float(os.environ.get("DRIVE_SYNC_LOAD_CEIL", "20.0"))


def _today() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%d")


def _setup_logging() -> logging.Logger:
    # Try Drive logs/ first; on any error (FUSE perm denied under launchd,
    # Drive unreachable, etc.) fall back to /tmp. Logs in /tmp will be
    # rsynced to Drive on the next batch cycle if they live under STAGING_ROOT/logs/.
    candidates = [LOG_DIR, STAGING_ROOT / "logs", Path("/tmp")]
    log_path = None
    for cand in candidates:
        try:
            cand.mkdir(parents=True, exist_ok=True)
            test_path = cand / f"drive_sync_batch_{_today()}.log"
            # try opening append-mode to verify write
            with open(test_path, "a"):
                pass
            log_path = test_path
            break
        except OSError:
            continue
    if log_path is None:
        log_path = Path(f"/tmp/drive_sync_batch_{_today()}.log")
    logger = logging.getLogger("drive_sync_batch")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_path, mode="a")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(fh)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(sh)
    return logger


def _read_load_1min() -> float:
    try:
        out = subprocess.check_output(["uptime"], text=True, timeout=5)
        # macOS: " 1:08  up ..., load averages: 106.85 64.07 44.96"
        if "load averages" in out:
            tail = out.split("load averages:")[1].strip()
            parts = tail.split()
            return float(parts[0])
        if "load average" in out:
            tail = out.split("load average:")[1].strip()
            return float(tail.split(",")[0])
    except (subprocess.SubprocessError, ValueError, IndexError):
        pass
    return 0.0


def _acquire_lock(logger: logging.Logger) -> bool:
    if LOCK_FILE.exists():
        try:
            pid = int(LOCK_FILE.read_text().strip())
            os.kill(pid, 0)
            logger.warning("previous run PID=%s still alive - skipping", pid)
            return False
        except (ValueError, OSError):
            logger.info("stale lock - clearing")
            LOCK_FILE.unlink(missing_ok=True)
    LOCK_FILE.write_text(str(os.getpid()))
    return True


def _release_lock() -> None:
    LOCK_FILE.unlink(missing_ok=True)


def _enumerate_subtrees(staging: Path) -> list[Path]:
    if not staging.exists():
        return []
    return [p for p in staging.iterdir() if not p.name.startswith(".")]


def _rsync(src: Path, dst: Path, dry_run: bool, logger: logging.Logger) -> dict:
    """Run rsync src/ -> dst/ (trailing slash semantics)."""
    dst.mkdir(parents=True, exist_ok=True)
    cmd = [
        "/usr/bin/rsync",
        "-a",
        "--update",
        "--stats",
        "--itemize-changes",
        # explicitly NOT --delete: never remove files from Drive
    ]
    if dry_run:
        cmd.append("--dry-run")
    cmd.append(f"{src}/")
    cmd.append(f"{dst}/")
    logger.info("rsync %s/ -> %s/", src, dst)
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "transferred_files": 0,
            "transferred_bytes": 0,
            "elapsed_sec": 600,
            "cmd": " ".join(shlex.quote(c) for c in cmd),
            "stderr": "TIMEOUT after 600s",
        }
    elapsed = time.time() - t0
    files = 0
    nbytes = 0
    for line in proc.stdout.splitlines():
        if "files transferred" in line.lower():
            try:
                files = int(line.split(":")[-1].strip().split()[0].replace(",", ""))
            except (ValueError, IndexError):
                pass
        if "total transferred file size" in line.lower():
            try:
                nbytes = int(line.split(":")[-1].strip().split()[0].replace(",", ""))
            except (ValueError, IndexError):
                pass
    return {
        "ok": proc.returncode == 0,
        "rc": proc.returncode,
        "transferred_files": files,
        "transferred_bytes": nbytes,
        "elapsed_sec": round(elapsed, 2),
        "cmd": " ".join(shlex.quote(c) for c in cmd),
        "stderr": proc.stderr.strip()[-500:] if proc.stderr else "",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="show what would sync, no transfers")
    parser.add_argument("--force", action="store_true",
                        help="ignore load cap (DRIVE_SYNC_LOAD_CEIL)")
    parser.add_argument("--staging", type=Path, default=STAGING_ROOT,
                        help="staging root (default: /tmp/ai-tools-staging)")
    parser.add_argument("--drive-root", type=Path, default=DRIVE_ROOT,
                        help="Drive AI-Tools root (default: My Drive/AI-Tools)")
    args = parser.parse_args()

    logger = _setup_logging()

    if os.environ.get("DRIVE_SYNC_BATCH_DISABLE") == "1":
        logger.warning("DRIVE_SYNC_BATCH_DISABLE=1 - exiting without action")
        return 0

    if not args.staging.exists():
        logger.info("staging %s does not exist - nothing to sync", args.staging)
        return 0

    load = _read_load_1min()
    if load > LOAD_CEILING and not args.force:
        logger.warning("load=%.2f > ceiling=%.2f - deferring this cycle",
                       load, LOAD_CEILING)
        return 0

    if not args.dry_run and not _acquire_lock(logger):
        return 0

    summary: dict = {
        "started_at": _dt.datetime.utcnow().isoformat() + "Z",
        "staging": str(args.staging),
        "drive_root": str(args.drive_root),
        "load_1min": load,
        "subtrees": [],
        "total_files": 0,
        "total_bytes": 0,
        "ok": True,
    }

    try:
        subtrees = _enumerate_subtrees(args.staging)
        if not subtrees:
            logger.info("staging is empty")
        for sub in subtrees:
            rel = sub.relative_to(args.staging)
            dst = args.drive_root / rel
            if sub.is_dir():
                res = _rsync(sub, dst, args.dry_run, logger)
                summary["subtrees"].append({"path": str(rel), **res})
                summary["total_files"] += res["transferred_files"]
                summary["total_bytes"] += res["transferred_bytes"]
                if not res["ok"]:
                    summary["ok"] = False
                    logger.error("rsync failed for %s rc=%s stderr=%s",
                                 rel, res.get("rc"), res.get("stderr"))
            elif sub.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                res = _rsync(sub.parent / sub.name, dst, args.dry_run, logger)
                summary["subtrees"].append({"path": str(rel), "file": True, **res})
    finally:
        if not args.dry_run:
            _release_lock()

    summary["finished_at"] = _dt.datetime.utcnow().isoformat() + "Z"
    logger.info("SUMMARY %s", json.dumps(summary, default=str))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
