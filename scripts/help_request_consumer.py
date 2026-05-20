#!/usr/bin/env python3
"""help_request_consumer.py — drains watchdog help_requests by spawning helper sub-agents.

Designed to be invoked by cron every 3 minutes. Each run is stateless — state is
captured by the filesystem (file location: main dir = pending, consumed/ = done,
archive_*/ = stale).

Architecture:
  watchdog/help_requests/<id>.json   — flagged by agent_watchdog_daemon.py
       |
       v (this script)
  parse → spawn helper(s) via `claude -p` subprocess
       |
       v (move file)
  watchdog/help_requests/consumed/<id>.json    (helpers spawned)
  watchdog/help_requests/archive_YYYY-MM-DD/   (too old / stale)

Rules:
  - Files <10 min old (by mtime) → spawn 2-3 helpers, move to consumed/
  - Files >=10 min old → mark stale, move to archive without spawning
  - Cap MAX_SPAWNS_PER_TICK (default 5) — avoid fork-bombing
  - Skip files that don't parse as JSON or lack agent_id
  - Each helper is spawned via `claude -p` (Option A from spec):
        claude -p --print --model sonnet --output-format text "<helper brief>"

CLI:
  --dry-run         Scan but do not spawn or move
  --max-spawns N    Override max spawns per tick
  --max-helpers N   Override helpers per help_request (default 2)
  --once <file>     Process a single help_request file (for testing)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Drive-safe move helper
# ---------------------------------------------------------------------------


def _safe_move(src: str, dst: str, *, retries: int = 3, log=None) -> bool:
    """shutil.move with retry/backoff to survive Google Drive sync races.
    Returns True on success, False on final failure (caller should log + continue)."""
    delays = [0.25, 0.5, 1.0]
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            shutil.move(src, dst)
            return True
        except (FileNotFoundError, PermissionError, OSError) as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(delays[attempt])
    msg = f"_safe_move FAILED after {retries} retries: {src} -> {dst}: {last_err!r}"
    if log:
        log(msg)
    else:
        print(msg, flush=True)
    return False


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

AI_ROOT = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools"
)
HELP_DIR = AI_ROOT / "watchdog" / "help_requests"
CONSUMED_DIR = HELP_DIR / "consumed"
LOG_DIR = AI_ROOT / "s&p500-ticker-mastery" / "logs"
LOG_FILE = LOG_DIR / "help_consumer.log"
HELPER_OUTPUT_DIR = HELP_DIR / "helper_outputs"
KILL_AND_RESPAWN_DIR = AI_ROOT / "watchdog" / "kill_and_respawn"

CLAUDE_BIN = (
    AI_ROOT / "ClaudeCode" / "npm-global" / "bin" / "claude"
)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

FRESH_MAX_AGE_MIN = 10  # files older than this are stale
DEFAULT_MAX_SPAWNS_PER_TICK = 5
DEFAULT_HELPERS_PER_REQUEST = 2
HELPER_TIMEOUT_SEC = 240  # 4 min per helper
HELPER_MODEL = "haiku"  # cheap default for low-signal triage

WATCHDOG_HELPER_TAG = "WATCHDOG_HELPER"  # propagated so daemon ignores them


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stdout),
        ],
    )


def get_age_minutes(path: Path) -> float:
    mtime = path.stat().st_mtime
    return (time.time() - mtime) / 60.0


def parse_help_request(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logging.warning("Could not parse %s: %s", path.name, exc)
        return None
    if not data.get("agent_id"):
        logging.warning("No agent_id in %s — skipping", path.name)
        return None
    return data


def build_helper_brief(req: dict, angle_idx: int) -> str:
    """Build a focused, narrow brief for one helper sub-agent."""
    agent_id = req.get("agent_id", "unknown")
    task_desc = (req.get("task_description") or "").strip()
    if len(task_desc) > 500:
        task_desc = task_desc[:500] + "…"
    decomp = req.get("suggested_decomposition") or []
    chosen = decomp[angle_idx] if angle_idx < len(decomp) else (
        decomp[0] if decomp else "Investigate and summarize the parent task; "
        "report what can be salvaged."
    )

    brief = (
        f"{WATCHDOG_HELPER_TAG} — helper spawned by help_request_consumer.\n\n"
        f"Parent agent: {agent_id}\n"
        f"Parent task (truncated): {task_desc}\n\n"
        f"YOUR NARROW SUB-TASK: {chosen}\n\n"
        "Scope is intentionally narrow. Do exactly this and return a concise "
        "result (<=10 lines). If the parent task is incoherent or trivial "
        "(e.g., a log fragment), say so in one line and stop. Do not spawn "
        "further sub-agents."
    )
    return brief


def spawn_helper(brief: str, agent_id: str, idx: int) -> dict:
    """Spawn one helper via `claude -p`. Returns dict with status + output path."""
    HELPER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = HELPER_OUTPUT_DIR / f"{agent_id}_h{idx}_{ts}.txt"

    cmd = [
        str(CLAUDE_BIN),
        "-p",
        "--model", HELPER_MODEL,
        "--output-format", "text",
        "--permission-mode", "bypassPermissions",
        brief,
    ]

    logging.info("Spawning helper idx=%d for agent_id=%s (model=%s)",
                 idx, agent_id, HELPER_MODEL)
    started = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=HELPER_TIMEOUT_SEC,
        )
        elapsed = time.time() - started
        out_path.write_text(
            f"# helper for agent_id={agent_id} idx={idx}\n"
            f"# elapsed_sec={elapsed:.1f}\n"
            f"# returncode={result.returncode}\n"
            f"# ---STDOUT---\n{result.stdout}\n"
            f"# ---STDERR---\n{result.stderr}\n"
        )
        status = "ok" if result.returncode == 0 else "fail"
        logging.info("Helper idx=%d finished in %.1fs (%s)", idx, elapsed, status)
        return {"status": status, "elapsed_sec": elapsed,
                "out": str(out_path), "rc": result.returncode}
    except subprocess.TimeoutExpired:
        elapsed = time.time() - started
        out_path.write_text(
            f"# helper for agent_id={agent_id} idx={idx}\n"
            f"# TIMEOUT after {elapsed:.1f}s\n"
        )
        logging.warning("Helper idx=%d TIMEOUT after %.1fs", idx, elapsed)
        return {"status": "timeout", "elapsed_sec": elapsed,
                "out": str(out_path), "rc": -1}
    except Exception as exc:  # noqa: BLE001
        logging.exception("Helper idx=%d crashed: %s", idx, exc)
        return {"status": "crash", "elapsed_sec": 0,
                "out": str(out_path), "rc": -2}


def move_to(path: Path, dest_dir: Path) -> Optional[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / path.name
    # If something with that name already exists, suffix
    if dest.exists():
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dest = dest_dir / f"{path.stem}_{ts}{path.suffix}"
    if not _safe_move(str(path), str(dest), log=logging.warning):
        return None
    return dest


def _guess_subagent_type(matched_pattern: str) -> str:
    """Best-effort guess of original subagent type from the matched_pattern string."""
    p = (matched_pattern or "").lower()
    mapping = [
        ("python", "python-pro"),
        ("typescript", "typescript-pro"),
        ("javascript", "javascript-pro"),
        ("react", "react-specialist"),
        ("node", "node-specialist"),
        ("rust", "rust-engineer"),
        ("go", "golang-pro"),
        ("java", "java-architect"),
        ("sql", "sql-pro"),
        ("data", "data-scientist"),
        ("ml", "ml-engineer"),
        ("devops", "devops-engineer"),
        ("docker", "docker-expert"),
        ("k8s", "kubernetes-specialist"),
        ("terraform", "terraform-engineer"),
    ]
    for keyword, subagent in mapping:
        if keyword in p:
            return subagent
    return "unknown"


def handle_mcp_stripped(req: dict, path: Path, dry_run: bool) -> dict:
    """Handle a help_request with type==MCP_STRIPPED.

    Writes a KILL_AND_RESPAWN instruction file instead of spawning a helper.
    Idempotent: if an unconsumed instruction file already exists, logs and skips.
    """
    agent_id = req["agent_id"]
    instruction_path = KILL_AND_RESPAWN_DIR / f"{agent_id}.json"

    # De-dup: skip if unconsumed instruction already exists
    if instruction_path.exists():
        try:
            existing = json.loads(instruction_path.read_text(encoding="utf-8"))
            if not existing.get("consumed", True):
                logging.info(
                    "MCP_STRIPPED: unconsumed instruction already exists for %s — skipping",
                    agent_id,
                )
                return {"spawned": 0, "status": "dedup_skipped", "agent_id": agent_id}
        except Exception as exc:  # noqa: BLE001
            logging.warning(
                "MCP_STRIPPED: could not read existing instruction for %s: %s — will overwrite",
                agent_id, exc,
            )

    matched_pattern = req.get("matched_pattern", "")
    instruction = {
        "agent_id": agent_id,
        "action": "KILL_AND_RESPAWN",
        "reason": req.get("reason", ""),
        "matched_pattern": matched_pattern,
        "respawn_as": "general-purpose",
        "original_subagent_type": _guess_subagent_type(matched_pattern),
        "original_help_request_path": str(path.resolve()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "consumed": False,
    }

    if dry_run:
        logging.info(
            "[DRY] MCP_STRIPPED: would write kill_and_respawn instruction for %s at %s",
            agent_id, instruction_path,
        )
        return {"spawned": 0, "status": "dry", "agent_id": agent_id}

    KILL_AND_RESPAWN_DIR.mkdir(parents=True, exist_ok=True)
    instruction_path.write_text(json.dumps(instruction, indent=2), encoding="utf-8")
    logging.info(
        "MCP_STRIPPED: wrote KILL_AND_RESPAWN instruction for %s at %s",
        agent_id, instruction_path,
    )

    # Move original help_request to consumed/
    dest = move_to(path, CONSUMED_DIR)
    if dest is None:
        logging.warning(
            "MCP_STRIPPED: could not move %s to consumed/ — instruction file written but source not moved",
            path.name,
        )
        return {"spawned": 0, "status": "move_failed", "agent_id": agent_id,
                "instruction_path": str(instruction_path)}

    # Annotate the consumed copy with consumer metadata
    try:
        data = json.loads(dest.read_text(encoding="utf-8"))
        data["_consumer"] = {
            "consumed_at": datetime.now(timezone.utc).isoformat(),
            "action": "KILL_AND_RESPAWN",
            "instruction_path": str(instruction_path),
            "helpers_spawned": 0,
        }
        dest.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logging.warning(
            "MCP_STRIPPED: could not annotate consumed copy %s: %s",
            dest.name, exc,
        )

    return {
        "spawned": 0,
        "status": "kill_and_respawn_written",
        "agent_id": agent_id,
        "instruction_path": str(instruction_path),
    }


def process_one(path: Path, max_helpers: int, dry_run: bool) -> dict:
    """Process one fresh help_request — returns dict {spawned: n, status: ...}."""
    req = parse_help_request(path)
    if req is None:
        # Unparseable — quarantine to archive
        if not dry_run:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if move_to(path, HELP_DIR / f"archive_{today}" / "unparseable") is None:
                logging.warning("Could not archive unparseable %s — skipping item", path.name)
                return {"spawned": 0, "status": "move_failed"}
        return {"spawned": 0, "status": "unparseable"}

    # MCP_STRIPPED requests get a kill-and-respawn instruction file, not a helper
    if req.get("type") == "MCP_STRIPPED":
        logging.info("MCP_STRIPPED request detected for agent_id=%s — routing to kill_and_respawn",
                     req["agent_id"])
        return handle_mcp_stripped(req, path, dry_run)

    agent_id = req["agent_id"]
    decomp = req.get("suggested_decomposition") or []
    n_helpers = min(max_helpers, max(1, len(decomp)) if decomp else 1)

    results = []
    if dry_run:
        logging.info("[DRY] would spawn %d helpers for agent_id=%s",
                     n_helpers, agent_id)
        # Return the counterfactual count so caller can enforce cap correctly
        return {"spawned": n_helpers, "status": "dry",
                "agent_id": agent_id, "results": []}
    else:
        for idx in range(n_helpers):
            brief = build_helper_brief(req, idx)
            results.append(spawn_helper(brief, agent_id, idx))

    # Move source to consumed/
    if not dry_run:
        dest = move_to(path, CONSUMED_DIR)
        if dest is None:
            logging.warning("Could not move %s to consumed/ after retries — skipping metadata write", path.name)
        else:
            # Augment consumed file with helper result metadata
            try:
                data = json.loads(dest.read_text(encoding="utf-8"))
                data["_consumer"] = {
                    "consumed_at": datetime.now(timezone.utc).isoformat(),
                    "helpers_spawned": len(results),
                    "helper_results": results,
                }
                dest.write_text(json.dumps(data, indent=2), encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                logging.warning("Could not write consumer metadata to %s: %s",
                                dest.name, exc)

    return {"spawned": len(results), "status": "ok",
            "agent_id": agent_id, "results": results}


def archive_stale(paths: list[Path], dry_run: bool) -> int:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    dest_dir = HELP_DIR / f"archive_{today}"
    moved = 0
    for p in paths:
        if dry_run:
            logging.info("[DRY] would archive stale %s", p.name)
            moved += 1
            continue
        if move_to(p, dest_dir) is not None:
            moved += 1
        else:
            logging.warning("Could not archive stale %s — skipping item", p.name)
    return moved


def scan_dir() -> tuple[list[Path], list[Path]]:
    """Return (fresh, stale) help_request files in main dir."""
    if not HELP_DIR.exists():
        return [], []
    fresh: list[Path] = []
    stale: list[Path] = []
    for p in HELP_DIR.iterdir():
        if not p.is_file() or p.suffix != ".json":
            continue
        # Skip files in subdirs (iterdir is top-level only — defensive)
        age = get_age_minutes(p)
        if age < FRESH_MAX_AGE_MIN:
            fresh.append(p)
        else:
            stale.append(p)
    # Process newest first so we work on the most relevant tasks
    fresh.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return fresh, stale


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-spawns", type=int,
                        default=DEFAULT_MAX_SPAWNS_PER_TICK)
    parser.add_argument("--max-helpers", type=int,
                        default=DEFAULT_HELPERS_PER_REQUEST)
    parser.add_argument("--once", type=str, default=None,
                        help="Process exactly this file (absolute path)")
    args = parser.parse_args(argv)

    setup_logging()
    logging.info("=== help_request_consumer start "
                 "(dry=%s max_spawns=%d helpers=%d) ===",
                 args.dry_run, args.max_spawns, args.max_helpers)

    if args.once:
        p = Path(args.once)
        if not p.exists():
            logging.error("--once file not found: %s", p)
            return 2
        out = process_one(p, args.max_helpers, args.dry_run)
        logging.info("once result: %s", json.dumps(out, default=str))
        return 0

    fresh, stale = scan_dir()
    logging.info("scan: fresh=%d stale=%d", len(fresh), len(stale))

    # Archive stale first
    archived = archive_stale(stale, args.dry_run)
    logging.info("archived stale=%d", archived)

    # Process fresh, up to MAX_SPAWNS_PER_TICK helpers total
    helpers_spawned = 0
    files_processed = 0
    for p in fresh:
        if helpers_spawned >= args.max_spawns:
            logging.info("hit max_spawns=%d — stopping fresh loop",
                         args.max_spawns)
            break
        remaining = args.max_spawns - helpers_spawned
        n_helpers = min(args.max_helpers, remaining)
        if n_helpers <= 0:
            break
        out = process_one(p, n_helpers, args.dry_run)
        helpers_spawned += out.get("spawned", 0)
        files_processed += 1

    logging.info(
        "=== done: files_processed=%d helpers_spawned=%d archived=%d ===",
        files_processed, helpers_spawned, archived,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
