"""
cloud_dispatch.py — Standard enqueue interface for all strategy/sweep scripts.

The dispatcher daemon (multi_cloud_dispatcher.py --daemon) polls
sweeps/queue.txt every 30 s and routes each job to the best available cloud.
This module is the single, canonical way to submit a job — never invoke
backtest scripts directly from orchestrators.

Usage:
    from cloud_dispatch import enqueue_job, enqueue_batch, check_status

    # Single job
    job_id = enqueue_job(
        ticker="AAPL",
        strategy="orb",
        script="scripts/backtest_xgb_v8.py",
    )

    # Bulk
    ids = enqueue_batch([
        {"ticker": "MSFT", "strategy": "orb",  "script": "scripts/backtest_xgb_v8.py"},
        {"ticker": "TSLA", "strategy": "vwap", "script": "scripts/backtest_xgb_v8.py"},
    ])

    # Status check (reads sweeps/dispatched.jsonl)
    info = check_status(job_id)
    # → {"status": "queued"|"submitted"|"complete"|"failed",
    #    "cloud": str|None, "result_path": str|None}

    # Backward-compat: fall through to local subprocess if no cloud available
    job_id = enqueue_job("AAPL", "orb", "scripts/backtest_xgb_v8.py",
                         subprocess_fallback=True)

Queue file format (sweeps/queue.txt):
    Each enqueued job appends ONE line:
        <script> <ticker> <strategy>
    This matches the legacy format the dispatcher already expects.

Status ledger (sweeps/dispatched.jsonl):
    One JSON line per job, written by this module at enqueue time and updated
    by the dispatcher (when it picks up a job it appends a new line with the
    same job_id and updated fields; check_status returns the latest record).
    Schema:
        {
          "id":           "<uuid8>",
          "ticker":       "AAPL",
          "strategy":     "orb",
          "script":       "scripts/backtest_xgb_v8.py",
          "out_path":     "backtests/AAPL/orb/result.json",
          "priority":     5,
          "extra_env":    {},
          "enqueued_at":  "2026-05-16T19:00:00+00:00",
          "status":       "queued",
          "cloud":        null,
          "result_path":  null
        }

Concurrency safety:
    All writes to queue.txt and dispatched.jsonl use fcntl.flock(LOCK_EX)
    so multiple producer processes can enqueue simultaneously without
    corrupting either file.

When to bypass (do NOT use this module):
    - Data-fetching utilities (download_ohlcv.py, fetch_fundamentals.py, etc.)
    - Quick diagnostic / one-off scripts that complete in <5 s
    - Scripts that are themselves orchestrators — they call enqueue_job, they
      don't *get* enqueued
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths (all relative to this file's parent — the project root)
# ---------------------------------------------------------------------------
_SCRIPT_DIR  = Path(__file__).parent          # …/scripts/
_PROJECT_ROOT = _SCRIPT_DIR.parent            # …/s&p500-ticker-mastery/
_QUEUE_FILE   = _PROJECT_ROOT / "sweeps" / "queue.txt"
_STATUS_FILE  = _PROJECT_ROOT / "sweeps" / "dispatched.jsonl"
_SWEEPS_DIR   = _PROJECT_ROOT / "sweeps"
_SWEEPS_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] cloud_dispatch — %(message)s",
        stream=sys.stdout,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _short_id() -> str:
    return str(uuid.uuid4())[:8]


def _default_out_path(ticker: str, strategy: str) -> str:
    return f"backtests/{ticker}/{strategy}/result.json"


def _append_locked(path: Path, text: str) -> None:
    """Append *text* to *path* atomically using an exclusive flock."""
    path.touch(exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            fh.write(text)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _write_status_record(record: dict[str, Any]) -> None:
    """Append one JSON record to sweeps/dispatched.jsonl."""
    line = json.dumps(record, ensure_ascii=False) + "\n"
    _append_locked(_STATUS_FILE, line)


def _queue_line(record: dict[str, Any]) -> str:
    """
    Build the 4-token line that multi_cloud_dispatcher.py expects:
        <script> <ticker> <strategy> <job_id>

    The job_id (4th token) is used by the dispatcher to correlate queue lines
    with dispatched.jsonl records so that processed lines can be atomically
    removed from the queue after each dispatch pass.  Older 3-token lines
    (written before this change) are still accepted by Job.from_line().
    """
    return f"{record['script']} {record['ticker']} {record['strategy']} {record['id']}\n"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enqueue_job(
    ticker: str,
    strategy: str,
    script: str,
    out_path: str | None = None,
    priority: int = 5,
    extra_env: dict[str, str] | None = None,
    subprocess_fallback: bool = False,
) -> str:
    """Enqueue one job for dispatch by the cloud dispatcher daemon.

    Appends a line to ``sweeps/queue.txt`` (the format the dispatcher reads)
    and a JSON record to ``sweeps/dispatched.jsonl`` (used by check_status).

    Args:
        ticker:              Ticker symbol, e.g. ``"AAPL"``.
        strategy:            Strategy code, e.g. ``"orb"`` or ``"vwap"``.
        script:              Repo-relative path to the backtest script,
                             e.g. ``"scripts/backtest_xgb_v8.py"``.
        out_path:            Where the script writes its result.  Defaults to
                             ``backtests/<ticker>/<strategy>/result.json``.
        priority:            Integer 1–10 (higher = more urgent).  Stored in
                             the status ledger; the dispatcher currently uses
                             FIFO ordering regardless of this field.
        extra_env:           Extra environment variables to store alongside the
                             job record (informational; adapters don't use them
                             yet unless the dispatcher is extended).
        subprocess_fallback: If ``True`` and the dispatcher daemon is NOT
                             running, run the script directly via
                             ``subprocess.run`` on this Mac instead of just
                             queuing.  Use this in scripts that need a result
                             before they can continue.

    Returns:
        The 8-character hex job ID string.

    Raises:
        ValueError: If ticker, strategy, or script are empty strings.
        RuntimeError: If subprocess_fallback=True and the subprocess fails.
    """
    if not ticker or not strategy or not script:
        raise ValueError("ticker, strategy, and script must all be non-empty strings")

    job_id   = _short_id()
    resolved_out = out_path or _default_out_path(ticker, strategy)
    env_dict = extra_env or {}
    now      = _now_iso()

    record: dict[str, Any] = {
        "id":          job_id,
        "ticker":      ticker,
        "strategy":    strategy,
        "script":      script,
        "out_path":    resolved_out,
        "priority":    priority,
        "extra_env":   env_dict,
        "enqueued_at": now,
        "status":      "queued",
        "cloud":       None,
        "result_path": None,
    }

    # 1. Append to queue.txt (dispatcher picks it up on next poll)
    _append_locked(_QUEUE_FILE, _queue_line(record))

    # 2. Write status record to dispatched.jsonl
    _write_status_record(record)

    log.info("Enqueued job_id=%s  %s/%s  script=%s", job_id, ticker, strategy, script)

    # 3. Optional local fallback when daemon is not running
    if subprocess_fallback and not _daemon_is_running():
        log.warning(
            "subprocess_fallback=True and daemon not running — "
            "executing %s locally for %s/%s", script, ticker, strategy
        )
        _run_local_fallback(record)

    return job_id


def enqueue_batch(jobs: list[dict[str, Any]]) -> list[str]:
    """Enqueue multiple jobs in a single locked append.

    Each dict in *jobs* must contain at minimum ``ticker``, ``strategy``,
    and ``script`` keys.  Optional keys: ``out_path``, ``priority``,
    ``extra_env``.

    Args:
        jobs: List of job-spec dicts.

    Returns:
        List of job ID strings (same order as *jobs*).

    Example::

        ids = enqueue_batch([
            {"ticker": "AAPL", "strategy": "orb",  "script": "scripts/backtest_xgb_v8.py"},
            {"ticker": "MSFT", "strategy": "vwap", "script": "scripts/backtest_xgb_v8.py"},
        ])
    """
    if not jobs:
        return []

    now      = _now_iso()
    records  = []
    job_ids  = []

    for spec in jobs:
        ticker   = spec["ticker"]
        strategy = spec["strategy"]
        script   = spec["script"]
        if not ticker or not strategy or not script:
            raise ValueError(
                f"Each job dict must have non-empty 'ticker', 'strategy', 'script'. Got: {spec!r}"
            )
        job_id = _short_id()
        job_ids.append(job_id)
        records.append({
            "id":          job_id,
            "ticker":      ticker,
            "strategy":    strategy,
            "script":      script,
            "out_path":    spec.get("out_path") or _default_out_path(ticker, strategy),
            "priority":    spec.get("priority", 5),
            "extra_env":   spec.get("extra_env") or {},
            "enqueued_at": now,
            "status":      "queued",
            "cloud":       None,
            "result_path": None,
        })

    # --- Atomic bulk append to queue.txt ---
    queue_block = "".join(_queue_line(r) for r in records)
    _append_locked(_QUEUE_FILE, queue_block)

    # --- Atomic bulk append to dispatched.jsonl ---
    status_block = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    _append_locked(_STATUS_FILE, status_block)

    log.info("Enqueued batch of %d jobs: ids=%s", len(records), job_ids)
    return job_ids


def check_status(job_id: str) -> dict[str, Any]:
    """Return the latest known status for a job.

    Reads ``sweeps/dispatched.jsonl`` and returns the **last** record whose
    ``id`` matches *job_id* (the dispatcher appends updated records as it
    processes jobs, so the last entry is authoritative).

    Args:
        job_id: The 8-character hex ID returned by :func:`enqueue_job`.

    Returns:
        Dict with keys:
            - ``status``:      ``"queued"`` | ``"submitted"`` | ``"complete"`` | ``"failed"``
            - ``cloud``:       Cloud name string or ``None``
            - ``result_path``: Path string or ``None``
            - ``ticker``:      Ticker symbol
            - ``strategy``:    Strategy code
            - ``enqueued_at``: ISO-8601 timestamp
        Returns ``{"status": "unknown", "cloud": None, "result_path": None}``
        if the job_id is not found.
    """
    if not _STATUS_FILE.exists():
        return {"status": "unknown", "cloud": None, "result_path": None}

    # Collect ALL rows for this job_id, then return the one with the latest
    # timestamp.  The dispatcher writes append-only event rows (one per status
    # transition), so "latest ts" is always authoritative regardless of line order.
    # Falls back to last-seen-row ordering when no ts field is present (old rows).
    matching: list[dict[str, Any]] = []
    with open(_STATUS_FILE, encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if rec.get("id") == job_id:
                matching.append(rec)

    if not matching:
        return {"status": "unknown", "cloud": None, "result_path": None}

    # Sort by ts descending; rows without a ts field sort before those with one
    # (i.e. old enqueue rows are treated as oldest).
    def _ts_key(r: dict[str, Any]) -> str:
        # ISO-8601 strings sort lexicographically correctly; empty string < any ts
        return r.get("ts") or r.get("enqueued_at") or ""

    latest = max(matching, key=_ts_key)

    return {
        "status":      latest.get("status", "queued"),
        "cloud":       latest.get("cloud"),
        "result_path": latest.get("result_path"),
        "ticker":      latest.get("ticker"),
        "strategy":    latest.get("strategy"),
        "enqueued_at": latest.get("enqueued_at"),
        "out_path":    latest.get("out_path"),
        "ts":          latest.get("ts"),
    }


# ---------------------------------------------------------------------------
# Subprocess fallback helpers (used when subprocess_fallback=True)
# ---------------------------------------------------------------------------

def _daemon_is_running() -> bool:
    """Return True if multi_cloud_dispatcher.py --daemon is running."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "multi_cloud_dispatcher.py"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def _run_local_fallback(record: dict[str, Any]) -> None:
    """Run the job's script directly via subprocess (blocking).

    Uses the sp500-mastery venv Python.  Raises RuntimeError on non-zero exit.

    Forwards record["extra_env"] (added 2026-05-20) into the subprocess env
    so per-job overrides (XGB_NO_TOPK etc.) survive the fallback path. Prior
    to this fix the fallback inherited only the dispatcher's env, silently
    dropping extra_env overrides.
    """
    python = "/Users/orginal/.venvs/sp500-mastery/bin/python"
    if not Path(python).exists():
        python = sys.executable

    script_path = str(_PROJECT_ROOT / record["script"])
    cmd = [
        python, script_path,
        "--ticker",   record["ticker"],
        "--strategy", record["strategy"],
        "--job-id",   record["id"],
    ]

    # Merge per-job extra_env into subprocess env (added 2026-05-20).
    sub_env = os.environ.copy()
    extra_env = record.get("extra_env") or {}
    if isinstance(extra_env, dict) and extra_env:
        for k, v in extra_env.items():
            if k and v is not None:
                sub_env[str(k)] = str(v)
        log.info("subprocess_fallback: forwarding %d extra_env keys: %s",
                 len(extra_env), sorted(extra_env.keys()))

    log.info("subprocess_fallback: running %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(_PROJECT_ROOT), env=sub_env)
    if proc.returncode != 0:
        raise RuntimeError(
            f"subprocess_fallback failed (rc={proc.returncode}) for "
            f"job_id={record['id']} {record['ticker']}/{record['strategy']}"
        )
    # Update status record in dispatched.jsonl to reflect local completion
    updated = dict(record)
    updated["status"]      = "complete"
    updated["cloud"]       = "mac_local_fallback"
    updated["result_path"] = str(_PROJECT_ROOT / record["out_path"])
    _write_status_record(updated)
    log.info(
        "subprocess_fallback complete: job_id=%s  result=%s",
        record["id"], updated["result_path"],
    )
