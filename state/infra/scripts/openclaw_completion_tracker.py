#!/usr/bin/env python3
"""OpenClaw completion tracker (REPO-LOCAL).

Polls running OpenClaw processes (pgrep for `openclaw`) every 10s. On exit,
parses the corresponding /tmp/openclaw_*_run.log (or *_run.log style) for
toolSummary / completion / stopReason / output paths and appends a single
JSON record per completion to ~/.claude/state/openclaw_completions.jsonl
with reported_to_user=false.

A companion UserPromptSubmit hook (openclaw-completion-inject) reads
unreported rows, emits them as additionalContext on the next user prompt,
and marks them reported_to_user=true.
"""
from __future__ import annotations

import json
import os
import re
import signal
import sys
import time
from pathlib import Path
from typing import Any

POLL_INTERVAL_S = 10
HOME = Path(os.environ.get("HOME", os.path.expanduser("~")))
STATE_DIR = HOME / ".claude" / "state"
STATE_FILE = STATE_DIR / "openclaw_completions.jsonl"
SEEN_FILE = STATE_DIR / "openclaw_tracker_seen.json"
TMP_DIR = Path("/tmp")

PAT_TOOLSUMMARY = re.compile(r"toolSummary[\"']?\s*[:=]\s*({[^}]*})", re.I)
PAT_STOPREASON = re.compile(r"stopReason[\"']?\s*[:=]\s*[\"']?([a-zA-Z_\-]+)", re.I)
PAT_COMPLETION = re.compile(r"completion[\"']?\s*[:=]\s*[\"']([^\"']{0,200})", re.I)
PAT_FILE = re.compile(
    r"(?:wrote|created|saved|output|path)[^\n]*?([/~][\w\-./ &]+\.[\w]{1,8})",
    re.I,
)
PAT_TOOLCALLS = re.compile(r"tool[_\s\-]?calls?[\"']?\s*[:=]\s*(\d+)", re.I)
PAT_ERROR = re.compile(r"\b(error|exception|failed|timeout)\b", re.I)


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    sys.stderr.write(f"[{ts}] {msg}\n")
    sys.stderr.flush()


def list_openclaw_pids() -> dict[int, dict[str, Any]]:
    import subprocess
    out: dict[int, dict[str, Any]] = {}
    try:
        r = subprocess.run(
            ["ps", "-axo", "pid=,lstart=,command="],
            capture_output=True, text=True, timeout=5,
        )
    except Exception as e:
        log(f"ps failed: {e}")
        return out
    for line in r.stdout.splitlines():
        if "openclaw" not in line.lower():
            continue
        if "openclaw_completion_tracker" in line:
            continue
        parts = line.strip().split(None, 6)
        if len(parts) < 7:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        lstart = " ".join(parts[1:6])
        cmd = parts[6]
        out[pid] = {"cmdline": cmd, "lstart": lstart}
    return out


def find_log_for_pid(pid: int, cmdline: str) -> Path | None:
    m = re.search(r"(/tmp/openclaw_[^\s]+\.log)", cmdline)
    if m:
        p = Path(m.group(1))
        if p.exists():
            return p
    candidates = sorted(
        TMP_DIR.glob("openclaw_*.log"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )
    return candidates[0] if candidates else None


def parse_log(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "log_path": str(path),
        "tool_calls": None,
        "stop_reason": None,
        "completion_excerpt": None,
        "files_written": [],
        "errors": [],
        "tool_summary": None,
    }
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > 262_144:
                fh.seek(size - 262_144)
            blob = fh.read().decode("utf-8", errors="replace")
    except Exception as e:
        result["errors"].append(f"read_failed:{e}")
        return result

    m = PAT_STOPREASON.search(blob)
    if m:
        result["stop_reason"] = m.group(1)
    m = PAT_TOOLCALLS.search(blob)
    if m:
        try:
            result["tool_calls"] = int(m.group(1))
        except ValueError:
            pass
    m = PAT_COMPLETION.search(blob)
    if m:
        result["completion_excerpt"] = m.group(1)[:200]
    m = PAT_TOOLSUMMARY.search(blob)
    if m:
        result["tool_summary"] = m.group(1)[:500]

    files = set()
    for fm in PAT_FILE.finditer(blob):
        fp = fm.group(1).strip().rstrip(".,;:)")
        if len(fp) < 200:
            files.add(fp)
    result["files_written"] = sorted(files)[:20]

    errs = set()
    for em in PAT_ERROR.finditer(blob):
        start = max(0, em.start() - 40)
        end = min(len(blob), em.end() + 80)
        snippet = blob[start:end].replace("\n", " ").strip()
        errs.add(snippet[:200])
    result["errors"] = sorted(errs)[:5]

    if not result["stop_reason"]:
        result["stop_reason"] = "error" if result["errors"] else "clean"
    return result


def load_seen() -> dict[int, dict[str, Any]]:
    if not SEEN_FILE.exists():
        return {}
    try:
        with SEEN_FILE.open() as fh:
            d = json.load(fh)
        return {int(k): v for k, v in d.items()}
    except Exception:
        return {}


def save_seen(seen: dict[int, dict[str, Any]]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = SEEN_FILE.with_suffix(".tmp")
    with tmp.open("w") as fh:
        json.dump({str(k): v for k, v in seen.items()}, fh)
    tmp.replace(SEEN_FILE)


# gabriel_self twin feedback path (added 2026-05-20). Every twin completion
# row is mirrored into state/gabriel_self/outcomes.jsonl so the capability
# map + reflexion loop see twin outcomes alongside daemon spawn outcomes.
# autosolve_skip: feature wiring
GABRIEL_SELF_DIR = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools/state/gabriel_self"
)


def _mirror_to_gabriel_self(rec: dict[str, Any]) -> None:
    """Mirror this completion to state/gabriel_self/outcomes.jsonl. Best effort."""
    try:
        GABRIEL_SELF_DIR.mkdir(parents=True, exist_ok=True)
        outcomes = GABRIEL_SELF_DIR / "outcomes.jsonl"
        stop = (rec.get("stop_reason") or "").lower()
        is_failure = bool(rec.get("errors")) or stop in ("error", "no_log", "timeout")
        row = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": "openclaw_twin",
            "task_id": rec.get("task_id"),
            "duration_s": rec.get("duration_s"),
            "stop_reason": rec.get("stop_reason"),
            "outcome": "loss" if is_failure else "win",
            "tool_calls": rec.get("tool_calls"),
            "files_written_count": len(rec.get("files_written", []) or []),
            "errors_count": len(rec.get("errors", []) or []),
            "excerpt": (rec.get("completion_excerpt") or "")[:120],
            "cmdline_excerpt": (rec.get("cmdline") or "")[:120],
        }
        with outcomes.open("a") as ofh:
            ofh.write(json.dumps(row) + "\n")
    except Exception as e:  # noqa: BLE001
        log(f"gabriel_self mirror failed: {e}")


def append_completion(rec: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with STATE_FILE.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    _mirror_to_gabriel_self(rec)


def process_once(seen: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """One poll cycle. Exposed for smoke-testing."""
    alive = list_openclaw_pids()
    now = int(time.time())
    for pid, info in alive.items():
        if pid not in seen:
            seen[pid] = {
                "cmdline": info["cmdline"],
                "lstart": info["lstart"],
                "started_at": now,
                "log_path": str(find_log_for_pid(pid, info["cmdline"]) or ""),
            }
            log(f"new openclaw pid={pid}")

    exited = [pid for pid in seen if pid not in alive]
    for pid in exited:
        info = seen.pop(pid)
        finished_at = int(time.time())
        duration_s = finished_at - int(info.get("started_at", finished_at))
        log_path = info.get("log_path") or ""
        if not log_path or not Path(log_path).exists():
            lp = find_log_for_pid(pid, info.get("cmdline", ""))
            log_path = str(lp) if lp else ""

        if log_path and Path(log_path).exists():
            parsed = parse_log(Path(log_path))
        else:
            parsed = {
                "log_path": "", "tool_calls": None, "stop_reason": "no_log",
                "files_written": [], "errors": [], "completion_excerpt": None,
                "tool_summary": None,
            }

        task_id = f"openclaw_{pid}_{int(info.get('started_at', finished_at))}"
        rec = {
            "task_id": task_id,
            "pid": pid,
            "cmdline": info.get("cmdline", "")[:300],
            "started_at": int(info.get("started_at", finished_at)),
            "finished_at": finished_at,
            "duration_s": duration_s,
            "stop_reason": parsed.get("stop_reason"),
            "tool_calls": parsed.get("tool_calls"),
            "tool_summary": parsed.get("tool_summary"),
            "files_written": parsed.get("files_written", []),
            "errors": parsed.get("errors", []),
            "completion_excerpt": parsed.get("completion_excerpt"),
            "log_path": parsed.get("log_path", log_path),
            "reported_to_user": False,
        }
        try:
            append_completion(rec)
            log(
                f"completion pid={pid} dur={duration_s}s "
                f"stop={rec['stop_reason']} files={len(rec['files_written'])}"
            )
        except Exception as e:
            log(f"append_completion failed pid={pid}: {e}")

    save_seen(seen)
    return seen


def main() -> int:
    log("openclaw_completion_tracker starting")
    seen = load_seen()
    while True:
        try:
            seen = process_once(seen)
        except Exception as e:
            log(f"process_once exception: {e}")
        time.sleep(POLL_INTERVAL_S)


def _handle_sigterm(signum, frame):  # type: ignore[no-untyped-def]
    log(f"received signal {signum}, exiting")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)
    sys.exit(main())
