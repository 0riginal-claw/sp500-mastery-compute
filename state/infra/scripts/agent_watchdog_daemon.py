#!/usr/bin/env python3
"""agent_watchdog_daemon.py — monitors running sub-agents and queues help requests.

Designed to be invoked by a LaunchAgent every 2 minutes (StartInterval=120).
Each run is stateless — state is persisted in watchdog/flagged.json,
watchdog/escalations.json, and watchdog/queue.jsonl so runs are fully idempotent.

Rules:
  - Never flags files with age < 5 min (too young / still actively writing)
  - Never flags files with age >= 60 min (stale / completed)
  - Never re-flags an agent_id already in flagged.json (initial flag guard)
  - Never flags its own helper agents (marked with WATCHDOG_HELPER tag)
  - Calls DeepSeek directly via REST API to get task decomposition
  - Falls back to heuristic decomposition if API call fails
  - Idempotent: running 100x without new long-tasks creates no new help_requests

Escalation behaviour (new in v1.2):
  - Tracks per-agent escalation_level in watchdog/escalations.json
  - Each 5-min scan cycle where the agent is still running: level += 1
  - At level L, write 2^L helper requests (level 1->2, level 2->4, level 3->8, cap 16)
  - Each helper gets a UNIQUE micro-task drawn from a different decomposition angle
  - escalation_level is reset when the agent's .output file disappears (completed)
  - Global safety cap: skip escalation when total in-flight help_requests > 100

CLI args:
  --max-flags N   Cap new flags per run (0 = unlimited; use 5 for smoke tests)
  --dry-run       Scan and log but do not write any output files
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

AI_ROOT = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools"
)
TASKS_GLOB = (
    "/private/tmp/claude-501/"
    "-Users-orginal*/"
    "0e03dc82*/"
    "tasks/*.output"
)
WATCHDOG_DIR = AI_ROOT / "watchdog"
FLAGGED_JSON = WATCHDOG_DIR / "flagged.json"
ESCALATIONS_JSON = WATCHDOG_DIR / "escalations.json"
QUEUE_JSONL = WATCHDOG_DIR / "queue.jsonl"
HELP_DIR = WATCHDOG_DIR / "help_requests"
LOG_FILE = AI_ROOT / "logs" / "agent_watchdog.log"

# DeepSeek credentials stored by OpenClaw
AUTH_PROFILES = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools/home/.openclaw/agents/main/agent/auth-profiles.json"
)
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
# deepseek-chat maps to DeepSeek-V3 — cheapest production model, no gateway needed
DEEPSEEK_MODEL = "deepseek-chat"

# Thresholds (minutes)
MIN_AGE_MINUTES: int = 5
MAX_AGE_MINUTES: int = 60

# Tag in .output first-line content that identifies watchdog-spawned helpers.
# Agents whose prompt contains this string are NEVER re-flagged (anti-recursion).
WATCHDOG_HELPER_TAG = "WATCHDOG_HELPER"

# Escalation caps
MAX_HELPERS_PER_AGENT: int = 16
MAX_INFLIGHT_HELP_REQUESTS: int = 100

WATCHDOG_VERSION = "1.5"

# Regex for plain-text completion markers (ELAPSED=390s, ELAPSED=27s, etc.)
_RE_ELAPSED = re.compile(r"ELAPSED=\d+")

# ---------------------------------------------------------------------------
# MCP_STRIPPED detection constants
# ---------------------------------------------------------------------------

# (pattern_name, compiled_regex) pairs — checked in order; first match wins.
_MCP_STRIPPED_PATTERNS: list[tuple[str, "re.Pattern[str]"]] = [
    ("mcp_tool_unavailable",   re.compile(r"MCP tool unavailable", re.IGNORECASE)),
    ("tool_not_available",     re.compile(r"tool not available", re.IGNORECASE)),
    ("tool_is_not_available",  re.compile(r"tool\s+\S+\s+is not available", re.IGNORECASE)),
    ("subagent_type_stripped", re.compile(
        r"subagent_type[^a-zA-Z0-9]{0,5}"
        r"(python-pro|javascript-pro|typescript-pro|golang-pro|rust-pro"
        r"|java-pro|c-pro|cpp-pro|ruby-pro|php-pro)",
        re.IGNORECASE,
    )),
    ("no_such_tool",           re.compile(r"no such tool", re.IGNORECASE)),
    ("tool_not_found",         re.compile(r"tool\s+\S+\s+not found", re.IGNORECASE)),
]

# Matches any Task-tool invocation in a JSON transcript (sub-agent spawn indicator).
# Covers both native "Task" and plugin "mcp__plugin_...__Task" variants.
_RE_TASK_TOOL = re.compile(r'"name"\s*:\s*"[^"]*Task[^"]*"')

# Suffix that distinguishes MCP_STRIPPED help_request files from escalation helpers.
MCP_STRIPPED_SUFFIX = "_mcp_stripped"

# Max lines to read from a transcript when scanning for MCP_STRIPPED patterns.
_MCP_STRIPPED_SCAN_MAX_LINES: int = 2000

# ---------------------------------------------------------------------------
# Distinct micro-task templates per decomposition angle (8 angles, rotated by level)
# ---------------------------------------------------------------------------
# These are used to ensure helpers at each escalation level attack a DIFFERENT angle.
# Level 1 (2 helpers) → angles 0,1
# Level 2 (4 helpers) → angles 0,1,2,3
# Level 3 (8 helpers) → angles 0-7
# Level 4+ (16 helpers) → all 8 angles, doubled with "aggressive" prefix

DECOMP_ANGLES = [
    "Feature pruning: identify and remove the lowest-value inputs or computations "
    "to cut runtime by >50% without material quality loss. Output: pruned feature list "
    "and estimated speedup.",

    "Parallelization: decompose the workload into independent shards that can run "
    "concurrently. Specify shard boundaries, merge strategy, and expected wall-clock "
    "reduction.",

    "Partial-result salvage: extract and persist whatever intermediate results exist "
    "so work is not lost if the agent is killed. Write salvage output to disk and "
    "report what fraction of the goal is covered.",

    "Alternative approach: propose and begin executing a fundamentally different "
    "algorithm or data path that avoids the current bottleneck entirely.",

    "Bottleneck profiling: instrument the running computation (timing, memory, I/O) "
    "and pinpoint the single largest bottleneck. Produce a ranked list with "
    "fix recommendations.",

    "Incremental checkpoint: implement a checkpoint/resume mechanism so future "
    "runs skip already-completed work. Write checkpoint files and validate resume "
    "logic.",

    "Data subsetting: run the full pipeline on a statistically representative 10% "
    "sample to produce fast preliminary results. Report sample fidelity vs full run.",

    "Error triage: scan logs and intermediate outputs for warnings, exceptions, "
    "or silent failures that may be stalling progress. Produce a triage report "
    "with remediation steps.",
]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(str(LOG_FILE), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("watchdog")


# ---------------------------------------------------------------------------
# State helpers — flagged.json
# ---------------------------------------------------------------------------


def load_flagged() -> dict[str, str]:
    """Return {agent_id: iso_timestamp} from flagged.json."""
    WATCHDOG_DIR.mkdir(parents=True, exist_ok=True)
    if FLAGGED_JSON.exists():
        try:
            return json.loads(FLAGGED_JSON.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            log.warning("flagged.json unreadable — starting fresh")
    return {}


def save_flagged(flagged: dict[str, str]) -> None:
    """Persist flagged dict. Callers must only add keys, never remove."""
    WATCHDOG_DIR.mkdir(parents=True, exist_ok=True)
    FLAGGED_JSON.write_text(
        json.dumps(flagged, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# State helpers — escalations.json
# ---------------------------------------------------------------------------


def load_escalations() -> dict[str, dict]:
    """Return {agent_id: {level, first_flagged_at, last_escalated_at}} from escalations.json."""
    WATCHDOG_DIR.mkdir(parents=True, exist_ok=True)
    if ESCALATIONS_JSON.exists():
        try:
            return json.loads(ESCALATIONS_JSON.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            log.warning("escalations.json unreadable — starting fresh")
    return {}


def save_escalations(escalations: dict[str, dict]) -> None:
    """Persist escalations dict atomically."""
    WATCHDOG_DIR.mkdir(parents=True, exist_ok=True)
    ESCALATIONS_JSON.write_text(
        json.dumps(escalations, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def reset_escalation(agent_id: str, escalations: dict[str, dict]) -> bool:
    """Remove agent from escalations (called when agent .output file disappears).

    Returns True if the entry existed and was removed.
    """
    if agent_id in escalations:
        log.info("Agent %s completed — resetting escalation_level", agent_id)
        del escalations[agent_id]
        return True
    return False


# ---------------------------------------------------------------------------
# State helpers — queue + help_requests
# ---------------------------------------------------------------------------


def append_queue(record: dict) -> None:
    """Append a single JSON record to queue.jsonl (append-only)."""
    WATCHDOG_DIR.mkdir(parents=True, exist_ok=True)
    with QUEUE_JSONL.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_help_request(agent_id: str, suffix: str, payload: dict) -> bool:
    """Write help_requests/<agent_id><suffix>.json.

    Returns True if the file was newly created, False if it already existed.
    suffix distinguishes escalation helpers: "" for initial, "_e1_h0", "_e2_h0", etc.
    """
    HELP_DIR.mkdir(parents=True, exist_ok=True)
    path = HELP_DIR / f"{agent_id}{suffix}.json"
    if path.exists():
        return False
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return True


def count_inflight_help_requests() -> int:
    """Count all .json files currently in help_requests/."""
    HELP_DIR.mkdir(parents=True, exist_ok=True)
    return len(list(HELP_DIR.glob("*.json")))


# ---------------------------------------------------------------------------
# DeepSeek API key
# ---------------------------------------------------------------------------


def load_deepseek_api_key() -> Optional[str]:
    """Read DeepSeek API key from OpenClaw auth-profiles.json."""
    try:
        data = json.loads(AUTH_PROFILES.read_text(encoding="utf-8"))
        for profile in data.get("profiles", {}).values():
            if profile.get("provider") == "deepseek" and profile.get("key"):
                return str(profile["key"])
    except (OSError, json.JSONDecodeError, AttributeError) as exc:
        log.warning("Could not read DeepSeek API key: %s", exc)
    return None


# ---------------------------------------------------------------------------
# .output file parsing
# ---------------------------------------------------------------------------


def read_first_line_full(path: Path) -> str:
    """Return the full first non-empty line of a file with NO length cap."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for raw_line in fh:
                stripped = raw_line.strip()
                if stripped:
                    return stripped
    except OSError:
        pass
    return ""


def extract_task_desc(first_line: str) -> str:
    """Extract human-readable task description from a raw .output first line."""
    if first_line.startswith("{"):
        try:
            obj = json.loads(first_line)
            content = (
                obj.get("message", {}).get("content", "")
                or obj.get("content", "")
                or ""
            )
            if isinstance(content, list):
                parts = [b.get("text", "") for b in content if isinstance(b, dict)]
                content = " ".join(parts)
            return str(content)[:200]
        except (json.JSONDecodeError, AttributeError):
            pass
    return first_line[:200]


def _is_json_shaped(path: Path) -> bool:
    """Return True if the first non-empty line of *path* starts with '{' or '['.

    On any read error returns False (false-negative is safer than false-alarm).
    """
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for raw_line in fh:
                stripped = raw_line.strip()
                if stripped:
                    return stripped[0] in ("{", "[")
    except OSError:
        pass
    return False


def agent_id_from_path(path: Path) -> str:
    """Derive agent_id from filename stem (e.g. 'a3b0e24cf9fac0fb2')."""
    return path.stem


# ---------------------------------------------------------------------------
# Transcript completion-marker check (false-positive suppression)
# ---------------------------------------------------------------------------

# Maximum lines to scan from the END of a file when looking for completion markers.
# Completion markers always appear at or near the end, so scanning the tail is
# sufficient and avoids reading multi-MB transcripts fully into memory.
_COMPLETION_SCAN_TAIL_LINES: int = 500


def agent_has_completed_transcript(path: Path) -> bool:
    """Return True if the .output transcript contains a recognised completion marker.

    Scans up to the last ``_COMPLETION_SCAN_TAIL_LINES`` lines for any of:

    * JSON record with ``stop_reason == "end_turn"``   — Claude Code sub-agent done
    * JSON record with ``stop_reason == "max_tokens"`` — hit token limit (still done)
    * JSON record with ``type == "model.completed"``   — OpenClaw / DeepSeek done
    * Plain-text line containing ``"THRESHOLD MET"``   — poll-script goal reached
    * Plain-text line matching ``ELAPSED=<digits>``    — timer/poll script finished

    When a marker is found the agent is considered **finished** and must not be
    flagged.  If no marker is found the agent may genuinely be stuck.

    Args:
        path: Absolute path to the ``.output`` file.

    Returns:
        True if the agent has a completion marker; False otherwise.
    """
    _DONE_STOP_REASONS = frozenset({"end_turn", "max_tokens", "stop_sequence"})

    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            all_lines = fh.readlines()
    except OSError as exc:
        log.debug("Could not read transcript %s: %s", path.name, exc)
        return False

    # Scan only the tail for performance — markers appear at/near EOF.
    tail = all_lines[-_COMPLETION_SCAN_TAIL_LINES:]

    for raw_line in tail:
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("{"):
            # Attempt JSON parse for structured markers.
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                pass
            else:
                stop_reason = obj.get("stop_reason", "")
                if stop_reason in _DONE_STOP_REASONS:
                    log.debug(
                        "Transcript marker 'stop_reason=%s' found in %s — skipping",
                        stop_reason,
                        path.name,
                    )
                    return True
                if obj.get("type") == "model.completed":
                    log.debug(
                        "Transcript marker 'model.completed' found in %s — skipping",
                        path.name,
                    )
                    return True
        else:
            # Plain-text markers from shell poll / timer scripts.
            if "THRESHOLD MET" in line:
                log.debug(
                    "Transcript marker 'THRESHOLD MET' found in %s — skipping",
                    path.name,
                )
                return True
            if _RE_ELAPSED.search(line):
                log.debug(
                    "Transcript marker 'ELAPSED=N' found in %s — skipping",
                    path.name,
                )
                return True

    return False


def _sibling_jsonl_has_completion(path: Path) -> bool:
    """Return True if a sibling <stem>.jsonl transcript shows a completion marker.

    Looks for ``"stop_reason"`` or ``"type": "result"`` substrings in the last
    _COMPLETION_SCAN_TAIL_LINES lines of <stem>.jsonl.  Returns False if the
    sibling file does not exist.
    """
    jsonl_path = path.with_suffix(".jsonl")
    if not jsonl_path.exists():
        return False

    try:
        with jsonl_path.open("r", encoding="utf-8", errors="replace") as fh:
            all_lines = fh.readlines()
    except OSError as exc:
        log.debug("Could not read sibling jsonl %s: %s", jsonl_path.name, exc)
        return False

    tail = all_lines[-_COMPLETION_SCAN_TAIL_LINES:]
    for raw_line in tail:
        line = raw_line.strip()
        if not line:
            continue
        if '"stop_reason"' in line or '"type": "result"' in line:
            log.debug(
                "Sibling .jsonl completion marker found in %s — skipping %s",
                jsonl_path.name,
                path.name,
            )
            return True
    return False


# ---------------------------------------------------------------------------
# DeepSeek decomposition via direct REST call (no subprocess, no gateway)
# ---------------------------------------------------------------------------


def call_deepseek_decompose(
    agent_id: str,
    task_desc: str,
    api_key: Optional[str],
    n_tasks: int = 15,
    timeout: int = 25,
) -> list[str]:
    """Call DeepSeek API to decompose a long-running task into micro-tasks.

    Uses only stdlib (urllib) — no extra dependencies.
    Falls back to heuristic decomposition on any failure.

    Billing note: ~200 input + ~200 output tokens ~= $0.0001 per call.
    """
    if not api_key:
        log.info(
            "No DeepSeek API key — using heuristic decomposition for %s", agent_id
        )
        return _heuristic_decompose(task_desc)

    prompt = (
        f"A sub-agent (id={agent_id}) has been running >5 minutes.\n"
        f"Task: {task_desc!r}\n\n"
        f"Break this into {n_tasks} concrete, independent micro-tasks that helper agents "
        "can run in parallel. Output ONLY a JSON array of strings — no prose, no "
        "markdown code fences. Example: [\"task1\", \"task2\", \"task3\"]"
    )

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.3,
        "stream": False,
    }

    req = urllib.request.Request(
        f"{DEEPSEEK_BASE_URL}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    log.info("Calling DeepSeek for agent %s", agent_id)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
        data = json.loads(body)
        text = data["choices"][0]["message"]["content"].strip()

        # Strip markdown code fences if model wraps output despite instructions
        if text.startswith("```"):
            inner_lines = [
                ln for ln in text.split("\n") if not ln.startswith("```")
            ]
            text = "\n".join(inner_lines).strip()

        tasks = json.loads(text)
        if isinstance(tasks, list):
            result = [str(t) for t in tasks[:n_tasks]]
            log.info("DeepSeek returned %d micro-tasks for %s", len(result), agent_id)
            return result

        log.warning(
            "DeepSeek non-list response for %s: %r", agent_id, text[:100]
        )
        return _heuristic_decompose(task_desc)

    except urllib.error.HTTPError as exc:
        log.warning(
            "DeepSeek HTTP %s %s for %s", exc.code, exc.reason, agent_id
        )
    except urllib.error.URLError as exc:
        log.warning("DeepSeek URL error for %s: %s", agent_id, exc.reason)
    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        log.warning("DeepSeek parse error for %s: %s", agent_id, exc)
    except Exception as exc:  # noqa: BLE001
        log.warning("DeepSeek call failed for %s: %s", agent_id, exc)

    return _heuristic_decompose(task_desc)


def _heuristic_decompose(task_desc: str) -> list[str]:
    """Fallback decomposition when DeepSeek is unreachable."""
    short = task_desc[:60].rstrip()
    return [
        f"Identify current state and blockers for: {short}",
        "Scan intermediate outputs and logs for progress signals",
        "Complete the highest-priority pending sub-component",
        "Validate outputs against expected format and correctness criteria",
        "Write final results to disk and emit a completion signal",
    ]


# ---------------------------------------------------------------------------
# Escalation helper builder
# ---------------------------------------------------------------------------


def build_escalation_helpers(
    agent_id: str,
    task_desc: str,
    escalation_level: int,
    api_key: Optional[str],
    now: datetime,
    dry_run: bool,
) -> int:
    """Write helper request files for one escalation cycle.

    At escalation level L:
      - n_helpers = min(2**L, MAX_HELPERS_PER_AGENT)
      - Each helper is assigned a unique decomposition angle from DECOMP_ANGLES
      - Suffix format: _eL_hN  (e.g. _e2_h3)

    Returns the number of helpers newly written.
    """
    n_helpers = min(2 ** escalation_level, MAX_HELPERS_PER_AGENT)
    log.info(
        "Escalating agent %s to level %d — writing %d helper(s)",
        agent_id,
        escalation_level,
        n_helpers,
    )

    if dry_run:
        return n_helpers

    # Pull micro-tasks from DeepSeek for context-aware task descriptions,
    # using a larger request at higher levels.
    micro_tasks = call_deepseek_decompose(
        agent_id, task_desc, api_key, n_tasks=max(n_helpers, 8)
    )

    written = 0
    for h_idx in range(n_helpers):
        angle = DECOMP_ANGLES[h_idx % len(DECOMP_ANGLES)]
        suffix = f"_e{escalation_level}_h{h_idx}"

        # Compose a unique micro-task: angle description + optional DeepSeek task
        ds_hint = micro_tasks[h_idx % len(micro_tasks)] if micro_tasks else ""
        micro_task = (
            f"[WATCHDOG_HELPER | escalation_level={escalation_level} | "
            f"helper={h_idx + 1}/{n_helpers} | angle={h_idx % len(DECOMP_ANGLES)}]\n\n"
            f"ESCALATION ANGLE: {angle}\n\n"
            f"PARENT TASK: {task_desc}\n\n"
            f"SPECIFIC SUB-TASK: {ds_hint}"
        )

        payload = {
            "agent_id": agent_id,
            "helper_suffix": suffix,
            "escalation_level": escalation_level,
            "helper_index": h_idx,
            "total_helpers_this_level": n_helpers,
            "escalated_at": now.isoformat(),
            "decomposition_angle": h_idx % len(DECOMP_ANGLES),
            "angle_description": angle,
            "task_description": task_desc,
            "micro_task": micro_task,
            "watchdog_version": WATCHDOG_VERSION,
        }

        newly_written = write_help_request(agent_id, suffix, payload)
        if newly_written:
            append_queue(
                {
                    "agent_id": agent_id,
                    "helper_suffix": suffix,
                    "escalation_level": escalation_level,
                    "helper_index": h_idx,
                    "micro_task_preview": micro_task[:120],
                    "escalated_at": now.isoformat(),
                }
            )
            written += 1

    log.info(
        "Agent %s level %d: %d/%d helper files newly written",
        agent_id,
        escalation_level,
        written,
        n_helpers,
    )
    return written


# ---------------------------------------------------------------------------
# Priority classification
# ---------------------------------------------------------------------------


def classify_priority(age_minutes: float, task_desc: str) -> str:
    """Heuristic: return 'P0' or 'P1' based on age and task keywords."""
    if age_minutes >= 30:
        return "P0"
    keywords_p0 = {
        "mastery", "backtest", "train", "pipeline", "critical", "urgent",
        "watchdog", "overseer", "daemon", "continuous", "loop",
    }
    desc_lower = task_desc.lower()
    if any(kw in desc_lower for kw in keywords_p0):
        return "P0"
    return "P1"


# ---------------------------------------------------------------------------
# MCP_STRIPPED detection helpers
# ---------------------------------------------------------------------------


def read_transcript_for_mcp_check(path: Path) -> str:
    """Read up to _MCP_STRIPPED_SCAN_MAX_LINES lines of transcript text.

    Returns content as a single string.  Limited read prevents OOM on large files.
    Returns empty string on any read error.
    """
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            lines: list[str] = []
            for i, line in enumerate(fh):
                if i >= _MCP_STRIPPED_SCAN_MAX_LINES:
                    break
                lines.append(line)
        return "".join(lines)
    except OSError as exc:
        log.debug("Could not read transcript for MCP check %s: %s", path.name, exc)
        return ""


def count_task_tool_calls(transcript_text: str) -> int:
    """Count Task-tool invocations in transcript (proxy for sub-agent spawns)."""
    return len(_RE_TASK_TOOL.findall(transcript_text))


def detect_mcp_stripped(
    agent_id: str,
    transcript_text: str,
    age_seconds: float,
) -> Optional[dict]:
    """Check whether an agent appears to be running with MCP tools stripped.

    Trigger conditions (ALL must be true):
      1. One of _MCP_STRIPPED_PATTERNS matches anywhere in transcript_text.
      2. Agent age > 180 seconds (3 minutes).
      3. Zero Task-tool calls found in transcript (no helper sub-agents spawned).

    Args:
        agent_id:        Agent identifier (for log messages).
        transcript_text: Content read from the agent's .output file.
        age_seconds:     Elapsed seconds since last mtime of the .output file.

    Returns:
        dict with keys {pattern_name, matched_text, excerpt} if detected, else None.
    """
    if age_seconds <= 180:
        return None

    task_calls = count_task_tool_calls(transcript_text)
    if task_calls > 0:
        log.debug(
            "MCP_STRIPPED skipped for %s — found %d Task call(s) in transcript",
            agent_id,
            task_calls,
        )
        return None

    for pattern_name, pattern in _MCP_STRIPPED_PATTERNS:
        match = pattern.search(transcript_text)
        if match:
            # Extract ~500 chars of context around the match for evidence
            start = max(0, match.start() - 200)
            end = min(len(transcript_text), match.end() + 300)
            excerpt = transcript_text[start:end]
            log.debug(
                "MCP_STRIPPED pattern '%s' matched in %s transcript",
                pattern_name,
                agent_id,
            )
            return {
                "pattern_name": pattern_name,
                "matched_text": match.group(0),
                "excerpt": excerpt,
            }

    return None


def _handle_mcp_stripped(
    agent_id: str,
    path: Path,
    age_seconds: float,
    detection: dict,
    now: datetime,
    dry_run: bool,
) -> int:
    """Write an MCP_STRIPPED help_request for the given agent.

    De-duplicates: if a help_request with suffix MCP_STRIPPED_SUFFIX already
    exists for this agent_id the function logs and returns 0 without writing.

    Args:
        agent_id:    Agent identifier.
        path:        Absolute path to the agent's .output file.
        age_seconds: Elapsed seconds since last mtime.
        detection:   Dict returned by detect_mcp_stripped.
        now:         Current UTC datetime (for detected_at timestamp).
        dry_run:     If True, log intent but do not write files.

    Returns:
        1 if a new help_request was written (or would be in dry_run), 0 if skipped.
    """
    existing = HELP_DIR / f"{agent_id}{MCP_STRIPPED_SUFFIX}.json"
    if not dry_run and existing.exists():
        log.debug("MCP_STRIPPED help_request already exists for %s — skipping", agent_id)
        return 0

    age_min = age_seconds / 60.0
    reason = (
        f"Agent {agent_id} appears to have MCP tools stripped. "
        f"Pattern '{detection['pattern_name']}' matched text: "
        f"{detection['matched_text']!r}. "
        f"Age: {age_min:.1f} min. "
        f"No Task tool calls found in transcript (0 helper sub-agents spawned)."
    )

    log.warning(
        "MCP_STRIPPED detected | agent=%s | age=%.1f min | pattern=%s | match=%r",
        agent_id,
        age_min,
        detection["pattern_name"],
        detection["matched_text"],
    )

    if dry_run:
        log.info("dry_run — would write MCP_STRIPPED help_request for %s", agent_id)
        return 1

    payload = {
        "agent_id": agent_id,
        "type": "MCP_STRIPPED",
        "reason": reason,
        "recommendation": "kill_and_respawn_as_general_purpose",
        "matched_pattern": detection["pattern_name"],
        "matched_text": detection["matched_text"],
        "detected_at": now.isoformat(),
        "transcript_excerpt": detection["excerpt"],
        "age_minutes": round(age_min, 1),
        "output_file": str(path),
        "watchdog_version": WATCHDOG_VERSION,
    }

    newly_written = write_help_request(agent_id, MCP_STRIPPED_SUFFIX, payload)
    if newly_written:
        append_queue(
            {
                "agent_id": agent_id,
                "type": "MCP_STRIPPED",
                "detected_at": now.isoformat(),
                "matched_pattern": detection["pattern_name"],
            }
        )
        log.info("MCP_STRIPPED help_request written for agent %s", agent_id)
        return 1

    # write_help_request returned False → file appeared between our exists() check
    # and the write (race condition). Treat as already-handled.
    log.debug("MCP_STRIPPED help_request race-condition skip for %s", agent_id)
    return 0


# ---------------------------------------------------------------------------
# Core scan logic
# ---------------------------------------------------------------------------


def run_scan(
    api_key: Optional[str],
    max_flags: int = 0,
    dry_run: bool = False,
) -> tuple[int, int, int, int]:
    """Run one full scan pass.

    Args:
        api_key:    DeepSeek API key (or None for heuristic fallback).
        max_flags:  If > 0, stop after flagging this many new agents.
                    0 means unlimited (normal production mode).
        dry_run:    If True, log findings but do not write any files.

    Returns:
        (n_scanned, n_in_window, n_flagged_new, n_escalated)
    """
    now = datetime.now(timezone.utc)
    now_ts = now.timestamp()

    flagged = load_flagged()
    escalations = load_escalations()
    output_files = glob.glob(TASKS_GLOB)

    # Build set of currently active agent_ids from file system
    active_agent_ids: set[str] = set()
    for file_str in output_files:
        active_agent_ids.add(agent_id_from_path(Path(file_str)))

    # Reset escalation for agents whose .output file has disappeared (completed)
    reset_happened = False
    for agent_id in list(escalations.keys()):
        if agent_id not in active_agent_ids:
            reset_escalation(agent_id, escalations)
            reset_happened = True
    if reset_happened and not dry_run:
        save_escalations(escalations)

    n_scanned = 0
    n_in_window = 0
    n_flagged_new = 0
    n_escalated = 0
    n_mcp_stripped = 0

    log.info(
        "Scan start — glob found %d .output files | dry_run=%s | max_flags=%d",
        len(output_files),
        dry_run,
        max_flags,
    )

    for file_str in sorted(output_files):  # sorted for determinism
        if max_flags > 0 and n_flagged_new >= max_flags:
            log.info("Reached --max-flags=%d cap, stopping early", max_flags)
            break

        path = Path(file_str)
        agent_id = agent_id_from_path(path)
        n_scanned += 1

        # --- mtime age ---
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        age_minutes = (now_ts - mtime) / 60.0

        # --- age window filter: 5 min <= age < 60 min ---
        if age_minutes < MIN_AGE_MINUTES:
            continue  # still actively writing output
        if age_minutes >= MAX_AGE_MINUTES:
            continue  # completed long ago
        n_in_window += 1

        # --- read full first line (no truncation) ---
        first_line = read_first_line_full(path)
        if not first_line:
            continue

        # --- watchdog helper guard: never recurse ---
        if WATCHDOG_HELPER_TAG in first_line:
            log.debug("Skipping watchdog helper agent %s", agent_id)
            continue

        # --- JSON-shape guard: skip raw shell stdout (not JSON agent payloads) ---
        if not _is_json_shaped(path):
            log.info("Skipping non-JSON shell output: %s", path.name)
            continue

        # --- sibling .jsonl completion check ---
        if _sibling_jsonl_has_completion(path):
            log.info(
                "Skipping agent %s — sibling .jsonl transcript shows completion",
                agent_id,
            )
            continue

        # --- transcript completion-marker check: skip already-finished agents ---
        # Reads last _COMPLETION_SCAN_TAIL_LINES lines; O(tail) not O(file size).
        if agent_has_completed_transcript(path):
            log.info(
                "Skipping agent %s — transcript shows completion (false-positive guard)",
                agent_id,
            )
            continue

        # --- MCP_STRIPPED detection ---
        # Checks for tool-unavailability patterns + age > 3 min + 0 Task calls.
        # Runs for all non-completed agents regardless of flagged/escalation state.
        _transcript_text = read_transcript_for_mcp_check(path)
        _mcp_detection = detect_mcp_stripped(agent_id, _transcript_text, now_ts - mtime)
        if _mcp_detection:
            n_mcp_stripped += _handle_mcp_stripped(
                agent_id=agent_id,
                path=path,
                age_seconds=now_ts - mtime,
                detection=_mcp_detection,
                now=now,
                dry_run=dry_run,
            )
        del _transcript_text  # release memory before continuing main loop

        # --- extract human-readable description ---
        task_desc = extract_task_desc(first_line)
        priority = classify_priority(age_minutes, task_desc)

        # ---------------------------------------------------------------
        # ESCALATION PATH: agent already flagged — increment level
        # ---------------------------------------------------------------
        if agent_id in flagged:
            # Global in-flight cap
            inflight = count_inflight_help_requests()
            if inflight > MAX_INFLIGHT_HELP_REQUESTS:
                log.warning(
                    "In-flight help_requests=%d > %d cap — skipping escalation for %s",
                    inflight,
                    MAX_INFLIGHT_HELP_REQUESTS,
                    agent_id,
                )
                continue

            # Determine new escalation level
            prev_level = escalations.get(agent_id, {}).get("level", 0)
            new_level = prev_level + 1
            n_helpers_this_level = min(2 ** new_level, MAX_HELPERS_PER_AGENT)

            log.info(
                "Agent %s still running | age=%.1f min | escalation %d->%d | "
                "writing %d helper(s) | inflight=%d",
                agent_id,
                age_minutes,
                prev_level,
                new_level,
                n_helpers_this_level,
                inflight,
            )

            helpers_written = build_escalation_helpers(
                agent_id=agent_id,
                task_desc=task_desc,
                escalation_level=new_level,
                api_key=api_key,
                now=now,
                dry_run=dry_run,
            )

            if not dry_run:
                escalations[agent_id] = {
                    "level": new_level,
                    "first_flagged_at": escalations.get(agent_id, {}).get(
                        "first_flagged_at", flagged.get(agent_id, now.isoformat())
                    ),
                    "last_escalated_at": now.isoformat(),
                    "helpers_written_this_level": helpers_written,
                    "priority": priority,
                }
                save_escalations(escalations)

            n_escalated += 1
            continue

        # ---------------------------------------------------------------
        # INITIAL FLAG PATH: first time we see this agent
        # ---------------------------------------------------------------

        # Idempotency guard: help_request file already on disk
        if not dry_run and (HELP_DIR / f"{agent_id}.json").exists():
            # Sync flagged.json (crash-recovery path)
            flagged[agent_id] = now.isoformat()
            save_flagged(flagged)
            continue

        log.info(
            "Flagging agent %s | age=%.1f min | %s | desc=%.80r",
            agent_id,
            age_minutes,
            priority,
            task_desc,
        )

        if dry_run:
            n_flagged_new += 1
            continue  # skip all writes

        # --- DeepSeek decomposition (~10s, ~$0.0001) ---
        micro_tasks = call_deepseek_decompose(agent_id, task_desc, api_key)

        # --- Build initial help_request payload ---
        flagged_at = now.isoformat()
        help_request = {
            "agent_id": agent_id,
            "flagged_at": flagged_at,
            "age_minutes": round(age_minutes, 1),
            "priority": priority,
            "task_description": task_desc,
            "output_file": file_str,
            "suggested_decomposition": micro_tasks,
            "escalation_level": 0,
            "watchdog_version": WATCHDOG_VERSION,
        }

        # --- Persist (crash-safe ordering) ---
        # 1. Write help_request first
        write_help_request(agent_id, "", help_request)

        # 2. Append to queue.jsonl
        append_queue(
            {
                "agent_id": agent_id,
                "task_desc_first_line": task_desc,
                "age_minutes": round(age_minutes, 1),
                "suggested_decomposition": micro_tasks,
                "flagged_at": flagged_at,
                "escalation_level": 0,
            }
        )

        # 3. Mark as flagged + initialize escalation record
        flagged[agent_id] = flagged_at
        save_flagged(flagged)

        escalations[agent_id] = {
            "level": 0,
            "first_flagged_at": flagged_at,
            "last_escalated_at": flagged_at,
            "helpers_written_this_level": 1,
            "priority": priority,
        }
        save_escalations(escalations)

        n_flagged_new += 1

    log.info(
        "Scan complete — total=%d in_window=%d newly_flagged=%d escalated=%d mcp_stripped=%d",
        n_scanned,
        n_in_window,
        n_flagged_new,
        n_escalated,
        n_mcp_stripped,
    )
    return n_scanned, n_in_window, n_flagged_new, n_escalated, n_mcp_stripped


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _write_heartbeat(status: str = "running") -> None:
    """Atomic heartbeat write (six-fail-fix F7 — 2026-05-20)."""
    try:
        import tempfile
        hb_dir = AI_ROOT / "state" / "agent_watchdog"
        hb_dir.mkdir(parents=True, exist_ok=True)
        hb = hb_dir / "heartbeat.json"
        payload = json.dumps({"ts": int(time.time()), "pid": os.getpid(), "status": status})
        with tempfile.NamedTemporaryFile(dir=str(hb_dir), delete=False, mode="w") as tmp:
            tmp.write(payload)
            tmp_path = tmp.name
        os.replace(tmp_path, hb)
    except Exception:
        pass


def main() -> None:
    """Entry point for LaunchAgent (StartInterval=120) and manual invocation."""
    _write_heartbeat("start")
    parser = argparse.ArgumentParser(
        description="Agent watchdog daemon — single-pass scan with escalation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--max-flags",
        type=int,
        default=0,
        metavar="N",
        help="Max new agents to flag per run (0=unlimited); use 5 for smoke tests",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and log but do not write help_requests or update state files",
    )
    args = parser.parse_args()

    log.info(
        "=== agent_watchdog_daemon START | v%s | pid=%d | python=%s | "
        "max_flags=%d | dry_run=%s ===",
        WATCHDOG_VERSION,
        os.getpid(),
        sys.executable,
        args.max_flags,
        args.dry_run,
    )

    api_key = load_deepseek_api_key()
    if api_key:
        log.info("DeepSeek API key loaded (%s...)", api_key[:6])
    else:
        log.warning("DeepSeek API key not found — will use heuristic decomposition")

    try:
        n_scanned, n_in_window, n_flagged, n_escalated, n_mcp_stripped = run_scan(
            api_key, max_flags=args.max_flags, dry_run=args.dry_run
        )
        log.info(
            "RESULT: scanned=%d in_window=%d newly_flagged=%d escalated=%d mcp_stripped=%d",
            n_scanned,
            n_in_window,
            n_flagged,
            n_escalated,
            n_mcp_stripped,
        )
        if not args.dry_run:
            (WATCHDOG_DIR / "health.json").write_text(
                json.dumps(
                    {
                        "last_run": datetime.now(timezone.utc).isoformat(),
                        "scanned": n_scanned,
                        "in_window": n_in_window,
                        "newly_flagged": n_flagged,
                        "escalated": n_escalated,
                        "mcp_stripped": n_mcp_stripped,
                        "pid": os.getpid(),
                        "python": sys.executable,
                        "deepseek_key_loaded": api_key is not None,
                        "watchdog_version": WATCHDOG_VERSION,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
    except Exception:  # noqa: BLE001
        log.exception("Unhandled exception in watchdog scan")
        sys.exit(1)

    log.info("=== agent_watchdog_daemon END ===")
    _write_heartbeat("done")


if __name__ == "__main__":
    main()
