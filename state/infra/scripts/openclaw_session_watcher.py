#!/usr/bin/env python3
"""
openclaw_session_watcher.py — poll OC session files; emit §8/§3 violation signals
to $HOME/.claude/state/openclaw_session_violations.jsonl so autosolve-require
hook can gate Claude Code tool calls.

Actual OC JSONL event types (observed 2026-05-19):
  type=session          → session started (has id, timestamp ISO)
  type=message          → conversation turn; role=assistant content[].type=toolCall
                          for tool calls; role=toolResult for results
  type=custom           → misc; customType=openclaw:prompt-error for errors
"""
# autosolve_skip: building watcher infra

import argparse
import glob
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

# autosolve_skip: REPO-LOCAL solver  # fanout_skip: bounded single-file path fix
_AT_FALLBACK = (
    "/Users/orginal/Library/CloudStorage/"
    "GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools"
)
AT = os.environ.get("AI_TOOLS_ROOT", _AT_FALLBACK)
# In launchd: HOME env var is overridden to AT/home.
# When run directly, HOME is real user home; we still use AT/home for OC/Claude state.
HOME_REDIR = os.path.join(AT, "home")

OC_SESSIONS_GLOB = os.path.join(HOME_REDIR, ".openclaw", "agents", "*", "sessions", "*.jsonl")
VIOLATIONS_FILE = os.path.join(HOME_REDIR, ".claude", "state", "openclaw_session_violations.jsonl")
LOG_FILE = os.path.join(AT, "logs", "openclaw_session_watcher.log")

ACTIVE_MTIME_WINDOW = 30 * 60       # sessions modified within 30 min are "active"
FAN_OUT_ELAPSED_THRESHOLD = 300      # 5 min
FAN_OUT_SPAWN_THRESHOLD = 3          # need ≥3 sub-spawns to satisfy §3
VIOLATION_SIGNAL_TYPES = ("tool_error", "fan_out_needed")

# Sub-spawn tool name patterns (OC tool calls that spawn child agents)
SUB_SPAWN_NAMES = ("llm-task", "task", "spawn", "agent")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("oc_watcher")


def _parse_ts(ts_str):
    """ISO or epoch float → epoch float. Returns None on failure."""
    if ts_str is None:
        return None
    if isinstance(ts_str, (int, float)):
        v = float(ts_str)
        # epoch ms vs epoch s heuristic
        return v / 1000.0 if v > 1e12 else v
    try:
        s = str(ts_str).rstrip("Z")
        dt = datetime.fromisoformat(s)
        return dt.replace(tzinfo=timezone.utc).timestamp()
    except Exception:
        pass
    try:
        return float(ts_str) / 1000.0 if float(ts_str) > 1e12 else float(ts_str)
    except Exception:
        return None


def _is_error_in_text(text):
    """Heuristic: does a toolResult text body indicate an error?"""
    lower = text.lower()
    for pat in ('"status": "error"', '"status":"error"', ": error", "no such file",
                "command aborted", "timed out", "traceback", "exception:",
                " failed", "not found", "permission denied"):
        if pat in lower:
            return True
    return False


def scan_files(offsets, sessions):
    """
    Scan OC session JSONL files. Update `sessions` in-place.
    offsets: {path: byte_offset}
    sessions: {session_id: {start_ts, tool_error_count, sub_spawns, path, pid}}
    Returns list of new violation dicts to emit.
    """
    now = time.time()
    violations = []

    for path in glob.glob(OC_SESSIONS_GLOB):
        # skip trajectory files and lock files
        basename = os.path.basename(path)
        if "trajectory" in basename or basename.endswith(".lock"):
            continue

        try:
            st = os.stat(path)
        except OSError:
            continue

        # only process recently modified files (active sessions)
        mtime_age = now - st.st_mtime
        if mtime_age > ACTIVE_MTIME_WINDOW and path in offsets:
            continue  # inactive and already partially read — skip

        offset = offsets.get(path, 0)
        try:
            with open(path, "rb") as fh:
                fh.seek(offset)
                new_bytes = fh.read()
                offsets[path] = fh.tell()
        except OSError:
            continue

        if not new_bytes:
            continue

        for raw_line in new_bytes.split(b"\n"):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                evt = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            etype = evt.get("type", "")
            data = evt.get("data", {})

            # ── SESSION STARTED ──────────────────────────────────────────────
            # actual format: type="session" with id, timestamp
            # brief format:  type="session.started" with ts, sessionId, data.pid
            if etype in ("session", "session.started"):
                sid = evt.get("id") or evt.get("sessionId") or (data.get("sessionId") if isinstance(data, dict) else None)
                ts_raw = evt.get("timestamp") or evt.get("ts") or (data.get("ts") if isinstance(data, dict) else None)
                pid = (data.get("pid") if isinstance(data, dict) else None)
                ts = _parse_ts(ts_raw)
                if sid and ts and sid not in sessions:
                    sessions[sid] = {
                        "start_ts": ts,
                        "tool_error_count": 0,
                        "sub_spawns": 0,
                        "path": path,
                        "pid": pid,
                    }
                continue

            # ── TOOL CALL (sub-spawn detection) ──────────────────────────────
            # brief format: type="tool_call"
            if etype == "tool_call":
                tool_name = evt.get("name", "") or str(evt.get("tool", ""))
                sid = evt.get("sessionId") or evt.get("session_id")
                if sid and any(p in tool_name.lower() for p in SUB_SPAWN_NAMES):
                    if sid in sessions:
                        sessions[sid]["sub_spawns"] += 1
                continue

            # ── TOOL RESULT ──────────────────────────────────────────────────
            # brief format: type="tool_result" with is_error, exit_code
            if etype == "tool_result":
                sid = evt.get("sessionId") or evt.get("session_id")
                is_error = (
                    evt.get("is_error") is True
                    or evt.get("exit_code") not in (0, None)
                    or "Error" in json.dumps(data)
                )
                if is_error and sid and sid in sessions:
                    sessions[sid]["tool_error_count"] += 1
                continue

            # ── MESSAGE (actual format) ───────────────────────────────────────
            if etype == "message":
                msg = evt.get("message", {})
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role", "")
                content = msg.get("content", [])
                if not isinstance(content, list):
                    content = []

                # Infer session_id from path: JSONL filename is <session_id>.jsonl
                inferred_sid = basename.replace(".jsonl", "")

                # Sub-spawn: assistant toolCall with spawn-like name
                if role == "assistant":
                    for item in content:
                        if not isinstance(item, dict):
                            continue
                        if item.get("type") == "toolCall":
                            tool_name = str(item.get("name", ""))
                            if any(p in tool_name.lower() for p in SUB_SPAWN_NAMES):
                                if inferred_sid in sessions:
                                    sessions[inferred_sid]["sub_spawns"] += 1

                # Tool result: check for error indicators
                if role == "toolResult":
                    has_error = False
                    for item in content:
                        if isinstance(item, dict):
                            text = item.get("text", "")
                            if isinstance(text, str) and _is_error_in_text(text):
                                has_error = True
                                break
                    if has_error and inferred_sid in sessions:
                        sessions[inferred_sid]["tool_error_count"] += 1
                continue

            # ── CUSTOM ERROR (actual format) ─────────────────────────────────
            if etype == "custom" and evt.get("customType") == "openclaw:prompt-error":
                sid = (data.get("sessionId") if isinstance(data, dict) else None)
                if sid and sid in sessions:
                    sessions[sid]["tool_error_count"] += 1
                elif sid and sid not in sessions:
                    # session start may have been missed; synthesize minimal entry
                    ts_raw = data.get("timestamp") if isinstance(data, dict) else None
                    ts = _parse_ts(ts_raw) or now
                    sessions[sid] = {
                        "start_ts": ts,
                        "tool_error_count": 1,
                        "sub_spawns": 0,
                        "path": path,
                        "pid": None,
                    }
                continue

        # After reading new bytes, check active sessions whose file is this path
        if mtime_age <= ACTIVE_MTIME_WINDOW:
            for sid, s in list(sessions.items()):
                if s.get("path") != path:
                    continue
                elapsed = now - s["start_ts"]
                # tool_error signal
                if s["tool_error_count"] > 0:
                    violations.append((sid, "tool_error", elapsed, s))
                # fan_out_needed signal
                if elapsed > FAN_OUT_ELAPSED_THRESHOLD and s["sub_spawns"] < FAN_OUT_SPAWN_THRESHOLD:
                    violations.append((sid, "fan_out_needed", elapsed, s))

    return violations


def append_violations(new_violations, emitted):
    """Write new violations to JSONL; dedupe via emitted set."""
    for sid, signal_type, elapsed, s in new_violations:
        key = (sid, signal_type)
        if key in emitted:
            continue
        emitted.add(key)
        row = {
            "session_id": sid,
            "pid": s.get("pid"),
            "signal_type": signal_type,
            "timestamp": int(time.time()),
            "elapsed_s": round(elapsed, 1),
            "tool_error_count": s["tool_error_count"],
            "sub_spawns": s["sub_spawns"],
            "session_path": s["path"],
        }
        try:
            os.makedirs(os.path.dirname(VIOLATIONS_FILE), exist_ok=True)
            with open(VIOLATIONS_FILE, "a") as fh:
                fh.write(json.dumps(row) + "\n")
                fh.flush()
        except OSError as exc:
            log.error("cannot write violations: %s", exc)
        log.info("VIOLATION session=%s signal=%s elapsed=%.0fs errors=%d spawns=%d",
                 sid[:8], signal_type, elapsed, s["tool_error_count"], s["sub_spawns"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="run one scan and exit")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    log.info("START smoke=%s", args.smoke)

    offsets = {}
    sessions = {}
    emitted = set()

    if args.smoke:
        violations = scan_files(offsets, sessions)
        append_violations(violations, emitted)
        print(f"smoke: sessions={len(sessions)} violations={len(violations)}")
        # print each violation
        for sid, sig, elapsed, s in violations:
            print(f"  {sid[:8]} {sig} elapsed={elapsed:.0f}s errors={s['tool_error_count']} spawns={s['sub_spawns']}")
        sys.exit(0)

    while True:
        try:
            violations = scan_files(offsets, sessions)
            append_violations(violations, emitted)
        except Exception as exc:
            log.exception("scan error: %s", exc)
        time.sleep(5)


if __name__ == "__main__":
    main()
