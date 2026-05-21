#!/usr/bin/env python3
"""hook_error_watcher.py - Phase B of hook-error spec.

Tails ~/.claude/projects/*/*.jsonl session transcripts and scans new
lines for hook errors ("PreToolUse|PostToolUse|Stop hook error").

Findings appended to $HOME/.claude/state/hook_errors/<YYYY-MM-DD>.jsonl.
Tracks per-file byte cursors at hook_errors/_cursors.json.
Heartbeat at hook_errors/heartbeat.json (guardrail-grade).

Cycle = 10s (HOOK_ERR_CYCLE_SEC override). Daemonizes via launchd.
"""
from __future__ import annotations
import json, os, re, sys, time, signal
from pathlib import Path
from datetime import datetime, timezone

HOME = Path(os.environ.get("HOOK_ERR_HOME") or os.path.expanduser("~"))
STATE_DIR = HOME / ".claude" / "state" / "hook_errors"
PROJECTS_DIR = HOME / ".claude" / "projects"
CURSOR_FILE = STATE_DIR / "_cursors.json"
HEARTBEAT_FILE = STATE_DIR / "heartbeat.json"

HOOK_ERR_RE = re.compile(
    r"(PreToolUse|PostToolUse|UserPromptSubmit|Stop|SessionStart|SubagentStart)[:\s][^\n]{0,80}?hook\s+error",
    re.IGNORECASE,
)
EXTRA_PATTERNS = [
    re.compile(r"No such file or directory.*\.(sh|py|js)\b", re.IGNORECASE),
    re.compile(r"hook command timed out", re.IGNORECASE),
]
CYCLE_SEC = float(os.environ.get("HOOK_ERR_CYCLE_SEC", "10"))
SNIPPET_LEN = 400

def now_unix() -> int: return int(time.time())
def now_iso() -> str: return datetime.now(tz=timezone.utc).isoformat()

def atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)
    os.replace(tmp, path)

def load_cursors() -> dict:
    if not CURSOR_FILE.exists(): return {}
    try: return json.loads(CURSOR_FILE.read_text(encoding="utf-8"))
    except Exception: return {}

def save_cursors(cur: dict) -> None: atomic_write_json(CURSOR_FILE, cur)

def write_heartbeat(cycle_id: str, status: str, extra=None) -> None:
    obj = {"ts": now_unix(), "iso": now_iso(), "pid": os.getpid(),
           "cycle_id": cycle_id, "status": status}
    if extra: obj.update(extra)
    atomic_write_json(HEARTBEAT_FILE, obj)

def append_finding(finding: dict) -> None:
    date = datetime.fromtimestamp(finding["ts"], tz=timezone.utc).strftime("%Y-%m-%d")
    out = STATE_DIR / f"{date}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(finding) + "\n")

def scan_line(line: str):
    if HOOK_ERR_RE.search(line):
        m = HOOK_ERR_RE.search(line)
        phase = (m.group(1) if m else "?").strip()
        snippet = line[max(0, m.start()-60): m.start()+SNIPPET_LEN].strip() if m else line[:SNIPPET_LEN]
        return phase, snippet
    for p in EXTRA_PATTERNS:
        m = p.search(line)
        if m:
            return "?", line[max(0, m.start()-60): m.start()+SNIPPET_LEN].strip()
    return None, None

def extract_tool(line: str) -> str:
    m = re.search(r'"tool(?:_name)?"\s*:\s*"([^"]{1,60})"', line)
    return m.group(1) if m else "?"

def scan_file(path: Path, cursor: int):
    n = 0
    try: size = path.stat().st_size
    except OSError: return cursor, 0
    if size < cursor: cursor = 0
    if size == cursor: return cursor, 0
    try:
        with open(path, "rb") as fh:
            fh.seek(cursor)
            chunk = fh.read(size - cursor)
            new_cursor = size
    except OSError:
        return cursor, 0
    try: text = chunk.decode("utf-8", errors="replace")
    except Exception: return new_cursor, 0
    offset = cursor
    for line in text.split("\n"):
        line_len = len(line.encode("utf-8", errors="replace")) + 1
        if line.strip():
            phase, snippet = scan_line(line)
            if phase is not None:
                tool = extract_tool(line)
                append_finding({
                    "ts": now_unix(), "iso": now_iso(),
                    "session_id": path.stem, "project_dir": path.parent.name,
                    "phase": phase, "tool": tool,
                    "snippet": (snippet or "")[:SNIPPET_LEN],
                    "raw_line_offset": offset,
                })
                n += 1
        offset += line_len
    return new_cursor, n

def cycle_once(cursors: dict):
    if not PROJECTS_DIR.exists(): return 0, 0
    n_files = n_findings = 0
    for proj_dir in PROJECTS_DIR.iterdir():
        if not proj_dir.is_dir(): continue
        for jsonl in proj_dir.glob("*.jsonl"):
            n_files += 1
            key = f"{proj_dir.name}/{jsonl.name}"
            cur = int(cursors.get(key, 0))
            new_cur, found = scan_file(jsonl, cur)
            if new_cur != cur or found:
                cursors[key] = new_cur
                n_findings += found
    return n_files, n_findings

def main_loop() -> int:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    cycle = 0
    cursors = load_cursors()
    write_heartbeat("startup", "starting", {"projects_dir": str(PROJECTS_DIR)})
    def _term(_s,_f): write_heartbeat(f"c{cycle}", "shutdown"); sys.exit(0)
    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)
    while True:
        cycle += 1
        cid = f"c{cycle}"
        try:
            n_files, n_findings = cycle_once(cursors)
            save_cursors(cursors)
            write_heartbeat(cid, "ok", {"files_scanned": n_files, "findings": n_findings})
        except Exception as e:
            write_heartbeat(cid, "error", {"err": repr(e)[:200]})
        time.sleep(CYCLE_SEC)

def one_shot() -> int:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    cursors = load_cursors()
    n_files, n_findings = cycle_once(cursors)
    save_cursors(cursors)
    write_heartbeat("oneshot", "ok", {"files_scanned": n_files, "findings": n_findings})
    print(json.dumps({"files_scanned": n_files, "findings": n_findings}))
    return 0

if __name__ == "__main__":
    if "--once" in sys.argv: sys.exit(one_shot())
    sys.exit(main_loop())
