#!/usr/bin/env bash
# autosolve-detect — PostToolUse hook (§8 auto-solve enforcement)
#
# Detects errors / failures in tool responses and registers a pending issue
# in the state file. The companion PreToolUse hook (autosolve-require) tracks
# turns_since and blocks further tool use if no 3-solver fan-out occurred
# within 5 turns.
#
# State file schema (JSONL, append-only):
#   {issue_id, detected_at, error_signature, solvers_spawned, turns_since, status}
#
# Matcher: ^(Bash|Read|Write|Edit|Task|Agent|mcp__.*)$
# Always exits 0 (non-blocking).
#
# Tunable:
#   AUTOSOLVE_DISABLE=1   → bypass

set +e
LC_ALL=C

STATE_DIR="$HOME/.claude/state"
mkdir -p "$STATE_DIR" 2>/dev/null
STATE_FILE="$STATE_DIR/autosolve_pending.jsonl"
LOG_DIR="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/logs/auto_solve"
mkdir -p "$LOG_DIR" 2>/dev/null
LOG_FILE="$LOG_DIR/autosolve_enforce.log"

if [[ "${AUTOSOLVE_DISABLE:-0}" == "1" ]]; then
  exit 0
fi

PAYLOAD="$(cat 2>/dev/null)"
[[ -z "$PAYLOAD" ]] && exit 0

# Parse + decide via python (robust to nested JSON / quotes).
# NOTE: cannot use `printf | python3 <<HEREDOC` — bash gives the heredoc to
# python's stdin and the piped payload is silently discarded. Pass via env.
export AUTOSOLVE_PAYLOAD="$PAYLOAD"
STATE_FILE="$STATE_FILE" LOG_FILE="$LOG_FILE" python3 <<'PY'
import sys, json, os, re, time, hashlib

raw = os.environ.get("AUTOSOLVE_PAYLOAD", "")
try:
    d = json.loads(raw)
except Exception:
    sys.exit(0)

tool = (d.get("tool_name") or "").strip()
sid  = (d.get("session_id") or "").strip()

# Matcher: Bash|Read|Write|Edit|Task|Agent|mcp__.*
if not re.match(r"^(Bash|Read|Write|Edit|Task|Agent|mcp__.*)$", tool):
    sys.exit(0)

# Bug-2 fix 2026-05-20: honor `# autosolve_skip:` annotations from parent / transcript
# Skip DETECT for tool calls whose tool_input OR session transcript contains an
# autosolve_skip marker. Sub-agents inherit the marker from their spawn brief.
def _scan_skip_marker(payload):
    blob_parts = []
    ti = payload.get("tool_input") or {}
    for k in ("prompt", "description", "command", "content", "new_string"):
        v = ti.get(k)
        if isinstance(v, str):
            blob_parts.append(v)
    tp = payload.get("transcript_path") or payload.get("transcriptPath") or ""
    if tp and os.path.isfile(tp):
        try:
            with open(tp, "rb") as fh:
                try:
                    fh.seek(-20480, 2)
                except OSError:
                    fh.seek(0)
                blob_parts.append(fh.read().decode("utf-8", "replace"))
        except Exception:
            pass
    sess = (payload.get("session_id") or "").strip()
    if sess:
        marker = os.path.join(os.path.expanduser("~/.claude/state"),
                              f"autosolve_skip_{sess}.marker")
        if os.path.isfile(marker):
            try:
                with open(marker) as fh:
                    blob_parts.append(fh.read())
            except Exception:
                pass
    return "# autosolve_skip:" in "\n".join(blob_parts)

if _scan_skip_marker(d):
    sys.exit(0)

resp = d.get("tool_response") or {}
if not isinstance(resp, dict):
    resp = {"output": str(resp)}

is_err = bool(resp.get("is_error", False))
interrupted = bool(resp.get("interrupted", False))

exit_code = resp.get("exit_code")
if exit_code is None: exit_code = resp.get("returncode")

# Gather text for regex scan
parts = []
for k in ("output", "stdout", "stderr", "error", "message", "content"):
    v = resp.get(k)
    if isinstance(v, str):
        parts.append(v)
    elif isinstance(v, list):
        for item in v:
            if isinstance(item, dict):
                t = item.get("text") or item.get("content") or ""
                if isinstance(t, str): parts.append(t)
            elif isinstance(item, str):
                parts.append(item)
text = "\n".join(parts)[:8000]

# Detect Python-source reads: tool is Read/Edit/Write AND content looks like Python code
# Heuristic: contains Python keywords like import, def, class, except, etc.
_IS_PYTHON_SOURCE = False
if tool in ("Read", "Edit", "Write"):
    _py_pattern = re.compile(
        r"(^|\n)\s*(import |from |def |class |# |@|except |async |elif |else:)",
        re.MULTILINE,
    )
    if _py_pattern.search(text):
        _IS_PYTHON_SOURCE = True

# Error regex — STRICT; no IGNORECASE; requires structural shape, not bare words.
# Matches:
#   - PythonStyle errors with [A-Z][a-z]+ prefix:  ValueError, RuntimeError, KeyError, RuntimeException
#   - Full traceback header:                       Traceback (most recent call last)
#   - Line-start error labels:                     ^FAIL:, ^FAILED:, ^ERROR:
#   - BLOCKED: prefix from our own hooks
#   - exit code N>=1
#   - Resource exhaustion phrases
#   - Common explicit error names
# Does NOT match: bare "Error", "Exception", "Traceback" in arbitrary content
# (file diffs, log lines containing the word, comments, docstrings, etc.)
ERR_RE = re.compile(
    r"\b[A-Z][a-z]+(?:Error|Exception)\b"          # ValueError, TypeError, RuntimeException
    r"|Traceback \(most recent call last\)"        # full python traceback header
    r"|\bexit code [1-9]\d*\b"                     # exit code N>=1
    r"|^(?:FAIL(?:ED)?|ERROR)[: ]"                 # line-start FAIL:/FAILED:/ERROR:
    r"|\bBLOCKED:"                                 # BLOCKED: prefix
    r"|\bout of (?:memory|disk|usage|quota)\b"     # resource exhaustion
    r"|\bSyntaxError\b|\bIndentationError\b"
    r"|\bsegmentation fault\b|\bcore dumped\b",
    re.MULTILINE,  # ^ matches line-start
)

# Architectural gate: only scan regex on tools whose response is actual command
# output (Bash/Task/Agent/MCP). Read/Edit/Write return verbatim file content —
# NEVER scan that for error patterns. The Python-source skip below is a secondary
# safety net for legacy fall-through cases.
SCAN_REGEX_TOOLS = {"Bash", "Task", "Agent"}
scan_regex = tool in SCAN_REGEX_TOOLS or tool.startswith("mcp__")

# Skip if the text starts with our own hook stderr prefix (recursion guard)
RECURSION_PREFIXES = ("[autosolve]", "BLOCKED: pending issue", "BLOCKED (OC violation)")
is_recursion = any(p in text for p in RECURSION_PREFIXES)

triggered = False
reason = ""
if is_err:
    triggered, reason = True, "is_error=true"
elif interrupted:
    triggered, reason = True, "interrupted=true"
elif tool == "Bash" and isinstance(exit_code, int) and exit_code >= 2:
    # exit_code >= 2 is a real error (grep/diff return 1 normally)
    triggered, reason = True, f"exit_code={exit_code}"
elif scan_regex and not _IS_PYTHON_SOURCE and not is_recursion and ERR_RE.search(text):
    m = ERR_RE.search(text)
    triggered, reason = True, f"regex:{m.group(0)[:40]}"

if not triggered:
    sys.exit(0)

# Build error signature: tool + first regex match + truncated text hash
sig_source = (tool + "|" + reason + "|" + text[:200]).encode("utf-8", "replace")
sig = hashlib.sha1(sig_source).hexdigest()[:16]
issue_id = f"iss_{int(time.time())}_{sig[:8]}"

# Dedupe: if last 5 records contain same signature within 60s, skip new row
state_file = os.environ["STATE_FILE"]
recent = []
try:
    with open(state_file, "r") as fh:
        for line in fh.readlines()[-20:]:
            try: recent.append(json.loads(line))
            except: pass
except FileNotFoundError:
    pass

now = int(time.time())
for r in recent[-5:]:
    if (r.get("error_signature") == sig
        and r.get("status") == "pending"
        and (now - int(r.get("detected_at", 0))) < 60):
        sys.exit(0)  # dedupe

row = {
    "issue_id": issue_id,
    "detected_at": now,
    "session_id": sid,
    "tool": tool,
    "error_signature": sig,
    "reason": reason,
    "snippet": text[:200].replace("\n", " "),
    "solvers_spawned": 0,
    "turns_since": 0,
    "status": "pending",
}

with open(state_file, "a") as fh:
    fh.write(json.dumps(row) + "\n")

# Audit log
try:
    with open(os.environ["LOG_FILE"], "a") as fh:
        fh.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(now))}] "
                 f"DETECT {issue_id} tool={tool} reason={reason}\n")
except Exception:
    pass

# stderr nudge (PostToolUse — non-blocking)
sys.stderr.write(
    f"[autosolve] pending issue {issue_id} detected ({reason}). "
    f"§8 requires 3 solver spawns (INTERNET + GITHUB + REPO-LOCAL) within 5 turns.\n"
)
PY

exit 0
