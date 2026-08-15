#!/usr/bin/env bash
# auto-solve-on-error — PostToolUse hook
#
# Per CLAUDE.md §8 (auto-solve-on-issue) + feedback_auto_solve_on_issue.md, ANY
# error / blocker MUST trigger an immediate 3-helper solver fan-out — not a stop
# back to the user. This hook fires AFTER every Bash / Read / WebFetch / Edit / Task
# tool, inspects the result for failure signals, and:
#
#   - logs the failure to logs/auto_solve_triggers/<UTC>.log
#   - emits a stderr warning that the orchestrator sees as tool feedback,
#     instructing it to spawn solver helpers IMMEDIATELY
#
# Non-blocking: always exits 0. The Stop hook (auto-solve-violation-detector) is
# what actually enforces — this hook is the early warning so the orchestrator can
# fan-out BEFORE composing a "would you like to retry" reply.
#
# Cooldown: any single session_id only emits one warning per 60 seconds to avoid
# flooding tool results with stderr spam during a known-bad chain.
#
# Schema (PostToolUse):
#   { "session_id", "tool_name", "tool_input", "tool_response": {
#       "output": "...", "stderr": "...", "is_error": bool, "interrupted": bool, ...
#     }, ... }
#
# Detection signals (any one fires):
#   1. tool_response.is_error == true
#   2. tool_response.interrupted == true
#   3. Bash exit_code != 0 AND tool_name == "Bash"  (excluding grep rc=1 / diff rc=1)
#   4. stdout / stderr matches:
#        Traceback|^[[:space:]]*Error:|ERROR:|FATAL:|CRITICAL:|BLOCKED:|
#        Permission denied|ModuleNotFoundError|ImportError|HTTP\s+(4|5)\d\d|
#        ConnectionError|TimeoutError|RateLimitError
#
# Tunables (env):
#   AUTO_SOLVE_ERROR_DISABLE=1     → bypass detection (emergency escape hatch)
#   AUTO_SOLVE_ERROR_COOLDOWN_SEC  → override 60s default

set +e
LC_ALL=C

LOG_DIR="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/logs/auto_solve_triggers"
mkdir -p "$LOG_DIR" 2>/dev/null
LOG_FILE="$LOG_DIR/$(date -u +%Y-%m-%d).log"
TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
NOW=$(date +%s)

if [[ "${AUTO_SOLVE_ERROR_DISABLE:-0}" == "1" ]]; then
  exit 0
fi

PAYLOAD="$(cat 2>/dev/null)"
if [[ -z "$PAYLOAD" ]]; then exit 0; fi

# Parse fields via python (jq sometimes misses nested objects with embedded quotes)
read -r SID TOOL IS_ERROR INTERRUPTED EXIT_CODE TEXT < <(printf '%s' "$PAYLOAD" | python3 -c '
import sys, json
try:
    d = json.load(sys.stdin)
except Exception:
    print("","","","","","")
    sys.exit(0)

sid   = (d.get("session_id") or "").strip()
tool  = (d.get("tool_name") or "").strip()
resp  = d.get("tool_response") or {}
if not isinstance(resp, dict):
    resp = {"output": str(resp)}

is_err = bool(resp.get("is_error", False))
intr   = bool(resp.get("interrupted", False))
# Bash result schema: stdout/stderr/exit_code (Claude Code) or output (others)
exit_code = resp.get("exit_code")
if exit_code is None: exit_code = resp.get("returncode")
if exit_code is None: exit_code = ""

# Concatenate all textual fields for regex scanning, cap at 8000 chars
parts = []
for key in ("output", "stdout", "stderr", "error", "message", "content"):
    v = resp.get(key)
    if isinstance(v, str):
        parts.append(v)
    elif isinstance(v, list):
        for item in v:
            if isinstance(item, dict):
                t = item.get("text") or item.get("content") or ""
                if isinstance(t, str): parts.append(t)
            elif isinstance(item, str):
                parts.append(item)
text = " ".join(parts)[:8000].replace("\n", " ").replace("\r", " ")

# Quote-protect text so shell read sees one field
def shesc(s):
    return s.replace(" ", "\x01")  # we will split on space later; encode

print(sid, tool, str(is_err).lower(), str(intr).lower(), str(exit_code), shesc(text))
' 2>/dev/null)

# Decode the spaces back
TEXT=$(printf '%s' "$TEXT" | tr '\001' ' ')

if [[ -z "$SID" || -z "$TOOL" ]]; then exit 0; fi

# Cooldown per session
COOLDOWN="${AUTO_SOLVE_ERROR_COOLDOWN_SEC:-60}"
STATE_DIR="/tmp/cc-auto-solve-on-error"
mkdir -p "$STATE_DIR" 2>/dev/null
LAST_FILE="$STATE_DIR/${SID}.lastfire"

# Decide if this counts as an error
TRIGGER=""

if [[ "$IS_ERROR" == "true" ]]; then
  TRIGGER="is_error=true"
elif [[ "$INTERRUPTED" == "true" ]]; then
  TRIGGER="interrupted=true"
elif [[ "$TOOL" == "Bash" && -n "$EXIT_CODE" && "$EXIT_CODE" != "0" ]]; then
  # Exclude well-known benign non-zero exit codes from grep / diff
  CMD=$(printf '%s' "$PAYLOAD" | python3 -c 'import sys,json
try: d=json.load(sys.stdin); print((d.get("tool_input") or {}).get("command","") or "")
except Exception: print("")' 2>/dev/null)
  CMD_LOWER=$(printf '%s' "$CMD" | tr '[:upper:]' '[:lower:]')
  # Benign rc=1: a *first* token that is grep/egrep/fgrep/diff/test/cmp/find (with -quit)
  # We deliberately keep this conservative — any pipeline w/ multiple commands flags.
  if [[ "$EXIT_CODE" == "1" ]] && \
     [[ "$CMD_LOWER" =~ ^[[:space:]]*(grep|egrep|fgrep|diff|test|cmp)[[:space:]] ]] && \
     [[ ! "$CMD_LOWER" =~ '|'|';'|'&&'|'||' ]]; then
    : # benign — fall through
  else
    TRIGGER="bash exit_code=$EXIT_CODE"
  fi
fi

# Regex scan of output text (covers cases where tool reports success but text shows error)
if [[ -z "$TRIGGER" && -n "$TEXT" ]]; then
  if printf '%s' "$TEXT" | grep -qE 'Traceback \(most recent call last\)|ModuleNotFoundError|ImportError:|AttributeError:|TypeError:|FATAL:|^ERROR:|: ERROR |HTTP [45][0-9][0-9]\b| 4[0-9][0-9] (Bad|Unauthorized|Forbidden|Not Found)| 5[0-9][0-9] (Internal|Bad Gateway|Service Unavailable)|BLOCKED:|Permission denied|RateLimit|ConnectionError|TimeoutError'; then
    TRIGGER="regex-match"
  fi
fi

if [[ -z "$TRIGGER" ]]; then
  exit 0
fi

# Cooldown gate
LAST=0
if [[ -f "$LAST_FILE" ]]; then LAST=$(cat "$LAST_FILE" 2>/dev/null || echo 0); fi
SINCE=$(( NOW - LAST ))
if (( SINCE < COOLDOWN )); then
  echo "$TS sid=$SID tool=$TOOL trigger=\"$TRIGGER\" suppressed cooldown=${SINCE}s" >> "$LOG_FILE" 2>/dev/null
  exit 0
fi
echo "$NOW" > "$LAST_FILE"

# Log + emit warning
SNIP=$(printf '%s' "$TEXT" | head -c 300)
echo "$TS sid=$SID tool=$TOOL trigger=\"$TRIGGER\" snippet=\"$SNIP\"" >> "$LOG_FILE" 2>/dev/null

cat >&2 <<EOF
AUTO-SOLVE TRIGGER — tool=$TOOL reported $TRIGGER

Per CLAUDE.md §8 (auto-solve-on-issue) you MUST NOT surface this to the user as
a question. Spawn 3 parallel solver helpers in ONE message right now:

  1. INTERNET solver  — WebSearch + WebFetch for the exact error text
  2. GITHUB solver    — gh search code / issues / PRs for matching fix
  3. REPO-LOCAL solver — grep AI-Tools registry / cloned-repos for prior fix

Use native Agent (subagent_type: general-purpose) for each, in parallel. Each
helper inherits §3 + §8. Aggregate, apply lowest-risk fix silently, log to
AI-Tools/logs/auto_solve/<issue>_<UTC>.md.

Only escalate to user if (a) all 3 helpers report no known fix, (b) requires
money/messages/external accounts, (c) touches credentials, or (d) irreversible.

Log: $LOG_FILE
EOF

exit 0
