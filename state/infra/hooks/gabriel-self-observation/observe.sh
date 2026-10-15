#!/usr/bin/env bash
# gabriel-self-observation — PostToolUse signal (Hook 3 of 6)
#
# Append every tool-use event (after it executes) to
# state/gabriel_self/observations_<DATE>.jsonl so the autonomous-mode
# daemon's _reflect() loop has raw material for learning.
#
# Cheap (~1ms). Atomic single-line append (POSIX guarantee <4KB).

set +e
LC_ALL=C

ROOT="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools"
STATE_DIR="$ROOT/state/gabriel_self"
TODAY=$(date -u +%Y-%m-%d)
OBS_FILE="$STATE_DIR/observations_${TODAY}.jsonl"

mkdir -p "$STATE_DIR" 2>/dev/null

# Read JSON payload from PostToolUse (tool_name, tool_input, tool_response, etc.)
INPUT="$(cat 2>/dev/null || true)"

NOW_ISO=$(date -u +%Y-%m-%dT%H:%M:%SZ)
NOW_UNIX=$(date +%s)

# Use python to parse + summarize without leaking large payloads
ENTRY=$(INPUT="$INPUT" NOW_ISO="$NOW_ISO" NOW_UNIX="$NOW_UNIX" python3 - <<'PY' 2>/dev/null
import json, os, sys
raw = os.environ.get("INPUT", "") or "{}"
try:
    payload = json.loads(raw) if raw.strip() else {}
except json.JSONDecodeError:
    payload = {}

tool_name = payload.get("tool_name") or payload.get("toolName") or ""
tool_input = payload.get("tool_input") or payload.get("toolInput") or {}
tool_response = payload.get("tool_response") or payload.get("toolResponse") or {}

# Success heuristic: response present + no error field
success = True
if isinstance(tool_response, dict):
    if tool_response.get("is_error") or tool_response.get("error"):
        success = False

# Param summary (truncated, names + lengths only — avoid leaking secrets/large blobs)
param_summary = {}
if isinstance(tool_input, dict):
    for k, v in list(tool_input.items())[:8]:
        if isinstance(v, str):
            param_summary[k] = f"str[{len(v)}]"
        elif isinstance(v, (list, tuple)):
            param_summary[k] = f"list[{len(v)}]"
        elif isinstance(v, dict):
            param_summary[k] = f"dict[{len(v)}]"
        elif isinstance(v, bool):
            param_summary[k] = bool(v)
        elif isinstance(v, (int, float)):
            param_summary[k] = v
        else:
            param_summary[k] = type(v).__name__

out = {
    "ts": os.environ.get("NOW_ISO"),
    "unix": int(os.environ.get("NOW_UNIX", "0") or 0),
    "tool_name": tool_name,
    "success": success,
    "params_summary": param_summary,
    "source": "gabriel-self-observation",
}
print(json.dumps(out, separators=(",", ":")))
PY
)

if [ -n "$ENTRY" ]; then
  # POSIX-atomic for <4KB lines
  echo "$ENTRY" >> "$OBS_FILE" 2>/dev/null
fi

exit 0
