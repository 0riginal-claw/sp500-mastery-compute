#!/usr/bin/env bash
# session-resume-subagent-inject (SubagentStart): inject prior-session
# resume hint into spawned sub-agent so they inherit context awareness.
# Non-blocking; never fails the hook chain.
# Created 2026-05-20 by six-fail-fix F4b.

set +e
LC_ALL=C

# Drain stdin
cat >/dev/null 2>&1

LOCAL_DIR="/Users/orginal/.zg/state/session_resume"
DRIVE_DIR="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/state/session_resume"

HB="$LOCAL_DIR/heartbeat.json"
[[ ! -s "$HB" ]] && HB="$DRIVE_DIR/heartbeat.json"

CTX=$(python3 <<PYEOF 2>/dev/null
import json, time, os
hb_path = "$HB"
if not os.path.exists(hb_path) or os.path.getsize(hb_path) == 0:
    raise SystemExit(0)
try:
    hb = json.loads(open(hb_path).read())
except Exception:
    raise SystemExit(0)
age = int(time.time() - hb.get("ts", 0))
sid = hb.get("session_id", "?")
ndaemons = len(hb.get("daemons", []) or [])
lines = [
    "## Session-resume inheritance (sub-agent context)",
    f"- parent session_id: {sid}, heartbeat age: {age}s, daemons_tracked: {ndaemons}",
    "- Resume protocol: write 30-60s checkpoints to state/session_resume/ if mission >5 min.",
    "- Universal-resume daemon will checkpoint your tool-use into the parent session's tool_use.jsonl.",
]
out = {"hookSpecificOutput": {"hookEventName": "SubagentStart", "additionalContext": "\n".join(lines)}}
print(json.dumps(out))
PYEOF
)

[[ -n "$CTX" ]] && echo "$CTX"
exit 0
