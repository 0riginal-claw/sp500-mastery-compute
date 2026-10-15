#!/usr/bin/env bash
# universal-resume-subagent-inject (SubagentStart): inject universal-resume
# status hint so spawned sub-agents inherit awareness of the checkpointer
# protocol. Non-blocking.
# Created 2026-05-20 by guardrail-100pct remediation.

set +e
LC_ALL=C
cat >/dev/null 2>&1

LOCAL_HB="/Users/orginal/.zg/state/universal_resume/heartbeat.json"
DRIVE_HB="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/state/universal_resume/heartbeat.json"

HB="$LOCAL_HB"
[[ ! -s "$HB" ]] && HB="$DRIVE_HB"

CTX=$(/usr/bin/python3 <<PYEOF 2>/dev/null
import json, os, time
hb_path = "$HB"
if not os.path.exists(hb_path) or os.path.getsize(hb_path) == 0:
    raise SystemExit(0)
try:
    hb = json.load(open(hb_path))
except Exception:
    raise SystemExit(0)
age = int(time.time() - hb.get("ts", 0))
sid = hb.get("session_id", "?")
lines = [
    "## Universal-resume inheritance (sub-agent posture)",
    f"- daemon parent session: {sid}, heartbeat age: {age}s",
    "- All 5 agent classes (claude_main/subagents, openclaw_main/subagents, ollama) are mirrored every 5s (post-2026-05-20 hardening).",
    "- Any worker is universally-resumable — write your durable state under AI-Tools/state/ or /Users/orginal/.zg/state/ and the daemon will checkpoint it.",
    "- If you crash mid-mission, the next session's SessionStart hook will surface LOST_<class>.md cards.",
]
out = {"hookSpecificOutput": {"hookEventName": "SubagentStart", "additionalContext": "\n".join(lines)}}
print(json.dumps(out))
PYEOF
)

[[ -n "$CTX" ]] && echo "$CTX"
exit 0
