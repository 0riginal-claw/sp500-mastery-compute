#!/usr/bin/env bash
# sync-hooks-local-subagent-inject (SubagentStart): tell child agents about mirror.
# Created 2026-05-20 by guardrail-100pct remediation.
set +e
LC_ALL=C
cat >/dev/null 2>&1
STATE_DIR="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/state/sync_hooks_local"
HB="$STATE_DIR/heartbeat.json"
CTX=$(/usr/bin/python3 <<'PY' 2>/dev/null
import json, os, time
hb_path = "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/state/sync_hooks_local/heartbeat.json"
age = "?"
if os.path.exists(hb_path) and os.path.getsize(hb_path) > 0:
    try:
        hb = json.load(open(hb_path))
        age = int(time.time() - hb.get("ts", 0))
    except Exception:
        pass
lines = [
    "## Sync-hooks-local inheritance",
    f"- mirror heartbeat age: {age}s",
    "- Hook scripts mirrored from Drive canonical path to /Users/orginal/.zg/ every 5 min for fast cold-cache reads.",
]
out = {"hookSpecificOutput": {"hookEventName": "SubagentStart", "additionalContext": "\n".join(lines)}}
print(json.dumps(out))
PY
)
[[ -n "$CTX" ]] && echo "$CTX"
exit 0
