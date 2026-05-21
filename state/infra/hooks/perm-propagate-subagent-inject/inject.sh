#!/usr/bin/env bash
# perm-propagate-subagent-inject (SubagentStart): tell child agents the
# perm-propagation chain is enforced by autosolve-require + spawn-validator
# hooks (3-solver fan-out + #model_reason in every spawn).
# Created 2026-05-20 by guardrail-100pct remediation.
set +e
LC_ALL=C
cat >/dev/null 2>&1
CTX=$(/usr/bin/python3 <<'PY' 2>/dev/null
import json, os, time
hb_path = "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/state/perm_propagate/heartbeat.json"
age = "?"
if os.path.exists(hb_path) and os.path.getsize(hb_path) > 0:
    try:
        hb = json.load(open(hb_path))
        age = int(time.time() - hb.get("ts", 0))
    except Exception:
        pass
lines = [
    "## Perm-propagate inheritance (grandchild permission marker)",
    f"- marker heartbeat age: {age}s",
    "- Permission/protocol propagation is enforced at the hook layer (autosolve-require, spawn-validator, model-routing-check).",
    "- Every spawn MUST include `# model_reason: <one-line>` near the top.",
    "- ANY error/blocker → spawn 3 helpers (INTERNET + GITHUB + REPO-LOCAL) within 5 turns, or include `# autosolve_skip: <reason>`.",
]
out = {"hookSpecificOutput": {"hookEventName": "SubagentStart", "additionalContext": "\n".join(lines)}}
print(json.dumps(out))
PY
)
[[ -n "$CTX" ]] && echo "$CTX"
exit 0
