#!/usr/bin/env bash
# tcc-admin-canceler-subagent-inject (SubagentStart): inform child agents
# the admin-elevation dialog auto-cancel patch is live.
# Created 2026-05-20 by guardrail-100pct remediation.
set +e
LC_ALL=C
cat >/dev/null 2>&1
CTX=$(/usr/bin/python3 <<'PY' 2>/dev/null
import json, os, time
hb_path = "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/state/tcc_admin_canceler/heartbeat.json"
age = "?"
patch = "?"
if os.path.exists(hb_path) and os.path.getsize(hb_path) > 0:
    try:
        hb = json.load(open(hb_path))
        age = int(time.time() - hb.get("ts", 0))
        patch = hb.get("patch_present", "?")
    except Exception:
        pass
lines = [
    "## TCC-admin-canceler inheritance",
    f"- heartbeat age: {age}s, patch_present: {patch}",
    "- 'administer your computer' SecurityAgent prompts are auto-CANCELLED (NEVER granted) by tcc-dialog-detect/scan.applescript.",
    "- This is a security boundary: child agents must NEVER bypass admin elevation; the cancel is intentional.",
]
out = {"hookSpecificOutput": {"hookEventName": "SubagentStart", "additionalContext": "\n".join(lines)}}
print(json.dumps(out))
PY
)
[[ -n "$CTX" ]] && echo "$CTX"
exit 0
