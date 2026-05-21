#!/usr/bin/env bash
# session-resume-bootstrap (SessionStart): emit additionalContext if a prior
# session checkpoint exists. Non-blocking; never fails the hook chain.
# Created 2026-05-20 by mega-builder Fix 4.

set +e
LC_ALL=C

LOCAL_STATE="/Users/orginal/.zg/state/session_resume"
DRIVE_STATE="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/state/session_resume"
LAST="$LOCAL_STATE/last_known_session.json"
[[ ! -s "$LAST" ]] && LAST="$DRIVE_STATE/last_known_session.json"
[[ ! -s "$LAST" ]] && exit 0

CTX=$(python3 <<'PY' 2>/dev/null
import json, time, os, glob
local = "/Users/orginal/.zg/state/session_resume"
drive = "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/state/session_resume"

last_path = None
for c in (local + "/last_known_session.json", drive + "/last_known_session.json"):
    if os.path.exists(c):
        last_path = c
        break
if not last_path:
    raise SystemExit(0)

with open(last_path) as f:
    last = json.load(f)
sid = last.get("session_id", "?")

cp_path = None
for d in (local, drive):
    cand = os.path.join(d, f"checkpoint_{sid}.json")
    if os.path.exists(cand):
        cp_path = cand
        break

ctx = f"## Session-resume context\n- prior session_id: {sid}\n- last checkpoint ts: {last.get('ts', '?')}\n"
if cp_path:
    try:
        with open(cp_path) as f:
            cp = json.load(f)
        host = cp.get("host", {})
        ctx += f"- load: {host.get('load1', '?')}/{host.get('load5', '?')}/{host.get('load15', '?')}\n"
        ctx += f"- drive_fuse_ok: {host.get('drive_fuse_ok', '?')}\n"
        ctx += f"- daemons_tracked: {len(cp.get('daemons', []))}\n"
    except Exception:
        pass

print(json.dumps({"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": ctx}}))
PY
)

[[ -n "$CTX" ]] && echo "$CTX"
exit 0
