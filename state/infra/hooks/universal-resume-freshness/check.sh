#!/usr/bin/env bash
# universal-resume-freshness (PreToolUse): verify universal_resume daemon
# heartbeat is fresh (<10 min). Auto-respawn via launchctl bootstrap if
# missing or stale. Non-blocking — never fails the hook chain.
# Created 2026-05-20 by guardrail-100pct remediation.

set +e
LC_ALL=C
# Consume stdin once, retain for tool-id parsing (Phase D)
HOOK_STDIN=$(cat 2>/dev/null)

# --- Phase D: tool-call inflight ledger -------------------------------
# Record this tool call as "started" so a crash mid-call leaves a ledger
# entry that SessionStart can surface as "resume from tool call X".
INFLIGHT_DIR="/Users/orginal/.zg/state/universal_resume/_inflight"
mkdir -p "$INFLIGHT_DIR" 2>/dev/null
if [[ -n "$HOOK_STDIN" ]]; then
  /usr/bin/python3 - <<PYEOF >/dev/null 2>&1 &
import json, os, sys, time
try:
    data = json.loads('''$HOOK_STDIN''')
except Exception:
    raise SystemExit(0)
tool_name = data.get("tool_name") or data.get("toolName") or "unknown"
tool_id = data.get("tool_use_id") or data.get("toolUseId") or f"pid_{os.getpid()}_{int(time.time()*1000)}"
sess = data.get("session_id") or os.environ.get("CLAUDE_SESSION_ID") or "unknown"
ledger = {
    "tool_id": tool_id,
    "tool_name": tool_name,
    "session_id": sess,
    "started_ts": time.time(),
    "phase": "pre",
}
fn = "/Users/orginal/.zg/state/universal_resume/_inflight/" + str(tool_id).replace("/", "_") + ".json"
tmp = fn + ".tmp"
try:
    with open(tmp, "w") as f:
        f.write(json.dumps(ledger))
    os.replace(tmp, fn)
except Exception:
    pass
PYEOF
fi

LOCAL_HB="/Users/orginal/.zg/state/universal_resume/heartbeat.json"
DRIVE_HB="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/state/universal_resume/heartbeat.json"
PLIST="/Users/orginal/Library/LaunchAgents/com.zg.universal_resume_guardrail.plist"
LOG="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/logs/universal_resume_freshness.log"
mkdir -p "$(dirname "$LOG")" 2>/dev/null

HB="$LOCAL_HB"
[[ ! -s "$HB" ]] && HB="$DRIVE_HB"

NEED_RESPAWN=0
if [[ ! -s "$HB" ]]; then
  NEED_RESPAWN=1
  REASON="heartbeat_missing"
else
  AGE=$(/usr/bin/python3 -c "
import json, time
try:
    hb = json.load(open('$HB'))
    print(int(time.time() - hb.get('ts', 0)))
except Exception:
    print(99999)
" 2>/dev/null)
  if [[ "$AGE" =~ ^[0-9]+$ ]] && [[ "$AGE" -gt 600 ]]; then
    NEED_RESPAWN=1
    REASON="stale_${AGE}s"
  fi
fi

if [[ "$NEED_RESPAWN" == "1" ]]; then
  if ! /bin/launchctl list 2>/dev/null | /usr/bin/grep -q "com.zg.universal_resume_guardrail"; then
    if [[ -f "$PLIST" ]]; then
      /bin/launchctl bootstrap "gui/$(id -u)" "$PLIST" 2>>"$LOG"
      echo "[$(date +%FT%TZ)] universal-resume-freshness: respawned daemon ($REASON)" >> "$LOG"
    fi
  else
    echo "[$(date +%FT%TZ)] universal-resume-freshness: daemon loaded but $REASON; kickstart" >> "$LOG"
    /bin/launchctl kickstart -k "gui/$(id -u)/com.zg.universal_resume_guardrail" 2>>"$LOG"
  fi
fi

exit 0
