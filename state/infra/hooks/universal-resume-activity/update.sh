#!/usr/bin/env bash
# universal-resume-activity (PostToolUse): touch last_session_activity.unix
# so daemon knows session is active. Non-blocking.
# Created 2026-05-20 by guardrail-100pct remediation.

set +e
LC_ALL=C
# Consume stdin once, retain for tool-id parsing (Phase D)
HOOK_STDIN=$(cat 2>/dev/null)

# --- Phase D: clear inflight tool-call ledger entry ------------------
if [[ -n "$HOOK_STDIN" ]]; then
  /usr/bin/python3 - <<PYEOF >/dev/null 2>&1 &
import json, os, time
try:
    data = json.loads('''$HOOK_STDIN''')
except Exception:
    raise SystemExit(0)
tool_id = data.get("tool_use_id") or data.get("toolUseId")
if not tool_id:
    raise SystemExit(0)
fn = "/Users/orginal/.zg/state/universal_resume/_inflight/" + str(tool_id).replace("/", "_") + ".json"
try:
    # Mark complete by writing 'done' phase first (audit trail), then unlink
    if os.path.exists(fn):
        with open(fn) as f:
            entry = json.load(f)
        entry["completed_ts"] = time.time()
        entry["phase"] = "done"
        # Move to _completed/ rather than delete: keeps short audit trail
        done_dir = "/Users/orginal/.zg/state/universal_resume/_inflight/_completed"
        os.makedirs(done_dir, exist_ok=True)
        # Cap _completed to 50 most-recent (lazy gc)
        try:
            existing = sorted(os.listdir(done_dir), key=lambda n: os.path.getmtime(os.path.join(done_dir, n)))
            for old in existing[:-49]:
                try: os.remove(os.path.join(done_dir, old))
                except Exception: pass
        except Exception:
            pass
        dst = os.path.join(done_dir, os.path.basename(fn))
        try:
            with open(dst, "w") as f:
                f.write(json.dumps(entry))
        except Exception:
            pass
        try: os.remove(fn)
        except Exception: pass
except Exception:
    pass
PYEOF
fi

LOCAL_DIR="/Users/orginal/.zg/state/universal_resume"
DRIVE_DIR="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/state/universal_resume"

mkdir -p "$LOCAL_DIR" 2>/dev/null
TS_FILE="$LOCAL_DIR/last_session_activity.unix"
TMP="$TS_FILE.tmp.$$"
date +%s > "$TMP" 2>/dev/null && mv "$TMP" "$TS_FILE" 2>/dev/null

# Mirror to Drive (best-effort, non-blocking)
mkdir -p "$DRIVE_DIR" 2>/dev/null
cp "$TS_FILE" "$DRIVE_DIR/last_session_activity.unix" 2>/dev/null &

exit 0
