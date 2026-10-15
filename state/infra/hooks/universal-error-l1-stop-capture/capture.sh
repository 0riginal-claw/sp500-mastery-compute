#!/usr/bin/env bash
# L1 — Stop hook: scan current session JSONL for hook errors / blocked tool calls / etc.
# and write them into the error pile so the watcher daemon sees them too.
#
# Schema for Stop hook payload: {"session_id", "transcript_path", "stop_hook_active", ...}
# We read the transcript JSONL (last 200 lines) and grep for error markers.

set +e
LC_ALL=C
LOCAL_PILE="/Users/orginal/.zg/state/error_pile"
DRIVE_PILE="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/state/error_pile"
mkdir -p "$LOCAL_PILE" "$DRIVE_PILE" 2>/dev/null
TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
DATE=$(date -u +"%Y-%m-%d")

PAYLOAD="$(cat 2>/dev/null)"
if [[ -z "$PAYLOAD" ]]; then exit 0; fi

# Extract session id + transcript path with python
read -r SID TRANSCRIPT < <(printf '%s' "$PAYLOAD" | python3 -c '
import sys, json
try:
    d = json.load(sys.stdin)
except Exception:
    print("",""); sys.exit(0)
print(d.get("session_id","").strip(), d.get("transcript_path","").strip())
' 2>/dev/null)

if [[ -z "$TRANSCRIPT" || ! -f "$TRANSCRIPT" ]]; then
  exit 0
fi

# Scan last 300 lines of transcript for error markers, emit one pile entry per match (capped)
tail -n 300 "$TRANSCRIPT" 2>/dev/null | python3 - "$SID" "$TS" "$LOCAL_PILE" "$DRIVE_PILE" "$DATE" <<'PYEOF'
import sys, json, re, hashlib, os
sid, ts, local_pile, drive_pile, date = sys.argv[1:6]

patterns = [
    ("hook_error", re.compile(r"hook error|PreToolUse[^\n]{0,120}error", re.IGNORECASE)),
    ("blocked", re.compile(r"^BLOCKED:|: BLOCKED:")),
    ("traceback", re.compile(r"Traceback \(most recent call last\)")),
    ("tool_error", re.compile(r'"is_error":\s*true')),
    ("interrupted", re.compile(r'"interrupted":\s*true')),
]
seen = set()
count = 0
for line in sys.stdin:
    body = line.strip()
    if not body:
        continue
    for kind, rx in patterns:
        if rx.search(body):
            h = hashlib.sha256(("L1"+body[:500]+sid).encode()).hexdigest()[:16]
            if h in seen:
                break
            seen.add(h)
            entry = {
                "ts": ts,
                "layer": "L1_stop_capture",
                "source": f"stop_hook_session={sid}",
                "kind": kind,
                "severity": "error",
                "body": body[:1000],
                "hash": h,
                "session_id": sid,
            }
            jline = json.dumps(entry, separators=(",",":")) + "\n"
            for pd in (local_pile, drive_pile):
                try:
                    with open(os.path.join(pd, f"{date}.jsonl"), "a", encoding="utf-8") as f:
                        f.write(jline)
                except Exception:
                    pass
            count += 1
            break
    if count >= 50:
        break
PYEOF

exit 0
