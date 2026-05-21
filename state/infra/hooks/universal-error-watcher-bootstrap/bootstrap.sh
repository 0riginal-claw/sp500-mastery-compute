#!/usr/bin/env bash
# SessionStart hook: bootstrap universal_error_watcher daemon if not loaded,
# AND inject "errors since last checkin (last 30 min)" as additionalContext.

set +u
LABEL="com.zg.universal_error_watcher"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"

if ! /bin/launchctl list 2>/dev/null | grep -q "$LABEL"; then
  if [[ -f "$PLIST" ]]; then
    /bin/launchctl bootstrap "gui/$(id -u)" "$PLIST" >/dev/null 2>&1 || \
      /bin/launchctl load "$PLIST" >/dev/null 2>&1 || true
  fi
fi

PILE_LOCAL="/Users/orginal/.zg/state/error_pile"
PILE_DRIVE="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/state/error_pile"
TODAY=$(date -u +"%Y-%m-%d")

DIGEST=$(python3 - "$PILE_LOCAL" "$PILE_DRIVE" "$TODAY" 2>/dev/null <<'PYEOF'
import sys, json, os
from datetime import datetime, timezone, timedelta
pl, pd, today = sys.argv[1:4]
cutoff = datetime.now(timezone.utc) - timedelta(minutes=30)
seen = set()
rows = []
for piledir in (pl, pd):
    fp = os.path.join(piledir, f"{today}.jsonl")
    if not os.path.exists(fp): continue
    try:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception: continue
                h = d.get("hash","")
                if h in seen: continue
                seen.add(h)
                try:
                    ts = datetime.strptime(d.get("ts",""), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                except Exception: continue
                if ts < cutoff: continue
                rows.append(d)
    except OSError: continue
rows.sort(key=lambda r: r.get("ts",""), reverse=True)
rows = rows[:20]
if not rows:
    sys.exit(0)
print(f"WARNING: ERRORS SINCE LAST CHECKIN (last 30 min, top {len(rows)} of {len(seen)})")
print()
for r in rows:
    print(f"  [{r.get('ts','?')}] {r.get('layer','?')} {r.get('kind','?')} severity={r.get('severity','?')}")
    body = (r.get('body','') or '').replace('\n',' ')[:160]
    print(f"      {body}")
print()
print("Auto-fix briefs (if classified) live in:")
print("  AI-Tools/logs/auto_solve_engine/<hash>_<UTC>.md")
print()
print("Per mandates.md section 8: classified errors require 3-solver triplet dispatch.")
PYEOF
)

if [[ -z "$DIGEST" ]]; then
  exit 0
fi

python3 - "$DIGEST" <<'PYEOF'
import json, sys
ctx = sys.argv[1]
print(json.dumps({
  "hookSpecificOutput": {
    "hookEventName": "SessionStart",
    "additionalContext": ctx
  }
}))
PYEOF
