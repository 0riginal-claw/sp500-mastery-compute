#!/usr/bin/env bash
# SubagentStart hook: inject the error-handling rule + recent-pending-fixes
# summary into every sub-agent spawn so children/grandchildren inherit the
# error-pile awareness.

set +u
PILE_LOCAL="/Users/orginal/.zg/state/error_pile"
PILE_DRIVE="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/state/error_pile"
TODAY=$(date -u +"%Y-%m-%d")

PENDING=$(python3 - "$PILE_LOCAL" "$PILE_DRIVE" "$TODAY" 2>/dev/null <<'PYEOF'
import sys, os, json
pl, pd, today = sys.argv[1:4]
seen = set()
total = 0
errors_h = 0
for piledir in (pl, pd):
    fp = os.path.join(piledir, f"{today}.jsonl")
    if not os.path.exists(fp): continue
    try:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                try: d = json.loads(line)
                except: continue
                h = d.get("hash","")
                if h in seen: continue
                seen.add(h)
                total += 1
                if d.get("severity") == "error":
                    errors_h += 1
    except OSError: continue
fixes_l = os.path.expanduser("~/.zg/state/error_pile/fixes")
pending = 0
if os.path.isdir(fixes_l):
    for fn in os.listdir(fixes_l):
        if not fn.endswith(".json"): continue
        try:
            d = json.load(open(os.path.join(fixes_l, fn)))
            if d.get("status") == "pending_triplet": pending += 1
        except: pass
print(f"{total},{errors_h},{pending}")
PYEOF
)

IFS=',' read -r TOTAL ERRORS PENDING <<< "$PENDING"
TOTAL="${TOTAL:-0}"; ERRORS="${ERRORS:-0}"; PENDING="${PENDING:-0}"

CTX=$(cat <<EOF
=== UNIVERSAL ERROR PILE (inherit from parent) ===

Today's error_pile snapshot (UTC ${TODAY}):
  total entries (deduped): ${TOTAL}
  severity=error:           ${ERRORS}
  pending triplet dispatch: ${PENDING}

If you encounter ANY error during your task, it will be auto-captured by:
  L1 Stop hook       (universal-error-l1-stop-capture)
  L2 PostToolUse     (universal-error-l2-posttool-failure)
  L3-L7 daemon       (com.zg.universal_error_watcher, cycle 10s)

Per ~/.zg/mandates.md sections 3 and 8: known-pattern errors auto-classify and
emit a brief at AI-Tools/logs/auto_solve_engine/<hash>_<UTC>.md describing the
3-helper triplet (INTERNET + GITHUB + REPO-LOCAL) to dispatch.

Apply lowest-risk fix silently. Mark resolved by touching:
  ~/.zg/state/error_pile/fixes/<hash>.resolved

Then run:
  python3 "AI-Tools/scripts/universal_error_autofix.py" --smoke-test <hash>

=== END UNIVERSAL ERROR PILE ===
EOF
)

python3 - "$CTX" <<'PYEOF'
import sys, json
print(json.dumps({
  "hookSpecificOutput": {
    "hookEventName": "SubagentStart",
    "additionalContext": sys.argv[1]
  }
}))
PYEOF
