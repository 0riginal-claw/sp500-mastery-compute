#!/usr/bin/env bash
# gabriel-context-inject — SubagentStart guardrail (Hook 4 of 6)
#
# Emit additionalContext JSON injecting Gabriel's current capability map,
# recent reflexions, and top goal-tree leaves so every spawned sub-agent
# inherits the self-awareness posture.
#
# Output: single JSON object on stdout matching the
# `hookSpecificOutput.additionalContext` schema.

set -u

ROOT="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools"
STATE_DIR="$ROOT/state/gabriel_self"
LOG_DIR="$ROOT/logs/gabriel_self"
LOG_FILE="$LOG_DIR/context_inject.log"

mkdir -p "$LOG_DIR" 2>/dev/null

# Drain stdin
INPUT="$(cat 2>/dev/null || true)"

# Build brief from current state files (best-effort, never block)
BRIEF=$(STATE_DIR="$STATE_DIR" python3 - <<'PY' 2>/dev/null
import json, os, sys, time
from pathlib import Path

state_dir = Path(os.environ["STATE_DIR"])

def safe_load(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default

# Capability map summary
cap = safe_load(state_dir / "capability_map.json", {})
cap_list = cap.get("capabilities", []) if isinstance(cap, dict) else []
cap_names = []
for c in cap_list[:10]:
    if isinstance(c, dict):
        cap_names.append(c.get("name") or c.get("id") or "?")
    elif isinstance(c, str):
        cap_names.append(c)
cap_summary = ", ".join(cap_names) if cap_names else "(empty)"

# Last 3 reflexions
reflex = []
try:
    with open(state_dir / "reflexions.jsonl") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        for ln in lines[-3:]:
            try:
                row = json.loads(ln)
                lesson = row.get("lesson") or row.get("text") or row.get("summary") or ""
                if lesson:
                    reflex.append(lesson[:160])
            except json.JSONDecodeError:
                continue
except OSError:
    pass
reflex_summary = " | ".join(reflex) if reflex else "(none)"

# Top goal-tree leaves
goal = safe_load(state_dir / "goal_tree.json", {})
leaves = []
def walk(node, depth=0):
    if not isinstance(node, dict):
        return
    children = node.get("children", []) or []
    if not children:
        label = node.get("label") or node.get("id") or ""
        if label:
            leaves.append(label[:80])
    else:
        for c in children:
            walk(c, depth + 1)
root = goal.get("root") if isinstance(goal, dict) else None
if root:
    walk(root)
goal_summary = " | ".join(leaves[:3]) if leaves else "(no leaves)"

# User predictor (top preferences)
user_p = safe_load(state_dir / "user_predictor.json", {})
prefs = user_p.get("preferences", {}) if isinstance(user_p, dict) else {}
pref_names = list(prefs.keys())[:5] if isinstance(prefs, dict) else []
pref_summary = ", ".join(pref_names) if pref_names else "(none)"

brief = f"""=== GABRIEL-SELF INHERITANCE (workspace self-awareness posture) ===

You inherit the workspace's running self-model. Act in alignment with it.

Capability map (top): {cap_summary}
Recent reflexions (last 3): {reflex_summary}
Current goal-tree leaves (top 3): {goal_summary}
Tracked user preferences: {pref_summary}

Rules:
1. If your work uncovers a new capability or limitation, append a reflexion
   to state/gabriel_self/reflexions.jsonl (one JSON line: ts, lesson, evidence).
2. If a goal-tree leaf is satisfied, mark it done by appending an inbox entry
   to state/autonomous_mode/user_inbox.jsonl with kind=goal_completed.
3. State files live at state/gabriel_self/ — append-only JSONL or atomic
   atomic-replace JSON. Never partial-write.
4. Freshness is enforced by PreToolUse hook (10-min stale window). If you
   plan a >10min run, write a heartbeat into capability_map.json.

=== END GABRIEL-SELF INHERITANCE ==="""

print(brief)
PY
)

# Emit hook output
BRIEF="${BRIEF:-}" python3 - <<'PY'
import json, os, sys
brief = os.environ.get("BRIEF", "") or ""
out = {
    "hookSpecificOutput": {
        "hookEventName": "SubagentStart",
        "additionalContext": brief,
    }
}
sys.stdout.write(json.dumps(out))
PY

{
  printf '[%s] gabriel-context-inject fired (brief=%d bytes, input=%d bytes)\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${#BRIEF}" "${#INPUT}"
} >> "$LOG_FILE" 2>/dev/null

exit 0
