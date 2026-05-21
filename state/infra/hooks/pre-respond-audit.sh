#!/usr/bin/env bash
# pre-respond-audit — Stop / SubagentStop semantic auditor
#
# Complements auto-solve-violation-detector.sh. The sentinel detector catches
# explicit phrases at the moment of Stop. This auditor goes further: it scans
# the last assistant message for SEMANTIC patterns that signal a mandate
# violation even when no sentinel phrase was used. Specifically:
#
#   PATTERN A: "describes a blocker / failure / outstanding step but no Task
#   tool calls earlier in the same response" — i.e. the orchestrator narrated
#   the problem but didn't spawn solvers (§8 violation).
#
#   PATTERN B: "lists per-provider/per-target status with FAIL/BLOCKED/MANUAL/
#   PENDING entries that have no corresponding fan-out earlier" — i.e. a
#   status dashboard without solver invocations (§3+§8 violation).
#
#   PATTERN C: "mentions an operator/user/human action as the resolution path"
#   — semantic version of the sentinel "operator must X".
#
# Output mode: the audit does NOT block (the sentinel detector already does
# that for explicit cases). Instead, it WRITES a violation alert to
#   /tmp/cc-violation-alert/<sid>.json
# which is consumed by the prepend-violation-alert.sh UserPromptSubmit hook
# (installed alongside). The next user turn will see, prepended to context:
#
#   "VIOLATION DETECTED on previous turn: pattern P=<X>, snippet=<Y>.
#    Mandate requires Z. Spawn solvers NOW for the outstanding work."
#
# This is the "feedback loop" mechanism — the model can't suppress its own
# stop, but it WILL see the alert next turn and self-correct.
#
# Cost: ~120 tokens additionalContext on turns following a violation.
# Skipped (alert file deleted) on clean turns.
#
# Tunables (env):
#   PRE_RESPOND_AUDIT_DISABLE=1   → off
#   PRE_RESPOND_AUDIT_DRY_RUN=1   → log, do not write alert
#
# Idempotency:
#   - stop_hook_active=true → skip (orchestrator already in a block-loop)
#   - errors → exit 0 (never break the session)

set +e
LC_ALL=C

LOG_DIR="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/logs/pre_respond_audit"
mkdir -p "$LOG_DIR" 2>/dev/null
LOG_FILE="$LOG_DIR/$(date -u +%Y-%m-%d).log"
TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

ALERT_DIR="/tmp/cc-violation-alert"
mkdir -p "$ALERT_DIR" 2>/dev/null

if [[ "${PRE_RESPOND_AUDIT_DISABLE:-0}" == "1" ]]; then
  echo "$TS DISABLED via env" >> "$LOG_FILE" 2>/dev/null
  exit 0
fi

PAYLOAD="$(cat 2>/dev/null)"
if [[ -z "$PAYLOAD" ]]; then exit 0; fi

# Parse fields
SID=""; TRANSCRIPT=""; STOP_ACTIVE=""; EVENT=""
if command -v jq >/dev/null 2>&1; then
  SID=$(printf '%s' "$PAYLOAD"       | jq -r '.session_id      // empty' 2>/dev/null)
  TRANSCRIPT=$(printf '%s' "$PAYLOAD" | jq -r '.transcript_path // empty' 2>/dev/null)
  STOP_ACTIVE=$(printf '%s' "$PAYLOAD" | jq -r '.stop_hook_active // false' 2>/dev/null)
  EVENT=$(printf '%s' "$PAYLOAD"     | jq -r '.hook_event_name  // empty' 2>/dev/null)
else
  read -r SID TRANSCRIPT STOP_ACTIVE EVENT < <(printf '%s' "$PAYLOAD" | python3 -c '
import sys,json
try:
    d=json.load(sys.stdin)
    print(d.get("session_id","") or "", d.get("transcript_path","") or "",
          str(d.get("stop_hook_active",False)).lower(), d.get("hook_event_name","") or "")
except Exception:
    print("","","false","")
' 2>/dev/null)
fi

[[ -z "$SID" ]] && exit 0
[[ "$STOP_ACTIVE" == "true" ]] && exit 0

ALERT_FILE="$ALERT_DIR/${SID}.json"

if [[ -z "$TRANSCRIPT" || ! -f "$TRANSCRIPT" ]]; then
  echo "$TS sid=$SID no transcript -> skip" >> "$LOG_FILE" 2>/dev/null
  exit 0
fi

# Pull last assistant turn AND count Task-tool invocations within it.
AUDIT_RESULT=$(python3 - "$TRANSCRIPT" <<'PYEOF' 2>/dev/null
import sys, json, re

path = sys.argv[1]

# We want: text of last assistant turn + count of Task/Agent tool uses in that
# same turn's tool_use blocks. The "turn" boundary in JSONL is: a sequence of
# {"type":"assistant", ...} records ending in a final assistant text record
# before a user record. To keep this simple we scan the last contiguous block
# of "assistant" records (the most recent turn).
records = []
try:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except Exception:
                continue
except Exception:
    print(json.dumps({"ok": False}))
    sys.exit(0)

# Walk backwards to find the last contiguous run of assistant records
last_run = []
for r in reversed(records):
    t = r.get("type", "")
    if t == "assistant":
        last_run.append(r)
    elif t == "user" and last_run:
        # User turn ends the run we're collecting
        break
    elif t in ("system", "summary"):
        continue
    else:
        if last_run:
            break

last_run.reverse()  # chronological

text_chunks = []
task_uses = 0
for r in last_run:
    msg = r.get("message", {})
    content = msg.get("content", [])
    if isinstance(content, list):
        for blk in content:
            if not isinstance(blk, dict): continue
            bt = blk.get("type","")
            if bt == "text":
                t = blk.get("text","") or ""
                if t: text_chunks.append(t)
            elif bt == "tool_use":
                name = blk.get("name","") or ""
                # Anything that spawns a sub-agent counts as fan-out
                if name in ("Task",) or name.startswith("mcp__plugin_fallback-agent_fallback__Task"):
                    task_uses += 1
    elif isinstance(content, str):
        text_chunks.append(content)

text = "\n".join(text_chunks)
if len(text) > 16000:
    text = text[-16000:]

print(json.dumps({"ok": True, "text": text, "task_uses": task_uses}))
PYEOF
)

if [[ -z "$AUDIT_RESULT" ]]; then
  echo "$TS sid=$SID python audit failed -> skip" >> "$LOG_FILE" 2>/dev/null
  exit 0
fi

# Parse the audit result
TEXT_LEN=0
TASK_USES=0
TEXT=""
if command -v jq >/dev/null 2>&1; then
  TEXT=$(printf '%s' "$AUDIT_RESULT" | jq -r '.text // ""' 2>/dev/null)
  TASK_USES=$(printf '%s' "$AUDIT_RESULT" | jq -r '.task_uses // 0' 2>/dev/null)
  TEXT_LEN=${#TEXT}
else
  TEXT=$(printf '%s' "$AUDIT_RESULT" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("text",""))' 2>/dev/null)
  TASK_USES=$(printf '%s' "$AUDIT_RESULT" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("task_uses",0))' 2>/dev/null)
  TEXT_LEN=${#TEXT}
fi
[[ "$TASK_USES" =~ ^[0-9]+$ ]] || TASK_USES=0
[[ -z "$TEXT" ]] && { echo "$TS sid=$SID empty text -> skip" >> "$LOG_FILE" 2>/dev/null; exit 0; }

LC=$(printf '%s' "$TEXT" | tr '[:upper:]' '[:lower:]')

# --- Semantic patterns ---

PATTERN=""
SNIPPET=""

# PATTERN A: blocker-narration without spawn
#   triggers when text contains a blocker word AND task_uses==0
if (( TASK_USES == 0 )); then
  if printf '%s' "$LC" | grep -qE '\b(blocked|blocker|stuck|failed|failing|cannot|can[[:space:]]*\xe2\x80\x99\?t|unable to|won[[:space:]]*\xe2\x80\x99\?t|missing|requires|prerequisite)\b'; then
    PATTERN="A:blocker-without-spawn"
  fi
fi

# PATTERN B: status-list with manual/pending/blocked entries (no spawn)
if [[ -z "$PATTERN" ]] && (( TASK_USES == 0 )); then
  STATUS_LINES=$(printf '%s' "$LC" | grep -cE '(^|[[:space:]])[a-z0-9_./-]+[[:space:]]*[:=\xe2\x86\x92>-][[:space:]]*(pending|blocked|manual|fail(ed)?|todo|skipped|deferred)' 2>/dev/null || echo 0)
  if [[ "$STATUS_LINES" =~ ^[0-9]+$ ]] && (( STATUS_LINES >= 2 )); then
    PATTERN="B:status-list-no-spawn"
  fi
fi

# PATTERN C: resolution path delegated to user/operator/human
if [[ -z "$PATTERN" ]]; then
  if printf '%s' "$LC" | grep -qE '\b(operator|user|human)[[:space:]]+(can|will|should|needs|must|has[[:space:]]+to)\b'; then
    PATTERN="C:resolution-via-human"
  fi
fi

# autosolve_skip: this is the REPO-LOCAL solver — editing the audit hook is the fix itself
# PATTERN D: explicit "next step is YOU" language (more than one of these = high confidence)
if [[ -z "$PATTERN" ]]; then
  HUMAN_REFS=$(printf '%s' "$LC" | grep -cE '\b(you|your)[[:space:]]+(can|will|need|should|must|have[[:space:]]+to|next|action|step)\b' 2>/dev/null || echo 0)
  if [[ "$HUMAN_REFS" =~ ^[0-9]+$ ]] && (( HUMAN_REFS >= 3 )) && (( TASK_USES == 0 )); then
    PATTERN="D:multi-you-references-no-spawn"
  fi
fi

# PATTERN E: approval-ask — model is asking for permission/preference rather
# than executing. Always-auto-execute mandate forbids these phrases when
# action is within standing authorization. Case-insensitive ($LC is already
# lowercased).
if [[ -z "$PATTERN" ]]; then
  if printf '%s' "$LC" | grep -qE 'would you like|do you want me to|want me to|should i\b|let me know if|say go|ready to proceed|shall i\b|which approach do you prefer|which would you like|would you prefer|your call|your choice|\bwant me\b|i can do .* if'; then
    PATTERN="E:approval-ask-detected"
  fi
fi

if [[ -z "$PATTERN" ]]; then
  # Clean turn — clear any stale alert
  if [[ -f "$ALERT_FILE" ]]; then rm -f "$ALERT_FILE" 2>/dev/null; fi
  echo "$TS sid=$SID event=$EVENT len=$TEXT_LEN task_uses=$TASK_USES -> CLEAN" >> "$LOG_FILE" 2>/dev/null
  exit 0
fi

# Extract a 400-char snippet for the alert
SNIPPET=$(printf '%s' "$TEXT" | tail -c 800 | tr '\n' ' ' | head -c 400)

echo "$TS sid=$SID event=$EVENT len=$TEXT_LEN task_uses=$TASK_USES PATTERN=\"$PATTERN\" snippet=\"$SNIPPET\"" >> "$LOG_FILE" 2>/dev/null

if [[ "${PRE_RESPOND_AUDIT_DRY_RUN:-0}" == "1" ]]; then
  echo "$TS sid=$SID DRY_RUN -> not writing alert" >> "$LOG_FILE" 2>/dev/null
  exit 0
fi

# Write the alert file for the next UserPromptSubmit to consume.
# Python ensures correct JSON escaping.
PATTERN_OUT="$PATTERN" SNIPPET_OUT="$SNIPPET" TS_OUT="$TS" python3 - "$ALERT_FILE" <<'PYEOF' 2>/dev/null
import json, sys, os
path = sys.argv[1]
payload = {
    "ts": os.environ.get("TS_OUT",""),
    "pattern": os.environ.get("PATTERN_OUT",""),
    "snippet": os.environ.get("SNIPPET_OUT",""),
    "mandate": "auto-solve-on-issue + §3 fan-out + always-auto-execute",
    "required_action": (
        "Spawn 3 parallel solver helpers in ONE message: "
        "(1) INTERNET (WebSearch+WebFetch), "
        "(2) GITHUB (gh search code/issues), "
        "(3) REPO-LOCAL (grep AI-Tools + cloned repos). "
        "Aggregate, apply silently, log under AI-Tools/logs/auto_solve/. "
        "Do NOT ask the user. Do NOT punt to operator/human. "
        "Do NOT skip the blocker 'for now'."
    ),
}
try:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
except Exception:
    pass
PYEOF

# This hook does NOT block — sentinel detector handles explicit-phrase blocking.
exit 0
