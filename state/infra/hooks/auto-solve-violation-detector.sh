#!/usr/bin/env bash
# auto-solve-violation-detector — Stop / SubagentStop hook
#
# Catches the orchestrator's "ask the user instead of fan-out / auto-solve" pattern at
# the moment it would surface to the user. Scans the last assistant message in the
# session transcript for sentinel phrases that signal a mandate violation
# (CLAUDE.md "Always auto-execute", §8 auto-solve-on-issue, §3 fan-out, no-restart):
#
#   "would you like"          → §3 / always-auto-execute violation (offering as choice)
#   "shall i"                 → same
#   "do you want"             → same
#   "let me know if"          → same
#   "say go"                  → same
#   "options:"                → multi-option presentation when one should be chosen
#   "you should manually"     → §8 violation (punting work to user)
#   "60-sec manual" / "60 sec manual" / "manual operator" / "operator action"
#   "you need to"             → §8 violation (telling user to act)
#   "you should generate"     → §8 violation
#   "fastest path = manual"   → §8 violation
#   "please restart"          → no-restart-mandate violation
#   "restart claude code"     → same
#
# If detected → exit 2 with JSON output that BLOCKS the stop and forces the
# orchestrator to re-spawn solver helpers (per §8 default 3-helper fan-out) instead
# of surfacing the question.
#
# Schema (Stop event):
#   { "session_id", "stop_hook_active", "transcript_path", "hook_event_name": "Stop", ... }
#
# Idempotency:
#   - if stop_hook_active==true → ALWAYS exit 0 (Claude already in stop-block loop; do
#     not block twice or we deadlock the session)
#   - errors in this hook → exit 0 (never break the session)
#
# Tunables (env):
#   AUTO_SOLVE_DETECTOR_DISABLE=1   → bypass detection (emergency escape hatch)
#   AUTO_SOLVE_DETECTOR_DRY_RUN=1   → log a match but exit 0 (no block)

set +e
LC_ALL=C

LOG_DIR="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/logs/auto_solve_detector"
mkdir -p "$LOG_DIR" 2>/dev/null
LOG_FILE="$LOG_DIR/$(date -u +%Y-%m-%d).log"
TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Emergency escape hatch
if [[ "${AUTO_SOLVE_DETECTOR_DISABLE:-0}" == "1" ]]; then
  echo "$TS DISABLED via env" >> "$LOG_FILE" 2>/dev/null
  exit 0
fi

PAYLOAD="$(cat 2>/dev/null)"
if [[ -z "$PAYLOAD" ]]; then exit 0; fi

# Parse fields (jq preferred, python fallback)
SID=""
TRANSCRIPT=""
STOP_ACTIVE=""
EVENT=""
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

# Loop-prevention: if stop_hook_active already true, never block again
if [[ "$STOP_ACTIVE" == "true" ]]; then
  echo "$TS sid=$SID stop_hook_active=true -> bypass" >> "$LOG_FILE" 2>/dev/null
  exit 0
fi

# Need a transcript to scan
if [[ -z "$TRANSCRIPT" || ! -f "$TRANSCRIPT" ]]; then
  echo "$TS sid=$SID no transcript ($TRANSCRIPT) -> pass" >> "$LOG_FILE" 2>/dev/null
  exit 0
fi

# Extract the LAST assistant text message from the JSONL transcript.
# Schema per line: {"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"..."}]}, ...}
LAST_ASSISTANT=$(python3 - "$TRANSCRIPT" <<'PYEOF' 2>/dev/null
import sys, json
path = sys.argv[1]
last = ""
try:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            # Match a few common shapes — record only assistant turn text.
            if rec.get("type") == "assistant":
                msg = rec.get("message", {})
                content = msg.get("content", [])
                if isinstance(content, list):
                    chunks = []
                    for blk in content:
                        if isinstance(blk, dict) and blk.get("type") == "text":
                            t = blk.get("text", "")
                            if t: chunks.append(t)
                    if chunks:
                        last = "\n".join(chunks)
                elif isinstance(content, str) and content:
                    last = content
    # Trim very long messages to last ~12000 chars (the closing part is where asks live)
    if len(last) > 12000:
        last = last[-12000:]
    sys.stdout.write(last)
except Exception:
    pass
PYEOF
)

if [[ -z "$LAST_ASSISTANT" ]]; then
  echo "$TS sid=$SID no assistant text -> pass" >> "$LOG_FILE" 2>/dev/null
  exit 0
fi

# Lowercase for case-insensitive match
LC=$(printf '%s' "$LAST_ASSISTANT" | tr '[:upper:]' '[:lower:]')

# Sentinel phrases. Keep these conservative — match the phrase IN CONTEXT of
# offering something to the user (not just any occurrence). We match raw substring;
# false-positive analysis covered in the install report.
declare -a SENTINELS=(
  # --- original choice/option-asking patterns ---
  "would you like"
  "do you want me to"
  "shall i "
  "let me know if you want"
  "say go to start"
  "say 'go' to"
  'say "go" to'
  # --- original manual-punt patterns ---
  "you should manually"
  "you'll need to manually"
  "you need to manually"
  "60-sec manual"
  "60 sec manual"
  "manual operator"
  "operator action required"
  "operator must"
  "operator should"
  "operator needs to"
  "operator will need"
  "you should generate"
  "fastest path = manual"
  "fastest path is manual"
  "please restart claude"
  "restart claude code"
  "please run this command"
  "please execute"
  # --- 2026-05-17 expansion: real-world false-negatives from CI/deploy workflow ---
  "manual token paste"
  "manual token-paste"
  "manual paste"
  "needs first-deploy"
  "needs first deploy"
  "requires first-deploy"
  "requires first deploy"
  "first-deploy requirement"
  "first deploy requirement"
  "needs manual"
  "need manual"
  "manual intervention"
  "manual step"
  "manual setup"
  "manually paste"
  "manually generate"
  "manually create"
  "manually configure"
  "manually visit"
  "needs a human"
  "need a human"
  "human intervention"
  "you can do this"
  "you can do that"
  "the user can"
  "user can run"
  "user can paste"
  "user can visit"
  "you'll need to"
  "you will need to"
  "you need to generate"
  "you need to visit"
  "you need to paste"
  "you need to create"
  "you need to configure"
  "you need to provide"
  "blocked on first-deploy"
  "blocked on first deploy"
  "blocked on manual"
  "blocked on operator"
  "blocked on token"
  "blocked on user"
  "blocked on the user"
  "blocked on human"
  "blocked on a human"
  # skip/circle-back/punt patterns
  "skip for now"
  "skipping for now"
  "skip this for now"
  "skip this provider"
  "circle back when"
  "come back when"
  "revisit when"
  "deferred until"
  "deferring until"
  "leave for later"
  "leave this for later"
  "leave it for later"
  "set aside"
  "park this"
  "parking this"
  # generic "blocker/blocked" + scoped clause
  "is blocked on"
  "are blocked on"
  "blocker:"
  "blockers:"
  "this is blocked"
  # token / api key / dashboard handoffs
  "paste the token"
  "paste this token"
  "paste your token"
  "paste the api key"
  "paste this api key"
  "paste your api key"
  "visit the dashboard"
  "go to the dashboard"
  "visit dash."
  "visit the browser"
  "open the browser"
  "open the dashboard"
  "in your browser"
  "from your browser"
  "from the browser"
  "via the browser"
  "via the dashboard"
  "via the ui"
  "via the web ui"
  "log in to"
  "log into"
  "sign in to"
  "sign into"
  "sign up at"
  "sign up for"
  # restart / reload escalations
  "you should restart"
  "you'll need to restart"
  "you need to restart"
  "consider restarting"
)

MATCH=""
for p in "${SENTINELS[@]}"; do
  if [[ "$LC" == *"$p"* ]]; then
    MATCH="$p"
    break
  fi
done

# Special multi-option pattern: lines starting with "options:" + numbered/lettered
# list — only fires if we did NOT already match a sentence-level sentinel
if [[ -z "$MATCH" ]]; then
  if printf '%s' "$LC" | grep -qE '^(options|choices):[[:space:]]*$' && \
     printf '%s' "$LC" | grep -qE '^[[:space:]]*[1-9a-z][\.)]'; then
    MATCH="options:-list"
  fi
fi

# Regex patterns — catch templated handoff/skip/blocker phrasings the literal
# sentinels can't enumerate. Added 2026-05-17.
if [[ -z "$MATCH" ]]; then
  # "skip <word(s)> for now"  (e.g. "Skip deno_deploy for now", "skip the X for now")
  if printf '%s' "$LC" | grep -qE 'skip[[:space:]]+[a-z0-9_./-]+([[:space:]]+[a-z0-9_./-]+)*[[:space:]]+for[[:space:]]+now'; then
    MATCH="regex:skip-for-now"
  # "blocked on <word(s)> requirement"
  elif printf '%s' "$LC" | grep -qE 'blocked[[:space:]]+on[[:space:]]+[a-z0-9_./-]+([[:space:]]+[a-z0-9_./-]+)*[[:space:]]+requirement'; then
    MATCH="regex:blocked-on-X-requirement"
  # "you/the user/operator (must|need to|should|can) <verb> ..." — handoff intent
  elif printf '%s' "$LC" | grep -qE '\b(you|the[[:space:]]+user|operator)[[:space:]]+(must|need[[:space:]]+to|should|will[[:space:]]+need[[:space:]]+to|can|have[[:space:]]+to)[[:space:]]+(paste|generate|create|configure|visit|open|run|provide|copy|enter|setup|set[[:space:]]+up|sign|log|install|deploy|click)'; then
    MATCH="regex:user-must-do-X"
  # "needs <X> to be (pasted|generated|created|configured|provided) by (you|user|operator|human)"
  elif printf '%s' "$LC" | grep -qE 'need[s]?[[:space:]]+.{1,60}(pasted|generated|created|configured|provided|entered)[[:space:]]+by[[:space:]]+(you|the[[:space:]]+user|operator|a[[:space:]]+human)'; then
    MATCH="regex:needs-X-by-user"
  # "<verb> from (you|user|operator|human|browser|dashboard)" — handoff intent
  elif printf '%s' "$LC" | grep -qE '\b(paste|input|provide|enter|generate|fetch)[[:space:]]+(it[[:space:]]+)?from[[:space:]]+(you|the[[:space:]]+user|operator|a[[:space:]]+human|the[[:space:]]+browser|the[[:space:]]+dashboard)'; then
    MATCH="regex:fetch-from-user-or-ui"
  fi
fi

if [[ -z "$MATCH" ]]; then
  # Log every clean turn at debug-level (1 line) so audit can confirm hook ran
  echo "$TS sid=$SID event=$EVENT len=${#LAST_ASSISTANT} -> CLEAN" >> "$LOG_FILE" 2>/dev/null
  exit 0
fi

# Match -> log + decide block-or-dry-run
SNIPPET=$(printf '%s' "$LAST_ASSISTANT" | tail -c 400 | tr '\n' ' ' | head -c 400)
echo "$TS sid=$SID event=$EVENT MATCH=\"$MATCH\" snippet=\"$SNIPPET\"" >> "$LOG_FILE" 2>/dev/null

if [[ "${AUTO_SOLVE_DETECTOR_DRY_RUN:-0}" == "1" ]]; then
  echo "$TS sid=$SID DRY_RUN -> would block, exiting 0" >> "$LOG_FILE" 2>/dev/null
  exit 0
fi

# Block with reason. exit 2 sends the reason back to the assistant as a stop-block.
# Per Claude Code hooks spec, stderr on exit-2 from Stop hook is fed back into the
# model context so it can react. We also emit JSON additionalContext via stdout for
# compatibility with the newer hook-output shape.
REASON="MANDATE VIOLATION — auto-solve-on-issue / always-auto-execute / §3 fan-out.

Detected sentinel phrase: \"$MATCH\"

You were about to surface a question / option / manual instruction to the user. Per
CLAUDE.md mandates this is not allowed:

  - feedback_always_auto_execute  : never propose 'I can do X if you want'
  - feedback_auto_solve_on_issue  : ANY blocker -> spawn 3 parallel solver helpers
  - feedback_dont_ask_just_do     : stop asking, never ask, just do
  - §3 (AGENT_BRIEF_TEMPLATE.md)  : if >5min / multi-slice -> fan out
  - feedback_no_restart_mandate   : never suggest restart

REQUIRED ACTION NOW:
  Spawn 3 parallel helpers in a single message via native Agent
  (subagent_type: general-purpose) OR mcp__plugin_fallback-agent_fallback__Task:
    1. INTERNET solver  — WebSearch + WebFetch for the blocker
    2. GITHUB solver    — gh search code / issues / PRs for fix patterns
    3. REPO-LOCAL solver — grep AI-Tools / cloned-repos for prior solutions
  Aggregate, pick lowest-risk highest-leverage fix, apply silently, log to
    AI-Tools/logs/auto_solve/<issue>_<UTC>.md

Escape hatch (use ONLY for genuinely unanswerable / safety-rule-protected items):
  Re-issue your final reply with the literal token  [auto-solve:exempt]  on its
  own line — the next Stop will bypass detection (stop_hook_active=true after
  this block sets the loop guard).

Log: $LOG_FILE"

# Emit JSON additionalContext on stdout AND reason on stderr (belt-and-braces)
printf '{"hookSpecificOutput":{"hookEventName":"Stop","decision":"block","reason":%s}}\n' \
  "$(printf '%s' "$REASON" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))')" 2>/dev/null

printf '%s\n' "$REASON" >&2

exit 2
