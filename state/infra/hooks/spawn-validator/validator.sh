#!/usr/bin/env bash
# spawn-validator — §3 enforcement + plugin Task permission-mode check at spawn-time
#
# Part A (§3 decomposition mandate):
#   When a parent agent invokes Task / Agent / mcp__plugin_fallback-agent_fallback__Task,
#   the spawn prompt MUST include AT LEAST ONE of:
#     (a) # decomposition_plan: <comma or arrow-separated list of N slices>
#     (b) # scope_estimate_min: <integer>
#     (c) # inline_justification: <reason this single helper is appropriate, max ~150 chars>
#   If scope_estimate_min > 5 AND no decomposition_plan, emit STRONG warning.
#
# Part B (plugin Task permission inheritance, added 2026-05-17):
#   The mcp__plugin_fallback-agent_fallback__Task tool spawns a fresh `claude -p`
#   subprocess. Per upstream Anthropic bugs #37442 / #5465 / #58663 / #40241 / #26479 /
#   #19077, the parent's bypassPermissions mode does NOT propagate to that subprocess
#   unless `--permission-mode` is passed explicitly. The plugin has been patched to
#   default permissionMode to "bypassPermissions" (see plugin_task_permission_fix
#   report), but for defense-in-depth we ALSO warn any spawn that omits both
#   `permissionMode` and `allowWrite`, and warn any spawn that omits `addDirs`
#   (children that need workspace access outside cwd will be blocked).
#
# Output stderr is surfaced to orchestrator's context; never exit 2 (don't break
# in-flight orchestrations). The §3 mandate is still upheld via the in-helper
# recursion-fanout-tracker hook.
#
# Idempotent + safe: missing-jq, malformed-JSON, empty stdin all → exit 0.

set +e

PAYLOAD="$(cat 2>/dev/null)"
if [[ -z "$PAYLOAD" ]]; then exit 0; fi

TOOL=""
PROMPT=""
PERM_MODE=""
ALLOW_WRITE=""
ADD_DIRS=""
if command -v jq >/dev/null 2>&1; then
  TOOL=$(printf '%s' "$PAYLOAD" | jq -r '.tool_name // empty' 2>/dev/null)
  PROMPT=$(printf '%s' "$PAYLOAD" | jq -r '.tool_input.prompt // empty' 2>/dev/null)
  PERM_MODE=$(printf '%s' "$PAYLOAD" | jq -r '.tool_input.permissionMode // empty' 2>/dev/null)
  ALLOW_WRITE=$(printf '%s' "$PAYLOAD" | jq -r '.tool_input.allowWrite // empty' 2>/dev/null)
  ADD_DIRS=$(printf '%s' "$PAYLOAD" | jq -r '.tool_input.addDirs // empty | if type=="array" then length else 0 end' 2>/dev/null)
else
  TOOL=$(printf '%s' "$PAYLOAD" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("tool_name",""))' 2>/dev/null)
  PROMPT=$(printf '%s' "$PAYLOAD" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("tool_input",{}).get("prompt",""))' 2>/dev/null)
  PERM_MODE=$(printf '%s' "$PAYLOAD" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("tool_input",{}).get("permissionMode") or "")' 2>/dev/null)
  ALLOW_WRITE=$(printf '%s' "$PAYLOAD" | python3 -c 'import sys,json; v=json.load(sys.stdin).get("tool_input",{}).get("allowWrite"); print("" if v is None else str(v).lower())' 2>/dev/null)
  ADD_DIRS=$(printf '%s' "$PAYLOAD" | python3 -c 'import sys,json; v=json.load(sys.stdin).get("tool_input",{}).get("addDirs"); print(len(v) if isinstance(v,list) else 0)' 2>/dev/null)
fi

case "$TOOL" in
  Task|Agent|mcp__plugin_fallback-agent_fallback__Task) ;;
  *) exit 0 ;;
esac

if [[ -z "$PROMPT" ]]; then exit 0; fi

HAS_DECOMP=0
HAS_SCOPE=0
HAS_INLINE=0
SCOPE_VAL=0

if echo "$PROMPT" | grep -qiE '^[[:space:]]*#[[:space:]]*decomposition_plan[[:space:]]*:'; then
  HAS_DECOMP=1
fi
if echo "$PROMPT" | grep -qiE '^[[:space:]]*#[[:space:]]*scope_estimate_min[[:space:]]*:'; then
  HAS_SCOPE=1
  SCOPE_VAL=$(echo "$PROMPT" | grep -iE '^[[:space:]]*#[[:space:]]*scope_estimate_min[[:space:]]*:' | head -1 | sed -E 's/.*:[[:space:]]*([0-9]+).*/\1/' )
  if ! [[ "$SCOPE_VAL" =~ ^[0-9]+$ ]]; then SCOPE_VAL=0; fi
fi
if echo "$PROMPT" | grep -qiE '^[[:space:]]*#[[:space:]]*inline_justification[[:space:]]*:'; then
  HAS_INLINE=1
fi

# log every spawn to a dispositions log (helps audit drift)
LOGDIR="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/logs"
mkdir -p "$LOGDIR" 2>/dev/null
LOGFILE="$LOGDIR/spawn_validator.log"
TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "$TS tool=$TOOL decomp=$HAS_DECOMP scope=$HAS_SCOPE($SCOPE_VAL) inline=$HAS_INLINE perm_mode=${PERM_MODE:-(unset)} allow_write=${ALLOW_WRITE:-(unset)} add_dirs=${ADD_DIRS:-0} prompt_len=${#PROMPT}" >> "$LOGFILE"

# Decision tree — warnings only (non-blocking)
if (( HAS_DECOMP == 0 && HAS_SCOPE == 0 && HAS_INLINE == 0 )); then
  cat >&2 <<EOF
PROTOCOL WARNING — spawn prompt missing §3 decomposition metadata.
Required: one of these lines near the top of the helper prompt:
  # decomposition_plan: slice1, slice2, slice3      (preferred for any >5-min scope)
  # scope_estimate_min: 7                            (helper will fan if estimate >5)
  # inline_justification: <why this is single-helper>
Tool: $TOOL
This is a WARN-ONLY check (the spawn proceeds). Future violations: add the line.
EOF
elif (( HAS_DECOMP == 0 && HAS_INLINE == 0 && SCOPE_VAL > 5 )); then
  cat >&2 <<EOF
PROTOCOL WARNING — scope_estimate_min=${SCOPE_VAL} > 5 but no decomposition_plan.
Per §3, scopes >5 min MUST be pre-decomposed by the orchestrator into N parallel
helpers. Add: # decomposition_plan: <slice1, slice2, slice3> and split this spawn.
Spawn proceeds, but tracker will warn helper at 5-min boundary regardless.
EOF
fi

# Part B — plugin Task permission inheritance check (added 2026-05-17)
# Only fires for the plugin Task tool; native Task/Agent inherit parent mode correctly.
if [[ "$TOOL" == "mcp__plugin_fallback-agent_fallback__Task" ]]; then
  # Treat allowWrite=true OR an explicit permissionMode value as "permission handled".
  PERM_OK=0
  if [[ -n "$PERM_MODE" ]]; then PERM_OK=1; fi
  if [[ "$ALLOW_WRITE" == "true" || "$ALLOW_WRITE" == "True" ]]; then PERM_OK=1; fi

  if (( PERM_OK == 0 )); then
    cat >&2 <<EOF
PLUGIN TASK WARNING — mcp__plugin_fallback-agent_fallback__Task spawn missing permissionMode + allowWrite.
The plugin spawns a fresh \`claude -p\` subprocess. Parent bypassPermissions does NOT
propagate (upstream Anthropic bugs #37442 / #5465 / #58663 / #40241 / #26479 / #19077).
The plugin has been PATCHED 2026-05-17 to default permissionMode="bypassPermissions"
when caller omits it, so this spawn should still succeed — but you should pass it
EXPLICITLY for clarity and for compatibility with un-patched plugin versions:
  permissionMode: "bypassPermissions"
  addDirs: ["/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools"]
Spawn proceeds. Log: $LOGFILE
EOF
  fi

  if [[ -z "$ADD_DIRS" || "$ADD_DIRS" == "0" ]]; then
    cat >&2 <<EOF
PLUGIN TASK WARNING — mcp__plugin_fallback-agent_fallback__Task spawn missing addDirs.
The child runs with cwd as its only allowed directory unless addDirs is passed OR
FALLBACK_AGENT_DEFAULT_ADD_DIRS env is set on the plugin server. If the child needs
to read/write outside cwd it will be blocked. Recommended addDirs:
  ["/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools"]
Spawn proceeds. Log: $LOGFILE
EOF
  fi
fi

exit 0
