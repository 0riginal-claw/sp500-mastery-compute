#!/usr/bin/env bash
# mandate-violation-block — PreToolUse hook (universal mandate auto-popup)
#
# Inspects every tool call against the 11 universal mandates in ~/.zg/mandates.md.
# Block with stderr "MANDATE X VIOLATED: <fix>" + exit 2 when a violation is
# detected AND no skip marker is present.
#
# Skip markers (any of these in the prompt/command/description bypasses ALL
# checks for the current tool call):
#   # autosolve_skip:  # tokensavers_skip:  # fanout_skip:  # karpathy_skip:
#   # cloud_routing_skip:  # repo_intel_skip:  # mandate_skip:
#
# Specific checks:
#   §3 fanout      — Task/Agent spawns w/o `# decomposition_plan:` OR
#                    `# scope_estimate_min:` OR `# inline_justification:`
#   §5a cloud      — Bash containing `AUTO_CLOUD_DISPATCH=0` AND no
#                    `# cloud_routing_skip:` marker AND no `smoke` in cmd
#   §8 auto-solve  — handled by separate autosolve-require hook; we only nudge
#   §9 model-reason— Task/Agent spawns must have `# model_reason:` marker
#   safety         — `rm -rf /`, `rm -rf ~`, `rm -rf $HOME` always blocked
#                    regardless of skip markers (non-negotiable)
#
# Matcher (set in settings.json): ^(Bash|Task|Agent|Write|Edit|mcp__.*)$

set +e
LC_ALL=C

LOG_DIR="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/logs/auto_solve"
mkdir -p "$LOG_DIR" 2>/dev/null
LOG_FILE="$LOG_DIR/mandate_violation_block.log"

# Emergency bypass
if [[ "${MANDATE_BLOCK_DISABLE:-0}" == "1" ]]; then
  exit 0
fi

PAYLOAD="$(cat 2>/dev/null)"
[[ -z "$PAYLOAD" ]] && exit 0

export MANDATE_PAYLOAD="$PAYLOAD"
export MANDATE_LOG_FILE="$LOG_FILE"

DECISION="$(python3 <<'PY' 2>&1
import json, os, re, sys, time

raw = os.environ.get("MANDATE_PAYLOAD", "")
log_file = os.environ.get("MANDATE_LOG_FILE", "/dev/null")

try:
    d = json.loads(raw)
except Exception:
    sys.exit(0)

tool = (d.get("tool_name") or "").strip()
ti = d.get("tool_input") or {}

# Collect every string field worth scanning
blob_parts = []
for k in ("prompt", "description", "command", "content", "new_string", "old_string", "subagent_type"):
    v = ti.get(k)
    if isinstance(v, str):
        blob_parts.append(v)
blob = "\n".join(blob_parts)
blob_lower = blob.lower()

# --- Universal skip markers: any of these aborts ALL non-safety checks -----
SKIP_PATTERNS = [
    r"#\s*autosolve_skip\s*:",
    r"#\s*tokensavers_skip\s*:",
    r"#\s*fanout_skip\s*:",
    r"#\s*karpathy_skip\s*:",
    r"#\s*cloud_routing_skip\s*:",
    r"#\s*repo_intel_skip\s*:",
    r"#\s*mandate_skip\s*:",
    r"#\s*model_reason_skip\s*:",
]
has_skip = any(re.search(p, blob, re.IGNORECASE) for p in SKIP_PATTERNS)

violations = []  # list of (mandate, fix_msg, force_block)

# ----- §safety: non-negotiable rm-rf root --------------------------------
SAFETY_RX = [
    (r"\brm\s+(-[rRfv]*\s*)+/(\s|$)", "rm -rf / (root)"),
    (r"\brm\s+(-[rRfv]*\s*)+~(\s|$)", "rm -rf ~ (HOME)"),
    (r"\brm\s+(-[rRfv]*\s*)+\$HOME(\s|$)", "rm -rf $HOME"),
]
for rx, label in SAFETY_RX:
    if re.search(rx, blob):
        violations.append(("SAFETY", f"forbidden destructive cmd ({label}) — no skip marker can bypass this", True))

# ----- §3 fanout: Task/Agent spawn must have one decomposition marker -----
if not has_skip and tool in ("Task", "Agent") or tool.endswith("Task"):
    if tool in ("Task", "Agent") or "mcp__plugin_fallback-agent_fallback__Task" in tool:
        markers_3 = [
            r"#\s*decomposition_plan\s*:",
            r"#\s*scope_estimate_min\s*:",
            r"#\s*inline_justification\s*:",
            r"#\s*fanout_skip\s*:",
        ]
        if not any(re.search(p, blob, re.IGNORECASE) for p in markers_3):
            violations.append((
                "§3-FANOUT",
                "spawn prompt missing `# decomposition_plan:` OR "
                "`# scope_estimate_min:` OR `# inline_justification:` near top",
                False,
            ))

# ----- §5a cloud: AUTO_CLOUD_DISPATCH=0 without smoke or skip ------------
if not has_skip and tool == "Bash":
    if re.search(r"AUTO_CLOUD_DISPATCH\s*=\s*0", blob):
        if not re.search(r"smoke", blob_lower):
            violations.append((
                "§5a-CLOUD-ROUTING",
                "AUTO_CLOUD_DISPATCH=0 only allowed for smoke tests <60s; "
                "add `# cloud_routing_skip: <reason>` to bypass",
                False,
            ))

# ----- §9 model-reason: Claude spawns must declare model_reason ----------
if not has_skip and (tool in ("Task", "Agent") or "Task" in tool):
    # Only nudge if no marker present; not a hard block (model-routing-check
    # hook already does stderr warns). Make this a soft warning unless missing
    # AND opus is selected (where reasoning matters most).
    if not re.search(r"#\s*model_reason\s*:", blob, re.IGNORECASE):
        # Soft nudge — warning only
        with open(log_file, "a") as fh:
            fh.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] "
                     f"WARN §9 model_reason missing for {tool}\n")

# ----- Emit decision -----------------------------------------------------
# Hard block on any forced violation (safety) regardless of skip markers.
forced = [v for v in violations if v[2]]
if forced:
    msgs = "; ".join(f"{m}: {fix}" for m, fix, _ in forced)
    sys.stderr.write(f"BLOCKED (UNIVERSAL MANDATE): {msgs}\n")
    with open(log_file, "a") as fh:
        fh.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] "
                 f"BLOCK forced tool={tool} violations={msgs}\n")
    sys.exit(2)

# Non-safety violations block unless skip marker present
real_violations = [v for v in violations if not v[2]]
if real_violations and not has_skip:
    msgs = "; ".join(f"MANDATE {m} VIOLATED: {fix}" for m, fix, _ in real_violations)
    sys.stderr.write(f"BLOCKED: {msgs}\n")
    with open(log_file, "a") as fh:
        fh.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] "
                 f"BLOCK tool={tool} violations={msgs}\n")
    sys.exit(2)

# All clear or skip marker present
if real_violations and has_skip:
    with open(log_file, "a") as fh:
        fh.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] "
                 f"SKIP tool={tool} violations={[v[0] for v in real_violations]} "
                 f"(skip marker present)\n")

sys.exit(0)
PY
)"
RC=$?

if [[ $RC -eq 2 ]]; then
  echo "$DECISION" >&2
  exit 2
fi

exit 0
