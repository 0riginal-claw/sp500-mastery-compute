#!/usr/bin/env bash
# universal-resume-bootstrap (SessionStart): emit per-class LOST_<class>.md
# context cards if prior universal-resume state exists. Diff claimed-vs-actual
# per class; surface manifests + diff totals as additionalContext.
# Non-blocking; never fails the hook chain.
# Created 2026-05-20 by §8 REPO-LOCAL universal resume mission.

set +e
LC_ALL=C

LOCAL_STATE="/Users/orginal/.zg/state/universal_resume"
DRIVE_STATE="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/state/universal_resume"

# Ensure daemon is loaded (idempotent) — auto-respawn if missing
if ! /bin/launchctl list 2>/dev/null | /usr/bin/grep -q "com.zg.universal_resume_guardrail"; then
  PLIST="/Users/orginal/Library/LaunchAgents/com.zg.universal_resume_guardrail.plist"
  if [[ -f "$PLIST" ]]; then
    /bin/launchctl bootstrap "gui/$(id -u)" "$PLIST" 2>/dev/null
  fi
fi

# Heartbeat path resolution: prefer fresh local, fall back to Drive
HB="$LOCAL_STATE/heartbeat.json"
[[ ! -s "$HB" ]] && HB="$DRIVE_STATE/heartbeat.json"
[[ ! -s "$HB" ]] && exit 0

CTX=$(/usr/bin/python3 <<'PY' 2>/dev/null
import json, os, time
from pathlib import Path

LOCAL = Path("/Users/orginal/.zg/state/universal_resume")
DRIVE = Path("/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/state/universal_resume")

def pick(*paths):
    for p in paths:
        if p.exists() and p.stat().st_size > 0:
            return p
    return None

hb_path = pick(LOCAL / "heartbeat.json", DRIVE / "heartbeat.json")
if not hb_path:
    raise SystemExit(0)
hb = json.loads(hb_path.read_text())
age_sec = int(time.time() - hb.get("ts", 0))

lines = ["## Universal-resume context (5 agent classes)"]
lines.append(f"- daemon pid: {hb.get('pid')}, session_id: {hb.get('session_id')}, heartbeat age: {age_sec}s")
lines.append(f"- cycle summary: {len(hb.get('cycle_summary', {}))-1} classes mirrored")
lines.append("")

# Per-class status + LOST cards
state_root = LOCAL if (LOCAL / "claude_main").exists() else DRIVE
classes = ["claude_main", "claude_subagents", "openclaw_main", "openclaw_subagents", "ollama"]
lost_dir = state_root / "_lost_reports"
lost_dir.mkdir(parents=True, exist_ok=True)

for cls in classes:
    cls_dir = state_root / cls
    mp = cls_dir / "manifest.json"
    dp = cls_dir / "diff.json"
    if not mp.exists():
        lines.append(f"- {cls}: NOT FOUND")
        continue
    try:
        m = json.loads(mp.read_text())
        d = json.loads(dp.read_text()) if dp.exists() else {}
    except Exception:
        lines.append(f"- {cls}: manifest unreadable")
        continue
    files = len(m.get("files", []))
    missing = len(d.get("missing_from_disk", []))
    orphans = len(d.get("orphan_on_disk", []))
    errors = m.get("errors", 0)
    marker = " LOST!" if missing > 0 or errors > 0 else ""
    lines.append(f"- {cls}: files={files} errors={errors} missing={missing} orphans={orphans}{marker}")
    # write per-class LOST_<cls>.md when there is loss
    if missing > 0 or errors > 0:
        lost_md = lost_dir / f"LOST_{cls}.md"
        body = [f"# Lost items in {cls} as of {m.get('ts')}",
                f"- session_id: {m.get('session_id')}",
                f"- errors: {errors}",
                f"- missing from disk ({missing}):"]
        for x in d.get("missing_from_disk", [])[:20]:
            body.append(f"  - {x}")
        body.append(f"- orphan on disk ({orphans}):")
        for x in d.get("orphan_on_disk", [])[:20]:
            body.append(f"  - {x}")
        lost_md.write_text("\n".join(body) + "\n")

lines.append("")

# --- Phase D: surface in-flight tool calls from prior session --------
inflight_dir = Path("/Users/orginal/.zg/state/universal_resume/_inflight")
inflight = []
if inflight_dir.exists():
    for entry in sorted(inflight_dir.glob("*.json")):
        try:
            j = json.loads(entry.read_text())
            # Skip stale (>10min — likely already cleaned, or false-positive)
            age = time.time() - j.get("started_ts", 0)
            if 0 < age < 600:
                inflight.append(j)
        except Exception:
            pass
if inflight:
    lines.append(f"## In-flight tool calls from prior session ({len(inflight)}):")
    for j in inflight[:5]:
        age = int(time.time() - j.get("started_ts", 0))
        lines.append(f"- {j.get('tool_name')} (id={j.get('tool_id','?')[:24]}, started {age}s ago, session={j.get('session_id','?')[:12]})")
    lines.append("If continuing prior work, re-issue the tool call(s) listed above; ledger entries auto-clear on completion.")
    lines.append("")

lines.append("Resume note: every helper, OC call, Ollama call writes to this checkpointer at 5s cadence. Any worker is universally-resumable.")

ctx = "\n".join(lines)
print(json.dumps({"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": ctx}}))
PY
)

[[ -n "$CTX" ]] && echo "$CTX"
exit 0
