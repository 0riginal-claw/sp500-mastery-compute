#!/usr/bin/env bash
# openclaw-completion-inject — UserPromptSubmit hook
#
# Reads ~/.claude/state/openclaw_completions.jsonl, finds rows with
# reported_to_user=false, emits them as additionalContext on the user's next
# prompt, then marks them reported_to_user=true.
#
# Output protocol: print JSON {"hookSpecificOutput":{"additionalContext":"..."}}.
# Exit 0 always (non-blocking).

set +e
LC_ALL=C

STATE_FILE="$HOME/.claude/state/openclaw_completions.jsonl"
LOG_DIR="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/logs"
LOG_FILE="$LOG_DIR/openclaw_completion_inject.log"
mkdir -p "$LOG_DIR" 2>/dev/null

# Drain stdin so caller doesn't block.
cat >/dev/null 2>&1

[[ ! -s "$STATE_FILE" ]] && exit 0

OUT=$(STATE_FILE="$STATE_FILE" LOG_FILE="$LOG_FILE" python3 <<'PY'
import json, os, sys, time

sf = os.environ["STATE_FILE"]
log_file = os.environ["LOG_FILE"]

rows = []
try:
    with open(sf) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
except FileNotFoundError:
    sys.exit(0)

unreported = [r for r in rows if not r.get("reported_to_user")]
if not unreported:
    sys.exit(0)

lines = ["OpenClaw COMPLETIONS (new since last prompt):"]
for r in unreported:
    files = r.get("files_written") or []
    files_short = ",".join(f.split("/")[-1] for f in files[:5])
    if len(files) > 5:
        files_short += f",+{len(files)-5}"
    excerpt = (r.get("completion_excerpt") or "").strip()
    if excerpt:
        excerpt = " excerpt=" + excerpt[:80].replace("\n", " ")
    err_n = len(r.get("errors") or [])
    lines.append(
        "- task={tid} pid={pid} dur={dur}s stop={stop} tool_calls={tc} "
        "files=[{files}] errors={errn}{exc}".format(
            tid=r.get("task_id", "?"),
            pid=r.get("pid", "?"),
            dur=r.get("duration_s", "?"),
            stop=r.get("stop_reason", "?"),
            tc=r.get("tool_calls"),
            files=files_short,
            errn=err_n,
            exc=excerpt,
        )
    )
ctx = "\n".join(lines)

now = int(time.time())
for r in rows:
    if not r.get("reported_to_user"):
        r["reported_to_user"] = True
        r["reported_at"] = now

tmp = sf + ".tmp"
with open(tmp, "w") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")
os.replace(tmp, sf)

try:
    with open(log_file, "a") as fh:
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))
        fh.write(f"[{ts}] injected {len(unreported)} completion(s)\n")
except Exception:
    pass

payload = {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": ctx}}
sys.stdout.write(json.dumps(payload))
PY
)

if [[ -n "$OUT" ]]; then
  printf '%s' "$OUT"
fi
exit 0
