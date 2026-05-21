#!/usr/bin/env bash
# universal-resume-stop-validate (Stop): verify daemon produced fresh
# heartbeat + at least one class manifest during this turn. Log to
# universal_resume_validate.log. Non-blocking.
# Created 2026-05-20 by guardrail-100pct remediation.

set +e
LC_ALL=C
cat >/dev/null 2>&1

LOCAL_DIR="/Users/orginal/.zg/state/universal_resume"
DRIVE_DIR="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/state/universal_resume"
LOG_DIR="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/logs"
mkdir -p "$LOG_DIR" 2>/dev/null
LOG="$LOG_DIR/universal_resume_validate.log"

HB="$LOCAL_DIR/heartbeat.json"
[[ ! -s "$HB" ]] && HB="$DRIVE_DIR/heartbeat.json"

if [[ ! -s "$HB" ]]; then
  echo "[$(date +%FT%TZ)] universal-resume-stop-validate: NO HEARTBEAT" >> "$LOG"
  exit 0
fi

AGE=$(/usr/bin/python3 -c "
import json, time
try:
    hb = json.load(open('$HB'))
    print(int(time.time() - hb.get('ts', 0)))
except Exception:
    print(-1)
" 2>/dev/null)

# Count class manifests present
ROOT="$LOCAL_DIR"
[[ ! -d "$ROOT/claude_main" ]] && ROOT="$DRIVE_DIR"
CLASSES_PRESENT=0
for cls in claude_main claude_subagents openclaw_main openclaw_subagents ollama; do
  [[ -s "$ROOT/$cls/manifest.json" ]] && CLASSES_PRESENT=$((CLASSES_PRESENT + 1))
done

if [[ "$AGE" =~ ^-?[0-9]+$ ]] && [[ "$AGE" -gt 300 ]]; then
  echo "[$(date +%FT%TZ)] universal-resume-stop-validate: STALE heartbeat age=${AGE}s classes=${CLASSES_PRESENT}/5" >> "$LOG"
elif [[ "$AGE" =~ ^-?[0-9]+$ ]] && [[ "$AGE" -ge 0 ]]; then
  echo "[$(date +%FT%TZ)] universal-resume-stop-validate: OK age=${AGE}s classes=${CLASSES_PRESENT}/5" >> "$LOG"
fi

exit 0
