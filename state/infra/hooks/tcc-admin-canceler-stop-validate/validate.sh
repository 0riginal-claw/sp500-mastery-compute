#!/usr/bin/env bash
# tcc-admin-canceler-stop-validate (Stop): confirm patch is still in place.
# Created 2026-05-20 by guardrail-100pct remediation.
set +e
LC_ALL=C
cat >/dev/null 2>&1
SCAN="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/home/.claude/hooks/tcc-dialog-detect/scan.applescript"
LOG="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/logs/tcc_admin_canceler_validate.log"
mkdir -p "$(dirname "$LOG")" 2>/dev/null
if [[ -f "$SCAN" ]] && /usr/bin/grep -q "administer your computer" "$SCAN" 2>/dev/null; then
  echo "[$(date +%FT%TZ)] tcc-admin-canceler-stop-validate: OK patch present" >> "$LOG"
else
  echo "[$(date +%FT%TZ)] tcc-admin-canceler-stop-validate: PATCH MISSING" >> "$LOG"
fi
exit 0
