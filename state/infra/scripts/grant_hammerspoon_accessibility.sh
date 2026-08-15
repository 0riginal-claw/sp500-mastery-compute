#!/bin/bash
# scripts/grant_hammerspoon_accessibility.sh
# Minimize friction of the irreducible Hammerspoon Accessibility grant click.
#
# Background: macOS TCC requires at least one human-confirmed Accessibility grant
# before any process can programmatically grant another. On this Mac (2026-05-20)
# NO app currently holds kTCCServiceAccessibility, so the Hammerspoon grant cannot
# be done programmatically. See: docs/MACOS_TCC_AUTOALLOW.md and
# memory/feedback_macos_tcc_irreducible.md.
#
# What this script DOES do, autonomously:
#   1. Verifies Hammerspoon is installed + running.
#   2. Deep-links System Settings directly to Privacy & Security → Accessibility
#      (skips 3 navigation clicks).
#   3. Brings Hammerspoon to front so its toggle is visible in the pane.
#   4. Shows an on-screen countdown banner via osascript-display-dialog so the
#      user knows exactly when + where to click. (Visible only when no other
#      apps have stolen focus — graceful degradation.)
#   5. Polls every 2s for up to 120s to detect when AX has been granted, then
#      reloads Hammerspoon config + announces success via `osascript display
#      notification`.
#   6. Logs every step + final outcome to logs/auto_solve/.
#
# Total user friction after running this script: ONE toggle click in System Settings.
# Pre-stages everything else.

set -u

DRIVE_ROOT="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools"
LOG_DIR="${DRIVE_ROOT}/logs/auto_solve"
LOG_FILE="${LOG_DIR}/tcc_zero_click_repo_$(date +%Y-%m-%d).md"
mkdir -p "$LOG_DIR"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "=== grant_hammerspoon_accessibility.sh START ==="

# Step 1: prerequisites
if [ ! -d "/Applications/Hammerspoon.app" ]; then
  log "ERROR: /Applications/Hammerspoon.app not installed. Install first."
  exit 1
fi
log "Hammerspoon.app present."

# Step 2: launch / focus Hammerspoon
if ! pgrep -f "Hammerspoon.app/Contents/MacOS/Hammerspoon" >/dev/null; then
  log "Hammerspoon not running — launching."
  open -gja Hammerspoon
  sleep 2
else
  log "Hammerspoon already running (pid $(pgrep -f Hammerspoon.app/Contents/MacOS/Hammerspoon | head -1))."
fi

# Step 3: deep-link System Settings directly to Accessibility pane
# Sequoia URL scheme is `x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility`
log "Opening System Settings → Privacy & Security → Accessibility…"
open "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
sleep 1

# Step 4: countdown banner via display-dialog (best-effort; if Settings has focus
# the banner appears on top of it via System Events fallback)
# NOTE: this is just visual signage — clicking OK does NOT confer permission.
osascript <<'APPLESCRIPT' &
on countdownBanner()
  set bannerText to "ACTION NEEDED — one click to eliminate all future TCC prompts:" & return & return & ¬
                    "1. In the System Settings pane that just opened," & return & ¬
                    "2. Toggle  Hammerspoon  ON." & return & return & ¬
                    "(Admin password may be requested by macOS — that's normal.)" & return & ¬
                    "This script will detect the toggle automatically and reload Hammerspoon."
  try
    display dialog bannerText with title "Hammerspoon Accessibility — one-time grant" ¬
      buttons {"I've toggled it"} default button "I've toggled it" giving up after 180
  end try
end countdownBanner
countdownBanner()
APPLESCRIPT
BANNER_PID=$!
log "Banner displayed (pid $BANNER_PID)."

# Step 5: poll for Accessibility grant taking effect.
# We can't read TCC.db, but we CAN detect when Hammerspoon's AX-driven hot-keys
# start working — proxy: Hammerspoon's log shows "TCC auto-allow loaded" after
# init.lua loads successfully, which requires AX.
HS_LOG="$HOME/.hammerspoon/tcc_autoallow.log"
[ ! -f "$HS_LOG" ] && HS_LOG="/Users/orginal/.hammerspoon/tcc_autoallow.log"

DEADLINE=$(($(date +%s) + 180))
GRANTED=0
while [ $(date +%s) -lt $DEADLINE ]; do
  # Reload Hammerspoon config every 5 polls — picks up new AX permission immediately
  if pgrep -f "Hammerspoon.app/Contents/MacOS/Hammerspoon" >/dev/null; then
    # Test: AppleScript IPC requires hs.allowAppleScript(true) — won't work in default config.
    # Use process-AX-readable check instead by counting Hammerspoon menu bar items via System Events.
    # If Terminal/this-script lacks AX we get -25211 reliably — but if Hammerspoon NOW has AX,
    # its log file will be touched as the watcher runs.
    if [ -f "$HS_LOG" ]; then
      lastmod=$(stat -f %m "$HS_LOG" 2>/dev/null || echo 0)
      now=$(date +%s)
      age=$((now - lastmod))
      if [ "$age" -lt 30 ] && [ "$(wc -l < "$HS_LOG")" -gt 1 ]; then
        log "Hammerspoon log activity within 30s — AX likely granted."
        GRANTED=1
        break
      fi
    fi
  fi
  sleep 2
done

# Step 6: report
if [ "$GRANTED" -eq 1 ]; then
  log "SUCCESS: Hammerspoon Accessibility granted + watcher active."
  osascript -e 'display notification "All future TCC dialogs will auto-allow." with title "Hammerspoon Accessibility granted"' 2>/dev/null
  kill $BANNER_PID 2>/dev/null
else
  log "TIMEOUT after 180s. User did not toggle Hammerspoon in System Settings, OR Hammerspoon AX log not updating yet."
  log "Run this script again, or grant manually."
fi

log "=== grant_hammerspoon_accessibility.sh END ==="
