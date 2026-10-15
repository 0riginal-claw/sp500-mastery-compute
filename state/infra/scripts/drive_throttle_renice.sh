#!/bin/bash
# drive_throttle_renice.sh
#
# Permanent nice-value enforcement for Google Drive + supporting macOS
# fileproviderd/mds_stores/corespotlightd. Run every 60s by launchd
# com.zg.drive_throttle_renice so newly-spawned Drive worker processes
# inherit the throttle.
#
# Drive workers are user-owned (renice OK without sudo).
# System indexers are root-owned (renice needs sudo — logged + skipped).
#
# Logs to /tmp/com.zg.drive_throttle_renice.log (rotated by launchd).

set -u

LOG=/tmp/com.zg.drive_throttle_renice.log
TS=$(date -u "+%Y-%m-%dT%H:%M:%SZ")

DRIVE_NICE=${DRIVE_NICE:-15}
SYS_NICE=${SYS_NICE:-10}

drive_count=0
drive_renamed=0

# Google Drive user-owned PIDs
for pid in $(pgrep -f "Google Drive" 2>/dev/null); do
    drive_count=$((drive_count + 1))
    cur=$(ps -p "$pid" -o nice= 2>/dev/null | tr -d ' ')
    if [ -z "$cur" ]; then continue; fi
    if [ "$cur" -lt "$DRIVE_NICE" ] 2>/dev/null; then
        if renice +"$DRIVE_NICE" -p "$pid" >/dev/null 2>&1; then
            drive_renamed=$((drive_renamed + 1))
        fi
    fi
done

# fileproviderd / mds_stores / corespotlightd — these are usually root-owned.
# Try without sudo; if it fails (Operation not permitted), skip silently —
# the user can run this script with sudo via launchd RootJob if approved.
sys_attempted=0
sys_succeeded=0
for proc in fileproviderd mds_stores corespotlightd; do
    for pid in $(pgrep -f "$proc" 2>/dev/null); do
        sys_attempted=$((sys_attempted + 1))
        if renice +"$SYS_NICE" -p "$pid" >/dev/null 2>&1; then
            sys_succeeded=$((sys_succeeded + 1))
        fi
    done
done

echo "$TS drive_pids=$drive_count drive_reniced=$drive_renamed sys_pids=$sys_attempted sys_reniced=$sys_succeeded" >> "$LOG"
exit 0
