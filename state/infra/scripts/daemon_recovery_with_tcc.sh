#!/usr/bin/env bash
# daemon_recovery.sh — with TCC auto-fix guard rail
# Run after every Drive remount to prevent crashdaemon recurrence

set -euo pipefail

PROJECT="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools"
DRIVE="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive"
LA="$HOME/Library/LaunchAgents"

echo "=== Daemon Recovery + TCC Guard $(date) ==="

# Step 1: Copy scripts out of Drive to bypass TCC
mkdir -p "$HOME/scripts"
for script in openclaw_session_watcher.py stale_memory_audit.py; do
    src="$PROJECT/scripts/$script"
    dst="$HOME/scripts/$script"
    if [ -f "$src" ]; then
        cp "$src" "$dst"
        chmod +x "$dst"
        echo "Copied $script to $dst (TCC bypass)"
    fi
done

# Step 2: Rebuild plists with local paths
python3 -c "
import plistlib, os, shutil
REAL_LA = os.path.expanduser('~/Library/LaunchAgents')
LOCAL = os.path.expanduser('~/scripts')
PROJ = '$PROJECT'
DRV = '$DRIVE'
for label, script_name in [('openclaw_session_watcher', 'openclaw_session_watcher.py'), 
                            ('stale_memory_audit', 'stale_memory_audit.py')]:
    path = os.path.join(REAL_LA, f'com.zg.{label}.plist')
    if not os.path.exists(path):
        continue
    with open(path, 'rb') as f:
        p = plistlib.load(f)
    p['ProgramArguments'] = ['/usr/bin/python3', os.path.join(LOCAL, script_name)]
    p['WorkingDirectory'] = LOCAL
    if 'WatchPaths' not in p:
        p['WatchPaths'] = [DRV]
    if 'KeepAlive' not in p:
        p['KeepAlive'] = True
    with open(path, 'wb') as f:
        plistlib.dump(f)
    print(f'Fixed {label} plist -> {LOCAL}/{script_name}')
"

# Step 3: Kickstart all IDLE daemons
for label in $(launchctl list | grep com.zg. | awk '$1 == "-" && $2 == 0 {print $3}'); do
    launchctl start "$label" 2>/dev/null && echo "Kickstarted $label" || true
done

echo "=== Recovery complete ==="
