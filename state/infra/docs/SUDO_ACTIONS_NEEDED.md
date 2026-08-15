# SUDO_ACTIONS_NEEDED — actions that require user-supplied sudo to complete

These are *optional hardening steps* for the Drive throttle. The system works without them; these would push the load reduction further by also throttling root-owned macOS indexers.

## Why these need sudo

`fileproviderd`, `mds_stores`, and `corespotlightd` run as root. `renice` of a root-owned PID by an unprivileged user returns `setpriority: Operation not permitted`. macOS also won't let an unprivileged user run `mdutil -i off` on a system volume.

## Action 1 — root-domain launchd for renice helper

Promote `drive_throttle_renice.sh` to a root-domain launchd so it can renice the system indexers. Two steps:

```bash
sudo cp /Users/orginal/.zg/bin/drive_throttle_renice.sh /usr/local/sbin/drive_throttle_renice.sh
sudo chmod 755 /usr/local/sbin/drive_throttle_renice.sh

sudo tee /Library/LaunchDaemons/com.zg.drive_throttle_renice.plist >/dev/null <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>com.zg.drive_throttle_renice</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>/usr/local/sbin/drive_throttle_renice.sh</string>
    </array>
    <key>StartInterval</key><integer>60</integer>
    <key>RunAtLoad</key><true/>
    <key>StandardOutPath</key><string>/var/log/com.zg.drive_throttle_renice.stdout.log</string>
    <key>StandardErrorPath</key><string>/var/log/com.zg.drive_throttle_renice.stderr.log</string>
</dict>
</plist>
EOF

sudo chmod 644 /Library/LaunchDaemons/com.zg.drive_throttle_renice.plist
sudo launchctl bootstrap system /Library/LaunchDaemons/com.zg.drive_throttle_renice.plist

# Then unload the user-domain version so we don't duplicate:
launchctl unload ~/Library/LaunchAgents/com.zg.drive_throttle_renice.plist
# Optionally rename: mv ~/Library/LaunchAgents/com.zg.drive_throttle_renice.plist ~/Library/LaunchAgents/com.zg.drive_throttle_renice.plist.disabled
```

Expected impact: `sys_reniced` count goes from 3 → 4 every minute in the log, and `corespotlightd` / `mds_stores` won't burn 30-60% CPU during file churn.

## Action 2 — Spotlight exclusion on Drive root

```bash
sudo mdutil -i off "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive"
sudo mdutil -E   "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive"
```

Note: Drive is FUSE so Spotlight can't actually index its contents — but disabling stops `corespotlightd` from retrying constantly when Drive paths appear in FSEvents, which silences a chunk of the load.

## Action 3 (alternative to Action 1) — limited sudoers

If you don't want a root-domain launchd, you can give the user passwordless `renice` for specific PIDs via:

```bash
sudo visudo
# Add line:
orginal ALL=(root) NOPASSWD: /usr/bin/renice
```

Then modify `drive_throttle_renice.sh` to call `sudo /usr/bin/renice +N -p PID` for root-owned PIDs. **More invasive — Action 1 is recommended.**

## How to verify after applying

```bash
# Check root-domain plist is loaded
sudo launchctl print system/com.zg.drive_throttle_renice | head -20

# Watch the log; sys_reniced should be 4
tail -f /var/log/com.zg.drive_throttle_renice.stdout.log
```

## Rollback

```bash
sudo launchctl bootout system /Library/LaunchDaemons/com.zg.drive_throttle_renice.plist
sudo rm /Library/LaunchDaemons/com.zg.drive_throttle_renice.plist
sudo rm /usr/local/sbin/drive_throttle_renice.sh
launchctl load ~/Library/LaunchAgents/com.zg.drive_throttle_renice.plist
```
