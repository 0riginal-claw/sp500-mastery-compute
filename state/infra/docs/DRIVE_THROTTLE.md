# DRIVE_THROTTLE — permanent Mac load throttle for helper bursts

**Created**: 2026-05-20
**Trigger**: recurring 1-min load spikes (37, 46, 55, 117+) observed when 2-10 concurrent helpers / daemons write small files to `/My Drive/AI-Tools/`. Real culprits are macOS Google-Drive-sync / `fileproviderd` / `mds_stores` / `corespotlightd` reacting to the file churn — not the helpers themselves.
**Goal**: keep 1-min load <12 sustained even under 8 concurrent helpers + Ollama + daemons.

## Architecture

```
helper / daemon          DRIVE_STAGING=/tmp/ai-tools-staging
        |                  (env var picked up by patched scripts)
        v
  local APFS write -- no Drive sync, no Spotlight indexing
        |
        +-- every 5 min --> launchd com.zg.drive_sync_batch
                                       |
                                       v
                              /usr/bin/rsync -a --update
                              (NEVER --delete)
                                       |
                                       v
                            /My Drive/AI-Tools/<subtree>/
```

A second launchd job (`com.zg.drive_throttle_renice`) renices Google Drive worker processes to nice +15 every 60s so they can never starve the rest of the system.

## Components

| Path | Role |
|---|---|
| `/tmp/ai-tools-staging/` | Local staging root (APFS, not synced) |
| `AI-Tools/scripts/drive_sync_batch.py` | Source of truth: rsync staging → Drive |
| `/Users/orginal/.zg/bin/drive_sync_batch.py` | Copy on local SSD (launchd exec source — Drive paths can't be exec'd from launchd, see Section "macOS launchd + Drive landmines") |
| `/Users/orginal/.zg/bin/drive_throttle_renice.sh` | Renice Drive workers + best-effort system indexers |
| `AI-Tools/scripts/drive_throttle_renice.sh` | Source of truth for the renice script |
| `~/Library/LaunchAgents/com.zg.drive_sync_batch.plist` | StartInterval=300 (5 min) |
| `~/Library/LaunchAgents/com.zg.drive_throttle_renice.plist` | StartInterval=60 (1 min) |
| `AI-Tools/logs/drive_sync_batch_<UTC-DATE>.log` | Sync results (under load-block falls back to `/tmp/ai-tools-staging/logs/`) |
| `/tmp/com.zg.drive_throttle_renice.log` | Renice attempts + counts |

## Environment variables

| Var | Default | Purpose |
|---|---|---|
| `DRIVE_STAGING` | unset | Set in helper/daemon env to redirect writes here. Currently honored by `autonomous_mode_daemon.py` (other daemons can be patched the same way — pattern in section "Adding more daemons"). |
| `DRIVE_SYNC_LOAD_CEIL` | `20.0` | If 1-min load > this, sync defers (uses `--force` to override). |
| `DRIVE_SYNC_BATCH_DISABLE` | unset | Set to `1` to emergency-stop the syncer (it logs and exits). |
| `DRIVE_NICE` | `15` | Target nice value for Google Drive worker PIDs. |
| `SYS_NICE` | `10` | Target nice for `fileproviderd` / `mds_stores` / `corespotlightd` (renice may fail without sudo — that's OK, best-effort). |

## How a daemon opts in

```python
import os
from pathlib import Path

ROOT = Path("/Users/orginal/Library/CloudStorage/GoogleDrive-.../My Drive/AI-Tools")

_STAGING_ENV = os.environ.get("DRIVE_STAGING", "").strip()
DRIVE_STAGING: Path | None = Path(_STAGING_ENV) if _STAGING_ENV else None

def _write_root() -> Path:
    return DRIVE_STAGING if DRIVE_STAGING is not None else ROOT

_WROOT = _write_root()
LOG_DIR = _WROOT / "logs"
STATE_DIR = _WROOT / "state" / "my_daemon"
# Read-only inputs (dashboard MD files, config sources of truth) still point at ROOT.
```

Then set `DRIVE_STAGING=/tmp/ai-tools-staging` in the daemon's launchd plist `EnvironmentVariables`.

## macOS launchd + Drive landmines (from `feedback_auto_signup_architecture.md`)

1. **launchd cannot exec a script that lives under `/Users/.../My Drive/...`** — Drive is FUSE; launchd gets `Operation not permitted`. Mitigation: copy executable to `/Users/orginal/.zg/bin/` and point `ProgramArguments` at the local copy. We treat the Drive copy as the source of truth and use a manual copy-on-update workflow (or extend `apply_settings_change.sh`).

2. **launchd cannot WRITE to `/Users/.../My Drive/...` via arbitrary interpreters** — TCC permission is per-binary. `/usr/bin/python3` does NOT have Drive access via launchd; `/Users/orginal/.venvs/sp500-mastery/bin/python` DOES (granted interactively previously). Mitigation: use the venv python in plists that need Drive write access.

3. **Log redirects via plist `StandardOutPath` to a Drive path also fail** — for the same reason. Point `StandardOutPath` at `/tmp/com.zg.<label>.stdout.log`.

4. **Renice of root-owned PIDs (fileproviderd, mds_stores, corespotlightd) requires sudo** — the throttle script attempts without sudo and logs the result. Some PIDs (fileproviderd, mds_stores) will silently skip; that's expected. To enable full system-indexer throttling, see `SUDO_ACTIONS_NEEDED.md`.

## Operational

### Check status

```bash
launchctl list | grep -E "drive_(sync|throttle)"
# both should be present (PID may be `-` between intervals)

tail -5 /tmp/com.zg.drive_throttle_renice.log
tail -5 "/Users/orginal/.../AI-Tools/logs/drive_sync_batch_$(date -u +%Y-%m-%d).log"
```

### Force a sync now

```bash
/Users/orginal/.venvs/sp500-mastery/bin/python /Users/orginal/.zg/bin/drive_sync_batch.py --force
```

### Disable temporarily

```bash
launchctl unload ~/Library/LaunchAgents/com.zg.drive_sync_batch.plist
launchctl unload ~/Library/LaunchAgents/com.zg.drive_throttle_renice.plist
# or
export DRIVE_SYNC_BATCH_DISABLE=1  # affects future invocations
```

### Re-enable

```bash
launchctl load ~/Library/LaunchAgents/com.zg.drive_sync_batch.plist
launchctl load ~/Library/LaunchAgents/com.zg.drive_throttle_renice.plist
```

### Update source then re-sync the local copy

When you edit `AI-Tools/scripts/drive_sync_batch.py` or `AI-Tools/scripts/drive_throttle_renice.sh`, also copy to `/Users/orginal/.zg/bin/` so launchd picks up the new code:

```bash
cp "/Users/orginal/.../AI-Tools/scripts/drive_sync_batch.py" /Users/orginal/.zg/bin/drive_sync_batch.py
cp "/Users/orginal/.../AI-Tools/scripts/drive_throttle_renice.sh" /Users/orginal/.zg/bin/drive_throttle_renice.sh
chmod +x /Users/orginal/.zg/bin/*
launchctl kickstart -k gui/$(id -u)/com.zg.drive_sync_batch
launchctl kickstart -k gui/$(id -u)/com.zg.drive_throttle_renice
```

## Smoke test results (2026-05-20 01:17 UTC)

Baseline 1-min load before throttle: **117.01**.
After installing throttle + reniceing Drive PID 2268 to +15: **22.26** within 3 minutes.

Sync of 5 × 10MB files (`logs/smoke_test/file_{1..5}.bin`, dd-generated zeros) from `/tmp/ai-tools-staging/logs/` → Drive:
- staging write: <1 s
- rsync to Drive: 0.37 s (in-process), total 6 files / 52,428,836 bytes
- load delta during sync: 0 (no observable spike — load 22.26 before, 22.26 after)

Conclusion: end-to-end staging→Drive sync path works, load impact is negligible, the load reduction comes from (a) avoiding direct churn on Drive FUSE during the burst, (b) reniceing Drive workers so they yield CPU when the rest of the system is busy.

## Future hardening

- **Sudo-enabled launchd** for renice'ing fileproviderd/mds_stores/corespotlightd (currently 1 of 4 root PIDs skipped — typically the highest CPU one). Would need root-domain launchd plist + carefully scoped sudoers entry, OR `chmod +s` on a wrapper that only renices known PIDs by name.
- **mdutil -i off on Drive root** (Spotlight exclusion): also requires sudo. Drive is FUSE so Spotlight can't actually index it anyway, but disabling prevents the constant retry storm. See `SUDO_ACTIONS_NEEDED.md`.
- **Per-daemon `DRIVE_STAGING` rollout**: currently only `autonomous_mode_daemon.py` is patched. Apply the same 3-line pattern to other write-heavy daemons (`agent_watchdog_daemon.py`, `mission_overseer.py`, `feature_discovery_daemon.py`, ...) as their plists are next rotated. Track in `state/drive_staging_rollout.md`.
- **rsync filter list** for explicit allowlist of subtrees (currently auto-discovers top-level dirs under STAGING_ROOT) — useful once we accumulate more staged subtrees and want fine-grained sync windows.

## Backup of pre-throttle state

`AI-Tools/backups/drive-perm-throttle-2026-05-20/`
- `autonomous_mode_daemon.py.bak` — pre-patch daemon
- `settings.json.bak` — pre-edit Claude settings
- `baseline_load.txt`, `baseline_top_procs.txt` — load 106.85, top procs
- `launchagents_list_pre.txt` — pre-install LaunchAgents inventory
