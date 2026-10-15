#!/usr/bin/env python3
"""session_resume_checkpointer - 60s atomic checkpointing.

Writes to local SSD first, mirrors to Drive opportunistically.
Created 2026-05-20 by mega-builder Fix 4.
"""
import json, os, subprocess, sys, tempfile, time, uuid
from pathlib import Path
from datetime import datetime, timezone

DRIVE_ROOT = Path("/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools")
DRIVE_STATE = DRIVE_ROOT / "state" / "session_resume"
LOCAL_STATE = Path("/Users/orginal/.zg/state/session_resume")
HEARTBEAT = DRIVE_STATE / "heartbeat.json"
LOCAL_HEARTBEAT = LOCAL_STATE / "heartbeat.json"
CYCLE_SEC = 60
SESSION_ID = os.environ.get("CLAUDE_SESSION_ID") or str(uuid.uuid4())[:8]


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(path.parent), delete=False) as tmp:
        tmp.write(data); tmp.flush()
        try: os.fsync(tmp.fileno())
        except OSError: pass
        tmp_name = tmp.name
    os.replace(tmp_name, path)


def fuse_alive():
    try:
        r = subprocess.run(["cat", str(DRIVE_STATE / "fuse_sentinel.txt")],
                           timeout=2, capture_output=True)
        return r.returncode == 0
    except Exception:
        return False


def daemons_status():
    try:
        r = subprocess.run(["launchctl", "list"], capture_output=True, text=True, timeout=5)
        out = []
        for line in r.stdout.splitlines()[:300]:
            if "com.zg." not in line: continue
            parts = line.split()
            if len(parts) >= 3:
                out.append({"label": parts[2], "pid": parts[0], "exit": parts[1]})
        return out[:50]
    except Exception:
        return []


def build_checkpoint():
    try:
        l1, l5, l15 = os.getloadavg()
    except Exception:
        l1 = l5 = l15 = -1
    return {
        "schema_version": 1,
        "session_id": SESSION_ID,
        "ts": datetime.now(timezone.utc).isoformat(),
        "host": {"load1": l1, "load5": l5, "load15": l15, "drive_fuse_ok": fuse_alive()},
        "daemons": daemons_status(),
        "pid": os.getpid(),
    }


def write_heartbeat():
    hb = {"ts": int(time.time()), "pid": os.getpid(),
          "session_id": SESSION_ID, "status": "alive", "cycle_id": SESSION_ID}
    payload = json.dumps(hb).encode()
    for p in (LOCAL_HEARTBEAT, HEARTBEAT):
        try: atomic_write(p, payload)
        except Exception: pass


def sweep_orphans(state_dir, max_age_sec=300):
    """Sweep tmp* files older than max_age_sec (default 5min).

    Atomic-write race in os.replace can orphan tmpfiles on Drive FUSE.
    Run at every cycle start before writing new checkpoints.
    """
    now = time.time()
    swept = 0
    try:
        for tmp in state_dir.glob("tmp*"):
            try:
                age = now - tmp.stat().st_mtime
                if age > max_age_sec:
                    tmp.unlink()
                    swept += 1
            except Exception:
                pass
    except Exception:
        pass
    return swept


def main():
    LOCAL_STATE.mkdir(parents=True, exist_ok=True)
    try: DRIVE_STATE.mkdir(parents=True, exist_ok=True)
    except Exception: pass
    while True:
        try:
            # sweep orphan tmp* files before writing new checkpoints (atomic-write race fix)
            sweep_orphans(LOCAL_STATE)
            try: sweep_orphans(DRIVE_STATE)
            except Exception: pass
            cp = build_checkpoint()
            payload = json.dumps(cp, indent=2).encode()
            atomic_write(LOCAL_STATE / f"checkpoint_{SESSION_ID}.json", payload)
            atomic_write(LOCAL_STATE / "last_known_session.json",
                         json.dumps({"session_id": SESSION_ID, "ts": cp["ts"]}).encode())
            try:
                atomic_write(DRIVE_STATE / f"checkpoint_{SESSION_ID}.json", payload)
                atomic_write(DRIVE_STATE / "last_known_session.json",
                             json.dumps({"session_id": SESSION_ID, "ts": cp["ts"]}).encode())
            except Exception:
                pass
            write_heartbeat()
        except Exception as e:
            sys.stderr.write(f"checkpoint error: {e}\n")
        time.sleep(CYCLE_SEC)
    return 0


if __name__ == "__main__":
    sys.exit(main())
