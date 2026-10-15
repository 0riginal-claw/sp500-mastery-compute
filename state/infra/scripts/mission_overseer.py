#!/usr/bin/env python3
"""
mission_overseer.py — autonomous mission overseer daemon.

Polls every 60s:
  1. Tasks API state via tasks_api_helper.py (pending + in_progress)
  2. Background agent output dir: /private/tmp/claude-501/.../tasks/*.output
  3. launchctl list | grep com.zg.* (verify daemons up)
  4. Cloud dispatch queue: s&p500-ticker-mastery/queue/
  5. OpenClaw completions tail: ~/.claude/state/openclaw_completions.jsonl

Detects:
  - in_progress task w/ no file/log activity >30min  → STALL
  - pending task >2h w/ no spawn                     → BLOCKED
  - daemon PID=- AND exit_status!=0                  → CRASHED
  - background agent duration >20min still running   → §3 KILL violation

On detection:
  - Append JSONL to logs/mission_overseer/stalls_<UTC_DATE>.jsonl
  - Spawn 3 §8 solvers (INTERNET/GITHUB/REPO-LOCAL) via openclaw-gdrive
    (falls back to a marker file consumed by the next Claude session if no
    openclaw available)
  - Mark issue NOTED in state file to avoid double-spawn

Every 60s writes:
  dashboard/MISSION_PROGRESS.md  — snapshot for at-a-glance status

Critical alerts (Pushbullet if $PUSHBULLET_TOKEN, else Drive status file):
  - daemon CRASHED
  - task BLOCKED >4h
  - signal-gen failure during US market hours
  - model version drift detected in logs
  - Mac load >12 (cloud-routing mandate breach)

Author: spawned by Claude Code parent session per §3 §8 §5a mandates.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
import traceback
import urllib.parse
import urllib.request
from collections import deque
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

# ----------------------------------------------------------------------------
# Constants / paths
# ----------------------------------------------------------------------------
AI_ROOT = Path(
    "/Users/orginal/Library/CloudStorage/"
    "GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools"
)
DASHBOARD_DIR = AI_ROOT / "dashboard"
DASHBOARD_FILE = DASHBOARD_DIR / "MISSION_PROGRESS.md"
LOG_DIR = AI_ROOT / "logs" / "mission_overseer"
STATE_DIR = AI_ROOT / "state" / "mission_overseer"
NOTED_FILE = STATE_DIR / "noted_issues.json"
ALERT_HISTORY_FILE = STATE_DIR / "alert_history.jsonl"
CRITICAL_ALERT_FILE = DASHBOARD_DIR / "CRITICAL_ALERTS.md"
LOAD_HISTORY_FILE = STATE_DIR / "load_history.json"
# 5min avg + 3-reading hysteresis added 2026-05-20 — eliminates false-positive
# alerts from transient ps-snapshot CPU spikes (see helper a667211d report).
LOAD_HYSTERESIS_N = 3

TASKS_FILE = AI_ROOT / ".claude" / "tasks.json"
TASKS_HELPER = AI_ROOT / "scripts" / "tasks_api_helper.py"

OPENCLAW_LAUNCHER = AI_ROOT / "bin" / "openclaw-gdrive"
OPENCLAW_COMPLETIONS = Path.home() / ".claude" / "state" / "openclaw_completions.jsonl"
SOLVER_QUEUE_DIR = AI_ROOT / "state" / "mission_overseer" / "pending_solvers"

BG_AGENT_OUTPUT_GLOB = Path("/private/tmp/claude-501")
QUEUE_DIR = AI_ROOT / "s&p500-ticker-mastery" / "queue"

POLL_INTERVAL_SEC = 60
STALL_THRESHOLD_MIN = 30
BLOCKED_THRESHOLD_MIN = 120
BG_AGENT_KILL_THRESHOLD_MIN = 20
TASK_BLOCKED_CRITICAL_MIN = 240  # 4h
MAC_LOAD_CRITICAL = 12.0

# US equity market hours (rough — ignores half-days/holidays, good enough for alert)
MARKET_OPEN_UTC = (13, 30)   # 09:30 ET ≈ 13:30 UTC (DST-naive approximation)
MARKET_CLOSE_UTC = (20, 0)   # 16:00 ET ≈ 20:00 UTC


# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------
def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ts_iso() -> str:
    return utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_dirs() -> None:
    for d in (DASHBOARD_DIR, LOG_DIR, STATE_DIR, SOLVER_QUEUE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_noted() -> dict:
    if not NOTED_FILE.exists():
        return {}
    try:
        return json.loads(NOTED_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def save_noted(noted: dict) -> None:
    try:
        NOTED_FILE.write_text(json.dumps(noted, indent=2))
    except OSError:
        pass


def append_jsonl(path: Path, record: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError:
        pass


def human_ago(dt: datetime) -> str:
    delta = utc_now() - dt
    secs = int(delta.total_seconds())
    if secs < 60:
        return f"{secs}s ago"
    if secs < 3600:
        return f"{secs // 60}min ago"
    if secs < 86400:
        return f"{secs // 3600}h{(secs % 3600) // 60}m ago"
    return f"{secs // 86400}d ago"


def is_us_market_hours(now: Optional[datetime] = None) -> bool:
    now = now or utc_now()
    if now.weekday() >= 5:
        return False
    open_h, open_m = MARKET_OPEN_UTC
    close_h, close_m = MARKET_CLOSE_UTC
    after_open = (now.hour, now.minute) >= (open_h, open_m)
    before_close = (now.hour, now.minute) <= (close_h, close_m)
    return after_open and before_close


# ----------------------------------------------------------------------------
# Pollers
# ----------------------------------------------------------------------------
def poll_tasks() -> list[dict]:
    """Read tasks.json directly (mirrors tasks_api_helper output)."""
    if not TASKS_FILE.exists():
        return []
    try:
        return json.loads(TASKS_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def poll_bg_agents() -> list[dict]:
    """
    Inspect /private/tmp/claude-501/<session>/tasks/*.output (and *.json).
    Returns list of {path, size, mtime, age_min}.
    """
    out: list[dict] = []
    if not BG_AGENT_OUTPUT_GLOB.exists():
        return out
    try:
        for sess in BG_AGENT_OUTPUT_GLOB.iterdir():
            if not sess.is_dir():
                continue
            tdir = sess / "tasks"
            if not tdir.exists():
                continue
            for f in tdir.iterdir():
                try:
                    st = f.stat()
                except OSError:
                    continue
                mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)
                age = (utc_now() - mtime).total_seconds() / 60.0
                out.append({
                    "path": str(f),
                    "size": st.st_size,
                    "mtime": mtime.isoformat(),
                    "age_min": round(age, 1),
                    "session": sess.name,
                })
    except OSError:
        pass
    return out


def poll_launchctl() -> list[dict]:
    """Parse launchctl list output for com.zg.* daemons."""
    rows: list[dict] = []
    try:
        res = subprocess.run(
            ["launchctl", "list"],
            capture_output=True, text=True, timeout=10
        )
    except (subprocess.SubprocessError, OSError):
        return rows
    if res.returncode != 0:
        return rows
    for line in res.stdout.splitlines():
        if "com.zg." not in line:
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            parts = line.split()
        if len(parts) < 3:
            continue
        pid_str, status_str, label = parts[0].strip(), parts[1].strip(), parts[2].strip()
        try:
            status = int(status_str)
        except ValueError:
            status = 0
        pid: Optional[int] = None
        if pid_str != "-":
            try:
                pid = int(pid_str)
            except ValueError:
                pid = None
        rows.append({"label": label, "pid": pid, "status": status})
    return rows


def poll_queue() -> dict:
    """Cloud dispatch queue snapshot."""
    if not QUEUE_DIR.exists():
        return {"exists": False, "count": 0, "oldest_age_min": None}
    try:
        files = [f for f in QUEUE_DIR.iterdir() if f.is_file()]
    except OSError:
        return {"exists": True, "count": 0, "oldest_age_min": None, "error": "iterdir"}
    if not files:
        return {"exists": True, "count": 0, "oldest_age_min": None}
    oldest = min((f.stat().st_mtime for f in files), default=time.time())
    age_min = (time.time() - oldest) / 60.0
    return {"exists": True, "count": len(files), "oldest_age_min": round(age_min, 1)}


def poll_openclaw_completions(tail: int = 5) -> list[dict]:
    """Tail recent OpenClaw completions."""
    if not OPENCLAW_COMPLETIONS.exists():
        return []
    try:
        lines = OPENCLAW_COMPLETIONS.read_text().splitlines()[-tail:]
    except OSError:
        return []
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return out


def poll_mac_load() -> float:
    """5-minute load average (was 1-min — switched 2026-05-20 to reduce
    false-positive alerts from transient ps-snapshot CPU spikes)."""
    try:
        return os.getloadavg()[1]
    except OSError:
        return 0.0


def load_load_history() -> deque:
    """Load persisted 3-reading history deque (survives daemon restarts)."""
    if not LOAD_HISTORY_FILE.exists():
        return deque(maxlen=LOAD_HYSTERESIS_N)
    try:
        data = json.loads(LOAD_HISTORY_FILE.read_text())
        readings = data.get("readings", [])[-LOAD_HYSTERESIS_N:]
        return deque(readings, maxlen=LOAD_HYSTERESIS_N)
    except (json.JSONDecodeError, OSError):
        return deque(maxlen=LOAD_HYSTERESIS_N)


def save_load_history(history: deque) -> None:
    try:
        LOAD_HISTORY_FILE.write_text(json.dumps({
            "readings": list(history),
            "updated_at": ts_iso(),
        }))
    except OSError:
        pass


# ----------------------------------------------------------------------------
# Detectors
# ----------------------------------------------------------------------------
def detect_stalls(tasks: list[dict], bg_agents: list[dict]) -> list[dict]:
    """
    Stall conditions per the brief.
    Each issue gets a deterministic key for noted-deduplication.
    """
    issues: list[dict] = []
    now = utc_now()

    for t in tasks:
        status = t.get("status", "")
        updated = t.get("updated_at") or t.get("created_at") or ""
        try:
            updated_dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            continue
        age_min = (now - updated_dt).total_seconds() / 60.0

        if status == "in_progress" and age_min > STALL_THRESHOLD_MIN:
            issues.append({
                "kind": "STALL",
                "key": f"task_stall_{t.get('id', 'unknown')}_{int(updated_dt.timestamp())}",
                "task_id": t.get("id"),
                "title": t.get("title"),
                "age_min": round(age_min, 1),
                "summary": f"in_progress task #{t.get('id')} idle {round(age_min)}min",
            })
        elif status == "pending" and age_min > BLOCKED_THRESHOLD_MIN:
            issues.append({
                "kind": "BLOCKED",
                "key": f"task_blocked_{t.get('id', 'unknown')}_{int(updated_dt.timestamp())}",
                "task_id": t.get("id"),
                "title": t.get("title"),
                "age_min": round(age_min, 1),
                "summary": f"pending task #{t.get('id')} unspawned {round(age_min)}min",
            })

    # §3 KILL — background agents >20min still running (size growing = active)
    for ag in bg_agents:
        if ag["age_min"] > BG_AGENT_KILL_THRESHOLD_MIN and ag["age_min"] < BG_AGENT_KILL_THRESHOLD_MIN * 60:
            # Still recent enough to mean "active long-runner", not historical
            if ag["age_min"] < 60:  # 20-60 min window = active KILL violation
                issues.append({
                    "kind": "BG_AGENT_KILL_VIOLATION",
                    "key": f"bg_kill_{Path(ag['path']).name}_{int(ag['age_min'])}",
                    "path": ag["path"],
                    "age_min": ag["age_min"],
                    "summary": f"bg agent {Path(ag['path']).name} active {ag['age_min']}min — §3 KILL",
                })

    return issues


def detect_crashed_daemons(daemons: list[dict]) -> list[dict]:
    """Daemon w/ PID=None AND exit_status != 0 → CRASHED."""
    out = []
    for d in daemons:
        if d["pid"] is None and d["status"] != 0:
            out.append({
                "kind": "CRASHED",
                "key": f"daemon_crashed_{d['label']}_{d['status']}",
                "label": d["label"],
                "status": d["status"],
                "summary": f"daemon {d['label']} exit_status={d['status']}, PID=-",
            })
    return out


# ----------------------------------------------------------------------------
# Solver spawning (best-effort — daemon can't directly invoke Claude sub-agents,
# but can drop a marker file for the next Claude session + invoke openclaw)
# ----------------------------------------------------------------------------
def spawn_solvers(issue: dict) -> dict:
    """
    Spawn 3 parallel §8 solvers.

    The overseer is a launchd daemon, not a Claude session — it cannot directly
    call mcp__plugin_fallback-agent_fallback__Task. Two channels:

      1. OpenClaw + DeepSeek (--local) — fire-and-forget for each angle
      2. Drop a marker file at SOLVER_QUEUE_DIR/<issue_key>.json so the next
         Claude session reading the dashboard knows to spawn 3 Task helpers.

    Returns dict with channels attempted.
    """
    result = {"openclaw_calls": [], "marker_file": None, "errors": []}

    angles = [
        ("INTERNET", "Search the internet for known fixes."),
        ("GITHUB", "Search GitHub issues/PRs for known fixes."),
        ("REPO-LOCAL", "Search repo-local code/logs for known fixes."),
    ]

    # Channel 1: OpenClaw fire-and-forget (best effort, capped at 60s each)
    if OPENCLAW_LAUNCHER.exists():
        for label, angle_brief in angles:
            msg = (
                f"# autosolve_skip: overseer-spawned §8 solver\n"
                f"# model_reason: deepseek-v4-flash for cheap independent research\n"
                f"AUTOSOLVE {label} angle:\n"
                f"Issue: {issue.get('summary', issue)}\n"
                f"Kind: {issue.get('kind')}\n"
                f"Context: spawned by mission_overseer daemon. {angle_brief}\n"
                f"Return: 1 best fix in 200 words max, citing source."
            )
            log_dir = AI_ROOT / "logs" / "openclaw_fanout"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"overseer_{issue.get('key', 'unknown')}_{label}.log"
            try:
                # Fire-and-forget — don't block the daemon
                with log_path.open("w") as logfh:
                    subprocess.Popen(
                        [
                            str(OPENCLAW_LAUNCHER),
                            "agent", "--local",
                            "--model", "deepseek/deepseek-v4-flash",
                            "--json", "--timeout", "300",
                            "--message", msg,
                        ],
                        stdout=logfh, stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                result["openclaw_calls"].append({
                    "label": label, "log": str(log_path),
                })
            except (OSError, subprocess.SubprocessError) as exc:
                result["errors"].append(f"{label}: {exc}")

    # Channel 2: drop marker for next Claude session
    try:
        marker = SOLVER_QUEUE_DIR / f"{issue.get('key', 'unknown')}.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({
            "created_at": ts_iso(),
            "issue": issue,
            "instruction": (
                "Next Claude session reading dashboard/MISSION_PROGRESS.md should "
                "spawn 3 §8 solvers (INTERNET/GITHUB/REPO-LOCAL) for this issue if "
                "not yet resolved."
            ),
        }, indent=2))
        result["marker_file"] = str(marker)
    except OSError as exc:
        result["errors"].append(f"marker: {exc}")

    return result


# ----------------------------------------------------------------------------
# Alerting
# ----------------------------------------------------------------------------
def send_pushbullet(title: str, body: str) -> bool:
    token = os.environ.get("PUSHBULLET_TOKEN", "").strip()
    if not token:
        return False
    try:
        data = urllib.parse.urlencode({
            "type": "note", "title": title, "body": body,
        }).encode()
        req = urllib.request.Request(
            "https://api.pushbullet.com/v2/pushes",
            data=data,
            headers={"Access-Token": token},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def emit_critical_alert(title: str, body: str, dedup_key: str) -> dict:
    """
    Critical-channel alert. Always writes to:
      - state/mission_overseer/alert_history.jsonl (dedup window 1h)
      - dashboard/CRITICAL_ALERTS.md (rolling 24h)
    Plus Pushbullet if $PUSHBULLET_TOKEN set.
    """
    # Dedup: skip if same key fired in last hour
    if ALERT_HISTORY_FILE.exists():
        try:
            cutoff = utc_now() - timedelta(hours=1)
            for ln in ALERT_HISTORY_FILE.read_text().splitlines()[-200:]:
                try:
                    prev = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if prev.get("key") == dedup_key:
                    try:
                        prev_ts = datetime.fromisoformat(prev["ts"].replace("Z", "+00:00"))
                        if prev_ts > cutoff:
                            return {"deduped": True, "key": dedup_key}
                    except (ValueError, KeyError):
                        pass
        except OSError:
            pass

    record = {"ts": ts_iso(), "key": dedup_key, "title": title, "body": body}
    append_jsonl(ALERT_HISTORY_FILE, record)

    pb_ok = send_pushbullet(title, body)

    # Append to rolling CRITICAL_ALERTS.md
    try:
        line = f"- **{ts_iso()}** [{('PB-OK' if pb_ok else 'PB-skip')}] **{title}** — {body}\n"
        if not CRITICAL_ALERT_FILE.exists():
            CRITICAL_ALERT_FILE.write_text("# Critical Alerts (rolling)\n\n")
        # Prune > 24h
        try:
            cur = CRITICAL_ALERT_FILE.read_text().splitlines()
            cutoff = utc_now() - timedelta(hours=24)
            kept = ["# Critical Alerts (rolling)", ""]
            for cl in cur[2:]:
                m = re.match(r"- \*\*(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\*\*", cl)
                if not m:
                    continue
                try:
                    cl_ts = datetime.fromisoformat(m.group(1).replace("Z", "+00:00"))
                except ValueError:
                    continue
                if cl_ts > cutoff:
                    kept.append(cl)
            kept.append(line.rstrip())
            CRITICAL_ALERT_FILE.write_text("\n".join(kept) + "\n")
        except OSError:
            with CRITICAL_ALERT_FILE.open("a") as fh:
                fh.write(line)
    except OSError:
        pass

    return {"deduped": False, "pushbullet": pb_ok, "key": dedup_key}


# ----------------------------------------------------------------------------
# Dashboard writer
# ----------------------------------------------------------------------------
def write_dashboard(
    tasks: list[dict],
    bg_agents: list[dict],
    daemons: list[dict],
    queue: dict,
    completions: list[dict],
    mac_load: float,
    issues: list[dict],
    noted: dict,
) -> None:
    now = ts_iso()
    lines: list[str] = []
    lines.append(f"# Mission Progress — {now}")
    lines.append("")
    lines.append(f"_Refreshed every {POLL_INTERVAL_SEC}s by `com.zg.mission_overseer`._")
    lines.append("")

    # Active
    active = [t for t in tasks if t.get("status") == "in_progress"]
    lines.append(f"## Active (in_progress) — {len(active)}")
    if not active:
        lines.append("- _none_")
    for t in active:
        try:
            dt = datetime.fromisoformat(
                (t.get("updated_at") or t.get("created_at") or "").replace("Z", "+00:00")
            )
            ago = human_ago(dt)
        except (ValueError, AttributeError):
            ago = "unknown"
        lines.append(f"- **#{t.get('id')}** {t.get('title', '?')} — last activity {ago}")
    lines.append("")

    # Pending
    pending = [t for t in tasks if t.get("status") == "pending"]
    lines.append(f"## Pending — {len(pending)}")
    if not pending:
        lines.append("- _none_")
    for t in pending[:10]:
        lines.append(f"- #{t.get('id')} {t.get('title', '?')}")
    if len(pending) > 10:
        lines.append(f"- _… +{len(pending) - 10} more_")
    lines.append("")

    # Stalled / blocked / crashed
    if issues:
        lines.append(f"## Issues detected — {len(issues)}")
        for iss in issues:
            noted_at = noted.get(iss["key"], {}).get("noted_at", "just now")
            lines.append(f"- **{iss['kind']}** — {iss['summary']} (noted {noted_at})")
    else:
        lines.append("## Issues detected — 0")
        lines.append("- _clean_")
    lines.append("")

    # Background agents
    lines.append(f"## Background agents — {len(bg_agents)}")
    if not bg_agents:
        lines.append("- _none_")
    long_runners = sorted([a for a in bg_agents if a["age_min"] > 5], key=lambda a: -a["age_min"])
    for ag in long_runners[:8]:
        kill_mark = " **§3 KILL?**" if ag["age_min"] > BG_AGENT_KILL_THRESHOLD_MIN else ""
        lines.append(f"- `{Path(ag['path']).name}` — age {ag['age_min']}min ({ag['size']}B){kill_mark}")
    lines.append("")

    # Daemons
    up_ct = sum(1 for d in daemons if d["pid"] is not None)
    crashed = [d for d in daemons if d["pid"] is None and d["status"] != 0]
    idle = [d for d in daemons if d["pid"] is None and d["status"] == 0]
    lines.append(f"## Daemons (com.zg.*) — {up_ct}/{len(daemons)} up")
    for d in sorted(daemons, key=lambda x: x["label"]):
        if d["pid"] is not None:
            state = f"PID {d['pid']} **UP**"
        elif d["status"] != 0:
            state = f"PID - exit={d['status']} **CRASHED**"
        else:
            state = "PID - **IDLE**"
        lines.append(f"- `{d['label']}` {state}")
    lines.append("")

    # Cloud dispatch
    lines.append("## Cloud dispatch queue")
    if not queue.get("exists"):
        lines.append("- queue dir not found")
    else:
        lines.append(f"- {queue['count']} item(s) in queue" + (
            f", oldest {queue['oldest_age_min']}min" if queue['oldest_age_min'] is not None else ""
        ))
    lines.append("")

    # OpenClaw completions
    lines.append("## OpenClaw completions (tail 5)")
    if not completions:
        lines.append("- _none_")
    for c in completions[-5:]:
        agent = c.get("agent_id", c.get("session_id", "?"))
        tstamp = c.get("completed_at", c.get("ts", "?"))
        lines.append(f"- `{agent}` @ {tstamp}")
    lines.append("")

    # Mac load
    cap_breach = " **CAP BREACH**" if mac_load > MAC_LOAD_CRITICAL else ""
    lines.append(f"## Mac load — {mac_load:.2f}{cap_breach}")
    lines.append(f"_cap = {MAC_LOAD_CRITICAL} per §5a cloud-routing mandate_")
    lines.append("")

    # Critical alerts (last 24h)
    lines.append("## Critical alerts (last 24h)")
    if CRITICAL_ALERT_FILE.exists():
        try:
            cnt = sum(
                1 for cl in CRITICAL_ALERT_FILE.read_text().splitlines()
                if cl.startswith("- **")
            )
            lines.append(f"- {cnt} alert(s) — see `dashboard/CRITICAL_ALERTS.md`")
        except OSError:
            lines.append("- (could not read alert file)")
    else:
        lines.append("- _none_")
    lines.append("")

    lines.append("---")
    lines.append(f"_Last write: {now} · host {socket.gethostname()} · py {platform.python_version()}_")

    try:
        DASHBOARD_FILE.write_text("\n".join(lines) + "\n")
    except OSError:
        pass


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
def one_iteration() -> dict:
    ensure_dirs()

    tasks = poll_tasks()
    bg_agents = poll_bg_agents()
    daemons = poll_launchctl()
    queue = poll_queue()
    completions = poll_openclaw_completions()
    mac_load = poll_mac_load()

    issues = detect_stalls(tasks, bg_agents)
    issues.extend(detect_crashed_daemons(daemons))

    noted = load_noted()
    new_issues = []
    for iss in issues:
        if iss["key"] in noted:
            continue
        noted[iss["key"]] = {
            "noted_at": ts_iso(),
            "summary": iss["summary"],
            "kind": iss["kind"],
        }
        new_issues.append(iss)

        # Log to daily JSONL
        day = utc_now().strftime("%Y-%m-%d")
        append_jsonl(LOG_DIR / f"stalls_{day}.jsonl", {
            "ts": ts_iso(),
            **iss,
        })

        # Spawn solvers
        spawn_result = spawn_solvers(iss)
        append_jsonl(LOG_DIR / f"spawns_{day}.jsonl", {
            "ts": ts_iso(),
            "issue_key": iss["key"],
            **spawn_result,
        })

        # Critical alert if appropriate
        if iss["kind"] == "CRASHED":
            emit_critical_alert(
                title=f"DAEMON CRASHED: {iss['label']}",
                body=f"exit_status={iss['status']} — see dashboard/MISSION_PROGRESS.md",
                dedup_key=iss["key"],
            )
        elif iss["kind"] == "BLOCKED" and iss.get("age_min", 0) > TASK_BLOCKED_CRITICAL_MIN:
            emit_critical_alert(
                title=f"TASK BLOCKED >4h: #{iss.get('task_id')}",
                body=f"{iss['title']} — see dashboard/MISSION_PROGRESS.md",
                dedup_key=iss["key"],
            )

    # Mac load critical alert (5min avg + 3-reading hysteresis added 2026-05-20
    # — eliminates false-positive alerts from transient ps-snapshot CPU spikes.
    # Require ALL 3 last readings >cap before emitting CRITICAL. Otherwise log INFO.).
    load_history = load_load_history()
    load_history.append(mac_load)
    save_load_history(load_history)
    if (
        len(load_history) >= LOAD_HYSTERESIS_N
        and all(r > MAC_LOAD_CRITICAL for r in load_history)
    ):
        readings_str = ",".join(f"{r:.2f}" for r in load_history)
        emit_critical_alert(
            title=f"MAC LOAD CRITICAL (sustained): {mac_load:.2f}",
            body=(
                f"5min-load >{MAC_LOAD_CRITICAL} for {LOAD_HYSTERESIS_N} consecutive "
                f"readings ({readings_str}) — §5a breach. Switch to cloud-route."
            ),
            dedup_key=f"mac_load_sustained_{int(mac_load)}",
        )
    elif mac_load > MAC_LOAD_CRITICAL:
        # Transient spike — log INFO to boot.log, no alert.
        try:
            (LOG_DIR / "boot.log").open("a").write(
                f"[{ts_iso()}] INFO mac_load={mac_load:.2f} > cap={MAC_LOAD_CRITICAL} "
                f"but hysteresis not met (history={list(load_history)}); no alert\n"
            )
        except OSError:
            pass

    # Market-hours signal-gen failure detector
    if is_us_market_hours():
        sig_log = AI_ROOT / "logs" / "signal_gen_errors.log"
        if sig_log.exists():
            try:
                st = sig_log.stat()
                age_min = (time.time() - st.st_mtime) / 60.0
                if age_min < 5:
                    emit_critical_alert(
                        title="SIGNAL-GEN error during market hours",
                        body=f"logs/signal_gen_errors.log updated {age_min:.1f}min ago",
                        dedup_key=f"siggen_{int(time.time() // 300)}",
                    )
            except OSError:
                pass

    save_noted(noted)
    write_dashboard(tasks, bg_agents, daemons, queue, completions, mac_load, issues, noted)

    return {
        "tasks": len(tasks),
        "bg_agents": len(bg_agents),
        "daemons": len(daemons),
        "issues_total": len(issues),
        "issues_new": len(new_issues),
        "mac_load": mac_load,
    }


def main() -> None:
    ensure_dirs()
    # Boot heartbeat
    try:
        (LOG_DIR / "boot.log").open("a").write(f"[{ts_iso()}] mission_overseer booted pid={os.getpid()}\n")
    except OSError:
        pass

    # Allow single-iteration mode for smoke test
    if "--once" in sys.argv:
        try:
            summary = one_iteration()
            print(json.dumps({"ok": True, **summary}, indent=2))
        except Exception as exc:
            print(json.dumps({"ok": False, "error": str(exc), "tb": traceback.format_exc()}))
            sys.exit(1)
        return

    while True:
        t0 = time.time()
        try:
            one_iteration()
        except Exception as exc:
            try:
                (LOG_DIR / "boot.log").open("a").write(
                    f"[{ts_iso()}] iteration ERROR: {exc}\n{traceback.format_exc()}\n"
                )
            except OSError:
                pass
        # Atomic heartbeat write (six-fail-fix F7 — 2026-05-20)
        try:
            import tempfile
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            _hb = STATE_DIR / "heartbeat.json"
            _payload = json.dumps({"ts": int(time.time()), "pid": os.getpid(), "status": "running"})
            with tempfile.NamedTemporaryFile(dir=str(STATE_DIR), delete=False, mode="w") as _tmp:
                _tmp.write(_payload)
                _tmp_path = _tmp.name
            os.replace(_tmp_path, _hb)
        except Exception:
            pass
        elapsed = time.time() - t0
        sleep_for = max(1.0, POLL_INTERVAL_SEC - elapsed)
        time.sleep(sleep_for)


if __name__ == "__main__":
    main()
