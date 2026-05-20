"""
live_dashboard_daemon.py — Always-fresh live dashboard daemon.

Runs as a continuous infinite loop (30-second intervals).
Writes:
  - dashboard/live.md         — human-readable (overwritten each iteration)
  - dashboard/live.json       — machine-readable (overwritten each iteration)
  - dashboard/live_history.jsonl — append-only retrospective log

Installed via LaunchAgent with KeepAlive=true for auto-restart.
Coexists with existing 10-min cron dashboard (progress_dashboard.py).
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORK = Path(
    "/Users/orginal/Library/CloudStorage/"
    "GoogleDrive-zachgladstone@gmail.com/My Drive/"
    "AI-Tools/s&p500-ticker-mastery"
)
DASH = WORK / "dashboard"
LOGS = WORK / "logs"
LIVE_MD = DASH / "live.md"
LIVE_JSON = DASH / "live.json"
LIVE_HISTORY = DASH / "live_history.jsonl"
LOG_FILE = LOGS / "live_dashboard.log"

INTERVAL_SEC = 30
MASTERY_TOTAL = 502

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
def _setup_logging() -> logging.Logger:
    LOGS.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("live_dashboard")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(LOG_FILE)
        fh.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(sh)
    return logger


log = _setup_logging()

# ---------------------------------------------------------------------------
# State collection helpers
# ---------------------------------------------------------------------------

def _age_seconds(path: Path) -> float | None:
    """Return seconds since path was last modified, or None if not found."""
    try:
        return time.time() - path.stat().st_mtime
    except Exception:
        return None


def _age_label(seconds: float | None) -> str:
    if seconds is None:
        return "N/A"
    if seconds < 60:
        return f"{int(seconds)}s ago"
    if seconds < 3600:
        return f"{int(seconds / 60)}m ago"
    return f"{seconds / 3600:.1f}h ago"


def _count_mastery_files() -> int:
    """Count unique mastered tickers by disk files."""
    try:
        mdir = WORK / "mastery_files"
        if not mdir.exists():
            return 0
        files = list(mdir.glob("*mastered*.md"))
        return len({p.stem.split("_")[0] for p in files})
    except Exception:
        return 0


def _active_backtest_procs() -> int:
    """Count running backtest_xgb processes via pgrep."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "backtest_xgb"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        pids = [p.strip() for p in result.stdout.splitlines() if p.strip()]
        return len(pids)
    except Exception:
        return 0


def _active_python_procs() -> int:
    """Count total python processes (broad)."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "python"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        pids = [p.strip() for p in result.stdout.splitlines() if p.strip()]
        return len(pids)
    except Exception:
        return 0


def _latest_discovery_report_age() -> float | None:
    """Age in seconds of most recent discovery report markdown."""
    try:
        rep_dir = WORK / "feature_discovery" / "reports"
        if not rep_dir.exists():
            return None
        reports = sorted(rep_dir.glob("*.md"))
        if not reports:
            return None
        return _age_seconds(reports[-1])
    except Exception:
        return None


def _latest_overseer_age() -> float | None:
    """Age in seconds of most recent overseer cycle JSON."""
    try:
        hist_dir = WORK / "overseer" / "history"
        if not hist_dir.exists():
            return None
        cycles = sorted(hist_dir.glob("*.json"))
        if not cycles:
            return None
        return _age_seconds(cycles[-1])
    except Exception:
        return None


def _latest_paper_trade_signal_age() -> float | None:
    """Age in seconds of most recent paper-trade signal file."""
    try:
        sig_dir = WORK / "paper_trade" / "signals"
        if not sig_dir.exists():
            return None
        signals = sorted(sig_dir.iterdir())
        if not signals:
            return None
        return _age_seconds(signals[-1])
    except Exception:
        return None


def _proactive_loop_stats() -> tuple[int, float | None]:
    """Return (iteration_count, seconds_since_last_iter) from live_history."""
    try:
        if not LIVE_HISTORY.exists():
            return 0, None
        lines = LIVE_HISTORY.read_text().splitlines()
        count = len([l for l in lines if l.strip()])
        if count == 0:
            return 0, None
        # Parse last line for ts
        last = json.loads(lines[-1])
        ts = datetime.fromisoformat(last["ts"])
        age = (datetime.now(timezone.utc) - ts).total_seconds()
        return count, age
    except Exception:
        return 0, None


# ---------------------------------------------------------------------------
# Snapshot builder
# ---------------------------------------------------------------------------

def collect_snapshot() -> dict:
    now = datetime.now(timezone.utc)
    mastery_count = _count_mastery_files()
    backtest_procs = _active_backtest_procs()
    python_procs = _active_python_procs()
    discovery_age = _latest_discovery_report_age()
    overseer_age = _latest_overseer_age()
    signal_age = _latest_paper_trade_signal_age()
    iter_count, last_iter_age = _proactive_loop_stats()

    return {
        "ts": now.isoformat(),
        "mastery_count": mastery_count,
        "mastery_total": MASTERY_TOTAL,
        "mastery_pct": round(100.0 * mastery_count / MASTERY_TOTAL, 1),
        "active_backtests": backtest_procs,
        "active_python_procs": python_procs,
        "discovery_report_age_sec": discovery_age,
        "overseer_cycle_age_sec": overseer_age,
        "paper_trade_signal_age_sec": signal_age,
        "proactive_loop_iterations": iter_count,
        "last_iteration_age_sec": last_iter_age,
    }


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

def render_md(snap: dict) -> str:
    ts_local = datetime.fromisoformat(snap["ts"]).strftime("%Y-%m-%d %H:%M:%S UTC")
    disc_label = _age_label(snap["discovery_report_age_sec"])
    ov_label = _age_label(snap["overseer_cycle_age_sec"])
    sig_label = _age_label(snap["paper_trade_signal_age_sec"])
    iter_count = snap["proactive_loop_iterations"]
    last_iter_label = _age_label(snap["last_iteration_age_sec"])

    return (
        f"# LIVE DASHBOARD — {ts_local}\n\n"
        f"Mastery: {snap['mastery_count']}/{snap['mastery_total']} "
        f"({snap['mastery_pct']}%)\n"
        f"Active backtests: {snap['active_backtests']}\n"
        f"Active python processes: {snap['active_python_procs']}\n"
        f"Latest discovery report: {disc_label}\n"
        f"Latest overseer cycle: {ov_label}\n"
        f"Latest paper-trade signal: {sig_label}\n"
        f"Proactive loop iterations: {iter_count} (last {last_iter_label})\n"
    )


# ---------------------------------------------------------------------------
# Single iteration
# ---------------------------------------------------------------------------

def run_iteration() -> None:
    snap = collect_snapshot()

    DASH.mkdir(parents=True, exist_ok=True)

    # Overwrite live files
    LIVE_MD.write_text(render_md(snap))
    LIVE_JSON.write_text(json.dumps(snap, indent=2, default=str))

    # Append to history (compact line)
    history_entry = {
        "ts": snap["ts"],
        "mastery_count": snap["mastery_count"],
        "active_backtests": snap["active_backtests"],
        "active_python_procs": snap["active_python_procs"],
        "discovery_report_age_sec": snap["discovery_report_age_sec"],
        "overseer_cycle_age_sec": snap["overseer_cycle_age_sec"],
        "paper_trade_signal_age_sec": snap["paper_trade_signal_age_sec"],
    }
    with open(LIVE_HISTORY, "a") as fh:
        fh.write(json.dumps(history_entry) + "\n")

    log.info(
        "iter OK | mastery=%d/%d (%.1f%%) | backtests=%d | python_procs=%d | "
        "disc=%s | overseer=%s | signal=%s",
        snap["mastery_count"],
        snap["mastery_total"],
        snap["mastery_pct"],
        snap["active_backtests"],
        snap["active_python_procs"],
        _age_label(snap["discovery_report_age_sec"]),
        _age_label(snap["overseer_cycle_age_sec"]),
        _age_label(snap["paper_trade_signal_age_sec"]),
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("live_dashboard_daemon starting — interval=%ds", INTERVAL_SEC)
    DASH.mkdir(parents=True, exist_ok=True)

    while True:
        try:
            run_iteration()
        except Exception as exc:  # noqa: BLE001
            log.error("iteration failed: %s", exc, exc_info=True)

        time.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    main()
