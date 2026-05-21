#!/usr/bin/env python3
"""
system_status.py — One-screen real-time system dashboard.
Run manually or via cron (*/5 * * * *).
Writes to: AI-Tools/dashboard/system_status.md  and stdout.
"""

import os
import json
import time
import datetime
import glob
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
DRIVE = Path("/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive")
AI   = DRIVE / "AI-Tools"
SP   = AI / "s&p500-ticker-mastery"
TTM  = AI / "trading-ticker-mastery"
LOGS = AI / "logs"
WD   = AI / "watchdog"
DASH = AI / "dashboard"

CLAUDE_TMP = Path("/private/tmp/claude-501")
NOW = datetime.datetime.now()
TODAY = NOW.strftime("%Y-%m-%d")


# ── Helpers ────────────────────────────────────────────────────────────────────
def safe_ls(p, pattern="*"):
    try:
        return list(Path(p).glob(pattern))
    except Exception:
        return []

def mtime(p):
    try:
        return os.path.getmtime(p)
    except Exception:
        return 0

def age_min(p):
    return (time.time() - mtime(p)) / 60

def count_dir(p, pattern="*"):
    return len([f for f in safe_ls(p, pattern) if not f.name.startswith(".")])

def count_json(p):
    return len([f for f in safe_ls(p, "*.json")])

def last_mtime_str(p):
    files = [f for f in safe_ls(p, "*") if f.is_file()]
    if not files:
        return "never"
    latest = max(files, key=lambda f: mtime(f))
    age = age_min(latest)
    if age < 60:
        return f"{int(age)}m ago ({latest.name})"
    elif age < 1440:
        return f"{int(age/60)}h ago ({latest.name})"
    else:
        return f"{int(age/1440)}d ago ({latest.name})"


# ── Section 1 — Mastery counts ─────────────────────────────────────────────────
def section_mastery():
    lines = []

    # Daily mastered — check mastered/daily/<today>.json
    daily_f = SP / "mastered" / "daily" / f"{TODAY}.json"
    daily_count = 0
    if daily_f.exists():
        try:
            data = json.loads(daily_f.read_text())
            daily_count = len(data) if isinstance(data, list) else (
                len(data.get("tickers", data.get("mastered", []))) if isinstance(data, dict) else 0
            )
        except Exception:
            daily_count = -1  # file exists but unreadable

    # Total mastered — count subdirs in backtests_xgb_v8 + v8b (proxy for mastered tickers)
    v8_dirs   = count_dir(SP / "backtests_xgb_v8") - 1   # subtract .DS_Store proxy
    v8b_dirs  = count_dir(SP / "backtests_xgb_v8b")
    total_mastered = max(v8_dirs, 0) + max(v8b_dirs, 0)

    # Intraday / incremental_bars today
    intraday_f = SP / "mastered" / "incremental_bars"
    intraday_today = len([f for f in safe_ls(intraday_f, f"*{TODAY}*")])

    lines.append(f"  Daily mastered today  : {daily_count}")
    lines.append(f"  Intraday bars today   : {intraday_today}")
    lines.append(f"  Total v8/v8b runs     : {total_mastered}")
    return lines


# ── Section 2 — Daemons live ───────────────────────────────────────────────────
CRON_DAEMONS = [
    ("feature_discovery", LOGS / "feature_discovery_cron.log"),
    ("overseer",          LOGS / "overseer_cron.log"),
    ("broadcast_daemon",  LOGS / "broadcast_daemon.log"),
    ("paper_trade",       SP  / "logs" / "paper_trade_cron.log"),
    ("progress_dashboard",LOGS / "dashboard_cron.log"),
]

def section_daemons():
    lines = []
    for name, log_path in CRON_DAEMONS:
        if Path(log_path).exists():
            age = age_min(log_path)
            status = "LIVE" if age < 15 else ("STALE" if age < 120 else "DEAD")
            lines.append(f"  {name:<22} [{status}]  last={int(age)}m ago")
        else:
            lines.append(f"  {name:<22} [NO LOG]")
    return lines


# ── Section 3 — Sub-agents in flight ──────────────────────────────────────────
def section_subagents():
    # Walk all session dirs under claude-501 project
    output_files = []
    try:
        for session_dir in CLAUDE_TMP.iterdir():
            task_dir = session_dir / "tasks"
            if task_dir.is_dir():
                for f in task_dir.glob("*.output"):
                    if age_min(f) < 10:
                        output_files.append(f)
    except Exception:
        pass
    return [f"  In-flight (<10 min)   : {len(output_files)} output file(s)"]


# ── Section 4 — Watchdog flagged ───────────────────────────────────────────────
def section_watchdog():
    hr_dir = WD / "help_requests"
    flagged_f = WD / "flagged.json"

    hr_count = count_json(hr_dir)
    flagged_count = 0
    if flagged_f.exists():
        try:
            data = json.loads(flagged_f.read_text())
            flagged_count = len(data) if isinstance(data, list) else len(data.get("flagged", []))
        except Exception:
            flagged_count = -1

    return [
        f"  Help requests (total) : {hr_count}",
        f"  Flagged agents        : {flagged_count}",
    ]


# ── Section 5 — v8 backtest_xgb runs ──────────────────────────────────────────
def section_v8_backtests():
    lines = []
    for ver in ["backtests_xgb_v8", "backtests_xgb_v8b", "backtests_xgb_v7",
                "backtests_xgb_v6", "backtests_xgb_v5"]:
        d = SP / ver
        n = count_dir(d)
        if n > 0:
            lines.append(f"  {ver:<28}: {n} runs")
    if not lines:
        lines.append("  No v8 backtest dirs found")
    return lines


# ── Section 6 — Intraday backtests by strategy ────────────────────────────────
def section_intraday():
    lines = []
    strategy_counts: dict = {}

    # trading-ticker-mastery/backtests/<ticker>/<strategy_dir>
    bt_root = TTM / "backtests"
    try:
        for ticker_dir in bt_root.iterdir():
            if not ticker_dir.is_dir():
                continue
            for strat_dir in ticker_dir.iterdir():
                if strat_dir.is_dir():
                    key = strat_dir.name
                    strategy_counts[key] = strategy_counts.get(key, 0) + 1
    except Exception:
        pass

    # Also scan s&p500-ticker-mastery/backtests_ml for threshold variants
    ml_sweep = SP / "backtests_ml_sweep"
    ml_sweep_count = count_dir(ml_sweep)
    if ml_sweep_count:
        strategy_counts["ml_sweep_variants"] = ml_sweep_count

    ml_v3 = SP / "backtests_ml_v3"
    ml_v3_count = count_dir(ml_v3)
    if ml_v3_count:
        strategy_counts["ml_v3_variants"] = ml_v3_count

    if strategy_counts:
        for strat, cnt in sorted(strategy_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {strat:<38}: {cnt}")
    else:
        lines.append("  No intraday backtest results found")
    return lines


# ── Section 7 — Recent DeepSeek calls (last hour) ─────────────────────────────
def section_deepseek():
    log_files = [
        LOGS / "broadcast_daemon.log",
        LOGS / "proactive_loop.log",
        LOGS / "overseer_cron.log",
        LOGS / "proactive_loop_stdout.log",
    ]
    cutoff = time.time() - 3600
    ds_calls = 0

    for lf in log_files:
        if not Path(lf).exists():
            continue
        try:
            lines = Path(lf).read_text(errors="replace").splitlines()
            for line in lines:
                # Lines with a timestamp + DeepSeek reference
                if "deepseek" in line.lower() or "DeepSeek" in line:
                    # Try to extract ISO timestamp from the line
                    if line.startswith("[2026-"):
                        try:
                            ts_str = line[1:27]  # e.g. 2026-05-16T18:00:51.390723
                            ts = datetime.datetime.fromisoformat(ts_str.replace("+00:00", ""))
                            # Treat as UTC, compare vs local now (rough)
                            ts_epoch = ts.timestamp()
                            if abs(ts_epoch - time.time()) < 7200 + 3600:  # within 4h window, filter by age
                                if time.time() - ts_epoch < 3600:
                                    ds_calls += 1
                        except Exception:
                            pass
        except Exception:
            pass

    return [f"  DeepSeek calls (1h)   : {ds_calls}"]


# ── Section 8 — Paper-trade signals today ─────────────────────────────────────
def section_paper_trade():
    sig_f = SP / "paper_trade" / "signals" / f"{TODAY}.json"
    long_count = short_count = total = 0
    if sig_f.exists():
        try:
            data = json.loads(sig_f.read_text())
            if isinstance(data, list):
                total = len(data)
                long_count  = sum(1 for s in data if s.get("signal") in ("BUY", "LONG", 1, "1"))
                short_count = sum(1 for s in data if s.get("signal") in ("SELL", "SHORT", -1, "-1"))
        except Exception:
            total = -1

    state_f = SP / "paper_trade" / "state" / f"{TODAY}_state.json"
    positions = 0
    if state_f.exists():
        try:
            state = json.loads(state_f.read_text())
            positions = len(state.get("positions", state.get("open_positions", {})))
        except Exception:
            pass

    return [
        f"  Signals filed today   : {total}  (long={long_count}, short={short_count})",
        f"  Open positions (state): {positions}",
    ]


# ── Assemble report ────────────────────────────────────────────────────────────
def build_report():
    ts = NOW.strftime("%Y-%m-%d %H:%M:%S")
    sep = "─" * 60

    sections = [
        f"SYSTEM STATUS  {ts}",
        sep,
        "[ 1 ] MASTERY",
        *section_mastery(),
        sep,
        "[ 2 ] DAEMONS LIVE",
        *section_daemons(),
        sep,
        "[ 3 ] SUB-AGENTS IN FLIGHT",
        *section_subagents(),
        sep,
        "[ 4 ] WATCHDOG",
        *section_watchdog(),
        sep,
        "[ 5 ] v8 BACKTEST_XGB RUNS",
        *section_v8_backtests(),
        sep,
        "[ 6 ] INTRADAY BACKTESTS BY STRATEGY",
        *section_intraday(),
        sep,
        "[ 7 ] DEEPSEEK CALLS",
        *section_deepseek(),
        sep,
        "[ 8 ] PAPER-TRADE SIGNALS TODAY",
        *section_paper_trade(),
        sep,
    ]
    return "\n".join(sections)


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    report = build_report()

    # Print to stdout
    print(report)

    # Save to dashboard/system_status.md
    DASH.mkdir(parents=True, exist_ok=True)
    out_path = DASH / "system_status.md"
    out_path.write_text(f"```\n{report}\n```\n")
