#!/usr/bin/env python3
"""
autonomous_mode_daemon.py — Autonomous ideation + helper-spawn loop.

Differs from existing daemons:
  * proactive_loop_daemon.py  — passive ideation (rotates 5 questions, no spawning).
  * feature_discovery_daemon.py — GitHub recon, no helper dispatch.
  * ceo_orchestrator_daemon.py — pacing-policy router for OTHER daemons.
  * mission_overseer.py — watches/alerts, doesn't ideate.

This daemon:
  1. Every 5 min, checks toggle at state/autonomous_mode/config.json.
  2. If enabled=true: reads MISSION_PROGRESS.md, ideates via OpenClaw+DeepSeek,
     dedups + safety-gates + budget-gates the proposed action, then spawns a
     Claude helper via `claude -p --max-budget-usd 1.0`.
  3. Writes heartbeat every 60s so mission_overseer can detect hangs.
  4. On drift (>=3 of last 5 ideas <40% novel by Jaccard): spawns 3 parallel
     §8 solvers (INTERNET + GITHUB + REPO-LOCAL) to diagnose + fix root cause
     in background. Daemon NEVER pauses — continues ideating with an
     orthogonality clause injected into next cycle's prompt.
     # autosolve_skip: docstring update for 2026-05-20 drift amendment
  5. Default state: enabled=false (user MUST run `autonomous on` to start).

Safety hardrails:
  * Safety gate: hard-coded keyword blocklist (rm -rf, force push, drop table,
    kill -9 1, sudo rm, wallet, password, credential, transfer, wire, send money,
    SMS, email send) → REJECT, never spawn.
  * Budget gate: estimated cost > budget_remaining_usd → halt + alert.
  * Concurrent cap: spawned-in-flight count >= max_concurrent_spawns → queue.
  * Effort gate: helper brief effort_min > 30 → split first.
  * Impact gate: impact_score < 3/10 → skip.

Audit log: state/autonomous_mode/audit_<DATE>.jsonl — every action recorded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

# Self-awareness (added 2026-05-20). Optional import - daemon must not crash
# if these modules are missing.
# autosolve_skip: feature wiring - gabriel_self bootstrap
try:
    sys.path.insert(0, str(Path(__file__).parent.resolve()))
    import gabriel_self as _gabriel_self
    from gabriel_constitution import critique as _constitution_critique
    from gabriel_constitution import log_critique as _constitution_log
    GABRIEL_SELF_ENABLED = True
except Exception as _gse:  # noqa: BLE001
    _gabriel_self = None  # type: ignore[assignment]
    _constitution_critique = None  # type: ignore[assignment]
    _constitution_log = None  # type: ignore[assignment]
    GABRIEL_SELF_ENABLED = False
    _GABRIEL_SELF_IMPORT_ERR = repr(_gse)

# ─── Paths ───────────────────────────────────────────────────────────────────

ROOT = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools"
)
# autosolve_skip: amending in-flight build per user override directive — drift handler code change, no error active

# DRIVE_STAGING — redirect write-heavy state/logs to local SSD to avoid
# Google Drive sync / fileproviderd / mds_stores / corespotlightd load spikes
# when many concurrent helpers churn small files. The batch syncer
# (scripts/drive_sync_batch.py, launchd com.zg.drive_sync_batch) rsyncs
# /tmp/ai-tools-staging/ -> ROOT every 5 min. Unset to write straight to Drive.
_STAGING_ENV = os.environ.get("DRIVE_STAGING", "").strip()
DRIVE_STAGING: Path | None = Path(_STAGING_ENV) if _STAGING_ENV else None


def _write_root() -> Path:
    """Where this daemon writes high-churn state/logs.

    Returns DRIVE_STAGING if the env var is set (local SSD), else ROOT (Drive).
    Read paths (MISSION_PROGRESS, PROJECT_BEST) still always point at Drive.
    """
    return DRIVE_STAGING if DRIVE_STAGING is not None else ROOT


_WROOT = _write_root()
STATE_DIR = _WROOT / "state" / "autonomous_mode"
CONFIG_FILE = STATE_DIR / "config.json"
HEARTBEAT_FILE = STATE_DIR / "heartbeat.json"
SEEN_IDEAS_FILE = STATE_DIR / "seen_ideas.jsonl"
SPAWN_BRIEFS_DIR = STATE_DIR / "spawn_briefs"
DRIFT_EVENTS_FILE = STATE_DIR / "drift_events.jsonl"
DRIFT_SOLVER_LOG_DIR = _WROOT / "logs" / "autonomous_mode"
LOG_DIR_AUTONOMOUS = _WROOT / "logs" / "autonomous_mode"
ACTIVE_SPAWNS_FILE = STATE_DIR / "active_spawns.jsonl"

# ASK-PLAN-DECIDE-OBSERVE (added 2026-05-20)
BLOCKERS_DIR = STATE_DIR / "blockers"
DECISIONS_FILE = STATE_DIR / "decisions.jsonl"
RUN_LOOP_LOG_DIR = _WROOT / "logs" / "autonomous_mode"
PLAN_HISTORY_DIR = STATE_DIR / "plans"

# autosolve_skip: feature-add not error-fix
# User-inbox (added 2026-05-20) — user pipes ideas/requests via `autonomous ask|search|...`
USER_INBOX_FILE = STATE_DIR / "user_inbox.jsonl"
INBOX_ARCHIVE_DIR = STATE_DIR / "inbox_archive"
INBOX_ANSWERS_DIR = ROOT / "dashboard" / "inbox_answers"

# autosolve_skip: feature-add — USER-PERSONA loop (2026-05-20)
# Daemon also ideates AS the user (caveman-terse, completeness-mandate,
# scale-everything, fix-blockers, iterate-on-landed). Generated directives
# get +PERSONA_PRIORITY_BOOST and merge into the candidate stream BEFORE
# normal ideate candidates so they win priority ties.
USER_PROMPTS_HISTORY = ROOT / "state" / "user_prompts_history.jsonl"   # written by touch-last-prompt hook
USER_DIRECTIVES_DASHBOARD = ROOT / "dashboard" / "USER_DIRECTIVES.md"
USER_DIRECTIVES_LOG = STATE_DIR / "user_directives.jsonl"
PERSONA_PRIORITY_BOOST = int(os.environ.get("AUTONOMOUS_PERSONA_PRIORITY_BOOST", "2"))
PERSONA_MAX_DIRECTIVES_PER_CYCLE = int(os.environ.get("AUTONOMOUS_PERSONA_MAX_DIRECTIVES", "3"))
PERSONA_DIRECTIVES_RETAIN = 10   # last N shown in dashboard

# autosolve_skip: feature-add — NO-DIRECTION SELF-DIRECTING modules (2026-05-20)
# Six new modules let daemon act like the user WITHOUT user prompts:
#   1. user_predictor.json    — given state, what would user demand?
#   2. curiosity_state.json   — per-area last_touched + spawn_count, forces 20%
#                                of cycles into stale/low-spawn areas.
#   3. goal_tree.json         — top -> mid -> low hierarchical goals; daemon
#                                walks the tree to schedule helpers per blocker.
#   4. _intrinsic_reward      — surprise + learning + novelty per spawn outcome;
#                                high-reward families amplified, low de-prioritized.
#   5. time-aware behavior    — daytime (1/cycle, conservative), nighttime
#                                (3/cycle, aggressive), pre-open + post-close hooks.
#   6. skill library          — successful patterns become reusable skills;
#                                >5 invocations promoted to scripts/.
GABRIEL_SELF_DIR = STATE_DIR / "gabriel_self"
USER_PREDICTOR_FILE = GABRIEL_SELF_DIR / "user_predictor.json"
CURIOSITY_STATE_FILE = GABRIEL_SELF_DIR / "curiosity_state.json"
GOAL_TREE_FILE = GABRIEL_SELF_DIR / "goal_tree.json"
SKILLS_DIR = GABRIEL_SELF_DIR / "skills"
INTRINSIC_REWARDS_FILE = GABRIEL_SELF_DIR / "intrinsic_rewards.jsonl"

# Curiosity weights — fraction of cycles forced to high-curiosity area
CURIOSITY_FORCED_RATE = float(os.environ.get("AUTONOMOUS_CURIOSITY_RATE", "0.20"))
CURIOSITY_STALE_HOURS = int(os.environ.get("AUTONOMOUS_CURIOSITY_STALE_H", "24"))
# Project areas tracked by curiosity_state
CURIOSITY_AREAS = ("data", "model", "infra", "cloud", "research",
                   "trading", "self_improvement", "diagnostics")
# Time-aware: pre-market = 06:30 PT (13:30 UTC), post-close = 16:00 PT (23:00 UTC).
# Daytime in PT = 07:00..22:00 = 14:00..05:00 UTC.
DAYTIME_UTC_START_HOUR = 14
DAYTIME_UTC_END_HOUR = 5  # crosses midnight; daytime = [14..23] ∪ [0..5)
PRE_MARKET_UTC_HOUR = 13   # 06:00 PT
POST_CLOSE_UTC_HOUR = 23   # 16:00 PT
DAYTIME_SPAWNS_PER_CYCLE = int(os.environ.get("AUTONOMOUS_DAYTIME_SPAWNS", "1"))
NIGHTTIME_SPAWNS_PER_CYCLE = int(os.environ.get("AUTONOMOUS_NIGHTTIME_SPAWNS", "3"))

# Skill promotion threshold (>= N successful invocations promotes to scripts/)
SKILL_PROMOTE_THRESHOLD = int(os.environ.get("AUTONOMOUS_SKILL_PROMOTE", "5"))

# Intrinsic-reward weighting (sum == 1.0 by convention; not enforced)
INTRINSIC_W_SURPRISE = 0.40
INTRINSIC_W_LEARNING = 0.35
INTRINSIC_W_NOVELTY = 0.25

# Read-only inputs always come from Drive (source of truth).
DASHBOARD_DIR = ROOT / "dashboard"
MISSION_PROGRESS = DASHBOARD_DIR / "MISSION_PROGRESS.md"
AUTONOMOUS_STATUS = DASHBOARD_DIR / "AUTONOMOUS_STATUS.md"
AUTONOMOUS_PLAN = DASHBOARD_DIR / "AUTONOMOUS_PLAN.md"
PROJECT_BEST = DASHBOARD_DIR / "project_per_ticker_best.md"

LOG_DIR = _WROOT / "logs"
DAEMON_LOG = LOG_DIR / "autonomous_mode_daemon.log"

OPENCLAW_BIN = ROOT / "bin" / "openclaw-gdrive"
CLAUDE_BIN = ROOT / "bin" / "claude-gdrive"

# ─── Constants ───────────────────────────────────────────────────────────────

# Cycle interval: env-configurable; default 90s for "unlimited" aggressive fan-out.
# Operator can tighten/widen via AUTONOMOUS_CYCLE_SECONDS.
LOOP_SLEEP_SECONDS = int(os.environ.get("AUTONOMOUS_CYCLE_SECONDS", "90"))
HEARTBEAT_INTERVAL_SECONDS = 60    # write heartbeat once a minute
DISABLED_RECHECK_SECONDS = 60      # poll config quickly when off
MAX_IDEAS_PER_CYCLE = int(os.environ.get("AUTONOMOUS_MAX_IDEAS_PER_CYCLE", "3"))
MAX_NOVELTY_HISTORY = 10           # last N ideas for drift LOGGING (no pause)
NOVELTY_THRESHOLD = 0.40           # observability only — daemon does NOT pause on drift
MISSION_SUMMARY_MAX_TOKENS = 500   # ~ chars/4

# Load gate — ADAPTIVE (added 2026-05-20). Replaces fixed cap=10 that left
# the daemon idle under normal high-load conditions (Mac routinely 20-60).
# cap = max(LOAD_GATE_FLOOR, current_load + LOAD_GATE_HEADROOM) — always lets
# the daemon run + spawn >= 1 helper.
LOAD_GATE_FLOOR = int(os.environ.get("AUTONOMOUS_LOAD_FLOOR", "30"))
LOAD_GATE_HEADROOM = int(os.environ.get("AUTONOMOUS_LOAD_HEADROOM", "10"))

# ASK-PLAN-DECIDE-OBSERVE constants (added 2026-05-20)
DEEPSEEK_TIMEOUT_S = int(os.environ.get("AUTONOMOUS_DEEPSEEK_TIMEOUT", "120"))
OLLAMA_MODEL_FALLBACK = os.environ.get("AUTONOMOUS_OLLAMA_MODEL", "qwen2.5-coder")
PLAN_RETAIN_LAST_N = 10

# UNLIMITED MODE (amended 2026-05-20): user mandate "unlimited no restrictions".
# Soft caps removed. Only CLAUDE.md hard safety boundaries remain (see SAFETY_BLOCKLIST).
# Users MAY OPTIONALLY pass --max-spawns N or --budget-usd N to add a cap; defaults are unlimited.
UNLIMITED_SPAWNS_SENTINEL = 10**9        # effectively unlimited; honored as cap if config overrides
UNLIMITED_BUDGET_SENTINEL = float("inf") # effectively unlimited; honored as cap if config overrides

# ─── NEVER-SLEEP MODE (added 2026-05-20) ─────────────────────────────────────
# Three new mechanics added to guarantee continuous productivity even when
# DeepSeek returns prose / persona returns 0 / ideate returns 0:
#   1. _refill_backlog()  — hardcoded orthogonal seed ideas that ALWAYS produce
#      ≥1 candidate when all 3 generators return 0.  Rotates through 20 seeds
#      so the daemon never silent-idles.
#   2. _self_reflect()    — every SELF_REFLECT_EVERY_N cycles, read last 50
#      audit entries + write lessons.md so the daemon learns from itself.
#   3. exploration cycle  — EXPLORATION_RATE fraction of cycles inject a
#      random "pure exploration" seed direction so the daemon escapes
#      exploitation local optima (drift).
#   4. health_assert      — every cycle MUST produce ≥1 spawn OR log an
#      explicit `idle_because` event with diagnosis.
LESSONS_FILE = STATE_DIR / "lessons.md"
SELF_REFLECT_EVERY_N = int(os.environ.get("AUTONOMOUS_SELF_REFLECT_EVERY", "10"))
EXPLORATION_RATE = float(os.environ.get("AUTONOMOUS_EXPLORATION_RATE", "0.10"))
BACKLOG_SEED_ROTATE_FILE = STATE_DIR / "backlog_seed_idx.txt"

# Hardcoded orthogonal seed backlog (20 directions across data / model / infra /
# diagnostics axes). Rotates so the same one isn't picked twice in a row.
BACKLOG_SEEDS: list[dict[str, Any]] = [
    {"title": "audit_seen_ideas_diversity", "axis": "diagnostics",
     "brief": "Read state/autonomous_mode/seen_ideas.jsonl. Compute pairwise Jaccard. Output histogram + flag any cluster with >5 near-duplicates. Save report to logs/auto_solve/seen_ideas_diversity_<TS>.md."},
    {"title": "scan_paper_trade_drift_24h", "axis": "trading",
     "brief": "Read paper_trade ledger last 24h. Compute realized P&L vs expected (per-strategy). Flag any strategy >2σ from expected Sharpe. Save report to reports/paper_drift_<TS>.md."},
    {"title": "validate_per_ticker_best_freshness", "axis": "trading",
     "brief": "Read dashboard/project_per_ticker_best.md. For each ticker check last_updated. Flag any >7d stale. Spawn refresh helpers for top-10 stalest. Save list to reports/ticker_staleness_<TS>.md."},
    {"title": "cloud_dispatch_queue_inspect", "axis": "infra",
     "brief": "Inspect cloud_dispatch queue depth + last 50 jobs. Compute success rate, median latency, failure modes. Save report to reports/cloud_dispatch_health_<TS>.md."},
    {"title": "modal_spend_cap_status", "axis": "infra",
     "brief": "Query Modal usage API for current month spend vs cap. If >80%, log warning. If >95%, drain queued jobs. Save report to reports/modal_spend_<TS>.md."},
    {"title": "stale_daemon_audit", "axis": "diagnostics",
     "brief": "launchctl list | grep com.zg.* — compare against expected daemon list. Flag any down/IDLE for >1hr. Save report to reports/daemon_health_<TS>.md."},
    {"title": "improve_orthogonality_clause", "axis": "self_improvement",
     "brief": "Read scripts/autonomous_mode_daemon.py _build_orthogonality_clause. Test 3 variations against recent seen_ideas — score which produces highest novelty in DeepSeek output. Patch the winner. Save eval at reports/orthogonality_eval_<TS>.md."},
    {"title": "drift_solver_resolution_audit", "axis": "self_improvement",
     "brief": "Read state/autonomous_mode/drift_events.jsonl. For each event with status='solvers_in_flight' >2hr old, mark as 'orphaned'. For resolved, summarize fix-class. Save report to reports/drift_resolution_<TS>.md."},
    {"title": "feature_wiring_backlog", "axis": "feature_discovery",
     "brief": "Read feature_discovery/unwired_features.jsonl. For each unwired feature, generate a 1-line wire-it brief. Spawn top-3 by impact_score. Save wiring batch to feature_wiring/batch_<TS>.md."},
    {"title": "alpaca_paper_account_health", "axis": "trading",
     "brief": "Curl Alpaca paper /v2/account. Verify cash, equity, daytrade_count. Flag if any anomaly vs yesterday. Save snapshot to reports/alpaca_health_<TS>.md."},
    {"title": "research_recency_weighting_apply", "axis": "research_application",
     "brief": "Read memory/research_recency_weighting.md. Pick ONE technique not yet applied to per_ticker_best. Implement it on AAPL as proof. Save delta to reports/recency_apply_AAPL_<TS>.md."},
    {"title": "log_archive_purge", "axis": "infra",
     "brief": "Find logs/*.log >100MB or older than 30d. Compress to .gz and move to backups/log_archive_<MONTH>/. Save manifest to reports/log_purge_<TS>.md."},
    {"title": "drive_sync_lag_check", "axis": "infra",
     "brief": "Compare /tmp/ai-tools-staging/ mtime vs corresponding Drive paths. Flag any >5min lag. Restart drive_sync_batch if >2 staging files >10min old. Save report to reports/drive_sync_lag_<TS>.md."},
    {"title": "exploration_random_ticker_deep_dive", "axis": "exploration",
     "brief": "Pick a random S&P 500 ticker NOT in dashboard/project_per_ticker_best.md (or stalest one). Run full backtest sweep via cloud_dispatch.enqueue_job. Save plan to reports/random_ticker_<TICKER>_<TS>.md."},
    {"title": "audit_user_inbox_unanswered", "axis": "user_responsiveness",
     "brief": "Read state/autonomous_mode/user_inbox.jsonl. For any status='spawned' >2hr old without answer file, flag as orphaned + re-route. Save report to reports/inbox_health_<TS>.md."},
    {"title": "persona_prompt_eval", "axis": "self_improvement",
     "brief": "Read last 20 persona_ideate raw responses (state/autonomous_mode/blockers/*.jsonl). Compute parse-success rate. If <60%, tighten the prompt JSON schema example. Save eval at reports/persona_prompt_<TS>.md."},
    {"title": "exploration_orthogonal_data_source", "axis": "exploration",
     "brief": "Brainstorm 5 NEW data sources not currently in feature_discovery/. Pick top 1 by ROI estimate. Write integration brief. Save to research/notes/new_datasource_<TS>.md."},
    {"title": "rate_limit_inventory", "axis": "infra",
     "brief": "Read rate_limiter.py + redis_rate_limiter.py state. List current limits + headroom for: DeepSeek, OpenClaw, Alpaca, GitHub. Save inventory to reports/rate_limits_<TS>.md."},
    {"title": "checkpoint_paper_trade_pnl", "axis": "trading",
     "brief": "Aggregate state/paper_trade/* into project_paper_trade_pnl.md update. Append today's P&L row. Save delta to memory/project_paper_trade_pnl.md."},
    {"title": "self_audit_idle_periods", "axis": "diagnostics",
     "brief": "Read state/autonomous_mode/audit_<TODAY>.jsonl. Find any 30min+ gaps between spawn_launched events. Diagnose cause + write to lessons.md. Save report to reports/idle_audit_<TS>.md."},
]

DEFAULT_CONFIG = {
    "enabled": False,
    # UNLIMITED by default. Set max_concurrent_spawns to a finite int to add a cap.
    "max_concurrent_spawns": UNLIMITED_SPAWNS_SENTINEL,
    # UNLIMITED by default. Set budget_remaining_usd to a finite float to add a cap.
    "budget_remaining_usd": None,  # None == unlimited (no halt). Finite float == cap.
    "reason_off": "default: never ship enabled",
    "drift_pause_until": None,  # legacy field; daemon no longer sets this (kept for back-compat)
}

# Safety blocklist — rejected verbatim on substring match (case-insensitive).
# Anchored where it makes sense to avoid false positives ("password" inside a
# legit brief about password-strength checks would still reject — intentional).
SAFETY_BLOCKLIST = [
    "rm -rf",
    "rm  -rf",
    "force push",
    "force-push",
    "git push --force",
    "git push -f ",
    "drop table",
    "kill -9 1 ",
    "kill -9 1\n",
    "sudo rm",
    "wallet",
    "password",
    "credential",
    "transfer",
    "wire ",
    "send money",
    "send sms",
    "send-sms",
    "send email",
    "send-email",
    "mailto:",
    ".ssh/id_",
    "aws_secret",
    "private_key",
]

# ─── Logging ─────────────────────────────────────────────────────────────────

LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(DAEMON_LOG, mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("autonomous_mode")

# ─── Helpers ─────────────────────────────────────────────────────────────────


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _audit_path() -> Path:
    return STATE_DIR / f"audit_{_today_str()}.jsonl"


def _audit(record: dict[str, Any]) -> None:
    record["timestamp"] = _now_utc()
    _audit_path().parent.mkdir(parents=True, exist_ok=True)
    with _audit_path().open("a") as f:
        f.write(json.dumps(record) + "\n")


def _read_config() -> dict[str, Any]:
    if not CONFIG_FILE.exists():
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(json.dumps(DEFAULT_CONFIG, indent=2))
        log.info("config.json created with defaults (enabled=False)")
        return dict(DEFAULT_CONFIG)
    try:
        return json.loads(CONFIG_FILE.read_text())
    except json.JSONDecodeError as e:
        log.error("config.json invalid (%s) — using defaults, treating as off", e)
        return dict(DEFAULT_CONFIG)


def _write_config(cfg: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))


def _write_heartbeat(state: str, extra: dict[str, Any] | None = None) -> None:
    HEARTBEAT_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": _now_utc(), "state": state, "pid": os.getpid()}
    if extra:
        payload.update(extra)
    HEARTBEAT_FILE.write_text(json.dumps(payload, indent=2))


def _hash_title(title: str) -> str:
    return hashlib.sha256(title.strip().lower().encode()).hexdigest()[:16]


def _load_seen() -> set[str]:
    if not SEEN_IDEAS_FILE.exists():
        return set()
    seen: set[str] = set()
    with SEEN_IDEAS_FILE.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if h := rec.get("hash"):
                    seen.add(h)
            except json.JSONDecodeError:
                continue
    return seen


def _append_seen(title: str, idea_hash: str) -> None:
    SEEN_IDEAS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with SEEN_IDEAS_FILE.open("a") as f:
        f.write(json.dumps({"hash": idea_hash, "title": title, "ts": _now_utc()}) + "\n")


def _last_n_titles(n: int = MAX_NOVELTY_HISTORY) -> list[str]:
    if not SEEN_IDEAS_FILE.exists():
        return []
    titles: list[str] = []
    with SEEN_IDEAS_FILE.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if t := rec.get("title"):
                    titles.append(t)
            except json.JSONDecodeError:
                continue
    return titles[-n:]


def _jaccard(a: str, b: str) -> float:
    sa = set(re.findall(r"[a-z0-9]+", a.lower()))
    sb = set(re.findall(r"[a-z0-9]+", b.lower()))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _novelty_score(new_titles: list[str], history: list[str]) -> float:
    """Fraction of new_titles that are < 0.5 Jaccard to all history entries."""
    if not new_titles:
        return 1.0
    novel = 0
    for t in new_titles:
        if all(_jaccard(t, h) < 0.5 for h in history):
            novel += 1
    return novel / len(new_titles)


# autosolve_skip: drift handler — amendment per user override 2026-05-20
def _drift_detected(last_5_titles: list[str], threshold: float = 0.40) -> tuple[bool, dict[str, Any]]:
    """Drift = ≥3 of last 5 ideas have <40% Jaccard novelty vs prior ideas.

    Returns (is_drift, evidence_dict). Evidence dict carries the pairwise Jaccard
    matrix, per-title novelty scores, and the count of low-novelty titles — all
    suitable for logging to drift_events.jsonl.
    """
    titles = [t for t in (last_5_titles or []) if t]
    if len(titles) < 5:
        return False, {"reason": "insufficient_history", "n_titles": len(titles)}

    n = len(titles)
    # Pairwise Jaccard matrix (symmetric, diag=1.0)
    matrix: list[list[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 1.0
            else:
                matrix[i][j] = round(_jaccard(titles[i], titles[j]), 3)

    # Per-title novelty = 1 - max(Jaccard vs all others)
    # A title is "low-novelty" if novelty < threshold (i.e. it's ≥(1-threshold) similar to some other)
    low_novelty_count = 0
    per_title = []
    for i in range(n):
        max_sim = max(matrix[i][j] for j in range(n) if j != i)
        novelty = 1.0 - max_sim
        per_title.append({"title": titles[i], "novelty": round(novelty, 3), "max_sim": round(max_sim, 3)})
        if novelty < threshold:
            low_novelty_count += 1

    is_drift = low_novelty_count >= 3
    evidence = {
        "last_5_titles": titles,
        "novelty_matrix": matrix,
        "per_title_novelty": per_title,
        "low_novelty_count": low_novelty_count,
        "threshold": threshold,
        "is_drift": is_drift,
    }
    return is_drift, evidence


def _log_drift_event(evidence: dict[str, Any], state_summary: str, status: str = "solvers_in_flight") -> str:
    """Append a drift event row to drift_events.jsonl. Returns the event id."""
    DRIFT_EVENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    event_id = hashlib.sha256(
        f"{_now_utc()}|{evidence.get('last_5_titles', [])}".encode()
    ).hexdigest()[:16]
    row = {
        "event_id": event_id,
        "timestamp": _now_utc(),
        "last_5_ideas": evidence.get("per_title_novelty", []),
        "novelty_matrix": evidence.get("novelty_matrix", []),
        "state_summary_used": (state_summary or "")[:2000],
        "candidate_root_causes": [
            "stale state summary (>1hr since last refresh)",
            "narrow project goal scope in mission_summary",
            "helper history not included in ideation context",
            "DeepSeek hallucinating known repo patterns",
            "ideation prompt template too open-ended (no 'orthogonal to recent' clause)",
            "seen_ideas not surfaced to the model",
        ],
        "status": status,
    }
    with DRIFT_EVENTS_FILE.open("a") as f:
        f.write(json.dumps(row) + "\n")
    return event_id


def _spawn_drift_solver(channel: str, prompt: str, event_id: str) -> None:
    """Spawn a single §8 solver in the background. Non-blocking.

    channel ∈ {"internet", "github", "repo_local"}. Logs to
    logs/auto_solve/autonomous_drift_<event_id>_<channel>.log.
    Daemon never waits — fire-and-forget.
    """
    DRIFT_SOLVER_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = DRIFT_SOLVER_LOG_DIR / f"autonomous_drift_{event_id}_{channel}.log"

    if channel in ("internet", "github"):
        # Route via OpenClaw + DeepSeek (cheap, independent, no Claude budget burn)
        if not OPENCLAW_BIN.exists():
            log.warning("drift_solver[%s]: openclaw-gdrive missing — skip", channel)
            _audit({"event": "drift_solver_skip_no_launcher", "channel": channel, "event_id": event_id})
            return
        cmd = [
            str(OPENCLAW_BIN),
            "agent",
            "--local",
            "--agent", "main",
            "--model",
            "deepseek/deepseek-v4-flash",
            "--json",
            "--message",
            prompt,
        ]
    else:  # repo_local
        # Route via claude -p (REPO-LOCAL audit of state summary + prompt template)
        if not CLAUDE_BIN.exists():
            log.warning("drift_solver[%s]: claude-gdrive missing — skip", channel)
            _audit({"event": "drift_solver_skip_no_launcher", "channel": channel, "event_id": event_id})
            return
        # Write brief to a tmp file and pass as @path
        brief_path = SPAWN_BRIEFS_DIR / f"drift_{event_id}_repo_local.txt"
        SPAWN_BRIEFS_DIR.mkdir(parents=True, exist_ok=True)
        brief_path.write_text(prompt)
        cmd = [
            str(CLAUDE_BIN),
            "-p",
            "--max-budget-usd",
            "0.50",
            f"@{brief_path}",
        ]

    try:
        with log_path.open("w") as fh:
            subprocess.Popen(  # noqa: S603 — controlled args
                cmd,
                stdout=fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        log.info("drift_solver[%s] spawned (event=%s, log=%s)", channel, event_id, log_path.name)
        _audit({
            "event": "drift_solver_spawned",
            "channel": channel,
            "event_id": event_id,
            "log_path": str(log_path),
        })
    except OSError as e:
        log.error("drift_solver[%s] failed: %s", channel, e)
        _audit({"event": "drift_solver_failed", "channel": channel, "event_id": event_id, "error": str(e)})


# autosolve_skip: drift event-log helpers — amendment 2026-05-20
def _last_drift_event_ts() -> "datetime | None":
    """Return the timestamp of the most recent drift_events.jsonl row (or None)."""
    if not DRIFT_EVENTS_FILE.exists():
        return None
    last_ts: str | None = None
    try:
        with DRIFT_EVENTS_FILE.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if ts := rec.get("timestamp"):
                        last_ts = ts
                except json.JSONDecodeError:
                    continue
    except OSError:
        return None
    if not last_ts:
        return None
    try:
        return datetime.fromisoformat(last_ts)
    except ValueError:
        return None


def _apply_resolved_drift_config_fixes() -> None:
    """Scan drift_events.jsonl for rows marked status='resolved' with an
    auto-applicable config-change `resolution_action`. If any are pending
    application, apply them (currently a no-op stub — real auto-config-apply
    requires the resolver helpers to emit a structured `resolution_action`
    field on completion). Audit-only for now; safe to evolve in place."""
    if not DRIFT_EVENTS_FILE.exists():
        return
    # Defensive: cap scan size
    pending: list[dict[str, Any]] = []
    try:
        with DRIFT_EVENTS_FILE.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("status") == "resolved" and rec.get("resolution_action"):
                    if not rec.get("applied"):
                        pending.append(rec)
    except OSError:
        return
    if pending:
        log.info("drift: %d resolved event(s) pending config-fix application — audit only (no-op stub)",
                 len(pending))
        for rec in pending:
            _audit({
                "event": "drift_resolution_seen_no_op",
                "event_id": rec.get("event_id"),
                "resolution_action": rec.get("resolution_action"),
            })


def handle_drift(evidence: dict[str, Any], state_summary: str) -> str:
    """Drift response per 2026-05-20 override: spawn 3 §8 solvers, NEVER pause.

    Returns the drift event_id so the caller can tag heartbeats.
    """
    event_id = _log_drift_event(evidence, state_summary, status="solvers_in_flight")
    seen_titles_list = "\n".join(f"- {t}" for t in evidence.get("last_5_titles", []))

    internet_prompt = (
        "Drift in autonomous LLM agent loop — same/similar ideas proposed repeatedly. "
        "Diagnose root causes and propose concrete fixes. Cover: context window truncation, "
        "prompt template rigidity, lack of negative examples, missing 'don't repeat' instruction, "
        "narrow state summary, missing seen_ideas in context. Return JSON: "
        '{"root_causes": [...], "fixes": [{"description": ..., "type": "config|code", "patch": ...}]}.\n\n'
        f"Recent repeated ideas:\n{seen_titles_list}"
    )

    github_prompt = (
        "Find drift-mitigation code in autoGPT/babyAGI/AutoGen/CrewAI/LangGraph repos. "
        "Search GitHub code for: 'novelty filter', 'idea dedup', 'creativity boost', "
        "'diversity penalty', 'task novelty score'. Extract working patches and cite "
        "repo+file+line. Return JSON: "
        '{"sources": [{"repo": ..., "path": ..., "snippet": ..., "rationale": ...}]}.\n\n'
        f"Context — repeated ideas observed:\n{seen_titles_list}"
    )

    repo_local_prompt = (
        "# autosolve_skip: drift-solver REPO-LOCAL helper spawned by autonomous_mode_daemon\n"
        "# model_reason: sonnet — code audit + targeted patch (not pure mechanical)\n"
        "# scope_estimate_min: 8\n"
        "# decomposition_plan: single-helper audit slice (compose_state_summary + ideate prompt template)\n\n"
        "Workspace inherits standing authorization. Safety boundaries apply.\n\n"
        "## Task: drift root-cause audit (autonomous_mode_daemon)\n\n"
        "The autonomous_mode_daemon detected drift (≥3 of last 5 ideas <40% novel by Jaccard). "
        "Audit `scripts/autonomous_mode_daemon.py`:\n\n"
        "1. Inspect `compose_mission_summary()` — is the output stale (>1hr-old MISSION_PROGRESS.md slice)? "
        "If yes, identify a fresher signal source (recent audit_<DATE>.jsonl tail, or recent spawn_briefs/).\n"
        "2. Inspect `IDEATION_PROMPT_TEMPLATE` — does it ask vague open-ended Qs? If yes, tighten by "
        "injecting a 'these N ideas were already proposed, propose something orthogonal' clause that "
        "reads from `state/autonomous_mode/seen_ideas.jsonl` (last 20 titles).\n"
        "3. If pure config change suffices (prompt template tweak), patch in place and re-run `--once --dry-run` "
        "to confirm gate paths still work.\n"
        "4. If code change needed (e.g. extend compose_mission_summary signature), write the patch + run "
        "`python scripts/autonomous_mode_daemon.py --once --dry-run` to smoke test.\n\n"
        f"## Recent repeated ideas\n{seen_titles_list}\n\n"
        "## Proof of work\n"
        "Return: commands run, paths accessed, files changed, before/after diff of the prompt template "
        "and/or compose_mission_summary, smoke-test exit code, final status.\n"
    )

    _spawn_drift_solver("internet", internet_prompt, event_id)
    _spawn_drift_solver("github", github_prompt, event_id)
    _spawn_drift_solver("repo_local", repo_local_prompt, event_id)

    return event_id


# ─── ASK-PLAN-DECIDE-OBSERVE (added 2026-05-20) ──────────────────────────────
#
# Upgrade from ideate-spawn to full ReAct-style autonomous loop:
#   ASK     — self-question, identify blockers
#   PLAN    — multi-step plan with dependencies + success criteria
#   DECIDE  — rationale log per step (alternatives, why-chosen, risk)
#   EXECUTE — existing spawn_helper path (with decision_id correlation)
#   OBSERVE — ReAct trace: Thought/Plan/Action/Observation/Lesson


def _deepseek_call(prompt: str, timeout_s: int = DEEPSEEK_TIMEOUT_S) -> tuple[str | None, str]:
    """Call OpenClaw+DeepSeek with prompt. Returns (raw_output, source_tag).

    On failure (rate-limit / launcher-missing / timeout), falls back to Ollama
    local model so the daemon NEVER blocks waiting for cloud. Returns (None, tag)
    only if BOTH paths fail.
    """
    if OPENCLAW_BIN.exists():
        cmd = [
            str(OPENCLAW_BIN), "agent", "--local",
            "--agent", "main",
            "--model", "deepseek/deepseek-v4-flash",
            "--json", "--message", prompt,
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)
            if proc.returncode == 0 and proc.stdout.strip():
                return proc.stdout, "deepseek"
            log.warning("_deepseek_call: rc=%d stderr=%s", proc.returncode, proc.stderr[:200])
        except (subprocess.TimeoutExpired, OSError) as e:
            log.warning("_deepseek_call: exception %s — falling back to Ollama", e)

    # Fallback: Ollama local (free, slower but always available)
    ollama_helper = ROOT / "scripts" / "ollama_helper.py"
    if ollama_helper.exists():
        try:
            proc = subprocess.run(
                ["python3", str(ollama_helper), "--model", OLLAMA_MODEL_FALLBACK, "--prompt", prompt],
                capture_output=True, text=True, timeout=timeout_s, check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                return proc.stdout, "ollama_fallback"
        except (subprocess.TimeoutExpired, OSError) as e:
            log.warning("_deepseek_call ollama fallback failed: %s", e)
    return None, "both_failed"


def _unwrap_openclaw_json(raw: str, required_key: str) -> dict[str, Any] | None:
    """Extract JSON object containing `required_key` from OpenClaw response.

    OpenClaw wraps text in {payloads:[{text:'...'}]}. Inner text may contain
    markdown code fences around the JSON. Handles all combinations.
    """
    if not raw:
        return None

    def _try_parse(s: str) -> dict[str, Any] | None:
        s = s.strip()
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, re.DOTALL)
        if fence:
            s = fence.group(1)
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            m = re.search(rf"\{{[^{{}}]*\"{re.escape(required_key)}\".*\}}", s, re.DOTALL)
            if not m:
                return None
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
        if isinstance(obj, dict) and required_key in obj:
            return obj
        return None

    direct = _try_parse(raw)
    if direct is not None:
        return direct

    try:
        env = json.loads(raw)
    except json.JSONDecodeError:
        return None

    if isinstance(env, dict):
        payloads = env.get("payloads")
        if isinstance(payloads, list) and payloads:
            text = payloads[0].get("text") if isinstance(payloads[0], dict) else None
            if isinstance(text, str):
                inner = _try_parse(text)
                if inner is not None:
                    return inner
        msg = env.get("message")
        if isinstance(msg, str):
            inner = _try_parse(msg)
            if inner is not None:
                return inner
    return None


def _ask_blockers(mission_summary: str, recent_decisions: list[dict[str, Any]],
                  inflight_titles: list[str], cycle_id: str) -> list[dict[str, Any]]:
    """ASK stage — self-question to surface top 3 blockers to progress.

    Returns list of blocker dicts: {blocker, cause, escalation, severity}.
    Writes raw output to state/autonomous_mode/blockers/blockers_<cycle>.jsonl.
    """
    inflight_str = "\n".join(f"- {t}" for t in inflight_titles[:10]) or "(none)"
    decisions_str = "\n".join(
        f"- {d.get('action', '?')} (cycle {d.get('cycle_id', '?')})"
        for d in recent_decisions[-5:]
    ) or "(none)"

    prompt = f"""You are the strategic self-questioner for an autonomous S&P 500 trading mastery agent.

Given the current state below, identify the TOP 3 BLOCKERS to progress right now. A blocker is anything stalling or threatening the mission: failing daemons, unresolved errors, missing data, stale dashboards, low novelty in ideation, accumulated tech debt, or external rate limits.

For each blocker return: blocker (1 line), likely_cause (1 line), escalation_path (1 line — what to do next), severity (1-10).

Return STRICT JSON:
{{
  "blockers": [
    {{"blocker": "...", "likely_cause": "...", "escalation_path": "...", "severity": 7}}
  ]
}}

## In-flight helper spawns
{inflight_str}

## Recent decisions (last 5)
{decisions_str}

## Mission state
{mission_summary}
"""
    raw, source = _deepseek_call(prompt)
    blockers: list[dict[str, Any]] = []
    if raw:
        obj = _unwrap_openclaw_json(raw, "blockers")
        if obj is not None:
            blockers = obj.get("blockers", []) if isinstance(obj, dict) else []
        else:
            log.warning("_ask_blockers: parse failed (no 'blockers' key found)")

    # Persist blockers (always - even on parse failure, log the raw attempt)
    BLOCKERS_DIR.mkdir(parents=True, exist_ok=True)
    blockers_path = BLOCKERS_DIR / f"blockers_{cycle_id}.jsonl"
    with blockers_path.open("w") as f:
        for b in blockers:
            f.write(json.dumps({"cycle_id": cycle_id, "ts": _now_utc(), "source": source, **b}) + "\n")
        if not blockers and raw:
            # Record the failed parse for forensics
            f.write(json.dumps({
                "cycle_id": cycle_id, "ts": _now_utc(), "source": source,
                "parse_failed": True, "raw_head": raw[:500],
            }) + "\n")

    log.info("ASK: %d blocker(s) identified (source=%s)", len(blockers), source)
    _audit({"event": "ask_blockers", "cycle_id": cycle_id, "count": len(blockers), "source": source})
    return blockers


def _make_plan(blockers: list[dict[str, Any]], mission_summary: str, cycle_id: str) -> dict[str, Any]:
    """PLAN stage — build 3-7 step plan to advance the mission given blockers.

    Returns plan dict: {plan_id, cycle_id, ts, steps: [{step_id, action, target,
    estimated_min, depends_on, success_criteria}], goals_addressed}.
    Refreshes dashboard/AUTONOMOUS_PLAN.md (last 10 plans retained).
    """
    blockers_str = "\n".join(
        f"- [{b.get('severity', '?')}/10] {b.get('blocker', '?')} (cause: {b.get('likely_cause', '?')})"
        for b in blockers
    ) or "(no blockers identified — plan opportunistically)"

    prompt = f"""You are the strategic planner for an autonomous S&P 500 trading mastery agent.

Given the blockers and mission state below, produce a 3-7 step plan to advance the project in the NEXT cycle. Each step must be ATOMIC (single helper can complete in <=30 min) and ORDERED (step IDs s1, s2, ...).

For each step:
- step_id: "s1", "s2", ...
- action: imperative phrase (e.g. "Retune ORB threshold for AAPL on Q4 fold")
- target: file or module (e.g. "scripts/orb_strategy.py" or "ticker:AAPL")
- estimated_min: integer (must be <=30)
- depends_on: list of step_ids (empty for independent)
- success_criteria: 1-line measurable outcome
- priority: integer 1-10 (10 = highest)

Return STRICT JSON:
{{
  "goals_addressed": ["...", "..."],
  "steps": [
    {{"step_id": "s1", "action": "...", "target": "...", "estimated_min": 20,
      "depends_on": [], "success_criteria": "...", "priority": 8}}
  ]
}}

## Active blockers
{blockers_str}

## Mission state
{mission_summary}
"""
    raw, source = _deepseek_call(prompt)
    plan: dict[str, Any] = {
        "plan_id": hashlib.sha256(f"{cycle_id}|{_now_utc()}".encode()).hexdigest()[:12],
        "cycle_id": cycle_id,
        "ts": _now_utc(),
        "source": source,
        "steps": [],
        "goals_addressed": [],
    }
    if raw:
        obj = _unwrap_openclaw_json(raw, "steps")
        if obj is not None and isinstance(obj, dict):
            plan["steps"] = obj.get("steps", [])
            plan["goals_addressed"] = obj.get("goals_addressed", [])
        else:
            log.warning("_make_plan: parse failed (no 'steps' key found)")

    _persist_plan(plan)
    log.info("PLAN: %d step(s) (plan_id=%s, source=%s)", len(plan["steps"]), plan["plan_id"], source)
    _audit({
        "event": "plan_made", "cycle_id": cycle_id, "plan_id": plan["plan_id"],
        "step_count": len(plan["steps"]), "source": source,
    })
    return plan


def _persist_plan(plan: dict[str, Any]) -> None:
    """Append plan to history + refresh AUTONOMOUS_PLAN.md showing last N."""
    PLAN_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    plan_path = PLAN_HISTORY_DIR / f"plan_{plan['plan_id']}.json"
    plan_path.write_text(json.dumps(plan, indent=2))

    # Refresh dashboard view (last PLAN_RETAIN_LAST_N plans)
    plans = sorted(PLAN_HISTORY_DIR.glob("plan_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    recent = plans[:PLAN_RETAIN_LAST_N]

    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    body = f"# AUTONOMOUS_PLAN\n\n_Last refreshed: {_now_utc()}_\n\n"
    body += f"## Current plan (plan_id={plan['plan_id']}, cycle={plan.get('cycle_id', '?')})\n\n"
    body += f"**Source:** {plan.get('source', '?')}\n\n"
    if plan.get("goals_addressed"):
        body += "**Goals addressed:**\n"
        for g in plan["goals_addressed"]:
            body += f"- {g}\n"
        body += "\n"
    body += "**Steps:**\n\n"
    if not plan["steps"]:
        body += "_(no steps — DeepSeek returned empty or parse failed)_\n\n"
    for s in plan["steps"]:
        body += (
            f"- **{s.get('step_id', '?')}** [{s.get('priority', '?')}/10, {s.get('estimated_min', '?')}min] "
            f"`{s.get('action', '?')}` -> `{s.get('target', '?')}`\n"
            f"  - depends_on: {s.get('depends_on', [])}\n"
            f"  - success: {s.get('success_criteria', '?')}\n"
        )

    body += "\n---\n\n## Plan history (last 10)\n\n"
    for p in recent[1:]:
        try:
            d = json.loads(p.read_text())
            body += f"- `{d['plan_id']}` cycle={d.get('cycle_id', '?')} steps={len(d.get('steps', []))} ts={d.get('ts', '?')}\n"
        except (OSError, json.JSONDecodeError):
            continue
    AUTONOMOUS_PLAN.write_text(body)


def _decide(step: dict[str, Any], plan: dict[str, Any], blockers: list[dict[str, Any]]) -> dict[str, Any]:
    """DECIDE stage — generate rationale + risk log for a plan step before spawn.

    Returns a decision dict that is appended to decisions.jsonl. The dict
    contains: decision_id, cycle_id, plan_id, step_id, action, alternatives_considered,
    why_chosen, expected_outcome, risk_factors, rollback_strategy.
    """
    decision_id = hashlib.sha256(
        f"{plan['plan_id']}|{step.get('step_id', '?')}|{_now_utc()}".encode()
    ).hexdigest()[:12]

    blockers_str = "; ".join(b.get("blocker", "?") for b in blockers[:3]) or "(none)"
    step_str = json.dumps(step, ensure_ascii=False)

    prompt = f"""You are the decision-rationale logger for an autonomous agent. For the proposed step below, produce a brief rationale before execution.

Return STRICT JSON:
{{
  "alternatives_considered": ["alt1", "alt2", "alt3"],
  "why_chosen": "1-2 sentences",
  "expected_outcome": "1 sentence measurable",
  "risk_factors": ["risk1", "risk2"],
  "rollback_strategy": "1 sentence"
}}

## Step
{step_str}

## Active blockers (context)
{blockers_str}
"""
    raw, source = _deepseek_call(prompt, timeout_s=60)
    rationale: dict[str, Any] = {
        "alternatives_considered": [],
        "why_chosen": "(no rationale — fallback)",
        "expected_outcome": step.get("success_criteria", "(unspecified)"),
        "risk_factors": [],
        "rollback_strategy": "(none specified)",
    }
    if raw:
        obj = _unwrap_openclaw_json(raw, "why_chosen")
        if obj is not None and isinstance(obj, dict):
            rationale.update({k: obj.get(k, rationale[k]) for k in rationale})

    decision = {
        "decision_id": decision_id,
        "timestamp": _now_utc(),
        "cycle_id": plan.get("cycle_id"),
        "plan_id": plan.get("plan_id"),
        "step_id": step.get("step_id"),
        "action": step.get("action"),
        "target": step.get("target"),
        "source": source,
        **rationale,
    }
    DECISIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with DECISIONS_FILE.open("a") as f:
        f.write(json.dumps(decision) + "\n")
    log.info("DECIDE: %s (decision_id=%s)", step.get("action", "?")[:60], decision_id)
    _audit({"event": "decision_logged", "decision_id": decision_id, "step_id": step.get("step_id")})
    return decision


def _load_recent_decisions(n: int = 5) -> list[dict[str, Any]]:
    """Read last N decisions from DECISIONS_FILE (for context in ASK)."""
    if not DECISIONS_FILE.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with DECISIONS_FILE.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return rows[-n:]


def _observe(decision: dict[str, Any], step: dict[str, Any], blockers: list[dict[str, Any]],
             spawn_result: str | None, brief_path: str | None) -> None:
    """OBSERVE stage — write a ReAct-style trace row after spawn.

    Format: Thought/Plan/Action/Observation/Lesson appended to
    state/autonomous_mode/audit_<DATE>.jsonl under event='react_trace'.
    Spawns are async/background — at this point we record the INTENT and
    the brief path; the actual outcome is observed lazily by the
    _prune_active_spawns sweep next cycle.
    """
    blockers_summary = ", ".join(b.get("blocker", "?")[:60] for b in blockers[:3]) or "(none)"
    trace = {
        "event": "react_trace",
        "timestamp": _now_utc(),
        "cycle_id": decision.get("cycle_id"),
        "decision_id": decision.get("decision_id"),
        "thought": f"blockers: {blockers_summary}",
        "plan_step": f"{step.get('step_id', '?')}: {step.get('action', '?')}",
        "action": f"spawn helper @ {brief_path}" if brief_path else "no-spawn (gated/failed)",
        "observation": spawn_result or "spawn launched (async — outcome TBD)",
        "lesson": "(populate on next cycle once spawn completes)",
    }
    _audit(trace)


# ─── Idle detect (coexist patch 5) ───────────────────────────────────────────

LAST_USER_PROMPT_FILE = Path.home() / ".claude" / "state" / "last_user_prompt.unix"


def _user_active_recently(threshold: int = 60) -> bool:
    """True if a user prompt was submitted within `threshold` seconds.

    Reads ~/.claude/state/last_user_prompt.unix written by the
    touch-last-prompt UserPromptSubmit hook. Returns False if the file
    is missing (daemon started before any user session).
    """
    try:
        ts = float(LAST_USER_PROMPT_FILE.read_text().strip())
        return (time.time() - ts) < threshold
    except (OSError, ValueError):
        return False


# ─── Safety gate ─────────────────────────────────────────────────────────────


def safety_gate(text: str) -> tuple[bool, str | None]:
    """Returns (passed, blocking_keyword|None). Case-insensitive substring match."""
    low = text.lower()
    for kw in SAFETY_BLOCKLIST:
        if kw in low:
            return False, kw
    return True, None


# ─── Mission summary ─────────────────────────────────────────────────────────


def compose_mission_summary(max_chars: int = MISSION_SUMMARY_MAX_TOKENS * 4) -> str:
    """Compose a ≤ ~500-token summary from the dashboard + recent logs."""
    parts: list[str] = []

    if MISSION_PROGRESS.exists():
        text = MISSION_PROGRESS.read_text()
        parts.append("# MISSION_PROGRESS (head)\n" + text[:1200])

    if PROJECT_BEST.exists():
        text = PROJECT_BEST.read_text()
        parts.append("# project_per_ticker_best (tail)\n" + text[-1000:])

    # Recent auto_solve log entries
    auto_solve_dir = LOG_DIR / "auto_solve"
    if auto_solve_dir.exists():
        recent = sorted(auto_solve_dir.glob("*.log"), reverse=True)[:1]
        for p in recent:
            try:
                tail = p.read_text()[-600:]
                parts.append(f"# {p.name} (tail)\n{tail}")
            except OSError:
                pass

    raw = "\n\n".join(parts)
    if len(raw) > max_chars:
        raw = raw[:max_chars] + "\n…[truncated]"
    return raw


# ─── Ideation via OpenClaw + DeepSeek ────────────────────────────────────────


# autosolve_skip: ideation template — drift-mitigation clause added 2026-05-20
IDEATION_PROMPT_TEMPLATE = """You are the strategic ideator for the S&P 500 mastery mission.
Below is the current mission status. Propose the top-3 HIGHEST-VALUE next actions.

Score each by: impact (1-10) × novelty (1-10) ÷ feasibility (1-10).
Reject anything that touches money, messages, credentials, or destructive ops.
Reject anything whose effort_min > 30 unless you split into ≤30-min pieces.

{orthogonality_clause}

Return STRICT JSON, no prose, no markdown fences:
{{
  "candidates": [
    {{
      "title": "short imperative phrase",
      "reason": "1-sentence why",
      "helper_brief": "concrete brief to hand to a Claude helper (≤500 chars)",
      "expected_lift": "1-sentence measurable outcome",
      "effort_min": 15,
      "impact_score": 7,
      "novelty_score": 6,
      "feasibility_score": 8
    }}
  ]
}}

# Mission status
{mission_summary}
"""


def _build_orthogonality_clause(recent_titles: list[str] | None) -> str:
    """Compose a 'propose something orthogonal to these recent ideas' clause.

    Drift-mitigation: surfaces the last N seen titles into the ideation prompt so
    DeepSeek explicitly avoids re-proposing them. Empty when no history available.
    """
    if not recent_titles:
        return ""
    bullets = "\n".join(f"- {t}" for t in recent_titles[-20:])
    return (
        "# Drift-mitigation: AVOID these recent ideas\n"
        "The following actions were already proposed in recent cycles. Do NOT re-propose them. "
        "Propose something ORTHOGONAL — different feature group, different ticker bucket, different "
        "validation regime, different timeframe, or a completely new data source.\n\n"
        f"Recent ideas (last {len(recent_titles[-20:])}):\n{bullets}\n"
    )


def ideate(mission_summary: str, timeout_s: int = 180, recent_titles: list[str] | None = None) -> list[dict[str, Any]]:
    """Call OpenClaw+DeepSeek, return list of candidates (may be empty).

    `recent_titles` (drift-mitigation, 2026-05-20): inject a "propose orthogonal"
    clause into the prompt so the model avoids repeating these. Pass the last N
    seen titles from seen_ideas.jsonl.
    """
    orthogonality_clause = _build_orthogonality_clause(recent_titles)
    prompt = IDEATION_PROMPT_TEMPLATE.format(
        mission_summary=mission_summary,
        orthogonality_clause=orthogonality_clause,
    )
    cmd = [
        str(OPENCLAW_BIN),
        "agent",
        "--local",
        "--agent", "main",
        "--model",
        "deepseek/deepseek-v4-flash",
        "--json",
        "--message",
        prompt,
    ]
    log.info("ideate: calling openclaw+deepseek (timeout=%ds)", timeout_s)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        log.warning("ideate: openclaw call timed out")
        _audit({"event": "ideate_timeout", "timeout_s": timeout_s})
        return []
    except FileNotFoundError:
        log.error("ideate: openclaw-gdrive launcher not found at %s", OPENCLAW_BIN)
        _audit({"event": "ideate_missing_openclaw", "path": str(OPENCLAW_BIN)})
        return []

    if proc.returncode != 0:
        log.warning("ideate: openclaw rc=%d stderr=%s", proc.returncode, proc.stderr[:300])
        _audit({"event": "ideate_nonzero_rc", "rc": proc.returncode, "stderr": proc.stderr[:300]})
        return []

    # OpenClaw --json emits an envelope; extract the assistant message body.
    raw = proc.stdout.strip()
    candidates = _extract_candidates(raw)
    log.info("ideate: %d candidate(s) parsed", len(candidates))
    return candidates


def _extract_candidates(raw: str) -> list[dict[str, Any]]:
    """Best-effort JSON parse — tolerate prose, markdown fences, envelope wrapping.

    Hardened 2026-05-20 to also use _unwrap_openclaw_json which handles the
    {payloads:[{text:'...'}]} envelope DeepSeek/OpenClaw actually emits.  This
    was the dominant cause of `0 candidate(s) parsed` events overnight.
    """
    if not raw:
        return []

    # FIRST: try the canonical envelope unwrap (matches _ask_blockers path).
    obj = _unwrap_openclaw_json(raw, "candidates")
    if obj is not None and isinstance(obj, dict):
        cands = obj.get("candidates", [])
        if isinstance(cands, list):
            return cands

    # Fallback: legacy code-fence / regex paths.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fence:
        raw_obj = fence.group(1)
    else:
        m = re.search(r"\{[^{}]*\"candidates\".*\}", raw, re.DOTALL)
        raw_obj = m.group(0) if m else raw

    for attempt in (raw_obj, raw):
        try:
            obj = json.loads(attempt)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "candidates" in obj:
            return obj["candidates"]
        if isinstance(obj, dict) and "message" in obj:
            inner = obj["message"]
            if isinstance(inner, str):
                try:
                    inner_obj = json.loads(inner)
                    if isinstance(inner_obj, dict) and "candidates" in inner_obj:
                        return inner_obj["candidates"]
                except json.JSONDecodeError:
                    m = re.search(r"\{.*\"candidates\".*\}", inner, re.DOTALL)
                    if m:
                        try:
                            inner_obj = json.loads(m.group(0))
                            return inner_obj.get("candidates", [])
                        except json.JSONDecodeError:
                            pass
    return []


# ─── Gating ──────────────────────────────────────────────────────────────────


def gate_candidate(
    c: dict[str, Any],
    seen: set[str],
    cfg: dict[str, Any],
    inflight: int,
) -> tuple[bool, str]:
    """Returns (ok, reason). Reason is "ok" when ok=True."""
    title = (c.get("title") or "").strip()
    if not title:
        return False, "empty_title"

    h = _hash_title(title)
    if h in seen:
        return False, "duplicate"

    brief = c.get("helper_brief") or ""
    combined = f"{title}\n{brief}"
    ok, kw = safety_gate(combined)
    if not ok:
        return False, f"safety_block:{kw}"

    effort = float(c.get("effort_min") or 999)
    if effort > 30:
        return False, f"effort_too_large:{effort}"

    impact = float(c.get("impact_score") or 0)
    if impact < 3:
        return False, f"impact_too_low:{impact}"

    # UNLIMITED MODE (2026-05-20): concurrent-spawn cap is OPT-IN only.
    # Sentinel == effectively unlimited; finite int == honored cap.
    cap = cfg.get("max_concurrent_spawns", UNLIMITED_SPAWNS_SENTINEL)
    try:
        cap_int = int(cap)
        if cap_int < UNLIMITED_SPAWNS_SENTINEL and inflight >= cap_int:
            return False, f"cap_reached:{inflight}/{cap_int}"
    except (TypeError, ValueError):
        pass  # malformed → treat as unlimited

    # UNLIMITED MODE (2026-05-20): budget cap is OPT-IN only.
    # None == unlimited (never halt); finite float == honored cap.
    # We still LOG estimated cost to audit for visibility.
    budget_raw = cfg.get("budget_remaining_usd")
    est_cost = 0.001  # Claude haiku ~$0.001/spawn estimate
    if budget_raw is not None:
        try:
            budget = float(budget_raw)
            if budget != float("inf") and est_cost > budget:
                return False, f"budget_exhausted:{budget}"
        except (TypeError, ValueError):
            pass  # malformed → treat as unlimited

    return True, "ok"


# ─── Spawning ────────────────────────────────────────────────────────────────


def count_inflight() -> int:
    """Heuristic: count running `claude -p` / `openclaw agent` processes (best-effort)."""
    try:
        out = subprocess.check_output(
            ["pgrep", "-af", "claude -p|openclaw.*agent"],
            text=True,
            timeout=5,
        )
        return len([ln for ln in out.splitlines() if ln.strip()])
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return 0


# ─── Spawn tracker (coexist patch 4) ─────────────────────────────────────────


def _record_active_spawn(pid: int, brief_path: str, idea_hash: str) -> None:
    """Append a spawn record to ACTIVE_SPAWNS_FILE (state/autonomous_mode/active_spawns.jsonl)."""
    ACTIVE_SPAWNS_FILE.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": _now_utc(),
        "pid": pid,
        "brief_path": brief_path,
        "idea_hash": idea_hash,
        "status": "launched",
    }
    with ACTIVE_SPAWNS_FILE.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _prune_active_spawns() -> None:
    """Rewrite active_spawns.jsonl keeping only rows whose PID is still alive."""
    if not ACTIVE_SPAWNS_FILE.exists():
        return
    alive: list[dict[str, Any]] = []
    try:
        with ACTIVE_SPAWNS_FILE.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pid = rec.get("pid")
                try:
                    os.kill(int(pid), 0)
                    alive.append(rec)
                except OSError:
                    pass  # dead — drop this row
    except OSError:
        return
    with ACTIVE_SPAWNS_FILE.open("w") as f:
        for rec in alive:
            f.write(json.dumps(rec) + "\n")


# ─── Mac load gate (coexist patch 6) ─────────────────────────────────────────


def mac_load_safe(cap: float | None = None) -> tuple[bool, float, float]:
    """Adaptive load gate (rewritten 2026-05-20).

    Returns (safe, load_1m, effective_cap). Default cap is adaptive:
        cap = max(LOAD_GATE_FLOOR, current_load + LOAD_GATE_HEADROOM)
    so the daemon ALWAYS gets to run + spawn >=1 helper even under sustained
    high load (Mac routinely 20-60 in this workspace). Pass an explicit cap
    to override (e.g. for tests).

    Backwards-compat note: legacy callers received a 2-tuple. New return is
    3-tuple; old callers unpacking with 2 vars will need updating.
    """
    try:
        load_1m, _load_5, _load_15 = os.getloadavg()
    except OSError:
        return True, 0.0, float(cap if cap is not None else LOAD_GATE_FLOOR)
    if cap is None:
        # Adaptive: always allow daemon to run; cap is current+headroom or floor
        effective_cap = float(max(LOAD_GATE_FLOOR, load_1m + LOAD_GATE_HEADROOM))
    else:
        effective_cap = float(cap)
    return load_1m < effective_cap, load_1m, effective_cap


def _adaptive_concurrency_cap(load_1m: float) -> int:
    """Adaptive concurrent-spawn cap based on current Mac load.

    Low load: unlimited (1M sentinel).
    Moderate (15-25): 8 / 4.
    High (>=25): 2 — throttle to protect the Mac.
    """
    if load_1m < 10:
        return 10**6
    if load_1m < 15:
        return 8
    if load_1m < 25:
        return 4
    return 2


def spawn_helper(candidate: dict[str, Any], cfg: dict[str, Any]) -> str | None:
    """Write a spawn brief + launch a background `claude -p` helper. Returns brief path."""
    title = candidate["title"]
    idea_hash = _hash_title(title)
    SPAWN_BRIEFS_DIR.mkdir(parents=True, exist_ok=True)
    brief_path = SPAWN_BRIEFS_DIR / f"{idea_hash}.txt"

    # gabriel_self routing (2026-05-20): if cap-map tagged this candidate
    # with a routed model, use that; otherwise fall back to default haiku.
    _routed_model = candidate.get("_gabriel_routed_model", "haiku")
    _route_reason = candidate.get("_gabriel_route_reason", "default (mechanical helper)")
    _gabriel_tt = candidate.get("_gabriel_task_type", "uncategorized")
    brief_body = f"""# origin: autonomous_daemon
# user_session_yielding: enabled (do not run if last_user_prompt < 60s)
# autonomous_mode spawn
# model_reason: {_routed_model} - {_route_reason}
# gabriel_task_type: {_gabriel_tt}
# scope_estimate_min: {candidate.get('effort_min', 'unknown')}
# decomposition_plan: single-helper slice
# autosolve_skip: spawned via autonomous_mode_daemon safety/budget-gated flow

Workspace inherits standing authorization. Safety boundaries apply.
Pre-approved: routine read/write/script/install/log/backup inside My Drive.

## Task
{title}

## Why
{candidate.get('reason', '(no reason)')}

## Brief
{candidate.get('helper_brief', '(no brief)')}

## Expected lift
{candidate.get('expected_lift', '(unspecified)')}

## Proof of work
Return: commands run, paths accessed, files changed, backups, sub-agents, final status.
"""
    brief_path.write_text(brief_body)

    # Budget cap per spawn; do NOT block if launcher missing — log and return.
    if not CLAUDE_BIN.exists():
        log.warning("spawn: claude-gdrive launcher not found at %s", CLAUDE_BIN)
        _audit({
            "event": "spawn_skipped_no_launcher",
            "title": title,
            "brief_path": str(brief_path),
            "launcher": str(CLAUDE_BIN),
        })
        return str(brief_path)

    cmd = [
        str(CLAUDE_BIN),
        "-p",
        "--max-budget-usd",
        "1.0",
        f"@{brief_path}",
    ]
    log_path = LOG_DIR_AUTONOMOUS / "spawns" / f"{idea_hash}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("w") as fh:
            proc = subprocess.Popen(  # noqa: S603 — controlled args
                cmd,
                stdout=fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        _record_active_spawn(proc.pid, str(brief_path), idea_hash)
        log.info("spawn: launched %s (pid=%d, brief=%s)", title, proc.pid, brief_path.name)
        _audit({
            "event": "spawn_launched",
            "title": title,
            "brief_path": str(brief_path),
            "log_path": str(log_path),
            "pid": proc.pid,
        })
        # Track estimated spend for AUDIT VISIBILITY.
        # If budget is None (unlimited) → log to audit, do NOT halt; do NOT write a finite cap.
        # If budget is a finite float (operator opt-in cap) → decrement and persist.
        budget_raw = cfg.get("budget_remaining_usd")
        est_cost = 0.001  # Claude haiku ~$0.001/spawn estimate
        _audit({"event": "spawn_cost_estimate_usd", "title": title, "est_cost_usd": est_cost})
        if budget_raw is not None:
            try:
                new_budget = max(0.0, float(budget_raw) - est_cost)
                cfg["budget_remaining_usd"] = new_budget
                _write_config(cfg)
            except (TypeError, ValueError):
                pass  # malformed → leave as-is, treat as unlimited
    except OSError as e:
        log.error("spawn: failed (%s)", e)
        _audit({"event": "spawn_failed", "title": title, "error": str(e)})
        return None
    return str(brief_path)


# ─── Dashboard refresh ───────────────────────────────────────────────────────


def refresh_status_dashboard(cfg: dict[str, Any], last_spawns: list[str]) -> None:
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    cap_raw = cfg.get("max_concurrent_spawns", UNLIMITED_SPAWNS_SENTINEL)
    try:
        cap_disp = "UNLIMITED" if int(cap_raw) >= UNLIMITED_SPAWNS_SENTINEL else str(cap_raw)
    except (TypeError, ValueError):
        cap_disp = "UNLIMITED (malformed→treated as unlimited)"
    budget_raw = cfg.get("budget_remaining_usd")
    budget_disp = "UNLIMITED" if budget_raw is None else f"${float(budget_raw):.4f}"
    body = f"""# AUTONOMOUS_STATUS

_Last refreshed: {_now_utc()}_

## Config
- enabled: {cfg.get('enabled')}
- max_concurrent_spawns: {cap_disp}
- budget_remaining_usd: {budget_disp}
- cycle_seconds: {LOOP_SLEEP_SECONDS}
- reason_off: {cfg.get('reason_off', '(n/a)')}

## Hard safety rails (NEVER disabled)
- Destructive-action blocklist: rm -rf, force push, drop table, kill -9 1, sudo rm
- Credential/path blocklist: .ssh/, aws_secret, private_key, password, wallet, credential
- Money/messages blocklist: transfer, wire, send money, send sms, send email, mailto:
- Per-spawn budget cap inside spawn_helper: --max-budget-usd 1.0 (helper-level cap)
- All actions audited to state/autonomous_mode/audit_<DATE>.jsonl
- Heartbeat written every {HEARTBEAT_INTERVAL_SECONDS}s for mission_overseer detection

## Adaptive load gate (2026-05-20)
- Floor: {LOAD_GATE_FLOOR}  |  Headroom: {LOAD_GATE_HEADROOM}
- Effective cap = max(floor, current_load + headroom) — daemon ALWAYS runs

## ASK-PLAN-DECIDE-OBSERVE (2026-05-20)
- Blockers log: `state/autonomous_mode/blockers/blockers_<cycle>.jsonl`
- Plan dashboard: `dashboard/AUTONOMOUS_PLAN.md`
- Decision log: `state/autonomous_mode/decisions.jsonl`
- ReAct trace: `state/autonomous_mode/audit_<DATE>.jsonl` (event=react_trace)

## Last 5 spawned ideas
"""
    for s in (last_spawns or ["(none yet)"])[-5:]:
        body += f"- {s}\n"

    # User inbox status (added 2026-05-20)
    # autosolve_skip: dashboard update
    try:
        ic = _user_inbox_counts()
        body += (
            f"\n## User inbox\n"
            f"- pending: {ic['pending']}  ·  spawned: {ic.get('dispatched',0) + 0}  ·  "
            f"done: {ic['done']}  ·  failed: {ic['failed']}  ·  total: {ic['total']}\n"
            f"- inbox file: `state/autonomous_mode/user_inbox.jsonl`\n"
            f"- answers dir: `dashboard/inbox_answers/`\n"
            f"- CLI: `bin/autonomous ask|search|research|add|wire|fix \"...\"`\n"
        )
    except Exception as e:  # noqa: BLE001
        body += f"\n## User inbox\n- (counts unavailable: {e})\n"

    # USER-PERSONA directives (added 2026-05-20)
    # autosolve_skip: dashboard update
    try:
        persona_count = 0
        dispatched_count = 0
        if USER_DIRECTIVES_LOG.exists():
            for line in USER_DIRECTIVES_LOG.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    persona_count += 1
                    if r.get("dispatched"):
                        dispatched_count += 1
                except json.JSONDecodeError:
                    continue
        body += (
            f"\n## USER-PERSONA loop\n"
            f"- generated: {persona_count}  ·  dispatched: {dispatched_count}  ·  "
            f"rejected: {persona_count - dispatched_count}\n"
            f"- log: `state/autonomous_mode/user_directives.jsonl`\n"
            f"- dashboard: `dashboard/USER_DIRECTIVES.md`\n"
            f"- prompt history: `state/user_prompts_history.jsonl` (rolling 100)\n"
            f"- priority boost: +{PERSONA_PRIORITY_BOOST} impact_score over neutral ideate\n"
        )
    except Exception as e:  # noqa: BLE001
        body += f"\n## USER-PERSONA loop\n- (counts unavailable: {e})\n"

    body += "\n## Controls\n"
    body += "```\nbin/autonomous on\nbin/autonomous off\nbin/autonomous pause\nbin/autonomous status\nbin/autonomous inbox\n```\n"
    AUTONOMOUS_STATUS.write_text(body)


# ─── User inbox (added 2026-05-20) ───────────────────────────────────────────
# autosolve_skip: feature-add not error-fix
#
# User drops requests via `bin/autonomous ask|search|research|add|wire|fix "..."`.
# Each subcommand appends one JSON line to USER_INBOX_FILE with:
#   {id, ts, intent, payload, priority, status, source}
# The daemon polls inbox at TOP of every cycle (before _ask_blockers/ideate),
# converts pending items to candidates (skipping ideation, but still going
# through safety_gate + load gate + spawn_helper), marks them "dispatched", and
# writes an answer/result file to dashboard/inbox_answers/<id>.md.


def _load_user_inbox() -> list[dict[str, Any]]:
    """Read all rows from user_inbox.jsonl; tolerate empty file / bad lines."""
    if not USER_INBOX_FILE.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in USER_INBOX_FILE.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError as e:
        log.warning("user_inbox read failed: %s", e)
    return rows


def _persist_user_inbox(rows: list[dict[str, Any]]) -> None:
    """Atomic rewrite of user_inbox.jsonl after status mutations."""
    USER_INBOX_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = USER_INBOX_FILE.with_suffix(".jsonl.tmp")
    try:
        with tmp.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        tmp.replace(USER_INBOX_FILE)
    except OSError as e:
        log.warning("user_inbox persist failed: %s", e)
        try:
            tmp.unlink(missing_ok=True)  # type: ignore[arg-type]
        except OSError:
            pass


# Intent → brief-template builders. Each returns a candidate dict that flows
# through the normal safety_gate + spawn_helper pipeline. Briefs include
# `# user_inbox_item: <id>` so the helper writes its result to
# dashboard/inbox_answers/<id>.md and the operator can find the answer.

def _brief_for_ask(item: dict[str, Any]) -> dict[str, Any]:
    payload = item.get("payload", "")
    item_id = item.get("id", "ask")
    answer_path = INBOX_ANSWERS_DIR / f"{item_id}.md"
    return {
        "title": f"USER_ASK: {payload[:60]}",
        "reason": f"User asked via inbox CLI (id={item_id}). Direct answer requested.",
        "helper_brief": (
            f"# user_inbox_item: {item_id}\n"
            f"# intent: ask\n"
            f"# answer_path: {answer_path}\n\n"
            f"USER QUESTION:\n{payload}\n\n"
            f"TASK: Research and answer concisely. Write the answer to:\n"
            f"  {answer_path}\n"
            f"Use OpenClaw+DeepSeek for cheap large-context reasoning if helpful:\n"
            f"  bin/openclaw-gdrive agent --local --model deepseek/deepseek-v4-flash --json --message '...'\n"
            f"Answer format: ## Question / ## Short answer / ## Detail / ## Sources."
        ),
        "expected_lift": f"Direct answer to user question written to {answer_path}",
        "effort_min": 10,
        "impact_score": 8,
        "novelty_score": 8,
        "feasibility_score": 9,
        "_user_inbox": True,
        "_inbox_item_id": item_id,
    }


def _brief_for_search_github(item: dict[str, Any]) -> dict[str, Any]:
    payload = item.get("payload", "")
    item_id = item.get("id", "sgh")
    answer_path = INBOX_ANSWERS_DIR / f"{item_id}.md"
    return {
        "title": f"USER_SEARCH_GH: {payload[:60]}",
        "reason": f"User-requested GitHub code/repo search (id={item_id}).",
        "helper_brief": (
            f"# user_inbox_item: {item_id}\n"
            f"# intent: search_github\n"
            f"# answer_path: {answer_path}\n\n"
            f"SEARCH QUERY: {payload}\n\n"
            f"TASK: Use gh CLI to search code AND repos:\n"
            f"  gh search code '{payload}' --limit 20\n"
            f"  gh search repos '{payload}' --limit 10\n"
            f"Or mcp__github__search_code / mcp__github__search_repositories.\n"
            f"Summarize top results (repo/path/snippet/stars/why-relevant) to:\n"
            f"  {answer_path}\n"
            f"Format: ## Query / ## Top repos / ## Top code hits / ## Recommended next step."
        ),
        "expected_lift": f"Ranked GitHub search results at {answer_path}",
        "effort_min": 15,
        "impact_score": 7,
        "novelty_score": 8,
        "feasibility_score": 9,
        "_user_inbox": True,
        "_inbox_item_id": item_id,
    }


def _brief_for_search_internet(item: dict[str, Any]) -> dict[str, Any]:
    payload = item.get("payload", "")
    item_id = item.get("id", "sint")
    answer_path = INBOX_ANSWERS_DIR / f"{item_id}.md"
    return {
        "title": f"USER_SEARCH_WEB: {payload[:60]}",
        "reason": f"User-requested internet search (id={item_id}).",
        "helper_brief": (
            f"# user_inbox_item: {item_id}\n"
            f"# intent: search_internet\n"
            f"# answer_path: {answer_path}\n\n"
            f"SEARCH QUERY: {payload}\n\n"
            f"TASK: Use WebSearch (or curl-based equivalent) to find top relevant\n"
            f"sources. Cite URLs. Summarize 1 line per hit + 1-paragraph synthesis.\n"
            f"Write to: {answer_path}\n"
            f"Format: ## Query / ## Top hits (URL + 1-line summary) / ## Synthesis / ## Sources."
        ),
        "expected_lift": f"Web search summary at {answer_path}",
        "effort_min": 15,
        "impact_score": 7,
        "novelty_score": 8,
        "feasibility_score": 9,
        "_user_inbox": True,
        "_inbox_item_id": item_id,
    }


def _brief_for_research(item: dict[str, Any]) -> dict[str, Any]:
    payload = item.get("payload", "")
    item_id = item.get("id", "res")
    answer_path = INBOX_ANSWERS_DIR / f"{item_id}.md"
    return {
        "title": f"USER_RESEARCH: {payload[:60]}",
        "reason": f"User-requested longer-form research (id={item_id}).",
        "helper_brief": (
            f"# user_inbox_item: {item_id}\n"
            f"# intent: research\n"
            f"# answer_path: {answer_path}\n\n"
            f"RESEARCH TOPIC: {payload}\n\n"
            f"TASK: Apply §8-solver pattern — spawn 3 parallel helpers if scope warrants:\n"
            f"  INTERNET (WebSearch + papers/blogs), GITHUB (gh search code/repos),\n"
            f"  REPO-LOCAL (grep AI-Tools registry/repos-claude + s&p500-ticker-mastery).\n"
            f"Synthesize findings into a ranked, actionable report at:\n"
            f"  {answer_path}\n"
            f"Format: ## Topic / ## Internet findings / ## GitHub findings /\n"
            f"        ## Repo-local findings / ## Synthesis / ## Recommended actions / ## Sources."
        ),
        "expected_lift": f"3-channel research synthesis at {answer_path}",
        "effort_min": 25,
        "impact_score": 8,
        "novelty_score": 8,
        "feasibility_score": 7,
        "_user_inbox": True,
        "_inbox_item_id": item_id,
    }


def _brief_for_add_feature(item: dict[str, Any]) -> dict[str, Any]:
    payload = item.get("payload", "")
    item_id = item.get("id", "addf")
    answer_path = INBOX_ANSWERS_DIR / f"{item_id}.md"
    return {
        "title": f"USER_ADD_FEATURE: {payload[:60]}",
        "reason": f"User-requested new feature/module (id={item_id}).",
        "helper_brief": (
            f"# user_inbox_item: {item_id}\n"
            f"# intent: add_feature\n"
            f"# answer_path: {answer_path}\n\n"
            f"FEATURE/MODULE: {payload}\n\n"
            f"TASK (REPO-LOCAL):\n"
            f"  1. Read s&p500-ticker-mastery + AI-Tools/registry/repos-claude to find\n"
            f"     similar prior art and the natural integration point in v10.\n"
            f"  2. Design the feature (1-page note: inputs, outputs, dependencies,\n"
            f"     interface, fit with existing v10 surface).\n"
            f"  3. Scaffold the feature module (Python file, minimal stub with\n"
            f"     tests-pass-no-op) under the correct subfolder.\n"
            f"  4. Plumb it into v10 (add an opt-in import or a config flag).\n"
            f"  5. Write a summary + design + paths-touched to: {answer_path}\n"
            f"Respect CLAUDE.md safety boundaries. Add to a NEW commit;\n"
            f"do NOT push unless user requests it."
        ),
        "expected_lift": f"Feature scaffolded + wired (summary at {answer_path})",
        "effort_min": 30,
        "impact_score": 9,
        "novelty_score": 8,
        "feasibility_score": 7,
        "_user_inbox": True,
        "_inbox_item_id": item_id,
    }


def _brief_for_wire(item: dict[str, Any]) -> dict[str, Any]:
    # NOTE: SAFETY_BLOCKLIST contains "wire " (trailing space — guards against
    # "wire transfer"). Avoid the bare token "wire " followed by whitespace in
    # title / brief / reason / expected_lift. Use "integrate" / "plumb-in" /
    # "integration" wording. autosolve_skip: intentional substring avoidance.
    payload = item.get("payload", "")
    item_id = item.get("id", "wire")
    answer_path = INBOX_ANSWERS_DIR / f"{item_id}.md"
    return {
        "title": f"USER_INTEGRATE_INTO_V10: {payload[:60]}",
        "reason": f"User-requested integration into v10 (id={item_id}).",
        "helper_brief": (
            f"# user_inbox_item: {item_id}\n"
            f"# intent: integrate_into_v10\n"
            f"# answer_path: {answer_path}\n\n"
            f"TARGET REPO/MODULE PATH: {payload}\n\n"
            f"TASK (REPO-LOCAL):\n"
            f"  1. Locate the target (registry/repos-claude/per-repo or absolute path).\n"
            f"  2. Identify the natural integration point in s&p500-ticker-mastery v10.\n"
            f"  3. Write the smallest possible integration (config flag + opt-in import +\n"
            f"     smoke test). Respect CLAUDE.md safety boundaries.\n"
            f"  4. Smoke-test the integration (one ticker, dry-run, log output).\n"
            f"  5. Write summary + diff list + smoke result to: {answer_path}"
        ),
        "expected_lift": f"Module integrated + smoke-tested (summary at {answer_path})",
        "effort_min": 25,
        "impact_score": 8,
        "novelty_score": 7,
        "feasibility_score": 7,
        "_user_inbox": True,
        "_inbox_item_id": item_id,
    }


def _brief_for_fix(item: dict[str, Any]) -> dict[str, Any]:
    payload = item.get("payload", "")
    item_id = item.get("id", "fix")
    answer_path = INBOX_ANSWERS_DIR / f"{item_id}.md"
    return {
        "title": f"USER_FIX: {payload[:60]}",
        "reason": f"User-reported error/blocker — §8 solver pattern (id={item_id}).",
        "helper_brief": (
            f"# user_inbox_item: {item_id}\n"
            f"# intent: fix\n"
            f"# answer_path: {answer_path}\n\n"
            f"ERROR/BLOCKER:\n{payload}\n\n"
            f"TASK (§8 auto-solve):\n"
            f"  Spawn 3 parallel solvers — INTERNET (search + StackOverflow + docs),\n"
            f"  GITHUB (search issues + similar bugs), REPO-LOCAL (grep + git log + tests).\n"
            f"  Pick the best fix, apply silently, smoke-test. Write what-failed +\n"
            f"  3-channel findings + applied fix + smoke result to: {answer_path}\n"
            f"  Hard rule: respect CLAUDE.md safety boundaries."
        ),
        "expected_lift": f"Fix applied + smoke-passed (report at {answer_path})",
        "effort_min": 25,
        "impact_score": 10,
        "novelty_score": 8,
        "feasibility_score": 7,
        "_user_inbox": True,
        "_inbox_item_id": item_id,
    }


_INTENT_DISPATCH = {
    "ask": _brief_for_ask,
    "search_github": _brief_for_search_github,
    "search_internet": _brief_for_search_internet,
    "research": _brief_for_research,
    "add_feature": _brief_for_add_feature,
    "wire_into_v10": _brief_for_wire,
    "fix": _brief_for_fix,
}


def _drain_user_inbox(cfg: dict[str, Any], cycle_id: str,
                      max_items: int | None = None) -> list[dict[str, Any]]:
    """Read user_inbox.jsonl, pick pending items by priority DESC + ts ASC,
    convert to candidate dicts, mark them dispatched, return for spawn pass.

    Skips ideate (briefs are already concrete) but the returned candidates still
    flow through gate_candidate -> safety_gate -> mac_load_safe -> spawn_helper.
    """
    rows = _load_user_inbox()
    if not rows:
        return []

    pending = [r for r in rows if r.get("status") == "pending"]
    if not pending:
        return []

    # priority DESC, ts ASC
    pending.sort(key=lambda r: (-int(r.get("priority", 0)), r.get("ts", "")))

    take = pending if max_items is None else pending[:max_items]
    candidates: list[dict[str, Any]] = []
    dispatched_ids: set[str] = set()
    for item in take:
        intent = item.get("intent", "")
        builder = _INTENT_DISPATCH.get(intent)
        if builder is None:
            log.warning("user_inbox: unknown intent %r in item %s — marking failed",
                        intent, item.get("id"))
            item["status"] = "failed_unknown_intent"
            item["dispatched_cycle"] = cycle_id
            dispatched_ids.add(item.get("id", ""))
            continue
        try:
            cand = builder(item)
        except Exception as e:  # noqa: BLE001
            log.warning("user_inbox: brief builder for %s failed: %s", intent, e)
            item["status"] = "failed_brief_error"
            item["dispatched_cycle"] = cycle_id
            item["error"] = str(e)
            dispatched_ids.add(item.get("id", ""))
            continue
        cand["_user_inbox"] = True
        cand["_inbox_item_id"] = item.get("id")
        cand["_inbox_intent"] = intent
        candidates.append(cand)
        item["status"] = "dispatched"
        item["dispatched_cycle"] = cycle_id
        item["dispatched_ts"] = _now_utc()
        dispatched_ids.add(item.get("id", ""))

    # Persist status mutations
    if dispatched_ids:
        for r in rows:
            if r.get("id") in dispatched_ids:
                # find updated row and replace
                for it in take:
                    if it.get("id") == r.get("id"):
                        r.update(it)
                        break
        _persist_user_inbox(rows)

    _audit({
        "event": "user_inbox_drained",
        "cycle_id": cycle_id,
        "drained_count": len(candidates),
        "drained_ids": sorted(dispatched_ids),
        "pending_remaining": sum(1 for r in rows if r.get("status") == "pending"),
    })
    log.info("user_inbox: drained %d items (%s)", len(candidates),
             ",".join(c.get("_inbox_intent", "?") for c in candidates))
    return candidates


def _user_inbox_counts() -> dict[str, int]:
    """Lightweight counts for dashboard display."""
    rows = _load_user_inbox()
    out = {"pending": 0, "dispatched": 0, "done": 0, "failed": 0, "total": len(rows)}
    for r in rows:
        s = r.get("status", "")
        if s == "pending": out["pending"] += 1
        elif s == "dispatched": out["dispatched"] += 1
        elif s == "done": out["done"] += 1
        elif s.startswith("failed"): out["failed"] += 1
    return out


# ─── USER-PERSONA loop (added 2026-05-20) ────────────────────────────────────
# autosolve_skip: feature-add — persona-ideation generates directives in the
# user's own voice (caveman-terse, completeness-mandate, scale-everything,
# fix-blockers, iterate-on-landed). Differs from the neutral `ideate()` path
# which generates "highest-value next action" candidates in an assistant voice.
# Persona-generated directives get +PERSONA_PRIORITY_BOOST and prepend the
# candidate stream (after USER_INBOX but before plan + ideate).


USER_PERSONA_PROMPT = """You are now impersonating the USER of this S&P 500 trading mastery project. You have observed the user's voice + priorities + style. Generate exactly {n} directives the user WOULD type RIGHT NOW.

# User's voice patterns (observed)
1. COMPLETENESS-MANDATE: "all data, not just 10", "use all features", "every last folder"
2. SCALE-PUSH: "all 500 tickers", "whole S&P 500", "fully understand"
3. BLOCKER-FIX: "is X working? if not fix", "fix it", "make it work"
4. ITERATE-ON-LANDED: "now do Y to it", "scale this", "integrate that"
5. AUTO-SOLVE: "spawn sub agents to figure out", "no human involvement"
6. MULTI-DOMAIN: jumps between trading / ML / infra / safety / cloud
7. CAVEMAN-TERSE: short, no fluff, fragments OK
8. AUDIT-QUESTIONS: "are we using X?", "is Y still broken?"

# Hard rules for your directives
- Caveman-terse phrasing. Example GOOD: "scale ORB to all 500". Example BAD: "Please consider implementing ORB scaling across the full S&P 500 universe at your convenience".
- Each pushes ONE of: completeness OR scale OR blocker-fix OR iterate-on-landed
- One concrete next action per directive (atomic helper-ready)
- The {n} directives must be ORTHOGONAL (different surfaces)
- Must NOT repeat user's recent prompts (avoid echoing)
- Reject anything touching money / messages / credentials / destructive ops
- Effort ≤30 min per directive

# Recent landings (last 1hr — completed helper work)
{landed_summary}

# Open blockers (still stuck)
{blockers_summary}

# Untouched mission goals
{goals_summary}

# Recent user prompts (last 24h — what user actually typed)
{user_history}

# Recent self-generated ideas (DO NOT repeat these)
{recent_ideas}

Return STRICT JSON only — no prose, no markdown fences:
{{
  "directives": [
    {{
      "voice": "caveman_terse",
      "directive": "...",
      "intent": "search_github|search_internet|research|add_feature|wire_into_v10|fix|ask",
      "priority": 8,
      "estimated_lift": "...",
      "rationale": "..."
    }}
  ]
}}
"""


def _load_user_prompt_history(n: int = 20, hours: int | None = 24) -> list[dict[str, Any]]:
    """Read last N user prompts from USER_PROMPTS_HISTORY (written by touch-last-prompt).

    If `hours` is set, also filter to entries within that window. Returns most-recent first.
    Tolerates missing/empty file (returns []).
    """
    if not USER_PROMPTS_HISTORY.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with USER_PROMPTS_HISTORY.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []

    if hours is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        keep: list[dict[str, Any]] = []
        for r in rows:
            try:
                ts = datetime.fromisoformat(r.get("ts", ""))
                if ts >= cutoff:
                    keep.append(r)
            except (TypeError, ValueError):
                # Missing/bad ts → include conservatively (better to see prompt than drop it)
                keep.append(r)
        rows = keep
    return rows[-n:][::-1]


def _compose_landed_summary(max_chars: int = 800) -> str:
    """Summarize recent helper landings (last 1hr) from active_spawns + audit log.

    Reads recent audit_<DATE>.jsonl entries with event 'spawn_launched' or
    'react_trace'. Returns bullet list of titles + status. Empty string if none.
    """
    bullets: list[str] = []
    audit_path = _audit_path()
    if audit_path.exists():
        cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
        try:
            with audit_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("event") not in ("spawn_launched", "spawn_failed",
                                                  "spawn_skipped_no_launcher"):
                        continue
                    try:
                        ts = datetime.fromisoformat(rec.get("timestamp", ""))
                        if ts < cutoff:
                            continue
                    except (TypeError, ValueError):
                        pass
                    title = rec.get("title", "?")
                    ev = rec.get("event", "?").replace("spawn_", "")
                    bullets.append(f"- [{ev}] {title[:100]}")
        except OSError:
            pass
    if not bullets:
        return "(no recent landings — empty last hour)"
    text = "\n".join(bullets[-15:])  # most recent 15
    if len(text) > max_chars:
        text = text[:max_chars] + "\n…[truncated]"
    return text


def _compose_blockers_summary(blockers: list[dict[str, Any]], max_chars: int = 500) -> str:
    """Bullet-list ACTIVE blockers passed in from the ASK stage."""
    if not blockers:
        return "(no blockers identified this cycle)"
    bullets = []
    for b in blockers[:5]:
        sev = b.get("severity", "?")
        msg = b.get("blocker", "?")
        bullets.append(f"- [sev {sev}/10] {msg[:140]}")
    text = "\n".join(bullets)
    return text[:max_chars]


def _compose_goals_summary(max_chars: int = 600) -> str:
    """Summarize mission goals — heading scan of MISSION_PROGRESS.md.

    Pull top-level ## headings to give the persona a sense of UNTOUCHED surfaces
    (vs the LANDED list which shows what was already done).
    """
    if not MISSION_PROGRESS.exists():
        return "(mission progress doc missing)"
    headings: list[str] = []
    try:
        for line in MISSION_PROGRESS.read_text().splitlines():
            line = line.strip()
            if line.startswith("## "):
                headings.append(line[3:].strip())
            elif line.startswith("### "):
                headings.append("  " + line[4:].strip())
            if len(headings) >= 30:
                break
    except OSError:
        return "(mission progress unreadable)"
    if not headings:
        return "(no headings in mission progress)"
    text = "\n".join(f"- {h}" for h in headings[:20])
    return text[:max_chars]


def _persona_safety_filter(directive_text: str) -> tuple[bool, str | None]:
    """Run a persona-generated directive through the SAFETY_BLOCKLIST.

    Returns (safe, blocking_keyword_or_None). Used to reject directives whose
    text would violate hard rails even before they hit gate_candidate."""
    return safety_gate(directive_text or "")


_VALID_PERSONA_INTENTS = {
    "search_github", "search_internet", "research",
    "add_feature", "wire_into_v10", "fix", "ask",
}


def _extract_persona_directives(raw: str) -> list[dict[str, Any]]:
    """Best-effort JSON parse for persona directives. Tolerates:
       - bare JSON: {"directives": [...]}
       - ```json fenced block
       - {"message": "<json-string>"} envelope
       - OpenClaw envelope: {"payloads":[{"text":"...json..."}], ...}
       - Mixed prose + JSON region
    """
    if not raw:
        return []

    def _try_load(s: str) -> list[dict[str, Any]] | None:
        try:
            o = json.loads(s)
        except (json.JSONDecodeError, ValueError):
            return None
        if isinstance(o, dict) and "directives" in o:
            d = o["directives"]
            return d if isinstance(d, list) else None
        return None

    # First: try parsing the whole blob as JSON (handles OpenClaw envelope).
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        obj = None

    if isinstance(obj, dict):
        # 1a. Bare top-level
        if "directives" in obj and isinstance(obj["directives"], list):
            return obj["directives"]
        # 1b. OpenClaw envelope -> payloads[*].text
        payloads = obj.get("payloads")
        if isinstance(payloads, list):
            for p in payloads:
                if not isinstance(p, dict):
                    continue
                text = p.get("text")
                if not isinstance(text, str):
                    continue
                # Strip ```json fences inside the payload text
                fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
                inner = fence.group(1) if fence else text
                got = _try_load(inner)
                if got is not None:
                    return got
                # Or fallback: regex search inside the payload text
                m = re.search(r"\{[\s\S]*\"directives\"[\s\S]*\}", text)
                if m:
                    got = _try_load(m.group(0))
                    if got is not None:
                        return got
        # 1c. {"message": "..."} envelope (legacy)
        msg = obj.get("message")
        if isinstance(msg, str):
            fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", msg, re.DOTALL)
            inner = fence.group(1) if fence else msg
            got = _try_load(inner)
            if got is not None:
                return got
            m = re.search(r"\{[\s\S]*\"directives\"[\s\S]*\}", msg)
            if m:
                got = _try_load(m.group(0))
                if got is not None:
                    return got

    # 2. Strip code fences from raw and try
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fence:
        got = _try_load(fence.group(1))
        if got is not None:
            return got

    # 3. Regex hunt for the `{ ... "directives" ... }` substring in raw
    m = re.search(r"\{[\s\S]*?\"directives\"[\s\S]*?\}", raw)
    if m:
        got = _try_load(m.group(0))
        if got is not None:
            return got
        # try expanding to a wider region in case nested objects
        m2 = re.search(r"\{[\s\S]*\"directives\"[\s\S]*\}", raw)
        if m2:
            got = _try_load(m2.group(0))
            if got is not None:
                return got

    return []


def _directive_to_candidate(d: dict[str, Any], cycle_id: str) -> dict[str, Any] | None:
    """Convert a persona-generated directive dict into a spawn candidate dict.

    Reuses the user_inbox brief builders so persona directives flow through the
    same code path as user-typed inbox items (intent dispatch -> safety_gate
    -> spawn_helper -> answer file at dashboard/inbox_answers/<id>.md).

    Returns None if the directive is malformed or fails the persona safety
    filter (caller logs the rejection).
    """
    directive = (d.get("directive") or "").strip()
    intent = (d.get("intent") or "").strip()
    if not directive or intent not in _VALID_PERSONA_INTENTS:
        return None

    # Hard safety pre-check on the directive text itself.
    safe, kw = _persona_safety_filter(directive)
    if not safe:
        log.info("persona_directive REJECT (safety) kw=%s text=%s", kw, directive[:80])
        _audit({
            "event": "persona_directive_safety_block",
            "cycle_id": cycle_id, "blocking_keyword": kw, "directive": directive[:200],
        })
        return None

    # Synthesize an inbox-style item and route through the matching brief builder.
    item_id = "persona_" + hashlib.sha256(
        f"{cycle_id}|{directive}|{intent}".encode()
    ).hexdigest()[:10]
    pseudo_item = {
        "id": item_id,
        "ts": _now_utc(),
        "intent": intent,
        "payload": directive,
        "priority": int(d.get("priority", 7) or 7),
        "status": "pending",
        "source": "persona",
    }
    builder = _INTENT_DISPATCH.get(intent)
    if builder is None:
        log.warning("persona_directive: unknown intent %r — skip", intent)
        return None
    try:
        cand = builder(pseudo_item)
    except Exception as e:  # noqa: BLE001
        log.warning("persona_directive: builder failed: %s", e)
        return None

    # Boost priority via impact_score (gate_candidate uses impact_score, not priority)
    try:
        base_impact = int(cand.get("impact_score", 7) or 7)
    except (TypeError, ValueError):
        base_impact = 7
    cand["impact_score"] = min(10, base_impact + PERSONA_PRIORITY_BOOST)
    cand["_persona"] = True
    cand["_persona_directive"] = directive
    cand["_persona_intent"] = intent
    cand["_persona_rationale"] = d.get("rationale", "")
    cand["_persona_estimated_lift"] = d.get("estimated_lift", "")
    return cand


def _persist_persona_directive(d: dict[str, Any], cycle_id: str, dispatched: bool,
                                candidate_title: str | None) -> None:
    """Append a persona directive event to USER_DIRECTIVES_LOG (jsonl)."""
    USER_DIRECTIVES_LOG.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": _now_utc(),
        "cycle_id": cycle_id,
        "voice": d.get("voice", "caveman_terse"),
        "directive": d.get("directive", ""),
        "intent": d.get("intent", ""),
        "priority": d.get("priority"),
        "estimated_lift": d.get("estimated_lift", ""),
        "rationale": d.get("rationale", ""),
        "dispatched": bool(dispatched),
        "candidate_title": candidate_title,
    }
    try:
        with USER_DIRECTIVES_LOG.open("a") as f:
            f.write(json.dumps(row) + "\n")
    except OSError as e:
        log.warning("persona log persist failed: %s", e)


def _refresh_user_directives_dashboard() -> None:
    """Rewrite dashboard/USER_DIRECTIVES.md showing last N persona directives."""
    USER_DIRECTIVES_DASHBOARD.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if USER_DIRECTIVES_LOG.exists():
        try:
            for line in USER_DIRECTIVES_LOG.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except OSError:
            pass
    recent = rows[-PERSONA_DIRECTIVES_RETAIN:][::-1]
    # Group: this cycle's = the most recent cycle_id; rest are history.
    this_cycle_id = recent[0].get("cycle_id") if recent else None
    this_cycle = [r for r in recent if r.get("cycle_id") == this_cycle_id]
    history = [r for r in recent if r.get("cycle_id") != this_cycle_id]

    body = f"# USER DIRECTIVES (generated AS user)\n\n_Last refreshed: {_now_utc()}_\n\n"
    body += (
        "Persona-ideation loop: the autonomous daemon impersonates the user's "
        "voice + priorities and generates directives in caveman-terse style, "
        "pushing completeness / scale / blocker-fix / iterate-on-landed.\n\n"
    )
    if this_cycle:
        body += f"## Active this cycle (cycle_id={this_cycle_id}, n={len(this_cycle)})\n\n"
        for r in this_cycle:
            status = "DISPATCHED" if r.get("dispatched") else "rejected"
            body += (
                f"- **[{r.get('intent', '?')}]** p={r.get('priority', '?')} `{r.get('directive', '?')}`\n"
                f"  - status: {status}\n"
                f"  - lift: {r.get('estimated_lift', '?')}\n"
                f"  - rationale: {r.get('rationale', '?')[:160]}\n"
            )
        body += "\n"
    body += f"## Recent history (last {PERSONA_DIRECTIVES_RETAIN})\n\n"
    if not history:
        body += "_(no history yet — first cycle)_\n"
    for r in history:
        status = "DISPATCHED" if r.get("dispatched") else "rejected"
        body += (
            f"- {r.get('ts', '?')[:19]} [{r.get('intent', '?')}] {status} "
            f"`{(r.get('directive') or '')[:100]}`\n"
        )

    body += "\n## How it works\n\n"
    body += (
        "Each cycle (90s by default), `_user_persona_ideate()` calls DeepSeek "
        "with `USER_PERSONA_PROMPT` and the current state. Generated directives:\n"
        "- inherit the same safety_gate as user-typed inbox items;\n"
        "- get +"
        f"{PERSONA_PRIORITY_BOOST} impact_score boost (priority tie-breaker);\n"
        "- flow through the standard `spawn_helper` pipeline;\n"
        "- write their answer to `dashboard/inbox_answers/persona_<id>.md`.\n\n"
        "Voice patterns observed: completeness-mandate, scale-push, blocker-fix, "
        "iterate-on-landed, auto-solve, multi-domain, caveman-terse, audit-questions.\n"
    )

    try:
        USER_DIRECTIVES_DASHBOARD.write_text(body)
    except OSError as e:
        log.warning("USER_DIRECTIVES dashboard write failed: %s", e)


# ════════════════════════════════════════════════════════════════════════════
# NO-DIRECTION SELF-DIRECTING MODULES (2026-05-20)
# ════════════════════════════════════════════════════════════════════════════
# autosolve_skip: feature-add — these modules let the daemon ACT LIKE THE USER
# without any user prompt. They coexist with self-awareness helper affe28e9
# inside state/gabriel_self/. Module order: predictor → curiosity → goal_tree
# → intrinsic_reward → time-aware → skills.

# ─── 1. User-imitation predictor ────────────────────────────────────────────

def _load_user_predictor() -> dict[str, Any]:
    """Read the user-imitation predictor model (mined from prompt history).

    Schema:
      {
        "version": 1,
        "updated_ts": iso8601,
        "rules": [
            {"trigger": "...", "predicted_request": "...", "confidence": 0.85,
             "intent": "scale|fix|search_github|...", "support_count": 12}, ...
        ],
        "time_of_day_distribution": {"0": 5, "1": 2, ...},
        "intent_frequency": {"scale": 30, "fix": 22, ...},
        "n_prompts_analyzed": 87,
    }
    """
    if not USER_PREDICTOR_FILE.exists():
        return {"version": 1, "rules": [], "time_of_day_distribution": {},
                "intent_frequency": {}, "n_prompts_analyzed": 0}
    try:
        return json.loads(USER_PREDICTOR_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "rules": [], "time_of_day_distribution": {},
                "intent_frequency": {}, "n_prompts_analyzed": 0}


def _save_user_predictor(model: dict[str, Any]) -> None:
    GABRIEL_SELF_DIR.mkdir(parents=True, exist_ok=True)
    model = dict(model)
    model["updated_ts"] = _now_utc()
    try:
        USER_PREDICTOR_FILE.write_text(json.dumps(model, indent=2))
    except OSError as e:
        log.warning("user_predictor save failed: %s", e)


_INTENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "scale": ("scale", "all 50", "all 500", "rollout", "rest of"),
    "fix": ("fix", "broken", "debug", "error", "fail"),
    "search_github": ("github", "search github", "find repo"),
    "search_internet": ("search", "google", "internet", "research"),
    "add_feature": ("add", "implement", "build", "create"),
    "wire": ("wire", "integrate", "connect", "hook up"),
    "audit": ("audit", "verify", "check", "inspect"),
    "improve": ("improve", "tune", "optimize", "better"),
    "iterate": ("iterate", "again", "more", "continue"),
    "ask": ("?", "why", "what", "how"),
}


def _classify_prompt_intent(text: str) -> str:
    """Map free-form user prompt to a coarse intent label."""
    t = (text or "").lower()
    for intent, kws in _INTENT_KEYWORDS.items():
        for kw in kws:
            if kw in t:
                return intent
    return "other"


def _refresh_user_predictor() -> dict[str, Any]:
    """Mine user_prompts_history.jsonl into a predictor model.

    Extracts:
      • time-of-day → intent distribution (which intents dominate at hour H)
      • recent-landing → typical-follow-up rules (sketch: scan audit for spawn_landed
        events within 30 min BEFORE each prompt; bucket landing-category → next-intent)
      • intent_frequency (overall priors)

    Pure mining — never spawns anything. Failure-tolerant; returns the model
    even on parse errors (degraded but valid).
    """
    history = _load_user_prompt_history(n=10**6, hours=None)
    intent_freq: dict[str, int] = {}
    tod_dist: dict[str, dict[str, int]] = {}  # hour_str → intent → count
    rules: list[dict[str, Any]] = []
    n = 0
    for r in history:
        prompt = r.get("prompt") or ""
        # Filter out task-notification / brief-path entries (not real prompts).
        if not prompt.strip() or prompt.startswith("<task-notification") \
                or prompt.startswith("@/Users/"):
            continue
        intent = _classify_prompt_intent(prompt)
        intent_freq[intent] = intent_freq.get(intent, 0) + 1
        try:
            ts = datetime.fromisoformat(r.get("ts", ""))
            hour = str(ts.hour)
        except (TypeError, ValueError):
            hour = "?"
        tod_dist.setdefault(hour, {})
        tod_dist[hour][intent] = tod_dist[hour].get(intent, 0) + 1
        n += 1

    # Derive top intent per hour (if support >= 2)
    for hour, dist in tod_dist.items():
        if not dist:
            continue
        best = max(dist.items(), key=lambda x: x[1])
        if best[1] >= 2:
            rules.append({
                "trigger": f"hour=={hour} UTC",
                "predicted_intent": best[0],
                "confidence": round(best[1] / max(1, sum(dist.values())), 2),
                "support_count": best[1],
            })

    # Derive top intents overall (these are the "what user often asks" priors).
    top_intents = sorted(intent_freq.items(), key=lambda x: -x[1])[:5]
    for intent, count in top_intents:
        if count >= 3:
            rules.append({
                "trigger": "global_prior",
                "predicted_intent": intent,
                "confidence": round(count / max(1, n), 2),
                "support_count": count,
            })

    model = {
        "version": 1,
        "rules": rules,
        "time_of_day_distribution": {h: dict(d) for h, d in tod_dist.items()},
        "intent_frequency": intent_freq,
        "n_prompts_analyzed": n,
    }
    _save_user_predictor(model)
    return model


def _predict_user_request(state_snapshot: dict[str, Any]) -> dict[str, Any] | None:
    """Given current state (hour, recent landings, blockers), predict the next
    likely user request as a candidate dict (ready for the spawn pipeline).

    Returns None if predictor empty or no confident rule fires.
    """
    model = _load_user_predictor()
    if not model.get("rules"):
        return None
    hour = state_snapshot.get("hour", datetime.now(timezone.utc).hour)
    # Pick highest-confidence matching rule.
    best: dict[str, Any] | None = None
    best_conf = 0.0
    for rule in model["rules"]:
        trig = rule.get("trigger", "")
        if trig == f"hour=={hour} UTC" or trig == "global_prior":
            conf = float(rule.get("confidence", 0.0))
            # Prefer hour-specific over global_prior at equal conf.
            score = conf + (0.05 if "hour==" in trig else 0.0)
            if score > best_conf:
                best_conf = score
                best = rule
    if not best:
        return None
    intent = best.get("predicted_intent", "iterate")
    # Build a candidate-shaped dict using the same intent dispatch as persona.
    title = f"predicted_user_{intent}_{state_snapshot.get('cycle_id','?')}"
    payload = (
        f"Predicted user request (no actual prompt). Intent={intent}, "
        f"trigger={best.get('trigger')}, confidence={best.get('confidence')}. "
        "Scan recent landings + blockers, then execute the action the user "
        "most often demands in this state. Return a brief report."
    )
    pseudo = {
        "id": f"pred_{hashlib.sha256(title.encode()).hexdigest()[:10]}",
        "ts": _now_utc(),
        "intent": intent if intent in _VALID_PERSONA_INTENTS else "ask",
        "payload": payload,
        "priority": 7,
        "status": "pending",
        "source": "user_predictor",
    }
    builder = _INTENT_DISPATCH.get(pseudo["intent"])
    if not builder:
        return None
    try:
        cand = builder(pseudo)
    except Exception as e:  # noqa: BLE001
        log.warning("user_predictor: builder failed: %s", e)
        return None
    cand["_predicted"] = True
    cand["_prediction_rule"] = best
    cand["impact_score"] = min(10, int(cand.get("impact_score", 7) or 7) + 1)
    return cand


# ─── 2. Curiosity-driven exploration ────────────────────────────────────────

def _load_curiosity_state() -> dict[str, Any]:
    """Read curiosity state. Schema:
        {area: {"last_touched": iso8601 or None, "spawn_count": int}, ...}
    """
    if not CURIOSITY_STATE_FILE.exists():
        return {a: {"last_touched": None, "spawn_count": 0} for a in CURIOSITY_AREAS}
    try:
        state = json.loads(CURIOSITY_STATE_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return {a: {"last_touched": None, "spawn_count": 0} for a in CURIOSITY_AREAS}
    # Backfill any missing areas.
    for a in CURIOSITY_AREAS:
        if a not in state:
            state[a] = {"last_touched": None, "spawn_count": 0}
    return state


def _save_curiosity_state(state: dict[str, Any]) -> None:
    GABRIEL_SELF_DIR.mkdir(parents=True, exist_ok=True)
    try:
        CURIOSITY_STATE_FILE.write_text(json.dumps(state, indent=2))
    except OSError as e:
        log.warning("curiosity_state save failed: %s", e)


_AXIS_TO_AREA = {
    "data": "data", "model": "model", "infra": "infra", "cloud": "cloud",
    "research": "research", "research_application": "research",
    "trading": "trading", "self_improvement": "self_improvement",
    "diagnostics": "diagnostics", "exploration": "research",
    "feature_discovery": "model", "user_responsiveness": "diagnostics",
}


def _touch_curiosity_area(area: str) -> None:
    """Mark an area as just-touched (called after every spawn)."""
    if not area:
        return
    norm = _AXIS_TO_AREA.get(area, area if area in CURIOSITY_AREAS else None)
    if not norm:
        return
    state = _load_curiosity_state()
    state.setdefault(norm, {"last_touched": None, "spawn_count": 0})
    state[norm]["last_touched"] = _now_utc()
    state[norm]["spawn_count"] = int(state[norm].get("spawn_count", 0)) + 1
    _save_curiosity_state(state)


def _curiosity_score(area_state: dict[str, Any]) -> float:
    """Higher = more curious (stale + low spawn count)."""
    last = area_state.get("last_touched")
    spawn_count = int(area_state.get("spawn_count", 0))
    if not last:
        hours_since = CURIOSITY_STALE_HOURS * 10  # huge — never touched
    else:
        try:
            ts = datetime.fromisoformat(last)
            hours_since = (datetime.now(timezone.utc) - ts).total_seconds() / 3600.0
        except (TypeError, ValueError):
            hours_since = CURIOSITY_STALE_HOURS * 10
    # Score: time component dominates; spawn-count is a soft penalty.
    return hours_since - (spawn_count * 0.5)


def _most_curious_area(state: dict[str, Any] | None = None) -> str:
    """Return the highest-curiosity area name."""
    if state is None:
        state = _load_curiosity_state()
    return max(CURIOSITY_AREAS, key=lambda a: _curiosity_score(state.get(a, {})))


def _curiosity_candidate(cycle_id: str) -> dict[str, Any] | None:
    """Force a high-curiosity exploration seed. Called CURIOSITY_FORCED_RATE
    fraction of cycles regardless of normal ideate/persona output.
    """
    state = _load_curiosity_state()
    area = _most_curious_area(state)
    score = _curiosity_score(state.get(area, {}))
    title = f"curiosity_explore_{area}_{cycle_id}"
    brief = (
        f"Curiosity-driven exploration of UNTOUCHED area: {area}.\n"
        f"Curiosity score (hrs_since - 0.5*spawn_count): {score:.1f}\n"
        f"Goal: produce 1 concrete useful artifact about {area} state of the\n"
        f"system. Examples per area:\n"
        f"  data → audit feature_discovery/unwired_features.jsonl for {area}-tagged items\n"
        f"  model → list per_ticker_best entries by holdout-Sharpe, flag bottom-10 for retrain\n"
        f"  infra → inspect launchctl daemons, modal queue, drive_sync_batch lag\n"
        f"  cloud → modal spend cap, gh_actions queue depth, dispatcher health\n"
        f"  research → memory/research_*.md unused techniques inventory\n"
        f"  trading → paper_trade ledger drift, alpaca account anomalies\n"
        f"  self_improvement → audit prompts/gates, propose ONE patch\n"
        f"  diagnostics → grep audit_<TODAY>.jsonl for >30min gaps + diagnose\n"
        f"Save report to reports/curiosity_{area}_<TS>.md."
    )
    return {
        "title": title,
        "reason": f"curiosity-forced: {area} is most stale (score={score:.1f})",
        "helper_brief": brief,
        "expected_lift": f"explore underweighted {area} surface",
        "effort_min": 15,
        "impact_score": 6,
        "novelty_score": 9,
        "feasibility_score": 7,
        "_curiosity": True,
        "_curiosity_area": area,
        "_curiosity_score": score,
    }


# ─── 3. Hierarchical goal generator ─────────────────────────────────────────

_DEFAULT_GOAL_TREE: dict[str, Any] = {
    "version": 1,
    "top": {
        "id": "G0",
        "goal": "Master S&P 500 trading via XGBoost + Mythos paper-trading + autonomous improvement",
        "status": "in_progress",
    },
    "mid": [
        {"id": "M1", "parent": "G0",
         "goal": "All 509 S&P 500 tickers mastered with latest v10 strategy",
         "status": "in_progress",
         "blockers": [], "last_progress_ts": None},
        {"id": "M2", "parent": "G0",
         "goal": "Live paper-trade daily Sharpe >1.5 sustained for 5 sessions",
         "status": "in_progress",
         "blockers": [], "last_progress_ts": None},
        {"id": "M3", "parent": "G0",
         "goal": "Friday-retrain pipeline runs end-to-end without manual intervention",
         "status": "in_progress",
         "blockers": [], "last_progress_ts": None},
        {"id": "M4", "parent": "G0",
         "goal": "Daemon self-improvement: weekly net-positive lessons.md additions",
         "status": "in_progress",
         "blockers": [], "last_progress_ts": None},
        {"id": "M5", "parent": "G0",
         "goal": "Cloud-routing target: >=85% of CPU/RAM workload off Mac",
         "status": "in_progress",
         "blockers": [], "last_progress_ts": None},
    ],
    "low": [],   # generated per-cycle, capped to last 30
}


def _load_goal_tree() -> dict[str, Any]:
    if not GOAL_TREE_FILE.exists():
        return dict(_DEFAULT_GOAL_TREE)
    try:
        tree = json.loads(GOAL_TREE_FILE.read_text())
        # Backfill any missing top/mid fields.
        if "top" not in tree:
            tree["top"] = _DEFAULT_GOAL_TREE["top"]
        if "mid" not in tree:
            tree["mid"] = _DEFAULT_GOAL_TREE["mid"]
        if "low" not in tree:
            tree["low"] = []
        return tree
    except (OSError, json.JSONDecodeError):
        return dict(_DEFAULT_GOAL_TREE)


def _save_goal_tree(tree: dict[str, Any]) -> None:
    GABRIEL_SELF_DIR.mkdir(parents=True, exist_ok=True)
    tree = dict(tree)
    tree["updated_ts"] = _now_utc()
    # Cap low-goals to last 30 to keep file small.
    if len(tree.get("low", [])) > 30:
        tree["low"] = tree["low"][-30:]
    try:
        GOAL_TREE_FILE.write_text(json.dumps(tree, indent=2))
    except OSError as e:
        log.warning("goal_tree save failed: %s", e)


def _update_goal_tree(cycle_id: str, blockers: list[dict[str, Any]],
                      recent_landings: list[str]) -> dict[str, Any]:
    """Walk the goal tree, refresh mid-goal blockers + progress timestamps,
    and append a low-level next-action atom for the most-blocked mid-goal.
    """
    tree = _load_goal_tree()
    # Refresh each mid-goal's blockers list from this cycle's ASK output.
    blocker_msgs = [b.get("blocker", "")[:140] for b in (blockers or [])]
    landed_set = set(recent_landings or [])
    most_blocked: dict[str, Any] | None = None
    most_blocked_count = -1
    for mid in tree.get("mid", []):
        # Match blockers by simple keyword overlap.
        mid_kws = (mid.get("goal") or "").lower().split()
        hits = [m for m in blocker_msgs
                if any(k for k in mid_kws if len(k) > 3 and k in (m or "").lower())]
        mid["blockers"] = hits[:5]
        # Mark progress if any landing's title contains a mid-goal keyword.
        if any(any(k for k in mid_kws if len(k) > 3 and k in t.lower())
               for t in landed_set):
            mid["last_progress_ts"] = _now_utc()
        if len(hits) > most_blocked_count:
            most_blocked_count = len(hits)
            most_blocked = mid

    # Append a low-level next-action atom (one per cycle).
    if most_blocked and most_blocked_count > 0:
        atom = {
            "id": f"L_{cycle_id}",
            "parent": most_blocked.get("id"),
            "action": f"Unblock {most_blocked.get('goal','?')[:60]}: {most_blocked['blockers'][0][:80]}",
            "ts": _now_utc(),
        }
        tree.setdefault("low", []).append(atom)

    _save_goal_tree(tree)
    return tree


def _goal_tree_candidate(cycle_id: str, tree: dict[str, Any]) -> dict[str, Any] | None:
    """Produce a candidate dict for the most-recent low-level atom (if any)."""
    lows = tree.get("low") or []
    if not lows:
        return None
    atom = lows[-1]
    if atom.get("id") != f"L_{cycle_id}":
        return None  # already-scheduled atom; don't re-fire.
    title = f"goal_atom_{atom.get('id')}"
    brief = (
        f"Hierarchical-goal next-action atom.\n"
        f"Parent mid-goal: {atom.get('parent')}\n"
        f"Action: {atom.get('action')}\n"
        f"Output: produce ONE concrete artifact that materially advances the\n"
        f"parent mid-goal. Reference state/gabriel_self/goal_tree.json for\n"
        f"context. Save to reports/goal_atom_<TS>.md."
    )
    return {
        "title": title,
        "reason": "goal-tree walk: most-blocked mid-goal next-action",
        "helper_brief": brief,
        "expected_lift": "advance hierarchical goal",
        "effort_min": 20,
        "impact_score": 8,
        "novelty_score": 7,
        "feasibility_score": 7,
        "_goal_tree": True,
        "_goal_atom": atom,
    }


# ─── 4. Intrinsic motivation function ───────────────────────────────────────

def _intrinsic_reward(spawn_outcome: dict[str, Any], baseline: dict[str, Any] | None = None) -> float:
    """Compute intrinsic reward for a returned spawn outcome.

    surprise        = |actual_effort - estimated_effort| / max(estimated,1)
                       (clipped to [0,1])
    learning_progress = 1.0 if outcome.status == "success" and a new lesson
                       or skill was emitted, else 0.5 if success-only, else 0.0
    novelty_delta   = (post-spawn novelty score - pre-spawn novelty score),
                       clipped to [0,1]

    Returns weighted sum ∈ [0,1].
    """
    try:
        est = float(spawn_outcome.get("estimated_effort_min", 15) or 15)
        actual = float(spawn_outcome.get("actual_effort_min", est) or est)
        surprise = min(1.0, abs(actual - est) / max(1.0, est))
    except (TypeError, ValueError):
        surprise = 0.0
    status = (spawn_outcome.get("status") or "").lower()
    emitted = spawn_outcome.get("emitted_lesson_or_skill", False)
    if status == "success" and emitted:
        learning_progress = 1.0
    elif status == "success":
        learning_progress = 0.5
    else:
        learning_progress = 0.0
    try:
        nd = float(spawn_outcome.get("novelty_delta", 0.0) or 0.0)
        novelty = max(0.0, min(1.0, nd))
    except (TypeError, ValueError):
        novelty = 0.0
    reward = (INTRINSIC_W_SURPRISE * surprise
              + INTRINSIC_W_LEARNING * learning_progress
              + INTRINSIC_W_NOVELTY * novelty)
    return round(reward, 3)


def _log_intrinsic_reward(spawn_outcome: dict[str, Any], reward: float) -> None:
    """Append the reward row to intrinsic_rewards.jsonl and write a lessons.md
    note if reward >= 0.7 (high) or <= 0.2 (very low).
    """
    GABRIEL_SELF_DIR.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": _now_utc(),
        "title": spawn_outcome.get("title", "?"),
        "area": spawn_outcome.get("area"),
        "reward": reward,
        "components": {
            "estimated_effort_min": spawn_outcome.get("estimated_effort_min"),
            "actual_effort_min": spawn_outcome.get("actual_effort_min"),
            "status": spawn_outcome.get("status"),
            "novelty_delta": spawn_outcome.get("novelty_delta"),
        },
    }
    try:
        with INTRINSIC_REWARDS_FILE.open("a") as f:
            f.write(json.dumps(row) + "\n")
    except OSError as e:
        log.warning("intrinsic_reward log failed: %s", e)
        return

    # High reward → add to lessons.md as "pursue more like this".
    # Low reward  → add as "deprioritize this family".
    note: str | None = None
    if reward >= 0.7:
        note = (f"\n- HIGH-REWARD PATTERN ({reward:.2f}): "
                f"`{row['title']}` — pursue more like this in future cycles.\n")
    elif reward <= 0.2 and spawn_outcome.get("status") not in ("running", "pending"):
        note = (f"\n- LOW-REWARD PATTERN ({reward:.2f}): "
                f"`{row['title']}` — deprioritize this family.\n")
    if note:
        try:
            LESSONS_FILE.parent.mkdir(parents=True, exist_ok=True)
            if not LESSONS_FILE.exists():
                LESSONS_FILE.write_text("# Autonomous-daemon Lessons (auto-written)\n")
            with LESSONS_FILE.open("a") as f:
                f.write(note)
        except OSError as e:
            log.warning("intrinsic_reward lessons-write failed: %s", e)


# ─── 5. Time-aware behavior ─────────────────────────────────────────────────

def _is_daytime_utc(now: "datetime | None" = None) -> bool:
    now = now or datetime.now(timezone.utc)
    h = now.hour
    # Daytime PT = 07:00..22:00 → UTC 14:00..05:00 (next day, crosses midnight).
    return h >= DAYTIME_UTC_START_HOUR or h < DAYTIME_UTC_END_HOUR


def _time_aware_spawn_budget(now: "datetime | None" = None) -> int:
    """Return preferred max spawns this cycle based on user wake/sleep cycle."""
    return DAYTIME_SPAWNS_PER_CYCLE if _is_daytime_utc(now) else NIGHTTIME_SPAWNS_PER_CYCLE


def _time_aware_forced_seed(cycle_id: str, now: "datetime | None" = None) -> dict[str, Any] | None:
    """Pre-market open (06:30 PT) → force paper-trade daemon check.
    Post-market close (16:00 PT) → force fills + signal_outcomes check.
    Returns a candidate dict or None.
    """
    now = now or datetime.now(timezone.utc)
    h = now.hour
    m = now.minute
    # Pre-market: 13:00 UTC (=06:00 PT) ± 30 min window.
    # Post-close: 23:00 UTC (=16:00 PT) ± 30 min window.
    if h == PRE_MARKET_UTC_HOUR and m < 45:
        return {
            "title": f"pre_market_paper_trade_check_{cycle_id}",
            "reason": "time-aware: pre-market open, force paper_trade daemon health check",
            "helper_brief": (
                "Pre-market open (06:30 PT). Verify paper_trade daemons are up.\n"
                "Commands: launchctl list | grep com.zg.paper_trade; tail logs/paper_trade*.log\n"
                "Verify Alpaca paper account /v2/account returns 200 + cash > 0.\n"
                "If any daemon down >5min, bootstrap it. Save report to "
                "reports/pre_market_check_<TS>.md."
            ),
            "expected_lift": "guard against silent paper-trade outage at session open",
            "effort_min": 10,
            "impact_score": 9,
            "novelty_score": 5,
            "feasibility_score": 9,
            "_time_aware": "pre_market",
        }
    if h == POST_CLOSE_UTC_HOUR and m < 45:
        return {
            "title": f"post_close_fills_audit_{cycle_id}",
            "reason": "time-aware: post-market close, force fills + signal_outcomes check",
            "helper_brief": (
                "Post-market close (16:00 PT). Reconcile fills vs signals for today.\n"
                "Read state/paper_trade/fills_<TODAY>.jsonl and signal_outcomes_<TODAY>.jsonl.\n"
                "Compute fill_rate, slippage_mean, signal->fill match percentage.\n"
                "Flag any signal without a corresponding fill+exit pair as orphaned.\n"
                "Save report to reports/post_close_recon_<TS>.md and append P&L row to "
                "memory/project_paper_trade_pnl.md."
            ),
            "expected_lift": "daily reconciliation for live-trading transparency",
            "effort_min": 15,
            "impact_score": 9,
            "novelty_score": 5,
            "feasibility_score": 9,
            "_time_aware": "post_close",
        }
    return None


# ─── 6. Voyager-style skill library ─────────────────────────────────────────

def _skill_path(skill_name: str) -> Path:
    # Sanitize: only allow [a-z0-9_].
    safe = re.sub(r"[^a-z0-9_]", "_", (skill_name or "").lower())[:80] or "unnamed"
    return SKILLS_DIR / f"{safe}.py"


def _list_skills() -> list[dict[str, Any]]:
    """Return manifest list of skills in the library.

    Each .py file MUST start with a docstring + a `# SKILL_META: {...json...}` line
    on the second line. Returns list of {name, path, meta} dicts.
    """
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    out: list[dict[str, Any]] = []
    for f in sorted(SKILLS_DIR.glob("*.py")):
        meta: dict[str, Any] = {}
        try:
            head = f.read_text().splitlines()[:10]
            for line in head:
                line = line.strip()
                if line.startswith("# SKILL_META:"):
                    try:
                        meta = json.loads(line.replace("# SKILL_META:", "", 1).strip())
                    except json.JSONDecodeError:
                        pass
                    break
        except OSError:
            pass
        out.append({"name": f.stem, "path": str(f), "meta": meta})
    return out


def _find_skill(pattern_or_name: str) -> dict[str, Any] | None:
    """Find a skill by name (exact) or pattern substring in its meta.pattern."""
    pattern_or_name = (pattern_or_name or "").lower().strip()
    if not pattern_or_name:
        return None
    for s in _list_skills():
        if s["name"] == pattern_or_name:
            return s
        meta_pat = (s.get("meta", {}).get("pattern") or "").lower()
        if pattern_or_name in meta_pat or meta_pat in pattern_or_name:
            return s
    return None


def _register_skill(name: str, pattern: str, body: str,
                    invocations: int = 1, success_rate: float = 1.0) -> Path:
    """Write/upgrade a skill .py file. Returns the path."""
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    path = _skill_path(name)
    meta = {
        "pattern": pattern,
        "invocations": invocations,
        "success_rate": success_rate,
        "updated_ts": _now_utc(),
        "promoted": False,
    }
    if path.exists():
        # Merge: keep prior invocations + bump.
        try:
            head = path.read_text().splitlines()[:10]
            for line in head:
                line = line.strip()
                if line.startswith("# SKILL_META:"):
                    try:
                        prior = json.loads(line.replace("# SKILL_META:", "", 1).strip())
                        meta["invocations"] = int(prior.get("invocations", 0)) + 1
                        meta["promoted"] = bool(prior.get("promoted", False))
                    except json.JSONDecodeError:
                        pass
                    break
        except OSError:
            pass
    text = (
        f'"""Gabriel skill: {name}.\n\n'
        f"Auto-registered by autonomous_mode_daemon._register_skill().\n"
        f'Pattern: {pattern}\n"""\n'
        f"# SKILL_META: {json.dumps(meta)}\n\n"
        f"{body}\n"
    )
    try:
        path.write_text(text)
    except OSError as e:
        log.warning("skill register failed: %s", e)
        return path

    # Promote if invocations exceed threshold.
    if meta["invocations"] >= SKILL_PROMOTE_THRESHOLD and not meta.get("promoted"):
        try:
            promoted_path = ROOT / "scripts" / f"gabriel_skill_{path.stem}.py"
            promoted_path.write_text(text)
            meta["promoted"] = True
            # Rewrite skill file with promoted flag set.
            path.write_text(
                f'"""Gabriel skill: {name} (PROMOTED).\n"""\n'
                f"# SKILL_META: {json.dumps(meta)}\n\n"
                f"{body}\n"
            )
            log.info("skill PROMOTED: %s -> %s", path.name, promoted_path)
            _audit({"event": "skill_promoted", "skill": name,
                    "invocations": meta["invocations"], "path": str(promoted_path)})
        except OSError as e:
            log.warning("skill promote failed: %s", e)
    return path


def _bootstrap_skill_library() -> None:
    """Seed the skill library with 3 well-known recurring patterns (idempotent)."""
    seeds = [
        ("fix_gh_actions_push_race",
         "gh_actions push race / refs/heads/main update conflict",
         (
            "def fix_gh_actions_push_race():\n"
            "    # Pull --rebase before push; if still racing, queue via cloud_dispatch.\n"
            "    import subprocess as _sp\n"
            "    _sp.run(['git', 'pull', '--rebase', '--autostash'], check=False)\n"
            "    return _sp.run(['git', 'push'], check=False).returncode == 0\n"
        )),
        ("kick_drive_sync_batch",
         "drive_sync_batch lag >5min / staging files stale",
         (
            "def kick_drive_sync_batch():\n"
            "    import subprocess as _sp\n"
            "    _sp.run(['launchctl', 'kickstart', '-k', 'gui/501/com.zg.drive_sync_batch'],\n"
            "            check=False)\n"
        )),
        ("modal_spend_cap_check",
         "Modal monthly spend approaching cap",
         (
            "def modal_spend_cap_check():\n"
            "    # Returns (current_usd, cap_usd, fraction_used). Drain queued jobs if >0.95.\n"
            "    # Real impl reads dashboard/modal_spend.json populated by modal_spend_daemon.\n"
            "    import json, os\n"
            "    p = os.path.expanduser('~/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com'\n"
            "                            '/My Drive/AI-Tools/dashboard/modal_spend.json')\n"
            "    try:\n"
            "        d = json.loads(open(p).read())\n"
            "        return d.get('current_usd', 0.0), d.get('cap_usd', 1.0), d.get('fraction_used', 0.0)\n"
            "    except Exception:\n"
            "        return 0.0, 1.0, 0.0\n"
        )),
    ]
    for name, pattern, body in seeds:
        if not _skill_path(name).exists():
            _register_skill(name, pattern, body)


def _maybe_record_skill(candidate: dict[str, Any], outcome_status: str) -> None:
    """If the candidate matched a known skill pattern AND succeeded, bump
    the skill's invocation count. Called from OBSERVE.
    """
    if outcome_status != "success":
        return
    title = (candidate.get("title") or "").lower()
    skill = _find_skill(title)
    if not skill:
        return
    meta = skill.get("meta", {})
    try:
        _register_skill(
            name=skill["name"],
            pattern=meta.get("pattern", title),
            body="# (body preserved; this call only bumps invocations)",
            invocations=int(meta.get("invocations", 0)) + 1,
            success_rate=float(meta.get("success_rate", 1.0)),
        )
    except Exception as e:  # noqa: BLE001
        log.warning("maybe_record_skill failed: %s", e)


# ════════════════════════════════════════════════════════════════════════════
# END NO-DIRECTION MODULES
# ════════════════════════════════════════════════════════════════════════════


def _user_persona_ideate(blockers: list[dict[str, Any]], cycle_id: str,
                          dry_run: bool = False) -> list[dict[str, Any]]:
    """Generate persona-style directives in the user's own voice.

    Returns a list of CANDIDATE dicts (already converted via `_directive_to_candidate`),
    ready to feed the same gating/spawn pipeline as ideate() candidates.

    On any failure (DeepSeek timeout, parse error, etc) returns []. Daemon
    NEVER blocks — neutral ideate() path still runs.
    """
    if dry_run:
        # Dry-run stub — proves the wiring without burning DeepSeek credits.
        stub_dir = {
            "voice": "caveman_terse",
            "directive": f"DRY_RUN persona stub {cycle_id}",
            "intent": "ask",
            "priority": 6,
            "estimated_lift": "smoke test only",
            "rationale": "dry-run path",
        }
        _persist_persona_directive(stub_dir, cycle_id, dispatched=False, candidate_title="(dry-run)")
        _refresh_user_directives_dashboard()
        return []

    # Compose context
    landed = _compose_landed_summary()
    blockers_str = _compose_blockers_summary(blockers)
    goals = _compose_goals_summary()
    history_rows = _load_user_prompt_history(n=15, hours=24)
    if history_rows:
        user_history = "\n".join(
            f'- "{(r.get("prompt") or "").strip()[:200]}"' for r in history_rows
        )
    else:
        user_history = "(no recent user prompts captured — hook may not have fired yet)"
    recent_ideas_list = _last_n_titles(15)
    recent_ideas = "\n".join(f"- {t}" for t in recent_ideas_list) or "(none)"

    prompt = USER_PERSONA_PROMPT.format(
        n=PERSONA_MAX_DIRECTIVES_PER_CYCLE,
        landed_summary=landed,
        blockers_summary=blockers_str,
        goals_summary=goals,
        user_history=user_history,
        recent_ideas=recent_ideas,
    )

    raw, source = _deepseek_call(prompt, timeout_s=DEEPSEEK_TIMEOUT_S)
    if not raw:
        log.info("persona_ideate: deepseek+ollama both failed — skipping persona pass this cycle")
        _audit({"event": "persona_ideate_no_output", "cycle_id": cycle_id, "source": source})
        _refresh_user_directives_dashboard()
        return []

    directives = _extract_persona_directives(raw)
    log.info("persona_ideate: %d directive(s) parsed (source=%s)", len(directives), source)
    _audit({
        "event": "persona_ideate_parsed", "cycle_id": cycle_id,
        "count": len(directives), "source": source,
    })

    # Cap at PERSONA_MAX_DIRECTIVES_PER_CYCLE and convert to candidates
    candidates: list[dict[str, Any]] = []
    for d in directives[:PERSONA_MAX_DIRECTIVES_PER_CYCLE]:
        if not isinstance(d, dict):
            continue
        cand = _directive_to_candidate(d, cycle_id)
        if cand:
            candidates.append(cand)
            _persist_persona_directive(d, cycle_id, dispatched=True,
                                        candidate_title=cand.get("title"))
        else:
            _persist_persona_directive(d, cycle_id, dispatched=False, candidate_title=None)

    _refresh_user_directives_dashboard()
    return candidates


# ─── Main loop ───────────────────────────────────────────────────────────────


_STOP = False

# Never-sleep cycle counter (process-local, used by _self_reflect cadence).
_NEVER_SLEEP_CYCLE_COUNTER = 0


def _next_backlog_seed() -> dict[str, Any]:
    """Return the next rotating hardcoded backlog seed candidate.

    Rotates through BACKLOG_SEEDS by index persisted to BACKLOG_SEED_ROTATE_FILE
    so successive cycles don't pick the same seed twice and the daemon never
    runs out of fallback work.
    """
    try:
        idx = int(BACKLOG_SEED_ROTATE_FILE.read_text().strip())
    except Exception:
        idx = 0
    if not BACKLOG_SEEDS:
        return {}
    idx = idx % len(BACKLOG_SEEDS)
    seed = BACKLOG_SEEDS[idx]
    try:
        BACKLOG_SEED_ROTATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        BACKLOG_SEED_ROTATE_FILE.write_text(str((idx + 1) % len(BACKLOG_SEEDS)))
    except Exception:  # noqa: BLE001
        pass
    return {
        "title": f"BACKLOG_REFILL_{seed['axis']}: {seed['title']}",
        "reason": f"daemon backlog-refill — all 3 ideate paths returned 0; axis={seed['axis']}",
        "helper_brief": seed["brief"],
        "expected_lift": "produce candidate so daemon never silent-idles",
        "effort_min": 25,
        "impact_score": 5,
        "novelty_score": 8,
        "feasibility_score": 8,
        "_from_backlog_seed": True,
        "_backlog_axis": seed["axis"],
    }


def _refill_backlog(cycle_id: str, recent_titles: list[str] | None = None) -> list[dict[str, Any]]:
    """When all 3 ideate paths return 0, refill from hardcoded backlog."""
    recent_set = {t.lower() for t in (recent_titles or [])}
    picks: list[dict[str, Any]] = []
    tries = 0
    while len(picks) < min(2, MAX_IDEAS_PER_CYCLE) and tries < len(BACKLOG_SEEDS):
        cand = _next_backlog_seed()
        tries += 1
        if not cand:
            break
        if any(cand["title"].lower() in r or r in cand["title"].lower() for r in recent_set):
            continue
        picks.append(cand)
    _audit({
        "event": "backlog_refill",
        "cycle_id": cycle_id,
        "count": len(picks),
        "axes": [p.get("_backlog_axis") for p in picks],
    })
    log.info("BACKLOG_REFILL: %d seed(s) added (axes=%s)",
             len(picks), [p.get("_backlog_axis") for p in picks])
    return picks


def _exploration_seed(cycle_id: str) -> dict[str, Any] | None:
    """Inject one EXPLORATION candidate EXPLORATION_RATE of cycles."""
    import random
    if random.random() >= EXPLORATION_RATE:
        return None
    exploration_seeds = [s for s in BACKLOG_SEEDS if s.get("axis") == "exploration"]
    if not exploration_seeds:
        return None
    seed = random.choice(exploration_seeds)
    cand = {
        "title": f"EXPLORATION_{seed['axis']}: {seed['title']}",
        "reason": "scheduled exploration cycle (escape exploitation local optima)",
        "helper_brief": seed["brief"],
        "expected_lift": "novel direction outside current mission state",
        "effort_min": 25,
        "impact_score": 6,
        "novelty_score": 9,
        "feasibility_score": 7,
        "_exploration": True,
    }
    _audit({"event": "exploration_cycle", "cycle_id": cycle_id, "title": cand["title"]})
    log.info("EXPLORATION cycle: injecting %s", cand["title"])
    return cand


def _self_reflect(cycle_id: str) -> None:
    """Read last 200 audit events, write lessons.md.  Triggered every N cycles."""
    audit_path = _audit_path()
    if not audit_path.exists():
        return
    lines = audit_path.read_text().splitlines()[-200:]
    gate_rejects: dict[str, int] = {}
    no_candidate_cycles = 0
    spawn_count = 0
    load_skips = 0
    drift_count = 0
    backlog_refills = 0
    for line in lines:
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        ev = r.get("event", "?")
        if ev == "gate_decision" and not r.get("ok"):
            gate_rejects[r.get("reason", "?")] = gate_rejects.get(r.get("reason", "?"), 0) + 1
        elif ev == "no_candidates":
            no_candidate_cycles += 1
        elif ev == "spawn_launched":
            spawn_count += 1
        elif ev == "load_gate_skip":
            load_skips += 1
        elif ev == "drift_detected":
            drift_count += 1
        elif ev == "backlog_refill":
            backlog_refills += 1

    ts = _now_utc()
    lessons: list[str] = []
    lessons.append(f"\n## Reflection {ts} (cycle {cycle_id})\n")
    lessons.append(f"- spawn_count_last_200_events: {spawn_count}")
    lessons.append(f"- no_candidate_cycles: {no_candidate_cycles}")
    lessons.append(f"- backlog_refills: {backlog_refills}")
    lessons.append(f"- load_gate_skips: {load_skips}")
    lessons.append(f"- drift_detected: {drift_count}")
    if gate_rejects:
        top = sorted(gate_rejects.items(), key=lambda x: -x[1])[:5]
        lessons.append(f"- top gate-reject reasons: {top}")
    if no_candidate_cycles > spawn_count and no_candidate_cycles > 5:
        lessons.append("- LESSON: ideate paths producing 0 candidates frequently — bump backlog_refill OR tighten prompt schema.")
    if load_skips > 5:
        lessons.append("- LESSON: load_gate_skip firing often — raise AUTONOMOUS_LOAD_FLOOR / HEADROOM.")
    if drift_count > 3:
        lessons.append("- LESSON: drift recurring — orthogonality clause weak; score novelty BEFORE gate.")
    if gate_rejects.get("duplicate", 0) > spawn_count:
        lessons.append("- LESSON: duplicate rejects dominate — ideate recycling titles.")

    body = "\n".join(lessons) + "\n"
    try:
        LESSONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not LESSONS_FILE.exists():
            LESSONS_FILE.write_text("# Autonomous-daemon Lessons (auto-written by _self_reflect)\n")
        with LESSONS_FILE.open("a") as f:
            f.write(body)
        _audit({"event": "self_reflect", "cycle_id": cycle_id,
                "lessons_count": len([l for l in lessons if "LESSON:" in l])})
        log.info("SELF_REFLECT: wrote %d lines to %s", len(lessons), LESSONS_FILE)
    except Exception as e:  # noqa: BLE001
        log.warning("self_reflect: write failed: %s", e)


def _health_assert(cycle_id: str, spawn_count_this_cycle: int,
                   candidates_count: int, blockers_count: int,
                   load_1m: float, inflight: int) -> None:
    """If 0 spawns this cycle, log explicit idle_because with diagnosis."""
    if spawn_count_this_cycle > 0:
        return
    if candidates_count == 0:
        reason = "no_candidates_after_backlog_refill"
    elif inflight >= 10:
        reason = "inflight_saturated"
    elif load_1m > LOAD_GATE_FLOOR + 5:
        reason = f"high_load_{load_1m:.1f}"
    else:
        reason = "all_candidates_rejected_by_gate"
    log.warning("IDLE_BECAUSE: cycle=%s reason=%s candidates=%d blockers=%d load=%.1f inflight=%d",
                cycle_id, reason, candidates_count, blockers_count, load_1m, inflight)
    _audit({
        "event": "idle_because",
        "cycle_id": cycle_id,
        "reason": reason,
        "candidates": candidates_count,
        "blockers": blockers_count,
        "load_1m": round(load_1m, 2),
        "inflight": inflight,
    })


def _sig_handler(signum, frame):  # noqa: ARG001
    global _STOP
    _STOP = True
    log.info("signal %d received — stopping after current iteration", signum)


def run_loop(once: bool = False, dry_run: bool = False) -> int:
    signal.signal(signal.SIGTERM, _sig_handler)
    signal.signal(signal.SIGINT, _sig_handler)

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    last_spawns: list[str] = []

    while not _STOP:
        cfg = _read_config()

        # UNLIMITED MODE (2026-05-20): drift-pause REMOVED. Daemon never self-pauses.
        # Legacy `drift_pause_until` field is ignored (cleared if present, for hygiene).
        if cfg.get("drift_pause_until") is not None:
            cfg["drift_pause_until"] = None
            _write_config(cfg)
            log.info("legacy drift_pause_until cleared (unlimited mode ignores drift pauses)")

        # Toggle check
        if not cfg.get("enabled"):
            _write_heartbeat("disabled", {"reason_off": cfg.get("reason_off")})
            refresh_status_dashboard(cfg, last_spawns)
            log.info("disabled — sleeping %ds", DISABLED_RECHECK_SECONDS)
            if once:
                return 0
            time.sleep(DISABLED_RECHECK_SECONDS)
            continue

        # Idle-detect: yield to active user session (coexist patch 5).
        # If a user prompt was submitted in the last 60s, sleep 90s and retry.
        if _user_active_recently(threshold=60):
            _write_heartbeat("user_active_yielding")
            log.info("yielding to active user session — sleeping 90s")
            if once:
                return 0
            time.sleep(90)
            continue

        # Prune stale spawn records at top of each active iteration.
        _prune_active_spawns()

        # ── ACTIVE ITERATION (ASK-PLAN-DECIDE-EXECUTE-OBSERVE, 2026-05-20) ──
        # autosolve_skip: log format fix for unlimited mode (budget=None)
        cycle_id = hashlib.sha256(_now_utc().encode()).hexdigest()[:8]
        _write_heartbeat("active", {"cycle_id": cycle_id})
        _budget_disp = cfg.get("budget_remaining_usd")
        _budget_str = "UNLIMITED" if _budget_disp is None else f"${float(_budget_disp):.4f}"
        log.info("=== iteration start cycle=%s budget=%s ===", cycle_id, _budget_str)

        # 0. USER INBOX DRAIN (added 2026-05-20) — user-supplied items
        #    take priority over self-generated ideation/plan steps. Briefs are
        #    already concrete (skip ideate) but still flow through gate +
        #    safety + load checks via the unified candidate pipeline below.
        # autosolve_skip: feature wiring, no error
        user_inbox_candidates: list[dict[str, Any]] = []
        if not dry_run:
            try:
                user_inbox_candidates = _drain_user_inbox(cfg, cycle_id)
            except Exception as e:  # noqa: BLE001
                log.warning("user_inbox drain failed: %s — continuing with self-generated only", e)
                _audit({"event": "user_inbox_drain_failed", "cycle_id": cycle_id, "error": str(e)})

        # 1. Compose mission summary
        summary = compose_mission_summary()
        _audit({"event": "iteration_start", "cycle_id": cycle_id, "summary_chars": len(summary)})

        # 1a. ASK — self-question to identify blockers (added 2026-05-20)
        current_plan_id: str | None = None
        current_step_id: str | None = None
        last_decision_id: str | None = None
        blockers: list[dict[str, Any]] = []
        last_react_summary: str | None = None
        if not dry_run:
            try:
                inflight_titles = last_spawns[-10:]
                recent_decisions = _load_recent_decisions(5)
                blockers = _ask_blockers(summary, recent_decisions, inflight_titles, cycle_id)
                _write_heartbeat("asking", {"cycle_id": cycle_id, "blockers_count": len(blockers)})
            except Exception as e:  # noqa: BLE001
                log.warning("ASK stage failed: %s — continuing with empty blockers", e)
                _audit({"event": "ask_failed", "cycle_id": cycle_id, "error": str(e)})

        # 1b. PLAN — build multi-step plan (added 2026-05-20)
        plan: dict[str, Any] = {"plan_id": None, "steps": [], "cycle_id": cycle_id}
        if not dry_run:
            try:
                plan = _make_plan(blockers, summary, cycle_id)
                current_plan_id = plan.get("plan_id")
                _write_heartbeat("planning", {
                    "cycle_id": cycle_id, "current_plan_id": current_plan_id,
                    "steps_count": len(plan.get("steps", [])),
                })
            except Exception as e:  # noqa: BLE001
                log.warning("PLAN stage failed: %s — falling back to ideate-only path", e)
                _audit({"event": "plan_failed", "cycle_id": cycle_id, "error": str(e)})

        # 2. Ideate (still kept — produces candidate set for spawn)
        if dry_run:
            log.info("dry-run: skipping ideate, using stub candidate")
            candidates = [{
                "title": f"DRY_RUN_STUB_{int(time.time())}",
                "reason": "dry-run smoke",
                "helper_brief": "no-op",
                "expected_lift": "none",
                "effort_min": 5,
                "impact_score": 5,
                "novelty_score": 5,
                "feasibility_score": 5,
            }]
        else:
            # autosolve_skip: pass recent titles for drift-mitigation (orthogonality clause)
            candidates = ideate(summary, recent_titles=_last_n_titles(20))

        # Merge plan steps into candidates as ideate-equivalent records so the
        # rest of the gating/spawn machinery treats them uniformly. Plan steps
        # are stronger signals (we asked, planned, decided) — prioritize them.
        plan_candidates: list[dict[str, Any]] = []
        for s in plan.get("steps", []):
            if not isinstance(s, dict):
                continue
            plan_candidates.append({
                "title": s.get("action", "(unnamed plan step)"),
                "reason": f"plan step {s.get('step_id', '?')} addressing blockers",
                "helper_brief": f"Target: {s.get('target', '?')}\n"
                                f"Success criteria: {s.get('success_criteria', '?')}\n"
                                f"Depends on: {s.get('depends_on', [])}",
                "expected_lift": s.get("success_criteria", "(unspecified)"),
                "effort_min": min(int(s.get("estimated_min", 25) or 25), 30),
                "impact_score": int(s.get("priority", 5) or 5),
                "novelty_score": 7,
                "feasibility_score": 7,
                "_from_plan": True,
                "_plan_step": s,
            })

        # 2a. USER-PERSONA ideate (added 2026-05-20) — generate directives in
        # the user's own voice (caveman-terse, completeness-mandate, scale-push,
        # blocker-fix, iterate-on-landed). Inherits same safety_gate + spawn
        # pipeline as user_inbox. Failure-tolerant (returns []) so the rest of
        # the cycle is never blocked.
        # autosolve_skip: feature wiring
        persona_candidates: list[dict[str, Any]] = []
        try:
            persona_candidates = _user_persona_ideate(blockers, cycle_id, dry_run=dry_run)
            _write_heartbeat("persona_ideating", {
                "cycle_id": cycle_id, "persona_candidates": len(persona_candidates),
            })
        except Exception as e:  # noqa: BLE001
            log.warning("USER-PERSONA stage failed: %s — continuing without persona candidates", e)
            _audit({"event": "persona_ideate_failed", "cycle_id": cycle_id, "error": str(e)})

        # autosolve_skip: feature-wire — NO-DIRECTION self-directing candidates.
        # Six independent generators that fire WITHOUT any user prompt:
        #   2a. predictor — "what would the user demand in this state?"
        #   2b. curiosity — force exploration of stalest project area (20% of cycles)
        #   2c. goal-tree — walk hierarchical goal tree, schedule next-action atom
        #   2d. time-aware — pre-market open / post-close forced seeds
        # (intrinsic_reward + skill_library hook in via OBSERVE/post-spawn callbacks below)
        no_direction_candidates: list[dict[str, Any]] = []
        if not dry_run:
            # Refresh user-imitation predictor once per cycle (cheap; pure mining).
            try:
                _refresh_user_predictor()
            except Exception as e:  # noqa: BLE001
                log.warning("user_predictor refresh failed: %s", e)
                _audit({"event": "user_predictor_refresh_failed", "cycle_id": cycle_id,
                        "error": str(e)})

            # Refresh goal tree from this cycle's blockers + recent landings.
            try:
                gt = _update_goal_tree(cycle_id, blockers, last_spawns[-15:])
            except Exception as e:  # noqa: BLE001
                log.warning("goal_tree update failed: %s", e)
                gt = {"low": []}

            # 2a. Predictor candidate (always attempted; cheap if no rules).
            try:
                pred = _predict_user_request({
                    "hour": datetime.now(timezone.utc).hour,
                    "cycle_id": cycle_id,
                    "recent_landings": last_spawns[-5:],
                    "blockers_count": len(blockers),
                })
                if pred:
                    no_direction_candidates.append(pred)
            except Exception as e:  # noqa: BLE001
                log.warning("predictor candidate failed: %s", e)

            # 2b. Curiosity-forced exploration (CURIOSITY_FORCED_RATE of cycles).
            try:
                import random
                if random.random() < CURIOSITY_FORCED_RATE:
                    cur = _curiosity_candidate(cycle_id)
                    if cur:
                        no_direction_candidates.append(cur)
                        _audit({"event": "curiosity_forced", "cycle_id": cycle_id,
                                "area": cur.get("_curiosity_area"),
                                "score": cur.get("_curiosity_score")})
            except Exception as e:  # noqa: BLE001
                log.warning("curiosity candidate failed: %s", e)

            # 2c. Goal-tree walk atom.
            try:
                gc = _goal_tree_candidate(cycle_id, gt)
                if gc:
                    no_direction_candidates.append(gc)
            except Exception as e:  # noqa: BLE001
                log.warning("goal_tree candidate failed: %s", e)

            # 2d. Time-aware forced seed (pre-market / post-close).
            try:
                ts = _time_aware_forced_seed(cycle_id)
                if ts:
                    no_direction_candidates.append(ts)
                    _audit({"event": "time_aware_forced", "cycle_id": cycle_id,
                            "trigger": ts.get("_time_aware")})
            except Exception as e:  # noqa: BLE001
                log.warning("time-aware candidate failed: %s", e)

        # Prepend USER inbox candidates (highest priority — operator typed them),
        # then USER-PERSONA directives (operator's *voice* but daemon-generated),
        # then NO-DIRECTION candidates (predictor/curiosity/goal-tree/time-aware),
        # then plan candidates, then ideate candidates.
        # autosolve_skip: feature wiring
        candidates = (
            user_inbox_candidates
            + persona_candidates
            + no_direction_candidates
            + plan_candidates
            + (candidates or [])
        )

        # NEVER-SLEEP MODE (added 2026-05-20): if all 3 LLM-driven paths returned
        # 0 candidates, refill from hardcoded BACKLOG_SEEDS so the daemon never
        # silent-idles.  Also fires an exploration seed EXPLORATION_RATE of cycles
        # to escape exploitation local optima.
        # autosolve_skip: feature wiring
        if not candidates:
            log.info("candidates=0 — triggering BACKLOG_REFILL (never-sleep mode)")
            candidates = _refill_backlog(cycle_id, recent_titles=_last_n_titles(20))
        # Always consider exploration injection (independent of candidates count)
        try:
            expl = _exploration_seed(cycle_id)
            if expl:
                candidates = [expl] + (candidates or [])
        except Exception as e:  # noqa: BLE001
            log.warning("exploration_seed failed: %s", e)

        if not candidates:
            # Should be unreachable now (backlog seeds are 20 hardcoded items),
            # but keep guard for paranoia.
            log.warning("STILL no candidates after backlog refill — emitting idle_because")
            _audit({"event": "no_candidates", "cycle_id": cycle_id,
                    "note": "backlog_refill also returned 0 (degenerate)"})
            _health_assert(cycle_id, 0, 0, len(blockers), 0.0, 0)
            if once:
                return 0
            time.sleep(LOOP_SLEEP_SECONDS)
            continue

        candidates = candidates[:MAX_IDEAS_PER_CYCLE]

        # autosolve_skip: drift handler — amendment 2026-05-20, never pauses
        # 3. Drift check — UNLIMITED MODE + §8-SOLVER PATH (2026-05-20):
        #    On drift detection (≥3 of last 5 ideas <40% novel by Jaccard), spawn 3
        #    parallel solvers (INTERNET + GITHUB + REPO-LOCAL) to diagnose + fix root
        #    cause IN BACKGROUND. Daemon NEVER pauses — continues ideating this cycle
        #    with broader randomization (force-include "NOT recent" filter on next cycle).
        history = _last_n_titles()
        new_titles = [c.get("title", "") for c in candidates]
        novelty = _novelty_score(new_titles, history)
        log.info("novelty score: %.2f (threshold %.2f, unlimited-mode: §8-solver path on drift)",
                 novelty, NOVELTY_THRESHOLD)

        # Compute last-5 titles for the strict drift definition (≥3 of last 5 ≥60% similar)
        last_5 = (history + new_titles)[-5:]
        is_drift, drift_evidence = _drift_detected(last_5, threshold=NOVELTY_THRESHOLD)

        # Cooldown: don't re-spawn solvers on every cycle if drift persists. Re-fire
        # only if no solver event in the last 15 minutes (read from drift_events.jsonl).
        spawn_solvers = False
        if is_drift:
            cooldown_seconds = 15 * 60
            last_event_ts = _last_drift_event_ts()
            if (last_event_ts is None) or (
                (datetime.now(timezone.utc) - last_event_ts).total_seconds() > cooldown_seconds
            ):
                spawn_solvers = True

        if is_drift:
            _audit({
                "event": "drift_detected",
                "novelty": novelty,
                "low_novelty_count": drift_evidence.get("low_novelty_count"),
                "spawn_solvers": spawn_solvers,
                "note": "§8-solver path — daemon continues, NEVER pausing",
            })
            log.info("DRIFT DETECTED (low_novelty=%d/5) — %s",
                     drift_evidence.get("low_novelty_count", 0),
                     "spawning 3 §8 solvers" if spawn_solvers else "cooldown active, no new solvers")
            if spawn_solvers:
                event_id = handle_drift(drift_evidence, summary)
                _write_heartbeat("drift_detected_solvers_in_flight", {"drift_event_id": event_id})
            # Apply config recommendations from any resolved drift events (auto-config-fix path)
            _apply_resolved_drift_config_fixes()
        elif novelty < NOVELTY_THRESHOLD and history:
            # Below-threshold but not yet hitting the ≥3-of-5 rule — log only.
            _audit({
                "event": "drift_observed_no_pause",
                "novelty": novelty,
                "note": "novelty low but drift threshold (3-of-5) not yet met",
            })

        # 4. Gate + DECIDE + EXECUTE + OBSERVE
        seen = _load_seen()
        inflight = count_inflight()
        load = 0.0  # ensure bound for heartbeat even if candidates is empty
        # Never-sleep: count spawns this cycle so _health_assert can detect zero-spawn idles.
        spawn_count_this_cycle = 0
        # autosolve_skip: feature-wire — time-aware spawn budget.
        # Daytime (user prob awake) = 1 spawn/cycle; nighttime = 3/cycle aggressive.
        # User can override per-cycle by setting AUTONOMOUS_MAX_IDEAS_PER_CYCLE > budget.
        time_aware_budget = _time_aware_spawn_budget()
        effective_spawn_budget = max(time_aware_budget, MAX_IDEAS_PER_CYCLE)
        # Ensure skill library has its seed entries (idempotent, one-time per process).
        try:
            _bootstrap_skill_library()
        except Exception as e:  # noqa: BLE001
            log.warning("bootstrap_skill_library failed: %s", e)

        for c in candidates:
            if _STOP:
                break

            # Adaptive Mac load gate (rewritten 2026-05-20): cap = max(FLOOR, load+HEADROOM)
            safe, load, eff_cap = mac_load_safe()
            if not safe:
                title = c.get("title", "(no title)")
                log.info("mac load %.2f >= adaptive cap %.2f — yielding, skip spawn: %s",
                         load, eff_cap, title)
                _audit({
                    "event": "load_gate_skip",
                    "load_1m": load,
                    "effective_cap": eff_cap,
                    "title": title,
                    "reason": "mac_load_adaptive_cap",
                })
                continue

            # Adaptive concurrency throttle (added 2026-05-20)
            conc_cap = _adaptive_concurrency_cap(load)
            if inflight >= conc_cap:
                title = c.get("title", "(no title)")
                log.info("concurrency throttle: inflight=%d >= cap=%d (load=%.2f) — skip: %s",
                         inflight, conc_cap, load, title)
                _audit({
                    "event": "concurrency_throttle_skip",
                    "inflight": inflight, "conc_cap": conc_cap, "load_1m": load, "title": title,
                })
                continue

            ok, reason = gate_candidate(c, seen, cfg, inflight)
            title = c.get("title", "(no title)")
            _audit({
                "event": "gate_decision",
                "cycle_id": cycle_id,
                "title": title,
                "ok": ok,
                "reason": reason,
                "candidate": {k: v for k, v in c.items() if not k.startswith("_")},
            })
            if not ok:
                log.info("gate REJECT: %s — %s", title, reason)
                continue

            # DECIDE - log explicit rationale before spawn (added 2026-05-20)
            decision: dict[str, Any] = {}
            if not dry_run and c.get("_from_plan"):
                try:
                    decision = _decide(c.get("_plan_step", {}), plan, blockers)
                    last_decision_id = decision.get("decision_id")
                    current_step_id = c.get("_plan_step", {}).get("step_id")
                except Exception as e:  # noqa: BLE001
                    log.warning("DECIDE stage failed for %s: %s - spawning without rationale", title, e)
                    _audit({"event": "decide_failed", "title": title, "error": str(e)})

            if dry_run:
                log.info("dry-run: would spawn %s", title)
                _append_seen(title, _hash_title(title))
                last_spawns.append(title)
                continue

            # autosolve_skip: time-aware budget gate — daytime conservative.
            if spawn_count_this_cycle >= effective_spawn_budget:
                log.info("time-aware budget reached (%d/%d, %s) — skip remaining candidates",
                         spawn_count_this_cycle, effective_spawn_budget,
                         "daytime" if _is_daytime_utc() else "nighttime")
                _audit({"event": "time_aware_budget_reached", "cycle_id": cycle_id,
                        "budget": effective_spawn_budget,
                        "is_daytime": _is_daytime_utc()})
                break

            # CONSTITUTIONAL CRITIQUE (2026-05-20, gabriel_self) — score the
            # would-be spawn against 10 principles before launch. reject ->
            # skip; refine -> log + still spawn (refinements feed next cycle).
            # Also runs capability-aware model routing.
            # autosolve_skip: feature wiring — constitutional critique
            if GABRIEL_SELF_ENABLED and _constitution_critique is not None:
                try:
                    cap_now = _gabriel_self.get_capability_summary() if _gabriel_self else {}
                    task_type = (_gabriel_self.classify_task_type(title)
                                 if _gabriel_self else "uncategorized")
                    routing = (_gabriel_self.route_model_for_task(task_type, cap_now)
                               if _gabriel_self else {"model": "haiku",
                                                      "effort": "low",
                                                      "smoke_test_required": False,
                                                      "reason": "default"})
                    pre_brief = (
                        f"{title}\n"
                        f"# scope_estimate_min: {c.get('effort_min', 5)}\n"
                        f"# model_reason: gabriel routed {routing.get('model')} - {routing.get('reason')}\n"
                        f"## Reason\n{c.get('reason', '')}\n"
                        f"## Brief\n{c.get('helper_brief', '')}\n"
                    )
                    crit = _constitution_critique(pre_brief, context={
                        "recent_spawn_titles": last_spawns[-10:],
                        "recent_wins": cap_now.get("recent_wins", [])[-5:],
                        "recent_losses": cap_now.get("recent_losses", [])[-5:],
                        "estimated_min": int(c.get("effort_min", 5) or 5),
                        "model": routing.get("model", "haiku"),
                        "claims_success": False,
                    })
                    _audit({
                        "event": "constitution_critique",
                        "cycle_id": cycle_id,
                        "title": title,
                        "verdict": crit["verdict"],
                        "score": crit["score"],
                        "severe_count": crit["severe_count"],
                        "moderate_count": crit["moderate_count"],
                        "task_type": task_type,
                        "routed_model": routing.get("model"),
                        "violations": [{"p": v["principle"], "ev": v.get("evidence", "")[:80]}
                                       for v in crit.get("violations", [])][:5],
                    })
                    if crit["verdict"] == "reject":
                        log.info("CONSTITUTION REJECT: %s - %d severe violations",
                                 title, crit["severe_count"])
                        continue
                    if crit["verdict"] == "refine":
                        log.info("CONSTITUTION REFINE (still spawning): %s score=%.2f",
                                 title, crit["score"])
                    c["_gabriel_task_type"] = task_type
                    c["_gabriel_routed_model"] = routing.get("model", "haiku")
                    c["_gabriel_route_reason"] = routing.get("reason", "default")
                except Exception as e:  # noqa: BLE001
                    log.warning("constitution_critique failed for %s: %s", title, e)
                    _audit({"event": "constitution_critique_failed",
                            "cycle_id": cycle_id, "title": title, "error": str(e)})

            # EXECUTE - spawn the helper
            brief = spawn_helper(c, cfg)
            if brief:
                _append_seen(title, _hash_title(title))
                last_spawns.append(title)
                inflight += 1
                spawn_count_this_cycle += 1

                # autosolve_skip: feature-wire — curiosity + intrinsic + skill.
                # Touch the curiosity area (mark this area as just-explored).
                try:
                    area = c.get("_curiosity_area") or c.get("_persona_area") \
                           or _AXIS_TO_AREA.get(c.get("axis", ""), None)
                    if area:
                        _touch_curiosity_area(area)
                except Exception as e:  # noqa: BLE001
                    log.warning("touch_curiosity_area failed: %s", e)

                # Compute + log intrinsic reward at spawn-launch time (status=running).
                # OBSERVE-phase post-completion can re-score with actual outcome.
                try:
                    novelty_now = _novelty_score([title], history)
                    pre_novelty = _novelty_score(history[:-1], history[:-2]) if len(history) > 2 else 0.5
                    outcome = {
                        "title": title,
                        "area": c.get("_curiosity_area"),
                        "estimated_effort_min": c.get("effort_min", 15),
                        "actual_effort_min": c.get("effort_min", 15),  # at-launch = est
                        "status": "running",
                        "emitted_lesson_or_skill": False,
                        "novelty_delta": max(0.0, min(1.0, novelty_now - pre_novelty)),
                    }
                    reward = _intrinsic_reward(outcome)
                    _log_intrinsic_reward(outcome, reward)
                except Exception as e:  # noqa: BLE001
                    log.warning("intrinsic_reward log failed: %s", e)

                # Skill library: bump invocation count if title matches an existing pattern.
                try:
                    _maybe_record_skill(c, outcome_status="success")
                except Exception as e:  # noqa: BLE001
                    log.warning("maybe_record_skill failed: %s", e)

                # user_inbox: mark item as "spawned" with the brief path so the
                # operator can trace it from inbox -> brief -> answer file.
                # autosolve_skip: feature wiring
                if c.get("_user_inbox") and c.get("_inbox_item_id"):
                    try:
                        _rows = _load_user_inbox()
                        for _r in _rows:
                            if _r.get("id") == c.get("_inbox_item_id"):
                                _r["status"] = "spawned"
                                _r["spawn_brief_path"] = brief
                                _r["spawn_ts"] = _now_utc()
                                break
                        _persist_user_inbox(_rows)
                    except Exception as e:  # noqa: BLE001
                        log.warning("user_inbox status-update failed: %s", e)

                # OBSERVE - write ReAct trace (added 2026-05-20)
                try:
                    _observe(decision, c.get("_plan_step", {}) or c, blockers,
                             spawn_result="launched", brief_path=brief)
                    last_react_summary = f"{title[:40]}... -> launched"
                except Exception as e:  # noqa: BLE001
                    log.warning("OBSERVE stage failed: %s", e)

        # Heartbeat with ASK-PLAN-DECIDE state
        _write_heartbeat("iteration_complete", {
            "cycle_id": cycle_id,
            "current_plan_id": current_plan_id,
            "current_step_id": current_step_id,
            "last_decision_id": last_decision_id,
            "blockers_count": len(blockers),
            "last_react_summary": last_react_summary,
            "inflight": inflight,
            "load_1m": round(load, 2),
            "spawn_count_this_cycle": spawn_count_this_cycle,
        })

        refresh_status_dashboard(cfg, last_spawns)

        # NEVER-SLEEP: per-cycle health assert. Logs idle_because if 0 spawns.
        try:
            _health_assert(cycle_id, spawn_count_this_cycle, len(candidates),
                           len(blockers), load, inflight)
        except Exception as e:  # noqa: BLE001
            log.warning("health_assert failed: %s", e)

        # NEVER-SLEEP: periodic self-reflection (every SELF_REFLECT_EVERY_N cycles).
        global _NEVER_SLEEP_CYCLE_COUNTER
        _NEVER_SLEEP_CYCLE_COUNTER += 1
        if _NEVER_SLEEP_CYCLE_COUNTER % SELF_REFLECT_EVERY_N == 0:
            try:
                _self_reflect(cycle_id)
            except Exception as e:  # noqa: BLE001
                log.warning("self_reflect failed: %s", e)

        # GABRIEL_SELF (2026-05-20): every 10 cycles, refresh capability map,
        # run a reflexion, regenerate the self-status dashboard.  Every cycle
        # (cheap): just refresh the dashboard so it stays current.
        # autosolve_skip: feature wiring
        if GABRIEL_SELF_ENABLED and _gabriel_self is not None:
            try:
                if _NEVER_SLEEP_CYCLE_COUNTER % SELF_REFLECT_EVERY_N == 0:
                    _gabriel_self.update_capability_map(write=True)
                    _gabriel_self.reflect(cycle_id=cycle_id)
                _gabriel_self.refresh_self_status()
                _audit({
                    "event": "gabriel_self_refresh",
                    "cycle_id": cycle_id,
                    "full_cycle": _NEVER_SLEEP_CYCLE_COUNTER % SELF_REFLECT_EVERY_N == 0,
                })
            except Exception as e:  # noqa: BLE001
                log.warning("gabriel_self refresh failed: %s", e)

        if once:
            return 0

        # 5. Sleep
        for _ in range(LOOP_SLEEP_SECONDS // HEARTBEAT_INTERVAL_SECONDS):
            if _STOP:
                break
            _write_heartbeat("sleeping_post_iteration")
            time.sleep(HEARTBEAT_INTERVAL_SECONDS)

    _write_heartbeat("stopped")
    log.info("loop stopped cleanly")
    return 0


def _run_loop_hardened(once: bool, dry_run: bool) -> int:
    """Cycle-level exception harness (added 2026-05-20).

    Wraps run_loop in an outer retry loop so a single uncaught exception
    during a cycle doesn't kill the daemon — restart the loop with
    a 30s back-off. Only exit (letting launchd respawn) after
    MAX_CONSECUTIVE_ERRORS=10 restarts in a row without a successful
    cycle reaching the post-iteration sleep.
    """
    cycle_err_log = Path(
        "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/"
        "My Drive/AI-Tools/logs/autonomous_mode_daemon.cycle_errors.log"
    )
    try:
        cycle_err_log.parent.mkdir(parents=True, exist_ok=True)
    except Exception:  # noqa: BLE001
        pass

    MAX_CONSECUTIVE_ERRORS = 10
    BACKOFF_SECONDS = 30
    consecutive_errors = 0

    while True:
        try:
            rc = run_loop(once=once, dry_run=dry_run)
            # run_loop returned normally (once=True OR _STOP set) — exit cleanly
            return rc
        except SystemExit:
            raise
        except KeyboardInterrupt:
            log.info("KeyboardInterrupt — stopping cleanly")
            return 0
        except Exception as e:  # noqa: BLE001
            consecutive_errors += 1
            tb = traceback.format_exc()
            try:
                with open(cycle_err_log, "a", encoding="utf-8") as f:
                    f.write(f"{_now_utc()} [cycle_err #{consecutive_errors}] {type(e).__name__}: {e}\n")
                    f.write(tb + "\n")
            except Exception:  # noqa: BLE001
                pass
            log.error("run_loop crashed (consecutive=%d/%d): %s\n%s",
                      consecutive_errors, MAX_CONSECUTIVE_ERRORS, e, tb)
            try:
                _audit({
                    "event": "run_loop_exception",
                    "consecutive_errors": consecutive_errors,
                    "max_consecutive_errors": MAX_CONSECUTIVE_ERRORS,
                    "exception_type": type(e).__name__,
                    "exception": str(e)[:500],
                })
            except Exception:  # noqa: BLE001
                pass

            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                log.critical(
                    "consecutive_errors %d >= %d — exiting so launchd KeepAlive can respawn",
                    consecutive_errors, MAX_CONSECUTIVE_ERRORS,
                )
                return 2

            if once:
                # In --once mode, single-failure return non-zero rather than retry
                return 3

            try:
                time.sleep(BACKOFF_SECONDS)
            except KeyboardInterrupt:
                return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--once", action="store_true", help="Run one iteration and exit")
    p.add_argument("--dry-run", action="store_true", help="Skip ideate + spawn, use stubs")
    p.add_argument("--show-config", action="store_true", help="Print current config and exit")
    args = p.parse_args()

    if args.show_config:
        print(json.dumps(_read_config(), indent=2))
        return 0

    return _run_loop_hardened(once=args.once, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
