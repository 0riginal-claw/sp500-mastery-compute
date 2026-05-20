"""
auto_action_daemon.py — Reads OpenClaw/DeepSeek recommendations and automatically
queues/executes new work.

Runs every 5 minutes via LaunchAgent (com.zg.auto_action).

Cycle logic:
  1. Read overseer/recommendations.json
  2. Read feature_discovery/inbox/queue.json
  3. Read broadcasts/live_stream.md
  4. Pick top-N ACTIONABLE recommendations not yet in auto_action/executed.jsonl
  5. Match to executable patterns and dispatch
  6. Log every action to auto_action/executed.jsonl

Safety rails:
  - Max 5 executions per cycle
  - Skip if same action executed within last 30 min
  - Only run scripts already present in scripts/
  - Never delete files, never overwrite scripts
  - Feature additions go to feature_queue.txt for manual review

Pacing integration (ADDITIVE — 2026-05-16):
  - Loads pacing policy from ceo_orchestrator_daemon.get_pacing_policy() each cycle.
  - In "emergency" regime, non-essential task types are skipped.
  - Logs a pacing_routing_decision event to the event bus whenever the pacing
    regime changes a model or defers a dispatch.

Router integration (ADDITIVE — 2026-05-16):
  - All model selection now flows through unified_model_router.route() when the
    router is importable.  The chosen model is logged in executed.jsonl and
    published to the event bus; the backtest script commands are not changed
    (those scripts do not accept a --model flag).
  - Use --router-policy flag to inspect routing decisions without running work.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Pacing policy integration (best-effort; never crash if import fails)
# ---------------------------------------------------------------------------
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from ceo_orchestrator_daemon import (  # type: ignore[import]
        get_pacing_policy as _get_pacing_policy,
        PacingPolicy as _PacingPolicy,
    )
    _PACING_AVAILABLE = True
except Exception as _pacing_import_err:
    _PACING_AVAILABLE = False
    _get_pacing_policy = None  # type: ignore[assignment]
    _PacingPolicy = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Unified model router integration (best-effort; never crash if import fails)
# ---------------------------------------------------------------------------
try:
    _scripts_dir = str(Path(__file__).parent)
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
    from unified_model_router import (  # type: ignore[import]
        route as _router_route,
        TaskRequest as _RouterTaskRequest,
        BackendChoice as _RouterBackendChoice,
    )
    _ROUTER_AVAILABLE = True
except Exception as _router_import_err:
    _ROUTER_AVAILABLE = False
    _router_route = None  # type: ignore[assignment]
    _RouterTaskRequest = None  # type: ignore[assignment]
    _RouterBackendChoice = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Event bus for pacing routing decisions (best-effort)
# ---------------------------------------------------------------------------
try:
    from event_bus import EventBus as _EventBus  # type: ignore[import]
    _EB = _EventBus
except Exception:
    _EB = None  # type: ignore[assignment]

# Action types that are considered non-essential and are deferred in
# "emergency" pacing regime.
_NON_ESSENTIAL_ACTION_TYPES = frozenset(["feature", "investigate"])

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORK = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools/s&p500-ticker-mastery"
)
SCRIPTS_DIR = WORK / "scripts"
AUTO_ACTION_DIR = WORK / "auto_action"
EXECUTED_LOG = AUTO_ACTION_DIR / "executed.jsonl"
FEATURE_QUEUE = AUTO_ACTION_DIR / "feature_queue.txt"
DAEMON_LOG = AUTO_ACTION_DIR / "daemon.log"

RECS_FILE = WORK / "overseer" / "recommendations.json"
FD_QUEUE = WORK / "feature_discovery" / "inbox" / "queue.json"
BROADCAST = WORK / "broadcasts" / "live_stream.md"

BACKTESTS_V8_DIR = WORK / "backtests_xgb_v8"

PYTHON = "/Users/orginal/.venvs/sp500-mastery/bin/python"

# Safety limits
MAX_EXECUTIONS_PER_CYCLE = 5
COOLDOWN_SECONDS = 1800  # 30 min per identical action

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
AUTO_ACTION_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(DAEMON_LOG),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("auto_action_daemon")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Any:
    """Load JSON file; return None on failure."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        log.warning("Could not load %s: %s", path, exc)
        return None


def _load_executed() -> list[dict]:
    """Return all records in executed.jsonl."""
    records: list[dict] = []
    if not EXECUTED_LOG.exists():
        return records
    with open(EXECUTED_LOG, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _append_executed(record: dict) -> None:
    """Append a single execution record to executed.jsonl."""
    with open(EXECUTED_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def _cooldown_ok(action_key: str, executed: list[dict]) -> bool:
    """True if no identical action_key in the last COOLDOWN_SECONDS."""
    now_ts = time.time()
    for rec in reversed(executed):
        if rec.get("action_key") == action_key:
            rec_ts = rec.get("epoch_ts", 0)
            age = now_ts - rec_ts
            if age < COOLDOWN_SECONDS:
                log.info("  Cooldown active for '%s' (%.0f s ago)", action_key, age)
                return False
    return True


def _script_exists(name: str) -> bool:
    """True if scripts/<name> exists on disk — never run arbitrary paths."""
    candidate = SCRIPTS_DIR / name
    return candidate.is_file()


# ---------------------------------------------------------------------------
# Pacing helpers (ADDITIVE — 2026-05-16)
# ---------------------------------------------------------------------------


def _load_pacing_policy() -> Any:
    """Load the current pacing policy via ceo_orchestrator_daemon.

    Returns the PacingPolicy dataclass on success, or None if the import is
    not yet available (the sibling daemon is still being built).  Failures are
    always non-fatal.
    """
    if not _PACING_AVAILABLE or _get_pacing_policy is None:
        return None
    try:
        return _get_pacing_policy()
    except Exception as exc:
        log.debug("Could not load pacing policy: %s", exc)
        return None


def _should_defer_for_pacing(
    action: dict,
    policy: Any,
) -> bool:
    """Return True if *action* should be skipped due to pacing emergency.

    Only applies when regime == "emergency" AND the action type is non-essential.
    Also publishes a pacing_routing_decision event to the event bus.

    Args:
        action: The classified action dict from _classify_action().
        policy: PacingPolicy from _load_pacing_policy(), or None.

    Returns:
        True if the action should be deferred.
    """
    if policy is None:
        return False
    if getattr(policy, "regime", "on") != "emergency":
        return False
    action_type = action.get("action_type", "")
    if action_type not in _NON_ESSENTIAL_ACTION_TYPES:
        return False

    log.info(
        "[PACING] DEFERRED action_key=%s action_type=%s — deferred due to pacing emergency",
        action.get("action_key"),
        action_type,
    )
    _publish_pacing_routing_event(
        task_id=action.get("action_key", "unknown"),
        original_model="sonnet",
        new_model="sonnet",
        regime="emergency",
        deferred=True,
    )
    return True


def _publish_pacing_routing_event(
    task_id: str,
    original_model: str,
    new_model: str,
    regime: str,
    deferred: bool = False,
) -> None:
    """Publish pacing_routing_decision to the event bus (best-effort).

    Args:
        task_id: Unique key for the action being routed.
        original_model: Model that would be used without pacing.
        new_model: Model chosen after applying pacing policy.
        regime: Current pacing regime string.
        deferred: True if the task was skipped entirely.
    """
    if _EB is None:
        return
    try:
        _EB.publish_from_anywhere(
            "pacing_routing_decision",
            {
                "task_id": task_id,
                "original_model": original_model,
                "new_model": new_model,
                "regime": regime,
                "deferred": deferred,
                "routing_influenced": (original_model != new_model) or deferred,
            },
            source="auto_action_daemon",
        )
    except Exception as exc:
        log.debug("Event bus publish failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Router-based model selection (ADDITIVE — 2026-05-16)
# ---------------------------------------------------------------------------

# Maps auto_action action_type values to unified_model_router ComplexityLevel.
_ACTION_TYPE_TO_COMPLEXITY: dict[str, str] = {
    "retrain": "coding",
    "backtest": "mechanical",
    "investigate": "reasoning",
    "feature": "coding",
}

# Default complexity if action_type not mapped.
_DEFAULT_COMPLEXITY = "coding"


def _pick_model_via_router(action: dict) -> tuple[str, str | None]:
    """Select a model for the action using unified_model_router.route().

    Returns the chosen model name (haiku/sonnet/opus or a full model id) and
    the router's reason string.  Falls back to "sonnet" (safe default) if the
    router is not available or raises.

    Args:
        action: Classified action dict from _classify_action().

    Returns:
        Tuple of (model_name, reason_string_or_None).
    """
    if not _ROUTER_AVAILABLE or _router_route is None or _RouterTaskRequest is None:
        return "sonnet", None  # safe default when router not yet available

    action_type = action.get("action_type", "backtest")
    complexity = _ACTION_TYPE_TO_COMPLEXITY.get(action_type, _DEFAULT_COMPLEXITY)
    description = action.get("description", action.get("action_key", "auto_action task"))

    try:
        req = _RouterTaskRequest(
            prompt=description,
            complexity=complexity,  # type: ignore[arg-type]
            independence_required=False,
            context_tokens_estimate=0,
            cost_ceiling_usd=0.10,
            deadline_seconds=60.0,
            allow_local=True,
        )
        choice = _router_route(req)
        return choice.model, choice.reason
    except Exception as exc:
        log.debug("unified_model_router.route() failed (%s) — defaulting to sonnet", exc)
        return "sonnet", None


# ---------------------------------------------------------------------------
# Recommendation collection
# ---------------------------------------------------------------------------

def _collect_recommendations() -> list[str]:
    """
    Gather recommendation strings from all three sources.
    Returns a list of raw text items, highest-priority first.
    """
    recs: list[str] = []

    # 1. overseer/recommendations.json
    rdata = _load_json(RECS_FILE)
    if rdata:
        top_actions = rdata.get("recommendations", {}).get("top_actions", [])
        for item in top_actions:
            if isinstance(item, str) and item.strip():
                recs.append(item.strip())

    # 2. broadcasts/live_stream.md — pull first 10 tickers mentioned alongside
    #    explicit action verbs (re-run, retrain, backtest, investigate)
    try:
        text = BROADCAST.read_text(encoding="utf-8", errors="ignore")
        for match in re.finditer(
            r"(?:re-run|retrain|backtest|test|investigate)\s+(?:the\s+)?v\d+\s+"
            r"(?:ensemble\s+)?on\s+([A-Z]{1,5})",
            text,
            re.IGNORECASE,
        ):
            ticker = match.group(1).upper()
            verb_raw = match.group(0).split()[0].lower()
            verb = "retrain" if verb_raw in ("retrain", "re-run") else verb_raw
            recs.append(f"{verb} ticker {ticker}")
    except Exception as exc:
        log.debug("Broadcast parse error: %s", exc)

    # 3. feature_discovery/inbox/queue.json — only "add feature" style
    fd_data = _load_json(FD_QUEUE)
    if isinstance(fd_data, list):
        for item in fd_data:
            if isinstance(item, dict):
                name = item.get("name", "")
                impact = str(item.get("impact", "")).lower()
                if name and "high" in impact:
                    recs.append(f"add feature {name}")

    return recs


# ---------------------------------------------------------------------------
# Pattern matching → action dispatch
# ---------------------------------------------------------------------------

def _extract_ticker(text: str) -> str | None:
    """Pull first S&P 500-style ticker (1-5 uppercase letters, possibly with dot) from text."""
    match = re.search(r"\b([A-Z]{1,5}(?:\.[A-B])?)\b", text)
    return match.group(1) if match else None


def _classify_action(rec_text: str) -> dict | None:
    """
    Map recommendation text to a concrete action dict, or None if not executable.

    Returns:
        {
            "action_key": str,         # dedup / cooldown key
            "action_type": str,        # "backtest" | "retrain" | "investigate" | "feature"
            "description": str,        # human-readable
            "cmd": list[str] | None,   # subprocess command (None for feature additions)
            "logfile": Path | None,
        }
    """
    text_lower = rec_text.lower()

    # --- "add feature Z" → queue for manual review (never auto-build) ---
    if re.search(r"\badd\s+feature\b", text_lower):
        feature_name = rec_text.strip().replace("add feature ", "").strip()
        return {
            "action_key": f"feature:{feature_name}",
            "action_type": "feature",
            "description": f"Queue feature for manual review: {feature_name}",
            "cmd": None,
            "logfile": None,
        }

    # --- "retrain ticker X" → run v8 with --sweep-threshold ---
    if re.search(r"\bretrain\b", text_lower) or re.search(r"\bre-?train\b", text_lower):
        ticker = _extract_ticker(rec_text)
        if not ticker:
            return None
        script = "backtest_xgb_v8.py"
        if not _script_exists(script):
            log.warning("Script %s not found — skipping retrain for %s", script, ticker)
            return None
        out_dir = BACKTESTS_V8_DIR / ticker
        out_dir.mkdir(parents=True, exist_ok=True)
        logfile = AUTO_ACTION_DIR / f"retrain_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        return {
            "action_key": f"retrain:{ticker}",
            "action_type": "retrain",
            "description": f"Retrain v8 with --sweep-threshold on {ticker}",
            "cmd": [
                PYTHON,
                str(SCRIPTS_DIR / script),
                "--ticker", ticker,
                "--output-dir", str(out_dir),
                "--sweep-threshold",
            ],
            "logfile": logfile,
        }

    # --- "test ticker X with strategy Y" / "backtest X" ---
    if re.search(r"\bbacktest\b|\btest\s+ticker\b", text_lower):
        ticker = _extract_ticker(rec_text)
        if not ticker:
            return None
        script = "backtest_xgb_v8.py"
        if not _script_exists(script):
            log.warning("Script %s not found — skipping backtest for %s", script, ticker)
            return None
        out_dir = BACKTESTS_V8_DIR / ticker
        out_dir.mkdir(parents=True, exist_ok=True)
        logfile = AUTO_ACTION_DIR / f"backtest_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        return {
            "action_key": f"backtest:{ticker}",
            "action_type": "backtest",
            "description": f"Backtest v8 on {ticker}",
            "cmd": [
                PYTHON,
                str(SCRIPTS_DIR / script),
                "--ticker", ticker,
                "--output-dir", str(out_dir),
            ],
            "logfile": logfile,
        }

    # --- "investigate ticker X" → feature importance dump ---
    if re.search(r"\binvestigate\b|\binference\b", text_lower):
        ticker = _extract_ticker(rec_text)
        if not ticker:
            return None
        script = "backtest_xgb_v8.py"
        if not _script_exists(script):
            log.warning("Script %s not found — skipping investigate for %s", script, ticker)
            return None
        out_dir = BACKTESTS_V8_DIR / ticker
        out_dir.mkdir(parents=True, exist_ok=True)
        logfile = AUTO_ACTION_DIR / f"investigate_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        # Investigate = run with default threshold (no sweep), output lands in out_dir
        return {
            "action_key": f"investigate:{ticker}",
            "action_type": "investigate",
            "description": f"Investigate (feature importance dump) v8 on {ticker}",
            "cmd": [
                PYTHON,
                str(SCRIPTS_DIR / script),
                "--ticker", ticker,
                "--output-dir", str(out_dir),
            ],
            "logfile": logfile,
        }

    return None


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def _execute_action(action: dict) -> str:
    """
    Execute the action.  Returns status string: "dispatched" | "queued" | "skipped".
    For subprocess actions, fires in background (non-blocking).
    """
    if action["action_type"] == "feature":
        # Log to feature_queue.txt
        with open(FEATURE_QUEUE, "a", encoding="utf-8") as fh:
            fh.write(
                f"{_utcnow()} | {action['description']}\n"
            )
        log.info("  Feature queued for manual review: %s", action["description"])
        return "queued"

    cmd = action["cmd"]
    logfile: Path = action["logfile"]

    # Safety: every token in cmd must not reference a path outside WORK or system dirs
    for token in cmd:
        if token.startswith("/") and not (
            token.startswith(str(WORK))
            or token.startswith(str(SCRIPTS_DIR))
            or token == PYTHON
        ):
            log.error("  SAFETY: refusing command with external path token: %s", token)
            return "skipped_safety"

    log.info("  Dispatching: %s", " ".join(str(t) for t in cmd))
    log.info("  Log → %s", logfile)

    with open(logfile, "w", encoding="utf-8") as out_fh:
        proc = subprocess.Popen(
            cmd,
            stdout=out_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,   # detach so daemon doesn't block
        )
    log.info("  PID %d launched (background)", proc.pid)
    return f"dispatched:pid={proc.pid}"


# ---------------------------------------------------------------------------
# Main cycle
# ---------------------------------------------------------------------------

def run_cycle() -> None:
    log.info("=== auto_action_daemon cycle start ===")

    # ── Pacing policy (ADDITIVE — 2026-05-16) ───────────────────────────────
    pacing_policy = _load_pacing_policy()
    if pacing_policy is not None:
        log.info(
            "[PACING] Policy loaded: regime=%s default=%s",
            getattr(pacing_policy, "regime", "unknown"),
            getattr(pacing_policy, "default_model", "unknown"),
        )
    else:
        log.debug("[PACING] Policy not available — proceeding without pacing constraints")
    # ────────────────────────────────────────────────────────────────────────

    recs = _collect_recommendations()
    log.info("Collected %d raw recommendation strings", len(recs))

    executed = _load_executed()
    already_keyed = {r.get("action_key") for r in executed}

    dispatched_this_cycle = 0

    for rec_text in recs:
        if dispatched_this_cycle >= MAX_EXECUTIONS_PER_CYCLE:
            log.info("Max executions per cycle (%d) reached — stopping.", MAX_EXECUTIONS_PER_CYCLE)
            break

        action = _classify_action(rec_text)
        if action is None:
            log.debug("No pattern match for: %r", rec_text)
            continue

        action_key = action["action_key"]

        # Skip if already executed (persistent log)
        if action_key in already_keyed:
            log.info("  Already executed: %s — skipping", action_key)
            continue

        # Cooldown check
        if not _cooldown_ok(action_key, executed):
            continue

        # ── Pacing emergency deferral (ADDITIVE — 2026-05-16) ───────────────
        if _should_defer_for_pacing(action, pacing_policy):
            record = {
                "rec_text": rec_text,
                "action_key": action_key,
                "action_type": action["action_type"],
                "action_taken": action["description"],
                "status": "deferred:pacing_emergency",
                "ts": _utcnow(),
                "epoch_ts": time.time(),
                "pacing_regime": getattr(pacing_policy, "regime", "unknown"),
                "router_available": _ROUTER_AVAILABLE,
            }
            _append_executed(record)
            executed.append(record)
            already_keyed.add(action_key)
            continue
        # ────────────────────────────────────────────────────────────────────

        # ── Router model selection (ADDITIVE — 2026-05-16) ──────────────────
        router_model, router_reason = _pick_model_via_router(action)
        if _ROUTER_AVAILABLE:
            log.info(
                "[ROUTER] action_key=%s → model=%s reason=%r",
                action_key,
                router_model,
                (router_reason or "")[:100],
            )
        # ────────────────────────────────────────────────────────────────────

        # Execute
        status = _execute_action(action)

        record = {
            "rec_text": rec_text,
            "action_key": action_key,
            "action_type": action["action_type"],
            "action_taken": action["description"],
            "status": status,
            "ts": _utcnow(),
            "epoch_ts": time.time(),
            "router_model": router_model,
            "router_reason": router_reason,
            "router_available": _ROUTER_AVAILABLE,
        }
        # Stamp the pacing regime onto every executed record (ADDITIVE — 2026-05-16)
        if pacing_policy is not None:
            record["pacing_regime"] = getattr(pacing_policy, "regime", "unknown")

        _append_executed(record)
        executed.append(record)          # update in-memory copy for cooldown checks
        already_keyed.add(action_key)   # prevent double-fire within same cycle

        dispatched_this_cycle += 1
        log.info("  Logged: %s → %s", action_key, status)

    log.info(
        "=== cycle complete: %d action(s) dispatched/queued ===",
        dispatched_this_cycle,
    )


# ---------------------------------------------------------------------------
# --router-policy inspector
# ---------------------------------------------------------------------------


def _show_router_policy() -> None:
    """Print what unified_model_router would pick for each action type.

    Iterates over all recognised action_type values, builds a sample action dict
    for each, calls _pick_model_via_router(), and prints the full decision as
    JSON.  Useful for debugging routing behaviour without running real work.
    """
    pacing_policy = _load_pacing_policy()
    regime = getattr(pacing_policy, "regime", "unknown") if pacing_policy else "unknown"

    sample_actions = [
        {
            "action_key": f"sample-{atype}:AAPL",
            "action_type": atype,
            "description": f"Sample {atype} action for --router-policy inspection on AAPL",
        }
        for atype in ["retrain", "backtest", "investigate", "feature"]
    ]

    decisions: list[dict] = []
    for action in sample_actions:
        model, reason = _pick_model_via_router(action)
        decisions.append(
            {
                "action_type": action["action_type"],
                "complexity_mapped": _ACTION_TYPE_TO_COMPLEXITY.get(
                    action["action_type"], _DEFAULT_COMPLEXITY
                ),
                "router_model": model,
                "router_reason": reason,
                "router_available": _ROUTER_AVAILABLE,
            }
        )

    output = {
        "regime": regime,
        "router_available": _ROUTER_AVAILABLE,
        "routing_decisions": decisions,
        "note": (
            "Router is active — model selection flows through unified_model_router.route()"
            if _ROUTER_AVAILABLE
            else "Router NOT imported — default 'sonnet' fallback is active"
        ),
    }
    print(json.dumps(output, indent=2, default=str))


# ---------------------------------------------------------------------------
# Entry point (runs one cycle then exits; LaunchAgent restarts every 5 min)
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="auto_action_daemon.py",
        description="Auto-action daemon for S&P 500 ML Mastery system.",
    )
    parser.add_argument(
        "--router-policy",
        action="store_true",
        help=(
            "Print what unified_model_router would pick for each action type given "
            "current pacing state, then exit.  Useful for debugging routing without "
            "firing real work."
        ),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    _args = _parse_args()

    if _args.router_policy:
        _show_router_policy()
        sys.exit(0)

    run_cycle()
