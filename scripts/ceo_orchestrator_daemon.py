"""
ceo_orchestrator_daemon.py — Top-level orchestration daemon for the S&P 500
ML Mastery system.

Responsibilities
----------------
* Read the pacing policy from dashboard/pacing_state.json (written by the
  sibling usage_pacing_daemon.py) and expose it as ``get_pacing_policy()``.
* Dispatch sub-agent work-items using model selection governed by the current
  pacing regime:
    - mechanical tasks  → policy["mechanical_model"]
    - reasoning tasks   → policy["escalation_model"]
    - everything else   → policy["default_model"]
* In "emergency" regime, defer non-essential tasks rather than spawning them.
* Publish a ``pacing_routing_decision`` event to the event bus every time the
  pacing state influences a routing choice.
* Exposes a --show-policy CLI flag for quick policy inspection.
* Exposes a --router-policy CLI flag to show what unified_model_router would
  pick for a sample TaskRequest given current pacing state (debugging aid).

Router integration (ADDITIVE — 2026-05-16):
  All model selection now flows through unified_model_router.route() when the
  router is importable.  The original select_model() is retained as a fallback
  and is invoked transparently if the import fails.

This module is ADDITIVE — it does not replace overseer_daemon, auto_action_daemon,
proactive_loop_daemon, or any other existing daemon.  It is a coordination layer
on top of them.

Usage
-----
    # Print current pacing policy and exit
    python ceo_orchestrator_daemon.py --show-policy

    # Show what the unified router would choose for each task complexity
    python ceo_orchestrator_daemon.py --router-policy

    # Run one orchestration cycle
    python ceo_orchestrator_daemon.py

    # Run one orchestration cycle in dry-run mode (no spawns, no mutations)
    python ceo_orchestrator_daemon.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

WORK = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools/s&p500-ticker-mastery"
)
SCRIPTS_DIR = WORK / "scripts"
DASHBOARD_DIR = WORK / "dashboard"
PACING_STATE_FILE = DASHBOARD_DIR / "pacing_state.json"
CEO_LOG_PATH = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools/logs/ceo_orchestrator.log"
)
CEO_STATE_FILE = WORK / "overseer" / "ceo_state.json"

PYTHON = "/Users/orginal/.venvs/sp500-mastery/bin/python"

# How old (seconds) pacing_state.json may be before we treat it as stale.
PACING_STALE_THRESHOLD_SECONDS = 1800  # 30 minutes

# ── PARALLEL IMPACT-RANKING (2026-05-17) ──────────────────────────────────
# Each cycle now fires 3 DeepSeek angle-ranking queries concurrently on the
# current task queue, producing impact assessments from 3 perspectives.
# Direct API call (~2s/each), $0.000001 ea — burn liberally.
IMPACT_RANK_ANGLES = [
    "Rank these tasks by EXPECTED MASTERY UNLOCK (which fills the biggest gap in the 502-ticker mission?). Be specific and ruthless.",
    "Rank these tasks by RISK / FAILURE MODE EXPOSURE (which one prevents the biggest hidden bug or look-ahead leak?).",
    "Rank these tasks by COST / EFFORT EFFICIENCY (highest impact per hour of compute or human time).",
]
IMPACT_RANK_TIMEOUT = 30
IMPACT_RANK_MAX_WORKERS = 3

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

CEO_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(CEO_LOG_PATH),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("ceo_orchestrator")

# ---------------------------------------------------------------------------
# Unified model router integration (best-effort — never crash if missing)
# ---------------------------------------------------------------------------

try:
    sys.path.insert(0, str(SCRIPTS_DIR))
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
# Event bus (best-effort)
# ---------------------------------------------------------------------------

try:
    sys.path.insert(0, str(SCRIPTS_DIR))
    from event_bus import EventBus as _EventBus  # type: ignore[import]
    _EB = _EventBus
except Exception:
    _EB = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Wire-candidate emitter (best-effort)
# ---------------------------------------------------------------------------

try:
    sys.path.insert(0, str(SCRIPTS_DIR))
    from wire_candidate import (  # type: ignore[import]
        emit as _wire_emit,
        parse_markdown_blocks as _wire_parse,
    )
    _WIRE_AVAILABLE = True
except Exception:
    _WIRE_AVAILABLE = False
    _wire_emit = None  # type: ignore[assignment]
    _wire_parse = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Pacing regime types
# ---------------------------------------------------------------------------

Regime = Literal["under", "on", "over", "emergency"]
ModelName = Literal["opus", "sonnet", "haiku"]

# Model selection tables per regime.
# Each regime maps task-class -> model name.
_REGIME_MODEL_TABLE: dict[str, dict[str, ModelName]] = {
    "under": {
        # Plenty of headroom — use best models freely.
        "default": "sonnet",
        "escalation": "opus",
        "mechanical": "haiku",
    },
    "on": {
        # Normal operating mode — balanced.
        "default": "sonnet",
        "escalation": "opus",
        "mechanical": "haiku",
    },
    "over": {
        # Over budget — step down one tier everywhere.
        "default": "haiku",
        "escalation": "sonnet",
        "mechanical": "haiku",
    },
    "emergency": {
        # Emergency — absolute minimum; reasoning tasks still need sonnet.
        "default": "haiku",
        "escalation": "sonnet",
        "mechanical": "haiku",
    },
}

# Task kinds that are considered "non-essential" and will be deferred in
# emergency regime.
_NON_ESSENTIAL_TASK_KINDS = frozenset(
    ["proactive_ideation", "feature_discovery", "report_generation", "broadcast"]
)

# ---------------------------------------------------------------------------
# Pacing policy
# ---------------------------------------------------------------------------


@dataclass
class PacingPolicy:
    """Resolved pacing policy for the current cycle.

    Attributes:
        regime: The pacing regime from pacing_state.json (or default fallback).
        default_model: Model to use for unclassified tasks.
        escalation_model: Model to use for reasoning/complex tasks.
        mechanical_model: Model to use for mechanical/scripted tasks.
        source: "live" if read from disk, "default" if fallback was used.
        pacing_state_ts: ISO-8601 timestamp of the pacing_state.json file, or None.
    """

    regime: Regime
    default_model: ModelName
    escalation_model: ModelName
    mechanical_model: ModelName
    source: Literal["live", "default"] = "default"
    pacing_state_ts: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_DEFAULT_POLICY = PacingPolicy(
    regime="on",
    default_model="sonnet",
    escalation_model="opus",
    mechanical_model="haiku",
    source="default",
    pacing_state_ts=None,
)


def get_pacing_policy() -> PacingPolicy:
    """Read the current pacing policy from dashboard/pacing_state.json.

    Falls back to a safe default (regime="on", default_model="sonnet") if:
    * The file does not exist.
    * The file cannot be parsed.
    * The file is stale (older than PACING_STALE_THRESHOLD_SECONDS = 30 min).

    Returns:
        PacingPolicy dataclass with regime and per-task-class model names.
    """
    if not PACING_STATE_FILE.exists():
        log.info(
            "pacing_state.json not found at %s — using default policy (regime=on)",
            PACING_STATE_FILE,
        )
        return _DEFAULT_POLICY

    # Staleness check
    try:
        age_seconds = time.time() - PACING_STATE_FILE.stat().st_mtime
        if age_seconds > PACING_STALE_THRESHOLD_SECONDS:
            log.warning(
                "pacing_state.json is %.0f min old (threshold=30 min) — using default policy",
                age_seconds / 60,
            )
            return _DEFAULT_POLICY
    except OSError as exc:
        log.warning("Cannot stat pacing_state.json: %s — using default policy", exc)
        return _DEFAULT_POLICY

    # Load and parse
    try:
        with open(PACING_STATE_FILE, "r", encoding="utf-8") as fh:
            raw: dict[str, Any] = json.load(fh)
    except Exception as exc:
        log.warning("Cannot parse pacing_state.json: %s — using default policy", exc)
        return _DEFAULT_POLICY

    regime_raw: str = raw.get("regime", "on")
    recommended_model_raw: str = raw.get("recommended_model_default", "sonnet")

    # Validate regime
    valid_regimes: set[str] = {"under", "on", "over", "emergency"}
    if regime_raw not in valid_regimes:
        log.warning(
            "Unknown regime %r in pacing_state.json — defaulting to 'on'",
            regime_raw,
        )
        regime_raw = "on"

    # Validate recommended model
    valid_models: set[str] = {"opus", "sonnet", "haiku"}
    if recommended_model_raw not in valid_models:
        log.warning(
            "Unknown recommended_model_default %r in pacing_state.json — defaulting to 'sonnet'",
            recommended_model_raw,
        )
        recommended_model_raw = "sonnet"

    regime: Regime = regime_raw  # type: ignore[assignment]
    model_table = _REGIME_MODEL_TABLE.get(regime, _REGIME_MODEL_TABLE["on"])

    # The pacing daemon's recommended_model_default overrides the table's default
    # but not mechanical/escalation (those are regime-driven).
    policy = PacingPolicy(
        regime=regime,
        default_model=recommended_model_raw,  # type: ignore[arg-type]
        escalation_model=model_table["escalation"],
        mechanical_model=model_table["mechanical"],
        source="live",
        pacing_state_ts=raw.get("ts") or raw.get("timestamp"),
    )

    log.info(
        "Pacing policy loaded: regime=%s default=%s escalation=%s mechanical=%s (age=%.0fs)",
        policy.regime,
        policy.default_model,
        policy.escalation_model,
        policy.mechanical_model,
        age_seconds,
    )
    return policy


# ---------------------------------------------------------------------------
# Task definition
# ---------------------------------------------------------------------------


@dataclass
class Task:
    """A unit of work to be dispatched to a sub-agent or script.

    Attributes:
        task_id: Unique identifier for this task.
        kind: Task classification — "mechanical" | "reasoning" | "discovery"
              | "proactive_ideation" | "feature_discovery" | "report_generation"
              | "broadcast" | "default".
        description: Human-readable summary.
        script: Optional script name (relative to scripts/) to execute.
        args: Additional CLI arguments for the script.
        essential: If False, may be deferred in emergency regime.
        requested_model: Explicit model override (None = let policy decide).
        metadata: Arbitrary extra fields.
    """

    task_id: str
    kind: str
    description: str
    script: str | None = None
    args: list[str] = field(default_factory=list)
    essential: bool = True
    requested_model: ModelName | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------


def select_model(task: Task, policy: PacingPolicy) -> tuple[ModelName, bool]:
    """Determine which model to use for *task* given the current *policy*.

    NOTE: This legacy function is retained as a fallback.  When unified_model_router
    is importable, ``select_model_via_router()`` is called instead and this
    function is only invoked if the router import fails.

    Args:
        task: The task to be dispatched.
        policy: The current pacing policy.

    Returns:
        Tuple of (chosen_model, was_overridden_by_pacing).
        ``was_overridden_by_pacing`` is True when the pacing regime caused the
        model to differ from what would have been chosen in the "on" baseline.
    """
    # Baseline (what we'd choose in the "on" regime)
    baseline_table = _REGIME_MODEL_TABLE["on"]
    if task.kind == "mechanical":
        baseline_model: ModelName = baseline_table["mechanical"]  # type: ignore[assignment]
    elif task.kind == "reasoning":
        baseline_model = baseline_table["escalation"]  # type: ignore[assignment]
    else:
        baseline_model = baseline_table["default"]  # type: ignore[assignment]

    # Honour explicit override from the task itself (but log it)
    if task.requested_model is not None:
        return task.requested_model, (task.requested_model != baseline_model)

    # Apply pacing policy
    if task.kind == "mechanical":
        chosen: ModelName = policy.mechanical_model
    elif task.kind == "reasoning":
        chosen = policy.escalation_model
    else:
        chosen = policy.default_model

    return chosen, (chosen != baseline_model)


# ---------------------------------------------------------------------------
# Unified router integration
# ---------------------------------------------------------------------------

# Maps CEO task kinds to unified_model_router ComplexityLevel values.
_KIND_TO_COMPLEXITY: dict[str, str] = {
    "mechanical": "mechanical",
    "reasoning": "reasoning",
    "feature_discovery": "coding",
    "proactive_ideation": "reasoning",
    "report_generation": "mechanical",
    "broadcast": "mechanical",
    "default": "coding",
}


def select_model_via_router(
    task: Task,
    policy: PacingPolicy,
) -> tuple[ModelName, bool, str | None]:
    """Choose a model by delegating to unified_model_router.route().

    Falls back to the legacy ``select_model()`` if the router is not available.

    Rationale: the unified router incorporates pacing regime directly (it reads
    pacing_state.json itself), adds cost-ceiling and context-size guards, and
    knows about DeepSeek/Ollama backends.  The CEO daemon only needs the
    three-tier Claude model name (haiku/sonnet/opus) from the router's
    BackendChoice.model field.  Non-Claude backends are mapped to "haiku" as a
    conservative local-equivalent.

    Args:
        task: The task to route.
        policy: Current pacing policy (used for fallback and for baseline calc).

    Returns:
        Tuple of (chosen_model, was_pacing_influenced, router_reason).
        ``router_reason`` is the BackendChoice.reason string from the router,
        or None if the legacy fallback was used.
    """
    if not _ROUTER_AVAILABLE or _router_route is None or _RouterTaskRequest is None:
        log.debug("Router not available — falling back to legacy select_model()")
        model, influenced = select_model(task, policy)
        return model, influenced, None

    # Honour explicit task override before hitting the router
    if task.requested_model is not None:
        baseline_table = _REGIME_MODEL_TABLE["on"]
        if task.kind == "mechanical":
            baseline_model: ModelName = baseline_table["mechanical"]  # type: ignore[assignment]
        elif task.kind == "reasoning":
            baseline_model = baseline_table["escalation"]  # type: ignore[assignment]
        else:
            baseline_model = baseline_table["default"]  # type: ignore[assignment]
        return (
            task.requested_model,
            (task.requested_model != baseline_model),
            "task.requested_model override (bypassed router)",
        )

    complexity = _KIND_TO_COMPLEXITY.get(task.kind, "coding")

    try:
        req = _RouterTaskRequest(
            prompt=task.description or f"task:{task.task_id}",
            complexity=complexity,  # type: ignore[arg-type]
            independence_required=False,
            context_tokens_estimate=0,
            cost_ceiling_usd=0.10,
            deadline_seconds=60.0,
            allow_local=True,
        )
        choice = _router_route(req)
    except Exception as exc:
        log.warning(
            "unified_model_router.route() raised %s — falling back to legacy select_model()",
            exc,
        )
        model, influenced = select_model(task, policy)
        return model, influenced, None

    # Extract claude model tier from router's choice.
    # The router may choose deepseek_openclaw or ollama_local; map those to
    # "haiku" (cheapest Claude equivalent) so the daemon can still build the
    # `--model haiku` flag for the subprocess command.
    router_model_raw = choice.model  # e.g. "claude-sonnet-4-6", "deepseek-v4-flash"
    if "opus" in router_model_raw:
        chosen: ModelName = "opus"
    elif "sonnet" in router_model_raw:
        chosen = "sonnet"
    else:
        # haiku, deepseek, ollama → haiku as local-equivalent
        chosen = "haiku"

    # Determine baseline (on-regime) for "was pacing influenced?" flag
    baseline_table_on = _REGIME_MODEL_TABLE["on"]
    if task.kind == "mechanical":
        baseline: ModelName = baseline_table_on["mechanical"]  # type: ignore[assignment]
    elif task.kind == "reasoning":
        baseline = baseline_table_on["escalation"]  # type: ignore[assignment]
    else:
        baseline = baseline_table_on["default"]  # type: ignore[assignment]

    was_influenced = chosen != baseline

    log.debug(
        "Router chose %s/%s (mapped to %s) for task=%s kind=%s complexity=%s reason=%r",
        choice.backend,
        choice.model,
        chosen,
        task.task_id,
        task.kind,
        complexity,
        choice.reason[:120],
    )
    return chosen, was_influenced, choice.reason


# ---------------------------------------------------------------------------
# Pacing routing event
# ---------------------------------------------------------------------------


def _publish_routing_event(
    task: Task,
    original_model: ModelName,
    chosen_model: ModelName,
    regime: Regime,
    deferred: bool = False,
) -> None:
    """Publish a pacing_routing_decision event to the event bus (best-effort).

    Args:
        task: The task being routed.
        original_model: What would have been used without pacing.
        chosen_model: What was actually chosen.
        regime: Current pacing regime.
        deferred: True if the task was skipped due to emergency regime.
    """
    if _EB is None:
        return
    try:
        _EB.publish_from_anywhere(
            "pacing_routing_decision",
            {
                "task_id": task.task_id,
                "task_kind": task.kind,
                "task_description": task.description[:200],
                "original_model": original_model,
                "new_model": chosen_model,
                "regime": regime,
                "deferred": deferred,
                "routing_influenced": (original_model != chosen_model) or deferred,
            },
            source="ceo_orchestrator_daemon",
        )
    except Exception as exc:
        log.debug("Event bus publish failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Task dispatch
# ---------------------------------------------------------------------------


def dispatch_task(
    task: Task,
    policy: PacingPolicy,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Dispatch a single task, honouring the pacing policy.

    This is the central routing function.  Every model decision flows through
    here so that pacing is applied uniformly.

    Args:
        task: The task to dispatch.
        policy: Current pacing policy.
        dry_run: If True, log the decision but do not spawn any process.

    Returns:
        Result dict with keys: task_id, status, model_used, regime, deferred.
    """
    # Baseline model (on-regime, no pacing)
    baseline_table = _REGIME_MODEL_TABLE["on"]
    if task.kind == "mechanical":
        baseline_model: ModelName = baseline_table["mechanical"]  # type: ignore[assignment]
    elif task.kind == "reasoning":
        baseline_model = baseline_table["escalation"]  # type: ignore[assignment]
    else:
        baseline_model = baseline_table["default"]  # type: ignore[assignment]

    # Emergency deferral for non-essential tasks
    if policy.regime == "emergency" and task.kind in _NON_ESSENTIAL_TASK_KINDS:
        log.info(
            "[PACING] DEFERRED task=%s kind=%s — deferred due to pacing emergency",
            task.task_id,
            task.kind,
        )
        _publish_routing_event(
            task=task,
            original_model=baseline_model,
            chosen_model=baseline_model,
            regime=policy.regime,
            deferred=True,
        )
        return {
            "task_id": task.task_id,
            "status": "deferred",
            "model_used": None,
            "regime": policy.regime,
            "deferred": True,
            "reason": "pacing_emergency",
        }

    # Delegate to unified_model_router when available; legacy select_model() as fallback.
    chosen_model, was_pacing_influenced, router_reason = select_model_via_router(task, policy)

    if was_pacing_influenced:
        log.info(
            "[PACING] Routing influenced for task=%s kind=%s: %s -> %s (regime=%s)%s",
            task.task_id,
            task.kind,
            baseline_model,
            chosen_model,
            policy.regime,
            f" router={router_reason[:80]!r}" if router_reason else "",
        )
        _publish_routing_event(
            task=task,
            original_model=baseline_model,
            chosen_model=chosen_model,
            regime=policy.regime,
            deferred=False,
        )
    else:
        log.debug(
            "task=%s kind=%s model=%s (regime=%s, no routing change)%s",
            task.task_id,
            task.kind,
            chosen_model,
            policy.regime,
            f" router={router_reason[:80]!r}" if router_reason else "",
        )

    if task.script is None:
        log.info("task=%s has no script — metadata-only task, marking dispatched", task.task_id)
        return {
            "task_id": task.task_id,
            "status": "dispatched_metadata",
            "model_used": chosen_model,
            "regime": policy.regime,
            "deferred": False,
        }

    script_path = SCRIPTS_DIR / task.script
    if not script_path.is_file():
        log.warning("Script %s not found — skipping task=%s", task.script, task.task_id)
        return {
            "task_id": task.task_id,
            "status": "skipped_no_script",
            "model_used": chosen_model,
            "regime": policy.regime,
            "deferred": False,
        }

    cmd = [PYTHON, str(script_path), "--model", chosen_model] + task.args

    if dry_run:
        log.info("[DRY-RUN] Would spawn: %s", " ".join(str(t) for t in cmd))
        return {
            "task_id": task.task_id,
            "status": "dry_run",
            "model_used": chosen_model,
            "regime": policy.regime,
            "deferred": False,
        }

    log.info("Spawning task=%s model=%s cmd=%s", task.task_id, chosen_model, cmd[:5])
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        log.info("task=%s PID=%d launched", task.task_id, proc.pid)
        return {
            "task_id": task.task_id,
            "status": f"dispatched:pid={proc.pid}",
            "model_used": chosen_model,
            "regime": policy.regime,
            "deferred": False,
            "pid": proc.pid,
        }
    except Exception as exc:
        log.error("Failed to spawn task=%s: %s", task.task_id, exc)
        return {
            "task_id": task.task_id,
            "status": f"spawn_failed:{exc}",
            "model_used": chosen_model,
            "regime": policy.regime,
            "deferred": False,
        }


# ---------------------------------------------------------------------------
# Orchestration cycle
# ---------------------------------------------------------------------------


def build_task_queue() -> list[Task]:
    """Produce the set of tasks for this orchestration cycle.

    Tasks are discovered from:
    * overseer/recommendations.json (top_actions → reasoning/retrain tasks)
    * proactive/urgent.json (high-priority proactive insights)
    * feature_discovery/inbox/queue.json (feature additions)

    Returns:
        Ordered list of Task objects, highest priority first.
    """
    tasks: list[Task] = []
    now_ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")

    # -- overseer recommendations -------------------------------------------
    recs_file = WORK / "overseer" / "recommendations.json"
    if recs_file.exists():
        try:
            with open(recs_file, "r", encoding="utf-8") as fh:
                rdata = json.load(fh)
            for idx, action in enumerate(
                rdata.get("recommendations", {}).get("top_actions", [])
            ):
                if not isinstance(action, str) or not action.strip():
                    continue
                text_lower = action.lower()
                if any(kw in text_lower for kw in ("retrain", "backtest", "run")):
                    kind = "mechanical"
                elif any(kw in text_lower for kw in ("investigate", "analyse", "analyze", "why")):
                    kind = "reasoning"
                else:
                    kind = "default"
                tasks.append(
                    Task(
                        task_id=f"overseer-{now_ts}-{idx}",
                        kind=kind,
                        description=action.strip()[:200],
                        essential=True,
                        metadata={"source": "overseer_recommendations"},
                    )
                )
        except Exception as exc:
            log.warning("Could not parse overseer recommendations: %s", exc)

    # -- proactive urgent insights -------------------------------------------
    urgent_file = WORK / "proactive" / "urgent.json"
    if urgent_file.exists():
        try:
            with open(urgent_file, "r", encoding="utf-8") as fh:
                udata = json.load(fh)
            question = udata.get("question", "")
            response = udata.get("response", "")
            tasks.append(
                Task(
                    task_id=f"urgent-{now_ts}",
                    kind="reasoning",
                    description=f"Urgent insight: {question[:100]}",
                    essential=True,
                    metadata={"source": "proactive_urgent", "response": response[:300]},
                )
            )
        except Exception as exc:
            log.warning("Could not parse proactive/urgent.json: %s", exc)

    # -- feature discovery queue (non-essential) -----------------------------
    fd_queue = WORK / "feature_discovery" / "inbox" / "queue.json"
    if fd_queue.exists():
        try:
            with open(fd_queue, "r", encoding="utf-8") as fh:
                fd_data = json.load(fh)
            if isinstance(fd_data, list):
                for idx, item in enumerate(fd_data[:5]):
                    if isinstance(item, dict) and "high" in str(item.get("impact", "")).lower():
                        tasks.append(
                            Task(
                                task_id=f"fd-{now_ts}-{idx}",
                                kind="feature_discovery",
                                description=f"Feature: {item.get('name', 'unknown')[:100]}",
                                essential=False,
                                metadata={"source": "fd_queue", "item": item},
                            )
                        )
        except Exception as exc:
            log.warning("Could not parse feature_discovery queue: %s", exc)

    return tasks


def _ceo_rank_burst(tasks: list) -> dict:
    """Fire 3 DeepSeek angle-ranking queries concurrently on the task queue.

    Writes results to overseer/ceo_impact_ranking_<TS>.json. Returns a summary.
    Non-fatal — never raises.
    """
    import concurrent.futures
    if not tasks:
        return {"calls": 0, "ok": 0, "errors": 0, "wrote": None}

    # Try the direct DeepSeek API caller (fast, parallel-friendly).
    try:
        sys.path.insert(0, str(SCRIPTS_DIR))
        from deepseek_direct import call_deepseek_direct as _ds_direct  # type: ignore
    except Exception:
        log.warning("CEO: deepseek_direct unavailable — skipping rank burst")
        return {"calls": 0, "ok": 0, "errors": 1, "wrote": None}

    task_lines = []
    for i, t in enumerate(tasks[:20]):
        try:
            task_lines.append(f"{i+1}. [{t.kind}] {t.description}"[:200])
        except Exception:
            continue
    task_block = "\n".join(task_lines) or "(empty)"

    def _one_angle(angle: str) -> dict:
        prompt = (
            "TASK QUEUE (ceo_orchestrator):\n"
            f"{task_block}\n\n"
            f"INSTRUCTION: {angle}\n\n"
            "Output: numbered ranking 1-N + 1-line justification each. Be terse."
        )
        t0 = time.monotonic()
        try:
            text = _ds_direct(prompt, timeout=IMPACT_RANK_TIMEOUT, max_tokens=600, temperature=0.3)
            return {"ok": bool(text), "angle": angle, "text": text or "", "elapsed_s": round(time.monotonic() - t0, 2)}
        except Exception as exc:
            return {"ok": False, "angle": angle, "text": "", "error": str(exc), "elapsed_s": round(time.monotonic() - t0, 2)}

    results: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=IMPACT_RANK_MAX_WORKERS) as ex:
        futs = {ex.submit(_one_angle, a): a for a in IMPACT_RANK_ANGLES}
        for fut in concurrent.futures.as_completed(futs, timeout=IMPACT_RANK_TIMEOUT + 10):
            try:
                results.append(fut.result())
            except Exception as exc:
                results.append({"ok": False, "angle": futs[fut], "text": "", "error": str(exc)})

    ok = sum(1 for r in results if r.get("ok"))
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = WORK / "overseer" / f"ceo_impact_ranking_{ts}.json"
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"ts": ts, "tasks": task_lines, "rankings": results}, indent=2, default=str))
    except Exception as exc:
        log.warning("CEO: rank burst write failed: %s", exc)
        out_path = None  # type: ignore[assignment]

    summary = {"calls": len(results), "ok": ok, "errors": len(results) - ok, "wrote": str(out_path) if out_path else None}
    log.info("CEO RANK BURST: %s", summary)
    return summary


def run_cycle(dry_run: bool = False) -> None:
    """Execute one full orchestration cycle.

    Args:
        dry_run: If True, do not actually spawn processes.
    """
    log.info("=== ceo_orchestrator cycle start (dry_run=%s) ===", dry_run)

    policy = get_pacing_policy()
    log.info(
        "Policy: regime=%s default=%s escalation=%s mechanical=%s source=%s",
        policy.regime,
        policy.default_model,
        policy.escalation_model,
        policy.mechanical_model,
        policy.source,
    )

    tasks = build_task_queue()
    log.info("Task queue: %d tasks", len(tasks))

    # ── PARALLEL IMPACT RANK BURST (2026-05-17) ──
    # Fire 3 DeepSeek angle-rankings concurrently. Non-fatal, additive.
    if not dry_run:
        try:
            _ceo_rank_burst(tasks)
        except Exception as exc:
            log.warning("CEO rank burst failed: %s", exc)

    results: list[dict[str, Any]] = []
    for task in tasks:
        result = dispatch_task(task, policy, dry_run=dry_run)
        results.append(result)
        log.info(
            "  task=%s kind=%s status=%s model=%s",
            task.task_id,
            task.kind,
            result["status"],
            result.get("model_used"),
        )

    # Re-emit high-impact feature_discovery queue items as WIRE_CANDIDATE
    # markers (the CEO's contribution: pulling forward queue items that
    # discovery surfaced but didn't auto-wire).  Writes BOTH md and jsonl.
    wire_emitted_this_cycle = 0
    if _WIRE_AVAILABLE and _wire_emit is not None:
        fd_queue = WORK / "feature_discovery" / "inbox" / "queue.json"
        try:
            if fd_queue.exists():
                with open(fd_queue, "r", encoding="utf-8") as fh:
                    fd_data = json.load(fh)
                cands: list[dict[str, Any]] = []
                # Only forward queue items the CEO has not previously processed
                # this calendar day — keeps the daily MD report tight.
                seen_today: set[str] = set()
                if CEO_STATE_FILE.exists():
                    try:
                        prev = json.loads(CEO_STATE_FILE.read_text())
                        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                        prev_ts = (prev.get("ts") or "")[:10]
                        if prev_ts == today:
                            seen_today = set(prev.get("wire_seen_today", []))
                    except Exception:
                        pass
                if isinstance(fd_data, list):
                    for item in fd_data[-30:]:  # last 30 items only
                        if not isinstance(item, dict):
                            continue
                        impact = str(item.get("impact", "")).lower()
                        if "high" not in impact and "med" not in impact:
                            continue
                        name = str(item.get("name", "")).strip()
                        if not name or name in seen_today:
                            continue
                        seen_today.add(name)
                        cands.append({
                            "feature_name": name,
                            "description": str(item.get("why", item.get("recipe", "")))[:160],
                            "data_source": str(item.get("url", item.get("source", "ceo_orchestrator"))),
                            "data_source_license": str(item.get("license") or "UNKNOWN"),
                            "features_added": int(item.get("features_added", 1) or 1),
                            "shift_1_safe": "unclear",
                            "integration_cost": str(item.get("cost", "MED")).upper(),
                            "requires_paid_api": "no",
                            "requires_human_review": "yes",
                            "expected_lift_pct": str(item.get("impact", "unknown")),
                            "citations": [item.get("url")] if item.get("url") else [],
                            "discovered_at": str(item.get("discovered_at",
                                                          datetime.now(timezone.utc).isoformat())),
                        })
                if cands:
                    res = _wire_emit(
                        cands,
                        discovered_by="ceo_orchestrator",
                        write_md=True,
                        write_jsonl=True,
                    )
                    wire_emitted_this_cycle = res["emitted"]
                    log.info(
                        "WIRE: CEO emitted %d candidates → md=%s jsonl=%s",
                        wire_emitted_this_cycle, res["md_path"], res["jsonl_path"],
                    )
        except Exception as exc:
            log.warning("CEO WIRE emit failed: %s", exc)

    # Persist cycle state
    state = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "regime": policy.regime,
        "policy": policy.to_dict(),
        "tasks_total": len(tasks),
        "tasks_deferred": sum(1 for r in results if r.get("deferred")),
        "tasks_dispatched": sum(
            1 for r in results if str(r.get("status", "")).startswith("dispatched")
        ),
        "wire_candidates_emitted": wire_emitted_this_cycle,
        "wire_seen_today": sorted(locals().get("seen_today", set())),
        "results": results,
    }
    CEO_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CEO_STATE_FILE.write_text(json.dumps(state, indent=2, default=str))

    log.info(
        "=== cycle complete: %d tasks, %d deferred, %d dispatched ===",
        len(tasks),
        state["tasks_deferred"],
        state["tasks_dispatched"],
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _show_policy() -> None:
    """Print the current pacing policy as JSON and exit."""
    policy = get_pacing_policy()
    output = {
        "pacing_policy": policy.to_dict(),
        "pacing_state_file": str(PACING_STATE_FILE),
        "pacing_state_exists": PACING_STATE_FILE.exists(),
        "pacing_state_age_seconds": (
            round(time.time() - PACING_STATE_FILE.stat().st_mtime, 1)
            if PACING_STATE_FILE.exists()
            else None
        ),
        "stale_threshold_seconds": PACING_STALE_THRESHOLD_SECONDS,
        "notes": {
            "under": "Plenty of API headroom — best models used freely.",
            "on": "Normal operating mode — balanced model selection.",
            "over": "Over budget — step down one tier everywhere.",
            "emergency": "Emergency — non-essential tasks deferred, minimum models used.",
        }[policy.regime],
    }
    print(json.dumps(output, indent=2, default=str))


def _show_router_policy() -> None:
    """Print what unified_model_router would pick for each task kind and exit.

    Iterates over all recognised task kinds, builds a sample TaskRequest for
    each, calls the router, and prints the full routing decision as JSON.
    Useful for validating router behaviour against the current pacing state
    without firing any real work.
    """
    policy = get_pacing_policy()
    regime = policy.regime

    sample_tasks = [
        Task(
            task_id=f"sample-{kind}",
            kind=kind,
            description=f"Sample {kind} task for --router-policy inspection",
        )
        for kind in [
            "mechanical",
            "reasoning",
            "feature_discovery",
            "proactive_ideation",
            "report_generation",
            "broadcast",
            "default",
        ]
    ]

    decisions: list[dict[str, Any]] = []
    for t in sample_tasks:
        chosen_model, influenced, router_reason = select_model_via_router(t, policy)
        decisions.append(
            {
                "task_kind": t.kind,
                "complexity_mapped": _KIND_TO_COMPLEXITY.get(t.kind, "coding"),
                "chosen_model": chosen_model,
                "was_pacing_influenced": influenced,
                "router_available": _ROUTER_AVAILABLE,
                "router_reason": router_reason,
            }
        )

    output = {
        "regime": regime,
        "router_available": _ROUTER_AVAILABLE,
        "pacing_policy": policy.to_dict(),
        "routing_decisions": decisions,
        "note": (
            "Router is active — all sub-agent spawns flow through unified_model_router.route()"
            if _ROUTER_AVAILABLE
            else "Router NOT imported — legacy select_model() fallback is active"
        ),
    }
    print(json.dumps(output, indent=2, default=str))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ceo_orchestrator_daemon.py",
        description="CEO orchestration daemon for S&P 500 ML Mastery system.",
    )
    parser.add_argument(
        "--show-policy",
        action="store_true",
        help="Print the current pacing policy as JSON and exit.",
    )
    parser.add_argument(
        "--router-policy",
        action="store_true",
        help=(
            "Print what unified_model_router would pick for each task kind given "
            "current pacing state, then exit.  Useful for debugging routing without "
            "firing real work."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run one cycle without spawning any processes.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()

    if args.show_policy:
        _show_policy()
        sys.exit(0)

    if args.router_policy:
        _show_router_policy()
        sys.exit(0)

    try:
        run_cycle(dry_run=args.dry_run)
    except Exception as exc:
        log.exception("FATAL: %s: %s", type(exc).__name__, exc)
        sys.exit(1)
