"""
usage_pacing_daemon.py — Claude Code token-pacing daemon.

Reads live usage stats via `ccusage weekly --json` (or a fallback sample JSON)
and computes a *model-tier policy* that ensures:
  - Tokens are never left on the table at the weekly reset.
  - The weekly quota is never burned out early (throttle cliff avoided).

PACE RATIO REGIMES
------------------
pace_ratio = quota_used_pct / max(time_elapsed_pct, 0.01)

  < 0.7   UNDER-PACE  → Use opus for all reasoning/coding; burn more tokens.
  0.7-1.3 ON-PACE     → Sonnet default; normal operations.
  1.3-1.8 OVER-PACE   → Haiku for mechanical tasks, sonnet only for must-reason.
  > 1.8   EMERGENCY   → Haiku only; pause non-essential daemons.

OUTPUT FILES
------------
  dashboard/pacing_state.json   — Latest state snapshot (overwritten each run).
  dashboard/pacing_history.jsonl — Append-only trend log (one line per run).

USAGE
-----
  python usage_pacing_daemon.py               # run once, exit 0
  python usage_pacing_daemon.py --daemon      # loop every 5 min
  python usage_pacing_daemon.py --simulate under      # inject synthetic regime
  python usage_pacing_daemon.py --simulate on
  python usage_pacing_daemon.py --simulate over
  python usage_pacing_daemon.py --simulate emergency

CRON REGISTRATION
-----------------
  # Run every 5 minutes (add via: crontab -e)
  # */5 * * * * /usr/bin/env python3 "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/s&p500-ticker-mastery/scripts/usage_pacing_daemon.py" >> "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/s&p500-ticker-mastery/logs/usage_pacing.log" 2>&1
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DRIVE = Path(
    "/Users/orginal/Library/CloudStorage/"
    "GoogleDrive-zachgladstone@gmail.com/My Drive"
)
WORK = _DRIVE / "AI-Tools" / "s&p500-ticker-mastery"
SCRIPTS = WORK / "scripts"
DASH = WORK / "dashboard"
LOGS = WORK / "logs"

PACING_STATE = DASH / "pacing_state.json"
PACING_HISTORY = DASH / "pacing_history.jsonl"
LOG_FILE = LOGS / "usage_pacing.log"

# Sample output path written by the sibling installer agent.
SAMPLE_OUTPUT_PATH = _DRIVE / "AI-Tools" / "reports" / "claude_usage" / "sample_output.json"

# ---------------------------------------------------------------------------
# Configuration — override via environment variables
# ---------------------------------------------------------------------------

# Weekly cost limit in USD.  Adjust to match your Anthropic plan:
#   Claude Pro:       $100 (approx)
#   Claude Max 5×:    $100/month — weekly ≈ $25
#   Claude Max 20×:   $200/month — weekly ≈ $50
# Set CLAUDE_WEEKLY_COST_LIMIT to your real limit before running.
WEEKLY_COST_LIMIT_USD: float = float(os.environ.get("CLAUDE_WEEKLY_COST_LIMIT", "100.0"))

# Week starts on Monday (ISO standard). Override with CLAUDE_WEEK_START_DAY=0..6
# 0=Monday, 6=Sunday
WEEK_START_WEEKDAY: int = int(os.environ.get("CLAUDE_WEEK_START_DAY", "0"))

# Daemon loop interval in seconds.
DAEMON_INTERVAL_SEC: int = int(os.environ.get("CLAUDE_PACING_INTERVAL", "300"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

RegimeType = Literal["under", "on", "over", "emergency"]
ModelName = Literal["opus", "sonnet", "haiku"]


def _setup_logging() -> logging.Logger:
    LOGS.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("usage_pacing")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(LOG_FILE)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(fh)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(sh)
    return logger


log = _setup_logging()

# ---------------------------------------------------------------------------
# Event-bus integration (best-effort — never crash the daemon)
# ---------------------------------------------------------------------------

try:
    sys.path.insert(0, str(SCRIPTS))
    from event_bus import EventBus as _EventBus  # type: ignore[import]

    _EB_AVAILABLE = True
except Exception:
    _EB_AVAILABLE = False
    _EventBus = None  # type: ignore[assignment,misc]


def _publish_event(regime: RegimeType, state: dict[str, Any]) -> None:
    """Publish a usage_pacing_changed event to the event bus (best-effort)."""
    if not _EB_AVAILABLE or _EventBus is None:
        return
    try:
        _EventBus.publish_from_anywhere(
            "usage_pacing_changed",
            {
                "regime": regime,
                "pace_ratio": state.get("pace_ratio"),
                "recommended_model_default": state.get("recommended_model_default"),
                "week_used_pct": state.get("week_used_pct"),
                "week_elapsed_pct": state.get("week_elapsed_pct"),
            },
            source="usage_pacing_daemon",
        )
    except Exception as exc:
        log.warning("event_bus publish failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Week boundary helpers
# ---------------------------------------------------------------------------


def _week_start_utc(now: datetime) -> datetime:
    """Return the most recent week-start boundary (UTC, time=00:00:00).

    The week starts on WEEK_START_WEEKDAY (0=Monday by default).
    """
    days_since_start = (now.weekday() - WEEK_START_WEEKDAY) % 7
    week_start = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(
        days=days_since_start
    )
    return week_start


def _week_reset_utc(now: datetime) -> datetime:
    """Return the upcoming week-reset boundary (UTC, time=00:00:00)."""
    return _week_start_utc(now) + timedelta(days=7)


# ---------------------------------------------------------------------------
# Usage data acquisition
# ---------------------------------------------------------------------------


def _run_ccusage_weekly() -> dict[str, Any] | None:
    """Run `npx ccusage weekly --json` and return parsed JSON or None."""
    try:
        result = subprocess.run(
            ["npx", "ccusage", "weekly", "--json", "--order", "desc"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            log.warning("ccusage exited %d: %s", result.returncode, result.stderr[:200])
            return None
        return json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        log.warning("ccusage timed out after 30s")
        return None
    except json.JSONDecodeError as exc:
        log.warning("ccusage JSON parse error: %s", exc)
        return None
    except FileNotFoundError:
        log.warning("npx not found; ccusage unavailable")
        return None
    except Exception as exc:
        log.warning("ccusage invocation error: %s", exc)
        return None


def _load_sample_output() -> dict[str, Any] | None:
    """Load the sample output file written by the installer agent, if present."""
    if SAMPLE_OUTPUT_PATH.exists():
        try:
            with open(SAMPLE_OUTPUT_PATH) as fh:
                return json.load(fh)
        except Exception as exc:
            log.warning("Failed to read sample_output.json: %s", exc)
    return None


def _extract_weekly_cost_from_json(data: dict[str, Any]) -> tuple[float, float] | None:
    """Extract (week_cost_usd, week_cost_limit_usd) from ccusage JSON.

    Tries multiple key patterns for forward-compatibility:
      weekly[0].totalCost        — standard ccusage weekly --json
      totals.totalCost           — ccusage totals block
      weekly_cost / week_cost    — hypothetical future keys
      tokens_used / weekly_quota — token-based quota (normalised to cost)

    Returns (used_cost, limit_cost) or None if not parseable.
    """
    # Pattern 1: ccusage weekly --json (primary)
    weekly = data.get("weekly") or data.get("weeks") or []
    if weekly:
        # Most-recent week is first when --order desc
        latest = weekly[0] if isinstance(weekly, list) else None
        if latest:
            cost = (
                latest.get("totalCost")
                or latest.get("total_cost")
                or latest.get("cost")
            )
            if cost is not None:
                return float(cost), WEEKLY_COST_LIMIT_USD

    # Pattern 2: totals block
    totals = data.get("totals") or {}
    cost = totals.get("totalCost") or totals.get("total_cost")
    if cost is not None:
        return float(cost), WEEKLY_COST_LIMIT_USD

    # Pattern 3: direct flat keys (future ccusage versions or custom schema)
    for key in ("weekly_cost", "week_cost", "cost_usd"):
        if key in data:
            return float(data[key]), WEEKLY_COST_LIMIT_USD

    # Pattern 4: token-based quota — normalise to cost using $15/M tokens (opus ballpark)
    tokens_used = data.get("tokens_used") or data.get("weekly_used")
    weekly_quota = data.get("weekly_quota") or data.get("tokens_remaining")
    if tokens_used is not None and weekly_quota is not None:
        # tokens_remaining → total quota = used + remaining
        if data.get("tokens_remaining") is not None:
            total_quota = float(tokens_used) + float(weekly_quota)
            approx_cost = float(tokens_used) * 15.0 / 1_000_000
            approx_limit = total_quota * 15.0 / 1_000_000
        else:
            approx_cost = float(tokens_used) * 15.0 / 1_000_000
            approx_limit = float(weekly_quota) * 15.0 / 1_000_000
        return approx_cost, approx_limit

    return None


def _extract_from_plain_text(text: str) -> tuple[float, float] | None:
    """Regex fallback: extract cost figures from plain-text ccusage output."""
    # Look for patterns like "$123.45" or "123.45 USD"
    cost_matches = re.findall(r"\$?([\d,]+\.[\d]{2})", text)
    costs = [float(m.replace(",", "")) for m in cost_matches]
    if costs:
        # Largest number is likely the weekly total
        return max(costs), WEEKLY_COST_LIMIT_USD
    return None


def _get_current_week_cost() -> tuple[float, float]:
    """Return (week_cost_used_usd, week_cost_limit_usd).

    Tries in order:
      1. ccusage weekly --json
      2. sample_output.json (written by installer agent)
      3. Returns (0.0, limit) with a warning so the daemon never crashes.
    """
    # Try live ccusage first
    raw = _run_ccusage_weekly()
    if raw is not None:
        result = _extract_weekly_cost_from_json(raw)
        if result is not None:
            return result

    # Try sample output from installer agent
    sample = _load_sample_output()
    if sample is not None:
        result = _extract_weekly_cost_from_json(sample)
        if result is not None:
            log.info("Using sample_output.json for usage data")
            return result

    log.warning(
        "Could not retrieve usage data from ccusage or sample_output.json. "
        "Defaulting to 0.0 used / %.1f limit. Check CLAUDE_WEEKLY_COST_LIMIT.",
        WEEKLY_COST_LIMIT_USD,
    )
    return 0.0, WEEKLY_COST_LIMIT_USD


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------


def _classify_regime(pace_ratio: float) -> RegimeType:
    """Map pace_ratio to regime label.

    Args:
        pace_ratio: quota_used_pct / max(time_elapsed_pct, 0.01)

    Returns:
        One of "under", "on", "over", "emergency".
    """
    if pace_ratio < 0.7:
        return "under"
    if pace_ratio < 1.3:
        return "on"
    if pace_ratio < 1.8:
        return "over"
    return "emergency"


def _model_policy(regime: RegimeType) -> tuple[ModelName, dict[str, ModelName]]:
    """Return (default_model, per_task_model_map) for the given regime.

    Args:
        regime: One of the four pace regimes.

    Returns:
        Tuple of (default_model, {"coding": ..., "mechanical": ..., "reasoning": ...})
    """
    policies: dict[RegimeType, tuple[ModelName, dict[str, ModelName]]] = {
        "under": (
            "opus",
            {"coding": "opus", "mechanical": "sonnet", "reasoning": "opus"},
        ),
        "on": (
            "sonnet",
            {"coding": "sonnet", "mechanical": "haiku", "reasoning": "sonnet"},
        ),
        "over": (
            "haiku",
            {"coding": "sonnet", "mechanical": "haiku", "reasoning": "sonnet"},
        ),
        "emergency": (
            "haiku",
            {"coding": "haiku", "mechanical": "haiku", "reasoning": "haiku"},
        ),
    }
    return policies[regime]


# ---------------------------------------------------------------------------
# Simulate helpers
# ---------------------------------------------------------------------------

_SIMULATE_PARAMS: dict[str, tuple[float, float]] = {
    # (quota_used_pct, time_elapsed_pct) → each produces the target regime
    "under": (0.15, 0.50),       # pace_ratio = 0.30 → under
    "on": (0.50, 0.50),          # pace_ratio = 1.00 → on
    "over": (0.70, 0.50),        # pace_ratio = 1.40 → over
    "emergency": (0.90, 0.45),   # pace_ratio = 2.00 → emergency
}


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_pacing_state(
    simulate: str | None = None,
) -> dict[str, Any]:
    """Compute the full pacing state dict.

    Args:
        simulate: If set, one of "under"|"on"|"over"|"emergency" — bypasses
                  ccusage and injects synthetic quota/time values for testing.

    Returns:
        Pacing state dict ready to write to pacing_state.json.
    """
    now = datetime.now(timezone.utc)
    week_start = _week_start_utc(now)
    week_reset = _week_reset_utc(now)

    week_elapsed_sec = (now - week_start).total_seconds()
    week_total_sec = 7 * 24 * 3600
    time_elapsed_pct = min(week_elapsed_sec / week_total_sec, 1.0)

    hours_until_reset = (week_reset - now).total_seconds() / 3600

    if simulate is not None:
        simulate_lower = simulate.lower()
        if simulate_lower not in _SIMULATE_PARAMS:
            raise ValueError(
                f"--simulate must be one of: {list(_SIMULATE_PARAMS.keys())}"
            )
        quota_used_pct, time_elapsed_pct = _SIMULATE_PARAMS[simulate_lower]
        week_cost_used = quota_used_pct * WEEKLY_COST_LIMIT_USD
        week_cost_limit = WEEKLY_COST_LIMIT_USD
        log.info(
            "[SIMULATE=%s] quota_used_pct=%.2f time_elapsed_pct=%.2f",
            simulate_lower,
            quota_used_pct,
            time_elapsed_pct,
        )
    else:
        week_cost_used, week_cost_limit = _get_current_week_cost()
        quota_used_pct = week_cost_used / max(week_cost_limit, 0.01)

    pace_ratio = quota_used_pct / max(time_elapsed_pct, 0.01)
    regime = _classify_regime(pace_ratio)
    default_model, task_models = _model_policy(regime)

    state: dict[str, Any] = {
        "ts": now.isoformat(),
        "week_used_pct": round(quota_used_pct, 6),
        "week_elapsed_pct": round(time_elapsed_pct, 6),
        "pace_ratio": round(pace_ratio, 6),
        "regime": regime,
        "recommended_model_default": default_model,
        "recommended_models_by_task_type": task_models,
        "reset_at": week_reset.isoformat(),
        "hours_until_reset": round(hours_until_reset, 4),
        # Audit fields — no raw token counts written here (private to user only)
        "_week_start": week_start.isoformat(),
        "_cost_limit_usd": round(week_cost_limit, 2),
        "_simulate": simulate,
    }
    return state


# ---------------------------------------------------------------------------
# I/O writers
# ---------------------------------------------------------------------------


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON to path atomically using a temp file + rename.

    Args:
        path: Target file path.
        data: Dict to serialise.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    content = json.dumps(data, indent=2, default=str) + "\n"
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON line to a .jsonl file with file locking.

    Args:
        path: Target .jsonl path.
        record: Dict to append.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, default=str) + "\n"
    with open(path, "a", encoding="utf-8") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX)
            fh.write(line)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Regime-transition tracking
# ---------------------------------------------------------------------------

_LAST_REGIME: RegimeType | None = None


def _check_and_publish_transition(state: dict[str, Any]) -> None:
    """Publish event_bus event only on regime transitions.

    Args:
        state: Current pacing state dict.
    """
    global _LAST_REGIME
    regime: RegimeType = state["regime"]
    if regime != _LAST_REGIME:
        log.info(
            "Regime transition: %s → %s (pace_ratio=%.3f)",
            _LAST_REGIME,
            regime,
            state["pace_ratio"],
        )
        _publish_event(regime, state)
        _LAST_REGIME = regime


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------


def run_once(simulate: str | None = None) -> dict[str, Any]:
    """Compute pacing state, write outputs, publish events, return state.

    Args:
        simulate: Optional regime name for dry-run testing.

    Returns:
        The pacing state dict that was written.
    """
    state = compute_pacing_state(simulate=simulate)

    # Always write the latest state
    _atomic_write_json(PACING_STATE, state)

    # Append to history (strip internal audit fields from history line to keep it lean)
    history_record: dict[str, Any] = {
        k: v for k, v in state.items() if not k.startswith("_")
    }
    _append_jsonl(PACING_HISTORY, history_record)

    # Publish event on regime change
    _check_and_publish_transition(state)

    log.info(
        "regime=%-10s pace_ratio=%.3f used=%.1f%% elapsed=%.1f%% reset_in=%.1fh model=%s",
        state["regime"],
        state["pace_ratio"],
        state["week_used_pct"] * 100,
        state["week_elapsed_pct"] * 100,
        state["hours_until_reset"],
        state["recommended_model_default"],
    )
    return state


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="usage_pacing_daemon",
        description="Claude Code token-pacing daemon. Recommends model tier policy.",
    )
    p.add_argument(
        "--daemon",
        action="store_true",
        help="Run in loop mode (every CLAUDE_PACING_INTERVAL seconds, default 300).",
    )
    p.add_argument(
        "--simulate",
        metavar="REGIME",
        choices=["under", "on", "over", "emergency"],
        default=None,
        help="Inject synthetic usage data to test JSON writer + recommendations.",
    )
    p.add_argument(
        "--limit",
        type=float,
        default=None,
        metavar="USD",
        help="Override weekly cost limit (USD). Default: CLAUDE_WEEKLY_COST_LIMIT env var or 100.0.",
    )
    return p


def main() -> int:
    """CLI entrypoint.

    Returns:
        Exit code (0 on success).
    """
    parser = _build_parser()
    args = parser.parse_args()

    # Allow --limit override
    if args.limit is not None:
        global WEEKLY_COST_LIMIT_USD
        WEEKLY_COST_LIMIT_USD = args.limit
        log.info("Weekly cost limit overridden to $%.2f via --limit", WEEKLY_COST_LIMIT_USD)

    if args.daemon:
        log.info(
            "Daemon mode: interval=%ds, limit=$%.2f",
            DAEMON_INTERVAL_SEC,
            WEEKLY_COST_LIMIT_USD,
        )
        while True:
            try:
                run_once(simulate=args.simulate)
            except Exception as exc:
                log.error("run_once failed (will retry): %s", exc, exc_info=True)
            time.sleep(DAEMON_INTERVAL_SEC)
    else:
        run_once(simulate=args.simulate)

    return 0


if __name__ == "__main__":
    sys.exit(main())


# ---------------------------------------------------------------------------
# CRON REGISTRATION INSTRUCTIONS
# ---------------------------------------------------------------------------
#
# To run this daemon every 5 minutes via cron, open your crontab with:
#
#   crontab -e
#
# Then add this single line (adjust CLAUDE_WEEKLY_COST_LIMIT to your plan):
#
#   */5 * * * * CLAUDE_WEEKLY_COST_LIMIT=100.0 /usr/bin/env python3 "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/s&p500-ticker-mastery/scripts/usage_pacing_daemon.py" >> "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/s&p500-ticker-mastery/logs/usage_pacing.log" 2>&1
#
# To verify the cron entry was added:
#   crontab -l | grep usage_pacing
#
# To remove:
#   crontab -e  → delete the line
#
# Alternatively, use the --daemon flag via a LaunchAgent plist (like other daemons
# in this project). Template:
#
#   Label: com.mastery.usage_pacing_daemon
#   ProgramArguments:
#     - /usr/bin/python3
#     - /path/to/usage_pacing_daemon.py
#     - --daemon
#   RunAtLoad: true
#   KeepAlive: true
#   EnvironmentVariables:
#     CLAUDE_WEEKLY_COST_LIMIT: "100.0"
#
# ---------------------------------------------------------------------------
