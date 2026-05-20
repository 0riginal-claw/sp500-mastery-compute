"""
event_bus.py — Append-only event bus for the S&P 500 ML Mastery system.

All system daemons publish structured events here. The broadcast_daemon.py
reads them every 10 minutes and streams to DeepSeek for real-time awareness.

Usage from any script:
    from event_bus import EventBus
    EventBus.publish_from_anywhere("mastery_added", {"ticker": "AAPL", "pf": 1.62})

Event types:
    mastery_added           — a ticker graduated to mastered
    feature_module_built    — a new feature module was created
    sweep_completed         — optuna or grid sweep finished for a ticker
    daemon_health           — periodic heartbeat from any daemon
    agent_spawned           — a Claude/OpenClaw sub-agent was spawned
    paper_trade_signal      — a live paper-trade signal fired
    overseer_cycle_complete — overseer_daemon.py finished one cycle
    discovery_report_written — feature_discovery_daemon.py wrote a report
    mastery_count_changed   — progress_dashboard.py detected a count change
    proactive_idea_generated — proactive_loop_daemon.py generated an idea
    paper_signal_generated  — live_paper_trade.py wrote today's signals
    paper_position_opened   — live_paper_trade.py opened a position
    paper_position_flattened — live_paper_trade.py closed all positions
    pacing_routing_decision — ceo_orchestrator chose a model influenced by pacing regime
    usage_pacing_changed    — usage_pacing_daemon.py detected a regime transition
"""

from __future__ import annotations

import json
import os
import time
import fcntl
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths — all relative to the project root; override via MASTERY_ROOT env var.
# ---------------------------------------------------------------------------

_DEFAULT_ROOT = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools/s&p500-ticker-mastery"
)

VALID_EVENT_TYPES: frozenset[str] = frozenset(
    [
        "mastery_added",
        "feature_module_built",
        "sweep_completed",
        "daemon_health",
        "agent_spawned",
        "paper_trade_signal",
        "overseer_cycle_complete",
        "discovery_report_written",
        "mastery_count_changed",
        "proactive_idea_generated",
        "paper_signal_generated",
        "paper_position_opened",
        "paper_position_flattened",
        "pacing_routing_decision",
        "usage_pacing_changed",
    ]
)


class EventBus:
    """Thread-safe, append-only event bus writing to events/stream.jsonl.

    All methods are classmethods so callers need no instantiation.
    """

    @classmethod
    def _get_stream_path(cls) -> Path:
        root = Path(os.environ.get("MASTERY_ROOT", str(_DEFAULT_ROOT)))
        return root / "events" / "stream.jsonl"

    @classmethod
    def publish(cls, event_type: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Append one event to events/stream.jsonl (never truncates).

        Args:
            event_type: One of VALID_EVENT_TYPES (unknown types are accepted
                        but flagged with a ``_unknown`` boolean).
            data: Arbitrary payload dict. Merged into the event envelope.

        Returns:
            The full event record that was written.
        """
        if data is None:
            data = {}

        event: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "_unknown": event_type not in VALID_EVENT_TYPES,
        }
        event.update(data)

        stream_path = cls._get_stream_path()
        try:
            stream_path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(event, default=str) + "\n"
            # File-level lock so concurrent daemons don't interleave writes.
            with open(stream_path, "a") as fh:
                try:
                    fcntl.flock(fh, fcntl.LOCK_EX)
                    fh.write(line)
                finally:
                    fcntl.flock(fh, fcntl.LOCK_UN)
        except Exception as exc:
            # Never crash the caller — event bus is best-effort.
            import sys
            print(f"[event_bus] WARNING: could not write event: {exc}", file=sys.stderr)

        return event

    @classmethod
    def publish_from_anywhere(
        cls,
        event_type: str,
        data: dict[str, Any] | None = None,
        source: str = "",
    ) -> dict[str, Any]:
        """Convenience wrapper that also stamps the caller source.

        Args:
            event_type: Event type string.
            data: Payload dict.
            source: Human-readable label for the calling script/daemon.

        Returns:
            The written event record.
        """
        payload = dict(data or {})
        if source:
            payload["_source"] = source
        return cls.publish(event_type, payload)

    @classmethod
    def read_last_n_minutes(cls, minutes: int = 10) -> list[dict[str, Any]]:
        """Return all events from the last ``minutes`` minutes.

        Corrupt/malformed lines are skipped silently.

        Args:
            minutes: Lookback window in minutes.

        Returns:
            List of event dicts ordered oldest-first.
        """
        stream_path = cls._get_stream_path()
        if not stream_path.exists():
            return []

        cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        events: list[dict[str, Any]] = []

        try:
            with open(stream_path, "r") as fh:
                for raw_line in fh:
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    try:
                        ev = json.loads(raw_line)
                        ts_str = ev.get("ts", "")
                        if not ts_str:
                            continue
                        ev_ts = datetime.fromisoformat(ts_str)
                        # Ensure timezone-aware comparison
                        if ev_ts.tzinfo is None:
                            ev_ts = ev_ts.replace(tzinfo=timezone.utc)
                        if ev_ts >= cutoff:
                            events.append(ev)
                    except (json.JSONDecodeError, ValueError):
                        continue
        except (OSError, IOError):
            return []

        return events

    @classmethod
    def summary_24h(cls) -> dict[str, int]:
        """Count events per type in the last 24 hours.

        Returns:
            Dict mapping event_type -> count, sorted descending.
        """
        stream_path = cls._get_stream_path()
        if not stream_path.exists():
            return {}

        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        counts: Counter[str] = Counter()

        try:
            with open(stream_path, "r") as fh:
                for raw_line in fh:
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    try:
                        ev = json.loads(raw_line)
                        ts_str = ev.get("ts", "")
                        if not ts_str:
                            continue
                        ev_ts = datetime.fromisoformat(ts_str)
                        if ev_ts.tzinfo is None:
                            ev_ts = ev_ts.replace(tzinfo=timezone.utc)
                        if ev_ts >= cutoff:
                            counts[ev.get("event_type", "unknown")] += 1
                    except (json.JSONDecodeError, ValueError):
                        continue
        except (OSError, IOError):
            return {}

        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    @classmethod
    def tail(cls, n: int = 20) -> list[dict[str, Any]]:
        """Return the last ``n`` events regardless of timestamp.

        Args:
            n: Number of tail events to return.

        Returns:
            List of event dicts, oldest first.
        """
        stream_path = cls._get_stream_path()
        if not stream_path.exists():
            return []

        lines: list[str] = []
        try:
            with open(stream_path, "r") as fh:
                # Memory-efficient tail using deque-like approach
                from collections import deque
                dq: deque[str] = deque(maxlen=n)
                for line in fh:
                    line = line.strip()
                    if line:
                        dq.append(line)
        except (OSError, IOError):
            return []

        events: list[dict[str, Any]] = []
        for raw_line in dq:
            try:
                events.append(json.loads(raw_line))
            except json.JSONDecodeError:
                continue
        return events


# ---------------------------------------------------------------------------
# CLI helper — python event_bus.py <event_type> [key=value ...]
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python event_bus.py <event_type> [key=value ...]\n")
        print("Valid types:", sorted(VALID_EVENT_TYPES))
        sys.exit(1)

    etype = sys.argv[1]
    payload: dict[str, Any] = {}
    for arg in sys.argv[2:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            payload[k] = v

    ev = EventBus.publish(etype, payload)
    print(json.dumps(ev, indent=2, default=str))
    print(f"\n24h summary: {EventBus.summary_24h()}")
