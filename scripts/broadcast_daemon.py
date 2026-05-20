"""
broadcast_daemon.py — 10-minute event broadcast to DeepSeek/OpenClaw.

Designed to run every 10 minutes via macOS LaunchAgent or cron.

Each cycle:
  1. Read events from events/stream.jsonl in the last 10 minutes.
  2. Bundle them into a structured context string.
  3. Call DeepSeek via OpenClaw ("capability model run --thinking medium").
  4. Save the response to broadcasts/{ts}.md.
  5. Append to broadcasts/live_stream.md (rolling window of last 50 broadcasts).

Robust against: missing events file, empty event window, DeepSeek timeouts,
corrupt broadcasts/live_stream.md, or missing broadcasts/ directory.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

WORK = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools/s&p500-ticker-mastery"
)
SCRIPTS_DIR = WORK / "scripts"
BROADCASTS_DIR = WORK / "broadcasts"
EVENTS_DIR = WORK / "events"
LOG_PATH = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools/logs/broadcast_daemon.log"
)
OPENCLAW = (
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools/bin/openclaw-gdrive"
)

LIVE_STREAM_PATH = BROADCASTS_DIR / "live_stream.md"
LOOKBACK_MINUTES = 2
MAX_LIVE_BROADCASTS = 50
DEEPSEEK_TIMEOUT = 25  # must finish before next 2-min cycle


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    line = f"[{ts}] {msg}\n"
    with open(LOG_PATH, "a") as fh:
        fh.write(line)
    print(line, end="", flush=True)


# ---------------------------------------------------------------------------
# Event reading (inline so broadcast_daemon.py is self-contained)
# ---------------------------------------------------------------------------

def _read_recent_events(minutes: int) -> list[dict]:
    """Read events from the last ``minutes`` minutes from stream.jsonl."""
    stream_path = EVENTS_DIR / "stream.jsonl"
    if not stream_path.exists():
        return []

    from datetime import timedelta
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    events: list[dict] = []

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
                        from datetime import timezone as _tz
                        ev_ts = ev_ts.replace(tzinfo=_tz.utc)
                    if ev_ts >= cutoff:
                        events.append(ev)
                except (json.JSONDecodeError, ValueError):
                    continue
    except (OSError, IOError):
        return []

    return events


# ---------------------------------------------------------------------------
# Event bundle formatter
# ---------------------------------------------------------------------------

def _bundle_events(events: list[dict]) -> str:
    """Format events into a readable context string for DeepSeek."""
    if not events:
        return "(no events in this window)"

    counts: Counter[str] = Counter(e.get("event_type", "unknown") for e in events)
    lines: list[str] = [
        f"Events in last {LOOKBACK_MINUTES}min: {len(events)} total",
        "By type: " + ", ".join(f"{k}={v}" for k, v in counts.most_common()),
        "",
        "Recent events (newest last):",
    ]
    for ev in events[-25:]:  # cap at 25 lines to stay within prompt budget
        ts_short = ev.get("ts", "")[:19].replace("T", " ")
        etype = ev.get("event_type", "unknown")
        # Build a compact payload summary (exclude meta fields)
        payload = {
            k: v
            for k, v in ev.items()
            if k not in ("ts", "event_type", "_unknown", "_source")
        }
        source = ev.get("_source", "")
        source_label = f" [{source}]" if source else ""
        payload_str = json.dumps(payload, default=str)
        if len(payload_str) > 200:
            payload_str = payload_str[:200] + "…"
        lines.append(f"  {ts_short}{source_label}  {etype}  {payload_str}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# DeepSeek call
# ---------------------------------------------------------------------------

def _call_deepseek(bundle: str, ts_label: str) -> str:
    """Call DeepSeek via OpenClaw and return the response text."""
    # Keep prompt short for fast 2-min cadence — cap bundle to 300 chars
    bundle_short = bundle[:300] if len(bundle) > 300 else bundle
    prompt = (
        f"S&P500 ML daemon activity ({ts_label}):\n{bundle_short}\n\n"
        "2 sentences: what's happening + 1 action. Signal over noise."
    )

    cmd = [
        OPENCLAW,
        "capability",
        "model",
        "run",
        "--local",
        "--model",
        "deepseek/deepseek-v4-flash",
        "--json",
        "--prompt",
        prompt,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=DEEPSEEK_TIMEOUT,
        )
        stdout = result.stdout.strip()
        if not stdout:
            return f"[empty response; stderr={result.stderr[:300]}]"

        try:
            parsed = json.loads(stdout)
            outputs = parsed.get("outputs") or []
            if outputs and isinstance(outputs, list):
                text = outputs[0].get("text", "")
                if text:
                    return text.strip()
            for key in ("response", "text", "content", "message", "result"):
                if key in parsed and parsed[key]:
                    return str(parsed[key]).strip()
            return stdout[:2000]
        except json.JSONDecodeError:
            return stdout[:2000]

    except subprocess.TimeoutExpired:
        return f"[timeout after {DEEPSEEK_TIMEOUT}s]"
    except FileNotFoundError:
        return f"[openclaw not found at {OPENCLAW}]"
    except Exception as exc:
        return f"[subprocess error: {exc}]"


# ---------------------------------------------------------------------------
# Live stream update (rolling last 50 broadcasts)
# ---------------------------------------------------------------------------

def _update_live_stream(ts_label: str, summary: str, event_count: int) -> None:
    """Prepend newest broadcast to live_stream.md; keep last 50."""
    BROADCASTS_DIR.mkdir(parents=True, exist_ok=True)

    new_block = (
        f"## {ts_label} ({event_count} events)\n\n"
        f"{summary}\n\n"
        f"---\n\n"
    )

    existing = ""
    if LIVE_STREAM_PATH.exists():
        try:
            existing = LIVE_STREAM_PATH.read_text()
        except (OSError, IOError):
            existing = ""

    # Count existing broadcast blocks (separated by "---")
    # Rebuild: newest on top, trim to MAX_LIVE_BROADCASTS
    all_blocks = [b.strip() for b in existing.split("---\n") if b.strip()]
    # Prepend new block (without trailing ---)
    clean_new = new_block.rstrip("\n").rstrip("---").strip()
    all_blocks.insert(0, clean_new)
    all_blocks = all_blocks[:MAX_LIVE_BROADCASTS]

    header = (
        "# Live Broadcast Stream\n\n"
        f"_Last updated: {ts_label} | Rolling window: {MAX_LIVE_BROADCASTS} broadcasts_\n\n"
    )
    content = header + "\n\n---\n\n".join(all_blocks) + "\n"
    LIVE_STREAM_PATH.write_text(content)


# ---------------------------------------------------------------------------
# Main cycle
# ---------------------------------------------------------------------------

def main() -> None:
    ts = datetime.now(timezone.utc)
    ts_label = ts.strftime("%Y-%m-%d %H:%M UTC")
    ts_file = ts.strftime("%Y%m%d-%H%M")

    log(f"=== broadcast_daemon cycle start — {ts_label} ===")

    # Ensure output dirs exist
    BROADCASTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Read recent events
    events = _read_recent_events(LOOKBACK_MINUTES)
    log(f"events in last {LOOKBACK_MINUTES}min: {len(events)}")

    # 2. Bundle
    bundle = _bundle_events(events)

    # 3. Call DeepSeek
    log("calling DeepSeek (thinking=medium)...")
    summary = _call_deepseek(bundle, ts_label)
    log(f"DeepSeek returned {len(summary)} chars")

    # 4. Save individual broadcast
    broadcast_md = (
        f"# Broadcast — {ts_label}\n\n"
        f"**Events:** {len(events)}\n\n"
        f"## Event bundle\n\n"
        f"```\n{bundle}\n```\n\n"
        f"## DeepSeek summary\n\n"
        f"{summary}\n"
    )
    broadcast_path = BROADCASTS_DIR / f"{ts_file}.md"
    broadcast_path.write_text(broadcast_md)
    log(f"wrote broadcast: {broadcast_path}")

    # 5. Update rolling live_stream.md
    _update_live_stream(ts_label, summary, len(events))
    log(f"updated live_stream.md")

    log(f"=== broadcast_daemon cycle complete ===\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"FATAL: {type(exc).__name__}: {exc}")
        import traceback
        log(traceback.format_exc())
        sys.exit(1)
