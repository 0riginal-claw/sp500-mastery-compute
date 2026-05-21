#!/usr/bin/env python3
"""test_watchdog_completion_check.py — Helper C tests for agent_has_completed_transcript().

Injects synthetic .output files (completed and stuck) and verifies that:
  - A completed transcript (with end_turn marker) is detected as finished → skipped
  - A stuck transcript (no completion markers) is detected as stuck → flagged

Run: python3 test_watchdog_completion_check.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: add the scripts/ dir to sys.path so we can import the daemon module
# ---------------------------------------------------------------------------
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))

from agent_watchdog_daemon import agent_has_completed_transcript  # noqa: E402

# ---------------------------------------------------------------------------
# Synthetic transcript builders
# ---------------------------------------------------------------------------

def make_completed_end_turn_jsonl() -> str:
    """JSONL transcript that ends with stop_reason=end_turn (Claude Code agent done)."""
    lines = [
        # First line: initial user message
        json.dumps({
            "type": "user",
            "agentId": "test_completed_001",
            "message": {"role": "user", "content": "Run some analysis task"}
        }),
        # Middle: assistant tool use
        json.dumps({
            "type": "assistant",
            "message": {"role": "assistant", "content": "I'll analyze the data."}
        }),
        # Final: stop_reason end_turn — this is the completion marker
        json.dumps({
            "type": "assistant",
            "stop_reason": "end_turn",
            "message": {"role": "assistant", "content": "Analysis complete."}
        }),
    ]
    return "\n".join(lines) + "\n"


def make_completed_model_completed_jsonl() -> str:
    """JSONL transcript with model.completed type (OpenClaw/DeepSeek agent done)."""
    lines = [
        json.dumps({
            "type": "prompt.submitted",
            "agentId": "test_openclaw_001",
        }),
        json.dumps({
            "type": "model.completed",
            "data": {
                "aborted": False,
                "timedOut": False,
                "stopReason": "end_turn",
            }
        }),
    ]
    return "\n".join(lines) + "\n"


def make_completed_threshold_met_plaintext() -> str:
    """Plain-text transcript from a poll script that reached its threshold."""
    return textwrap.dedent("""\
        [2026-05-16 14:00:01] poll #1: 0/5 deliverables present
        [2026-05-16 14:01:01] poll #2: 2/5 deliverables present
        [2026-05-16 14:02:01] poll #3: 4/5 deliverables present
        THRESHOLD MET: 4/5 — exiting poll
        FINAL STATUS:
          result_a.py: PRESENT
          result_b.py: PRESENT
    """)


def make_completed_elapsed_plaintext() -> str:
    """Plain-text transcript from a timer script that logged ELAPSED=Ns."""
    return textwrap.dedent("""\
        Starting analysis...
        Processing batch 1/3
        Processing batch 2/3
        Processing batch 3/3
        ELAPSED=390s
    """)


def make_stuck_transcript_jsonl() -> str:
    """JSONL transcript where agent is mid-flight — no completion markers."""
    lines = [
        json.dumps({
            "type": "user",
            "agentId": "test_stuck_001",
            "message": {"role": "user", "content": "Run long training pipeline"}
        }),
        # Only tool_use stop_reason — agent is still running tools
        json.dumps({
            "type": "assistant",
            "stop_reason": "tool_use",
            "message": {"role": "assistant", "content": "Starting training..."}
        }),
        json.dumps({
            "type": "tool_result",
            "content": "Training epoch 1/50 complete..."
        }),
    ]
    return "\n".join(lines) + "\n"


def make_stuck_transcript_plaintext() -> str:
    """Plain-text transcript with no completion markers — agent is stuck."""
    return textwrap.dedent("""\
        [2026-05-16 14:00:00] Starting heavy computation
        [2026-05-16 14:01:00] Still running...
        [2026-05-16 14:05:00] Processing large dataset, please wait...
    """)


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_tests() -> int:
    """Run all test cases. Returns number of failures."""
    failures = 0

    test_cases = [
        # (description, content, expected_result)
        ("Completed: stop_reason=end_turn (JSONL)",
         make_completed_end_turn_jsonl(), True),
        ("Completed: type=model.completed (JSONL)",
         make_completed_model_completed_jsonl(), True),
        ("Completed: THRESHOLD MET (plain text)",
         make_completed_threshold_met_plaintext(), True),
        ("Completed: ELAPSED=Ns (plain text)",
         make_completed_elapsed_plaintext(), True),
        ("Stuck: only tool_use stop_reason (JSONL)",
         make_stuck_transcript_jsonl(), False),
        ("Stuck: no markers (plain text)",
         make_stuck_transcript_plaintext(), False),
    ]

    print("=" * 70)
    print("agent_has_completed_transcript() — test suite")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, (desc, content, expected) in enumerate(test_cases):
            fake_path = Path(tmpdir) / f"test_agent_{i:02d}_fake.output"
            fake_path.write_text(content, encoding="utf-8")

            result = agent_has_completed_transcript(fake_path)
            status = "PASS" if result == expected else "FAIL"
            if result != expected:
                failures += 1

            print(f"  [{status}] {desc}")
            print(f"         expected={expected!r}  got={result!r}")

    print("=" * 70)
    if failures == 0:
        print(f"All {len(test_cases)} tests PASSED.")
    else:
        print(f"{failures}/{len(test_cases)} tests FAILED.")
    print("=" * 70)
    return failures


if __name__ == "__main__":
    sys.exit(run_tests())
