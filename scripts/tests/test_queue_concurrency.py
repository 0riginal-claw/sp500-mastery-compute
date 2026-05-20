"""
test_queue_concurrency.py — Proves that concurrent enqueue_job calls do not
corrupt queue.txt or dispatched.jsonl.

Test strategy
-------------
- Spawn 10 threads; each calls enqueue_job 10 times → 100 total enqueues.
- After all threads complete: assert 100 well-formed lines in queue.txt and
  100 well-formed JSON records in dispatched.jsonl, with no duplicate job IDs.
- The "without lock" variant temporarily monkey-patches _append_locked to use
  a bare open/write (no flock) and asserts that corruption is detected OR that
  it at least documents the risk.  Because the Python GIL partially serialises
  writes, corruption is not guaranteed on every run — the test therefore only
  checks structural integrity and reports whether corruption occurred, rather
  than asserting it fails.  The "with lock" variant must always pass.

Usage
-----
    pytest scripts/tests/test_queue_concurrency.py -v

Or run standalone:
    python scripts/tests/test_queue_concurrency.py
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Resolve project root so we can import cloud_dispatch even when pytest is
# invoked from an arbitrary working directory.
# ---------------------------------------------------------------------------
_TESTS_DIR   = Path(__file__).parent
_SCRIPTS_DIR = _TESTS_DIR.parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent

if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import cloud_dispatch  # noqa: E402  (must follow sys.path fix)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_queue(tmp_path: Path) -> tuple[Path, Path]:
    """Return (queue_file, status_file) inside a fresh temp directory."""
    queue_file  = tmp_path / "sweeps" / "queue.txt"
    status_file = tmp_path / "sweeps" / "dispatched.jsonl"
    (tmp_path / "sweeps").mkdir(parents=True, exist_ok=True)
    queue_file.touch()
    status_file.touch()
    return queue_file, status_file


# ---------------------------------------------------------------------------
# Helper: run concurrent enqueues against the given paths
# ---------------------------------------------------------------------------

_THREADS       = 10
_ENQUEUES_EACH = 10
_TOTAL         = _THREADS * _ENQUEUES_EACH

TICKER   = "AAPL"
STRATEGY = "orb"
SCRIPT   = "scripts/backtest_xgb_v8.py"


def _run_concurrent_enqueues(
    queue_file: Path,
    status_file: Path,
    patched_append: bool = False,
) -> list[str]:
    """
    Spawn _THREADS threads, each enqueuing _ENQUEUES_EACH jobs.

    If patched_append=True, replaces _append_locked with a bare write
    (no flock) to simulate what would happen without locking.

    Returns the list of all job IDs produced.
    """
    job_ids: list[str] = []
    lock = threading.Lock()

    errors: list[Exception] = []

    def _worker(thread_idx: int) -> None:
        for i in range(_ENQUEUES_EACH):
            try:
                jid = cloud_dispatch.enqueue_job(
                    ticker=f"{TICKER}_{thread_idx}_{i}",
                    strategy=STRATEGY,
                    script=SCRIPT,
                )
                with lock:
                    job_ids.append(jid)
            except Exception as exc:
                with lock:
                    errors.append(exc)

    if patched_append:
        # Replace _append_locked with a non-atomic write that has no flock.
        # We add a tiny sleep to maximise interleaving.
        def _unsafe_append(path: Path, text: str) -> None:
            time.sleep(0.0001)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(text)

        ctx = patch.object(cloud_dispatch, "_append_locked", _unsafe_append)
    else:
        ctx = patch.object(
            cloud_dispatch, "_QUEUE_FILE", queue_file
        )  # just a no-op context to keep code uniform

    # Always redirect writes to the tmp paths regardless of patching
    with (
        patch.object(cloud_dispatch, "_QUEUE_FILE",  queue_file),
        patch.object(cloud_dispatch, "_STATUS_FILE", status_file),
    ):
        if patched_append:
            with patch.object(cloud_dispatch, "_append_locked", _unsafe_append):
                threads = [
                    threading.Thread(target=_worker, args=(i,), daemon=True)
                    for i in range(_THREADS)
                ]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join(timeout=30)
        else:
            threads = [
                threading.Thread(target=_worker, args=(i,), daemon=True)
                for i in range(_THREADS)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)

    if errors:
        raise RuntimeError(f"{len(errors)} enqueue errors: {errors[:3]}")

    return job_ids


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------

def _parse_queue(queue_file: Path) -> tuple[int, int, list[str]]:
    """Return (total_lines, malformed_count, queue_job_ids)."""
    lines  = [ln for ln in queue_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    total  = len(lines)
    bad    = 0
    job_ids: list[str] = []
    for line in lines:
        parts = line.split()
        if len(parts) not in (3, 4):
            bad += 1
        elif len(parts) == 4:
            job_ids.append(parts[3])
    return total, bad, job_ids


def _parse_status(status_file: Path) -> tuple[int, int, list[str]]:
    """Return (total_records, malformed_count, job_ids)."""
    raw_lines = [ln for ln in status_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    total  = len(raw_lines)
    bad    = 0
    job_ids: list[str] = []
    for raw in raw_lines:
        try:
            rec = json.loads(raw)
            job_ids.append(rec["id"])
        except (json.JSONDecodeError, KeyError):
            bad += 1
    return total, bad, job_ids


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestQueueConcurrencyWithLock:
    """All writes use fcntl.flock — must always produce clean files."""

    def test_queue_line_count(self, tmp_queue: tuple[Path, Path]) -> None:
        queue_file, status_file = tmp_queue
        job_ids = _run_concurrent_enqueues(queue_file, status_file, patched_append=False)

        total, bad, _ = _parse_queue(queue_file)
        assert total == _TOTAL, (
            f"Expected {_TOTAL} queue lines, got {total}. "
            f"Possible lost writes under concurrency."
        )
        assert bad == 0, f"{bad} malformed queue lines detected."

    def test_status_line_count(self, tmp_queue: tuple[Path, Path]) -> None:
        queue_file, status_file = tmp_queue
        _run_concurrent_enqueues(queue_file, status_file, patched_append=False)

        total, bad, _ = _parse_status(status_file)
        assert total == _TOTAL, (
            f"Expected {_TOTAL} status records, got {total}."
        )
        assert bad == 0, f"{bad} malformed/non-JSON status records."

    def test_no_duplicate_job_ids_in_status(self, tmp_queue: tuple[Path, Path]) -> None:
        queue_file, status_file = tmp_queue
        _run_concurrent_enqueues(queue_file, status_file, patched_append=False)

        _, _, ids = _parse_status(status_file)
        assert len(ids) == _TOTAL, f"Expected {_TOTAL} IDs, found {len(ids)}"
        assert len(set(ids)) == _TOTAL, (
            f"Duplicate job IDs detected: "
            f"{_TOTAL - len(set(ids))} collisions out of {_TOTAL} jobs."
        )

    def test_no_partial_json_records(self, tmp_queue: tuple[Path, Path]) -> None:
        """Every line in dispatched.jsonl must be valid JSON with required keys."""
        queue_file, status_file = tmp_queue
        _run_concurrent_enqueues(queue_file, status_file, patched_append=False)

        raw_lines = [
            ln for ln in status_file.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        required_keys = {"id", "ticker", "strategy", "script", "status", "enqueued_at"}
        for i, raw in enumerate(raw_lines):
            rec: dict[str, Any]
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as exc:
                pytest.fail(f"Line {i+1} is not valid JSON: {exc}\nContent: {raw!r}")
            missing = required_keys - rec.keys()
            assert not missing, (
                f"Line {i+1} missing keys {missing}: {raw!r}"
            )


class TestQueueConcurrencyWithoutLock:
    """Documents what happens with unsafe (unflocked) writes.

    We cannot guarantee corruption on every run (GIL partially serialises),
    so these tests *report* corruption rather than asserting it must happen.
    They exist to show the test infrastructure works and to catch cases where
    something does go wrong.
    """

    def test_documents_unsafe_write_behavior(
        self, tmp_queue: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
    ) -> None:
        queue_file, status_file = tmp_queue
        _run_concurrent_enqueues(queue_file, status_file, patched_append=True)

        q_total, q_bad, _ = _parse_queue(queue_file)
        s_total, s_bad, s_ids = _parse_status(status_file)

        corruption_detected = q_bad > 0 or s_bad > 0 or q_total != _TOTAL or s_total != _TOTAL
        dup_ids = _TOTAL - len(set(s_ids)) if s_ids else 0

        report = (
            f"\n[WITHOUT LOCK] queue lines={q_total}/{_TOTAL} malformed={q_bad} | "
            f"status records={s_total}/{_TOTAL} malformed={s_bad} | "
            f"duplicate IDs={dup_ids} | corruption_detected={corruption_detected}"
        )
        print(report)

        # We do NOT assert corruption here — but we assert the test ran at all.
        assert q_total > 0 or s_total > 0, "No output produced — something went wrong."


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    print("=" * 70)
    print("Queue concurrency test — standalone mode")
    print(f"Threads={_THREADS}  enqueues_each={_ENQUEUES_EACH}  total={_TOTAL}")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        (tmp / "sweeps").mkdir()
        qf = tmp / "sweeps" / "queue.txt"
        sf = tmp / "sweeps" / "dispatched.jsonl"
        qf.touch(); sf.touch()

        print("\n--- WITH LOCK ---")
        t0 = time.monotonic()
        _run_concurrent_enqueues(qf, sf, patched_append=False)
        elapsed = time.monotonic() - t0
        qt, qb, _ = _parse_queue(qf)
        st, sb, sids = _parse_status(sf)
        dups = _TOTAL - len(set(sids))
        print(f"  queue lines : {qt}/{_TOTAL}  malformed={qb}")
        print(f"  status recs : {st}/{_TOTAL}  malformed={sb}")
        print(f"  dup IDs     : {dups}")
        print(f"  elapsed     : {elapsed:.2f}s")
        locked_ok = (qt == _TOTAL and qb == 0 and st == _TOTAL and sb == 0 and dups == 0)
        print(f"  RESULT: {'PASS' if locked_ok else 'FAIL'}")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        (tmp / "sweeps").mkdir()
        qf2 = tmp / "sweeps" / "queue.txt"
        sf2 = tmp / "sweeps" / "dispatched.jsonl"
        qf2.touch(); sf2.touch()

        print("\n--- WITHOUT LOCK ---")
        t0 = time.monotonic()
        _run_concurrent_enqueues(qf2, sf2, patched_append=True)
        elapsed = time.monotonic() - t0
        qt2, qb2, _ = _parse_queue(qf2)
        st2, sb2, sids2 = _parse_status(sf2)
        dups2 = _TOTAL - len(set(sids2)) if sids2 else 0
        corruption = qb2 > 0 or sb2 > 0 or qt2 != _TOTAL or st2 != _TOTAL
        print(f"  queue lines : {qt2}/{_TOTAL}  malformed={qb2}")
        print(f"  status recs : {st2}/{_TOTAL}  malformed={sb2}")
        print(f"  dup IDs     : {dups2}")
        print(f"  elapsed     : {elapsed:.2f}s")
        print(f"  corruption detected: {corruption}")
        print(f"  NOTE: GIL may prevent corruption on CPython — absence of")
        print(f"        corruption here does NOT mean the lock is unnecessary.")

    print("\n" + "=" * 70)
    sys.exit(0 if locked_ok else 1)
