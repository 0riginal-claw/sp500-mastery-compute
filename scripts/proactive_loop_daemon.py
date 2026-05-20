"""
proactive_loop_daemon.py — Continuous-loop ideation daemon (24/7).

Runs as a macOS LaunchAgent (KeepAlive=true). Differs from overseer_daemon.py
(15-min cron, structured recommendations) and feature_discovery_daemon.py
(30-min cron, GitHub recon). This daemon is a TRUE continuous loop:

  * Infinite while-True with 120s sleep between iterations.
  * Rotates through 5 specific questions each cycle.
  * Appends every response to proactive/stream.jsonl (append-only).
  * High-priority insights written to proactive/urgent.json (overwrites).
  * Health state written to proactive/health.json each iteration.
  * NEVER crashes — catches and logs all exceptions.
  * Memory leak prevention via gc.collect() every 100 iterations.

Outputs:
    proactive/stream.jsonl   — full chronological log (append-only)
    proactive/urgent.json    — latest high-priority insight
    proactive/health.json    — liveness probe
    logs/proactive_loop.log  — plain-text daemon log

Author: python-pro sub-agent
"""

from __future__ import annotations

import concurrent.futures
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── event bus (best-effort) ──────────────────────────────────────────────────
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from event_bus import EventBus as _EventBus
    _EB = _EventBus
except Exception:
    _EB = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

WORK = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools/s&p500-ticker-mastery"
)
PROACTIVE_DIR = WORK / "proactive"
LOG_PATH = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools/logs/proactive_loop.log"
)
sys.path.insert(0, str(WORK / "scripts"))
from deepseek_direct import call_deepseek_direct  # noqa: E402

# ── wire_candidate emitter (structured output for consumer daemon) ───────────
try:
    from wire_candidate import (
        emit as _wire_emit,
        parse_markdown_blocks as _wire_parse,
        WIRE_CANDIDATE_PROMPT_SUFFIX as _WIRE_SUFFIX,
    )
    _WIRE = True
except Exception:
    _WIRE = False
    _WIRE_SUFFIX = ""

STREAM_JSONL = PROACTIVE_DIR / "stream.jsonl"
URGENT_JSON = PROACTIVE_DIR / "urgent.json"
HEALTH_JSON = PROACTIVE_DIR / "health.json"

SLEEP_SECONDS = 120  # 2026-05-17 load-recovery: 30→120s (duty cycle 30%→1.7%)
API_TIMEOUT = 30   # direct API call is ~2s; 30s is generous
GC_EVERY = 100  # iterations

# ── PARALLEL BURST CONFIG (2026-05-17) ──────────────────────────────────────
# Each cycle now fires N parallel angle-questions in addition to the rotating
# single question — multiplies idea-generation throughput ~5x at constant
# wall-clock (DeepSeek is concurrent-friendly; cost is still ~$0.0001/cycle).
# 2026-05-17 load-recovery tune: BURST_N 5→2, MAX_WORKERS 5→2 to cut sustained
# load contribution. Combined with SLEEP_SECONDS 30→120 yields ~1.7% duty cycle.
PARALLEL_BURST_N = 2         # questions per cycle (in addition to rotating Q)
PARALLEL_BURST_MAX_WORKERS = 2  # ThreadPoolExecutor cap
PARALLEL_BURST_TIMEOUT = 45  # per-question

PARALLEL_BURST_ANGLES = [
    "What new factor model could unlock additional mastery for our mean-reversion XGBoost pipeline?",
    "Which microstructure pattern (VPIN, Kyle's lambda, tick-imbalance, order-book imbalance) would we most benefit from adding?",
    "What alt-data source (congressional trades, insider Form 4, lobbying, EDGAR filings, news) is under-utilized?",
    "What regime-detection method (HMM, change-point, vol clustering) could improve our current model?",
    "What sentiment or alternative-text signal would best complement our 722-feature stack?",
]

# ---------------------------------------------------------------------------
# Rotating questions (cycle index 0..4 mod 5)
# ---------------------------------------------------------------------------

QUESTIONS = [
    "What are we doing RIGHT NOW that's working?",
    "What can we do NEXT given current state?",
    "What can we EXPECT in the next 24 hours?",
    "What's the biggest risk we're ignoring?",
    "What 1 thing should we change immediately?",
]

# Indices of QUESTIONS that should receive the WIRE_CANDIDATE prompt suffix
# (DeepSeek will respond with structured feature blocks).  This routes ~20%
# of existing calls through the structured path without adding new API calls.
WIRE_QUESTION_INDICES = {1}  # "What can we do NEXT given current state?"

# High-priority keywords that trigger urgent.json write
HIGH_PRIORITY_KEYWORDS = [
    "immediately",
    "critical",
    "urgent",
    "risk",
    "fail",
    "broken",
    "blocker",
    "stop",
    "wrong",
    "danger",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    """Append timestamped line to daemon log and stdout."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    line = f"[{ts}] {msg}\n"
    try:
        with open(LOG_PATH, "a") as fh:
            fh.write(line)
    except Exception:
        pass  # Never crash the daemon over a log write
    print(line, end="", flush=True)


# ---------------------------------------------------------------------------
# State snapshot
# ---------------------------------------------------------------------------


def snapshot_state() -> dict[str, Any]:
    """Collect lightweight mission-state snapshot — no heavy I/O."""
    state: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
    }

    # Mastery count from mastery_files dir
    mastery_dir = WORK / "mastery_files"
    try:
        mastery_count = len(list(mastery_dir.glob("*mastered*.md")))
    except Exception:
        mastery_count = 0
    state["mastery_count"] = mastery_count

    # Queue depth
    queue_dir = WORK / "queue"
    try:
        queue_depth = len(list(queue_dir.iterdir())) if queue_dir.exists() else 0
    except Exception:
        queue_depth = 0
    state["queue_depth"] = queue_depth

    # Latest overseer report timestamp
    overseer_recs = WORK / "overseer" / "recommendations.json"
    try:
        if overseer_recs.exists():
            state["overseer_report_ts"] = datetime.fromtimestamp(
                overseer_recs.stat().st_mtime, tz=timezone.utc
            ).isoformat()
        else:
            state["overseer_report_ts"] = None
    except Exception:
        state["overseer_report_ts"] = None

    # Latest feature-discovery report timestamp
    fd_reports = WORK / "feature_discovery" / "reports"
    try:
        if fd_reports.exists():
            latest = max(fd_reports.glob("*.md"), key=lambda p: p.stat().st_mtime, default=None)
            state["fd_report_ts"] = (
                datetime.fromtimestamp(latest.stat().st_mtime, tz=timezone.utc).isoformat()
                if latest
                else None
            )
        else:
            state["fd_report_ts"] = None
    except Exception:
        state["fd_report_ts"] = None

    # Latest stream.jsonl entry count (cheap line-count)
    try:
        if STREAM_JSONL.exists():
            with open(STREAM_JSONL) as fh:
                state["stream_entries"] = sum(1 for _ in fh)
        else:
            state["stream_entries"] = 0
    except Exception:
        state["stream_entries"] = 0

    return state


# ---------------------------------------------------------------------------
# DeepSeek call via OpenClaw
# ---------------------------------------------------------------------------


def call_deepseek(
    question: str,
    state: dict[str, Any],
    timeout: int = API_TIMEOUT,
    *,
    wire_mode: bool = False,
) -> str:
    """
    Call DeepSeek API directly (no subprocess).
    Returns the response text, or an error string.
    Replaces openclaw subprocess (80-90s) with direct urllib call (~2s).

    When `wire_mode=True`, the WIRE_CANDIDATE prompt suffix is appended so the
    response is machine-parseable feature blocks (same call count, better prompt).
    """
    mastery_count = state.get("mastery_count", "?")
    queue_depth = state.get("queue_depth", "?")
    overseer_ts = state.get("overseer_report_ts") or "unknown"
    fd_ts = state.get("fd_report_ts") or "unknown"

    state_summary = (
        f"Mission: S&P 500 XGBoost ML strategy, {mastery_count}/502 tickers mastered. "
        f"Queue depth: {queue_depth}. "
        f"Latest overseer report: {overseer_ts}. "
        f"Latest feature-discovery report: {fd_ts}. "
        f"Pipelines: v7/v8 (~722 features)."
    )
    if wire_mode and _WIRE:
        prompt = (
            f"{state_summary}\n\n"
            f"QUESTION: {question}\n\n"
            f"Translate your answer into 1-3 concrete NEW FEATURE proposals — "
            f"each wired as a WIRE_CANDIDATE block per the schema below.\n"
            f"{_WIRE_SUFFIX}"
        )
        # WIRE_CANDIDATE blocks are denser than 3 sentences — give it more tokens.
        max_tokens = 800
    else:
        prompt = (
            f"{state_summary}\n\n"
            f"QUESTION: {question}\n\n"
            f"Answer in 3 sentences MAX. Be specific and actionable."
        )
        max_tokens = 256

    try:
        return call_deepseek_direct(prompt, timeout=timeout, max_tokens=max_tokens, temperature=0.3)
    except RuntimeError as exc:
        log(f"[WARN] DeepSeek call failed: {exc}")
        return f"[api error: {exc}]"
    except Exception as exc:
        return f"[unexpected error: {exc}]"


# ---------------------------------------------------------------------------
# Parallel angle burst (5 simultaneous DeepSeek queries / cycle)
# ---------------------------------------------------------------------------


def _single_angle_call(idx: int, angle: str, state: dict[str, Any]) -> dict[str, Any]:
    """Fire one WIRE-mode DeepSeek query for a single angle. Used by burst()."""
    t0 = time.monotonic()
    try:
        if _WIRE:
            prompt = (
                f"Mission: S&P 500 XGBoost ML, {state.get('mastery_count', '?')}/502 mastered. "
                f"Pipelines v7/v8 (~722 features).\n\n"
                f"QUESTION: {angle}\n\n"
                f"Translate your answer into 1-3 concrete NEW FEATURE proposals — "
                f"each wired as a WIRE_CANDIDATE block per the schema below.\n"
                f"{_WIRE_SUFFIX}"
            )
        else:
            prompt = f"QUESTION: {angle}\n\nAnswer in 3 sentences MAX. Specific + actionable."
        response = call_deepseek_direct(
            prompt, timeout=PARALLEL_BURST_TIMEOUT, max_tokens=600, temperature=0.4
        )
        return {
            "ok": True,
            "idx": idx,
            "angle": angle,
            "response": response,
            "elapsed_s": round(time.monotonic() - t0, 2),
        }
    except Exception as exc:
        return {
            "ok": False,
            "idx": idx,
            "angle": angle,
            "error": str(exc),
            "elapsed_s": round(time.monotonic() - t0, 2),
        }


def run_parallel_burst(state: dict[str, Any]) -> dict[str, Any]:
    """Fire N angle-questions to DeepSeek in parallel and emit WIRE_CANDIDATEs.

    Returns a summary dict: {"calls": N, "ok": k, "errors": e, "wire_emitted": m,
    "total_elapsed_s": float}.
    """
    burst_t0 = time.monotonic()
    angles = PARALLEL_BURST_ANGLES[:PARALLEL_BURST_N]
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL_BURST_MAX_WORKERS) as ex:
        futures = {ex.submit(_single_angle_call, i, a, state): (i, a) for i, a in enumerate(angles)}
        for fut in concurrent.futures.as_completed(futures, timeout=PARALLEL_BURST_TIMEOUT + 10):
            try:
                results.append(fut.result())
            except Exception as exc:
                idx, angle = futures[fut]
                results.append({"ok": False, "idx": idx, "angle": angle, "error": str(exc)})

    wire_emitted = 0
    ok_count = 0
    err_count = 0
    if _WIRE:
        all_cands: list[dict[str, Any]] = []
        for r in results:
            if not r.get("ok") or not r.get("response"):
                err_count += 1
                continue
            ok_count += 1
            try:
                cands = _wire_parse(r["response"], discovered_by="proactive_loop")
                all_cands.extend(cands)
            except Exception:
                pass
        if all_cands:
            try:
                res = _wire_emit(
                    all_cands,
                    discovered_by="proactive_loop",
                    write_md=True,
                    write_jsonl=True,
                )
                wire_emitted = res["emitted"]
            except Exception as exc:
                log(f"[BURST] emit failed: {exc}")
    else:
        ok_count = sum(1 for r in results if r.get("ok"))
        err_count = len(results) - ok_count

    total_elapsed = time.monotonic() - burst_t0
    summary = {
        "calls": len(results),
        "ok": ok_count,
        "errors": err_count,
        "wire_emitted": wire_emitted,
        "total_elapsed_s": round(total_elapsed, 2),
    }
    log(f"[BURST] {summary}")
    return summary


# ---------------------------------------------------------------------------
# Priority detection
# ---------------------------------------------------------------------------


def is_high_priority(response: str) -> bool:
    """Return True if response contains any high-priority keyword."""
    lowered = response.lower()
    return any(kw in lowered for kw in HIGH_PRIORITY_KEYWORDS)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def append_stream(entry: dict[str, Any]) -> None:
    """Append one JSON line to stream.jsonl."""
    PROACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(STREAM_JSONL, "a") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        log(f"[ERROR] append_stream failed: {exc}")


def write_urgent(entry: dict[str, Any]) -> None:
    """Overwrite urgent.json with the latest high-priority insight."""
    PROACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(URGENT_JSON, "w") as fh:
            json.dump(entry, fh, indent=2, ensure_ascii=False)
    except Exception as exc:
        log(f"[ERROR] write_urgent failed: {exc}")


def write_health(
    iteration: int,
    errors: int,
    last_question: str,
    last_run_ts: str,
) -> None:
    """Update health.json — liveness probe for external monitoring."""
    PROACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    health = {
        "last_run": last_run_ts,
        "iterations_total": iteration,
        "errors_total": errors,
        "last_question": last_question,
        "pid": os.getpid(),
        "sleep_seconds": SLEEP_SECONDS,
    }
    try:
        with open(HEALTH_JSON, "w") as fh:
            json.dump(health, fh, indent=2)
    except Exception as exc:
        log(f"[ERROR] write_health failed: {exc}")


# ---------------------------------------------------------------------------
# Single iteration (extracted for manual test-run)
# ---------------------------------------------------------------------------


def run_one_iteration(
    iteration: int,
    errors_total: int,
    question_index: int,
) -> tuple[int, int, int]:
    """
    Execute one ideation cycle.

    Returns updated (iteration, errors_total, question_index).
    NEVER raises — all exceptions are caught internally.
    """
    question = QUESTIONS[question_index % len(QUESTIONS)]
    next_q_index = (question_index + 1) % len(QUESTIONS)
    now_ts = datetime.now(timezone.utc).isoformat()

    log(f"[iter {iteration}] Q[{question_index}]: {question}")

    try:
        # a. State snapshot
        state = snapshot_state()

        # b. Call DeepSeek (wire_mode triggers WIRE_CANDIDATE structured output)
        wire_mode = question_index in WIRE_QUESTION_INDICES
        t_start = time.monotonic()
        response = call_deepseek(question, state, wire_mode=wire_mode)
        elapsed = time.monotonic() - t_start
        log(f"[iter {iteration}] DeepSeek answered in {elapsed:.1f}s (wire_mode={wire_mode}): {response[:120]}")

        # b2. Parse + emit WIRE_CANDIDATE blocks (additive, idempotent).
        wire_emitted = 0
        if _WIRE:
            try:
                cands = _wire_parse(response, discovered_by="proactive_loop")
                if cands:
                    # In proactive loop, default to JSONL only (cheap, dense) — the
                    # feature_discovery daemon already curates the daily MD report.
                    # But on wire_mode iterations, also write to MD for visibility.
                    res = _wire_emit(
                        cands,
                        discovered_by="proactive_loop",
                        write_md=wire_mode,
                        write_jsonl=True,
                    )
                    wire_emitted = res["emitted"]
                    log(f"[iter {iteration}] WIRE: emitted {wire_emitted} candidates")
            except Exception as exc:
                log(f"[iter {iteration}] WIRE: parse/emit failed: {exc}")

        # c. Build stream entry
        entry: dict[str, Any] = {
            "ts": now_ts,
            "iteration": iteration,
            "question_index": question_index,
            "question": question,
            "response": response,
            "elapsed_s": round(elapsed, 2),
            "mastery_count": state.get("mastery_count"),
            "queue_depth": state.get("queue_depth"),
            "high_priority": is_high_priority(response),
            "wire_mode": wire_mode,
            "wire_candidates_emitted": wire_emitted,
        }

        # c. Append to stream.jsonl
        append_stream(entry)

        # d. High-priority insight → urgent.json
        if entry["high_priority"]:
            log(f"[iter {iteration}] HIGH-PRIORITY insight detected — writing urgent.json")
            write_urgent(entry)

        # e. Publish event bus entry (best-effort)
        if _EB:
            try:
                _EB.publish_from_anywhere("proactive_idea_generated", {
                    "iteration": iteration,
                    "question": question[:200],
                    "response_excerpt": response[:300],
                    "high_priority": entry["high_priority"],
                    "elapsed_s": entry["elapsed_s"],
                }, source="proactive_loop_daemon")
            except Exception:
                pass

        # f. PARALLEL BURST — fire N angle-questions to DeepSeek concurrently.
        # Added 2026-05-17 (max-out OpenClaw/DeepSeek mandate). Non-fatal.
        try:
            burst = run_parallel_burst(state)
            entry["burst"] = burst
            # rewrite stream entry tail with burst summary
            append_stream({
                "ts": now_ts,
                "iteration": iteration,
                "event": "parallel_burst",
                "burst_summary": burst,
            })
        except Exception as exc:
            log(f"[iter {iteration}] BURST failed: {exc}")

    except Exception as exc:
        errors_total += 1
        log(f"[iter {iteration}] UNHANDLED exception (errors={errors_total}): {exc}")
        # Still update health so external monitors see we tried
        entry = {
            "ts": now_ts,
            "iteration": iteration,
            "question_index": question_index,
            "question": question,
            "response": f"[error: {exc}]",
            "elapsed_s": 0,
            "high_priority": False,
            "error": str(exc),
        }
        append_stream(entry)

    # e. Health update (outside inner try so it always runs)
    try:
        write_health(iteration, errors_total, question, now_ts)
    except Exception:
        pass

    iteration += 1
    return iteration, errors_total, next_q_index


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point: infinite ideation loop."""
    PROACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    log(f"[START] proactive_loop_daemon pid={os.getpid()} python={sys.version.split()[0]}")
    log(f"[START] sleep={SLEEP_SECONDS}s, api_timeout={API_TIMEOUT}s (direct API mode)")

    iteration = 1
    errors_total = 0
    question_index = 0

    while True:
        try:
            iteration, errors_total, question_index = run_one_iteration(
                iteration, errors_total, question_index
            )
        except Exception as exc:
            # Absolute last-resort — run_one_iteration should never raise,
            # but if it does, log and keep going.
            errors_total += 1
            log(f"[FATAL CATCH] outer loop exception #{errors_total}: {exc}")

        # Memory leak prevention every GC_EVERY iterations
        if iteration % GC_EVERY == 0:
            collected = gc.collect()
            log(f"[GC] iteration {iteration}: collected {collected} objects")

        # Sleep between iterations
        try:
            time.sleep(SLEEP_SECONDS)
        except KeyboardInterrupt:
            log("[STOP] KeyboardInterrupt — exiting cleanly")
            sys.exit(0)


# ---------------------------------------------------------------------------
# CLI: --test runs 3 iterations with no sleep (for manual verification)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("=== TEST MODE: 3 iterations, no sleep ===", flush=True)
        PROACTIVE_DIR.mkdir(parents=True, exist_ok=True)
        it, err, qi = 1, 0, 0
        for _ in range(3):
            it, err, qi = run_one_iteration(it, err, qi)
        print(f"\n=== TEST DONE: iterations={it - 1}, errors={err} ===", flush=True)
        # Print last 3 lines of stream.jsonl
        print("\n--- stream.jsonl (last 3 entries) ---")
        try:
            with open(STREAM_JSONL) as fh:
                lines = fh.readlines()
            for line in lines[-3:]:
                obj = json.loads(line)
                print(f"  iter={obj['iteration']} q={obj['question_index']} "
                      f"elapsed={obj.get('elapsed_s')}s hp={obj['high_priority']}")
                print(f"  RESPONSE: {obj['response'][:300]}\n")
        except Exception as e:
            print(f"  (could not read stream.jsonl: {e})")
        # Print health.json
        print("\n--- health.json ---")
        try:
            with open(HEALTH_JSON) as fh:
                print(json.dumps(json.load(fh), indent=2))
        except Exception as e:
            print(f"  (could not read health.json: {e})")
    else:
        main()
