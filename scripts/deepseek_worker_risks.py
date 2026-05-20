"""
deepseek_worker_risks.py — Lane: RISKS

Runs every 60s via macOS LaunchAgent (KeepAlive=true, StartInterval=60).
Each cycle asks DeepSeek: "what's 1 hidden risk in current state?"

Outputs:
    deepseek_workers/risks/{ts}.jsonl  — one JSONL line per call (append-only)
    deepseek_workers/risks/latest.json — most-recent result (overwrite)
    logs/dw_risks.log

Design:
- Direct DeepSeek API call via deepseek_direct.py (~2s) — replaces openclaw subprocess (80-90s).
- Exponential backoff on HTTP 429 / empty response (2^n * 2s, max 4 retries).
- Append-only JSONL — no overwrite races.
- Never crashes: all exceptions caught and logged.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORK = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools/s&p500-ticker-mastery"
)
LANE = "risks"
OUT_DIR = WORK / "deepseek_workers" / LANE
LOG_PATH = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools/logs/dw_risks.log"
)
sys.path.insert(0, str(WORK / "scripts"))
from deepseek_direct import call_deepseek_direct  # noqa: E402

API_TIMEOUT = 30
BACKOFF_BASE = 2.0
MAX_RETRIES = 4

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    line = f"[{ts}] [risks] {msg}\n"
    try:
        with open(LOG_PATH, "a") as fh:
            fh.write(line)
    except Exception:
        pass
    print(line, end="", flush=True)


# ---------------------------------------------------------------------------
# State snapshot (lightweight)
# ---------------------------------------------------------------------------


def _mastery_count() -> int:
    mastery_dir = WORK / "mastery_files"
    try:
        return len(list(mastery_dir.glob("*mastered*.md")))
    except Exception:
        return 0


def _v3_run_count() -> int:
    try:
        return len(list((WORK / "backtests_xgb_v3").glob("*/run_meta.json")))
    except Exception:
        return 0


def _queue_depth() -> int:
    queue_dir = WORK / "queue"
    try:
        return len(list(queue_dir.iterdir())) if queue_dir.exists() else 0
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# DeepSeek call with exponential backoff
# ---------------------------------------------------------------------------


def _call_deepseek(prompt: str) -> str:
    """Call DeepSeek API directly (no subprocess). Returns response text or '' on failure."""
    try:
        return call_deepseek_direct(prompt, timeout=API_TIMEOUT, max_tokens=256, temperature=0.3)
    except RuntimeError as exc:
        log(f"api error: {exc}")
        return ""
    except Exception as exc:
        log(f"unexpected error: {exc}")
        return ""


def call_deepseek_with_backoff(prompt: str) -> str:
    """Call DeepSeek with exponential backoff on empty/error (rate-limit safety)."""
    for attempt in range(MAX_RETRIES + 1):
        result = _call_deepseek(prompt)
        if result:
            return result
        if attempt < MAX_RETRIES:
            delay = BACKOFF_BASE ** attempt * 2
            log(f"empty attempt {attempt+1}/{MAX_RETRIES+1} — backoff {delay:.1f}s")
            time.sleep(delay)
    return ""


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _append_jsonl(entry: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts_str = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_file = OUT_DIR / f"{ts_str}.jsonl"
    try:
        with open(out_file, "a") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        log(f"append_jsonl error: {exc}")


def _write_latest(entry: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(OUT_DIR / "latest.json", "w") as fh:
            json.dump(entry, fh, indent=2, ensure_ascii=False)
    except Exception as exc:
        log(f"write_latest error: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    log(f"cycle start pid={os.getpid()}")
    mastered = _mastery_count()
    v3_runs = _v3_run_count()
    queue_depth = _queue_depth()

    # Keep prompt under 400 chars for fast inference within 50s timeout
    prompt = (
        f"S&P500 XGBoost mastery pipeline: {mastered}/502 mastered, "
        f"{v3_runs} v3 backtests run, queue depth={queue_depth}. "
        "What is 1 HIDDEN or overlooked risk — data leakage, overfitting, "
        "infrastructure failure, or market regime shift — that could silently "
        "undermine results? Answer in 2 sentences max. Be specific."
    )

    t0 = time.monotonic()
    response = call_deepseek_with_backoff(prompt)
    elapsed = round(time.monotonic() - t0, 2)

    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "lane": LANE,
        "mastered": mastered,
        "v3_runs": v3_runs,
        "queue_depth": queue_depth,
        "prompt_chars": len(prompt),
        "response": response[:500],
        "elapsed_s": elapsed,
    }
    _append_jsonl(entry)
    _write_latest(entry)
    log(f"done in {elapsed}s — response: {response[:120]}")
    gc.collect()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"FATAL: {type(exc).__name__}: {exc}")
        import traceback
        log(traceback.format_exc())
        sys.exit(1)
