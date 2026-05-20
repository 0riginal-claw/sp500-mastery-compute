"""
deepseek_worker_features.py — Lane: FEATURES

Runs every 60s via macOS LaunchAgent (KeepAlive=true, StartInterval=60).
Each cycle asks DeepSeek: "what's 1 new feature we haven't built?"

Outputs:
    deepseek_workers/features/{ts}.jsonl  — one JSONL line per call (append-only)
    deepseek_workers/features/latest.json — most-recent result (overwrite)
    logs/dw_features.log

Design:
- Direct DeepSeek API call via deepseek_direct.py (~2s) — replaces openclaw subprocess (80-90s).
- Exponential backoff on empty/429 (2^n * 2s, max 4 retries).
- Append-only JSONL — no overwrite races.
- Never crashes.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

WORK = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools/s&p500-ticker-mastery"
)
LANE = "features"
OUT_DIR = WORK / "deepseek_workers" / LANE
LOG_PATH = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools/logs/dw_features.log"
)
sys.path.insert(0, str(WORK / "scripts"))
from deepseek_direct import call_deepseek_direct  # noqa: E402

API_TIMEOUT = 30
BACKOFF_BASE = 2.0
MAX_RETRIES = 4

FEATURE_STACK_SUMMARY = (
    "53 technicals (RSI/EMA/MACD/BBands/ADX), 22 intraday (VWAP/gap/ORB), "
    "9 alt-data (EDGAR/congress/lobbying), sentiment, options-flow, macro."
)


def log(msg: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    line = f"[{ts}] [features] {msg}\n"
    try:
        with open(LOG_PATH, "a") as fh:
            fh.write(line)
    except Exception:
        pass
    print(line, end="", flush=True)


def _mastery_count() -> int:
    mastery_dir = WORK / "mastery_files"
    try:
        return len(list(mastery_dir.glob("*mastered*.md")))
    except Exception:
        return 0


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
    for attempt in range(MAX_RETRIES + 1):
        result = _call_deepseek(prompt)
        if result:
            return result
        if attempt < MAX_RETRIES:
            delay = BACKOFF_BASE ** attempt * 2
            log(f"empty attempt {attempt+1}/{MAX_RETRIES+1} — backoff {delay:.1f}s")
            time.sleep(delay)
    return ""


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


def main() -> None:
    log(f"cycle start pid={os.getpid()}")
    mastered = _mastery_count()

    prompt = (
        f"S&P500 daily mean-reversion XGBoost: {mastered}/502 mastered. "
        f"Current features: {FEATURE_STACK_SUMMARY} "
        "What is 1 NEW feature we haven't built that would most improve mastery? "
        "Name it, explain in 1 sentence, and give the Python data source."
    )

    t0 = time.monotonic()
    response = call_deepseek_with_backoff(prompt)
    elapsed = round(time.monotonic() - t0, 2)

    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "lane": LANE,
        "mastered": mastered,
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
