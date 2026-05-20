"""
deepseek_worker_market.py — Lane: MARKET

Runs every 60s via macOS LaunchAgent (KeepAlive=true, StartInterval=60).
Each cycle asks DeepSeek: "what's current market regime + impact on our mastery?"

Outputs:
    deepseek_workers/market/{ts}.jsonl  — one JSONL line per call (append-only)
    deepseek_workers/market/latest.json — most-recent result (overwrite)
    logs/dw_market.log

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
LANE = "market"
OUT_DIR = WORK / "deepseek_workers" / LANE
LOG_PATH = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools/logs/dw_market.log"
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
    line = f"[{ts}] [market] {msg}\n"
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


def _recent_mastered_tickers(n: int = 5) -> list[str]:
    """Most recently added mastery files — proxy for what's currently working."""
    mastery_dir = WORK / "mastery_files"
    try:
        files = sorted(
            mastery_dir.glob("*mastered*.md"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return [p.stem.split("_")[0] for p in files[:n]]
    except Exception:
        return []


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
    recent = _recent_mastered_tickers(5)
    now_utc = datetime.now(timezone.utc)
    # Day-of-week context for regime awareness (Mon=0, Fri=4)
    day_name = now_utc.strftime("%A")
    hour_et = (now_utc.hour - 4) % 24  # rough ET offset (no DST handling)

    # Keep prompt under 450 chars for fast inference within 50s timeout
    prompt = (
        f"S&P500 mean-reversion XGBoost: {mastered}/502 mastered. "
        f"Recent mastery: {recent}. Now: {day_name} ~{hour_et:02d}:00 ET. "
        "What is the CURRENT equity market regime (trend/range/volatile/event-driven) "
        "and how does it specifically impact our mean-reversion mastery rate? "
        "Answer in 2 sentences max. Be specific."
    )

    t0 = time.monotonic()
    response = call_deepseek_with_backoff(prompt)
    elapsed = round(time.monotonic() - t0, 2)

    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "lane": LANE,
        "mastered": mastered,
        "recent_mastered": recent,
        "day": day_name,
        "hour_et_approx": hour_et,
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
