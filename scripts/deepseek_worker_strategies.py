"""
deepseek_worker_strategies.py — Lane: STRATEGIES

Runs every 60s via macOS LaunchAgent (KeepAlive=true, StartInterval=60).
Each cycle asks DeepSeek: "which 1 ticker should we focus on next for intraday mastery?"

Outputs:
    deepseek_workers/strategies/{ts}.jsonl  — one JSONL line per call (append-only)
    deepseek_workers/strategies/latest.json — most-recent result (overwrite)
    logs/dw_strategies.log

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
LANE = "strategies"
OUT_DIR = WORK / "deepseek_workers" / LANE
LOG_PATH = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools/logs/dw_strategies.log"
)
sys.path.insert(0, str(WORK / "scripts"))
from deepseek_direct import call_deepseek_direct  # noqa: E402

API_TIMEOUT = 30   # seconds — generous for direct API call (~2s typical)
BACKOFF_BASE = 2.0
MAX_RETRIES = 4

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    line = f"[{ts}] [strategies] {msg}\n"
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


def _failing_tickers(n: int = 5) -> list[str]:
    import glob as _glob
    rows: list[tuple[float, str]] = []
    try:
        for p in _glob.glob(str(WORK / "backtests_xgb_v3/*/run_meta.json"))[:300]:
            tk = Path(p).parent.name.replace("_v3", "")
            with open(p) as fh:
                m = json.load(fh).get("metrics_oos_aggregate", {})
            pf = m.get("profit_factor") or 0.0
            wr = m.get("win_rate") or 0.0
            gap = max(0, 1.5 - pf) + max(0, 0.53 - wr)
            rows.append((gap, tk))
    except Exception:
        return []
    rows.sort()
    return [r[1] for r in rows[:n]]


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
            log(f"empty response attempt {attempt+1}/{MAX_RETRIES+1} — backoff {delay:.1f}s")
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
    failing = _failing_tickers(5)

    # Keep prompt under 400 chars for fast inference
    prompt = (
        f"S&P500 XGBoost mastery: {mastered}/502 done. "
        f"Near-mastery tickers: {failing}. "
        "Which 1 ticker should we focus on NEXT for intraday mastery and WHY? "
        "Answer in 2 sentences max. Be specific."
    )

    t0 = time.monotonic()
    response = call_deepseek_with_backoff(prompt)
    elapsed = round(time.monotonic() - t0, 2)

    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "lane": LANE,
        "mastered": mastered,
        "failing_sample": failing,
        "prompt_chars": len(prompt),
        "response": response[:500],
        "elapsed_s": elapsed,
    }
    _append_jsonl(entry)
    _write_latest(entry)
    log(f"done in {elapsed}s — response[:{min(120,len(response))}]: {response[:120]}")
    gc.collect()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"FATAL: {type(exc).__name__}: {exc}")
        import traceback
        log(traceback.format_exc())
        sys.exit(1)
