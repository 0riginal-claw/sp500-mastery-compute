"""
ollama_worker_daemon.py — Continuous local-Ollama rater for WIRE_CANDIDATEs.

Polls feature_discovery/wiring_queue.jsonl (and proactive/stream.jsonl as
fallback) for WIRE_CANDIDATE entries, asks Ollama (free, local, $0/call) to:
  - rate quality on 0-10 scale
  - suggest 1-2 improvements
  - flag integration risks

Writes ratings JSONL to:
    feature_discovery/ollama_ratings.jsonl

Consumers (feature_wiring_consumer_daemon.py) can join on feature_name and
treat rating>=7 as SAFE/AUTO-WIRE.

Designed to run as a LaunchAgent (KeepAlive=true). Loop sleep: 60s.
Ollama concurrency cap: 4 parallel calls (single GPU/CPU bottleneck).

Author: 2026-05-17 (max-out Ollama mandate)
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

# ── Paths ────────────────────────────────────────────────────────────────────
WORK = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools/s&p500-ticker-mastery"
)
AI_TOOLS = WORK.parent  # …/My Drive/AI-Tools
SCRIPTS_DIR = WORK / "scripts"

WIRING_QUEUE = WORK / "feature_discovery" / "wiring_queue.jsonl"
STREAM_JSONL = WORK / "proactive" / "stream.jsonl"

# Per the task: ratings go to AI-Tools/feature_discovery/ollama_ratings.jsonl
# (parallel to the s&p500 dir, accessible to all consumers under AI-Tools).
RATINGS_DIR = AI_TOOLS / "feature_discovery"
RATINGS_JSONL = RATINGS_DIR / "ollama_ratings.jsonl"
STATE_FILE = RATINGS_DIR / "ollama_worker_state.json"

LOG_PATH = AI_TOOLS / "logs" / "ollama_worker.log"

# ── Ollama helper ────────────────────────────────────────────────────────────
sys.path.insert(0, str(AI_TOOLS / "scripts"))
try:
    from ollama_helper import call_ollama  # type: ignore[import]
    _HAS_OLLAMA = True
except Exception as _exc:
    _HAS_OLLAMA = False
    call_ollama = None  # type: ignore[assignment]

# ── Tunables ─────────────────────────────────────────────────────────────────
SLEEP_SECONDS = 60
BATCH_SIZE = 4              # per cycle (== Ollama concurrency cap)
OLLAMA_TIMEOUT_S = 120
OLLAMA_MODEL = "qwen2.5-coder:7b"


def log(msg: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    line = f"[{ts}] {msg}\n"
    try:
        with open(LOG_PATH, "a") as fh:
            fh.write(line)
    except Exception:
        pass
    print(line, end="", flush=True)


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"rated_names": [], "rated_count": 0, "last_run": None}


def save_state(s: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    s["last_run"] = datetime.now(timezone.utc).isoformat()
    STATE_FILE.write_text(json.dumps(s, indent=2, default=str))


# ── Candidate pull from queues ──────────────────────────────────────────────


def _read_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except Exception:
        return


def pull_unrated_candidates(rated_names: set[str], cap: int = 64) -> list[dict]:
    """Collect candidates from both queue files; dedupe and drop already-rated."""
    bag: list[dict] = []
    seen: set[str] = set()

    for path in (WIRING_QUEUE, STREAM_JSONL):
        for obj in _read_jsonl(path):
            # Wire events look like {"event": "wire_candidate", "feature_name": ...}
            ev = obj.get("event")
            name = obj.get("feature_name") or obj.get("name")
            if not name:
                continue
            if not (ev == "wire_candidate" or "feature_name" in obj):
                continue
            if name in rated_names or name in seen:
                continue
            seen.add(name)
            bag.append(obj)
            if len(bag) >= cap:
                return bag
    return bag


def _rate_prompt(cand: dict) -> str:
    fname = cand.get("feature_name", "?")
    desc = cand.get("description", "")[:300]
    src = cand.get("data_source", "")
    sig = cand.get("function_signature", "")
    shift_safe = cand.get("shift_1_safe", "unclear")
    cost = cand.get("integration_cost", "MED")
    paid = cand.get("requires_paid_api", "no")
    return (
        "Rate this proposed trading-ML feature for a S&P 500 XGBoost mean-reversion pipeline "
        "(722 existing features). Be terse and structured.\n\n"
        f"FEATURE NAME: {fname}\n"
        f"DESCRIPTION : {desc}\n"
        f"DATA SOURCE : {src}\n"
        f"SIGNATURE   : {sig}\n"
        f"SHIFT_1_SAFE: {shift_safe}\n"
        f"INTEG_COST  : {cost}\n"
        f"REQ_PAID_API: {paid}\n\n"
        "Output EXACTLY this format and nothing else:\n"
        "RATING: <integer 0-10>\n"
        "IMPROVEMENT_1: <one-line specific change>\n"
        "IMPROVEMENT_2: <one-line specific change>\n"
        "INTEGRATION_RISK: <one-line, or NONE>\n"
        "NOTES: <one-line, or NONE>\n"
    )


def _parse_rating(text: str) -> dict:
    out: dict[str, Any] = {"rating": None, "improvement_1": "", "improvement_2": "", "integration_risk": "", "notes": ""}
    if not text:
        return out
    m = re.search(r"RATING\s*:\s*(\d+(?:\.\d+)?)", text)
    if m:
        try:
            out["rating"] = float(m.group(1))
        except Exception:
            pass
    for key, regex in (
        ("improvement_1", r"IMPROVEMENT_1\s*:\s*(.+)"),
        ("improvement_2", r"IMPROVEMENT_2\s*:\s*(.+)"),
        ("integration_risk", r"INTEGRATION_RISK\s*:\s*(.+)"),
        ("notes", r"NOTES\s*:\s*(.+)"),
    ):
        m = re.search(regex, text)
        if m:
            out[key] = m.group(1).strip()
    return out


def rate_one(cand: dict) -> dict:
    fname = cand.get("feature_name", "unknown")
    t0 = time.monotonic()
    prompt = _rate_prompt(cand)
    if not _HAS_OLLAMA:
        return {
            "feature_name": fname,
            "rated_at": datetime.now(timezone.utc).isoformat(),
            "ok": False,
            "error": "ollama_helper unavailable",
            "elapsed_s": round(time.monotonic() - t0, 2),
        }
    try:
        res = call_ollama(prompt, model=OLLAMA_MODEL, max_tokens=300, timeout=OLLAMA_TIMEOUT_S)
        if not res.get("success"):
            return {
                "feature_name": fname,
                "rated_at": datetime.now(timezone.utc).isoformat(),
                "ok": False,
                "error": res.get("error", "unknown"),
                "latency_s": res.get("latency_s"),
                "elapsed_s": round(time.monotonic() - t0, 2),
            }
        parsed = _parse_rating(res.get("text", ""))
        return {
            "feature_name": fname,
            "rated_at": datetime.now(timezone.utc).isoformat(),
            "ok": True,
            "model": res.get("model", OLLAMA_MODEL),
            "cost_usd": 0.0,
            "latency_s": res.get("latency_s"),
            "rating": parsed["rating"],
            "improvement_1": parsed["improvement_1"],
            "improvement_2": parsed["improvement_2"],
            "integration_risk": parsed["integration_risk"],
            "notes": parsed["notes"],
            "raw_response": res.get("text", "")[:600],
            "elapsed_s": round(time.monotonic() - t0, 2),
        }
    except Exception as exc:
        return {
            "feature_name": fname,
            "rated_at": datetime.now(timezone.utc).isoformat(),
            "ok": False,
            "error": str(exc),
            "elapsed_s": round(time.monotonic() - t0, 2),
        }


def write_ratings(rows: list[dict]) -> None:
    if not rows:
        return
    RATINGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RATINGS_JSONL, "a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def run_one_cycle(state: dict) -> dict:
    """One iteration: pull up to BATCH_SIZE unrated candidates, rate in parallel."""
    rated_names = set(state.get("rated_names", []))
    candidates = pull_unrated_candidates(rated_names, cap=BATCH_SIZE * 4)
    if not candidates:
        log("no unrated candidates in queues — sleeping")
        return {"rated": 0, "skipped": 0}

    # Take a random subset of BATCH_SIZE (avoids head-of-line starvation on a
    # growing queue). Random sampling biases toward freshness if queue is big.
    batch = random.sample(candidates, min(BATCH_SIZE, len(candidates)))

    log(f"rating batch of {len(batch)} candidates (queue size: {len(candidates)})")
    rows: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=BATCH_SIZE) as ex:
        futs = {ex.submit(rate_one, c): c for c in batch}
        for fut in concurrent.futures.as_completed(futs, timeout=OLLAMA_TIMEOUT_S + 30):
            try:
                rows.append(fut.result())
            except Exception as exc:
                c = futs[fut]
                rows.append({
                    "feature_name": c.get("feature_name", "unknown"),
                    "ok": False,
                    "error": str(exc),
                    "rated_at": datetime.now(timezone.utc).isoformat(),
                })

    write_ratings(rows)
    ok_rows = [r for r in rows if r.get("ok")]
    for r in ok_rows:
        rated_names.add(r["feature_name"])
    # Cap rated_names list (state file stays small)
    state["rated_names"] = sorted(rated_names)[-2000:]
    state["rated_count"] = state.get("rated_count", 0) + len(ok_rows)
    save_state(state)

    safe_count = sum(1 for r in ok_rows if isinstance(r.get("rating"), (int, float)) and r["rating"] >= 7)
    log(f"cycle complete — rated={len(ok_rows)}/{len(batch)} safe(rating>=7)={safe_count}")
    return {"rated": len(ok_rows), "skipped": len(batch) - len(ok_rows), "safe": safe_count}


def main() -> None:
    log(f"[START] ollama_worker_daemon pid={os.getpid()} model={OLLAMA_MODEL} sleep={SLEEP_SECONDS}s")
    if not _HAS_OLLAMA:
        log("[FATAL] ollama_helper.py not importable — exiting after 30s")
        time.sleep(30)
        sys.exit(1)
    state = load_state()
    while True:
        try:
            run_one_cycle(state)
        except Exception as exc:
            log(f"[ERROR] cycle exception: {exc}")
        try:
            time.sleep(SLEEP_SECONDS)
        except KeyboardInterrupt:
            log("[STOP] KeyboardInterrupt")
            sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        log("=== TEST MODE: 1 cycle, no loop ===")
        st = load_state()
        res = run_one_cycle(st)
        print(json.dumps(res, indent=2))
    else:
        main()
