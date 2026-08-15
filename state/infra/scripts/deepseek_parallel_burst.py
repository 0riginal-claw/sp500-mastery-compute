"""
deepseek_parallel_burst.py — Fire N parallel DeepSeek queries on diverse trading topics.

Runs concurrent.futures over 10 topics:
  trading strategies, factor models, alt-data sources, microstructure patterns,
  regime detection methods, sentiment signals, optionsflow, congressional trades,
  insider patterns, news embeddings

For each topic, asks DeepSeek for 3 concrete WIRE_CANDIDATE-formatted feature
proposals. All results are appended as WIRE_CANDIDATE blocks to:
    AI-Tools/feature_discovery/reports/wire_candidates_deepseek_burst_<DATE>.md

Designed to be invoked ad-hoc (one-shot) OR on a cron schedule. The
feature_wiring_consumer_daemon will auto-pick-up emitted candidates.

Cost: ~10 calls * $0.000005 = $0.00005/run (rounds to $0.0001/run).

Usage:
    python scripts/deepseek_parallel_burst.py                  # 10 topics
    python scripts/deepseek_parallel_burst.py --n 20           # 20 topics (repeats angles with seed jitter)
    python scripts/deepseek_parallel_burst.py --max-workers 5  # cap concurrency
    python scripts/deepseek_parallel_burst.py --dry-run        # build prompts but don't call

Author: 2026-05-17 (max-out DeepSeek mandate)
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Paths ────────────────────────────────────────────────────────────────────
AI_TOOLS = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools"
)
SP_WORK = AI_TOOLS / "s&p500-ticker-mastery"
SCRIPTS_DIR = SP_WORK / "scripts"
REPORTS_DIR = AI_TOOLS / "feature_discovery" / "reports"

# ── Imports: direct DeepSeek + WIRE_CANDIDATE ────────────────────────────────
sys.path.insert(0, str(SCRIPTS_DIR))
from deepseek_direct import call_deepseek_direct  # noqa: E402

try:
    from wire_candidate import (  # type: ignore[import]
        emit as _wire_emit,
        parse_markdown_blocks as _wire_parse,
        WIRE_CANDIDATE_PROMPT_SUFFIX as _WIRE_SUFFIX,
    )
    _HAS_WIRE = True
except Exception:
    _HAS_WIRE = False
    _WIRE_SUFFIX = ""

# ── Topics ────────────────────────────────────────────────────────────────────
TOPICS = [
    "trading strategies (mean-reversion + momentum hybrids)",
    "factor models (Fama-French, Q-factor, MSCI Barra equivalents)",
    "alternative data sources (satellite, credit-card transactions, web traffic)",
    "market microstructure patterns (VPIN, Kyle's lambda, tick-imbalance, order book depth)",
    "regime detection methods (HMM, change-point, vol clustering, structural breaks)",
    "sentiment signals (StockTwits, Reddit/WallStreetBets, news headlines, FinBERT)",
    "options flow patterns (unusual options activity, put/call skew, dark pool flow)",
    "congressional + senate trades (PTR disclosures, lag patterns, sector concentration)",
    "insider transaction patterns (Form 4 cluster buys, 10b5-1 abandonments)",
    "news embeddings (BERT/sentence transformers on financial news, topic clustering)",
]

# ── Tunables ──────────────────────────────────────────────────────────────────
DEFAULT_MAX_WORKERS = 10  # all 10 topics in parallel
PER_CALL_TIMEOUT = 90
PER_CALL_MAX_TOKENS = 800
PER_CALL_TEMPERATURE = 0.5


def _build_prompt(topic: str) -> str:
    suffix = _WIRE_SUFFIX if _HAS_WIRE else ""
    return (
        "You are a quant feature engineering analyst. The pipeline is an S&P 500 daily "
        "mean-reversion XGBoost model with ~722 features (v7/v8) covering technical "
        "indicators, intraday/VWAP, macro (VIX/DXY/yields), alt-data (EDGAR, "
        "congressional, lobbying, insider Form 4), Qlib Alpha158, news sentiment, "
        "Featuretools DFS, multi-timeframe (5/15/60/240min), and Yang-Zhang vol.\n\n"
        f"FOCUS AREA: {topic}\n\n"
        "Give 3 SPECIFIC, NEW feature proposals in this area that would complement "
        "(not duplicate) the existing 722. Each must include:\n"
        "  1. A unique snake_case feature_name\n"
        "  2. A 1-line description of what it captures\n"
        "  3. The data source (github_url, drive_path, or api_endpoint)\n"
        "  4. The compute recipe (terse pseudocode)\n"
        "  5. shift_1_safe assessment (yes/no/unclear)\n"
        "  6. Integration cost (LOW/MED/HIGH)\n"
        "  7. Expected PF lift (float or 'unknown')\n\n"
        f"{suffix}"
    )


def fire_one(topic: str, timeout: int = PER_CALL_TIMEOUT) -> dict[str, Any]:
    t0 = time.monotonic()
    try:
        text = call_deepseek_direct(
            _build_prompt(topic),
            timeout=timeout,
            max_tokens=PER_CALL_MAX_TOKENS,
            temperature=PER_CALL_TEMPERATURE,
        )
        return {
            "topic": topic,
            "ok": bool(text),
            "text": text,
            "elapsed_s": round(time.monotonic() - t0, 2),
        }
    except Exception as exc:
        return {
            "topic": topic,
            "ok": False,
            "text": "",
            "error": str(exc),
            "elapsed_s": round(time.monotonic() - t0, 2),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=len(TOPICS), help="Number of topics to fire (default 10).")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--timeout", type=int, default=PER_CALL_TIMEOUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    topics = (TOPICS * ((args.n // len(TOPICS)) + 1))[:args.n]
    print(f"[burst] topics={len(topics)} max_workers={args.max_workers} timeout={args.timeout}s dry_run={args.dry_run}")

    if args.dry_run:
        for t in topics:
            print(f"  - {t}")
        return 0

    burst_t0 = time.monotonic()
    results: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {ex.submit(fire_one, t, args.timeout): t for t in topics}
        for fut in concurrent.futures.as_completed(futs, timeout=args.timeout + 60):
            try:
                results.append(fut.result())
            except Exception as exc:
                results.append({"topic": futs[fut], "ok": False, "text": "", "error": str(exc)})

    ok = sum(1 for r in results if r.get("ok"))
    err = len(results) - ok
    print(f"[burst] DeepSeek phase done — ok={ok} err={err} elapsed={time.monotonic() - burst_t0:.1f}s")

    # ── Parse + emit WIRE_CANDIDATEs ─────────────────────────────────────────
    wire_emitted = 0
    md_path_used = None
    jsonl_path_used = None
    if _HAS_WIRE:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        md_path = REPORTS_DIR / f"wire_candidates_deepseek_burst_{date_str}.md"
        all_cands: list[dict] = []
        for r in results:
            if not r.get("ok") or not r.get("text"):
                continue
            try:
                cands = _wire_parse(r["text"], discovered_by="feature_discovery")
                # Tag the topic so consumers can group
                for c in cands:
                    c["citations"] = (c.get("citations") or []) + [f"deepseek_topic:{r['topic']}"]
                all_cands.extend(cands)
            except Exception as exc:
                print(f"  parse failed for topic {r['topic']}: {exc}")

        if all_cands:
            try:
                res = _wire_emit(
                    all_cands,
                    discovered_by="feature_discovery",
                    write_md=True,
                    write_jsonl=True,
                    md_path=md_path,
                )
                wire_emitted = res["emitted"]
                md_path_used = res["md_path"]
                jsonl_path_used = res["jsonl_path"]
                print(f"[burst] WIRE: emitted {wire_emitted} → md={md_path_used}")
                print(f"[burst] WIRE: jsonl={jsonl_path_used}")
            except Exception as exc:
                print(f"[burst] WIRE emit failed: {exc}")
    else:
        # No wire_candidate available — dump raw responses to a markdown file
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        md_path = REPORTS_DIR / f"deepseek_burst_raw_{date_str}.md"
        md_path.parent.mkdir(parents=True, exist_ok=True)
        with open(md_path, "a", encoding="utf-8") as fh:
            fh.write(f"# DeepSeek burst — {datetime.now(timezone.utc).isoformat()}\n\n")
            for r in results:
                fh.write(f"## TOPIC: {r['topic']}\n\n```\n{r.get('text', '')[:4000]}\n```\n\n")
        md_path_used = str(md_path)
        print(f"[burst] WIRE unavailable — dumped raw responses to {md_path_used}")

    summary = {
        "n_topics": len(results),
        "ok": ok,
        "errors": err,
        "wire_emitted": wire_emitted,
        "md_path": md_path_used,
        "jsonl_path": jsonl_path_used,
        "total_elapsed_s": round(time.monotonic() - burst_t0, 2),
    }
    print("\n[burst] SUMMARY:")
    print(json.dumps(summary, indent=2, default=str))
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
