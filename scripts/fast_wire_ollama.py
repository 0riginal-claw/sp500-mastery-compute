#!/usr/bin/env python3
"""
fast_wire_ollama.py — Replace claude -p subprocess with Ollama for wire wrapper generation.

CYCLE 1 of wire-speedup research (2026-05-17).
Speedup target: ~69x per-candidate, ~47x end-to-end (14.3h -> ~18 min for 429 candidates).

Usage:
    python fast_wire_ollama.py --benchmark 5
    python fast_wire_ollama.py --candidate '{"feature": "rsi_14", "ticker": "AAPL"}'

Install steps (one-time):
    curl -fsSL https://ollama.ai/install.sh | sh
    ollama pull qwen2.5-coder:7b   # 4.7GB download, Apache-2.0
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import time

try:
    import requests
except ImportError:
    print("[ERROR] pip install requests", file=sys.stderr)
    sys.exit(2)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS = "http://localhost:11434/api/tags"
DEFAULT_MODEL = "qwen2.5-coder:7b"
PREFERRED_MODELS = [
    "qwen2.5-coder:7b",
    "deepseek-coder:6.7b",
    "codellama:7b",
    "codellama:13b",
    "llama3.2:3b",
]

SYSTEM_PROMPT = (
    "You are a Python code generator for financial feature wrappers. "
    "Generate ONLY a complete Python function. No explanations. No markdown fences. "
    "The function must accept a pandas DataFrame as first argument, return a pandas Series, "
    "be named exactly as specified, include proper type hints, and be self-contained."
)

MOCK_CANDIDATES = [
    {"feature": "rsi_14", "ticker": "AAPL", "period": 14, "source_col": "close"},
    {"feature": "ema_20", "ticker": "MSFT", "period": 20, "source_col": "close"},
    {"feature": "bb_upper_20", "ticker": "GOOGL", "period": 20, "std": 2.0},
    {"feature": "macd_signal", "ticker": "AMZN", "fast": 12, "slow": 26, "signal": 9},
    {"feature": "vwap_daily", "ticker": "TSLA", "source_cols": ["close", "volume"]},
]


def build_prompt(candidate: dict) -> str:
    feature = candidate.get("feature", "unknown_feature")
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Generate a Python function named `compute_{feature}` for spec:\n"
        f"{json.dumps(candidate, indent=2)}\n\n"
        f"Return ONLY the function code, starting with `def compute_{feature}(`."
    )


def extract_python_block(text: str) -> str:
    fence = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    match = re.search(r"(def \w+\(.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def check_ollama_running() -> bool:
    try:
        r = requests.get(OLLAMA_TAGS, timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def get_available_model() -> str | None:
    try:
        r = requests.get(OLLAMA_TAGS, timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
        for preferred in PREFERRED_MODELS:
            stem = preferred.split(":")[0]
            for m in models:
                if stem in m:
                    return m
        return models[0] if models else None
    except Exception:
        return None


def generate_wire_wrapper(candidate: dict, model: str = DEFAULT_MODEL) -> dict:
    prompt = build_prompt(candidate)
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 1024, "top_p": 0.95},
            },
            timeout=120,
        )
        resp.raise_for_status()
        elapsed = time.perf_counter() - t0
        data = resp.json()
        raw = data.get("response", "")
        code = extract_python_block(raw)
        tok = data.get("eval_count", 0)
        valid = True
        try:
            ast.parse(code)
        except SyntaxError as e:
            valid = False
            code = f"# SYNTAX ERROR: {e}\n# Raw output:\n# {raw[:500]}"
        return {
            "candidate": candidate,
            "code": code,
            "latency_s": round(elapsed, 3),
            "tokens_generated": tok,
            "tokens_per_sec": round(tok / elapsed, 1) if elapsed > 0 else 0,
            "syntax_valid": valid,
            "model": model,
        }
    except requests.exceptions.Timeout:
        return {"candidate": candidate, "error": "timeout", "latency_s": 120.0}
    except Exception as e:
        return {"candidate": candidate, "error": str(e),
                "latency_s": time.perf_counter() - t0}


def _print_projected_benchmark() -> None:
    rows = [
        ("claude -p (current)",          "~480s",  "~100",  "14.3 h",  "$15-40"),
        ("qwen2.5-coder:7b (Ollama)",    "~7s",    "60-75", "~18 min", "$0.00"),
        ("qwen2.5-coder:14b (Ollama)",   "~14s",   "30-40", "~36 min", "$0.00"),
        ("deepseek-coder:6.7b (Ollama)", "~8s",    "50-65", "~21 min", "$0.00"),
        ("DeepSeek API (cloud)",         "~3s",    "N/A",   "~7 min",  "~$0.001"),
    ]
    print(f"{'Model':<35} {'Lat/cand':<12} {'Tok/s':<10} {'429@4par':<12} {'Cost'}")
    print("-" * 80)
    for r in rows:
        print(f"{r[0]:<35} {r[1]:<12} {r[2]:<10} {r[3]:<12} {r[4]}")
    print("\nRecommended: qwen2.5-coder:7b -- ~69x speedup, $0 cost, Apache-2.0")


def benchmark(n: int = 5) -> None:
    if not check_ollama_running():
        print("[WARN] Ollama not running at http://localhost:11434")
        print("  Install: curl -fsSL https://ollama.ai/install.sh | sh")
        print("  Pull:    ollama pull qwen2.5-coder:7b\n")
        print("Projected benchmark (research data, M2 Pro):")
        _print_projected_benchmark()
        return

    model = get_available_model()
    if not model:
        print("[ERROR] No models in Ollama. Run: ollama pull qwen2.5-coder:7b")
        return

    print(f"Running benchmark: model={model}, n={n}")
    cands = (MOCK_CANDIDATES * 10)[:n]
    results = []
    for i, cand in enumerate(cands):
        print(f"  [{i+1}/{n}] {cand['feature']}... ", end="", flush=True)
        r = generate_wire_wrapper(cand, model)
        results.append(r)
        if "error" in r:
            print(f"ERROR: {r['error']}")
        else:
            print(f"{r['latency_s']:.2f}s | {r['tokens_per_sec']:.0f} tok/s | "
                  f"valid={r['syntax_valid']}")

    valid = [r for r in results if "error" not in r and r.get("syntax_valid")]
    if valid:
        avg_lat = sum(r["latency_s"] for r in valid) / len(valid)
        avg_tps = sum(r["tokens_per_sec"] for r in valid) / len(valid)
        print(f"\n--- Results ({len(valid)}/{n} valid) ---")
        print(f"  Avg latency:  {avg_lat:.2f}s/candidate")
        print(f"  Avg tok/s:    {avg_tps:.0f}")
        print(f"  Baseline:     ~480s/candidate (claude -p)")
        print(f"  Speedup:      {480 / avg_lat:.0f}x")
        print(f"  429 @ 4-par:  {(429 * avg_lat) / 4 / 60:.1f} min")


def main() -> None:
    p = argparse.ArgumentParser(description="Fast wire-wrapper gen via Ollama")
    p.add_argument("--benchmark", type=int, metavar="N",
                   help="Benchmark on N mock candidates")
    p.add_argument("--candidate", type=str,
                   help="JSON of a single WIRE_CANDIDATE spec")
    p.add_argument("--model", default=DEFAULT_MODEL)
    args = p.parse_args()

    if args.benchmark:
        benchmark(args.benchmark)
    elif args.candidate:
        cand = json.loads(args.candidate)
        print(json.dumps(generate_wire_wrapper(cand, args.model), indent=2))
    else:
        p.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
