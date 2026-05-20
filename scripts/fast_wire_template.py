"""
fast_wire_template.py -- Template-based WIRE_CANDIDATE wrapper generator.

CYCLE 2 of wire-speedup research (2026-05-17).
No LLM required. Sub-millisecond per wrapper. Falls back gracefully to LLM
when no template matches (estimated ~8% of 429 candidates).

Usage:
    from fast_wire_template import generate_wrapper
    result = generate_wrapper(candidate_spec)
    if result["template_match"]:
        exec(result["code"])
    else:
        # escalate to LLM (cycle-1 fast_wire_ollama.py)
        pass

Benchmark:
    python fast_wire_template.py
"""
from __future__ import annotations

import ast
import statistics
import string
import time
from typing import Any

# ---------------------------------------------------------------------------
# Template registry -- 9 families covering ~92% of rolling/stat candidates
# ---------------------------------------------------------------------------
_TEMPLATES: dict[str, string.Template] = {
    "rolling_stat": string.Template(
        "def feat_${name}(df, col=\"${col}\", window=${window}):\n"
        "    return df[col].rolling(${window}).${stat}()\n"
    ),
    "zscore": string.Template(
        "def feat_${name}(df, col=\"${col}\", window=${window}):\n"
        "    r = df[col].rolling(${window})\n"
        "    return (df[col] - r.mean()) / (r.std() + 1e-9)\n"
    ),
    "momentum": string.Template(
        "def feat_${name}(df, col=\"${col}\", period=${period}):\n"
        "    lag = df[col].shift(${period})\n"
        "    return (df[col] - lag) / (lag + 1e-9)\n"
    ),
    "lag": string.Template(
        "def feat_${name}(df, col=\"${col}\", period=${period}):\n"
        "    return df[col].shift(${period})\n"
    ),
    "ewma": string.Template(
        "def feat_${name}(df, col=\"${col}\", span=${span}):\n"
        "    return df[col].ewm(span=${span}, adjust=False).mean()\n"
    ),
    "ratio_to_mean": string.Template(
        "def feat_${name}(df, col=\"${col}\", window=${window}):\n"
        "    return df[col] / (df[col].rolling(${window}).mean() + 1e-9)\n"
    ),
    "rolling_rank": string.Template(
        "def feat_${name}(df, col=\"${col}\", window=${window}):\n"
        "    return df[col].rolling(${window}).apply(\n"
        "        lambda x: (x.argsort().argsort()[-1] + 1) / len(x), raw=True\n"
        "    )\n"
    ),
    "spread": string.Template(
        "def feat_${name}(df, col_a=\"${col_a}\", col_b=\"${col_b}\"):\n"
        "    return df[col_a] - df[col_b]\n"
    ),
    "bollinger_pct_b": string.Template(
        "def feat_${name}(df, col=\"${col}\", window=${window}, n_std=${n_std}):\n"
        "    r = df[col].rolling(${window})\n"
        "    mid = r.mean()\n"
        "    band = r.std() * ${n_std}\n"
        "    return (df[col] - (mid - band)) / (2 * band + 1e-9)\n"
    ),
}


# ---------------------------------------------------------------------------
# Classifier -- maps a spec dict -> template family
# ---------------------------------------------------------------------------
def _classify(spec: dict[str, Any]) -> str | None:
    kind = (spec.get("kind") or spec.get("type") or "").lower()
    stat = (spec.get("stat") or "").lower()

    if kind in ("rolling_rank", "rank"):
        return "rolling_rank"
    if kind in ("bollinger", "pct_b", "bollinger_pct_b"):
        return "bollinger_pct_b"
    if kind in ("zscore", "z_score", "standardize"):
        return "zscore"
    if kind in ("momentum", "roc", "rate_of_change", "pct_change"):
        return "momentum"
    if kind in ("lag", "shift", "lagged"):
        return "lag"
    if kind in ("ewma", "ema", "ewm"):
        return "ewma"
    if kind in ("ratio", "ratio_to_mean", "normalized"):
        return "ratio_to_mean"
    if kind in ("spread", "diff_cols", "cross_diff"):
        return "spread"
    valid_stats = {"mean", "std", "min", "max", "sum", "median", "kurt", "skew", "var"}
    if kind in ("rolling", "rolling_stat") and stat in valid_stats:
        return "rolling_stat"
    if stat in valid_stats:
        return "rolling_stat"
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_wrapper(spec: dict[str, Any], validate: bool = True) -> dict[str, Any]:
    t0 = time.perf_counter()
    family = _classify(spec)
    if family is None:
        return {
            "template_match": False, "family": None, "code": None,
            "needs_llm": True, "gen_us": (time.perf_counter() - t0) * 1e6,
        }

    tmpl = _TEMPLATES[family]
    col = spec.get("col") or spec.get("column") or "close"
    window = spec.get("window", 20)
    period = spec.get("period", 14)
    span = spec.get("span", 14)
    name = spec.get("name") or f"{family}_{col}_{window or period or span}"

    subs = {
        "name": name, "col": col, "window": window, "period": period, "span": span,
        "stat": spec.get("stat", "mean"),
        "col_a": spec.get("col_a", "close"), "col_b": spec.get("col_b", "open"),
        "n_std": spec.get("n_std", 2),
    }

    try:
        code = tmpl.substitute(subs)
    except KeyError as exc:
        return {
            "template_match": False, "family": family, "code": None,
            "needs_llm": True, "gen_us": (time.perf_counter() - t0) * 1e6,
            "error": str(exc),
        }

    if validate:
        try:
            ast.parse(code, mode="exec")
        except SyntaxError as e:
            return {
                "template_match": False, "family": family, "code": code,
                "needs_llm": True, "gen_us": (time.perf_counter() - t0) * 1e6,
                "error": f"syntax: {e}",
            }

    return {
        "template_match": True, "family": family, "code": code,
        "needs_llm": False, "gen_us": (time.perf_counter() - t0) * 1e6,
    }


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def _benchmark() -> None:
    candidates = [
        {"kind": "rolling_stat", "stat": "mean",  "col": "close",   "window": 20, "name": "ma20"},
        {"kind": "zscore",       "col": "volume", "window": 30,     "name": "vol_z30"},
        {"kind": "momentum",     "col": "close",  "period": 5,      "name": "mom5"},
        {"kind": "ewma",         "col": "close",  "span": 12,       "name": "ema12"},
        {"kind": "bollinger",    "col": "close",  "window": 20, "n_std": 2, "name": "bb_pct"},
        {"kind": "lag",          "col": "close",  "period": 1,      "name": "lag1"},
        {"kind": "rolling_rank", "col": "returns","window": 60,     "name": "rank60"},
        {"kind": "spread",       "col_a": "high", "col_b": "low",   "name": "hl_spread"},
        {"kind": "ratio_to_mean","col": "close",  "window": 50,     "name": "norm50"},
        {"kind": "exotic_neural_feature"},
    ]
    n_reps = 10_000
    print(f"{'Candidate':<22} {'Family':<18} {'Match':<6} {'us':>8}")
    print("-" * 58)
    matched = []
    for spec in candidates:
        reps = [generate_wrapper(spec, validate=False)["gen_us"] for _ in range(n_reps)]
        med = statistics.median(reps)
        r = generate_wrapper(spec)
        name = spec.get("name") or spec.get("kind") or "unknown"
        print(f"{name:<22} {str(r.get('family') or 'N/A'):<18} "
              f"{str(r['template_match']):<6} {med:>8.3f}")
        if r["template_match"]:
            matched.append(med)
    avg = statistics.mean(matched)
    n = 429
    total_ms = avg * n / 1000
    print(f"\nTemplate avg: {avg:.1f} us  |  {n} cands: {total_ms:.2f} ms")
    print(f"vs LLM serial (~8min/cand): {8 * n / 60:.1f} h")
    print(f"Speedup vs LLM serial: {(8 * n * 60 * 1e6) / (avg * n):,.0f}x")


if __name__ == "__main__":
    _benchmark()
