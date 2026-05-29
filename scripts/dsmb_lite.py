"""dsmb_lite.py — Independent Adjudicator Panel (DSMB-Lite).

Convenes a 3-advisor DeepSeek panel that mechanically scores a SAP → paper
promotion request against the frozen rubric at:
  AI-Tools/reports/adjudicator_rubric.md

R4 chairman demand: the right to skip independent adjudication has NOT been
earned. Every promotion must be scored against 5 dimensions (0-10 each):

  1. Methodological soundness
  2. Out-of-sample evidence
  3. Operational readiness
  4. Risk exposure
  5. Pre-registration honored

Pass = total >= 40 / 50  AND  every dimension >= 6.

Backend: direct DeepSeek HTTP (no openclaw subprocess — established pattern
from autonomous_mode_daemon._deepseek_direct, ~2s per call vs 80-90s via
openclaw chain).

Persistence: every decision lands at
  state/dsmb_lite/decisions/<request_id>_<utc>.json
with the rubric sha256 pinned so future audits can detect rubric drift.

CLI usage:
  python dsmb_lite.py --request-json /path/to/request.json
  python dsmb_lite.py --tool-context-stdin   # for the PreToolUse hook
  python dsmb_lite.py --smoke-apa            # synthetic APA test

Library usage:
  from dsmb_lite import convene_panel
  result = convene_panel(request_dict)
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
DRIVE_BASE = Path(
    os.environ.get(
        "DRIVE_BASE",
        "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive",
    )
)
AI_TOOLS = DRIVE_BASE / "AI-Tools"
RUBRIC_PATH = AI_TOOLS / "reports" / "adjudicator_rubric.md"
DECISIONS_DIR = AI_TOOLS / "state" / "dsmb_lite" / "decisions"
ZG_TMP = Path("/Volumes/ZG-2TB/zg/tmp/dsmb_lite")
STATUS_PATH = ZG_TMP / "status.md"

# DeepSeek (mirror autonomous_mode_daemon._deepseek_direct)
_DEEPSEEK_KEYFILE = (
    AI_TOOLS / "home" / ".openclaw" / "agents" / "main" / "agent" / "auth-profiles.json"
)
_DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
_DEEPSEEK_MODEL = "deepseek-chat"
_DEEPSEEK_CACHED_KEY: str | None = None

# Rubric (cached) — sha256 + raw text
_RUBRIC_CACHED: tuple[str, str] | None = None  # (text, sha256)

PASS_TOTAL_THRESHOLD = 40
PASS_DIM_THRESHOLD = 6
DIMENSIONS = [
    "methodological",
    "oos",
    "operational",
    "risk",
    "pre_reg",
]
ADVISOR_TIMEOUT_S = 90


# ─────────────────────────────────────────────────────────────────────────────
# DeepSeek direct (copied from autonomous_mode_daemon for zero-dep import)
# ─────────────────────────────────────────────────────────────────────────────
def _deepseek_get_api_key() -> str | None:
    global _DEEPSEEK_CACHED_KEY
    if _DEEPSEEK_CACHED_KEY:
        return _DEEPSEEK_CACHED_KEY
    env_key = os.environ.get("DEEPSEEK_API_KEY")
    if env_key:
        _DEEPSEEK_CACHED_KEY = env_key
        return env_key
    try:
        with open(_DEEPSEEK_KEYFILE) as f:
            d = json.load(f)
        key = d.get("profiles", {}).get("deepseek:default", {}).get("key")
        if key:
            _DEEPSEEK_CACHED_KEY = key
            return key
    except (OSError, json.JSONDecodeError) as e:
        print(f"[dsmb_lite] keyfile load failed: {e}", file=sys.stderr)
    return None


def _deepseek_direct(
    prompt: str,
    timeout_s: int = ADVISOR_TIMEOUT_S,
    max_tokens: int = 1200,
    temperature: float = 0.2,
) -> str | None:
    """Direct DeepSeek HTTP call. Returns text or None. Never raises."""
    api_key = _deepseek_get_api_key()
    if not api_key:
        print("[dsmb_lite] no DeepSeek API key available", file=sys.stderr)
        return None
    try:
        payload = json.dumps({
            "model": _DEEPSEEK_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode("utf-8")
        req = urllib.request.Request(
            _DEEPSEEK_API_URL,
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 — fixed HTTPS URL
            result = json.loads(resp.read().decode("utf-8"))
        choices = result.get("choices") or []
        if choices:
            content = (choices[0].get("message") or {}).get("content", "").strip()
            return content or None
        return None
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8")[:200]
        except Exception:  # noqa: BLE001
            pass
        print(f"[dsmb_lite] HTTP {e.code}: {body}", file=sys.stderr)
        return None
    except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError) as e:
        print(f"[dsmb_lite] exception {type(e).__name__}: {e}", file=sys.stderr)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Rubric
# ─────────────────────────────────────────────────────────────────────────────
def _load_rubric() -> tuple[str, str]:
    """Return (rubric_text, sha256). Cached for the process lifetime."""
    global _RUBRIC_CACHED
    if _RUBRIC_CACHED is not None:
        return _RUBRIC_CACHED
    text = RUBRIC_PATH.read_text(encoding="utf-8")
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    _RUBRIC_CACHED = (text, sha)
    return _RUBRIC_CACHED


# ─────────────────────────────────────────────────────────────────────────────
# Advisor call
# ─────────────────────────────────────────────────────────────────────────────
_ADVISOR_ROLES = {
    "A": "Methodologist (focus: pre-registration discipline, researcher degrees of freedom)",
    "B": "Risk officer (focus: position sizing, tail loss, DSMB sign-off)",
    "C": "Operations specialist (focus: SIV completeness, kill-criterion, failure-mode plans)",
}


def _build_advisor_prompt(advisor_id: str, request: dict[str, Any]) -> str:
    rubric_text, rubric_sha = _load_rubric()
    role = _ADVISOR_ROLES.get(advisor_id, "General adjudicator")
    request_json = json.dumps(request, indent=2, default=str)
    return f"""You are advisor {advisor_id} on a DSMB-Lite independent adjudication panel.
Your role: {role}

Score the promotion request against the 5 rubric dimensions (0-10 each).
LEAN INTO YOUR ROLE — your job is to apply healthy skepticism from your angle.
Other advisors will cover other angles; you do NOT need to be balanced.

==================== FROZEN RUBRIC (DO NOT REINTERPRET) ====================
{rubric_text}
============================================================================

Rubric sha256: {rubric_sha}

==================== PROMOTION REQUEST ====================
{request_json}
===========================================================

Return ONLY a single JSON object on one line (no markdown, no preamble):
{{"score_methodological": <int 0-10>, "score_oos": <int 0-10>, "score_operational": <int 0-10>, "score_risk": <int 0-10>, "score_pre_reg": <int 0-10>, "justification": "<one paragraph, <=200 words, naming the SPECIFIC weakest dimension and why>"}}
"""


_SCORE_KEYS = {
    "methodological": "score_methodological",
    "oos": "score_oos",
    "operational": "score_operational",
    "risk": "score_risk",
    "pre_reg": "score_pre_reg",
}


def _parse_advisor_response(raw: str | None) -> dict[str, Any]:
    """Extract scores + justification from advisor response. Defensive parse."""
    if not raw:
        return {
            "scores": dict.fromkeys(DIMENSIONS, 0),
            "justification": "ERROR: advisor returned no content",
            "parse_ok": False,
        }
    # Find the JSON object in the response (advisors sometimes wrap in ```json)
    m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    obj_str = m.group(0) if m else raw.strip()
    try:
        obj = json.loads(obj_str)
    except json.JSONDecodeError:
        return {
            "scores": dict.fromkeys(DIMENSIONS, 0),
            "justification": f"ERROR: could not parse JSON: {raw[:200]}",
            "parse_ok": False,
        }
    scores: dict[str, int] = {}
    for dim, key in _SCORE_KEYS.items():
        val = obj.get(key, 0)
        try:
            ival = int(val)
        except (TypeError, ValueError):
            ival = 0
        scores[dim] = max(0, min(10, ival))
    return {
        "scores": scores,
        "justification": str(obj.get("justification", "")).strip()[:1000],
        "parse_ok": True,
    }


def _call_advisor(advisor_id: str, request: dict[str, Any]) -> dict[str, Any]:
    """Single advisor call. Returns advisor record dict."""
    prompt = _build_advisor_prompt(advisor_id, request)
    t0 = time.time()
    raw = _deepseek_direct(prompt, timeout_s=ADVISOR_TIMEOUT_S)
    latency_s = round(time.time() - t0, 2)
    parsed = _parse_advisor_response(raw)
    return {
        "advisor_id": advisor_id,
        "role": _ADVISOR_ROLES.get(advisor_id, ""),
        "scores": parsed["scores"],
        "justification": parsed["justification"],
        "parse_ok": parsed["parse_ok"],
        "latency_s": latency_s,
        "raw_excerpt": (raw[:500] if raw else None),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Chairman (deterministic — no LLM)
# ─────────────────────────────────────────────────────────────────────────────
def _chairman_synthesize(panel: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute per-dimension means, total, verdict. Zero LLM discretion."""
    n = max(1, len(panel))
    means: dict[str, float] = {}
    for dim in DIMENSIONS:
        vals = [a["scores"].get(dim, 0) for a in panel]
        means[dim] = round(sum(vals) / n, 2)
    # Integer-floor breakdown for the public output (rubric uses 0-10 integers)
    breakdown: dict[str, int] = {dim: int(round(means[dim])) for dim in DIMENSIONS}
    total_score = int(round(sum(means.values())))

    dim_pass = {dim: means[dim] >= PASS_DIM_THRESHOLD for dim in DIMENSIONS}
    total_pass = total_score >= PASS_TOTAL_THRESHOLD
    overall_pass = total_pass and all(dim_pass.values())

    failing_dims = [d for d, ok in dim_pass.items() if not ok]
    dissent: list[str] = []
    if not total_pass:
        dissent.append(
            f"Total {total_score} < {PASS_TOTAL_THRESHOLD} — below mechanical pass threshold."
        )
    for d in failing_dims:
        dissent.append(f"Dimension '{d}' mean {means[d]} < {PASS_DIM_THRESHOLD} — gate-blocking.")
    # Include each advisor's named-weakest-dimension as their dissent voice
    for a in panel:
        j = a.get("justification", "")
        if j:
            dissent.append(f"[advisor {a['advisor_id']}] {j[:300]}")

    summary_lines = [
        f"Per-dimension means: {means}",
        f"Total: {total_score} / 50  (threshold {PASS_TOTAL_THRESHOLD})",
        f"All dimensions >= {PASS_DIM_THRESHOLD}: {all(dim_pass.values())}",
        f"Failing dimensions: {failing_dims if failing_dims else 'none'}",
        f"Verdict: {'PASS' if overall_pass else 'REJECTED'}",
    ]
    if failing_dims:
        summary_lines.append(
            "Rationale: per the frozen rubric, ANY dimension below 6 or total below 40 "
            "is a mechanical REJECTED. Appeal requires: fix underlying issue, re-submit "
            "as new request id, convene fresh panel."
        )
    chairman_synthesis = "\n".join(summary_lines)

    return {
        "means": means,
        "breakdown": breakdown,
        "total_score": total_score,
        "pass": overall_pass,
        "dim_pass": dim_pass,
        "failing_dims": failing_dims,
        "dissent": dissent,
        "chairman_synthesis": chairman_synthesis,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def convene_panel(
    request_dict: dict[str, Any],
    n_advisors: int = 3,
    request_id: str | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    """Convene n_advisors DeepSeek calls in parallel, score, synthesize, persist.

    Returns the decision JSON (see module docstring).
    """
    if not isinstance(request_dict, dict):
        raise TypeError("request_dict must be a dict")

    request_id = request_id or request_dict.get("request_id") or f"req-{uuid.uuid4().hex[:8]}"
    decision_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    _, rubric_sha = _load_rubric()

    # 3 advisors in parallel (A=methodologist, B=risk, C=ops)
    advisor_ids = ["A", "B", "C"][:n_advisors]
    panel: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_advisors) as ex:
        futures = {ex.submit(_call_advisor, aid, request_dict): aid for aid in advisor_ids}
        for fut in concurrent.futures.as_completed(futures):
            try:
                panel.append(fut.result())
            except Exception as e:  # noqa: BLE001
                aid = futures[fut]
                panel.append({
                    "advisor_id": aid,
                    "role": _ADVISOR_ROLES.get(aid, ""),
                    "scores": dict.fromkeys(DIMENSIONS, 0),
                    "justification": f"ERROR: advisor crashed: {type(e).__name__}: {e}",
                    "parse_ok": False,
                    "latency_s": None,
                    "raw_excerpt": None,
                })
    panel.sort(key=lambda a: a["advisor_id"])

    chair = _chairman_synthesize(panel)

    decision = {
        "request_id": request_id,
        "decision_utc": decision_utc,
        "rubric_version": "1.0",
        "rubric_sha256": rubric_sha,
        "request": request_dict,
        "panel_responses": panel,
        "n_advisors": n_advisors,
        "breakdown": chair["breakdown"],
        "means": chair["means"],
        "total_score": chair["total_score"],
        "dim_pass": chair["dim_pass"],
        "failing_dims": chair["failing_dims"],
        "dissent": chair["dissent"],
        "chairman_synthesis": chair["chairman_synthesis"],
        "pass": chair["pass"],
        "verdict": "PASS" if chair["pass"] else "REJECTED",
        "pass_thresholds": {
            "total_min": PASS_TOTAL_THRESHOLD,
            "per_dimension_min": PASS_DIM_THRESHOLD,
        },
    }

    if persist:
        _persist_decision(decision)
        _write_status(decision)
    return decision


def _persist_decision(decision: dict[str, Any]) -> Path:
    DECISIONS_DIR.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(decision["request_id"]))
    path = DECISIONS_DIR / f"{safe_id}_{decision['decision_utc']}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(decision, indent=2, default=str))
    tmp.replace(path)
    return path


def _write_status(decision: dict[str, Any]) -> None:
    try:
        ZG_TMP.mkdir(parents=True, exist_ok=True)
        lines = [
            "# DSMB-Lite status",
            "",
            f"- Last decision UTC : {decision['decision_utc']}",
            f"- Request id        : {decision['request_id']}",
            f"- Verdict           : {decision['verdict']}",
            f"- Total score       : {decision['total_score']} / 50",
            f"- Per-dim means     : {decision['means']}",
            f"- Failing dims      : {decision['failing_dims']}",
            f"- Rubric sha256     : {decision['rubric_sha256']}",
        ]
        STATUS_PATH.write_text("\n".join(lines) + "\n")
    except OSError as e:
        print(f"[dsmb_lite] status write failed: {e}", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# Tool-context helpers (used by the PreToolUse hook)
# ─────────────────────────────────────────────────────────────────────────────
_PAPER_TRADE_PATTERNS = [
    re.compile(r"paper_trade\.commit", re.I),
    re.compile(r"promote_to_paper", re.I),
    re.compile(r"promote\.to\.paper", re.I),
    re.compile(r"sap[_-]?promote", re.I),
]
_COHORT_PAPER_PATTERN = re.compile(r"['\"]cohort['\"]\s*:\s*['\"]paper['\"]", re.I)


def _extract_request_from_tool_context(ctx: dict[str, Any]) -> dict[str, Any] | None:
    """Pull a promotion request out of a tool-call JSON if it looks like one.

    Hook contract: hook reads PreToolUse JSON from stdin. We look for command
    text and file edits that indicate a paper-trade promotion. Returns request
    dict or None.
    """
    tool_name = ctx.get("tool_name") or ""
    tool_input = ctx.get("tool_input") or {}
    blob = json.dumps(tool_input, default=str)

    is_promo = False
    if tool_name == "Bash":
        cmd = str(tool_input.get("command", ""))
        if any(p.search(cmd) for p in _PAPER_TRADE_PATTERNS):
            is_promo = True
    if tool_name in {"Write", "Edit", "NotebookEdit"}:
        new_blob = str(tool_input.get("new_string", "")) + str(tool_input.get("content", ""))
        if _COHORT_PAPER_PATTERN.search(new_blob):
            is_promo = True
    if not is_promo and any(p.search(blob) for p in _PAPER_TRADE_PATTERNS):
        is_promo = True
    if not is_promo and _COHORT_PAPER_PATTERN.search(blob):
        is_promo = True

    if not is_promo:
        return None
    return {
        "request_id": f"hook-{uuid.uuid4().hex[:8]}",
        "source": "PreToolUse hook",
        "tool_name": tool_name,
        "tool_input_excerpt": blob[:2000],
        "ticker": ctx.get("ticker", "UNKNOWN"),
        "sap_id": ctx.get("sap_id", "UNKNOWN"),
        "target_cohort": "paper",
        "note": "Auto-extracted from tool call; advisors should score CONSERVATIVELY when sap evidence is sparse.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
_SMOKE_APA_REQUEST = {
    "request_id": "SAP-APA-006-synthetic-2026-05-29",
    "ticker": "APA",
    "sap_id": "SAP-APA-006",
    "target_cohort": "paper",
    "perm_survival": 0.046,
    "holdout_sharpe": 3.40,
    "siv_status": "5/6 (DSMB present after this commit; blinding STILL MISSING)",
    "pre_registration_hash": "sha256:placeholder_apa_006_pre_reg",
    "kill_criterion_signed_off": True,
    "dsmb_lineage": "none — first request after charter lock-in",
    "operational_notes": (
        "Paper-trade integration paths NOT smoke-tested. Failure-mode plan exists "
        "but has not been exercised. Position-size cap implemented at 1% NAV."
    ),
    "risk_notes": (
        "Holdout max DD 18% (above 15% target). No DSMB sign-off on risk envelope yet."
    ),
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--request-json", type=str, default=None,
                    help="Path to a JSON file containing the promotion request.")
    ap.add_argument("--request-id", type=str, default=None,
                    help="Override request id (otherwise derived from request or auto).")
    ap.add_argument("--tool-context-stdin", action="store_true",
                    help="Read PreToolUse JSON from stdin; gate on paper-trade patterns.")
    ap.add_argument("--smoke-apa", action="store_true",
                    help="Run synthetic APA promotion smoke test.")
    ap.add_argument("--n-advisors", type=int, default=3)
    ap.add_argument("--no-persist", action="store_true",
                    help="Skip writing decision JSON to disk (testing only).")
    args = ap.parse_args(argv)

    request: dict[str, Any] | None = None
    if args.smoke_apa:
        request = dict(_SMOKE_APA_REQUEST)
    elif args.request_json:
        request = json.loads(Path(args.request_json).read_text(encoding="utf-8"))
    elif args.tool_context_stdin:
        try:
            ctx = json.loads(sys.stdin.read() or "{}")
        except json.JSONDecodeError:
            print("[dsmb_lite] stdin is not valid JSON; allowing tool call (no-op)", file=sys.stderr)
            return 0
        request = _extract_request_from_tool_context(ctx)
        if request is None:
            # Not a paper-trade promotion → don't block.
            return 0
    else:
        ap.print_help()
        return 2

    decision = convene_panel(
        request,
        n_advisors=args.n_advisors,
        request_id=args.request_id,
        persist=not args.no_persist,
    )
    print(json.dumps(decision, indent=2, default=str))
    return 0 if decision["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
