"""
wire_candidate.py — Shared schema + emitter for WIRE_CANDIDATE markers.

Producers (feature_discovery_daemon, proactive_loop_daemon, ceo_orchestrator_daemon)
emit STRUCTURED WIRE_CANDIDATE blocks when they identify a useful new feature.
A separate consumer daemon parses these blocks and auto-wires features into the
backtest pipeline.

Two output forms (both kept in sync):
  1. Markdown block — appended to feature_discovery/reports/wire_candidates_<DATE>.md
  2. JSONL event   — appended to proactive/stream.jsonl as one JSON line

Schema (all keys required, missing keys filled with safe defaults):
  feature_name             snake_case unique id                       (str)
  description              one-line human summary                      (str)
  data_source              github URL | drive path | api endpoint     (str)
  data_source_license      MIT|Apache|BSD|GPL|UNKNOWN                  (str)
  function_signature       `def add_<name>_features(df, ticker) -> df`(str)
  features_added           how many columns the function adds          (int)
  shift_1_safe             yes|no|unclear (lookahead guard)            (str)
  integration_cost         LOW|MED|HIGH                                (str)
  requires_paid_api        yes|no                                      (str)
  requires_human_review    yes|no                                      (str)
  expected_lift_pct        float or 'unknown'                          (str|float)
  citations                list of URLs                                (list[str])
  discovered_by            feature_discovery|proactive_loop|ceo_orchestrator
  discovered_at            ISO-8601 UTC timestamp                      (str)

Designed to be backward compatible — old freeform reports still work; this is
ADDITIVE.

The DeepSeek prompt suffix `WIRE_CANDIDATE_PROMPT_SUFFIX` instructs the model to
emit blocks directly in machine-parseable form; producers paste it onto their
existing prompts.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Canonical output paths (the daemons already use these — keep in sync)
# ---------------------------------------------------------------------------

WORK = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools/s&p500-ticker-mastery"
)
PROACTIVE_DIR = WORK / "proactive"
STREAM_JSONL = PROACTIVE_DIR / "stream.jsonl"
WIRE_REPORTS_DIR = WORK / "feature_discovery" / "reports"

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

REQUIRED_KEYS: tuple[str, ...] = (
    "feature_name",
    "description",
    "data_source",
    "data_source_license",
    "function_signature",
    "features_added",
    "shift_1_safe",
    "integration_cost",
    "requires_paid_api",
    "requires_human_review",
    "expected_lift_pct",
    "citations",
    "discovered_by",
    "discovered_at",
)

VALID_LICENSES = {"MIT", "Apache", "BSD", "GPL", "UNKNOWN"}
VALID_TRISTATE = {"yes", "no", "unclear"}
VALID_BISTATE = {"yes", "no"}
VALID_COST = {"LOW", "MED", "HIGH"}
VALID_DISCOVERERS = {"feature_discovery", "proactive_loop", "ceo_orchestrator"}

_SNAKE_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _coerce(d: dict[str, Any], discovered_by: str) -> dict[str, Any]:
    """Fill missing keys with safe defaults; normalize a few common variants."""
    out = dict(d)
    out.setdefault("feature_name", "unknown_feature")
    out.setdefault("description", "")
    out.setdefault("data_source", "")
    lic = str(out.get("data_source_license", "UNKNOWN")).strip()
    if lic.upper() in {x.upper() for x in VALID_LICENSES}:
        out["data_source_license"] = lic if lic in VALID_LICENSES else lic.upper()
    else:
        out["data_source_license"] = "UNKNOWN"
    name = str(out["feature_name"])
    if not _SNAKE_RE.match(name):
        name = re.sub(r"[^a-z0-9_]", "_", name.lower()).strip("_") or "unknown_feature"
        out["feature_name"] = name
    out.setdefault("function_signature",
                   f"def add_{out['feature_name']}_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:")
    try:
        out["features_added"] = int(out.get("features_added", 1))
    except Exception:
        out["features_added"] = 1
    shift = str(out.get("shift_1_safe", "unclear")).lower()
    out["shift_1_safe"] = shift if shift in VALID_TRISTATE else "unclear"
    cost = str(out.get("integration_cost", "MED")).upper()
    out["integration_cost"] = cost if cost in VALID_COST else "MED"
    paid = str(out.get("requires_paid_api", "no")).lower()
    out["requires_paid_api"] = paid if paid in VALID_BISTATE else "no"
    review = str(out.get("requires_human_review", "yes")).lower()
    out["requires_human_review"] = review if review in VALID_BISTATE else "yes"
    lift = out.get("expected_lift_pct", "unknown")
    if isinstance(lift, (int, float)):
        out["expected_lift_pct"] = float(lift)
    else:
        try:
            out["expected_lift_pct"] = float(str(lift).strip().rstrip("%"))
        except Exception:
            out["expected_lift_pct"] = "unknown"
    cites = out.get("citations", [])
    if isinstance(cites, str):
        cites = [c.strip() for c in re.split(r"[,\s]+", cites) if c.strip()]
    if not isinstance(cites, list):
        cites = []
    out["citations"] = [str(c) for c in cites][:10]
    out["discovered_by"] = discovered_by if discovered_by in VALID_DISCOVERERS else "feature_discovery"
    out.setdefault("discovered_at", datetime.now(timezone.utc).isoformat())
    return out


def render_markdown_block(cand: dict[str, Any]) -> str:
    """Format a single WIRE_CANDIDATE as a markdown block matching the schema."""
    c = cand
    cites = ", ".join(c.get("citations", []) or []) or "[]"
    return (
        f"## WIRE_CANDIDATE: {c['feature_name']}\n\n"
        f"- feature_name: {c['feature_name']}\n"
        f"- description: {c['description']}\n"
        f"- data_source: {c['data_source']}\n"
        f"- data_source_license: {c['data_source_license']}\n"
        f"- function_signature: `{c['function_signature']}`\n"
        f"- features_added: {c['features_added']}\n"
        f"- shift_1_safe: {c['shift_1_safe']}\n"
        f"- integration_cost: {c['integration_cost']}\n"
        f"- requires_paid_api: {c['requires_paid_api']}\n"
        f"- requires_human_review: {c['requires_human_review']}\n"
        f"- expected_lift_pct: {c['expected_lift_pct']}\n"
        f"- citations: [{cites}]\n"
        f"- discovered_by: {c['discovered_by']}\n"
        f"- discovered_at: {c['discovered_at']}\n"
    )


def parse_markdown_blocks(text: str, discovered_by: str = "feature_discovery") -> list[dict[str, Any]]:
    """Extract WIRE_CANDIDATE blocks from free-form text (typically DeepSeek output).

    Robust to formatting variation: accepts `## WIRE_CANDIDATE: <name>` headers
    followed by `- key: value` lines, until the next `## ` header or EOF.
    """
    out: list[dict[str, Any]] = []
    pattern = re.compile(r"##\s*WIRE_CANDIDATE\s*:\s*([^\n]+)\n(.*?)(?=\n##\s|\Z)", re.DOTALL | re.IGNORECASE)
    for m in pattern.finditer(text):
        name = m.group(1).strip()
        body = m.group(2)
        d: dict[str, Any] = {"feature_name": name}
        for line in body.splitlines():
            line = line.strip()
            if not line.startswith("-"):
                continue
            kv = line.lstrip("-").strip()
            if ":" not in kv:
                continue
            k, _, v = kv.partition(":")
            k = k.strip().lower().replace(" ", "_")
            v = v.strip().strip("`")
            if k == "citations":
                v = [c.strip() for c in re.split(r"[,\s]+", v.strip("[]")) if c.strip()]
            d[k] = v
        out.append(_coerce(d, discovered_by))
    return out


def emit(
    candidates: Iterable[dict[str, Any]],
    *,
    discovered_by: str,
    write_md: bool = True,
    write_jsonl: bool = True,
    md_path: Path | None = None,
    jsonl_path: Path | None = None,
) -> dict[str, Any]:
    """Emit one or more WIRE_CANDIDATE markers to disk.

    Returns a summary dict: {"emitted": n, "md_path": str|None, "jsonl_path": str|None}.

    Idempotent on append-only files; safe to call repeatedly.
    """
    cands = [_coerce(dict(c), discovered_by) for c in candidates]
    if not cands:
        return {"emitted": 0, "md_path": None, "jsonl_path": None}

    md_written: str | None = None
    jsonl_written: str | None = None

    if write_md:
        if md_path is None:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            md_path = WIRE_REPORTS_DIR / f"wire_candidates_{date_str}.md"
        md_path.parent.mkdir(parents=True, exist_ok=True)
        header_needed = not md_path.exists() or md_path.stat().st_size == 0
        with open(md_path, "a", encoding="utf-8") as fh:
            if header_needed:
                fh.write(f"# Wire Candidates — {datetime.now(timezone.utc).strftime('%Y-%m-%d')}\n\n")
                fh.write(
                    "_Auto-emitted by producer daemons. Each `## WIRE_CANDIDATE:` block "
                    "is machine-parseable by the wire-consumer daemon._\n\n"
                )
            for c in cands:
                fh.write(render_markdown_block(c))
                fh.write("\n")
        md_written = str(md_path)

    if write_jsonl:
        if jsonl_path is None:
            jsonl_path = STREAM_JSONL
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(jsonl_path, "a", encoding="utf-8") as fh:
            for c in cands:
                event = {"event": "wire_candidate", **c}
                fh.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        jsonl_written = str(jsonl_path)

    return {"emitted": len(cands), "md_path": md_written, "jsonl_path": jsonl_written}


# ---------------------------------------------------------------------------
# Prompt suffix — producers paste this onto their DeepSeek prompts so the model
# returns directly machine-parseable WIRE_CANDIDATE blocks (no JSON repair).
# ---------------------------------------------------------------------------

WIRE_CANDIDATE_PROMPT_SUFFIX = """

OUTPUT FORMAT — RESPOND ONLY WITH WIRE_CANDIDATE BLOCKS (one per feature):

## WIRE_CANDIDATE: <snake_case_unique_name>

- feature_name: <snake_case>
- description: <1-line>
- data_source: <github_url OR drive_path OR api_endpoint>
- data_source_license: <MIT|Apache|BSD|GPL|UNKNOWN>
- function_signature: `def add_<name>_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:`
- features_added: <int>
- shift_1_safe: <yes|no|unclear>
- integration_cost: <LOW|MED|HIGH>
- requires_paid_api: <yes|no>
- requires_human_review: <yes|no>
- expected_lift_pct: <float or 'unknown'>
- citations: [<url1>, <url2>]
- discovered_by: <feature_discovery|proactive_loop|ceo_orchestrator>
- discovered_at: <ISO-8601 UTC>

Emit at least 1 and at most 5 blocks. No prose, no preamble, no JSON envelope —
just the blocks back-to-back. Use 'UNKNOWN' for license if unsure. Use 'unclear'
for shift_1_safe if the source repo doesn't make it obvious. Always include the
discovered_at and discovered_by fields.
"""


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = """
Some preamble that should be ignored.

## WIRE_CANDIDATE: foo_bar_features

- feature_name: foo_bar_features
- description: A test feature for self-testing the parser.
- data_source: https://github.com/example/foo
- data_source_license: MIT
- function_signature: `def add_foo_bar_features(df, ticker) -> df:`
- features_added: 3
- shift_1_safe: yes
- integration_cost: LOW
- requires_paid_api: no
- requires_human_review: no
- expected_lift_pct: 1.2
- citations: [https://example.com/paper.pdf]
- discovered_by: feature_discovery
- discovered_at: 2026-05-17T00:00:00+00:00
"""
    parsed = parse_markdown_blocks(sample, discovered_by="feature_discovery")
    print(f"Parsed {len(parsed)} candidates")
    for c in parsed:
        print(json.dumps(c, indent=2, default=str))
    # Round-trip
    print("\n--- Round-trip markdown ---")
    print(render_markdown_block(parsed[0]))
