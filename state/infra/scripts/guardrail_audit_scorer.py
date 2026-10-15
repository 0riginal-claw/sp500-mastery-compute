#!/usr/bin/env python3
"""Score every workspace feature on the 10-point guardrail-grade checklist.

Output: markdown table to stdout + JSON to docs/guardrail_audit.json.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

ROOT = Path("/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools")

plists = [Path(p).stem.replace("com.zg.", "") for p in glob.glob(str(ROOT / "home/Library/LaunchAgents/com.zg.*.plist"))]
state_dirs = [p.name for p in (ROOT / "state").iterdir() if p.is_dir()]
hook_dirs = [p.name for p in (ROOT / "home/.claude/hooks").iterdir() if p.is_dir() and not p.name.startswith("_")]

hook_features: set[str] = set()
SUFFIXES = [
    "-bootstrap",
    "-freshness",
    "-heartbeat",
    "-session-activity",
    "-subagent-inject",
    "-stop-validate",
    "-context-inject",
    "-validate",
    "-observation",
    "-direction-validate",
    "-action-guard",
    "-activity",
    "-dialog-detect",
]
for h in hook_dirs:
    for suffix in SUFFIXES:
        if h.endswith(suffix):
            base = h[: -len(suffix)].replace("-", "_")
            hook_features.add(base)
            break

features = set(plists) | set(state_dirs) | hook_features

SKIP = {
    "user_prompts_history.jsonl",
    "patterns",
    "scan_commit",
    "scan_secrets",
    "auto_sandbox",
    "sensitive_path_block",
    "prompt_injection_defender",
    "model_reason_block",
    "model_routing_check",
    "openclaw_routing_block",
    "cloud_routing_block",
    "karpathy_guidelines_block",
    "auto_spawn_compress",
    "spawn_prompt_compress",
    "spawn_validator",
    "prompt_to_inbox",
    "subagent_rules_inject",
    "recursion_fanout_tracker",
    "touch_last_prompt",
    "auto_solve_violation_detector",
}
features = {f for f in features if f not in SKIP}


def score(f: str) -> tuple[int, dict[str, str]]:
    fdash = f.replace("_", "-")
    s = 0
    parts: dict[str, str] = {}

    plist_path = ROOT / f"home/Library/LaunchAgents/com.zg.{f}.plist"
    if plist_path.exists():
        s += 1
        parts["plist"] = "Y"
    else:
        parts["plist"] = "-"

    state_path = ROOT / "state" / f
    hb_path = state_path / "heartbeat.json"
    if hb_path.exists() or state_path.exists():
        s += 1
        parts["state"] = "Y"
    else:
        parts["state"] = "-"

    fresh_candidates = [
        f"{fdash}-freshness",
        f"{fdash}-heartbeat",
        f"{fdash}-daemon-heartbeat",
    ]
    if any((ROOT / f"home/.claude/hooks/{c}").exists() for c in fresh_candidates):
        s += 1
        parts["fresh_hook"] = "Y"
    else:
        parts["fresh_hook"] = "-"

    boot_candidates = [f"{fdash}-bootstrap", f"{fdash}-daemon-bootstrap"]
    if any((ROOT / f"home/.claude/hooks/{c}").exists() for c in boot_candidates):
        s += 1
        parts["boot_hook"] = "Y"
    else:
        parts["boot_hook"] = "-"

    act_candidates = [f"{fdash}-activity", f"{fdash}-session-activity"]
    if any((ROOT / f"home/.claude/hooks/{c}").exists() for c in act_candidates):
        s += 1
        parts["act_hook"] = "Y"
    else:
        parts["act_hook"] = "-"

    sa_candidates = [f"{fdash}-subagent-inject", f"{fdash}-context-inject"]
    if any((ROOT / f"home/.claude/hooks/{c}").exists() for c in sa_candidates):
        s += 1
        parts["sa_hook"] = "Y"
    else:
        parts["sa_hook"] = "-"

    stop_candidates = [
        f"{fdash}-stop-validate",
        f"{fdash}-validate",
        f"{fdash}-action-guard",
    ]
    if any((ROOT / f"home/.claude/hooks/{c}").exists() for c in stop_candidates):
        s += 1
        parts["stop_hook"] = "Y"
    else:
        parts["stop_hook"] = "-"

    if any(v == "Y" for k, v in parts.items() if "hook" in k):
        s += 1
        parts["settings"] = "Y"
    else:
        parts["settings"] = "-"

    doc_candidates = [
        f"docs/{f.upper()}.md",
        f"docs/{f}.md",
        f"docs/{fdash.upper()}.md",
        f"docs/{f.replace('_', '-').upper()}.md",
    ]
    doc_present = any((ROOT / d).exists() for d in doc_candidates)
    if doc_present:
        s += 1
        parts["doc"] = "Y"
    else:
        parts["doc"] = "-"

    backups = list((ROOT / "backups").glob(f"*{f}*")) + list((ROOT / "backups").glob(f"*{fdash}*"))
    if backups:
        s += 1
        parts["backup"] = "Y"
    else:
        parts["backup"] = "-"

    return s, parts


rows = []
for f in sorted(features):
    sc, p = score(f)
    rows.append((f, sc, p))
rows.sort(key=lambda r: (r[1], r[0]))

print("Feature | Score | plist | state | fresh | boot | act | s/a | stop | settings | doc | backup")
print("--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---")
for f, sc, p in rows:
    print(
        f"{f} | {sc}/10 | {p['plist']} | {p['state']} | {p['fresh_hook']} | "
        f"{p['boot_hook']} | {p['act_hook']} | {p['sa_hook']} | {p['stop_hook']} | "
        f"{p['settings']} | {p['doc']} | {p['backup']}"
    )

out = {f: {"score": s, "parts": p} for f, s, p in rows}
(ROOT / "docs/guardrail_audit.json").write_text(json.dumps(out, indent=2))
print()
print(f"# Total features: {len(rows)}")
print(f"# Fully guardrail-grade (10/10): {sum(1 for _, s, _ in rows if s == 10)}")
print(f"# Mostly guardrail-grade (>=7/10): {sum(1 for _, s, _ in rows if s >= 7)}")
print(f"# Brittle (<5/10): {sum(1 for _, s, _ in rows if s < 5)}")
