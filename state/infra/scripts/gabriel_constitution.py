#!/usr/bin/env python3
"""gabriel_constitution.py - Constitutional self-critique for Gabriel.

10 concrete principles. critique(brief, context) returns:
    {
        "ok": bool,            # False if any principle violated severely
        "violations": [...],   # list of {principle, severity, evidence, suggestion}
        "score": float,        # 1.0 - (severe_violations / 10)
        "refinements": [...],  # concrete tweaks to apply before respawn
        "verdict": "approve" | "refine" | "reject",
    }

Designed to be cheap (pure-Python regex/keyword) so it can fire on EVERY spawn
without latency. No LLM call. Inspired by Anthropic Constitutional AI: critique
first, refine, then act.

USAGE FROM DAEMON:
    from scripts.gabriel_constitution import critique
    result = critique(brief_text, context={
        "recent_spawn_titles": [...],
        "elapsed_min": 7.2,
        "task_keywords": ["backtest", "AAPL"],
        "is_destructive": False,
        "claims_success": True,
    })
    if result["verdict"] == "reject":
        # do not spawn
    elif result["verdict"] == "refine":
        # apply result["refinements"] before spawn

Author: gabriel_self bootstrap (2026-05-20)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools"
)
GABRIEL_SELF_DIR = ROOT / "state" / "gabriel_self"


@dataclass
class Principle:
    id: str
    name: str
    description: str
    severity_default: str  # "severe" | "moderate" | "advisory"

    def check(self, brief: str, ctx: dict[str, Any]) -> tuple[bool, str | None, str | None]:
        """Returns (passed, evidence_if_violated, suggestion). Override in subclass."""
        raise NotImplementedError


# Principles


class P01NoRepeatFailedApproach(Principle):
    def check(self, brief: str, ctx: dict[str, Any]) -> tuple[bool, str | None, str | None]:
        title = (brief.split("\n", 1)[0] or "").lower()
        recent_losses = [t.lower() for t in ctx.get("recent_losses", [])]
        for loser in recent_losses:
            new_tokens = set(re.findall(r"[a-z][a-z0-9_]{2,}", title))
            old_tokens = set(re.findall(r"[a-z][a-z0-9_]{2,}", loser))
            overlap = new_tokens & old_tokens
            if len(overlap) >= 4:
                return (False,
                        f"shares {len(overlap)} tokens with recent loss '{loser[:60]}'",
                        "change at least one independent variable (data window, model, threshold, ticker)")
        return (True, None, None)


class P02FanOutOver5Min(Principle):
    def check(self, brief: str, ctx: dict[str, Any]) -> tuple[bool, str | None, str | None]:
        est_min = ctx.get("estimated_min", 0) or 0
        has_fanout = any(tag in brief for tag in ("# decomposition_plan:", "# fan_out:", "# inline_justification:"))
        if est_min and est_min > 5 and not has_fanout:
            return (False,
                    f"estimated_min={est_min} >5 but brief lacks decomposition_plan/inline_justification",
                    "add `# decomposition_plan: <N grandchildren slices>` OR `# inline_justification: <why solo>` near top")
        return (True, None, None)


class P03NoSimilarToRecent(Principle):
    def check(self, brief: str, ctx: dict[str, Any]) -> tuple[bool, str | None, str | None]:
        title = (brief.split("\n", 1)[0] or "").lower()
        recent = [t.lower() for t in ctx.get("recent_spawn_titles", [])][-10:]
        new_tokens = set(re.findall(r"[a-z][a-z0-9_]{2,}", title))
        for t in recent:
            old_tokens = set(re.findall(r"[a-z][a-z0-9_]{2,}", t))
            if not old_tokens or not new_tokens:
                continue
            j = len(new_tokens & old_tokens) / max(1, len(new_tokens | old_tokens))
            if j > 0.6:
                return (False,
                        f"Jaccard={j:.2f} with recent '{t[:60]}'",
                        "rephrase with different angle OR target different artifact")
        return (True, None, None)


class P04NoPuntToUser(Principle):
    PUNT_PHRASES = (
        "ask user", "should i", "would you like", "let me know if",
        "if you want me to", "say go to", "please confirm",
    )

    def check(self, brief: str, ctx: dict[str, Any]) -> tuple[bool, str | None, str | None]:
        low = brief.lower()
        for p in self.PUNT_PHRASES:
            if p in low:
                return (False, f"contains punt-phrase '{p}'",
                        "rewrite as direct action - daemon has standing authorization")
        return (True, None, None)


class P05CloudRouteHeavyCompute(Principle):
    HEAVY_KEYWORDS = ("backtest", "sweep", "monte carlo", "ml train", "hyperparam",
                      "batch", "100 ticker", "all ticker", "vwap_batch", "orb_batch")

    def check(self, brief: str, ctx: dict[str, Any]) -> tuple[bool, str | None, str | None]:
        low = brief.lower()
        is_heavy = any(k in low for k in self.HEAVY_KEYWORDS)
        if not is_heavy:
            return (True, None, None)
        has_cloud = ("cloud_dispatch" in low or "modal" in low or
                     "gh_actions" in low or "enqueue_job" in low)
        if not has_cloud:
            return (False,
                    "heavy-compute keywords present but no cloud_dispatch/Modal mention",
                    "add `cloud_dispatch.enqueue_job(...)` OR explain why local <60s smoke")
        return (True, None, None)


class P06BackupBeforeDestructive(Principle):
    DESTRUCTIVE = ("rm -rf", "rm  -rf", "git push --force", "drop table",
                   "truncate", "delete from", "shutil.rmtree", "force-push")

    def check(self, brief: str, ctx: dict[str, Any]) -> tuple[bool, str | None, str | None]:
        low = brief.lower()
        for d in self.DESTRUCTIVE:
            if d in low:
                if "backup" not in low and "timestamp" not in low:
                    return (False, f"destructive op '{d}' without 'backup'/'timestamp' nearby",
                            "add explicit 'backup to backups/<ts>/ first' step")
        return (True, None, None)


class P07SafetyBoundaries(Principle):
    HARD_FORBIDS = ("wire money", "send sms", "post tweet", "email send",
                    "wallet seed", "private key", "credit card", "transfer funds")

    def check(self, brief: str, ctx: dict[str, Any]) -> tuple[bool, str | None, str | None]:
        low = brief.lower()
        for f in self.HARD_FORBIDS:
            if f in low:
                return (False, f"hard-forbidden phrase '{f}'",
                        "REJECT - safety boundary cannot be bypassed by daemon")
        return (True, None, None)


class P08IterateOnLanded(Principle):
    def check(self, brief: str, ctx: dict[str, Any]) -> tuple[bool, str | None, str | None]:
        recent_wins = ctx.get("recent_wins", [])
        if not recent_wins:
            return (True, None, None)
        title = (brief.split("\n", 1)[0] or "").lower()
        new_tokens = set(re.findall(r"[a-z][a-z0-9_]{2,}", title))
        if not new_tokens:
            return (True, None, None)
        for w in recent_wins:
            old_tokens = set(re.findall(r"[a-z][a-z0-9_]{2,}", w.lower()))
            if new_tokens & old_tokens:
                return (True, None, None)
        return (True,
                f"no overlap with recent_wins={recent_wins[:3]} - drifting away from landed work",
                "consider extending one of recent_wins instead of greenfield")


class P09UseCheapestModel(Principle):
    OPUS_KEYWORDS = ("architecture", "design", "synthesize final", "trading risk", "strategy design")
    MECHANICAL_KEYWORDS = ("grep", "list files", "format json", "rename", "inventory", "scan logs")

    def check(self, brief: str, ctx: dict[str, Any]) -> tuple[bool, str | None, str | None]:
        low = brief.lower()
        model = (ctx.get("model") or "").lower()
        if model == "opus":
            if any(k in low for k in self.MECHANICAL_KEYWORDS) and not any(k in low for k in self.OPUS_KEYWORDS):
                return (False,
                        "opus selected for mechanical task (keywords: mechanical)",
                        "downgrade to haiku - mechanical work doesn't need opus")
        if model == "haiku":
            if any(k in low for k in self.OPUS_KEYWORDS):
                return (False,
                        "haiku selected for high-reasoning task (keywords: opus-class)",
                        "upgrade to sonnet or opus - task needs deeper reasoning")
        return (True, None, None)


class P10VerifyResultJson(Principle):
    def check(self, brief: str, ctx: dict[str, Any]) -> tuple[bool, str | None, str | None]:
        low = brief.lower()
        claims_success = ctx.get("claims_success", False)
        if claims_success and "result.json" not in low and "verify" not in low and "smoke" not in low:
            return (False,
                    "claims_success=True but no verification step (result.json / smoke / verify)",
                    "add explicit verification step before claiming done")
        return (True, None, None)


PRINCIPLES: list[Principle] = [
    P01NoRepeatFailedApproach(
        "P01", "no-repeat-failed-approach",
        "Don't repeat failed approaches without changing variables", "severe"),
    P02FanOutOver5Min(
        "P02", "fan-out-over-5min",
        "Don't spawn helpers that take >5min w/o fan-out", "severe"),
    P03NoSimilarToRecent(
        "P03", "no-similar-to-recent",
        "Don't propose ideas similar to last 10", "moderate"),
    P04NoPuntToUser(
        "P04", "no-punt-to-user",
        "Don't punt to user", "severe"),
    P05CloudRouteHeavyCompute(
        "P05", "cloud-route-heavy-compute",
        "Cloud-route heavy compute", "moderate"),
    P06BackupBeforeDestructive(
        "P06", "backup-before-destructive",
        "Backup before destructive", "severe"),
    P07SafetyBoundaries(
        "P07", "maintain-safety-boundaries",
        "Maintain safety boundaries (no money/messages/credentials)", "severe"),
    P08IterateOnLanded(
        "P08", "iterate-on-landed",
        "Iterate on what just landed, don't drop it", "advisory"),
    P09UseCheapestModel(
        "P09", "use-cheapest-model",
        "Use cheapest model for task", "moderate"),
    P10VerifyResultJson(
        "P10", "verify-before-claim",
        "Verify result.json before claiming success", "moderate"),
]


# Public API


def critique(brief: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Score a spawn brief against all 10 principles."""
    ctx = context or {}
    violations: list[dict[str, Any]] = []
    refinements: list[str] = []
    severe_count = 0
    moderate_count = 0

    for p in PRINCIPLES:
        try:
            passed, evidence, suggestion = p.check(brief, ctx)
        except Exception as e:  # noqa: BLE001
            violations.append({
                "principle": p.id, "name": p.name, "severity": "advisory",
                "evidence": f"principle-check-exception: {e}",
                "suggestion": "(principle check raised - review impl)",
            })
            continue
        if passed:
            continue
        violations.append({
            "principle": p.id, "name": p.name,
            "severity": p.severity_default,
            "evidence": evidence,
            "suggestion": suggestion,
        })
        if suggestion:
            refinements.append(f"[{p.id}] {suggestion}")
        if p.severity_default == "severe":
            severe_count += 1
        elif p.severity_default == "moderate":
            moderate_count += 1

    score = max(0.0, 1.0 - (severe_count + 0.5 * moderate_count) / 10.0)
    if severe_count >= 1:
        verdict = "reject"
    elif moderate_count >= 2:
        verdict = "refine"
    elif moderate_count == 1 or any(v.get("severity") == "advisory" for v in violations):
        verdict = "refine"
    else:
        verdict = "approve"

    return {
        "ok": severe_count == 0,
        "violations": violations,
        "score": round(score, 3),
        "refinements": refinements,
        "verdict": verdict,
        "severe_count": severe_count,
        "moderate_count": moderate_count,
    }


def log_critique(brief_path: str | Path, result: dict[str, Any]) -> None:
    """Append the critique result to state/gabriel_self/critiques.jsonl."""
    out = GABRIEL_SELF_DIR / "critiques.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    import time as _t
    rec = {
        "ts": _t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime()),
        "brief_path": str(brief_path),
        "verdict": result.get("verdict"),
        "score": result.get("score"),
        "severe_count": result.get("severe_count"),
        "moderate_count": result.get("moderate_count"),
        "violations": result.get("violations", []),
    }
    with out.open("a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    import sys
    test_briefs = [
        ("APPROVE",
         "audit_seen_ideas_diversity\n# decomposition_plan: solo, single-file slice\n"
         "# scope_estimate_min: 3\nRead seen_ideas.jsonl. Compute Jaccard. Save report.",
         {"recent_spawn_titles": [], "estimated_min": 3, "model": "haiku", "claims_success": False}),
        ("REFINE",
         "scan_paper_trade_drift_24h\nGrep logs. List drifting strategies.",
         {"recent_spawn_titles": [], "estimated_min": 8, "model": "opus", "claims_success": False}),
        ("REJECT",
         "wire money to alpaca account for live trade boost\n# scope: 10min, no decomposition",
         {"recent_spawn_titles": [], "estimated_min": 10, "model": "sonnet"}),
    ]
    for label, brief, ctx in test_briefs:
        r = critique(brief, ctx)
        print(f"--- {label} ---")
        print(f"  verdict={r['verdict']} score={r['score']} severe={r['severe_count']} mod={r['moderate_count']}")
        for v in r["violations"]:
            print(f"  - {v['principle']} ({v['severity']}): {v['evidence']}")
        print()
    sys.exit(0)
