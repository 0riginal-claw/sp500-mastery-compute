#!/usr/bin/env python3
"""gabriel_self.py - Self-awareness layer for the autonomous_mode_daemon.

Modules:
    1. update_capability_map() - reads audit log, classifies spawns into
       strengths/weaknesses/recent_wins/recent_losses by task type.
    2. reflect() - every N cycles, runs DeepSeek call "given these 10 actions
       + outcomes, what 1 lesson would I tell my future self?" Appends to
       reflexions.jsonl.
    3. refresh_self_status() - regenerates dashboard/GABRIEL_SELF_STATUS.md.
    4. get_capability_summary() - cheap dict for spawn-routing decisions.
    5. classify_task_type() - heuristic taxonomy.

State files:
    state/gabriel_self/capability_map.json
    state/gabriel_self/reflexions.jsonl
    state/gabriel_self/lessons.md
    state/gabriel_self/outcomes.jsonl

Author: gabriel_self bootstrap (2026-05-20)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools"
)
GABRIEL_SELF_DIR = ROOT / "state" / "gabriel_self"
CAPABILITY_MAP = GABRIEL_SELF_DIR / "capability_map.json"
REFLEXIONS = GABRIEL_SELF_DIR / "reflexions.jsonl"
LESSONS_MD = GABRIEL_SELF_DIR / "lessons.md"
OUTCOMES = GABRIEL_SELF_DIR / "outcomes.jsonl"

STATE_AUTONOMOUS = ROOT / "state" / "autonomous_mode"
DASHBOARD_SELF_STATUS = ROOT / "dashboard" / "GABRIEL_SELF_STATUS.md"
OPENCLAW_BIN = ROOT / "bin" / "openclaw-gdrive"

STRENGTH_MIN_SUCCESS_RATE = 0.7
STRENGTH_MIN_ATTEMPTS = 3
WEAKNESS_MAX_SUCCESS_RATE = 0.3
RECENT_LIMIT = 5
REFLEXION_WINDOW = 10
DEEPSEEK_TIMEOUT_S = 90

TASK_TYPE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("backtest_sweep", re.compile(r"\b(backtest|sweep|monte carlo|hyperparam|optimi[sz]e)\b", re.I)),
    ("ticker_scan", re.compile(r"\bticker\b.*\b(scan|inspect|audit|check)\b", re.I)),
    ("audit_diagnostic", re.compile(r"\b(audit|diagnose|inspect|health|status|drift)\b", re.I)),
    ("data_integration", re.compile(r"\b(integrate|source|fetch|ingest|crawl|new datasource)\b", re.I)),
    ("refactor_code", re.compile(r"\b(refactor|cleanup|tidy|reorganize|simplify)\b", re.I)),
    ("self_improvement", re.compile(r"\b(self.improvement|improve|tune|orthogonality|reflexion)\b", re.I)),
    ("infra_ops", re.compile(r"\b(daemon|launchctl|cron|launchd|systemd|drive sync|modal|gh.actions)\b", re.I)),
    ("alpaca_trading", re.compile(r"\b(alpaca|paper.trade|live.trade|order|portfolio)\b", re.I)),
    ("report_synthesis", re.compile(r"\b(report|summari[sz]e|aggregate|rollup|digest)\b", re.I)),
    ("file_mechanical", re.compile(r"\b(grep|list files|format json|rename|inventory|scan logs)\b", re.I)),
    ("feature_discovery", re.compile(r"\b(feature.discovery|new feature|capability|untried)\b", re.I)),
]
DEFAULT_TASK_TYPE = "uncategorized"


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def classify_task_type(text: str) -> str:
    if not text:
        return DEFAULT_TASK_TYPE
    for name, pat in TASK_TYPE_PATTERNS:
        if pat.search(text):
            return name
    return DEFAULT_TASK_TYPE


def _read_audit_lines(days: int = 1) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    audit_dir = STATE_AUTONOMOUS
    files = sorted(audit_dir.glob("audit_*.jsonl"))[-days:]
    for fp in files:
        try:
            for line in fp.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except OSError:
            continue
    return out


def _read_openclaw_completions() -> list[dict[str, Any]]:
    home = Path(os.environ.get("HOME", os.path.expanduser("~")))
    completions = home / ".claude" / "state" / "openclaw_completions.jsonl"
    out: list[dict[str, Any]] = []
    if not completions.exists():
        return out
    try:
        for line in completions.read_text().splitlines()[-500:]:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError:
        pass
    return out


def _classify_outcome(audit_record: dict[str, Any]) -> str | None:
    ev = audit_record.get("event", "")
    if ev == "spawn_launched":
        return "pending"
    if ev == "spawn_failed":
        return "loss"
    if ev == "gate_decision" and not audit_record.get("ok"):
        return "loss"
    if ev == "spawn_skipped_no_launcher":
        return "loss"
    return None


def update_capability_map(write: bool = True) -> dict[str, Any]:
    audits = _read_audit_lines(days=2)
    twin = _read_openclaw_completions()

    stats: dict[str, dict[str, int]] = {}
    recent_wins: list[str] = []
    recent_losses: list[str] = []
    untried_seen: set[str] = set()

    for rec in audits:
        outcome = _classify_outcome(rec)
        if outcome is None:
            continue
        title = rec.get("title") or (rec.get("candidate") or {}).get("title") or ""
        tt = classify_task_type(title)
        stats.setdefault(tt, {"attempts": 0, "wins": 0, "losses": 0, "pending": 0})
        stats[tt]["attempts"] += 1
        if outcome == "loss":
            stats[tt]["losses"] += 1
            recent_losses.append(title[:80])
        elif outcome == "pending":
            stats[tt]["pending"] += 1
            stats[tt]["wins"] += 1
            recent_wins.append(title[:80])

    for c in twin:
        title = c.get("completion_excerpt") or c.get("cmdline", "")[:80]
        tt = classify_task_type(c.get("cmdline", "") + " " + (title or ""))
        stats.setdefault(tt, {"attempts": 0, "wins": 0, "losses": 0, "pending": 0})
        stats[tt]["attempts"] += 1
        stop = (c.get("stop_reason") or "").lower()
        if c.get("errors") or stop in ("error", "no_log", "timeout"):
            stats[tt]["losses"] += 1
            recent_losses.append(f"[twin] {title[:70]}")
        else:
            stats[tt]["wins"] += 1
            recent_wins.append(f"[twin] {title[:70]}")

    strengths: list[dict[str, Any]] = []
    weaknesses: list[dict[str, Any]] = []
    all_types: list[dict[str, Any]] = []
    for tt, s in stats.items():
        att = s["attempts"]
        win_rate = s["wins"] / att if att else 0.0
        entry = {
            "task_type": tt,
            "attempts": att,
            "wins": s["wins"],
            "losses": s["losses"],
            "success_rate": round(win_rate, 3),
        }
        all_types.append(entry)
        if att >= STRENGTH_MIN_ATTEMPTS and win_rate >= STRENGTH_MIN_SUCCESS_RATE:
            strengths.append(entry)
        elif att >= STRENGTH_MIN_ATTEMPTS and win_rate <= WEAKNESS_MAX_SUCCESS_RATE:
            weaknesses.append(entry)

    known = set(stats.keys())
    untried = [name for (name, _pat) in TASK_TYPE_PATTERNS if name not in known]

    user_hist = ROOT / "state" / "user_prompts_history.jsonl"
    if user_hist.exists():
        try:
            for line in user_hist.read_text().splitlines()[-200:]:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    txt = r.get("prompt") or r.get("text") or r.get("message") or ""
                    tt = classify_task_type(txt)
                    if tt not in known:
                        untried_seen.add(tt)
                except json.JSONDecodeError:
                    continue
        except OSError:
            pass
    untried = sorted(set(untried) | untried_seen)

    cap_map = {
        "last_updated": _now_utc(),
        "strengths": sorted(strengths, key=lambda x: -x["success_rate"])[:10],
        "weaknesses": sorted(weaknesses, key=lambda x: x["success_rate"])[:10],
        "untried": untried[:15],
        "recent_wins": recent_wins[-RECENT_LIMIT:],
        "recent_losses": recent_losses[-RECENT_LIMIT:],
        "all_types": sorted(all_types, key=lambda x: -x["attempts"])[:20],
        "total_attempts": sum(s["attempts"] for s in stats.values()),
        "audit_records_scanned": len(audits),
        "twin_records_scanned": len(twin),
    }

    if write:
        GABRIEL_SELF_DIR.mkdir(parents=True, exist_ok=True)
        tmp = CAPABILITY_MAP.with_suffix(".tmp")
        tmp.write_text(json.dumps(cap_map, indent=2))
        tmp.replace(CAPABILITY_MAP)
        # Atomic heartbeat write (guardrail point 2)
        try:
            import os as _os
            from datetime import datetime as _dt, timezone as _tz
            hb = GABRIEL_SELF_DIR / "heartbeat.json"
            hb_tmp = hb.with_suffix(".tmp")
            hb_payload = {
                "ts": _dt.now(_tz.utc).isoformat(),
                "pid": _os.getpid(),
                "cycle_id": _os.environ.get("CYCLE_ID", "manual"),
                "status": "ok",
                "total_attempts": cap_map.get("total_attempts", 0),
                "strengths": len(cap_map.get("strengths", [])),
                "weaknesses": len(cap_map.get("weaknesses", [])),
            }
            hb_tmp.write_text(json.dumps(hb_payload, indent=2))
            hb_tmp.replace(hb)
        except Exception:
            pass
    return cap_map


def get_capability_summary() -> dict[str, Any]:
    if not CAPABILITY_MAP.exists():
        return {"strengths": [], "weaknesses": [], "untried": [],
                "recent_wins": [], "recent_losses": []}
    try:
        return json.loads(CAPABILITY_MAP.read_text())
    except (OSError, json.JSONDecodeError):
        return {"strengths": [], "weaknesses": [], "untried": [],
                "recent_wins": [], "recent_losses": []}


def route_model_for_task(task_type: str, cap_map: dict[str, Any] | None = None) -> dict[str, Any]:
    cap = cap_map or get_capability_summary()
    # exclude uncategorized from strength-based fast-path routing
    strengths = {e["task_type"] for e in cap.get("strengths", []) if e["task_type"] != DEFAULT_TASK_TYPE}
    weaknesses = {e["task_type"] for e in cap.get("weaknesses", []) if e["task_type"] != DEFAULT_TASK_TYPE}
    untried = set(cap.get("untried", []))

    if task_type in strengths:
        return {"model": "haiku", "effort": "low", "smoke_test_required": False,
                "reason": f"task_type={task_type} is a known strength (cheap routing)"}
    if task_type in weaknesses:
        return {"model": "opus", "effort": "high", "smoke_test_required": True,
                "reason": f"task_type={task_type} is a known weakness (escalate + verify)"}
    if task_type in untried:
        return {"model": "sonnet", "effort": "medium", "smoke_test_required": True,
                "reason": f"task_type={task_type} is untried (exploration budget)"}
    return {"model": "sonnet", "effort": "medium", "smoke_test_required": False,
            "reason": "task_type unclassified (default routing)"}


def _deepseek_oneshot(prompt: str, timeout_s: int = DEEPSEEK_TIMEOUT_S) -> str | None:
    if not OPENCLAW_BIN.exists():
        return None
    cmd = [
        str(OPENCLAW_BIN), "agent", "--local",
        "--agent", "main",
        "--model", "deepseek/deepseek-v4-flash",
        "--json", "--message", prompt,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout_s, check=False)
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout
    except (subprocess.TimeoutExpired, OSError):
        return None
    return None


def _extract_lesson_from_deepseek(raw: str) -> str | None:
    if not raw:
        return None
    try:
        env = json.loads(raw)
        if isinstance(env, dict):
            payloads = env.get("payloads")
            if isinstance(payloads, list) and payloads:
                t = payloads[0].get("text", "") if isinstance(payloads[0], dict) else ""
                if t:
                    return t.strip().splitlines()[0][:300]
            if "message" in env:
                return str(env["message"]).strip().splitlines()[0][:300]
    except json.JSONDecodeError:
        pass
    for line in raw.splitlines():
        line = line.strip()
        if line:
            return line[:300]
    return None


def _heuristic_lesson(events: list[dict[str, Any]], cap: dict[str, Any]) -> str:
    fail_count = sum(1 for e in events if e.get("event") in ("spawn_failed", "idle_because"))
    reject_count = sum(1 for e in events if e.get("event") == "gate_decision" and not e.get("ok"))
    if fail_count >= 3:
        return "3+ spawn failures in last window - diagnose root cause before more spawns"
    if reject_count >= 5:
        reasons: dict[str, int] = {}
        for e in events:
            if e.get("event") == "gate_decision" and not e.get("ok"):
                r = e.get("reason", "?")
                reasons[r] = reasons.get(r, 0) + 1
        top = max(reasons.items(), key=lambda x: x[1])[0] if reasons else "?"
        return f"Gate-reject reason '{top}' dominates - tighten ideate to avoid this class"
    weaknesses = [w["task_type"] for w in cap.get("weaknesses", [])]
    if weaknesses:
        return f"Weakness areas {weaknesses[:3]} need higher-tier model + smoke test next cycle"
    return "Recent cycle stable - explore one untried task type to expand capability"


def reflect(cycle_id: str | None = None, window: int = REFLEXION_WINDOW) -> dict[str, Any] | None:
    audits = _read_audit_lines(days=1)
    spawn_events = [r for r in audits if r.get("event") in
                    ("spawn_launched", "spawn_failed", "gate_decision", "idle_because")][-window:]
    if not spawn_events:
        return None

    summary_lines: list[str] = []
    for r in spawn_events:
        ev = r.get("event")
        title = r.get("title") or (r.get("candidate") or {}).get("title") or "(no title)"
        if ev == "spawn_launched":
            summary_lines.append(f"- LAUNCHED: {title[:70]}")
        elif ev == "spawn_failed":
            summary_lines.append(f"- FAILED: {title[:70]} ({(r.get('error') or '?')[:50]})")
        elif ev == "gate_decision":
            ok = r.get("ok")
            reason = r.get("reason", "?")
            summary_lines.append(f"- GATE {'PASS' if ok else 'REJECT'}: {title[:60]} ({reason[:40]})")
        elif ev == "idle_because":
            summary_lines.append(f"- IDLE: cycle={r.get('cycle_id', '?')} reason={r.get('reason', '?')}")

    cap = get_capability_summary()
    prompt = (
        "You are Gabriel, an autonomous Claude+OpenClaw agent. Reflect on your last "
        f"{len(spawn_events)} actions and write ONE concrete lesson to your future self.\n\n"
        f"Strengths: {[s['task_type'] for s in cap.get('strengths', [])[:5]]}\n"
        f"Weaknesses: {[w['task_type'] for w in cap.get('weaknesses', [])[:5]]}\n"
        f"Untried: {cap.get('untried', [])[:5]}\n\n"
        "Recent actions:\n" + "\n".join(summary_lines) + "\n\n"
        "Output ONE LINE of actionable advice (start with a verb, no preamble). "
        "Format: just the lesson text, no JSON wrapping."
    )

    raw = _deepseek_oneshot(prompt)
    if raw:
        lesson = _extract_lesson_from_deepseek(raw)
        source = "deepseek"
    else:
        lesson = _heuristic_lesson(spawn_events, cap)
        source = "heuristic_fallback"

    if not lesson:
        return None

    rec = {
        "ts": _now_utc(),
        "cycle_id": cycle_id,
        "window": len(spawn_events),
        "lesson": lesson,
        "source": source,
        "strengths_snapshot": [s["task_type"] for s in cap.get("strengths", [])[:5]],
        "weaknesses_snapshot": [w["task_type"] for w in cap.get("weaknesses", [])[:5]],
    }

    GABRIEL_SELF_DIR.mkdir(parents=True, exist_ok=True)
    with REFLEXIONS.open("a") as f:
        f.write(json.dumps(rec) + "\n")

    if not LESSONS_MD.exists():
        LESSONS_MD.write_text("# Gabriel lessons (auto-written by gabriel_self.reflect)\n\n")
    with LESSONS_MD.open("a") as f:
        f.write(f"\n- {rec['ts']} (cycle {cycle_id}, src={source}): {lesson}\n")

    return rec


def last_n_lessons(n: int = 5) -> list[str]:
    if not REFLEXIONS.exists():
        return []
    try:
        lines = REFLEXIONS.read_text().splitlines()[-n:]
        out = []
        for line in reversed(lines):
            try:
                r = json.loads(line)
                lesson = r.get("lesson")
                if lesson:
                    out.append(lesson)
            except json.JSONDecodeError:
                continue
        return out
    except OSError:
        return []


def refresh_self_status() -> None:
    cap = get_capability_summary()
    lessons = last_n_lessons(3)

    parts: list[str] = []
    parts.append("# Gabriel Self-Status\n")
    parts.append(f"_auto-refreshed: {_now_utc()}_\n")
    parts.append(f"\n_Capability map last updated: {cap.get('last_updated', '(never)')}_\n")
    parts.append(f"_Total audited spawns: {cap.get('total_attempts', 0)}_\n")
    parts.append(f"_Audit records scanned: {cap.get('audit_records_scanned', 0)} (autonomous_mode), "
                 f"{cap.get('twin_records_scanned', 0)} (openclaw twin)_\n")

    parts.append("\n## Top 5 strengths (success_rate >= 0.7, attempts >= 3)\n")
    if cap.get("strengths"):
        for s in cap["strengths"][:5]:
            parts.append(f"- **{s['task_type']}** - {s['wins']}/{s['attempts']} "
                         f"({s['success_rate']:.1%})")
    else:
        parts.append("_(none yet - need >=3 attempts at any task type)_")

    parts.append("\n\n## Top 5 weaknesses (success_rate <= 0.3, attempts >= 3)\n")
    if cap.get("weaknesses"):
        for w in cap["weaknesses"][:5]:
            parts.append(f"- **{w['task_type']}** - {w['wins']}/{w['attempts']} "
                         f"({w['success_rate']:.1%})")
    else:
        parts.append("_(none yet)_")

    parts.append("\n\n## Untried task types\n")
    if cap.get("untried"):
        parts.append(", ".join(f"`{t}`" for t in cap["untried"][:10]))
    else:
        parts.append("_(none)_")

    parts.append("\n\n## Last 3 lessons (reflexion)\n")
    if lessons:
        for i, l in enumerate(lessons, 1):
            parts.append(f"{i}. {l}")
    else:
        parts.append("_(no reflexions yet)_")

    parts.append("\n\n## Recent wins (last 5)\n")
    for w in cap.get("recent_wins", [])[-5:]:
        parts.append(f"- {w}")

    parts.append("\n\n## Recent losses (last 5)\n")
    for l in cap.get("recent_losses", [])[-5:]:
        parts.append(f"- {l}")

    parts.append("\n\n## All task-type stats\n")
    parts.append("| task_type | attempts | wins | losses | success_rate |")
    parts.append("|---|---|---|---|---|")
    for t in cap.get("all_types", [])[:15]:
        parts.append(f"| `{t['task_type']}` | {t['attempts']} | {t['wins']} | "
                     f"{t['losses']} | {t['success_rate']:.1%} |")

    DASHBOARD_SELF_STATUS.parent.mkdir(parents=True, exist_ok=True)
    DASHBOARD_SELF_STATUS.write_text("\n".join(parts) + "\n")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--update-map", action="store_true")
    p.add_argument("--reflect", action="store_true")
    p.add_argument("--refresh-status", action="store_true")
    p.add_argument("--all", action="store_true")
    p.add_argument("--cycle-id", default=None)
    args = p.parse_args()

    did = False
    if args.all or args.update_map:
        cap = update_capability_map(write=True)
        print(f"[capability_map] wrote {CAPABILITY_MAP} - "
              f"{len(cap['strengths'])} strengths, {len(cap['weaknesses'])} weaknesses, "
              f"{cap['total_attempts']} attempts")
        did = True
    if args.all or args.reflect:
        r = reflect(cycle_id=args.cycle_id or "smoke")
        if r:
            print(f"[reflect] wrote lesson: {r['lesson'][:80]} (src={r['source']})")
        else:
            print("[reflect] no spawn events - no lesson written")
        did = True
    if args.all or args.refresh_status:
        refresh_self_status()
        print(f"[refresh_status] wrote {DASHBOARD_SELF_STATUS}")
        did = True
    if not did:
        p.print_help()
