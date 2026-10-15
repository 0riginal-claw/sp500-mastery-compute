#!/usr/bin/env python3
"""Classify user chat prompt + append to autonomous_mode/user_inbox.jsonl.

Invoked by inject.sh. Reads JSON {"prompt": "..."} (Claude Code UserPromptSubmit
contract) from stdin. Writes a single inbox row with:

    {id, ts, intent, payload, priority, status: "pending", source: "chat_prompt_hook"}

Intents map 1:1 with autonomous_mode_daemon._INTENT_DISPATCH:
    ask, search_github, search_internet, research, add_feature, wire_into_v10, fix

Never raises. Failures log + exit 0.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone


def _log(log_path: str, msg: str) -> None:
    try:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with open(log_path, "a") as fh:
            fh.write(f"[{ts}] {msg}\n")
    except Exception:
        pass


def _read_prompt() -> str:
    try:
        raw = sys.stdin.read()
    except Exception:
        return ""
    if not raw or not raw.strip():
        return ""
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return (data.get("prompt") or "").strip()
    except Exception:
        # Stdin wasn't JSON — treat raw as prompt (defensive fallback)
        pass
    return raw.strip()


# Status / no-op chat fragments — skip
_STATUS_RE = re.compile(
    r"^\s*(update|update on .*|status|current state.*|what'?s happening.*|"
    r"how (are|is) (it|things) going|are you (there|alive)|ping)\s*[?.]?\s*$",
    re.IGNORECASE,
)

# Sub-agent spawn brief markers — if 2+ present in first 2k chars, this isn't
# real user chat, it's a helper prompt that happened to reach this hook.
_SPAWN_MARKERS = (
    "# scope_estimate_min:",
    "# autosolve_skip:",
    "# model_reason:",
    "# decomposition_plan:",
    "# inline_justification:",
    "# fanout_skip:",
    "# karpathy_checked:",
    "# justify_claude:",
    "# pre_compressed:",
)

# Intent classification table (order matters — first match wins)
_PATTERNS: list[tuple[str, str, int]] = [
    # fix / debug
    (r"\b(fix|repair|debug|resolve|broken|failing)\b.{0,80}\b(error|bug|crash|issue|test|build|hook|daemon)\b",
     "fix", 10),
    (r"^\s*fix\b", "fix", 10),
    # wire — integrate existing module
    (r"\bwire\b.{0,40}\b(into|to|with|up)\b", "wire_into_v10", 9),
    (r"^\s*wire\b", "wire_into_v10", 9),
    # add feature
    (r"\b(let'?s\s+)?(add|build|create|implement)\b.{0,80}\b(feature|module|skill|capability|tool|agent|hook|daemon|component|integration|script)\b",
     "add_feature", 9),
    # search github
    (r"\bsearch\s+github\b", "search_github", 8),
    (r"\b(find|look\s+up|gh\s+search)\b.{0,40}\b(github|gh|repo|repository|repos)\b", "search_github", 8),
    # search web/internet
    (r"\bsearch\s+(the\s+)?(internet|web|online|google)\b", "search_internet", 8),
    (r"\b(web|google)\s+search\b", "search_internet", 8),
    # research
    (r"\bresearch\b", "research", 7),
    (r"\b(investigate|explore|look\s+into|deep\s+dive)\b.{0,80}\b(topic|approach|method|technique|strategy|paper|literature)\b",
     "research", 7),
    # question
    (r"^\s*(what|who|when|where|why|how|which|can\s+you|could\s+you|do\s+you|is\s+there|are\s+there)\b",
     "ask", 7),
    (r"\?\s*$", "ask", 7),
]


def _classify(prompt: str) -> tuple[str, int]:
    for pat, name, prio in _PATTERNS:
        if re.search(pat, prompt, re.IGNORECASE | re.DOTALL):
            return name, prio
    # Fallback: low-priority ask
    return "ask", 6


def main() -> int:
    if len(sys.argv) < 3:
        return 0
    inbox_path, log_path = sys.argv[1], sys.argv[2]

    prompt = _read_prompt()
    if not prompt:
        return 0

    L = len(prompt)
    if L < 12 or L > 4000:
        _log(log_path, f"SKIP len={L} (out of range 12..4000)")
        return 0

    if _STATUS_RE.match(prompt):
        _log(log_path, f"SKIP status_query: {prompt[:80]!r}")
        return 0

    first_2k_lower = prompt[:2000].lower()
    spawn_hits = sum(1 for m in _SPAWN_MARKERS if m.lower() in first_2k_lower)
    if spawn_hits >= 2:
        _log(log_path, f"SKIP spawn_brief (markers={spawn_hits})")
        return 0

    intent, priority = _classify(prompt)

    ts = datetime.now(timezone.utc).isoformat()
    raw_id = f"{ts}|{intent}|{prompt[:200]}|{os.getpid()}"
    item_id = "u_" + hashlib.sha256(raw_id.encode()).hexdigest()[:12]

    rec = {
        "id": item_id,
        "ts": ts,
        "intent": intent,
        "payload": prompt[:2000],
        "priority": priority,
        "status": "pending",
        "source": "chat_prompt_hook",
    }

    try:
        os.makedirs(os.path.dirname(inbox_path), exist_ok=True)
        with open(inbox_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        _log(log_path, f"QUEUED id={item_id} intent={intent} priority={priority} len={L}")
    except Exception as e:  # noqa: BLE001
        _log(log_path, f"FAIL write {e!r}")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
