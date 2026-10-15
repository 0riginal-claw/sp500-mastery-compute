#!/usr/bin/env bash
# L2 — PostToolUse hook for any tool: detect failure signals, append to error_pile.

set +e
LC_ALL=C
LOCAL_PILE="/Users/orginal/.zg/state/error_pile"
DRIVE_PILE="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/state/error_pile"
mkdir -p "$LOCAL_PILE" "$DRIVE_PILE" 2>/dev/null
TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
DATE=$(date -u +"%Y-%m-%d")

if [[ "${UNIVERSAL_ERROR_L2_DISABLE:-0}" == "1" ]]; then exit 0; fi

PAYLOAD="$(cat 2>/dev/null)"
if [[ -z "$PAYLOAD" ]]; then exit 0; fi

printf '%s' "$PAYLOAD" | python3 - "$TS" "$LOCAL_PILE" "$DRIVE_PILE" "$DATE" <<'PYEOF'
import sys, json, hashlib, re, os
ts, local_pile, drive_pile, date = sys.argv[1:5]
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(0)

sid = (d.get("session_id") or "").strip()
tool = (d.get("tool_name") or "").strip()
resp = d.get("tool_response") or {}
if not isinstance(resp, dict):
    resp = {"output": str(resp)}

is_err = bool(resp.get("is_error", False))
intr = bool(resp.get("interrupted", False))
exit_code = resp.get("exit_code")
if exit_code is None: exit_code = resp.get("returncode")

parts = []
for k in ("output","stdout","stderr","error","message","content"):
    v = resp.get(k)
    if isinstance(v, str):
        parts.append(v)
    elif isinstance(v, list):
        for it in v:
            if isinstance(it, dict):
                t = it.get("text") or it.get("content") or ""
                if isinstance(t, str): parts.append(t)
            elif isinstance(it, str):
                parts.append(it)
text = " ".join(parts)[:8000]

triggers = []
if is_err: triggers.append("is_error")
if intr: triggers.append("interrupted")
if tool == "Bash" and exit_code not in (None, 0, "0", "", "None"):
    try:
        ec = int(exit_code)
        if ec != 0:
            triggers.append(f"exit_code={ec}")
    except (ValueError, TypeError):
        pass

err_rx = re.compile(
    r"Traceback \(most recent call last\)|ModuleNotFoundError|"
    r"ImportError:|AttributeError:|TypeError:|FATAL:|^ERROR:|"
    r"HTTP\s*[45]\d\d|BLOCKED:|Permission denied|RateLimit|"
    r"ConnectionError|TimeoutError|hook error",
    re.MULTILINE | re.IGNORECASE,
)
if not triggers and err_rx.search(text):
    triggers.append("regex_match")

if not triggers:
    sys.exit(0)

body = (text or "").strip()[:1000]
h = hashlib.sha256(("L2"+tool+body[:500]).encode()).hexdigest()[:16]
entry = {
    "ts": ts,
    "layer": "L2_posttool",
    "source": f"tool={tool} session={sid}",
    "kind": triggers[0],
    "severity": "error" if "is_error" in triggers or "regex_match" in triggers else "warn",
    "body": body or f"tool={tool} {','.join(triggers)}",
    "hash": h,
    "session_id": sid,
}
line = json.dumps(entry, separators=(",",":")) + "\n"
for pd in (local_pile, drive_pile):
    try:
        with open(os.path.join(pd, f"{date}.jsonl"), "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass
PYEOF

exit 0
