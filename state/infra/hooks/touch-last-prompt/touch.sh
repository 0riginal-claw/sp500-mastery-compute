#!/usr/bin/env bash
# autosolve_skip: REPO-LOCAL feature — adding user-prompt content archive for persona loop (2026-05-20)
# touch-last-prompt — UserPromptSubmit hook
#
# Two purposes:
#   1. Idle-detect: write Unix ts to ~/.claude/state/last_user_prompt.unix
#      (autonomous_mode_daemon yields when user is actively typing).
#   2. Persona-loop feed (added 2026-05-20): append (ts, prompt) JSONL row
#      to AI-Tools/state/user_prompts_history.jsonl so the daemon's
#      USER-PERSONA ideator can learn from the user's actual voice.
#      Keeps rolling tail of 100 entries.
#
# Hook stdin is the Claude Code UserPromptSubmit payload (JSON):
#   {"prompt": "...", "session_id": "...", "transcript_path": "...", ...}
# We read it, extract `prompt`, and persist the rolling tail.
#
# Safety: never echo prompt to stdout/stderr (hook output appears in UI).
# All failures swallow silently — never break the user's prompt submission.

set +e
LC_ALL=C

# 1. Idle-detect timestamp
mkdir -p ~/.claude/state 2>/dev/null
date +%s > ~/.claude/state/last_user_prompt.unix

# 2. Capture prompt content for persona loop
ROOT="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools"
HIST="$ROOT/state/user_prompts_history.jsonl"
MAX_ENTRIES=100

# Read stdin (payload). If empty (or non-JSON), skip silently.
PAYLOAD="$(cat 2>/dev/null)"
if [ -z "$PAYLOAD" ]; then
    exit 0
fi

# Pass to python for safe JSON extraction + rolling-tail rewrite. Never fail.
HIST_PATH="$HIST" MAX_ENTRIES="$MAX_ENTRIES" PAYLOAD="$PAYLOAD" \
python3 - <<'PY' 2>/dev/null
import json, os, sys
from datetime import datetime, timezone

hist_path = os.environ.get("HIST_PATH")
max_n = int(os.environ.get("MAX_ENTRIES", "100"))
payload_raw = os.environ.get("PAYLOAD", "")

try:
    payload = json.loads(payload_raw) if payload_raw else {}
except (json.JSONDecodeError, ValueError):
    payload = {}

prompt = payload.get("prompt") if isinstance(payload, dict) else None
if not isinstance(prompt, str) or not prompt.strip():
    sys.exit(0)

# Truncate very long prompts so the history file stays small (head 4k chars)
prompt_short = prompt[:4000]

row = {
    "ts": datetime.now(timezone.utc).isoformat(),
    "prompt": prompt_short,
    "len": len(prompt),
}

# Read existing rows (best-effort)
rows = []
try:
    with open(hist_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
except FileNotFoundError:
    pass
except OSError:
    sys.exit(0)

rows.append(row)
# Keep only the last MAX_ENTRIES (rolling tail)
rows = rows[-max_n:]

# Atomic rewrite
tmp = hist_path + ".tmp"
try:
    os.makedirs(os.path.dirname(hist_path), exist_ok=True)
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, hist_path)
except OSError:
    try:
        os.unlink(tmp)
    except OSError:
        pass
    sys.exit(0)
PY

exit 0
