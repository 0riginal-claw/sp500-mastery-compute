#!/usr/bin/env bash
# autosolve_skip: hook construction — internal mechanical work, no external blocker
# UserPromptSubmit hook — classifies user prompt + injects into autonomous_mode user_inbox.jsonl
# so the daemon spawns helper(s) in parallel with Claude's response.
#
# Schema matches bin/autonomous (id=u_<12-hex sha256>, ts=ISO UTC, intents from _INTENT_DISPATCH).
# Source tag: "chat_prompt_hook" (vs "user_cli" from bin/autonomous).
#
# Skip rules (return 0, no inbox write):
#   - empty prompt
#   - prompt < 12 chars or > 4000 chars
#   - literal status query ("status", "update", "what's happening", ...)
#   - sub-agent spawn brief markers (# scope_estimate_min:, # autosolve_skip:, ...)
#   - already-classified system payloads
#
# Never blocks. Hook exits 0 always (silent best-effort).

set -uo pipefail

LOG_DIR="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/logs/auto_solve"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/prompt_to_inbox.log"
INBOX="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/state/autonomous_mode/user_inbox.jsonl"
mkdir -p "$(dirname "$INBOX")"

# Read JSON from stdin (Claude Code UserPromptSubmit hook contract)
INPUT=$(cat 2>/dev/null || true)
if [[ -z "$INPUT" ]]; then exit 0; fi

PYTHON="${VENV_PY:-/Users/orginal/.venvs/sp500-mastery/bin/python}"
if [[ ! -x "$PYTHON" ]]; then PYTHON="$(command -v python3 || echo /usr/bin/python3)"; fi

# Pipe INPUT to python; pass INBOX + LOG_FILE as argv
printf '%s' "$INPUT" | "$PYTHON" "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/home/.claude/hooks/prompt-to-inbox/_classify.py" "$INBOX" "$LOG_FILE" 2>/dev/null || true

exit 0
