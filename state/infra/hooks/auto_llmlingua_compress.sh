#!/usr/bin/env bash
# UserPromptSubmit hook: auto-compress prompts >2000 tokens with LLMLingua
# Passes through unchanged if under threshold or if compression fails.
# Logs every invocation for diagnostics.

set -uo pipefail

LOG_DIR="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/logs/hooks"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/llmlingua_compress.log"

SCRIPTS_DIR="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/scripts"
LLMLINGUA_SCRIPT="$SCRIPTS_DIR/llmlingua_compress.py"
VENV_PYTHON="/Users/orginal/.venvs/sp500-mastery/bin/python"

TS=$(date -u '+%Y-%m-%d %H:%M:%S UTC')

# Read full JSON input
INPUT=$(cat)

# Extract prompt via python3 — handles unicode and escaping correctly
PROMPT=$(python3 -c "
import json, sys
try:
    data = json.loads(sys.stdin.read())
    sys.stdout.write(data.get('prompt', ''))
except Exception:
    pass
" <<< "$INPUT" 2>/dev/null || true)

if [[ -z "$PROMPT" ]]; then
    exit 0
fi

# Estimate tokens (chars / 4 — close enough for gating)
CHAR_COUNT=${#PROMPT}
TOKEN_ESTIMATE=$(( CHAR_COUNT / 4 ))

printf '[%s] UserPromptSubmit: ~%d tokens (%d chars)\n' "$TS" "$TOKEN_ESTIMATE" "$CHAR_COUNT" >> "$LOG_FILE"

# Passthrough if under threshold (lowered 2000→1500→800 on 2026-05-19 after
# audit: 197KB log, 966 passthrough vs 5 compressions. Avg prompt 200-700
# tokens. 800-token bar catches every nontrivial planning/research prompt
# while still ignoring trivial chatter. Env override available.
THRESHOLD=${LLMLINGUA_THRESHOLD_TOKENS:-800}
if [[ $TOKEN_ESTIMATE -le $THRESHOLD ]]; then
    printf '[%s]   Below %d-token threshold. Passthrough.\n' "$TS" "$THRESHOLD" >> "$LOG_FILE"
    exit 0
fi

# Resolve python interpreter
if [[ ! -x "$VENV_PYTHON" ]]; then
    VENV_PYTHON=$(command -v python3 2>/dev/null || true)
fi

if [[ -z "$VENV_PYTHON" ]] || [[ ! -f "$LLMLINGUA_SCRIPT" ]]; then
    printf '[%s]   LLMLingua not available. Passthrough. (python=%s script=%s)\n' \
        "$TS" "$VENV_PYTHON" "$LLMLINGUA_SCRIPT" >> "$LOG_FILE"
    exit 0
fi

# Run compression via temp file (avoids arg-length limits on large prompts)
PROMPT_FILE=$(mktemp /tmp/llmlingua_in.XXXXXX)
RESULT_FILE=$(mktemp /tmp/llmlingua_out.XXXXXX)
trap 'rm -f "$PROMPT_FILE" "$RESULT_FILE"' EXIT

printf '%s' "$PROMPT" > "$PROMPT_FILE"

COMPRESS_OK=0
# Claude Code kills this hook at 90s (set in settings.json); no internal timeout needed
"$VENV_PYTHON" "$LLMLINGUA_SCRIPT" \
    --target-ratio 0.5 \
    --text "$(cat "$PROMPT_FILE")" \
    > "$RESULT_FILE" 2>>"$LOG_FILE" && COMPRESS_OK=1 || true

COMPRESSED=""
[[ -s "$RESULT_FILE" ]] && COMPRESSED=$(cat "$RESULT_FILE")

if [[ -z "$COMPRESSED" ]] || [[ $COMPRESS_OK -eq 0 ]]; then
    printf '[%s]   Compression failed or empty. Passthrough.\n' "$TS" >> "$LOG_FILE"
    exit 0
fi

COMP_CHARS=${#COMPRESSED}
COMP_TOKENS=$(( COMP_CHARS / 4 ))
SAVED=$(( TOKEN_ESTIMATE - COMP_TOKENS ))
printf '[%s]   Compressed: %d → %d tokens (saved %d)\n' "$TS" "$TOKEN_ESTIMATE" "$COMP_TOKENS" "$SAVED" >> "$LOG_FILE"

# Output additionalContext with compressed prompt
python3 - "$COMPRESSED" "$TOKEN_ESTIMATE" "$COMP_TOKENS" <<'PYEOF'
import json, sys
try:
    compressed = sys.argv[1]
    orig_tok = sys.argv[2]
    comp_tok = sys.argv[3]
    header = f"[LLMLingua auto-compressed: {orig_tok}→{comp_tok} tokens]\n"
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": header + compressed
        }
    }))
except Exception:
    pass
PYEOF
