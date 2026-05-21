#!/usr/bin/env bash
# PreToolUse hook for Task / mcp__plugin_fallback-agent_fallback__Task spawns.
# If the spawn's prompt exceeds ~2000 tokens, run it through LLMLingua at
# --target-ratio 0.5 and return the compressed version via
# hookSpecificOutput.updatedInput.prompt so Claude Code mutates the
# outgoing tool_input before dispatching the sub-agent.
#
# Passes through unchanged (exit 0, no JSON) if:
#  - tool input is unreadable
#  - prompt is empty or under 2000-token threshold
#  - LLMLingua script / venv python missing
#  - compression fails or returns empty
#  - compression yields a longer result (sanity)
#
# Logs every invocation to logs/hooks/auto_spawn_compress.log for diagnostics.

set -uo pipefail

LOG_DIR="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/logs/hooks"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/auto_spawn_compress.log"

SCRIPTS_DIR="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/scripts"
LLMLINGUA_SCRIPT="$SCRIPTS_DIR/llmlingua_compress.py"
VENV_PYTHON="/Users/orginal/.venvs/sp500-mastery/bin/python"

TS=$(date -u '+%Y-%m-%d %H:%M:%S UTC')

# Persist hook input to a temp file so we can re-read it from multiple python invocations
INPUT_FILE=$(mktemp /tmp/spawn_compress_input.XXXXXX)
PROMPT_FILE=$(mktemp /tmp/spawn_compress_in.XXXXXX)
RESULT_FILE=$(mktemp /tmp/spawn_compress_out.XXXXXX)
META_FILE=$(mktemp /tmp/spawn_compress_meta.XXXXXX)
trap 'rm -f "$INPUT_FILE" "$PROMPT_FILE" "$RESULT_FILE" "$META_FILE"' EXIT

cat > "$INPUT_FILE"

# Extract tool_name + prompt length. Writes "<tool_name> <prompt_len>" to META_FILE.
python3 - "$INPUT_FILE" "$PROMPT_FILE" "$META_FILE" <<'PYEOF' 2>>"$LOG_FILE"
import json, sys
in_path, prompt_path, meta_path = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    with open(in_path, 'r', encoding='utf-8') as f:
        data = json.loads(f.read())
    tn = data.get('tool_name', 'unknown') or 'unknown'
    ti = data.get('tool_input', {}) or {}
    prompt = ti.get('prompt', '') or ''
    with open(prompt_path, 'w', encoding='utf-8') as f:
        f.write(prompt)
    with open(meta_path, 'w', encoding='utf-8') as f:
        f.write(f"{tn} {len(prompt)}")
except Exception as e:
    with open(meta_path, 'w', encoding='utf-8') as f:
        f.write("unknown 0")
    sys.stderr.write(f"auto-spawn-compress parse error: {e}\n")
PYEOF

META=$(cat "$META_FILE" 2>/dev/null || echo "unknown 0")
TOOL_NAME=$(echo "$META" | awk '{print $1}')
PROMPT_LEN=$(echo "$META" | awk '{print $2}')
PROMPT_LEN=${PROMPT_LEN:-0}
TOKEN_ESTIMATE=$(( PROMPT_LEN / 4 ))

printf '[%s] %s: ~%d tokens (%d chars)\n' "$TS" "$TOOL_NAME" "$TOKEN_ESTIMATE" "$PROMPT_LEN" >> "$LOG_FILE"

if [[ $PROMPT_LEN -eq 0 ]]; then
    printf '[%s]   Empty prompt or unreadable input. Passthrough.\n' "$TS" >> "$LOG_FILE"
    exit 0
fi

# Threshold lowered 2000→1500 on 2026-05-18 after audit:
# 297 spawn events, only 3 (1.0%) crossed 2000. Avg=675, p95=~1500, max=6400.
# Lowering to 1500 captures the long-tail bursts without latency on small spawns.
THRESHOLD=${LLMLINGUA_SPAWN_THRESHOLD_TOKENS:-1500}
if [[ $TOKEN_ESTIMATE -le $THRESHOLD ]]; then
    printf '[%s]   Below %d-token threshold. Passthrough.\n' "$TS" "$THRESHOLD" >> "$LOG_FILE"
    exit 0
fi

if [[ ! -x "$VENV_PYTHON" ]]; then
    VENV_PYTHON=$(command -v python3 2>/dev/null || true)
fi

if [[ -z "$VENV_PYTHON" ]] || [[ ! -f "$LLMLINGUA_SCRIPT" ]]; then
    printf '[%s]   LLMLingua unavailable (python=%s script=%s). Passthrough.\n' \
        "$TS" "$VENV_PYTHON" "$LLMLINGUA_SCRIPT" >> "$LOG_FILE"
    exit 0
fi

COMPRESS_OK=0
"$VENV_PYTHON" "$LLMLINGUA_SCRIPT" \
    --target-ratio 0.5 \
    < "$PROMPT_FILE" \
    > "$RESULT_FILE" 2>>"$LOG_FILE" && COMPRESS_OK=1 || true

if [[ $COMPRESS_OK -eq 0 ]] || [[ ! -s "$RESULT_FILE" ]]; then
    printf '[%s]   Compression failed or empty. Passthrough.\n' "$TS" >> "$LOG_FILE"
    exit 0
fi

COMP_CHARS=$(wc -c < "$RESULT_FILE" | tr -d ' ')
COMP_TOKENS=$(( COMP_CHARS / 4 ))
SAVED=$(( TOKEN_ESTIMATE - COMP_TOKENS ))
printf '[%s]   Compressed: %d -> %d tokens (saved %d)\n' "$TS" "$TOKEN_ESTIMATE" "$COMP_TOKENS" "$SAVED" >> "$LOG_FILE"

if [[ $COMP_TOKENS -ge $TOKEN_ESTIMATE ]]; then
    printf '[%s]   Compressed >= original. Passthrough.\n' "$TS" >> "$LOG_FILE"
    exit 0
fi

# Emit hookSpecificOutput with updatedInput preserving all other tool_input fields.
python3 - "$INPUT_FILE" "$RESULT_FILE" "$TOKEN_ESTIMATE" "$COMP_TOKENS" <<'PYEOF' 2>>"$LOG_FILE"
import json, sys
in_path, result_path, orig_tok, comp_tok = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
try:
    with open(in_path, 'r', encoding='utf-8') as f:
        data = json.loads(f.read())
    tool_input = (data.get('tool_input') or {}).copy()
    with open(result_path, 'r', encoding='utf-8') as f:
        compressed = f.read()
    header = f"[LLMLingua auto-compressed on spawn: {orig_tok}->{comp_tok} tokens]\n"
    tool_input['prompt'] = header + compressed
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "updatedInput": tool_input
        }
    }))
except Exception as e:
    sys.stderr.write(f"auto-spawn-compress emit error: {e}\n")
PYEOF

exit 0
