#!/usr/bin/env bash
# PreToolUse BLOCKING gate for spawn prompts.
# Matcher (settings.json): ^(Task|Agent|mcp__plugin_fallback-agent_fallback__Task)$
#
# Behavior:
#   - Reads PreToolUse JSON from stdin.
#   - If tool_input.prompt length > 2000 chars AND prompt does NOT start with
#     a compression marker (regex `^(\[LLMLingua compressed|# pre_compressed:)`),
#     emit a stderr instruction and exit 2 (Claude Code treats exit 2 as block).
#   - Otherwise exit 0 (pass-through).
#
# Why this exists:
#   - auto-spawn-compress.sh silently mutates large prompts via hookSpecificOutput.
#   - This gate runs FIRST and forces the *caller* to compress explicitly when
#     they're sending oversized prompts, so the compression is auditable in the
#     parent transcript instead of being invisible.
#   - Together they form defense-in-depth: this blocks loud, auto-compress
#     handles the silent fallback if this is ever disabled.
#
# Threshold: 2000 chars (~500 tokens). Tighter than auto-spawn-compress (1500
# tokens) because this fires first and represents intent, not panic-mode.
#
# Escape hatch: prepend prompt with `# pre_compressed: <note>` or
# `[LLMLingua compressed: <orig>->...]` to bypass.
#
# Logs to logs/hooks/spawn_prompt_compress.log.

set -uo pipefail

LOG_DIR="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/logs/hooks"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/spawn_prompt_compress.log"

THRESHOLD_CHARS=${SPAWN_PROMPT_COMPRESS_CHAR_THRESHOLD:-2000}
TS=$(date -u '+%Y-%m-%d %H:%M:%S UTC')

INPUT_FILE=$(mktemp /tmp/spc_input.XXXXXX)
META_FILE=$(mktemp /tmp/spc_meta.XXXXXX)
trap 'rm -f "$INPUT_FILE" "$META_FILE"' EXIT

cat > "$INPUT_FILE"

# Extract tool_name + prompt length + first-128-char prefix (for marker check)
python3 - "$INPUT_FILE" "$META_FILE" <<'PYEOF' 2>>"$LOG_FILE"
import json, sys
in_path, meta_path = sys.argv[1], sys.argv[2]
try:
    with open(in_path, 'r', encoding='utf-8') as f:
        data = json.loads(f.read())
    tn = data.get('tool_name', 'unknown') or 'unknown'
    ti = data.get('tool_input', {}) or {}
    prompt = ti.get('prompt', '') or ''
    # Strip leading whitespace before marker check so indented/blank-line prompts still pass
    head = prompt.lstrip()[:128].replace('\n', ' ')
    with open(meta_path, 'w', encoding='utf-8') as f:
        # Tab-separated: tool_name<TAB>prompt_len<TAB>head_prefix
        f.write(f"{tn}\t{len(prompt)}\t{head}")
except Exception as e:
    with open(meta_path, 'w', encoding='utf-8') as f:
        f.write("unknown\t0\t")
    sys.stderr.write(f"spawn-prompt-compress parse error: {e}\n")
PYEOF

META=$(cat "$META_FILE" 2>/dev/null || echo $'unknown\t0\t')
TOOL_NAME=$(printf '%s' "$META" | awk -F '\t' '{print $1}')
PROMPT_LEN=$(printf '%s' "$META" | awk -F '\t' '{print $2}')
HEAD=$(printf '%s' "$META" | awk -F '\t' '{print $3}')
PROMPT_LEN=${PROMPT_LEN:-0}

printf '[%s] %s len=%s\n' "$TS" "$TOOL_NAME" "$PROMPT_LEN" >> "$LOG_FILE"

if [[ "$PROMPT_LEN" -le "$THRESHOLD_CHARS" ]]; then
    exit 0
fi

# Compression marker check — case-sensitive, anchored at start of stripped prompt
if [[ "$HEAD" =~ ^(\[LLMLingua\ compressed|\#\ pre_compressed:) ]]; then
    printf '[%s]   Marker present. Allow.\n' "$TS" >> "$LOG_FILE"
    exit 0
fi

printf '[%s]   BLOCK: %s chars, no marker.\n' "$TS" "$PROMPT_LEN" >> "$LOG_FILE"

cat <<EOF >&2
BLOCKED: spawn prompt ${PROMPT_LEN} chars must be compressed first. Run:
  echo '<prompt>' | python3 scripts/llmlingua_compress.py --target-ratio 0.5 --text -
Then prepend output with \`[LLMLingua compressed: <orig>->&lt;new&gt; tokens]\`.
Or prepend \`# pre_compressed: <note>\` to bypass (e.g. when prompt is already terse JSON).
EOF
exit 2
