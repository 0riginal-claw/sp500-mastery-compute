#!/usr/bin/env bash
# SessionStart hook: inject mandates + CLAUDE.md + recent session log summaries
# Mirrors what /resume does — reads project memory + last 3 session logs,
# prefixed with ~/.zg/mandates.md so every session bootstraps universal rules.

set -uo pipefail

CONTEXT_FILE=$(mktemp /tmp/auto_resume_ctx.XXXXXX)
trap 'rm -f "$CONTEXT_FILE"' EXIT

# --- 1. Mandates (universal-resume rule) -------------------------------------
for mp in \
  "/Users/orginal/.zg/mandates.md" \
  "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/home/.zg/mandates.md" \
  "$HOME/.zg/mandates.md"; do
  if [[ -r "$mp" && -s "$mp" ]]; then
    printf '=== ~/.zg/mandates.md (universal mandates) ===\n' >> "$CONTEXT_FILE"
    cat "$mp" >> "$CONTEXT_FILE"
    printf '\n' >> "$CONTEXT_FILE"
    break
  fi
done

# --- 2. CLAUDE.md (walk up to find it) ---------------------------------------
find_claude_md() {
    local dir="$PWD"
    while [[ "$dir" != "/" ]]; do
        for name in "CLAUDE.md" "Claude.md" ".claude/CLAUDE.md" "docs/CLAUDE.md"; do
            [[ -f "$dir/$name" ]] && echo "$dir/$name" && return 0
        done
        dir=$(dirname "$dir")
    done
    # Fallback: AI-Tools workspace CLAUDE.md
    local ws="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools"
    [[ -f "$ws/CLAUDE.md" ]] && echo "$ws/CLAUDE.md" && return 0
    return 1
}

CLAUDE_MD=$(find_claude_md 2>/dev/null || true)

if [[ -n "$CLAUDE_MD" ]]; then
    printf '=== CLAUDE.md (%s) ===\n' "$(dirname "$CLAUDE_MD")" >> "$CONTEXT_FILE"
    head -80 "$CLAUDE_MD" >> "$CONTEXT_FILE" 2>/dev/null || true
    printf '\n' >> "$CONTEXT_FILE"
fi

# --- 3. Recent session logs --------------------------------------------------
PROJECT_ROOT=$(dirname "${CLAUDE_MD:-$PWD}")
SESSION_LOGS_DIR="$PROJECT_ROOT/CC-Session-Logs"

if [[ -d "$SESSION_LOGS_DIR" ]]; then
    mapfile -t RECENT_LOGS < <(ls -1t "$SESSION_LOGS_DIR"/*.md 2>/dev/null | head -3)
    if [[ ${#RECENT_LOGS[@]} -gt 0 ]]; then
        printf '=== RECENT SESSIONS (%d) ===\n' "${#RECENT_LOGS[@]}" >> "$CONTEXT_FILE"
        for logfile in "${RECENT_LOGS[@]}"; do
            printf '--- %s ---\n' "$(basename "$logfile")" >> "$CONTEXT_FILE"
            sed '/^## Raw Session Log/Q' "$logfile" 2>/dev/null | head -40 >> "$CONTEXT_FILE" || true
            printf '\n' >> "$CONTEXT_FILE"
        done
    fi
fi

# Nothing to inject — exit silently
[[ ! -s "$CONTEXT_FILE" ]] && exit 0

# --- 4. Emit JSON additionalContext (cap at 16k chars; mandates ~7k + ctx) ---
python3 - "$CONTEXT_FILE" <<'PYEOF'
import json, sys
try:
    with open(sys.argv[1], encoding='utf-8', errors='replace') as f:
        ctx = f.read(16000)  # cap to keep context budget reasonable
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": ctx
        }
    }))
except Exception:
    pass
PYEOF
