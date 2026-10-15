#!/usr/bin/env bash
# apply_settings_change.sh — helper that classifies a settings.json edit by its
# hot-reload status and tells the caller what (if anything) is needed to make
# the change take effect.
#
# Usage:
#   ./apply_settings_change.sh <path-to-settings.json> [--diff-against <previous>] [--apply]
#
# Modes:
#   default     analyze only — print classification + recommended action
#   --apply     also perform the recommended action automatically when safe
#               (e.g. invoke /reload-plugins via a heredoc-able marker file)
#
# It backs up the target settings.json before any modifications.
#
# Reference: AI-Tools/docs/HOT_RELOAD_PATTERNS.md

set -uo pipefail

SETTINGS_PATH="${1:-}"
shift || true
PREV_PATH=""
APPLY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --diff-against) PREV_PATH="$2"; shift 2;;
        --apply)        APPLY=1; shift;;
        -h|--help)      head -30 "$0" | sed -n 's/^# //p'; exit 0;;
        *)              echo "unknown arg: $1" >&2; exit 2;;
    esac
done

if [[ -z "$SETTINGS_PATH" ]] || [[ ! -f "$SETTINGS_PATH" ]]; then
    echo "ERROR: provide a path to settings.json (not found: $SETTINGS_PATH)" >&2
    exit 2
fi

TS=$(date '+%Y%m%d_%H%M%S')
BACKUP_DIR="$(dirname "$SETTINGS_PATH")/.settings_backups"
mkdir -p "$BACKUP_DIR"
BACKUP_PATH="$BACKUP_DIR/settings.json.$TS.bak"
cp "$SETTINGS_PATH" "$BACKUP_PATH"
echo "[apply_settings_change] backup: $BACKUP_PATH"

# If no previous snapshot supplied, just print structural summary and the
# canonical "what hot-reloads" table for the caller's reference.

python3 - "$SETTINGS_PATH" "$PREV_PATH" "$APPLY" <<'PYSCRIPT'
import json, sys, os, shutil
from pathlib import Path

target_p = Path(sys.argv[1])
prev_p = Path(sys.argv[2]) if sys.argv[2] else None
apply = sys.argv[3] == "1"

def load(p):
    with open(p) as f:
        return json.load(f)

cur = load(target_p)

def keyset(d, *path):
    cur = d
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
        if cur is None:
            return None
    return cur

print("=" * 70)
print("CURRENT SETTINGS STRUCTURE")
print("=" * 70)
print(f"  top-level keys: {sorted(cur.keys())}")
print(f"  hook events:    {sorted(cur.get('hooks', {}).keys())}")
print(f"  mcpServers:     {sorted((cur.get('mcpServers') or {}).keys())}")
print(f"  permissions:    {sorted((cur.get('permissions') or {}).keys())}")
print(f"  plugins:        {len(cur.get('enabledPlugins', []) or [])} enabled")
print()

# Detect what diff areas exist if prev_p
hot_reloads = []
needs_action = []

if prev_p and prev_p.exists():
    prev = load(prev_p)

    # Hooks comparison
    def hook_signatures(d):
        sigs = []
        for evt, entries in (d.get("hooks", {}) or {}).items():
            for entry in entries or []:
                if isinstance(entry, dict):
                    for h in entry.get("hooks", []) or []:
                        sigs.append((evt, entry.get("matcher", ""), h.get("type", ""), h.get("command", "")))
        return set(sigs)

    cur_hooks = hook_signatures(cur)
    prev_hooks = hook_signatures(prev)
    added_hooks = cur_hooks - prev_hooks
    removed_hooks = prev_hooks - cur_hooks
    if added_hooks or removed_hooks:
        needs_action.append(
            f"settings.json HOOK REGISTRATION changed (+{len(added_hooks)}/-{len(removed_hooks)})\n"
            f"  → /reload-plugins (if hooks come from a plugin) OR /reload (SIGHUP) OR sub-agent spawn"
        )

    # MCP comparison
    cur_mcp = set((cur.get("mcpServers") or {}).keys())
    prev_mcp = set((prev.get("mcpServers") or {}).keys())
    if cur_mcp != prev_mcp:
        needs_action.append(
            f"mcpServers list changed (+{cur_mcp - prev_mcp} / -{prev_mcp - cur_mcp})\n"
            f"  → /reload-plugins (refreshes MCP connections in-process) OR /mcp to retry. New stdio MCPs need fresh process."
        )

    # Permissions comparison
    cur_allow = set((cur.get("permissions") or {}).get("allow", []))
    cur_deny  = set((cur.get("permissions") or {}).get("deny", []))
    prev_allow = set((prev.get("permissions") or {}).get("allow", []))
    prev_deny  = set((prev.get("permissions") or {}).get("deny", []))
    if cur_allow != prev_allow or cur_deny != prev_deny:
        needs_action.append(
            f"permissions changed (allow +{len(cur_allow - prev_allow)}/-{len(prev_allow - cur_allow)}, "
            f"deny +{len(cur_deny - prev_deny)}/-{len(prev_deny - cur_deny)})\n"
            f"  → snapshotted at session start. /reload (SIGHUP) or sub-agent spawn. In bypassPermissions, deny is mostly inert; sensitive-path-block hook is the real enforcer."
        )

    # env comparison
    cur_env = cur.get("env") or {}
    prev_env = prev.get("env") or {}
    if cur_env != prev_env:
        needs_action.append(
            f"env vars changed\n"
            f"  → set at process start. Restart needed for current session. Sub-agents can override per-spawn."
        )

    # statusLine, theme, alwaysThinkingEnabled (these reload live for some)
    for live_key in ("theme", "statusLine", "keybindings"):
        if cur.get(live_key) != prev.get(live_key):
            hot_reloads.append(f"{live_key} changed — hot-reloads (no action needed)")

else:
    print("(no previous snapshot supplied — skipping diff)")
    print()

print("=" * 70)
print("HOT-RELOAD CLASSIFICATION")
print("=" * 70)
if hot_reloads:
    print("HOT-RELOADS (no action needed):")
    for h in hot_reloads:
        print(f"  + {h}")
    print()
if needs_action:
    print("REQUIRES ACTION (NOT automatic):")
    for n in needs_action:
        print(f"  ! {n}")
    print()
    print("RECOMMENDED ORDER:")
    print("  1. Try /reload-plugins from inside Claude  (covers plugin hooks/MCP/skills/agents)")
    print("  2. If that doesn't suffice, run /reload (SIGHUP) — costs ~1s, --continue preserves session")
    print("  3. Universal fallback: spawn fresh sub-agent for the work needing the change")
    print()
elif not hot_reloads:
    print("No diff or no detectable changes between snapshots.")
    print()

print("=" * 70)
print("REFERENCE — what hot-reloads vs what doesn't")
print("=" * 70)
print("""
HOT-RELOADS (just save the file):
  - Bash hook script content (.sh under hooks/)  [VERIFIED 2026-05-17]
  - CLAUDE.md, MEMORY.md, memory notes
  - Python scripts invoked via Bash tool
  - sitecustomize.py (next Python subprocess)
  - theme, keybindings
  - MCP server list_changed notifications

REQUIRES /reload-plugins:
  - Skill body edits
  - Agent definition edits (frontmatter, tools, model)
  - Plugin-shipped hook bash content (live anyway, but reload-plugins is cleaner)
  - Plugin-shipped MCP server config

REQUIRES /reload (SIGHUP) OR restart:
  - settings.json hook registrations (new/removed/matcher change)
  - settings.json mcpServers list (new entries)
  - settings.json permissions changes
  - settings.json env section
  - Launcher env vars (bin/claude-gdrive exports)
  - New slash command names (Issue #37862)
""")

if apply:
    print("=" * 70)
    print("APPLY MODE — leaving a marker for the caller")
    print("=" * 70)
    marker = Path("/tmp/claude-native-features-2026-05-17/apply_settings_change.marker")
    marker.parent.mkdir(parents=True, exist_ok=True)
    actions = []
    if needs_action:
        actions = ["/reload-plugins", "/reload"]
    marker.write_text("\n".join(actions))
    print(f"Marker written: {marker}")
    print(f"Recommended slash commands: {' then '.join(actions) if actions else '(none)'}")
PYSCRIPT

rc=$?
if [[ $rc -ne 0 ]]; then
    echo "[apply_settings_change] python helper failed rc=$rc — restoring backup" >&2
    cp "$BACKUP_PATH" "$SETTINGS_PATH"
    exit $rc
fi

echo
echo "[apply_settings_change] done. backup retained at: $BACKUP_PATH"
echo "[apply_settings_change] full reference: AI-Tools/docs/HOT_RELOAD_PATTERNS.md"
