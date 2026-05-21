# Claude Code Hot-Reload Patterns

**Last updated:** 2026-05-17
**Status:** Empirically tested against the running session (PID 266) in this workspace.
**Purpose:** Stop disrupting mid-session work by restarting Claude Code every time we patch a plugin / MCP server / setting / hook. Document exactly what hot-reloads, what does not, and the workarounds for everything that does not.

---

## TL;DR — what we discovered

| Change type | Hot-reload? | Mechanism | Workaround if no |
|---|---|---|---|
| **Bash hook script body** (`.sh` file content) | **YES** | Hook scripts are exec'd fresh each invocation; bash re-reads source every fire | None needed — just save the file. **VERIFIED in this session.** |
| **CLAUDE.md** (user/project memory) | **YES** | Re-read on next prompt | None needed |
| **Keybindings, theme** | YES | Live | None needed |
| **`settings.json` hook *registration*** (matchers, new entries, new events) | **NO** | Snapshotted at session start for security | A) `/reload-plugins` for plugin-shipped hooks, B) `kill -HUP $PPID` + wrapper for full reload, C) spawn fresh sub-agent which reads current settings, D) full restart |
| **`settings.json` permissions.allow / .deny** | NO (mostly) | Snapshotted at session start | Same workarounds. Note: in `bypassPermissions` mode the deny list is mostly ignored anyway; the `sensitive-path-block` hook re-implements it |
| **`settings.json` env vars** | NO | Process env captured at startup | Restart or spawn sub-agent (sub-agent re-reads settings & re-launches with new env) |
| **MCP server config (`mcpServers` section)** | PARTIAL | New MCP **server registrations** require restart. **Tool/prompt/resource lists** within an *already-connected* server auto-refresh when the server sends `list_changed` notification | `/mcp` to view, custom `/reload` skill, or spawn sub-agent for the new MCP |
| **MCP server crashed/disconnected (HTTP/SSE)** | YES (auto) | Exponential backoff, up to 5 retries (1s→2s→4s→8s→16s); marked `failed` after | `/mcp` to manually retry |
| **MCP server crashed (stdio)** | NO | Not auto-reconnected | `/mcp` to manually retry; sub-agent spawn relaunches the server fresh |
| **Plugin install/uninstall (`/plugin install`)** | PARTIAL | `/reload-plugins` re-runs skill/agent/hook/MCP registration in-process, BUT does **not** rebuild slash-command parser index (Issue #37862) | Full restart for new slash command names; `/reload-plugins` for everything else |
| **Plugin code (Python/JS source in a plugin dir)** | DEPENDS | Subprocess code (stdio MCP, hook bash) re-reads on next invocation. In-process plugin Python — needs `/reload-plugins` | `/reload-plugins`, then verify |
| **Sub-agent definition (`agents/*.md` frontmatter, tools, model)** | NO | Captured at session start | `/reload-plugins` partially; spawn-fresh-agent picks up new agent file via `mcp__plugin_fallback-agent_fallback__Task` which forks a fresh `claude -p` process |
| **Skill definition (`skills/*/SKILL.md`)** | YES via `/reload-plugins` | Skills re-injected into context | `/reload-plugins` then re-invoke skill |
| **Slash command (`commands/*.md`)** | DEPENDS | Newly created commands NOT recognized by parser without restart (Issue #37862). Edits to existing command bodies — re-read each invocation | Restart needed for *new* command names. Edits to existing commands — live |
| **Launcher env vars** (`bin/claude-gdrive` exports) | NO | Set before `claude` exec — only new sessions | Restart launcher |
| **`sitecustomize.py` (venv site-packages patch)** | YES per-subprocess | Re-imported on every Python subprocess start. So any new Python subprocess invoked from this session loads the latest sitecustomize | None — already live |

---

## Empirical test results (this session, 2026-05-17 20:20-20:22)

### Test 1 — adding a new hook registration to `settings.json`
- Created `hooks/hotreload-probe/probe.sh` (chmod +x).
- Appended `PostToolUse:Bash → probe.sh` entry to `settings.json`.
- Fired a Bash tool (`echo TRIGGER POST...`).
- **Result:** `/tmp/claude-native-features-2026-05-17/hook_probe.log` was NOT created. The newly-registered hook did NOT fire.
- **Conclusion:** Settings.json hook registrations are snapshotted at session start. **NO hot reload for new registrations.**

### Test 2 — mutating the body of an ALREADY-REGISTERED hook
- `hooks/cache-control-injector.sh` is registered as `PreToolUse:Task|mcp__plugin_fallback-agent_fallback__Task`.
- Edited the script body to append a side-channel marker to `/tmp/claude-native-features-2026-05-17/hot_reload_marker.log`.
- Spawned a Task sub-agent (a trivial Haiku echo).
- **Result:** `hot_reload_marker.log` contains `MARKER_V2 20:21:18`.
- **Conclusion:** **Bash hook script content hot-reloads** — the script is re-read by bash on every invocation. We can patch hook logic without restart.

### Test 3 — `bypassPermissions` masking permission allow changes
- `bypassPermissions` mode makes deny rules in settings.json mostly inert; `permissions.allow` becomes uninformative because everything is auto-allowed.
- This test was inconclusive in `bypassPermissions` mode — we couldn't tell whether the new allow entry was being read or whether bypass was just letting the command through.
- **Recommendation:** Trust GitHub issue #33829 (closed-as-duplicate) and #30737 — permission changes in `settings.local.json` do NOT hot-reload in normal mode.

---

## What ACTUALLY hot-reloads — exploit these

### 1. Bash hook script content
**This is the big one.** Every PreToolUse / PostToolUse / UserPromptSubmit / SessionStart / PreCompact / SubagentStart hook that runs a bash script — you can edit the `.sh` file mid-session and the next tool call will see the new content.

**Implication:** All our token-reduction / safety / observability logic lives in bash scripts under `home/.claude/hooks/` — these can be tweaked freely without restart. Examples:
- `auto_llmlingua_compress.sh` — adjust token threshold (currently 2000), change compression target ratio, change passthrough rules → live.
- `cache-control-injector.sh` — adjust observability fields, add new metrics → live.
- `scan-secrets.sh` — add new regex patterns → live (next prompt will be scanned with new patterns).
- `sensitive-path-block.sh` — add new path patterns → live.
- `model-routing-check/check.sh` — adjust the keyword→model mapping → live.

**Caveat:** You CANNOT change the matcher (which tool the hook fires on), the event (PreToolUse vs PostToolUse), the `type`, or add a *new* hook by editing `settings.json`. Only the script body itself is live-mutable.

### 2. CLAUDE.md and memory files
Edit `My Drive/AI-Tools/CLAUDE.md` or any `MEMORY.md` and the next prompt picks it up (re-read every turn).

### 3. Sub-agent execution context
When you spawn a sub-agent via `mcp__plugin_fallback-agent_fallback__Task`, the spawned process is a **fresh `claude -p`** that re-reads `settings.json`, re-launches MCP servers, reloads plugins. **This is the universal workaround:** if you can't hot-reload in the current session, spawn a sub-agent that needs the change and it will load the current state.

### 4. Python scripts invoked from Bash
Anything under `AI-Tools/scripts/*.py` invoked via the Bash tool — Python imports fresh every subprocess. Edit `llmlingua_compress.py`, `mem0_helper.py`, `claude_native_features.py` etc. freely.

### 5. `sitecustomize.py` (venv site-packages)
Every Python subprocess launched from this session re-imports `sitecustomize.py`. The `auto_cloud_dispatcher` monkey-patch installs on each subprocess start. You can edit the patch mid-session.

### 6. MCP `list_changed` notifications
Per docs: "When an MCP server sends a list_changed notification, Claude Code automatically refreshes the available capabilities from that server." So if you're authoring an MCP server, emit `notifications/tools/list_changed` after adding tools — they appear without restart.

---

## What does NOT hot-reload — workarounds

### A. Adding a new hook to `settings.json` mid-session
**Workaround 1 — defer to a future session:** patch `settings.json`, knowing it activates on next start.
**Workaround 2 — spawn-fresh sub-agent:** if the hook is needed for a specific task, spawn a sub-agent via `mcp__plugin_fallback-agent_fallback__Task`. The sub-agent loads `settings.json` from scratch, so the new hook fires in its context.
**Workaround 3 — full reload:** use the `/reload` skill described below (sends `SIGHUP` to claude, restarts with `claude -c` to preserve session ID).

### B. Adding / changing / removing an MCP server
**Workaround 1 — `/mcp` slash command:** shows current server status, may allow retry of failed servers.
**Workaround 2 — spawn-fresh sub-agent:** if you need a new MCP for a specific task, brief a sub-agent that needs it — the sub-agent's fresh `claude` process will register the new MCP.
**Workaround 3 — `/reload` skill (SIGHUP):** for adding to the main session.

### C. Updating a plugin's Python/JS source (not bash)
**Workaround 1 — `/reload-plugins`:** This is the canonical command. Re-registers skills/agents/hooks/MCP. But note Issue #37862: does NOT rebuild slash command parser index, so newly-added command names still won't autocomplete.
**Workaround 2 — sub-agent spawn:** fresh subprocess picks up new plugin code.

### D. Adding a new slash command (new name)
**Workaround:** Full restart unfortunately required (Issue #37862). `/reload-plugins` will load the skill content but the `/` parser won't recognize the new name.

### E. Changing launcher env vars (bin/claude-gdrive exports)
**No workaround for current session** — env is set before `claude` exec. Either restart, or use sub-agent spawn which has its own env via the Task tool parameters.

### F. Changing settings.json permissions (allow/deny)
**Workaround 1 — `/reload` skill:** full SIGHUP-style reload.
**Workaround 2 — bypass mode:** running in `bypassPermissions` (our default) makes most permission changes moot; the `sensitive-path-block` PreToolUse hook is the real enforcer and IT hot-reloads (bash hook content).

---

## Universal escape hatch: the `/reload` skill (SIGHUP pattern)

Adopted from Anthony Panozzo's pattern (panozzaj.com/blog/2026/02/07).

### Install
Save as `~/.claude/commands/reload.md` (which in our launcher = `home/.claude/commands/reload.md`):

```markdown
# Reload Claude Code (kill parent with SIGHUP — wrapper restarts with --continue)

!`kill -HUP $PPID`
```

The `!` prefix forces immediate execution without LLM intervention.

### Launcher wrapper (bash function)
Already partially aligned with our launcher pattern. Add to `bin/claude-gdrive` or a separate wrapper function:

```bash
function CL_RELOADABLE {
  local continue_flag=""
  local rc
  while true; do
    "$HOME/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/bin/claude-gdrive" \
      $continue_flag "$@"
    rc=$?
    [ $rc -eq 129 ] || return $rc
    echo "[CL_RELOADABLE] reload requested — restarting with --continue"
    sleep 0.5
    continue_flag="--continue"
  done
}
```

Then user types `/reload` from within Claude → claude exits with code 129 (128 + SIGHUP signal 1) → wrapper detects, restarts with `--continue` (preserves session ID, continues conversation), re-reads everything.

**Cost:** ~1 second wall-clock per reload. Conversation context is preserved by `--continue`. Background tasks die.

### When NOT to use
- Background TaskCreate jobs running — they're terminated.
- Mid-stream tool call — partial output lost.
- If you only changed a bash hook script — don't bother, it's already live.

---

## `/reload-plugins` — native command (use this first)

Built-in. Reloads skills + agents + hooks + MCP servers in-process. **Does NOT restart the session.** Faster than SIGHUP because no process recycle.

```
/reload-plugins
```

**Known limitations (per docs + community):**
1. Does **not** rebuild the slash-command parser index — new command *names* added via plugin install still need restart.
2. Output ("Reloaded: N plugins · K skills · M agents") is misleading — implies more was reloaded than actually was.
3. Desktop App does not support this command (Issue #52967) — must Cmd+Q. We're on CLI so this doesn't affect us.

**Use it for:**
- After `/plugin install <name>` — to activate.
- After editing a `skills/*/SKILL.md` body.
- After editing an `agents/*.md` body.
- After adding a new MCP server to `mcpServers` section (in conjunction with `/mcp`).

**Don't bother for:**
- Bash hook script edits (already live).
- CLAUDE.md edits (already live).

---

## Decision tree — "I just changed X, do I need to restart?"

```
1. Did you edit a .sh file under home/.claude/hooks/ (just the script body)?
   → NO RESTART NEEDED. Next hook fire picks it up.

2. Did you edit CLAUDE.md, MEMORY.md, or a memory note?
   → NO RESTART NEEDED. Next prompt picks it up.

3. Did you edit a Python script under AI-Tools/scripts/ that's invoked via Bash?
   → NO RESTART NEEDED. Next subprocess re-imports.

4. Did you change ONLY the body of an existing skill (.md) or agent (.md)?
   → Run /reload-plugins. NO restart needed.

5. Did you add a NEW skill, agent, or MCP server to a plugin?
   → Run /reload-plugins. May need restart for new slash command names.

6. Did you change settings.json hook registrations (matchers / new entries / events)?
   → Either:
     (a) Live without it this session — spawn fresh sub-agent for the work that needs it.
     (b) /reload via SIGHUP if installed — costs ~1 sec, --continue preserves session.

7. Did you change settings.json mcpServers section?
   → Same as #6. Sub-agent spawn or /reload.

8. Did you change settings.json permissions?
   → Only matters in non-bypass mode. /reload to apply.

9. Did you change bin/claude-gdrive env exports?
   → Must restart launcher. Sub-agents spawned with Task() can override env per-spawn.

10. Did you install a new MCP server with `claude mcp add`?
    → Restart needed (it's written to ~/.claude.json but not loaded). Use /reload skill.

11. Did you change slash command parser (new / name)?
    → Restart needed (Issue #37862, no workaround short of /reload).
```

---

## Workspace-specific notes

### Our session as of 2026-05-17
- 5 hook events registered in `settings.json`: SessionStart, PreCompact, PreToolUse, UserPromptSubmit, PostToolUse.
- 47 hook-related entries total — bash scripts under `home/.claude/hooks/`.
- All bash hook *script* edits live. Settings.json hook *registration* edits dormant until restart.

### Our launcher
`bin/claude-gdrive` exports 11 feature flags (DAEMON/BG_SESSIONS/BRIDGE_MODE/ULTRAPLAN/TEMPLATES=1; KAIROS/COORDINATOR_MODE/BUDDY/VOICE_MODE/WEB_BROWSER=0) and appends `--max-budget-usd 50`. These env vars and CLI args only apply on new sessions. Sub-agents inherit only the `CLAUDE_*` env vars that propagate naturally.

### Our `auto_cloud_dispatcher`
Patched into venv `sitecustomize.py`. Re-imported on every Python subprocess. Edits to `auto_cloud_dispatcher.py` go live for any subprocess started after the edit — including subprocesses started by hook scripts and sub-agents.

### Our PreCompact / SessionStart hooks
`auto_preserve.sh` (PreCompact) and `auto_resume.sh` (SessionStart) — bash content is hot-reloadable. The events themselves only fire on session lifecycle boundaries.

---

## References

- Building a /reload Command for Claude Code (Anthony Panozzo, 2026-02-07): https://www.panozzaj.com/blog/2026/02/07/building-a-reload-command-for-claude-code/
- Claude Code Plugins reference: https://code.claude.com/docs/en/plugins-reference
- Claude Code MCP docs: https://code.claude.com/docs/en/mcp
- Claude Code Hooks docs: https://code.claude.com/docs/en/hooks-guide
- Issue #5513 — /reloadSettings feature request
- Issue #17127 — /reload command without restart
- Issue #33829 — Hot-reload permissions from settings.local.json (closed as dup)
- Issue #29822 — /reload command request
- Issue #30737 — Reload permissions in running session
- Issue #15858 — RFC: Config Hot-Reload
- Issue #6605 — Hot-reload SDK config
- Issue #42251 — Hot reload env vars and runtime code
- Issue #46426 — Hot-reload MCP without session restart
- Issue #40059 — Reload MCP without restart
- Issue #36847 — SIGHUP handler for config reload
- Issue #53125 — Remote control: MCP reload
- Issue #52967 — Desktop /reload-plugins parity
- Issue #37862 — /reload-plugins doesn't rebuild slash-command index
- Issue #28685 — /restart-session command
- Issue #11632 — /reload-commands feature request
- Issue #18174 — Plugin hot-reload without session restart
- Issue #24057 — MCP/hooks/plugins auto-reload on config change
- mcp-hot-reload (data-goblin): https://github.com/data-goblin/claude-code-mcp-reload
- mcpmon: https://mcpservers.org/servers/neilopet/mcp-server-hmr

---

## Zero-Restart Strategies (2026-05-17 enhanced)

User mandate: **NO MORE RESTARTS**. This section consolidates the 4-tier hierarchy for avoiding restarts while still picking up config changes. Full research deliverable at `AI-Tools/research/zero_restart_internet_2026-05-17.md`.

### 4-Tier Zero-Restart Hierarchy

Apply in order; use the first tier that works for your blocker.

**TIER 1 — No reload needed (auto-detected)**
- `settings.json` edits to *existing* hook entries → hot-reloaded since v1.0.90; saves on next tool call.
- `CLAUDE.md` edits → hot-reloaded on next prompt.
- Bash subprocess env vars → `export VAR=val` inside a Bash tool call (subprocess inherits).
- May 2026 update fixed symlinked-settings hot-reload bug (per Releasebot changelog).

**TIER 2 — `/reload-plugins` (~50ms, no session interruption)**
Covers per [code.claude.com/docs/en/plugins-reference](https://code.claude.com/docs/en/plugins-reference):
- Plugin skills, agents
- Plugin-shipped hooks (PreToolUse / PostToolUse / etc.)
- Plugin-shipped MCP servers (re-pointed to new path)
- Plugin-shipped LSP servers
Excluded: plugin monitors (need full restart); brand-new slash-command parser index (Issue #37862 — skill loads into context but `/` autocomplete doesn't show it).

**TIER 3 — Fresh sub-agent spawn (~1-3s, parent context preserved)**
- Use `mcp__plugin_fallback-agent_fallback__Task`.
- Child `claude -p` process forks fresh through launcher → inherits CURRENT launcher env + reads CURRENT settings.json/MCP config.
- Use for: NEW launcher env exports, NEW settings.json hook blocks, NEW MCP server entries.
- Limitation: only the CHILD sees the new config; parent orchestrator does not. Good when the work can be slice-delegated.

**TIER 4 — SIGHUP wrapper restart (~1s, session preserved via `claude -c`)**
- Universal fallback covering all 4 blockers, including launcher env vars cached at process startup.
- Install `/reload` skill at `~/.claude/commands/reload.md`:
  ```
  # Reload Claude Code (restart Claude)
  !`kill -HUP $PPID`
  ```
- Wrap launcher with shell function detecting exit code 129 and looping with `-c` (resume):
  ```bash
  function CL {
    local continue_flag=""
    local restart_msg=""
    local rc
    while true; do
      "$AI_TOOLS_ROOT/bin/claude-gdrive" $continue_flag "$@" $restart_msg
      rc=$?
      [ $rc -eq 129 ] || return $rc
      echo "Reloading Claude Code..."
      sleep 0.5
      continue_flag="-c"
      restart_msg="restarted"
    done
  }
  ```
- Source: [Panozzo 2026-02-07](https://www.panozzaj.com/blog/2026/02/07/building-a-reload-command-for-claude-code/)
- Conversation context preserved; ~1s gap; user sees "Reloading…" line.

### Per-Blocker Decision Matrix

| Blocker | Best workaround | Fallback |
|---|---|---|
| settings.json edit (existing hook block) | Tier 1 — save and trigger any tool | Tier 4 |
| settings.json NEW hook block | Tier 3 (sub-agent spawn) | Tier 4 |
| Plugin code / skill / agent change | Tier 2 — `/reload-plugins` | Tier 4 |
| Plugin MCP server change | Tier 2 — `/reload-plugins` | Tier 4 |
| Standalone MCP server change | Tier 3 (sub-agent) + `mcpmon` proxy on the server side | Tier 4 |
| Launcher env export added | Tier 3 (sub-agent inherits) | Tier 4 (launcher restart) |
| Plugin monitor change | — (hard limit) | Tier 4 |
| Brand-new slash command (new install) | — (Issue #37862 limit) | Tier 4 |

### Hard limits (no zero-restart possible)

1. **OS env vars cached at startup** — Unix process env is immutable from outside; only `lldb attach + setenv()` works, and only for runtime `getenv()` calls (cached vars unaffected). SIGHUP wrapper (Tier 4) is the floor.
2. **Plugin monitors** — explicit "session restart required" per official docs.
3. **Brand-new slash-command parser entries** — Issue #37862, `/reload-plugins` doesn't rebuild the `/` parser index.
4. **Claude Desktop app plugin updates** — no `/reload-plugins` parity (Issue #52967); only Cmd+Q + reopen.

### Bottom-line workflow rules

1. **Default for ANY config edit:** Save → continue working. Hot-reload likely handles it.
2. **If it doesn't take effect:** Run `/reload-plugins`. Covers plugin/MCP/hook plugin path.
3. **If still not picked up AND you can delegate:** Spawn sub-agent via `mcp__plugin_fallback-agent_fallback__Task`. Child sees fresh config.
4. **If orchestrator itself needs the new config:** Type `/reload` (SIGHUP). ~1s restart, no context loss.
5. **Never run `claude` without the SIGHUP wrapper** — restart cost drops from "lose session + 30s" to "~1s, session preserved".

### Additional sources (2026-05-17 research)
- Issue #6497 — Hot reload agents/slash commands
- Issue #7841 — Signal sending for background commands
- Issue #17975 — MCP tool caching/hot-reload
- Issue #20365 — Dynamic MCP server management
- Issue #22050 — Agent definitions hot-reload
- Issue #38707 — Auto-reload skills across sessions
- Issue #11632 — /reload-commands feature request
- Claude Code 2.1 fixes summary: https://paddo.dev/blog/claude-code-21-pain-points-addressed/
- Releasebot May 2026 changelog: https://releasebot.io/updates/anthropic/claude-code
- LLDB env injection technique: https://blog.merovius.de/posts/2013-10-11-inject-environment-variables-int/
- launchctl setenv semantics: https://ss64.com/mac/launchctl.html

---

## Zero-Restart Tools Installed (2026-05-17)

Three complementary tools wired in so the user never needs to type "restart Claude Code" again.

### Tool 1 — `/reload` Skill (Panozzo SIGHUP pattern)

- **Skill file:** `home/.claude/skills/reload/SKILL.md` (new — created 2026-05-17)
- **Slash command:** `home/.claude/commands/reload.md` (pre-existing, kept)
- **Mechanism:** `kill -HUP $PPID` → claude exits with rc=129 → launcher `claude-gdrive` while-loop relaunches with `--continue` (preserves session id + conversation)
- **Cost:** ~1s wall-clock, no token cost (Skill is `!`-prefix bash, never reaches model)
- **Source:** Panozzo blog 2026-02-07 (https://www.panozzaj.com/blog/2026/02/07/building-a-reload-command-for-claude-code/). The upstream `panozzaj/claude-skills` GitHub repo does NOT exist (404 verified 2026-05-17); we authored the equivalent Skill ourselves with full attribution.

### Tool 2 — yacb2/claude-restart (Zero-Token Restart)

- **Repo:** https://github.com/yacb2/claude-restart (MIT, cloned to `/tmp/claude-restart-install`)
- **Scripts installed at `home/.claude/scripts/`:**
  - `capture-session-id.sh` — SessionStart hook, writes session_id to per-PID file
  - `restart-hook.sh` — UserPromptSubmit hook, intercepts the literal prompt "restart" before it reaches the model, creates a flag, sends SIGTERM to PPID. **Zero tokens consumed.**
  - `claude-wrapper.sh` — reference copy (not invoked; our launcher integrates the logic directly to avoid touching `.zshrc` which is deny-listed)
  - `LICENSE.claude-restart` — MIT attribution
- **Slash command:** `home/.claude/commands/restart.md` — fallback path that goes through the model (uses tokens), kept as documented backup
- **Hooks added to `ClaudeCode/config/settings.json`:**
  - `SessionStart` → `~/.claude/scripts/capture-session-id.sh` (timeout 5)
  - `UserPromptSubmit` → `~/.claude/scripts/restart-hook.sh` (timeout 5)
- **Launcher integration:** `bin/claude-gdrive` patched to (a) export `CLAUDE_RESTART_ID=$$` so the hooks scope per-launcher-instance, (b) check for `~/.claude/tmp/restart-flag-${CLAUDE_RESTART_ID}` after each claude exit and re-exec with `--resume <session_id>` if found.
- **Usage:** Type the literal prompt `restart` (no slash, lowercase). The UserPromptSubmit hook intercepts before model — zero tokens. `/restart` slash command is the fallback that DOES use tokens (because the slash parser routes through the model).

### Tool 3 — mcpmon (MCP Hot-Reload Proxy)

- **Repo:** https://github.com/neilopet/mcpmon (MIT, installed via npm)
- **Install path:** `ClaudeCode/npm-global/bin/mcpmon` (npm global prefix scoped to workspace)
- **Runtime dependency:** `bun` (installed at `ClaudeCode/npm-global/bin/bun` v1.3.14, also via npm)
- **Verified:** `mcpmon --help` returns full usage on 2026-05-17
- **Status:** Installed-and-ready. **Not yet wired into any MCP server** because the active `.claude.json` only references published packages (`mcp-memory-keeper` npm + `github` HTTP), and mcpmon is only useful for stdio MCP servers whose source code lives in the workspace and is being actively edited.
- **Future wiring (when a local MCP server is added):**
  ```json
  "mcpServers": {
    "your-local-server": {
      "type": "stdio",
      "command": "mcpmon",
      "args": ["-w", "/path/to/server/src", "-e", "py,ts", "--", "python", "/path/to/server/src/server.py"]
    }
  }
  ```
  mcpmon watches the source directory and sends `notifications/tools/list_changed` after each restart — Claude Code auto-refreshes its tool cache without restart.

### Verification table

| Tool | Install status | Smoke test | Mechanism |
|---|---|---|---|
| `/reload` Skill | INSTALLED (`home/.claude/skills/reload/SKILL.md`) | Skill file present + launcher rc=129 loop confirmed in `bin/claude-gdrive` | SIGHUP → launcher `--continue` |
| yacb2/claude-restart | INSTALLED (scripts + hooks + slash command + launcher patch) | Hook scripts present + executable + jq-valid settings.json entries | UserPromptSubmit intercept → restart-flag → launcher `--resume` |
| mcpmon | INSTALLED (CLI present, `mcpmon --help` returns usage) | `mcpmon --help` → full help text printed | stdio proxy + `notifications/tools/list_changed` |

### Combined effect — the 3-tool hierarchy

| Need | Tool to use | Token cost | Wall-clock |
|---|---|---|---|
| Pick up new MCP server source code | mcpmon proxy (passive — auto on file change) | 0 | <1s |
| Pick up settings.json hook block edit | save (no action) | 0 | 0 |
| Pick up plugin component change | `/reload-plugins` | small | ~50ms |
| Pick up new MCP server / new hook block / launcher env / new slash command (orchestrator-side) | type `restart` (yacb2 — zero-token) OR `/reload` Skill (Panozzo — zero-token) | 0 | ~1s |
| Last resort | `/restart` slash command (yacb2 fallback) | ~N input tokens (full conversation re-read) | ~1s |

### Backups

All pre-install configs backed up to `AI-Tools/backups/zero-restart-tools-2026-05-17/`:
- `settings.json.bak`
- `.claude.json.bak`
- `claude-gdrive.bak`
