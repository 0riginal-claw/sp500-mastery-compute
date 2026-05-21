# Operating Manual — 2026-05-20

**Synthesized from:** 6 forensic docs + 4 log-reread slices in `logs/auto_solve/`
**Source authority:** crash_forensics_2026-05-20/{chunk1-4, recovery_guardrail_design, wholelog_patterns}.md + log_reread_slice{1,2,3,4}_2026-05-20.md
**Crash anchor:** Mac kernel-watchdog panic 2026-05-20 16:37:11 CDT (VM compressor 100% → cwd nuked by Drive FUSE → 170 shell-quote failures → hook cascade → reboot)
**Status:** governing rules for the post-crash recovery session

---

## 0. Pre-crash vs post-crash diff (what changed)

| Surface | Pre-crash (2026-05-19) | Post-crash (2026-05-20) | Reason |
|---|---|---|---|
| Mandate enforcement | Soft (CLAUDE.md text only) | 10 PreToolUse hard blockers + stateful counters | 144 §3 violations in 3-day audit |
| Autosolve hook | Advisory stderr | Stateful `autosolve_pending.jsonl` + 5-turn deadline + 3-distinct-role enforce | Helpers ignored stderr warnings |
| Fan-out tracker | None | 5/10/15-min escalating + 20-min KILL | Wave A's 33-min/83-tool solo run |
| Hook script location | Drive FUSE | **MUST move to `~/.zg/hooks/` (local SSD)** — DESIGNED, not yet shipped | 170× `/bin/sh: ".../My: No such file"` cascade |
| OpenClaw routing | Optional | Hard block — non-Alpaca research MUST route via `bin/openclaw-gdrive` agent | User mandate 2026-05-18 |
| Model routing | Hand-picked | `# model_reason:` REQUIRED in every spawn prompt | Drift toward opus-by-default |
| Sub-agent visibility | Native Agent default | Native Agent for 95%; plugin path only for MCP-needing grandchildren | UI widget needs native path |
| Cloud-routing | Soft default | 85-90% MUST go Modal/gh_actions; Mac cap = load 12 | Mac crash at load 14-23 |
| Session resume | Best-effort | `session_resume_guardrail` DESIGNED (2-store local+Drive, 60s checkpoints) — NOT YET INSTALLED | Compaction + crash lost in-flight helpers |
| Keychain default | "login" persisted | `security default-keychain` reports "not found" on 15.7.7 — cosmetic only; dump works | macOS Sequoia CLI quirk |

---

## 1. Cadence

### 1.1 Per-turn (every assistant response)

1. Read inherited mandates (`§3 FAN-OUT`, `§5a CLOUD-ROUTING`, `§8 AUTO-SOLVE`, MODEL-REASON, OPENCLAW ROUTING, LLMLINGUA, NEVER RESTART, AUTO-EXECUTE, NEVER PUNT, PROOF OF WORK, SAFETY).
2. Read inherited self-model (Gabriel-Self capability map + reflexions).
3. If any new capability/limitation discovered, append JSONL line to `state/gabriel_self/reflexions.jsonl`.
4. End with proof-of-work.

### 1.2 Per-task (5-minute tick)

- Estimated wall-clock check at T+0. If >5 min AND >2 logical slices → decompose + spawn N=2-6 helpers BEFORE doing any work.
- Audit slow children at T+5, +10, +15 min.
- Helper that has not fanned out by T+20 min → MUST self-terminate with partial report.

### 1.3 Per-session

- SessionStart hooks: auto_resume (loads CLAUDE.md + recent logs), auto-sandbox reminder, session-resume-bootstrap (once shipped).
- PreCompact: auto_preserve appends session learnings to CLAUDE.md.
- Stop: Stop-validate hooks shame next turn on miss (PostToolUse exit-2 is broken — bug #19009).
- Session-end: write `project_resume_<DATE>.md` to memory.

### 1.4 Per-day

- 09:00 PT: `claude-code-ultimate-guide:ccguide:daily` (Anthropic docs diff + version digest).
- 15:00 PT: dashboard/MISSION_PROGRESS.md auto-rollup (autonomous_mode daemon).
- Friday 06:30 PT: weekly full retrain (XSEC mega-job, gh_actions or Modal).

### 1.5 Per-incident

§8 triplet IMMEDIATELY: INTERNET + GITHUB + REPO-LOCAL solver. Stateful counter under `~/.claude/state/autosolve_pending.jsonl` registers issue and blocks tool use until 3 distinct roles spawn within 5 turns. Bypass per-call with `# autosolve_skip: <reason>`.

---

## 2. Routing decision tree

```
TASK INCOMING
│
├─ Alpaca live-trading code? ───────→ Claude (sonnet default, opus if hard reasoning)
│
├─ Mechanical (grep/list/format)? ──→ Haiku OR ollama_helper.py ($0) OR deepseek_helper.py ($0.000001)
│
├─ Research / synthesis / audit / 
│  bucket / WebSearch / survey? ────→ OpenClaw + DeepSeek (--local --model deepseek/deepseek-v4-flash)
│                                     bin/openclaw-gdrive agent --json --message "..."
│                                     (BLOCKED by openclaw-routing-block hook otherwise)
│
├─ Backtest sweep / ticker batch / 
│  parallel ML run? ────────────────→ cloud_dispatch.enqueue_job(ticker=X, strategy=Y, script=Z)
│                                     → Modal (preferred) OR gh_actions xsec.yml fallback
│                                     (AUTO_CLOUD_DISPATCH=0 FORBIDDEN unless smoke <60s)
│
├─ Multi-step / multi-target? ──────→ dispatching-parallel-agents skill OR native Agent fan-out N=2-6
│
├─ ML / quant / backtest /
│  hyperparam / signal-gen? ────────→ karpathy-guidelines skill FIRST (assumptions + verifiable criteria)
│                                     (BLOCKED by karpathy-guidelines-block hook otherwise)
│
├─ >3 file edits OR architecture? ──→ /ultraplan via claude_native_features.py
│
└─ Same transform on 3+ files? ─────→ /batch via claude_native_features.py
```

### 2a Sub-agent spawn type (95/5 rule)

- **95% native Agent with `subagent_type: general-purpose`** — UI widget, Ctrl+B background, MCP preserved.
- **5% mcp__plugin_fallback-agent_fallback__Task** — only when helper itself will spawn MCP-using grandchildren.

### 2b Mac safety cap

Load >12 → switch any new heavy work to cloud-route immediately. Don't launch new local heavy procs. Existing daemons (consumer, dispatcher, paper_trade, memory_save, watchdog) stay on Mac (lightweight).

---

## 3. §8 AUTO-SOLVE protocol

Triggered by: ANY error / blocker / failure / unknown library / HTTP 4xx-5xx / daemon crash / CAPTCHA / auth fail / timeout / version conflict.

**Mandatory spawn pattern** (within 5 turns of trigger):

```
INTERNET solver:    research best-practice from docs / SO / blogs
GITHUB solver:      search repos for known fixes (issues, PRs, commits)
REPO-LOCAL solver:  apply fix in this workspace, backup first, smoke test
```

Each spawn prompt MUST include one of:
- `# decomposition_plan: <slices>`
- `# scope_estimate_min: <N>`
- `# inline_justification: <one-line>`

PLUS:
- `# model_reason: <why this model>`

Bypass per-call: `# autosolve_skip: <reason>` in prompt header.
Escalate to user ONLY if all 3 solvers return "no known fix" OR fix touches money/messages/credentials/destructive actions.

---

## 4. Communication style

### 4.1 Comm-style commandments

1. **No emojis** in code or natural-language output unless user explicitly asks.
2. **No "I can do X if you want"** — execute silently within standing authorization.
3. **No "say go to start"** — auto-execute via tools/sub-agent/OpenClaw.
4. **No "restart Claude Code"** — use 4-tier hot-reload hierarchy.
5. **No punt to user** for routine research / MCP calls / paid OpenClaw / installs / sub-agent spawns. Only stop for destructive/irreversible/money/messages/credentials/external-account actions.
6. **No colon-before-tool-call** — "Let me read the file." NOT "Let me read the file:".
7. **No re-reading just-edited file** — Edit/Write would have errored if change failed.
8. **No relative paths in proof-of-work** — always absolute.

### 4.2 Proof-of-work fields (every task)

1. Commands run (verbatim)
2. Paths accessed (absolute)
3. Files created/changed (absolute + one-line note)
4. Backups created (timestamped paths)
5. OpenClaw usage (commands + outputs, or "none")
6. Sub-agent usage (name, task, model, reason, paths, files, status, or "none")
7. Final status (done / pending / Mac-local files w/ path+reason)

### 4.3 Output style routing

- Grandchild summaries → Explanatory
- Machine handoffs → Concise
- Trading code review → Explanatory

---

## 5. Active hooks (10 PreToolUse blockers + 5 lifecycle)

| Hook | Event | Bypass token |
|---|---|---|
| autosolve-require | PreToolUse | `# autosolve_skip: <reason>` |
| recursion-fanout-tracker | PreToolUse | `# fanout_skip: <reason>` |
| openclaw-routing-block | PreToolUse Task/Agent | `# justify_claude: <reason>` |
| karpathy-guidelines-block | PreToolUse (ML keywords) | `# karpathy_checked: <reason>` |
| model-reason-block | PreToolUse Task/Agent | (header `# model_reason:` required) |
| cloud-routing-block | PreToolUse Bash (heavy worker) | `AUTO_CLOUD_DISPATCH=0` + smoke <60s |
| spawn-prompt-compress | PreToolUse Task/Agent | (>2k tokens auto-compresses) |
| sensitive-path-block | PreToolUse Read/Write/Edit/Bash | (40-rule deny list) |
| scan-commit | PreToolUse Bash `git commit` | (no bypass for staged secrets) |
| destructive-delete + direct-push + pipe-to-shell + exfil + escalation | PreToolUse Bash | (no bypass; safety boundary) |

Lifecycle:
- auto_resume (SessionStart) — loads CLAUDE.md + recent session logs
- auto_preserve (PreCompact) — appends learnings to CLAUDE.md
- auto_llmlingua_compress (UserPromptSubmit, >800 tokens)
- cache-control-injector (PreToolUse Task, observe-only)
- auto-sandbox (SessionStart) — reminds to `/sandbox` for untrusted repos

---

## 6. Auto-execute (NEVER PUNT)

Within standing authorization (read/write/script/install/log/backup anywhere under `My Drive`), execute without confirmation. Examples that DO NOT require user permission:

- Spawning sub-agents (any count, any model)
- OpenClaw / DeepSeek calls (paid is OK, log in proof-of-work)
- MCP invocations
- Installing project dependencies
- Creating/moving project files
- Running scripts / tests / diagnostics
- Reading the entire `My Drive` tree
- Writing reports/logs/artifacts/backups
- launchctl bootstrap/bootout for `com.zg.*` daemons (with backup)

DO require user permission:
- Permanent file deletes (must move to timestamped backup folder first)
- Credential/wallet/financial-doc modification
- Money / messages / external account / paid plan / trading orders
- macOS system folder modification (`/System`, `/usr` outside `/usr/local`)
- `git push --force` to main
- Hook bypass flags (`--no-verify`, `--no-gpg-sign`)

---

## 7. Inherited mandates (one-line each)

- **§3 FAN-OUT**: >5min+2slices → 2-6 helpers, 20-min ceiling absolute.
- **§5a CLOUD-ROUTING**: 85-90% Modal/gh_actions; cloud_dispatch.enqueue_job; Mac cap load 12.
- **§8 AUTO-SOLVE**: error → 3 solvers (INTERNET+GITHUB+REPO-LOCAL) within 5 turns.
- **MODEL-REASON**: every spawn prompt has `# model_reason:`; default to cheaper if unjustified.
- **OPENCLAW**: non-Alpaca research → openclaw-gdrive + deepseek-v4-flash.
- **LLMLINGUA**: spawn prompt >800 tokens auto-compressed; >2k MUST compress.
- **NEVER RESTART**: 4-tier hot-reload (settings reload → /reload-plugins → sub-agent → SIGHUP).
- **AUTO-EXECUTE**: no "if you want"; execute silently within authorization.
- **NEVER PUNT**: routine research/MCP/installs/spawns proceed without permission.
- **PROOF OF WORK**: 7 fields at end of every task.
- **SAFETY**: no permanent deletes, no credential mods, no money/messages/external-accounts, no macOS system mods.

---

## 8. Known landmines (verified 2026-05-20)

| Landmine | Symptom | Workaround |
|---|---|---|
| Drive FUSE evicts cwd | `/bin/sh: ".../My: No such file"` cascade | Move all hooks to `~/.zg/hooks/` (local SSD) — DESIGNED, not yet shipped |
| launchd cannot exec Node from Drive | RSS frozen 88KB, script never runs | Execution copies on local disk; data on Drive is fine |
| PostToolUse exit-2 broken (#19009) | Hook fires after tool already ran | Use PreToolUse for blocking |
| Stop hook fires post-reply | Cannot retroactively gate just-delivered msg | Shame NEXT turn via violation reminder |
| `enablePromptCaching: false` on native Task (#29966) | Cache disabled for sub-agent spawns | Caching only on `claude -p` router calls (~36-39% theoretical) |
| `security default-keychain` "not found" on 15.7.7 | Cosmetic only — keychain functional | `security dump-keychain` works; ignore the "not found" message |
| Modal workspace spend cap | Not auto-reset at billing cycle | Manual dashboard action only |
| GH Actions billing block | `account payments have failed or your spending limit needs to be increased` | User action required at billing.github.com — agent cannot fix |
| `+suffix` aliases NOT canonical | Provider treats `a+x@yahoo.com` as new | Per-provider passwords; HF requires uppercase |
| patchright + macOS DNS | `--no-sandbox --disable-features=IsolateOrigins` breaks DNS | Drop both; keep only `--disable-blink-features=AutomationControlled` |
| `add_init_script` triggers ERR_NAME_NOT_RESOLVED | All hosts fail | `page.evaluate(script)` AFTER first nav instead |

---

## 9. Diff: pre-crash session vs post-crash session

**Pre-crash (last 24h before 16:37):**
- Heavy load (1:07 → 55.34, 13:37 → 9.91, 16:07 → 14.86)
- Drive FUSE intermittent (170 cascade failures in last 600 lines of log)
- Mission overseer LIVE (pid 71162) but ignored its own 21.49 load warnings
- 22 buckets / 7 task2 categories / 3 OpenClaw deep-dives all firing concurrently
- 6 stub adapters in 11 cloud-dispatch adapters (oracle_a1, gcp_ssh, aws_ssh, render_api, railway_api, fly_api pass-through unimplemented)
- 30 docstring-only stubs rejected by feature_auto_promote
- 429 trading repos cloned but unwired
- Modal workspace spend cap hit
- xsec gh_actions fallback DESIGNED + tested (run 26141858086 OK, then later runs cancelled/failed/billing-blocked)

**Post-crash (this session):**
- Load 5.16 at boot; trending back up
- Heartbeats stale (autonomous_mode 15min, gabriel_self 29min)
- `dashboard/pacing_state.json` MISSING (mandate references it)
- `state/cloud_dispatch/job_queue.json` MISSING (consumer expects it)
- session_resume_guardrail DESIGNED (`recovery_guardrail_design.md` 25 KB) but NOT INSTALLED
- 22 daemons crash-looping (exit 78, 126)
- Drive→local rsync hooks DESIGNED, NOT INSTALLED
- GH Actions billing-blocked (xsec re-run 26196717999 "not started" — user action required)

---

## 10. Open critical items (top of stack)

1. **GH Actions billing** — user action at billing.github.com (cannot agent-fix)
2. **Modal workspace spend cap** — user action at modal.com dashboard (cannot agent-fix)
3. **Hook FUSE migration** — rsync to `~/.zg/hooks/`, install `com.zg.sync_hooks_local` daemon, patch `settings.json` paths
4. **session_resume_guardrail install** — 10-point checklist ready, gated on aggregator
5. **22 daemons crash-looping** — `mission_overseer`, `agent_watchdog`, `memory_auto_save` all NOT running with Nice=-15
6. **dashboard/pacing_state.json missing** — mandate references nonexistent file; create stub or update mandate
7. **30 docstring-only stubs** — need extraction pass (recipes in docstrings, not function bodies)
8. **429 unwired trading repos** — triage WIRE_NOW / WIRE_LATER / RETIRE (see `unwired_repos_inventory.md`)
9. **Magic-link signup engine** — Yahoo+NopeCHA IP rate-limit chokepoint (24h reset OR paid key OR --use-tor)
10. **xsec workflow CPU runtime** — when billing unblocked, prefer Modal A10G over 4-core CPU 2-6h fallback

---

## 11. Quick-reference paths

```
Project root         /Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools
Trading repo         AI-Tools/s&p500-ticker-mastery/
Mastery GH repo      0riginal-claw/sp500-mastery-compute (private)
Logs                 AI-Tools/logs/
Auto-solve logs      AI-Tools/logs/auto_solve/
Crash forensics      AI-Tools/logs/auto_solve/crash_forensics_2026-05-20/
Backups              AI-Tools/backups/<feature>-<DATE>/
State (Drive)        AI-Tools/state/<feature>/
State (local SSD)    ~/.zg/state/<feature>/
Hooks (Drive)        AI-Tools/home/.claude/hooks/<name>/<script>.sh
Hooks (local, planned) ~/.zg/hooks/<name>/<script>.sh
Settings             AI-Tools/ClaudeCode/config/settings.json
LaunchAgents         ~/Library/LaunchAgents/com.zg.<feature>.plist
Pending issues       ~/.claude/state/autosolve_pending.jsonl
Memory index         AI-Tools/ClaudeCode/config/projects/.../memory/MEMORY.md
Launcher (Claude)    AI-Tools/bin/claude-gdrive
Launcher (OpenClaw)  AI-Tools/bin/openclaw-gdrive
Cloud dispatch       AI-Tools/s&p500-ticker-mastery/scripts/cloud_dispatch.py
Workflow xsec        AI-Tools/s&p500-ticker-mastery/.github/workflows/xsec.yml
Feature manifest     AI-Tools/s&p500-ticker-mastery/scripts/feature_manifest.json
Generated stubs      AI-Tools/s&p500-ticker-mastery/scripts/_generated/
Trading repos clones AI-Tools/external-repos/trading-free-clones/<category>/<repo>/
```

---

**End Operating Manual 2026-05-20.** Update on every material protocol change.
