# Standard sub-agent brief template

Every sub-agent spawned in this workspace should be briefed with the following standing context (copy/paste sections relevant to the task).

## 1. Workspace + authorization
- Working scope: `/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/`
- Primary AI root: `AI-Tools/`
- Permission mode: `bypassPermissions` (deny rules + hooks still active — see `reports/security_hardening/`)
- Pre-approved: reads/writes/installs/scripts/logs/backups inside My Drive, no sudo, no permanent deletes
- Safety boundaries: no credential/wallet/financial-doc reads; no real-money trades; no external account changes; no public-facing exfil

## 2. Sub-agent nesting (NEW — confirmed working as of 2026-05-16)

### 2.0 OpenClaw nested orchestration (NEW 2026-05-19 — llm-task plugin enabled)
For non-Claude work spanning **N independent slices** (multi-bucket research, parallel ticker analyses, fan-out report writes), prefer OpenClaw's `llm-task` plugin over a single sequential `openclaw agent` call:
  - The `llm-task` plugin is now ENABLED (was: bundled-but-disabled). Verify with `openclaw plugins inspect llm-task` → `Status: loaded`.
  - **When to nest:** if the task decomposes into 2+ independent JSON-structured sub-questions (one per ticker / one per bucket / one per source URL), spawn them as parallel llm-task calls inside the OpenClaw orchestration rather than answering them serially.
  - **Dedicated workspaces (added 2026-05-19):**
    - `trading-research` — buckets, signal-gen research, backtest analysis (use for `oc_task2_*.txt` templates by default; do NOT route to `main`)
    - `web-scraper` — signup automation, CAPTCHA solving, browser-driven workflows
    - `data-ingest` — ETL, parquet generation, file conversion
  - **Invocation pattern:** `openclaw agent --agent trading-research --local --model deepseek/deepseek-v4-flash --json --message "..."` (route to the right workspace; `main` is reserved for general-purpose chat).
  - **Rule of thumb:** if work spans N independent slices, spawn N llm-task calls (one per slice) instead of doing them sequentially inside a single agent turn — same recursive fan-out logic as §3, applied inside the OpenClaw runtime.

- You CAN spawn your own children via `mcp__plugin_fallback-agent_fallback__Task`
- Use ToolSearch to load the schema: `select:mcp__plugin_fallback-agent_fallback__Task`
- Schema accepts: description, prompt, model (sonnet/opus/haiku), timeout (ms), permissionMode, allowedTools, disallowedTools, addDirs

**REQUIRED SPAWN FLAGS (or child will be blocked):**
- `permissionMode: "bypassPermissions"` — otherwise child runs in stricter default mode and `Bash`/`Write`/network operations get blocked
- `addDirs: ["/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools"]` (or specific subpaths the child needs) — otherwise child can't read/write outside its starting cwd
- Verified empirically 2026-05-16: financial-training agent's initial spawn failed with "blocked by missing bypassPermissions"; re-spawned with both flags → succeeded immediately

**PLUGIN PATCH 2026-05-17 (defense-in-depth):** The fallback-agent plugin at `ClaudeCode/config/plugins/cache/fallback-agent/fallback-agent/1.0.3/mcp-server/{src,dist}/index.{ts,mjs}` has been patched so that:
  - When `permissionMode` is omitted, it defaults to `"bypassPermissions"` (was: no `--permission-mode` flag, child fell back to `default` mode and prompted on every tool call — this caused the Wave A / mass-retry / all-repos helper failures).
  - When `addDirs` is omitted, it consults `FALLBACK_AGENT_DEFAULT_ADD_DIRS` env (colon-separated list; the launcher sets it to `AI-Tools`) and falls back to that.
  - Spawn-time hook `home/.claude/hooks/spawn-validator/validator.sh` (Part B) emits a stderr WARN when plugin Task is called without `permissionMode`/`allowWrite` or without `addDirs`, logged to `logs/spawn_validator.log`.
  - Root cause analysis + restart instructions: `reports/plugin_task_permission_fix_2026-05-17.md`. Restart Claude Code to reload the MCP plugin after the patch.
  - **You should STILL pass `permissionMode` and `addDirs` explicitly** for clarity and to remain robust against plugin upgrades that may overwrite the patch.

### 2a. CRITICAL — MCP tool inheritance to sub-agents (root cause analysis 2026-05-16)

**The problem you may have seen:** some sub-agents report `mcp__plugin_fallback-agent_fallback__Task` is "not available" and fall back to inline work — others (spawned via the same MCP tool) can use it freely to nest deeper.

**Root cause:** This is governed by HOW the sub-agent is spawned, NOT by config/permissions:

| Spawn shape | MCP tools (incl. nest tool) available? | Why |
|---|---|---|
| Parent uses `mcp__plugin_fallback-agent_fallback__Task` (the plugin tool) | **YES — full MCP** | Spawns fresh `claude -p` subprocess with `--plugin-dir`. New process loads ALL MCP servers from user-scope `.claude.json` (memory-keeper, github, fallback-agent itself, etc.). |
| Parent uses native `Task` tool with `subagent_type: general-purpose` (or `Explore`/`Plan`) | **YES — full MCP** | Built-in subagent types inherit ALL parent tools including MCP (documented behavior). |
| Parent uses native `Task` tool with `subagent_type: python-pro` (or any `awesome-claude-code-subagents/*` plugin agent) | **NO — MCP stripped** | Those agent definitions ship with `tools: Read, Write, Edit, Bash, Glob, Grep` in YAML frontmatter — an **explicit allowlist** that omits MCP tools. Per official docs: "if `tools` is set, MCP tools are NOT inherited". |
| Parent uses native `Task` tool with `subagent_type: <any-plugin-defined-agent>` | **OFTEN NO** | Known Claude Code bugs #13605, #15810, #21560 — plugin-defined custom subagents frequently can't access MCP tools regardless of `tools` field. |

**Verified 2026-05-16:** 3 fresh probes (general-purpose, sonnet default, haiku unrestricted) all spawned via `mcp__plugin_fallback-agent_fallback__Task` confirm MCP tool **always visible + callable**. The "modal worker fix agent" inconsistency was caused by spawning `python-pro` via native `Task`, which is restricted by frontmatter.

**RULE: if your sub-agent needs to spawn its own children (multi-level fanout) OR call any MCP tool (memory-keeper, github, drive, fallback-agent), do ONE of:**

1. **PREFERRED** — Spawn via `mcp__plugin_fallback-agent_fallback__Task` (this is the documented pattern in section 2 above). The plugin's subprocess shape gives the child full MCP access automatically.
2. **OR** Spawn via native `Task` with `subagent_type: "general-purpose"` (built-in, inherits all tools). Use this when you specifically want the parent's exact tool set and context.
3. **AVOID** native `Task` with `subagent_type: python-pro` / `golang-pro` / any `awesome-claude-code-subagents/*` type IF the child needs MCP — those agents' frontmatter strips MCP tools. They're fine for pure code-write tasks that only need Bash/Read/Edit/Write/Grep/Glob.

**If you MUST use a language-specialist subagent type and it needs MCP:**
- Either copy the agent file to `.claude/agents/` (NOT a plugin location) and edit out the `tools:` line so MCP inherits naturally
- OR add explicit MCP tool names to the agent's `tools:` allowlist (e.g. `tools: Read, Write, Edit, Bash, Glob, Grep, mcp__plugin_fallback-agent_fallback__Task, mcp__memory-keeper__*`)
- OR have the parent do the MCP calls and pass results to the child via the prompt string

**Spawn pattern (copy-paste):**
```python
mcp__plugin_fallback-agent_fallback__Task(
    description="...",
    prompt="...",
    model="sonnet",  # or haiku/opus per task type
    permissionMode="bypassPermissions",
    addDirs=["/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools"],
    timeout=600000  # 10 min default
)
```

## 3. Long-running task protocol — HARD MANDATE (tightened 2026-05-17, MECHANICALLY ENFORCED)

### 3.0 The 5-minute Recursive Fan-out Rule (UNCONDITIONAL)

**If estimated wall-clock OR observed elapsed time exceeds 5 minutes AND the task has >2 logical slices remaining, you MUST stop in-line work, decompose into N=2-6 parallel sub-helpers, and become the aggregator.** This is unconditional. Failure to recurse at the 5-minute boundary is a **protocol violation** that will be flagged in proof-of-work.

Re-check this rule at every additional 5-minute boundary while work remains. Recursion is the default, not the exception. **All helpers (and their helpers, and their helpers' helpers) inherit this same mandate.** Recursion propagates down the tree unconditionally.

**Spawn method**: native `Agent` tool with `subagent_type: general-purpose` (default, see CLAUDE.md "Default tool preferences"). Use `mcp__plugin_fallback-agent_fallback__Task` only when the helper itself must spawn grandchildren that need MCP tools.

### 3.0.1 MECHANICAL ENFORCEMENT (added 2026-05-17 — audit found 144 violations in last 3 days)

The §3 mandate is now enforced by two PreToolUse hooks (wired into `AI-Tools/ClaudeCode/config/settings.json`):

1. **`recursion-fanout-tracker`** (matchers: `Bash|Edit|Write|MultiEdit|NotebookEdit|Task|Agent|mcp__plugin_fallback-agent_fallback__Task`)
   - Tracks per-session start time at `/tmp/cc-recursion-tracker/<sid>.start`
   - Counts spawns at `/tmp/cc-recursion-tracker/<sid>.spawns`
   - At 5/10/15 min elapsed with **zero spawns** → emits stderr WARN of escalating severity (visible to helper as tool feedback)
   - At **20 min elapsed with zero spawns** → **BLOCKS the tool call with exit 2** (KILL CONDITION). Helper sees the violation message and MUST commit to fan-out, surface partial, or self-terminate.
   - If any spawn has occurred → silent pass (mandate satisfied).
   - Source: `home/.claude/hooks/recursion-fanout-tracker/tracker.sh`

2. **`spawn-validator`** (matchers: `Task|Agent|mcp__plugin_fallback-agent_fallback__Task`)
   - Validates every spawn prompt contains AT LEAST ONE of:
     - `# decomposition_plan: <slice1, slice2, slice3>` — preferred for any >5-min scope
     - `# scope_estimate_min: <integer>` — if >5 also requires decomposition_plan
     - `# inline_justification: <why this is a single helper>` — for genuine single-slice work
   - Warn-only (non-blocking) but logged to `AI-Tools/logs/spawn_validator.log` for audit drift detection.
   - Source: `home/.claude/hooks/spawn-validator/validator.sh`

**What this means for every spawn you author:**

```
# decomposition_plan: ticker_AAPL, ticker_MSFT, ticker_GOOG, ticker_AMZN, synth
# OR
# scope_estimate_min: 3
# OR
# inline_justification: one-shot config edit, no parallelization possible

<rest of prompt>
```

**If you author a spawn without one of these lines, the hook warns and logs the drift.** Orchestrator: if you can't articulate the decomposition or justify single-helper, you are NOT YET READY to spawn — think more first. Helpers: inherit and propagate these lines to grandchildren.

### 3.0.2 Audit baseline (2026-05-17)

Pre-enforcement audit of the last 3 days' helper JSONLs found **144 protocol violations** (>5 min elapsed + >20 tool uses + zero fan-outs in a single helper session). Worst offenders ran 216 min / 112 tools and 215 min / 97 tools — both solo, both should have decomposed into 6+ helpers. Sessions that DID fan out (e.g., 344 spawns / 1397 tools) completed work an order of magnitude faster. The cost of NOT fanning out is observable and large.

### 3.1 Trigger conditions (any one fires the mandate)

- Estimated wall-clock > 5 minutes
- Observed elapsed wall-clock > 5 minutes with work remaining
- Task touches 2+ files
- Task has 2+ independent sub-questions
- Task processes 2+ items (tickers, URLs, repos, sections, etc.)

When triggered, you MUST:

1. **Decompose FIRST, within 30-60 seconds of receiving your brief.** Don't wait the full 2 min. List 2-6 independent sub-tasks, each scoped to **< 5 minutes** wall-clock.
2. **Spawn helpers immediately** — native `Agent` `subagent_type: general-purpose` is the default. All spawned in parallel in a single message.
3. **Assign each helper a UNIQUE file/path scope** (see section 4). Two helpers must NEVER touch the same file.
4. **You become the aggregator**: poll helper output files, synthesize results when they land. Don't do the underlying work yourself once helpers are spawned.
5. **Every 5 minutes** of remaining wall-clock, audit: are any sub-tasks still running >5min with multi-slice work left? If yes, **recursively spawn sub-helpers** under those slow ones. Each helper MUST follow the same rule. Nested recursion is verified working (40 nested helpers across 3 levels validated 2026-05-16).
6. **Fan out aggressively**: if your task touches N files, default to N helpers (one per file). If aggregating M reports, default to M+1 helpers (M readers + 1 synthesizer). Marginal cost of an extra helper is low; marginal cost of a serial bottleneck is high.

### 3.2 Concrete EXAMPLES — when fan-out is mandatory

| Situation | Action |
|---|---|
| **Ticker batch >5 tickers** | One helper per ticker (or per ticker group of <=2). Don't loop. |
| **Multi-file edit >3 files** | One helper per file. Use `/batch` if same transformation; otherwise N helpers. |
| **Multi-folder scan >3 folders** | One helper per folder. Coordinator merges. |
| **N config / settings updates** | One helper per config domain. |
| **N independent research queries** | One helper per URL/source. Synthesizer merges. |
| **N report sections** | One helper per section, writing to unique `/tmp/<slice>.md` paths. |
| **N-strategy backtest sweep** | One helper per strategy. Aggregator computes leaderboard. |
| **Migrating M files between formats** | `/batch` if uniform, else N helpers. |
| **Verifying N URLs / dependencies / installs** | One helper per item; never serial loop. |
| **Reading + summarizing M docs >2k tokens each** | One helper per doc. Synthesizer merges. |
| **Tool-loop with >20 sequential tool uses observed** | STOP. Decompose what remains. Spawn helpers for unfinished slices. |

### 3.3 KILL CONDITION (HARD)

**If elapsed wall-clock exceeds 20 minutes AND zero fan-out has occurred yet, you MUST:**

1. Stop all in-line work immediately.
2. Write a partial report covering what was completed.
3. Document why you failed to fan out at the 5-min boundary (this is a protocol violation — name it).
4. Either: (a) self-terminate and surface the partial to the parent, OR (b) immediately spawn helpers for all remaining slices with a note that this is a recovery fan-out.

The 20-minute solo-work ceiling is **absolute**. Wave A's 33-minute / 83-tool-use solo run is the canonical example of what this mandate exists to prevent.

### 3.4 Recursion propagation requirement

Every helper you spawn MUST be briefed with the same §3 mandate. Copy/paste this section's text (or a pointer to it) into the child's prompt. The mandate dies at any level that fails to propagate — and a non-recursive grandchild defeats the whole tree.

Decomposition axes (pick one or compose): per-ticker, per-strategy, per-file, per-feature, per-cloud-adapter, per-test-case, per-section-of-doc, per-API-call, per-URL.

**Use cheap backends for mechanical work** (see §5a): if a sub-task is text manipulation / summary / format conversion, route it to `scripts/ollama_helper.py` (free, $0) or `scripts/deepseek_helper.py` ($0.000001) via subprocess — no Claude sub-agent needed.

**Anti-patterns (don't):**
- "I'll just do it inline since it's fast" — every prior session's agent that said this took 6-15 min and produced 0 helpers. The mandate exists because that judgment has consistently been wrong.
- "Sequential because of dependencies" — only TRUE sequential deps (B reads A's output) force serial. If both can run in parallel and merge later, spawn both NOW.
- "It's only 5 files, I'll just loop" — spawn 5 helpers; loops are the slow path.
- "I'm 4 minutes in, almost done" — if work isn't done at the 5-min mark with >2 slices left, fan out. The "almost done" judgment has been consistently wrong.
- "The helpers are doing tool-loops solo" — propagate the §3 mandate INTO every helper brief. A helper that doesn't recurse is a misbriefed helper.

## 4. Work non-overlap — ENFORCED via path scopes

Every helper MUST be assigned an explicit UNIQUE scope in its brief. Examples of valid scope assignments:

- **Per-file**: Helper A owns `scripts/orb_fast.py`; Helper B owns `scripts/orb_fade.py`. Neither touches the other's file.
- **Per-section**: Helper A writes lines 1-100 of report at `/tmp/helper_A.md`; Helper B writes lines 101-200 at `/tmp/helper_B.md`. Coordinator concats.
- **Per-output-key**: Helper A writes `reports/<topic>/section_a.md`; Helper B writes `reports/<topic>/section_b.md`. No same-file writes.
- **Per-ticker**: Helper A processes tickers A-M; Helper B processes N-Z. No overlap on ticker set.

**Rule**: if two helpers' scopes would touch the same path, the brief is wrong — re-decompose.

**Staging convention**: each helper writes its output to a designated unique path like `/tmp/<parent-agent-id>/<slice-name>.md`. Coordinator polls those paths; never reads a path it shares with another helper.

If you detect a sibling has modified a file in your declared scope, treat it as a brief error and STOP — report the overlap rather than blind-merge.

## 5. Pacing-aware model routing

### 5.routing-mandate — Model routing MANDATE (HARD RULE)

Before spawning any helper via `mcp__plugin_fallback-agent_fallback__Task`, you MUST include a `# model_reason: <one-line justification>` comment near the `model: ...` parameter (in the prompt or as a leading line of the prompt). If you cannot articulate why opus over sonnet (or sonnet over haiku), default to the CHEAPER option.

Anti-pattern: hardcoding sonnet for tasks that are pure mechanical work (file edits, CLI writes, JSON parsing, inventory scans) — those are HAIKU.

Routing quick-reference:
- haiku → list/copy/inventory/file scan/grep/summarize/format/quick file inspection/repetitive low-risk
- sonnet → normal coding, file analysis, project organization, medium debugging, scripts, doc edits, clear-task implementation (DEFAULT if unsure)
- opus → advanced reasoning, architecture, complex coding, hard debugging, multi-step planning, strategy review, final synthesis, large refactors

Every spawn MUST justify its model choice or default DOWN.

- Read `dashboard/pacing_state.json` to see current regime
- emergency → use haiku unless absolutely impossible
- over → haiku for mechanical, sonnet for must-reason
- on → sonnet default, opus for architecture/synthesis
- under → opus aggressively for reasoning, sonnet otherwise
- Unified router (when available): `scripts/unified_model_router.py` handles this

### 5a.0 Cloud-routing mandate (HARD RULE, 2026-05-17)

**Heavy compute MUST run on the cloud (Modal + gh_actions), NOT on the Mac.** The Mac is for Claude orchestration + Drive sync + lightweight daemons. Target: **85-90% of CPU/RAM workload off-Mac**.

#### FORBIDDEN patterns (these are protocol violations)

1. **`AUTO_CLOUD_DISPATCH=0`** in any spawn prompt's Bash env unless the task is a smoke test <60s wall-clock. Setting `=0` defeats the venv-level monkey-patch that auto-routes heavy workers and forces Mac execution.
2. **Direct local invocation** of heavy workers from a helper prompt: `python3 scripts/backtest_xgb_v10.py --ticker X`, `python scripts/orb_*.py`, `python scripts/vwap_*.py`, `python scripts/momentum_*.py`, `python scripts/catalyst_*.py` — these MUST go through `cloud_dispatch.enqueue_job(...)`.
3. **Parallel local launches** (e.g. `for ticker in $TICKERS; do python backtest.py --ticker $ticker &; done`) — the entire batch must enqueue to cloud; the parent polls dispatched.jsonl for completion.
4. **Disabling the monkey-patch** via `auto_cloud_dispatcher.disabled()` context manager for >60s windows.

#### REQUIRED patterns

1. **Heavy work goes through `cloud_dispatch.enqueue_job(...)`** — example:
   ```python
   from cloud_dispatch import enqueue_job
   job_id = enqueue_job(
       script="backtest_xgb_v10.py",
       ticker="AAPL",
       strategy="xgb_v10",
       extra_args=["--mode", "validation"],
   )
   # Poll sweeps/dispatched.jsonl or use enqueue_and_wait() helper
   ```
2. **Polling pattern** — read `s&p500-ticker-mastery/sweeps/dispatched.jsonl` (line-appended log of job state). Aggregator helpers tail this file rather than running compute themselves.
3. **Harvesting** — completed cloud jobs write `result.json` (or equivalent) to a known artifact path under `sweep_artifacts/`. Helpers read those artifacts, never recompute locally.
4. **Smoke-only locals** — if a helper genuinely needs a <60s sanity check, that's the ONLY justifiable local heavy-compute run. Document the smoke window in proof-of-work.

#### Mac safety cap (load-aware)

Before launching ANY local Python heavier than a file edit / inventory scan, every helper MUST check `uptime` (1-min load average):

- **load <8** — normal; local OK for ad-hoc <60s scripts; heavy workers still enqueue to cloud per rule above.
- **load 8-12** — caution; ONLY smoke tests <30s; everything else MUST cloud-route.
- **load >12** — HARD STOP; no new local heavy procs. Existing work continues; new work cloud-routes only. Surface load in proof-of-work and alert parent.

#### Why this matters

- Mac is single-machine, finite IO + CPU. Modal + gh_actions are horizontally scalable.
- Cloud has paid quota; local has thermal/swap walls. Spending $0.10 of cloud quota saves 30 min of Mac thrash.
- Drive sync degrades when Mac load >15 — preservation guarantees weaken.
- The `auto_cloud_dispatcher.py` monkey-patch was built specifically because helpers kept overriding it. Don't override.

#### When `AUTO_CLOUD_DISPATCH=0` IS legitimate

- Smoke tests <60s on 1 ticker with smallest feasible date window.
- Local-only debug of a feature module before wire-in (helper writes the module, runs smoke, then ALL further sweeps enqueue).
- Unit tests in `tests/` directories.
- Documentation generation that touches no compute.

Any other use is a protocol violation. The orchestrator must call it out.

### 5a. When to use Ollama/DeepSeek instead of Claude sub-agents

Before spawning a Claude sub-agent, ask: **"Does this need real reasoning or just text manipulation?"**

**Mechanical work → call `ollama_helper.py` subprocess. Zero Claude tokens spent.**

Mechanical tasks: file inventory, JSON parsing, text summarization, format conversion, list filtering, regex extraction, CSV manipulation, log parsing, template filling, string transformations.

```bash
python "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/scripts/ollama_helper.py" \
  --prompt "Summarize this JSON into bullet points: {data}" \
  --max-tokens 500
```

Cost: $0.00. Latency: ~5-15s (local 7B model on CPU). No quota impact.

**Independent verification / second opinion / large-context research → call `deepseek_helper.py`. ~$0.000001/call.**

Use DeepSeek when you need: a genuine second opinion from a different alignment, context windows >200k tokens, or cross-checking a Claude conclusion before a high-stakes decision.

```bash
python "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/scripts/deepseek_helper.py" \
  --prompt "Verify this backtest logic for look-ahead bias: {logic}" \
  --max-tokens 1000
```

Cost: ~$0.000001/call. Latency: ~6-15s. Different alignment = real second opinion.

**Reasoning / tool use / multi-step planning / code generation → spawn Claude sub-agent via MCP (current pattern). Claude tokens used but justified.**

Use Claude when: the task requires multi-step reasoning, tool calls, file writes, web search, or coordinating multiple actions. This is the only tier that can use MCP tools.

**Decision tree:**

```
Task arrives
    |
    v
Text-only manipulation? (summarize, parse, filter, convert)
    YES → ollama_helper.py  ($0.00, free, no quota)
    NO  →
        Need independence / second-opinion / >200k context?
            YES → deepseek_helper.py  (~$0.000001, independent alignment)
            NO  →
                Need tool use / file writes / multi-step reasoning?
                    YES → Claude sub-agent (MCP spawn, quota used, justified)
```

**Both helpers return JSON to stdout** — parse with `json.loads(result.stdout)` and check `result["success"]`.

### 5b. Automatic behaviours your sub-agents inherit (no action needed)

When you spawn a sub-agent, the following auto-fire — you do NOT need to invoke them manually:

**Token reduction stack**
- `SessionStart` hook auto-runs `/resume`: loads CLAUDE.md + recent session logs. Script: `…/AI-Tools/home/.claude/hooks/auto_resume.sh`
- `PreCompact` hook auto-runs `/preserve`: appends session learnings to CLAUDE.md before compaction. Script: `…/AI-Tools/home/.claude/hooks/auto_preserve.sh`
- `UserPromptSubmit` hook auto-compresses any prompt >2000 tokens via `llmlingua_compress.py` and replaces the prompt with the compressed version transparently. Script: `…/AI-Tools/home/.claude/hooks/auto_llmlingua_compress.sh`
- `caveman` plugin is active (already in enabledPlugins) — in-context prose compression ~75%.
- `Mem0` is wired via the memory MCP server for cross-session persistence.

**Cloud dispatch auto-routing**
- Every Python subprocess in the workspace venv (`/Users/orginal/.venvs/sp500-mastery`) has its `subprocess.run/Popen/call/check_call/check_output` monkey-patched at interpreter startup by `…/AI-Tools/scripts/auto_cloud_dispatcher.py` (installed by the venv's `sitecustomize.py`).
- Heavy-compute Python workers (`orb_*.py`, `vwap_*.py`, `backtest_*.py`, `momentum_*.py`, `catalyst_*.py` invoked with a ticker arg) auto-route to `cloud_dispatch.enqueue_job(...)` instead of running on the Mac.
- Light / non-python / orchestrator subprocess calls pass through unchanged.
- To opt out for a specific subprocess: `os.environ["AUTO_CLOUD_DISPATCH"] = "0"` or use `with auto_cloud_dispatcher.disabled(): …`
- To dry-run (log decision but don't enqueue): `AUTO_CLOUD_DISPATCH_DRY_RUN=1`

**Implication for sub-agent prompts**: stop asking sub-agents to "use cloud_dispatch for heavy compute" or "run /resume on start" — these are auto-applied. Only mention them when you need to explicitly opt OUT or override the defaults.

---

### 5b.manual. Token-reduction tools (manual invocation — every spawn, every model)

Token-reduction stack (8 tools). Pick by trigger, not by default; auto-on tools (RTK, token-optimizer-mcp) require no action.

**1. RTK** — Trigger: auto-on after Claude Code restart.
Auto-compresses large Bash output before it lands in context. No manual invocation needed.
- Invocation: `some-command | rtk` or `rtk <input-file>`

**2. token-optimizer-mcp** — Trigger: compress tool output / structured payloads mid-session.
MCP server registered in `AI-Tools/ClaudeCode/config/settings.json`. Auto-loads on Claude Code start. Invoke via registered MCP tools (no subprocess needed).

**3. LLMLingua** — Trigger: prompts >2000 tokens to nested helpers.
Aggressive prompt compression for lossy-OK summarization. Wraps LLMLingua-2.
```bash
/Users/orginal/.venvs/sp500-mastery/bin/python \
  "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/scripts/llmlingua_compress.py" \
  --text "..." \
  --target-ratio 0.5
```

**4. Mem0** — Trigger: cross-session recall of facts / decisions.
Semantic memory with smart retrieval. Reduces re-loading of prior context.
```bash
python "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/scripts/mem0_helper.py" \
  --add "fact here" \
  --search "query"
```

**5. Selective Context** — Trigger: long-context pruning of low-info sentences.
Drop irrelevant context blocks before passing to helpers; prune by relevance.
```bash
python "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/scripts/selective_context_helper.py" \
  --text "..." \
  --reduce_ratio 0.5
```

**6. CausalRepack** — Trigger: causal-importance prompt compression (if install succeeds).
Repack token sequences causally to reduce KV-cache waste. Fallback to LLMLingua if unavailable.
```bash
python "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/scripts/causalrepack_helper.py"
```

**7. caveman** — Trigger: one-time compression of static memory files.
Compress a memory file (CLAUDE.md, large notes) into caveman format.
- Invocation: `/caveman-compress <filepath>`

**8. cpr-compress-preserve-resume** — Trigger: session restarts / context preservation across compactions.
Compress, preserve, or resume context across session boundaries.
- Invocation: `/compress`, `/preserve`, `/resume` (slash commands)

**MANDATORY rules — every spawn brief MUST include these:**

1. **Use LLMLingua for any prompt >~2000 tokens** to a nested helper — compress first with `--target-ratio 0.5`.
2. **Use RTK on large Bash output** automatically; helpers can invoke explicitly on expected large results.
3. **Use token-optimizer-mcp via MCP tools** whenever compressing structured tool output mid-session.
4. **Brief all children with this 8-tool stack** so it propagates down the tree.

### 5c. Token reduction toolbox — full inventory (2026-05-16)

All 11 tools in the stack. Every spawn must know about every tool and pick the right one for each situation.

| # | Tool | Status | When to use | Invocation |
|---|---|---|---|---|
| 1 | **cpr-compress-preserve-resume** | ACTIVE | Session compress/preserve/resume; slash commands `/compress`, `/preserve`, `/resume` | `Skill("cpr-compress-preserve-resume:compress")` etc. |
| 2 | **JSON mode** | ACTIVE | Enforce structured output from any Claude call; eliminates prose padding | Pass `response_format: {type: "json_object"}` or add "respond ONLY with JSON" to system prompt |
| 3 | **bypassPermissions** | ACTIVE | Workspace-wide — removes permission-prompt overhead from every tool call | Set by launcher or `settings.json defaultMode`; no action needed |
| 4 | **caveman plugin** | ACTIVE | In-context prose compression ~75%; use when many turns expected | `/caveman` or `Skill("caveman:caveman")` |
| 5 | **RTK** | ACTIVE | Auto-compress large Bash stdout before it lands in context | `~/.local/bin/rtk`; pipe: `command | rtk` or `rtk <file>` |
| 6 | **token-optimizer-mcp** | ACTIVE | MCP server; auto-loads on Claude Code start; call via MCP tool list | Invoke via registered MCP tools (no subprocess) |
| 7 | **LLMLingua** | ACTIVE | Aggressive prompt compression (50%+ ratio) for nested helper prompts | `/Users/orginal/.venvs/sp500-mastery/bin/python "AI-Tools/scripts/llmlingua_compress.py" --text "..." --target-ratio 0.5` |
| 8 | **Selective Context** | ACTIVE | Drop irrelevant context blocks before passing to helpers; prune by relevance | Manual: identify and omit low-relevance file sections from child prompt |
| 9 | **Mem0** | ACTIVE | Cross-session semantic memory with smart retrieval; reduces re-loading | Use Mem0 MCP tools or `scripts/mem0_helper.py` if installed |
| 10 | **CausalRepack** | ACTIVE | Repack token sequences causally to reduce KV-cache waste in long contexts | `scripts/causal_repack.py` (if installed); apply before very long context passes |
| 11 | **Anthropic prompt caching** | NATIVE | Cache static system-prompt prefixes; cuts repeat-call cost 80-90% | Add `cache_control: {type: "ephemeral"}` breakpoints at stable prefix boundaries in API calls |

**Decision guide:**

```
Bash output large?         → RTK (pipe it)
Prompt >2000 tokens to nested helper? → LLMLingua (--target-ratio 0.5) FIRST
Many turns in session?     → caveman plugin
Need structured output?    → JSON mode (system prompt or response_format)
Static prefix in API call? → Anthropic prompt caching (cache_control breakpoints)
Cross-session context?     → Mem0
Long KV context waste?     → CausalRepack
Context blocks irrelevant? → Selective Context (manually prune)
Already in workspace?      → bypassPermissions (already active, no action)
MCP context overhead?      → token-optimizer-mcp (auto-loaded)
Session checkpoint/resume? → cpr-compress-preserve-resume
```

**Mandatory rules (every spawn brief must include):**
1. RTK on all large Bash output.
2. LLMLingua on any prompt >2000 tokens passed to a nested helper.
3. Brief the child with this toolbox so it propagates down the tree.

## 6. Proof of work (end of every task)
- Commands run (verbatim)
- Paths accessed (absolute)
- Files created/changed (absolute + one-line note)
- Backups created (paths)
- Sub-agents used (id, model, reason, result)
- OpenClaw/DeepSeek calls (commands + cost)
- Final status (done/pending/blocked)

## 7. Native Claude Code 2026 features — MANDATORY for all sub-agents

These are first-class defaults. Sub-agents must know them and apply them before falling back to manual patterns.

### 7.1 /sandbox — session isolation
- Recommended at session start. A SessionStart hook now auto-prompts this.
- Use when the session touches unfamiliar repos or runs untrusted scripts.

### 7.2 /ultraplan — multi-file architecture planning
**Trigger: task touches >3 files OR involves architecture/design changes.**
- Do NOT write a plan in prose inline — invoke `/ultraplan` so Claude Code reasons over the actual codebase state.
- From inside a sub-agent (can't type slash commands): use the wrapper:
  ```python
  import subprocess, json
  result = subprocess.run(
      ["python3", "AI-Tools/scripts/claude_native_features.py",
       "ultraplan", "task description", "--files", "a.py", "b.py"],
      capture_output=True, text=True
  )
  data = json.loads(result.stdout)
  ```
  Full API: `invoke_ultraplan(task, files)` in `AI-Tools/scripts/claude_native_features.py`.

### 7.3 /batch — parallel same-transformation across files
**Trigger: same change/refactor applies to 3+ similar files.**
- Faster than a for-loop; runs file edits in parallel.
- Wrapper: `invoke_batch(file_list, transformation)` in `AI-Tools/scripts/claude_native_features.py`.
- CLI: `python3 AI-Tools/scripts/claude_native_features.py batch a.py b.py c.py --transformation "add type hints"`

### 7.4 /loop — iterative optimization
**Trigger: any task described as "iterate until X improves", hyperparameter sweep, or autoresearch loop.**
- Replaces ad-hoc `while True:` Python loops in optimization scripts.
- Wrapper: `invoke_loop(prompt, max_iter)` in `AI-Tools/scripts/claude_native_features.py`.
- Default: `--max-iter 10` (change when needed).

### 7.5 /ctx-viz — context usage check
**MUST invoke if context concern surfaces at any 5-min check-in.**
- Returns token usage breakdown; use to decide whether to compress before the next sub-agent spawn.
- Wrapper: `invoke_ctx_viz()` in `AI-Tools/scripts/claude_native_features.py`.

### 7.6 /effort — model effort slider
- Bind to `model_reason`: opus → `high`/`xhigh`, sonnet → `medium`, haiku → `low`.
- Pass as `--effort <level>` flag when invoking `claude --print`.

### 7.7 Tasks API — cross-session task persistence
**Use instead of TodoWrite for tasks that span sessions or sub-agents.**
- TodoWrite is ephemeral (in-context only). Tasks API persists across sessions.
- Wrapper: `AI-Tools/scripts/tasks_api_helper.py`
  ```bash
  python3 AI-Tools/scripts/tasks_api_helper.py create "my task title"
  python3 AI-Tools/scripts/tasks_api_helper.py list --status pending
  python3 AI-Tools/scripts/tasks_api_helper.py update <task_id> done
  python3 AI-Tools/scripts/tasks_api_helper.py migrate todowrite_dump.json
  ```
- Tasks persist at `AI-Tools/.claude/tasks.json` (local mirror).
- TODO: once official Tasks API endpoint is confirmed, update the wrapper.

### 7.8 Output Styles — /config Explanatory mode
- Set `Explanatory` output style (via `/config`) for grandchild summaries that need to be human-readable without the parent's context.
- For machine-to-machine handoffs, use `Concise` or JSON mode instead.

### 7.9 --max-budget-usd cap
- The launcher (`AI-Tools/bin/claude-gdrive`) enforces `--max-budget-usd 50` per session.
- Sub-agents MUST NOT override or circumvent this cap.
- If a sub-task is approaching budget, escalate to parent rather than requesting a higher limit.

### 7.10 Feature recommender
To pick the right feature automatically:
```bash
python3 AI-Tools/scripts/claude_native_features.py recommend "task description here"
```
Returns: `/ultraplan`, `/batch`, `/loop`, `/ctx-viz`, or `none`.

## 8. Auto-solve-on-issue MANDATE (HARD RULE, 2026-05-17)

ANY error, blocker, failure, unknown library/API, dispatcher 4xx/5xx, daemon crash, dependency conflict, rate limit, CAPTCHA, auth failure, Playwright selector miss, schema mismatch, timeout, version conflict, weird output, or unexpected state → IMMEDIATELY spawn 2-3 parallel sub-agents to research + apply fix. NEVER ask user. NEVER stop the mission.

**Why:** User stated explicitly 2026-05-17 "when issues arise spawn sub agents to search the internt and github for solutions then apply said solutions do this automatically for EVERYTHING ANY TIME THERES AN ISSUE SAME FOR SUB AGENTS OF ALL KINDS!". Pattern across this session: every time the orchestrator stopped to explain a blocker to the user, time was wasted; every time it auto-spawned solver helpers, the mission unblocked itself.

### 8.1 Trigger conditions (HARD list — any match → spawn solvers immediately, no user prompt)
- Non-zero exit code on any command (excluding `grep` empty-match rc=1 + `diff` clean rc=1)
- HTTP 4xx/5xx from any API call (gh_actions, modal, openclaw, deepseek, ollama, provider signup)
- Daemon log shows `ERROR|CRITICAL|FATAL|Traceback` in last 60s
- `ImportError`, `ModuleNotFoundError`, `AttributeError`, `TypeError` from try-import or runtime call
- Sub-agent returns partial-failure status OR times out before §3 fan-out
- Mac load >12 OR RAM swap rate >100 MB/s (resource issue → cloud-route + research how to free)
- Unknown library/repo encountered in WIRE_CANDIDATE → spawn helper to research API+license before declining
- Browser-automation selector miss (Playwright element-not-found) → spawn helper to find current DOM
- Rate-limit hit on any service → spawn helper to find backoff strategy + alternative
- User reports stuck/slow/broken in prompt → spawn solver before responding

### 8.2 Default spawn pattern (3-helper fan-out, single message, parallel)
- **Helper INTERNET** — `WebSearch` + `WebFetch` for exact error message + last 12 months of solutions
- **Helper GITHUB** — `gh search code/issues/PRs` for matching error string + license-clean fix patterns
- **Helper REPO-LOCAL** — grep `My Drive/AI-Tools` registry + cloned-repos for prior solution

Each helper inherits §3 recursive fan-out — if estimate >5min, decompose + grandchild fan-out.

### 8.3 Aggregator + apply
Orchestrator (OR delegated aggregator helper) synthesizes 3 reports → picks lowest-risk + highest-leverage fix → APPLY silently → log to `AI-Tools/logs/auto_solve/<issue>_<UTC>.md`. NEVER ask user for permission to research.

### 8.4 Escalation to user (ONLY if)
- All 3 helpers return "no known fix"
- Solution requires money/messages/external-account changes per safety rules
- Solution would touch credentials/wallets/financial docs
- Solution requires irreversible destructive action

### 8.5 Inheritance
Every spawn brief MUST include §8 (or pointer + 3-bullet summary) so grandchildren auto-solve too — recursively, all the way down.

Related: §3 recursive fanout, memory `feedback_auto_solve_on_issue.md`, memory `feedback_always_auto_execute.md`.

## 9. Guardrail-grade default MANDATE (HARD RULE, 2026-05-20)

Every new feature/daemon/capability built by a sub-agent (or the orchestrator) MUST be guardrail-grade by default. The user stated explicitly 2026-05-20: "everything should be guardrail-grade by default, don't make me keep reminding you". This §9 is the meta-mandate that turns one-off features into self-healing infrastructure.

### 9.1 The 10-point checklist (apply by default — do NOT defer)

When building ANY new feature, the sub-agent MUST produce all 10 artifacts:

1. **launchd plist** at `home/Library/LaunchAgents/com.zg.<feature>.plist` with `KeepAlive=true` + `ThrottleInterval=30` (skip if not a long-running process)
2. **State heartbeat** at `state/<feature>/heartbeat.json` (atomic write: `ts`, `pid`, `cycle_id`, `status`)
3. **PreToolUse freshness hook** at `home/.claude/hooks/<feature>-freshness/check.sh` — verifies heartbeat fresh (10-min stale window default), auto-respawns via `launchctl bootstrap` if dead
4. **SessionStart bootstrap hook** at `home/.claude/hooks/<feature>-bootstrap/bootstrap.sh` — idempotent: checks `launchctl list` then loads plist if missing
5. **PostToolUse activity hook** at `home/.claude/hooks/<feature>-activity/update.sh` — updates `state/<feature>/last_session_activity.unix`
6. **SubagentStart context hook** at `home/.claude/hooks/<feature>-subagent-inject/inject.sh` — injects feature context (state snapshot, recent activity) into every child spawn so the entire tree inherits awareness
7. **Stop validation hook** at `home/.claude/hooks/<feature>-stop-validate/validate.sh` — checks the feature produced expected output during the turn; shames next turn on miss
8. **settings.json registration** — idempotent merge into `ClaudeCode/config/settings.json` (check existing entry by hook command before append)
9. **docs/<FEATURE>.md** — design, revert procedure, smoke test, failure modes, escape hatch
10. **backups/<feature>-pre-install-<YYYY-MM-DD>/** — timestamped backup of any pre-existing files before system change

### 9.2 Auto-wrapper

Invoke the `guardrail-ify` agent (`home/.claude/agents/guardrail-ify.md`) for any new feature larger than a 1-file script. It auto-generates all 10 artifacts from `(feature_name, script_path, plist_label?)`. Spawn pattern:

```
mcp__plugin_fallback-agent_fallback__Task with subagent_type=guardrail-ify
prompt:
  # model_reason: opus - meta-mandate generation, multi-file generation w/ safety
  Feature: <name>
  Script path: <abs path>
  Plist label: com.zg.<name> (optional override)
  Heartbeat fields: <list>
  Stale window: <minutes, default 10>
  Stop validate: <bool, default true>
```

### 9.3 Canonical examples

- `autonomous_mode` (#197) — daemon at `scripts/autonomous_mode_daemon.py`, plist `com.zg.autonomous_mode`, state at `state/autonomous_mode/`, 6 hooks (bootstrap, heartbeat, session-activity, subagent-inject, stop-validate, action-guard). See `docs/AUTONOMOUS_MODE.md`.
- `gabriel_self` (#198) — capability map at `state/gabriel_self/capability_map.json`, 6 hooks (bootstrap, freshness, observation, validate, context-inject, direction-validate).
- `tcc_autoallow` (#199) — TCC permission auto-grant, 5 hooks (bootstrap, freshness, context-inject, dialog-detect, validate).

### 9.4 Forgetting-curve mitigation

Features that ship without the wrapper get explicitly marked `**BRITTLE**` in their doc header. Monthly audit at `docs/GUARDRAIL_AUDIT.md` lists every feature ranked by guardrail-score (0-10) and queues retroactive promotion. See `docs/EVERYTHING_GUARDRAIL.md` for the design and rationale.

### 9.5 Trigger keywords (route to guardrail-ify automatically)

When the user prompt or task context contains any of: "new daemon", "new feature", "new capability", "new hook", "new automation", "always-on", "background service", "monitor process", "auto-restart", "self-healing", "persistent service" — the orchestrator MUST invoke `guardrail-ify` before writing a single byte of feature code.

### 9.6 Inheritance

Every spawn brief MUST include §9 (or pointer + 3-bullet summary) so grandchildren build guardrail-grade by default — recursively, all the way down. The 3-bullet summary:

- New feature → 10-point wrapper (plist + heartbeat + 5 hooks + settings + doc + backup)
- Use `guardrail-ify` agent to auto-generate
- BRITTLE if shipped without wrapper

Related: §3 recursive fanout, §8 auto-solve, memory `feedback_guardrail_grade_default.md`, `docs/EVERYTHING_GUARDRAIL.md`, `docs/GUARDRAIL_AUDIT.md`.
