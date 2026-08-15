# CLAUDE.md history — Session Preservation 2026-05-16

_Extracted from CLAUDE.md 2026-05-20 by audit-gap-closure Fix 10 to reduce CLAUDE.md below 40k threshold._

## Session Preservation — 2026-05-16 (Claude Code feature catalog install + automation wave)

**Installed (under `home/.claude/<type>/<namespace>/` from 7 catalog repos cloned to `repos-claude-clones/`):** 190 agents (all `tools:` commented to preserve MCP) + 130 slash commands + 70 skills + 120 plugins + 15 rules + 8 templates. 104 hooks copied **reference-only** under `_reference_<namespace>/` — NOT auto-registered. New Python wrappers: `scripts/claude_native_features.py` (/ultraplan, /batch, /loop, /ctx-viz, `recommend` heuristic CLI), `scripts/tasks_api_helper.py` (Tasks API + local-mirror at `.claude/tasks.json`).

**Auto-fire hooks live:** `auto_resume.sh` (SessionStart, 6596-char context inject), `auto_preserve.sh` (PreCompact), `auto_llmlingua_compress.sh` (UserPromptSubmit, **measured 44% token reduction** 2812→1574), `auto-sandbox.sh` (SessionStart /sandbox reminder), `cache-control-injector.sh` (PreToolUse Task, observe-only, logs to `logs/cache_control_hook.log`).

**Model-routing 100% coverage:** PreToolUse matcher = `^(Task|mcp__plugin_fallback-agent_fallback__Task)$`. Router CLI: `s&p500-ticker-mastery/scripts/unified_model_router.py --complexity <alias> --task-kind "<free>" --json` returns single-token model name on stdout + rationale on stderr. Hook warnings are operator-facing (stderr, not agent-self-correctable) — explicit `# model_reason:` in every spawn prompt is still mandatory.

**Auto cloud dispatch:** `scripts/auto_cloud_dispatcher.py` (22.2K, 15/15 unit tests pass) monkey-patches subprocess.{run,Popen,call,check_call,check_output}; `sitecustomize.py` (789B) in `~/.venvs/sp500-mastery/lib/python3.11/site-packages/` auto-installs patches. 9 scripts have native `--dispatch-mode {local,cloud}` (cloud default) + `--dry-run`. Reroute log: `logs/auto_cloud_dispatch/<DATE>.log`. Opt-out: `AUTO_CLOUD_DISPATCH=0` env or `auto_cloud_dispatcher.disabled()` ctx manager.

**Launcher (`bin/claude-gdrive`):** exports 11 Claude Code feature flags (DAEMON/BG_SESSIONS/BRIDGE_MODE/ULTRAPLAN/TEMPLATES=1; KAIROS/COORDINATOR_MODE/BUDDY/VOICE_MODE/WEB_BROWSER=0) + appends `--max-budget-usd 50` per-session cap.

**Deliberately OFF (gated/unsafe leaked flags):** `KAIROS` (experimental), `COORDINATOR_MODE` (partial-built, conflicts with existing dispatcher), `BUDDY` (novelty), `USER_TYPE=ant` (Anthropic-internal, unauthorized), `DISABLE_COMMAND_INJECTION_CHECK` / `CLAUDE_CODE_ABLATION_BASELINE` / `DISABLE_INTERLEAVED_THINKING` (safety bypass — refuse), `CLAUDE_CODE_UNDERCOVER` (strips AI evidence from commits — ethical refuse), Anti-Distillation (poisons competitor training — ethical refuse).

**Active blockers / caveats:**
1. **Bug #29966** — Claude Code hardcodes `enablePromptCaching: false` for native Task spawns. Caching works only on router-driven `claude -p` calls (~36–39% theoretical input savings, not empirically measured) until Anthropic upstream fix.
2. **104 hooks reference-only** — opt in by editing `settings.json`. Top picks: `reports/install_disler_hooks_mastery.md` recommends SubagentStart, SessionEnd, PostToolUseFailure with snippets.
3. **16 MCP configs not auto-merged** — see `reports/mcp_configs_to_review.md`. Top-3 for trading/ML: `finance.json` (helium live stocks/options + chart-library 24M pattern embeddings, both free, no key), `data-science.json` (Jupyter MCP executes notebook cells directly), `research.json` (BGPT scientific papers, 50 free).
4. **sitecustomize.py** only patches sp500-mastery venv; other venvs (proactive-agent, etc.) won't auto-route.
5. **vwap_batch_resilient.py** lacks native `--dispatch-mode` (covered by monkey-patch fallback only).

**Restart command:** `"/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/bin/claude-gdrive" --continue` then `/sandbox` at the prompt. `--resume` opens picker if `--continue` selects wrong session.

**Reports written this session (all under `reports/`):** `install_florianbruniaux_ultimate_guide.md`, `install_rohitg00_toolkit.md`, `install_wesammustafa_everything.md`, `install_claude_howto.md`, `install_disler_hooks_mastery.md`, `cheatsheet_njengah_reference.md`, `mcp_configs_to_review.md`, `model_routing_100pct.md`, `anthropic_prompt_caching.md`, `AUTOMATION_VERIFICATION.md`.
