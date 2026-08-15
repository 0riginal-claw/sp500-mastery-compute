=== INHERITED MANDATES (workspace standing rules — apply automatically) ===

ALL agent classes (Claude main+sub, OpenClaw main+sub, Ollama main+sub) inherit
these mandates. Source-of-truth: ~/.zg/mandates.md. Edit ONE place; all entry
points read from here.

---

## 1. §3 FAN-OUT (HARD)

If estimated OR observed wall-clock > 5 min AND > 2 logical slices remain, you
MUST decompose and spawn N=2-6 helpers (native Agent general-purpose, or
mcp__plugin_fallback-agent_fallback__Task for grandchild MCP). Become the
aggregator. 20-min solo ceiling is absolute — self-terminate with partial
report if no fan-out by then. Every spawn prompt MUST carry one of
`# decomposition_plan:` OR `# scope_estimate_min:` OR `# inline_justification:`
near the top. Propagate this rule into every child brief.

## 2. §5a CLOUD-ROUTING (HARD)

85-90% of compute runs on Modal + gh_actions, NOT Mac. NEVER set
AUTO_CLOUD_DISPATCH=0 unless smoke test < 60s. Use
`cloud_dispatch.enqueue_job(ticker=X, strategy=Y, script=Z, ...)` for backtest
sweeps / ticker batches / parallel ML runs. Mac load cap = 12; if exceeded,
switch to cloud-route immediately. Heavy workers (orb_*, vwap_*, backtest_*,
momentum_*, catalyst_*) are auto-routed by sitecustomize.py — do not bypass.

## 3. §8 AUTO-SOLVE (HARD)

ANY error / blocker / failure / unknown library / HTTP 4xx-5xx / daemon crash /
CAPTCHA / auth fail / timeout / version conflict → IMMEDIATELY spawn 3 parallel
solvers (INTERNET + GITHUB + REPO-LOCAL) to research and apply fix. NEVER ask
user permission. NEVER stop the mission to explain. Escalate ONLY if all 3
return "no known fix" OR fix touches money / messages / credentials /
destructive actions. Propagate into every child. ENFORCED: errors → spawn 3
solvers per §8. The `autosolve-detect` (PostToolUse) + `autosolve-require`
(PreToolUse) hook pair registers pending issues to
`~/.claude/state/autosolve_pending.jsonl` and BLOCKS further tool use (exit 2)
if 3 distinct solver spawns (prompts tagged INTERNET / GITHUB / REPO-LOCAL) do
not occur within 5 turns. Bypass single call with `# autosolve_skip: <why>`.

## 4. AUTO-EXECUTE (HARD)

Never propose "I can do X if you want" or "say go to start" — silently execute
via tools / sub-agent / OpenClaw. Within standing authorization (read / write /
script / install / log / backup under My Drive), proceed without confirmation.
NEVER PUNT TO USER: do not stop the mission to ask permission for routine
research, MCP calls, paid OpenClaw/DeepSeek calls, installs, or sub-agent
spawns. Execute and report in proof-of-work. Only stop for destructive /
irreversible / money / messages / credentials / external-account actions.

## 5. NEVER RESTART (HARD)

Never suggest "restart Claude Code". Use the 4-tier hierarchy: (1) settings
hot-reload, (2) `/reload-plugins`, (3) sub-agent spawn for fresh config, (4)
SIGHUP wrapper (~1s, session preserved). See `docs/HOT_RELOAD_PATTERNS.md`.

## 6. TOKEN-SAVERS (HARD)

Every agent MUST use:
- **RTK** (`~/.local/bin/rtk`) — rate-limiter + token-budget on Bash output
- **LLMLingua compress** — prompts > 800 tokens MUST run
  `python scripts/llmlingua_compress.py --target-ratio 0.5` before passing to
  a nested helper. Auto-compress hook handles UserPromptSubmit at 2k;
  spawn-prompt-compress handles Task spawns.
- **caveman ultra-compressed mode** — 3-line summaries for grandchild output
- **CPR session-preserve** — `/compress` `/preserve` `/resume` for restarts
Bypass with `# tokensavers_skip: <why>`.

## 7. GUARDRAIL-GRADE DEFAULT (HARD)

Every new daemon / hook / signal-gen / feature MUST ship with the full
10-point guardrail wrapper:
1. launchd plist `KeepAlive=true` + `ThrottleInterval=30`
2. `state/<feature>/heartbeat.json` atomic write
3. PreToolUse freshness hook
4. SessionStart bootstrap hook
5. PostToolUse activity hook
6. SubagentStart inject hook
7. Stop validate hook
8. settings.json registration (idempotent)
9. `docs/<FEATURE>.md`
10. `backups/<feature>-pre-install-<DATE>/`

See `docs/EVERYTHING_GUARDRAIL.md`. Use `guardrail-ify` agent to automate.

## 8. KARPATHY-GUIDELINES PRE-FLIGHT (HARD)

Any ML / quant / backtest / hyperparam / signal-gen / live-paper-trade work
MUST surface assumptions + verifiable success criteria BEFORE execution.
Spawn prompt or Bash command must contain `# karpathy_checked: <summary>`.
Enforced by `karpathy-guidelines-block` PreToolUse hook (exit 2 if BLOCK
patterns matched AND no marker AND not a smoke/test).

## 9. OPENCLAW-FIRST ROUTING (HARD, non-Alpaca)

For non-Alpaca research, independent cross-check, large context (>200k), or
second-opinion review, prefer OpenClaw + DeepSeek over a Claude sub-agent:

```bash
bin/openclaw-gdrive agent --local --model deepseek/deepseek-v4-flash --json \
  --message "..."
```

Cheaper, independent, large-ctx. Use the `llm-task` plugin for
N-independent-slice fan-out (one llm-task call per slice, routed to dedicated
workspace by topic: trading-research, web-scraper, data-ingest).

MODEL-REASON HEADER: every Claude spawn prompt MUST include
`# model_reason: <why this model>` near the top. If you cannot articulate WHY
opus over sonnet (or sonnet over haiku), default to the CHEAPER option.
Mechanical work (grep/list/format/file-edit) = HAIKU; pure reasoning = OPUS;
default = SONNET.

## 10. REPO-INTEL-LAYER (HARD)

Before ANY non-trivial code edit / architecture change / skill install /
config write, scan:
- `registry/` (ticker lists, timelines, seeds)
- `repos-claude-clones/` (cloned catalog repos)
- `registry/repos-claude/QUICK_INDEX.md` (capability_map.json triggers)

Do not rewrite patterns that already exist. Use `grep -r` across `registry/` +
`repos-claude-clones/` before creating new files. Register new findings in
`registry/`.

## 11. UNIVERSAL-RESUME (HARD)

Every session bootstrap MUST inject:
- Latest `CLAUDE.md` head + recent session log summaries (via auto_resume hook)
- `~/.zg/mandates.md` (this file) as SessionStart `additionalContext`
- Current pacing state (`dashboard/pacing_state.json`) for model-routing bias

Sub-agents inherit via the `subagent-rules-inject` SubagentStart hook (reads
from `~/.zg/mandates.md`). OpenClaw inherits via `bin/openclaw-gdrive` (prepends
mandates to `--message`). Ollama inherits via `scripts/ollama_helper.py`
(`--system` defaults to mandates content unless explicitly overridden).

---

## PROOF OF WORK

End every task with: commands run, paths accessed, files changed, backups
created, OpenClaw usage, sub-agent usage (name/task/model/reason/result),
final status.

## SAFETY (non-negotiable)

- No permanent deletes (timestamped backup first).
- No credential / wallet / financial-doc modification.
- No money / messages / external account changes unless explicitly requested.
- No macOS system folder mods.

=== END INHERITED MANDATES ===
