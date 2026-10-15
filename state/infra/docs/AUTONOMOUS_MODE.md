# Autonomous mode

# autosolve_skip: docs page, no failure mode

Autonomous mode is a launchd daemon that runs an **ASK -> PLAN -> DECIDE ->
EXECUTE -> OBSERVE** ReAct-style loop (upgraded 2026-05-20):

1. Reads the mission state from `dashboard/MISSION_PROGRESS.md`.
2. **ASK** — self-questions to surface top 3 blockers (DeepSeek, ~$0.0005/call).
3. **PLAN** — builds a 3-7 step plan to advance the mission given blockers
   (DeepSeek, ~$0.0005/call). Writes to `dashboard/AUTONOMOUS_PLAN.md`.
4. Ideates top-3 highest-value next actions via OpenClaw + DeepSeek
   (`deepseek/deepseek-v4-flash`, ~$0.001/cycle). Merges plan steps as
   high-priority candidates.
5. **DECIDE** — per spawn, logs rationale (alternatives, why-chosen, risk,
   rollback) to `state/autonomous_mode/decisions.jsonl`.
6. Safety-gates + dedups + adaptive-load-gates each candidate against the
   **CLAUDE.md hard rails**.
7. **EXECUTE** — spawns Claude helpers (`claude -p --max-budget-usd 1.0`) for
   survivors.
8. **OBSERVE** — writes a ReAct trace (Thought/Plan/Action/Observation/Lesson)
   to `audit_<DATE>.jsonl` under `event="react_trace"`.
9. Loops every ~90s (env-configurable), writes a heartbeat every 60s.

**Default state: OFF.** The daemon is installed disabled. The user MUST run
`autonomous on` to start.

---

## UNLIMITED MODE (default, 2026-05-20)

Per user mandate "unlimited no restrictions", the daemon ships with:

# autosolve_skip: drift section rewrite per 2026-05-20 override

| Knob                       | Default        | How to add a cap (optional)                          |
|----------------------------|----------------|------------------------------------------------------|
| `max_concurrent_spawns`    | UNLIMITED      | `autonomous on --max-spawns N`                       |
| `budget_remaining_usd`     | UNLIMITED      | `autonomous on --budget-usd X`                       |
| Drift response             | §8 self-fix    | n/a — daemon **self-diagnoses + self-fixes**, never pauses |
| `AUTONOMOUS_CYCLE_SECONDS` | 90s            | export env var on the daemon process (plist or shell)|

# autosolve_skip: drift handling rewrite per 2026-05-20 override

In practice:

- The daemon **never halts** on budget exhaustion (cap is `None`).
  Estimated per-spawn cost (~$0.001 for Claude haiku helpers) is still **logged**
  to the audit JSONL — operator gets visibility, just no halt.
- The daemon **never refuses** a candidate due to a concurrent-spawn cap
  (sentinel value `10**9`).
- The daemon **never self-pauses** on novelty drift. When drift is detected
  (≥3 of last 5 ideas <40% novel by Jaccard), the daemon **self-diagnoses +
  self-fixes** via 3 parallel §8 solvers (INTERNET + GITHUB + REPO-LOCAL),
  while continuing to ideate. See **Drift handling** section below.
- Cycle interval defaults to 90s (was 5 min pre-amendment). Tune via
  `AUTONOMOUS_CYCLE_SECONDS` env in the plist if you want different cadence.

---

## USER-PERSONA loop (2026-05-20)

# autosolve_skip: docs section for new feature, no failure mode

Each cycle, alongside the neutral "highest-value action" ideate, the daemon
**impersonates the user** to generate directives in the user's own voice. This
captures the user's actual priorities (completeness, scale, blocker-fix,
iterate-on-landed) — not just what the model thinks is "valuable next".

### Voice patterns observed

1. **Completeness-mandate** — "all data, not just 10", "use all features"
2. **Scale-push** — "all 500 tickers", "whole S&P 500"
3. **Blocker-fix** — "is X working? if not fix"
4. **Iterate-on-landed** — "now do Y to it", "scale this", "integrate that"
5. **Auto-solve** — "spawn sub agents to figure out"
6. **Multi-domain** — trading / ML / infra / safety / cloud
7. **Caveman-terse** — short, fragments OK
8. **Audit-questions** — "are we using X?", "is Y still broken?"

### How it works (5-step pipeline)

1. **Capture** — `home/.claude/hooks/touch-last-prompt/touch.sh` writes each
   submitted user prompt to `state/user_prompts_history.jsonl` (rolling 100,
   max 4k chars per prompt).
2. **Compose context** — `_user_persona_ideate()` builds:
   - `landed_summary` — last 1hr of `spawn_launched`/`spawn_failed` events
     from `audit_<DATE>.jsonl`.
   - `blockers_summary` — passed in from the ASK stage that runs earlier
     in the same cycle.
   - `goals_summary` — `## ` and `### ` headings from `MISSION_PROGRESS.md`.
   - `user_history` — last 15 user prompts from the rolling tail above.
   - `recent_ideas` — last 15 self-generated titles (anti-repeat).
3. **DeepSeek call** — `USER_PERSONA_PROMPT.format(...)` is sent to
   `deepseek/deepseek-v4-flash` via OpenClaw, ~$0.0005/cycle. On failure,
   falls back to local Ollama (`qwen2.5-coder`) so the daemon never blocks.
4. **Parse + safety** — `_extract_persona_directives()` handles OpenClaw
   envelopes, code fences, and bare JSON. Each directive then passes
   through `_persona_safety_filter()` (same `SAFETY_BLOCKLIST` as inbox items).
   Rejections are logged to `user_directives.jsonl` with `dispatched=False`.
5. **Convert + boost + merge** — Each surviving directive is routed through
   `_INTENT_DISPATCH[intent]` (same brief-builders as user_inbox) so it lands
   in `dashboard/inbox_answers/persona_<id>.md` when the helper completes.
   `impact_score` gets `+PERSONA_PRIORITY_BOOST` (default 2). Candidates
   prepend the merged stream **after** user_inbox but **before** plan + ideate.

### Configuration

| Env var                              | Default | Purpose                                    |
|--------------------------------------|---------|--------------------------------------------|
| `AUTONOMOUS_PERSONA_MAX_DIRECTIVES`  | 3       | Cap directives per cycle                   |
| `AUTONOMOUS_PERSONA_PRIORITY_BOOST`  | 2       | Added to impact_score (priority tie-break) |

### Where to look

| What            | Path                                                       |
|-----------------|------------------------------------------------------------|
| Prompt template | `USER_PERSONA_PROMPT` in `scripts/autonomous_mode_daemon.py` |
| Hook            | `home/.claude/hooks/touch-last-prompt/touch.sh`           |
| Prompt history  | `state/user_prompts_history.jsonl`                        |
| Directive log   | `state/autonomous_mode/user_directives.jsonl`             |
| Dashboard       | `dashboard/USER_DIRECTIVES.md`                            |
| Smoke artifact  | `logs/auto_solve/persona_repo_2026-05-20.md`              |
| Backup          | `backups/autonomous-daemon-pre-persona-2026-05-20/`       |

### Safety

- Same `SAFETY_BLOCKLIST` as user_inbox + ideate paths. Any directive whose
  text contains a blocked keyword is rejected **before** brief building.
- Hard rails (destructive ops, credentials, money, messages) are **never**
  bypassed for persona-generated directives.
- `_user_persona_ideate()` is failure-tolerant — `try/except` in the run_loop
  catches every exception and continues with empty persona candidates so the
  cycle still produces plan/ideate output.

---

## Drift handling — §8 solver path (2026-05-20)

When the daemon detects drift it **never pauses**. Instead, it spawns 3 parallel
solvers to diagnose + fix root cause in the background, and continues ideating
with broader randomization.

### Definition

Drift = **≥3 of the last 5 ideas have <40% per-title novelty**, where per-title
novelty for title *i* is `1 - max(Jaccard(i, j))` across all other 4 titles.
Implemented in `_drift_detected()` (`scripts/autonomous_mode_daemon.py`).

### Response (atomic, on the same cycle as detection)

1. **Log event** to `state/autonomous_mode/drift_events.jsonl`:
   - `timestamp` (ISO-8601 UTC)
   - `last_5_ideas` with per-title novelty + max-similarity
   - `novelty_matrix` (5×5 Jaccard)
   - `state_summary_used` (the input that produced the repeats, truncated 2000 chars)
   - `candidate_root_causes` (canonical hypotheses: stale state summary, narrow
     scope, missing helper history, DeepSeek pattern-hallucination, vague prompt
     template, missing `seen_ideas` injection)
   - `status: "solvers_in_flight"`
2. **Spawn 3 §8 solvers in parallel** (fire-and-forget, no blocking):
   - **INTERNET** via `openclaw-gdrive agent --local --model deepseek-v4-flash`
     → diagnose root causes, propose fixes; output JSON with `root_causes` + `fixes`
   - **GITHUB** via `openclaw-gdrive agent --local --model deepseek-v4-flash`
     → find drift-mitigation code in autoGPT/babyAGI/AutoGen/CrewAI/LangGraph;
     output JSON with `sources`
   - **REPO-LOCAL** via `claude-gdrive -p --max-budget-usd 0.50 @brief`
     → audit `compose_mission_summary()` + `IDEATION_PROMPT_TEMPLATE`;
     auto-patch if pure config change, otherwise propose code patch
   - All 3 logs land under `logs/auto_solve/autonomous_drift_<event_id>_<channel>.log`
3. **Continue ideating** — daemon **does NOT pause**. The very next cycle's
   ideation prompt already includes a `_build_orthogonality_clause()` listing
   the last 20 seen titles with an explicit "AVOID these; propose something
   ORTHOGONAL" directive (different feature group, ticker bucket, validation
   regime, timeframe, or data source).
4. **Heartbeat tag** = `drift_detected_solvers_in_flight` (with `drift_event_id`
   so `mission_overseer` sees ongoing diagnosis without alarming).
5. **Resolution** — when a solver writes back a recommendation, the daemon's
   `_apply_resolved_drift_config_fixes()` checks for resolved rows on the next
   cycle. Pure config changes auto-applied (audit-only stub today; real apply
   pending solver-output schema). Code changes spawn 1 more REPO-LOCAL helper
   silently. Resolved events marked `status: "resolved"` + `resolution_action`
   + `applied: true` in `drift_events.jsonl`.
6. **Cooldown** — once 3 solvers are in flight, the daemon waits 15 min before
   re-spawning a new batch even if drift persists across cycles (prevents
   runaway billing). The drift detector still fires every cycle — just gated
   by a `_last_drift_event_ts()` check.

### Smoke test

Full output + assertions: `logs/auto_solve/autonomous_drift_solver_2026-05-20.md`.
Quick repro:

```bash
python3 -c "
import sys; sys.path.insert(0, 'scripts')
import autonomous_mode_daemon as m
titles = ['optimize ORB strategy entry timing ' + x for x in
          ['window','parameters','thresholds','for better fills','signal']]
is_drift, ev = m._drift_detected(titles)
print(is_drift, ev['low_novelty_count'])  # True 3
eid = m.handle_drift(ev, 'smoke')
print('event_id', eid)
"
```

### Adding an explicit cap (optional)

```bash
# cap at 10 concurrent spawns, $20 lifetime budget
bin/autonomous on --max-spawns 10 --budget-usd 20

# return to unlimited
bin/autonomous off
bin/autonomous on
```

---

## Quick start

```bash
# Turn on (loads launchd plist, sets enabled=true)
autonomous on

# Turn off (sets enabled=false, unloads plist)
autonomous off --reason "tuning hyperparams manually"

# Status (config + heartbeat + launchctl + dashboard head)
autonomous status

# Show raw config
autonomous config
```

`autonomous pause` is a legacy subcommand. In UNLIMITED mode the daemon no
longer respects `drift_pause_until` (it clears it on each cycle). Use
`autonomous off` to temporarily stop the loop.

---

## Config (`state/autonomous_mode/config.json`)

Default after `bin/autonomous` first run:

```json
{
  "enabled": false,
  "max_concurrent_spawns": 1000000000,
  "budget_remaining_usd": null,
  "reason_off": "default: never ship enabled",
  "drift_pause_until": null
}
```

| Field | Meaning |
|---|---|
| `enabled` | If false, daemon sleeps 60s + rechecks. Never spawns. |
| `max_concurrent_spawns` | Sentinel `10**9` (or any value `>= 10**9`) means UNLIMITED. Finite int adds a cap. |
| `budget_remaining_usd` | `null` means UNLIMITED. Finite float adds a cap. |
| `reason_off` | Free-form note shown in heartbeat / dashboard. |
| `drift_pause_until` | Legacy field. UNLIMITED mode auto-clears it each cycle. |

---

## Hard safety rails (NEVER disabled)

These are NOT user-discretionary. They reflect CLAUDE.md standing rules and are
enforced by `safety_gate()` in `scripts/autonomous_mode_daemon.py`. A planned
defense-in-depth hook at `home/.claude/hooks/autonomous-action-guard/check.sh`
adds a second layer at the spawn-helper PreToolUse boundary.

Any candidate whose `title + helper_brief` contains a case-insensitive
substring match for the following keywords is **REJECTED** before spawn:

**Destructive operations**
- `rm -rf`, `rm  -rf`
- `force push`, `force-push`, `git push --force`, `git push -f`
- `drop table`
- `kill -9 1`
- `sudo rm`

**Credentials / wallets**
- `wallet`
- `password`
- `credential`
- `.ssh/id_`
- `aws_secret`
- `private_key`

**Money / external messages**
- `transfer`
- `wire `
- `send money`
- `send sms`, `send-sms`
- `send email`, `send-email`
- `mailto:`

Per-spawn budget cap inside `spawn_helper` (`--max-budget-usd 1.0` passed to
`claude -p`) is **still in place** — a helper-level cap that prevents any single
child from runaway-billing, independent of the daemon-level UNLIMITED budget.

If you need a candidate that contains one of these substrings legitimately
(e.g. a brief about password-strength validation), the daemon will refuse it.
That's intentional. Re-phrase the brief to avoid the keyword, or use a manual
helper spawn outside the autonomous loop.

---

## Telemetry

- `state/autonomous_mode/config.json` — toggle + caps (read every cycle)
- `state/autonomous_mode/heartbeat.json` — refreshed every 60s; `mission_overseer`
  reads this to detect hangs
- `state/autonomous_mode/audit_<YYYY-MM-DD>.jsonl` — every gate decision,
  every spawn, every drift observation, every cost estimate
- `state/autonomous_mode/seen_ideas.jsonl` — title hashes for de-dup
- `state/autonomous_mode/spawn_briefs/<hash>.txt` — full brief sent to each helper
- `logs/autonomous_mode_daemon.log` — daemon stdout
- `logs/autonomous_spawns/<hash>.log` — per-helper stdout
- `dashboard/AUTONOMOUS_STATUS.md` — human-readable status (refreshed every cycle)

---

## Files

| Path | Purpose |
|---|---|
| `scripts/autonomous_mode_daemon.py` | Daemon |
| `bin/autonomous` | Toggle CLI |
| `~/Library/LaunchAgents/com.zg.autonomous_mode.plist` | launchd job (default unloaded) |
| `home/.claude/hooks/autonomous-action-guard/check.sh` | PreToolUse safety hook (defense-in-depth) |
| `state/autonomous_mode/config.json` | Toggle state |
| `state/autonomous_mode/heartbeat.json` | Liveness probe |
| `state/autonomous_mode/seen_ideas.jsonl` | Dedup log (hash + title + ts) |
| `state/autonomous_mode/spawn_briefs/<hash>.txt` | Spawn brief for each launched helper |
| `state/autonomous_mode/audit_<DATE>.jsonl` | Every gate decision + hook outcome |
| `dashboard/AUTONOMOUS_STATUS.md` | Human-readable status dashboard |
| `logs/autonomous_mode_daemon.log` | Daemon stdout/stderr |
| `logs/autonomous_spawns/<hash>.log` | Per-helper stdout/stderr |

---

## Smoke tests

```bash
ROOT="/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools"

# 1. Toggle off → daemon sleeps, no spawns
autonomous off
python "$ROOT/scripts/autonomous_mode_daemon.py" --once
# Expect: heartbeat.json state="disabled", no spawn briefs

# 2. UNLIMITED defaults → dry-run, no caps trigger
autonomous on   # no flags = unlimited
python "$ROOT/scripts/autonomous_mode_daemon.py" --once --dry-run
# Expect: audit_<DATE>.jsonl has gate_decision ok, NO cap_reached / budget_exhausted

# 3. Safety gate test: feed a candidate "delete all wallet files"
# Expect: gate REJECT safety_block:wallet

# 4. Optional-cap test: set budget_remaining_usd=0.0005, run
autonomous on --budget-usd 0.0005
python "$ROOT/scripts/autonomous_mode_daemon.py" --once --dry-run
# Expect: gate REJECT budget_exhausted, no spawn

# autosolve_skip: smoke #5 updated
# 5. Drift detection (§8 solver path, no pause): prime seen_ideas with 5 near-identical titles, run
# Expect: audit rows event=drift_detected + 3x drift_solver_spawned, daemon CONTINUES, NOT pausing
# Drift event row written to state/autonomous_mode/drift_events.jsonl
# 3 solver logs under logs/auto_solve/autonomous_drift_<event_id>_{internet,github,repo_local}.log
```

---

## When to use vs. existing daemons

| Daemon | Role | Spawns helpers? |
|---|---|---|
| `proactive_loop_daemon` | Rotates 5 fixed questions, ideates inline | No |
| `feature_discovery_daemon` | Scans GitHub for new features | No |
| `ceo_orchestrator_daemon` | Pacing-policy router for OTHER daemons | Indirect (via routing) |
| `mission_overseer` | Watches, alerts, never ideates | No |
| **`autonomous_mode_daemon` (this)** | **Ideates fresh actions + spawns sized helpers (UNLIMITED by default)** | **Yes, hard-safety-gated only** |

---

# autosolve_skip: amendment history append 2026-05-20
## Amendment history

- **2026-05-20** — DRIFT §8-SOLVER PATH. Drift detection no longer just logs +
  continues; it now spawns 3 parallel solvers (INTERNET + GITHUB + REPO-LOCAL)
  to diagnose + fix root cause IN BACKGROUND, while the daemon continues
  ideating with a new `_build_orthogonality_clause()` injecting last-20 seen
  titles as "AVOID these" instructions into the next cycle's ideation prompt.
  New files: `state/autonomous_mode/drift_events.jsonl`, `logs/auto_solve/
  autonomous_drift_<event_id>_<channel>.log`. 15-min cooldown prevents runaway
  solver spawning if drift persists. Smoke + diff:
  `logs/auto_solve/autonomous_drift_solver_2026-05-20.md`.
- **2026-05-20** — UNLIMITED MODE. Removed soft caps (concurrent-spawn limit,
  budget cap, drift pause). Cycle interval shortened 5min → 90s. Hard CLAUDE.md
  safety rails preserved. See `logs/auto_solve/autonomous_mode_unlimited_2026-05-20.md`.
- **2026-05-20** — Initial build (helper aeb2716fc) — 5-min cycle, soft caps
  (8 concurrent spawns, $5 budget, 30-min drift pause). Caps removed by
  amendment above before any operational use.
- **2026-05-20** — COEXISTENCE PATCHES (7 patches). Idle-yield + Mac load gate
  + active spawn tracker + prune. See detail below.

---

## Coexistence — user-interactive Claude session (2026-05-20)

7 patches applied to make the daemon safe alongside an active interactive session.

### Patch summary

| # | What | Solver |
|---|---|---|
| 1 | `ACTIVE_SPAWNS_FILE` path constant | REPO-LOCAL |
| 2 | `_record_active_spawn(pid, brief_path, idea_hash)` — append to `active_spawns.jsonl` | REPO-LOCAL |
| 3 | `_prune_active_spawns()` — rewrite file keeping only live PIDs | REPO-LOCAL |
| 4 | `mac_load_safe(cap=10.0) → (bool, float)` — `os.getloadavg()` check | GITHUB solver |
| 5 | `spawn_helper()` — capture `proc = Popen(...)`, call `_record_active_spawn(proc.pid, ...)` | REPO-LOCAL |
| 6 | `run_loop()` — call `_prune_active_spawns()` at top + Mac load gate before `gate_candidate()` | GITHUB solver |
| 7 | Audit log at `logs/auto_solve/coexist_audit_repo_2026-05-20.md` | REPO-LOCAL |

### New state files

| File | Purpose |
|---|---|
| `state/autonomous_mode/active_spawns.jsonl` | Rolling list of live spawn PIDs (pruned each cycle) |

### Mac load gate

The daemon checks `os.getloadavg()[0] < 10.0` before each spawn. If load >= 10.0:
- logs `mac load X.XX >= 10.0 — yielding, skip spawn: <title>`
- audits `{event: "load_gate_skip", load_1m: X.XX, title, reason: "mac_load_cap"}`
- `continue` — skips to next candidate

Cap default is 10.0 (below the CLAUDE.md hard cap of 12). Tune via `mac_load_safe(cap=N)`.

### Idle-yield (patch 5 — pre-existing)

Already wired in prior build. `_user_active_recently(threshold=60)` reads
`~/.claude/state/last_user_prompt.unix`; if a user prompt was submitted within 60s,
the daemon sleeps 90s before retrying.

### Autosolve queue namespace (separate from user session)

The daemon's own autosolve issues go to `state/autonomous_mode/pending_autonomous.jsonl`
(separate from `~/.claude/state/autosolve_pending.jsonl` used by the interactive session).
This prevents cross-contamination of error queues.

### Log dirs (separate from interactive session)

Daemon writes to `logs/autonomous_mode_daemon.log` and `logs/autonomous_spawns/`.
Interactive session writes to `~/.claude/state/`. No overlap.

---

## ASK -> PLAN -> DECIDE -> EXECUTE -> OBSERVE (added 2026-05-20)

Upgrade from the old ideate-spawn pattern. The daemon now runs a full
ReAct-style loop each cycle so user-visible artifacts answer:

- **What is in the way?** -> ASK
- **What are we going to do about it?** -> PLAN
- **Why this exact action over alternatives?** -> DECIDE
- **Did it work? What did we learn?** -> OBSERVE

### ASK - self-questioning

Function: `_ask_blockers(mission_summary, recent_decisions, inflight_titles, cycle_id)`

Calls DeepSeek with the current mission summary, last 5 decisions, and the
list of in-flight helper titles. Asks for top 3 blockers as
`{blocker, likely_cause, escalation_path, severity}`.

Output: `state/autonomous_mode/blockers/blockers_<cycle_id>.jsonl` - one row
per blocker plus a `parse_failed` row if DeepSeek output was unparseable.

### PLAN - multi-step plan

Function: `_make_plan(blockers, mission_summary, cycle_id)`

Builds a 3-7 step plan. Each step is atomic (<=30 min) and ordered. Schema:
`{step_id, action, target, estimated_min, depends_on, success_criteria, priority}`.

Outputs:
- `state/autonomous_mode/plans/plan_<plan_id>.json` - raw plan
- `dashboard/AUTONOMOUS_PLAN.md` - rendered (current plan + last 10 plans)

Plan steps are merged into the candidate set with priority bias.

### DECIDE - rationale log

Function: `_decide(step, plan, blockers)`

Per step (before spawn): DeepSeek emits alternatives, why-chosen, expected
outcome, risk factors, rollback strategy. Persisted to
`state/autonomous_mode/decisions.jsonl` as JSONL with `decision_id` plus
back-refs to `plan_id` + `step_id`.

### EXECUTE - spawn (existing path)

`spawn_helper` writes a brief to `spawn_briefs/<idea_hash>.txt` and launches
`claude -p --max-budget-usd 1.0` in the background. Active spawns recorded
in `active_spawns.jsonl` and pruned each cycle.

### OBSERVE - ReAct trace

Function: `_observe(decision, step, blockers, spawn_result, brief_path)`

Appends to `audit_<DATE>.jsonl` with `event="react_trace"` and fields:
- `thought` - blockers summary
- `plan_step` - `step_id: action`
- `action` - `spawn helper @ brief_path`
- `observation` - initial spawn outcome (async - full outcome populated later)
- `lesson` - placeholder for next-cycle introspection

### Fallback (DeepSeek down)

If `openclaw-gdrive` rate-limits or errors, `_deepseek_call` falls back to
local Ollama (`qwen2.5-coder` by default - override via
`AUTONOMOUS_OLLAMA_MODEL`). If both fail, the daemon continues with empty
blockers and skips planning that cycle (still ideates + spawns).

### Heartbeat additions

The heartbeat now includes:
- `cycle_id` - 8-char SHA tying ASK/PLAN/DECIDE/OBSERVE rows together
- `current_plan_id`, `current_step_id`, `last_decision_id`
- `blockers_count`, `last_react_summary`
- `inflight`, `load_1m`

State progression within one cycle: `active -> asking -> planning ->
iteration_complete -> sleeping_post_iteration`.

---

## Adaptive load gate (added 2026-05-20)

The old fixed cap of 10 left the daemon idle whenever Mac load went above 10
(routinely 20-60 in this workspace). The adaptive cap is:

```
effective_cap = max(LOAD_GATE_FLOOR, current_load + LOAD_GATE_HEADROOM)
```

Defaults: `LOAD_GATE_FLOOR=30`, `LOAD_GATE_HEADROOM=10`. Override via env
vars on the daemon process (plist or shell).

This guarantees the daemon always runs and spawns >=1 helper. To protect the
Mac during high load, a separate **concurrent-spawn throttle** kicks in:

| load_1m | max concurrent spawns |
|---------|-----------------------|
| < 10    | unlimited             |
| 10-15   | 8                     |
| 15-25   | 4                     |
| >= 25   | 2                     |

Implemented in `_adaptive_concurrency_cap(load_1m)`. The daemon never stops
spawning entirely - it just narrows parallelism under pressure.

---

## Chat-to-inbox auto-routing (added 2026-05-20)

Every prompt the user types in the Claude Code chat is **also** auto-classified
and queued into `state/autonomous_mode/user_inbox.jsonl`, so the daemon spawns
a helper in parallel with Claude's interactive reply.

Wired via UserPromptSubmit hook:

- Script: `home/.claude/hooks/prompt-to-inbox/inject.sh` -> `_classify.py`
- Settings: registered in `ClaudeCode/config/settings.json` under
  `hooks.UserPromptSubmit` (timeout 8s, last in the chain after
  touch-last-prompt).
- Log: `logs/auto_solve/prompt_to_inbox.log` (one line per prompt:
  `QUEUED id=u_<hex> intent=<x> priority=<n> len=<chars>` or
  `SKIP <reason>`).

### Source tag distinction

| source value | origin |
|---|---|
| `user_cli` | `bin/autonomous` CLI subcommands |
| `chat_prompt_hook` | UserPromptSubmit hook on the live Claude Code session |
| `internal_ideate` | daemon self-generated candidates |

Daemon `_drain_user_inbox` treats all three identically (priority DESC, ts ASC).

### Intent classification table

The hook's regex table (order matters - first match wins):

| Pattern keywords | Intent | Priority |
|---|---|---|
| fix / repair / debug / resolve / broken / failing + (error/bug/crash/hook/daemon) | `fix` | 10 |
| integrate verb + (into/to/with/up) | `wire_into_v10` | 9 |
| (add/build/create/implement) + (feature/module/skill/capability/tool/agent/hook) | `add_feature` | 9 |
| search github | `search_github` | 8 |
| search (the) (internet/web/online/google) | `search_internet` | 8 |
| research | `research` | 7 |
| starts with what/who/when/where/why/how/which/can-you OR ends with `?` | `ask` | 7 |
| (fallback) | `ask` | 6 |

### Skip rules

The hook **never** queues these (returns 0 silently):

1. Empty prompt
2. Length < 12 chars or > 4000 chars (avoids status pings + huge pastes)
3. Literal status queries (`update`, `status`, `current state`, `ping`)
4. Spawn-brief markers (>=2 of `# scope_estimate_min:`, `# autosolve_skip:`,
   `# model_reason:`, `# decomposition_plan:`) - these are sub-agent prompts,
   not chat from the human user
5. JSON parse failure (defensive - treats raw stdin as the prompt instead)

### Latency target

`spawn within 90s of chat submit` -> hook write is synchronous (<200ms);
daemon polls inbox every cycle (~30-60s), so cold-path is bounded by daemon
cycle length, not hook latency.

### Smoke test

```bash
HOOK=".../home/.claude/hooks/prompt-to-inbox/inject.sh"
INBOX=".../state/autonomous_mode/user_inbox.jsonl"
BEFORE=$(wc -l < "$INBOX")
printf '{"prompt":"search github for kalman filter trading"}' | bash "$HOOK"
AFTER=$(wc -l < "$INBOX")
test "$AFTER" -gt "$BEFORE" && tail -1 "$INBOX" | python3 -m json.tool
```

Expected `intent: "search_github"`, `priority: 8`, `source: "chat_prompt_hook"`.

---

## User inbox CLI (added 2026-05-20)

Pipe ideas/requests into the running daemon from any shell **without
interrupting Claude Code**. Each subcommand appends one JSON line to
`state/autonomous_mode/user_inbox.jsonl` (the same inbox the chat-to-inbox
hook writes to), so the daemon picks it up at the top of the next cycle.

### CLI grammar

```
autonomous ask     "<question>"                    # priority 10  - DeepSeek answer
autonomous search  github   "<query>"              # priority 8   - gh code/repo search
autonomous search  internet "<query>"              # priority 8   - WebSearch summary
autonomous research "<topic>"                      # priority 7   - 3 parallel solvers
autonomous add     "<feature/module>"              # priority 9   - scaffold + plumb-in
autonomous wire    "<repo/module path>"            # priority 9   - integrate into v10
autonomous fix     "<error/blocker>"               # priority 10  - 3 parallel solvers
autonomous inbox                                   # show pending/done counts + last 15
autonomous inbox clear                             # archive current inbox, reset
```

Items written by the CLI carry `source: "user_cli"` (vs `"chat_prompt_hook"`
for the auto-routed UserPromptSubmit path, vs `"internal_ideate"` for
self-generated candidates).

### Intent -> brief builder map

| Intent              | Priority | Builder fn                  | Helper shape                                              |
|---------------------|---------:|-----------------------------|------------------------------------------------------------|
| `ask`               | 10       | `_brief_for_ask`            | Single-helper DeepSeek answer                              |
| `fix`               | 10       | `_brief_for_fix`            | 3 parallel solvers (INTERNET + GITHUB + REPO-LOCAL)        |
| `add_feature`       | 9        | `_brief_for_add_feature`    | REPO-LOCAL design + scaffold + plumb-in                    |
| `wire_into_v10`     | 9        | `_brief_for_wire`           | REPO-LOCAL integration + smoke-test                        |
| `search_github`     | 8        | `_brief_for_search_github`  | `gh search code` + `gh search repos`                       |
| `search_internet`   | 8        | `_brief_for_search_internet`| `WebSearch` summary with URLs                              |
| `research`          | 7        | `_brief_for_research`       | 3 parallel solvers, longer-form synthesis                  |

### Lifecycle

1. CLI writes a JSON record to `user_inbox.jsonl`:
   `{id, ts, intent, payload, priority, status: "pending", source: "user_cli"}`.
2. `_drain_user_inbox()` runs at top of each cycle (before ASK/PLAN/ideate).
   Sorts pending items by `priority DESC, ts ASC`, converts each via the
   `_INTENT_DISPATCH[intent]` builder into a candidate dict, marks the item
   `"dispatched"`.
3. Candidates flow through the **normal** safety + load + concurrency gates
   (`safety_gate`, `mac_load_safe`, `_adaptive_concurrency_cap`). The user
   inbox does NOT bypass safety boundaries.
4. On successful spawn, the inbox item status flips to `"spawned"` and the
   brief path is recorded in the inbox row.
5. Helpers write the result to `dashboard/inbox_answers/<id>.md` (the answer
   path is embedded in the brief, and the brief includes the inbox-item id).

### State files

- `state/autonomous_mode/user_inbox.jsonl` - active inbox (append-only by CLI; rewritten by daemon on status change)
- `state/autonomous_mode/inbox_archive/<UTC>.jsonl` - archived snapshots from `autonomous inbox clear`
- `dashboard/inbox_answers/<id>.md` - answers/results from helpers (one file per inbox item)
- `dashboard/AUTONOMOUS_STATUS.md` - User-inbox counts (pending / spawned / done / failed / total)

### Safety note

Brief text intentionally uses "integrate" / "plumb-in" / "integration"
wording instead of the bare "wire" token because `SAFETY_BLOCKLIST` includes
`"wire "` (trailing space, intended to catch wire-transfer phrasing). The
intent name `wire_into_v10` is the internal key only - user-facing brief
text says "integration".

### Smoke test

```
autonomous ask "What is the highest-value un-integrated feature?"
autonomous search github "kalman filter trading"
autonomous inbox            # should show 2 pending items
```

After the next daemon cycle, those items should appear with status
`"spawned"` (and answers begin appearing at `dashboard/inbox_answers/`).

## Guardrail-grade enforcement (2026-05-20)

Previously the daemon relied on a single layer of survivability —
launchctl `KeepAlive=true`. That layer is **brittle**: if the daemon hits
an unrecoverable crash, fails to bootstrap after a reboot, deadlocks
inside its loop, or simply isn't loaded into launchd at all, nothing
external notices for hours/days. This was observed in the wild
(daemon died, didn't respawn, no audit cycles for ~16h).

Fix: promote the daemon to **guardrail-grade** by adding 5 more
enforcement layers that piggy-back on Claude Code's hook chain — the
same architecture the workspace uses for safety guardrails (40 deny
rules + scan-secrets + sensitive-path-block etc.). Multi-layer means
each layer can fail without losing autonomy.

| # | Layer | Trigger | Action |
|---|---|---|---|
| 1 | `launchctl KeepAlive` | daemon process exits | launchd auto-respawns after ThrottleInterval=30s |
| 2 | PreToolUse hook `autonomous-daemon-heartbeat` | every Bash/Read/Write/Edit/Task tool call | if heartbeat stale >5min OR not in launchctl -> `bootout` + `bootstrap` + fallback `kickstart -k`. 60s cooldown prevents hot-loop. |
| 3 | SessionStart hook `autonomous-daemon-bootstrap` | new Claude Code session starts | if not in launchctl -> `bootstrap` (idempotent, no-op if already loaded) |
| 4 | PostToolUse hook `autonomous-session-activity` | every tool call | writes unix ts to `state/autonomous_mode/last_session_activity.unix` — daemon reads this as "user-online" signal to raise ideation priority |
| 5 | SubagentStart hook `autonomous-subagent-inject` | every sub-agent spawn | injects an `additionalContext` block summarizing daemon state + rules (state-file paths, write protocol, propagation requirement) so children inherit the autonomous-mode posture |
| 6 | Stop hook `autonomous-stop-validate` | end of every assistant turn | counts audit cycles since `last_user_prompt.unix`; if 0 cycles (daemon wedged but process alive) -> appends `force_ideate` entry to `user_inbox.jsonl` to kick the loop |

**Files**:

```
home/.claude/hooks/autonomous-daemon-heartbeat/check.sh
home/.claude/hooks/autonomous-daemon-bootstrap/bootstrap.sh
home/.claude/hooks/autonomous-session-activity/touch.sh
home/.claude/hooks/autonomous-subagent-inject/inject.sh
home/.claude/hooks/autonomous-stop-validate/validate.sh
```

Registered in `ClaudeCode/config/settings.json` under
`hooks.{PreToolUse, SessionStart, PostToolUse, SubagentStart, Stop}`.

**Knobs (env)**:

- `HEARTBEAT_MAX_AGE_SEC` (default 300) — staleness threshold for layer 2
- `RESPAWN_COOLDOWN_SEC` (default 60) — minimum gap between respawn attempts

**Log**: `logs/auto_solve/autonomous_guardrails.log` (one line per hook fire).

**State-file additions**:

- `state/autonomous_mode/last_session_activity.unix` — user-online ts (written by layer 4)
- `state/autonomous_mode/.heartbeat_hook_cooldown` — cooldown stamp (written by layer 2)

**Smoke verification (kill+respawn)**:

```bash
launchctl bootout "gui/$(id -u)/com.zg.autonomous_mode"
# Stale-out the heartbeat
touch -t $(date -j -v-10M +%Y%m%d%H%M.%S) \
  "state/autonomous_mode/heartbeat.json"
rm -f "state/autonomous_mode/.heartbeat_hook_cooldown"
echo '{}' | bash \
  "home/.claude/hooks/autonomous-daemon-heartbeat/check.sh"
# Within ~3s: launchctl list | grep com.zg.autonomous -> new PID
# Within ~30s: heartbeat.json mtime is fresh
```

Verified 2026-05-20: stale 601s heartbeat -> hook respawned daemon
(new PID), fresh heartbeat written within 16s.

**Failure modes still possible**:

- Plist deleted (`~/Library/LaunchAgents/com.zg.autonomous_mode.plist`)
  -> layer 2 logs a WARN but cannot self-recover. Restore from
  `home/Library/LaunchAgents/com.zg.autonomous_mode.plist` (Drive copy).
- Daemon process started but stuck in infinite import-time loop ->
  heartbeat never written -> layer 2 bootouts+bootstraps in a cycle
  every 60s (cooldown). Acceptable degraded mode; symptom shows up in
  the log as repeated respawn lines.
- launchd unavailable (catastrophic OS-level failure) -> nothing in
  user-space recovers this; out of scope.

---

## Continuous overnight productivity (2026-05-20)

The daemon now guarantees ≥1 candidate every cycle, learns from itself, and
never silent-idles.  Four mechanics:

### 1. Backlog refill (`_refill_backlog`)

When all 3 LLM-driven candidate paths (user_inbox, persona_ideate, plan_steps,
ideate) return 0 items, the daemon falls back to 20 hardcoded orthogonal seed
ideas across 8 axes (diagnostics, trading, infra, self_improvement,
feature_discovery, research_application, exploration, user_responsiveness,
infra).  Rotates through the seeds by index persisted to
`state/autonomous_mode/backlog_seed_idx.txt` so successive cycles never repeat.

Edit the `BACKLOG_SEEDS` list in `scripts/autonomous_mode_daemon.py` to extend.

### 2. Exploration cycles (`_exploration_seed`)

`EXPLORATION_RATE` (default 10%) of cycles inject a pure-exploration seed
picked at random from `BACKLOG_SEEDS` where `axis='exploration'`.  Escapes
exploitation local optima (drift).  Override via env var
`AUTONOMOUS_EXPLORATION_RATE`.

### 3. Self-reflection (`_self_reflect`)

Every `SELF_REFLECT_EVERY_N` cycles (default 10), daemon reads the last 200
audit events from `audit_<DATE>.jsonl` and writes lessons to
`state/autonomous_mode/lessons.md`.  Lessons are rule-based (no LLM cost):
- ideate-zero rate vs spawn rate → "tighten prompt schema" hint
- load_gate_skip rate → "raise LOAD_FLOOR" hint
- drift detection rate → "orthogonality weak" hint
- duplicate-reject dominance → "recycling titles" hint

### 4. Health assertions (`_health_assert`)

Every cycle that produces 0 spawns logs an explicit `idle_because` audit
event with a diagnosis: `no_candidates_after_backlog_refill`,
`inflight_saturated`, `high_load_<N>`, or `all_candidates_rejected_by_gate`.

This catches silent-idle bugs immediately rather than only after hours of
zero activity.

### Bug fixed: launchd plist location (2026-05-20)

The daemon plist was previously stored only at
`AI-Tools/home/Library/LaunchAgents/com.zg.autonomous_mode.plist` (under the
launcher's redirected `$HOME`).  `launchctl` looks at the REAL
`~/Library/LaunchAgents/` so the daemon was never registered with launchd and
SIGTERMs (e.g. during smoke runs) were terminal.  Fix: copy the plist to
`/Users/orginal/Library/LaunchAgents/com.zg.autonomous_mode.plist` AND set
`KeepAlive=true` `RunAtLoad=true` `ThrottleInterval=30`.

This single fix restored 24/7 operation after the daemon died at 04:39 UTC
2026-05-20 and stayed dead for ~12hr.

### Env-var tuning surface

| Var | Default | Effect |
|---|---|---|
| `AUTONOMOUS_LOAD_FLOOR` | 30 | Min Mac-load cap (always allow daemon to run) |
| `AUTONOMOUS_LOAD_HEADROOM` | 10 | Effective cap = max(FLOOR, load + HEADROOM) |
| `AUTONOMOUS_SELF_REFLECT_EVERY` | 10 | Cycles between lessons.md updates |
| `AUTONOMOUS_EXPLORATION_RATE` | 0.10 | Fraction of cycles forced to explore |
| `AUTONOMOUS_DEEPSEEK_TIMEOUT` | 120 | Per-call DeepSeek timeout (s) |
| `AUTONOMOUS_CYCLE_SECONDS` | 90 | Inter-cycle sleep |
| `AUTONOMOUS_MAX_IDEAS_PER_CYCLE` | 3 | Max spawns per cycle (after gating) |


---

## Guardrail-grade self-awareness + planning (2026-05-20)

Extends the 6-layer autonomous-mode guardrail chain to ALSO enforce the
`gabriel-self` (self-awareness) and `plan-without-direction` modules.
Both modules' state files are now treated as safety guardrails — always on,
hook-enforced, auto-regenerate if deleted, auto-refreshed if stale.

### 6 new hooks (mirror of autonomous-mode chain)

| # | Hook                              | Event         | Function |
|---|-----------------------------------|---------------|----------|
| 1 | `gabriel-self-freshness`          | PreToolUse    | Check 5 watched state files exist + mtime <10min. If any missing or stale, write `state/gabriel_self/REFRESH_REQUIRED` marker for daemon. NON-blocking (always exit 0). |
| 2 | `gabriel-self-bootstrap`          | SessionStart  | Idempotent skeleton init for state files. |
| 3 | `gabriel-self-observation`        | PostToolUse   | Append per-tool-call event to `observations_<DATE>.jsonl` for `_reflect()`. Type+length only, never raw param values. |
| 4 | `gabriel-context-inject`          | SubagentStart | `additionalContext` injection: cap-map summary + last 3 reflexions + top 3 goal-tree leaves + tracked user prefs. Sub-agents inherit self-awareness. |
| 5 | `gabriel-self-validate`           | Stop          | If neither capability_map nor reflexions touched since last user prompt → inject `force_reflect` into `state/autonomous_mode/user_inbox.jsonl`. |
| 6 | `gabriel-direction-validate`      | Stop          | If goal_tree.json stale >1h → inject `force_plan` into `state/autonomous_mode/user_inbox.jsonl`. |

### Watched state files

```
state/gabriel_self/
  capability_map.json    (atomic-replace, 10-min stale window)
  reflexions.jsonl       (append-only)
  user_predictor.json    (atomic-replace, 10-min stale window)
  curiosity_state.json   (atomic-replace, 10-min stale window)
  goal_tree.json         (atomic-replace, 60-min stale window for direction-validate)

  observations_<DATE>.jsonl    (auto-created by observation hook)
  REFRESH_REQUIRED             (auto-created by freshness hook; consumed by daemon)
```

### Inbox kinds produced by these hooks

The autonomous-mode daemon reads `state/autonomous_mode/user_inbox.jsonl`
and acts on the `kind` field. New kinds added:

- `force_reflect` — daemon should run `_reflect()` next cycle (self-awareness module updates)
- `force_plan` — daemon should walk `goal_tree` and emit next-action proposals (plan-without-direction module)

### Smoke verification

```bash
# Delete state file → freshness hook → REFRESH_REQUIRED appears
mv state/gabriel_self/capability_map.json /tmp/
rm -f state/gabriel_self/REFRESH_REQUIRED
./home/.claude/hooks/gabriel-self-freshness/check.sh </dev/null
test -f state/gabriel_self/REFRESH_REQUIRED && echo "PASS"

# Bootstrap restores it
./home/.claude/hooks/gabriel-self-bootstrap/bootstrap.sh </dev/null
test -f state/gabriel_self/capability_map.json && echo "PASS"
```

10/10 sub-tests passed at install. Full breakdown:
`logs/auto_solve/gabriel_module_guardrails_repo_2026-05-20.md`.

### Pre-install backup

`backups/settings-pre-gabriel-guardrails-2026-05-20/`

Contains `settings.json` + `AUTONOMOUS_MODE.md` snapshots.
