# EVERYTHING_GUARDRAIL — workspace meta-mandate (2026-05-20)

> "Everything should be guardrail-grade by default — don't make me keep reminding you." — user, 2026-05-20

This doc is the design document, rationale, and operational reference for the **Guardrail-grade default** mandate added to `CLAUDE.md` and `docs/AGENT_BRIEF_TEMPLATE.md` section 9 on 2026-05-20.

## Why this mandate exists

Across the last 3 days of workspace evolution we added three large self-healing features — `autonomous_mode` (#197), `gabriel_self` (#198), `tcc_autoallow` (#199) — each one needing the same 10 artifacts (plist + state + 5 hooks + settings + doc + backup). Each implementation was reinvented from scratch because there was no canonical template.

The user observed that every NEW feature still requires a manual prompt of "and please add a hook, and a heartbeat, and a doc, and..." This mandate flips the default: no feature ships without the 10-point wrapper. The `guardrail-ify` agent automates the boilerplate so the marginal cost of "make this feature self-healing" approaches zero.

## The 10-point checklist (canonical)

| # | Artifact | Path | Purpose |
|---|---|---|---|
| 1 | launchd plist | `home/Library/LaunchAgents/com.zg.<feature>.plist` | KeepAlive process supervisor |
| 2 | State heartbeat | `state/<feature>/heartbeat.json` | Liveness signal for freshness check |
| 3 | PreToolUse freshness hook | `home/.claude/hooks/<feature>-freshness/check.sh` | Pre-tool auto-respawn if dead |
| 4 | SessionStart bootstrap hook | `home/.claude/hooks/<feature>-bootstrap/bootstrap.sh` | Boot feature on every session |
| 5 | PostToolUse activity hook | `home/.claude/hooks/<feature>-activity/update.sh` | Touch last-online timestamp |
| 6 | SubagentStart context-inject hook | `home/.claude/hooks/<feature>-subagent-inject/inject.sh` | Propagate awareness to children |
| 7 | Stop validation hook | `home/.claude/hooks/<feature>-stop-validate/validate.sh` | End-of-turn output check |
| 8 | settings.json registration | `ClaudeCode/config/settings.json` | Wire hooks into Claude Code |
| 9 | Documentation | `docs/<FEATURE>.md` | Revert + smoke + failure modes |
| 10 | Pre-install backup | `backups/<feature>-pre-install-<DATE>/` | Rollback safety |

Each artifact has a default template in the `guardrail-ify` agent spec (`home/.claude/agents/guardrail-ify.md` v2 section).

## Layer cake (why this many?)

| Layer | Failure mode addressed | If absent |
|---|---|---|
| 1 plist | Process crashes | Feature dies silently, never restarts |
| 2 heartbeat | Process hung but not crashed | No way to detect "alive but stuck" |
| 3 freshness | Heartbeat stale at next session | Session continues unaware feature is dead |
| 4 bootstrap | Process never started this session | Feature absent for entire session |
| 5 activity | Feature can't tell if user is active | Feature wastes cycles when no one's home |
| 6 subagent-inject | Children don't know feature exists | Tree fragments — children duplicate work |
| 7 stop-validate | Feature ran but produced no output | Silent regressions |
| 8 settings reg | Hooks exist but don't fire | Hooks are dead code |
| 9 doc | New contributor breaks it | Revert + smoke require tribal knowledge |
| 10 backup | Failed install corrupts existing state | Permanent damage |

Skipping any layer leaves a known failure mode unhandled.

## The 5-min cost vs the multi-hour debugging cost

Each layer takes ~30 seconds to generate via `guardrail-ify` (5 min total). Without these layers, a single silent failure can eat 2-4 hours of debugging the next time the user notices "wait, why isn't X running?". The cost-benefit is overwhelming.

## Canonical examples to study before building new features

1. **`autonomous_mode`** (#197) — full 9-10/10 wrap. Daemon at `scripts/autonomous_mode_daemon.py`. 6 hooks. Doc at `docs/AUTONOMOUS_MODE.md`.
2. **`gabriel_self`** (#198) — capability map + reflexions. 6 hooks. State at `state/gabriel_self/`.
3. **`tcc_autoallow`** (#199) — TCC dialog auto-grant. 5 hooks.

Read these three before generating new features. Match their structure exactly.

## When to invoke guardrail-ify (auto-trigger keywords)

The orchestrator MUST invoke `guardrail-ify` (rather than write the scaffold manually) when the user prompt or sub-task context contains any of:

- "new daemon"
- "new feature"
- "new capability"
- "new hook"
- "new automation"
- "always-on"
- "background service"
- "monitor process"
- "auto-restart"
- "self-healing"
- "persistent service"

Auto-trigger means: spawn `guardrail-ify` BEFORE writing a single byte of feature code.

## BRITTLE marker convention

Features that ship without the wrapper (one-offs, prototypes, "we'll wrap it later" debt) MUST have their `docs/<FEATURE>.md` header start with:

```
# FEATURE_NAME
BRITTLE — not yet guardrail-grade. See docs/GUARDRAIL_AUDIT.md for promotion queue.
```

The monthly audit at `docs/GUARDRAIL_AUDIT.md` lists every BRITTLE feature and ranks them for retroactive promotion.

## Retroactive audit + wrap process

1. Run `python3 scripts/guardrail_audit_scorer.py` — produces fresh table
2. Update `docs/GUARDRAIL_AUDIT.md` if scores changed
3. For each feature scoring <5/10 (or P0/P1 in priorities table), invoke `guardrail-ify` to fill gaps
4. Re-run audit to confirm score improved
5. Log to `logs/guardrail_audit/<DATE>.md` so we have a longitudinal record

## Sub-agent inheritance

Every spawn brief MUST copy/paste the 3-bullet summary of section 9 from `AGENT_BRIEF_TEMPLATE.md`:

- New feature -> 10-point wrapper (plist + heartbeat + 5 hooks + settings + doc + backup)
- Use `guardrail-ify` agent to auto-generate
- BRITTLE if shipped without wrapper

Grandchildren inherit from children. The tree is recursive. No feature anywhere in the tree should bypass section 9 silently.

## Escape hatch

If a feature genuinely cannot be guardrail-grade (e.g. one-shot script that runs <5 sec and exits), explicitly mark it in its doc:

```
# QUICK_SCRIPT_NAME
EXEMPT from guardrail-grade default — reason: one-shot, runs <5s, no state, no recurrence.
```

The audit will skip these from the brittle queue. But the exemption must be documented in writing — there is no silent skip.

## Related references

- `CLAUDE.md` — "Guardrail-grade default (HARD MANDATE, 2026-05-20)" section (top of "Default tool preferences")
- `docs/AGENT_BRIEF_TEMPLATE.md` section 9 — full inheritance rules + spawn-template
- `docs/GUARDRAIL_AUDIT.md` — current scorecard + retroactive priorities
- `docs/AUTONOMOUS_MODE.md` — canonical example #197
- `home/.claude/agents/guardrail-ify.md` — auto-wrapper agent spec
- `scripts/guardrail_audit_scorer.py` — audit scorer (run monthly)
- Memory: `feedback_guardrail_grade_default.md` — user-facing rationale

## Changelog

- 2026-05-20 — Mandate established. `guardrail-ify` agent extended to 10-point v2 (was 5-hook v1). Audit script + GUARDRAIL_AUDIT.md generated. CLAUDE.md + AGENT_BRIEF_TEMPLATE.md updated.
