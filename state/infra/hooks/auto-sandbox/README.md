# auto-sandbox SessionStart hook

Fires at every session start. Injects a context reminder recommending
the user run `/sandbox` for file/network isolation.

**What it does:** prints a JSON payload matching the SessionStart hook
schema. Claude Code injects the `additionalContext` string into the
agent's context at session start — the agent sees it and can act.

**Why:** `/sandbox` reduces permission prompts ~84% and isolates file
and network side-effects. Easy to forget; this hook ensures it's surfaced.

**To disable:** remove (or comment out) the matching entry in
`AI-Tools/ClaudeCode/config/settings.json` under `hooks.SessionStart`.
