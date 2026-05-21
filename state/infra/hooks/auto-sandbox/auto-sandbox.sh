#!/bin/bash
# SessionStart hook — injects sandbox reminder into session context.
# Claude Code hooks cannot invoke slash commands directly; this prints
# a JSON context-injection that the agent reads at session start.
printf '{"hookSpecificOutput":{"hookEventName":"SessionStart","additionalContext":"Reminder: this session auto-recommends /sandbox for file/network isolation (84%% fewer permission prompts). Run /sandbox at the next natural pause if you need file or network isolation for this session."}}'
