"""
unified_model_router.py — Routes each task to the optimal LLM backend.

Supported backends:
  - claude_max      : Claude via nested-subagent CLI bridge (requires active Claude Code session)
  - deepseek_openclaw : DeepSeek v4-flash via openclaw-gdrive subprocess
  - ollama_local    : Qwen2.5-Coder-7B at http://localhost:11434/v1

Claude access model:
  The user has a Claude Max subscription (flat-rate, OAuth via Claude Code), NOT an
  Anthropic API key. The Max subscription is accessed by shelling out to ``claude -p``
  which inherits the OAuth credentials from the active Claude Code session. This means:

  - Claude is ONLY available when running inside a Claude Code session
    (detected via CLAUDE_CODE_SESSION_ID / CLAUDECODE env vars or /tmp/claude-501).
  - When running in headless / daemon mode (no session), Claude is unavailable and
    ``_invoke_claude`` raises ``ClaudeNotAvailable`` immediately so the fallback chain
    moves to DeepSeek or Ollama.
  - Cost for the Claude path is $0.00 per call from the billing perspective (covered by
    the flat-rate subscription). The cost ledger records ``source="max_subscription"``
    to distinguish this from paid API calls. The pacing daemon still tracks Max usage
    against the subscription's weekly message cap — $0.00 cost does NOT mean unlimited.

Routing is driven by:
  1. Task complexity classification (mechanical / coding / reasoning / architecture)
  2. Pacing regime read from dashboard/pacing_state.json (under/on/over/emergency)
  3. Hard constraints (independence_required, context_tokens_estimate, cost_ceiling_usd)
  4. Automatic fallback chain on HTTP/timeout errors (up to 3 total attempts)

All calls are appended to dashboard/model_router_ledger.jsonl for cost tracking.

Usage (CLI)
-----------
    python scripts/unified_model_router.py --prompt "Reverse a string" --complexity coding
    python scripts/unified_model_router.py --prompt "Design a distributed cache" --complexity architecture
    python scripts/unified_model_router.py --prompt "Summarize JSON" --complexity mechanical --independence

Environment
-----------
    CLAUDE_CODE_SESSION_ID — set by Claude Code; presence means claude -p is available
    CLAUDECODE             — also set by Claude Code (value "1")
    OLLAMA_BASE_URL        — optional override (default: http://localhost:11434)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import httpx

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("unified_model_router")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DRIVE = Path(
    "/Users/orginal/Library/CloudStorage/"
    "GoogleDrive-zachgladstone@gmail.com/My Drive"
)
_PROJECT_ROOT = _DRIVE / "AI-Tools" / "s&p500-ticker-mastery"
_PACING_STATE = _PROJECT_ROOT / "dashboard" / "pacing_state.json"
_LEDGER = _PROJECT_ROOT / "dashboard" / "model_router_ledger.jsonl"
_OPENCLAW_BIN = _DRIVE / "AI-Tools" / "bin" / "openclaw-gdrive"
_OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# ---------------------------------------------------------------------------
# Static system-prompt prefix for claude -p cache warming
#
# Passed via --append-system-prompt on every _invoke_claude call so that the
# SAME prefix hash reaches the server on repeated calls. The main REPL path
# (used by claude -p) auto-injects cache_control breakpoints (mechanism d in
# A_research.md). Prefix must exceed per-model token minimums to be cached:
#   Sonnet 4.6: 1,024 tokens  (~4,096 chars)
#   Opus 4.7 / Haiku 4.5: 4,096 tokens  (~16,384 chars)
# This prefix targets the Sonnet threshold. Opus/Haiku will only cache if
# total system context (CLAUDE.md injected by Claude Code + this prefix)
# exceeds 4,096 tokens.
# TODO: verify --append-system-prompt is available in the installed Claude
# Code version: run `claude --help | grep append-system` to confirm.
# ---------------------------------------------------------------------------

_STATIC_SYSTEM_PREFIX: str = (
    "You are a quantitative trading assistant operating within the "
    "s&p500-ticker-mastery project (AI-Tools workspace, Google Drive, Mac execution environment).\n\n"
    "WORKSPACE CONTEXT\n"
    "-----------------\n"
    "Project root: /Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools/s&p500-ticker-mastery\n"
    "Key directories:\n"
    "  scripts/      — Python scripts including this router and strategy code\n"
    "  dashboard/    — Pacing state (pacing_state.json), cost ledger (model_router_ledger.jsonl)\n"
    "  data/         — Ticker price data and feature matrices\n"
    "  strategies/   — Per-ticker strategy configurations\n"
    "  backups/      — Timestamped backups before risky changes (pattern: YYYY-MM-DD-HHMM)\n"
    "  reports/      — Analysis outputs, backtests, evaluation reports\n"
    "Parent AI-Tools root: .../My Drive/AI-Tools/ (primary AI/project working root)\n\n"
    "ROUTING POLICY\n"
    "--------------\n"
    "This assistant is invoked via unified_model_router.py. Routing decisions are driven by:\n"
    "1. Task complexity: mechanical / coding / reasoning / architecture\n"
    "2. Pacing regime: under / on / over / emergency (read from dashboard/pacing_state.json)\n"
    "3. Hard constraints: independence_required forces DeepSeek; context_tokens > 200k forces DeepSeek\n"
    "4. Cost ceiling: default $0.10/call (Claude Max = $0.00, DeepSeek = paid, Ollama = free)\n\n"
    "BACKEND CAPABILITIES\n"
    "--------------------\n"
    "claude-opus-4-7:   Best reasoning, architecture, complex analysis. ~$5/M input (API), $0 (Max).\n"
    "claude-sonnet-4-6: Balanced speed + intelligence, coding and reasoning. ~$3/M input (API), $0 (Max).\n"
    "claude-haiku-4-5:  Fastest, cheapest, mechanical tasks. ~$1/M input (API), $0 (Max).\n"
    "deepseek-v4-flash: Independent second opinion, large context (977k tokens). ~$0.07/M input.\n"
    "qwen2.5-coder:7b:  Local Ollama model. Free. No external API needed. Best for simple coding tasks.\n\n"
    "PACING REGIME BEHAVIOR\n"
    "-----------------------\n"
    "emergency : Use Ollama for mechanical/coding/reasoning. Only Sonnet for architecture. Conserve Max quota.\n"
    "over      : Use Ollama for mechanical/coding. Sonnet for reasoning/architecture. Avoid Opus.\n"
    "on        : Haiku for mechanical. Sonnet for coding/reasoning. Opus for architecture. Standard operation.\n"
    "under     : Opus aggressively for reasoning/architecture/coding. Sonnet for mechanical. Quota headroom.\n\n"
    "SAFETY AND AUTHORIZATION RULES\n"
    "-------------------------------\n"
    "Pre-approved: read files, write files, search, run scripts, create logs/backups, spawn sub-agents.\n"
    "Prohibited: permanent deletes (backup first to backups/YYYY-MM-DD-HHMM/), credential modification,\n"
    "external messages/money/account changes, macOS system folder modification without explicit instruction.\n"
    "Proof of work required at end of every task: commands run, paths accessed, files changed, backups\n"
    "created, OpenClaw/sub-agent usage (model, reason, result), final status.\n\n"
    "COST LEDGER SCHEMA\n"
    "------------------\n"
    "All calls appended to dashboard/model_router_ledger.jsonl (JSONL, one entry per call).\n"
    "Fields: ts, request_hash, backend, model, prompt_tokens, response_tokens, cost_usd, cost_source,\n"
    "latency_s, success, regime, cache_creation_tokens, cache_read_tokens, cache_status.\n"
    "cost_source values: 'max_subscription' (Claude via Max, $0.00), 'api_pay_per_token' (DeepSeek).\n"
    "cache_status: 'hit' (cache_read > 0), 'write' (cache_creation > 0),\n"
    "              'unknown_cli_path' (claude -p output lacks cache counters), 'n/a' (non-Claude).\n\n"
    "CACHE BEHAVIOR\n"
    "--------------\n"
    "Claude calls via `claude -p` use the main REPL path which auto-injects cache_control breakpoints.\n"
    "This static prefix is passed via --append-system-prompt on every call so the server-side prefix\n"
    "hash matches across repeated calls, enabling cache hits after the first write.\n"
    "Cache TTL: ~5 minutes ephemeral. Do not modify this prefix text between back-to-back calls.\n"
    "Minimum token thresholds: Sonnet 1,024 tokens; Opus/Haiku 4,096 tokens.\n"
    "Note: Claude Code sub-agent path has bug #29966 (enablePromptCaching hardcoded false).\n"
    "This prefix mechanism uses the claude -p REPL path (mechanism d) which is unaffected.\n\n"
    "SUB-AGENT MODEL ROUTING\n"
    "------------------------\n"
    "opus:   Advanced reasoning, deep architecture, complex coding, hard debugging, synthesis.\n"
    "sonnet: Normal coding, analysis, file review, medium debugging, scripts, reports. Default.\n"
    "haiku:  File inspection, search, log scanning, simple formatting, repetitive low-risk tasks.\n"
    "Include model_reason in spawn payload. Escalate model if stuck. Route mechanical work to Ollama first.\n\n"
    "QUANTITATIVE TRADING CONTEXT\n"
    "-----------------------------\n"
    "This project operates on S&P 500 tickers. Critical constraints to always enforce:\n"
    "- No lookahead bias: features must use only data available at signal time (shifted >=1 bar).\n"
    "- Validation split must precede test split chronologically. No random shuffles on time-series data.\n"
    "- Per-ticker optimization: tune strategy, threshold, TP-SL per ticker, not globally.\n"
    "- Backtest integrity: transaction costs, slippage, and position sizing must be realistic.\n"
    "- Feature engineering: use only OHLCV, volume, derived indicators, fundamentals pre-signal.\n"
    "- Risk management: max drawdown, Sharpe ratio, and win rate are primary evaluation metrics.\n"
    "- Data leakage check: verify no future data leaks into training features before final evaluation.\n"
    "- Walk-forward validation: use rolling or expanding window; never fit on the full dataset.\n\n"
    "TOOLING PREFERENCES\n"
    "--------------------\n"
    "GitHub work: use github-mcp-server (MCP) over gh/git shell commands.\n"
    "Persistent memory across sessions: use mcp-memory-keeper (MCP, local SQLite).\n"
    "Token reduction tools (mandatory for every sub-agent spawn):\n"
    "  RTK: ~/.local/bin/rtk (auto-loads on Bash output; no manual invocation needed)\n"
    "  token-optimizer-mcp: MCP-based, auto-loaded at session start\n"
    "  LLMLingua: python scripts/llmlingua_compress.py --target-ratio 0.5 (for prompts >2k tokens)\n"
    "Session compress/preserve/resume: cpr-compress-preserve-resume skill.\n"
    "Parallel orchestration: dispatching-parallel-agents skill (fan out 6-10 agents).\n"
    "Pre-flight assumption check: karpathy-guidelines skill (before any ML/quant/backtest task).\n\n"
    "BEHAVIORAL GUIDELINES (Karpathy principles)\n"
    "--------------------------------------------\n"
    "1. Minimal code: don't add code not needed for the current task. No premature abstractions.\n"
    "2. Surface assumptions before implementing. Verify with data and metrics, not intuition.\n"
    "3. Surgical changes: minimal diff. No collateral cleanup in bug-fix PRs.\n"
    "4. No silent failures: propagate errors, don't swallow exceptions. Log with context.\n"
    "5. Test with real data, not mocks, for financial/backtest code. Mock/prod divergence causes failures.\n"
    "6. One logical change per PR. No bundling unrelated fixes.\n"
    "7. Before optimizing: measure first. Never guess the bottleneck.\n"
    "8. Three similar lines is better than a premature abstraction. Resist DRY at the wrong level.\n"
)

# ---------------------------------------------------------------------------
# Pricing table (USD per million tokens)
# ---------------------------------------------------------------------------

_PRICING: dict[str, dict[str, float]] = {
    # Claude models: accessed via Max subscription (claude -p shell-out).
    # Billing cost = $0.00 per call. _estimate_cost() returns 0.0 for all claude-* models.
    # These rates are the published Anthropic API rates, kept here for reference only
    # (e.g., if a direct API key path is ever re-enabled for a different deployment).
    "claude-opus-4-7":    {"input": 5.00,  "output": 25.00},
    "claude-sonnet-4-6":  {"input": 3.00,  "output": 15.00},
    "claude-haiku-4-5":   {"input": 1.00,  "output": 5.00},
    "deepseek-v4-flash":  {"input": 0.07,  "output": 0.28},  # DeepSeek V3 flash pricing
    "qwen2.5-coder:7b":   {"input": 0.00,  "output": 0.00},  # Local — free
}

_LATENCY_ESTIMATES: dict[str, float] = {
    "claude-opus-4-7":   15.0,
    "claude-sonnet-4-6": 8.0,
    "claude-haiku-4-5":  3.0,
    "deepseek-v4-flash": 6.0,   # direct API; openclaw adds ~5s subprocess overhead
    "qwen2.5-coder:7b":  12.0,  # local 7B model on CPU
}

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ComplexityLevel = Literal["mechanical", "coding", "reasoning", "architecture"]
BackendName = Literal["claude_api", "deepseek_openclaw", "ollama_local"]
PacingRegime = Literal["under", "on", "over", "emergency"]

# ---------------------------------------------------------------------------
# Routing-query helpers (CLI mode: --task-kind / --complexity aliases)
# ---------------------------------------------------------------------------

_COMPLEXITY_ALIAS_MAP: dict[str, ComplexityLevel] = {
    # low tier
    "low": "mechanical", "simple": "mechanical", "trivial": "mechanical",
    # mid tier
    "medium": "coding", "normal": "coding",
    # high tier
    "high": "architecture", "complex": "architecture",
    # pass-through existing values
    "mechanical": "mechanical", "coding": "coding",
    "reasoning": "reasoning", "architecture": "architecture",
}

# Ordered rules: first match wins; checked case-insensitively via 'in'
_TASK_KIND_RULES: list[tuple[list[str], ComplexityLevel]] = [
    (["architecture", "synthesis", "system design", "distributed", "orchestrat"], "architecture"),
    (["reasoning", "backtest", "strategy", "security", "debug", "analyze", "analysis", "review"], "reasoning"),
    (["coding", "code", "implement", "refactor", "fix", "script", "api", "test", "build"], "coding"),
    (["list", "grep", "search", "scan", "format", "summarize", "summary", "inspect", "read", "log", "file"], "mechanical"),
]

_MODEL_SHORT_NAMES: dict[str, str] = {
    "claude-opus-4-7":   "opus",
    "claude-sonnet-4-6": "sonnet",
    "claude-haiku-4-5":  "haiku",
    "deepseek-v4-flash": "sonnet",  # treat as sonnet-tier externally
    "qwen2.5-coder:7b":  "haiku",   # treat as haiku-tier externally
}


def _normalize_complexity(alias: str) -> ComplexityLevel:
    """Map a user-supplied complexity alias (low/medium/high/...) to ComplexityLevel."""
    result = _COMPLEXITY_ALIAS_MAP.get(alias.lower().strip())
    if result is None:
        valid = ", ".join(sorted(_COMPLEXITY_ALIAS_MAP))
        raise ValueError(f"Unknown complexity alias {alias!r}. Valid values: {valid}")
    return result


def _resolve_task_kind(task_kind: str) -> ComplexityLevel:
    """Infer ComplexityLevel from a task-kind string via case-insensitive contains-matching."""
    tk = task_kind.lower()
    for keywords, level in _TASK_KIND_RULES:
        if any(kw in tk for kw in keywords):
            return level
    return "coding"  # default if no keyword matches


def _short_model_name(model: str) -> str:
    """Return canonical short name (haiku/sonnet/opus) for a model ID."""
    return _MODEL_SHORT_NAMES.get(model, model)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class TaskRequest:
    """Specifies a task to be routed and invoked."""

    prompt: str
    complexity: ComplexityLevel
    independence_required: bool = False
    """If True, must use a non-Claude backend (real second opinion)."""
    context_tokens_estimate: int = 0
    """Rough count of tokens in context window needed."""
    cost_ceiling_usd: float = 0.10
    """Maximum allowable cost per call in USD."""
    deadline_seconds: float = 60.0
    """Wall-clock budget for the call."""
    allow_local: bool = True
    """Whether ollama_local is a permitted backend."""


@dataclass
class BackendChoice:
    """Describes the selected backend and the rationale."""

    backend: BackendName
    model: str
    estimated_cost_usd: float
    estimated_latency_seconds: float
    reason: str


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class RouterFailure(Exception):
    """Raised when all fallback attempts are exhausted."""

    def __init__(self, message: str, attempts: list[dict]) -> None:
        super().__init__(message)
        self.attempts = attempts


class BackendError(Exception):
    """Raised by individual backend invokers on recoverable errors."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class ClaudeNotAvailable(BackendError):
    """Raised when the Claude path cannot be used.

    This happens when:
    - No active Claude Code session is detected (headless/daemon mode)
    - The ``claude -p`` subprocess is not found on PATH
    - The shell-out to ``claude -p`` exits non-zero

    The fallback chain in ``invoke()`` catches this and moves to DeepSeek or Ollama.
    ClaudeNotAvailable is treated as non-retryable within the claude_api backend
    but the *chain itself* still falls through to the next backend.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=None)
        # Override _should_retry: we want the chain to continue, so treat as
        # a connection-style error (status_code=None → _should_retry returns True).
        # The chain will try the next backend rather than aborting.


# ---------------------------------------------------------------------------
# Pacing regime reader
# ---------------------------------------------------------------------------

def _read_pacing_regime() -> PacingRegime:
    """Read current pacing regime from pacing_state.json.

    Returns:
        One of "under", "on", "over", "emergency". Falls back to "on" on any error.
    """
    try:
        with open(_PACING_STATE) as f:
            state = json.load(f)
        regime = state.get("regime", "on")
        valid: set[str] = {"under", "on", "over", "emergency"}
        return regime if regime in valid else "on"  # type: ignore[return-value]
    except Exception as exc:
        log.warning("Could not read pacing state (%s); defaulting to 'on'.", exc)
        return "on"


# ---------------------------------------------------------------------------
# Claude session detection
# ---------------------------------------------------------------------------

def _in_claude_session() -> bool:
    """Detect whether we are running inside an active Claude Code session.

    When True, ``claude -p`` is available on PATH and inherits the Max subscription
    OAuth credentials, making it usable as a zero-marginal-cost LLM backend.

    When False (headless daemon, cron job, plain Python script), ``claude -p`` will
    either not be on PATH or will not have valid credentials, so we skip Claude
    entirely and fall through to DeepSeek/Ollama.

    Detection strategy (ordered by reliability):
    1. ``CLAUDE_CODE_SESSION_ID`` — set by Claude Code for every session; most reliable.
    2. ``CLAUDECODE=1``           — also set by Claude Code launcher; secondary check.
    3. ``/tmp/claude-501``        — Claude Code socket directory; present when the
                                    daemon is active on this Mac (UID 501).

    Any one marker is sufficient to return True.

    Returns:
        True if an active Claude Code session is detected, False otherwise.
    """
    if os.environ.get("CLAUDE_CODE_SESSION_ID"):
        return True
    if os.environ.get("CLAUDECODE") == "1":
        return True
    if os.path.exists("/tmp/claude-501"):
        return True
    return False


# ---------------------------------------------------------------------------
# Cost estimator
# ---------------------------------------------------------------------------

def _estimate_cost(model: str, context_tokens: int) -> float:
    """Rough cost estimate assuming ~50% input / 50% output split of total tokens.

    Claude models accessed via the Max subscription (``claude -p`` shell-out) are
    **free per call** from the billing perspective — the subscription is flat-rate.
    This function returns 0.0 for all Claude model IDs so the cost ledger and routing
    cost-ceiling guard treat them correctly.

    Note: $0.00 does NOT mean unlimited. The pacing daemon still tracks Max subscription
    weekly message usage independently. This function only handles USD billing cost.

    Args:
        model: Model identifier string.
        context_tokens: Total token budget (prompt + expected response).

    Returns:
        Estimated cost in USD (0.0 for Claude/Max models, token-based for paid APIs).
    """
    # Claude models via Max subscription — flat-rate, no per-token billing cost.
    if model.startswith("claude-"):
        return 0.0
    pricing = _PRICING.get(model, {"input": 5.00, "output": 25.00})
    half = max(context_tokens, 500) / 2
    return (half / 1_000_000) * pricing["input"] + (half / 1_000_000) * pricing["output"]


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def route(req: TaskRequest) -> BackendChoice:
    """Determine the optimal backend and model for this TaskRequest.

    Priority order:
      1. independence_required → DeepSeek (non-Claude alignment for real second opinion)
      2. context_tokens_estimate > 200_000 → DeepSeek (977k context advantage)
      3. Pacing regime table (see below)
      4. Cost ceiling guard — downgrade if cheapest viable option still exceeds cap

    Pacing regime → model mapping:
      emergency : Ollama for mechanical/coding/reasoning; Sonnet for architecture
      over      : Ollama for mechanical/coding; Sonnet for reasoning; Opus for architecture
      on        : Haiku for mechanical; Sonnet for coding/reasoning; Opus for architecture
      under     : Opus for reasoning/architecture/coding; Sonnet for mechanical

    Args:
        req: The task request to route.

    Returns:
        A BackendChoice describing the selected backend, model, and reason.
    """
    # --- Hard constraint: independence_required --------------------------------
    if req.independence_required:
        model = "deepseek-v4-flash"
        return BackendChoice(
            backend="deepseek_openclaw",
            model=model,
            estimated_cost_usd=_estimate_cost(model, req.context_tokens_estimate),
            estimated_latency_seconds=_LATENCY_ESTIMATES[model],
            reason="independence_required=True — DeepSeek provides non-Claude alignment for genuine second opinion",
        )

    # --- Hard constraint: huge context ----------------------------------------
    if req.context_tokens_estimate > 200_000:
        model = "deepseek-v4-flash"
        return BackendChoice(
            backend="deepseek_openclaw",
            model=model,
            estimated_cost_usd=_estimate_cost(model, req.context_tokens_estimate),
            estimated_latency_seconds=_LATENCY_ESTIMATES[model],
            reason=f"context_tokens_estimate={req.context_tokens_estimate} > 200k — DeepSeek has 977k context window",
        )

    # --- Pacing regime --------------------------------------------------------
    regime: PacingRegime = _read_pacing_regime()
    complexity = req.complexity

    if regime == "emergency":
        if complexity == "architecture":
            backend: BackendName = "claude_api"
            model = "claude-sonnet-4-6"
            reason = "emergency regime: architecture still needs Claude Sonnet; Ollama insufficient"
        elif req.allow_local:
            backend = "ollama_local"
            model = "qwen2.5-coder:7b"
            reason = f"emergency regime: routing {complexity} to free local Ollama to conserve quota"
        else:
            backend = "claude_api"
            model = "claude-haiku-4-5"
            reason = f"emergency regime: Ollama not allowed; using cheapest Claude (Haiku) for {complexity}"

    elif regime == "over":
        if complexity in ("mechanical", "coding"):
            if req.allow_local:
                backend = "ollama_local"
                model = "qwen2.5-coder:7b"
                reason = f"over regime: {complexity} work routed to local Ollama to reduce spend"
            else:
                backend = "claude_api"
                model = "claude-haiku-4-5"
                reason = f"over regime: Ollama not allowed; Haiku for {complexity}"
        elif complexity == "reasoning":
            backend = "claude_api"
            model = "claude-sonnet-4-6"
            reason = "over regime: reasoning still needs Sonnet; local models not reliable enough"
        else:  # architecture
            backend = "claude_api"
            model = "claude-sonnet-4-6"
            reason = "over regime: architecture needs Sonnet even when over quota"

    elif regime == "on":
        if complexity == "mechanical":
            backend = "claude_api"
            model = "claude-haiku-4-5"
            reason = "on regime: mechanical tasks → Haiku (fastest, cheapest)"
        elif complexity == "architecture":
            backend = "claude_api"
            model = "claude-opus-4-7"
            reason = "on regime: architecture → Opus (deepest reasoning)"
        else:  # coding, reasoning
            backend = "claude_api"
            model = "claude-sonnet-4-6"
            reason = f"on regime: {complexity} → Sonnet (best speed/intelligence balance)"

    else:  # under
        if complexity in ("reasoning", "architecture"):
            backend = "claude_api"
            model = "claude-opus-4-7"
            reason = f"under regime: {complexity} → Opus aggressively (quota headroom available)"
        elif complexity == "coding":
            backend = "claude_api"
            model = "claude-opus-4-7"
            reason = "under regime: coding → Opus aggressively (quota headroom available)"
        else:  # mechanical
            backend = "claude_api"
            model = "claude-sonnet-4-6"
            reason = "under regime: mechanical → Sonnet (slight upgrade; Haiku not worth saving here)"

    # --- Cost ceiling guard ---------------------------------------------------
    estimated_cost = _estimate_cost(model, req.context_tokens_estimate)
    if estimated_cost > req.cost_ceiling_usd and backend == "claude_api":
        # Downgrade to next cheaper tier
        if model == "claude-opus-4-7":
            model = "claude-sonnet-4-6"
            estimated_cost = _estimate_cost(model, req.context_tokens_estimate)
            reason += " [downgraded Opus→Sonnet: cost ceiling]"
        if estimated_cost > req.cost_ceiling_usd:
            if req.allow_local:
                backend = "ollama_local"
                model = "qwen2.5-coder:7b"
                estimated_cost = 0.0
                reason += " [downgraded to Ollama: cost ceiling]"
            else:
                model = "claude-haiku-4-5"
                estimated_cost = _estimate_cost(model, req.context_tokens_estimate)
                reason += " [downgraded to Haiku: cost ceiling]"

    return BackendChoice(
        backend=backend,
        model=model,
        estimated_cost_usd=_estimate_cost(model, req.context_tokens_estimate),
        estimated_latency_seconds=_LATENCY_ESTIMATES.get(model, 10.0),
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Backend invocation functions
# ---------------------------------------------------------------------------

def _invoke_claude(prompt: str, model: str) -> tuple[str, float, float, dict]:
    """Invoke Claude via the ``claude -p`` headless CLI bridge.

    This function uses the nested-subagent pattern: it shells out to ``claude -p``
    which inherits the active Claude Code session's OAuth credentials (Max
    subscription). No Anthropic API key is needed or used.

    The cost is always $0.00 per call (flat-rate Max subscription). The pacing
    daemon tracks weekly usage volume separately.

    Cache warming:
        ``_STATIC_SYSTEM_PREFIX`` is passed via ``--append-system-prompt`` so the
        same stable prefix hash reaches the server on every call. The ``claude -p``
        path uses the main REPL query path which auto-injects ``cache_control``
        breakpoints (mechanism d; unaffected by Claude Code sub-agent bug #29966).
        Cache token counts are extracted from the stream-json ``result`` event's
        ``usage`` field when available; otherwise ``cache_status`` is set to
        ``"unknown_cli_path"`` in the ledger.

    Session requirement:
        This function first checks ``_in_claude_session()``. If no session is
        detected (headless daemon mode), it raises ``ClaudeNotAvailable``
        immediately so the fallback chain can move to DeepSeek or Ollama.

    Model name mapping:
        The router uses full model IDs (e.g. ``claude-sonnet-4-6``). The
        ``claude -p`` CLI accepts short names (``sonnet``, ``opus``, ``haiku``)
        or the full model string. We strip the version suffix to produce the
        canonical short alias the CLI accepts.

    Args:
        prompt: The user prompt string.
        model: Full model ID (e.g. ``"claude-sonnet-4-6"``).

    Returns:
        Tuple of (response_text, actual_cost_usd=0.0, actual_latency_seconds,
        cache_info). cache_info keys: cache_creation_input_tokens (int),
        cache_read_input_tokens (int), cache_status (str: "hit"/"write"/"unknown_cli_path").

    Raises:
        ClaudeNotAvailable: When no Claude Code session is detected, or when the
            ``claude`` binary is not found on PATH, or when the subprocess exits
            non-zero with a meaningful error.
        BackendError: On timeout (retryable) or other subprocess errors.
    """
    if not _in_claude_session():
        raise ClaudeNotAvailable(
            "No active Claude Code session detected "
            "(CLAUDE_CODE_SESSION_ID not set, CLAUDECODE!=1, /tmp/claude-501 absent) — "
            "claude -p unavailable in this execution context; falling through to next backend"
        )

    # Map full model ID to the short alias accepted by the claude CLI.
    # claude -p also accepts full model strings, but short names are more robust.
    _MODEL_ALIAS: dict[str, str] = {
        "claude-opus-4-7":   "opus",
        "claude-sonnet-4-6": "sonnet",
        "claude-haiku-4-5":  "haiku",
    }
    cli_model = _MODEL_ALIAS.get(model, model)

    # Build the command.
    # --output-format stream-json requires --verbose when used with --print.
    # --model sets the model. --print (-p) is headless mode (no interactive UI).
    # --append-system-prompt passes the stable workspace prefix so repeated calls
    # hash to the same prefix on the server, enabling cache hits (mechanism d).
    # TODO: confirm --append-system-prompt is supported: `claude --help | grep append-system`
    cmd = [
        "claude",
        "--print",
        "--output-format", "stream-json",
        "--verbose",
        "--model", cli_model,
        "--append-system-prompt", _STATIC_SYSTEM_PREFIX,
        prompt,
    ]

    t0 = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired as exc:
        raise BackendError(
            f"claude -p timed out after 120s for model {cli_model}"
        ) from exc
    except FileNotFoundError as exc:
        raise ClaudeNotAvailable(
            "claude binary not found on PATH — Claude Code may not be installed "
            "or the session PATH is not inherited"
        ) from exc

    latency = time.monotonic() - t0

    if result.returncode != 0:
        stderr_snippet = result.stderr.strip()[:400]
        # Auth errors suggest the session is invalid — surface as ClaudeNotAvailable
        if any(kw in stderr_snippet.lower() for kw in ("auth", "unauthorized", "login", "credentials")):
            raise ClaudeNotAvailable(
                f"claude -p exited {result.returncode} with auth-related error: {stderr_snippet}"
            )
        raise BackendError(
            f"claude -p exited {result.returncode}: {stderr_snippet}",
            status_code=result.returncode,
        )

    # Parse stream-json output: find the final "result" event which carries the
    # complete assistant text and usage data (including cache token counts).
    # Each line is a JSON object with a "type" field.
    response_text = ""
    cache_info: dict = {
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_status": "unknown_cli_path",
    }
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        # stream-json events: type="result" has the final text and usage data.
        if event.get("type") == "result":
            response_text = event.get("result", "")
            usage = event.get("usage", {})
            cache_creation = usage.get("cache_creation_input_tokens", 0) or 0
            cache_read = usage.get("cache_read_input_tokens", 0) or 0
            cache_info = {
                "cache_creation_input_tokens": cache_creation,
                "cache_read_input_tokens": cache_read,
                "cache_status": (
                    "hit" if cache_read > 0
                    else "write" if cache_creation > 0
                    else "unknown_cli_path"
                ),
            }
            break
        # Fallback: some builds emit type="assistant" with content blocks
        if event.get("type") == "assistant":
            content = event.get("message", {}).get("content", [])
            if content and isinstance(content, list):
                texts = [c.get("text", "") for c in content if c.get("type") == "text"]
                if texts:
                    response_text = "".join(texts)

    if not response_text:
        # Last-resort: if stream-json parsing found nothing, try raw stdout
        raw = result.stdout.strip()
        if raw:
            response_text = raw
        else:
            raise BackendError(
                f"claude -p returned no parseable response text (stdout was empty)"
            )

    # Cost is $0.00 — Max subscription is flat-rate, no per-token billing.
    return response_text, 0.0, latency, cache_info


def _invoke_deepseek(prompt: str) -> tuple[str, float, float, dict]:
    """Invoke DeepSeek via the openclaw-gdrive subprocess.

    Command pattern:
        openclaw-gdrive capability model run --local --model deepseek/deepseek-v4-flash
            --json --prompt "<text>"

    The JSON output contains outputs[0].text with the response.

    Args:
        prompt: The user prompt string.

    Returns:
        Tuple of (response_text, actual_cost_usd, actual_latency_seconds, cache_info).
        cache_info is always {"cache_status": "n/a"} — DeepSeek has no cache_control.

    Raises:
        BackendError: On non-zero exit, timeout, or malformed JSON.
    """
    openclaw = str(_OPENCLAW_BIN)
    cmd = [
        openclaw,
        "capability", "model", "run",
        "--local",
        "--model", "deepseek/deepseek-v4-flash",
        "--json",
        "--prompt", prompt,
    ]
    t0 = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired as exc:
        raise BackendError("DeepSeek/openclaw timed out after 120s") from exc
    except FileNotFoundError as exc:
        raise BackendError(f"openclaw-gdrive not found at {openclaw}") from exc

    latency = time.monotonic() - t0

    if result.returncode != 0:
        raise BackendError(
            f"openclaw exited {result.returncode}: {result.stderr[:300]}",
            status_code=result.returncode,
        )

    raw = result.stdout.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try to find first JSON object in output (openclaw may emit preamble)
        start = raw.find("{")
        if start >= 0:
            try:
                data = json.loads(raw[start:])
            except json.JSONDecodeError as exc:
                raise BackendError(f"openclaw JSON parse failed: {raw[:200]}") from exc
        else:
            raise BackendError(f"openclaw non-JSON output: {raw[:200]}")

    outputs = data.get("outputs", [])
    if not outputs:
        raise BackendError(f"openclaw returned no outputs: {data}")

    response_text = outputs[0].get("text", "")

    # DeepSeek pricing approximation — token counts not reliably returned by openclaw
    word_count = len(prompt.split()) + len(response_text.split())
    approx_tokens = int(word_count * 1.33)
    pricing = _PRICING["deepseek-v4-flash"]
    half = approx_tokens / 2
    actual_cost = (half / 1_000_000) * pricing["input"] + (half / 1_000_000) * pricing["output"]

    return response_text, actual_cost, latency, {"cache_status": "n/a"}


def _invoke_ollama(prompt: str, model: str = "qwen2.5-coder:7b") -> tuple[str, float, float, dict]:
    """Invoke a local Ollama model via the Anthropic-compatible messages endpoint.

    Targets: POST http://localhost:11434/v1/messages

    Args:
        prompt: The user prompt string.
        model: Ollama model tag (default: "qwen2.5-coder:7b").

    Returns:
        Tuple of (response_text, actual_cost_usd, actual_latency_seconds, cache_info).
        Cost is always 0.0 for local models. cache_info is always {"cache_status": "n/a"}.

    Raises:
        BackendError: On connection error, timeout, or non-2xx response.
    """
    url = f"{_OLLAMA_BASE_URL}/v1/messages"
    payload = {
        "model": model,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
    }
    t0 = time.monotonic()
    try:
        with httpx.Client(timeout=90.0) as client:
            resp = client.post(url, json=payload)
    except httpx.ConnectError as exc:
        raise BackendError(f"Ollama not reachable at {url}: {exc}") from exc
    except httpx.TimeoutException as exc:
        raise BackendError(f"Ollama request timed out: {exc}") from exc

    latency = time.monotonic() - t0

    if resp.status_code in (401, 403):
        raise BackendError(f"Ollama auth error {resp.status_code}", status_code=resp.status_code)
    if resp.status_code == 429:
        raise BackendError("Ollama rate limit (unusual)", status_code=429)
    if resp.status_code >= 500:
        raise BackendError(f"Ollama server error {resp.status_code}: {resp.text[:200]}", status_code=resp.status_code)
    if resp.status_code != 200:
        raise BackendError(f"Ollama unexpected status {resp.status_code}: {resp.text[:200]}", status_code=resp.status_code)

    try:
        data = resp.json()
    except Exception as exc:
        raise BackendError(f"Ollama non-JSON response: {resp.text[:200]}") from exc

    # Anthropic message format: content[0].text
    content = data.get("content", [])
    if content:
        response_text = content[0].get("text", "")
    else:
        # Fallback: OpenAI format (some Ollama builds)
        choices = data.get("choices", [])
        response_text = (choices[0].get("message", {}).get("content", "") if choices else "")

    return response_text, 0.0, latency, {"cache_status": "n/a"}


# ---------------------------------------------------------------------------
# Fallback ordering
# ---------------------------------------------------------------------------

def _fallback_chain(primary: BackendChoice, req: TaskRequest) -> list[BackendChoice]:
    """Build an ordered list of backends to try, starting from primary.

    Args:
        primary: The originally selected backend.
        req: The original task request (used to check allow_local).

    Returns:
        List of BackendChoice instances to attempt in order (max 3).
    """
    all_options: list[BackendChoice] = []

    # Priority: primary first, then ordered by cost (cheapest fallback preferred)
    candidates: list[tuple[BackendName, str]] = [
        ("claude_api", "claude-sonnet-4-6"),
        ("deepseek_openclaw", "deepseek-v4-flash"),
    ]
    if req.allow_local:
        candidates.append(("ollama_local", "qwen2.5-coder:7b"))

    # Insert primary at front, remove duplicates
    full_order: list[tuple[BackendName, str]] = [(primary.backend, primary.model)]
    for bk, mdl in candidates:
        if (bk, mdl) != (primary.backend, primary.model):
            full_order.append((bk, mdl))

    for backend, model in full_order[:3]:
        all_options.append(
            BackendChoice(
                backend=backend,
                model=model,
                estimated_cost_usd=_estimate_cost(model, req.context_tokens_estimate),
                estimated_latency_seconds=_LATENCY_ESTIMATES.get(model, 10.0),
                reason=f"fallback chain entry (primary was {primary.backend}/{primary.model})",
            )
        )
    return all_options


def _should_retry(exc: BackendError) -> bool:
    """Return True if the error is worth retrying on a different backend."""
    if exc.status_code is None:
        return True  # Connection errors, timeouts — always retry
    return exc.status_code in (429, 500, 502, 503, 504)


# ---------------------------------------------------------------------------
# Cost ledger
# ---------------------------------------------------------------------------

def _append_ledger(
    *,
    request_hash: str,
    backend: BackendName,
    model: str,
    prompt_tokens: int,
    response_tokens: int,
    cost_usd: float,
    latency_s: float,
    success: bool,
    regime: str,
    cost_source: str = "api_pay_per_token",
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_status: str = "unknown",
) -> None:
    """Append one record to the JSONL cost ledger.

    Args:
        request_hash: Short SHA256 of the prompt for deduplication.
        backend: Which backend was used.
        model: The exact model string.
        prompt_tokens: Estimated input token count.
        response_tokens: Estimated output token count.
        cost_usd: Actual or estimated cost in USD. Always 0.0 for Claude/Max calls.
        latency_s: Wall-clock seconds for the call.
        success: Whether the call succeeded.
        regime: Pacing regime at time of call.
        cost_source: How the cost is billed. ``"max_subscription"`` for Claude via
            Max (flat-rate, $0 per-token); ``"api_pay_per_token"`` for DeepSeek
            and any paid API. Defaults to ``"api_pay_per_token"``.
        cache_creation_tokens: Tokens written to the prompt cache on this call
            (from response.usage.cache_creation_input_tokens). 0 if unavailable.
        cache_read_tokens: Tokens read from the prompt cache on this call
            (from response.usage.cache_read_input_tokens). 0 if unavailable.
        cache_status: "hit" (cache_read > 0), "write" (cache_creation > 0),
            "unknown_cli_path" (claude -p; no counters exposed in output),
            "n/a" (non-Claude backends), "unknown" (default/unset).
    """
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "request_hash": request_hash,
        "backend": backend,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "cost_usd": round(cost_usd, 8),
        "cost_source": cost_source,
        "latency_s": round(latency_s, 3),
        "success": success,
        "regime": regime,
        "cache_creation_tokens": cache_creation_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_status": cache_status,
    }
    try:
        _LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with open(_LEDGER, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as exc:
        log.warning("Could not write ledger entry: %s", exc)


def get_spend(backend: Optional[BackendName] = None, window_hours: float = 24.0) -> float:
    """Return total cost in USD for a given backend over the past window_hours.

    Args:
        backend: Filter by backend name, or None for all backends combined.
        window_hours: How many hours back to look.

    Returns:
        Total USD spent. Returns 0.0 if ledger is missing or unreadable.
    """
    try:
        if not _LEDGER.exists():
            return 0.0
        now = datetime.now(timezone.utc).timestamp()
        cutoff = now - window_hours * 3600
        total = 0.0
        with open(_LEDGER) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    ts = datetime.fromisoformat(entry["ts"]).timestamp()
                    if ts < cutoff:
                        continue
                    if backend and entry.get("backend") != backend:
                        continue
                    if entry.get("success", False):
                        total += entry.get("cost_usd", 0.0)
                except Exception:
                    continue
        return round(total, 6)
    except Exception as exc:
        log.warning("Could not read ledger for spend: %s", exc)
        return 0.0


# ---------------------------------------------------------------------------
# Main invoke function with fallback chain
# ---------------------------------------------------------------------------

def invoke(req: TaskRequest) -> tuple[str, BackendChoice]:
    """Route and invoke the task, with automatic fallback on failure.

    Attempts up to 3 backends in priority order. On HTTP 401/403/429/5xx
    or timeout, falls through to the next backend in the chain.

    Args:
        req: The task request to fulfill.

    Returns:
        Tuple of (response_text, BackendChoice that succeeded).

    Raises:
        RouterFailure: If all fallback attempts are exhausted.
    """
    primary = route(req)
    chain = _fallback_chain(primary, req)
    regime = _read_pacing_regime()
    request_hash = hashlib.sha256(req.prompt.encode()).hexdigest()[:12]

    attempts: list[dict] = []

    # In emergency regime, skip Claude entirely (Max subscription cap conservation).
    # The routing logic may still pick claude_api for architecture tasks even in
    # emergency — override that here to avoid wasting Max messages when quota is critical.
    if regime == "emergency":
        chain = [c for c in chain if c.backend != "claude_api"]
        if not chain:
            raise RouterFailure(
                "Emergency regime: no non-Claude backends available for this request",
                attempts=[],
            )

    for choice in chain:
        t0 = time.monotonic()
        try:
            log.info(
                "Attempting %s/%s (complexity=%s, regime=%s)",
                choice.backend, choice.model, req.complexity, regime,
            )
            if choice.backend == "claude_api":
                # Claude via Max subscription (claude -p shell-out, $0.00 billing cost)
                text, cost, latency, cache_info = _invoke_claude(req.prompt, choice.model)
            elif choice.backend == "deepseek_openclaw":
                text, cost, latency, cache_info = _invoke_deepseek(req.prompt)
            elif choice.backend == "ollama_local":
                text, cost, latency, cache_info = _invoke_ollama(req.prompt, choice.model)
            else:
                raise BackendError(f"Unknown backend: {choice.backend}")

            # Success
            prompt_tokens = int(len(req.prompt.split()) * 1.33)
            response_tokens = int(len(text.split()) * 1.33)
            # For Claude/Max, annotate source so the ledger captures the subscription model.
            is_claude_max = choice.backend == "claude_api"
            _append_ledger(
                request_hash=request_hash,
                backend=choice.backend,
                model=choice.model,
                prompt_tokens=prompt_tokens,
                response_tokens=response_tokens,
                cost_usd=cost,  # 0.0 for claude_api (Max subscription)
                latency_s=latency,
                success=True,
                regime=regime,
                cost_source="max_subscription" if is_claude_max else "api_pay_per_token",
                cache_creation_tokens=cache_info.get("cache_creation_input_tokens", 0),
                cache_read_tokens=cache_info.get("cache_read_input_tokens", 0),
                cache_status=cache_info.get("cache_status", "unknown"),
            )
            log.info(
                "Success: %s/%s — cost=$%.6f%s latency=%.2fs",
                choice.backend, choice.model, cost,
                " (Max subscription)" if is_claude_max else "",
                latency,
            )
            # Update BackendChoice with actual values
            choice.estimated_cost_usd = cost
            choice.estimated_latency_seconds = latency
            choice.reason = primary.reason  # preserve original reason
            return text, choice

        except ClaudeNotAvailable as exc:
            # Claude path unavailable (no session / binary missing).
            # Log clearly and continue chain — this is expected in headless mode.
            latency = time.monotonic() - t0
            log.info(
                "Claude path unavailable (no active session or binary missing): %s — "
                "falling through to next backend in chain",
                exc,
            )
            attempts.append({
                "backend": choice.backend,
                "model": choice.model,
                "error": f"ClaudeNotAvailable: {exc}",
                "status_code": None,
                "latency_s": round(latency, 3),
            })
            _append_ledger(
                request_hash=request_hash,
                backend=choice.backend,
                model=choice.model,
                prompt_tokens=0,
                response_tokens=0,
                cost_usd=0.0,
                latency_s=latency,
                success=False,
                regime=regime,
                cost_source="max_subscription",
                cache_status="unknown_cli_path",
            )
            continue  # Always continue chain on ClaudeNotAvailable

        except BackendError as exc:
            latency = time.monotonic() - t0
            log.warning(
                "Backend %s/%s failed: %s (status=%s)",
                choice.backend, choice.model, exc, exc.status_code,
            )
            attempts.append({
                "backend": choice.backend,
                "model": choice.model,
                "error": str(exc),
                "status_code": exc.status_code,
                "latency_s": round(latency, 3),
            })
            _append_ledger(
                request_hash=request_hash,
                backend=choice.backend,
                model=choice.model,
                prompt_tokens=int(len(req.prompt.split()) * 1.33),
                response_tokens=0,
                cost_usd=0.0,
                latency_s=latency,
                success=False,
                regime=regime,
                cost_source="api_pay_per_token",
            )
            if not _should_retry(exc):
                log.error("Non-retryable error from %s: %s", choice.backend, exc)
                break
            continue

    raise RouterFailure(
        f"All {len(chain)} backend attempts failed for prompt hash {request_hash}",
        attempts=attempts,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified model router — routes tasks to Claude/DeepSeek/Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Routing-query mode (returns model name; no invocation):
  python scripts/unified_model_router.py --complexity low --task-kind "list files"
  python scripts/unified_model_router.py --complexity high --task-kind "architecture synthesis" --json
  python scripts/unified_model_router.py --complexity medium --task-kind "coding"

Invoke mode (runs the prompt against the selected backend):
  python scripts/unified_model_router.py --prompt "Reverse a string in Python" --complexity coding
  python scripts/unified_model_router.py --prompt "Design a new subsystem" --complexity architecture
  python scripts/unified_model_router.py --prompt "Summarize JSON" --complexity mechanical --independence
  python scripts/unified_model_router.py --prompt "Verify backtest" --complexity reasoning --independence --dry-run
        """,
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Task prompt to send to the model (invoke mode). Omit when using --task-kind.",
    )
    parser.add_argument(
        "--task-kind",
        default=None,
        dest="task_kind",
        help=(
            "Short description of the task kind for routing-query mode "
            "(e.g. 'list files', 'architecture synthesis'). "
            "Matched case-insensitively to known categories. "
            "Omit when using --prompt."
        ),
    )
    parser.add_argument(
        "--complexity",
        default=None,
        help=(
            "Task complexity. Aliases accepted: "
            "low/simple/trivial → mechanical (haiku); "
            "medium/normal → coding (sonnet); "
            "high/complex/architecture → architecture (opus). "
            "Also accepts the internal values directly: mechanical, coding, reasoning, architecture."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        dest="json_output",
        help="Output routing decision as JSON {model, task_kind, complexity, rationale} (routing-query mode).",
    )
    parser.add_argument(
        "--independence",
        action="store_true",
        default=False,
        help="Require a non-Claude backend for true independence",
    )
    parser.add_argument(
        "--context-tokens",
        type=int,
        default=0,
        help="Estimated context token count",
    )
    parser.add_argument(
        "--cost-ceiling",
        type=float,
        default=0.10,
        help="Maximum cost per call in USD (default: 0.10)",
    )
    parser.add_argument(
        "--deadline",
        type=float,
        default=60.0,
        help="Wall-clock deadline in seconds (default: 60)",
    )
    parser.add_argument(
        "--no-local",
        action="store_true",
        default=False,
        help="Disallow Ollama local backend",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Show routing decision only; do not invoke the backend (invoke mode)",
    )
    args = parser.parse_args()

    # --- Resolve complexity ---------------------------------------------------
    if args.complexity is not None:
        try:
            complexity: ComplexityLevel = _normalize_complexity(args.complexity)
        except ValueError as exc:
            parser.error(str(exc))
    elif args.task_kind is not None:
        complexity = _resolve_task_kind(args.task_kind)
    else:
        complexity = "coding"  # default for invoke mode when neither is set

    # --- Routing-query mode: --task-kind given, no --prompt needed ------------
    if args.task_kind is not None:
        req = TaskRequest(
            prompt=args.task_kind,  # used only for ledger hash; not sent to any backend
            complexity=complexity,
            independence_required=args.independence,
            context_tokens_estimate=args.context_tokens,
            cost_ceiling_usd=args.cost_ceiling,
            deadline_seconds=args.deadline,
            allow_local=not args.no_local,
        )
        choice = route(req)
        short_name = _short_model_name(choice.model)

        if args.json_output:
            out = {
                "model": short_name,
                "task_kind": args.task_kind,
                "complexity": complexity,
                "rationale": choice.reason,
            }
            print(json.dumps(out))
        else:
            # Single token to stdout; rationale to stderr so callers can capture model name cleanly
            print(short_name, flush=True)
            print(choice.reason, file=sys.stderr)
        return

    # --- Invoke mode: --prompt required ---------------------------------------
    if args.prompt is None:
        parser.error(
            "Provide either --prompt (invoke mode) or --task-kind (routing-query mode)."
        )

    req = TaskRequest(
        prompt=args.prompt,
        complexity=complexity,
        independence_required=args.independence,
        context_tokens_estimate=args.context_tokens,
        cost_ceiling_usd=args.cost_ceiling,
        deadline_seconds=args.deadline,
        allow_local=not args.no_local,
    )

    choice = route(req)
    regime = _read_pacing_regime()

    print("\n=== Routing Decision ===")
    print(f"  Pacing regime  : {regime}")
    print(f"  Backend        : {choice.backend}")
    print(f"  Model          : {choice.model}")
    print(f"  Est. cost      : ${choice.estimated_cost_usd:.6f}")
    print(f"  Est. latency   : {choice.estimated_latency_seconds:.1f}s")
    print(f"  Reason         : {choice.reason}")

    if args.dry_run:
        print("\n[dry-run] Skipping invocation.")
        return

    print("\n=== Invoking... ===")
    try:
        response_text, final_choice = invoke(req)
        print(f"\n=== Response (via {final_choice.backend}/{final_choice.model}) ===")
        print(response_text)
        print(f"\n  Actual cost    : ${final_choice.estimated_cost_usd:.6f}")
        print(f"  Actual latency : {final_choice.estimated_latency_seconds:.2f}s")
        spend_24h = get_spend(window_hours=24)
        print(f"  24h total spend: ${spend_24h:.4f}")
    except RouterFailure as exc:
        print(f"\n[ERROR] All backends failed: {exc}")
        for i, attempt in enumerate(exc.attempts, 1):
            print(f"  Attempt {i}: {attempt['backend']}/{attempt['model']} — {attempt['error']}")
        sys.exit(1)


if __name__ == "__main__":
    _cli_main()
