"""
ollama_helper.py — Single-call CLI interface to local Ollama.

Calls Ollama at http://localhost:11434 and returns JSON to stdout.
Tries POST /v1/chat/completions (OpenAI-compatible) first; falls back
to POST /api/generate (native Ollama) if the completions endpoint returns 404.

Usage
-----
    python scripts/ollama_helper.py --prompt "Summarize this JSON: {...}" --max-tokens 500
    python scripts/ollama_helper.py --prompt "List all .py files" --model qwen2.5-coder:7b
    python scripts/ollama_helper.py --prompt "Convert CSV to JSON" --system "You are a data converter."

Output (stdout, always JSON)
-----------------------------
    Success:
        {
            "success": true,
            "text": "...",
            "model": "qwen2.5-coder:7b",
            "latency_s": 8.34,
            "cost_usd": 0.0,
            "backend": "ollama_local"
        }
    Error:
        {
            "success": false,
            "error": "Ollama not reachable at http://localhost:11434: ...",
            "backend": "ollama_local"
        }

When to use
-----------
    Use for mechanical text work where no real reasoning or tool use is needed:
    - JSON parsing / summarization
    - Format conversion (CSV↔JSON, YAML↔TOML)
    - List filtering / sorting
    - Log parsing and extraction
    - Template filling
    - String transformations

    Do NOT use for:
    - Tasks requiring tool use or file writes (spawn Claude sub-agent)
    - Tasks requiring independent alignment cross-check (use deepseek_helper.py)
    - Complex multi-step reasoning (spawn Claude sub-agent)

Requirements
------------
    pip install httpx  (httpx is already installed in this workspace)
    Ollama must be running: brew install ollama && ollama serve
    Model must be pulled: ollama pull qwen2.5-coder:7b
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

import httpx

_OLLAMA_BASE = "http://localhost:11434"

# Universal mandates: resolve ~/.zg/mandates.md from real-HOME or Drive-HOME.
# Returned as the default `system` prompt unless the caller explicitly passes
# one or sets OLLAMA_SKIP_MANDATES=1.
_MANDATES_CANDIDATES = [
    "/Users/orginal/.zg/mandates.md",
    (
        "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone"
        "@gmail.com/My Drive/AI-Tools/home/.zg/mandates.md"
    ),
    os.path.expanduser("~/.zg/mandates.md"),
]


def _load_mandates() -> str:
    """Return the contents of ~/.zg/mandates.md, or empty string if missing.

    Skipped entirely if OLLAMA_SKIP_MANDATES=1 in the env.
    """
    if os.environ.get("OLLAMA_SKIP_MANDATES") == "1":
        return ""
    for path in _MANDATES_CANDIDATES:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                content = fh.read()
                if content.strip():
                    return content
        except (FileNotFoundError, PermissionError, OSError):
            continue
    return ""


def call_ollama(
    prompt: str,
    *,
    model: str = "qwen2.5-coder:7b",
    max_tokens: int = 1000,
    system: Optional[str] = None,
    timeout: float = 90.0,
    base_url: str = _OLLAMA_BASE,
) -> dict:
    """Call local Ollama and return a result dict.

    Tries the OpenAI-compatible /v1/chat/completions endpoint first.
    Falls back to the native /api/generate endpoint if the first returns 404
    or any non-2xx status, since older Ollama builds may not expose /v1/.

    Args:
        prompt: The user prompt string.
        model: Ollama model tag (e.g. "qwen2.5-coder:7b").
        max_tokens: Maximum tokens to generate.
        system: Optional system prompt injected as a system message.
        timeout: HTTP request timeout in seconds.
        base_url: Ollama base URL (default: http://localhost:11434).

    Returns:
        A dict with keys: success (bool), text or error (str), model (str),
        latency_s (float), cost_usd (float), backend (str).
    """
    t0 = time.monotonic()

    # Universal mandates: if caller didn't supply a system prompt, inject the
    # workspace mandates so Ollama-routed agents inherit them. Skip with
    # OLLAMA_SKIP_MANDATES=1 env var.
    effective_system = system
    if effective_system is None or effective_system == "":
        mandates = _load_mandates()
        if mandates:
            effective_system = mandates

    # --- Attempt 1: OpenAI-compatible /v1/chat/completions -------------------
    messages: list[dict] = []
    if effective_system:
        messages.append({"role": "system", "content": effective_system})
    messages.append({"role": "user", "content": prompt})

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(
                f"{base_url}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "stream": False,
                },
            )
        if resp.status_code == 200:
            latency = time.monotonic() - t0
            data = resp.json()
            text = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            return {
                "success": True,
                "text": text,
                "model": model,
                "latency_s": round(latency, 3),
                "cost_usd": 0.0,
                "backend": "ollama_local",
                "endpoint": "v1/chat/completions",
            }
        # 404 → fall through to /api/generate
        if resp.status_code != 404:
            latency = time.monotonic() - t0
            return {
                "success": False,
                "error": f"Ollama /v1/chat/completions returned {resp.status_code}: {resp.text[:300]}",
                "backend": "ollama_local",
                "latency_s": round(latency, 3),
            }
    except httpx.ConnectError as exc:
        latency = time.monotonic() - t0
        return {
            "success": False,
            "error": f"Ollama not reachable at {base_url}: {exc}",
            "backend": "ollama_local",
            "latency_s": round(latency, 3),
        }
    except httpx.TimeoutException as exc:
        latency = time.monotonic() - t0
        return {
            "success": False,
            "error": f"Ollama request timed out after {timeout}s: {exc}",
            "backend": "ollama_local",
            "latency_s": round(latency, 3),
        }

    # --- Attempt 2: Native /api/generate fallback ----------------------------
    generate_payload: dict = {
        "model": model,
        "prompt": (
            prompt if not effective_system else f"{effective_system}\n\n{prompt}"
        ),
        "stream": False,
        "options": {"num_predict": max_tokens},
    }

    try:
        with httpx.Client(timeout=timeout) as client:
            resp2 = client.post(f"{base_url}/api/generate", json=generate_payload)
        latency = time.monotonic() - t0
        if resp2.status_code == 200:
            data2 = resp2.json()
            text2 = data2.get("response", "")
            return {
                "success": True,
                "text": text2,
                "model": model,
                "latency_s": round(latency, 3),
                "cost_usd": 0.0,
                "backend": "ollama_local",
                "endpoint": "api/generate",
            }
        return {
            "success": False,
            "error": f"Ollama /api/generate returned {resp2.status_code}: {resp2.text[:300]}",
            "backend": "ollama_local",
            "latency_s": round(latency, 3),
        }
    except httpx.ConnectError as exc:
        latency = time.monotonic() - t0
        return {
            "success": False,
            "error": f"Ollama not reachable at {base_url}: {exc}",
            "backend": "ollama_local",
            "latency_s": round(latency, 3),
        }
    except httpx.TimeoutException as exc:
        latency = time.monotonic() - t0
        return {
            "success": False,
            "error": f"Ollama /api/generate timed out after {timeout}s: {exc}",
            "backend": "ollama_local",
            "latency_s": round(latency, 3),
        }


def _cli_main() -> None:
    """Entry point for command-line invocation."""
    parser = argparse.ArgumentParser(
        description="Call local Ollama and return JSON to stdout.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/ollama_helper.py --prompt "Summarize: {data}" --max-tokens 300
  python scripts/ollama_helper.py --prompt "Convert CSV to JSON" --model qwen2.5-coder:7b
  python scripts/ollama_helper.py --prompt "Filter these items" --system "You are a data filter."

Parse output in Python:
  import subprocess, json
  r = subprocess.run(["python", "scripts/ollama_helper.py", "--prompt", "task"], capture_output=True, text=True)
  result = json.loads(r.stdout)
  if result["success"]:
      print(result["text"])
        """,
    )
    parser.add_argument("--prompt", required=True, help="Task prompt")
    parser.add_argument(
        "--model",
        default="qwen2.5-coder:7b",
        help="Ollama model tag (default: qwen2.5-coder:7b)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1000,
        help="Maximum tokens to generate (default: 1000)",
    )
    parser.add_argument(
        "--system",
        default=None,
        help="Optional system prompt",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=90.0,
        help="HTTP timeout in seconds (default: 90)",
    )
    parser.add_argument(
        "--base-url",
        default=_OLLAMA_BASE,
        help=f"Ollama base URL (default: {_OLLAMA_BASE})",
    )
    args = parser.parse_args()

    result = call_ollama(
        args.prompt,
        model=args.model,
        max_tokens=args.max_tokens,
        system=args.system,
        timeout=args.timeout,
        base_url=args.base_url,
    )
    print(json.dumps(result, ensure_ascii=False))
    sys.exit(0 if result.get("success") else 1)


if __name__ == "__main__":
    _cli_main()
