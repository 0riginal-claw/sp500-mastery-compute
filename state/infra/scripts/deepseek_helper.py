"""
deepseek_helper.py — Single-call CLI interface to DeepSeek via openclaw-gdrive.

Calls DeepSeek v4-flash through the openclaw-gdrive subprocess and returns
JSON to stdout. Uses only stdlib — no extra dependencies.

Usage
-----
    python scripts/deepseek_helper.py --prompt "Verify this logic for errors" --max-tokens 1000
    python scripts/deepseek_helper.py --prompt "Second-opinion on this backtest" --model deepseek/deepseek-v4-flash
    python scripts/deepseek_helper.py --prompt "Summarize 300k tokens of data" --timeout 180

Output (stdout, always JSON)
-----------------------------
    Success:
        {
            "success": true,
            "text": "...",
            "model": "deepseek/deepseek-v4-flash",
            "latency_s": 7.21,
            "cost_usd": 0.0000014,
            "backend": "deepseek_openclaw"
        }
    Error:
        {
            "success": false,
            "error": "openclaw exited 1: ...",
            "backend": "deepseek_openclaw"
        }

When to use
-----------
    Use for tasks where you need:
    - A genuine second opinion from a different alignment (not just delegation)
    - Context windows >200k tokens (DeepSeek has 977k context)
    - Independent cross-check before high-stakes decisions
    - Research where DeepSeek's training provides a distinct perspective

    Cost: ~$0.000001/call (blended deepseek-v4-flash rate, based on word count).
    This is ~1000-5000x cheaper than Claude Haiku via API.

    Do NOT use for:
    - Tasks requiring tool use or file writes (spawn Claude sub-agent)
    - Simple text manipulation where alignment doesn't matter (use ollama_helper.py, free)

Requirements
------------
    openclaw-gdrive must exist at the standard path (see _OPENCLAW_BIN below).
    DeepSeek API credentials must be configured in openclaw.
    No extra Python packages required (stdlib only).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

_OPENCLAW_BIN = Path(
    "/Users/orginal/Library/CloudStorage/"
    "GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/bin/openclaw-gdrive"
)

# DeepSeek v4-flash pricing (USD per million tokens, blended input+output average)
# Input: $0.07/M, Output: $0.28/M — blended at 60% input / 40% output split
_DEEPSEEK_BLENDED_PER_M = 0.154  # ($0.07 * 0.6) + ($0.28 * 0.4)


def _estimate_cost(prompt: str, response_text: str) -> float:
    """Estimate DeepSeek call cost in USD.

    Uses word-count approximation (words * 1.33 ≈ tokens) for both prompt
    and response. Applies the blended deepseek-v4-flash rate.

    Args:
        prompt: The prompt string sent to DeepSeek.
        response_text: The response text received.

    Returns:
        Estimated cost in USD (typically $0.000001 - $0.000010 per call).
    """
    total_words = len(prompt.split()) + len(response_text.split())
    approx_tokens = total_words * 1.33
    return round((approx_tokens / 1_000_000) * _DEEPSEEK_BLENDED_PER_M, 9)


def _parse_openclaw_output(raw: str) -> Optional[str]:
    """Extract response text from openclaw JSON output.

    openclaw may return JSON in several formats depending on version:
    - {"outputs": [{"text": "..."}]}  (capability model run format)
    - {"result": {"text": "..."}}
    - {"text": "..."}
    - Preamble text before the first "{" (preamble is skipped)

    Args:
        raw: Raw stdout string from openclaw subprocess.

    Returns:
        Extracted text string, or None if no parseable text found.
    """
    # Strip any preamble before first JSON object
    start = raw.find("{")
    if start < 0:
        return None

    try:
        data = json.loads(raw[start:])
    except json.JSONDecodeError:
        # Try to find end of first JSON object by scanning for balanced braces
        # This handles cases where openclaw emits trailing garbage after JSON
        depth = 0
        end = start
        for i, ch in enumerate(raw[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        try:
            data = json.loads(raw[start:end])
        except json.JSONDecodeError:
            return None

    # Format 1: outputs[0].text  (capability model run)
    outputs = data.get("outputs", [])
    if outputs and isinstance(outputs, list):
        text = outputs[0].get("text", "")
        if text:
            return text

    # Format 2: result.text
    result = data.get("result", {})
    if isinstance(result, dict):
        text = result.get("text", "")
        if text:
            return text

    # Format 3: top-level text
    text = data.get("text", "")
    if text:
        return text

    # Format 4: choices[0].message.content  (OpenAI-compatible)
    choices = data.get("choices", [])
    if choices:
        text = choices[0].get("message", {}).get("content", "")
        if text:
            return text

    return None


def call_deepseek(
    prompt: str,
    *,
    model: str = "deepseek/deepseek-v4-flash",
    max_tokens: int = 1000,
    timeout: float = 120.0,
    openclaw_bin: Path = _OPENCLAW_BIN,
) -> dict:
    """Call DeepSeek via openclaw-gdrive subprocess.

    Constructs and runs the standard openclaw capability model run command,
    parses the JSON output, and estimates the cost.

    Args:
        prompt: The task prompt string.
        model: DeepSeek model identifier (default: deepseek/deepseek-v4-flash).
        max_tokens: Maximum tokens to generate (passed to openclaw if supported).
        timeout: Subprocess timeout in seconds.
        openclaw_bin: Path to the openclaw-gdrive binary.

    Returns:
        A dict with keys: success (bool), text or error (str), model (str),
        latency_s (float), cost_usd (float), backend (str).
    """
    if not openclaw_bin.exists():
        return {
            "success": False,
            "error": f"openclaw-gdrive not found at {openclaw_bin}",
            "backend": "deepseek_openclaw",
            "latency_s": 0.0,
        }

    cmd = [
        str(openclaw_bin),
        "capability", "model", "run",
        "--local",
        "--model", model,
        "--json",
        "--prompt", prompt,
    ]

    t0 = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        latency = time.monotonic() - t0
        return {
            "success": False,
            "error": f"openclaw timed out after {timeout}s",
            "backend": "deepseek_openclaw",
            "latency_s": round(latency, 3),
        }
    except FileNotFoundError:
        latency = time.monotonic() - t0
        return {
            "success": False,
            "error": f"openclaw-gdrive binary not executable at {openclaw_bin}",
            "backend": "deepseek_openclaw",
            "latency_s": round(latency, 3),
        }

    latency = time.monotonic() - t0

    if result.returncode != 0:
        stderr_snippet = result.stderr.strip()[:400]
        return {
            "success": False,
            "error": f"openclaw exited {result.returncode}: {stderr_snippet}",
            "backend": "deepseek_openclaw",
            "latency_s": round(latency, 3),
        }

    raw = result.stdout.strip()
    if not raw:
        return {
            "success": False,
            "error": "openclaw returned empty stdout",
            "backend": "deepseek_openclaw",
            "latency_s": round(latency, 3),
        }

    text = _parse_openclaw_output(raw)
    if text is None:
        return {
            "success": False,
            "error": f"Could not parse openclaw output: {raw[:300]}",
            "backend": "deepseek_openclaw",
            "latency_s": round(latency, 3),
        }

    cost = _estimate_cost(prompt, text)

    return {
        "success": True,
        "text": text,
        "model": model,
        "latency_s": round(latency, 3),
        "cost_usd": cost,
        "backend": "deepseek_openclaw",
    }


def _cli_main() -> None:
    """Entry point for command-line invocation."""
    parser = argparse.ArgumentParser(
        description="Call DeepSeek via openclaw-gdrive and return JSON to stdout.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/deepseek_helper.py --prompt "Verify this backtest for look-ahead bias: ..."
  python scripts/deepseek_helper.py --prompt "Second-opinion: is this logic correct?" --max-tokens 1500
  python scripts/deepseek_helper.py --prompt "Summarize 500k tokens of logs" --timeout 180

Parse output in Python:
  import subprocess, json
  r = subprocess.run(
      ["python", "scripts/deepseek_helper.py", "--prompt", "task"],
      capture_output=True, text=True
  )
  result = json.loads(r.stdout)
  if result["success"]:
      print(result["text"])
      print(f"Cost: ${result['cost_usd']:.8f}")
        """,
    )
    parser.add_argument("--prompt", required=True, help="Task prompt")
    parser.add_argument(
        "--model",
        default="deepseek/deepseek-v4-flash",
        help="DeepSeek model identifier (default: deepseek/deepseek-v4-flash)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1000,
        help="Maximum tokens to generate (default: 1000)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Subprocess timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--openclaw-bin",
        default=str(_OPENCLAW_BIN),
        help=f"Path to openclaw-gdrive binary (default: {_OPENCLAW_BIN})",
    )
    args = parser.parse_args()

    result = call_deepseek(
        args.prompt,
        model=args.model,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        openclaw_bin=Path(args.openclaw_bin),
    )
    print(json.dumps(result, ensure_ascii=False))
    sys.exit(0 if result.get("success") else 1)


if __name__ == "__main__":
    _cli_main()
