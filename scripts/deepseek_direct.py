"""
deepseek_direct.py — Lightweight direct DeepSeek API caller.

Replaces openclaw subprocess calls (which take 80-90s due to Drive filesystem
latency during Node.js startup) with direct Python urllib calls (~2s round-trip).

API key is read at runtime from the openclaw auth-profiles.json in Drive home.
No third-party dependencies — stdlib only (urllib, json).

Usage:
    from deepseek_direct import call_deepseek_direct

    text = call_deepseek_direct("your prompt here", timeout=30)
    # Returns response text, or "" on any error.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

_KEYFILE = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools/home/.openclaw/agents/main/agent/auth-profiles.json"
)
_API_URL = "https://api.deepseek.com/v1/chat/completions"
_MODEL = "deepseek-chat"  # deepseek-v4-flash maps to deepseek-chat on their API
_CACHED_KEY: str | None = None


def _get_api_key() -> str:
    global _CACHED_KEY
    if _CACHED_KEY:
        return _CACHED_KEY
    with open(_KEYFILE) as f:
        d = json.load(f)
    key = d["profiles"]["deepseek:default"]["key"]
    _CACHED_KEY = key
    return key


def call_deepseek_direct(
    prompt: str,
    *,
    timeout: int = 30,
    max_tokens: int = 512,
    temperature: float = 0.3,
    model: str = _MODEL,
) -> str:
    """
    Call DeepSeek API directly.  Returns response text or "" on failure.
    Never raises — all exceptions are caught and return "".
    """
    try:
        api_key = _get_api_key()
        payload = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode("utf-8")
        req = urllib.request.Request(
            _API_URL,
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        choices = result.get("choices") or []
        if choices:
            return (choices[0].get("message") or {}).get("content", "").strip()
        return ""
    except urllib.error.HTTPError as e:
        # Surface rate-limit or auth errors clearly in logs
        body = ""
        try:
            body = e.read().decode("utf-8")[:200]
        except Exception:
            pass
        raise RuntimeError(f"DeepSeek HTTP {e.code}: {body}") from e
    except Exception as exc:
        raise RuntimeError(f"DeepSeek call error: {exc}") from exc
