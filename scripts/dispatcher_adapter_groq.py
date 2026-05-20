"""dispatcher_adapter_groq.py — Groq LPU inference API adapter stub.

Free tier (verified 2026-05-17): 30 RPM / 6,000 TPM / 14,400 RPD on Llama 3.1
8B Instant. Other models have varying limits (Llama 4 Maverick: 15 RPM / 3k TPM
/ 500 RPD; Gemma 2 9B: 30 RPM / 15k TPM / 1k RPD). 1,000 RPD on most others.
NO credit card required. Free tier is PERMANENT (not a trial). OpenAI-compatible
endpoint.

Signup: https://console.groq.com — sign in with GitHub/Google. Generate API key.

Auth: Standard Authorization: Bearer GROQ_API_KEY header.

USE CASE FOR THIS DISPATCHER: Inference-only — not for backtesting compute.
Suitable for: signal-generation prompts on news/filings, daily commentary
summarization, structured data extraction from headlines, sentiment scoring,
agent-orchestrated decision steps. NOT suitable for: pandas/numpy compute,
heavy backtest sweeps, file I/O — Groq is API-call-only.

Submit model: POST /openai/v1/chat/completions — OpenAI-compatible. Models
include `llama-3.3-70b-versatile`, `llama-4-scout-17b-16e-instruct`,
`llama-4-maverick-17b-128e-instruct`, `qwen-3-32b`, `gpt-oss-20b`,
`groq/compound` (agentic — limited to 250 req/day).

KEY ADVANTAGES:
 - Permanent free tier, no CC.
 - 300-1000+ tokens/sec output speed (10x typical GPU inference).
 - OpenAI-SDK-compatible (one-line base_url change).
 - Sub-200ms time-to-first-token.

LIMITATIONS:
 - Only open-weight models (no GPT/Claude/Gemini).
 - 14,400 req/day cap = 10 req/min sustained avg.
 - No private model deployment on free tier.
 - 8k-128k context depending on model.

Docs:
 - https://console.groq.com/docs/quickstart
 - https://console.groq.com/docs/rate-limits
"""
from __future__ import annotations

import os
import json
import urllib.request
import urllib.error


def _api_key() -> str:
    t = os.environ.get("GROQ_API_KEY")
    if not t:
        raise RuntimeError("GROQ_API_KEY not set")
    return t


def _model() -> str:
    return os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")


def submit_job(ticker: str, strategy: str, job_spec: dict, dry_run: bool = False) -> dict:
    """Submit an inference job. job_spec must contain `messages` or `prompt`."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    messages = job_spec.get("messages")
    if messages is None:
        prompt = job_spec.get("prompt", f"Analyze {ticker} for strategy {strategy}.")
        messages = [{"role": "user", "content": prompt}]
    payload = {
        "model": job_spec.get("model", _model()),
        "messages": messages,
        "temperature": job_spec.get("temperature", 0.2),
        "max_tokens": job_spec.get("max_tokens", 1024),
    }
    if dry_run:
        return {"job_id": "DRY-RUN", "status": "would_submit",
                "url": url, "payload": payload}
    data = json.dumps(payload).encode()
    headers = {"Authorization": f"Bearer {_api_key()}",
               "Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode())
            return {"job_id": body.get("id"),
                    "status": "completed",
                    "result": body.get("choices", [{}])[0].get("message", {}).get("content"),
                    "usage": body.get("usage")}
    except urllib.error.HTTPError as e:
        return {"job_id": None,
                "status": ("rate_limited" if e.code == 429 else
                           "auth_failure" if e.code in (401, 403) else "submit_error"),
                "code": e.code, "body": e.read().decode(errors="ignore")}


def check_status(job_id: str) -> dict:
    # Synchronous API — submit_job returns the final result inline.
    return {"status": "completed", "note": "groq inference is synchronous"}


if __name__ == "__main__":
    print(json.dumps(submit_job("AAPL", "D1_REV",
                                {"prompt": "Test"},
                                dry_run=True), indent=2))
