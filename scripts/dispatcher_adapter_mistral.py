"""dispatcher_adapter_mistral.py — Mistral La Plateforme inference adapter stub.

Free tier (verified 2026-05-17): 1,000,000,000 tokens/MONTH FREE across all
Mistral models (Large, Small, Embed, Codestral) at 2 RPM / ~1 req/sec hard cap.
NO credit card required. Permanent free tier. EU-based (data residency).

Signup: https://console.mistral.ai — sign up with email, no CC. Generate API key
under "API Keys".

Auth: Standard Authorization: Bearer MISTRAL_API_KEY header.

USE CASE: Inference (same category as Groq/Cerebras/Gemini/SambaNova). Mistral's
strengths over peers: (1) 1B tokens/month — the largest monthly token budget on
any free tier as of 2026; (2) EU data residency (GDPR-friendly); (3) Codestral
supports Fill-in-the-Middle (FIM) for code completion. Mistral's weakness:
2 RPM = throughput-bound, so high-concurrency agent workflows time out.

Submit model: POST /v1/chat/completions — Mistral-native (also OpenAI-compatible
via SDK). Models: `mistral-large-latest`, `mistral-small-latest`,
`codestral-latest`, `mistral-embed`.

KEY ADVANTAGES:
 - 1B tokens/month free — best-in-class monthly volume.
 - EU jurisdiction (GDPR).
 - FIM via Codestral — useful for code-gen workloads.
 - Embeddings included in the free quota.

LIMITATIONS:
 - 2 RPM hard cap (very low for concurrent agent loops).
 - No multimodal (text-only).
 - Tier 2 free tier (per industry rankings) — less production-ready than
   Groq/Cerebras/Gemini.

Docs:
 - https://docs.mistral.ai
 - https://console.mistral.ai
"""
from __future__ import annotations

import os
import json
import urllib.request
import urllib.error


def _api_key() -> str:
    t = os.environ.get("MISTRAL_API_KEY")
    if not t:
        raise RuntimeError("MISTRAL_API_KEY not set")
    return t


def _model() -> str:
    return os.environ.get("MISTRAL_MODEL", "mistral-small-latest")


def submit_job(ticker: str, strategy: str, job_spec: dict, dry_run: bool = False) -> dict:
    url = "https://api.mistral.ai/v1/chat/completions"
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
    return {"status": "completed", "note": "mistral inference is synchronous"}


if __name__ == "__main__":
    print(json.dumps(submit_job("AAPL", "D1_REV",
                                {"prompt": "Test"},
                                dry_run=True), indent=2))
