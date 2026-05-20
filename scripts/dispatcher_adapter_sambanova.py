"""dispatcher_adapter_sambanova.py — SambaNova Cloud RDU inference adapter stub.

Free tier (verified 2026-05-17): $5 starter credits (expire ~3 months) + a
permanent rate-limited free tier (10-30 RPM depending on model). NO credit card
required. Reaches 435+ tokens/sec on frontier open-weight models (Llama 3.1
405B, DeepSeek V3.1, Qwen, MiniMax M2).

Signup: https://cloud.sambanova.ai — sign in (Google/email), no CC.

Auth: Standard Authorization: Bearer SAMBANOVA_API_KEY header.

USE CASE: Inference (same category as Groq/Cerebras/Gemini). SambaNova's edge
is that it serves the LARGEST open-weight models on its free tier — including
Llama 3.1 405B and DeepSeek V3.1 671B — that Groq and Cerebras don't serve.
Reach for SambaNova when model quality > volume.

Submit model: POST /v1/chat/completions — OpenAI-compatible (same SDK as Groq).

KEY ADVANTAGES:
 - Largest open-weight models accessible on free tier (405B, 671B).
 - 435+ tok/s on frontier models — fastest non-Cerebras provider.
 - OpenAI-compatible API.

LIMITATIONS:
 - Starter $5 credit only — sustained use depends on rate-limited free tier.
 - 10 RPM for 405B (very tight); 30 RPM for 8B-class models.
 - Free-tier token cap is undocumented but practical — burst-only.
 - No fine-tuning on free tier.

Docs:
 - https://cloud.sambanova.ai/plans
 - https://docs.sambanova.ai
"""
from __future__ import annotations

import os
import json
import urllib.request
import urllib.error


def _api_key() -> str:
    t = os.environ.get("SAMBANOVA_API_KEY")
    if not t:
        raise RuntimeError("SAMBANOVA_API_KEY not set")
    return t


def _model() -> str:
    return os.environ.get("SAMBANOVA_MODEL", "Meta-Llama-3.1-8B-Instruct")


def submit_job(ticker: str, strategy: str, job_spec: dict, dry_run: bool = False) -> dict:
    url = "https://api.sambanova.ai/v1/chat/completions"
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
    return {"status": "completed", "note": "sambanova inference is synchronous"}


if __name__ == "__main__":
    print(json.dumps(submit_job("AAPL", "D1_REV",
                                {"prompt": "Test"},
                                dry_run=True), indent=2))
