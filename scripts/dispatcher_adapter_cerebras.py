"""dispatcher_adapter_cerebras.py — Cerebras Inference Cloud adapter stub.

Free tier (verified 2026-05-17): 1,000,000 tokens/day FREE, 30 req/min, 60k-100k
TPM, 8k context cap (across all free-tier models). NO credit card required.
Free tier is PERMANENT (not a trial). Industry-leading 2,500-2,600 tokens/sec
output speed on wafer-scale CS-3 hardware.

Signup: https://cloud.cerebras.ai (or https://inference.cerebras.ai) —
sign up with email, no CC, no waitlist. API key issued immediately.

Models on free tier: Llama 3.3 70B, Llama 4 Scout, Qwen 3 32B, Qwen 3 235B,
GPT-OSS 120B (OpenAI open weights).

USE CASE FOR THIS DISPATCHER: Same as Groq (inference, not compute) but with
~10x daily token budget. Best for high-volume agentic workflows, batch
classification, content generation pipelines, daily report automation.

Submit model: POST /v1/chat/completions — OpenAI-compatible. SDK: `cerebras-cloud-sdk`
or use the OpenAI Python SDK with `base_url=https://api.cerebras.ai/v1`.

KEY ADVANTAGES:
 - 1M tokens/day permanent free — the most generous LLM daily volume in 2026.
 - 2,500+ tokens/sec output speed (3x Groq, 10x typical GPU).
 - No CC, no waitlist, instant signup.
 - OpenAI SDK compatible.

LIMITATIONS:
 - 8k context cap on free tier (paid tier raises to 128k+).
 - 30 RPM rate limit — must batch concurrent calls within the cap.
 - Only open-weight models.
 - No fine-tuning on free tier.
 - No batch API or async/scheduled inference on free tier.

Docs:
 - https://inference-docs.cerebras.ai
 - https://www.cerebras.ai/pricing
"""
from __future__ import annotations

import os
import json
import urllib.request
import urllib.error


def _api_key() -> str:
    t = os.environ.get("CEREBRAS_API_KEY")
    if not t:
        raise RuntimeError("CEREBRAS_API_KEY not set")
    return t


def _model() -> str:
    return os.environ.get("CEREBRAS_MODEL", "llama-3.3-70b")


def submit_job(ticker: str, strategy: str, job_spec: dict, dry_run: bool = False) -> dict:
    url = "https://api.cerebras.ai/v1/chat/completions"
    messages = job_spec.get("messages")
    if messages is None:
        prompt = job_spec.get("prompt", f"Analyze {ticker} for strategy {strategy}.")
        messages = [{"role": "user", "content": prompt}]
    payload = {
        "model": job_spec.get("model", _model()),
        "messages": messages,
        "temperature": job_spec.get("temperature", 0.2),
        "max_completion_tokens": job_spec.get("max_tokens", 1024),
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
    return {"status": "completed", "note": "cerebras inference is synchronous"}


if __name__ == "__main__":
    print(json.dumps(submit_job("AAPL", "D1_REV",
                                {"prompt": "Test"},
                                dry_run=True), indent=2))
