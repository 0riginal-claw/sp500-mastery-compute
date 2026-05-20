"""dispatcher_adapter_gemini_api.py — Google AI Studio Gemini API adapter stub.

Free tier (verified 2026-05-17, post April-2026 enforcement):
 - Gemini 2.5 Pro: 5 RPM / 100 RPD (preview — Pro paywalled for free users)
 - Gemini 2.5 Flash: 10 RPM / 250 RPD
 - Gemini 2.5 Flash-Lite: 15 RPM / 1,000 RPD
 - Universal 250k TPM cap across all models.

NO credit card required for the free tier. Free tier limits reduced ~50-80% in
December 2025; effective April 2026 the Pro models are paid-only for new free
accounts. The free tier remains for Flash/Flash-Lite.

Signup: https://aistudio.google.com — sign in with Google account, click
"Get API key". No CC.

Auth: Standard API key in `x-goog-api-key` header or `?key=` query param.

USE CASE: Inference (same category as Groq/Cerebras). Gemini's strengths over
Groq/Cerebras: 1M-token context window, multimodal (image, audio, video), and
native Google Search grounding for current-events / news.

Submit model: POST /v1/models/<model>:generateContent — Google's native API,
NOT OpenAI-compatible (use the genai SDK or raw REST).

KEY ADVANTAGES:
 - Permanent free tier, no CC, multimodal.
 - 1M-token context (largest among free-tier providers).
 - Native search grounding (Tavily-equivalent, free).
 - Flash-Lite at 1,000 RPD = 1 req per 86 sec sustained.

LIMITATIONS:
 - **Data privacy: free-tier prompts ARE used for model training.** Paid tier
   ($) and Vertex AI free of this clause. DO NOT send proprietary trading
   strategy details on free tier.
 - Pro models paywalled (April 2026 change).
 - 250 RPD on Flash — burst-friendly, not high-volume.
 - Per-project, not per-key; multiple keys do NOT raise the cap.
 - Limits change frequently (Dec 2025 cut by 50-80%); pin via your dashboard.

Docs:
 - https://ai.google.dev/gemini-api/docs/rate-limits
 - https://ai.google.dev/gemini-api/docs/pricing
"""
from __future__ import annotations

import os
import json
import urllib.request
import urllib.error


def _api_key() -> str:
    t = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not t:
        raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) not set")
    return t


def _model() -> str:
    return os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


def submit_job(ticker: str, strategy: str, job_spec: dict, dry_run: bool = False) -> dict:
    model = job_spec.get("model", _model())
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    prompt = job_spec.get("prompt", f"Analyze {ticker} for strategy {strategy}.")
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": job_spec.get("temperature", 0.2),
            "maxOutputTokens": job_spec.get("max_tokens", 1024),
        },
    }
    if dry_run:
        return {"job_id": "DRY-RUN", "status": "would_submit",
                "url": url, "payload": payload}
    data = json.dumps(payload).encode()
    headers = {"x-goog-api-key": _api_key(),
               "Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode())
            try:
                text = body["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError):
                text = None
            return {"job_id": body.get("responseId", "gemini-sync"),
                    "status": "completed",
                    "result": text,
                    "usage": body.get("usageMetadata")}
    except urllib.error.HTTPError as e:
        return {"job_id": None,
                "status": ("rate_limited" if e.code == 429 else
                           "auth_failure" if e.code in (401, 403) else "submit_error"),
                "code": e.code, "body": e.read().decode(errors="ignore")}


def check_status(job_id: str) -> dict:
    return {"status": "completed", "note": "gemini api is synchronous"}


if __name__ == "__main__":
    print(json.dumps(submit_job("AAPL", "D1_REV",
                                {"prompt": "Test"},
                                dry_run=True), indent=2))
