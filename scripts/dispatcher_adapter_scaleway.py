"""dispatcher_adapter_scaleway.py — Scaleway Serverless Functions adapter stub.

Free tier (verified 2026-05-17): 1,000,000 requests/month + 400,000 GB-s/month
free per account (€0.15/M req + €1.20/100k GB-s after). No credit card required
for the free quota; payment method only required when you exceed it. EU/GDPR
hosting (Paris, Amsterdam, Warsaw, Milan).

Signup: https://console.scaleway.com/register — verify email, generate API key
under "Credentials". No CC needed.

Auth: Scaleway API key (X-Auth-Token header) + secret_key + organization_id.
Submit model: POST /functions/v1beta1/regions/:region/functions/:fn_id/invocations
  or use the OpenWhisk-compatible REST API. Functions can be created via
  POST /functions/v1beta1/regions/:region/functions.

Python runtime: Python 3.10+ supported natively. Handler format `src/handler.handle`
(file path + function name). Dependencies bundled in a package/ folder.

KEY ADVANTAGES:
 - Generous free monthly quota.
 - EU-only data residency.
 - Native Python support with multiple runtime versions.
 - OpenAI/Knative-compatible architecture.

LIMITATIONS:
 - Functions cold-start adds ~500ms latency.
 - 400k GB-s/mo = roughly 444 GPU-equivalent-seconds at 1GB memory.
 - No GPU support in serverless tier (use Scaleway Instances for GPU).
 - Native deps cannot be cross-compiled locally for Python runtime.

Docs:
 - https://www.scaleway.com/en/docs/serverless-functions/
 - https://www.scaleway.com/en/developers/api/serverless-functions
"""
from __future__ import annotations

import os
import json
import urllib.request
import urllib.error


def _region() -> str:
    return os.environ.get("SCALEWAY_REGION", "fr-par")


def _auth_token() -> str:
    t = os.environ.get("SCALEWAY_API_KEY")
    if not t:
        raise RuntimeError("SCALEWAY_API_KEY not set")
    return t


def _function_id() -> str:
    fid = os.environ.get("SCALEWAY_FUNCTION_ID")
    if not fid:
        raise RuntimeError("SCALEWAY_FUNCTION_ID not set")
    return fid


def submit_job(ticker: str, strategy: str, job_spec: dict, dry_run: bool = False) -> dict:
    fn_id = _function_id() if not dry_run else "DRY"
    region = _region()
    url = (f"https://api.scaleway.com/functions/v1beta1/regions/{region}"
           f"/functions/{fn_id}/invocations")
    payload = {
        "ticker": ticker,
        "strategy": strategy,
        "job_spec": job_spec,
    }
    if dry_run:
        return {"job_id": "DRY-RUN", "status": "would_submit",
                "url": url, "payload": payload}
    data = json.dumps(payload).encode()
    headers = {"X-Auth-Token": _auth_token(), "Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode())
            return {"job_id": str(body.get("id", "scw-async")),
                    "status": "submitted"}
    except urllib.error.HTTPError as e:
        return {"job_id": None,
                "status": "auth_failure" if e.code in (401, 403) else "submit_error",
                "code": e.code, "body": e.read().decode(errors="ignore")}


def check_status(job_id: str) -> dict:
    # Scaleway Functions invocations are typically synchronous; for async use
    # Scaleway Jobs (different API).
    return {"status": "completed", "note": "scaleway functions are synchronous"}


if __name__ == "__main__":
    print(json.dumps(submit_job("AAPL", "D1_REV", {"thresh": 0.5}, dry_run=True), indent=2))
