"""dispatcher_adapter_digitalocean_functions.py — DigitalOcean Functions adapter stub.

Free tier (verified 2026-05-17): 90,000 GiB-seconds/month free across all
functions in your team. After that, $0.0000185/GiB-s ($0.07/GiB-hr). NO charge
per invocation. Python 3.11+ supported natively.

**Credit card status: REQUIRED for identity verification** (charged $1-2 temporarily
then refunded; PayPal $5 one-time non-refundable alternative). This is NOT a
no-CC provider, but the free quota itself is genuinely free.

Signup: https://cloud.digitalocean.com/registrations/new — verify payment method
(CC or PayPal), then enable Functions namespace via Console.

Auth: DO Personal Access Token (Bearer) issued from
https://cloud.digitalocean.com/account/api/tokens.

Submit model: doctl serverless deploy + invoke via REST API at
https://faas-<region>-<namespace>.doserverless.co/api/v1/web/<fn_path>.

KEY ADVANTAGES:
 - 90,000 GiB-s/mo free — enough for ~3.6M invocations at 100ms × 256MB.
 - Python 3.11+ native runtime.
 - Multiple regions (NYC, SFO, AMS, LON, FRA, SGP, BLR, TOR, SYD).
 - Predictable pricing past the free tier.

LIMITATIONS:
 - **CC/PayPal verification required at signup.**
 - 15-minute max function timeout.
 - 512MiB max memory per function on standard plan.
 - No GPU on Functions tier.

Docs:
 - https://docs.digitalocean.com/products/functions/
 - https://docs.digitalocean.com/products/functions/details/pricing/
"""
from __future__ import annotations

import os
import json
import urllib.request
import urllib.error


def _token() -> str:
    t = os.environ.get("DO_TOKEN")
    if not t:
        raise RuntimeError("DO_TOKEN not set")
    return t


def _function_url() -> str:
    u = os.environ.get("DO_FUNCTION_URL")
    if not u:
        raise RuntimeError("DO_FUNCTION_URL not set (https://faas-<region>-<ns>.doserverless.co/api/v1/web/<fn>)")
    return u


def submit_job(ticker: str, strategy: str, job_spec: dict, dry_run: bool = False) -> dict:
    url = _function_url() if not dry_run else "https://faas-DRY.doserverless.co/api/v1/web/DRY"
    payload = {"ticker": ticker, "strategy": strategy, "job_spec": job_spec}
    if dry_run:
        return {"job_id": "DRY-RUN", "status": "would_submit",
                "url": url, "payload": payload}
    data = json.dumps(payload).encode()
    headers = {"Authorization": f"Bearer {_token()}",
               "Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:  # up to 15min
            body = json.loads(resp.read().decode())
            return {"job_id": body.get("activationId", "do-sync"),
                    "status": "completed",
                    "result": body}
    except urllib.error.HTTPError as e:
        return {"job_id": None,
                "status": "auth_failure" if e.code in (401, 403) else "submit_error",
                "code": e.code, "body": e.read().decode(errors="ignore")}


def check_status(job_id: str) -> dict:
    return {"status": "completed", "note": "do functions web invoke is sync"}


if __name__ == "__main__":
    print(json.dumps(submit_job("AAPL", "D1_REV", {"thresh": 0.5}, dry_run=True), indent=2))
