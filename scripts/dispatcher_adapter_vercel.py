"""dispatcher_adapter_vercel.py — Vercel Serverless Functions adapter stub.

Free tier (verified 2026-05-17): Hobby plan, 100 GB-hours/mo serverless function
execution, 10s max function duration on Hobby (60s on Pro). 100 GB bandwidth/mo.
No credit card required.

Signup: https://vercel.com/signup (GitHub OAuth).

Auth model: Vercel deploy token from https://vercel.com/account/tokens. Functions
are deployed automatically from a connected git repo or via `vercel deploy`. This
adapter assumes a pre-deployed Python function at /api/run_sweep that wraps
run_sweep.py with a 10s budget.

Submit model: Simple HTTPS POST to the function URL with JSON body. Vercel runs
the Python via the official @vercel/python builder. Synchronous response.

LIMITATIONS:
 - 10s wall-clock cap is the killer constraint. Only suitable for very short
   strategy evaluations (single-day backtests, threshold spot-checks). Long
   sweeps will time out.
 - 1024 MB memory cap on Hobby.
 - 100 GB-hr/mo quota = 100 hours of full-memory compute. Plenty for short jobs.
"""
from __future__ import annotations

import os
import json
import time
import urllib.request
import urllib.error


def _function_url() -> str:
    url = os.environ.get("VERCEL_FUNCTION_URL")  # e.g. "https://myapp.vercel.app/api/run_sweep"
    if not url:
        raise RuntimeError("VERCEL_FUNCTION_URL not set")
    return url


def _bearer_token() -> str:
    """Return the Vercel deploy token, or empty string for public functions."""
    return os.environ.get("VERCEL_TOKEN", "")


def submit_job(ticker: str, strategy: str, job_spec: dict, dry_run: bool = False) -> dict:
    url = _function_url()
    payload = {"ticker": ticker, "strategy": strategy, **job_spec}
    if dry_run:
        return {"job_id": "DRY-RUN", "status": "would_submit",
                "url": url, "payload": payload}

    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    bt = _bearer_token()
    if bt:
        headers["Authorization"] = f"Bearer {bt}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read().decode())
            return {"job_id": body.get("invocation_id", str(int(time.time()))),
                    "status": "completed",
                    "data": body}
    except urllib.error.HTTPError as e:
        return {"job_id": None,
                "status": "auth_failure" if e.code in (401, 403) else "submit_error",
                "code": e.code, "body": e.read().decode(errors="ignore")}


def check_status(job_id: str) -> dict:
    return {"status": "completed", "note": "Vercel serverless functions are synchronous"}


if __name__ == "__main__":
    print(json.dumps(submit_job("AAPL", "D1_REV", {"thresh": 0.5}, dry_run=True), indent=2))
