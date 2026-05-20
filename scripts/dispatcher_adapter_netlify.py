"""dispatcher_adapter_netlify.py — Netlify Functions adapter stub.

Free tier (verified 2026-05-17): 125,000 function invocations/mo, 100 hours
runtime, 100 GB bandwidth. Functions run on AWS Lambda. 10s wall-clock cap on
synchronous, 15 min on background functions. No credit card required.

Signup: https://app.netlify.com/signup (GitHub/GitLab/Bitbucket OAuth).

Auth model: Netlify access token from https://app.netlify.com/user/applications.
Function deployed via netlify-cli to /.netlify/functions/run_sweep. This adapter
assumes a pre-deployed Python function with the @netlify/functions-python runtime.

Submit model: Synchronous POST to the function URL; or background-function POST
to /.netlify/functions/run_sweep-background returns 202 immediately and runs up
to 15 min asynchronously.

LIMITATIONS:
 - Python on Netlify Functions is in beta as of 2026. JavaScript/Go runtimes
   are more mature. Treat this as experimental.
 - 1024 MB memory cap, 6 MB request body cap, 6 MB response body cap.
"""
from __future__ import annotations

import os
import json
import time
import urllib.request
import urllib.error


def _function_url() -> str:
    url = os.environ.get("NETLIFY_FUNCTION_URL")  # e.g. "https://myapp.netlify.app/.netlify/functions/run_sweep"
    if not url:
        raise RuntimeError("NETLIFY_FUNCTION_URL not set")
    return url


def _background() -> bool:
    return os.environ.get("NETLIFY_USE_BACKGROUND", "0") == "1"


def submit_job(ticker: str, strategy: str, job_spec: dict, dry_run: bool = False) -> dict:
    url = _function_url()
    if _background() and not url.endswith("-background"):
        url = url + "-background"
    payload = {"ticker": ticker, "strategy": strategy, **job_spec}
    if dry_run:
        return {"job_id": "DRY-RUN", "status": "would_submit",
                "url": url, "payload": payload, "background": _background()}

    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    token = os.environ.get("NETLIFY_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    timeout = 900 if _background() else 15
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            code = resp.getcode()
            if code == 202:  # background accepted
                return {"job_id": resp.headers.get("X-Nf-Request-Id", str(int(time.time()))),
                        "status": "submitted"}
            body = json.loads(resp.read().decode())
            return {"job_id": str(int(time.time())),
                    "status": "completed", "data": body}
    except urllib.error.HTTPError as e:
        return {"job_id": None,
                "status": "auth_failure" if e.code in (401, 403) else "submit_error",
                "code": e.code, "body": e.read().decode(errors="ignore")}


def check_status(job_id: str) -> dict:
    # Netlify does not expose a per-invocation status endpoint on the free tier.
    # Background functions write results to Netlify Blobs; poll the blob key.
    blob_url = os.environ.get("NETLIFY_BLOB_BASE_URL")
    if not blob_url:
        return {"status": "unknown", "note": "Set NETLIFY_BLOB_BASE_URL to poll background results"}
    full = f"{blob_url}/{job_id}.json"
    try:
        with urllib.request.urlopen(full, timeout=15) as resp:
            return {"status": "completed", "data": json.loads(resp.read().decode())}
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {"status": "running"}
        return {"status": "poll_error", "code": e.code}


if __name__ == "__main__":
    print(json.dumps(submit_job("AAPL", "D1_REV", {"thresh": 0.5}, dry_run=True), indent=2))
