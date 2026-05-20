"""dispatcher_adapter_cloudflare_workers.py — Cloudflare Workers adapter stub.

Free tier (verified 2026-05-17): 100,000 requests/day, 10 ms CPU time per request
(free) or 30 s wall-clock (paid). Python support via Pyodide is in open beta as
of 2026. No credit card required to sign up; CC needed only to upgrade.

Signup: https://dash.cloudflare.com/sign-up (no CC).

Auth model: Cloudflare API token (account-scoped) from https://dash.cloudflare.com/profile/api-tokens
with "Workers Scripts:Edit" permission. The Worker is pre-deployed via wrangler
CLI and invoked over HTTPS at <name>.<sub>.workers.dev.

Submit model: Simple POST to the Worker URL. The Worker shells out to the
sp500-backtest WASM bundle compiled with Pyodide. Synchronous response under
10 ms CPU budget.

LIMITATIONS:
 - 10 ms CPU budget is the killer. This is suitable ONLY for lookup/filter
   helpers, NOT full backtests. Use it for high-frequency cheap operations
   (e.g., quote-lookup, signal-threshold checks against precomputed parquet).
 - Python in beta — expect runtime quirks.
 - No persistent filesystem; reads must come from KV / R2 / D1.
"""
from __future__ import annotations

import os
import json
import time
import urllib.request
import urllib.error


def _worker_url() -> str:
    url = os.environ.get("CF_WORKER_URL")  # e.g. "https://sp500.user.workers.dev"
    if not url:
        raise RuntimeError("CF_WORKER_URL not set")
    return url


def submit_job(ticker: str, strategy: str, job_spec: dict, dry_run: bool = False) -> dict:
    url = _worker_url()
    payload = {"ticker": ticker, "strategy": strategy, **job_spec}
    if dry_run:
        return {"job_id": "DRY-RUN", "status": "would_submit",
                "url": url, "payload": payload}
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    token = os.environ.get("CF_API_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode())
            return {"job_id": resp.headers.get("CF-Ray", str(int(time.time()))),
                    "status": "completed", "data": body}
    except urllib.error.HTTPError as e:
        return {"job_id": None,
                "status": "auth_failure" if e.code in (401, 403) else "submit_error",
                "code": e.code, "body": e.read().decode(errors="ignore")}


def check_status(job_id: str) -> dict:
    return {"status": "completed", "note": "CF Workers are synchronous"}


if __name__ == "__main__":
    print(json.dumps(submit_job("AAPL", "D1_REV", {"thresh": 0.5}, dry_run=True), indent=2))
