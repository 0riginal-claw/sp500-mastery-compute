"""dispatcher_adapter_snapdeploy.py — SnapDeploy Docker-native adapter stub.

Free tier (verified 2026-05-17): 4 containers max, 512 MB RAM + 0.25 vCPU each,
10 deploys/day. Auto-sleep on idle, auto-wake on traffic. NO credit card required.
Python-friendly via Dockerfile.

Signup: https://snapdeploy.dev — Sign up with GitHub OAuth. No CC.

Auth: SnapDeploy personal access token (Bearer).
Submit model: POST /api/v1/containers — push image reference + env vars to
trigger a deploy; container becomes accessible at a unique subdomain.

For batch/compute jobs (one-shot tickers), push a Docker image that exits on
completion: SnapDeploy treats it as a deploy event that runs once.

KEY ADVANTAGES:
 - No CC, Docker-native (any Python image runs).
 - 10 deploys/day = 10 batch jobs/day per account.
 - Auto-sleep means dormant containers don't burn quota.

LIMITATIONS:
 - 512MB RAM cap — fine for many quant workloads, tight for ML training.
 - 4-container concurrency cap.
 - 10 deploys/day caps batch-job velocity.
 - REST API surface is small; some interactions require dashboard.

Docs:
 - https://snapdeploy.dev/docs (anchor TBD — verify on signup)
 - https://snapdeploy.dev/blog/host-python-web-app-free-2026-guide
"""
from __future__ import annotations

import os
import json
import urllib.request
import urllib.error


def _server() -> str:
    return os.environ.get("SNAPDEPLOY_SERVER", "https://api.snapdeploy.dev")


def _token() -> str:
    t = os.environ.get("SNAPDEPLOY_TOKEN")
    if not t:
        raise RuntimeError("SNAPDEPLOY_TOKEN not set")
    return t


def submit_job(ticker: str, strategy: str, job_spec: dict, dry_run: bool = False) -> dict:
    image = os.environ.get("SNAPDEPLOY_IMAGE", "ghcr.io/0riginal-claw/sp500-compute:latest")
    url = f"{_server()}/api/v1/containers"
    payload = {
        "image": image,
        "name": f"job-{ticker}-{strategy}".lower(),
        "env": {
            "TICKER": ticker,
            "STRATEGY": strategy,
            "JOB_SPEC_JSON": json.dumps(job_spec),
        },
        "command": ["python", "-m", "worker", "--ticker", ticker, "--strategy", strategy],
    }
    if dry_run:
        return {"job_id": "DRY-RUN", "status": "would_submit",
                "url": url, "payload": payload}
    data = json.dumps(payload).encode()
    headers = {"Authorization": f"Bearer {_token()}", "Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode())
            return {"job_id": body.get("id"), "status": "submitted"}
    except urllib.error.HTTPError as e:
        return {"job_id": None,
                "status": "auth_failure" if e.code in (401, 403) else "submit_error",
                "code": e.code, "body": e.read().decode(errors="ignore")}


def check_status(job_id: str) -> dict:
    url = f"{_server()}/api/v1/containers/{job_id}"
    headers = {"Authorization": f"Bearer {_token()}"}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode())
            st = body.get("status", "unknown")
            mapping = {"running": "running", "starting": "pending",
                       "exited": "completed", "failed": "failed",
                       "sleeping": "completed"}
            return {"status": mapping.get(st, "unknown"), "snapdeploy_status": st}
    except urllib.error.HTTPError as e:
        return {"status": "poll_error", "code": e.code}


if __name__ == "__main__":
    print(json.dumps(submit_job("AAPL", "D1_REV", {"thresh": 0.5}, dry_run=True), indent=2))
