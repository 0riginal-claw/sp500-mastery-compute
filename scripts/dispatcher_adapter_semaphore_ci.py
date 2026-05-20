"""dispatcher_adapter_semaphore_ci.py — Semaphore CI adapter stub.

Free tier (verified 2026-05-17): 100 private builds/month free, FREE FOREVER
for open-source projects (unlimited). Source-available enterprise features
are free for self-hosting. Self-hosted agents are free unlimited.

Signup: https://semaphoreci.com — sign in with GitHub/GitLab/Bitbucket.
For private repo CI: free tier gives 100 builds/month with shared workers.
For OSS: unlimited.

Auth: Semaphore API token (Bearer) issued from the account settings page.

Submit model: POST /api/v2/projects/:project_id/workflows — trigger a workflow
by branch + pipeline name + environment variables. Semaphore polls the repo and
runs the matching .semaphore/semaphore.yml pipeline.

KEY ADVANTAGES:
 - Free 100 builds/mo for private repos.
 - **Unlimited for OSS** (matches GitHub Actions for public repos).
 - Self-hosted agents free + unlimited.
 - Monorepo-first design (selective pipeline execution by changed paths).

LIMITATIONS:
 - 100 builds/mo cap for private (vs. 2000 min on GitHub Actions Free private).
 - Build duration cap on shared runners (typically 1hr).
 - Less mature SDK ecosystem than GitHub/GitLab.

Docs:
 - https://docs.semaphoreci.com/
 - https://semaphore.io/blog/enterprise-ci-cd-free
"""
from __future__ import annotations

import os
import json
import urllib.request
import urllib.error


def _server() -> str:
    return os.environ.get("SEMAPHORE_SERVER", "https://YOUR-ORG.semaphoreci.com")


def _token() -> str:
    t = os.environ.get("SEMAPHORE_TOKEN")
    if not t:
        raise RuntimeError("SEMAPHORE_TOKEN not set")
    return t


def _project_id() -> str:
    p = os.environ.get("SEMAPHORE_PROJECT_ID")
    if not p:
        raise RuntimeError("SEMAPHORE_PROJECT_ID not set")
    return p


def submit_job(ticker: str, strategy: str, job_spec: dict, dry_run: bool = False) -> dict:
    project = _project_id() if not dry_run else "DRY"
    url = f"{_server()}/api/v2/projects/{project}/workflows"
    branch = os.environ.get("SEMAPHORE_BRANCH", "main")
    payload = {
        "reference": f"refs/heads/{branch}",
        "pipeline_file": ".semaphore/semaphore.yml",
        "parameters": {
            "TICKER": ticker,
            "STRATEGY": strategy,
            "JOB_SPEC_JSON": json.dumps(job_spec),
        },
    }
    if dry_run:
        return {"job_id": "DRY-RUN", "status": "would_submit",
                "url": url, "payload": payload}
    data = json.dumps(payload).encode()
    headers = {"Authorization": f"Token {_token()}", "Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode())
            return {"job_id": str(body.get("wf_id") or body.get("id")),
                    "status": "submitted"}
    except urllib.error.HTTPError as e:
        return {"job_id": None,
                "status": "auth_failure" if e.code in (401, 403) else "submit_error",
                "code": e.code, "body": e.read().decode(errors="ignore")}


def check_status(job_id: str) -> dict:
    url = f"{_server()}/api/v2/workflows/{job_id}"
    headers = {"Authorization": f"Token {_token()}"}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode())
            st = (body.get("state") or body.get("status") or "").lower()
            mapping = {"pending": "pending", "running": "running",
                       "passed": "completed", "failed": "failed",
                       "stopped": "failed", "canceled": "failed"}
            return {"status": mapping.get(st, "unknown"), "semaphore_status": st}
    except urllib.error.HTTPError as e:
        return {"status": "poll_error", "code": e.code}


if __name__ == "__main__":
    print(json.dumps(submit_job("AAPL", "D1_REV", {"thresh": 0.5}, dry_run=True), indent=2))
