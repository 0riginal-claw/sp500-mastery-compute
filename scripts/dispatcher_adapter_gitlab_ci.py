"""dispatcher_adapter_gitlab_ci.py — GitLab CI pipeline dispatch adapter stub.

Free tier (verified 2026-05-17): 400 minutes/month free on GitLab SaaS shared
runners (small ubuntu, 1 vCPU). Self-hosted runners are UNLIMITED on any plan.
No credit card required.

Signup: https://gitlab.com/users/sign_up (GitHub OAuth or email).

Auth model: GitLab personal access token with `api` scope from
https://gitlab.com/-/user_settings/personal_access_tokens.

Submit model: Triggers a pipeline via POST /projects/:id/trigger/pipeline with
variables ticker/strategy/job_spec_json. The .gitlab-ci.yml in the target repo
runs run_sweep.py. Mirrors the github_actions adapter pattern but against the
GitLab REST API.

KEY ADVANTAGE: self-hosted runners are FREE AND UNLIMITED. If we register the
Mac (or Oracle A1) as a GitLab runner, this becomes a no-cap cloud cloud worker.

LIMITATIONS (shared runners only):
 - 400 min/mo cap on free SaaS shared runners.
 - 1 vCPU / 3.75 GB on `saas-linux-small-amd64`.
 - 1 hour max job duration.
"""
from __future__ import annotations

import os
import json
import time
import urllib.request
import urllib.parse
import urllib.error


GL_API_BASE = "https://gitlab.com/api/v4"


def _token() -> str:
    t = os.environ.get("GITLAB_TOKEN")
    if not t:
        raise RuntimeError("GITLAB_TOKEN not set")
    return t


def _project_id() -> str:
    p = os.environ.get("GITLAB_PROJECT_ID")
    if not p:
        raise RuntimeError("GITLAB_PROJECT_ID not set (numeric ID or URL-encoded path)")
    return p


def submit_job(ticker: str, strategy: str, job_spec: dict, dry_run: bool = False) -> dict:
    pid = _project_id()
    ref = os.environ.get("GITLAB_REF", "main")
    url = f"{GL_API_BASE}/projects/{pid}/pipeline"
    variables = [
        {"key": "TICKER", "value": ticker},
        {"key": "STRATEGY", "value": strategy},
        {"key": "JOB_SPEC_JSON", "value": json.dumps(job_spec)},
    ]
    payload = {"ref": ref, "variables": variables}
    if dry_run:
        return {"job_id": "DRY-RUN", "status": "would_submit",
                "url": url, "payload": payload}
    data = json.dumps(payload).encode()
    headers = {"PRIVATE-TOKEN": _token(),
               "Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode())
            return {"job_id": str(body.get("id")), "status": "submitted",
                    "web_url": body.get("web_url")}
    except urllib.error.HTTPError as e:
        return {"job_id": None,
                "status": "auth_failure" if e.code in (401, 403) else "submit_error",
                "code": e.code, "body": e.read().decode(errors="ignore")}


def check_status(job_id: str) -> dict:
    pid = _project_id()
    url = f"{GL_API_BASE}/projects/{pid}/pipelines/{job_id}"
    headers = {"PRIVATE-TOKEN": _token()}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode())
            st = body.get("status")
            mapping = {"created": "pending", "waiting_for_resource": "pending",
                       "preparing": "pending", "pending": "pending",
                       "running": "running", "success": "completed",
                       "failed": "failed", "canceled": "failed", "skipped": "failed",
                       "manual": "pending", "scheduled": "pending"}
            return {"status": mapping.get(st, "unknown"),
                    "gitlab_status": st,
                    "duration": body.get("duration"),
                    "web_url": body.get("web_url")}
    except urllib.error.HTTPError as e:
        return {"status": "poll_error", "code": e.code}


if __name__ == "__main__":
    print(json.dumps(submit_job("AAPL", "D1_REV", {"thresh": 0.5}, dry_run=True), indent=2))
