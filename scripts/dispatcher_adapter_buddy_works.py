"""dispatcher_adapter_buddy_works.py — Buddy.works CI/CD adapter stub.

Free tier (verified 2026-05-17): 5 free projects, 1 concurrent run, 120 pipeline
executions/month free. NO credit card required. 100+ ready-to-use actions.

Signup: https://buddy.works — sign in with GitHub/GitLab/Bitbucket. No CC.

Auth: Buddy.works personal access token (Bearer) issued from
https://app.buddy.works/<workspace>/applications.

Submit model: POST /workspaces/:domain/projects/:project_name/pipelines/:pipeline_id/executions
— trigger a pipeline execution with optional ENV var overrides.

KEY ADVANTAGES:
 - No CC.
 - 120 executions/month = 4 batch runs/day sustained.
 - 100+ pre-built actions (Docker, Python, SSH, S3, etc.).
 - Visual pipeline editor.

LIMITATIONS:
 - 120 executions/mo is tight — exhausted by ~4 sweeps/day.
 - 1 concurrent run cap on free tier (no parallel batches).
 - 5-project cap.
 - Less programmable than GitLab CI / Semaphore (visual-first design).

Docs:
 - https://buddy.works/docs/api
 - https://buddy.works/pricing
"""
from __future__ import annotations

import os
import json
import urllib.request
import urllib.error


def _server() -> str:
    return os.environ.get("BUDDY_SERVER", "https://api.buddy.works")


def _token() -> str:
    t = os.environ.get("BUDDY_TOKEN")
    if not t:
        raise RuntimeError("BUDDY_TOKEN not set")
    return t


def _workspace() -> str:
    w = os.environ.get("BUDDY_WORKSPACE")
    if not w:
        raise RuntimeError("BUDDY_WORKSPACE not set")
    return w


def _project() -> str:
    p = os.environ.get("BUDDY_PROJECT")
    if not p:
        raise RuntimeError("BUDDY_PROJECT not set")
    return p


def _pipeline_id() -> str:
    pid = os.environ.get("BUDDY_PIPELINE_ID")
    if not pid:
        raise RuntimeError("BUDDY_PIPELINE_ID not set")
    return pid


def submit_job(ticker: str, strategy: str, job_spec: dict, dry_run: bool = False) -> dict:
    workspace = _workspace() if not dry_run else "DRY"
    project = _project() if not dry_run else "DRY"
    pipeline = _pipeline_id() if not dry_run else "DRY"
    url = (f"{_server()}/workspaces/{workspace}/projects/{project}"
           f"/pipelines/{pipeline}/executions")
    payload = {
        "to_revision": {"branch": os.environ.get("BUDDY_BRANCH", "main")},
        "variables": [
            {"key": "TICKER", "value": ticker},
            {"key": "STRATEGY", "value": strategy},
            {"key": "JOB_SPEC_JSON", "value": json.dumps(job_spec)},
        ],
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
            return {"job_id": str(body.get("id")), "status": "submitted"}
    except urllib.error.HTTPError as e:
        return {"job_id": None,
                "status": "auth_failure" if e.code in (401, 403) else "submit_error",
                "code": e.code, "body": e.read().decode(errors="ignore")}


def check_status(job_id: str) -> dict:
    workspace = _workspace()
    project = _project()
    pipeline = _pipeline_id()
    url = (f"{_server()}/workspaces/{workspace}/projects/{project}"
           f"/pipelines/{pipeline}/executions/{job_id}")
    headers = {"Authorization": f"Bearer {_token()}"}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode())
            st = (body.get("status") or "").upper()
            mapping = {"ENQUEUED": "pending", "INITIAL": "pending",
                       "INPROGRESS": "running", "IN_PROGRESS": "running",
                       "SUCCESSFUL": "completed", "FAILED": "failed",
                       "TERMINATED": "failed", "SKIPPED": "failed"}
            return {"status": mapping.get(st, "unknown"), "buddy_status": st}
    except urllib.error.HTTPError as e:
        return {"status": "poll_error", "code": e.code}


if __name__ == "__main__":
    print(json.dumps(submit_job("AAPL", "D1_REV", {"thresh": 0.5}, dry_run=True), indent=2))
