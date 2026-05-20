"""dispatcher_adapter_codeberg_woodpecker.py — Codeberg Woodpecker CI adapter stub.

Free tier (verified 2026-05-17): Codeberg is a Berlin-based non-profit Gitea
instance. Their Woodpecker CI is donation-funded, no published per-user minute
cap, but is gated access — request access to ci.codeberg.org via
codeberg-e.V. issue tracker for OSS projects.

Signup: https://codeberg.org/user/sign_up; CI access at
https://codeberg.org/Codeberg-CI; Woodpecker docs:
https://woodpecker-ci.org/docs/usage/intro.

Auth model: Codeberg personal access token with `repo` scope. Woodpecker uses
the same token via OAuth federated identity. Adapter triggers a manual pipeline
run via Woodpecker REST API.

Submit model: POST /api/repos/:owner/:repo/pipelines with branch + variables.
Functionally similar to gitlab_ci adapter.

KEY ADVANTAGE: For OSS projects, this is fully FREE forever with no cap, no CC,
and runs in the EU (GDPR-friendly).

LIMITATIONS:
 - Manual signup + OSS-approval gate (apply via Codeberg issue tracker).
 - Shared community runners; expect contention during EU business hours.
 - Self-hosted Woodpecker agent on a €5/mo Hetzner VPS gets you unlimited
   private capacity if needed.
"""
from __future__ import annotations

import os
import json
import time
import urllib.request
import urllib.error


def _server() -> str:
    return os.environ.get("WOODPECKER_SERVER", "https://ci.codeberg.org")


def _token() -> str:
    t = os.environ.get("WOODPECKER_TOKEN")
    if not t:
        raise RuntimeError("WOODPECKER_TOKEN not set")
    return t


def _repo() -> tuple:
    owner = os.environ.get("WOODPECKER_OWNER")
    repo = os.environ.get("WOODPECKER_REPO")
    if not owner or not repo:
        raise RuntimeError("WOODPECKER_OWNER and WOODPECKER_REPO must be set")
    return owner, repo


def submit_job(ticker: str, strategy: str, job_spec: dict, dry_run: bool = False) -> dict:
    owner, repo = _repo()
    branch = os.environ.get("WOODPECKER_BRANCH", "main")
    url = f"{_server()}/api/repos/{owner}/{repo}/pipelines"
    payload = {
        "branch": branch,
        "variables": {
            "TICKER": ticker,
            "STRATEGY": strategy,
            "JOB_SPEC_JSON": json.dumps(job_spec),
        },
    }
    if dry_run:
        return {"job_id": "DRY-RUN", "status": "would_submit",
                "url": url, "payload": payload}
    data = json.dumps(payload).encode()
    headers = {"Authorization": f"Bearer {_token()}",
               "Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode())
            return {"job_id": str(body.get("number") or body.get("id")),
                    "status": "submitted"}
    except urllib.error.HTTPError as e:
        return {"job_id": None,
                "status": "auth_failure" if e.code in (401, 403) else "submit_error",
                "code": e.code, "body": e.read().decode(errors="ignore")}


def check_status(job_id: str) -> dict:
    owner, repo = _repo()
    url = f"{_server()}/api/repos/{owner}/{repo}/pipelines/{job_id}"
    headers = {"Authorization": f"Bearer {_token()}"}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode())
            st = body.get("status")
            mapping = {"pending": "pending", "running": "running",
                       "success": "completed", "failure": "failed",
                       "killed": "failed", "error": "failed",
                       "blocked": "pending", "declined": "failed",
                       "started": "running"}
            return {"status": mapping.get(st, "unknown"),
                    "woodpecker_status": st}
    except urllib.error.HTTPError as e:
        return {"status": "poll_error", "code": e.code}


if __name__ == "__main__":
    print(json.dumps(submit_job("AAPL", "D1_REV", {"thresh": 0.5}, dry_run=True), indent=2))
