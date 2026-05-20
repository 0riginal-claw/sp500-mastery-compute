"""dispatcher_adapter_deno_deploy.py — Deno Deploy / Subhosting adapter stub.

Free tier (verified 2026-05-17): Deno Deploy free tier with usage caps on CPU
seconds, GiB-seconds, egress, and 60 deploys/hr (50 active/day on subhosting).
Python support is via the Deno Subhosting v2 API + the `subhosting` PyPI client.
NO credit card required for free tier.

Signup: https://dash.deno.com or https://app.deno.com — sign in with GitHub.
Generate access token at https://dash.deno.com/account#access-tokens.

Auth: `DEPLOY_ACCESS_TOKEN` bearer.
Subhosting model: Submit code (or a manifest pointing to a GitHub deploy)
as a "deployment" tied to a project under your org. Each deployment runs in
an isolated sandbox.

NOTE: Deno Deploy primarily targets TypeScript/JavaScript. Python is supported
via WASM Python builds (RustPython/Pyodide) or via subhosting wrapping a Python
runtime in a Deno-deployable form. For pure-Python quant workloads this is
likely NOT a good fit; this adapter is provided for completeness and for cases
where the workload can be expressed as a Deno handler invoking a child Python
WASM bundle.

Python SDK install: `pip install subhosting`.
Docs:
 - https://docs.deno.com/subhosting/api/
 - https://pypi.org/project/subhosting/

KEY ADVANTAGES:
 - Generous free tier, no CC.
 - Excellent cold start (Deno Isolates, ~10ms).
 - Native global edge network.

LIMITATIONS:
 - **Python is NOT a first-class runtime.** Use WASM Python or subprocess.
 - Subhosting v1 API shuts down 2026-07-20 (use v2 only).
 - 50 active deployments/day cap (free tier).
 - No GPU.
"""
from __future__ import annotations

import os
import json
import urllib.request
import urllib.error


def _token() -> str:
    t = os.environ.get("DENO_DEPLOY_ACCESS_TOKEN")
    if not t:
        raise RuntimeError("DENO_DEPLOY_ACCESS_TOKEN not set")
    return t


def _org_id() -> str:
    o = os.environ.get("DENO_DEPLOY_ORG_ID")
    if not o:
        raise RuntimeError("DENO_DEPLOY_ORG_ID not set")
    return o


def _project_id() -> str:
    p = os.environ.get("DENO_DEPLOY_PROJECT_ID")
    if not p:
        raise RuntimeError("DENO_DEPLOY_PROJECT_ID not set")
    return p


def submit_job(ticker: str, strategy: str, job_spec: dict, dry_run: bool = False) -> dict:
    project = _project_id() if not dry_run else "DRY"
    url = f"https://api.deno.com/v2/projects/{project}/deployments"
    payload = {
        "entryPointUrl": "main.ts",
        "envVars": {
            "TICKER": ticker,
            "STRATEGY": strategy,
            "JOB_SPEC_JSON": json.dumps(job_spec),
        },
        "assets": {
            "main.ts": {
                "kind": "file",
                "content": (
                    "const t=Deno.env.get('TICKER');"
                    "const s=Deno.env.get('STRATEGY');"
                    "// invoke Python WASM runtime here\n"
                    "console.log({t,s});"
                ),
                "encoding": "utf-8",
            }
        },
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
    url = f"https://api.deno.com/v2/deployments/{job_id}"
    headers = {"Authorization": f"Bearer {_token()}"}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode())
            st = (body.get("status") or "").lower()
            mapping = {"pending": "pending", "deploying": "running",
                       "success": "completed", "failed": "failed"}
            return {"status": mapping.get(st, "unknown"), "deno_status": st}
    except urllib.error.HTTPError as e:
        return {"status": "poll_error", "code": e.code}


if __name__ == "__main__":
    print(json.dumps(submit_job("AAPL", "D1_REV", {"thresh": 0.5}, dry_run=True), indent=2))
