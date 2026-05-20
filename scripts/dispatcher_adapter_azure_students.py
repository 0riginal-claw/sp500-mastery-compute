"""dispatcher_adapter_azure_students.py — Azure for Students adapter stub.

Free tier (verified 2026-05-17): $100 in Azure credits per year for verified
students (age 18+, accredited 2/4-year institution). Renewable annually.
NO credit card required. Plus 55+ free always-on services (App Service free
tier, Functions free tier 1M req/mo).

Signup: https://azure.microsoft.com/en-us/free/students — sign in with a
PERSONAL Microsoft account (NOT institutional), provide school email for
verification, then complete the student verification flow.

Auth: Standard Azure Service Principal (client_id / client_secret / tenant_id)
issued via Azure AD. Once you have a subscription provisioned, this adapter
behaves like a generic Azure dispatcher.

Submit model: Azure Functions invoke (HTTP trigger) for short jobs, OR Azure
Container Instances `az container create` for batch jobs, OR Azure Batch for
parallel sweeps. This stub uses the Azure Functions HTTP-invoke path; for ACI
or Batch, swap the URL + payload.

KEY ADVANTAGES:
 - $100/yr no-CC + 55 always-free services.
 - Full Azure ML / Functions / Container Instances access.
 - Stackable with GitHub Student Developer Pack (which itself adds credits).
 - 12 months of validity per renewal.

LIMITATIONS:
 - Once $100 is exhausted, subscription disables until renewal (no overage).
 - Cannot purchase support plans, DevOps, Marketplace, or 3rd-party SKUs.
 - Sovereign clouds (US Gov, Azure Germany) excluded.
 - Student verification can require uploading transcript / ID.

Docs:
 - https://azure.microsoft.com/en-us/free/students
 - https://learn.microsoft.com/en-us/azure/education-hub/about-azure-for-students
"""
from __future__ import annotations

import os
import json
import urllib.request
import urllib.error
import base64
import urllib.parse


_TOKEN_CACHE: dict = {}


def _get_aad_token() -> str:
    """Acquire an Azure AD access token via client_credentials grant."""
    if _TOKEN_CACHE.get("access_token"):
        return _TOKEN_CACHE["access_token"]
    tenant = os.environ["AZURE_STUDENTS_TENANT_ID"]
    client_id = os.environ["AZURE_STUDENTS_CLIENT_ID"]
    client_secret = os.environ["AZURE_STUDENTS_CLIENT_SECRET"]
    url = f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"
    body = urllib.parse.urlencode({
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": "https://management.azure.com/.default",
    }).encode()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())
    _TOKEN_CACHE["access_token"] = data["access_token"]
    return data["access_token"]


def submit_job(ticker: str, strategy: str, job_spec: dict, dry_run: bool = False) -> dict:
    function_url = os.environ.get(
        "AZURE_STUDENTS_FUNCTION_URL",
        "https://<your-function-app>.azurewebsites.net/api/run-sweep",
    )
    function_key = os.environ.get("AZURE_STUDENTS_FUNCTION_KEY", "")
    payload = {"ticker": ticker, "strategy": strategy, "job_spec": job_spec}
    if dry_run:
        return {"job_id": "DRY-RUN", "status": "would_submit",
                "url": function_url, "payload": payload}
    url = f"{function_url}?code={urllib.parse.quote(function_key)}"
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode())
            return {"job_id": body.get("invocation_id", "azs-sync"),
                    "status": "submitted"}
    except urllib.error.HTTPError as e:
        return {"job_id": None,
                "status": "auth_failure" if e.code in (401, 403) else "submit_error",
                "code": e.code, "body": e.read().decode(errors="ignore")}


def check_status(job_id: str) -> dict:
    # Azure Functions HTTP invocations are synchronous by default; for async
    # use Durable Functions and poll the status endpoint.
    return {"status": "completed", "note": "azure functions http invoke is sync"}


if __name__ == "__main__":
    print(json.dumps(submit_job("AAPL", "D1_REV", {"thresh": 0.5}, dry_run=True), indent=2))
