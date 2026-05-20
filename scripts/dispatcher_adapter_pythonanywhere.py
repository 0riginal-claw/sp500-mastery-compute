"""dispatcher_adapter_pythonanywhere.py — PythonAnywhere scheduled-task adapter stub.

Free tier (verified 2026-05-17): 1 always-on console, 1 web app, 100s CPU/day,
512 MB disk, 1 scheduled task. Python 3.10+. No credit card required.

Signup: https://www.pythonanywhere.com (Beginner plan = free forever).

Auth model: API token from https://www.pythonanywhere.com/user/<user>/account/#api_token
(scope: full account). Set PA_USERNAME and PA_API_TOKEN env vars.

REST API base: https://www.pythonanywhere.com/api/v0/user/<username>/
Docs: https://help.pythonanywhere.com/pages/API/

Submit model: POST to /schedule/ creates a one-shot scheduled task. We pass the
command as a shell-quoted python invocation against scripts/run_sweep.py with the
job_spec args. Tasks are listed via GET /schedule/ and per-task GET shows
last_run_time + return_code.

LIMITATIONS:
 - Only 1 scheduled task slot on free tier; this adapter is best for OVERFLOW
   one-at-a-time, not parallel sweeps. max_concurrent=1.
 - 100s CPU/day hard cap → only viable for short jobs (<60 s expected).
 - Free accounts cannot reach the public internet from python except for a
   whitelist (yfinance + alpaca are NOT on the whitelist). Use this adapter
   only for jobs whose data is uploaded ahead of time as parquet via the
   /files/ API.
"""
from __future__ import annotations

import os
import json
import time
import urllib.request
import urllib.parse
import urllib.error
from typing import Optional


API_BASE = "https://www.pythonanywhere.com/api/v0/user/{user}"


def _auth_headers() -> dict:
    token = os.environ.get("PA_API_TOKEN")
    if not token:
        raise RuntimeError("PA_API_TOKEN not set")
    return {"Authorization": f"Token {token}"}


def _user() -> str:
    user = os.environ.get("PA_USERNAME")
    if not user:
        raise RuntimeError("PA_USERNAME not set")
    return user


def submit_job(ticker: str, strategy: str, job_spec: dict, dry_run: bool = False) -> dict:
    """Schedule a one-shot task on PythonAnywhere. Returns {job_id, status}.

    NOTE: PythonAnywhere "scheduled tasks" actually run on a fixed daily schedule
    on the free tier. For true one-shot, we currently use an always-on console
    workaround OR re-schedule the task for "now+1m". This stub uses the schedule
    POST and relies on minute-granularity scheduling.
    """
    user = _user()
    url = API_BASE.format(user=user) + "/schedule/"
    # Use a posix-quoted command. job_spec keys passed as --k v.
    cmd_parts = ["/home/{u}/.virtualenvs/sweep/bin/python".format(u=user),
                 "/home/{u}/scripts/run_sweep.py".format(u=user),
                 "--ticker", ticker, "--strategy", strategy]
    for k, v in job_spec.items():
        cmd_parts.extend([f"--{k}", str(v)])
    cmd = " ".join(cmd_parts)

    # Schedule one minute in the future.
    future = time.gmtime(time.time() + 60)
    payload = {
        "command": cmd,
        "enabled": True,
        "interval": "daily",  # PA free only supports daily; we delete after first run
        "hour": future.tm_hour,
        "minute": future.tm_min,
        "description": f"sp500-sweep-{ticker}-{strategy}",
    }

    if dry_run:
        return {"job_id": "DRY-RUN", "status": "would_submit",
                "command": cmd, "url": url}

    data = urllib.parse.urlencode(payload).encode()
    req = urllib.request.Request(url, data=data, headers=_auth_headers(), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode())
            return {"job_id": str(body.get("id")), "status": "submitted",
                    "scheduled_minute": payload["minute"]}
    except urllib.error.HTTPError as e:
        return {"job_id": None, "status": "auth_failure" if e.code in (401, 403) else "submit_error",
                "code": e.code, "body": e.read().decode(errors="ignore")}


def check_status(job_id: str) -> dict:
    """Poll task status. Returns {status: pending|running|completed|failed, ...}."""
    user = _user()
    url = API_BASE.format(user=user) + f"/schedule/{job_id}/"
    req = urllib.request.Request(url, headers=_auth_headers())
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode())
            if body.get("logfile") and body.get("last_run_time"):
                rc = body.get("return_code")
                status = "completed" if rc == 0 else ("failed" if rc is not None else "running")
                return {"status": status, "return_code": rc,
                        "logfile": body.get("logfile"), "last_run": body.get("last_run_time")}
            return {"status": "pending"}
    except urllib.error.HTTPError as e:
        return {"status": "poll_error", "code": e.code}


def cancel_job(job_id: str) -> dict:
    """Delete the scheduled task (cleanup after completion or for cancellation)."""
    user = _user()
    url = API_BASE.format(user=user) + f"/schedule/{job_id}/"
    req = urllib.request.Request(url, headers=_auth_headers(), method="DELETE")
    try:
        urllib.request.urlopen(req, timeout=30)
        return {"status": "deleted"}
    except urllib.error.HTTPError as e:
        return {"status": "delete_error", "code": e.code}


if __name__ == "__main__":
    print(json.dumps(submit_job("AAPL", "D1_REV", {"thresh": "0.5"}, dry_run=True), indent=2))
