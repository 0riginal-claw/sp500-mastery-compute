"""dispatcher_adapter_huggingface_spaces.py — Hugging Face Spaces inference adapter stub.

Free tier (verified 2026-05-17): unlimited Spaces, free CPU-basic (2 vCPU, 16 GB RAM)
that auto-sleeps after ~48 h idle and auto-wakes on first request. ZeroGPU is also
free for select accounts (queue-based H200 share). No credit card required.

Signup: https://huggingface.co/join → settings/tokens → fine-grained token with
"Read access to user info" + "Make calls to inference providers" + "Write to spaces".

Auth model: Bearer HF_TOKEN. Spaces are deployed as Gradio/FastAPI apps; we POST
to the Space's /run/predict endpoint with the job_spec as JSON.

Submit model: HF Space exposes /api/predict and /api/predict/<fn_name>. We pre-deploy
ONE Space called <user>/sp500-backtest that wraps run_sweep.py as a Gradio Interface.
This adapter posts the inputs and polls /queue/data for completion.

LIMITATIONS:
 - CPU-basic is shared with all free users; expect queueing under load.
 - 48h idle sleep → first request after idle has ~30 s cold start.
 - One Space per logical worker; HF allows unlimited Spaces, so max_concurrent
   can be raised by deploying replicas (sp500-backtest-1, -2, -3, ...).
"""
from __future__ import annotations

import os
import json
import time
import urllib.request
import urllib.error
from typing import Optional


def _token() -> str:
    t = os.environ.get("HF_TOKEN")
    if not t:
        raise RuntimeError("HF_TOKEN not set")
    return t


def _space_url() -> str:
    space = os.environ.get("HF_SPACE_ID")  # e.g. "youruser/sp500-backtest"
    if not space:
        raise RuntimeError("HF_SPACE_ID not set")
    return f"https://{space.replace('/', '-')}.hf.space"


def submit_job(ticker: str, strategy: str, job_spec: dict, dry_run: bool = False) -> dict:
    """POST to the Space's predict endpoint. Returns {job_id, status}."""
    url = _space_url() + "/run/predict"
    payload = {
        "data": [ticker, strategy, json.dumps(job_spec)],
        "fn_index": 0,
    }
    if dry_run:
        return {"job_id": "DRY-RUN", "status": "would_submit",
                "url": url, "payload": payload}

    data = json.dumps(payload).encode()
    headers = {"Authorization": f"Bearer {_token()}",
               "Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode())
            # Synchronous endpoint: response is already the result.
            return {"job_id": body.get("hash", str(int(time.time()))),
                    "status": "completed",
                    "data": body.get("data")}
    except urllib.error.HTTPError as e:
        return {"job_id": None,
                "status": "auth_failure" if e.code in (401, 403) else "submit_error",
                "code": e.code, "body": e.read().decode(errors="ignore")}


def check_status(job_id: str) -> dict:
    """HF Spaces /run/predict is synchronous; results are returned at submit time.

    If you switch to /queue/join + /queue/data SSE, this poll method becomes
    relevant. For now, jobs are completed-on-return.
    """
    return {"status": "completed", "note": "HF Space predict is synchronous"}


if __name__ == "__main__":
    print(json.dumps(submit_job("AAPL", "D1_REV", {"thresh": 0.5}, dry_run=True), indent=2))
