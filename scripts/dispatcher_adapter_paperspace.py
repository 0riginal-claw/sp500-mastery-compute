"""dispatcher_adapter_paperspace.py — Paperspace Gradient notebook/jobs adapter stub.

Free tier (verified 2026-05-17): Paperspace Gradient (now part of DigitalOcean)
offers free-GPU notebooks (M4000) with persistent storage. No credit card
required to sign up; CC needed only for paid tiers.

Signup: https://www.paperspace.com/account/signup.

Auth model: Paperspace API key from https://console.paperspace.com/account/api.
SDK = `pip install gradient`; uses GRADIENT_API_KEY env var.

Submit model: `gradient jobs create --container ... --command "python run_sweep.py"`
or via the REST API at api.paperspace.io/jobs/createJob. Result artifacts are
written to a job-specific workspace mounted at /artifacts.

KEY ADVANTAGE: Persistent /storage volume survives across jobs (10-15 GB free)
— don't have to re-download data each time.

LIMITATIONS:
 - "Free" GPU tier has long queue waits during peak hours.
 - 6 hour max job duration on free tier.
 - 1 concurrent job per account on free.
"""
from __future__ import annotations

import os
import json
import time
import subprocess


GRADIENT_BIN = os.environ.get("GRADIENT_BIN", "gradient")


def submit_job(ticker: str, strategy: str, job_spec: dict, dry_run: bool = False) -> dict:
    api_key = os.environ.get("GRADIENT_API_KEY")
    project_id = os.environ.get("GRADIENT_PROJECT_ID")
    container = os.environ.get("GRADIENT_CONTAINER", "paperspace/gradient-base:py311-fastai")
    machine_type = os.environ.get("GRADIENT_MACHINE_TYPE", "Free-CPU")

    job_args = " ".join(f"--{k} {v}" for k, v in job_spec.items())
    cmd = f"python run_sweep.py --ticker {ticker} --strategy {strategy} {job_args}"

    if dry_run:
        return {"job_id": "DRY-RUN", "status": "would_submit",
                "machine": machine_type, "container": container, "cmd": cmd}

    if not api_key or not project_id:
        return {"job_id": None, "status": "auth_failure",
                "error": "GRADIENT_API_KEY + GRADIENT_PROJECT_ID required"}

    env = os.environ.copy()
    env["PAPERSPACE_API_KEY"] = api_key
    try:
        out = subprocess.run(
            [GRADIENT_BIN, "jobs", "create",
             "--projectId", project_id,
             "--machineType", machine_type,
             "--container", container,
             "--command", cmd,
             "--name", f"sp500-{ticker}-{strategy}"],
            capture_output=True, text=True, timeout=120, env=env,
        )
        if out.returncode != 0:
            return {"job_id": None, "status": "submit_error",
                    "stderr": out.stderr[:500]}
        # Extract job ID from CLI output.
        job_id = None
        for line in out.stdout.splitlines():
            if line.lower().startswith("new jobs created with id:"):
                job_id = line.split(":")[-1].strip()
        return {"job_id": job_id or out.stdout.strip(),
                "status": "submitted"}
    except FileNotFoundError:
        return {"job_id": None, "status": "submit_error",
                "error": f"gradient CLI not found at {GRADIENT_BIN}"}


def check_status(job_id: str) -> dict:
    try:
        out = subprocess.run([GRADIENT_BIN, "jobs", "show", "--jobId", job_id],
                             capture_output=True, text=True, timeout=30)
        text = out.stdout.lower()
        if "stopped" in text or "succeeded" in text or "finished" in text:
            return {"status": "completed", "raw": out.stdout.strip()}
        if "failed" in text or "error" in text:
            return {"status": "failed", "raw": out.stdout.strip()}
        if "running" in text or "pending" in text or "queued" in text:
            return {"status": "running", "raw": out.stdout.strip()}
        return {"status": "unknown", "raw": out.stdout.strip()}
    except FileNotFoundError:
        return {"status": "poll_error", "error": "gradient CLI missing"}


if __name__ == "__main__":
    print(json.dumps(submit_job("AAPL", "D1_REV", {"thresh": 0.5}, dry_run=True), indent=2))
