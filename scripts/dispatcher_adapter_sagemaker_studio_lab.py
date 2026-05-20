"""dispatcher_adapter_sagemaker_studio_lab.py — Amazon SageMaker Studio Lab stub.

Free tier (verified 2026-05-17): No AWS account required, no credit card. Free
T4 GPU (4 hrs/session) + free CPU (12 hrs/session); 15 GB persistent storage.

Signup: https://studiolab.sagemaker.aws — email-based, no AWS root account.

Auth model: SSO via email; no programmatic API token. Interaction is through
a JupyterLab UI with SSH access disabled. Like Colab, this is interactive-only.

Submit model: Same Drive-folder-IPC pattern as the Colab adapter. A persistent
notebook polls a synced folder for jobs, writes results back. SageMaker Studio
Lab does NOT have native Drive sync, so we use the `git` workflow instead:
adapter commits job to a `sp500-jobs` GitHub repo; the notebook git-pulls every
60 s, runs the job, git-commits the result back.

KEY ADVANTAGE: 4 hr T4 sessions + 15 GB persistent disk; FREE without any AWS
billing relationship.

LIMITATIONS:
 - Sessions auto-stop after 4 hr (GPU) / 12 hr (CPU); manual restart required.
 - No public REST API; depends on git-loop convention.
 - max_concurrent=1 per Studio Lab account.
"""
from __future__ import annotations

import os
import json
import time
import subprocess
import pathlib


def _jobs_repo() -> pathlib.Path:
    p = os.environ.get("SMSL_JOBS_REPO_DIR")
    if not p:
        p = "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/sp500-jobs-repo"
    out = pathlib.Path(p)
    out.mkdir(parents=True, exist_ok=True)
    return out


def submit_job(ticker: str, strategy: str, job_spec: dict, dry_run: bool = False) -> dict:
    job_id = f"smsl-{ticker}-{strategy}-{int(time.time()*1000)}"
    repo = _jobs_repo()
    job_file = repo / "queue" / f"{job_id}.json"
    job_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {"job_id": job_id, "ticker": ticker, "strategy": strategy,
               "job_spec": job_spec, "submitted_at": time.time()}
    if dry_run:
        return {"job_id": job_id, "status": "would_submit",
                "queue_file": str(job_file)}
    job_file.write_text(json.dumps(payload))
    # Commit + push (best effort — if no git remote configured, just rely on
    # Drive sync to ferry the file to SMSL via gdrive client).
    try:
        subprocess.run(["git", "-C", str(repo), "add", str(job_file)],
                       capture_output=True, timeout=10)
        subprocess.run(["git", "-C", str(repo), "commit", "-m", f"enqueue {job_id}"],
                       capture_output=True, timeout=10)
        subprocess.run(["git", "-C", str(repo), "push"], capture_output=True, timeout=30)
    except Exception:  # noqa: BLE001
        pass
    return {"job_id": job_id, "status": "submitted",
            "queue_file": str(job_file),
            "note": "SMSL notebook must be running and git-pulling the queue"}


def check_status(job_id: str) -> dict:
    repo = _jobs_repo()
    result_file = repo / "results" / f"{job_id}.json"
    if result_file.exists():
        try:
            return {"status": "completed",
                    "data": json.loads(result_file.read_text())}
        except json.JSONDecodeError:
            return {"status": "poll_error", "error": "malformed result"}
    queue_file = repo / "queue" / f"{job_id}.json"
    if queue_file.exists():
        return {"status": "running"}
    return {"status": "unknown"}


if __name__ == "__main__":
    print(json.dumps(submit_job("AAPL", "D1_REV", {"thresh": 0.5}, dry_run=True), indent=2))
