"""dispatcher_adapter_colab.py — Google Colab notebook execution adapter stub.

Free tier (verified 2026-05-17): Free T4 GPU; ~15-30 hours/week dynamic compute
unit allocation; 12-hour session ceiling; auto-disconnect on idle (90 min).
No credit card required.

Signup: any Google account works.

Auth model: Colab is NOT a public REST API; it's interactive notebooks. Two
viable dispatch paths:
  (A) Colab + Google Drive sync: notebook polls a Drive folder for jobs, writes
      results back. Requires the notebook to be already RUNNING (manual start).
  (B) `google-colab` python pkg + `colab-connect` undocumented APIs (fragile).

This stub implements PATH (A) — write a JSON job into Drive folder
sp500_dispatch/queue/<job_id>.json; poll sp500_dispatch/results/<job_id>.json
for results. The notebook side is `scripts/colab_worker_notebook.ipynb`
(must be uploaded + opened + Run-All by the user once).

KEY ADVANTAGE: free T4 GPU + 12 hr sessions + 80 GB RAM (high-mem mode).

LIMITATIONS:
 - Requires interactive session; auto-disconnect kills the worker.
 - max_concurrent = number of open Colab tabs you keep running.
 - No CLI API; depends on Drive shared folder for job IPC.
 - Cannot be fully headless — at least one Chrome tab must be running.
"""
from __future__ import annotations

import os
import json
import time
import pathlib


def _queue_dir() -> pathlib.Path:
    p = os.environ.get("COLAB_QUEUE_DIR")
    if not p:
        # Default to the Drive-synced folder used by the workspace.
        p = "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/sp500_dispatch/queue"
    out = pathlib.Path(p)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _results_dir() -> pathlib.Path:
    p = os.environ.get("COLAB_RESULTS_DIR")
    if not p:
        p = "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/sp500_dispatch/results"
    out = pathlib.Path(p)
    out.mkdir(parents=True, exist_ok=True)
    return out


def submit_job(ticker: str, strategy: str, job_spec: dict, dry_run: bool = False) -> dict:
    job_id = f"{ticker}-{strategy}-{int(time.time()*1000)}"
    job_file = _queue_dir() / f"{job_id}.json"
    payload = {"job_id": job_id, "ticker": ticker, "strategy": strategy,
               "job_spec": job_spec, "submitted_at": time.time()}
    if dry_run:
        return {"job_id": job_id, "status": "would_submit",
                "queue_file": str(job_file), "payload": payload}
    job_file.write_text(json.dumps(payload))
    return {"job_id": job_id, "status": "submitted",
            "queue_file": str(job_file),
            "note": "Colab worker notebook must be running for pickup"}


def check_status(job_id: str) -> dict:
    result_file = _results_dir() / f"{job_id}.json"
    if result_file.exists():
        try:
            data = json.loads(result_file.read_text())
            return {"status": "completed", "data": data}
        except json.JSONDecodeError:
            return {"status": "poll_error", "error": "result file is malformed"}
    # If queue file still present, still pending.
    queue_file = _queue_dir() / f"{job_id}.json"
    if queue_file.exists():
        return {"status": "running"}
    return {"status": "unknown", "note": "queue file gone but no result yet"}


if __name__ == "__main__":
    print(json.dumps(submit_job("AAPL", "D1_REV", {"thresh": 0.5}, dry_run=True), indent=2))
