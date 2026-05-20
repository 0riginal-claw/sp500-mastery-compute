"""dispatcher_adapter_lightning_ai.py — Lightning AI Studios adapter stub.

Free tier (verified 2026-05-17): 22 GPU-hours/month across T4/L4/A10G;
unlimited CPU-only Studios; persistent file storage; full VS Code env with SSH.
No credit card required (signup is GitHub OAuth + email verify).

Signup: https://lightning.ai/ → "Get Started Free" → GitHub OAuth.

Auth model: Lightning API key from https://lightning.ai/account/api-keys.
SDK = `pip install lightning-sdk`; uses LIGHTNING_USER_ID + LIGHTNING_API_KEY.

Submit model: Studios are long-running VMs; the `lightning_sdk` Job abstraction
creates an ephemeral job from a Studio template, runs a command, persists
artifacts, and exits. Closest analog to Modal.

KEY ADVANTAGE: L4/A10G GPUs are way more capable than T4 for ML workloads
(24 GB VRAM, ~3x faster than T4). 22 hrs/mo is enough for serious finetune
experiments.

LIMITATIONS:
 - 22 GPU-hr cap is monthly; CPU-only is unlimited but slower than Colab CPU.
 - SDK is in active development; API shape changes occasionally.
"""
from __future__ import annotations

import os
import json
import time


def submit_job(ticker: str, strategy: str, job_spec: dict, dry_run: bool = False) -> dict:
    user_id = os.environ.get("LIGHTNING_USER_ID")
    api_key = os.environ.get("LIGHTNING_API_KEY")
    template = os.environ.get("LIGHTNING_STUDIO_TEMPLATE", "sp500-backtest")
    machine = os.environ.get("LIGHTNING_MACHINE", "CPU")  # or "T4", "L4", "A10G"

    if dry_run:
        return {"job_id": "DRY-RUN", "status": "would_submit",
                "template": template, "machine": machine,
                "cmd": f"python run_sweep.py --ticker {ticker} --strategy {strategy}"}

    if not user_id or not api_key:
        return {"job_id": None, "status": "auth_failure",
                "error": "LIGHTNING_USER_ID + LIGHTNING_API_KEY required"}

    try:
        # Lazy import — only require the SDK at actual submit time.
        from lightning_sdk import Job, Machine  # type: ignore
    except ImportError:
        return {"job_id": None, "status": "submit_error",
                "error": "lightning-sdk not installed (pip install lightning-sdk)"}

    machine_enum = getattr(Machine, machine, Machine.CPU)
    job_args = " ".join(f"--{k} {v}" for k, v in job_spec.items())
    cmd = f"python run_sweep.py --ticker {ticker} --strategy {strategy} {job_args}"
    try:
        job = Job.run(
            name=f"sp500-{ticker}-{strategy}-{int(time.time())}",
            command=cmd,
            studio=template,
            machine=machine_enum,
        )
        return {"job_id": job.name, "status": "submitted", "machine": machine}
    except Exception as e:  # noqa: BLE001
        return {"job_id": None, "status": "submit_error", "error": str(e)}


def check_status(job_id: str) -> dict:
    try:
        from lightning_sdk import Job  # type: ignore
    except ImportError:
        return {"status": "poll_error", "error": "lightning-sdk not installed"}
    try:
        job = Job(name=job_id)
        st = str(job.status).lower()
        mapping = {"running": "running", "pending": "pending",
                   "succeeded": "completed", "completed": "completed",
                   "failed": "failed", "stopped": "failed"}
        for k, v in mapping.items():
            if k in st:
                return {"status": v, "raw": st}
        return {"status": "unknown", "raw": st}
    except Exception as e:  # noqa: BLE001
        return {"status": "poll_error", "error": str(e)}


if __name__ == "__main__":
    print(json.dumps(submit_job("AAPL", "D1_REV", {"thresh": 0.5}, dry_run=True), indent=2))
