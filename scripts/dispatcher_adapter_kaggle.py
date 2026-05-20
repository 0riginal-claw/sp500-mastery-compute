"""dispatcher_adapter_kaggle.py — Kaggle Notebooks (Kernels) adapter stub.

Free tier (verified 2026-05-17): 30 hours/week of P100 GPU; 20 hours/week of
TPU v3-8; 4 CPU cores + 29 GB RAM on CPU-only kernels. No credit card required.

Signup: https://www.kaggle.com/account/login (Google OAuth → phone-verify for
GPU access). API token at https://www.kaggle.com/<user>/account → "Create API
Token" → downloads kaggle.json.

Auth model: Place kaggle.json at ~/.kaggle/kaggle.json (mode 600). The
official `kaggle` PyPI package wraps the REST API.

Submit model: We pre-create a private kernel `sp500-sweep` that reads
input dataset `sp500-jobs` and writes results to dataset `sp500-results`.
Adapter calls `kaggle kernels push` to update the kernel with new job
parameters and `kaggle kernels status` to poll.

KEY ADVANTAGE: P100 GPU + 29 GB RAM + 4 cores is the strongest free CPU/GPU
combo for ML — way beyond what GitHub Actions provides.

LIMITATIONS:
 - 30 hrs/wk hard cap on GPU; 9 hours max session duration.
 - One running kernel per user; serial execution (max_concurrent=1).
 - Cold start ~30 s for the kernel container.
 - Push-to-run cycle is slow (~1-2 min latency).
"""
from __future__ import annotations

import os
import json
import time
import subprocess
import tempfile
import pathlib


KAGGLE_BIN = os.environ.get("KAGGLE_BIN", "kaggle")


def _user() -> str:
    u = os.environ.get("KAGGLE_USERNAME")
    if not u:
        raise RuntimeError("KAGGLE_USERNAME not set (or place kaggle.json at ~/.kaggle/)")
    return u


def _kernel_slug() -> str:
    return os.environ.get("KAGGLE_KERNEL_SLUG", "sp500-sweep")


def submit_job(ticker: str, strategy: str, job_spec: dict, dry_run: bool = False) -> dict:
    """Push a new run of the existing kernel with the job_spec embedded as
    KERNEL_METADATA arguments. Returns {job_id, status}.
    """
    user = _user()
    slug = _kernel_slug()
    job_id = f"{slug}-{int(time.time())}"

    if dry_run:
        return {"job_id": job_id, "status": "would_submit",
                "kernel": f"{user}/{slug}",
                "args": {"ticker": ticker, "strategy": strategy, **job_spec}}

    # Write kernel-metadata.json + script into tempdir, push.
    with tempfile.TemporaryDirectory() as td:
        meta = {
            "id": f"{user}/{slug}",
            "title": slug,
            "code_file": "sweep.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": os.environ.get("KAGGLE_ENABLE_GPU", "false").lower() == "true",
            "enable_internet": True,
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
        }
        pathlib.Path(td, "kernel-metadata.json").write_text(json.dumps(meta))
        # The script reads ticker/strategy from env (set via kaggle UI doesn't support
        # env vars on kernels — we bake them into the script).
        script = (
            f"import os, json\n"
            f"TICKER = {ticker!r}\n"
            f"STRATEGY = {strategy!r}\n"
            f"JOB_SPEC = {json.dumps(job_spec)}\n"
            f"# user's run_sweep.py is committed to the kernel via dataset attach\n"
            f"from run_sweep import main\n"
            f"main(TICKER, STRATEGY, JOB_SPEC)\n"
        )
        pathlib.Path(td, "sweep.py").write_text(script)
        try:
            out = subprocess.run([KAGGLE_BIN, "kernels", "push", "-p", td],
                                 capture_output=True, text=True, timeout=120)
            if out.returncode != 0:
                return {"job_id": None, "status": "submit_error",
                        "stderr": out.stderr[:500]}
            return {"job_id": job_id, "status": "submitted",
                    "stdout": out.stdout.strip()}
        except FileNotFoundError:
            return {"job_id": None, "status": "submit_error",
                    "error": f"kaggle CLI not found at {KAGGLE_BIN}"}


def check_status(job_id: str) -> dict:
    user = _user()
    slug = _kernel_slug()
    try:
        out = subprocess.run([KAGGLE_BIN, "kernels", "status", f"{user}/{slug}"],
                             capture_output=True, text=True, timeout=30)
        text = out.stdout.lower()
        if "complete" in text:
            return {"status": "completed", "raw": out.stdout.strip()}
        if "error" in text or "fail" in text:
            return {"status": "failed", "raw": out.stdout.strip()}
        if "running" in text or "queued" in text:
            return {"status": "running", "raw": out.stdout.strip()}
        return {"status": "unknown", "raw": out.stdout.strip()}
    except FileNotFoundError:
        return {"status": "poll_error", "error": "kaggle CLI not on PATH"}


if __name__ == "__main__":
    print(json.dumps(submit_job("AAPL", "D1_REV", {"thresh": 0.5}, dry_run=True), indent=2))
