"""dispatcher_adapter_codespaces.py — GitHub Codespaces adapter stub.

Free tier (verified 2026-05-17): 120 core-hours/month + 15 GB storage. A 2-core
machine = 60 hours of active wall-clock per month. No credit card required.

Signup: any GitHub user account at https://github.com/codespaces.

Auth model: GH personal access token with `codespace` scope (separate from the
standard `workflow` scope) at https://github.com/settings/tokens.

Submit model: Codespaces are interactive long-running VMs by design. Programmatic
job dispatch uses the GitHub REST API to:
  1. POST /user/codespaces — create a new Codespace from a repo + branch.
  2. POST /user/codespaces/<name>/start — start it.
  3. POST /user/codespaces/<name>/exec (preview API) — run a command (or use
     the `gh codespace ssh -- <cmd>` CLI wrapper). For broad compat, this stub
     uses `gh codespace ssh` via subprocess.
  4. Poll exec output / job-result file via Drive sync or Codespace blob fetch.

Stub status: a real automated workflow requires either the `gh` CLI to be
installed AND a long-lived Codespace pre-warmed, OR full REST orchestration
including key registration. Stub records intent; integration deferred.

KEY ADVANTAGE: 4-core / 8 GB / 32 GB SSD machine for free; full SSH; pip/npm work.

LIMITATIONS:
 - 60 wall-clock-hours/mo on the default 2-core. 30 hrs on 4-core. Quota is
   tight for continuous sweep work.
 - 30-min idle timeout (configurable). After idle, restart cost ~30 s.
 - 1 concurrent Codespace on free; 2 on Pro.
"""
from __future__ import annotations

import os
import json
import time
import subprocess


def _gh() -> str:
    return os.environ.get("GH_BIN", "gh")


def submit_job(ticker: str, strategy: str, job_spec: dict, dry_run: bool = False) -> dict:
    name = os.environ.get("CODESPACES_NAME")  # pre-warmed codespace name
    repo = os.environ.get("CODESPACES_REPO", "0riginal-claw/sp500-mastery-compute")

    job_args = " ".join(f"--{k} {v}" for k, v in job_spec.items())
    cmd = f"cd /workspaces/sp500-mastery-compute && python run_sweep.py --ticker {ticker} --strategy {strategy} {job_args}"

    if dry_run:
        return {"job_id": "DRY-RUN", "status": "would_submit",
                "codespace": name or "(would create)", "repo": repo, "cmd": cmd}

    if not name:
        # Create a new Codespace (60-90 s warm-up).
        try:
            out = subprocess.run([_gh(), "codespace", "create", "--repo", repo, "--machine", "basicLinux32gb"],
                                 capture_output=True, text=True, timeout=180)
            if out.returncode != 0:
                return {"job_id": None, "status": "submit_error",
                        "stderr": out.stderr[:500]}
            name = out.stdout.strip().splitlines()[-1]
        except FileNotFoundError:
            return {"job_id": None, "status": "submit_error",
                    "error": "gh CLI not on PATH"}

    try:
        out = subprocess.run([_gh(), "codespace", "ssh", "-c", name, "--", cmd],
                             capture_output=True, text=True, timeout=600)
        return {"job_id": name, "status": "completed" if out.returncode == 0 else "failed",
                "stdout": out.stdout[-500:], "stderr": out.stderr[-500:]}
    except FileNotFoundError:
        return {"job_id": None, "status": "submit_error",
                "error": "gh CLI missing"}


def check_status(job_id: str) -> dict:
    """Codespaces ssh is synchronous; status is captured at submit time. This
    poll is only useful for long-lived async jobs (which we don't model)."""
    return {"status": "completed", "note": "Codespaces ssh is synchronous"}


if __name__ == "__main__":
    print(json.dumps(submit_job("AAPL", "D1_REV", {"thresh": 0.5}, dry_run=True), indent=2))
