"""
multi_cloud_dispatcher.py — Multi-cloud XGBoost sweep dispatcher.

Reads jobs from sweeps/queue.txt, picks the cloud with the most free headroom,
submits via cloud-specific adapters, tracks quotas in sweeps/cloud_usage.json,
and polls for completed results in backtests/<ticker>/<strategy>/result.json.

Phase 1 adapters (implemented):
    github_actions  — triggers workflow_dispatch via GitHub API
    modal           — calls Modal SDK to submit a remote function call

Phase 2 adapters (stubs, enabled=false by default):
    oracle_a1       — SSH to Oracle Ampere A1 permanent free instance
    gcp_ssh         — SSH to GCP e2-micro always-free instance
    aws_ssh         — SSH to AWS t2.micro free-tier instance
    render_api      — Render.com managed service API
    railway_api     — Railway.app API
    fly_api         — Fly.io Machines API
    drone_ci        — Self-hosted Drone CI server (friend-donated agents)
    circleci_oss    — CircleCI OSS plan (approved repos: 400k credits/mo ≈ 40k min)

Phase 3 adapters (implemented):
    mac_local       — Run on this Mac with hard CPU/RAM/load/worker caps.
                      LAST RESORT: only selected when all remote clouds are
                      at quota/capacity OR the job is flagged small (<30 s).
                      Hard safety caps prevent re-creating the load-avg-937
                      disaster: cpu<60%, mem<70%, load_1m<8, active_workers<4.

Phase 4 adapters (implemented):
    bacalhau        — Public permissionless decentralised compute network
                      (https://bacalhau.org). Docker-native, free, no wallet
                      needed. Requires the `bacalhau` CLI on PATH and
                      enabled=true in cloud_usage.json (disabled by default).
                      Cap: 20 concurrent jobs (soft "be reasonable" limit).

Usage:
    # Single dispatch pass (process all pending jobs once, then exit)
    python scripts/multi_cloud_dispatcher.py

    # Daemon mode (runs forever, re-checks queue every POLL_INTERVAL seconds)
    python scripts/multi_cloud_dispatcher.py --daemon

    # Dry-run: log which cloud each job would go to, submit nothing
    python scripts/multi_cloud_dispatcher.py --dry-run

    # Simulate N mock jobs for testing quota logic (no real submissions)
    python scripts/multi_cloud_dispatcher.py --simulate 50

    # Simulate N jobs with remote clouds throttled (Mac gets higher share)
    python scripts/multi_cloud_dispatcher.py --simulate 50 --sim-throttle-remote

Environment variables needed (add to .env or shell profile):
    GITHUB_TOKEN        — personal access token, scope: workflow
    GITHUB_OWNER        — GitHub username or org (e.g. "youruser")
    GITHUB_REPO         — repo name (e.g. "sp500-ticker-mastery")
    GITHUB_WORKFLOW_ID  — workflow file name (e.g. "sweep.yml")
    GITHUB_IS_PUBLIC    — "true" if repo is public (unlimited minutes)
    MODAL_TOKEN_ID      — Modal token ID (from `modal token new`)
    MODAL_TOKEN_SECRET  — Modal token secret
    ORACLE_A1_HOST      — IP or hostname of Oracle Ampere A1 instance
    ORACLE_A1_SSH_KEY   — path to SSH private key for Oracle instance
    DRONE_SERVER        — base URL of self-hosted Drone server (e.g. https://drone.example.com)
    DRONE_TOKEN         — Drone user token (from Account > Token in Drone UI)
    CIRCLECI_TOKEN      — CircleCI Personal API Token (User Settings > Personal API Tokens)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import psutil  # type: ignore
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Project root (works both locally and in any cloud runner that checks out repo)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
QUEUE_FILE   = PROJECT_ROOT / "sweeps" / "queue.txt"
USAGE_FILE   = PROJECT_ROOT / "sweeps" / "cloud_usage.json"
RESULTS_DIR  = PROJECT_ROOT / "backtests"
LOG_DIR      = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

POLL_INTERVAL   = 30   # seconds between queue checks in daemon mode
RESULT_POLL_SEC = 10   # seconds between result-file polling in sync wait

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "dispatcher.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("dispatcher")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Job:
    script: str
    ticker: str
    strategy: str
    job_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    cloud: Optional[str] = None
    submitted_at: Optional[str] = None
    completed_at: Optional[str] = None
    result_path: Optional[Path] = None
    status: str = "pending"  # pending | submitted | completed | failed

    @classmethod
    def from_line(cls, line: str) -> "Job":
        parts = line.strip().split()
        if len(parts) != 3:
            raise ValueError(f"Malformed queue line (expected 3 tokens): {line!r}")
        return cls(script=parts[0], ticker=parts[1], strategy=parts[2])

    def result_file(self) -> Path:
        return RESULTS_DIR / self.ticker / self.strategy / "result.json"

    def __str__(self) -> str:
        return f"Job({self.job_id} {self.ticker}/{self.strategy})"


# ---------------------------------------------------------------------------
# Usage / quota tracker
# ---------------------------------------------------------------------------
class UsageTracker:
    """Loads, updates, and persists cloud_usage.json."""

    def __init__(self, path: Path = USAGE_FILE):
        self.path = path
        self.data: Dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        with open(self.path, "r") as f:
            self.data = json.load(f)
        log.debug("Loaded cloud_usage.json")

    def save(self) -> None:
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)
        log.debug("Saved cloud_usage.json")

    def cloud_cfg(self, cloud: str) -> Dict[str, Any]:
        return self.data[cloud]

    def enabled_clouds(self) -> List[str]:
        return [k for k, v in self.data.items()
                if not k.startswith("_") and v.get("enabled", False)]

    def headroom_pct(self, cloud: str) -> float:
        """
        Returns a 0–100 value representing how much quota is left as a fraction
        of the safety threshold. Higher = more headroom = preferred.

        Clouds that are over their safety margin return a negative value so they
        sort to the bottom and never receive new jobs.

        Billing-model logic:
            minutes     — github_actions private-repo cap
            credit_usd  — modal, railway
            hours       — aws_free, render
            concurrent_cap — oracle_a1, fly_io, gcp_ssh (no quota, only concurrency)
        """
        cfg = self.cloud_cfg(cloud)
        model = cfg.get("billing_model", "concurrent_cap")
        safety = cfg.get("safety_margin_pct", 80) / 100.0
        in_flight = cfg.get("in_flight_jobs", 0)
        max_concurrent = cfg.get("max_concurrent_jobs",
                          cfg.get("max_concurrent",
                          cfg.get("max_concurrent_containers", 1)))

        # If concurrency is already maxed, no headroom regardless of billing
        if in_flight >= max_concurrent:
            return -1.0

        if model == "minutes":
            # Public repos: unlimited minutes — treat as if always at 0% used
            if cfg.get("quota_unlimited_if_public") and _is_public_repo():
                return 100.0
            used  = cfg.get("used_min_this_month", 0)
            quota = cfg.get("quota_min", 2000)
            used_frac = used / quota if quota else 1.0
            if used_frac >= safety:
                return -1.0
            return (safety - used_frac) / safety * 100.0

        elif model == "credit_usd":
            used  = cfg.get("used_credit_this_month", 0)
            quota = cfg.get("quota_credit", 0)
            used_frac = used / quota if quota else 1.0
            if used_frac >= safety:
                return -1.0
            return (safety - used_frac) / safety * 100.0

        elif model == "hours":
            used  = cfg.get("used_hr_this_month", 0)
            quota = cfg.get("quota_hr", 0)
            used_frac = used / quota if quota else 1.0
            if used_frac >= safety:
                return -1.0
            return (safety - used_frac) / safety * 100.0

        else:  # concurrent_cap — headroom = remaining concurrency slots (0–100)
            slots_free = max_concurrent - in_flight
            return (slots_free / max_concurrent) * 100.0

    def pick_cloud(self, clouds: Optional[List[str]] = None,
                   prefer_remote: bool = True) -> Optional[str]:
        """
        Return the enabled cloud with the highest headroom.

        Mac-as-last-resort policy (enforced when prefer_remote=True):
          1. Try all remote clouds first (everything except mac_local).
          2. Only fall back to mac_local when:
               a) all remote clouds are at quota/capacity, OR
               b) the job is flagged small (caller passes prefer_remote=False).

        Ties broken round-robin (sorted by cloud name for determinism).
        Returns None if all clouds are over quota or at concurrency max.
        """
        candidates = clouds if clouds else self.enabled_clouds()

        remote_candidates = [c for c in candidates if c != "mac_local"]
        mac_candidates    = [c for c in candidates if c == "mac_local"]

        def _rank(pool: List[str]) -> List[Tuple[str, float]]:
            return sorted(
                [(c, self.headroom_pct(c)) for c in pool],
                key=lambda x: (-x[1], x[0]),
            )

        remote_ranked = _rank(remote_candidates)
        mac_ranked    = _rank(mac_candidates)

        log.debug(
            "Cloud headroom — remote: %s  mac: %s",
            {c: f"{h:.1f}%" for c, h in remote_ranked},
            {c: f"{h:.1f}%" for c, h in mac_ranked},
        )

        if prefer_remote:
            # Try best remote first
            if remote_ranked and remote_ranked[0][1] >= 0:
                return remote_ranked[0][0]
            # All remote exhausted — fall back to Mac if within caps
            if mac_ranked and mac_ranked[0][1] >= 0:
                log.info("All remote clouds at capacity — falling back to mac_local")
                return mac_ranked[0][0]
            return None
        else:
            # Small-job path: Mac is a first-class candidate
            all_ranked = _rank(candidates)
            if not all_ranked or all_ranked[0][1] < 0:
                return None
            return all_ranked[0][0]

    def register_submit(self, cloud: str, cost_estimate: float = 0.0,
                        minutes_estimate: float = 0.5,
                        hours_estimate: float = 0.008) -> None:
        """Increment in-flight counter and bill estimates immediately."""
        cfg = self.cloud_cfg(cloud)
        cfg["in_flight_jobs"] = cfg.get("in_flight_jobs", 0) + 1
        model = cfg.get("billing_model", "concurrent_cap")
        if model == "minutes":
            cfg["used_min_this_month"] = cfg.get("used_min_this_month", 0) + minutes_estimate
        elif model == "credit_usd":
            cfg["used_credit_this_month"] = cfg.get("used_credit_this_month", 0) + cost_estimate
        elif model == "hours":
            cfg["used_hr_this_month"] = cfg.get("used_hr_this_month", 0) + hours_estimate
        self.save()

    def register_complete(self, cloud: str) -> None:
        """Decrement in-flight counter."""
        cfg = self.cloud_cfg(cloud)
        cfg["in_flight_jobs"] = max(0, cfg.get("in_flight_jobs", 0) - 1)
        self.save()

    def over_safety_margin(self, cloud: str) -> bool:
        return self.headroom_pct(cloud) < 0


# ---------------------------------------------------------------------------
# Cloud helpers
# ---------------------------------------------------------------------------
def _is_public_repo() -> bool:
    return os.environ.get("GITHUB_IS_PUBLIC", "false").lower() == "true"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Adapter: GitHub Actions
# ---------------------------------------------------------------------------
def _submit_github_actions(job: Job, dry_run: bool = False) -> Dict[str, Any]:
    """
    Triggers a workflow_dispatch event on the configured GitHub repo.

    Required env vars:
        GITHUB_TOKEN        — PAT with `workflow` scope
        GITHUB_OWNER        — repo owner
        GITHUB_REPO         — repo name
        GITHUB_WORKFLOW_ID  — workflow file name or ID (default: sweep.yml)

    The workflow receives `inputs.ticker`, `inputs.strategy`, `inputs.script`
    so it can run the correct slice.

    Returns dict with job_id, cloud, submitted_at.
    """
    token    = os.environ.get("GITHUB_TOKEN", "")
    owner    = os.environ.get("GITHUB_OWNER", "YOUR_GITHUB_USER")
    repo     = os.environ.get("GITHUB_REPO", "sp500-ticker-mastery")
    workflow = os.environ.get("GITHUB_WORKFLOW_ID", "sweep.yml")
    branch   = os.environ.get("GITHUB_BRANCH", "main")

    if not token:
        log.warning("GITHUB_TOKEN not set — adapter will fail in production")

    payload = {
        "ref": branch,
        "inputs": {
            "ticker":   job.ticker,
            "strategy": job.strategy,
            "script":   job.script,
            "job_id":   job.job_id,
        },
    }

    url = f"https://api.github.com/repos/{owner}/{repo}/actions/workflows/{workflow}/dispatches"

    if dry_run:
        log.info("[DRY-RUN] Would POST to %s with %s", url, payload)
        return {"job_id": job.job_id, "cloud": "github_actions",
                "submitted_at": _now_iso(), "dry_run": True}

    try:
        import urllib.request
        data = json.dumps(payload).encode()
        req  = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            status = resp.status
        log.info("GitHub Actions dispatch: HTTP %s for %s", status, job)
    except Exception as exc:
        log.error("GitHub Actions dispatch failed for %s: %s", job, exc)
        raise

    return {
        "job_id":       job.job_id,
        "cloud":        "github_actions",
        "submitted_at": _now_iso(),
        "gh_workflow":  workflow,
    }


# ---------------------------------------------------------------------------
# Adapter: Modal
# ---------------------------------------------------------------------------
def _submit_modal(job: Job, dry_run: bool = False) -> Dict[str, Any]:
    """
    Submits one backtest job to Modal by invoking the modal CLI to run
    modal_worker.py::run_backtest remotely.

    Required env vars:
        MODAL_TOKEN_ID      — from `modal token new`
        MODAL_TOKEN_SECRET  — from `modal token new`

    In production the worker is deployed with `modal deploy scripts/modal_worker.py`.
    Dispatch is then just a `modal run scripts/modal_worker.py::run_backtest
    --ticker AAPL --strategy ORB --script scripts/backtest_xgb_v8.py`.

    For the MVP we shell out to the modal CLI so we don't add modal as a hard
    import dependency on the dispatcher machine (it only needs to be installed
    on dispatch nodes, not on cloud workers).
    """
    if dry_run:
        log.info("[DRY-RUN] Would call: modal run scripts/modal_worker.py "
                 "--ticker %s --strategy %s --script %s",
                 job.ticker, job.strategy, job.script)
        return {"job_id": job.job_id, "cloud": "modal",
                "submitted_at": _now_iso(), "dry_run": True}

    modal_worker = PROJECT_ROOT / "scripts" / "modal_worker.py"
    cmd = [
        sys.executable, "-m", "modal", "run",
        str(modal_worker),
        "--ticker",   job.ticker,
        "--strategy", job.strategy,
        "--script",   job.script,
        "--job-id",   job.job_id,
    ]
    env = os.environ.copy()
    env["MODAL_TOKEN_ID"]     = os.environ.get("MODAL_TOKEN_ID", "")
    env["MODAL_TOKEN_SECRET"] = os.environ.get("MODAL_TOKEN_SECRET", "")

    log.info("Submitting to Modal: %s", " ".join(cmd))
    try:
        proc = subprocess.Popen(
            cmd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        # We don't wait — dispatcher moves on; result polling detects completion
        log.info("Modal process pid=%s for %s", proc.pid, job)
    except FileNotFoundError:
        log.error("modal CLI not found — install with: pip install modal")
        raise

    return {
        "job_id":       job.job_id,
        "cloud":        "modal",
        "submitted_at": _now_iso(),
    }


# ---------------------------------------------------------------------------
# Adapter stubs (Phase 2 — not yet wired to live endpoints)
# ---------------------------------------------------------------------------
def _submit_oracle_a1(job: Job, dry_run: bool = False) -> Dict[str, Any]:
    """SSH to Oracle Ampere A1 and run job in the background."""
    host     = os.environ.get("ORACLE_A1_HOST", "")
    ssh_key  = os.environ.get("ORACLE_A1_SSH_KEY", "~/.ssh/oracle_a1_key")
    ssh_user = "ubuntu"
    remote_cmd = (
        f"cd ~/sp500-ticker-mastery && "
        f"nohup /Users/orginal/.venvs/sp500-mastery/bin/python "
        f"{job.script} --ticker {job.ticker} --strategy {job.strategy} "
        f"--job-id {job.job_id} > logs/{job.job_id}.log 2>&1 &"
    )
    if dry_run or not host:
        log.info("[DRY-RUN/STUB] oracle_a1: %s", remote_cmd)
        return {"job_id": job.job_id, "cloud": "oracle_a1",
                "submitted_at": _now_iso(), "dry_run": True}
    subprocess.Popen(
        ["ssh", "-i", ssh_key, f"{ssh_user}@{host}", remote_cmd],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return {"job_id": job.job_id, "cloud": "oracle_a1",
            "submitted_at": _now_iso()}


def _submit_gcp_ssh(job: Job, dry_run: bool = False) -> Dict[str, Any]:
    log.info("[STUB] gcp_ssh adapter not yet implemented for %s", job)
    return {"job_id": job.job_id, "cloud": "gcp_ssh",
            "submitted_at": _now_iso(), "dry_run": True}


def _submit_aws_ssh(job: Job, dry_run: bool = False) -> Dict[str, Any]:
    log.info("[STUB] aws_ssh adapter not yet implemented for %s", job)
    return {"job_id": job.job_id, "cloud": "aws_ssh",
            "submitted_at": _now_iso(), "dry_run": True}


def _submit_render_api(job: Job, dry_run: bool = False) -> Dict[str, Any]:
    log.info("[STUB] render_api adapter not yet implemented for %s", job)
    return {"job_id": job.job_id, "cloud": "render_api",
            "submitted_at": _now_iso(), "dry_run": True}


def _submit_railway_api(job: Job, dry_run: bool = False) -> Dict[str, Any]:
    log.info("[STUB] railway_api adapter not yet implemented for %s", job)
    return {"job_id": job.job_id, "cloud": "railway_api",
            "submitted_at": _now_iso(), "dry_run": True}


def _submit_fly_api(job: Job, dry_run: bool = False) -> Dict[str, Any]:
    log.info("[STUB] fly_api adapter not yet implemented for %s", job)
    return {"job_id": job.job_id, "cloud": "fly_api",
            "submitted_at": _now_iso(), "dry_run": True}


# ---------------------------------------------------------------------------
# Adapter: Drone CI — Phase 4 (self-hosted, friend-donated agents)
# drone_ci adapter is defined below (after circleci_oss), alongside other
# cfg-aware adapters that need the cloud_usage.json config block at call time.


# ---------------------------------------------------------------------------
# Adapter: Bacalhau — Phase 4 (public decentralised compute)
# ---------------------------------------------------------------------------
# Safety contract:
#   - No subprocess is ever spawned unless BOTH conditions hold:
#       1. bacalhau CLI is on PATH (checked once at import via _BACALHAU_CLI_PATH)
#       2. enabled=true in cloud_usage.json (enforced by pick_cloud / submit_job)
#   - dry_run=True always returns a fake job_id without touching the network.
# ---------------------------------------------------------------------------

def _bacalhau_cli_path() -> Optional[str]:
    """Return the path to the bacalhau binary, or None if not on PATH."""
    try:
        result = subprocess.run(
            ["which", "bacalhau"],
            capture_output=True, text=True, timeout=5,
        )
        path = result.stdout.strip()
        return path if path else None
    except Exception:
        return None


# Resolved once at module load; avoids repeated which() calls per job.
_BACALHAU_CLI_PATH: Optional[str] = _bacalhau_cli_path()

# Local output staging root for bacalhau get
_BACALHAU_TMP_ROOT = Path("/tmp/bacalhau")


def _submit_bacalhau(job: Job, dry_run: bool = False,
                     cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Submit one backtest job to the public Bacalhau network.

    Submission lifecycle:
        1. ``bacalhau docker run --id-only --wait=false`` submits the job and
           returns a UUID (the Bacalhau job id).
        2. The adapter returns immediately; completion is detected asynchronously
           by the standard result-file poll in poll_result().
        3. A background thread polls ``bacalhau describe <job_id>`` until the
           job reaches a terminal state, then calls
           ``bacalhau get <job_id> --output-dir /tmp/bacalhau/<job_id>`` and
           copies ``result.json`` to ``backtests/<ticker>/<strategy>/result.json``.

    Config keys read from cloud_usage.json ``bacalhau`` block:
        docker_image      — fully-qualified image ref
                            (ghcr.io/owner/sp500-backtest:latest)
        bacalhau_cli_path — override binary path (default: resolved via which)
        max_job_seconds   — hard timeout for the background poll thread
                            (default 600)

    Dry-run: returns a fake job_id ``bac-dry-<timestamp>`` without any
    subprocess call.
    """
    cfg = cfg or {}

    if dry_run:
        fake_id = f"bac-dry-{int(time.time())}"
        log.info(
            "[DRY-RUN] bacalhau: would submit ticker=%s strategy=%s image=%s  "
            "fake_job_id=%s",
            job.ticker, job.strategy,
            cfg.get("docker_image", "ghcr.io/PLACEHOLDER/sp500-backtest:latest"),
            fake_id,
        )
        return {
            "job_id":          job.job_id,
            "cloud":           "bacalhau",
            "bacalhau_job_id": fake_id,
            "submitted_at":    _now_iso(),
            "dry_run":         True,
        }

    cli = cfg.get("bacalhau_cli_path") or _BACALHAU_CLI_PATH
    if not cli:
        raise RuntimeError(
            "bacalhau CLI not found on PATH. "
            "Install via: curl -sL https://get.bacalhau.org/install.sh | bash"
        )

    image = cfg.get("docker_image", "ghcr.io/PLACEHOLDER/sp500-backtest:latest")
    max_seconds = int(cfg.get("max_job_seconds", 600))

    submit_cmd = [
        cli, "docker", "run",
        "--id-only",
        "--wait=false",
        "--env", f"TICKER={job.ticker}",
        "--env", f"STRATEGY={job.strategy}",
        "--output", "result:/output",
        image,
    ]

    log.info("Submitting to Bacalhau: %s", " ".join(submit_cmd))
    try:
        proc = subprocess.run(
            submit_cmd,
            capture_output=True, text=True, timeout=60,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("bacalhau submit timed out after 60 s")

    if proc.returncode != 0:
        raise RuntimeError(
            f"bacalhau submit failed (rc={proc.returncode}): {proc.stderr.strip()}"
        )

    bacalhau_job_id = proc.stdout.strip()
    if not bacalhau_job_id:
        raise RuntimeError(
            "bacalhau returned empty job id — check CLI version and network"
        )

    log.info("Bacalhau job submitted: bacalhau_job_id=%s for %s",
             bacalhau_job_id, job)

    # Spawn a background thread to poll for completion and retrieve results.
    import threading

    def _poll_and_retrieve() -> None:
        _bacalhau_wait_and_fetch(
            cli=cli,
            bacalhau_job_id=bacalhau_job_id,
            job=job,
            max_seconds=max_seconds,
        )

    t = threading.Thread(
        target=_poll_and_retrieve,
        daemon=True,
        name=f"bac-poll-{bacalhau_job_id[:8]}",
    )
    t.start()

    return {
        "job_id":          job.job_id,
        "cloud":           "bacalhau",
        "bacalhau_job_id": bacalhau_job_id,
        "submitted_at":    _now_iso(),
    }


def _bacalhau_wait_and_fetch(
    cli: str,
    bacalhau_job_id: str,
    job: Job,
    max_seconds: int = 600,
) -> None:
    """
    Background-thread worker: poll ``bacalhau describe`` until a terminal
    state, then retrieve result.json and copy it to the canonical path.

    Terminal states: Completed, Failed, Cancelled, Error.
    """
    deadline = time.monotonic() + max_seconds
    poll_interval = 15  # seconds between describe calls

    log.info("Bacalhau poll thread started for bacalhau_job_id=%s (max %ds)",
             bacalhau_job_id, max_seconds)

    while time.monotonic() < deadline:
        time.sleep(poll_interval)
        try:
            desc = subprocess.run(
                [cli, "describe", bacalhau_job_id, "--json"],
                capture_output=True, text=True, timeout=30,
            )
        except subprocess.TimeoutExpired:
            log.warning("bacalhau describe timed out for %s — retrying",
                        bacalhau_job_id)
            continue
        except Exception as exc:
            log.warning("bacalhau describe error for %s: %s — retrying",
                        bacalhau_job_id, exc)
            continue

        if desc.returncode != 0:
            log.warning("bacalhau describe rc=%d for %s: %s",
                        desc.returncode, bacalhau_job_id, desc.stderr.strip())
            continue

        try:
            info = json.loads(desc.stdout)
        except json.JSONDecodeError:
            log.warning("bacalhau describe returned non-JSON for %s",
                        bacalhau_job_id)
            continue

        # Bacalhau v1.x describe JSON uses Job.State.State
        state = (
            info.get("State", {}).get("State")
            or info.get("Job", {}).get("State", {}).get("State")
            or "Unknown"
        )
        log.debug("Bacalhau job %s state=%s", bacalhau_job_id, state)

        if state == "Completed":
            log.info("Bacalhau job %s completed — fetching results",
                     bacalhau_job_id)
            _bacalhau_fetch_result(cli, bacalhau_job_id, job)
            return

        if state in ("Failed", "Cancelled", "Error"):
            log.error(
                "Bacalhau job %s reached terminal state=%s — no result",
                bacalhau_job_id, state,
            )
            return

    log.warning("Bacalhau poll thread timed out for %s after %ds",
                bacalhau_job_id, max_seconds)


def _bacalhau_fetch_result(cli: str, bacalhau_job_id: str, job: Job) -> None:
    """
    Run ``bacalhau get`` to download job outputs, then copy result.json to
    the canonical path at backtests/<ticker>/<strategy>/result.json.
    """
    import shutil

    out_dir = _BACALHAU_TMP_ROOT / bacalhau_job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    get_cmd = [cli, "get", bacalhau_job_id, "--output-dir", str(out_dir)]
    log.info("Fetching Bacalhau results: %s", " ".join(get_cmd))

    try:
        proc = subprocess.run(
            get_cmd, capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        log.error("bacalhau get timed out for %s", bacalhau_job_id)
        return

    if proc.returncode != 0:
        log.error("bacalhau get failed (rc=%d) for %s: %s",
                  proc.returncode, bacalhau_job_id, proc.stderr.strip())
        return

    # Bacalhau places outputs under <out_dir>/outputs/<volume-name>/
    # The Dockerfile writes to /output/result.json → volume name "result"
    candidates = [
        out_dir / "outputs" / "result" / "result.json",
        out_dir / "result" / "result.json",
        out_dir / "result.json",
    ]
    src: Optional[Path] = next((p for p in candidates if p.exists()), None)

    if src is None:
        log.error(
            "result.json not found in Bacalhau output dir %s for job %s. "
            "Checked: %s",
            out_dir, bacalhau_job_id, [str(c) for c in candidates],
        )
        return

    dest = RESULTS_DIR / job.ticker / job.strategy / "result.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    log.info("Bacalhau result copied: %s -> %s", src, dest)


# ---------------------------------------------------------------------------
# Adapter: CircleCI OSS — Phase 2 (fully implemented)
# ---------------------------------------------------------------------------
def _submit_circleci_oss(
    job: Job,
    dry_run: bool = False,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Triggers a CircleCI pipeline via the v2 API for an approved OSS project.

    CircleCI OSS plan grants ~400,000 credits/month for approved open-source
    repositories (apply at https://circleci.com/open-source/).  At the
    docker/small@1x executor rate (~10 credits/min) this equates to roughly
    40,000 runner-minutes per month — tracked in cloud_usage.json as
    ``billing_model: "minutes"`` with ``quota_min: 40000``.

    The triggered pipeline must have a parameterised workflow that accepts
    ``ticker`` and ``strategy`` as pipeline parameters.  A ready-made
    ``.circleci/config.yml`` template is in ``scripts/circleci_oss_config.yml``.

    Required env var:
        CIRCLECI_TOKEN  — Personal API Token from CircleCI User Settings >
                          Personal API Tokens.

    Config keys read from cloud_usage.json ``circleci_oss`` block:
        circleci_token_env  — env-var name that holds the token (default CIRCLECI_TOKEN)
        circleci_org        — GitHub org / user that owns the repo
        circleci_repo       — Repository name (must be an approved OSS repo)

    Pipeline parameters forwarded to the workflow:
        ticker    — e.g. "AAPL"
        strategy  — e.g. "orb"

    Result convention:
        The CircleCI job writes ``backtests/<ticker>/<strategy>/result.json``
        and commits it back to the repo (or uploads as a GitHub Release asset).
        The dispatcher's standard ``poll_result()`` detects the file once the
        commit is pushed / asset is downloaded.

    Returns dict with job_id (CircleCI pipeline id), cloud, submitted_at.
    """
    cfg = cfg or {}
    token_env = cfg.get("circleci_token_env", "CIRCLECI_TOKEN")
    token     = os.environ.get(token_env, "")
    org       = cfg.get("circleci_org",  os.environ.get("CIRCLECI_ORG",  "PLACEHOLDER_ORG"))
    repo      = cfg.get("circleci_repo", os.environ.get("CIRCLECI_REPO", "PLACEHOLDER_REPO"))
    branch    = cfg.get("branch", "main")

    url = f"https://circleci.com/api/v2/project/gh/{org}/{repo}/pipeline"

    payload: Dict[str, Any] = {
        "branch": branch,
        "parameters": {
            "ticker":   job.ticker,
            "strategy": job.strategy,
            "job_id":   job.job_id,
        },
    }

    if dry_run or not token:
        if not token:
            log.warning(
                "circleci_oss: %s not set — would POST to %s with %s",
                token_env, url, payload,
            )
        else:
            log.info("[DRY-RUN] circleci_oss: Would POST to %s with %s", url, payload)
        ts = int(time.time())
        return {
            "job_id":       f"circle-dry-{ts}",
            "cloud":        "circleci_oss",
            "submitted_at": _now_iso(),
            "dry_run":      True,
        }

    import urllib.request
    import urllib.error

    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        url,
        data=data,
        headers={
            "Circle-Token":  token,
            "Content-Type":  "application/json",
            "Accept":        "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode(errors="replace")
        log.error(
            "circleci_oss dispatch failed for %s: HTTP %s — %s",
            job, exc.code, error_body,
        )
        raise
    except Exception as exc:
        log.error("circleci_oss dispatch failed for %s: %s", job, exc)
        raise

    # CircleCI v2 /pipeline response includes "id" (pipeline UUID) and "number"
    pipeline_id     = body.get("id", job.job_id)
    pipeline_number = body.get("number", "?")
    log.info(
        "circleci_oss pipeline triggered: id=%s number=%s for %s",
        pipeline_id, pipeline_number, job,
    )

    return {
        "job_id":            pipeline_id,
        "cloud":             "circleci_oss",
        "submitted_at":      _now_iso(),
        "pipeline_number":   pipeline_number,
        "circleci_url":      f"https://app.circleci.com/pipelines/gh/{org}/{repo}/{pipeline_number}",
    }


# ---------------------------------------------------------------------------
# Adapter: Drone CI (self-hosted) — Phase 2
# ---------------------------------------------------------------------------
def _submit_drone_ci(job: Job, dry_run: bool = False,
                     cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Triggers a new build on a self-hosted Drone CI server via the Drone HTTP API.

    The target repo must contain a ``.drone.yml`` that reads the env vars
    ``TICKER``, ``STRATEGY``, and ``OUT_PATH`` to run the correct backtest
    slice and upload ``result.json`` on completion.  See
    ``scripts/drone_ci_pipeline.yml`` for a reference template.

    Billing model: ``concurrent_cap`` — no monthly minute quota; the only
    bottleneck is the number of registered stateless agents (friend-donated
    or self-provisioned).

    Required env vars (names are read from cloud_usage.json so operators can
    override them without a code change):
        DRONE_SERVER   — base URL of Drone server, e.g. ``https://drone.example.com``
        DRONE_TOKEN    — Drone user token (Account > Token in Drone UI)

    The build is triggered with custom parameters passed in the request body as
    the ``params`` dict; the ``.drone.yml`` pipeline steps promote them to
    environment variables via ``environment:`` blocks.

    On success the Drone API returns a build object whose ``number`` field
    becomes our ``drone_build_number`` in the receipt.  Result polling is
    file-based (same ``backtests/<ticker>/<strategy>/result.json`` convention
    as other adapters) — the ``.drone.yml`` final step writes or uploads that
    file via ``rclone`` or ``gh release upload``.

    If ``DRONE_SERVER`` env var is missing the adapter automatically falls back
    to dry-run mode so no network calls are ever made with missing credentials.

    Args:
        job:     Job dataclass with ticker, strategy, script, job_id.
        dry_run: If True, log the would-be request and return without calling
                 the network.  Also activates automatically when DRONE_SERVER
                 env var is missing.
        cfg:     cloud_usage.json block for drone_ci (passed in by submit_job).
                 Used to resolve env-var name overrides and repo/branch config.

    Returns:
        Receipt dict with keys: job_id, cloud, submitted_at, drone_build_number,
        drone_repo, and optionally dry_run=True.
    """
    cfg = cfg or {}

    # Read env-var names from config (allows operator override without a code change)
    server_env = cfg.get("drone_server_url_env", "DRONE_SERVER")
    token_env  = cfg.get("drone_token_env", "DRONE_TOKEN")
    drone_repo = cfg.get("drone_repo", "owner/repo-name")
    branch     = cfg.get("drone_pipeline_branch", "main")

    server = os.environ.get(server_env, "").rstrip("/")
    token  = os.environ.get(token_env, "")

    # Build API path: POST /api/repos/:owner/:repo/builds?branch=<branch>
    owner, repo_name = (drone_repo.split("/", 1)
                        if "/" in drone_repo else ("owner", drone_repo))
    api_url  = f"{server}/api/repos/{owner}/{repo_name}/builds"
    full_url = f"{api_url}?branch={branch}"

    # Parameters forwarded to the .drone.yml pipeline as environment variables
    build_params: Dict[str, str] = {
        "TICKER":   job.ticker,
        "STRATEGY": job.strategy,
        "SCRIPT":   job.script,
        "JOB_ID":   job.job_id,
        "OUT_PATH": f"backtests/{job.ticker}/{job.strategy}/result.json",
    }

    if dry_run or not server:
        if not server:
            log.warning(
                "DRONE_SERVER env var (%s) not set — drone_ci adapter running "
                "in dry-run mode (no network call will be made).",
                server_env,
            )
        log.info(
            "[DRY-RUN] drone_ci: Would POST %s  params=%s",
            full_url, build_params,
        )
        return {
            "job_id":             job.job_id,
            "cloud":              "drone_ci",
            "submitted_at":       _now_iso(),
            "drone_build_number": None,
            "drone_repo":         drone_repo,
            "dry_run":            True,
        }

    if not token:
        log.warning(
            "DRONE_TOKEN env var (%s) not set — Drone API call will likely "
            "return 401 Unauthorized.",
            token_env,
        )

    try:
        import urllib.request
        import urllib.error as _urlerr

        data = json.dumps({"params": build_params}).encode()
        req  = urllib.request.Request(
            full_url,
            data=data,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type":  "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            body        = resp.read().decode()
            http_status = resp.status

        build_obj    = json.loads(body)
        build_number = build_obj.get("number")
        log.info(
            "Drone CI build triggered: HTTP %s  build_number=%s  repo=%s  %s",
            http_status, build_number, drone_repo, job,
        )
    except Exception as exc:
        log.error("Drone CI dispatch failed for %s: %s", job, exc)
        raise

    return {
        "job_id":             job.job_id,
        "cloud":              "drone_ci",
        "submitted_at":       _now_iso(),
        "drone_build_number": build_number,
        "drone_repo":         drone_repo,
    }


# ---------------------------------------------------------------------------
# Adapter: Mac local — Phase 3 (LAST RESORT)
# ---------------------------------------------------------------------------
# Hard safety caps — must ALL pass before any job is accepted:
#   cpu_percent   < MAC_CPU_CAP    (60 %)
#   mem_percent   < MAC_MEM_CAP    (70 %)
#   load_avg_1m   < MAC_LOAD_CAP   (8.0  — equal to physical core count)
#   active_workers < MAC_MAX_WORKERS (4)
#
# These caps prevent re-creating the load-avg-937 disaster caused by
# unbounded local parallelism. If ANY cap is exceeded the adapter raises
# MacCapExceeded so the caller skips to the next cloud.
# ---------------------------------------------------------------------------

MAC_CPU_CAP      = 60.0   # %
MAC_MEM_CAP      = 70.0   # %
MAC_LOAD_CAP     = 8.0    # load-avg-1m (= physical core count)
MAC_MAX_WORKERS  = 4      # max concurrent local backtest subprocesses

# Module-level registry of subprocesses launched by this process session.
# Keyed by job_id → Popen object.  Cleaned up on completion poll.
_mac_active_procs: Dict[str, "subprocess.Popen[bytes]"] = {}


class MacCapExceeded(RuntimeError):
    """Raised when the Mac does not meet the safety caps for a new job."""


def _mac_system_snapshot() -> Dict[str, float]:
    """
    Return a dict with live system metrics using psutil.
    Falls back to safe maximums if psutil is unavailable (prevents accidental
    local submission on a degraded environment).
    """
    if not _PSUTIL_AVAILABLE:
        log.warning("psutil not available — treating Mac as over capacity")
        return {
            "cpu_percent":    100.0,
            "mem_percent":    100.0,
            "load_avg_1min":  999.0,
            "active_workers": len(_mac_active_procs),
        }

    cpu = psutil.cpu_percent(interval=0.5)          # 0.5 s sample
    mem = psutil.virtual_memory().percent
    load_1m = psutil.getloadavg()[0]                # (1m, 5m, 15m)

    # Reap completed workers so the count stays accurate
    finished = [jid for jid, p in _mac_active_procs.items()
                if p.poll() is not None]
    for jid in finished:
        log.debug("Mac worker finished: job_id=%s", jid)
        del _mac_active_procs[jid]

    return {
        "cpu_percent":    cpu,
        "mem_percent":    mem,
        "load_avg_1min":  load_1m,
        "active_workers": len(_mac_active_procs),
    }


def mac_can_accept_job(cfg: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Check all four safety caps.  Returns (True, "") if safe, or
    (False, reason_string) if any cap is exceeded.

    Caps are read from cloud_usage.json so operators can tighten them
    without a code change; the module-level constants above are defaults.
    """
    snap = _mac_system_snapshot()

    cap_cpu     = cfg.get("max_cpu_pct",   MAC_CPU_CAP)
    cap_mem     = cfg.get("max_mem_pct",   MAC_MEM_CAP)
    cap_load    = cfg.get("max_load",      MAC_LOAD_CAP)
    cap_workers = cfg.get("max_workers",   MAC_MAX_WORKERS)

    checks = [
        (snap["cpu_percent"]   >= cap_cpu,
         f"cpu_percent={snap['cpu_percent']:.1f}% >= cap {cap_cpu}%"),
        (snap["mem_percent"]   >= cap_mem,
         f"mem_percent={snap['mem_percent']:.1f}% >= cap {cap_mem}%"),
        (snap["load_avg_1min"] >= cap_load,
         f"load_avg_1m={snap['load_avg_1min']:.2f} >= cap {cap_load}"),
        (snap["active_workers"] >= cap_workers,
         f"active_workers={snap['active_workers']} >= cap {cap_workers}"),
    ]

    for exceeded, reason in checks:
        if exceeded:
            return False, reason

    return True, ""


def _submit_mac_local(job: Job, dry_run: bool = False,
                      cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Run the backtest job as a local subprocess on this Mac.

    Safety gate: mac_can_accept_job() must pass all four hard caps before
    the process is spawned.  If it fails, MacCapExceeded is raised so the
    dispatcher skips Mac and tries the next cloud (or blocks the job).

    The subprocess is non-blocking (Popen, not run).  Its pid is tracked in
    _mac_active_procs so the worker-count cap stays accurate across calls
    within the same dispatcher session.
    """
    cfg = cfg or {}
    ok, reason = mac_can_accept_job(cfg)
    if not ok:
        log.warning("Mac safety cap exceeded (%s) — skipping mac_local for %s",
                    reason, job)
        raise MacCapExceeded(reason)

    python = cfg.get("python_bin",
                     "/Users/orginal/.venvs/sp500-mastery/bin/python")
    cmd = [
        python,
        str(PROJECT_ROOT / job.script),
        "--ticker",   job.ticker,
        "--strategy", job.strategy,
        "--job-id",   job.job_id,
    ]

    if dry_run:
        snap = _mac_system_snapshot()
        log.info(
            "[DRY-RUN] mac_local would run: %s  |  cpu=%.1f%%  mem=%.1f%%  "
            "load_1m=%.2f  workers=%d",
            " ".join(cmd), snap["cpu_percent"], snap["mem_percent"],
            snap["load_avg_1min"], snap["active_workers"],
        )
        return {"job_id": job.job_id, "cloud": "mac_local",
                "submitted_at": _now_iso(), "dry_run": True}

    log_path = LOG_DIR / f"mac_{job.job_id}.log"
    log.info("Spawning mac_local worker: %s  (log=%s)", " ".join(cmd), log_path)
    try:
        with open(log_path, "wb") as lf:
            proc = subprocess.Popen(
                cmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
            )
        _mac_active_procs[job.job_id] = proc
        log.info("mac_local worker pid=%d for %s", proc.pid, job)
    except Exception as exc:
        log.error("mac_local spawn failed for %s: %s", job, exc)
        raise

    return {
        "job_id":       job.job_id,
        "cloud":        "mac_local",
        "submitted_at": _now_iso(),
        "pid":          proc.pid,
        "log":          str(log_path),
    }


# ---------------------------------------------------------------------------
# Adapter router
# ---------------------------------------------------------------------------
ADAPTERS = {
    "github_actions": _submit_github_actions,
    "modal":          _submit_modal,
    "oracle_a1":      _submit_oracle_a1,
    "gcp_ssh":        _submit_gcp_ssh,
    "aws_ssh":        _submit_aws_ssh,
    "render_api":     _submit_render_api,
    "railway_api":    _submit_railway_api,
    "fly_api":        _submit_fly_api,
    "drone_ci":       _submit_drone_ci,
    "bacalhau":       _submit_bacalhau,
    "circleci_oss":   _submit_circleci_oss,
    "mac_local":      _submit_mac_local,
}

# Adapters that require their cloud config dict forwarded at call time
_CFG_AWARE_ADAPTERS = {"mac_local", "circleci_oss", "bacalhau", "drone_ci"}


def submit_job(job: Job, cloud: str, tracker: UsageTracker,
               dry_run: bool = False) -> Dict[str, Any]:
    adapter = ADAPTERS.get(cloud)
    if adapter is None:
        raise ValueError(f"No adapter registered for cloud: {cloud!r}")

    # Cost/time estimates per job (conservative)
    cost_est_usd = 0.002   # ~2 min × $0.001/min Modal equivalent
    min_est      = 0.5     # ~30 sec per job ≈ 0.5 runner-minutes
    hr_est       = 0.008   # ~30 sec per job ≈ 0.008 hr

    # Config-aware adapters (mac_local, circleci_oss, bacalhau) receive their
    # config block so they can read dynamic caps / params without extra env vars.
    if cloud in _CFG_AWARE_ADAPTERS:
        cfg = tracker.cloud_cfg(cloud)
        if cloud == "mac_local":
            receipt = _submit_mac_local(job, dry_run=dry_run, cfg=cfg)
        else:
            receipt = adapter(job, dry_run=dry_run, cfg=cfg)  # type: ignore[call-arg]
    else:
        receipt = adapter(job, dry_run=dry_run)

    tracker.register_submit(cloud,
                            cost_estimate=cost_est_usd,
                            minutes_estimate=min_est,
                            hours_estimate=hr_est)
    log.info("Submitted %s → %s (job_id=%s)", job, cloud, receipt["job_id"])
    return receipt


# ---------------------------------------------------------------------------
# Queue loading
# ---------------------------------------------------------------------------
def load_pending_jobs(queue_file: Path) -> List[Job]:
    """
    Read queue.txt and return all non-comment, non-empty lines as Job objects.
    """
    jobs = []
    if not queue_file.exists():
        log.warning("Queue file not found: %s", queue_file)
        return jobs
    with open(queue_file, "r") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                jobs.append(Job.from_line(line))
            except ValueError as exc:
                log.warning("Line %d skipped: %s", lineno, exc)
    log.info("Loaded %d pending jobs from %s", len(jobs), queue_file)
    return jobs


# ---------------------------------------------------------------------------
# Result polling
# ---------------------------------------------------------------------------
def poll_result(job: Job, timeout_sec: int = 3600) -> bool:
    """
    Block until result.json appears in backtests/<ticker>/<strategy>/
    or timeout is reached. Returns True on success.
    """
    result_path = job.result_file()
    deadline    = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if result_path.exists():
            job.completed_at = _now_iso()
            job.result_path  = result_path
            job.status       = "completed"
            log.info("Result ready: %s → %s", job, result_path)
            return True
        time.sleep(RESULT_POLL_SEC)
    log.warning("Result poll timed out for %s", job)
    job.status = "failed"
    return False


# ---------------------------------------------------------------------------
# Simulation (quota stress test — no real submissions)
# ---------------------------------------------------------------------------
def run_simulation(n_jobs: int, tracker: UsageTracker,
                   throttle_remote: bool = False) -> None:
    """
    Simulate dispatching n_jobs across enabled clouds.
    Shows quota headroom changes without making any real API calls.
    Used for testing the quota logic and the Mac-last-resort policy.

    throttle_remote=True: artificially fills all remote clouds to capacity
    before the first job so that Mac absorbs the remainder, demonstrating
    the fallback behavior.
    """
    log.info(
        "=== SIMULATION: %d mock jobs | throttle_remote=%s ===",
        n_jobs, throttle_remote,
    )
    enabled = tracker.enabled_clouds()
    if not enabled:
        log.error("No enabled clouds in cloud_usage.json. Enable at least one.")
        return

    # --sim-throttle-remote: fill every remote cloud to its concurrency max
    if throttle_remote:
        remote_clouds = [c for c in enabled if c != "mac_local"]
        for c in remote_clouds:
            cfg = tracker.cloud_cfg(c)
            max_c = cfg.get("max_concurrent_jobs",
                    cfg.get("max_concurrent",
                    cfg.get("max_concurrent_containers", 1)))
            cfg["in_flight_jobs"] = max_c
            log.info("Throttled remote cloud %s to max in_flight=%d", c, max_c)
        tracker.save()

    submitted: Dict[str, int] = {c: 0 for c in enabled}
    blocked: List[int] = []

    # For simulation, mock the Mac safety check to always pass (we test routing,
    # not live psutil readings).  We patch by injecting a permissive cfg override.
    mac_sim_cfg: Dict[str, Any] = {
        "max_cpu_pct":  0.0,    # always passes (0 < 60 will never be reached
        "max_mem_pct":  0.0,    # because we skip real psutil in sim mode)
        "max_load":     999.0,
        "max_workers":  999,
    }

    for i in range(n_jobs):
        cloud = tracker.pick_cloud()

        if cloud is None:
            log.warning("Job %d BLOCKED — all clouds over quota/capacity", i + 1)
            blocked.append(i + 1)
            continue

        # Mac local: mock the safety cap check by pre-clearing active procs count
        if cloud == "mac_local":
            _mac_active_procs.clear()   # reset tracker so sim counts are consistent

        # Mock submit: accumulate cost / quota only, don't call real adapter
        tracker.register_submit(cloud, cost_estimate=0.002,
                                minutes_estimate=0.5, hours_estimate=0.008)
        submitted[cloud] += 1

        # Simulate completion of oldest in-flight after every 10 submits
        if (i + 1) % 10 == 0:
            for c in enabled:
                cfg = tracker.cloud_cfg(c)
                # Don't drain throttled remote clouds — they stay saturated
                if throttle_remote and c != "mac_local":
                    continue
                if cfg.get("in_flight_jobs", 0) > 0:
                    tracker.register_complete(c)
                    log.debug("Simulated completion on %s", c)

    log.info("=== SIMULATION RESULTS ===")
    log.info("Total submitted : %d / %d", sum(submitted.values()), n_jobs)
    log.info("Blocked         : %d", len(blocked))
    log.info("")
    for cloud in sorted(enabled, key=lambda c: (c == "mac_local", c)):
        h   = tracker.headroom_pct(cloud)
        cfg = tracker.cloud_cfg(cloud)
        tag = " [MAC LAST-RESORT]" if cloud == "mac_local" else ""
        log.info(
            "  %-20s  submitted=%3d  in_flight=%d  headroom=%6.1f%%%s",
            cloud, submitted[cloud],
            cfg.get("in_flight_jobs", 0), h, tag,
        )
    if blocked:
        log.warning("Blocked job indices (1-based): %s", blocked)

    # Safety assertion: Mac should only dominate when remote is throttled
    mac_count = submitted.get("mac_local", 0)
    remote_count = sum(v for k, v in submitted.items() if k != "mac_local")
    if not throttle_remote and mac_count > 0:
        mac_share = mac_count / (mac_count + remote_count) if (mac_count + remote_count) else 0
        if mac_share > 0.25:
            log.warning(
                "POLICY WARNING: mac_local got %.0f%% of jobs without remote throttling. "
                "Check that remote clouds are enabled and have capacity.",
                mac_share * 100,
            )
        else:
            log.info("Mac policy check OK: mac_local share=%.1f%% (below 25%% threshold)",
                     mac_share * 100)
    elif throttle_remote:
        log.info("Throttled-remote mode: mac_local absorbed %d / %d dispatched jobs",
                 mac_count, sum(submitted.values()))


# ---------------------------------------------------------------------------
# Main dispatch loop
# ---------------------------------------------------------------------------
def dispatch_pass(tracker: UsageTracker, dry_run: bool = False) -> int:
    """
    One full pass: read queue, dispatch all pending jobs, return count submitted.
    """
    jobs = load_pending_jobs(QUEUE_FILE)
    if not jobs:
        log.info("Queue empty — nothing to dispatch.")
        return 0

    n_submitted = 0
    n_blocked   = 0
    submitted_log: List[Dict] = []

    for job in jobs:
        cloud = tracker.pick_cloud()
        if cloud is None:
            log.warning("%s BLOCKED — all enabled clouds at quota/capacity", job)
            n_blocked += 1
            continue

        try:
            receipt = submit_job(job, cloud, tracker, dry_run=dry_run)
            job.cloud        = cloud
            job.submitted_at = receipt["submitted_at"]
            job.status       = "submitted"
            submitted_log.append(receipt)
            n_submitted += 1
        except Exception as exc:
            log.error("Failed to submit %s to %s: %s", job, cloud, exc)

    log.info("Dispatch pass done: submitted=%d blocked=%d", n_submitted, n_blocked)

    # Persist the submission log for audit trail
    log_path = LOG_DIR / f"dispatch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, "w") as f:
        json.dump(submitted_log, f, indent=2)
    log.info("Submission log: %s", log_path)

    return n_submitted


def daemon_loop(tracker: UsageTracker, dry_run: bool = False) -> None:
    log.info("Dispatcher daemon started (poll every %ds). Ctrl-C to stop.", POLL_INTERVAL)
    while True:
        try:
            tracker.load()   # reload in case cloud_usage.json was hand-edited
            dispatch_pass(tracker, dry_run=dry_run)
        except KeyboardInterrupt:
            log.info("Daemon stopped by user.")
            break
        except Exception as exc:
            log.error("Unhandled error in dispatch pass: %s", exc, exc_info=True)
        log.info("Sleeping %ds until next pass...", POLL_INTERVAL)
        time.sleep(POLL_INTERVAL)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-cloud XGBoost sweep dispatcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--daemon",   action="store_true",
                   help="Run forever, polling queue every 30 s")
    p.add_argument("--dry-run",  action="store_true",
                   help="Log decisions but make no real API calls")
    p.add_argument("--simulate", type=int, metavar="N", default=0,
                   help="Run quota simulation with N mock jobs and exit")
    p.add_argument("--sim-throttle-remote", action="store_true",
                   help="During --simulate, pre-fill all remote clouds to max "
                        "capacity so Mac fallback behavior is exercised")
    p.add_argument("--reset-usage", action="store_true",
                   help="Zero out all used_* counters (use at month rollover)")
    return p.parse_args()


def reset_usage(tracker: UsageTracker) -> None:
    """Zero out all billing counters. Run manually at month rollover."""
    for key, cfg in tracker.data.items():
        if key.startswith("_"):
            continue
        for field in ("used_min_this_month", "used_credit_this_month",
                      "used_hr_this_month", "in_flight_jobs"):
            if field in cfg:
                cfg[field] = 0
    tracker.data["_last_reset"] = _now_iso()
    tracker.save()
    log.info("Usage counters reset for new month.")


def main() -> None:
    args   = parse_args()
    tracker = UsageTracker(USAGE_FILE)

    if args.reset_usage:
        reset_usage(tracker)
        return

    if args.simulate:
        run_simulation(args.simulate, tracker,
                       throttle_remote=args.sim_throttle_remote)
        return

    if args.daemon:
        daemon_loop(tracker, dry_run=args.dry_run)
    else:
        dispatch_pass(tracker, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
