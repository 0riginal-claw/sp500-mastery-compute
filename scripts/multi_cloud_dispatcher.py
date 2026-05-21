"""
multi_cloud_dispatcher.py — Multi-cloud XGBoost sweep dispatcher.

Reads jobs from sweeps/queue.txt, picks the cloud with the most free headroom,
submits via cloud-specific adapters, tracks quotas in sweeps/cloud_usage.json,
and polls for completed results in backtests/<ticker>/<strategy>/result.json.

Phase 1 adapters (implemented):
    github_actions  — triggers workflow_dispatch via GitHub API
    modal           — calls Modal SDK to submit a remote function call

Phase 2 adapters (stubs, enabled=false by default):
    oracle_a1          — SSH to Oracle Ampere A1 permanent free instance
    gcp_ssh            — SSH to GCP e2-micro always-free instance
    aws_ssh            — SSH to AWS t2.micro free-tier instance
    render_api         — Render.com managed service API
    railway_api        — Railway.app API
    fly_api            — Fly.io Machines API
    drone_ci           — Self-hosted Drone CI server (friend-donated agents)
    circleci_oss       — CircleCI OSS plan (approved repos: 400k credits/mo ≈ 40k min)
    ibm_code_engine    — IBM Code Engine batch jobs (Lite plan: 100k vCPU-s/mo free,
                         no CC). enabled=false until signup.
    firebase_functions — Google Cloud Functions Gen2 / Firebase (Spark plan: 2M
                         invocations/mo free). Bearer OIDC token auth. enabled=false
                         until user signs up and deploys the function.

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

Phase 5 adapters (implemented, enabled=false until user signup):
    northflank      — Northflank.com free tier (2 free jobs, no CC required).
                      Triggers a named Northflank job via the REST API at
                      api.northflank.com/v1. Signup with GitHub at
                      https://northflank.com. Stdlib urllib only — no extra deps.

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

    # Show each enabled cloud + its cost_tier + current headroom, then exit
    python scripts/multi_cloud_dispatcher.py --show-tiers

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
    IBMCLOUD_API_KEY    — IBM Cloud API key (IAM > API keys). Needed for ibm_code_engine adapter.
    IBM_CE_PROJECT_ID   — IBM Code Engine project ID (from the CE console).
    NORTHFLANK_API_TOKEN  — Northflank API token (Project > API tokens in the UI).
    NORTHFLANK_PROJECT_ID — Northflank project ID (visible in the UI URL).
    NORTHFLANK_JOB_ID     — Northflank job name/ID (the job to trigger per run).
    IBMCLOUD_API_KEY      — IBM Cloud API key (IAM > API keys). ibm_code_engine adapter.
    IBM_CE_PROJECT_ID     — IBM Code Engine project ID (from the CE console).
    FIREBASE_PROJECT_ID   — GCP/Firebase project ID. firebase_functions adapter.
    FIREBASE_ID_TOKEN     — Short-lived OIDC identity token for the invoker service
                            account. Obtain via: gcloud auth print-identity-token
                            --audiences=<function-url>
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

try:
    import psutil  # type: ignore
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Project root (works both locally and in any cloud runner that checks out repo)
# ---------------------------------------------------------------------------
PROJECT_ROOT  = Path(__file__).parent.parent
QUEUE_FILE    = PROJECT_ROOT / "sweeps" / "queue.txt"
USAGE_FILE    = PROJECT_ROOT / "sweeps" / "cloud_usage.json"
STATUS_FILE   = PROJECT_ROOT / "sweeps" / "dispatched.jsonl"
RESULTS_DIR   = PROJECT_ROOT / "backtests"
LOG_DIR       = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

POLL_INTERVAL   = 30   # seconds between queue checks in daemon mode
RESULT_POLL_SEC = 10   # seconds between result-file polling in sync wait

# ---------------------------------------------------------------------------
# Cloud-first 90% routing policy (2026-05-17)
# ---------------------------------------------------------------------------
# Per operator mandate: Mac CAN absorb jobs, BUT only after aggregated remote
# cloud capacity reaches 90%. Below that threshold the dispatcher SLEEPS for
# CLOUD_WAIT_SLEEP_S waiting for cloud capacity rather than falling to mac.
#
# Exception: jobs flagged priority="P1" (urgent) can use mac_local after at
# most P1_CLOUD_WAIT_MAX_S seconds of cloud waiting — this preserves a fast
# path for emergencies without leaking the local-fallback into normal flow.
#
# These constants can be overridden via cloud_usage.json["_policy"] block
# (preferred for prod tuning) or via CLI flags --cloud-first-pct / --cloud-
# wait-sleep / --no-cloud-first (operator escape hatch).
CLOUD_FIRST_MIN_USAGE_PCT = 90.0   # mac_local engaged only at/above this
CLOUD_WAIT_SLEEP_S        = 30     # sleep between retries when waiting for cloud
CLOUD_WAIT_MAX_RETRIES    = 20     # cap total wait at 10 min then give up
P1_CLOUD_WAIT_MAX_S       = 300    # P1 jobs can use mac after 5 min of waiting
CLOUD_FIRST_ENABLED       = True   # global default; --no-cloud-first overrides

# ---------------------------------------------------------------------------
# Tier-based cloud preference order
# ---------------------------------------------------------------------------
# TIER_ORDER defines the preference hierarchy used by pick_cloud().
# The dispatcher always exhausts earlier tiers before moving to later ones.
#   free   — always-free quota or unlimited public CI minutes; spend $0 ever
#   credit — starts with a free credit allowance then bills a CC on exhaustion
#   paid   — always-billed; reserved for future clouds
#   local  — mac_local; LAST RESORT behind all remote options
# Within each tier, the cloud with the highest headroom_pct() wins.
# This constant mirrors the "cost_tier" field in cloud_usage.json.
TIER_ORDER: List[str] = ["free", "credit", "paid", "local"]

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
# Concurrency helpers
# ---------------------------------------------------------------------------

@contextmanager
def _flocked(path: Path, mode: str = "a") -> Generator:
    """Acquire an exclusive OS-level lock on *path*; release on exit.

    Creates the file (and parent dirs) if absent.  The lock is held for the
    entire duration of the ``with`` block, which must be kept short so
    concurrent producers are not starved.

    Args:
        path: File to lock.
        mode: Open mode passed to ``open()``.  Use ``"r+"`` for read-modify-write
              and ``"a"`` for pure append.

    Yields:
        The open file object, seeked to the correct position for *mode*.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    fh = open(path, mode, encoding="utf-8")
    fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
    try:
        yield fh
    finally:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        fh.close()


def _write_status_event(
    job: "Job",
    status: str,
    cloud: Optional[str] = None,
    result_path: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Append one status-transition event to sweeps/dispatched.jsonl.

    Uses the same append-only event-log pattern as cloud_dispatch.enqueue_job:
    each call writes a NEW line with the same job_id and the updated status +
    timestamp.  ``cloud_dispatch.check_status`` scans all rows for a job_id
    and returns the one with the latest ``ts`` field, so calling this function
    is the only change needed to make status advances visible to consumers.

    Args:
        job:         The Job dataclass that just transitioned.
        status:      New status string: ``"submitted"`` | ``"complete"`` | ``"failed"``.
        cloud:       Cloud that accepted the job (or None if not yet assigned).
        result_path: Path to the result artifact once the job completes.
        extra:       Optional dict of additional fields to merge into the record.
                     Use this to store cloud-specific receipt fields (e.g.
                     ``{"gh_workflow": "sweep.yml"}``).
    """
    record: Dict[str, Any] = {
        "id":          job.job_id,
        "ticker":      job.ticker,
        "strategy":    job.strategy,
        "script":      job.script,
        "status":      status,
        "cloud":       cloud or job.cloud,
        "result_path": result_path,
        "ts":          _now_iso(),
    }
    if extra:
        record.update(extra)

    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _flocked(STATUS_FILE, mode="a") as fh:
        fh.write(line)

    log.debug(
        "Status event written: job_id=%s status=%s cloud=%s",
        job.job_id, status, record["cloud"],
    )


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
    # extra_env (added 2026-05-20): per-job env-var overrides forwarded to the
    # worker process in every adapter (Modal, gh_actions, mac_local, cerebras,
    # groq, lambda, replicate, runpod, vast, northflank, ibm_code_engine,
    # firebase_functions, drone_ci, bacalhau, circleci_oss). Populated by
    # load_pending_jobs() from the dispatched.jsonl ledger record matching this
    # job_id. Keys MUST be valid env var names (e.g. XGB_NO_TOPK,
    # INTERACTION_CONSTRAINTS_JSON, MONOTONIC_CONSTRAINTS_JSON). Values are
    # stringified before being passed to subprocess.Popen/HTTP-payload.
    extra_env: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_line(cls, line: str) -> "Job":
        """Parse a queue.txt line.

        Accepts both the legacy 3-token format ``<script> <ticker> <strategy>``
        and the 4-token format ``<script> <ticker> <strategy> <job_id>`` written
        by ``cloud_dispatch.enqueue_job`` (added 2026-05-16).  When the 4th
        token is present it is used as the job_id so that the dispatcher can
        match queue lines to dispatched.jsonl records by ID instead of only by
        (ticker, strategy).
        """
        parts = line.strip().split()
        if len(parts) == 4:
            return cls(script=parts[0], ticker=parts[1],
                       strategy=parts[2], job_id=parts[3])
        if len(parts) == 3:
            return cls(script=parts[0], ticker=parts[1], strategy=parts[2])
        raise ValueError(
            f"Malformed queue line (expected 3 or 4 tokens, got {len(parts)}): {line!r}"
        )

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

        # If the cloud is in auth-failure cooldown, treat as no headroom.
        cooldown_until = cfg.get("cooldown_until")
        if cooldown_until is not None and time.time() < cooldown_until:
            remaining_min = (cooldown_until - time.time()) / 60
            log.debug(
                "%s is in auth cooldown (%.1f min remaining) — headroom=-1",
                cloud, remaining_min,
            )
            return -1.0

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

        elif model == "invocations":
            # Firebase Functions / similar: quota is invocations per month
            used  = cfg.get("used_invocations_this_month", 0)
            quota = cfg.get("quota_invocations_per_month", 0)
            used_frac = used / quota if quota else 1.0
            if used_frac >= safety:
                return -1.0
            return (safety - used_frac) / safety * 100.0

        else:  # concurrent_cap — headroom = remaining concurrency slots (0–100)
            slots_free = max_concurrent - in_flight
            return (slots_free / max_concurrent) * 100.0

    def pick_cloud(self, clouds: Optional[List[str]] = None,
                   prefer_remote: bool = True) -> Optional[str]:
        """Return the best available cloud using tier-based preference ordering.

        Routing policy (in priority order):
            1. ``free`` tier  — always-free services (GitHub Actions, Oracle A1,
               GCP e2-micro, Bacalhau, Drone CI, Northflank, Firebase, IBM CE,
               AWS free tier). Drained aggressively (safety_margin_pct=90) before
               falling to any credit-bearing cloud.
            2. ``credit`` tier — services with a free credit allowance that bill a
               CC once exhausted (Modal $30, CircleCI OSS 400k credits, Railway
               $5/mo, Fly.io 3 VMs). Only selected when all free clouds are full.
            3. ``paid`` tier  — always-billed; reserved for future clouds.
            4. ``local`` tier — mac_local; absolute last resort regardless of
               prefer_remote flag.

        Within each tier, the cloud with the highest ``headroom_pct()`` is chosen.
        Ties are broken alphabetically for determinism.

        Backward-compatibility notes:
            - ``prefer_remote=False`` (small-job path) skips the tier gate and
              picks the highest-headroom cloud across all enabled clouds, which
              keeps the original small-job Mac-first behaviour.
            - When only one cloud is enabled the result is identical to the old
              headroom-only approach.
            - When all clouds share the same tier the result is identical to the
              old behaviour (highest headroom wins).

        Args:
            clouds:        Explicit candidate list; defaults to all enabled clouds.
            prefer_remote: When True (default) enforce tier ordering with mac_local
                           last. When False allow mac_local as a first-class
                           candidate (small-job / testing path).

        Returns:
            Name of the selected cloud, or ``None`` if nothing has headroom.
        """
        candidates = clouds if clouds else self.enabled_clouds()

        if not prefer_remote:
            # Small-job / legacy path: flat headroom ranking, Mac is first-class
            all_ranked = sorted(
                [(c, self.headroom_pct(c)) for c in candidates],
                key=lambda x: (-x[1], x[0]),
            )
            log.debug(
                "pick_cloud (prefer_remote=False) headroom: %s",
                {c: f"{h:.1f}%" for c, h in all_ranked},
            )
            if not all_ranked or all_ranked[0][1] < 0:
                return None
            return all_ranked[0][0]

        # Tier-aware path (default): group enabled clouds by cost_tier, then
        # iterate TIER_ORDER and pick the highest-headroom cloud from the first
        # non-empty tier that has at least one cloud with positive headroom.
        by_tier: Dict[str, List[Tuple[str, float]]] = {t: [] for t in TIER_ORDER}
        unrecognised: List[Tuple[str, float]] = []

        for name in candidates:
            tier = self.data[name].get("cost_tier", "credit")
            hp   = self.headroom_pct(name)
            if hp > 0:
                if tier in by_tier:
                    by_tier[tier].append((name, hp))
                else:
                    # Unknown tier: treat as credit so we still pick it
                    log.warning(
                        "Cloud %r has unrecognised cost_tier=%r — treating as 'credit'",
                        name, tier,
                    )
                    unrecognised.append((name, hp))

        # Merge unrecognised into credit bucket
        by_tier["credit"].extend(unrecognised)

        log.debug(
            "pick_cloud tier buckets: %s",
            {t: [(c, f"{h:.1f}%") for c, h in v] for t, v in by_tier.items() if v},
        )

        for tier in TIER_ORDER:
            bucket = by_tier[tier]
            if bucket:
                chosen, headroom = max(bucket, key=lambda x: (x[1], x[0]))
                if tier == "local":
                    log.info(
                        "All remote clouds at capacity — falling back to mac_local "
                        "(headroom=%.1f%%)",
                        headroom,
                    )
                elif tier == "credit":
                    log.info(
                        "All free clouds at capacity — escalating to credit tier: "
                        "%s (headroom=%.1f%%)",
                        chosen, headroom,
                    )
                return chosen

        return None  # all clouds exhausted

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
        elif model == "invocations":
            cfg["used_invocations_this_month"] = cfg.get("used_invocations_this_month", 0) + 1
        self.save()

    def register_complete(self, cloud: str) -> None:
        """Decrement in-flight counter."""
        cfg = self.cloud_cfg(cloud)
        cfg["in_flight_jobs"] = max(0, cfg.get("in_flight_jobs", 0) - 1)
        self.save()

    def over_safety_margin(self, cloud: str) -> bool:
        return self.headroom_pct(cloud) < 0

    # ------------------------------------------------------------------
    # Cloud-first 90% routing (2026-05-17 mandate)
    # ------------------------------------------------------------------
    def aggregate_remote_usage_pct(self) -> float:
        """Return SUM(in_flight / max_concurrent) across all ENABLED remote
        (non-mac_local) clouds, expressed as percent of total concurrency.

        Used by ``pick_cloud_cloud_first`` to decide whether mac_local is
        eligible: mac is engaged only when aggregate >= CLOUD_FIRST_MIN_USAGE_PCT.

        Implementation notes:
            - Auth-cooldown clouds are EXCLUDED from the denominator (they can
              not absorb jobs right now, so counting their slots would
              under-state real saturation and starve mac unnecessarily).
            - Returns 0.0 if no enabled remote clouds remain (forces caller
              to fall to mac immediately — there is nothing else to pick).
        """
        total_slots   = 0
        used_slots    = 0
        now = time.time()
        for name in self.enabled_clouds():
            if name == "mac_local":
                continue
            cfg = self.cloud_cfg(name)
            cooldown_until = cfg.get("cooldown_until")
            if cooldown_until is not None and now < cooldown_until:
                continue  # cloud is dead-air for now, don't count its slots
            max_c = cfg.get("max_concurrent_jobs",
                    cfg.get("max_concurrent",
                    cfg.get("max_concurrent_containers", 1)))
            in_flight = cfg.get("in_flight_jobs", 0)
            total_slots += max_c
            used_slots  += min(in_flight, max_c)
        if total_slots <= 0:
            return 0.0
        return (used_slots / total_slots) * 100.0

    def pick_cloud_cloud_first(
        self,
        clouds: Optional[List[str]] = None,
        priority: str = "P2",
        waited_s: float = 0.0,
        min_remote_usage_pct: float = CLOUD_FIRST_MIN_USAGE_PCT,
    ) -> Tuple[Optional[str], str]:
        """Cloud-first routing per the 2026-05-17 mandate.

        Behaviour vs the legacy ``pick_cloud``:
            * Returns a (cloud, action) pair. ``action`` is one of:
                - ``"submit"``    — cloud is valid, submit the job there
                - ``"wait"``      — no remote capacity AND aggregate <90%;
                                    caller should sleep and retry
                - ``"none"``      — all clouds (including mac) over quota /
                                    nothing usable; caller marks job BLOCKED
            * ``mac_local`` is suppressed unless ONE of:
                - aggregate remote usage >= min_remote_usage_pct (90% default), OR
                - priority == "P1" AND waited_s >= P1_CLOUD_WAIT_MAX_S, OR
                - no enabled remote clouds remain at all

        The standard ``pick_cloud`` is left intact so legacy callers (simulator,
        ``prefer_remote=False`` small-job path) keep their behaviour.

        Args:
            clouds:                 Explicit candidate list; defaults to enabled.
            priority:               "P1" allows mac after waited_s >= 5min;
                                    "P2" (default) follows strict cloud-first.
            waited_s:               How long the caller has already been waiting
                                    on this job (used only for P1 escalation).
            min_remote_usage_pct:   Override the 90% threshold (testing/ops).

        Returns:
            (cloud_name | None, action_string)
        """
        candidates = clouds if clouds else self.enabled_clouds()
        # Carve out mac_local — it gets gated on aggregate usage.
        remote_candidates = [c for c in candidates if c != "mac_local"]
        has_mac           = "mac_local" in candidates

        # First, try to pick a remote cloud using the existing tier logic.
        remote_pick = self.pick_cloud(clouds=remote_candidates, prefer_remote=True) \
                      if remote_candidates else None
        if remote_pick is not None and remote_pick != "mac_local":
            return remote_pick, "submit"

        # No remote pick — decide between WAIT and MAC fallback.
        agg_usage = self.aggregate_remote_usage_pct()
        any_remote_enabled = any(
            c != "mac_local" for c in self.enabled_clouds()
        )

        # Case A: no remote clouds enabled at all — mac is the only option.
        if not any_remote_enabled:
            if has_mac:
                return "mac_local", "submit"
            return None, "none"

        # Case B: P1 escalation after long wait.
        if priority == "P1" and waited_s >= P1_CLOUD_WAIT_MAX_S and has_mac:
            log.info(
                "P1 escalation: cloud waited %.0fs (>= %ds) — engaging mac_local",
                waited_s, P1_CLOUD_WAIT_MAX_S,
            )
            return "mac_local", "submit"

        # Case C: aggregate remote usage hit 90% — mac is now eligible.
        if agg_usage >= min_remote_usage_pct and has_mac:
            log.info(
                "Cloud-first threshold reached: agg_remote_usage=%.1f%% "
                ">= %.1f%% — falling to mac_local",
                agg_usage, min_remote_usage_pct,
            )
            return "mac_local", "submit"

        # Case D: under threshold — tell caller to wait for cloud capacity.
        log.info(
            "Cloud-first WAIT: agg_remote_usage=%.1f%% < %.1f%%; "
            "remote candidates over capacity, waiting %ds for cloud slot",
            agg_usage, min_remote_usage_pct, CLOUD_WAIT_SLEEP_S,
        )
        return None, "wait"

    # ------------------------------------------------------------------
    # Auth-failure cooldown helpers
    # ------------------------------------------------------------------
    def mark_auth_failure(self, cloud: str) -> None:
        """
        Record an auth failure (HTTP 401/403) for cloud and apply
        exponential backoff cooldown: 60 * 2^min(count-1, 5) seconds
        (1 min, 2 min, 4 min … capped at ~32 min).
        Persists state to cloud_usage.json.
        """
        cfg = self.cloud_cfg(cloud)
        count = cfg.get("auth_failure_count", 0) + 1
        cfg["auth_failure_count"] = count
        cooldown_sec = 60 * (2 ** min(count - 1, 5))
        cfg["cooldown_until"] = time.time() + cooldown_sec
        self.save()
        log.warning(
            "AUTH FAILURE on %s: cooldown %d min (failure #%d)",
            cloud, cooldown_sec // 60, count,
        )

    def mark_success(self, cloud: str) -> None:
        """
        Called after a successful submit on cloud.
        Clears auth_failure_count and cooldown_until.
        """
        cfg = self.cloud_cfg(cloud)
        if cfg.get("auth_failure_count", 0) > 0 or cfg.get("cooldown_until") is not None:
            cfg["auth_failure_count"] = 0
            cfg["cooldown_until"] = None
            self.save()
            log.info("Auth state cleared for %s after successful submit", cloud)

    def reset_cooldowns(self) -> None:
        """
        Operator override: clear all auth_failure_count and cooldown_until
        fields across every cloud.  Called via --reset-cooldowns CLI flag.
        """
        for key, cfg in self.data.items():
            if key.startswith("_"):
                continue
            cfg["auth_failure_count"] = 0
            cfg["cooldown_until"] = None
        self.save()
        log.info("All cloud cooldowns reset by operator.")


# ---------------------------------------------------------------------------
# Cloud helpers
# ---------------------------------------------------------------------------
def _load_github_actions_cfg() -> Dict[str, Any]:
    """
    Read the github_actions block from cloud_usage.json (schema v1.2+).
    Returns {} if file missing or block absent.

    Used as fallback when GITHUB_OWNER / GITHUB_REPO / GITHUB_BRANCH /
    GITHUB_WORKFLOW_ID env vars are not set. Env vars still take precedence
    when present (changelog v1.2 contract).
    """
    try:
        with open(USAGE_FILE, "r") as f:
            data = json.load(f)
        return data.get("github_actions", {}) or {}
    except Exception as exc:
        log.debug("Could not read github_actions block from %s: %s", USAGE_FILE, exc)
        return {}


def _is_public_repo() -> bool:
    """
    True if the configured GitHub repo is public (unlimited Actions minutes).

    Resolution order:
        1. GITHUB_IS_PUBLIC env var ("true"/"false") — explicit operator override
        2. cloud_usage.json github_actions.is_public — config-file default
        3. False — conservative default (treat as private-cap)
    """
    env_val = os.environ.get("GITHUB_IS_PUBLIC", "").lower()
    if env_val in ("true", "false"):
        return env_val == "true"
    cfg = _load_github_actions_cfg()
    return bool(cfg.get("is_public", False))


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
    # Resolve owner/repo/branch/workflow — env vars take precedence, then
    # fall back to cloud_usage.json github_actions block (schema v1.2+),
    # then to a clearly-invalid placeholder so failures are loud.
    cfg = _load_github_actions_cfg()

    # GITHUB_TOKEN preferred; GH_TOKEN accepted as alias (gh CLI convention).
    token    = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""
    owner    = os.environ.get("GITHUB_OWNER")        or cfg.get("owner")    or "YOUR_GITHUB_USER"
    repo     = os.environ.get("GITHUB_REPO")         or cfg.get("repo")     or "sp500-ticker-mastery"
    workflow = os.environ.get("GITHUB_WORKFLOW_ID")  or cfg.get("workflow") or "sweep.yml"
    branch   = os.environ.get("GITHUB_BRANCH")       or cfg.get("branch")   or "main"

    # XSEC branch (added 2026-05-20): if this is a full-S&P-500 cross-sectional
    # job (ticker="ALL" + strategy contains "xsec"), route to xsec.yml workflow
    # which uses a single 4-core ubuntu-latest runner with timeout-minutes=350
    # (max for free tier is 6h=360min) instead of the 20-parallel matrix shape
    # of sweep.yml. This is the Modal cap fallback: Modal's A10G GPU job (1-2hr)
    # is replaced by gh_actions CPU job (~2-6hr with tree_method=hist). Free
    # tier private repo = 2000 min/mo, well within budget for occasional
    # full-panel retrain.
    # autosolve_skip: leaf-task dispatcher branch — no error condition
    is_xsec_megajob = (
        (job.ticker or "").upper() == "ALL"
        and "xsec" in (job.strategy or "").lower()
    )
    if is_xsec_megajob:
        # Allow per-env override; fall back to xsec.yml.
        workflow = (
            os.environ.get("GITHUB_XSEC_WORKFLOW_ID")
            or cfg.get("xsec_workflow")
            or "xsec.yml"
        )
        log.info(
            "[gh_actions] XSEC mega-job detected — routing to %s workflow "
            "(CPU runner, tree_method=hist fallback for Modal A10G)",
            workflow,
        )

    if not token:
        log.warning("GITHUB_TOKEN/GH_TOKEN not set — adapter will fail in production")
    if owner == "YOUR_GITHUB_USER":
        log.error(
            "GITHUB_OWNER unresolved (env unset AND cloud_usage.json "
            "github_actions.owner missing) — dispatch will 404. "
            "Set env GITHUB_OWNER=<your-gh-user> or add 'owner' to "
            "sweeps/cloud_usage.json github_actions block."
        )

    if is_xsec_megajob:
        # xsec.yml uses a flatter input set: it always runs the full S&P-500
        # panel via registry/sp500_tickers.csv container-side, so we only
        # need to forward script/strategy/job_id (+ optional tickers-file
        # override + extra_env).
        payload = {
            "ref": branch,
            "inputs": {
                "strategy": job.strategy,
                "script":   job.script or "scripts/backtest_xgb_v10_xsec.py",
                "job_id":   job.job_id,
                "tickers_file": (job.extra_env or {}).get(
                    "TICKERS_FILE", "registry/sp500_tickers.csv"
                ),
            },
        }
    else:
        payload = {
            "ref": branch,
            "inputs": {
                "ticker":   job.ticker,
                "strategy": job.strategy,
                "script":   job.script,
                "job_id":   job.job_id,
            },
        }
    # Forward per-job extra_env (added 2026-05-20): encode as JSON string in
    # inputs.extra_env_json (workflow_dispatch inputs MUST be strings — GitHub
    # API rejects nested objects). The sweep.yml/xsec.yml workflow decodes this
    # back into individual env vars before running the backtest script.
    if job.extra_env:
        try:
            payload["inputs"]["extra_env_json"] = json.dumps(job.extra_env)
        except (TypeError, ValueError) as _exc:
            log.warning("Could not JSON-encode extra_env for %s: %s — omitting",
                        job, _exc)

    url = f"https://api.github.com/repos/{owner}/{repo}/actions/workflows/{workflow}/dispatches"

    # Log resolved URL once per dispatcher process so 404s are diagnosable.
    if not getattr(_submit_github_actions, "_url_logged", False):
        log.info(
            "GitHub Actions adapter resolved: owner=%s repo=%s branch=%s "
            "workflow=%s public=%s token_set=%s url=%s",
            owner, repo, branch, workflow, _is_public_repo(),
            bool(token), url,
        )
        _submit_github_actions._url_logged = True  # type: ignore[attr-defined]

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
# Module-level registry of Modal CLI subprocesses launched by this session.
# Keyed by job_id → Popen object. The completion poller (sweep) checks
# proc.poll() to detect when the `modal run` subprocess exits — this is the
# remote-equivalent of the result.json arrival signal, since Modal workers
# can't write to the dispatcher Mac's filesystem until the local entrypoint
# returns control. Fix for modal.in_flight_jobs pinning at max (2026-05-17,
# mirroring the gh_actions completion-poll pattern in _fetch_github_actions_runs_by_jobid).
_modal_active_procs: Dict[str, "subprocess.Popen[bytes]"] = {}


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
    # Modal CLI requires explicit entrypoint when modal_worker.py defines multiple
    # local_entrypoints (main, echo_smoke, xsec).  Without `::main` the CLI errors with
    # "Specify a Modal Function or local entrypoint to run" and returncode=1
    # (verified 2026-05-18: every retry submit was failing for this reason, not
    # for billing — see reports/budget_allocation_2026-05-18.md).
    #
    # XSEC branch (added 2026-05-20): if this is a full-S&P-500 cross-sectional
    # job (ticker="ALL" + strategy contains "xsec"), invoke ::xsec entrypoint
    # instead of ::main. The xsec entrypoint has gpu=A10G + memory=32GB and
    # the CLI shape that backtest_xgb_v10_xsec.py expects (--tickers-csv +
    # --output-dir, not --ticker single).
    # autosolve_skip: leaf-task dispatcher branch — no error condition
    is_xsec_megajob = (
        (job.ticker or "").upper() == "ALL"
        and "xsec" in (job.strategy or "").lower()
    )
    if is_xsec_megajob:
        log.info("[modal] XSEC mega-job detected — routing to ::xsec entrypoint")
        cmd = [
            sys.executable, "-m", "modal", "run",
            f"{modal_worker}::xsec",
            # Default tickers-csv path — xsec entrypoint resolves it container-side.
            "--tickers-csv", "registry/sp500_tickers.csv",
            "--strategy", job.strategy,
            "--script", job.script,
            "--job-id", job.job_id,
        ]
    else:
        cmd = [
            sys.executable, "-m", "modal", "run",
            f"{modal_worker}::main",
            "--ticker",   job.ticker,
            "--strategy", job.strategy,
            "--script",   job.script,
            "--job-id",   job.job_id,
        ]
    # Forward per-job extra_env (added 2026-05-20): encode as JSON and pass via
    # --extra-env-json CLI flag. The modal local_entrypoint::main reads it and
    # forwards into run_backtest.remote(extra_env=...) which merges into the
    # subprocess env inside the container.
    if job.extra_env:
        try:
            cmd.extend(["--extra-env-json", json.dumps(job.extra_env)])
        except (TypeError, ValueError) as _exc:
            log.warning("Could not JSON-encode extra_env for %s: %s — sending empty",
                        job, _exc)
    env = os.environ.copy()
    env["MODAL_TOKEN_ID"]     = os.environ.get("MODAL_TOKEN_ID", "")
    env["MODAL_TOKEN_SECRET"] = os.environ.get("MODAL_TOKEN_SECRET", "")
    # Also inject extra_env into the *local* modal CLI subprocess env. Some
    # adapters (e.g. when modal_worker is updated to read os.environ directly)
    # can pick them up from here. Harmless for current modal_worker which uses
    # the explicit --extra-env-json arg.
    for k, v in (job.extra_env or {}).items():
        if k and v is not None:
            env[str(k)] = str(v)

    log.info("Submitting to Modal: %s", " ".join(cmd))
    try:
        proc = subprocess.Popen(
            cmd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        # We don't wait — dispatcher moves on; result polling detects completion
        log.info("Modal process pid=%s for %s", proc.pid, job)
        # Track the Popen handle so the completion poller can detect exit
        # without depending solely on the result.json arriving locally
        # (which fails silently when the modal CLI subprocess dies early).
        _modal_active_procs[job.job_id] = proc
    except FileNotFoundError:
        log.error("modal CLI not found — install with: pip install modal")
        raise

    return {
        "job_id":       job.job_id,
        "cloud":        "modal",
        "submitted_at": _now_iso(),
        "pid":          proc.pid,
    }


# ---------------------------------------------------------------------------
# Tier-S #6 (2026-05-21): Modal batch fan-out via Function.map
# ---------------------------------------------------------------------------
# Instead of shelling out to the `modal` CLI once per job (which spawns a
# Python subprocess + bootstraps the Modal SDK every time — ~3-5s of pure
# overhead per submit), `_submit_modal_batch` calls `Function.map(args_iter)`
# ONCE, and the Modal control-plane fans the iterable out across up to
# `max_containers` workers concurrently. For a 500-ticker sweep this drops
# total submit latency from ~25-40min (sequential CLI shell-outs) to ~5-15s
# (single SDK call) and lets Modal schedule 100-500 containers in parallel.
#
# The function returns a list of result dicts shaped like _submit_modal's
# return so the rest of the dispatcher (status writer, sweep poller) can
# treat batch and single submissions uniformly.
#
# Notes:
#   * Requires `modal` SDK installed on the dispatcher (it already is —
#     auto_cloud_dispatcher imports it). We use the SDK directly here, not
#     the CLI subprocess.
#   * Uses `lookup()` to attach to the deployed `sp500-mastery` app. The
#     dispatcher cannot deploy at submit time; the operator must have run
#     `modal deploy scripts/modal_worker.py` once first.
#   * `order_outputs=False` lets Modal return results as they finish (not
#     in submit order) — fine because each result dict carries its own
#     job_id. This is the highest-throughput option.
#   * `return_exceptions=True` so a single failing job doesn't crash the
#     whole batch — failed jobs return as exception objects that the
#     caller can detect and re-route.
# autosolve_skip: tier-S #6 batch-enqueue addition — no error condition
def _submit_modal_batch(
    jobs: List[Job],
    dry_run: bool = False,
    max_containers: int = 500,
) -> List[Dict[str, Any]]:
    """
    Submit a batch of Modal jobs via Function.map for parallel container fan-out.

    Args:
        jobs            — list of Job to dispatch (same shape as _submit_modal expects)
        dry_run         — log-only, don't call the SDK
        max_containers  — upper bound on concurrent Modal containers
                         (Modal's hard cap is ~1000; we default to 500 to stay
                         well under quota for a single S&P 500 sweep)

    Returns:
        List of submission-result dicts (one per job), in input order.
    """
    if not jobs:
        return []

    if dry_run:
        for j in jobs:
            log.info("[DRY-RUN][modal-batch] Would map: %s/%s via %s",
                     j.ticker, j.strategy, j.script)
        return [
            {"job_id": j.job_id, "cloud": "modal", "submitted_at": _now_iso(),
             "dry_run": True, "batch": True}
            for j in jobs
        ]

    try:
        import modal  # local import — only required when batch path is hot
    except ImportError:
        log.error("modal SDK not installed — `pip install modal` required for batch submit")
        raise

    # Pull a handle on the deployed function. App name + function name must
    # match what `modal_worker.py` defines (modal.App("sp500-mastery") +
    # @app.function def run_backtest).
    app_name = os.environ.get("MODAL_APP_NAME", "sp500-mastery")
    fn_name  = "run_backtest"
    try:
        run_backtest_fn = modal.Function.from_name(app_name, fn_name)
    except Exception as exc:
        log.error("Could not look up Modal Function %s::%s — did you `modal deploy`? %s",
                  app_name, fn_name, exc)
        raise

    # Build the arg iterable. Modal's .map() accepts either positional args
    # iterable or kwargs via .starmap()/.map() with tuples. We use kwargs via
    # a list of tuples that match run_backtest's signature exactly.
    args_iter: List[Tuple[str, str, str, str, Optional[Dict[str, str]]]] = [
        (j.ticker, j.strategy, j.script, j.job_id, j.extra_env or None)
        for j in jobs
    ]

    log.info("[modal-batch] Dispatching %d jobs via Function.map (max_containers=%d)",
             len(jobs), max_containers)

    submitted_at = _now_iso()
    submission_results: List[Dict[str, Any]] = []

    # .spawn_map() — fire-and-forget — submits all jobs to Modal's queue and
    # returns immediately without blocking on results. The dispatcher's
    # existing completion poller (result.json arrival) handles the rest.
    # This matches the existing _submit_modal semantics (non-blocking) while
    # cutting per-job submit overhead from ~3-5s to ~0.01s.
    try:
        # spawn_map returns FunctionCall objects we don't strictly need;
        # we just want the submit to complete.
        run_backtest_fn.spawn_map.aio  # touch attribute to confirm SDK supports it
    except AttributeError:
        # Older Modal SDK — fall back to .spawn() per-job (still skips CLI subprocess)
        log.warning("[modal-batch] spawn_map unavailable; falling back to per-job .spawn()")
        for j, args in zip(jobs, args_iter):
            try:
                fc = run_backtest_fn.spawn(*args)
                submission_results.append({
                    "job_id": j.job_id, "cloud": "modal",
                    "submitted_at": submitted_at, "batch": True,
                    "modal_call_id": getattr(fc, "object_id", None),
                })
            except Exception as exc:
                log.error("[modal-batch] spawn failed for %s: %s", j.job_id, exc)
                submission_results.append({
                    "job_id": j.job_id, "cloud": "modal",
                    "submitted_at": submitted_at, "batch": True,
                    "error": str(exc),
                })
        return submission_results

    # Preferred path: single spawn_map call for all jobs at once
    try:
        run_backtest_fn.spawn_map(args_iter)
        for j in jobs:
            submission_results.append({
                "job_id": j.job_id, "cloud": "modal",
                "submitted_at": submitted_at, "batch": True,
            })
    except Exception as exc:
        log.error("[modal-batch] spawn_map failed (%s); falling back to per-job .spawn()", exc)
        for j, args in zip(jobs, args_iter):
            try:
                fc = run_backtest_fn.spawn(*args)
                submission_results.append({
                    "job_id": j.job_id, "cloud": "modal",
                    "submitted_at": submitted_at, "batch": True,
                    "modal_call_id": getattr(fc, "object_id", None),
                })
            except Exception as exc2:
                log.error("[modal-batch] fallback spawn failed for %s: %s", j.job_id, exc2)
                submission_results.append({
                    "job_id": j.job_id, "cloud": "modal",
                    "submitted_at": submitted_at, "batch": True,
                    "error": str(exc2),
                })

    return submission_results


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
    ]
    # Forward per-job extra_env (added 2026-05-20): bacalhau supports --env
    # KEY=VALUE repeatedly. Quote values defensively so spaces/JSON payloads
    # survive Bacalhau's CLI parsing.
    if job.extra_env:
        for k, v in job.extra_env.items():
            if k and v is not None:
                submit_cmd.extend(["--env", f"{k}={v}"])
    # Image MUST come last (positional arg)
    submit_cmd.append(image)

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
    # Forward per-job extra_env (added 2026-05-20): CircleCI pipeline params
    # must all be declared in .circleci/config.yml. To keep this dispatcher
    # backwards-compatible we pass a single JSON-encoded "extra_env_json"
    # parameter. The pipeline config decodes it via jq/python and exports
    # each key as an env var before running the backtest job.
    if job.extra_env:
        try:
            payload["parameters"]["extra_env_json"] = json.dumps(job.extra_env)
        except (TypeError, ValueError) as _exc:
            log.warning("Could not JSON-encode extra_env for %s: %s — omitting",
                        job, _exc)

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
    # Forward per-job extra_env (added 2026-05-20): Drone's "params" dict is
    # exposed as env vars in pipeline steps via environment: blocks, so we
    # can directly inject extra_env keys.
    if job.extra_env:
        for k, v in job.extra_env.items():
            if k and v is not None:
                build_params[str(k)] = str(v)

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

# 2026-05-17: operator mandate "Mac CAN be used even when maxed out" — caps
# raised so Mac stays a viable destination once aggregated remote usage hits
# 90% (see CLOUD_FIRST_MIN_USAGE_PCT). Previous v1 defaults were 60/70/8/4 —
# kept in commented form for reference / quick rollback. cloud_usage.json
# still overrides per-cloud, so operators can tighten via config without
# code change.
MAC_CPU_CAP      = 95.0   # %  (was 60 — raised per cloud-first-90 mandate)
MAC_MEM_CAP      = 95.0   # %  (was 70 — raised per cloud-first-90 mandate)
MAC_LOAD_CAP     = 20.0   # load-avg-1m (was 8 — Mac mini can handle higher)
MAC_MAX_WORKERS  = 8      # (was 4 — doubled per operator mandate)

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
    # Canonical output dir = RESULTS_DIR / <ticker> / <strategy>; backtest_xgb_v10
    # requires --output-dir / --out-dir (one-of, required) and writes result.json
    # there. Prior to 2026-05-21 this arg was omitted and every mac_local job
    # died at argparse with "one of the arguments --output-dir --out-dir is
    # required", producing 87k+ completion_poll_timeout failures (1h each).
    # Fix: derive from RESULTS_DIR + job.ticker + job.strategy and mkdir before
    # spawn so the worker can write result.json into the path the completion
    # poller checks (Job.result_file()).
    output_dir = RESULTS_DIR / job.ticker / job.strategy
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python,
        str(PROJECT_ROOT / job.script),
        "--ticker",   job.ticker,
        "--strategy", job.strategy,
        "--job-id",   job.job_id,
        "--output-dir", str(output_dir),
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

    # Forward per-job extra_env (added 2026-05-20): start from the current
    # process env (so PATH/PYTHONPATH/HOME stay sane) then overlay extra_env
    # so per-job overrides win.
    worker_env = os.environ.copy()
    if job.extra_env:
        for k, v in job.extra_env.items():
            if k and v is not None:
                worker_env[str(k)] = str(v)
        log.info("mac_local extra_env applied (%d keys): %s",
                 len(job.extra_env), list(job.extra_env.keys()))

    try:
        with open(log_path, "wb") as lf:
            proc = subprocess.Popen(
                cmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
                env=worker_env,
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
# Adapter: Northflank — Phase 5 (enabled=false until user signup)
# ---------------------------------------------------------------------------
def _submit_northflank(
    job: Job,
    dry_run: bool = False,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Trigger a Northflank job run via the Northflank REST API v1.

    Northflank free tier provides 2 free jobs (no credit card required).
    Sign up with GitHub at https://northflank.com to obtain credentials.

    The adapter POSTs to:
        https://api.northflank.com/v1/projects/{project_id}/jobs/{job_id}/runs

    with a JSON body that passes TICKER and STRATEGY as environment-variable
    overrides for the run.  The Northflank job must already be created in the
    UI or via the API before this adapter can trigger runs against it.  See
    ``scripts/northflank_job.json`` for the job-spec shape.

    Northflank returns a run object containing an ``id`` field (the run ID),
    which is used as the external reference for tracking.

    Dry-run and token-absent paths:
        - If ``dry_run=True``:  logs the would-be request and returns immediately
          with ``dry_run=True`` in the receipt — no network call is made.
        - If ``NORTHFLANK_API_TOKEN`` is absent: logs a warning and falls back
          to dry-run mode automatically (same safe no-op behaviour).

    Args:
        job:     Job dataclass containing ticker, strategy, script, job_id.
        dry_run: If True, log the request and return without calling the API.
        cfg:     cloud_usage.json block for northflank (injected by submit_job).
                 Supplies env-var name overrides and project/job ID defaults.

    Returns:
        Receipt dict with keys: job_id, cloud, submitted_at, northflank_run_id.
        dry_run=True is added when no real API call was made.

    Required env vars (names are configurable in cloud_usage.json):
        NORTHFLANK_API_TOKEN  — API token from Northflank UI (Project > API tokens)
        NORTHFLANK_PROJECT_ID — Northflank project identifier
        NORTHFLANK_JOB_ID     — Northflank job name to trigger
    """
    import urllib.request
    import urllib.error

    cfg = cfg or {}

    # Resolve env-var names from config so operators can override without code changes
    token_env      = cfg.get("northflank_token_env",      "NORTHFLANK_API_TOKEN")
    project_id_env = cfg.get("northflank_project_env",    "NORTHFLANK_PROJECT_ID")
    job_id_env     = cfg.get("northflank_job_id_env",     "NORTHFLANK_JOB_ID")

    token      = os.environ.get(token_env, "")
    project_id = os.environ.get(project_id_env, cfg.get("northflank_project_id", ""))
    nf_job_id  = os.environ.get(job_id_env,     cfg.get("northflank_job_id", ""))

    url = (
        f"https://api.northflank.com/v1/projects/{project_id}"
        f"/jobs/{nf_job_id}/runs"
    )

    # Environment-variable overrides injected into the job run
    payload: Dict[str, Any] = {
        "overrides": {
            "runtimeEnvironment": {
                "TICKER":    job.ticker,
                "STRATEGY":  job.strategy,
                "SCRIPT":    job.script,
                "JOB_ID":    job.job_id,
            }
        }
    }
    # Forward per-job extra_env (added 2026-05-20): Northflank
    # runtimeEnvironment is a flat {KEY: VALUE} map exposed as container env.
    if job.extra_env:
        for k, v in job.extra_env.items():
            if k and v is not None:
                payload["overrides"]["runtimeEnvironment"][str(k)] = str(v)

    # ------------------------------------------------------------------
    # Dry-run and token-absent paths — no network call in either case
    # ------------------------------------------------------------------
    if dry_run or not token:
        if not token:
            log.warning(
                "northflank: %s env var not set — adapter running in dry-run mode "
                "(no network call). Sign up at https://northflank.com to enable.",
                token_env,
            )
        else:
            log.info(
                "[DRY-RUN] northflank: would POST %s  payload=%s",
                url, payload,
            )
        ts = int(time.time())
        return {
            "job_id":             job.job_id,
            "cloud":              "northflank",
            "submitted_at":       _now_iso(),
            "northflank_run_id":  f"nf-dry-{ts}",
            "dry_run":            True,
        }

    # ------------------------------------------------------------------
    # Live path — POST to Northflank API
    # ------------------------------------------------------------------
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode(errors="replace")
        log.error(
            "northflank dispatch failed for %s: HTTP %s — %s",
            job, exc.code, error_body,
        )
        raise
    except Exception as exc:
        log.error("northflank dispatch failed for %s: %s", job, exc)
        raise

    # Northflank /runs response: {"data": {"id": "<run-id>", ...}}
    run_id = (
        body.get("data", {}).get("id")
        or body.get("id")
        or job.job_id
    )
    log.info(
        "Northflank run triggered: northflank_run_id=%s  project=%s  job=%s  %s",
        run_id, project_id, nf_job_id, job,
    )

    return {
        "job_id":             job.job_id,
        "cloud":              "northflank",
        "submitted_at":       _now_iso(),
        "northflank_run_id":  run_id,
        "northflank_url":     (
            f"https://app.northflank.com/s/project/{project_id}"
            f"/jobs/{nf_job_id}/runs"
        ),
    }


# ---------------------------------------------------------------------------
# Adapter: IBM Code Engine — Phase 2 (stub, enabled=false until user signup)
# ---------------------------------------------------------------------------
def _submit_ibm_code_engine(
    job: Job,
    dry_run: bool = False,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Submit a batch job to IBM Code Engine via the IBM CE Batch Jobs REST API.

    IBM Code Engine Lite plan provides 100,000 vCPU-seconds/month and
    200,000 GiB-seconds/month at no cost, with no credit card required.
    Sign up at https://cloud.ibm.com/codeengine/overview.

    The adapter:
        1. Exchanges the IBM Cloud API key for a short-lived IAM access token
           via ``https://iam.cloud.ibm.com/identity/token``.
        2. POSTs a batch-job-run to the Code Engine Jobs API:
               POST /v2/projects/{project_id}/job_runs
        3. Returns the job_run name from the response body.

    Result convention:
        The CE job writes ``backtests/{ticker}/{strategy}/result.json`` and
        uploads it to a Cloud Object Storage bucket (or commits it back via gh).
        The dispatcher's standard ``poll_result()`` detects the file once it
        arrives in the expected path.

    Required env vars (names configurable in cloud_usage.json):
        IBMCLOUD_API_KEY    — IBM Cloud API key (IAM > API keys)
        IBM_CE_PROJECT_ID   — IBM Code Engine project ID (from CE console)

    Config keys read from cloud_usage.json ``ibm_code_engine`` block:
        ibmcloud_api_key_env  — env-var holding the API key (default: IBMCLOUD_API_KEY)
        ibm_ce_project_env    — env-var holding the project ID (default: IBM_CE_PROJECT_ID)
        ibm_ce_region         — IBM Cloud region slug (default: "us-south")
        ibm_ce_job_name       — CE job definition name (default: "sp500-backtest")
        image                 — Container image ref (default: placeholder)

    Dry-run and token-absent paths:
        - If ``dry_run=True`` or the API key env var is absent: logs the
          would-be request and returns immediately with ``dry_run=True`` — no
          network call is made.

    Args:
        job:     Job dataclass with ticker, strategy, script, job_id.
        dry_run: If True, simulate without network calls.
        cfg:     cloud_usage.json block for ibm_code_engine (injected by submit_job).

    Returns:
        Receipt dict with keys: job_id, cloud, submitted_at, ibm_job_run_name.
    """
    import urllib.request
    import urllib.error
    import urllib.parse

    cfg = cfg or {}

    api_key_env = cfg.get("ibmcloud_api_key_env", "IBMCLOUD_API_KEY")
    project_env = cfg.get("ibm_ce_project_env",   "IBM_CE_PROJECT_ID")
    region      = cfg.get("ibm_ce_region",         "us-south")
    job_name    = cfg.get("ibm_ce_job_name",       "sp500-backtest")
    image       = cfg.get("image", "icr.io/PLACEHOLDER/sp500-backtest:latest")

    api_key    = os.environ.get(api_key_env, "")
    project_id = os.environ.get(project_env, cfg.get("ibm_ce_project_id", ""))

    ce_base_url = (
        f"https://api.{region}.codeengine.cloud.ibm.com/v2/projects"
        f"/{project_id}"
    )
    job_run_url = f"{ce_base_url}/job_runs"

    if dry_run or not api_key:
        if not api_key:
            log.warning(
                "ibm_code_engine: %s env var not set — adapter running in "
                "dry-run mode. Sign up at https://cloud.ibm.com/codeengine/overview",
                api_key_env,
            )
        else:
            log.info(
                "[DRY-RUN] ibm_code_engine: would POST %s  job=%s  ticker=%s",
                job_run_url, job_name, job.ticker,
            )
        ts = int(time.time())
        return {
            "job_id":           job.job_id,
            "cloud":            "ibm_code_engine",
            "submitted_at":     _now_iso(),
            "ibm_job_run_name": f"ibm-dry-{ts}",
            "dry_run":          True,
        }

    # Step 1: Exchange API key for IAM access token
    iam_url  = "https://iam.cloud.ibm.com/identity/token"
    iam_data = urllib.parse.urlencode({
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": api_key,
    }).encode()
    iam_req = urllib.request.Request(
        iam_url,
        data=iam_data,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(iam_req, timeout=20) as resp:
            iam_body  = json.loads(resp.read().decode())
            iam_token = iam_body.get("access_token", "")
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode(errors="replace")
        log.error(
            "ibm_code_engine: IAM token exchange failed HTTP %d — %s",
            exc.code, err_body,
        )
        raise
    except Exception as exc:
        log.error("ibm_code_engine: IAM token exchange error: %s", exc)
        raise

    if not iam_token:
        raise RuntimeError("ibm_code_engine: IAM token exchange returned empty token")

    # Step 2: Submit batch job run
    payload: Dict[str, Any] = {
        "job_name": job_name,
        "run_env_variables": [
            {"type": "literal", "name": "TICKER",   "value": job.ticker},
            {"type": "literal", "name": "STRATEGY", "value": job.strategy},
            {"type": "literal", "name": "SCRIPT",   "value": job.script},
            {"type": "literal", "name": "JOB_ID",   "value": job.job_id},
        ],
    }
    # Forward per-job extra_env (added 2026-05-20): IBM CE run_env_variables
    # is an array of {type, name, value} objects.
    if job.extra_env:
        for k, v in job.extra_env.items():
            if k and v is not None:
                payload["run_env_variables"].append({
                    "type": "literal",
                    "name": str(k),
                    "value": str(v),
                })

    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        job_run_url,
        data=data,
        headers={
            "Authorization": f"Bearer {iam_token}",
            "Content-Type":  "application/json",
            "Accept":        "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode(errors="replace")
        log.error(
            "ibm_code_engine: job_run POST failed HTTP %d — %s",
            exc.code, err_body,
        )
        raise
    except Exception as exc:
        log.error("ibm_code_engine: job_run POST error for %s: %s", job, exc)
        raise

    run_name = body.get("name", job.job_id)
    log.info(
        "IBM Code Engine job run submitted: name=%s  project=%s  %s",
        run_name, project_id, job,
    )

    return {
        "job_id":           job.job_id,
        "cloud":            "ibm_code_engine",
        "submitted_at":     _now_iso(),
        "ibm_job_run_name": run_name,
        "ibm_ce_url": (
            f"https://cloud.ibm.com/codeengine/projects/{project_id}/jobs"
        ),
    }


# ---------------------------------------------------------------------------
# Adapter: Firebase Functions (Cloud Functions Gen2 HTTPS) — Phase 2
# (stub, enabled=false until user signup)
# ---------------------------------------------------------------------------
def _submit_firebase_functions(
    job: Job,
    dry_run: bool = False,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Dispatch a backtest job to a Google Cloud Functions Gen2 (Firebase Functions)
    HTTPS endpoint via a Bearer-authenticated HTTP POST.

    Firebase Functions free tier (Spark plan) provides 2,000,000 invocations/month
    at no cost.  Sign up at https://firebase.google.com and enable Cloud Functions
    in the Firebase console.

    The HTTPS function receives a JSON body with ticker, strategy, script, and
    job_id.  It runs the backtest (or enqueues it as a Cloud Task / Pub/Sub
    message) and writes ``backtests/{ticker}/{strategy}/result.json`` to a Cloud
    Storage bucket or commits it back to the repo.

    Authentication:
        The adapter attaches a Firebase ID token (or a GCP identity token) in the
        ``Authorization: Bearer`` header.  The token must be provided via the
        ``FIREBASE_ID_TOKEN`` env var (typically a short-lived OIDC token issued by
        ``gcloud auth print-identity-token`` or the Firebase Admin SDK).

    Endpoint URL pattern:
        https://{region}-{project_id}.cloudfunctions.net/{function_name}
    or (Gen2 / Cloud Run backed):
        https://{function_name}-<hash>-{region}.a.run.app

    The full URL is read from the config's ``endpoint_url`` field or constructed
    from ``project_id`` + ``region`` + ``function_name``.

    Required env vars (names configurable in cloud_usage.json):
        FIREBASE_PROJECT_ID   — GCP/Firebase project ID
        FIREBASE_ID_TOKEN     — short-lived OIDC/identity token for the invoker SA

    Config keys read from cloud_usage.json ``firebase_functions`` block:
        firebase_project_env  — env-var holding project ID (default: FIREBASE_PROJECT_ID)
        firebase_token_env    — env-var holding ID token (default: FIREBASE_ID_TOKEN)
        firebase_region       — GCP region (default: "us-central1")
        firebase_function     — Cloud Function name (default: "run_backtest")
        endpoint_url          — full override URL (skips URL construction if set)

    Dry-run and token-absent paths:
        - If ``dry_run=True`` or ``FIREBASE_ID_TOKEN`` is absent: logs the
          would-be request and returns immediately with ``dry_run=True``.

    Args:
        job:     Job dataclass with ticker, strategy, script, job_id.
        dry_run: If True, simulate without network calls.
        cfg:     cloud_usage.json block for firebase_functions (injected by submit_job).

    Returns:
        Receipt dict with keys: job_id, cloud, submitted_at, firebase_invocation_id.
    """
    import urllib.request
    import urllib.error

    cfg = cfg or {}

    project_env  = cfg.get("firebase_project_env", "FIREBASE_PROJECT_ID")
    token_env    = cfg.get("firebase_token_env",   "FIREBASE_ID_TOKEN")
    region       = cfg.get("firebase_region",      "us-central1")
    func_name    = cfg.get("firebase_function",    "run_backtest")

    project_id = os.environ.get(project_env, cfg.get("firebase_project_id", ""))
    id_token   = os.environ.get(token_env, "")

    # Resolve endpoint URL: use explicit override if provided, else construct
    endpoint_url = cfg.get("endpoint_url") or (
        f"https://{region}-{project_id}.cloudfunctions.net/{func_name}"
    )

    payload: Dict[str, Any] = {
        "ticker":   job.ticker,
        "strategy": job.strategy,
        "script":   job.script,
        "job_id":   job.job_id,
    }
    # Forward per-job extra_env (added 2026-05-20): Firebase Function receives
    # a JSON body — embed extra_env as a nested object so the function code
    # can iterate and merge into the worker subprocess env.
    if job.extra_env:
        payload["extra_env"] = {str(k): str(v) for k, v in job.extra_env.items()
                                if k and v is not None}

    if dry_run or not id_token:
        if not id_token:
            log.warning(
                "firebase_functions: %s env var not set — adapter running in "
                "dry-run mode. Obtain a token via: "
                "gcloud auth print-identity-token --audiences=%s",
                token_env, endpoint_url,
            )
        else:
            log.info(
                "[DRY-RUN] firebase_functions: would POST %s  payload=%s",
                endpoint_url, payload,
            )
        ts = int(time.time())
        return {
            "job_id":                  job.job_id,
            "cloud":                   "firebase_functions",
            "submitted_at":            _now_iso(),
            "firebase_invocation_id":  f"fb-dry-{ts}",
            "dry_run":                 True,
        }

    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        endpoint_url,
        data=data,
        headers={
            "Authorization": f"Bearer {id_token}",
            "Content-Type":  "application/json",
            "Accept":        "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            http_status = resp.status
            raw_body    = resp.read().decode(errors="replace")
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode(errors="replace")
        log.error(
            "firebase_functions: POST failed HTTP %d — %s",
            exc.code, err_body,
        )
        raise
    except Exception as exc:
        log.error("firebase_functions: POST error for %s: %s", job, exc)
        raise

    # Cloud Functions return a plain text or JSON body; parse if JSON, else treat
    # the HTTP 2xx itself as confirmation.
    invocation_id: str = job.job_id
    try:
        body = json.loads(raw_body)
        invocation_id = body.get("invocation_id") or body.get("id") or job.job_id
    except (json.JSONDecodeError, AttributeError):
        # Non-JSON response body is fine for 2xx responses
        pass

    log.info(
        "Firebase Function invoked: HTTP %d  invocation_id=%s  endpoint=%s  %s",
        http_status, invocation_id, endpoint_url, job,
    )

    return {
        "job_id":                  job.job_id,
        "cloud":                   "firebase_functions",
        "submitted_at":            _now_iso(),
        "firebase_invocation_id":  invocation_id,
        "firebase_endpoint":       endpoint_url,
    }


# ---------------------------------------------------------------------------
# Codespaces adapter (alt-GPU backup route — added 2026-05-19)
# ---------------------------------------------------------------------------
def _submit_codespaces(job: Job, dry_run: bool = False) -> Dict[str, Any]:
    """Dispatch one job onto a GitHub Codespace via the standalone adapter.

    Wraps scripts/dispatcher_adapter_codespaces.submit_job so the codespaces
    adapter participates in the multi-cloud routing fabric. Uses the same
    job.ticker / job.strategy + extra_env contract as other adapters.

    Free quota: 120 core-hours/month on 2-core (or 30 hrs on 4-core). Sized
    to handle xsec/full500 backups when gh_actions hits its 2000-min cap.
    """
    try:
        from dispatcher_adapter_codespaces import submit_job as _cs_submit
    except ImportError as exc:
        log.error("codespaces adapter import failed: %s", exc)
        return {"job_id": None, "status": "submit_error",
                "error": f"adapter import failed: {exc}",
                "error_type": "config_error"}

    # Build job_spec from job.extra_env (the run_sweep.py compute repo accepts
    # arbitrary flags forwarded verbatim).
    job_spec: Dict[str, Any] = {}
    if job.extra_env:
        job_spec.update({k: v for k, v in job.extra_env.items() if v is not None})

    receipt = _cs_submit(job.ticker, job.strategy, job_spec, dry_run=dry_run)
    receipt.setdefault("cloud", "codespaces")
    receipt.setdefault("submitted_at", _now_iso())
    # Normalize keys to multi-cloud convention
    if "job_id" not in receipt and "name" in receipt:
        receipt["job_id"] = receipt["name"]
    return receipt


# ---------------------------------------------------------------------------
# Adapter router
# ---------------------------------------------------------------------------
ADAPTERS = {
    "github_actions":      _submit_github_actions,
    "modal":               _submit_modal,
    "oracle_a1":           _submit_oracle_a1,
    "gcp_ssh":             _submit_gcp_ssh,
    "aws_ssh":             _submit_aws_ssh,
    "render_api":          _submit_render_api,
    "railway_api":         _submit_railway_api,
    "fly_api":             _submit_fly_api,
    "drone_ci":            _submit_drone_ci,
    "bacalhau":            _submit_bacalhau,
    "circleci_oss":        _submit_circleci_oss,
    "northflank":          _submit_northflank,
    "ibm_code_engine":     _submit_ibm_code_engine,
    "firebase_functions":  _submit_firebase_functions,
    "mac_local":           _submit_mac_local,
    "codespaces":          _submit_codespaces,
}

# Adapters that require their cloud config dict forwarded at call time
_CFG_AWARE_ADAPTERS = {
    "mac_local", "circleci_oss", "bacalhau", "drone_ci",
    "northflank", "ibm_code_engine", "firebase_functions",
}


def submit_job(job: Job, cloud: str, tracker: UsageTracker,
               dry_run: bool = False) -> Dict[str, Any]:
    """
    Dispatch one job to the named cloud adapter.

    Auth-failure fallback (Bug Fix #1):
        If the adapter raises urllib.error.HTTPError with code 401/403, or
        returns a result dict containing ``error_type="auth_failure"``, the
        tracker marks an exponential-backoff cooldown on that cloud and
        re-raises an AuthFailureError so the caller can skip to the next
        cloud without aborting the whole dispatch pass.
    """
    import urllib.error as _urlerr

    adapter = ADAPTERS.get(cloud)
    if adapter is None:
        raise ValueError(f"No adapter registered for cloud: {cloud!r}")

    # Cost/time estimates per job (conservative)
    cost_est_usd = 0.002   # ~2 min × $0.001/min Modal equivalent
    min_est      = 0.5     # ~30 sec per job ≈ 0.5 runner-minutes
    hr_est       = 0.008   # ~30 sec per job ≈ 0.008 hr

    try:
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

        # Check for soft auth-failure signalled via result dict
        if receipt.get("error_type") == "auth_failure":
            tracker.mark_auth_failure(cloud)
            raise AuthFailureError(
                f"Auth failure (soft signal) on {cloud} for {job}"
            )

        # Successful submit — clear any prior auth failure state
        tracker.mark_success(cloud)

    except _urlerr.HTTPError as exc:
        if exc.code in (401, 403):
            tracker.mark_auth_failure(cloud)
            log.warning(
                "AUTH FAILURE (HTTP %d) on %s for %s — cloud placed in cooldown, "
                "skipping to next cloud.",
                exc.code, cloud, job,
            )
            raise AuthFailureError(
                f"HTTP {exc.code} auth failure on {cloud} for {job}"
            ) from exc
        # Non-auth HTTP error — re-raise as-is
        raise

    tracker.register_submit(cloud,
                            cost_estimate=cost_est_usd,
                            minutes_estimate=min_est,
                            hours_estimate=hr_est)
    log.info("Submitted %s → %s (job_id=%s)", job, cloud, receipt["job_id"])
    return receipt


class AuthFailureError(RuntimeError):
    """Raised by submit_job() when an adapter returns a 401/403 or auth_failure signal."""


# ---------------------------------------------------------------------------
# Completion-poll registry and background poller
# ---------------------------------------------------------------------------
# Global registry of submitted jobs awaiting completion.
# Keyed by job_id → (Job, cloud_name, submitted_epoch).
# The background completion thread scans this dict each cycle and calls
# register_complete() when result.json appears OR the job times out.
# Protected by a threading.Lock for safe concurrent access.
# ---------------------------------------------------------------------------
import threading as _threading

_inflight_registry: Dict[str, Tuple["Job", str, float]] = {}
_inflight_lock = _threading.Lock()

# Maximum seconds to wait for a remote result before force-marking as timed out.
# Reads from mac_local.max_job_seconds if present; falls back to this default.
_DEFAULT_MAX_JOB_SECONDS = 3600  # 1 hour


def _register_inflight(job: "Job", cloud: str) -> None:
    """Add a submitted job to the in-flight completion-poll registry.

    Called by :func:`dispatch_pass` immediately after a successful submit.
    The background thread returned by :func:`_start_completion_poller` will
    watch these entries and call ``tracker.register_complete()`` once the job
    reaches a terminal state.

    Args:
        job:   Submitted Job instance.  Must have ``job_id`` and ``result_file()``
               pointing at the canonical result path.
        cloud: Cloud name that accepted the job (e.g. ``"github_actions"``).
    """
    with _inflight_lock:
        _inflight_registry[job.job_id] = (job, cloud, time.time())
    log.debug(
        "Registered in-flight job %s on %s (registry size=%d)",
        job.job_id, cloud, len(_inflight_registry),
    )


def _start_completion_poller(
    tracker: "UsageTracker",
    poll_interval_sec: int = 15,
    max_job_seconds: int = _DEFAULT_MAX_JOB_SECONDS,
) -> _threading.Thread:
    """Start a daemon thread that polls the in-flight registry and calls
    ``tracker.register_complete()`` when each job finishes or times out.

    The thread runs until the process exits.  It sleeps *poll_interval_sec*
    between sweeps so it does not busy-spin.

    Completion detection (in priority order):
        1. ``result.json`` appears at the canonical path returned by
           ``job.result_file()``.  This works for every adapter that writes
           results locally (mac_local) or has a side-channel that deposits
           the file (bacalhau background thread, GitHub Actions commit-back).
        2. Age check: if the job has been in-flight longer than
           *max_job_seconds*, it is treated as failed/timed-out and
           ``register_complete()`` is still called so the slot is freed.

    The thread is intentionally non-reentrant per job_id — once a job_id is
    removed from ``_inflight_registry`` it will never be re-added under the
    same ID.

    Args:
        tracker:          UsageTracker instance shared with the dispatch loop.
        poll_interval_sec: Sleep duration between registry sweeps (seconds).
        max_job_seconds:   Hard timeout after which a job is force-completed.

    Returns:
        The started daemon Thread (callers do not normally need to join it).
    """

    def _poll_loop() -> None:
        log.info(
            "Completion-poller thread started (interval=%ds, timeout=%ds).",
            poll_interval_sec, max_job_seconds,
        )
        while True:
            time.sleep(poll_interval_sec)
            try:
                _sweep_completions(tracker, max_job_seconds)
            except Exception as exc:
                log.error(
                    "Completion-poller: unhandled error in sweep: %s",
                    exc, exc_info=True,
                )

    t = _threading.Thread(
        target=_poll_loop,
        daemon=True,
        name="completion-poller",
    )
    t.start()
    log.info("Completion-poller thread launched (daemon=True).")
    return t


def _fetch_github_actions_runs_by_jobid(
    snapshot: List[Tuple[str, Tuple["Job", str, float]]],
) -> Dict[str, Dict[str, Any]]:
    """Fetch recent github_actions workflow runs and index by inputs.job_id.

    Returns {} when no github_actions jobs are in flight (saves an API call)
    or on any HTTP/JSON failure (caller falls back to local result.json check
    and the max_job_seconds timeout — neither path can mis-attribute a slot).

    The dispatcher injects ``inputs.job_id`` into every workflow_dispatch
    payload, so each run carries that ID via the workflow's
    ``run-name`` / ``inputs.job_id`` field which surfaces in the runs API.
    """
    gh_jobs = [(jid, job) for jid, (job, c, _) in snapshot if c == "github_actions"]
    if not gh_jobs:
        return {}

    cfg = _load_github_actions_cfg()
    owner = os.environ.get("GITHUB_OWNER") or cfg.get("owner")
    repo = os.environ.get("GITHUB_REPO") or cfg.get("repo")
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""
    if not (owner and repo and token):
        return {}

    url = (
        f"https://api.github.com/repos/{owner}/{repo}/actions/runs"
        f"?per_page=100"
    )
    try:
        import urllib.request
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        log.debug("github_actions runs API fetch failed: %s", exc)
        return {}

    runs = data.get("workflow_runs", []) or []
    # Build {job_id: {id, status, conclusion}} by parsing run-name. Workflows
    # commonly format run-name like "sweep AAPL momentum <job_id>" or include
    # job_id in display_title. We scan name + display_title for a substring
    # match against every in-flight job_id. O(N_jobs * N_runs) but both are
    # small (<= 100 runs, <= max_concurrent in_flight jobs).
    out: Dict[str, Dict[str, Any]] = {}
    for run in runs:
        haystack = " ".join(str(run.get(k, "")) for k in (
            "name", "display_title", "head_branch", "head_commit",
        ))
        for jid, _job in gh_jobs:
            if jid in haystack and jid not in out:
                out[jid] = {
                    "id": run.get("id"),
                    "status": run.get("status"),
                    "conclusion": run.get("conclusion"),
                }
    return out


def _fetch_modal_completions(
    snapshot: List[Tuple[str, Tuple["Job", str, float]]],
) -> Dict[str, Dict[str, Any]]:
    """Poll Popen handles for in-flight Modal jobs and report exited ones.

    Mirrors the structural contract of :func:`_fetch_github_actions_runs_by_jobid`
    so :func:`_sweep_completions` can treat both clouds uniformly.

    The Modal adapter (:func:`_submit_modal`) launches the ``modal run`` CLI as
    a non-blocking subprocess and stashes the Popen handle in
    ``_modal_active_procs`` keyed by job_id. When that subprocess exits, the
    local entrypoint has finished writing ``backtests/<ticker>/<strategy>/
    result.json`` to the Drive path (success) OR died early (failure). This
    helper detects that exit transition via ``proc.poll()``.

    Returns ``{job_id: {"returncode": int, "status": "completed",
                        "conclusion": "success"|"failed"}}``
    for jobs whose subprocess has exited. Returns ``{}`` when no Modal jobs
    are in flight, or when none of them have exited yet (caller falls back to
    the result.json existence check and the max_job_seconds timeout — neither
    path can mis-attribute a slot).

    Note: only jobs submitted by *this* dispatcher process appear in
    ``_modal_active_procs``. Jobs orphaned by a previous dispatcher restart
    will fall through to the existing timeout path; this is intentional —
    re-attaching to remote Modal function-call IDs requires refactoring the
    adapter to use ``Function.spawn()`` instead of the ``modal run`` CLI.
    """
# autosolve_skip: blocker #116 modal-cap fallback patch
    modal_jobs = [(jid, job) for jid, (job, c, _) in snapshot if c == "modal"]
    if not modal_jobs:
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for jid, _job in modal_jobs:
        proc = _modal_active_procs.get(jid)
        if proc is None:
            # No Popen handle (orphaned from prior process) — let timeout handle it.
            continue
        rc = proc.poll()
        if rc is None:
            # Still running.
            continue
        # Subprocess has exited — terminal state.
        # Blocker #116 (2026-05-19): inspect stderr for Modal spend-cap /
        # workspace-quota signals. Modal CLI emits these strings when the
        # workspace spend cap is hit (separate from monthly credit quota —
        # see CLAUDE.md auto_signup feedback). Marking the cloud as
        # auth-failed engages the existing exponential cooldown so the
        # dispatcher routes around Modal until it recovers.
        cap_signal = False
        stderr_tail = ""
        try:
            if proc.stderr is not None:
                stderr_tail = (proc.stderr.read() or b"").decode("utf-8", "replace")
        except Exception:
            stderr_tail = ""
        try:
            cap_signal = _modal_cap_signal(stderr_tail, rc)
        except Exception:
            cap_signal = False

        out[jid] = {
            "returncode": rc,
            "status":     "completed",
            "conclusion": "success" if rc == 0 else "failed",
            "cap_signal": cap_signal,
            "stderr_tail": stderr_tail[-512:] if stderr_tail else "",
        }
    return out


# ---------------------------------------------------------------------------
# Blocker #116 — Modal spend-cap fallback (2026-05-19)
# ---------------------------------------------------------------------------
# Detect Modal workspace spend-cap / quota-exceeded signals from the CLI stderr
# tail. When detected, the completion-sweep marks Modal as auth-failed (using
# the existing exponential-cooldown machinery) so the dispatcher re-routes
# new jobs to gh_actions / cerebras / groq / gemini / mac_local.
_MODAL_CAP_TOKENS = (
    "spend cap",
    "spending cap",
    "workspace spend",
    "insufficient credit",
    "quota exceeded",
    "exceeded credit",
    "429",
    "rate limit",
)


def _modal_cap_signal(stderr_text: str, returncode: int) -> bool:
    """True iff Modal stderr tail looks like a spend-cap or quota-exceeded
    failure. Returncode-agnostic (Modal CLI exits 1 for many reasons)."""
    if not stderr_text:
        return False
    s = stderr_text.lower()
    return any(tok in s for tok in _MODAL_CAP_TOKENS)


# ---------------------------------------------------------------------------
# mac_local completion fetcher (2026-05-21)
# ---------------------------------------------------------------------------
def _fetch_mac_completions(
    snapshot: List[Tuple[str, Tuple["Job", str, float]]],
) -> Dict[str, Dict[str, Any]]:
    """Poll Popen handles for in-flight mac_local jobs and report exited ones.

    Mirrors :func:`_fetch_modal_completions`. The mac_local adapter
    (:func:`_submit_mac_local`) launches the backtest worker as a non-blocking
    subprocess and stashes the Popen handle in ``_mac_active_procs`` keyed by
    job_id. When the subprocess exits, the worker has either written
    ``result.json`` (success) OR crashed early (failure -- e.g. argparse error,
    missing data, OOM, segfault). This helper detects that exit transition via
    ``proc.poll()`` so the completion sweep can decrement the in-flight counter
    in seconds rather than waiting on the 1-hour ``max_job_seconds`` watchdog.

    Returns ``{job_id: {"returncode": int, "status": "completed",
                        "conclusion": "success"|"failed"}}``
    for jobs whose subprocess has exited. Returns ``{}`` when no mac_local jobs
    are in flight, or when none of them have exited yet (caller falls back to
    the result.json existence check and the max_job_seconds timeout).

    Note: only jobs submitted by *this* dispatcher process appear in
    ``_mac_active_procs``. Jobs orphaned by a previous dispatcher restart will
    fall through to the existing result.json / timeout path.

    Rationale (2026-05-21): pre-fix mac_local jobs all died at argparse with
    "one of the arguments --output-dir --out-dir is required" within ~0.5s of
    spawn, but the dispatcher waited 1h before timing them out, producing
    96k+ ``completion_poll_timeout`` failures. With this fetcher wired in,
    crashed workers are detected within one sweep tick (default ~30s) and the
    slot is freed immediately so the dispatcher can queue the next job.
    """
    mac_jobs = [(jid, job) for jid, (job, c, _) in snapshot if c == "mac_local"]
    if not mac_jobs:
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for jid, _job in mac_jobs:
        proc = _mac_active_procs.get(jid)
        if proc is None:
            # No Popen handle (orphaned from prior process) -- let result.json
            # or timeout handle it.
            continue
        rc = proc.poll()
        if rc is None:
            # Still running.
            continue
        # Subprocess has exited -- terminal state.
        out[jid] = {
            "returncode": rc,
            "status":     "completed",
            "conclusion": "success" if rc == 0 else "failed",
        }
    return out


def _log_modal_cap_fallback(job_id: str, stderr_tail: str) -> None:
    """Append a JSONL row to logs/cloud_dispatch/modal_cap_fallback.jsonl
    so the operator can see when Modal capped and the dispatcher rerouted."""
    import json as _json, time as _time
    p = "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/logs/cloud_dispatch/modal_cap_fallback.jsonl"
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "a") as fh:
            fh.write(_json.dumps({
                "ts": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
                "job_id": job_id,
                "stderr_tail": (stderr_tail or "")[-512:],
                "action": "marked_auth_failure_cooldown",
            }) + "\n")
    except Exception:
        pass


def _sweep_completions(
    tracker: "UsageTracker",
    max_job_seconds: int = _DEFAULT_MAX_JOB_SECONDS,
) -> None:
    """One sweep of the in-flight registry.

    Called by the completion-poller thread every *poll_interval_sec* seconds.
    Also callable directly (e.g., from tests or the drift-correction pass).

    For each in-flight job:
      - If ``result.json`` exists → mark complete, write status event, decrement.
      - If age > max_job_seconds → treat as timed-out, write failed event, decrement.
      - Otherwise → leave in registry and check again next cycle.

    Args:
        tracker:         UsageTracker holding per-cloud in-flight counters.
        max_job_seconds: Hard timeout before a job is force-completed.
    """
    with _inflight_lock:
        snapshot = list(_inflight_registry.items())

    completed_ids: List[str] = []

    # Pre-fetch a single GitHub Actions runs snapshot (one call serves all
    # github_actions in-flight jobs in this sweep). Indexed by inputs.job_id
    # which the dispatcher injects into the workflow_dispatch payload.
    gh_runs_by_jobid = _fetch_github_actions_runs_by_jobid(snapshot)

    # Pre-fetch Popen-exit snapshot for all in-flight Modal jobs (one pass).
    # Indexed by job_id; only present for subprocesses that have exited.
    modal_exits_by_jobid = _fetch_modal_completions(snapshot)

    # Pre-fetch Popen-exit snapshot for all in-flight mac_local jobs (2026-05-21).
    # Mirror of modal_exits_by_jobid -- detects exited mac_local subprocesses
    # within one sweep tick rather than waiting on max_job_seconds (1h).
    mac_exits_by_jobid = _fetch_mac_completions(snapshot)

    for job_id, (job, cloud, submitted_at_epoch) in snapshot:
        age_sec = time.time() - submitted_at_epoch
        result_path = job.result_file()

        # GitHub Actions cloud jobs: check upstream run state via API since
        # remote workers can't write to the Mac filesystem. Fix for
        # in_flight_jobs pinning at max_concurrent_jobs (2026-05-17).
        gh_state = gh_runs_by_jobid.get(job_id) if cloud == "github_actions" else None
        if gh_state and gh_state.get("status") == "completed":
            conclusion = gh_state.get("conclusion") or "unknown"
            terminal = (conclusion == "success")
            job.status = "completed" if terminal else "failed"
            job.completed_at = _now_iso()
            _write_status_event(
                job, "complete" if terminal else "failed", cloud=cloud,
                extra={"gh_run_id": gh_state.get("id"),
                       "gh_conclusion": conclusion,
                       "source": "github_actions_api_poll"},
            )
            tracker.load()
            tracker.register_complete(cloud)
            log.info(
                "register_complete (gh_actions API): job_id=%s run_id=%s "
                "conclusion=%s age=%.1fs",
                job_id, gh_state.get("id"), conclusion, age_sec,
            )
            completed_ids.append(job_id)
            continue

        # Modal cloud jobs: check Popen exit state since the `modal run`
        # subprocess returning is the local-equivalent of result.json arrival
        # (the local_entrypoint in modal_worker.py::main writes result.json
        # to the canonical Drive path on success). Fix for modal.in_flight_jobs
        # pinning at max_concurrent_containers (2026-05-17, mirroring the
        # gh_actions completion-poll pattern above).
        # autosolve_skip: blocker #116 modal-cap fallback wiring
        modal_state = modal_exits_by_jobid.get(job_id) if cloud == "modal" else None
        if modal_state:
            conclusion = modal_state.get("conclusion") or "unknown"
            terminal = (conclusion == "success")
            # Blocker #116 (2026-05-19): if Modal CLI stderr indicates a
            # workspace spend-cap / quota exceeded, engage the existing
            # auth-failure cooldown so the dispatcher reroutes to
            # gh_actions / cerebras / groq / gemini / mac_local.
            if modal_state.get("cap_signal"):
                try:
                    tracker.mark_auth_failure("modal")
                    _log_modal_cap_fallback(job_id, modal_state.get("stderr_tail", ""))
                    log.warning(
                        "MODAL CAP SIGNAL on job_id=%s — cooldown engaged; "
                        "dispatcher will reroute future jobs around modal.",
                        job_id,
                    )
                except Exception as exc:
                    log.warning("modal cap-signal handler failed: %s", exc)
            job.status = "completed" if terminal else "failed"
            job.completed_at = _now_iso()
            # Best-effort: capture result_path if the local entrypoint wrote
            # the file (success path); leave None on failure.
            rp_str: Optional[str] = None
            try:
                if result_path.exists():
                    job.result_path = result_path
                    rp_str = str(result_path)
            except Exception:
                pass
            _write_status_event(
                job, "complete" if terminal else "failed", cloud=cloud,
                result_path=rp_str,
                extra={"modal_returncode": modal_state.get("returncode"),
                       "modal_conclusion": conclusion,
                       "source": "modal_popen_poll"},
            )
            tracker.load()
            tracker.register_complete(cloud)
            # Drop the Popen handle so the dict doesn't grow unboundedly.
            _modal_active_procs.pop(job_id, None)
            log.info(
                "register_complete (modal Popen): job_id=%s rc=%s "
                "conclusion=%s age=%.1fs result=%s",
                job_id, modal_state.get("returncode"), conclusion, age_sec, rp_str,
            )
            completed_ids.append(job_id)
            continue

        # mac_local cloud jobs: check Popen exit state (2026-05-21).
        # Same pattern as modal/gh_actions above. Detects worker crashes
        # (argparse error, OOM, segfault) within one sweep tick instead of
        # waiting on the 1-hour max_job_seconds watchdog -- prior to this,
        # 96k+ mac_local jobs accumulated as completion_poll_timeout failures.
        mac_state = mac_exits_by_jobid.get(job_id) if cloud == "mac_local" else None
        if mac_state:
            conclusion = mac_state.get("conclusion") or "unknown"
            terminal = (conclusion == "success")
            job.status = "completed" if terminal else "failed"
            job.completed_at = _now_iso()
            # Best-effort: capture result_path if the worker wrote the file.
            rp_str: Optional[str] = None
            try:
                if result_path.exists():
                    job.result_path = result_path
                    rp_str = str(result_path)
            except Exception:
                pass
            _write_status_event(
                job, "complete" if terminal else "failed", cloud=cloud,
                result_path=rp_str,
                extra={"mac_returncode": mac_state.get("returncode"),
                       "mac_conclusion": conclusion,
                       "source": "mac_local_popen_poll"},
            )
            tracker.load()
            tracker.register_complete(cloud)
            # Drop the Popen handle so the dict doesn't grow unboundedly.
            _mac_active_procs.pop(job_id, None)
            log.info(
                "register_complete (mac_local Popen): job_id=%s rc=%s "
                "conclusion=%s age=%.1fs result=%s",
                job_id, mac_state.get("returncode"), conclusion, age_sec, rp_str,
            )
            completed_ids.append(job_id)
            continue

        if result_path.exists():
            # Success: result file landed
            job.status       = "completed"
            job.completed_at = _now_iso()
            job.result_path  = result_path
            _write_status_event(
                job, "complete", cloud=cloud,
                result_path=str(result_path),
            )
            tracker.load()          # re-read in case another process updated
            tracker.register_complete(cloud)
            log.info(
                "register_complete: job_id=%s cloud=%s age=%.1fs result=%s",
                job_id, cloud, age_sec, result_path,
            )
            completed_ids.append(job_id)

        elif age_sec > max_job_seconds:
            # Timeout: job took too long — free the slot anyway
            job.status = "failed"
            _write_status_event(
                job, "failed", cloud=cloud,
                extra={"reason": "completion_poll_timeout",
                       "age_sec": int(age_sec)},
            )
            tracker.load()
            tracker.register_complete(cloud)
            log.warning(
                "register_complete (timeout): job_id=%s cloud=%s age=%.1fs "
                "> max_job_seconds=%d — slot freed.",
                job_id, cloud, age_sec, max_job_seconds,
            )
            completed_ids.append(job_id)

    if completed_ids:
        with _inflight_lock:
            for jid in completed_ids:
                _inflight_registry.pop(jid, None)
        log.info(
            "Completion sweep: resolved %d jobs, %d still in flight.",
            len(completed_ids),
            len(_inflight_registry),
        )


# ---------------------------------------------------------------------------
# Queue loading
# ---------------------------------------------------------------------------
def _load_extra_env_map(status_file: Path = STATUS_FILE) -> Dict[str, Dict[str, str]]:
    """Build a {job_id -> extra_env dict} map from the dispatched.jsonl ledger.

    Added 2026-05-20 to plumb per-job env overrides (XGB_NO_TOPK,
    INTERACTION_CONSTRAINTS, MONOTONIC_CONSTRAINTS, etc.) from
    cloud_dispatch.enqueue_job(..., extra_env={...}) into every adapter's
    worker invocation.  Before this helper, ``extra_env`` was stored in the
    ledger record but never read back into the Job object, so adapters had no
    way to forward it.

    The ledger is append-only: each status transition writes a new line for
    the same job_id. The ENQUEUE row (status="queued") always carries the
    extra_env dict; later rows may omit it.  We therefore keep the FIRST seen
    non-empty extra_env per job_id (the enqueue row) and ignore subsequent
    rows that might overwrite it with {}.

    Acquires LOCK_SH on the ledger so the read is consistent w.r.t. the
    dispatcher's own write-back step.

    Returns:
        Dict mapping job_id -> extra_env dict.  Jobs without an entry get {}.
    """
    out: Dict[str, Dict[str, str]] = {}
    if not status_file.exists():
        return out
    try:
        with open(status_file, "r", encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_SH)
            try:
                lines = fh.readlines()
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    except OSError as exc:
        log.warning("Could not read dispatched.jsonl for extra_env map: %s", exc)
        return out

    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError:
            continue
        jid = rec.get("id")
        if not jid:
            continue
        env = rec.get("extra_env")
        if not isinstance(env, dict) or not env:
            continue
        # Keep first non-empty extra_env (typically the enqueue row); later
        # rows may legitimately omit it during status transitions.
        if jid not in out:
            # Coerce all values to str (env vars are strings) and skip
            # non-stringifiable / None values silently.
            clean = {}
            for k, v in env.items():
                if v is None:
                    continue
                try:
                    clean[str(k)] = str(v)
                except Exception:
                    continue
            out[jid] = clean
    return out


def load_pending_jobs(queue_file: Path) -> List[Job]:
    """Read queue.txt and return all non-comment, non-empty lines as Job objects.

    Acquires a shared (LOCK_SH) flock while reading so the file cannot be
    truncated by the dispatcher's own write-back step mid-read.  This is
    intentionally a *shared* lock — multiple read-only consumers can proceed
    concurrently; only the write-back step (``_truncate_dispatched_from_queue``)
    holds an exclusive lock.

    Each Job is enriched with its ``extra_env`` dict from dispatched.jsonl so
    adapters can forward per-job env overrides to worker processes (added
    2026-05-20 — see :func:`_load_extra_env_map`).
    """
    jobs: List[Job] = []
    if not queue_file.exists():
        log.warning("Queue file not found: %s", queue_file)
        return jobs
    with open(queue_file, "r", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_SH)
        try:
            lines = fh.readlines()
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    for lineno, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            jobs.append(Job.from_line(line))
        except ValueError as exc:
            log.warning("Line %d skipped: %s", lineno, exc)

    # Enrich with extra_env from dispatched.jsonl (added 2026-05-20).
    # Single ledger read per load_pending_jobs() call regardless of job count.
    if jobs:
        env_map = _load_extra_env_map(STATUS_FILE)
        enriched = 0
        for j in jobs:
            env = env_map.get(j.job_id)
            if env:
                j.extra_env = env
                enriched += 1
        if enriched:
            log.info("Enriched %d/%d jobs with extra_env from dispatched.jsonl",
                     enriched, len(jobs))

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

    total_submitted = sum(submitted.values())
    log.info("=== SIMULATION RESULTS ===")
    log.info("Total submitted : %d / %d", total_submitted, n_jobs)
    log.info("Blocked         : %d", len(blocked))
    log.info("")

    # Per-cloud breakdown, sorted by tier then name
    def _cloud_sort_key(c: str) -> Tuple[int, str]:
        tier = tracker.data[c].get("cost_tier", "credit")
        tier_idx = TIER_ORDER.index(tier) if tier in TIER_ORDER else len(TIER_ORDER)
        return (tier_idx, c)

    log.info("  %-22s %-10s %s  %s  %s",
             "CLOUD", "TIER", "submitted", "in_flight", "headroom")
    log.info("  %s", "-" * 70)
    for cloud in sorted(enabled, key=_cloud_sort_key):
        h   = tracker.headroom_pct(cloud)
        cfg = tracker.cloud_cfg(cloud)
        tier = cfg.get("cost_tier", "?")
        tag = " [MAC LAST-RESORT]" if cloud == "mac_local" else ""
        log.info(
            "  %-22s %-10s %9d  %9d  %6.1f%%%s",
            cloud, tier, submitted[cloud],
            cfg.get("in_flight_jobs", 0), h, tag,
        )

    # Tier-level summary
    log.info("")
    log.info("  Tier summary:")
    tier_totals: Dict[str, int] = {t: 0 for t in TIER_ORDER}
    for cloud, count in submitted.items():
        tier = tracker.data[cloud].get("cost_tier", "credit")
        if tier in tier_totals:
            tier_totals[tier] += count
    for tier in TIER_ORDER:
        pct = (tier_totals[tier] / total_submitted * 100) if total_submitted else 0
        log.info("    %-10s %3d jobs  (%5.1f%%)", tier, tier_totals[tier], pct)

    if blocked:
        log.warning("Blocked job indices (1-based): %s", blocked)

    # Safety assertions
    mac_count    = submitted.get("mac_local", 0)
    free_count   = tier_totals.get("free", 0)
    credit_count = tier_totals.get("credit", 0)
    remote_count = sum(v for k, v in submitted.items() if k != "mac_local")

    if not throttle_remote and mac_count > 0:
        mac_share = mac_count / total_submitted if total_submitted else 0
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
                 mac_count, total_submitted)

    if not throttle_remote and credit_count > 0 and free_count == 0:
        log.warning(
            "TIER POLICY WARNING: credit tier used (%d jobs) but no free-tier jobs "
            "dispatched — ensure free-tier clouds are enabled and have capacity.",
            credit_count,
        )
    elif free_count > 0 and credit_count > 0:
        free_pct = free_count / total_submitted * 100 if total_submitted else 0
        log.info(
            "Tier routing: %.0f%% free-tier, %d credit-tier — "
            "free clouds were saturated before escalating.",
            free_pct, credit_count,
        )


# ---------------------------------------------------------------------------
# Main dispatch loop
# ---------------------------------------------------------------------------
def _pick_cloud_with_wait(
    tracker: UsageTracker,
    job: "Job",
    cloud_first: bool,
    min_remote_usage_pct: float,
    wait_sleep_s: int,
    max_retries: int,
) -> Optional[str]:
    """Cloud-first picker with bounded wait loop.

    Returns the chosen cloud, or None if the job should be BLOCKED. When
    ``cloud_first`` is False this is a thin wrapper around the legacy
    ``tracker.pick_cloud()`` (behaviour-preserving fallback for operators
    who pass ``--no-cloud-first``).

    Priority handling: jobs carrying an attribute ``priority == "P1"``
    escalate to mac_local after P1_CLOUD_WAIT_MAX_S seconds of waiting.
    Other jobs strictly cloud-first until aggregate remote usage >= 90%.
    """
    if not cloud_first:
        return tracker.pick_cloud()

    priority = getattr(job, "priority", "P2") or "P2"
    waited_s = 0.0
    for attempt in range(max_retries + 1):
        chosen, action = tracker.pick_cloud_cloud_first(
            priority=priority,
            waited_s=waited_s,
            min_remote_usage_pct=min_remote_usage_pct,
        )
        if action == "submit":
            if attempt > 0:
                log.info(
                    "Cloud slot freed for %s after %.0fs wait — submitting to %s",
                    job, waited_s, chosen,
                )
            return chosen
        if action == "none":
            return None
        # action == "wait" — sleep and retry
        time.sleep(wait_sleep_s)
        waited_s += wait_sleep_s
    log.warning(
        "Cloud-first WAIT timeout for %s after %.0fs (%d retries) — BLOCKING job",
        job, waited_s, max_retries,
    )
    return None


def dispatch_pass(
    tracker: UsageTracker,
    dry_run: bool = False,
    cloud_first: bool = True,
    min_remote_usage_pct: float = CLOUD_FIRST_MIN_USAGE_PCT,
    wait_sleep_s: int = CLOUD_WAIT_SLEEP_S,
    max_retries: int = CLOUD_WAIT_MAX_RETRIES,
) -> int:
    """
    One full pass: read queue, dispatch all pending jobs, return count submitted.

    Cloud-first mode (2026-05-17 default): mac_local is engaged only when
    aggregate remote usage >= ``min_remote_usage_pct``. When all remote clouds
    are busy but aggregate usage is below the threshold, the dispatcher SLEEPS
    for ``wait_sleep_s`` per retry up to ``max_retries`` rather than falling
    to mac. Operators can disable via ``cloud_first=False``.

    Circuit-breaker + kill-switch (2026-05-21 emergency fix):
    - DISPATCHER_PAUSED=1 env var: exit immediately, log paused state.
    - sweeps/dispatch_blacklist.json: skip (ticker,strategy) pairs listed with
      reason=circuit_open until expires_utc passes (15-min TTL default).
    - Per-ticker circuit-breaker: if (ticker,strategy) has >=3 consecutive
      failures in last 5 min from dispatched.jsonl, skip with reason=circuit_open.
    """
    # ---------- Kill-switch: DISPATCHER_PAUSED=1 ----------
    if os.environ.get("DISPATCHER_PAUSED", "0") == "1":
        log.warning("DISPATCHER_PAUSED=1 — dispatch_pass exiting (kill-switch active)")
        return 0

    # ---------- Load blacklist (15-min TTL) ----------
    _blacklist_set: set = set()
    try:
        import json as _j
        from pathlib import Path as _P
        _bl_path = _P(QUEUE_FILE).parent / "dispatch_blacklist.json"
        if _bl_path.exists():
            with open(_bl_path) as _bf:
                _bl = _j.load(_bf)
            _exp = _bl.get("expires_utc", "")
            from datetime import datetime as _dt, timezone as _tz
            try:
                _exp_dt = _dt.fromisoformat(_exp.replace("Z", "+00:00"))
                if _dt.now(_tz.utc) < _exp_dt:
                    for _p in _bl.get("pairs", []):
                        _blacklist_set.add((_p.get("ticker"), _p.get("strategy")))
                    if _blacklist_set:
                        log.warning(
                            "Circuit-breaker blacklist active: %d (ticker,strategy) "
                            "pairs blocked until %s",
                            len(_blacklist_set), _exp,
                        )
            except Exception as _e:
                log.debug("Blacklist TTL parse failed: %s", _e)
    except Exception as _e:
        log.debug("Blacklist load failed (non-fatal): %s", _e)

    jobs = load_pending_jobs(QUEUE_FILE)
    if not jobs:
        log.info("Queue empty — nothing to dispatch.")
        return 0

    n_submitted = 0
    n_blocked   = 0
    n_circuit_open = 0
    submitted_log: List[Dict] = []

    for job in jobs:
        # ---------- Circuit-breaker check ----------
        _key = (getattr(job, "ticker", None), getattr(job, "strategy", None))
        if _key in _blacklist_set:
            n_circuit_open += 1
            if not dry_run:
                try:
                    _write_status_event(job, "circuit_open", cloud="-",
                                        extra={"reason": "blacklist_15min_ttl"})
                except Exception:
                    pass
            continue

        cloud = _pick_cloud_with_wait(
            tracker, job,
            cloud_first=cloud_first,
            min_remote_usage_pct=min_remote_usage_pct,
            wait_sleep_s=wait_sleep_s,
            max_retries=max_retries,
        )
        if cloud is None:
            log.warning("%s BLOCKED — all enabled clouds at quota/capacity", job)
            n_blocked += 1
            continue

        try:
            receipt = submit_job(job, cloud, tracker, dry_run=dry_run)
            job.cloud        = cloud
            job.submitted_at = receipt["submitted_at"]
            job.status       = "submitted"
            # Write status transition to dispatched.jsonl so check_status()
            # returns "submitted" instead of the stale "queued" entry.
            if not dry_run:
                _write_status_event(job, "submitted", cloud=cloud, extra=receipt)
                # Register the job for background completion polling so that
                # register_complete() is called once result.json appears —
                # this is the P0 fix for in_flight_jobs growing monotonically.
                _register_inflight(job, cloud)
            submitted_log.append(receipt)
            n_submitted += 1
        except AuthFailureError as exc:
            # Auth failure already logged + cooldown set inside submit_job().
            # Try the next best cloud for this job.
            log.info(
                "Auth failure on %s — retrying %s on next available cloud.", cloud, job
            )
            remaining_enabled = [
                c for c in tracker.enabled_clouds()
                if c != cloud and tracker.headroom_pct(c) >= 0
            ]
            retried = False
            for fallback_cloud in remaining_enabled:
                try:
                    receipt = submit_job(job, fallback_cloud, tracker,
                                        dry_run=dry_run)
                    job.cloud        = fallback_cloud
                    job.submitted_at = receipt["submitted_at"]
                    job.status       = "submitted"
                    # Write status transition for the fallback cloud too.
                    if not dry_run:
                        _write_status_event(
                            job, "submitted", cloud=fallback_cloud, extra=receipt
                        )
                        # Register fallback submit for completion polling.
                        _register_inflight(job, fallback_cloud)
                    submitted_log.append(receipt)
                    n_submitted += 1
                    retried = True
                    break
                except AuthFailureError:
                    continue
                except Exception as inner_exc:
                    log.error(
                        "Fallback submit of %s to %s also failed: %s",
                        job, fallback_cloud, inner_exc,
                    )
                    break
            if not retried:
                log.warning(
                    "%s could not be dispatched to any cloud after auth failure on %s.",
                    job, cloud,
                )
                if not dry_run:
                    _write_status_event(job, "failed", cloud=cloud)
                n_blocked += 1
        except Exception as exc:
            log.error("Failed to submit %s to %s: %s", job, cloud, exc)
            if not dry_run:
                _write_status_event(job, "failed", cloud=cloud)

    log.info("Dispatch pass done: submitted=%d blocked=%d", n_submitted, n_blocked)

    # Persist the submission log for audit trail
    log_path = LOG_DIR / f"dispatch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, "w") as f:
        json.dump(submitted_log, f, indent=2)
    log.info("Submission log: %s", log_path)

    # ------------------------------------------------------------------
    # Queue truncation (Bug Fix #2):
    # Remove dispatched job lines from queue.txt so they aren't re-submitted
    # on the next pass.  Uses fcntl.flock for safe concurrent access and an
    # atomic rename via a /tmp staging file to prevent partial writes.
    # ------------------------------------------------------------------
    if submitted_log and not dry_run:
        _truncate_dispatched_from_queue(QUEUE_FILE, submitted_log, jobs)

    return n_submitted


def _truncate_dispatched_from_queue(
    queue_file: Path,
    submitted_log: List[Dict],
    dispatched_jobs: List[Job],
) -> None:
    """Remove lines from queue.txt whose jobs were successfully submitted this pass.

    Algorithm (all steps execute while holding an exclusive flock):
        1. Acquire an exclusive flock on queue.txt — blocks until any concurrent
           producer's append is complete.
        2. Re-read queue.txt in full (another process may have appended after our
           initial ``load_pending_jobs`` call).
        3. Build match sets from the dispatched jobs:
               - job_id set  (for 4-token lines written by cloud_dispatch.py)
               - (ticker, strategy) set (legacy fallback for 3-token lines)
        4. Filter: keep blank/comment lines and any line NOT in the match sets.
        5. Write kept lines to a /tmp staging file, then ``shutil.move`` it over
           queue.txt atomically (same filesystem on macOS so this is a rename).
        6. Release flock by closing the file handle.

    The flock is held for the **entire** read-filter-write cycle so that a
    concurrent ``enqueue_job`` that appends between our read and our write-back
    is serialized — it will either complete before we lock (and we'll see its
    line during re-read) or block until after we rename (and its append will
    land on the freshly-written file).
    """
    if not queue_file.exists():
        return

    # shutil and tempfile imported at module level

    # Build match sets for dispatched jobs
    dispatched_ids: set[str] = {
        j.job_id for j in dispatched_jobs if j.status == "submitted"
    }
    dispatched_pairs: set[tuple[str, str]] = {
        (j.ticker, j.strategy) for j in dispatched_jobs if j.status == "submitted"
    }

    if not dispatched_ids and not dispatched_pairs:
        return

    try:
        # Hold the exclusive lock for the full read-filter-write cycle.
        # _flocked opens in "a" (append) mode by default; we need "r+" here
        # so we can read the existing content.  We use the raw fcntl approach
        # instead of _flocked() because we need to read content, not append.
        with open(queue_file, "r+", encoding="utf-8") as qf:
            fcntl.flock(qf.fileno(), fcntl.LOCK_EX)
            try:
                lines = qf.readlines()

                kept: List[str] = []
                removed = 0
                for raw in lines:
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        kept.append(raw)
                        continue
                    parts = line.split()
                    # 4-token line: match by job_id (precise)
                    if len(parts) >= 4 and parts[3] in dispatched_ids:
                        removed += 1
                        log.debug("Queue truncation: removing by job_id %s: %s",
                                  parts[3], line)
                        continue
                    # 3-token line: match by (ticker, strategy) (legacy)
                    if len(parts) >= 3:
                        pair = (parts[1], parts[2])
                        if pair in dispatched_pairs:
                            removed += 1
                            log.debug(
                                "Queue truncation: removing by (ticker,strategy) %s: %s",
                                pair, line,
                            )
                            continue
                    kept.append(raw)

                # Write to a /tmp staging file and atomically rename over queue.txt.
                # The rename is atomic on macOS (POSIX rename guarantee), so readers
                # always see either the old or the new file — never a partial write.
                tmp_fd, tmp_path = tempfile.mkstemp(
                    prefix="queue_", suffix=".txt", dir="/tmp"
                )
                try:
                    with os.fdopen(tmp_fd, "w", encoding="utf-8") as tmp_f:
                        tmp_f.writelines(kept)
                    shutil.move(tmp_path, str(queue_file))
                    log.info(
                        "Queue truncated: removed %d dispatched line(s), "
                        "%d remaining.",
                        removed, len(kept),
                    )
                except Exception:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                    raise
            finally:
                fcntl.flock(qf.fileno(), fcntl.LOCK_UN)

    except Exception as exc:
        log.warning("Queue truncation failed (queue.txt unchanged): %s", exc)


# ---------------------------------------------------------------------------
# In-flight drift correction
# ---------------------------------------------------------------------------
# Helper C: every DRIFT_CORRECTION_INTERVAL_SEC seconds, scan dispatched.jsonl
# for jobs that are still "submitted" (no subsequent "complete"/"failed" event),
# whose submitted_at timestamp is older than DRIFT_MAX_AGE_SEC, AND that have
# no result.json on disk.  These are orphaned in-flight slots — they were
# incremented by register_submit() but register_complete() was never called
# (e.g., the daemon was restarted mid-flight, or the completion-poller missed
# them on a previous run).  Force-decrement each orphan once.
#
# This is a safety net; the primary fix is the completion-poller above.
# The drift-correction pass prevents permanent starvation if any job slips
# through the poller (restart, crash, manual intervention).
# ---------------------------------------------------------------------------

DRIFT_CORRECTION_INTERVAL_SEC = 300   # 5 minutes
DRIFT_MAX_AGE_SEC              = 3600  # jobs older than 1 h with no result → orphan


def _load_dispatched_inflight() -> Dict[str, Dict[str, Any]]:
    """Read dispatched.jsonl and return a dict of job_id → latest event record
    for all jobs whose *latest* status event is "submitted" (i.e., not yet
    completed or failed).

    The JSONL file is an append-only event log; we collect all records per
    job_id and keep the one with the latest ``ts`` field.  Only those whose
    final status is "submitted" are considered in-flight.

    Returns:
        Dict mapping job_id to the most-recent event record for jobs that are
        still in the "submitted" state.
    """
    if not STATUS_FILE.exists():
        return {}

    latest_by_id: Dict[str, Dict[str, Any]] = {}
    try:
        with open(STATUS_FILE, "r", encoding="utf-8") as fh:
            for raw_line in fh:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    rec = json.loads(raw_line)
                    jid = rec.get("id")
                    if not jid:
                        continue
                    prev = latest_by_id.get(jid)
                    if prev is None or rec.get("ts", "") > prev.get("ts", ""):
                        latest_by_id[jid] = rec
                except (json.JSONDecodeError, KeyError):
                    continue
    except OSError:
        return {}

    return {
        jid: rec
        for jid, rec in latest_by_id.items()
        if rec.get("status") == "submitted"
    }


def _drift_correction_pass(tracker: "UsageTracker") -> None:
    """Scan dispatched.jsonl for orphaned in-flight jobs and force-decrement
    their cloud's in_flight counter.

    An orphaned job is one where:
      - The latest status event in dispatched.jsonl is "submitted".
      - The job is NOT in the live ``_inflight_registry`` (i.e., not being
        actively polled by the completion-poller thread).
      - The job's ``submitted_at`` timestamp is more than DRIFT_MAX_AGE_SEC
        old.
      - The job's ``result.json`` does NOT exist on disk.

    When all three hold, the job slot will never be freed by normal means, so
    we force-decrement and write a "failed" status event with reason
    "drift_correction".

    Args:
        tracker: UsageTracker instance.  ``tracker.load()`` is called before
                 making any changes so we see the freshest in_flight counts.
    """
    submitted_events = _load_dispatched_inflight()
    if not submitted_events:
        return

    now = time.time()
    orphans_found = 0
    tracker.load()   # fresh snapshot before mutating

    with _inflight_lock:
        live_ids = set(_inflight_registry.keys())

    for job_id, rec in submitted_events.items():
        # Skip if the completion-poller is already watching this job
        if job_id in live_ids:
            continue

        # Parse submitted_at timestamp → epoch float
        submitted_at_str = rec.get("ts", "")
        try:
            from datetime import datetime, timezone
            submitted_at_epoch = datetime.fromisoformat(
                submitted_at_str
            ).timestamp()
        except (ValueError, TypeError):
            continue  # can't parse timestamp; skip

        age_sec = now - submitted_at_epoch
        if age_sec < DRIFT_MAX_AGE_SEC:
            continue  # too fresh; give it more time

        # Check if result.json exists — if so, sweep_completions will handle it
        cloud = rec.get("cloud", "")
        ticker    = rec.get("ticker", "")
        strategy  = rec.get("strategy", "")
        result_path = RESULTS_DIR / ticker / strategy / "result.json"
        if result_path.exists():
            continue  # result landed; sweep_completions will clean it up

        # Orphan confirmed — force-decrement
        if cloud and cloud in tracker.data:
            tracker.register_complete(cloud)

        # Append a failed event so dispatched.jsonl shows final status
        orphan_job = Job(
            script=rec.get("script", "unknown"),
            ticker=ticker,
            strategy=strategy,
            job_id=job_id,
            cloud=cloud,
        )
        orphan_job.status = "failed"
        _write_status_event(
            orphan_job, "failed", cloud=cloud,
            extra={
                "reason": "drift_correction",
                "age_sec": int(age_sec),
                "drift_max_age_sec": DRIFT_MAX_AGE_SEC,
            },
        )
        log.warning(
            "DRIFT CORRECTION: freed in_flight slot for orphan job_id=%s "
            "cloud=%s age=%.0fs (no result.json, not in live registry).",
            job_id, cloud, age_sec,
        )
        orphans_found += 1

    if orphans_found:
        log.info(
            "Drift correction pass complete: freed %d orphaned in_flight slot(s).",
            orphans_found,
        )
    else:
        log.debug("Drift correction pass: no orphaned jobs found.")


def daemon_loop(
    tracker: UsageTracker,
    dry_run: bool = False,
    cloud_first: bool = True,
    min_remote_usage_pct: float = CLOUD_FIRST_MIN_USAGE_PCT,
    wait_sleep_s: int = CLOUD_WAIT_SLEEP_S,
    max_retries: int = CLOUD_WAIT_MAX_RETRIES,
) -> None:
    log.info(
        "Dispatcher daemon started (poll every %ds, cloud_first=%s, "
        "mac_engage_at=%.0f%%). Ctrl-C to stop.",
        POLL_INTERVAL, cloud_first, min_remote_usage_pct,
    )

    # Start the background completion-poller thread once per daemon session.
    # The thread is a daemon thread so it exits cleanly when the main process exits.
    # Poll every 15 s for result files; max_job_seconds read from mac_local config
    # (defaults to _DEFAULT_MAX_JOB_SECONDS = 3600 s if not configured).
    _max_job_sec = int(
        tracker.data.get("mac_local", {}).get("max_job_seconds",
        _DEFAULT_MAX_JOB_SECONDS)
    )
    _start_completion_poller(
        tracker,
        poll_interval_sec=15,
        max_job_seconds=_max_job_sec,
    )

    _last_drift_check: float = 0.0  # epoch of last drift-correction pass

    while True:
        # Honor DISPATCHER_PAUSED kill-switch at loop level (2026-05-21).
        if os.environ.get("DISPATCHER_PAUSED", "0") == "1":
            log.warning("DISPATCHER_PAUSED=1 — daemon_loop sleeping %ds (paused)", POLL_INTERVAL)
            time.sleep(POLL_INTERVAL)
            continue
        try:
            tracker.load()   # reload in case cloud_usage.json was hand-edited
            dispatch_pass(
                tracker, dry_run=dry_run,
                cloud_first=cloud_first,
                min_remote_usage_pct=min_remote_usage_pct,
                wait_sleep_s=wait_sleep_s,
                max_retries=max_retries,
            )

            # Run drift-correction every DRIFT_CORRECTION_INTERVAL_SEC (default 5 min).
            # This is a safety net that frees orphaned in_flight slots for jobs that
            # were submitted in a previous daemon session (before this fix was deployed)
            # or that somehow slipped past the completion-poller thread.
            now = time.time()
            if now - _last_drift_check >= DRIFT_CORRECTION_INTERVAL_SEC:
                if not dry_run:
                    try:
                        _drift_correction_pass(tracker)
                    except Exception as drift_exc:
                        log.error(
                            "Drift correction pass failed: %s", drift_exc,
                            exc_info=True,
                        )
                _last_drift_check = now

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
    p.add_argument("--reset-cooldowns", action="store_true",
                   help="Clear all auth_failure_count and cooldown_until fields "
                        "in cloud_usage.json (operator override after fixing creds)")
    p.add_argument("--show-tiers", action="store_true",
                   help="Print each enabled cloud with its cost_tier and current "
                        "headroom, then exit. Useful for debugging tier routing.")
    # ---- Cloud-first 90% routing (2026-05-17 operator mandate) -------------
    p.add_argument("--cloud-first-90", dest="cloud_first", action="store_true",
                   default=True,
                   help="Enforce cloud-first routing: mac_local only engaged "
                        "after aggregate remote usage >= 90%% (default ON).")
    p.add_argument("--no-cloud-first", dest="cloud_first", action="store_false",
                   help="Disable cloud-first routing; use legacy tier picker "
                        "(mac last-resort regardless of remote usage).")
    p.add_argument("--cloud-first-pct", type=float,
                   default=CLOUD_FIRST_MIN_USAGE_PCT, metavar="PCT",
                   help=f"Aggregate remote-usage %% required before engaging "
                        f"mac_local (default {CLOUD_FIRST_MIN_USAGE_PCT}).")
    p.add_argument("--cloud-wait-sleep", type=int,
                   default=CLOUD_WAIT_SLEEP_S, metavar="SEC",
                   help=f"Seconds to sleep per retry while waiting for cloud "
                        f"capacity (default {CLOUD_WAIT_SLEEP_S}).")
    p.add_argument("--cloud-wait-retries", type=int,
                   default=CLOUD_WAIT_MAX_RETRIES, metavar="N",
                   help=f"Max cloud-wait retries before BLOCKING a job "
                        f"(default {CLOUD_WAIT_MAX_RETRIES}).")
    return p.parse_args()


def reset_usage(tracker: UsageTracker) -> None:
    """Zero out all billing counters. Run manually at month rollover."""
    for key, cfg in tracker.data.items():
        if key.startswith("_"):
            continue
        for field in ("used_min_this_month", "used_credit_this_month",
                      "used_hr_this_month", "used_invocations_this_month",
                      "in_flight_jobs"):
            if field in cfg:
                cfg[field] = 0
    tracker.data["_last_reset"] = _now_iso()
    tracker.save()
    log.info("Usage counters reset for new month.")


def show_tiers(tracker: UsageTracker) -> None:
    """Print each enabled cloud with its cost_tier and current headroom.

    Invoked via ``--show-tiers`` CLI flag for debugging and capacity review.
    Clouds are grouped by tier in TIER_ORDER order; disabled clouds are listed
    separately at the bottom so operators can see the full picture.
    """
    enabled  = set(tracker.enabled_clouds())
    all_keys = [k for k in tracker.data if not k.startswith("_")]

    header = f"{'CLOUD':<25} {'TIER':<10} {'ENABLED':<9} {'HEADROOM':>10}  {'NOTES'}"
    print()
    print(header)
    print("-" * len(header))

    # Print enabled clouds grouped by tier
    for tier in TIER_ORDER + ["unknown"]:
        in_tier = [
            k for k in all_keys
            if k in enabled
            and (tracker.data[k].get("cost_tier", "credit") == tier
                 or (tier == "unknown"
                     and tracker.data[k].get("cost_tier", "credit") not in TIER_ORDER))
        ]
        for cloud in sorted(in_tier):
            cfg      = tracker.data[cloud]
            hp       = tracker.headroom_pct(cloud)
            hp_str   = f"{hp:.1f}%" if hp >= 0 else "OVER-QUOTA"
            label    = cfg.get("label", cloud)
            print(f"  {cloud:<23} {tier:<10} {'yes':<9} {hp_str:>10}  {label}")

    # Print disabled clouds in a collapsed section
    disabled = [k for k in all_keys if k not in enabled]
    if disabled:
        print()
        print(f"  {'--- disabled ---':<23}")
        for cloud in sorted(disabled):
            cfg   = tracker.data[cloud]
            tier  = cfg.get("cost_tier", "?")
            label = cfg.get("label", cloud)
            print(f"  {cloud:<23} {tier:<10} {'no':<9} {'---':>10}  {label}")
    print()


def main() -> None:
    args   = parse_args()
    tracker = UsageTracker(USAGE_FILE)

    if args.reset_usage:
        reset_usage(tracker)
        return

    if args.reset_cooldowns:
        tracker.reset_cooldowns()
        return

    if args.show_tiers:
        show_tiers(tracker)
        return

    if args.simulate:
        run_simulation(args.simulate, tracker,
                       throttle_remote=args.sim_throttle_remote)
        return

    if args.daemon:
        daemon_loop(
            tracker, dry_run=args.dry_run,
            cloud_first=args.cloud_first,
            min_remote_usage_pct=args.cloud_first_pct,
            wait_sleep_s=args.cloud_wait_sleep,
            max_retries=args.cloud_wait_retries,
        )
    else:
        dispatch_pass(
            tracker, dry_run=args.dry_run,
            cloud_first=args.cloud_first,
            min_remote_usage_pct=args.cloud_first_pct,
            wait_sleep_s=args.cloud_wait_sleep,
            max_retries=args.cloud_wait_retries,
        )


if __name__ == "__main__":
    main()
