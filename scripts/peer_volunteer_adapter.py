"""
peer_volunteer_adapter.py — PeerVolunteerAdapter SKELETON for the
multi_cloud_dispatcher.

Status: SKELETON / TODO. Not wired into the main dispatcher yet because
`multi_cloud_dispatcher.py` is under active edit by another agent. Once
that edit lands, merge this class into the dispatcher's `ADAPTERS` registry
and add a `peer_volunteer` entry to `sweeps/cloud_usage.json`.

PURPOSE
-------
Dispatch S&P-500 backtest jobs to a peer / volunteer / donor pool — e.g.
Drone CI / Buildkite self-hosted agents on friend hardware, Bacalhau /
Lilypad public Docker workers, or a self-hosted BOINC-Docker pool.

SAFETY KNOBS (required for every donor pathway)
-----------------------------------------------
1. `max_concurrent_jobs_per_donor`   — per-donor concurrency cap (default 2)
2. `sandboxed_payload_only`          — when True, refuse to ship any payload
                                       containing credentials, API keys,
                                       broker tokens, or proprietary alpha.
                                       Default True. Hard fail if violated.
3. `workload_signature`              — SHA-256 of (script_source + cmd_args
                                       + container_image_digest). Donors can
                                       verify exactly what they're being asked
                                       to run; mismatch → reject.
4. `per_donor_budget_seconds`        — max wall-clock per donor per job
                                       (default 3600 = 1 hour). Hard kill
                                       beyond this so a flaky donor can't
                                       hold a job hostage.

ADDITIONAL DESIGN NOTES
-----------------------
* Treat donor compute as UNTRUSTED. Results must be verifiable (either via
  duplicate execution on 2 donors and consensus, or via a deterministic
  re-check on the orchestrator after return).
* Never ship: .env, ~/.aws, ~/.config/gcloud, brokerage tokens, Alpaca keys,
  any file matched by SECRET_REGEX, any private CSV in /data unless flagged
  `public_ok=True`.
* Each donor identity = (donor_id, public_key). Jobs signed by orchestrator;
  results signed by donor. Mismatch = drop.
* Heartbeat every 30 s from donor agent → orchestrator. Missed 3 in a row →
  job re-queued.

POSSIBLE BACKENDS (per the peer_compute research bundle)
--------------------------------------------------------
A) Bacalhau public network    — Docker-native, permissionless, free
B) Lilypad IncentiveNet       — Bacalhau fork w/ token rewards on testnet
C) Drone CI self-hosted       — agents on friend boxes, no ToS gray
D) GitHub self-hosted runners — agents on friend boxes, integrates w/ existing
                                GitHub Actions adapter
E) Browser donors (WebAssembly) — TODO: research-pending
F) Friend P2P cluster (SSH)   — TODO: research-pending

The adapter is backend-agnostic; `backend` is a config knob.

EXAMPLE cloud_usage.json ENTRY (to add later)
---------------------------------------------
"peer_volunteer": {
    "enabled": false,
    "backend": "bacalhau",             // or drone | gh_selfhosted | lilypad
    "billing_model": "concurrent_cap",
    "max_concurrent_jobs": 8,          // total across all donors
    "max_concurrent_jobs_per_donor": 2,
    "in_flight_jobs": 0,
    "safety_margin_pct": 80,
    "sandboxed_payload_only": true,
    "per_donor_budget_seconds": 3600,
    "donors": [
        {"id": "friend_alice", "endpoint": "...", "pubkey": "...",
         "trust_tier": "high"},
        {"id": "bacalhau_pub", "endpoint": "https://bootstrap.production.bacalhau.org",
         "pubkey": null, "trust_tier": "untrusted"}
    ]
}
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("dispatcher.peer_volunteer")

# ---------------------------------------------------------------------------
# Hard-coded leak-prevention. Refuse to ship any file matching these patterns.
# ---------------------------------------------------------------------------
SECRET_PATH_PATTERNS = [
    re.compile(r"\.env(\..*)?$"),
    re.compile(r".*credentials\.json$"),
    re.compile(r".*[._-]token[._-]?.*$", re.IGNORECASE),
    re.compile(r".*alpaca.*key.*", re.IGNORECASE),
    re.compile(r".*api[._-]?key.*", re.IGNORECASE),
    re.compile(r".*\.pem$"),
    re.compile(r".*id_(rsa|ed25519|ecdsa)(\.pub)?$"),
    re.compile(r".*\.kdbx$"),                       # KeePass
    re.compile(r".*service[_-]account.*\.json$"),
    re.compile(r".*\.netrc$"),
]

SECRET_CONTENT_PATTERNS = [
    re.compile(rb"(?i)AKIA[0-9A-Z]{16}"),           # AWS key id
    re.compile(rb"(?i)sk-[A-Za-z0-9]{20,}"),        # OpenAI / Anthropic-ish
    re.compile(rb"(?i)ghp_[A-Za-z0-9]{30,}"),       # GitHub PAT
    re.compile(rb"(?i)BEGIN [A-Z ]*PRIVATE KEY"),
]


class PayloadSecurityError(RuntimeError):
    """Raised when payload validation refuses to ship a job to a donor."""


class DonorBudgetExceeded(RuntimeError):
    """Raised when a donor has spent more than per_donor_budget_seconds."""


# ---------------------------------------------------------------------------
# Donor accounting
# ---------------------------------------------------------------------------
@dataclass
class DonorState:
    donor_id: str
    endpoint: str
    pubkey: Optional[str]
    trust_tier: str = "untrusted"          # high | medium | untrusted
    in_flight: int = 0
    total_seconds_used: float = 0.0
    last_heartbeat: Optional[str] = None
    failure_count: int = 0


# ---------------------------------------------------------------------------
# Workload signature
# ---------------------------------------------------------------------------
def compute_workload_signature(script_path: Path, ticker: str, strategy: str,
                               container_image_digest: Optional[str] = None) -> str:
    """
    Deterministic SHA-256 covering exactly what the donor will execute.
    Anything that affects the run must be in the hash.
    """
    h = hashlib.sha256()
    h.update(Path(script_path).read_bytes() if Path(script_path).exists() else b"")
    h.update(b"\x00")
    h.update(ticker.encode())
    h.update(b"\x00")
    h.update(strategy.encode())
    h.update(b"\x00")
    h.update((container_image_digest or "").encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Payload validation (sandboxed_payload_only)
# ---------------------------------------------------------------------------
def validate_payload(payload_paths: List[Path]) -> None:
    """
    Walk every path that will be sent to a donor and raise PayloadSecurityError
    if anything looks like a secret. Hard-fails; never logs the secret itself.
    """
    for p in payload_paths:
        name = p.name
        for pat in SECRET_PATH_PATTERNS:
            if pat.search(name) or pat.search(str(p)):
                raise PayloadSecurityError(
                    f"Refusing to ship suspicious-named file to donor: {name}"
                )
        if p.is_file() and p.stat().st_size < 1_000_000:  # only scan small files
            try:
                data = p.read_bytes()
            except Exception:
                continue
            for pat in SECRET_CONTENT_PATTERNS:
                if pat.search(data):
                    raise PayloadSecurityError(
                        f"Secret-pattern match inside payload file: {p}"
                    )


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------
class PeerVolunteerAdapter:
    """
    Skeleton adapter. Real backends wired in subclasses or via the `backend`
    config knob.

    Public methods follow the same shape as other adapters in
    multi_cloud_dispatcher.py so we can register this with `ADAPTERS`.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.backend = cfg.get("backend", "bacalhau")
        self.max_total = cfg.get("max_concurrent_jobs", 8)
        self.max_per_donor = cfg.get("max_concurrent_jobs_per_donor", 2)
        self.sandboxed_only = cfg.get("sandboxed_payload_only", True)
        self.budget_sec = cfg.get("per_donor_budget_seconds", 3600)
        self.donors: Dict[str, DonorState] = {
            d["id"]: DonorState(
                donor_id=d["id"],
                endpoint=d["endpoint"],
                pubkey=d.get("pubkey"),
                trust_tier=d.get("trust_tier", "untrusted"),
            )
            for d in cfg.get("donors", [])
        }

    # ------- selection -------------------------------------------------------
    def pick_donor(self) -> Optional[DonorState]:
        """Return the donor with most headroom that's under all caps."""
        candidates = [
            d for d in self.donors.values()
            if d.in_flight < self.max_per_donor
               and d.total_seconds_used < self.budget_sec
               and d.failure_count < 3
        ]
        if not candidates:
            return None
        # Prefer high-trust donors; tie-break by least in-flight
        candidates.sort(key=lambda d: (
            {"high": 0, "medium": 1, "untrusted": 2}.get(d.trust_tier, 3),
            d.in_flight,
            d.donor_id,
        ))
        return candidates[0]

    # ------- submit ----------------------------------------------------------
    def submit(self, job, dry_run: bool = False) -> Dict[str, Any]:
        """
        TODO: wire backend-specific submission. Steps:
            1. validate_payload(payload_paths) if sandboxed_payload_only
            2. compute_workload_signature(...)
            3. pick_donor()
            4. dispatch to backend (bacalhau docker run / drone webhook / ssh /
               gh self-hosted dispatch / etc.)
            5. record in donor.in_flight += 1
            6. return receipt dict with job_id, cloud='peer_volunteer',
               donor_id, signature, submitted_at
        """
        # ---- Phase 0: skeleton-safe early returns ----
        donor = self.pick_donor()
        if donor is None:
            raise RuntimeError("No donor with headroom — peer_volunteer at cap")

        # Compute & log signature so donors can verify what they're running
        script_path = Path(job.script)
        sig = compute_workload_signature(
            script_path=script_path,
            ticker=job.ticker,
            strategy=job.strategy,
            container_image_digest=self.cfg.get("container_image_digest"),
        )

        # Validate payload doesn't leak secrets
        if self.sandboxed_only:
            payload_paths = [script_path]  # TODO: extend to data inputs once defined
            validate_payload(payload_paths)

        receipt = {
            "job_id":       job.job_id,
            "cloud":        "peer_volunteer",
            "backend":      self.backend,
            "donor_id":     donor.donor_id,
            "donor_trust":  donor.trust_tier,
            "signature":    sig,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "budget_sec":   self.budget_sec,
            "dry_run":      dry_run,
        }

        if dry_run:
            log.info("[DRY-RUN] peer_volunteer would dispatch %s → donor=%s "
                     "backend=%s sig=%s", job, donor.donor_id, self.backend,
                     sig[:12])
            return receipt

        # ---- Phase 1: TODO real dispatch -----------------------------------
        # if self.backend == "bacalhau":
        #     ... bacalhau docker run ... return job_id
        # elif self.backend == "drone":
        #     ... POST to drone /api/queue ...
        # elif self.backend == "gh_selfhosted":
        #     ... workflow_dispatch with `runs-on: [self-hosted, donor-pool]` ...
        # elif self.backend == "lilypad":
        #     ... lilypad CLI ...
        # elif self.backend == "browser_wasm":
        #     ... TODO: pending research input ...
        # else:
        #     raise NotImplementedError(self.backend)

        donor.in_flight += 1
        log.info("[STUB] peer_volunteer registered job %s on donor %s "
                 "(backend=%s) — real dispatch not yet implemented",
                 job, donor.donor_id, self.backend)

        return receipt

    # ------- completion ------------------------------------------------------
    def report_complete(self, donor_id: str, seconds_used: float,
                        success: bool) -> None:
        d = self.donors.get(donor_id)
        if d is None:
            return
        d.in_flight = max(0, d.in_flight - 1)
        d.total_seconds_used += seconds_used
        if not success:
            d.failure_count += 1

    def heartbeat(self, donor_id: str) -> None:
        d = self.donors.get(donor_id)
        if d is not None:
            d.last_heartbeat = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Wiring hook — to be called from multi_cloud_dispatcher.py once merged
# ---------------------------------------------------------------------------
def submit_peer_volunteer(job, dry_run: bool = False,
                          cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Top-level adapter function matching the other _submit_* signatures."""
    if cfg is None:
        # Minimal default config for skeleton mode
        cfg = {
            "backend": "bacalhau",
            "max_concurrent_jobs": 8,
            "max_concurrent_jobs_per_donor": 2,
            "sandboxed_payload_only": True,
            "per_donor_budget_seconds": 3600,
            "donors": [],
        }
    adapter = PeerVolunteerAdapter(cfg)
    return adapter.submit(job, dry_run=dry_run)


# To register in multi_cloud_dispatcher.ADAPTERS once that file's edit lands:
#
#     from peer_volunteer_adapter import submit_peer_volunteer
#     ADAPTERS["peer_volunteer"] = submit_peer_volunteer
#
# And add the cloud_usage.json entry shown in the docstring above.
