"""
modal_worker.py — Modal.com remote worker for the XGBoost backtest sweep.

Deploy once:
    modal deploy scripts/modal_worker.py

Dispatch a single job (from local machine or dispatcher):
    modal run scripts/modal_worker.py \
        --ticker AAPL --strategy ORB \
        --script scripts/backtest_xgb_v8.py --job-id abc12345

Dispatch many jobs in parallel (Modal handles fan-out):
    python scripts/dispatch_modal_batch.py   # optional convenience wrapper

Required environment variables (set via Modal Secrets or shell export):
    MODAL_TOKEN_ID      — from `modal token new`
    MODAL_TOKEN_SECRET  — from `modal token new`

Optional / passed at runtime:
    RESULT_STORE_PATH   — where to write result.json (default: /tmp/results)
    DATA_BUCKET         — GCS/S3 bucket for input data files (if used)

Notes:
  - This file uses the modal >= 0.60 API (App-based, not Stub-based).
  - Each function call runs in its own isolated container; no shared filesystem.
  - Results are written to /tmp inside the container then returned as a dict.
    The dispatcher writes that dict to the Drive results path.
  - A local "mock" mode (no Modal SDK) is available for unit testing.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Modal SDK import — graceful fallback so the file can be imported on machines
# that don't have modal installed (e.g. for unit tests / dry-run simulation).
# ---------------------------------------------------------------------------
try:
    import modal  # type: ignore

    _MODAL_AVAILABLE = True
except ImportError:
    _MODAL_AVAILABLE = False
    modal = None  # type: ignore

# ---------------------------------------------------------------------------
# App / image definition
# ---------------------------------------------------------------------------
# Image: base Python 3.12 + XGBoost stack.
# Pin versions to keep builds reproducible across runs.
_PACKAGES = [
    "xgboost==2.1.3",
    "pandas==2.2.3",
    "numpy==1.26.4",
    "scikit-learn==1.6.1",
    "pyarrow==17.0.0",
    "requests==2.32.3",
    "python-dotenv==1.0.1",
]

if _MODAL_AVAILABLE:
    _image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(_PACKAGES)
    )

    app = modal.App(
        name="sp500-xgb-sweep",
        image=_image,
    )
else:
    # Stubs for local import without Modal SDK
    app = None  # type: ignore
    _image = None  # type: ignore


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT_ENV = "SP500_PROJECT_ROOT"
DEFAULT_RESULT_DIR = Path("/tmp/sp500_results")

# Estimated cost per job (0.5 CPU-min at $0.0471/vCPU-hr = ~$0.0004)
COST_PER_JOB_USD = 0.0004

# Hard job-level timeout — prevents runaway containers from burning credit
JOB_TIMEOUT_SEC = 300   # 5 minutes; single-ticker backtests finish in <30 s


# ---------------------------------------------------------------------------
# Core backtest runner (cloud-side logic)
# ---------------------------------------------------------------------------
def _run_backtest_local(
    script: str,
    ticker: str,
    strategy: str,
    job_id: str,
    project_root: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes the backtest script as a subprocess inside the container.

    The script is expected to accept:
        python <script> --ticker TICKER --strategy STRATEGY [--job-id JOB_ID]
    and write its output to stdout as JSON on the last line (or exit 0 on success).

    Returns a result dict:
        {
          "job_id":        str,
          "ticker":        str,
          "strategy":      str,
          "cloud":         "modal",
          "status":        "completed" | "failed",
          "returncode":    int,
          "stdout_tail":   str,   # last 2000 chars of stdout
          "stderr_tail":   str,
          "elapsed_sec":   float,
          "completed_at":  ISO8601 str,
          "result":        dict | None,  # parsed JSON from last stdout line
        }
    """
    root = Path(project_root) if project_root else Path("/workspace")
    python_bin = sys.executable
    cmd = [
        python_bin,
        str(root / script),
        "--ticker",   ticker,
        "--strategy", strategy,
        "--job-id",   job_id,
    ]

    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=JOB_TIMEOUT_SEC,
            cwd=str(root),
        )
        elapsed = time.monotonic() - start
        stdout  = proc.stdout[-2000:] if proc.stdout else ""
        stderr  = proc.stderr[-2000:] if proc.stderr else ""

        # Try to parse last line as JSON result payload
        result_json: Optional[Dict] = None
        for line in reversed((proc.stdout or "").splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    result_json = json.loads(line)
                    break
                except json.JSONDecodeError:
                    pass

        status = "completed" if proc.returncode == 0 else "failed"
    except subprocess.TimeoutExpired:
        elapsed = JOB_TIMEOUT_SEC
        stdout  = ""
        stderr  = f"Process timed out after {JOB_TIMEOUT_SEC}s"
        result_json = None
        status  = "failed"
        proc    = type("P", (), {"returncode": -1})()

    return {
        "job_id":       job_id,
        "ticker":       ticker,
        "strategy":     strategy,
        "cloud":        "modal",
        "status":       status,
        "returncode":   getattr(proc, "returncode", -1),
        "stdout_tail":  stdout,
        "stderr_tail":  stderr,
        "elapsed_sec":  round(elapsed, 3),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "result":       result_json,
        "cost_usd_est": round(elapsed / 3600 * 0.0471, 6),
    }


# ---------------------------------------------------------------------------
# Modal-decorated remote function
# ---------------------------------------------------------------------------
if _MODAL_AVAILABLE and app is not None:

    @app.function(
        cpu=1.0,
        memory=768,          # MB — our jobs use ~200 MB; headroom for imports
        timeout=JOB_TIMEOUT_SEC + 30,
        retries=modal.Retries(
            max_retries=2,
            backoff_coefficient=2.0,
            initial_delay=5.0,
        ),
        # Mount the project code into the container.
        # In production, mount the repo root or use a Modal Volume.
        # For MVP: code is bundled via Modal's local_dir mount.
        mounts=[
            modal.Mount.from_local_dir(
                # Resolved at deploy time — this path must exist on the machine
                # running `modal deploy`.
                local_path=os.environ.get(
                    PROJECT_ROOT_ENV,
                    str(Path(__file__).parent.parent),
                ),
                remote_path="/workspace",
            )
        ],
        # Secrets — add your own via `modal secret create sp500-secrets`
        secrets=[
            modal.Secret.from_name(
                "sp500-secrets",
                # Gracefully skip if the secret hasn't been created yet
                # (avoids deploy failures during initial setup).
                required=False,
            )
        ],
    )
    def run_backtest(
        ticker: str,
        strategy: str,
        script: str = "scripts/backtest_xgb_v8.py",
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Entry point called by the dispatcher or directly via `modal run`.

        Args:
            ticker    : Ticker symbol, e.g. "AAPL"
            strategy  : Strategy code, e.g. "ORB" or "VWAP"
            script    : Relative path to backtest script from project root
            job_id    : Optional UUID for tracking; generated if omitted

        Returns:
            Result dict (see _run_backtest_local for schema).
        """
        if job_id is None:
            job_id = str(uuid.uuid4())[:8]

        print(f"[modal] Starting job {job_id}: {ticker}/{strategy} via {script}")
        result = _run_backtest_local(
            script=script,
            ticker=ticker,
            strategy=strategy,
            job_id=job_id,
            project_root="/workspace",
        )
        print(f"[modal] Completed job {job_id}: status={result['status']} "
              f"elapsed={result['elapsed_sec']}s cost_est=${result['cost_usd_est']:.6f}")

        # Write result.json inside the container (for artifact collection if needed)
        out_dir = DEFAULT_RESULT_DIR / ticker / strategy
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "result.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        return result

    # -----------------------------------------------------------------------
    # Local entrypoint (used by `modal run scripts/modal_worker.py`)
    # -----------------------------------------------------------------------
    @app.local_entrypoint()
    def main(
        ticker: str = "AAPL",
        strategy: str = "ORB",
        script: str = "scripts/backtest_xgb_v8.py",
        job_id: str = "",
    ) -> None:
        """
        CLI entrypoint for `modal run scripts/modal_worker.py --ticker X --strategy Y`.
        """
        if not job_id:
            job_id = str(uuid.uuid4())[:8]
        print(f"Dispatching job {job_id} to Modal: {ticker}/{strategy}")
        result = run_backtest.remote(
            ticker=ticker,
            strategy=strategy,
            script=script,
            job_id=job_id,
        )
        print(json.dumps(result, indent=2))

        # Write back to local Drive path so dispatcher can detect completion
        project_root = Path(os.environ.get(PROJECT_ROOT_ENV, Path(__file__).parent.parent))
        result_dir  = project_root / "backtests" / ticker / strategy
        result_dir.mkdir(parents=True, exist_ok=True)
        result_path = result_dir / "result.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Result written to {result_path}")


# ---------------------------------------------------------------------------
# Batch map helper — dispatch many jobs in parallel from local machine
# ---------------------------------------------------------------------------
def dispatch_batch(jobs: list[Dict[str, str]]) -> list[Dict[str, Any]]:
    """
    Submit a list of job dicts [{ticker, strategy, script, job_id}, ...] to Modal
    in parallel using .map(). Each call runs in its own container simultaneously.

    Called by the dispatcher's _submit_modal path when bulk dispatch is needed.
    Returns list of result dicts.

    Example:
        jobs = [
            {"ticker": "AAPL", "strategy": "ORB", "script": "scripts/backtest_xgb_v8.py"},
            {"ticker": "MSFT", "strategy": "VWAP", "script": "scripts/backtest_xgb_v8.py"},
        ]
        results = dispatch_batch(jobs)
    """
    if not _MODAL_AVAILABLE:
        raise ImportError("modal package not installed. Run: pip install modal")

    # Unpack into parallel argument lists for .starmap()
    results = list(
        run_backtest.starmap(
            [
                (
                    j["ticker"],
                    j["strategy"],
                    j.get("script", "scripts/backtest_xgb_v8.py"),
                    j.get("job_id", str(uuid.uuid4())[:8]),
                )
                for j in jobs
            ]
        )
    )
    return results


# ---------------------------------------------------------------------------
# Standalone local mock (no Modal SDK needed — for unit tests / dry-run)
# ---------------------------------------------------------------------------
def _mock_run(ticker: str, strategy: str,
              script: str = "scripts/backtest_xgb_v8.py",
              job_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Simulates the modal result dict without making any Modal API calls.
    Used by the dispatcher's --simulate and --dry-run modes.
    """
    job_id = job_id or str(uuid.uuid4())[:8]
    return {
        "job_id":       job_id,
        "ticker":       ticker,
        "strategy":     strategy,
        "cloud":        "modal",
        "status":       "completed",
        "returncode":   0,
        "stdout_tail":  f"[mock] {ticker}/{strategy} completed",
        "stderr_tail":  "",
        "elapsed_sec":  0.001,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "result":       {"sharpe": 1.23, "total_return": 0.087, "mock": True},
        "cost_usd_est": 0.0,
    }


# ---------------------------------------------------------------------------
# CLI shim (direct invocation without Modal SDK for testing)
# ---------------------------------------------------------------------------
if __name__ == "__main__" and not _MODAL_AVAILABLE:
    parser = argparse.ArgumentParser(description="Modal worker (local mock mode)")
    parser.add_argument("--ticker",   required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--script",   default="scripts/backtest_xgb_v8.py")
    parser.add_argument("--job-id",   default="")
    args = parser.parse_args()
    result = _mock_run(args.ticker, args.strategy, args.script, args.job_id or None)
    print(json.dumps(result, indent=2))
