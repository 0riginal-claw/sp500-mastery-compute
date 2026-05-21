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
  - This file uses the modal >= 1.0 API (App-based, no Stub, no modal.Mount).
  - Local code is bundled via Image.add_local_dir() / add_local_python_source()
    chained onto the Image definition — the deprecated modal.Mount API and the
    mounts=[] parameter on @app.function were removed in Modal 1.x.
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
# Constants  (defined early — used in image definition below)
# ---------------------------------------------------------------------------
PROJECT_ROOT_ENV = "SP500_PROJECT_ROOT"
DEFAULT_RESULT_DIR = Path("/tmp/sp500_results")

# Estimated cost per job (0.5 CPU-min at $0.0471/vCPU-hr = ~$0.0004)
COST_PER_JOB_USD = 0.0004

# Hard job-level timeout — prevents runaway containers from burning credit
JOB_TIMEOUT_SEC = 300   # 5 minutes; single-ticker backtests finish in <30 s


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
    # Determine project root at define-time (used for add_local_dir below).
    _project_root = Path(
        os.environ.get(PROJECT_ROOT_ENV, str(Path(__file__).parent.parent))
    )

    # Build image: base Python 3.12, pip-install packages, then bundle the
    # scripts/ directory into /workspace/scripts inside the container.
    #
    # We mount only scripts/ (not the full project root) because the project
    # workspace is a live environment — multiple daemons constantly write to
    # dashboard/, sweeps/, deepseek_workers/, logs/, etc. Mounting the full
    # root triggers modal.ExecutionError("file modified during build process").
    # Scripts are static at deploy time, so this is safe.
    #
    # If the backtest scripts need data files, mount additional static dirs:
    #     .add_local_dir(str(_project_root / "data-index"), "/workspace/data-index", copy=True)
    _scripts_dir = _project_root / "scripts"
    _registry_dir = _project_root / "registry"
    # autosolve_skip: image build patch — leaf-task

    _image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(_PACKAGES)
        .add_local_dir(
            str(_scripts_dir),
            remote_path="/workspace/scripts",
            copy=True,
        )
    )
    # Add registry/ if present — needed for xsec mega-job which reads
    # registry/sp500_tickers.csv. Build is robust to missing dir during dev.
    if _registry_dir.exists():
        _image = _image.add_local_dir(
            str(_registry_dir),
            remote_path="/workspace/registry",
            copy=True,
        )

    app = modal.App(
        name="sp500-xgb-sweep",
        image=_image,
    )
else:
    # Stubs for local import without Modal SDK
    _project_root = Path(__file__).parent.parent
    app = None  # type: ignore
    _image = None  # type: ignore


# ---------------------------------------------------------------------------
# Core backtest runner (cloud-side logic)
# ---------------------------------------------------------------------------
def _run_backtest_local(
    script: str,
    ticker: str,
    strategy: str,
    job_id: str,
    project_root: Optional[str] = None,
    extra_env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Executes the backtest script as a subprocess inside the container.

    The script is expected to accept:
        python <script> --ticker TICKER --strategy STRATEGY [--job-id JOB_ID]
    and write its output to stdout as JSON on the last line (or exit 0 on success).

    Args:
        extra_env: Optional per-job env overrides (XGB_NO_TOPK, etc.) merged
                   into the worker subprocess env. Added 2026-05-20.

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

    # Per-job env overrides (added 2026-05-20): start from container env,
    # overlay extra_env so per-job XGB_NO_TOPK / INTERACTION_CONSTRAINTS /
    # MONOTONIC_CONSTRAINTS / etc. land in the worker process.
    worker_env = os.environ.copy()
    if extra_env:
        for k, v in extra_env.items():
            if k and v is not None:
                worker_env[str(k)] = str(v)
        print(f"[modal] extra_env applied ({len(extra_env)} keys): "
              f"{sorted(extra_env.keys())}")

    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=JOB_TIMEOUT_SEC,
            cwd=str(root),
            env=worker_env,
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

    # Tier-S #7 (2026-05-21): Memory Snapshots + keep_warm pool.
    # - enable_memory_snapshot=True: Modal snapshots the post-import CPU/RAM
    #   state of the container and forks future cold starts from it (3-10x
    #   faster cold-start; cuts the ~6-8s import cost of xgboost/sklearn/pandas).
    # - min_containers=2: keeps 2 warm containers idle so fan-out bursts hit
    #   warm pool first (zero cold-start latency for the first 2 jobs).
    # - scaledown_window=300: idle containers live 5 min before scaledown,
    #   amortizing warm-pool cost across bursty sweeps. (Modal SDK ≥0.74:
    #   `container_idle_timeout` was renamed to `scaledown_window` and
    #   `keep_warm` was renamed to `min_containers`.)
    # If the running Modal SDK doesn't support these knobs, the decorator
    # call will TypeError at import time — fail-loud is desired so the
    # operator sees the SDK-version mismatch immediately.
    @app.function(
        cpu=1.0,
        memory=768,          # MB — our jobs use ~200 MB; headroom for imports
        timeout=JOB_TIMEOUT_SEC + 30,
        retries=modal.Retries(
            max_retries=2,
            backoff_coefficient=2.0,
            initial_delay=5.0,
        ),
        # Tier-S #7: memory snapshot + warm pool
        enable_memory_snapshot=True,
        min_containers=2,
        scaledown_window=300,
        # NOTE: mounts=[] was removed in Modal 1.x.  Local code is now bundled
        # into the image via .add_local_dir() above; no mounts= needed here.
        #
        # Secrets: uncomment once you create the secret in Modal:
        #   modal secret create sp500-secrets MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=...
        # Then replace the empty secrets=[] below with:
        #   secrets=[modal.Secret.from_name("sp500-secrets")],
        secrets=[],
    )
    def run_backtest(
        ticker: str,
        strategy: str,
        script: str = "scripts/backtest_xgb_v8.py",
        job_id: Optional[str] = None,
        extra_env: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Entry point called by the dispatcher or directly via `modal run`.

        Args:
            ticker    : Ticker symbol, e.g. "AAPL"
            strategy  : Strategy code, e.g. "ORB" or "VWAP"
            script    : Relative path to backtest script from project root
            job_id    : Optional UUID for tracking; generated if omitted
            extra_env : Optional per-job env overrides merged into worker env
                        (added 2026-05-20). Used for XGB_NO_TOPK,
                        INTERACTION_CONSTRAINTS_JSON, MONOTONIC_CONSTRAINTS_JSON
                        and other per-job tuning knobs.

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
            extra_env=extra_env,
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
        extra_env_json: str = "",
    ) -> None:
        """
        CLI entrypoint for `modal run scripts/modal_worker.py --ticker X --strategy Y`.

        --extra-env-json '{"XGB_NO_TOPK":"1","INTERACTION_CONSTRAINTS_JSON":"..."}'
        (added 2026-05-20) forwards per-job env overrides into the remote
        run_backtest call so the worker subprocess inside the Modal container
        sees them.
        """
        if not job_id:
            job_id = str(uuid.uuid4())[:8]
        # Decode extra_env_json once on the local side — invalid JSON is a
        # dispatcher bug, fail loud so the operator sees it immediately.
        extra_env: Optional[Dict[str, str]] = None
        if extra_env_json:
            try:
                parsed = json.loads(extra_env_json)
                if isinstance(parsed, dict):
                    extra_env = {str(k): str(v) for k, v in parsed.items()
                                 if k and v is not None}
                    print(f"[modal] extra_env decoded ({len(extra_env)} keys): "
                          f"{sorted(extra_env.keys())}")
                else:
                    print(f"[modal] WARN: --extra-env-json is not a dict "
                          f"(type={type(parsed).__name__}) — ignoring")
            except json.JSONDecodeError as exc:
                print(f"[modal] ERROR: invalid --extra-env-json: {exc} — ignoring")

        print(f"Dispatching job {job_id} to Modal: {ticker}/{strategy}")
        result = run_backtest.remote(
            ticker=ticker,
            strategy=strategy,
            script=script,
            job_id=job_id,
            extra_env=extra_env,
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

    # -----------------------------------------------------------------------
    # XSEC mega-job entrypoint — full S&P 500 cross-sectional pooling.
    # Added 2026-05-20. Uses A10G GPU + 32 GB RAM (vs the per-ticker function's
    # cpu=1.0/mem=768MB), longer timeout, and CLI shape that backtest_xgb_v10_xsec.py
    # expects: --tickers <comma> OR --tickers-file <path> + --output-dir.
    # autosolve_skip: leaf-task xsec entrypoint add — no error condition
    # -----------------------------------------------------------------------
    XSEC_JOB_TIMEOUT_SEC = 3600  # 1 hour — full-500 training on A10G expected 30–60 min

    @app.function(
        gpu="A10G",
        cpu=4.0,
        memory=32768,          # 32 GB — required for 500-ticker × 1633-feature panel
        timeout=XSEC_JOB_TIMEOUT_SEC + 60,
        retries=modal.Retries(
            max_retries=1,
            backoff_coefficient=2.0,
            initial_delay=10.0,
        ),
        secrets=[],
    )
    def run_xsec_backtest(
        tickers_csv: str,
        strategy: str,
        job_id: str,
        script: str = "scripts/backtest_xgb_v10_xsec.py",
        extra_env: Optional[Dict[str, str]] = None,
        wf_train_days: int = 252,
        wf_test_days: int = 21,
        wf_stride_days: int = 21,
    ) -> Dict[str, Any]:
        """
        XSEC mega-job entrypoint (full-500 cross-sectional XGB training).

        Args:
            tickers_csv  : Either a comma-separated ticker list
                           ("AAPL,MSFT,...") OR a path to a tickers file
                           (csv/txt) inside the container's /workspace/scripts
                           or sibling registry/ dir.
            strategy     : Strategy code (used for naming + run_meta).
            job_id       : Job id for tracking / artifact dir naming.
            script       : Path to xsec backtest script (default v10).
            extra_env    : Per-job env overrides (XGB_DEVICE=cuda etc.)
            wf_*_days    : Days-based walk-forward windows.

        Returns dict with status, returncode, stdout/stderr tails, elapsed,
        and parsed run_meta.json if present in --output-dir.
        """
        root = Path("/workspace")
        out_dir = root / "artifacts" / f"v10_xsec_full500_{job_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Distinguish comma-list vs file path. A path doesn't contain commas
        # and either starts with / or matches a known file inside the container.
        tickers_arg: list[str]
        if "," in tickers_csv:
            tickers_arg = ["--tickers", tickers_csv]
        else:
            # File path — resolve relative to /workspace if not absolute.
            tf = Path(tickers_csv)
            if not tf.is_absolute():
                # Common case: caller passes "registry/sp500_tickers.csv";
                # the registry/ dir lives outside scripts/ which is the only
                # dir we mount, so we expect the dispatcher / caller to have
                # added it. If missing, fall back to scripts/-side csv.
                candidate_paths = [
                    root / tf,
                    root / "scripts" / tf.name,
                    Path("/workspace") / "scripts" / tf.name,
                ]
                resolved = next((p for p in candidate_paths if p.exists()),
                                root / tf)
                tf = resolved
            tickers_arg = ["--tickers-file", str(tf)]

        cmd = [
            sys.executable,
            str(root / script),
            *tickers_arg,
            "--output-dir", str(out_dir),
            "--strategy", strategy,
            "--job-id", job_id,
            "--wf-train-days", str(wf_train_days),
            "--wf-test-days", str(wf_test_days),
            "--wf-stride-days", str(wf_stride_days),
        ]

        worker_env = os.environ.copy()
        if extra_env:
            for k, v in extra_env.items():
                if k and v is not None:
                    worker_env[str(k)] = str(v)
            print(f"[modal][xsec] extra_env applied ({len(extra_env)} keys): "
                  f"{sorted(extra_env.keys())}")

        print(f"[modal][xsec] Starting xsec job {job_id} strategy={strategy} "
              f"out={out_dir}")
        print(f"[modal][xsec] cmd: {' '.join(cmd)}")

        start = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=XSEC_JOB_TIMEOUT_SEC,
                cwd=str(root),
                env=worker_env,
            )
            elapsed = time.monotonic() - start
            stdout = proc.stdout[-5000:] if proc.stdout else ""
            stderr = proc.stderr[-5000:] if proc.stderr else ""
            status = "completed" if proc.returncode == 0 else "failed"
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            elapsed = XSEC_JOB_TIMEOUT_SEC
            stdout = ""
            stderr = f"xsec job timed out after {XSEC_JOB_TIMEOUT_SEC}s"
            status = "failed"
            returncode = -1

        # Try to parse run_meta.json if the script wrote one
        run_meta: Optional[Dict] = None
        run_meta_path = out_dir / "run_meta.json"
        if run_meta_path.exists():
            try:
                with open(run_meta_path) as f:
                    run_meta = json.load(f)
            except Exception as exc:
                print(f"[modal][xsec] WARN: failed to parse run_meta.json: {exc}")

        print(f"[modal][xsec] Completed job {job_id}: status={status} "
              f"elapsed={elapsed:.1f}s rc={returncode}")

        return {
            "job_id":       job_id,
            "ticker":       "ALL",
            "strategy":     strategy,
            "cloud":        "modal",
            "kind":         "xsec",
            "status":       status,
            "returncode":   returncode,
            "stdout_tail":  stdout,
            "stderr_tail":  stderr,
            "elapsed_sec":  round(elapsed, 3),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "output_dir":   str(out_dir),
            "run_meta":     run_meta,
            # A10G + 32GB pricing — rough $1.10/hr GPU + $0.45/hr CPU+mem
            "cost_usd_est": round(elapsed / 3600 * 1.55, 4),
        }

    @app.local_entrypoint()
    def xsec(
        tickers_csv: str = "registry/sp500_tickers.csv",
        strategy: str = "ML_XGB_v10_xsec_full500",
        job_id: str = "",
        script: str = "scripts/backtest_xgb_v10_xsec.py",
        extra_env_json: str = "",
        wf_train_days: int = 252,
        wf_test_days: int = 21,
        wf_stride_days: int = 21,
    ) -> None:
        """
        CLI entrypoint for full-S&P-500 xsec backtest.

        Usage:
            modal run scripts/modal_worker.py::xsec \\
              --tickers-csv "registry/sp500_tickers.csv" \\
              --strategy ML_XGB_v10_xsec_full500 \\
              --job-id <job_id> \\
              --extra-env-json '{"XGB_DEVICE":"cuda",...}'
        """
        if not job_id:
            job_id = str(uuid.uuid4())[:8]
        extra_env: Optional[Dict[str, str]] = None
        if extra_env_json:
            try:
                parsed = json.loads(extra_env_json)
                if isinstance(parsed, dict):
                    extra_env = {str(k): str(v) for k, v in parsed.items()
                                 if k and v is not None}
                    print(f"[modal][xsec] extra_env decoded ({len(extra_env)} keys): "
                          f"{sorted(extra_env.keys())}")
                else:
                    print(f"[modal][xsec] WARN: --extra-env-json not a dict — ignoring")
            except json.JSONDecodeError as exc:
                print(f"[modal][xsec] ERROR invalid --extra-env-json: {exc}")

        print(f"[modal][xsec] Dispatching xsec job {job_id} strategy={strategy}")
        result = run_xsec_backtest.remote(
            tickers_csv=tickers_csv,
            strategy=strategy,
            job_id=job_id,
            script=script,
            extra_env=extra_env,
            wf_train_days=wf_train_days,
            wf_test_days=wf_test_days,
            wf_stride_days=wf_stride_days,
        )
        # Truncate stdout/stderr tails for printable summary
        summary = {k: v for k, v in result.items()
                   if k not in ("stdout_tail", "stderr_tail")}
        print(json.dumps(summary, indent=2, default=str))

        # Persist to local Drive results dir
        project_root = Path(os.environ.get(PROJECT_ROOT_ENV,
                                           Path(__file__).parent.parent))
        result_dir = project_root / "backtests" / "ALL" / strategy
        result_dir.mkdir(parents=True, exist_ok=True)
        result_path = result_dir / f"result_{job_id}.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"[modal][xsec] Result written to {result_path}")

    # -----------------------------------------------------------------------
    # Smoke-test entrypoint — trivial echo workload, no real backtest.
    # Usage: HOME=/Users/orginal modal run scripts/modal_worker.py::echo_smoke
    # -----------------------------------------------------------------------
    @app.function(
        cpu=0.25,
        memory=256,
        timeout=60,
    )
    def _echo_remote(msg: str) -> Dict[str, Any]:
        """Trivial remote function that echoes its input — used for smoke tests."""
        import platform
        return {
            "echo": msg,
            "python": platform.python_version(),
            "status": "ok",
        }

    @app.local_entrypoint()
    def echo_smoke() -> None:
        """
        Zero-cost smoke test: boots a tiny container and echoes a string.
        Verifies the app definition, image build, and network path work end-to-end
        without running any real backtest or consuming meaningful Modal credit.

        Usage:
            HOME=/Users/orginal modal run scripts/modal_worker.py::echo_smoke
        """
        payload = f"smoke-{str(uuid.uuid4())[:8]}"
        print(f"[echo_smoke] Sending: {payload!r}")
        result = _echo_remote.remote(payload)
        print(f"[echo_smoke] Remote returned: {json.dumps(result, indent=2)}")
        assert result.get("status") == "ok", f"Unexpected result: {result}"
        assert result.get("echo") == payload, "Echo mismatch"
        print("[echo_smoke] PASS — Modal worker boots and responds correctly.")


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
