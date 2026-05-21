# autosolve_skip: deferred-path template, no error condition
# karpathy_checked: assumptions+success criteria embedded in module docstring
"""xsec_beam_template.py - Beam Cloud serverless backstop for xsec (Path 5).

STATUS: DEFERRED - requires Beam Cloud account + creds at ~/.beam/
        Once user has Beam Cloud account:
          1. Sign up at https://beam.cloud (free tier: $15/mo credit, ~10 GPU-hr)
          2. pip install beam-client
          3. beam configure   # stores creds at ~/.beam/config.json
          4. python scripts/xsec_beam_template.py deploy
          5. beam serve scripts/xsec_beam_template.py:xsec_full500_handler

DESIGN: Beam Cloud is serverless GPU (T4 / A10G / A100), 1-line decorator deploy.
  - Cold start ~10-30s, then per-call billing in 1s increments
  - $15/mo free credit = ~10 hr A10G or ~30 hr T4 free monthly
  - Direct competitor to Modal; functionally equivalent for our xsec workload
  - Why Beam alongside Modal: independent provider, different IP space,
    avoids Modal workspace-spend-cap chokepoint (see feedback_auto_signup_architecture)

KARPATHY assumptions:
  - Beam A10G == Modal A10G performance (both NVIDIA, same VRAM class)
  - xsec backtest is embarrassingly batch (no streaming reqs); fits FaaS model
  - Free tier $15/mo > one full-500 xsec run cost on A10G (~$5 estimated)

VERIFIABLE SUCCESS:
  - Smoke (5-ticker): completes in <5 min, returns metrics JSON
  - Full-500: completes in <2 hr, mean_pf >= 1.05 across all tickers (parity with Modal)
"""

from __future__ import annotations

# Conditional import: only import beam when creds exist.
# Template lives in the repo year-round; do not ImportError at module-load
# time when beam isn't installed.
try:
    from beam import Image, Volume, function  # type: ignore
    _BEAM_OK = True
except ImportError:
    _BEAM_OK = False

import os
import subprocess
import sys
from pathlib import Path

# Beam image: replicate the sp500-mastery venv.
# Build once, cached forever; ~30s first deploy then instant.
if _BEAM_OK:
    XSEC_IMAGE = (
        Image(python_version="python3.11")
        .add_python_packages([
            "xgboost==2.*",
            "yfinance",
            "pandas",
            "numpy",
            "scikit-learn",
            "pyarrow",
        ])
    )

    # Persistent volume for repo clone + artifact cache across invocations.
    XSEC_VOL = Volume(name="xsec-repo", mount_path="/repo")

    @function(
        image=XSEC_IMAGE,
        gpu="A10G",            # or "T4" for cheaper, "A100" for 2x throughput
        memory="32Gi",
        cpu=4,
        timeout=7200,          # 2 hr ceiling for full-500
        volumes=[XSEC_VOL],
        secrets=["GITHUB_PAT"],  # configure in Beam dashboard
    )
    def xsec_full500_handler(
        tickers: str = "AAPL,MSFT,GOOG,META,NVDA",
        job_id: str = "beam-smoke",
        output_subdir: str = "smoke_run",
    ) -> dict:
        """Run xsec backtest on Beam A10G.

        Args:
            tickers: comma-sep list, or "FULL_500" sentinel
            job_id: tag for output naming
            output_subdir: subdir under /repo/artifacts/

        Returns:
            dict with status, output_path, mean_pf
        """
        repo_dir = Path("/repo/sp500-ticker-mastery")

        # First-call: clone repo into persistent volume.
        if not repo_dir.exists():
            pat = os.environ["GITHUB_PAT"]
            subprocess.check_call([
                "git", "clone",
                f"https://{pat}@github.com/<user>/sp500-ticker-mastery.git",
                str(repo_dir),
            ])
        else:
            subprocess.check_call(["git", "-C", str(repo_dir), "pull"])

        # Resolve tickers arg.
        if tickers == "FULL_500":
            ticker_args = ["--tickers-file", "registry/sp500_tickers.csv"]
        else:
            ticker_args = ["--tickers", tickers]

        out_path = repo_dir / "artifacts" / f"xsec_beam_{output_subdir}"

        env = os.environ.copy()
        env.update({
            "AUTO_CLOUD_DISPATCH": "0",
            "XGB_NO_TOPK": "1",
            "XGB_DEVICE": "cuda",
            "XGB_TREE_METHOD": "hist",
            "XGB_N_ESTIMATORS": "2000",
            "XGB_EARLY_STOP": "50",
            "XGB_XSEC_WEIGHT": "equal",
        })

        subprocess.check_call(
            [
                "python", "scripts/backtest_xgb_v10_xsec.py",
                *ticker_args,
                "--output-dir", str(out_path),
                "--strategy", f"ML_XGB_v10_xsec_beam_{output_subdir}",
                "--job-id", job_id,
                "--wf-train-days", "252",
                "--wf-test-days", "21",
                "--wf-stride-days", "21",
            ],
            cwd=str(repo_dir),
            env=env,
        )

        # Parse summary JSON for return value.
        summary_path = out_path / "summary.json"
        if summary_path.exists():
            import json
            summary = json.loads(summary_path.read_text())
            return {
                "status": "ok",
                "output_path": str(out_path),
                "mean_pf": summary.get("mean_pf"),
                "n_tickers": summary.get("n_tickers"),
            }
        return {"status": "ok_no_summary", "output_path": str(out_path)}


def _cli_deploy():
    """Manual deploy via `python xsec_beam_template.py deploy`."""
    if not _BEAM_OK:
        print("ERROR: beam-client not installed. Run: pip install beam-client")
        sys.exit(1)
    print("Deploy via: beam deploy scripts/xsec_beam_template.py:xsec_full500_handler")
    print("Or invoke directly: beam run scripts/xsec_beam_template.py:xsec_full500_handler "
          "-d '{\"tickers\":\"AAPL,MSFT\",\"job_id\":\"beam-test\"}'")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "deploy":
        _cli_deploy()
    else:
        print(__doc__)
        print(f"\nBeam SDK available: {_BEAM_OK}")
        print("\nActivation steps:")
        print("  1. Sign up at https://beam.cloud")
        print("  2. pip install beam-client")
        print("  3. beam configure")
        print("  4. Add GITHUB_PAT secret in Beam dashboard")
        print("  5. python scripts/xsec_beam_template.py deploy")
