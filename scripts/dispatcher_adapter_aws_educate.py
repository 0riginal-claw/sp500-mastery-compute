"""dispatcher_adapter_aws_educate.py — AWS Educate Starter Account adapter stub.

Free tier (verified 2026-05-17): $75 (basic) or $100 (institution-affiliated)
in AWS credits per year for verified students/educators. NO credit card required.
Renewable annually while enrolled.

Signup: https://aws.amazon.com/education/awseducate/ — register with school
email or upload proof of student status. Approval typically within 30 minutes
to 3 days. Credits expire 1 year from issue.

Auth: AWS Educate Starter accounts run inside a sandboxed AWS sub-account
managed via qwikLABS. Use the AWS Educate Vocareum portal to spin up a session,
then AWS keys are issued for the duration of that session.

Submit model: Once you have the session AWS keys, dispatch is identical to the
existing `aws_free` adapter — Lambda invoke, ECS run-task, or EC2 launch.
This adapter is therefore a CONFIGURATION WRAPPER, not a separate code path:
it pulls AWS_EDUCATE_* env vars and rotates them into AWS_ACCESS_KEY_ID /
AWS_SECRET_ACCESS_KEY before delegating to the existing aws_free dispatcher.

KEY ADVANTAGES:
 - $75-100/yr free, no CC, renewable annually.
 - Full AWS Lambda + EC2 access within the sandbox.
 - Stackable with personal AWS free tier on a different account.

LIMITATIONS:
 - Session-based credentials expire (typically 3-9 hours per session).
 - Some AWS services blocked in sandbox (e.g., higher-tier GPU instances).
 - Credits do not stack with other AWS promo programs.
 - Hard cap at $75-100/yr — exceeding it freezes the account until annual reset.
 - Requires manual session refresh (qwikLABS does not expose a long-lived API key).

Docs:
 - https://aws.amazon.com/education/awseducate/
 - https://www.knowledgehut.com/blog/cloud-computing/ways-to-get-aws-credits
"""
from __future__ import annotations

import os
import json


def _has_session_creds() -> bool:
    return all(os.environ.get(k) for k in
               ("AWS_EDUCATE_ACCESS_KEY", "AWS_EDUCATE_SECRET_KEY", "AWS_EDUCATE_SESSION_TOKEN"))


def _rotate_into_aws_env() -> None:
    """Copy AWS_EDUCATE_* session creds into the standard AWS env vars so the
    existing aws_free adapter can pick them up."""
    os.environ["AWS_ACCESS_KEY_ID"] = os.environ["AWS_EDUCATE_ACCESS_KEY"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ["AWS_EDUCATE_SECRET_KEY"]
    os.environ["AWS_SESSION_TOKEN"] = os.environ["AWS_EDUCATE_SESSION_TOKEN"]
    if os.environ.get("AWS_EDUCATE_REGION"):
        os.environ["AWS_DEFAULT_REGION"] = os.environ["AWS_EDUCATE_REGION"]


def submit_job(ticker: str, strategy: str, job_spec: dict, dry_run: bool = False) -> dict:
    if not _has_session_creds():
        return {"job_id": None, "status": "no_session",
                "note": ("AWS_EDUCATE_ACCESS_KEY / SECRET_KEY / SESSION_TOKEN not set. "
                         "Refresh via the Vocareum console (sessions expire ~3-9h).")}
    if dry_run:
        return {"job_id": "DRY-RUN", "status": "would_submit",
                "delegates_to": "dispatcher_adapter_aws_free",
                "ticker": ticker, "strategy": strategy, "job_spec": job_spec}
    _rotate_into_aws_env()
    # Delegate to the existing aws_free adapter
    try:
        from . import dispatcher_adapter_aws_free as aws_free  # type: ignore
    except ImportError:
        import importlib
        aws_free = importlib.import_module("dispatcher_adapter_aws_free")
    return aws_free.submit_job(ticker, strategy, job_spec, dry_run=False)


def check_status(job_id: str) -> dict:
    if not _has_session_creds():
        return {"status": "no_session"}
    _rotate_into_aws_env()
    try:
        from . import dispatcher_adapter_aws_free as aws_free  # type: ignore
    except ImportError:
        import importlib
        aws_free = importlib.import_module("dispatcher_adapter_aws_free")
    return aws_free.check_status(job_id)


if __name__ == "__main__":
    print(json.dumps(submit_job("AAPL", "D1_REV", {"thresh": 0.5}, dry_run=True), indent=2))
