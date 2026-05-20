#!/usr/bin/env python3
"""
sync_compute_repo.py — Keep the GitHub Actions compute repo in sync with local mastery.

Iterates configured local source paths (scripts/*.py, .github/workflows/*.yml,
registry/*.csv), computes git-blob SHA-1, compares to the remote repo via the
GitHub Contents API, and PUTs an update when local differs. Runs as a launchd-
managed daemon on a 5-minute interval (StartInterval=300).

Why: cloud workers in 0riginal-claw/sp500-mastery-compute (target repo) must run
the same code as the local mastery dir. Without sync, cloud jobs run stale code
while local edits drift, leading to silent skew between local backtests and
dispatched runs.

Configuration (env, with defaults):
    GITHUB_OWNER         default "0riginal-claw"
    GITHUB_REPO          default "sp500-mastery-compute"
    GITHUB_BRANCH        default "main"
    GH_TOKEN/GITHUB_TOKEN required (gho_* or PAT with contents:write)
    SYNC_DRY_RUN         "1" = log actions, no PUT (default off)
    SYNC_INTERVAL_SEC    0 = one-shot (launchd handles cadence); >0 = internal loop

Behavior:
    - 404 on repo → log warning + exit clean (daemon idle until repo exists)
    - PUT failures → log + continue (don't abort the whole pass)
    - 0.2s sleep between PUTs to avoid secondary rate limits

Logs: $AI_ROOT/s&p500-ticker-mastery/logs/sync_compute_repo.log
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

AI_ROOT = Path(
    os.environ.get(
        "AI_ROOT",
        "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools",
    )
)
PROJECT_ROOT = AI_ROOT / "s&p500-ticker-mastery"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "sync_compute_repo.log"

OWNER = os.environ.get("GITHUB_OWNER", "0riginal-claw")
REPO = os.environ.get("GITHUB_REPO", "sp500-mastery-compute")
BRANCH = os.environ.get("GITHUB_BRANCH", "main")
TOKEN = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or ""
DRY_RUN = os.environ.get("SYNC_DRY_RUN", "0") == "1"
INTERVAL = int(os.environ.get("SYNC_INTERVAL_SEC", "0"))
PUT_DELAY_SEC = 0.2

SYNC_GLOBS = [
    ".github/workflows/*.yml",
    "scripts/**/*.py",            # recursive — pull nested scripts/*/foo.py
    "registry/*.csv",
]
# __init__.py files are excluded by default (package markers shouldn't sync to
# compute repo and overwrite remote scaffolding). However, the vendored
# historical_system package needs its __init__.py files synced because the
# scripts that import it require the package structure to be present remotely.
EXCLUDE_BASENAMES = {"__init__.py"}
ALLOWLIST_INIT_DIR_SUBSTR = {"scripts/historical_system"}
EXCLUDE_DIR_SUBSTR = {"__pycache__", "_generated", "_rescue"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] sync_compute_repo - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stderr)],
)
log = logging.getLogger()

API_BASE = "https://api.github.com"


def _gh_request(method, path, body=None):
    url = f"{API_BASE}{path}"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "sync-compute-repo/1.0",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
            try:
                return resp.status, json.loads(raw) if raw else None
            except json.JSONDecodeError:
                return resp.status, raw
    except urllib.error.HTTPError as e:
        try:
            err_body = json.loads(e.read())
        except Exception:
            err_body = None
        return e.code, err_body
    except urllib.error.URLError as e:
        log.error("Network error %s %s: %s", method, path, e)
        return 0, None


def repo_exists():
    status, _ = _gh_request("GET", f"/repos/{OWNER}/{REPO}")
    return status == 200


def get_remote_sha(remote_path):
    status, body = _gh_request(
        "GET", f"/repos/{OWNER}/{REPO}/contents/{remote_path}?ref={BRANCH}"
    )
    if status == 200 and isinstance(body, dict):
        return body.get("sha")
    return None


def put_remote_file(remote_path, content_bytes, existing_sha):
    payload = {
        "message": f"sync: {remote_path}",
        "content": base64.b64encode(content_bytes).decode("ascii"),
        "branch": BRANCH,
    }
    if existing_sha:
        payload["sha"] = existing_sha
    status, body = _gh_request(
        "PUT", f"/repos/{OWNER}/{REPO}/contents/{remote_path}", body=payload
    )
    if status in (200, 201):
        return True
    log.error("PUT %s failed status=%s body=%s", remote_path, status, body)
    return False


def sha1_blob_for_bytes(b):
    h = hashlib.sha1()
    h.update(f"blob {len(b)}\0".encode("ascii"))
    h.update(b)
    return h.hexdigest()


def enumerate_local():
    pairs = []
    seen = set()  # dedupe (recursive globs can yield overlap)
    for pattern in SYNC_GLOBS:
        for p in PROJECT_ROOT.glob(pattern):
            if not p.is_file():
                continue
            rel = p.relative_to(PROJECT_ROOT).as_posix()
            if rel in seen:
                continue
            if any(seg in rel for seg in EXCLUDE_DIR_SUBSTR):
                continue
            # __init__.py is excluded unless inside an allowlisted dir
            if p.name in EXCLUDE_BASENAMES:
                if not any(seg in rel for seg in ALLOWLIST_INIT_DIR_SUBSTR):
                    continue
            seen.add(rel)
            pairs.append((p, rel))
    return pairs


def run_once():
    summary = {"checked": 0, "synced": 0, "unchanged": 0, "failed": 0, "dry_run": DRY_RUN}
    if not TOKEN:
        log.error("No GH_TOKEN/GITHUB_TOKEN in env - aborting pass")
        summary["error"] = "no_token"
        return summary
    if not repo_exists():
        log.warning(
            "Repo %s/%s not accessible (404 or auth fail) - daemon idle until repo exists",
            OWNER, REPO,
        )
        summary["error"] = "repo_not_found"
        return summary

    pairs = enumerate_local()
    log.info("Enumerated %d local files for sync", len(pairs))

    for local, remote in pairs:
        summary["checked"] += 1
        try:
            local_bytes = local.read_bytes()
        except OSError as e:
            log.error("Cannot read %s: %s", local, e)
            summary["failed"] += 1
            continue

        local_sha = sha1_blob_for_bytes(local_bytes)
        remote_sha = get_remote_sha(remote)

        if remote_sha == local_sha:
            summary["unchanged"] += 1
            continue

        if DRY_RUN:
            log.info("[dry-run] would PUT %s (local=%s, remote=%s)",
                     remote, local_sha[:8], (remote_sha or "absent")[:8])
            summary["synced"] += 1
            continue

        ok = put_remote_file(remote, local_bytes, remote_sha)
        if ok:
            log.info("Synced %s (%s -> %s)", remote,
                     (remote_sha or "new")[:8], local_sha[:8])
            summary["synced"] += 1
        else:
            summary["failed"] += 1

        time.sleep(PUT_DELAY_SEC)

    log.info("Pass complete: checked=%d synced=%d unchanged=%d failed=%d",
             summary["checked"], summary["synced"], summary["unchanged"], summary["failed"])
    return summary


def main():
    log.info("sync_compute_repo starting - owner=%s repo=%s branch=%s dry_run=%s interval=%ds",
             OWNER, REPO, BRANCH, DRY_RUN, INTERVAL)
    if INTERVAL <= 0:
        run_once()
        return 0
    while True:
        try:
            run_once()
        except Exception as e:
            log.exception("Unhandled error in sync pass: %s", e)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    sys.exit(main())
