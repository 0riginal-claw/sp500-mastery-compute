#!/usr/bin/env python3
"""gh_app_repo_creator.py — create a repo via a GitHub App installation token.

Status (2026-05-18): READY but UNARMED.
No GitHub App credentials are stored locally. Audited 2026-05-18 — see
reports/gh_repo_create_repo_2026-05-18.md. Until the user installs a
GitHub App on the `0riginal-claw` account and drops the PEM + app_id +
installation_id into ~/.config/secrets/github_app/, this script returns
exit 2 with a "not configured" message.

Once configured, this script can mint an installation access token without
any browser interaction and create the target repo (subject to App
"Administration: Write" permission on the user account).

Expected layout:
    ~/.config/secrets/github_app/app.json   # {"app_id": 123, "installation_id": 456}
    ~/.config/secrets/github_app/key.pem    # PKCS#1 or PKCS#8 private key

Usage:
    python gh_app_repo_creator.py --owner 0riginal-claw \
        --name sp500-mastery-compute --public --description "..." [--auto-init]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import urllib.request
import urllib.error

CFG_DIR = Path.home() / ".config" / "secrets" / "github_app"
APP_JSON = CFG_DIR / "app.json"
KEY_PEM = CFG_DIR / "key.pem"

GITHUB_API = "https://api.github.com"


def _api(method: str, path: str, token: str, body: Dict[str, Any] | None = None) -> Dict[str, Any]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"{GITHUB_API}{path}",
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "gh_app_repo_creator/1.0",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode()
        try:
            return {"_error": True, "_status": exc.code, **json.loads(raw)}
        except Exception:
            return {"_error": True, "_status": exc.code, "message": raw}


def _load_creds() -> Dict[str, Any]:
    if not APP_JSON.exists() or not KEY_PEM.exists():
        return {}
    cfg = json.loads(APP_JSON.read_text())
    cfg["pem"] = KEY_PEM.read_text()
    return cfg


def _mint_jwt(app_id: int, pem: str) -> str:
    try:
        import jwt  # PyJWT
    except ImportError:
        sys.stderr.write(
            "[fatal] PyJWT not installed. Run: pip install 'PyJWT[crypto]'\n"
        )
        sys.exit(2)
    now = int(time.time())
    payload = {"iat": now - 60, "exp": now + 540, "iss": str(app_id)}
    return jwt.encode(payload, pem, algorithm="RS256")


def _installation_token(app_jwt: str, installation_id: int) -> str:
    resp = _api(
        "POST",
        f"/app/installations/{installation_id}/access_tokens",
        token=app_jwt,
    )
    if resp.get("_error"):
        raise RuntimeError(f"failed to mint installation token: {resp}")
    return resp["token"]


def create_repo(owner: str, name: str, *, public: bool, description: str, auto_init: bool) -> Dict[str, Any]:
    cfg = _load_creds()
    if not cfg:
        sys.stderr.write(
            "[not-configured] No GitHub App credentials at "
            f"{CFG_DIR}. See reports/gh_repo_create_repo_2026-05-18.md.\n"
            "Falling back: use scripts/gh_repo_create_manual_helper.md instead.\n"
        )
        sys.exit(2)

    app_jwt = _mint_jwt(int(cfg["app_id"]), cfg["pem"])
    inst_token = _installation_token(app_jwt, int(cfg["installation_id"]))

    body = {
        "name": name,
        "description": description,
        "private": not public,
        "auto_init": auto_init,
        "has_issues": False,
        "has_projects": False,
        "has_wiki": False,
    }
    # GitHub App user-installed: POST /user/repos works; for org-installed:
    # POST /orgs/{owner}/repos. We branch on owner type.
    user_resp = _api("GET", f"/users/{owner}", token=inst_token)
    if user_resp.get("type") == "Organization":
        return _api("POST", f"/orgs/{owner}/repos", token=inst_token, body=body)
    # User-account repos via app: requires the user installation token to be
    # scoped to the user, AND the App must have "Administration: Write" on
    # the user account. App spec: https://docs.github.com/en/rest/repos/repos
    return _api("POST", "/user/repos", token=inst_token, body=body)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--owner", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--description", default="")
    ap.add_argument("--auto-init", action="store_true")
    args = ap.parse_args()

    result = create_repo(
        owner=args.owner,
        name=args.name,
        public=args.public,
        description=args.description,
        auto_init=args.auto_init,
    )
    if result.get("_error"):
        sys.stderr.write(f"[error] {result}\n")
        return 1
    print(json.dumps({"created": True, "full_name": result.get("full_name"), "html_url": result.get("html_url")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
