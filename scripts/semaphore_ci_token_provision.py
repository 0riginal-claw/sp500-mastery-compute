#!/usr/bin/env python3
"""
semaphore_ci_token_provision.py — Auto-generate a Semaphore CI API token via
headless GitHub OAuth login, then write to a local .env file and flip the
adapter to enabled=true in cloud_usage.json.

NO MANUAL PASTE. Token is read from the DOM after Semaphore renders it once.

Flow
----
0. PRE-FLIGHT (one-time): operator runs `--bootstrap-github` once. That opens
   a visible browser pinned to the persistent profile, navigates to
   github.com/login, and waits up to 5 min for the operator to complete the
   GitHub login (password + 2FA / Touch ID). The session cookie is then
   persisted inside the profile so subsequent headless runs auto-authorize.
   We deliberately do NOT scrape GitHub's password form headlessly: 2FA-bearing
   accounts break, and pasting credentials into a headless browser violates the
   `no credentials in third-party tools` safety rule.
1. Launch patchright (Playwright fork, MIT) Chromium with a persistent profile
   at /Users/orginal/.config/auto_signup/playwright_profiles/semaphore_ci/.
2. _verify_github_session() opens github.com/settings/profile in a throwaway
   tab; if it 302s to /login, exit 4 with a clear bootstrap instruction.
3. Navigate https://id.semaphoreci.com/login.
4. Click "Sign in with GitHub" — opens GitHub OAuth. With session present from
   step 0, GitHub auto-redirects (or shows a single "Authorize" button that we
   click).
5. After OAuth redirects back, Semaphore may force first-time org creation
   (URL /new). Auto-create an org named `signup-bot-<UTC>`.
6. Navigate to https://me.semaphoreci.com/account (the canonical API tokens
   page as of 2026). Click "New Token" / "Generate", read the token value
   from the DOM.
7. Write token to /Users/orginal/.config/auto_signup/semaphore_ci.env with
   chmod 600, format: SEMAPHORE_API_TOKEN="<token>"
   Also write SEMAPHORE_ORG="<org-slug>" so the adapter knows the server URL.
8. Flip semaphore_ci.enabled=true in cloud_usage.json.
9. Smoke test: curl GET /api/v1alpha/projects with the new token — expect 200.

Safety
------
- Token value is NEVER echoed (only length + last 4 chars).
- Env file is chmod 600.
- Path is /Users/orginal/.config/... (Mac-local, NOT the launcher-redirected
  $HOME which lives in Drive).
- Screenshots on failure go to AI-Tools/logs/auto_signup/.
- If CAPTCHA, phone-verification, or selector-miss occurs, exits with a
  distinct code so the parent's §8 logic spawns the right grandchild solver.
- Does NOT submit a phone number even if Semaphore asks — bails instead.

Exit codes
----------
 0  success (token captured + adapter flipped + smoke 200)
 1  fatal (env/dep missing, unrecoverable)
 2  CAPTCHA / human challenge — solver needed
 3  phone verification requested — alt-path needed
 4  selector miss / DOM drift — UI hunt needed
 5  smoke test failed (token captured but API rejected it)

Usage
-----
    # First-time bootstrap (one-shot, opens visible browser; operator logs into
    # GitHub manually with Touch ID / 2FA; session saved in persistent profile):
    python semaphore_ci_token_provision.py --bootstrap-github

    # Production run (headless, reuses GitHub session from bootstrap):
    python semaphore_ci_token_provision.py

    # Force regen (token already exists but you want a fresh one):
    python semaphore_ci_token_provision.py --force

    # Debug a failing run (shows browser):
    python semaphore_ci_token_provision.py --headed
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import re
import sys
import urllib.request
import urllib.error
from pathlib import Path

from patchright.async_api import async_playwright, TimeoutError as PlaywrightTimeout

# Shared CAPTCHA-solver integration. Helper lives one tree over under
# AI-Tools/scripts/ (NOT this s&p500-ticker-mastery/scripts/ dir).
_AI_TOOLS_SCRIPTS = Path("/Users/orginal/Library/CloudStorage/"
                        "GoogleDrive-zachgladstone@gmail.com/My Drive/"
                        "AI-Tools/scripts")
sys.path.insert(0, str(_AI_TOOLS_SCRIPTS))
try:
    from captcha_solver_helper import maybe_solve_captcha  # type: ignore
except ImportError:
    maybe_solve_captcha = None  # type: ignore

# Launcher redirects $HOME into Drive; secrets MUST live on Mac-local path.
MAC_HOME = Path("/Users/orginal")
CFG_DIR = MAC_HOME / ".config" / "auto_signup"
ENV_PATH = CFG_DIR / "semaphore_ci.env"
PROFILE_DIR = CFG_DIR / "playwright_profiles" / "semaphore_ci"
LOG_DIR = MAC_HOME / "Library" / "CloudStorage" / "GoogleDrive-zachgladstone@gmail.com" / "My Drive" / "AI-Tools" / "logs" / "auto_signup"
CLOUD_USAGE = MAC_HOME / "Library" / "CloudStorage" / "GoogleDrive-zachgladstone@gmail.com" / "My Drive" / "AI-Tools" / "s&p500-ticker-mastery" / "sweeps" / "cloud_usage.json"

for d in (CFG_DIR, PROFILE_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)


def _ts() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _log(msg: str) -> None:
    print(f"[{dt.datetime.now(dt.timezone.utc).isoformat()}] {msg}", file=sys.stderr, flush=True)


async def _shot(page, name: str) -> Path:
    p = LOG_DIR / f"semaphore_ci_provision_{_ts()}_{name}.png"
    try:
        await page.screenshot(path=str(p), full_page=True)
        _log(f"shot:{p}")
    except Exception as e:
        _log(f"shot-fail:{e}")
    return p


def _write_env(token: str, org_slug: str) -> None:
    """Write env file atomically with chmod 600. Never echoes token value."""
    lines = []
    if ENV_PATH.exists():
        for ln in ENV_PATH.read_text().splitlines():
            if ln.startswith("SEMAPHORE_API_TOKEN=") or ln.startswith("export SEMAPHORE_API_TOKEN="):
                continue
            if ln.startswith("SEMAPHORE_ORG=") or ln.startswith("export SEMAPHORE_ORG="):
                continue
            lines.append(ln)
    lines.append(f'SEMAPHORE_API_TOKEN="{token}"')
    lines.append(f'SEMAPHORE_ORG="{org_slug}"')
    tmp = ENV_PATH.with_suffix(".env.tmp")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.chmod(0o600)
    tmp.replace(ENV_PATH)
    ENV_PATH.chmod(0o600)


def _flip_cloud_usage(org_slug: str) -> None:
    """Set semaphore_ci.enabled=true and record the org server URL."""
    if not CLOUD_USAGE.exists():
        _log(f"WARN: cloud_usage.json not at {CLOUD_USAGE}; skipping flip")
        return
    data = json.loads(CLOUD_USAGE.read_text())
    if "semaphore_ci" not in data:
        _log("WARN: semaphore_ci block missing in cloud_usage.json")
        return
    data["semaphore_ci"]["enabled"] = True
    data["semaphore_ci"]["server"] = f"https://{org_slug}.semaphoreci.com"
    data["semaphore_ci"]["org_slug"] = org_slug
    data["semaphore_ci"]["_provisioned_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    CLOUD_USAGE.write_text(json.dumps(data, indent=2))
    _log(f"flipped semaphore_ci.enabled=true (server=https://{org_slug}.semaphoreci.com)")


def _smoke_test(token: str, org_slug: str) -> bool:
    """curl -H 'Authorization: Token <t>' https://<org>.semaphoreci.com/api/v1alpha/projects.
    Returns True iff HTTP 200."""
    url = f"https://{org_slug}.semaphoreci.com/api/v1alpha/projects"
    req = urllib.request.Request(url, headers={"Authorization": f"Token {token}"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status == 200
    except urllib.error.HTTPError as e:
        _log(f"smoke HTTPError: {e.code} {e.reason}")
        return False
    except Exception as e:
        _log(f"smoke error: {e}")
        return False


async def _try_click(page_or_frame, selectors: list[str], timeout_ms: int = 4000) -> str | None:
    """Click first matching selector. Returns the selector that worked, or None."""
    for sel in selectors:
        try:
            loc = page_or_frame.locator(sel).first
            await loc.click(timeout=timeout_ms)
            return sel
        except Exception:
            continue
    return None


async def _is_blocked(page) -> tuple[int, str] | None:
    """Detect blockers. Returns (exit_code, reason) or None.

    If a CAPTCHA widget is detected AND a solver key is present in env, attempt
    auto-solve first; only return blocked if the solver fails.
    """
    body = (await page.content()).lower()
    # CAPTCHA detection — be specific to avoid false positives on marketing pages
    captcha_present = (
        "g-recaptcha" in body or "h-captcha" in body or "cf-turnstile" in body
    )
    if captcha_present:
        solver_key_set = bool(
            os.environ.get("CAPSOLVER_KEY") or os.environ.get("TWOCAPTCHA_KEY")
        )
        if solver_key_set and maybe_solve_captcha is not None:
            _log("blocker: CAPTCHA detected — invoking captcha_solver_helper")
            solver_env = {
                "CAPSOLVER_KEY": os.environ.get("CAPSOLVER_KEY", ""),
                "TWOCAPTCHA_KEY": os.environ.get("TWOCAPTCHA_KEY", ""),
                "CAPTCHA_TIMEOUT_S": os.environ.get("CAPTCHA_TIMEOUT_S", "180"),
            }
            try:
                solved = await maybe_solve_captcha(page, solver_env, _log)
            except Exception as e:
                _log(f"blocker: solver raised {e!r}")
                solved = False
            if solved:
                _log("blocker: CAPTCHA auto-solved; clearing block state")
                return None
            _log("blocker: solver failed; reporting as blocked")
        return (2, "CAPTCHA widget detected (recaptcha/hcaptcha/turnstile)")
    # Phone-verification probes
    for tok in ("verify your phone", "phone verification", "enter your phone number",
                "we'll send a code to your phone"):
        if tok in body:
            return (3, f"phone verification requested ({tok!r})")
    return None


async def _maybe_create_org(page) -> str | None:
    """If Semaphore shows the first-time org-creation form, fill + submit.
    Returns the org slug if created; None if no form (already has org)."""
    # Heuristic: URL contains '/new' or '/onboarding', or DOM has 'Create organization'/'org name' field
    url = page.url.lower()
    needs_org = ("/new" in url or "/onboarding" in url or "/setup" in url)
    if not needs_org:
        # Also check for explicit form
        try:
            form = page.locator(
                'input[name*="org" i], input[placeholder*="organization" i], '
                'input[placeholder*="company name" i]'
            ).first
            if await form.count() and await form.is_visible():
                needs_org = True
        except Exception:
            pass
    if not needs_org:
        return None
    org_slug = f"signup-bot-{_ts().lower()}"
    _log(f"creating org: {org_slug}")
    # Try common selectors
    filled = False
    for sel in [
        'input[name="organization[name]"]',
        'input[name="org_name"]',
        'input[name*="org" i]',
        'input[placeholder*="organization" i]',
        'input[placeholder*="company name" i]',
        'input[type="text"]',
    ]:
        try:
            loc = page.locator(sel).first
            if await loc.count() and await loc.is_visible():
                await loc.fill(org_slug, timeout=4000)
                filled = True
                break
        except Exception:
            continue
    if not filled:
        return None  # let caller screenshot + decide
    # Submit
    await _try_click(page, [
        'button:has-text("Create organization")',
        'button:has-text("Create")',
        'button[type="submit"]',
        'input[type="submit"]',
    ], timeout_ms=5000)
    await page.wait_for_load_state("domcontentloaded", timeout=15000)
    return org_slug


async def _detect_org_slug(page) -> str | None:
    """Detect the org slug after redirect to me.semaphoreci.com or <org>.semaphoreci.com."""
    url = page.url
    # Match https://<slug>.semaphoreci.com/...
    m = re.match(r"https?://([a-z0-9-]+)\.semaphoreci\.com", url)
    if m and m.group(1) not in ("me", "id", "www", "docs", "semaphore"):
        return m.group(1)
    # If still on me.semaphoreci.com, try to read the active-org dropdown
    for sel in [
        '[data-org-slug]',
        '[data-organization]',
        'a[href*=".semaphoreci.com"]',
    ]:
        try:
            loc = page.locator(sel).first
            if await loc.count():
                # Prefer attribute, fallback to href
                slug = await loc.get_attribute("data-org-slug") or await loc.get_attribute("data-organization")
                if slug:
                    return slug
                href = await loc.get_attribute("href") or ""
                m2 = re.search(r"https?://([a-z0-9-]+)\.semaphoreci\.com", href)
                if m2 and m2.group(1) not in ("me", "id", "www", "docs"):
                    return m2.group(1)
        except Exception:
            continue
    return None


async def _extract_token(page) -> str | None:
    """Extract a freshly-generated API token from the DOM.
    Semaphore tokens are alphanumeric, typically 32-64 chars."""
    # Selectors that commonly hold the freshly-revealed token
    selectors = [
        'input[readonly][value]',
        'code',
        'pre',
        '[data-token]',
        '[data-test="token"]',
        '[aria-label*="token" i]',
    ]
    candidates: list[str] = []
    for sel in selectors:
        try:
            loc = page.locator(sel)
            n = await loc.count()
            for i in range(min(n, 20)):
                el = loc.nth(i)
                val = await el.get_attribute("value")
                if val:
                    candidates.append(val.strip())
                txt = await el.inner_text()
                if txt:
                    candidates.append(txt.strip())
        except Exception:
            continue
    # Filter: alphanumeric (allow hyphens/underscores), length 24-128, no spaces
    for c in candidates:
        c = c.strip()
        # Skip obvious labels
        if any(w in c.lower() for w in ("token", "api", "copy", "regenerate", "click", "name")):
            continue
        if re.fullmatch(r"[A-Za-z0-9_\-]{24,128}", c):
            return c
    return None


async def bootstrap_github() -> int:
    """One-shot operator-mediated bootstrap: open the persistent profile in a
    visible browser, navigate to github.com/login, wait for human to log in
    (including any 2FA / passkey). Close window when finished. Session cookie
    + device-trust state is then persisted in the user_data_dir."""
    _log("BOOTSTRAP MODE: opening visible browser. Log into GitHub, then close the window.")
    async with async_playwright() as pw_drv:
        ctx = await pw_drv.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            headless=False,
            channel="chromium",
            viewport={"width": 1280, "height": 800},
            user_agent=("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/147.0.0.0 Safari/537.36"),
            locale="en-US",
            timezone_id="America/New_York",
            args=["--disable-blink-features=AutomationControlled"],
        )
        page = await ctx.new_page()
        try:
            await page.goto("https://github.com/login", timeout=30000)
        except Exception as e:
            _log(f"could not open github.com/login: {e}")
            await ctx.close()
            return 1
        _log("Waiting for you to finish logging in. The script will detect when")
        _log("github.com no longer shows the /login path and then save + exit.")
        # Poll for up to 5 minutes — exits when URL leaves /login or /sessions
        for i in range(300):
            await asyncio.sleep(1)
            url = page.url.lower()
            try:
                if "github.com" in url and "/login" not in url and "/sessions" not in url:
                    # Verify by hitting api: page should be able to see /settings
                    try:
                        await page.goto("https://github.com/settings/profile",
                                        timeout=15000, wait_until="domcontentloaded")
                        if "/login" not in page.url:
                            _log("OK: GitHub session captured in profile.")
                            await ctx.close()
                            return 0
                    except Exception:
                        continue
            except Exception:
                continue
        _log("BOOTSTRAP TIMEOUT (5 min). Closing — re-run --bootstrap-github to retry.")
        await ctx.close()
        return 1


async def _verify_github_session(ctx) -> bool:
    """Open a throwaway tab on github.com/settings/profile. If we hit /login,
    the profile is unauthenticated."""
    page = await ctx.new_page()
    try:
        await page.goto("https://github.com/settings/profile",
                        timeout=15000, wait_until="domcontentloaded")
        ok = "/login" not in page.url and "/sessions" not in page.url
        await page.close()
        return ok
    except Exception:
        await page.close()
        return False


async def main(headed: bool, force: bool) -> int:
    # Idempotency: skip if token already present and works
    if ENV_PATH.exists() and not force:
        existing = {}
        for ln in ENV_PATH.read_text().splitlines():
            if "=" in ln:
                k, v = ln.split("=", 1)
                existing[k.strip().replace("export ", "")] = v.strip().strip('"').strip("'")
        tok = existing.get("SEMAPHORE_API_TOKEN")
        org = existing.get("SEMAPHORE_ORG")
        if tok and org:
            _log(f"existing token present (len={len(tok)}, org={org}); smoke-testing")
            if _smoke_test(tok, org):
                _log("existing token still valid — nothing to do. Use --force to regenerate.")
                _flip_cloud_usage(org)
                return 0
            else:
                _log("existing token failed smoke test — regenerating")

    async with async_playwright() as pw_drv:
        ctx = await pw_drv.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            headless=not headed,
            channel="chromium",  # NOT headless-shell (most-detected variant)
            viewport={"width": 1280, "height": 800},
            user_agent=("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/147.0.0.0 Safari/537.36"),
            locale="en-US",
            timezone_id="America/New_York",
            # NOTE: --no-sandbox + --disable-dev-shm-usage are Linux/Docker patterns.
            # On macOS in this Chromium build, --no-sandbox breaks the DNS resolver
            # (every navigation returns net::ERR_NAME_NOT_RESOLVED). We keep only
            # the stealth flag here. yahoo_self_provision.py uses the Linux flags
            # because it's invoked in a different environment.
            args=[
                "--disable-blink-features=AutomationControlled",
            ],
        )

        # Pre-flight: confirm GitHub session is present in the persistent profile,
        # else we'd hit the login form mid-OAuth (and bail with exit 4).
        gh_ok = await _verify_github_session(ctx)
        if not gh_ok:
            _log("BLOCKED: no GitHub session in persistent profile. Run once with")
            _log(f"  python {Path(__file__).name} --bootstrap-github")
            _log("then re-run this script (no flags) for headless production.")
            await ctx.close()
            return 4
        _log("pre-flight: GitHub session present in persistent profile")

        # NOTE: patchright already provides built-in stealth (navigator.webdriver=undefined,
        # languages, plugins, etc. — verified via bot.sannysoft.com). Adding our own
        # add_init_script() on top of patchright's persistent_context corrupts the network
        # service in this Chromium build on macOS (every navigation returns
        # net::ERR_NAME_NOT_RESOLVED). Don't add init scripts. If site-specific shims
        # are needed later (e.g., shadow-DOM piercing), add them per-page after first
        # successful navigation.
        page = await ctx.new_page()

        # ---- Step 1: navigate to login ----
        _log("step 1: navigate id.semaphoreci.com/login")
        try:
            await page.goto("https://id.semaphoreci.com/login", timeout=30000,
                            wait_until="domcontentloaded")
        except Exception as e:
            _log(f"FATAL: cannot reach id.semaphoreci.com: {e}")
            await _shot(page, "nav_fail")
            await ctx.close()
            return 1

        # Blocker scan
        blk = await _is_blocked(page)
        if blk:
            await _shot(page, "blocked_login")
            _log(f"BLOCKED at login: {blk[1]}")
            await ctx.close()
            return blk[0]

        # ---- Step 2: click Sign in with GitHub ----
        _log("step 2: click 'Sign in with GitHub'")
        gh_sel = await _try_click(page, [
            'a:has-text("Sign in with GitHub")',
            'button:has-text("Sign in with GitHub")',
            'a:has-text("Continue with GitHub")',
            'a[href*="github"]',
            'a[href*="/auth/github"]',
            '[aria-label*="github" i]',
        ], timeout_ms=8000)
        if not gh_sel:
            await _shot(page, "no_github_button")
            _log("FATAL: 'Sign in with GitHub' button not found")
            await ctx.close()
            return 4
        _log(f"  clicked: {gh_sel}")

        # ---- Step 3: GitHub OAuth ----
        # Wait for redirect to github.com or back to semaphoreci.com
        _log("step 3: await GitHub OAuth redirect cascade")
        for _ in range(40):  # up to 40s
            await asyncio.sleep(1)
            url = page.url
            if "semaphoreci.com" in url and "id.semaphoreci.com" not in url:
                break  # already redirected back (cached oauth)
            if "github.com" in url:
                # On GitHub — may be: login form, 2FA, authorize-app, or auto-redirect
                # Check for the "Authorize" green button (already-logged-in path)
                authz = await _try_click(page, [
                    'button:has-text("Authorize")',
                    'input[value="Authorize"]',
                    'button[name="authorize"]',
                ], timeout_ms=2000)
                if authz:
                    _log(f"  clicked GitHub Authorize button: {authz}")
                    await page.wait_for_load_state("domcontentloaded", timeout=20000)
                    continue
                # Check for login form (not logged into GH in this profile yet)
                login_visible = False
                try:
                    login_visible = bool(await page.locator('input[name="login"]').count())
                except Exception:
                    pass
                if login_visible:
                    await _shot(page, "github_login_required")
                    _log("BLOCKED: GitHub login required in fresh profile. "
                         "Operator must log into github.com once via the persistent "
                         f"profile at {PROFILE_DIR}. Re-run after.")
                    await ctx.close()
                    return 4
                # 2FA gate?
                blk2 = await _is_blocked(page)
                if blk2:
                    await _shot(page, "github_2fa_or_captcha")
                    _log(f"BLOCKED on GitHub: {blk2[1]}")
                    await ctx.close()
                    return blk2[0]
        else:
            await _shot(page, "oauth_timeout")
            _log(f"FATAL: OAuth cascade did not complete in 40s. Final URL: {page.url}")
            await ctx.close()
            return 4

        _log(f"  post-OAuth URL: {page.url}")

        # ---- Step 4: maybe create org ----
        _log("step 4: check for first-time org creation")
        try:
            org_slug = await _maybe_create_org(page)
        except Exception as e:
            _log(f"  org-creation attempt errored: {e}")
            org_slug = None
        if org_slug:
            _log(f"  org created: {org_slug}")
            # Wait for redirect to org dashboard
            for _ in range(20):
                await asyncio.sleep(1)
                if re.match(rf"https?://{re.escape(org_slug)}\.semaphoreci\.com", page.url):
                    break

        # ---- Step 5: detect org slug ----
        if not org_slug:
            org_slug = await _detect_org_slug(page)
        # If still not detected, navigate to me.semaphoreci.com which is the
        # canonical post-login landing and re-detect from any visible org link.
        if not org_slug:
            try:
                await page.goto("https://me.semaphoreci.com/account",
                                timeout=20000, wait_until="domcontentloaded")
                await asyncio.sleep(2)
                org_slug = await _detect_org_slug(page)
            except Exception:
                pass
        if not org_slug:
            await _shot(page, "no_org_detected")
            _log("FATAL: could not detect organization slug")
            await ctx.close()
            return 4
        _log(f"  org slug: {org_slug}")

        # ---- Step 6: navigate to API tokens page ----
        # Token-management URL changed historically — try canonical first, then
        # known fallbacks within the org dashboard.
        _log("step 6: navigate to API tokens page")
        token_urls = [
            "https://me.semaphoreci.com/account",
            f"https://{org_slug}.semaphoreci.com/account",
            f"https://{org_slug}.semaphoreci.com/users/edit",
            f"https://{org_slug}.semaphoreci.com/settings",
        ]
        landed = False
        for u in token_urls:
            try:
                await page.goto(u, timeout=20000, wait_until="domcontentloaded")
                await asyncio.sleep(1.5)
                # Heuristic: page mentions "token" or "API"
                body = (await page.content()).lower()
                if "token" in body or "api" in body:
                    _log(f"  landed: {u}")
                    landed = True
                    break
            except Exception:
                continue
        if not landed:
            await _shot(page, "no_token_page")
            _log("FATAL: no token-management page reachable")
            await ctx.close()
            return 4

        # ---- Step 7: click "Generate" / "New token" ----
        _log("step 7: click Generate/New token button")
        gen_sel = await _try_click(page, [
            'button:has-text("Generate new token")',
            'button:has-text("Generate token")',
            'button:has-text("Generate")',
            'button:has-text("New token")',
            'a:has-text("Generate new token")',
            'a:has-text("New token")',
            'button:has-text("Create token")',
            'button:has-text("Regenerate")',  # if a token already exists
            '[data-test="generate-token"]',
        ], timeout_ms=5000)
        if not gen_sel:
            await _shot(page, "no_generate_button")
            _log("FATAL: no Generate-token button found on tokens page")
            await ctx.close()
            return 4
        _log(f"  clicked: {gen_sel}")

        # Some Semaphore UIs prompt for a token name field
        try:
            name_loc = page.locator('input[name*="token" i], input[placeholder*="name" i]').first
            if await name_loc.count() and await name_loc.is_visible():
                await name_loc.fill(f"signup-bot-{_ts()}", timeout=3000)
                # Confirm button if present
                await _try_click(page, [
                    'button:has-text("Create")',
                    'button:has-text("Generate")',
                    'button[type="submit"]',
                ], timeout_ms=3000)
        except Exception:
            pass

        await asyncio.sleep(2)

        # Some flows require confirming a 2nd dialog
        await _try_click(page, [
            'button:has-text("Confirm")',
            'button:has-text("Yes, regenerate")',
            'button:has-text("Continue")',
        ], timeout_ms=2000)
        await asyncio.sleep(1.5)

        # ---- Step 8: extract token ----
        _log("step 8: extract token from DOM")
        token = await _extract_token(page)
        if not token:
            await _shot(page, "no_token_in_dom")
            _log("FATAL: token not visible in DOM after Generate click")
            await ctx.close()
            return 4
        _log(f"  token captured (len={len(token)}, suffix=...{token[-4:]})")

        await ctx.close()

    # ---- Step 9: write env + flip cloud_usage ----
    _write_env(token, org_slug)
    _log(f"wrote {ENV_PATH} (chmod 600)")
    _flip_cloud_usage(org_slug)

    # ---- Step 10: smoke test ----
    _log("step 10: smoke-test the new token against the projects API")
    if _smoke_test(token, org_slug):
        _log(f"OK: smoke 200. semaphore_ci provisioned. org={org_slug}")
        return 0
    else:
        _log(f"WARN: token captured + env written, but smoke API call failed. "
             f"Token may need a moment to propagate; re-run smoke later.")
        return 5


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--headed", action="store_true",
                    help="Show browser (debug only; production stays headless)")
    ap.add_argument("--force", action="store_true",
                    help="Regenerate even if SEMAPHORE_API_TOKEN already present + valid")
    ap.add_argument("--bootstrap-github", action="store_true",
                    help="One-time operator step: open visible browser to "
                         "github.com/login, wait for human login, save session "
                         "cookies into the persistent profile, exit. "
                         "Required exactly once per fresh profile.")
    args = ap.parse_args()
    try:
        if args.bootstrap_github:
            sys.exit(asyncio.run(bootstrap_github()))
        sys.exit(asyncio.run(main(args.headed, args.force)))
    except KeyboardInterrupt:
        sys.exit(130)
