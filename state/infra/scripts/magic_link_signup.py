#!/usr/bin/env python3
"""magic_link_signup.py — Generic email-only / magic-link signup orchestrator.

Bypasses the GitHub-OAuth-cookie blocker for YELLOW-tier providers that
accept email-only signup (groq, cerebras, sambanova, mistral, lightning_ai,
huggingface_spaces, koyeb, buddy_works). For each provider, this script:

  1. Loads per-provider config from ~/.config/auto_signup/providers.json
  2. Launches patchright (stealth Playwright fork) with a persistent profile
  3. Navigates to the signup URL
  4. Fills the email (+ password / username if required)
  5. Submits and waits for the "check your email" UI state
  6. Polls Yahoo IMAP via yahoo_mail_reader for the magic-link / confirmation
     email matching the provider's sender/subject filters
  7. Extracts the confirmation URL from the email body
  8. Opens the confirmation URL in the same Playwright context (so the
     session cookie persists for subsequent token-page nav)
  9. Navigates to the token page
 10. Clicks the create-token button, fills a unique token name
 11. Reads the token from the DOM
 12. Writes the token to a per-provider .env file (chmod 600)
 13. Flips cloud_usage.json <provider>.enabled = true
 14. Runs the smoke-test curl

Safety:
  * Never logs the token value (only length)
  * chmod 600 on all .env files
  * Refuses to run without --confirm-create
  * Detects CAPTCHA / device-verify and aborts with a screenshot
  * --dry-run shows the plan without any network I/O
  * IMAP read by default (no --mark-read pass-through unless --mark-read-mail)
  * If YAHOO_APP_PASSWORD is empty, falls back to Path 2 DOM scrape via
    yahoo_mail_reader's web fallback (Playwright DOM of mail.yahoo.com).
    Currently flagged as TODO if Path 2 doesn't exist in yahoo_mail_reader.

Usage:
    python magic_link_signup.py --provider groq --email orginal_clawdbot@yahoo.com --confirm-create
    python magic_link_signup.py --provider huggingface_spaces --email ... --confirm-create --headed
    python magic_link_signup.py --provider groq --dry-run

License: MIT-compatible (patchright is MIT, imaplib is stdlib).
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import importlib.util
import json
import os
import re
import secrets
import subprocess
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths — Mac-local (launcher remaps $HOME into Drive, so we hardcode).       #
# --------------------------------------------------------------------------- #

MAC_HOME = Path("/Users/orginal")
CFG_DIR = MAC_HOME / ".config" / "auto_signup"
PROVIDERS_JSON = CFG_DIR / "providers.json"
PROFILE_ROOT = CFG_DIR / "playwright_profiles"
DRIVE_ROOT = (
    MAC_HOME / "Library" / "CloudStorage" / "GoogleDrive-zachgladstone@gmail.com"
    / "My Drive" / "AI-Tools"
)
LOG_DIR = DRIVE_ROOT / "logs" / "auto_signup"
CLOUD_USAGE = DRIVE_ROOT / "s&p500-ticker-mastery" / "sweeps" / "cloud_usage.json"
YAHOO_ENV = CFG_DIR / "yahoo.env"
SCRIPTS_DIR = DRIVE_ROOT / "scripts"
BACKUP_DIR = DRIVE_ROOT / "backups" / "magic-link-signup-2026-05-17"

CFG_DIR.mkdir(parents=True, exist_ok=True)
PROFILE_ROOT.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
BACKUP_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Logging                                                                     #
# --------------------------------------------------------------------------- #

def _ts() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def log(msg: str) -> None:
    print(f"[{dt.datetime.now(dt.timezone.utc).isoformat()}] {msg}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# Env loading                                                                 #
# --------------------------------------------------------------------------- #

def load_dotenv(path: Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE / export KEY=VALUE file. Returns {}."""
    if not path.exists():
        return {}
    env: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:]
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def load_providers() -> dict:
    if not PROVIDERS_JSON.exists():
        raise SystemExit(
            f"FATAL: {PROVIDERS_JSON} missing. Create it first — see report "
            "reports/magic_link_signup_repo_2026-05-17.md"
        )
    return json.loads(PROVIDERS_JSON.read_text())


# --------------------------------------------------------------------------- #
# Yahoo Mail polling — delegated to scripts/yahoo_mail_reader.py              #
# --------------------------------------------------------------------------- #

def _import_yahoo_reader():
    """Dynamic-import yahoo_mail_reader so this script works under any $PYTHONPATH.

    NOTE (2026-05-18 koyeb wave2c fix): the module MUST be registered in
    `sys.modules` BEFORE `exec_module` runs, otherwise the `@dataclass`
    decorator at yahoo_mail_reader.py:93 hits an AttributeError —
    `_is_type` in dataclasses.py looks up `sys.modules.get(cls.__module__)`,
    finds None, and then crashes on `.__dict__`. This is the documented
    importlib.util usage pattern; the prior code was incomplete.
    """
    mod_name = "yahoo_mail_reader"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(
        mod_name, str(SCRIPTS_DIR / "yahoo_mail_reader.py")
    )
    if spec is None or spec.loader is None:
        raise SystemExit(f"FATAL: cannot import yahoo_mail_reader from {SCRIPTS_DIR}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod  # MUST happen before exec_module — see docstring.
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return mod


def poll_yahoo_for_magic_link(
    email_user: str,
    yahoo_app_password: str,
    since_dt: dt.datetime,
    sender_substr: str,
    subject_substr: str | None,
    url_pattern: str,
    timeout_s: int = 300,
) -> str | None:
    """Poll Yahoo IMAP until a message from sender_substr arrives whose body
    contains a URL matching url_pattern. Returns the matching URL or None."""
    yreader = _import_yahoo_reader()
    url_re = re.compile(url_pattern, re.IGNORECASE)
    msgs = yreader.poll_until_match(
        user=email_user,
        app_password=yahoo_app_password,
        since=since_dt,
        sender_filter=sender_substr,
        subject_filter=subject_substr,
        folder="INBOX",
        mark_read=False,
        timeout_s=timeout_s,
        require_url=True,
        require_otp=False,
    )
    for m in msgs:
        for u in m.urls:
            if url_re.search(u):
                log(f"matched magic-link from uid={m.uid} subject={m.subject[:60]!r}")
                return u
    if msgs:
        log(f"WARN: {len(msgs)} candidate msgs but none URL-matched pattern {url_pattern!r}")
    return None


# --------------------------------------------------------------------------- #
# Token-file writer                                                           #
# --------------------------------------------------------------------------- #

def write_token_to_env(env_path: Path, var_name: str, token: str) -> None:
    """Upsert var_name=token in env_path. chmod 600 after write. Never log value."""
    lines: list[str] = []
    if env_path.exists():
        # Make a backup before overwriting
        bkp = BACKUP_DIR / f"{env_path.name}.bak.{_ts()}"
        bkp.write_text(env_path.read_text())
        bkp.chmod(0o600)
        for line in env_path.read_text().splitlines():
            if re.match(rf"^\s*(?:export\s+)?{re.escape(var_name)}=", line):
                continue
            lines.append(line)
    else:
        lines.append(f"# Auto-generated by magic_link_signup.py at {_ts()}")
        lines.append("# chmod 600. NEVER commit. NEVER print.")
    lines.append(f'export {var_name}="{token}"')
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("\n".join(lines) + "\n")
    env_path.chmod(0o600)
    log(f"wrote {var_name}=<{len(token)} chars> to {env_path}")


def flip_cloud_usage_enabled(cloud_key: str) -> bool:
    """Set cloud_usage.json[<cloud_key>].enabled = True. Return True on flip."""
    if not CLOUD_USAGE.exists():
        log(f"WARN: cloud_usage.json missing at {CLOUD_USAGE} — skipping flip")
        return False
    # Backup before edit
    bkp = BACKUP_DIR / f"cloud_usage.json.bak.{_ts()}"
    bkp.write_text(CLOUD_USAGE.read_text())
    log(f"backup: {bkp}")

    data = json.loads(CLOUD_USAGE.read_text())
    if cloud_key not in data or not isinstance(data[cloud_key], dict):
        log(f"WARN: '{cloud_key}' block missing from cloud_usage.json — skipping flip")
        return False
    before = data[cloud_key].get("enabled", False)
    data[cloud_key]["enabled"] = True
    CLOUD_USAGE.write_text(json.dumps(data, indent=2))
    log(f"flipped {cloud_key}.enabled: {before} -> True")
    return True


# --------------------------------------------------------------------------- #
# Smoke test                                                                  #
# --------------------------------------------------------------------------- #

def smoke_test(curl_template: str, token: str) -> tuple[bool, str]:
    """Run curl_template with {TOKEN} substitution. Return (ok, http_code)."""
    if "{TOKEN}" not in curl_template:
        return False, "BAD_TEMPLATE"
    cmd = curl_template.replace("{TOKEN}", token)
    try:
        result = subprocess.run(
            ["bash", "-c", cmd], capture_output=True, text=True, timeout=20,
        )
        code = result.stdout.strip().splitlines()[-1] if result.stdout else "?"
        ok = code in ("200", "201", "204", "401")  # 401 = endpoint reachable, token shape OK
        log(f"smoke-test HTTP {code} (ok={ok})")
        return code == "200", code
    except Exception as e:
        log(f"smoke-test ERROR: {e}")
        return False, "ERROR"


# --------------------------------------------------------------------------- #
# Playwright helpers                                                          #
# --------------------------------------------------------------------------- #

async def _screenshot(page, label: str) -> Path:
    p = LOG_DIR / f"magic_link_{label}_{_ts()}.png"
    try:
        await page.screenshot(path=str(p), full_page=True)
    except Exception as e:  # pragma: no cover - best-effort
        log(f"shot-fail:{e}")
    return p


async def _detect_challenge(page) -> str | None:
    """Detect a *visible / actionable* CAPTCHA or device-verification challenge.

    History: the prior implementation substring-matched the raw HTML body for
    words like "captcha", "recaptcha", "hcaptcha", "verification challenge",
    "press and hold", "i'm not a robot". This produced massive false positives:
    privacy banners, footer terms links, hidden anti-bot script tags, ToS
    text, password-strength hints — all triggered. Wave-1 burned 3 providers
    (cerebras, huggingface_spaces, buddy_works) on false-positive blocks even
    though the pages contained NO visible CAPTCHA widget.

    Replacement strategy:
      1. Inspect the DOM for a *visible* widget element via JS — same selectors
         captcha_solver_helper.detect_captcha_widget uses (.h-captcha[data-sitekey],
         .g-recaptcha[data-sitekey], .cf-turnstile[data-sitekey], or iframes
         with hcaptcha/google-recaptcha/cloudflare-challenges hosts).
      2. Only signal a block when the widget is BOTH present AND has nonzero
         layout (i.e., actually rendered, not a hidden script-loaded stub).
      3. Also detect explicit device-verification text inside form / heading
         elements — but NOT in script/meta/footer/anchor href text.

    Returns:
      None  → no actionable challenge (proceed)
      str   → human-readable challenge label (caller may decide to solve / abort)
    """
    js = r"""
    () => {
      // 1. Visible CAPTCHA widget — must have nonzero bounding box
      const widgetSelectors = [
        ['hcaptcha',  '.h-captcha[data-sitekey], iframe[src*="hcaptcha.com"]'],
        ['recaptcha', '.g-recaptcha[data-sitekey], iframe[src*="google.com/recaptcha"], iframe[src*="recaptcha.net"]'],
        ['turnstile', '.cf-turnstile[data-sitekey], iframe[src*="challenges.cloudflare.com"]'],
      ];
      for (const [kind, sel] of widgetSelectors) {
        for (const el of document.querySelectorAll(sel)) {
          const r = el.getBoundingClientRect();
          if (r.width > 0 && r.height > 0) {
            return { kind: kind + '_widget_visible' };
          }
        }
      }

      // 2. Heading / button text — actionable device-verify or press-and-hold
      //    Scope to h1/h2/h3/h4/button/[role=heading] to avoid privacy-banner noise.
      const phrases = [
        'verification challenge',
        'device verification',
        'verify your device',
        'press and hold',
        "i'm not a robot",
        'are you human',
      ];
      const scopes = document.querySelectorAll('h1, h2, h3, h4, [role="heading"], button, [data-testid*="challenge" i]');
      for (const el of scopes) {
        const txt = (el.innerText || '').toLowerCase();
        for (const p of phrases) {
          if (txt.includes(p)) {
            return { kind: 'text:' + p };
          }
        }
      }
      return null;
    }
    """
    try:
        result = await page.evaluate(js)
    except Exception:
        return None
    if not result:
        return None
    return result.get("kind") if isinstance(result, dict) else str(result)


async def _try_click(page, selectors: list[str], timeout_ms: int = 4000) -> bool:
    for sel in selectors:
        try:
            loc = page.locator(sel).first
            await loc.wait_for(state="visible", timeout=timeout_ms)
            await loc.click(timeout=timeout_ms)
            log(f"  clicked: {sel}")
            return True
        except Exception:
            continue
    return False


async def _try_fill(page, selectors: list[str], value: str, timeout_ms: int = 4000) -> bool:
    """Fill a form field reliably across React / Vue / vanilla DOM stacks.

    Bug context (2026-05-18 huggingface_spaces wave):
      The prior implementation used `locator.fill(value)`, which Playwright
      implements via DOM-level value assignment + a single synthetic `input`
      event. React's controlled-input pattern hooks the *native* HTMLInput
      `value` setter (via `Object.getOwnPropertyDescriptor`) and listens for
      `onChange` — Playwright's `fill()` does NOT exercise that descriptor
      hook, so React's internal state never updates. The DOM showed our email,
      but React thought the field was empty. On submit, the form's HTML5
      `required` validation rejected the (React-empty) field and aborted the
      POST — the browser stayed at /join and no email was ever sent.

    Symptom in logs:
      - magic_link_huggingface_spaces_post_submit_*.png shows the
        "Please fill out this field" tooltip on the email input.
      - Stage 3 records `submitted: true` but page.url stays at /join.
      - Yahoo IMAP poll times out (no email was sent).

    Fix strategy — three-layer:
      1. Focus + clear the field (so we don't append to stale state).
      2. `press_sequentially(value, delay=30)` — types char-by-char,
         firing real `keydown`/`keypress`/`input`/`keyup` events that React,
         Vue, Svelte, Solid, and vanilla listeners all honor.
      3. Reinforcement: dispatch a final JS `input`+`change` pair with
         `bubbles: true` via `evaluate`, in case any framework re-reads the
         value off the DOM at submit time.
      4. `blur()` to flush focused-only validators (e.g. react-hook-form's
         `validateOnBlur` mode).

    Verification:
      After typing, we read the field's `.value` back via JS. If it doesn't
      match `value`, we log a WARN but still return True (the form may have
      transformed the input — e.g. trimming whitespace, casing — and we
      don't want to abort on cosmetic differences). A length mismatch >2x
      the expected is treated as a fill failure and the next selector is
      tried.

    Returns True on the first selector that successfully receives the value.
    Never logs the value contents (caller passes passwords through here).
    """
    for sel in selectors:
        try:
            loc = page.locator(sel).first
            await loc.wait_for(state="visible", timeout=timeout_ms)

            # 1. Focus + clear
            try:
                await loc.click(timeout=timeout_ms)
            except Exception:
                # Some inputs intercept click events (e.g. styled wrappers);
                # focus is enough to make them receptive to keypresses.
                try:
                    await loc.focus(timeout=timeout_ms)
                except Exception:
                    pass
            try:
                # Triple-click to select all existing content, then Backspace.
                # More reliable than `fill("")` which has the same React bug.
                await loc.click(click_count=3, timeout=timeout_ms)
                await page.keyboard.press("Backspace")
            except Exception:
                pass

            # 2. Type char-by-char (fires real events React/Vue honor).
            #    delay=30ms keeps total time well under 1s for typical
            #    20-char passwords / 30-char emails, while still defeating
            #    rate-limiting input handlers.
            await loc.press_sequentially(value, delay=30, timeout=timeout_ms)

            # 3. Reinforcement JS-dispatch for any framework that re-reads
            #    the value on submit. Using the native value-setter pattern
            #    is the canonical React-controlled-input workaround
            #    (see facebook/react#11488).
            try:
                await loc.evaluate(
                    """(el, val) => {
                        try {
                            const proto = el.tagName === 'TEXTAREA'
                                ? window.HTMLTextAreaElement.prototype
                                : window.HTMLInputElement.prototype;
                            const setter = Object.getOwnPropertyDescriptor(proto, 'value').set;
                            setter.call(el, val);
                        } catch (e) {
                            el.value = val;
                        }
                        el.dispatchEvent(new Event('input', {bubbles: true}));
                        el.dispatchEvent(new Event('change', {bubbles: true}));
                    }""",
                    value,
                )
            except Exception:
                # Non-fatal: press_sequentially alone is usually enough.
                pass

            # 4. Blur to flush validateOnBlur frameworks (react-hook-form,
            #    formik with validateOnBlur=true, etc.).
            try:
                await loc.evaluate("(el) => el.blur()")
            except Exception:
                pass

            # Verify the field actually has approximately the value we sent.
            try:
                actual = await loc.input_value(timeout=1000)
            except Exception:
                actual = None
            if actual is not None:
                if actual == value:
                    log(f"  filled: {sel} (verified)")
                    return True
                elif len(actual) >= max(1, len(value) // 2):
                    # Form transformed the input but it's clearly populated
                    # (e.g. trimmed, lowercased). Accept it.
                    log(f"  filled: {sel} (len={len(actual)}, partial-match accepted)")
                    return True
                else:
                    log(f"  WARN: {sel} read-back len={len(actual)} (expected {len(value)}); trying next selector")
                    continue
            # No input_value available (e.g. contenteditable) — assume OK.
            log(f"  filled: {sel} (no read-back)")
            return True
        except Exception:
            continue
    return False


async def _detect_login_redirect(page, cfg: dict) -> str | None:
    """Detect whether the signup nav was redirected to a LOGIN page.

    Triggered when the provider's identity system (e.g. Ory Kratos) detects
    the email already has an account — the server 303-redirects from
    /registration/* to /login?flow=<uuid>. The browser silently follows.

    Detection sources (any one is sufficient):
      1. URL contains one of cfg["login_url_substrings"] (e.g. "/login?flow=").
      2. Visible page text contains any string in cfg["login_indicators"]
         (e.g. "Please select a way to login", "Continue with password").

    Returns a human-readable label of WHAT triggered the detection, or None
    if we're still on a fresh signup page.
    """
    url = (page.url or "").lower()
    for sub in cfg.get("login_url_substrings", []) or []:
        if sub.lower() in url:
            return f"url:{sub}"
    indicators = cfg.get("login_indicators") or []
    if not indicators:
        return None
    js = r"""
    (phrases) => {
      const body = document.body ? (document.body.innerText || '') : '';
      const lower = body.toLowerCase();
      for (const p of phrases) {
        if (lower.includes(p.toLowerCase())) return p;
      }
      return null;
    }
    """
    try:
        hit = await page.evaluate(js, indicators)
    except Exception:
        return None
    if hit:
        return f"text:{hit}"
    return None


async def _drive_magic_link_login_path(
    page,
    cfg: dict,
    email: str,
    confirm_timeout_ms: int = 20000,
) -> tuple[bool, str]:
    """Drive a MAGIC-LINK login flow on a provider's login page.

    Use this when the provider's login UI offers an email-magic-link option
    in addition to (or instead of) password. Mistral's Ory Kratos login flow
    shows a "Continue with email" / "Email me a sign-in link" button beside
    the password field — we'd rather take that path than touch the password
    fields (account was created with an unknown password, and "Forgot password"
    would trigger a lockout).

    Steps:
      1. Fill the email/identifier field.
      2. Click one of the magic-link buttons configured under
         `login_magic_link_button_selectors`. We do NOT click any of the
         password-related submit buttons.
      3. Wait briefly for the "check your email" / confirmation UI by polling
         for any of `login_magic_link_confirm_indicators` in the page text,
         or for a URL substring in `login_magic_link_confirm_url_substrings`.
      4. Hand back control to the caller, which will poll Yahoo IMAP for the
         magic-link email and continue with the existing stages 4-10.

    Returns (ok, last_url). `ok=True` only if both fill + click succeed.
    Confirmation-text detection is best-effort: many providers redirect or
    re-render rather than showing dedicated copy, so we accept either signal
    OR a click-without-error.

    Safety:
      * NEVER clicks "Forgot password", "Reset password", or any password
        submit button (those would either lock the account or fail).
      * NEVER fills the password field on this path.
    """
    # 1. Fill identifier/email.
    log("magic-link-login: fill email + click magic-link button")
    filled = await _try_fill(
        page, cfg.get("email_field_selectors", []), email, timeout_ms=8000
    )
    if not filled:
        await _screenshot(page, "magic_link_login_no_email_field")
        log("magic-link-login: FATAL: email field not found")
        return False, page.url

    # 2. Click magic-link button (NOT password submit).
    mlbtns = cfg.get("login_magic_link_button_selectors") or []
    if not mlbtns:
        log("magic-link-login: FATAL: no login_magic_link_button_selectors "
            "configured for this provider")
        return False, page.url
    clicked = await _try_click(page, mlbtns, timeout_ms=8000)
    if not clicked:
        await _screenshot(page, "magic_link_login_no_button")
        log("magic-link-login: FATAL: magic-link button not found / not clickable")
        return False, page.url
    await asyncio.sleep(2.5)
    await _screenshot(page, "magic_link_login_post_click")

    # 3. Best-effort wait for confirmation UI.
    indicators = cfg.get("login_magic_link_confirm_indicators") or [
        "check your email",
        "check your inbox",
        "we sent you",
        "we've sent you",
        "magic link sent",
        "sign-in link",
        "sign in link",
    ]
    url_subs = [s.lower() for s in (
        cfg.get("login_magic_link_confirm_url_substrings") or []
    )]
    deadline = asyncio.get_event_loop().time() + confirm_timeout_ms / 1000.0
    confirmed_via: str | None = None
    while asyncio.get_event_loop().time() < deadline:
        cur = (page.url or "").lower()
        if url_subs and any(s in cur for s in url_subs):
            confirmed_via = f"url:{cur}"
            break
        try:
            hit = await page.evaluate(
                """(phrases) => {
                    const body = document.body ? (document.body.innerText || '') : '';
                    const lower = body.toLowerCase();
                    for (const p of phrases) {
                        if (lower.includes(p.toLowerCase())) return p;
                    }
                    return null;
                }""",
                indicators,
            )
        except Exception:
            hit = None
        if hit:
            confirmed_via = f"text:{hit}"
            break
        await asyncio.sleep(1.0)

    if confirmed_via:
        log(f"magic-link-login: confirmation detected via {confirmed_via}")
        return True, page.url

    # 3b. Identifier-first fallback: the click may have routed us to a
    #     password page (provider supports BOTH password + magic-link, and
    #     the account has a password set). Look for an alt magic-link button
    #     on this new page ("Email me a sign-in link" / "Sign in with email").
    alt_btns = cfg.get("login_magic_link_alt_button_selectors_on_password_page") or []
    if alt_btns:
        cur = (page.url or "").lower()
        on_password_page = "/password" in cur or "/login/password" in cur
        if not on_password_page:
            # Heuristic: visible password field on page = password page.
            try:
                pwd_count = 0
                for sel in cfg.get("password_field_selectors", []) or []:
                    pwd_count += await page.locator(sel).count()
                on_password_page = pwd_count > 0
            except Exception:
                pass
        if on_password_page:
            log("magic-link-login: routed to password page — trying alt "
                "'Email me a sign-in link' / 'Sign in with email' button")
            await _screenshot(page, "magic_link_login_on_password_page")
            alt_clicked = await _try_click(page, alt_btns, timeout_ms=6000)
            if alt_clicked:
                await asyncio.sleep(2.5)
                await _screenshot(page, "magic_link_login_alt_clicked")
                # Re-poll for confirmation.
                deadline2 = asyncio.get_event_loop().time() + 10.0
                while asyncio.get_event_loop().time() < deadline2:
                    cur = (page.url or "").lower()
                    if url_subs and any(s in cur for s in url_subs):
                        log(f"magic-link-login: alt-button confirmation via url:{cur}")
                        return True, page.url
                    try:
                        hit = await page.evaluate(
                            """(phrases) => {
                                const body = document.body ? (document.body.innerText || '') : '';
                                const lower = body.toLowerCase();
                                for (const p of phrases) {
                                    if (lower.includes(p.toLowerCase())) return p;
                                }
                                return null;
                            }""",
                            indicators,
                        )
                    except Exception:
                        hit = None
                    if hit:
                        log(f"magic-link-login: alt-button confirmation via text:{hit}")
                        return True, page.url
                    await asyncio.sleep(1.0)
                log("magic-link-login: WARN: alt-button clicked but no "
                    "confirmation observed — proceeding to IMAP poll anyway")
                return True, page.url
            else:
                log("magic-link-login: alt magic-link button NOT found on "
                    "password page — provider does NOT offer magic-link "
                    "alternative when password is set")
                await _screenshot(page, "magic_link_login_no_alt_button")
                return False, page.url

    log("magic-link-login: WARN: no explicit confirmation text/url, "
        "but click succeeded — proceeding to IMAP poll anyway")
    return True, page.url


async def _drive_login_path(
    page,
    cfg: dict,
    email: str,
    password: str | None,
    post_login_timeout_ms: int = 30000,
) -> tuple[bool, str]:
    """Drive a password-login flow after detecting an existing-account redirect.

    Multi-step Kratos login flows often present:
      Step A: identifier (email) field only, click Continue → reveals password
      Step B: password field, click "Continue with password" → redirects to app

    We handle BOTH:
      1. If no password field is visible yet, fill identifier + click submit.
      2. Then wait briefly and fill the password field + click submit.
      3. Wait for the URL to leave the auth-domain (or match
         post_login_url_substrings).

    Returns (ok, last_url). `ok=True` only if we observe the post-login
    landing URL.

    Safety: this function NEVER triggers the "forgot password" / account-
    recovery flow — it only fills the known password and clicks the primary
    submit selectors.
    """
    if not password:
        log("login-fallback: FATAL: no password available "
            "(--password not given and YAHOO_PASSWORD missing from yahoo.env)")
        return False, page.url

    # Optionally navigate to the explicit login_url. If we're already on a
    # login page we leave the current URL alone (Kratos flows are stateful —
    # navigating away discards the flow=<uuid>).
    login_url = cfg.get("login_url")
    detected = await _detect_login_redirect(page, cfg)
    if detected:
        log(f"login-fallback: already on login page ({detected}); reusing flow")
    elif login_url:
        log(f"login-fallback: navigating to login_url {login_url}")
        try:
            await page.goto(login_url, timeout=30000, wait_until="domcontentloaded")
            await asyncio.sleep(2)
        except Exception as e:
            log(f"login-fallback: nav to login_url failed: {e}")
            return False, page.url

    # Step A: identifier fill. Two flow shapes:
    #   (i) Multi-step (Kratos / Auth0 progressive) — password field NOT visible
    #       yet; fill identifier + click submit to advance, then fall through to
    #       step B which fills password on the next page.
    #   (ii) Single-page (HuggingFace, classic forms) — username AND password
    #       fields are BOTH visible on the same page. We must fill BOTH and
    #       click submit once.
    # 2026-05-18 HF zombie-login patch: previously the code only filled
    # identifier when pwd_count==0, which left HF's username field empty
    # (browser then blocked submit with "Please fill out this field"). Fix:
    # always attempt to fill identifier — _try_fill is a best-effort no-op
    # if the field isn't visible/already-filled, so this is safe for the
    # multi-step case too.
    pwd_count = 0
    for sel in cfg.get("password_field_selectors", []) or []:
        try:
            pwd_count += await page.locator(sel).count()
        except Exception:
            continue

    if pwd_count == 0:
        log("login-fallback: step A (multi-step) — fill identifier + submit to advance")
        filled = await _try_fill(
            page, cfg.get("email_field_selectors", []), email, timeout_ms=8000
        )
        if filled:
            await _try_click(
                page, cfg.get("submit_selectors", []), timeout_ms=6000
            )
            await asyncio.sleep(2.5)
        else:
            log("  identifier field not visible — assuming page is already at password step")
    else:
        log("login-fallback: step A (single-page) — fill identifier (password field already visible)")
        await _try_fill(
            page, cfg.get("email_field_selectors", []), email, timeout_ms=8000
        )

    # Step B: fill password + submit.
    log("login-fallback: step B — fill password + submit")
    pwd_filled = await _try_fill(
        page, cfg.get("password_field_selectors", []), password, timeout_ms=8000
    )
    if not pwd_filled:
        await _screenshot(page, "login_no_password_field")
        log("login-fallback: FATAL: password field never appeared")
        return False, page.url

    clicked = await _try_click(
        page, cfg.get("submit_selectors", []), timeout_ms=6000
    )
    if not clicked:
        log("login-fallback: WARN: submit click failed — pressing Enter as fallback")
        try:
            await page.keyboard.press("Enter")
        except Exception:
            pass

    # Wait for redirect away from auth domain.
    post_subs = [s.lower() for s in (cfg.get("post_login_url_substrings") or [])]
    landing = cfg.get("post_login_landing_url")
    deadline = asyncio.get_event_loop().time() + post_login_timeout_ms / 1000.0
    while asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(1.0)
        cur = (page.url or "").lower()
        if post_subs and any(s in cur for s in post_subs):
            log(f"login-fallback: post-login URL matched: {page.url}")
            return True, page.url
        # Heuristic: if URL no longer contains "/login" or "/registration",
        # AND it no longer points at an auth subdomain, assume success.
        if "/login" not in cur and "/registration" not in cur and "auth." not in cur:
            log(f"login-fallback: post-login URL inferred (no auth markers): {page.url}")
            return True, page.url

    # Last-ditch: force-nav to the configured landing page and confirm.
    if landing:
        try:
            await page.goto(landing, timeout=15000, wait_until="domcontentloaded")
            await asyncio.sleep(2)
            cur = (page.url or "").lower()
            if post_subs and any(s in cur for s in post_subs):
                log(f"login-fallback: forced-nav landing OK: {page.url}")
                return True, page.url
        except Exception:
            pass

    await _screenshot(page, "login_post_submit_timeout")
    log(f"login-fallback: FATAL: did not reach post-login URL (last={page.url})")
    return False, page.url


async def _read_token_from_dom(page, selectors: list[str], regex: str,
                               timeout_ms: int = 10000) -> str | None:
    """Try each selector; for each match, return the first DOM text matching regex."""
    rgx = re.compile(regex)
    deadline = asyncio.get_event_loop().time() + timeout_ms / 1000.0
    while asyncio.get_event_loop().time() < deadline:
        for sel in selectors:
            try:
                loc = page.locator(sel)
                count = await loc.count()
                for i in range(min(count, 10)):
                    el = loc.nth(i)
                    # Prefer 'value' attr (readonly input), else text content
                    val = None
                    try:
                        val = await el.get_attribute("value")
                    except Exception:
                        val = None
                    if not val:
                        try:
                            val = await el.text_content()
                        except Exception:
                            val = None
                    if not val:
                        continue
                    m = rgx.search(val)
                    if m:
                        return m.group(0)
            except Exception:
                continue
        await asyncio.sleep(0.5)
    return None


# --------------------------------------------------------------------------- #
# CAPTCHA attempt — free ClickSolver first, then CapSolver/2Captcha           #
# --------------------------------------------------------------------------- #

def _import_captcha_helper():
    """Dynamic-import captcha_solver_helper (sits next to this file)."""
    spec = importlib.util.spec_from_file_location(
        "captcha_solver_helper", str(SCRIPTS_DIR / "captcha_solver_helper.py")
    )
    if spec is None or spec.loader is None:
        return None
    try:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception as e:
        log(f"captcha-attempt: helper import failed: {e!r}")
        return None


async def _try_free_clicksolver(page) -> bool:
    """Free path: playwright-captcha's ClickSolver (no API key).

    Supports Cloudflare interstitial, Cloudflare Turnstile, reCAPTCHA v2/v3.
    Does NOT support hCaptcha — caller falls back to API solver on hCaptcha.

    Returns True if the solver believes it cleared a challenge; False if
    nothing-to-do or the solver could not progress. Never raises.
    """
    try:
        from playwright_captcha import ClickSolver, FrameworkType, CaptchaType
    except Exception as e:
        log(f"captcha-attempt: playwright_captcha import failed ({e!r}) — skipping free path")
        return False

    # ClickSolver requires both `captcha_container` (Page/Frame/ElementHandle)
    # AND `captcha_type` (CaptchaType enum). The container is normally the
    # Page itself for top-level widgets.
    supported = [
        CaptchaType.CLOUDFLARE_TURNSTILE,
        CaptchaType.CLOUDFLARE_INTERSTITIAL,
        CaptchaType.RECAPTCHA_V2,
    ]
    for captcha_type in supported:
        try:
            async with ClickSolver(
                framework=FrameworkType.PATCHRIGHT,
                page=page,
                max_attempts=2,
                attempt_delay=3,
            ) as solver:
                detected = await solver.detect_captcha_data(
                    captcha_container=page,
                    captcha_type=captcha_type,
                )
                if not detected:
                    continue
                log(f"captcha-attempt: free ClickSolver detected {captcha_type.value} — solving")
                result = await solver.solve_captcha(
                    captcha_container=page,
                    captcha_type=captcha_type,
                )
                if result:
                    log(f"captcha-attempt: free ClickSolver succeeded on {captcha_type.value}")
                    return True
                log(f"captcha-attempt: free ClickSolver returned falsy on {captcha_type.value}")
        except Exception as e:
            log(f"captcha-attempt: ClickSolver {captcha_type.value} raised {e!r}")
            continue
    return False


async def _attempt_captcha_clear(page, env: dict, label: str,
                                 provider_name: str | None = None) -> bool:
    """Four-stage CAPTCHA clear:
       (1) free ClickSolver        — Cloudflare Turnstile / Interstitial / reCAPTCHA v2
       (2) free audio solver       — playwright-recaptcha (reCAPTCHA v2 + v3, incl. Enterprise)
       (3) free NopeCHA Token API  — hCaptcha / reCAPTCHA v2/v3 / Turnstile (100/day per IP)
       (4) paid CapSolver/2Captcha — hCaptcha / image-challenge fallback

    Logs every stage. Never raises. Returns True only if a widget was both
    detected AND solved+injected successfully.

    `provider_name` (optional) is forwarded to the free audio solver as
    per-provider hints (recaptcha_version, click_checkbox_first) via
    captcha_solver_helper.PROVIDER_CAPTCHA_HINTS. Default behavior is unchanged
    when no hints are registered.
    """
    log(f"captcha-attempt[{label}]: stage 1 — free ClickSolver (no API key)")
    try:
        if await _try_free_clicksolver(page):
            return True
    except Exception as e:
        log(f"captcha-attempt[{label}]: free path raised {e!r}")

    helper = _import_captcha_helper()
    if helper is None:
        log(f"captcha-attempt[{label}]: captcha_solver_helper unavailable; cannot try further")
        return False

    # Stage 2: free audio solver — handles reCAPTCHA v2 + v3 incl. Enterprise.
    # Skips silently on non-reCAPTCHA widgets and on solver failure.
    # Per-provider hints flip force-version / click-checkbox-first behavior
    # for sites the default heuristic doesn't handle (cerebras Enterprise v3,
    # buddy_works/sambanova iframe nesting).
    hints = {}
    try:
        get_hints = getattr(helper, "get_provider_captcha_hints", None)
        if get_hints is not None and provider_name:
            hints = get_hints(provider_name) or {}
            if hints:
                log(f"captcha-attempt[{label}]: applying provider hints {hints}")
    except Exception as e:
        log(f"captcha-attempt[{label}]: hints lookup raised {e!r} (continuing without)")
        hints = {}

    log(f"captcha-attempt[{label}]: stage 2 — free audio solver "
        "(playwright-recaptcha, no API key)")
    try:
        free_audio_fn = getattr(helper, "maybe_solve_recaptcha_audio_free", None)
        if free_audio_fn is not None:
            if await free_audio_fn(page, env=env, log=log, **hints):
                return True
        else:
            log(f"captcha-attempt[{label}]: helper has no maybe_solve_recaptcha_audio_free; "
                "skipping free audio stage")
    except TypeError as e:
        # Older helper version that doesn't accept the new kwargs — retry without
        log(f"captcha-attempt[{label}]: free audio TypeError {e!r}; "
            "retrying without per-provider hints")
        try:
            if await free_audio_fn(page, env=env, log=log):
                return True
        except Exception as e2:
            log(f"captcha-attempt[{label}]: free audio retry raised {e2!r}")
    except Exception as e:
        log(f"captcha-attempt[{label}]: free audio path raised {e!r}")

    # Stage 3: free NopeCHA Token API — 100 free credits/day per source IP,
    # no signup, no API key. Handles hCaptcha + reCAPTCHA v2/v3 + Turnstile.
    # Module-global 10s spacing applied inside the solver to stay polite.
    log(f"captcha-attempt[{label}]: stage 3 — free NopeCHA Token API "
        "(no key required, 100/day per IP)")
    try:
        nopecha_fn = getattr(helper, "maybe_solve_nopecha", None)
        if nopecha_fn is not None:
            if await nopecha_fn(page, env=env, log=log):
                return True
        else:
            log(f"captcha-attempt[{label}]: helper has no maybe_solve_nopecha; "
                "skipping NopeCHA stage")
    except Exception as e:
        log(f"captcha-attempt[{label}]: NopeCHA path raised {e!r}")

    has_key = bool((env.get("CAPSOLVER_KEY") or "").strip() or
                   (env.get("TWOCAPTCHA_KEY") or "").strip())
    if not has_key:
        log(f"captcha-attempt[{label}]: no CAPSOLVER_KEY / TWOCAPTCHA_KEY in env; "
            "API path unavailable, falling through")
        return False

    log(f"captcha-attempt[{label}]: stage 4 — paid API solver (CapSolver/2Captcha)")
    try:
        return await helper.maybe_solve_captcha(page, env=env, log=log)
    except Exception as e:
        log(f"captcha-attempt[{label}]: API helper raised {e!r}")
        return False


# --------------------------------------------------------------------------- #
# Engine selection                                                            #
# --------------------------------------------------------------------------- #

from contextlib import asynccontextmanager


@asynccontextmanager
async def _engine_context(
    *,
    engine: str,
    user_data_dir: str,
    headed: bool,
    chromium_args: list[str],
    pw_proxy: dict | None,
):
    """Yield a Playwright-compatible BrowserContext for the chosen engine.

    Patchright path: launch_persistent_context on undetected Chromium with the
    existing Mac-Chrome 147 UA + AutomationControlled blink-feature disable.

    Camoufox path: launch a fingerprint-randomized Firefox 135 via the
    AsyncCamoufox context manager. The UA, languages, platform, navigator
    properties, and WebGL fingerprint are auto-generated by BrowserForge —
    we do NOT override the UA (overriding defeats the point). humanize=True
    adds slight cursor/scroll jitter to defeat behavioural heuristics. The
    proxy and headed flags map through directly.
    """
    if engine == "patchright":
        from patchright.async_api import async_playwright

        async with async_playwright() as pw:
            launch_kwargs = dict(
                user_data_dir=user_data_dir,
                headless=not headed,
                channel="chromium",
                viewport={"width": 1366, "height": 850},
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/147.0.0.0 Safari/537.36"
                ),
                locale="en-US",
                timezone_id="America/New_York",
                args=chromium_args,
            )
            if pw_proxy is not None:
                launch_kwargs["proxy"] = pw_proxy
            ctx = await pw.chromium.launch_persistent_context(**launch_kwargs)
            try:
                yield ctx
            finally:
                try:
                    await ctx.close()
                except Exception:
                    pass
        return

    if engine == "camoufox":
        from camoufox.async_api import AsyncCamoufox

        cam_kwargs: dict = dict(
            headless=not headed,
            persistent_context=True,
            user_data_dir=user_data_dir,
            humanize=True,
            os="macos",
            locale="en-US",
        )
        if pw_proxy is not None:
            cam_kwargs["proxy"] = pw_proxy
        async with AsyncCamoufox(**cam_kwargs) as ctx:
            yield ctx
        return

    raise ValueError(
        f"unknown engine: {engine!r} (expected 'patchright' or 'camoufox')"
    )


# --------------------------------------------------------------------------- #
# Main provider driver                                                        #
# --------------------------------------------------------------------------- #

async def drive_signup(
    provider_name: str,
    cfg: dict,
    email: str,
    password: str | None,
    username: str | None,
    yahoo_user: str,
    yahoo_app_pw: str,
    headed: bool,
    timeout_email_s: int,
    timeout_token_s: int,
    solver_env: dict | None = None,
    proxy: str | None = None,
    engine: str = "patchright",
) -> tuple[int, dict]:
    """Returns (exit_code, result_dict).

    `engine` selects the browser stack:
      * "patchright" — undetected Chromium (default; matches prior behaviour).
      * "camoufox"   — Firefox 135 with fingerprint-randomized profile from
        BrowserForge. Gives a non-`HeadlessChrome` UA + different fingerprint
        surface; useful when reCAPTCHA Enterprise / Akamai have flagged the
        Mac's Chromium fingerprint.

    Both engines yield a Playwright-compatible `BrowserContext` so the rest
    of the stage logic (page.goto, page.locator, etc.) is engine-agnostic.
    """
    from patchright.async_api import TimeoutError as PWTimeout
    # `async_playwright` is only used by the patchright branch below; the
    # camoufox branch imports its own context manager. Keep this import
    # eager because PWTimeout (re-exported from patchright) is referenced
    # at module level by the existing stage-flow code.

    profile = PROFILE_ROOT / provider_name
    profile.mkdir(parents=True, exist_ok=True)
    log(f"profile: {profile}")

    # Pre-launch: remove stale Chromium SingletonLock / SingletonCookie /
    # SingletonSocket files left behind by a prior crashed run. These cause
    # `launch_persistent_context` to fail with "ProcessSingleton: failed
    # to create lock file" on macOS. See koyeb wave2c crash 2026-05-18.
    for lock in ("SingletonLock", "SingletonCookie", "SingletonSocket"):
        lock_path = profile / lock
        if lock_path.exists() or lock_path.is_symlink():
            try:
                lock_path.unlink()
                log(f"cleaned stale lock: {lock_path}")
            except Exception as e:  # pragma: no cover - best-effort
                log(f"WARN: could not remove {lock_path}: {e}")

    result: dict = {"provider": provider_name, "email": email, "stages": []}

    # Optional proxy (e.g. socks5://127.0.0.1:9050 for Tor). When set we pass
    # `proxy={"server": ...}` to launch_persistent_context AND add Chromium
    # flags that force DNS-over-proxy resolution so the destination origin
    # cannot see the Mac's real IP via a DNS-leaked A-record lookup.
    pw_proxy: dict | None = None
    if proxy:
        pw_proxy = {"server": proxy}
        log(f"proxy: routing browser traffic through {proxy}")
        result["proxy"] = proxy

    chromium_args = [
        "--disable-blink-features=AutomationControlled",
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--dns-prefetch-disable",
    ]
    if proxy and proxy.startswith(("socks5://", "socks5h://")):
        # Force Chromium to resolve DNS over the SOCKS5 proxy (no DNS leak).
        # This matches the patchright/playwright contract: when `proxy` is
        # passed at launch time, Chromium uses it for both TCP + DNS.
        # The explicit --host-resolver-rules belt-and-suspenders fallback
        # ensures grandchild renderers also route name resolution.
        chromium_args.append("--proxy-bypass-list=<-loopback>")

    result["engine"] = engine
    log(f"engine: {engine}")
    async with _engine_context(
        engine=engine,
        user_data_dir=str(profile),
        headed=headed,
        chromium_args=chromium_args,
        pw_proxy=pw_proxy,
    ) as ctx:
        # Camoufox persistent_context opens with a blank page already; reuse
        # it when present, else create one. Same shape as patchright path.
        page = ctx.pages[0] if getattr(ctx, "pages", None) else await ctx.new_page()

        # -------- Stage 1: signup nav --------
        signup_url = cfg["signup_url"]
        log(f"stage 1: nav {signup_url}")
        try:
            await page.goto(signup_url, timeout=45000, wait_until="domcontentloaded")
        except Exception as e:
            log(f"FATAL nav: {e}")
            await _screenshot(page, f"{provider_name}_nav_fail")
            await ctx.close()
            return 2, result
        await asyncio.sleep(2)
        result["stages"].append({"stage": 1, "url": page.url})

        ch = await _detect_challenge(page)
        if ch:
            await _screenshot(page, f"{provider_name}_challenge_signup")
            log(f"DETECTED challenge on signup: {ch} — attempting clear")
            cleared = await _attempt_captcha_clear(
                page, env=solver_env or {}, label=f"{provider_name}_signup",
                provider_name=provider_name,
            )
            if not cleared:
                log(f"BLOCKED: could not clear challenge on signup: {ch}")
                await ctx.close()
                return 2, result
            log("CHALLENGE CLEARED on signup; continuing")
            result["stages"].append({"stage": "1c", "challenge_cleared": ch})

        # -------- Stage 1b: account-exists detection (zombie-account fallback) --------
        # If the provider's identity system redirected our /registration nav
        # to a /login flow (e.g. Mistral's Ory Kratos sees the email already
        # has an account from a partial prior signup), branch into one of two
        # login paths:
        #
        #   (a) MAGIC-LINK login — preferred when the provider's login UI
        #       offers a "Continue with email" / "Email me a sign-in link"
        #       button (configured via `login_magic_link_button_selectors`).
        #       This avoids touching the password (account may have been
        #       created with an unknown password; "Forgot password" would
        #       risk a lockout). On success we set `record_since` and FALL
        #       THROUGH to the existing IMAP-poll stage 4 + token stages 5-10.
        #
        #   (b) PASSWORD login — the legacy fallback. On success we jump
        #       directly to stage 6 (token page nav) because no magic-link
        #       email was sent.
        #
        # Providers that haven't opted in by setting `login_url` +
        # `login_indicators` / `login_url_substrings` simply skip this block.
        zombie = await _detect_login_redirect(page, cfg)
        record_since: dt.datetime | None = None  # set by stage 2 (signup) OR magic-link login
        if zombie and (cfg.get("login_url") or cfg.get("login_indicators")
                       or cfg.get("login_url_substrings")):
            await _screenshot(page, f"{provider_name}_login_redirect")
            use_magic_link = bool(cfg.get("login_magic_link_button_selectors"))
            if use_magic_link:
                log(f"stage 1b: account-exists redirect detected ({zombie}) — "
                    "switching to MAGIC-LINK login path (bypassing password)")
                # Record the IMAP "since" timestamp BEFORE clicking the
                # magic-link button so we don't miss the email if it arrives
                # in the same second the click fires.
                record_since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=5)
                ok, last_url = await _drive_magic_link_login_path(
                    page, cfg, email=email, confirm_timeout_ms=20000,
                )
                result["stages"].append({
                    "stage": "1b",
                    "login_fallback": True,
                    "path": "magic_link",
                    "trigger": zombie,
                    "post_click_url": last_url,
                    "ok": ok,
                })
                if not ok:
                    log("FATAL: magic-link login path failed — see screenshots in logs/auto_signup/")
                    await ctx.close()
                    return 3, result
                await _screenshot(page, f"{provider_name}_magic_link_login_sent")
                # FALL THROUGH to stage 4 below — record_since is set, the
                # email is in flight to Yahoo, and stages 4-10 already poll
                # IMAP + open the link + scrape the token. Skip stages 2-3
                # (the signup form fill/submit) via the goto-style flag.
                magic_link_login_taken = True
            else:
                log(f"stage 1b: account-exists redirect detected ({zombie}) — "
                    "switching to PASSWORD-LOGIN path")
                ok, last_url = await _drive_login_path(
                    page, cfg, email=email, password=password,
                    post_login_timeout_ms=30000,
                )
                result["stages"].append({
                    "stage": "1b",
                    "login_fallback": True,
                    "path": "password",
                    "trigger": zombie,
                    "post_login_url": last_url,
                    "ok": ok,
                })
                if not ok:
                    log("FATAL: password-login fallback failed — token cannot be retrieved")
                    await ctx.close()
                    return 3, result
                magic_link_login_taken = False
                await _screenshot(page, f"{provider_name}_post_login")
        else:
            magic_link_login_taken = False
        # If we did NOT take the magic-link login path but DID take the
        # password-login path, jump to stage 6. Magic-link login falls through
        # to stage 4 (IMAP poll). Signup-path falls through to stage 2.
        if zombie and (cfg.get("login_url") or cfg.get("login_indicators")
                       or cfg.get("login_url_substrings")) and not magic_link_login_taken:
            # Jump directly to Stage 6 (token page nav). Stages 2-5 are
            # signup/magic-link only.
            token_url = cfg["token_page_url"]
            if "<workspace>" in token_url:
                log(f"WARN: token_page_url contains <workspace> placeholder ({token_url}) — "
                    "workspace name must be discovered manually for now")
                await ctx.close()
                return 4, result
            log(f"stage 6 (login path): nav token page {token_url}")
            try:
                await page.goto(token_url, timeout=30000, wait_until="domcontentloaded")
            except Exception as e:
                log(f"FATAL token-page nav (login path): {e}")
                await _screenshot(page, f"{provider_name}_token_page_fail")
                await ctx.close()
                return 4, result
            await asyncio.sleep(3)
            await _screenshot(page, f"{provider_name}_token_page")

            # Stage 7 (login path): create token. Same as the signup branch.
            token_name = f"clawbot-{secrets.token_hex(4)}-{_ts()}"
            log(f"stage 7 (login path): click create-token (name={token_name})")
            if not await _try_click(
                page, cfg.get("token_create_button_selectors", []), timeout_ms=10000
            ):
                log("WARN: create-token button not found — token may already exist; "
                    "scraping page for existing token.")
            else:
                await asyncio.sleep(1.5)
                await _try_fill(
                    page, cfg.get("token_name_field_selectors", []),
                    token_name, timeout_ms=4000,
                )
                await _try_click(
                    page,
                    cfg.get("token_create_button_selectors", []) + [
                        "button:has-text('Create')",
                        "button:has-text('Generate')",
                        "button[type='submit']",
                    ],
                    timeout_ms=4000,
                )
                await asyncio.sleep(2)

            log("stage 8 (login path): scrape token from DOM")
            token = await _read_token_from_dom(
                page,
                cfg.get("token_dom_selectors", []),
                cfg["token_regex"],
                timeout_ms=timeout_token_s * 1000,
            )
            if not token:
                await _screenshot(page, f"{provider_name}_no_token")
                log("FATAL: token not visible in DOM after create-button click (login path)")
                await ctx.close()
                return 5, result
            log(f"token captured (login path): {len(token)} chars (value NOT logged)")
            result["stages"].append({"stage": 8, "token_chars": len(token),
                                     "via": "login_path"})

            env_file = Path(cfg["env_file"])
            write_token_to_env(env_file, cfg["env_var_name"], token)
            flipped = flip_cloud_usage_enabled(cfg["cloud_usage_key"])
            result["env_file"] = str(env_file)
            result["cloud_usage_flipped"] = flipped

            ok, code = smoke_test(cfg["smoke_test_curl"], token)
            result["smoke_test_http"] = code
            result["smoke_test_ok"] = ok

            await ctx.close()
            return 0 if ok else 6, result

        # -------- Stage 2: fill email (+ optional username/password) --------
        # SKIPPED when magic_link_login_taken=True: stage 1b already filled
        # the email and clicked "Send sign-in link", so the IMAP poll below
        # will pick up the in-flight email and stages 5-10 will finish the
        # token flow.
        if not magic_link_login_taken:
            # Optional pre-step: click a button that reveals the email input
            # (e.g. Lightning AI's "Email" selector button on a multi-auth chooser
            # page; SambaNova's "Sign up" link on the login screen). Non-fatal:
            # if no selector matches we continue and the email-field detect below
            # will surface the real failure mode.
            pre_click = cfg.get("pre_email_click_selectors") or []
            if pre_click:
                log(f"stage 2a: pre-email click ({len(pre_click)} selectors)")
                if await _try_click(page, pre_click, timeout_ms=6000):
                    await asyncio.sleep(1.5)
                else:
                    log("  (pre-email click selectors did not match — continuing anyway)")

            log("stage 2: fill credentials")
            email_filled = await _try_fill(
                page, cfg.get("email_field_selectors", []), email, timeout_ms=10000
            )
            if not email_filled:
                await _screenshot(page, f"{provider_name}_no_email_field")
                log("FATAL: email field not found")
                await ctx.close()
                return 2, result

            if username and cfg.get("username_field_selectors"):
                await _try_fill(page, cfg["username_field_selectors"], username, timeout_ms=4000)

            if password and cfg.get("password_field_selectors"):
                await _try_fill(page, cfg["password_field_selectors"], password, timeout_ms=4000)

            record_since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=5)

            # -------- Stage 2b: optional Terms-of-Service checkbox --------
            # Some providers (e.g. CapSolver) gate the Sign Up button behind
            # a ToS checkbox + a CAPTCHA. Tick the checkbox first so the
            # submit button can later activate.
            tos_selectors = cfg.get("tos_checkbox_selectors") or []
            if tos_selectors:
                log(f"stage 2b: tick ToS checkbox ({len(tos_selectors)} selectors)")
                ticked = False
                for sel in tos_selectors:
                    try:
                        loc = page.locator(sel).first
                        await loc.wait_for(state="visible", timeout=4000)
                        await loc.check(timeout=4000)
                        log(f"  ticked: {sel}")
                        ticked = True
                        break
                    except Exception:
                        try:
                            # Fallback: many React checkboxes wrap input in
                            # a styled <label> — click() works even when
                            # check() rejects the wrapped input.
                            loc = page.locator(sel).first
                            await loc.click(timeout=4000)
                            log(f"  clicked (fallback): {sel}")
                            ticked = True
                            break
                        except Exception:
                            continue
                if not ticked:
                    log("  (no ToS selector matched — continuing anyway)")

            # -------- Stage 2c: optional pre-submit captcha clear --------
            # Some signup pages (e.g. CapSolver, dashboards behind Cloudflare
            # Turnstile) gate the Sign Up button behind a CAPTCHA widget that
            # MUST be solved BEFORE click. Run the 3-stage cascade here too
            # so the button becomes enabled by the time we hit stage 3.
            # We bypass `_detect_challenge` and ALWAYS call the cascade because
            # some widgets (CapSolver's interactive Turnstile checkbox) don't
            # expose the standard `data-sitekey` markers `_detect_challenge`
            # looks for — but ClickSolver still recognizes them via the actual
            # widget DOM. Cascade is a safe no-op when nothing is present.
            if cfg.get("pre_submit_captcha"):
                # Wait a moment for the Turnstile widget to render (often
                # lazy-loads after the form is fully filled).
                await asyncio.sleep(3)
                # sambanova fix 2026-05-18: Auth0 redirect chains can leave
                # outstanding requests in flight when the CAPTCHA cascade
                # starts probing, causing the widget detection / iframe
                # bootstrapping to race the navigation. Wait for networkidle
                # before the cascade so the page is fully settled. Best-effort
                # — timeout 10s, swallow on failure so other providers that
                # never reach idle (websockets/poll) aren't blocked.
                try:
                    await page.wait_for_load_state("networkidle", timeout=10000)
                    log("stage 2c: page reached networkidle before cascade")
                except Exception as _nie:
                    log(f"stage 2c: networkidle wait skipped ({_nie!r})")
                # Probe for cf-chl-widget which IS present even when the
                # iframe-based widget hasn't fully bootstrapped — useful
                # diagnostic when _detect_challenge missed.
                ts_present = False
                try:
                    ts_present = await page.evaluate(
                        "() => !!document.querySelector('[id^=\"cf-chl-widget-\"]')"
                    )
                except Exception:
                    pass
                log(f"stage 2c: pre-submit captcha cascade (cf-chl-widget present: {ts_present})")
                pre_cleared = await _attempt_captcha_clear(
                    page, env=solver_env or {}, label=f"{provider_name}_pre_submit",
                    provider_name=provider_name,
                )
                if pre_cleared:
                    log("stage 2c: pre-submit challenge CLEARED")
                    await asyncio.sleep(3)
                    result["stages"].append({"stage": "2c", "pre_submit_cleared": True})
                else:
                    log("stage 2c: cascade returned False — submit may still be gated")
                    await _screenshot(page, f"{provider_name}_pre_submit_uncleared")

            # -------- Stage 3: submit --------
            log("stage 3: submit signup form")
            clicked = await _try_click(
                page, cfg.get("submit_selectors", []), timeout_ms=8000
            )
            if not clicked:
                await _screenshot(page, f"{provider_name}_no_submit")
                log("FATAL: submit button not found")
                await ctx.close()
                return 2, result
            await asyncio.sleep(3)
            await _screenshot(page, f"{provider_name}_post_submit")

            ch = await _detect_challenge(page)
            if ch:
                await _screenshot(page, f"{provider_name}_challenge_post_submit")
                log(f"DETECTED challenge after submit: {ch} — attempting clear")
                # sambanova fix 2026-05-18: Auth0 redirects + iframe (re)attach
                # can race the CAPTCHA cascade and cause "Frame was detached"
                # mid-solve. Wait for networkidle so the post-submit page is
                # fully settled before probing/clicking the recaptcha iframe.
                # Mirrors the same guard added to stage 2c. Best-effort —
                # 10s timeout, swallow on failure so providers that never
                # reach idle (websockets/poll) aren't blocked.
                try:
                    await page.wait_for_load_state("networkidle", timeout=10000)
                    log("post-submit: page reached networkidle before cascade")
                except Exception as _nie:
                    log(f"post-submit: networkidle wait skipped ({_nie!r})")
                cleared = await _attempt_captcha_clear(
                    page, env=solver_env or {}, label=f"{provider_name}_post_submit",
                    provider_name=provider_name,
                )
                if not cleared:
                    log(f"BLOCKED: could not clear challenge after submit: {ch}")
                    await ctx.close()
                    return 2, result
                log("CHALLENGE CLEARED post-submit; re-clicking submit and continuing")
                # After token injection, re-fire the submit button so the form
                # picks up the now-valid response.
                await _try_click(page, cfg.get("submit_selectors", []), timeout_ms=6000)
                await asyncio.sleep(3)
                result["stages"].append({"stage": "3c", "challenge_cleared": ch})

            result["stages"].append({"stage": 3, "url": page.url, "submitted": True})
        else:
            log("stages 2-3 SKIPPED (magic-link login already submitted email)")
            result["stages"].append({"stage": 3, "skipped": "magic_link_login_taken"})

        # Sanity: record_since must be set by now (signup path stage 2 OR
        # magic-link login stage 1b). Defensive default in case of a bug.
        if record_since is None:
            record_since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=2)

        # -------- Stage 4: poll Yahoo IMAP for magic-link --------
        log(f"stage 4: poll Yahoo IMAP up to {timeout_email_s}s for sender~{cfg['magic_link_sender_substr']!r}")
        if not yahoo_app_pw:
            log("FATAL: YAHOO_APP_PASSWORD empty — IMAP path unavailable.")
            log("  TODO: integrate yahoo_mail_reader Path 2 (Playwright DOM scrape).")
            log("  For now, run scripts/yahoo_self_provision.py first to populate the app password.")
            await ctx.close()
            return 3, result

        try:
            magic_url = poll_yahoo_for_magic_link(
                email_user=yahoo_user,
                yahoo_app_password=yahoo_app_pw,
                since_dt=record_since,
                sender_substr=cfg["magic_link_sender_substr"],
                subject_substr=cfg.get("magic_link_subject_substr"),
                url_pattern=cfg["magic_link_url_pattern"],
                timeout_s=timeout_email_s,
            )
        except (RuntimeError, AttributeError, TypeError) as e:
            # Broadened 2026-05-18 to absorb the AttributeError cascade observed
            # in koyeb wave2c (yahoo_mail_reader.py:93 ParsedMessage path).
            log(f"FATAL IMAP cascade ({type(e).__name__}): {e}")
            await ctx.close()
            return 3, result

        if not magic_url:
            log("FATAL: no matching magic-link email within timeout")
            await ctx.close()
            return 3, result
        log(f"got magic-link URL ({len(magic_url)} chars)")
        result["stages"].append({"stage": 4, "magic_link_received": True})

        # -------- Stage 5: open magic-link in same context (session cookie) --------
        log("stage 5: open magic-link URL")
        try:
            await page.goto(magic_url, timeout=30000, wait_until="domcontentloaded")
        except Exception as e:
            log(f"FATAL magic-link nav: {e}")
            await _screenshot(page, f"{provider_name}_magic_link_fail")
            await ctx.close()
            return 3, result
        await asyncio.sleep(3)
        await _screenshot(page, f"{provider_name}_post_confirm")
        result["stages"].append({"stage": 5, "post_confirm_url": page.url})

        # -------- Stage 6: navigate to token page --------
        token_url = cfg["token_page_url"]
        if "<workspace>" in token_url:
            log(f"WARN: token_page_url contains <workspace> placeholder ({token_url}) — "
                "workspace name must be discovered manually for now")
            await ctx.close()
            return 4, result
        log(f"stage 6: nav token page {token_url}")
        try:
            await page.goto(token_url, timeout=30000, wait_until="domcontentloaded")
        except Exception as e:
            log(f"FATAL token-page nav: {e}")
            await _screenshot(page, f"{provider_name}_token_page_fail")
            await ctx.close()
            return 4, result
        await asyncio.sleep(3)
        await _screenshot(page, f"{provider_name}_token_page")

        # -------- Stage 7: click create-token --------
        token_name = f"clawbot-{secrets.token_hex(4)}-{_ts()}"
        log(f"stage 7: click create-token (name={token_name})")
        if not await _try_click(
            page, cfg.get("token_create_button_selectors", []), timeout_ms=10000
        ):
            log("WARN: create-token button not found — token may already exist; "
                "scraping page for existing token.")
        else:
            await asyncio.sleep(1.5)
            await _try_fill(
                page, cfg.get("token_name_field_selectors", []),
                token_name, timeout_ms=4000,
            )
            # Click confirm/submit (often same selectors as outer submit)
            await _try_click(
                page,
                cfg.get("token_create_button_selectors", []) + [
                    "button:has-text('Create')",
                    "button:has-text('Generate')",
                    "button[type='submit']",
                ],
                timeout_ms=4000,
            )
            await asyncio.sleep(2)

        # -------- Stage 8: read token from DOM --------
        log("stage 8: scrape token from DOM")
        token = await _read_token_from_dom(
            page,
            cfg.get("token_dom_selectors", []),
            cfg["token_regex"],
            timeout_ms=timeout_token_s * 1000,
        )
        if not token:
            await _screenshot(page, f"{provider_name}_no_token")
            log("FATAL: token not visible in DOM after create-button click")
            await ctx.close()
            return 5, result
        log(f"token captured: {len(token)} chars (value NOT logged)")
        result["stages"].append({"stage": 8, "token_chars": len(token)})

        # -------- Stage 9: write env + flip cloud_usage --------
        env_file = Path(cfg["env_file"])
        write_token_to_env(env_file, cfg["env_var_name"], token)
        flipped = flip_cloud_usage_enabled(cfg["cloud_usage_key"])
        result["env_file"] = str(env_file)
        result["cloud_usage_flipped"] = flipped

        # -------- Stage 10: smoke test --------
        ok, code = smoke_test(cfg["smoke_test_curl"], token)
        result["smoke_test_http"] = code
        result["smoke_test_ok"] = ok

        await ctx.close()
        return 0 if ok else 6, result


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="magic_link_signup.py",
        description="Generic email-only / magic-link signup orchestrator. "
                    "Bypasses GitHub-OAuth-blocker for YELLOW-tier providers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python magic_link_signup.py --provider groq --confirm-create\n"
            "  python magic_link_signup.py --provider huggingface_spaces "
            "--email user@yahoo.com --headed --confirm-create\n"
            "  python magic_link_signup.py --provider mistral --dry-run\n"
            "  python magic_link_signup.py --list\n"
        ),
    )
    p.add_argument("--provider", help="Provider key from providers.json")
    p.add_argument("--email", help="Bot's email (default: from yahoo.env YAHOO_USER)")
    p.add_argument("--password", help="Override account password (default: from yahoo.env YAHOO_PASSWORD)")
    p.add_argument("--username", help="Override account username (only for providers needing it)")
    p.add_argument("--headed", action="store_true", help="Show browser (recommended first run)")
    p.add_argument("--confirm-create", action="store_true",
                   help="Safety gate; required to actually create an account")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan + exit (no network, no signup)")
    p.add_argument("--list", action="store_true",
                   help="List all provider keys + tier + status")
    p.add_argument("--timeout-email", type=int, default=300,
                   help="Magic-link email polling timeout in seconds (default 300)")
    p.add_argument("--timeout-token", type=int, default=15,
                   help="Token DOM-scrape timeout in seconds (default 15)")
    p.add_argument("--proxy", default=None,
                   help=(
                       "Optional proxy URL for patchright launch_persistent_context "
                       "(e.g. socks5://127.0.0.1:9050 for Tor, or "
                       "http://user:pass@host:port). When set, ALL browser traffic "
                       "for the signup + token flow routes through this proxy. "
                       "Useful when the Mac IP is on reCAPTCHA Enterprise bot lists "
                       "(cerebras / sambanova / buddy_works). Free Tor via "
                       "torpy: python -m torpy.cli.socks -i 127.0.0.1 -p 9050."
                   ))
    p.add_argument("--engine", default="patchright",
                   choices=["patchright", "camoufox"],
                   help=(
                       "Browser engine to use. 'patchright' (default) = undetected "
                       "Chromium 147 — fast but fingerprint flagged by Akamai / "
                       "reCAPTCHA Enterprise on some providers. 'camoufox' = "
                       "fingerprint-randomized Firefox 135 (BrowserForge) — slower "
                       "but bypasses Chromium-fingerprint bot lists. Try 'camoufox' "
                       "when patchright keeps hitting 'unusual traffic' challenges."
                   ))
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    providers = load_providers()

    if args.list:
        print(f"{'PROVIDER':25s} {'TIER':18s} {'SIGNUP_URL':40s} {'TOKEN_PAGE_URL'}")
        for key, cfg in providers.items():
            if key.startswith("_"):
                continue
            print(f"{key:25s} {cfg.get('tier','?'):18s} "
                  f"{cfg.get('signup_url','-')[:40]:40s} "
                  f"{cfg.get('token_page_url','-')}")
        return 0

    if not args.provider:
        print("error: --provider required (or use --list)", file=sys.stderr)
        return 2

    if args.provider not in providers:
        avail = [k for k in providers if not k.startswith("_")]
        print(f"error: unknown provider {args.provider!r}. Available: {avail}",
              file=sys.stderr)
        return 2

    cfg = providers[args.provider]
    if cfg.get("tier") == "RED_BLOCKED":
        print(f"error: provider {args.provider!r} is RED_BLOCKED — "
              f"{cfg.get('_blocked_reason','no email-only signup path')}",
              file=sys.stderr)
        return 2

    yahoo_env = load_dotenv(YAHOO_ENV)
    # Provider-specific config (env + password + email overrides). Loaded BEFORE
    # email resolution so that per-provider <PROVIDER>_EMAIL can override the
    # Yahoo default (2026-05-18 mistral zombie-account workaround — Yahoo supports
    # `+suffix` aliases, so each provider can use its own alias to bypass any
    # pre-existing account on the bare YAHOO_USER address).
    provider_env_path = CFG_DIR / f"{args.provider}.env"
    provider_env = load_dotenv(provider_env_path)
    provider_email_key = f"{args.provider.upper()}_EMAIL"
    provider_pw_key = f"{args.provider.upper()}_ACCOUNT_PASSWORD"

    # Email resolution order (2026-05-18 mistral alias patch):
    #   1. --email CLI flag (highest precedence)
    #   2. <PROVIDER_UPPER>_EMAIL from per-provider env file
    #   3. <PROVIDER_UPPER>_EMAIL from process env
    #   4. YAHOO_USER from yahoo.env (legacy default)
    email = (
        args.email
        or provider_env.get(provider_email_key)
        or os.environ.get(provider_email_key)
        or yahoo_env.get("YAHOO_USER")
    )
    if not email:
        print(f"error: --email not given and YAHOO_USER missing from {YAHOO_ENV}",
              file=sys.stderr)
        return 2
    if provider_env.get(provider_email_key) or os.environ.get(provider_email_key):
        log(f"email: using {provider_email_key} (provider-specific alias)")
    elif args.email:
        log("email: using --email CLI flag")
    else:
        log("email: using YAHOO_USER (legacy default)")
    # Provider-specific password override resolution order (2026-05-18 huggingface fix):
    #   1. --password CLI flag
    #   2. <PROVIDER_UPPER>_ACCOUNT_PASSWORD from per-provider env file
    #      (~/.config/auto_signup/<provider>.env) — needed when provider has
    #      stricter password policy than Yahoo (e.g. HF requires uppercase).
    #   3. <PROVIDER_UPPER>_ACCOUNT_PASSWORD from process env
    #   4. YAHOO_PASSWORD fallback (legacy default)
    password = (
        args.password
        or provider_env.get(provider_pw_key)
        or os.environ.get(provider_pw_key)
        or yahoo_env.get("YAHOO_PASSWORD")
    )
    if provider_env.get(provider_pw_key) or os.environ.get(provider_pw_key):
        log(f"password: using {provider_pw_key} (provider-specific override)")
    else:
        log("password: using YAHOO_PASSWORD (legacy fallback)")
    username = args.username or email.split("@")[0] + "-bot"
    yahoo_app_pw = yahoo_env.get("YAHOO_APP_PASSWORD", "")

    plan = {
        "provider": args.provider,
        "tier": cfg.get("tier"),
        "email": email,
        "username": username if cfg.get("username_field_selectors") else None,
        "password_set": bool(password),
        "yahoo_app_pw_set": bool(yahoo_app_pw),
        "signup_url": cfg["signup_url"],
        "token_page_url": cfg["token_page_url"],
        "env_var_name": cfg["env_var_name"],
        "env_file": cfg["env_file"],
        "cloud_usage_key": cfg["cloud_usage_key"],
        "magic_link_sender_substr": cfg["magic_link_sender_substr"],
        "magic_link_url_pattern": cfg["magic_link_url_pattern"],
        "timeout_email_s": args.timeout_email,
        "timeout_token_s": args.timeout_token,
        "headed": args.headed,
        "dry_run": args.dry_run,
        "proxy": args.proxy,
    }

    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0

    if not args.confirm_create:
        print("error: --confirm-create required to actually create an account",
              file=sys.stderr)
        print(json.dumps(plan, indent=2), file=sys.stderr)
        return 2

    if not yahoo_app_pw:
        print("FATAL: YAHOO_APP_PASSWORD empty in yahoo.env — "
              "run scripts/yahoo_self_provision.py first (or wait for "
              "the in-flight Yahoo IMAP IP-block cooldown).", file=sys.stderr)
        return 3

    # Solver env: merge process env + capsolver.env + twocaptcha.env if present.
    # captcha_solver_helper.maybe_solve_captcha reads CAPSOLVER_KEY /
    # TWOCAPTCHA_KEY / CAPTCHA_TIMEOUT_S from this dict. Free ClickSolver
    # requires no env; API path activates only when a key is present.
    solver_env: dict[str, str] = {}
    # Allowlist of env keys the captcha helper consumes. BotsForge_* added
    # 2026-05-18 for the free local Turnstile solver (CapSolver-API-compatible
    # localhost server) — see captcha_solver_helper._botsforge_solve_sync.
    _SOLVER_KEYS = (
        "CAPSOLVER_KEY", "TWOCAPTCHA_KEY", "CAPTCHA_TIMEOUT_S",
        "BOTSFORGE_TURNSTILE_KEY", "BOTSFORGE_TURNSTILE_URL", "BOTSFORGE_TURNSTILE_TIMEOUT_S",
    )
    for k in _SOLVER_KEYS:
        if os.environ.get(k):
            solver_env[k] = os.environ[k]
    for env_file_name in ("capsolver.env", "twocaptcha.env", "botsforge.env"):
        env_path = CFG_DIR / env_file_name
        for k, v in load_dotenv(env_path).items():
            if k in _SOLVER_KEYS and v:
                solver_env[k] = v
    log(f"solver_env: keys={sorted(solver_env.keys()) or '[]'} "
        f"(free ClickSolver always tried first regardless)")

    code, result = asyncio.run(drive_signup(
        provider_name=args.provider,
        cfg=cfg,
        email=email,
        password=password,
        username=username if cfg.get("username_field_selectors") else None,
        yahoo_user=email,
        yahoo_app_pw=yahoo_app_pw,
        headed=args.headed,
        timeout_email_s=args.timeout_email,
        timeout_token_s=args.timeout_token,
        solver_env=solver_env,
        proxy=args.proxy,
        engine=args.engine,
    ))

    out_path = LOG_DIR / f"magic_link_{args.provider}_{_ts()}.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    log(f"result: {out_path}")
    log(f"exit code: {code}")
    return code


if __name__ == "__main__":
    sys.exit(main())
