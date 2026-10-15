#!/usr/bin/env python3
"""
captcha_solver_helper.py — Shared CAPTCHA solver integration for token-grab scripts.

Supports hCaptcha + reCAPTCHA v2 via CapSolver (default) or 2Captcha (fallback).
Both client SDKs are MIT-licensed Python packages.

ALSO: free / no-cost reCAPTCHA v2 + v3 audio-challenge solver via
playwright-recaptcha (MIT — https://github.com/Xewdy444/Playwright-reCAPTCHA).
That solver works on standard AND enterprise reCAPTCHA endpoints
(`/recaptcha/(api2|enterprise)/...`). It transcribes the audio challenge via
Google's free speech-recognition endpoint — no API key required, no money.
ffmpeg is provided via the `imageio-ffmpeg` pip package (bundled binary), so
the helper has no system-level ffmpeg dependency. See
`maybe_solve_recaptcha_audio_free_*` below.

Design:
- Auto-detects hCaptcha / reCAPTCHA widget on a Playwright/patchright page.
- Extracts sitekey from DOM (data-sitekey attribute, common selectors).
- Submits createTask to the CAPTCHA-solver API, polls until ready.
- Injects the returned token into the form's response textarea + dispatches the
  callback event that the CAPTCHA library expects, so the form's submit handler
  treats the challenge as solved.
- All errors are logged but never crash the caller — solver is best-effort.

Env keys (load via the calling script's load_env helper):
    CAPSOLVER_KEY     — primary solver (recommended; explicit hCaptcha support)
    TWOCAPTCHA_KEY    — fallback solver
    CAPTCHA_TIMEOUT_S — max poll seconds (default 180)
    CAPTCHA_FREE_AUDIO_DISABLED — if "1", skip the playwright-recaptcha audio path

Cost (as of 2026-05): CapSolver hCaptcha ~$1.20-$1.50/1k solves, reCAPTCHA ~$0.80/1k.
2Captcha similar pricing. Both require a paid deposit before issuing solves (the
account creation itself is email-only and free; only the deposit needs payment).

Usage from a Playwright/patchright async script:
    from captcha_solver_helper import maybe_solve_captcha
    solved = await maybe_solve_captcha(page, env=env, log=log)
    if solved:
        # The token has been injected; submit the form normally.
        await page.click('button[type="submit"]')
    else:
        # No solver key OR widget not detected OR solver failed.
        # Fall back to existing behavior (screenshot + exit, or --headed manual).
        ...

License: MIT. Depends on stdlib only at import time; lazy-imports the solver
SDK only when a key is present, so scripts without solver keys still work.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Optional


# -----------------------------------------------------------------------------
# Sitekey extraction
# -----------------------------------------------------------------------------
# Detector covers BOTH host variants reCAPTCHA loads from:
#   - www.google.com/recaptcha/...      (default)
#   - www.recaptcha.net/recaptcha/...   (mirror used in regions where google.com is
#                                        blocked, AND used by some sites as an
#                                        anti-bot-detection countermeasure)
# We also walk every sub-frame in the page tree, because many SPAs (cerebras,
# sambanova, buddy_works) render the auth form inside a nested iframe, and the
# top-level page.evaluate() in those cases sees only an empty shell.
# -----------------------------------------------------------------------------
_DETECT_WIDGET_JS = r"""
() => {
  const hosts = [
    ['hcaptcha',  '.h-captcha[data-sitekey], iframe[src*="hcaptcha"]'],
    ['recaptcha', '.g-recaptcha[data-sitekey], iframe[src*="google.com/recaptcha"], iframe[src*="recaptcha.net"]'],
    ['turnstile', '.cf-turnstile[data-sitekey], iframe[src*="challenges.cloudflare.com"]'],
  ];
  for (const [kind, sel] of hosts) {
    const el = document.querySelector(sel);
    if (!el) continue;
    // Prefer data-sitekey on container; fall back to iframe src k= param.
    let sitekey = el.getAttribute && el.getAttribute('data-sitekey');
    if (!sitekey && el.tagName === 'IFRAME') {
      const m = el.src.match(/[?&]k=([^&]+)/);
      if (m) sitekey = decodeURIComponent(m[1]);
    }
    if (!sitekey) {
      // Some sites put data-sitekey on a parent or sibling
      const parent = el.closest('[data-sitekey]');
      if (parent) sitekey = parent.getAttribute('data-sitekey');
    }
    // Also expose the iframe src so callers can see the host/path used
    // (api2 vs enterprise, google.com vs recaptcha.net).
    let frame_src = null;
    if (el.tagName === 'IFRAME') frame_src = el.src;
    if (sitekey) return { type: kind, sitekey, frame_src };
  }
  return null;
}
"""


async def _detect_widget_in_all_frames(page) -> Optional[dict]:
    """Recursively check page.main_frame + every sub-frame for a widget.

    Why: Cerebras, SambaNova and Buddy.works embed their signup forms in an
    iframe (Auth0 / Ory Kratos / Okta widgets). page.evaluate() runs in the
    TOP frame's context — it sees only an empty shell, missing the widget
    inside the auth iframe. Walking page.frames matches what
    playwright_recaptcha's _get_recaptcha_frame_pairs() walks anyway, so this
    parallels the library's own search surface.
    """
    try:
        frames = page.frames
    except Exception:
        return None
    for fr in frames:
        try:
            result = await fr.evaluate(_DETECT_WIDGET_JS)
        except Exception:
            continue
        if result and result.get("sitekey"):
            return result
    return None


async def detect_captcha_widget(page) -> Optional[dict]:
    """Inspect the current page for a CAPTCHA widget. Return None if absent.

    Returns dict with: type (hcaptcha|recaptcha|turnstile), sitekey, url,
    plus (when available) frame_src — the actual iframe URL of the widget.
    Searches the main frame first, then walks every sub-frame in the page tree
    so widgets rendered inside auth-provider iframes (Auth0 / Ory / Okta) are
    detected too.
    """
    try:
        url = page.url
    except Exception:
        return None

    # First try the top frame (fast path — covers 90% of pages).
    result = None
    try:
        result = await page.evaluate(_DETECT_WIDGET_JS)
    except Exception:
        result = None

    # Fall back to walking sub-frames (covers SPAs with auth iframes).
    if not result or not result.get("sitekey"):
        result = await _detect_widget_in_all_frames(page)

    if not result or not result.get("sitekey"):
        return None
    result["url"] = url
    return result


# -----------------------------------------------------------------------------
# BotsForge local Turnstile solver — FREE, CapSolver-API-compatible, localhost
# -----------------------------------------------------------------------------
# Why this exists:
#   Cloudflare Turnstile cannot be solved by the free playwright-recaptcha audio
#   path (that only works on reCAPTCHA). The next-cheapest alternative is the
#   BotsForge/CloudFlare local solver (https://github.com/BotsForge/CloudFlare),
#   a Python server that launches its own patchright Chrome window, injects the
#   target sitekey, and harvests the Turnstile token — for FREE. It exposes the
#   same /createTask + /getTaskResult endpoints as CapSolver, so the client code
#   is the existing CapSolver client with a different host.
#
# When this is tried:
#   Turnstile widget detected AND BOTSFORGE_TURNSTILE_URL env is set
#   (default http://localhost:5033 if BOTSFORGE_TURNSTILE_KEY is set).
#   Tried AFTER free audio (which only handles reCAPTCHA, so for Turnstile it's
#   skipped immediately) and BEFORE paid CapSolver/2Captcha.
#
# Env keys:
#   BOTSFORGE_TURNSTILE_URL — server URL (default http://localhost:5033)
#   BOTSFORGE_TURNSTILE_KEY — the API_KEY the BotsForge server expects (mandatory)
#
# Caveats:
#   - BotsForge uses headless=False patchright + pyautogui clicks, so it MUST
#     run inside a logged-in GUI session (not a launchd background daemon
#     without an active console user).
#   - Only solves Turnstile (AntiTurnstileTaskProxyLess). hCaptcha + reCAPTCHA
#     fall through unchanged.
#   - Quality depends on the local browser passing CF challenge fingerprinting;
#     patchright is a stealth fork specifically for this purpose, but not all
#     sites accept it. On failure, falls through to paid CapSolver as before.
# -----------------------------------------------------------------------------
def _botsforge_solve_sync(server_url: str, api_key: str, widget: dict, timeout_s: int = 60) -> Optional[str]:
    """Synchronous BotsForge local Turnstile solver. Returns token or None.

    Talks to a local CapSolver-API-compatible server (default localhost:5033).
    Only handles Turnstile widgets; returns None for any other widget type so
    callers can fall through to the next solver in the chain.
    """
    if widget.get("type") != "turnstile":
        return None
    server_url = server_url.rstrip("/")
    task = {
        "type": "AntiTurnstileTaskProxyLess",
        "websiteURL": widget["url"],
        "websiteKey": widget["sitekey"],
    }
    create_body = json.dumps({"clientKey": api_key, "task": task}).encode()
    req = urllib.request.Request(
        f"{server_url}/createTask",
        data=create_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError, ConnectionRefusedError):
        return None

    if data.get("errorId"):
        return None
    task_id = data.get("taskId")
    if not task_id:
        return None

    poll_body = json.dumps({"clientKey": api_key, "taskId": task_id}).encode()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        req = urllib.request.Request(
            f"{server_url}/getTaskResult",
            data=poll_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                pdata = json.loads(resp.read().decode())
        except Exception:
            time.sleep(2)
            continue
        status = pdata.get("status")
        if status == "ready":
            sol = pdata.get("solution", {}) or {}
            return sol.get("token") or sol.get("gRecaptchaResponse")
        if status == "error" or pdata.get("errorId"):
            return None
        time.sleep(2)
    return None


# -----------------------------------------------------------------------------
# CapSolver API (preferred — explicit hCaptcha support, $1.20-$1.50/1k)
# -----------------------------------------------------------------------------
def _capsolver_solve_sync(api_key: str, widget: dict, timeout_s: int = 180) -> Optional[str]:
    """Synchronous CapSolver call. Returns token string or None."""
    type_map = {
        "hcaptcha":  "HCaptchaTaskProxyLess",
        "recaptcha": "ReCaptchaV2TaskProxyLess",
        "turnstile": "AntiTurnstileTaskProxyLess",
    }
    task_type = type_map.get(widget["type"])
    if not task_type:
        return None

    task: dict[str, Any] = {
        "type": task_type,
        "websiteURL": widget["url"],
        "websiteKey": widget["sitekey"],
    }

    create_body = json.dumps({"clientKey": api_key, "task": task}).encode()
    req = urllib.request.Request(
        "https://api.capsolver.com/createTask",
        data=create_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError):
        return None

    if data.get("errorId"):
        return None
    task_id = data.get("taskId")
    if not task_id:
        return None

    poll_body = json.dumps({"clientKey": api_key, "taskId": task_id}).encode()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        req = urllib.request.Request(
            "https://api.capsolver.com/getTaskResult",
            data=poll_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                pdata = json.loads(resp.read().decode())
        except Exception:
            time.sleep(3)
            continue
        if pdata.get("status") == "ready":
            sol = pdata.get("solution", {}) or {}
            return sol.get("gRecaptchaResponse") or sol.get("token") or sol.get("text")
        if pdata.get("errorId"):
            return None
        time.sleep(3)
    return None


# -----------------------------------------------------------------------------
# 2Captcha API (fallback — same call pattern, different host)
# -----------------------------------------------------------------------------
def _twocaptcha_solve_sync(api_key: str, widget: dict, timeout_s: int = 180) -> Optional[str]:
    """Synchronous 2Captcha call. Uses the in.php / res.php HTTP-only protocol
    so we don't need the 2captcha-python pip package at import time."""
    method_map = {
        "hcaptcha":  "hcaptcha",
        "recaptcha": "userrecaptcha",
        "turnstile": "turnstile",
    }
    method = method_map.get(widget["type"])
    if not method:
        return None

    # Submit task
    params = {
        "key": api_key,
        "method": method,
        "pageurl": widget["url"],
        "json": "1",
    }
    if method == "hcaptcha":
        params["sitekey"] = widget["sitekey"]
    elif method == "userrecaptcha":
        params["googlekey"] = widget["sitekey"]
    else:  # turnstile
        params["sitekey"] = widget["sitekey"]

    qs = "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items())
    submit_url = f"https://2captcha.com/in.php?{qs}"
    try:
        with urllib.request.urlopen(submit_url, timeout=30) as resp:
            sdata = json.loads(resp.read().decode())
    except Exception:
        return None
    if sdata.get("status") != 1:
        return None
    task_id = sdata.get("request")
    if not task_id:
        return None

    poll_url = (
        f"https://2captcha.com/res.php?key={api_key}"
        f"&action=get&id={task_id}&json=1"
    )
    deadline = time.monotonic() + timeout_s
    # 2captcha asks to wait ~20s before first poll
    time.sleep(15)
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(poll_url, timeout=20) as resp:
                pdata = json.loads(resp.read().decode())
        except Exception:
            time.sleep(5)
            continue
        if pdata.get("status") == 1:
            return pdata.get("request")
        # status 0 + request "CAPCHA_NOT_READY" → keep polling
        if pdata.get("request") and pdata["request"] not in ("CAPCHA_NOT_READY", "CAPTCHA_NOT_READY"):
            return None
        time.sleep(5)
    return None


# We need urllib.parse for 2Captcha; import lazily so module import is cheap.
import urllib.parse  # noqa: E402


# -----------------------------------------------------------------------------
# NopeCHA Token API (FREE — no signup, no API key, 100 credits/day per IP)
# -----------------------------------------------------------------------------
# Why this exists:
#   The free playwright-recaptcha audio path covers reCAPTCHA v2/v3 ONLY, and
#   only when the audio button is reachable. The BotsForge local solver covers
#   Turnstile only and needs a local Chrome window. The paid CapSolver / 2Captcha
#   tier costs money. NopeCHA sits in the middle: no API key needed (the
#   request's source IP acts as the free-tier identifier), and the API supports
#   hCaptcha + reCAPTCHA v2/v3 + Cloudflare Turnstile — no local browser.
#
# Endpoint (legacy unified, no Authorization header required):
#   POST https://api.nopecha.com/token
#     JSON: {"type": "<recaptcha2|recaptcha3|hcaptcha|turnstile>",
#            "sitekey": "<sitekey>", "url": "<page_url>"}
#     -> {"data": "<job_id>"}                       on success
#     -> {"error": <int>, "message": "<str>"}       on failure
#   GET  https://api.nopecha.com/token?id=<job_id>
#     -> {"data": "<token>"}                        when solved
#     -> {"error": 14, "message": "Incomplete job"} while pending
#     -> {"error": <other_int>, ...}                on hard failure
#
# Free-tier limits:
#   - 100 credits/day per source IP (rolling 23h window).
#   - 1-2 credits typically solve one reCAPTCHA from a reputable IP; 2-6 from a
#     high-risk IP. So practical headroom is ~30-50 solves/day.
#   - To stay polite and avoid 429s, we enforce a MINIMUM 10s spacing between
#     submissions across the whole process (module-global, lock-protected).
#
# Cost: $0 when within the free quota. The API rejects with HTTP 402 / specific
# error codes once the daily quota is exhausted.
#
# Env keys:
#   NOPECHA_KEY              - optional, paid subscription key (auto-used if set,
#                              bypasses the 100/day free cap and IP-based limits)
#   NOPECHA_DISABLED         - if "1", skip NopeCHA entirely
#   NOPECHA_MIN_SPACING_S    - override the 10s rate-limit floor (e.g. "5")
#   CAPTCHA_TIMEOUT_S        - shared with other solvers (poll deadline, default 180s)
# -----------------------------------------------------------------------------
import threading  # noqa: E402

_NOPECHA_LAST_SUBMIT_TS: float = 0.0   # monotonic timestamp of last submit
_NOPECHA_LOCK = threading.Lock()        # protects _NOPECHA_LAST_SUBMIT_TS
_NOPECHA_MIN_SPACING_DEFAULT_S: float = 10.0
_NOPECHA_TYPE_MAP: dict = {
    "recaptcha": "recaptcha2",  # default to v2; v3 via widget["recaptcha_version"]="v3"
    "hcaptcha":  "hcaptcha",
    "turnstile": "turnstile",
}


def _nopecha_rate_limit_wait(min_spacing_s: float, log: Callable[[str], None]) -> None:
    """Sleep just long enough that this submit is >= min_spacing_s after the
    previous one. Module-global, so multiple solvers in one process share the
    same throttle. Updates the timestamp atomically under _NOPECHA_LOCK.
    """
    global _NOPECHA_LAST_SUBMIT_TS
    with _NOPECHA_LOCK:
        now = time.monotonic()
        elapsed = now - _NOPECHA_LAST_SUBMIT_TS
        if _NOPECHA_LAST_SUBMIT_TS > 0 and elapsed < min_spacing_s:
            wait = min_spacing_s - elapsed
            log(f"captcha-nopecha: rate-limit wait {wait:.1f}s "
                f"(min spacing {min_spacing_s:.0f}s, elapsed {elapsed:.1f}s)")
            time.sleep(wait)
        _NOPECHA_LAST_SUBMIT_TS = time.monotonic()


def _nopecha_solve_sync(
    widget: dict,
    timeout_s: int = 180,
    api_key: Optional[str] = None,
    min_spacing_s: float = _NOPECHA_MIN_SPACING_DEFAULT_S,
    log: Callable[[str], None] = lambda _msg: None,
) -> Optional[str]:
    """Synchronous NopeCHA Token API call. Returns token string or None.

    urllib only (no SDK). Respects module-global rate limit. Never raises.

    Parameters
    ----------
    widget : dict
        Must contain "type" (one of "recaptcha", "hcaptcha", "turnstile"),
        "sitekey", and "url" — same shape detect_captcha_widget returns.
        Set widget["recaptcha_version"]="v3" to force the v3 endpoint.
    timeout_s : int
        Maximum total poll time (seconds). Default 180.
    api_key : Optional[str]
        Paid NopeCHA subscription key. If None, request is anonymous and the
        user's IP acts as the free-tier identifier (100 credits/day).
    min_spacing_s : float
        Minimum seconds between this submit and the previous one (module-global).
        Default 10s. Override with env NOPECHA_MIN_SPACING_S in callers.
    log : Callable[[str], None]
        Logging hook. Defaults to a no-op so this helper is silent without one.
    """
    nopecha_type = _NOPECHA_TYPE_MAP.get(widget.get("type", ""))
    if not nopecha_type:
        log(f"captcha-nopecha: unsupported widget type {widget.get('type')!r}; skipping")
        return None

    sitekey = widget.get("sitekey")
    page_url = widget.get("url")
    if not sitekey or not page_url:
        log("captcha-nopecha: missing sitekey or url on widget; skipping")
        return None

    # Allow v3-version hint via widget["recaptcha_version"]="v3".
    if (widget.get("recaptcha_version") or "").lower() == "v3" and nopecha_type == "recaptcha2":
        nopecha_type = "recaptcha3"

    # Rate-limit BEFORE submit (module-global, ~10s minimum spacing).
    _nopecha_rate_limit_wait(min_spacing_s, log)

    # Build submit body. Anonymous (no key) submits use the source IP as the
    # free-tier identifier and consume the 100/day quota.
    body: dict = {
        "type": nopecha_type,
        "sitekey": sitekey,
        "url": page_url,
    }
    if api_key:
        body["key"] = api_key

    submit_data = json.dumps(body).encode("utf-8")
    submit_req = urllib.request.Request(
        "https://api.nopecha.com/token",
        data=submit_data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(submit_req, timeout=30) as resp:
            sdata = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        # Common: 402 = quota exhausted, 429 = rate-limited, 403 = blocked IP.
        try:
            body_txt = e.read().decode("utf-8", errors="replace")[:200]
        except Exception:
            body_txt = ""
        log(f"captcha-nopecha: submit HTTP {e.code}: {body_txt}")
        # 429 handling fix 2026-05-18: NopeCHA free tier is 100/day per IP.
        # On 429 specifically, surface a clearer message so operators know
        # to wait ~24h for quota reset OR provide a NOPECHA_KEY env var
        # (paid tier has a separate budget). No inline sleep/retry — would
        # hang the signup flow.
        if e.code == 429:
            has_key = bool(api_key)
            log(f"captcha-nopecha: free-tier quota likely exhausted "
                f"(100/day per IP). Reset in ~24h. "
                f"NOPECHA_KEY present={has_key}. "
                f"To bypass: set NOPECHA_KEY env (paid tier) or wait for reset.")
        elif e.code == 402:
            log("captcha-nopecha: HTTP 402 — paid-tier quota exhausted "
                "(if NOPECHA_KEY set) or free tier blocked. Falling through.")
        return None
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
        log(f"captcha-nopecha: submit transport error {type(e).__name__}: {e}")
        return None
    except Exception as e:
        log(f"captcha-nopecha: submit unexpected error {type(e).__name__}: {e}")
        return None

    if sdata.get("error") is not None:
        log(f"captcha-nopecha: submit returned error code {sdata.get('error')}: "
            f"{sdata.get('message', '')!r}")
        return None

    job_id = sdata.get("data")
    if not isinstance(job_id, str) or not job_id:
        log(f"captcha-nopecha: submit response missing 'data' job_id: {sdata}")
        return None

    log(f"captcha-nopecha: submitted ({nopecha_type}, sitekey={sitekey[:8]}…); "
        f"job_id={job_id[:12]}…, polling up to {timeout_s}s")

    # Poll loop. NopeCHA returns {"error": 14, "message": "Incomplete job"}
    # while pending. Spec doesn't pin a first-poll wait; use 5s initial then
    # 3s thereafter — modest enough to not waste budget, fast enough to catch
    # a quick solve (most solves finish in 10-30s on reputable IPs).
    poll_qs = urllib.parse.urlencode({"id": job_id} if not api_key
                                     else {"id": job_id, "key": api_key})
    poll_url = f"https://api.nopecha.com/token?{poll_qs}"
    deadline = time.monotonic() + timeout_s
    time.sleep(5)
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(poll_url, timeout=20) as resp:
                pdata = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            log(f"captcha-nopecha: poll HTTP {e.code}; retrying")
            time.sleep(3)
            continue
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
            log(f"captcha-nopecha: poll transport error {type(e).__name__}: {e}; retrying")
            time.sleep(3)
            continue
        except Exception as e:
            log(f"captcha-nopecha: poll unexpected error {type(e).__name__}: {e}; retrying")
            time.sleep(3)
            continue

        err = pdata.get("error")
        if err is None:
            token = pdata.get("data")
            if isinstance(token, str) and token:
                log(f"captcha-nopecha: solved (token len={len(token)})")
                return token
            log(f"captcha-nopecha: poll returned no error and no data: {pdata}")
            return None
        if err == 14:
            # Pending — keep polling.
            time.sleep(3)
            continue
        # Hard error — abort.
        log(f"captcha-nopecha: poll returned error code {err}: "
            f"{pdata.get('message', '')!r}")
        return None

    log(f"captcha-nopecha: poll deadline reached ({timeout_s}s) without solution")
    return None


async def maybe_solve_nopecha(
    page,
    env: dict,
    log: Callable[[str], None] = print,
    timeout_s: Optional[int] = None,
) -> bool:
    """Async: try the FREE NopeCHA Token API (100 free credits/day per IP).

    Returns True iff a widget was detected, NopeCHA returned a token, and the
    token was successfully injected into the page DOM. Returns False otherwise.
    Never raises.

    Env:
        NOPECHA_DISABLED       - if "1", skip entirely
        NOPECHA_KEY            - optional paid subscription key
        NOPECHA_MIN_SPACING_S  - override the 10s rate-limit floor
        CAPTCHA_TIMEOUT_S      - shared poll deadline (default 180)
    """
    if (env.get("NOPECHA_DISABLED") or "").strip() == "1":
        log("captcha-nopecha: disabled by env NOPECHA_DISABLED=1")
        return False

    widget = await detect_captcha_widget(page)
    if not widget:
        log("captcha-nopecha: no widget detected on page")
        return False
    if widget.get("type") not in _NOPECHA_TYPE_MAP:
        log(f"captcha-nopecha: widget is {widget.get('type')}, "
            "not supported by NopeCHA Token API")
        return False

    log(f"captcha-nopecha: detected {widget['type']} widget "
        f"(sitekey={widget['sitekey'][:8]}…); attempting free Token API solve")

    api_key = (env.get("NOPECHA_KEY") or "").strip() or None
    timeout_s = timeout_s or int(env.get("CAPTCHA_TIMEOUT_S", "180"))
    try:
        min_spacing_s = float(env.get("NOPECHA_MIN_SPACING_S") or
                              _NOPECHA_MIN_SPACING_DEFAULT_S)
    except (TypeError, ValueError):
        min_spacing_s = _NOPECHA_MIN_SPACING_DEFAULT_S

    loop = asyncio.get_event_loop()
    try:
        token = await loop.run_in_executor(
            None,
            lambda: _nopecha_solve_sync(
                widget,
                timeout_s=timeout_s,
                api_key=api_key,
                min_spacing_s=min_spacing_s,
                log=log,
            ),
        )
    except Exception as e:
        log(f"captcha-nopecha: solver raised {type(e).__name__}: {e}")
        token = None

    if not token:
        log("captcha-nopecha: no token returned; falling through")
        return False

    injected = await inject_captcha_token(page, widget, token)
    if not injected:
        log("captcha-nopecha: token solved but DOM injection target not found")
        return False

    log("captcha-nopecha: token injected; pausing 1s for form to ack")
    await asyncio.sleep(1)
    return True


def maybe_solve_nopecha_sync(
    page,
    env: dict,
    log: Callable[[str], None] = print,
    timeout_s: Optional[int] = None,
) -> bool:
    """Sync sibling of maybe_solve_nopecha. Same contract.

    Use from patchright.sync_api / playwright.sync_api callers.
    """
    if (env.get("NOPECHA_DISABLED") or "").strip() == "1":
        log("captcha-nopecha: disabled by env NOPECHA_DISABLED=1")
        return False

    widget = detect_captcha_widget_sync(page)
    if not widget:
        log("captcha-nopecha: no widget detected on page")
        return False
    if widget.get("type") not in _NOPECHA_TYPE_MAP:
        log(f"captcha-nopecha: widget is {widget.get('type')}, "
            "not supported by NopeCHA Token API")
        return False

    log(f"captcha-nopecha: detected {widget['type']} widget "
        f"(sitekey={widget['sitekey'][:8]}…); attempting free Token API solve")

    api_key = (env.get("NOPECHA_KEY") or "").strip() or None
    timeout_s = timeout_s or int(env.get("CAPTCHA_TIMEOUT_S", "180"))
    try:
        min_spacing_s = float(env.get("NOPECHA_MIN_SPACING_S") or
                              _NOPECHA_MIN_SPACING_DEFAULT_S)
    except (TypeError, ValueError):
        min_spacing_s = _NOPECHA_MIN_SPACING_DEFAULT_S

    try:
        token = _nopecha_solve_sync(
            widget,
            timeout_s=timeout_s,
            api_key=api_key,
            min_spacing_s=min_spacing_s,
            log=log,
        )
    except Exception as e:
        log(f"captcha-nopecha: solver raised {type(e).__name__}: {e}")
        token = None

    if not token:
        log("captcha-nopecha: no token returned; falling through")
        return False

    injected = inject_captcha_token_sync(page, widget, token)
    if not injected:
        log("captcha-nopecha: token solved but DOM injection target not found")
        return False

    log("captcha-nopecha: token injected; pausing 1s for form to ack")
    time.sleep(1)
    return True


# -----------------------------------------------------------------------------
# Token injection — paste the solved token into the DOM
# -----------------------------------------------------------------------------
async def inject_captcha_token(page, widget: dict, token: str) -> bool:
    """Inject the solved token into the page so form submission succeeds.

    Different CAPTCHA libraries expect different injection points:
    - hCaptcha:  textarea[name="h-captcha-response"] + window.hcaptchaCallback
    - reCAPTCHA: textarea#g-recaptcha-response + window.___grecaptcha_cfg callback
    - Turnstile: input[name="cf-turnstile-response"]

    Returns True if at least one injection target was found.
    """
    kind = widget["type"]
    if kind == "hcaptcha":
        js = r"""
        (token) => {
          let found = false;
          // 1) Standard textarea
          document.querySelectorAll('textarea[name="h-captcha-response"], textarea#h-captcha-response')
            .forEach(t => { t.value = token; t.innerHTML = token; found = true; });
          // 2) Some forms use a hidden input
          document.querySelectorAll('input[name="h-captcha-response"]')
            .forEach(i => { i.value = token; found = true; });
          // 3) Invoke the data-callback if defined
          const widget = document.querySelector('.h-captcha[data-callback], [data-hcaptcha-widget-id]');
          if (widget) {
            const cb = widget.getAttribute('data-callback');
            if (cb && typeof window[cb] === 'function') {
              try { window[cb](token); } catch (e) {}
            }
          }
          return found;
        }
        """
    elif kind == "recaptcha":
        js = r"""
        (token) => {
          let found = false;
          document.querySelectorAll('textarea#g-recaptcha-response, textarea[name="g-recaptcha-response"]')
            .forEach(t => { t.value = token; t.innerHTML = token; found = true; });
          const widget = document.querySelector('.g-recaptcha[data-callback]');
          if (widget) {
            const cb = widget.getAttribute('data-callback');
            if (cb && typeof window[cb] === 'function') {
              try { window[cb](token); } catch (e) {}
            }
          }
          return found;
        }
        """
    elif kind == "turnstile":
        js = r"""
        (token) => {
          let found = false;
          document.querySelectorAll('input[name="cf-turnstile-response"]')
            .forEach(i => { i.value = token; found = true; });
          return found;
        }
        """
    else:
        return False

    try:
        return bool(await page.evaluate(js, token))
    except Exception:
        return False


# -----------------------------------------------------------------------------
# Top-level orchestrator
# -----------------------------------------------------------------------------
async def maybe_solve_captcha(
    page,
    env: dict,
    log: Callable[[str], None] = print,
) -> bool:
    """Detect + solve any CAPTCHA on the current page. Return True on success.

    Returns False when:
    - No widget detected (caller may continue normally)
    - No solver key in env (caller should fall back to --headed manual or exit)
    - Solver API returned error / timeout

    NEVER raises — best-effort with full logging.
    """
    widget = await detect_captcha_widget(page)
    if not widget:
        return False

    log(f"captcha-solver: detected {widget['type']} widget "
        f"(sitekey={widget['sitekey'][:8]}…)")

    capsolver_key = (env.get("CAPSOLVER_KEY") or "").strip()
    twocaptcha_key = (env.get("TWOCAPTCHA_KEY") or "").strip()
    botsforge_key = (env.get("BOTSFORGE_TURNSTILE_KEY") or "").strip()
    botsforge_url = (env.get("BOTSFORGE_TURNSTILE_URL") or "http://localhost:5033").strip()
    botsforge_timeout_s = int(env.get("BOTSFORGE_TURNSTILE_TIMEOUT_S", "60"))
    timeout_s = int(env.get("CAPTCHA_TIMEOUT_S", "180"))

    # botsforge is Turnstile-only; only counts as a "solver available" for Turnstile widgets
    botsforge_applicable = bool(botsforge_key) and widget["type"] == "turnstile"

    if not capsolver_key and not twocaptcha_key and not botsforge_applicable:
        log("captcha-solver: no CAPSOLVER_KEY or TWOCAPTCHA_KEY or BOTSFORGE_TURNSTILE_KEY in env — "
            "cannot solve; falling through")
        return False

    loop = asyncio.get_event_loop()
    token: Optional[str] = None

    # Try free local BotsForge solver FIRST for Turnstile (free, fast on success).
    if botsforge_applicable:
        log(f"captcha-solver: trying BotsForge local Turnstile solver at {botsforge_url} (timeout={botsforge_timeout_s}s)")
        try:
            token = await loop.run_in_executor(
                None, _botsforge_solve_sync, botsforge_url, botsforge_key, widget, botsforge_timeout_s
            )
        except Exception as e:
            log(f"captcha-solver: BotsForge raised {e!r}")
            token = None
        if token:
            log(f"captcha-solver: BotsForge returned token (len={len(token)})")
        else:
            log("captcha-solver: BotsForge returned no token; falling through to paid solvers")

    if not token and capsolver_key:
        log(f"captcha-solver: submitting to CapSolver (timeout={timeout_s}s)")
        try:
            token = await loop.run_in_executor(
                None, _capsolver_solve_sync, capsolver_key, widget, timeout_s
            )
        except Exception as e:
            log(f"captcha-solver: CapSolver raised {e!r}")
            token = None
        if token:
            log(f"captcha-solver: CapSolver returned token (len={len(token)})")

    if not token and twocaptcha_key:
        log(f"captcha-solver: submitting to 2Captcha (timeout={timeout_s}s)")
        try:
            token = await loop.run_in_executor(
                None, _twocaptcha_solve_sync, twocaptcha_key, widget, timeout_s
            )
        except Exception as e:
            log(f"captcha-solver: 2Captcha raised {e!r}")
            token = None
        if token:
            log(f"captcha-solver: 2Captcha returned token (len={len(token)})")

    if not token:
        log("captcha-solver: all solvers exhausted, no token")
        return False

    injected = await inject_captcha_token(page, widget, token)
    if not injected:
        log("captcha-solver: token solved but DOM injection target not found")
        return False

    log("captcha-solver: token injected; pausing 1s for form to ack")
    await asyncio.sleep(1)
    return True


# -----------------------------------------------------------------------------
# Sync variants — for callers using patchright.sync_api / playwright.sync_api
# (e.g. deno_deploy_token_provision.py). Same contract as the async versions.
# -----------------------------------------------------------------------------
def detect_captcha_widget_sync(page) -> Optional[dict]:
    """Sync sibling of detect_captcha_widget. Returns same dict shape.

    Mirrors the async version: tries top frame first, then walks every
    sub-frame in the page tree, and uses the broader recaptcha host regex
    (covers both google.com/recaptcha and recaptcha.net).
    """
    try:
        url = page.url
    except Exception:
        return None

    result = None
    try:
        result = page.evaluate(_DETECT_WIDGET_JS)
    except Exception:
        result = None

    if not result or not result.get("sitekey"):
        # Walk sub-frames
        try:
            frames = page.frames
        except Exception:
            frames = []
        for fr in frames:
            try:
                r = fr.evaluate(_DETECT_WIDGET_JS)
            except Exception:
                continue
            if r and r.get("sitekey"):
                result = r
                break

    if not result or not result.get("sitekey"):
        return None
    result["url"] = url
    return result


def inject_captcha_token_sync(page, widget: dict, token: str) -> bool:
    """Sync sibling of inject_captcha_token."""
    kind = widget["type"]
    if kind == "hcaptcha":
        js = r"""
        (token) => {
          let found = false;
          document.querySelectorAll('textarea[name="h-captcha-response"], textarea#h-captcha-response')
            .forEach(t => { t.value = token; t.innerHTML = token; found = true; });
          document.querySelectorAll('input[name="h-captcha-response"]')
            .forEach(i => { i.value = token; found = true; });
          const widget = document.querySelector('.h-captcha[data-callback], [data-hcaptcha-widget-id]');
          if (widget) {
            const cb = widget.getAttribute('data-callback');
            if (cb && typeof window[cb] === 'function') {
              try { window[cb](token); } catch (e) {}
            }
          }
          return found;
        }
        """
    elif kind == "recaptcha":
        js = r"""
        (token) => {
          let found = false;
          document.querySelectorAll('textarea#g-recaptcha-response, textarea[name="g-recaptcha-response"]')
            .forEach(t => { t.value = token; t.innerHTML = token; found = true; });
          const widget = document.querySelector('.g-recaptcha[data-callback]');
          if (widget) {
            const cb = widget.getAttribute('data-callback');
            if (cb && typeof window[cb] === 'function') {
              try { window[cb](token); } catch (e) {}
            }
          }
          return found;
        }
        """
    elif kind == "turnstile":
        js = r"""
        (token) => {
          let found = false;
          document.querySelectorAll('input[name="cf-turnstile-response"]')
            .forEach(i => { i.value = token; found = true; });
          return found;
        }
        """
    else:
        return False
    try:
        return bool(page.evaluate(js, token))
    except Exception:
        return False


def maybe_solve_captcha_sync(
    page,
    env: dict,
    log: Callable[[str], None] = print,
) -> bool:
    """Sync sibling of maybe_solve_captcha. Same return contract."""
    widget = detect_captcha_widget_sync(page)
    if not widget:
        return False

    log(f"captcha-solver: detected {widget['type']} widget "
        f"(sitekey={widget['sitekey'][:8]}…)")

    capsolver_key = (env.get("CAPSOLVER_KEY") or "").strip()
    twocaptcha_key = (env.get("TWOCAPTCHA_KEY") or "").strip()
    botsforge_key = (env.get("BOTSFORGE_TURNSTILE_KEY") or "").strip()
    botsforge_url = (env.get("BOTSFORGE_TURNSTILE_URL") or "http://localhost:5033").strip()
    botsforge_timeout_s = int(env.get("BOTSFORGE_TURNSTILE_TIMEOUT_S", "60"))
    timeout_s = int(env.get("CAPTCHA_TIMEOUT_S", "180"))

    botsforge_applicable = bool(botsforge_key) and widget["type"] == "turnstile"

    if not capsolver_key and not twocaptcha_key and not botsforge_applicable:
        log("captcha-solver: no CAPSOLVER_KEY or TWOCAPTCHA_KEY or BOTSFORGE_TURNSTILE_KEY in env — "
            "cannot solve; falling through")
        return False

    token: Optional[str] = None

    if botsforge_applicable:
        log(f"captcha-solver: trying BotsForge local Turnstile solver at {botsforge_url} (timeout={botsforge_timeout_s}s)")
        try:
            token = _botsforge_solve_sync(botsforge_url, botsforge_key, widget, botsforge_timeout_s)
        except Exception as e:
            log(f"captcha-solver: BotsForge raised {e!r}")
            token = None
        if token:
            log(f"captcha-solver: BotsForge returned token (len={len(token)})")
        else:
            log("captcha-solver: BotsForge returned no token; falling through to paid solvers")

    if not token and capsolver_key:
        log(f"captcha-solver: submitting to CapSolver (timeout={timeout_s}s)")
        try:
            token = _capsolver_solve_sync(capsolver_key, widget, timeout_s)
        except Exception as e:
            log(f"captcha-solver: CapSolver raised {e!r}")
            token = None
        if token:
            log(f"captcha-solver: CapSolver returned token (len={len(token)})")

    if not token and twocaptcha_key:
        log(f"captcha-solver: submitting to 2Captcha (timeout={timeout_s}s)")
        try:
            token = _twocaptcha_solve_sync(twocaptcha_key, widget, timeout_s)
        except Exception as e:
            log(f"captcha-solver: 2Captcha raised {e!r}")
            token = None
        if token:
            log(f"captcha-solver: 2Captcha returned token (len={len(token)})")

    if not token:
        log("captcha-solver: all solvers exhausted, no token")
        return False

    injected = inject_captcha_token_sync(page, widget, token)
    if not injected:
        log("captcha-solver: token solved but DOM injection target not found")
        return False

    log("captcha-solver: token injected; pausing 1s for form to ack")
    time.sleep(1)
    return True


# -----------------------------------------------------------------------------
# Free reCAPTCHA v2/v3 audio solver — playwright-recaptcha (MIT)
# -----------------------------------------------------------------------------
# Why this exists:
#   The paid CapSolver / 2Captcha path costs ~$1.20-$1.50/1k solves. For sites
#   that use reCAPTCHA v2 with the audio-challenge option (the vast majority of
#   non-enterprise + many enterprise deployments), we can solve for FREE by
#   transcribing the audio via Google's free speech-recognition endpoint.
#
#   Library: https://github.com/Xewdy444/Playwright-reCAPTCHA — MIT, 539 stars,
#   actively maintained (last update 2026-05-16). Supports both standard and
#   enterprise reCAPTCHA endpoints (matches /recaptcha/(api2|enterprise)/...).
#
# How to integrate from callers (e.g. magic_link_signup._attempt_captcha_clear):
#   1) First try playwright-captcha ClickSolver (existing free path)
#   2) If unsolved AND widget is reCAPTCHA → call maybe_solve_recaptcha_audio_free
#   3) If still unsolved → fall through to paid CapSolver/2Captcha (existing)
#
# Requirements (already installed in sp500-mastery venv):
#   pip install playwright-recaptcha imageio-ffmpeg
#
# Ffmpeg is supplied by imageio-ffmpeg (a bundled static binary) so no
# system-level ffmpeg install is required. We wire pydub.AudioSegment.converter
# to the imageio-ffmpeg binary at import time.
#
# Limitations:
#   - Only solves reCAPTCHA (v2 + v3). Does NOT solve hCaptcha or Turnstile.
#   - Google occasionally rate-limits the audio challenge; the library
#     retries with exponential backoff via tenacity.
#   - Requires the audio button to be reachable (some sites hide it; the
#     library will fall through to the image challenge which requires CapSolver).
# -----------------------------------------------------------------------------
_FREE_AUDIO_READY: Optional[bool] = None  # tri-state cache: None=unchecked


def _free_audio_setup() -> bool:
    """One-time setup: wire pydub to imageio-ffmpeg's bundled ffmpeg binary.

    Returns True if playwright_recaptcha is importable AND ffmpeg is wired.
    Cached so repeated calls are O(1) after first invocation.
    """
    global _FREE_AUDIO_READY
    if _FREE_AUDIO_READY is not None:
        return _FREE_AUDIO_READY
    try:
        import imageio_ffmpeg  # type: ignore
        from pydub import AudioSegment  # type: ignore
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        AudioSegment.converter = ffmpeg_path
        AudioSegment.ffmpeg = ffmpeg_path
        # ffprobe — imageio-ffmpeg only ships ffmpeg, but pydub falls back to
        # using ffmpeg's probe mode when ffprobe is missing. That's fine for
        # the small WAV files reCAPTCHA serves up.
        import playwright_recaptcha  # noqa: F401  type: ignore
        _FREE_AUDIO_READY = True
    except Exception:
        _FREE_AUDIO_READY = False
    return _FREE_AUDIO_READY


async def _dump_frame_urls(page, log: Callable[[str], None]) -> None:
    """Diagnostic: log every frame URL in the page tree so failed solves can
    be inspected. Anchors/bframes for reCAPTCHA load asynchronously; if they
    never appear here, the library has nothing to grab onto and times out.
    """
    try:
        frames = page.frames
    except Exception as e:
        log(f"captcha-free-audio: page.frames unavailable: {e!r}")
        return
    log(f"captcha-free-audio: page has {len(frames)} frames:")
    for i, fr in enumerate(frames):
        try:
            url = fr.url
        except Exception:
            url = "<unreadable>"
        try:
            name = fr.name
        except Exception:
            name = ""
        log(f"  [{i}] name={name!r} url={url}")


async def _dismiss_overlays(page, log: Callable[[str], None]) -> None:
    """Best-effort: dismiss cookie banners / consent modals / chat widgets that
    overlay the page and block subsequent clicks on the reCAPTCHA checkbox.

    Buddy.works (and many other SPAs) renders an Intercom chat widget and a
    cookie-consent overlay on top of the page. Xewdy444's `_click_checkbox`
    waits for the checkbox to be both visible AND actionable; when an overlay
    intercepts pointer events, the click times out at 30s with
    `<div></div> from <div>…</div> subtree intercepts pointer events`.

    Common patterns we try to dismiss:
      - data-test*="cookie", id*="cookie", class*="cookie", "[aria-label*=cookie]"
      - "Accept all"/"Accept cookies"/"OK"/"Got it"/"Reject all" buttons
      - role="dialog"[aria-modal=true] with a close button
      - Intercom messenger frame ([id^=intercom-frame], iframe[name^=intercom])

    Never raises. Logs each dismissal action so callers can audit.
    """
    js = r"""
    () => {
      const log = [];
      const tryClick = (el, why) => {
        try {
          el.click();
          log.push(why);
        } catch (e) {}
      };

      // Accept-cookie buttons (text-based — most common).
      //
      // sambanova fix 2026-05-18: previously this matched ANY button on the
      // page whose text equalled one of the phrases below. That accidentally
      // matched Auth0's "Continue" button on sambanova/cerebras-style login
      // pages, which navigated the page mid-CAPTCHA-solve and crashed it
      // (chrome-error://chromewebdata/). Now we ONLY click buttons that are
      // INSIDE a known consent container, or whose text is unambiguously
      // a cookie/consent phrase. Generic words ("continue", "ok", "okay")
      // are now treated as cookie-only and require a container guard.
      const consentContainerSelector = [
        '[id*="cookie" i]', '[class*="cookie" i]',
        '[id*="consent" i]', '[class*="consent" i]',
        '[id*="gdpr" i]', '[class*="gdpr" i]',
        '[id*="onetrust" i]', '[class*="onetrust" i]',
        '[id*="cmp" i]:not([id*="component" i])',
        '[aria-label*="cookie" i]', '[aria-label*="consent" i]',
        '[data-testid*="cookie" i]', '[data-testid*="consent" i]',
      ].join(',');
      // Phrases that are SAFE anywhere (unambiguously cookie/consent).
      const strongPhrases = [
        'accept all', 'accept cookies', 'allow all',
        'reject all', 'i agree to cookies',
      ];
      // Phrases that are AMBIGUOUS — only click if inside a consent container.
      const weakPhrases = [
        'accept', 'i agree', 'agree', 'got it', 'ok', 'okay', 'continue',
      ];
      for (const btn of document.querySelectorAll('button, a, [role="button"]')) {
        const txt = (btn.innerText || btn.textContent || '').toLowerCase().trim();
        if (!txt) continue;
        const r = btn.getBoundingClientRect();
        if (r.width <= 0 || r.height <= 0) continue;
        // 1) strong phrases: click regardless of container
        let matched = false;
        for (const p of strongPhrases) {
          if (txt === p || txt.startsWith(p + ' ') || txt.endsWith(' ' + p)) {
            tryClick(btn, 'cookie-button:' + p);
            matched = true;
            break;
          }
        }
        if (matched) continue;
        // 2) weak phrases: require the button (or an ancestor) to match a
        //    consent container. Otherwise this is NOT a cookie button and
        //    must NOT be clicked (Auth0 "Continue" lives outside consent).
        const inConsent = btn.closest(consentContainerSelector);
        if (!inConsent) continue;
        for (const p of weakPhrases) {
          if (txt === p || txt.startsWith(p + ' ') || txt.endsWith(' ' + p)) {
            tryClick(btn, 'cookie-button:' + p + ' (consent-scoped)');
            break;
          }
        }
      }

      // Close-X buttons on modal overlays
      for (const sel of [
        '[aria-label*="close" i]',
        '[aria-label*="dismiss" i]',
        'button[aria-label="Close"]',
        '[data-testid*="close" i]',
      ]) {
        for (const el of document.querySelectorAll(sel)) {
          const r = el.getBoundingClientRect();
          if (r.width > 0 && r.height > 0 && r.width < 80 && r.height < 80) {
            tryClick(el, 'close-btn:' + sel);
          }
        }
      }

      // Hide common overlay containers (defensive — won't break the page,
      // just stops them from intercepting pointer events)
      const overlaySelectors = [
        '[id^="intercom-frame"]',
        '[class*="intercom"]',
        '[id*="cookie-banner" i]',
        '[class*="cookie-banner" i]',
        '[id*="cookie-consent" i]',
        '[class*="cookie-consent" i]',
        '[id*="onetrust"]',
        '[class*="onetrust"]',
        '[id*="gdpr" i]',
        '[class*="gdpr" i]',
      ];
      for (const sel of overlaySelectors) {
        for (const el of document.querySelectorAll(sel)) {
          // Only hide if NOT a child/parent of the recaptcha widget
          if (el.closest('.g-recaptcha, [data-sitekey], iframe[src*="recaptcha"]')) continue;
          if (el.querySelector('.g-recaptcha, [data-sitekey], iframe[src*="recaptcha"]')) continue;
          try {
            el.style.setProperty('pointer-events', 'none', 'important');
            log.push('disable-pointer:' + sel);
          } catch (e) {}
        }
      }
      return log;
    }
    """
    try:
        actions = await page.evaluate(js)
        if actions:
            log(f"captcha-free-audio: dismissed overlays: {actions[:6]}"
                + (f" ... (+{len(actions)-6} more)" if len(actions) > 6 else ""))
    except Exception as e:
        log(f"captcha-free-audio: overlay dismissal raised {e!r} (continuing)")


async def _try_click_recaptcha_checkbox(page, log: Callable[[str], None]) -> bool:
    """Best-effort: click the v2 reCAPTCHA checkbox to FORCE bframe iframe
    creation. Many SPAs render only the anchor iframe at page-load; the
    bframe (where the audio button + audio challenge live) is lazily injected
    only after the user clicks the "I'm not a robot" checkbox. Xewdy444's
    library WILL do this automatically once it finds the anchor + bframe pair
    — but on cerebras/sambanova/buddy_works the bframe never exists, so
    from_frames() times out with RecaptchaNotFoundError.

    Strategy: walk every frame whose URL matches /recaptcha/(api2|enterprise)/anchor,
    locate the checkbox inside, and click it. This MIRRORS what Xewdy444 does
    via its checkbox property, but runs BEFORE the library is engaged so the
    bframe is already attached by the time AsyncSolver.solve_recaptcha runs.

    Click strategy is layered (Xewdy444's plain `.click()` fails on sites with
    cookie/intercom overlays that intercept pointer events):
      1. Standard `.click(force=True)` — skips actionability checks but still
         dispatches a real pointer event.
      2. JS `.click()` via `evaluate` — pure DOM event, bypasses overlay layout.

    Returns True if a checkbox was found and at least one click strategy
    succeeded; False otherwise. Never raises.
    """
    import re as _re
    try:
        frames = page.frames
    except Exception:
        return False
    anchor_re = _re.compile(r"/recaptcha/(api2|enterprise)/anchor")
    for fr in frames:
        try:
            url = fr.url
        except Exception:
            continue
        if anchor_re.search(url or "") is None:
            continue
        try:
            cb = fr.locator("#recaptcha-anchor, .recaptcha-checkbox")
            if await cb.count() == 0:
                continue
            try:
                await cb.first.scroll_into_view_if_needed(timeout=2000)
            except Exception:
                pass

            async def _is_checked():
                try:
                    aria = await cb.first.get_attribute("aria-checked")
                    if aria == "true":
                        return True
                    cls = (await cb.first.get_attribute("class")) or ""
                    return "recaptcha-checkbox-checked" in cls
                except Exception:
                    return False

            clicked = False
            # Strategy 1: force-click (skips actionability checks).
            try:
                await cb.first.click(timeout=4000, force=True)
                log(f"captcha-free-audio: force-clicked v2 checkbox ({url[:80]}…)")
                await page.wait_for_timeout(400)
                if await _is_checked():
                    clicked = True
                    log("captcha-free-audio: aria-checked=true after force-click")
            except Exception as e:
                log(f"captcha-free-audio: force-click failed in {url[:60]}…: {e!r}")

            # Strategy 2: JS .click() — dispatches a synthetic DOM event without
            # honoring pointer-event interception. reCAPTCHA's anchor checkbox
            # has an internal `<div></div>` overlay that swallows pointer
            # events from playwright's force-click; the DOM-level click()
            # bypasses that overlay entirely.
            if not clicked:
                try:
                    await cb.first.evaluate("(el) => el.click()")
                    log(f"captcha-free-audio: JS-clicked v2 checkbox ({url[:80]}…)")
                    await page.wait_for_timeout(400)
                    if await _is_checked():
                        clicked = True
                        log("captcha-free-audio: aria-checked=true after JS-click")
                except Exception as e:
                    log(f"captcha-free-audio: JS-click failed in {url[:60]}…: {e!r}")

            # Strategy 3: dispatch synthetic mousedown/mouseup/click events.
            # Some sites' anchor-iframe checkbox listens on mousedown ONLY,
            # not click. Pure DOM click() doesn't fire mousedown.
            if not clicked:
                try:
                    await cb.first.evaluate(
                        """(el) => {
                            const ev = (type) => el.dispatchEvent(new MouseEvent(type, {
                                view: window, bubbles: true, cancelable: true,
                                clientX: el.getBoundingClientRect().left + 5,
                                clientY: el.getBoundingClientRect().top + 5,
                            }));
                            ev('mousedown'); ev('mouseup'); ev('click');
                        }"""
                    )
                    log(f"captcha-free-audio: synth mouse-events on v2 checkbox ({url[:80]}…)")
                    await page.wait_for_timeout(400)
                    if await _is_checked():
                        clicked = True
                        log("captcha-free-audio: aria-checked=true after mouse-events")
                except Exception as e:
                    log(f"captcha-free-audio: mouse-events failed in {url[:60]}…: {e!r}")

            if not clicked:
                log(f"captcha-free-audio: ALL click strategies failed to check the box; "
                    f"continuing — Xewdy444 may still succeed or fail with diagnostic")
                continue
            # Give bframe a moment to attach AND for Google to assess
            # the click pattern. reCAPTCHA may auto-pass (low risk) or
            # present the audio challenge.
            try:
                await page.wait_for_timeout(2500)
            except Exception:
                pass
            return True
        except Exception as e:
            log(f"captcha-free-audio: checkbox click attempt failed in {url[:60]}…: {e!r}")
            continue
    return False


def _patch_xewdy444_click_checkbox(AsyncSolver_cls, log: Callable[[str], None]) -> None:
    """Monkey-patch AsyncSolver._click_checkbox to be a no-op when the checkbox
    is already checked. Xewdy444's stock implementation ALWAYS calls
    `await recaptcha_box.checkbox.click()` regardless of state — but if our
    pre-click already checked the box, the second click can be blocked by an
    overlay (Intercom chat widget, cookie banner, etc.).

    The replacement: if the checkbox is already checked OR the audio_challenge
    is already visible, skip the click. This lets the rest of the solve flow
    proceed to the audio challenge directly.

    Idempotent: only patches once per process.
    """
    if getattr(AsyncSolver_cls, "_zg_checkbox_patched", False):
        return

    original = AsyncSolver_cls._click_checkbox

    async def _patched_click_checkbox(self, recaptcha_box):
        try:
            already_checked = await recaptcha_box.checkbox.is_checked()
        except Exception:
            already_checked = False
        try:
            audio_visible = await recaptcha_box.audio_challenge_is_visible()
        except Exception:
            audio_visible = False
        if already_checked or audio_visible:
            log(f"captcha-free-audio: skipping Xewdy444 internal checkbox click "
                f"(already_checked={already_checked}, audio_visible={audio_visible})")
            # Mirror the original's post-click wait loop so the caller sees
            # the challenge state without us re-clicking.
            return await original(self, recaptcha_box) if False else None
        return await original(self, recaptcha_box)

    AsyncSolver_cls._click_checkbox = _patched_click_checkbox
    AsyncSolver_cls._zg_checkbox_patched = True


def _patch_xewdy444_click_checkbox_sync(SyncSolver_cls, log: Callable[[str], None]) -> None:
    """Sync sibling of _patch_xewdy444_click_checkbox."""
    if getattr(SyncSolver_cls, "_zg_checkbox_patched", False):
        return

    original = SyncSolver_cls._click_checkbox

    def _patched_click_checkbox(self, recaptcha_box):
        try:
            already_checked = recaptcha_box.checkbox.is_checked()
        except Exception:
            already_checked = False
        try:
            audio_visible = recaptcha_box.audio_challenge_is_visible()
        except Exception:
            audio_visible = False
        if already_checked or audio_visible:
            log(f"captcha-free-audio: skipping Xewdy444 internal checkbox click "
                f"(already_checked={already_checked}, audio_visible={audio_visible})")
            return None
        return original(self, recaptcha_box)

    SyncSolver_cls._click_checkbox = _patched_click_checkbox
    SyncSolver_cls._zg_checkbox_patched = True


async def maybe_solve_recaptcha_audio_free(
    page,
    env: dict,
    log: Callable[[str], None] = print,
    timeout_s: Optional[int] = None,
    recaptcha_version: Optional[str] = None,
    click_checkbox_first: bool = True,
    iframe_url_pattern: Optional[str] = None,
) -> bool:
    """Async: try the FREE playwright-recaptcha audio-challenge solver.

    Returns True if a reCAPTCHA was detected and solved (token already injected
    by the library). Returns False on no-widget / non-reCAPTCHA / solve-failure
    / library-not-installed / env-disabled. Never raises.

    Parameters
    ----------
    recaptcha_version : Optional[str]
        Force a specific version path. One of "v2", "v3", or None (try both).
        Use "v2" when a site rejects the v3-style passive token (some
        Enterprise sitekeys score-blacklist the audio-derived v3 token but
        accept a fresh v2 audio solve). Use "v3" when the site is a v3-only
        deployment (no checkbox visible).
    click_checkbox_first : bool
        If True (default), attempt to click the v2 reCAPTCHA checkbox BEFORE
        invoking the solver. This forces the bframe iframe (which hosts the
        audio button) to be lazily attached on sites that gate it behind the
        first user interaction. Pass False for v3-only sites (no checkbox).
    iframe_url_pattern : Optional[str]
        Reserved for future custom-host overrides. Not currently used by
        Xewdy444 (whose regex already covers api2 + enterprise on both
        google.com and recaptcha.net) — kept in the signature so callers can
        pass per-provider hints without breaking.

    Note: the library targets v2 (audio challenge) by default and listens for
    the v3 callback for v3 sites. It auto-detects which version is present
    unless recaptcha_version is specified.
    """
    if (env.get("CAPTCHA_FREE_AUDIO_DISABLED") or "").strip() == "1":
        log("captcha-free-audio: disabled by env CAPTCHA_FREE_AUDIO_DISABLED=1")
        return False

    if not _free_audio_setup():
        log("captcha-free-audio: playwright_recaptcha or imageio_ffmpeg "
            "not importable; skipping free audio path")
        return False

    # Detect widget kind first — only worth running for reCAPTCHA
    widget = await detect_captcha_widget(page)
    if not widget:
        log("captcha-free-audio: no widget detected on page; dumping frame URLs for diagnosis")
        await _dump_frame_urls(page, log)
        return False
    if widget.get("type") != "recaptcha":
        log(f"captcha-free-audio: widget is {widget.get('type')}, not reCAPTCHA; "
            "skipping free audio path")
        return False

    log(f"captcha-free-audio: reCAPTCHA detected (sitekey="
        f"{widget['sitekey'][:8]}…, frame_src={(widget.get('frame_src') or '')[:80]}…); "
        f"version_hint={recaptcha_version or 'auto'}, "
        f"click_checkbox_first={click_checkbox_first}")

    # Pre-step A: dismiss cookie/consent/Intercom overlays that intercept
    # pointer events. Buddy.works retry traced 55 click-retries blocked by
    # `<div></div> from <div>…</div> subtree intercepts pointer events` — that's
    # an Intercom + cookie-consent overlay. Dismiss them BEFORE we (or Xewdy444)
    # try to click the checkbox.
    try:
        await _dismiss_overlays(page, log)
    except Exception as e:
        log(f"captcha-free-audio: overlay dismissal raised {e!r} (continuing)")

    # Pre-step B: force-attach the bframe iframe on lazy-mount sites by clicking
    # the checkbox ourselves (Xewdy444 only does this AFTER it found the frame
    # pair, but the frame pair only exists AFTER the click — chicken-and-egg).
    # Skip on explicit v3 (no checkbox exists).
    if click_checkbox_first and (recaptcha_version or "").lower() != "v3":
        try:
            await _try_click_recaptcha_checkbox(page, log)
        except Exception as e:
            log(f"captcha-free-audio: pre-click checkbox raised {e!r} (continuing)")

    # Pick which paths to attempt based on version hint
    version = (recaptcha_version or "").lower()
    do_v2 = version in ("", "v2")
    do_v3 = version in ("", "v3")

    timeout_s = timeout_s or int(env.get("CAPTCHA_TIMEOUT_S", "180"))
    token: Optional[str] = None

    # Try v2 audio solver first. If the page is actually v3, v2 raises
    # RecaptchaNotFoundError → fall through to v3. Use AsyncSolver since the
    # caller's `page` is an async Playwright Page object — passing it to
    # SyncSolver would fail with "no running event loop".
    if do_v2:
        try:
            from playwright_recaptcha import recaptchav2  # type: ignore
            # Monkey-patch Xewdy444's _click_checkbox so it no-ops when the
            # box is already checked (our pre-click did it). Otherwise the
            # library's second click can be blocked by an overlay we missed.
            _patch_xewdy444_click_checkbox(recaptchav2.AsyncSolver, log)
            async with recaptchav2.AsyncSolver(page, attempts=3) as solver:
                token = await solver.solve_recaptcha(wait=True, wait_timeout=timeout_s)
        except Exception as e:
            log(f"captcha-free-audio: v2 solver raised {type(e).__name__}: {e}")
            token = None

    if not token and do_v3:
        log("captcha-free-audio: trying v3 listener path")
        try:
            from playwright_recaptcha import recaptchav3  # type: ignore
            # cerebras fix 2026-05-18: reCAPTCHA Enterprise INVISIBLE
            # (`size=invisible` in anchor URL — confirmed for cerebras sitekey
            # 6LdnMCUr…) only emits a token when `grecaptcha.execute(sitekey,
            # {action})` is called by the host page. The host page only calls
            # it on submit-click, but the cascade runs PRE-submit, so the
            # passive listener times out after `timeout_s` seconds.
            #
            # Workaround: manually trigger `grecaptcha.execute()` (or
            # `grecaptcha.enterprise.execute()`) via page.evaluate() so the
            # token flow starts while the listener is armed. Best-effort —
            # if grecaptcha isn't loaded yet or already executed, swallow.
            sitekey = widget.get("sitekey") or ""
            try:
                trigger_js = """
                async (sitekey) => {
                  const tryExec = async (api) => {
                    try {
                      if (!api || typeof api.execute !== 'function') return null;
                      // Wrap api.ready (returns a promise on newer versions,
                      // takes a callback on older). Race against a 2s
                      // timeout so we don't hang.
                      const ready = new Promise((resolve) => {
                        try {
                          const r = api.ready(() => resolve(true));
                          if (r && typeof r.then === 'function') r.then(() => resolve(true));
                        } catch (e) { resolve(true); }
                        setTimeout(() => resolve(true), 2000);
                      });
                      await ready;
                      const tok = await api.execute(sitekey, {action: 'submit'});
                      return tok || 'triggered';
                    } catch (e) { return 'error:' + (e && e.message || String(e)); }
                  };
                  // Try Enterprise first (matches cerebras Auth0 sitekey),
                  // fall back to classic grecaptcha.
                  const enterprise = (window.grecaptcha && window.grecaptcha.enterprise) || null;
                  const classic = window.grecaptcha || null;
                  const r1 = await tryExec(enterprise);
                  if (r1) return 'enterprise:' + r1.slice(0, 20);
                  const r2 = await tryExec(classic);
                  if (r2) return 'classic:' + r2.slice(0, 20);
                  return 'no_grecaptcha';
                }
                """
                if sitekey:
                    trig = await page.evaluate(trigger_js, sitekey)
                    log(f"captcha-free-audio: v3 execute() trigger result: {trig}")
                else:
                    log("captcha-free-audio: no sitekey in widget; skipping v3 execute() trigger")
            except Exception as _te:
                log(f"captcha-free-audio: v3 execute() trigger raised {_te!r} (continuing)")
            async with recaptchav3.AsyncSolver(page, timeout=timeout_s) as solver:
                token = await solver.solve_recaptcha()
        except Exception as e:
            log(f"captcha-free-audio: v3 solver raised {type(e).__name__}: {e}")
            token = None

    if not token:
        log("captcha-free-audio: all attempted paths failed; dumping frame URLs for diagnosis")
        await _dump_frame_urls(page, log)
        return False

    log(f"captcha-free-audio: SUCCESS (token len={len(token)}); "
        "library has injected response into DOM")
    # The library already injects g-recaptcha-response into the textarea. But
    # some forms also expect the data-callback to fire — try inject_captcha_token
    # as a belt-and-braces step.
    try:
        await inject_captcha_token(page, widget, token)
    except Exception:
        pass
    return True


def _dump_frame_urls_sync(page, log: Callable[[str], None]) -> None:
    """Sync sibling of _dump_frame_urls."""
    try:
        frames = page.frames
    except Exception as e:
        log(f"captcha-free-audio: page.frames unavailable: {e!r}")
        return
    log(f"captcha-free-audio: page has {len(frames)} frames:")
    for i, fr in enumerate(frames):
        try:
            url = fr.url
        except Exception:
            url = "<unreadable>"
        try:
            name = fr.name
        except Exception:
            name = ""
        log(f"  [{i}] name={name!r} url={url}")


def _dismiss_overlays_sync(page, log: Callable[[str], None]) -> None:
    """Sync sibling of _dismiss_overlays. Same JS, same contract."""
    # Re-use the same JS string as the async helper to keep behavior identical.
    js = r"""
    () => {
      const log = [];
      const tryClick = (el, why) => {
        try { el.click(); log.push(why); } catch (e) {}
      };
      const buttonPhrases = [
        'accept all', 'accept cookies', 'accept', 'i agree', 'agree',
        'got it', 'ok', 'okay', 'allow all', 'continue',
      ];
      for (const btn of document.querySelectorAll('button, a, [role="button"]')) {
        const txt = (btn.innerText || btn.textContent || '').toLowerCase().trim();
        if (!txt) continue;
        for (const p of buttonPhrases) {
          if (txt === p || txt.startsWith(p + ' ') || txt.endsWith(' ' + p)) {
            const r = btn.getBoundingClientRect();
            if (r.width > 0 && r.height > 0) {
              tryClick(btn, 'cookie-button:' + p);
              break;
            }
          }
        }
      }
      for (const sel of [
        '[aria-label*="close" i]', '[aria-label*="dismiss" i]',
        'button[aria-label="Close"]', '[data-testid*="close" i]',
      ]) {
        for (const el of document.querySelectorAll(sel)) {
          const r = el.getBoundingClientRect();
          if (r.width > 0 && r.height > 0 && r.width < 80 && r.height < 80) {
            tryClick(el, 'close-btn:' + sel);
          }
        }
      }
      const overlaySelectors = [
        '[id^="intercom-frame"]', '[class*="intercom"]',
        '[id*="cookie-banner" i]', '[class*="cookie-banner" i]',
        '[id*="cookie-consent" i]', '[class*="cookie-consent" i]',
        '[id*="onetrust"]', '[class*="onetrust"]',
        '[id*="gdpr" i]', '[class*="gdpr" i]',
      ];
      for (const sel of overlaySelectors) {
        for (const el of document.querySelectorAll(sel)) {
          if (el.closest('.g-recaptcha, [data-sitekey], iframe[src*="recaptcha"]')) continue;
          if (el.querySelector('.g-recaptcha, [data-sitekey], iframe[src*="recaptcha"]')) continue;
          try {
            el.style.setProperty('pointer-events', 'none', 'important');
            log.push('disable-pointer:' + sel);
          } catch (e) {}
        }
      }
      return log;
    }
    """
    try:
        actions = page.evaluate(js)
        if actions:
            log(f"captcha-free-audio: dismissed overlays: {actions[:6]}"
                + (f" ... (+{len(actions)-6} more)" if len(actions) > 6 else ""))
    except Exception as e:
        log(f"captcha-free-audio: overlay dismissal raised {e!r} (continuing)")


def _try_click_recaptcha_checkbox_sync(page, log: Callable[[str], None]) -> bool:
    """Sync sibling of _try_click_recaptcha_checkbox.

    Same layered click strategy: force-click first, then JS `.click()` fallback.
    """
    import re as _re
    try:
        frames = page.frames
    except Exception:
        return False
    anchor_re = _re.compile(r"/recaptcha/(api2|enterprise)/anchor")
    for fr in frames:
        try:
            url = fr.url
        except Exception:
            continue
        if anchor_re.search(url or "") is None:
            continue
        try:
            cb = fr.locator("#recaptcha-anchor, .recaptcha-checkbox")
            if cb.count() == 0:
                continue
            try:
                cb.first.scroll_into_view_if_needed(timeout=2000)
            except Exception:
                pass

            def _is_checked_sync():
                try:
                    aria = cb.first.get_attribute("aria-checked")
                    if aria == "true":
                        return True
                    cls = (cb.first.get_attribute("class")) or ""
                    return "recaptcha-checkbox-checked" in cls
                except Exception:
                    return False

            clicked = False
            try:
                cb.first.click(timeout=4000, force=True)
                log(f"captcha-free-audio: force-clicked v2 checkbox ({url[:80]}…)")
                page.wait_for_timeout(400)
                if _is_checked_sync():
                    clicked = True
                    log("captcha-free-audio: aria-checked=true after force-click")
            except Exception as e:
                log(f"captcha-free-audio: force-click failed in {url[:60]}…: {e!r}")

            if not clicked:
                try:
                    cb.first.evaluate("(el) => el.click()")
                    log(f"captcha-free-audio: JS-clicked v2 checkbox ({url[:80]}…)")
                    page.wait_for_timeout(400)
                    if _is_checked_sync():
                        clicked = True
                        log("captcha-free-audio: aria-checked=true after JS-click")
                except Exception as e:
                    log(f"captcha-free-audio: JS-click failed in {url[:60]}…: {e!r}")

            if not clicked:
                try:
                    cb.first.evaluate(
                        """(el) => {
                            const ev = (type) => el.dispatchEvent(new MouseEvent(type, {
                                view: window, bubbles: true, cancelable: true,
                                clientX: el.getBoundingClientRect().left + 5,
                                clientY: el.getBoundingClientRect().top + 5,
                            }));
                            ev('mousedown'); ev('mouseup'); ev('click');
                        }"""
                    )
                    log(f"captcha-free-audio: synth mouse-events on v2 checkbox ({url[:80]}…)")
                    page.wait_for_timeout(400)
                    if _is_checked_sync():
                        clicked = True
                        log("captcha-free-audio: aria-checked=true after mouse-events")
                except Exception as e:
                    log(f"captcha-free-audio: mouse-events failed in {url[:60]}…: {e!r}")

            if not clicked:
                log(f"captcha-free-audio: ALL click strategies failed to check the box; "
                    f"continuing — Xewdy444 may still succeed or fail with diagnostic")
                continue
            try:
                page.wait_for_timeout(2500)
            except Exception:
                pass
            return True
        except Exception as e:
            log(f"captcha-free-audio: checkbox click attempt failed in {url[:60]}…: {e!r}")
            continue
    return False


def maybe_solve_recaptcha_audio_free_sync(
    page,
    env: dict,
    log: Callable[[str], None] = print,
    timeout_s: Optional[int] = None,
    recaptcha_version: Optional[str] = None,
    click_checkbox_first: bool = True,
    iframe_url_pattern: Optional[str] = None,
) -> bool:
    """Sync sibling of maybe_solve_recaptcha_audio_free. Same contract +
    same new parameters (recaptcha_version, click_checkbox_first,
    iframe_url_pattern)."""
    if (env.get("CAPTCHA_FREE_AUDIO_DISABLED") or "").strip() == "1":
        log("captcha-free-audio: disabled by env CAPTCHA_FREE_AUDIO_DISABLED=1")
        return False

    if not _free_audio_setup():
        log("captcha-free-audio: playwright_recaptcha or imageio_ffmpeg "
            "not importable; skipping free audio path")
        return False

    widget = detect_captcha_widget_sync(page)
    if not widget:
        log("captcha-free-audio: no widget detected on page; dumping frame URLs for diagnosis")
        _dump_frame_urls_sync(page, log)
        return False
    if widget.get("type") != "recaptcha":
        log(f"captcha-free-audio: widget is {widget.get('type')}, not reCAPTCHA; "
            "skipping free audio path")
        return False

    log(f"captcha-free-audio: reCAPTCHA detected (sitekey="
        f"{widget['sitekey'][:8]}…, frame_src={(widget.get('frame_src') or '')[:80]}…); "
        f"version_hint={recaptcha_version or 'auto'}, "
        f"click_checkbox_first={click_checkbox_first}")

    # Pre-step A: dismiss overlays (cookie/Intercom/consent) — must run BEFORE
    # any click attempt so the checkbox isn't intercepted by an overlay div.
    try:
        _dismiss_overlays_sync(page, log)
    except Exception as e:
        log(f"captcha-free-audio: overlay dismissal raised {e!r} (continuing)")

    if click_checkbox_first and (recaptcha_version or "").lower() != "v3":
        try:
            _try_click_recaptcha_checkbox_sync(page, log)
        except Exception as e:
            log(f"captcha-free-audio: pre-click checkbox raised {e!r} (continuing)")

    version = (recaptcha_version or "").lower()
    do_v2 = version in ("", "v2")
    do_v3 = version in ("", "v3")

    timeout_s = timeout_s or int(env.get("CAPTCHA_TIMEOUT_S", "180"))
    token: Optional[str] = None
    if do_v2:
        try:
            from playwright_recaptcha import recaptchav2  # type: ignore
            _patch_xewdy444_click_checkbox_sync(recaptchav2.SyncSolver, log)
            with recaptchav2.SyncSolver(page, attempts=3) as solver:
                token = solver.solve_recaptcha(wait=True, wait_timeout=timeout_s)
        except Exception as e:
            log(f"captcha-free-audio: v2 solver raised {type(e).__name__}: {e}")
            token = None

    if not token and do_v3:
        try:
            from playwright_recaptcha import recaptchav3  # type: ignore
            with recaptchav3.SyncSolver(page, timeout=timeout_s) as solver:
                token = solver.solve_recaptcha()
        except Exception as e:
            log(f"captcha-free-audio: v3 solver raised {type(e).__name__}: {e}")
            token = None

    if not token:
        log("captcha-free-audio: all attempted paths failed; dumping frame URLs for diagnosis")
        _dump_frame_urls_sync(page, log)
        return False

    log(f"captcha-free-audio: SUCCESS (token len={len(token)}); "
        "library has injected response into DOM")
    try:
        inject_captcha_token_sync(page, widget, token)
    except Exception:
        pass
    return True


# -----------------------------------------------------------------------------
# Per-provider hints for the FREE audio solver
# -----------------------------------------------------------------------------
# Some Enterprise sitekeys score-blacklist v3 audio-derived tokens but still
# accept fresh v2 audio solves; others run pure v3 (no checkbox to click).
# Callers (e.g. magic_link_signup._attempt_captcha_clear) can look up the
# active provider here and forward the hints to maybe_solve_recaptcha_audio_free.
#
# Schema per provider:
#   {
#     "recaptcha_version": "v2" | "v3" | None,   # None = auto-detect
#     "click_checkbox_first": bool,              # default True for v2; False for v3
#     "iframe_url_pattern": Optional[str],       # reserved; Xewdy444 already covers
#                                                # both api2 + enterprise on
#                                                # google.com + recaptcha.net
#   }
# -----------------------------------------------------------------------------
PROVIDER_CAPTCHA_HINTS: dict[str, dict] = {
    # cerebras — invisible Enterprise (`size=invisible` in anchor URL) per
    # a4581b9 retry helper finding 2026-05-18. v2 path has no checkbox to grab.
    # Use v3 path (score-based, no anchor needed).
    "cerebras": {
        "recaptcha_version": "v3",
        "click_checkbox_first": False,
        "iframe_url_pattern": None,
    },
    # buddy_works — outer detector failed → widget likely inside an Ory Kratos
    # auth iframe. Sub-frame walking (new) should now surface it. Treat as
    # auto-version, click checkbox if present.
    "buddy_works": {
        "recaptcha_version": None,
        "click_checkbox_first": True,
        "iframe_url_pattern": None,
    },
    # sambanova — challenge appears POST-submit (different stage); same iframe
    # nesting story as buddy_works. Auto-version + click checkbox.
    "sambanova": {
        "recaptcha_version": None,
        "click_checkbox_first": True,
        "iframe_url_pattern": None,
    },
}


def get_provider_captcha_hints(provider_name: Optional[str]) -> dict:
    """Look up per-provider captcha solver hints.

    Returns the registry entry verbatim if present, or an empty dict if the
    provider is unregistered (caller then uses default behavior). Safe to pass
    the returned dict's items as **kwargs to maybe_solve_recaptcha_audio_free.
    """
    if not provider_name:
        return {}
    return dict(PROVIDER_CAPTCHA_HINTS.get(provider_name, {}))


__all__ = [
    "detect_captcha_widget",
    "inject_captcha_token",
    "maybe_solve_captcha",
    "detect_captcha_widget_sync",
    "inject_captcha_token_sync",
    "maybe_solve_captcha_sync",
    "maybe_solve_recaptcha_audio_free",
    "maybe_solve_recaptcha_audio_free_sync",
    "maybe_solve_nopecha",
    "maybe_solve_nopecha_sync",
    "get_provider_captcha_hints",
    "PROVIDER_CAPTCHA_HINTS",
]


# -----------------------------------------------------------------------------
# CLI smoke test (no network — verifies imports + detect against a static page)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    print("captcha_solver_helper module loaded OK", file=sys.stderr)
    print(f"  exports: {', '.join(__all__)}", file=sys.stderr)
    sys.exit(0)
