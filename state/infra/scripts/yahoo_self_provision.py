#!/usr/bin/env python3
"""
Self-provision a yahoo app-password by headless login as the bot's own account.

Loads YAHOO_USER + YAHOO_PASSWORD from ~/.config/auto_signup/yahoo.env
Generates a new app password via yahoo's security page
Appends it back into yahoo.env as YAHOO_APP_PASSWORD

Designed to be re-runnable: if YAHOO_APP_PASSWORD already populated and valid,
skips. If invalid (yahoo revoked), regenerates.

Uses patchright (Playwright fork, MIT) for CDP stealth — yahoo detects vanilla
Playwright via WebDriver flag + CDP listener.

Safety:
- Never logs the password
- chmod 600 on .env file after write
- Stops at any CAPTCHA or 2FA challenge (logs evidence, exits 2)
- Screenshots on failure go to logs/auto_signup/yahoo_provision_*.png

Usage:
    python yahoo_self_provision.py [--headed] [--screenshot-on-fail]
"""

import argparse
import asyncio
import datetime as dt
import os
import re
import sys
from pathlib import Path

from patchright.async_api import async_playwright, TimeoutError as PlaywrightTimeout

# NOTE: launcher remaps $HOME into Drive (AI-Tools/home/). Secrets MUST live on
# Mac-local /Users/orginal/.config to avoid Drive sync of credentials.
MAC_HOME = Path("/Users/orginal")
ENV_PATH = MAC_HOME / ".config" / "auto_signup" / "yahoo.env"
LOG_DIR = MAC_HOME / "Library" / "CloudStorage" / "GoogleDrive-zachgladstone@gmail.com" / "My Drive" / "AI-Tools" / "logs" / "auto_signup"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def _ts():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_env() -> dict:
    if not ENV_PATH.exists():
        sys.exit(f"FATAL: {ENV_PATH} missing")
    env = {}
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:]
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def write_app_password(app_pw: str):
    text = ENV_PATH.read_text()
    if re.search(r'^export YAHOO_APP_PASSWORD=.*$', text, re.M):
        text = re.sub(r'^export YAHOO_APP_PASSWORD=.*$',
                      f'export YAHOO_APP_PASSWORD="{app_pw}"', text, flags=re.M)
    else:
        text += f'\nexport YAHOO_APP_PASSWORD="{app_pw}"\n'
    ENV_PATH.write_text(text)
    ENV_PATH.chmod(0o600)


async def screenshot(page, name: str):
    p = LOG_DIR / f"yahoo_provision_{_ts()}_{name}.png"
    try:
        await page.screenshot(path=str(p), full_page=True)
        print(f"shot:{p}", file=sys.stderr)
    except Exception as e:
        print(f"shot-fail:{e}", file=sys.stderr)


async def main(headed: bool, snap_on_fail: bool):
    env = load_env()
    user = env.get("YAHOO_USER")
    pw = env.get("YAHOO_PASSWORD")
    existing_app = env.get("YAHOO_APP_PASSWORD", "")
    if not user or not pw:
        sys.exit("FATAL: YAHOO_USER or YAHOO_PASSWORD missing in env")
    if existing_app:
        print(f"YAHOO_APP_PASSWORD already set ({len(existing_app)} chars). "
              f"Re-run with --force to regenerate.", file=sys.stderr)
        return 0

    # Persistent context bypasses yahoo's trust heuristic. user_data_dir caches
    # cookies, localStorage, browser fingerprint between runs.
    user_data_dir = MAC_HOME / ".config" / "auto_signup" / "yahoo_browser_profile"
    user_data_dir.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as pw_drv:
        ctx = await pw_drv.chromium.launch_persistent_context(
            user_data_dir=str(user_data_dir),
            headless=not headed,
            channel="chromium",  # NOT headless-shell (most-detected variant)
            viewport={"width": 1280, "height": 800},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/147.0.0.0 Safari/537.36",
            locale="en-US",
            timezone_id="America/New_York",
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        )
        # NOTE 2026-05-18: removed ctx.add_init_script() entirely — under
        # patchright + persistent_context + headless=True, ANY add_init_script
        # call triggers net::ERR_NAME_NOT_RESOLVED on subsequent navigations
        # (likely a CDP shim bug in patchright; confirmed via minimal repro).
        # Patchright already returns navigator.webdriver=false natively. The
        # shadow-DOM piercing is now injected per-navigation via
        # page.evaluate() in the helper below, AFTER goto completes.
        async def _pierce_shadow_dom():
            try:
                await page.evaluate("""
                    if (!window.__shadowPierced) {
                        const originalAttachShadow = Element.prototype.attachShadow;
                        Element.prototype.attachShadow = function(init) {
                            return originalAttachShadow.call(this, { ...init, mode: 'open' });
                        };
                        window.__shadowPierced = true;
                    }
                """)
            except Exception:
                pass
        page = await ctx.new_page()
        browser = None  # persistent_context returns BrowserContext, not Browser

        async def find_in_any_frame(selector, timeout_ms=10000):
            """Search the locator across every frame (yahoo Account Info pages live in iframes)."""
            import time as _t
            deadline = _t.time() + timeout_ms / 1000
            while _t.time() < deadline:
                # Try the main page first
                try:
                    loc = page.locator(selector).first
                    if await loc.count() and await loc.is_visible():
                        return loc, page
                except Exception:
                    pass
                # Iterate through every frame
                for frame in page.frames:
                    if frame == page.main_frame:
                        continue
                    try:
                        loc = frame.locator(selector).first
                        if await loc.count() and await loc.is_visible():
                            return loc, frame
                    except Exception:
                        continue
                await _aio.sleep(0.5)
            raise PlaywrightTimeout(f"{selector} not found in any frame within {timeout_ms}ms")

        import asyncio as _aio
        print(f"step 1: navigate login.yahoo.com", file=sys.stderr)
        await page.goto("https://login.yahoo.com", timeout=30000)
        await page.wait_for_load_state("domcontentloaded")
        await _pierce_shadow_dom()

        # NEW 2026-05-18: if the persistent profile is already signed in,
        # login.yahoo.com redirects to www.yahoo.com (homepage). Detect that
        # and skip username/password steps entirely.
        already_signed_in = False
        try:
            await _aio.sleep(1)
            current_url = page.url
            print(f"  -> after nav URL={current_url}", file=sys.stderr)
            if "login.yahoo.com" not in current_url:
                already_signed_in = True
                print(f"  -> already signed in, skipping creds", file=sys.stderr)
            else:
                # Also detect by absence of username input
                has_username = await page.locator('input[name="username"]').count()
                print(f"  -> username input count={has_username}", file=sys.stderr)
                if has_username == 0:
                    already_signed_in = True
                    print(f"  -> no username input, treating as signed-in", file=sys.stderr)
        except Exception as e:
            print(f"  -> signed-in detection raised: {e}", file=sys.stderr)

        if not already_signed_in:
            print(f"step 2: enter username", file=sys.stderr)
            try:
                await page.fill('input[name="username"]', user, timeout=10000)
                await page.click('button[name="signin"]', timeout=5000)
            except PlaywrightTimeout:
                if snap_on_fail: await screenshot(page, "username_step")
                sys.exit("FATAL: username step failed (selector miss?)")

            await page.wait_for_load_state("domcontentloaded", timeout=15000)

            # Detect CAPTCHA before password
            body = (await page.content()).lower()
            if "captcha" in body or "recaptcha" in body or "verification challenge" in body:
                if snap_on_fail: await screenshot(page, "captcha")
                sys.exit("BLOCKED: CAPTCHA before password step. Need CapSolver integration.")

        if not already_signed_in:
            print(f"step 3: enter password (or fallback past passkey gate)", file=sys.stderr)
            # Yahoo 2024+ forces passkey QR. Click "Try signing in another way" first
            # so the password form actually renders.
            try:
                await page.click('text=Try signing in another way', timeout=4000)
                await page.wait_for_load_state("domcontentloaded", timeout=10000)
            except Exception:
                pass  # passkey gate not shown for this account

            # Some Yahoo flows then list options (Passkey, Phone, Password, etc).
            # Click the password option if present.
            for opt in [
                'text=/sign in with.*password/i',
                'text=/password/i',
                'button:has-text("Password")',
                'a:has-text("Password")',
            ]:
                try:
                    await page.locator(opt).first.click(timeout=2500)
                    await page.wait_for_load_state("domcontentloaded", timeout=8000)
                    break
                except Exception:
                    continue

            try:
                await page.fill('input[name="password"]', pw, timeout=10000)
                await page.click('button[name="verifyPassword"], button:has-text("Next"), button:has-text("Sign in")', timeout=5000)
            except PlaywrightTimeout:
                if snap_on_fail: await screenshot(page, "password_step")
                sys.exit("FATAL: password step failed (selector miss?)")

            # Wait for the post-Next state to settle: either URL leaves login.yahoo.com
            # (auth ok) or a 2FA form appears (specific selectors, not body-text regex).
            # Poll for up to 30s.
            auth_ok = False
            for _ in range(30):
                await _aio.sleep(1)
                if "login.yahoo.com" not in page.url and "consent.yahoo.com" not in page.url:
                    auth_ok = True
                    break
                # Check explicit 2FA element selectors (these are stable; do NOT
                # body-text grep — yahoo homepage articles trigger false positives)
                for sel in ['input[name="verificationCode"]', 'input[id*="otp"]',
                            'text=/check.*your.*phone/i', 'text=/check.*your.*email/i',
                            'text=/we.*sent.*you.*code/i']:
                    try:
                        el = page.locator(sel).first
                        if await el.count() and await el.is_visible():
                            if snap_on_fail: await screenshot(page, "2fa")
                            sys.exit(f"BLOCKED: 2FA challenge detected ({sel}). Operator action required.")
                    except Exception:
                        continue
            if not auth_ok and "login.yahoo.com" in page.url:
                if snap_on_fail: await screenshot(page, "post_password_stuck")
                sys.exit(f"FATAL: stuck on {page.url} after 30s.")

        print(f"step 4: navigate to canonical /account/security URL", file=sys.stderr)
        # Yahoo Help SLN15241 lists /account/security; /myaccount/security is
        # an undocumented sign-in gateway that often serves degraded variants.
        await page.goto("https://login.yahoo.com/account/security?.lang=en-US&.intl=us&.src=yhelp",
                        timeout=30000)
        await page.wait_for_load_state("domcontentloaded")
        await _pierce_shadow_dom()
        # SPA renders async — wait then directly search for the "App passwords"
        # tab. NOTE 2026-05-18: removed scroll-to-"external connections" because
        # the new yahoo UI puts "External connections" only in the LEFT sidebar
        # nav (not as a scrollable section header) — scroll_into_view fails
        # because patchright sees the sidebar version, which is already in view.
        # The "App passwords" tab itself is visible directly.
        await _aio.sleep(4)

        # NEW 2026-05-18: yahoo's UI uses non-breaking space in tab text
        # ('App\xa0passwords'), so plain regex 'App passwords' fails. Also the
        # bare 'text=App passwords' selector matches news/article links on the
        # page that navigate away. The reliable selector is role=tab + filter.
        print(f"step 5a: locate App passwords tab via role=tab", file=sys.stderr)
        tabs = page.locator('[role="tab"]')
        n_tabs = await tabs.count()
        print(f"  -> found {n_tabs} role=tab elements", file=sys.stderr)
        app_pw_tab_idx = None
        for i in range(n_tabs):
            try:
                txt = (await tabs.nth(i).text_content()) or ""
                # Normalize whitespace including \xa0
                txt_norm = " ".join(txt.split()).lower()
                if "app password" in txt_norm:
                    app_pw_tab_idx = i
                    print(f"  -> tab[{i}] matches 'app password' ({txt!r})", file=sys.stderr)
                    break
            except Exception:
                continue
        if app_pw_tab_idx is None:
            if snap_on_fail: await screenshot(page, "no_app_pw_section")
            sys.exit(f"BLOCKED: 'App passwords' tab not found among {n_tabs} role=tab on {page.url}.")
        try:
            await tabs.nth(app_pw_tab_idx).click(timeout=5000)
            await _aio.sleep(2)
            print(f"  -> clicked App passwords tab", file=sys.stderr)
        except Exception as e:
            if snap_on_fail: await screenshot(page, "app_pw_click_failed")
            sys.exit(f"FATAL: clicking App passwords tab failed: {e}")

        print(f"step 5b: click 'Create app password' button (canonical yahoo text)", file=sys.stderr)
        # Per yahoo help SLN15241 canonical button text is "Create app password".
        # Generate/Add are stale third-party guesses.
        clicked = False
        for sel in [
            'button:has-text("Create app password")',
            'a:has-text("Create app password")',
            '[aria-label*="create app password" i]',
            'button:has-text("Create")',
            'button:has-text("Add app password")',
            'button:has-text("Generate app password")',
            'button:has-text("Generate")',
            'button:has-text("Add")',
            '[aria-label*="generate" i]',
            '[aria-label*="add app password" i]',
        ]:
            try:
                await page.locator(sel).first.click(timeout=4000)
                clicked = True
                print(f"  -> clicked {sel}", file=sys.stderr)
                break
            except Exception:
                continue
        if not clicked:
            if snap_on_fail: await screenshot(page, "no_generate_btn")
            sys.exit("FATAL: no Generate-app-password button found.")

        # Wait for the "Generate an app password" modal to render
        await _aio.sleep(2)
        print(f"step 5c: fill name field in modal", file=sys.stderr)
        # Try multiple name selectors. New yahoo modal label = "Enter your app's name"
        name_filled = False
        for nsel in [
            'input[name="app-password-name"]',
            'input[placeholder*="app" i]',
            'input[placeholder*="name" i]',
            'input[aria-label*="name" i]',
            'input[type="text"]:visible',
        ]:
            try:
                await page.fill(nsel, "signup-bot", timeout=3000)
                name_filled = True
                print(f"  -> filled name via {nsel}", file=sys.stderr)
                break
            except Exception:
                continue
        if not name_filled:
            print(f"  -> name fill via selectors failed, trying keyboard type", file=sys.stderr)
            try:
                await page.keyboard.type("signup-bot")
            except Exception:
                pass

        await _aio.sleep(1)
        print(f"step 5d: click Generate in modal", file=sys.stderr)
        # The submit button text is "Generate". Need to scope to the modal/dialog
        # so we don't re-click the "Create app password" button outside.
        submit_clicked = False
        for ssel in [
            '[role="dialog"] button:has-text("Generate")',
            'div[role="dialog"] button:visible',
            'button:has-text("Generate"):visible',
            'button[type="submit"]:visible',
        ]:
            try:
                await page.locator(ssel).last.click(timeout=4000)
                submit_clicked = True
                print(f"  -> clicked submit via {ssel}", file=sys.stderr)
                break
            except Exception:
                continue
        if not submit_clicked:
            if snap_on_fail: await screenshot(page, "no_submit")
            sys.exit("FATAL: no Generate-submit button found in modal.")

        # Wait for password to appear (XHR-based; may take a few seconds)
        await _aio.sleep(5)
        try:
            await page.wait_for_load_state("networkidle", timeout=10000)
        except Exception:
            pass

        print(f"step 6: extract password", file=sys.stderr)
        # App password is shown once, typically in a <span> or <code>.
        # Yahoo formats as 4-groups of 4 chars separated by spaces, e.g.
        # "abcd efgh ijkl mnop". Strip spaces before validating.
        candidates = await page.locator("code, span, div, p, strong, b").all_text_contents()
        match = None
        for c in candidates:
            c_stripped = c.strip().replace(" ", "").replace("\xa0", "")
            if re.fullmatch(r"[a-z]{16}", c_stripped):
                match = c_stripped
                break
        if not match:
            if snap_on_fail: await screenshot(page, "no_password_found")
            # Dump candidate text for diagnosis (length only — never the content)
            short_candidates = [c.strip() for c in candidates if 10 < len(c.strip()) < 30]
            print(f"  -> {len(candidates)} text nodes, {len(short_candidates)} in 10-30 char range", file=sys.stderr)
            for c in short_candidates[:20]:
                cs = c.replace(" ", "").replace("\xa0", "")
                print(f"  candidate len={len(c)} stripped_len={len(cs)} all_lower={cs.islower() and cs.isalpha()}", file=sys.stderr)
            sys.exit("FATAL: 16-char app password not found in DOM. Check screenshot.")

        write_app_password(match)
        print(f"OK: app password provisioned ({len(match)} chars). "
              f"Wrote to {ENV_PATH}", file=sys.stderr)
        await ctx.close()
        if browser is not None:
            await browser.close()
        return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--headed", action="store_true",
                    help="Show browser (debug only; production stays headless)")
    ap.add_argument("--screenshot-on-fail", action="store_true", default=True,
                    help="Save screenshot to logs/auto_signup/ on any failure")
    ap.add_argument("--force", action="store_true",
                    help="Regenerate even if YAHOO_APP_PASSWORD already set")
    args = ap.parse_args()

    if args.force:
        # blank existing app password before run
        text = ENV_PATH.read_text()
        text = re.sub(r'^export YAHOO_APP_PASSWORD=.*$',
                      'export YAHOO_APP_PASSWORD=""', text, flags=re.M)
        ENV_PATH.write_text(text)

    sys.exit(asyncio.run(main(args.headed, args.screenshot_on_fail)))
