#!/usr/bin/env python3
"""Probe lightning.ai/sign-in resend-magic-link flow.

Sign-up flow yields no new email after the first request (account exists).
The /sign-in flow may have a 'Resend verification' or 'Send magic link' path
that re-sends the email reliably. This probe drives that flow + screenshots
each stage. NO token is captured — purely a recon step.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import magic_link_signup as mls  # noqa: E402
from playwright.async_api import async_playwright  # noqa: E402


PROVIDER = "lightning_ai"


async def main() -> int:
    yenv = mls.load_dotenv(mls.YAHOO_ENV)
    email = yenv.get("YAHOO_USER")
    if not email:
        print("FATAL: YAHOO_USER missing", file=sys.stderr)
        return 2

    profile = mls.PROFILE_ROOT / PROVIDER
    profile.mkdir(parents=True, exist_ok=True)
    for lock in ("SingletonLock", "SingletonCookie", "SingletonSocket"):
        lp = profile / lock
        if lp.exists() or lp.is_symlink():
            try:
                lp.unlink()
            except OSError:
                pass

    result: dict = {"provider": PROVIDER, "mode": "signin_probe", "stages": []}

    async with async_playwright() as pw:
        ctx = await pw.chromium.launch_persistent_context(
            user_data_dir=str(profile),
            headless=True,
            channel="chromium",
            viewport={"width": 1366, "height": 850},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/147.0.0.0 Safari/537.36"
            ),
            locale="en-US",
            timezone_id="America/New_York",
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--dns-prefetch-disable",
            ],
        )
        page = await ctx.new_page()

        # Stage 1: nav /sign-in
        mls.log("probe stage 1: nav https://lightning.ai/sign-in")
        try:
            await page.goto("https://lightning.ai/sign-in",
                            timeout=45000, wait_until="domcontentloaded")
        except Exception as e:
            mls.log(f"FATAL nav /sign-in: {e}")
            await ctx.close()
            return 3
        await asyncio.sleep(3)
        await mls._screenshot(page, f"{PROVIDER}_probe_signin_landing")
        result["stages"].append({"stage": 1, "url": page.url})

        # Stage 2: click Email chooser (same as signup chooser)
        email_btn_selectors = [
            "button:has-text('Email'):not(:has-text('Google')):not(:has-text('GitHub'))",
            "button:has(svg) >> text=Email",
            "[role='button']:has-text('Email')",
            "a:has-text('Continue with Email')",
            "button:has-text('Continue with Email')",
        ]
        if await mls._try_click(page, email_btn_selectors, timeout_ms=10000):
            await asyncio.sleep(2)
            await mls._screenshot(page, f"{PROVIDER}_probe_after_email_click")
        else:
            mls.log("WARN: Email button not clicked on /sign-in")

        # Stage 3: fill email
        email_field_selectors = [
            "input[type='email']",
            "input[name='email']",
            "input[autocomplete='email']",
            "input[placeholder*='mail' i]",
            "input[placeholder*='work or .edu' i]",
        ]
        filled = await mls._try_fill(page, email_field_selectors,
                                     email, timeout_ms=10000)
        await asyncio.sleep(1)
        await mls._screenshot(page, f"{PROVIDER}_probe_email_filled")
        if not filled:
            mls.log("FATAL: email field not found on /sign-in")
            await ctx.close()
            return 4

        # Stage 4: submit (Log in / Continue / Send magic link)
        submit_selectors = [
            "button:has-text('Log in')",
            "button:has-text('Continue')",
            "button:has-text('Sign in')",
            "button:has-text('Send magic link')",
            "button:has-text('Send link')",
            "button[type='submit']",
        ]
        if not await mls._try_click(page, submit_selectors, timeout_ms=10000):
            mls.log("FATAL: submit button not clicked on /sign-in")
            await ctx.close()
            return 5
        await asyncio.sleep(5)
        await mls._screenshot(page, f"{PROVIDER}_probe_post_submit")
        result["stages"].append({"stage": 4, "post_submit_url": page.url})

        # Capture page text for confirmation message
        try:
            body_text = (await page.locator("body").inner_text(timeout=2000)) or ""
        except Exception:
            body_text = ""
        result["post_submit_text_snippet"] = body_text[:500]
        mls.log(f"  post-submit URL: {page.url}")
        mls.log(f"  post-submit text snippet: {body_text[:200]!r}")

        await ctx.close()

    out_json = mls.LOG_DIR / f"magic_link_{PROVIDER}_signin_probe_{mls._ts()}.json"
    out_json.write_text(json.dumps(result, indent=2))
    mls.log(f"result: {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
