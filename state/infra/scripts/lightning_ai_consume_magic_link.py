#!/usr/bin/env python3
"""Consume a pre-fetched Lightning AI magic-link URL and complete stages 5-9
of the magic_link_signup orchestrator: confirm session, navigate to api-keys
page, create + scrape token, write env file, flip cloud_usage.

Why this exists: Lightning AI rate-limits magic-link issuance for the same
email, AND the orchestrator's `record_since = now - 1 min` cutoff drops any
magic-link that arrived BEFORE the new submit attempt. This standalone
consumer accepts the URL via stdin (kept off the process arglist + shell
history so the token never leaks) and runs the post-IMAP stages directly.

USAGE:
    echo "<magic-link-url>" | python3 scripts/lightning_ai_consume_magic_link.py

Token VALUE is never logged. Length is logged.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import sys
from pathlib import Path

# Make scripts/ importable so we can reuse the orchestrator's helpers.
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

# Import everything we need from the orchestrator.
import magic_link_signup as mls  # noqa: E402

from playwright.async_api import async_playwright  # noqa: E402


PROVIDER = "lightning_ai"


async def main() -> int:
    magic_url = sys.stdin.read().strip()
    if not magic_url.startswith("https://"):
        print("FATAL: stdin must contain the magic-link URL (got "
              f"{len(magic_url)} chars, no https:// prefix)", file=sys.stderr)
        return 2

    with open(mls.PROVIDERS_JSON, encoding="utf-8") as f:
        cfg = json.load(f)[PROVIDER]

    profile = mls.PROFILE_ROOT / PROVIDER
    profile.mkdir(parents=True, exist_ok=True)
    mls.log(f"profile: {profile}")

    # Clear stale Chromium singleton locks (same as orchestrator does).
    for lock in ("SingletonLock", "SingletonCookie", "SingletonSocket"):
        lock_path = profile / lock
        if lock_path.exists() or lock_path.is_symlink():
            try:
                lock_path.unlink()
            except OSError:
                pass

    result: dict = {"provider": PROVIDER, "stages": [], "mode": "consume_prefetched"}

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

        # Stage 5: open the magic-link
        mls.log(f"stage 5: open magic-link URL ({len(magic_url)} chars)")
        try:
            await page.goto(magic_url, timeout=30000, wait_until="domcontentloaded")
        except Exception as e:
            mls.log(f"FATAL magic-link nav: {e}")
            await mls._screenshot(page, f"{PROVIDER}_consume_magic_link_fail")
            await ctx.close()
            return 3
        await asyncio.sleep(4)
        await mls._screenshot(page, f"{PROVIDER}_consume_post_confirm")
        post_url = page.url
        mls.log(f"  post-confirm URL: {post_url}")
        result["stages"].append({"stage": 5, "post_confirm_url": post_url})

        # Detect "token expired" / "invalid link" UIs early.
        try:
            page_text = (await page.locator("body").inner_text(timeout=2000)) or ""
        except Exception:
            page_text = ""
        bad_signals = ["expired", "invalid", "link is no longer", "try again"]
        if any(s in page_text.lower() for s in bad_signals):
            mls.log(f"FATAL: magic-link appears expired/invalid (text: "
                    f"{page_text[:200]!r})")
            await mls._screenshot(page, f"{PROVIDER}_consume_link_expired")
            await ctx.close()
            return 3

        # Stage 6: nav to token page
        token_url = cfg["token_page_url"]
        mls.log(f"stage 6: nav token page {token_url}")
        try:
            await page.goto(token_url, timeout=30000, wait_until="domcontentloaded")
        except Exception as e:
            mls.log(f"FATAL token-page nav: {e}")
            await mls._screenshot(page, f"{PROVIDER}_consume_token_page_fail")
            await ctx.close()
            return 4
        await asyncio.sleep(4)
        await mls._screenshot(page, f"{PROVIDER}_consume_token_page")
        result["stages"].append({"stage": 6, "token_page_url": page.url})

        # Stage 7: click create-token
        token_name = f"clawbot-{secrets.token_hex(4)}-{mls._ts()}"
        mls.log(f"stage 7: click create-token (name={token_name})")
        clicked_create = await mls._try_click(
            page, cfg.get("token_create_button_selectors", []), timeout_ms=10000,
        )
        if not clicked_create:
            mls.log("WARN: create-token button not found — may already exist; "
                    "scraping page for existing token.")
        else:
            await asyncio.sleep(1.5)
            await mls._try_fill(
                page, cfg.get("token_name_field_selectors", []),
                token_name, timeout_ms=4000,
            )
            await mls._try_click(
                page,
                cfg.get("token_create_button_selectors", []) + [
                    "button:has-text('Create')",
                    "button:has-text('Generate')",
                    "button[type='submit']",
                ],
                timeout_ms=4000,
            )
            await asyncio.sleep(2)

        # Stage 8: scrape token from DOM
        mls.log("stage 8: scrape token from DOM")
        token = await mls._read_token_from_dom(
            page,
            cfg.get("token_dom_selectors", []),
            cfg["token_regex"],
            timeout_ms=15000,
        )
        if not token:
            await mls._screenshot(page, f"{PROVIDER}_consume_no_token")
            mls.log("FATAL: token not visible in DOM after create-button click")
            await ctx.close()
            return 5
        mls.log(f"token captured: {len(token)} chars (value NOT logged)")
        result["stages"].append({"stage": 8, "token_chars": len(token)})

        # Stage 9: write env + flip cloud_usage
        env_file = Path(cfg["env_file"])
        mls.write_token_to_env(env_file, cfg["env_var_name"], token)
        flipped = mls.flip_cloud_usage_enabled(cfg["cloud_usage_key"])
        result["env_file"] = str(env_file)
        result["cloud_usage_flipped"] = flipped
        mls.log(f"env written: {env_file}")
        mls.log(f"cloud_usage flipped: {flipped}")

        await ctx.close()

    out_json = mls.LOG_DIR / f"magic_link_{PROVIDER}_consume_{mls._ts()}.json"
    out_json.write_text(json.dumps(result, indent=2))
    mls.log(f"result: {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
