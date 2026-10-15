#!/usr/bin/env python3
"""Camoufox vs Patchright smoke test on a reCAPTCHA Enterprise target.

Compares fingerprint-flag rate, reCAPTCHA widget detection, and time-to-load
on cerebras.ai signup (the hardest reCAPTCHA Enterprise gate the bot has
encountered).

Headless run — does NOT submit any form fields. Read-only DOM inspection.

License: MIT (camoufox), Apache-2.0 (patchright). Both workspace-acceptable.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

TARGET_URL = "https://cloud.cerebras.ai/?login_hint="
PROFILE_ROOT = Path("/tmp/camoufox_vs_patchright")
PROFILE_ROOT.mkdir(parents=True, exist_ok=True)

# Allow `--headed` to drive a visible window for manual inspection.
HEADED = "--headed" in sys.argv


async def _scan_page(page) -> Dict[str, Any]:
    """Inspect the page for reCAPTCHA presence + bot-detection hints."""
    # Wait for network mostly idle (cerebras loads heavy JS).
    try:
        await page.wait_for_load_state("networkidle", timeout=20000)
    except Exception:
        pass
    # SPA may still be hydrating — give it explicit head room.
    await asyncio.sleep(5)

    info = await page.evaluate(
        """() => {
            const has = (sel) => !!document.querySelector(sel);
            return {
                title: document.title,
                url: location.href,
                grecaptcha_global: typeof window.grecaptcha !== 'undefined',
                recaptcha_iframe: has('iframe[src*="recaptcha"]'),
                recaptcha_enterprise: has('iframe[src*="recaptcha/enterprise"]'),
                hcaptcha_iframe: has('iframe[src*="hcaptcha"]'),
                cloudflare_challenge: document.body.innerText.includes('Just a moment'),
                blocked_strings: ['blocked', 'forbidden', 'unusual traffic'].filter(
                    s => document.body.innerText.toLowerCase().includes(s)
                ),
                userAgent: navigator.userAgent,
                webdriver: navigator.webdriver,
                languages: navigator.languages,
                platform: navigator.platform,
                vendor: navigator.vendor,
                bodyLen: document.body.innerText.length,
            };
        }"""
    )
    return info


async def run_patchright() -> Dict[str, Any]:
    from patchright.async_api import async_playwright

    out: Dict[str, Any] = {"engine": "patchright"}
    t0 = time.monotonic()
    profile = PROFILE_ROOT / "patchright"
    profile.mkdir(parents=True, exist_ok=True)
    try:
        async with async_playwright() as pw:
            ctx = await pw.chromium.launch_persistent_context(
                user_data_dir=str(profile),
                channel="chromium",
                headless=not HEADED,
                no_viewport=True,
            )
            page = ctx.pages[0] if ctx.pages else await ctx.new_page()
            await page.goto(TARGET_URL, wait_until="domcontentloaded", timeout=30000)
            out["info"] = await _scan_page(page)
            await ctx.close()
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    out["elapsed_s"] = round(time.monotonic() - t0, 2)
    return out


async def run_camoufox() -> Dict[str, Any]:
    from camoufox.async_api import AsyncCamoufox

    out: Dict[str, Any] = {"engine": "camoufox"}
    t0 = time.monotonic()
    profile = PROFILE_ROOT / "camoufox"
    profile.mkdir(parents=True, exist_ok=True)
    try:
        # AsyncCamoufox is a context manager that yields a Playwright browser.
        async with AsyncCamoufox(
            headless=not HEADED,
            persistent_context=True,
            user_data_dir=str(profile),
            humanize=True,
            os="macos",
        ) as ctx:
            page = ctx.pages[0] if ctx.pages else await ctx.new_page()
            await page.goto(TARGET_URL, wait_until="domcontentloaded", timeout=30000)
            out["info"] = await _scan_page(page)
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    out["elapsed_s"] = round(time.monotonic() - t0, 2)
    return out


async def main() -> int:
    results = {}
    for name, fn in (("patchright", run_patchright), ("camoufox", run_camoufox)):
        print(f"[{name}] starting...", flush=True)
        results[name] = await fn()
        info = results[name].get("info") or {}
        err = results[name].get("error")
        if err:
            print(f"[{name}] ERROR: {err}", flush=True)
        else:
            print(
                f"[{name}] title={info.get('title')!r} "
                f"recap_enterprise={info.get('recaptcha_enterprise')} "
                f"recap_any={info.get('recaptcha_iframe')} "
                f"webdriver={info.get('webdriver')} "
                f"blocked={info.get('blocked_strings')} "
                f"elapsed={results[name]['elapsed_s']}s",
                flush=True,
            )

    print("\n--- JSON RESULTS ---")
    print(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
