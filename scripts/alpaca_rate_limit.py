"""
alpaca_rate_limit.py — Token-bucket + 429-aware rate-limit middleware for Alpaca REST.

Based on research report `AI-Tools/reports/alpaca_streaming_ratelimit_2026-05-18.md` §4 + §8.

Alpaca enforces 200 req/min on trade endpoints (free/Basic plan). This module:
  - Pre-throttles via token bucket (default 180 RPM → ~10% headroom under cap).
  - Detects 429 responses + honors `Retry-After` header when present.
  - Falls back to exponential backoff (2^n + 1 seconds) on 429 when no header.
  - Provides both sync (`call`) and async (`acall`) entry-points.
  - Single module-level singleton `ALPACA_RL` — import + use anywhere.

Usage:
    from scripts.alpaca_rate_limit import ALPACA_RL
    order = ALPACA_RL.call(client.submit_order, req)             # sync
    order = await ALPACA_RL.acall(client.submit_order, req)      # async

Bulk submit:
    async def submit_orders_bulk(orders, max_parallel=5):
        sem = asyncio.Semaphore(max_parallel)
        async def one(o):
            async with sem:
                return await ALPACA_RL.acall(client.submit_order, o)
        return await asyncio.gather(*(one(o) for o in orders), return_exceptions=True)
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any, Callable

log = logging.getLogger("alpaca_rl")


class RateLimitMiddleware:
    """Token-bucket + 429-aware wrapper for Alpaca REST calls.

    Thread-safe (sync `call`) and asyncio-safe (`acall`). Both use the same bucket
    so mixing sync + async clients in one process is safe.

    Args:
        rpm:    request budget per minute (default 180; cap is 200 on Basic plan).
        burst:  bucket capacity (default 10). Permits short bursts up to `burst`
                concurrent calls without throttling.
        max_retries: max 429 retries before giving up. Default 5.
    """

    def __init__(self, rpm: int = 180, burst: int = 10, max_retries: int = 5):
        self.rate = rpm / 60.0
        self.tokens = float(burst)
        self.capacity = float(burst)
        self.last = time.monotonic()
        self.max_retries = max_retries
        self._sync_lock = threading.Lock()
        self._async_lock: asyncio.Lock | None = None  # lazy — needs event loop
        self.rpm = rpm

    # ──────────────────────── token refill (shared logic) ────────────────────
    def _refill_locked(self) -> None:
        """Refill tokens proportional to elapsed wall-clock. Must be called under lock."""
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
        self.last = now

    def _take_or_compute_wait(self) -> float:
        """Try to consume one token. Returns 0.0 if consumed, else wait-seconds."""
        self._refill_locked()
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return 0.0
        wait = (1.0 - self.tokens) / self.rate
        return wait

    # ──────────────────────── sync path ────────────────────────
    def acquire_sync(self) -> None:
        """Block (sync) until one token is available."""
        while True:
            with self._sync_lock:
                wait = self._take_or_compute_wait()
                if wait == 0.0:
                    return
            # release lock before sleeping
            log.debug("rate-limit (sync) wait %.2fs", wait)
            time.sleep(wait)

    def call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Invoke a sync SDK fn under the bucket; honor 429 Retry-After if raised."""
        for attempt in range(self.max_retries):
            self.acquire_sync()
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                delay = self._parse_429_delay(e, attempt)
                if delay is None:
                    raise
                log.warning(
                    "429 — backing off %ds (attempt %d/%d)",
                    delay, attempt + 1, self.max_retries,
                )
                time.sleep(delay)
        raise RuntimeError(
            f"alpaca_rate_limit: 429 retries exhausted ({self.max_retries})"
        )

    # ──────────────────────── async path ────────────────────────
    def _get_async_lock(self) -> asyncio.Lock:
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()
        return self._async_lock

    async def acquire(self) -> None:
        """Async-acquire one token."""
        lock = self._get_async_lock()
        while True:
            async with lock:
                wait = self._take_or_compute_wait()
                if wait == 0.0:
                    return
            log.debug("rate-limit (async) wait %.2fs", wait)
            await asyncio.sleep(wait)

    async def acall(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Invoke a sync SDK fn from async code under the bucket; honor 429."""
        for attempt in range(self.max_retries):
            await self.acquire()
            try:
                return await asyncio.to_thread(fn, *args, **kwargs)
            except Exception as e:
                delay = self._parse_429_delay(e, attempt)
                if delay is None:
                    raise
                log.warning(
                    "429 — backing off %ds (attempt %d/%d)",
                    delay, attempt + 1, self.max_retries,
                )
                await asyncio.sleep(delay)
        raise RuntimeError(
            f"alpaca_rate_limit: 429 retries exhausted ({self.max_retries})"
        )

    # ──────────────────────── 429 detection ────────────────────────
    @staticmethod
    def _parse_429_delay(e: BaseException, attempt: int) -> int | None:
        """Return retry-delay in seconds if `e` looks like a 429, else None."""
        msg = str(e)
        is_429 = "429" in msg or "rate limit" in msg.lower()
        if not is_429:
            return None
        # default: exp backoff 2,3,5,9,17,...
        delay = 2 ** attempt + 1
        # try Retry-After header if SDK exposes it
        resp = getattr(e, "response", None)
        if resp is not None:
            try:
                hdrs = getattr(resp, "headers", {}) or {}
                ra_hdr = hdrs.get("Retry-After") or hdrs.get("retry-after")
                if ra_hdr is not None:
                    ra_str = str(ra_hdr).strip()
                    if ra_str.isdigit():
                        delay = max(delay, int(ra_str))
            except Exception:
                pass
        return delay

    def __repr__(self) -> str:
        return (
            f"<RateLimitMiddleware rpm={self.rpm} burst={int(self.capacity)} "
            f"tokens={self.tokens:.2f}>"
        )


# Module-level singleton — conservative 180 RPM (10% headroom under 200 cap)
ALPACA_RL = RateLimitMiddleware(rpm=180, burst=10)


# ─────────────────────── async bulk submit helper ───────────────────────
async def submit_orders_bulk(
    client: Any,
    requests: list[Any],
    max_parallel: int = 5,
) -> list[Any]:
    """Submit a list of MarketOrderRequest objects in parallel under the bucket.

    Returns list of results in same order; Exceptions are returned in-place
    (not raised) thanks to `return_exceptions=True`.

    Default max_parallel=5 matches the undocumented Alpaca order-creation
    sub-limit reported on the forum (see report §4).
    """
    sem = asyncio.Semaphore(max_parallel)

    async def one(req: Any) -> Any:
        async with sem:
            return await ALPACA_RL.acall(client.submit_order, req)

    return await asyncio.gather(
        *(one(r) for r in requests),
        return_exceptions=True,
    )
