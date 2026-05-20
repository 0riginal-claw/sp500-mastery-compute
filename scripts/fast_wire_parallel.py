"""
fast_wire_parallel.py -- asyncio batch processor for WIRE_CANDIDATE wraps.

CYCLE 3 of wire-speedup research (2026-05-17).
Eliminates per-candidate `claude -p` subprocess overhead (~300ms startup each)
by running all wraps cooperatively in a single asyncio event loop.

Pluggable: `wrap_fn` accepts any async (dict) -> any callable, so this stacks
cleanly with cycle-1 (Ollama) and cycle-2 (templates).

Usage:
    import asyncio
    from fast_wire_parallel import batch_wrap

    async def my_wrap(candidate: dict) -> dict:
        # composed: template-first, LLM fallback
        from fast_wire_template import generate_wrapper
        r = generate_wrapper(candidate)
        if r["template_match"]:
            return r
        # else escalate to Ollama (async-wrap a sync call)
        return await asyncio.to_thread(ollama_wrap, candidate)

    results = asyncio.run(batch_wrap(candidates, my_wrap, concurrency=50))

Benchmark:
    python fast_wire_parallel.py
"""
from __future__ import annotations

import asyncio
import math
import time
from typing import Any, Awaitable, Callable

WrapFn = Callable[[dict], Awaitable[Any]]


class AsyncRateLimiter:
    """Token-bucket rate limiter, asyncio-safe.
    rps=0.833 -> Anthropic Tier-1 (50 RPM)
    rps=16.67 -> Tier-2 (1000 RPM)
    rps=None  -> unlimited (local LLM / template path)
    """
    def __init__(self, rps: float):
        if rps <= 0:
            raise ValueError("rps must be > 0")
        self._interval = 1.0 / rps
        self._next_allowed = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._next_allowed = time.monotonic() + self._interval


async def batch_wrap(
    candidates: list[dict],
    wrap_fn: WrapFn,
    concurrency: int = 50,
    rps_limit: float | None = None,
    progress_every: int = 25,
    verbose: bool = False,
) -> list[dict]:
    """Run wrap_fn over all candidates concurrently.

    Args:
      candidates:    list of WIRE_CANDIDATE specs (dicts).
      wrap_fn:       async (dict) -> any. Plug in template/LLM logic.
      concurrency:   max in-flight tasks (default 50).
      rps_limit:     requests/sec ceiling, or None for unlimited.
      progress_every: log progress every N completions (if verbose).
      verbose:       print progress to stderr.

    Returns:
      list of dicts (one per candidate, same order):
        {"ok": bool, "result": any, "latency": float}
        on error: {"ok": False, "error": str, "latency": float}
    """
    sem = asyncio.Semaphore(concurrency)
    rl = AsyncRateLimiter(rps_limit) if rps_limit else None
    results: list[Any] = [None] * len(candidates)
    done = 0
    start = time.monotonic()

    async def _one(idx: int, cand: dict) -> None:
        nonlocal done
        async with sem:
            if rl is not None:
                await rl.acquire()
            t0 = time.monotonic()
            try:
                res = await wrap_fn(cand)
                results[idx] = {"ok": True, "result": res,
                                "latency": time.monotonic() - t0}
            except Exception as e:
                results[idx] = {"ok": False, "error": f"{type(e).__name__}: {e}",
                                "latency": time.monotonic() - t0}
            done += 1
            if verbose and done % progress_every == 0:
                elapsed = time.monotonic() - start
                rate = done / elapsed if elapsed > 0 else 0
                print(f"  [{done}/{len(candidates)}] {rate:.2f} req/s "
                      f"elapsed={elapsed:.1f}s", flush=True)

    await asyncio.gather(*(_one(i, c) for i, c in enumerate(candidates)))
    return results


# ---------------------------------------------------------------------------
# Benchmark -- mock wrap_fn with asyncio.sleep to simulate IO latency
# ---------------------------------------------------------------------------
async def _mock_wrap(candidate: dict) -> dict:
    """Mock async wrap with configurable latency via candidate['_sim_lat']."""
    lat = candidate.get("_sim_lat", 0.1)
    await asyncio.sleep(lat)
    return {"wrapped": candidate.get("name", "?")}


async def _run_scenario(n: int, lat: float, concurrency: int,
                        rps: float | None) -> tuple[float, int]:
    cands = [{"name": f"c{i}", "_sim_lat": lat} for i in range(n)]
    t0 = time.monotonic()
    results = await batch_wrap(cands, _mock_wrap,
                                concurrency=concurrency, rps_limit=rps)
    wall = time.monotonic() - t0
    ok = sum(1 for r in results if r["ok"])
    return wall, ok


def _projected(n: int, lat: float, concurrency: int,
                rps: float | None) -> float:
    """Analytical projection: wall = max(lat * ceil(n/conc), n/rps)."""
    conc_floor = lat * math.ceil(n / concurrency)
    rate_floor = n / rps if rps else 0.0
    return max(conc_floor, rate_floor)


def _benchmark() -> None:
    scenarios = [
        # (n,  lat,  conc, rps,    label)
        (5,   0.1,  50,   None,   "5 cands, 100ms, c=50 unlimited"),
        (5,   5.0,  50,   None,   "5 cands, 5s, c=50 unlimited"),
        (20,  0.1,  50,   None,   "20 cands, 100ms, c=50 unlimited"),
        (20,  0.1,  4,    None,   "20 cands, 100ms, c=4 (baseline)"),
        (429, 0.05, 50,   None,   "429 cands, 50ms (template path)"),
        (429, 5.0,  50,   None,   "429 cands, 5s (Ollama path, unlimited)"),
        (429, 5.0,  50,   0.833,  "429 cands, 5s, Tier-1 (50 RPM)"),
        (429, 5.0,  50,   16.67,  "429 cands, 5s, Tier-2 (1000 RPM)"),
    ]
    print(f"{'Scenario':<50} {'Wall(s)':>10} {'Proj(s)':>10} {'Throughput':>14}")
    print("-" * 90)
    for n, lat, conc, rps, label in scenarios:
        # Only RUN scenarios that finish quickly; project the rest.
        proj = _projected(n, lat, conc, rps)
        if proj <= 15:
            wall, ok = asyncio.run(_run_scenario(n, lat, conc, rps))
            thr = n / wall if wall > 0 else 0
            print(f"{label:<50} {wall:>10.2f} {proj:>10.2f} "
                  f"{thr:>10.1f} req/s")
        else:
            print(f"{label:<50} {'(skip)':>10} {proj:>10.2f} "
                  f"{n / proj:>10.1f} req/s")

    print("\n--- Wire-throughput summary (429 candidates) ---")
    base_wall_min = 8 * 429 / 4  # 4-parallel subprocess baseline
    print(f"  Baseline (subprocess, 4 parallel):       {base_wall_min:.0f} min")
    print(f"  Asyncio + Ollama (5s, c=50, no limit):   "
          f"{_projected(429, 5.0, 50, None) / 60:.1f} min")
    print(f"  Asyncio + template (50ms, c=50):         "
          f"{_projected(429, 0.05, 50, None):.1f} s")
    print(f"  Asyncio + Claude API Tier-2 (5s, c=50):  "
          f"{_projected(429, 5.0, 50, 16.67):.1f} s")


if __name__ == "__main__":
    _benchmark()
