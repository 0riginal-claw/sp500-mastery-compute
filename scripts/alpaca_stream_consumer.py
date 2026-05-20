"""
alpaca_stream_consumer.py — Supervised TradingStream wrapper for Alpaca trade_updates.

Based on research report `AI-Tools/reports/alpaca_streaming_ratelimit_2026-05-18.md` §3, §7.

Adds production-grade resilience that alpaca-py lacks:
  - Exponential backoff w/ jitter (1s → 60s cap) on reconnect.
  - Heartbeat watchdog — if no events for 30s, force-recreate stream
    (works around the known 1006 stuck-state bug — alpaca-py issue #491).
  - Auth-failure detection — refuses to retry on `unauthorized` (avoid hammering).
  - Persistent fill log — every fill/partial_fill written to
    `paper_trade/fills/<DATE>.jsonl` so `cmd_ingest` can reconcile exit prices.

Usage:
    from scripts.alpaca_stream_consumer import TradeUpdatesConsumer
    consumer = TradeUpdatesConsumer(api_key, secret_key, paper=True)
    asyncio.run(consumer.run())                # blocks until ctrl-c
    consumer.stop()                            # cooperative shutdown (from another task)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

log = logging.getLogger("alpaca_ws")

# ── default fills directory ──────────────────────────────────────────────────
DEFAULT_FILLS_DIR = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools/s&p500-ticker-mastery/paper_trade/fills"
)


def _fills_path(d: date | None = None, fills_dir: Path | None = None) -> Path:
    """Path to today's fills JSONL file."""
    fills_dir = fills_dir or DEFAULT_FILLS_DIR
    fills_dir.mkdir(parents=True, exist_ok=True)
    today = (d or date.today()).isoformat()
    return fills_dir / f"{today}.jsonl"


def write_fill_event(event: dict[str, Any], fills_dir: Path | None = None) -> None:
    """Append one fill event to today's JSONL file. Idempotent on disk format."""
    path = _fills_path(fills_dir=fills_dir)
    with open(path, "a") as f:
        f.write(json.dumps(event, default=str) + "\n")


# ── event normalization ──────────────────────────────────────────────────────
# Common Alpaca trade_update event types — see report §2a.
RECORDED_EVENTS = {
    "fill",
    "partial_fill",
    "canceled",
    "rejected",
    "expired",
    "done_for_day",
}


def _msg_to_dict(msg: Any) -> dict[str, Any]:
    """Coerce an alpaca-py TradeUpdate model (Pydantic) into a plain dict.

    Falls back to attribute access if `.dict()` isn't available.
    """
    # alpaca-py v0.x: pydantic .model_dump() ; older: .dict()
    for fn_name in ("model_dump", "dict"):
        fn = getattr(msg, fn_name, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass

    # last-resort manual extraction
    order = getattr(msg, "order", None)
    return {
        "event": getattr(msg, "event", None),
        "execution_id": getattr(msg, "execution_id", None),
        "timestamp": getattr(msg, "timestamp", None),
        "qty": getattr(msg, "qty", None),
        "price": getattr(msg, "price", None),
        "position_qty": getattr(msg, "position_qty", None),
        "order": {
            "id": getattr(order, "id", None),
            "symbol": getattr(order, "symbol", None),
            "side": getattr(order, "side", None),
            "qty": getattr(order, "qty", None),
            "filled_qty": getattr(order, "filled_qty", None),
            "filled_avg_price": getattr(order, "filled_avg_price", None),
            "status": getattr(order, "status", None),
            "filled_at": getattr(order, "filled_at", None),
        } if order is not None else None,
    }


def normalize_fill(msg_dict: dict[str, Any]) -> dict[str, Any]:
    """Return the canonical fill record we persist to JSONL.

    Schema:
        order_id, symbol, side, qty, filled_avg_price, status, event,
        execution_id, timestamp, position_qty, received_at (UTC ISO).
    """
    order = msg_dict.get("order") or {}
    return {
        "order_id": str(order.get("id") or ""),
        "symbol": str(order.get("symbol") or ""),
        "side": str(order.get("side") or ""),
        "qty": str(msg_dict.get("qty") or order.get("filled_qty") or ""),
        "filled_avg_price": str(order.get("filled_avg_price") or ""),
        "status": str(order.get("status") or ""),
        "event": str(msg_dict.get("event") or ""),
        "execution_id": str(msg_dict.get("execution_id") or ""),
        "timestamp": str(msg_dict.get("timestamp") or ""),
        "position_qty": str(msg_dict.get("position_qty") or ""),
        "received_at": datetime.now(timezone.utc).isoformat(),
    }


# ─────────────────────────── consumer ────────────────────────────────────────
class TradeUpdatesConsumer:
    """Supervised TradingStream wrapper with exp-backoff reconnect + watchdog.

    Args:
        api_key, secret_key: Alpaca paper credentials.
        on_event: optional async callback invoked AFTER persistence with the
                  normalized fill dict. Use to publish to event_bus, etc.
        paper:    True for paper-api endpoint (default).
        fills_dir: override fills JSONL directory (default = repo's paper_trade/fills/).
        idle_threshold: seconds with no events before forcing stream recreate
                  (workaround for code 1006 stuck-state). Default 30.
        max_backoff: max reconnect delay in seconds. Default 60.
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        on_event: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        paper: bool = True,
        fills_dir: Path | None = None,
        idle_threshold: float = 30.0,
        max_backoff: float = 60.0,
    ):
        self.api_key = api_key
        self.secret_key = secret_key
        self.on_event = on_event
        self.paper = paper
        self.fills_dir = fills_dir or DEFAULT_FILLS_DIR
        self.idle_threshold = idle_threshold
        self.max_backoff = max_backoff
        self._attempts = 0
        self._stop = False
        self._last_event_at = time.monotonic()
        self._stream: Any = None  # current TradingStream

    # ── handler ────────────────────────────────────────────────────────────
    async def _handler(self, msg: Any) -> None:
        """Receives every trade_update from alpaca-py. Catch-all so stream stays up."""
        self._last_event_at = time.monotonic()
        try:
            d = _msg_to_dict(msg)
            ev = (d.get("event") or "").lower()
            if ev in RECORDED_EVENTS:
                rec = normalize_fill(d)
                try:
                    write_fill_event(rec, fills_dir=self.fills_dir)
                except Exception:
                    log.exception("fill persistence failed")
                log.info(
                    "[WS] %s %s qty=%s avg_px=%s order_id=%s",
                    rec.get("event"),
                    rec.get("symbol"),
                    rec.get("qty"),
                    rec.get("filled_avg_price"),
                    rec.get("order_id"),
                )
                if self.on_event is not None:
                    try:
                        await self.on_event(rec)
                    except Exception:
                        log.exception("on_event callback raised — swallowed")
            else:
                # Log-and-ignore: new, accepted, pending_new, replaced, etc.
                log.debug("[WS] (untracked) event=%s", ev)
        except Exception:
            log.exception("_handler caught unexpected — stream stays up")

    # ── backoff ────────────────────────────────────────────────────────────
    def _backoff(self) -> float:
        delay = min(self.max_backoff, 2 ** self._attempts) + random.uniform(0, 1.0)
        return float(delay)

    # ── watchdog ───────────────────────────────────────────────────────────
    async def _watchdog(self) -> None:
        """Force-recreate the stream if no events in `idle_threshold` seconds.

        Cooperates with `run()` by setting an internal flag the supervisor checks.
        """
        while not self._stop:
            await asyncio.sleep(5.0)
            idle = time.monotonic() - self._last_event_at
            if idle > self.idle_threshold and self._stream is not None:
                log.warning(
                    "[WS] watchdog: no events for %.0fs (>%.0fs threshold) — "
                    "force-recreating stream",
                    idle, self.idle_threshold,
                )
                try:
                    # alpaca-py exposes async stop_ws
                    stop_fn = getattr(self._stream, "stop_ws", None)
                    if callable(stop_fn):
                        try:
                            await stop_fn()
                        except Exception:
                            pass
                    close_fn = getattr(self._stream, "close", None)
                    if callable(close_fn):
                        try:
                            await close_fn()
                        except Exception:
                            pass
                except Exception:
                    log.exception("watchdog: stream stop failed")
                # reset the timer so we don't recreate every 5s
                self._last_event_at = time.monotonic()

    # ── supervisor loop ────────────────────────────────────────────────────
    async def run(self, dry_run: bool = False) -> None:
        """Main supervisor loop. Reconnects forever w/ backoff until stop().

        Args:
            dry_run: if True, construct the stream + subscribe + return without
                     blocking on _run_forever (used for smoke tests).
        """
        from alpaca.trading.stream import TradingStream  # lazy import — only when needed

        watchdog_task: asyncio.Task | None = None
        if not dry_run:
            watchdog_task = asyncio.create_task(self._watchdog())

        try:
            while not self._stop:
                stream = TradingStream(
                    api_key=self.api_key,
                    secret_key=self.secret_key,
                    paper=self.paper,
                )
                stream.subscribe_trade_updates(self._handler)
                self._stream = stream
                self._last_event_at = time.monotonic()

                if dry_run:
                    log.info(
                        "[WS] dry-run: stream constructed + handler subscribed "
                        "(paper=%s); exiting without _run_forever",
                        self.paper,
                    )
                    # exercise close path so the smoke test covers it
                    try:
                        await stream.close()
                    except Exception:
                        pass
                    return

                try:
                    log.info(
                        "[WS] connecting to trade_updates (attempt %d, paper=%s)",
                        self._attempts + 1, self.paper,
                    )
                    await stream._run_forever()  # alpaca-py async entry-point
                    # clean exit (stop_ws or stop called) → reset counter
                    self._attempts = 0
                except Exception as e:
                    msg = str(e)
                    if "unauthorized" in msg.lower() or "auth" in msg.lower():
                        log.error("[WS] auth failure (%s) — NOT retrying", e)
                        break
                    self._attempts += 1
                    delay = self._backoff()
                    log.warning(
                        "[WS] disconnect (%s) — reconnect in %.1fs (attempt %d)",
                        e, delay, self._attempts,
                    )
                    await asyncio.sleep(delay)
                finally:
                    try:
                        await stream.close()
                    except Exception:
                        pass
                    self._stream = None
        finally:
            if watchdog_task is not None:
                watchdog_task.cancel()
                try:
                    await watchdog_task
                except (asyncio.CancelledError, Exception):
                    pass

    def stop(self) -> None:
        """Request cooperative shutdown. Run loop will exit after current cycle."""
        self._stop = True
