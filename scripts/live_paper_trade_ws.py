"""
live_paper_trade_ws.py — Long-running WebSocket daemon for Alpaca trade_updates.

Runs 09:25–16:05 ET on weekdays (auto-exits outside window). Single launchd entry:
    ~/Library/LaunchAgents/com.zg.paper_trade_ws.plist

Fills are persisted to:
    paper_trade/fills/<DATE>.jsonl

`cmd_ingest` in live_paper_trade.py later reconciles these JSONL fills back into
the daily state file + outcomes parquet — fixing the `exit_price=None` bug.

Usage:
    python live_paper_trade_ws.py            # run until 16:05 ET or signal
    python live_paper_trade_ws.py --dry-run  # smoke: construct stream + exit
    python live_paper_trade_ws.py --force    # bypass the market-hours gate
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# ── sys.path setup so sibling imports work both as script + as module ──────
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from alpaca_stream_consumer import TradeUpdatesConsumer  # noqa: E402

# ─────────────────── paths + logging ────────────────────────────────────────
WORK = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools/s&p500-ticker-mastery"
)
LOGS_DIR = WORK.parent / "logs"   # AI-Tools/logs
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOGS_DIR / "paper_trade_ws.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.FileHandler(str(LOG_FILE)),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("paper_trade_ws")


# ─────────────────── ET helpers ─────────────────────────────────────────────
def _et_now() -> datetime:
    """Return current time in US/Eastern timezone."""
    try:
        import pytz
        return datetime.now(pytz.timezone("America/New_York"))
    except ImportError:
        from datetime import timezone as _tz
        return datetime.now(_tz.utc).astimezone(_tz(timedelta(hours=-4)))


def _within_window() -> bool:
    """True if now is Mon-Fri between 09:25 and 16:05 ET."""
    now = _et_now()
    if now.weekday() > 4:  # Sat=5, Sun=6
        return False
    start = now.replace(hour=9, minute=25, second=0, microsecond=0)
    end = now.replace(hour=16, minute=5, second=0, microsecond=0)
    return start <= now <= end


def _seconds_until_window_open() -> float:
    """Seconds remaining until 09:25 ET today. Returns 0 if already past or weekend."""
    now = _et_now()
    if now.weekday() > 4:
        return 0.0
    start = now.replace(hour=9, minute=25, second=0, microsecond=0)
    if now >= start:
        return 0.0
    return (start - now).total_seconds()


def _seconds_until_window_close() -> float:
    """Seconds remaining until 16:05 ET today. Returns 0 if past."""
    now = _et_now()
    close = now.replace(hour=16, minute=5, second=0, microsecond=0)
    if now >= close:
        return 0.0
    return (close - now).total_seconds()


# ─────────────────── credentials ────────────────────────────────────────────
def _detect_credentials() -> tuple[str | None, str | None]:
    """Mirror of live_paper_trade._detect_mode (subset): env + alpaca.env file + Keychain.

    Returns (api_key, secret_key) or (None, None).
    """
    api_key = os.environ.get("ALPACA_PAPER_API_KEY")
    secret_key = os.environ.get("ALPACA_PAPER_SECRET_KEY")
    if api_key and secret_key:
        return api_key, secret_key

    env_file = "/Users/orginal/.config/auto_signup/alpaca.env"
    if os.path.exists(env_file):
        try:
            for line in open(env_file):
                line = line.strip()
                if line.startswith("export ") and "=" in line:
                    k, v = line[7:].split("=", 1)
                    v = v.strip('"').strip("'")
                    os.environ.setdefault(k, v)
            api_key = os.environ.get("ALPACA_PAPER_API_KEY")
            secret_key = os.environ.get("ALPACA_PAPER_SECRET_KEY")
            if api_key and secret_key:
                return api_key, secret_key
        except Exception as e:
            log.warning("env file read failed: %s", e)

    def _kc(service: str) -> str | None:
        try:
            r = subprocess.run(
                ["security", "find-generic-password", "-s", service, "-w"],
                capture_output=True, text=True, timeout=5,
            )
            v = r.stdout.strip()
            return v if v else None
        except Exception:
            return None

    api_key = _kc("alpaca-paper-api-key")
    secret_key = _kc("alpaca-paper-secret-key")
    return api_key, secret_key


# ─────────────────── event_bus best-effort publish ──────────────────────────
def _publish_fill(rec: dict[str, Any]) -> None:
    """Publish a normalized fill to event_bus (best-effort)."""
    try:
        # event_bus lives next to this script
        from event_bus import EventBus  # type: ignore
        EventBus.publish_from_anywhere(
            "paper_fill_received",
            rec,
            source="live_paper_trade_ws",
        )
    except Exception:
        pass


async def _on_fill(rec: dict[str, Any]) -> None:
    _publish_fill(rec)


# ─────────────────── main async ─────────────────────────────────────────────
async def _amain(args: argparse.Namespace) -> int:
    # Window handling: if launchd fires before 09:25 ET (e.g. plist schedules
    # 09:20 ET so the daemon is warm by open), block-sleep until 09:25 ET
    # rather than exiting. If today is a weekend OR past 16:05 ET, exit clean.
    if not args.force and not args.dry_run:
        if not _within_window():
            now = _et_now()
            if now.weekday() > 4:
                log.warning(
                    "[WS-daemon] weekend (weekday=%d) — exiting", now.weekday()
                )
                return 0
            wait_open = _seconds_until_window_open()
            if wait_open > 0:
                # Cap pre-open wait at 30 min — defensive in case launchd fires
                # very early (e.g. DST change throws the schedule). Beyond that,
                # exit and let the next scheduled launch handle it.
                if wait_open > 1800:
                    log.warning(
                        "[WS-daemon] window opens in %.0fs (>30min) — exiting; "
                        "next launchd run will pick up", wait_open,
                    )
                    return 0
                log.info(
                    "[WS-daemon] waiting %.0fs until 09:25 ET window open",
                    wait_open,
                )
                await asyncio.sleep(wait_open)
            else:
                log.warning(
                    "[WS-daemon] past 16:05 ET (now=%s ET) — exiting",
                    now.strftime("%Y-%m-%d %H:%M"),
                )
                return 0

    api_key, secret_key = _detect_credentials()
    if not api_key or not secret_key:
        log.error("[WS-daemon] ALPACA paper creds not found — refusing to start")
        return 2

    consumer = TradeUpdatesConsumer(
        api_key=api_key,
        secret_key=secret_key,
        on_event=_on_fill,
        paper=True,
    )

    # graceful shutdown via SIGTERM (launchd sends this on unload)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, consumer.stop)
        except NotImplementedError:
            pass  # Windows; not relevant here

    # auto-exit at 16:05 ET
    if not args.dry_run:
        async def _market_close_watcher() -> None:
            wait_s = _seconds_until_window_close()
            if wait_s <= 0:
                return
            log.info("[WS-daemon] will self-stop in %.0fs (at 16:05 ET)", wait_s)
            await asyncio.sleep(wait_s)
            log.info("[WS-daemon] 16:05 ET reached — cooperative stop")
            consumer.stop()
        asyncio.create_task(_market_close_watcher())

    log.info("[WS-daemon] starting (dry_run=%s, force=%s)", args.dry_run, args.force)
    await consumer.run(dry_run=args.dry_run)
    log.info("[WS-daemon] consumer.run() returned — exiting")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Long-running Alpaca trade_updates WebSocket daemon"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Smoke: construct stream + handlers, then exit without _run_forever",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Bypass the 09:25-16:05 ET market-hours gate",
    )
    args = parser.parse_args()

    try:
        return asyncio.run(_amain(args))
    except KeyboardInterrupt:
        log.info("[WS-daemon] KeyboardInterrupt — exiting clean")
        return 0
    except Exception as e:
        log.exception("[WS-daemon] FATAL: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
