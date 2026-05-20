"""
live_paper_trade_snapshots.py — Every-60s portfolio snapshot daemon.

Captures the live state of the paper-trade account every minute during market
hours (09:30–16:00 ET, weekdays). One JSONL row per minute appended to:
    paper_trade/snapshots/<DATE>.jsonl

Each row contains:
    timestamp_utc, equity, cash, buying_power, portfolio_value, last_equity,
    unrealized_pl, realized_pnl_from_equity, daytrade_count,
    pattern_day_trader, trading_blocked, account_blocked,
    positions[ticker] = {
        qty, avg_entry_price, current_price, market_value,
        cost_basis, unrealized_pl, unrealized_plpc, side, lastday_price,
        change_today, asset_marginable, qty_available,
    }

Why: live_paper_trade_ingest.py captures end-of-day daily bar only. To replay
XGBoost decisions and feed Mythos with the actual minute-by-minute equity
trajectory + per-position price evolution, we need this granularity.

Usage:
    python live_paper_trade_snapshots.py          # run until 16:05 ET
    python live_paper_trade_snapshots.py --once   # single snapshot then exit
    python live_paper_trade_snapshots.py --force  # bypass market-hours gate
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

WORK = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools/s&p500-ticker-mastery"
)
SNAPSHOT_DIR = WORK / "paper_trade" / "snapshots"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = WORK.parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOGS_DIR / "paper_trade_snapshots.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.FileHandler(str(LOG_FILE)),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("pt_snapshots")

INTERVAL_SECONDS = 60


def _et_now() -> datetime:
    try:
        import pytz
        return datetime.now(pytz.timezone("America/New_York"))
    except ImportError:
        return datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=-4)))


def _within_window() -> bool:
    now = _et_now()
    if now.weekday() > 4:
        return False
    start = now.replace(hour=9, minute=30, second=0, microsecond=0)
    end = now.replace(hour=16, minute=5, second=0, microsecond=0)
    return start <= now <= end


def _seconds_until_close() -> float:
    now = _et_now()
    close = now.replace(hour=16, minute=5, second=0, microsecond=0)
    if now >= close:
        return 0.0
    return (close - now).total_seconds()


def _detect_credentials() -> tuple[str | None, str | None]:
    api = os.environ.get("ALPACA_PAPER_API_KEY")
    sec = os.environ.get("ALPACA_PAPER_SECRET_KEY")
    if api and sec:
        return api, sec
    env_file = "/Users/orginal/.config/auto_signup/alpaca.env"
    if os.path.exists(env_file):
        try:
            for line in open(env_file):
                line = line.strip()
                if line.startswith("export ") and "=" in line:
                    k, v = line[7:].split("=", 1)
                    os.environ.setdefault(k, v.strip('"').strip("'"))
            api = os.environ.get("ALPACA_PAPER_API_KEY")
            sec = os.environ.get("ALPACA_PAPER_SECRET_KEY")
            if api and sec:
                return api, sec
        except Exception as e:
            log.warning(f"env file read failed: {e}")

    def _kc(svc: str) -> str | None:
        try:
            r = subprocess.run(
                ["security", "find-generic-password", "-s", svc, "-w"],
                capture_output=True, text=True, timeout=5,
            )
            v = r.stdout.strip()
            return v if v else None
        except Exception:
            return None
    return _kc("alpaca-paper-api-key"), _kc("alpaca-paper-secret-key")


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    v = getattr(obj, name, default)
    if v is None:
        return default
    if hasattr(v, "value"):
        return v.value
    return v


def _snapshot_once(tc: Any) -> dict[str, Any]:
    """Single account + positions snapshot."""
    acct = tc.get_account()
    positions = tc.get_all_positions()
    rec: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "timestamp_et": _et_now().isoformat(),
        "equity": str(_attr(acct, "equity", "")),
        "cash": str(_attr(acct, "cash", "")),
        "buying_power": str(_attr(acct, "buying_power", "")),
        "portfolio_value": str(_attr(acct, "portfolio_value", "")),
        "last_equity": str(_attr(acct, "last_equity", "")),
        "long_market_value": str(_attr(acct, "long_market_value", "")),
        "short_market_value": str(_attr(acct, "short_market_value", "")),
        "daytrade_count": _attr(acct, "daytrade_count", 0),
        "pattern_day_trader": bool(_attr(acct, "pattern_day_trader", False)),
        "trading_blocked": bool(_attr(acct, "trading_blocked", False)),
        "account_blocked": bool(_attr(acct, "account_blocked", False)),
        "positions": {},
        "num_positions": len(positions),
    }
    for p in positions:
        sym = _attr(p, "symbol", "")
        rec["positions"][sym] = {
            "qty": str(_attr(p, "qty", "")),
            "qty_available": str(_attr(p, "qty_available", "")),
            "side": str(_attr(p, "side", "")),
            "avg_entry_price": str(_attr(p, "avg_entry_price", "")),
            "current_price": str(_attr(p, "current_price", "")),
            "market_value": str(_attr(p, "market_value", "")),
            "cost_basis": str(_attr(p, "cost_basis", "")),
            "unrealized_pl": str(_attr(p, "unrealized_pl", "")),
            "unrealized_plpc": str(_attr(p, "unrealized_plpc", "")),
            "lastday_price": str(_attr(p, "lastday_price", "")),
            "change_today": str(_attr(p, "change_today", "")),
            "asset_marginable": bool(_attr(p, "asset_marginable", False)),
        }
    return rec


def _write_snapshot(rec: dict[str, Any]) -> Path:
    today = date.today().isoformat()
    path = SNAPSHOT_DIR / f"{today}.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    return path


async def _amain(args: argparse.Namespace) -> int:
    if not args.force and not args.once and not _within_window():
        now = _et_now()
        if now.weekday() > 4:
            log.warning(f"[snapshots] weekend (weekday={now.weekday()}) — exiting")
            return 0
        # if before open, wait
        start = now.replace(hour=9, minute=30, second=0, microsecond=0)
        if now < start:
            wait_s = (start - now).total_seconds()
            if wait_s > 1800:
                log.warning(f"[snapshots] window opens in {wait_s:.0f}s (>30min) — exiting")
                return 0
            log.info(f"[snapshots] waiting {wait_s:.0f}s until 09:30 ET")
            await asyncio.sleep(wait_s)
        else:
            log.warning(f"[snapshots] past 16:05 ET — exiting")
            return 0

    api, sec = _detect_credentials()
    if not api or not sec:
        log.error("[snapshots] ALPACA paper creds not found — refusing to start")
        return 2

    from alpaca.trading.client import TradingClient
    tc = TradingClient(api_key=api, secret_key=sec, paper=True)

    # graceful shutdown
    stop_flag = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stop_flag.set)
        except NotImplementedError:
            pass

    if args.once:
        rec = _snapshot_once(tc)
        path = _write_snapshot(rec)
        log.info(
            f"[snapshots] once → {path} (equity=${rec['equity']} positions={rec['num_positions']})"
        )
        return 0

    log.info(f"[snapshots] starting (interval={INTERVAL_SECONDS}s, force={args.force})")
    consecutive_errors = 0
    while not stop_flag.is_set():
        try:
            rec = _snapshot_once(tc)
            path = _write_snapshot(rec)
            log.info(
                f"[snapshots] {rec['timestamp_et']} equity=${rec['equity']} "
                f"cash=${rec['cash']} pos={rec['num_positions']}"
            )
            consecutive_errors = 0
        except Exception as e:
            consecutive_errors += 1
            log.warning(f"[snapshots] capture failed ({consecutive_errors}x): {e}")
            if consecutive_errors >= 10:
                log.error("[snapshots] 10 consecutive errors — exiting")
                return 1

        # auto-exit at 16:05 ET (unless --force)
        if not args.force and not _within_window():
            log.info("[snapshots] past 16:05 ET — exiting clean")
            return 0

        try:
            await asyncio.wait_for(stop_flag.wait(), timeout=INTERVAL_SECONDS)
        except asyncio.TimeoutError:
            pass

    log.info("[snapshots] stop flag set — exiting clean")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Paper trade 1-min snapshot daemon")
    p.add_argument("--once", action="store_true", help="Single snapshot then exit")
    p.add_argument("--force", action="store_true", help="Bypass market-hours gate")
    args = p.parse_args()
    try:
        return asyncio.run(_amain(args))
    except KeyboardInterrupt:
        log.info("[snapshots] KeyboardInterrupt — exiting clean")
        return 0
    except Exception as e:
        log.exception(f"[snapshots] FATAL: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
