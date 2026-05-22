"""
live_paper_trade.py — Main paper-trading orchestration script.

Subcommands (run at scheduled ET times):
    startup      09:00 ET — load models, fetch overnight data, regenerate signals
    open-trades  09:30 ET — place paper-trade orders for firing signals
    flatten      15:55 ET — close all open positions
    ingest       16:30 ET — save fills/P&L, append training data, trigger retrain

Mode detection:
    SIMULATED  — Alpaca credentials absent; simulates fills from yfinance bar data
    LIVE_PAPER — Alpaca credentials present; uses the Gabriel alpaca-system
                 wrapper (OrderManager / AccountManager / MarketDataManager)
                 against the Alpaca paper-trade API. Direct alpaca-py imports
                 have been removed — every API call now flows through the
                 wrapper's public manager surface.

Risk guardrails:
    MAX_POSITION_NOTIONAL    $100 per ticker  (5% of $2k synthetic budget,
                              matches Kelly half-K cap KELLY_MAX_FRACTION=0.05
                              and concentration gate MAX_PCT_PER_TICKER=0.05)
    MAX_TOTAL_EXPOSURE       $2,000 across all open positions
                              (synthetic budget — Alpaca paper account equity
                               ~$95k is IGNORED for sizing; used for reporting only)
    DAILY_LOSS_SOFT_HALT     -$120 → block NEW entries, exit-only mode
                              (~6% of $2k budget; tracks original 6% setting)
    DAILY_LOSS_HARD_HALT     -$200 → flat all positions, halt for day
                              (~10% of $2k budget)
    BATCH_OPEN_STAGGER_S     1.0     → between submit_order calls at 09:30 open
    ORDER_TYPE               MARKET only, regular-hours only
    NO SHORTS / NO OPTIONS / NO LEVERAGE / NO OVERNIGHT HOLDS

    Cap rationale (2026-05-22): user requested live paper-trade balance
    restricted to $2,000 starting July 2026 (sized for transition to real money).
    Alpaca paper account equity inflates PnL ~47×; capping at $2k via synthetic
    budget makes paper PnL representative of live $2k account. Per-ticker cap
    is 5% (matches Kelly + concentration gates) so a 20-name portfolio fully
    deploys the $2k budget without any risk gate refusing.

    Env overrides (read at startup):
      LIVE_BUDGET_USD              — total budget (default 2000)
      LIVE_MAX_POSITION_USD        — per-ticker cap (default 100, 5% of budget)
      LIVE_DAILY_LOSS_SOFT_USD     — soft halt (default -120, 6% of budget)
      LIVE_DAILY_LOSS_HARD_USD     — hard halt (default -200, 10% of budget)

    Thresholds source: paper_trade/alpaca_risk_config.yaml (SAFE-SUBSET applied
    2026-05-18 per reports/alpaca_risk_sizing_2026-05-18.md, re-capped 2026-05-22
    per cap-2k-july-2026-05-22). Half-Kelly/ATR sizing + limit-order migration
    are DEFERRED pending backtest validation.

Usage:
    python live_paper_trade.py startup
    python live_paper_trade.py open-trades
    python live_paper_trade.py flatten
    python live_paper_trade.py ingest
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

# ── Gabriel alpaca-system wrapper (canonical Alpaca interface) ────────────────
# The wrapper lives outside the venv site-packages, so prepend its src/ to
# sys.path before any wrapper imports. Importing the wrapper modules is
# deferred to the LIVE_PAPER helpers below so SIMULATED mode (no creds, or
# Drive unmounted) does not pay any wrapper-import cost.
_GABRIEL_WRAPPER_SRC = (
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/version_3 - Gabriel/Alpaca System - Gabriel/alpaca-system/src"
)
if _GABRIEL_WRAPPER_SRC not in sys.path:
    sys.path.insert(0, _GABRIEL_WRAPPER_SRC)

# ── event bus (best-effort) ──────────────────────────────────────────────────
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from event_bus import EventBus as _EventBus
    _EB = _EventBus
except Exception:
    _EB = None  # type: ignore[assignment]

# ── alpaca rate-limit middleware (best-effort) ───────────────────────────────
# Token-bucket + 429-aware wrapper. When imported, every REST call gets
# automatically paced under the 200-RPM Alpaca cap (180 RPM with 10% headroom).
# Falls back to direct calls if the module is missing.
# See `alpaca_rate_limit.py` + report
# `reports/alpaca_streaming_ratelimit_2026-05-18.md` §4 + §8.
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from alpaca_rate_limit import ALPACA_RL, submit_orders_bulk  # type: ignore
    _RL: Any = ALPACA_RL
    _RL_BULK: Any = submit_orders_bulk
except Exception:
    _RL = None
    _RL_BULK = None

# ── limit-order router (a6ead433 #1: capture 10-20bps retail limit lift) ─────
# Off by default; enable via LIMIT_ORDER_ROUTER=1 in env. When on, paper-trade
# BUY/SELL go through the smart-limit router (passive-mid OR marketable-limit
# depending on signal_strength) with a 60s TIF and market-fallback on no-fill.
# See `scripts/limit_order_router.py` + report
# `AI-Tools/research/limit_order_router_2026-05-21/repo_2026-05-21.md`.
try:
    from limit_order_router import route_order as _route_limit_order  # type: ignore
    _LIMIT_ROUTER_AVAILABLE = True
except Exception:
    _route_limit_order = None  # type: ignore
    _LIMIT_ROUTER_AVAILABLE = False

_LIMIT_ROUTER_ENABLED = os.environ.get("LIMIT_ORDER_ROUTER", "0") in ("1", "true", "yes")


def _rl_call(fn, *args, **kwargs):
    """Invoke `fn` via ALPACA_RL.call when available; else direct call."""
    if _RL is not None:
        return _RL.call(fn, *args, **kwargs)
    return fn(*args, **kwargs)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORK = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools/s&p500-ticker-mastery"
)
SCRIPTS_DIR = WORK / "scripts"
PAPER_DIR = WORK / "paper_trade"
SIGNALS_DIR = PAPER_DIR / "signals"
DAILY_DIR = PAPER_DIR / "daily"
STATE_DIR = PAPER_DIR / "state"
INCREMENTAL_DIR = PAPER_DIR / "incremental_bars"
LOGS_DIR = WORK / "logs"

# Ensure directories exist
for _d in [SIGNALS_DIR, DAILY_DIR, STATE_DIR, INCREMENTAL_DIR, LOGS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FILE = LOGS_DIR / "paper_trade.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.FileHandler(str(LOG_FILE)),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("paper_trade")

# ---------------------------------------------------------------------------
# Risk constants — synthetic $2k budget cap (2026-05-22)
# ---------------------------------------------------------------------------
# User mandate: live paper-trade balance restricted to $2,000 starting July
# 2026 (transition to real money). Alpaca paper account equity (~$95k) is now
# IGNORED for sizing; we use a synthetic budget. All caps below scale to this
# budget. Defaults honoured by env vars so an ops change ($1k / $5k / etc.)
# does not need a code edit.
#
# Defaults (calibrated to $2k):
#   total budget               $2,000           (LIVE_BUDGET_USD)
#   per-ticker max             $100  (5%)       (LIVE_MAX_POSITION_USD)
#   daily soft halt             -$120 (6%)      (LIVE_DAILY_LOSS_SOFT_USD)
#   daily hard halt             -$200 (10%)     (LIVE_DAILY_LOSS_HARD_USD)
#
# Per-ticker = 5% matches risk_engine.KELLY_MAX_FRACTION (0.05) and
# MAX_PCT_PER_TICKER (0.05). At 5%, a fully-deployed portfolio holds 20 names
# at $100 each = $2k. The Kelly + concentration gates therefore PASS the
# sized notional cleanly; if we used 20% per ticker ($400), Kelly would refuse
# every entry.
#
# Halt thresholds were previously -$1,500 / -$2,500 (1.6% / 2.6% of $95k).
# At $2k budget the equivalent %-of-budget figures are -$32 / -$53, but those
# are absurdly tight for any single-ticker move; we lift to 6%/10% which is
# consistent with the per-ticker DD bands in risk_engine.py.
LIVE_BUDGET_USD = float(os.environ.get("LIVE_BUDGET_USD", "2000"))
MAX_POSITION_NOTIONAL = float(
    os.environ.get("LIVE_MAX_POSITION_USD", str(LIVE_BUDGET_USD * 0.05))
)
MAX_TOTAL_EXPOSURE = LIVE_BUDGET_USD

DAILY_LOSS_SOFT_HALT = -abs(float(
    os.environ.get("LIVE_DAILY_LOSS_SOFT_USD", str(LIVE_BUDGET_USD * 0.06))
))  # $ → block NEW entries, allow exits/manage existing
DAILY_LOSS_HARD_HALT = -abs(float(
    os.environ.get("LIVE_DAILY_LOSS_HARD_USD", str(LIVE_BUDGET_USD * 0.10))
))  # $ → flat all positions, halt for day
# Back-compat alias (deprecated; refers to HARD threshold).
MAX_DAILY_LOSS = DAILY_LOSS_HARD_HALT

# Order pacing — 1s stagger between batch-open submits at 09:30 ET.
# Reduces mass-fill latency cost (~0.0930 per helper finding in
# reports/alpaca_risk_sizing_2026-05-18.md).
BATCH_OPEN_STAGGER_S = 1.0

ALPACA_BASE_URL_PAPER = "https://paper-api.alpaca.markets"

# ── Market-hours helpers (pytz optional; fallback to UTC offset) ─────────────
def _et_now() -> datetime:
    """Return current time in US/Eastern timezone."""
    try:
        import pytz
        return datetime.now(pytz.timezone("America/New_York"))
    except ImportError:
        # Summer: EDT = UTC-4; Winter: EST = UTC-5. Use UTC-4 as conservative default.
        from datetime import timezone as _tz
        return datetime.now(_tz.utc).astimezone(_tz(timedelta(hours=-4)))


def _assert_market_window(
    subcommand: str,
    open_hour: int,
    open_minute: int,
    close_hour: int,
    close_minute: int,
    grace_minutes: int = 15,
) -> None:
    """
    Log a WARNING (non-fatal) if the current ET time is outside the expected
    window for the given subcommand. Does NOT block execution — scheduling errors
    are caught by the warning, not by hard-stopping.
    """
    now = _et_now()
    start = now.replace(hour=open_hour, minute=open_minute, second=0, microsecond=0)
    end = now.replace(hour=close_hour, minute=close_minute, second=0, microsecond=0)
    if not (start <= now <= end):
        log.warning(
            f"[TIMING] {subcommand} called at {now.strftime('%H:%M')} ET — "
            f"expected window {open_hour:02d}:{open_minute:02d}–"
            f"{close_hour:02d}:{close_minute:02d} ET. "
            "Proceeding anyway — verify this is intentional."
        )


# ── NYSE market-day guard (weekday + holiday) ────────────────────────────────
# Hardcoded NYSE 2026 holiday calendar (full closes only — does not enumerate
# early-close days, since those still HAVE a market session).
# Source: https://www.nyse.com/markets/hours-calendars (verified 2026-05-22).
# Update yearly. If pandas_market_calendars becomes available it takes priority.
_NYSE_HOLIDAYS_2026 = {
    "2026-01-01",  # New Year's Day
    "2026-01-19",  # MLK Jr. Day
    "2026-02-16",  # Presidents Day
    "2026-04-03",  # Good Friday
    "2026-05-25",  # Memorial Day
    "2026-06-19",  # Juneteenth
    "2026-07-03",  # Independence Day observed (Sat Jul 4 → market closed Fri)
    "2026-09-07",  # Labor Day
    "2026-11-26",  # Thanksgiving Day
    "2026-12-25",  # Christmas Day
}
_NYSE_HOLIDAYS_2027 = {
    "2027-01-01",  # New Year's Day
    "2027-01-18",  # MLK Jr. Day
    "2027-02-15",  # Presidents Day
    "2027-03-26",  # Good Friday
    "2027-05-31",  # Memorial Day
    "2027-06-18",  # Juneteenth observed (Sat Jun 19 → market closed Fri)
    "2027-07-05",  # Independence Day observed (Sun Jul 4 → market closed Mon)
    "2027-09-06",  # Labor Day
    "2027-11-25",  # Thanksgiving Day
    "2027-12-24",  # Christmas Day observed (Sat Dec 25 → market closed Fri)
}
_NYSE_HOLIDAYS_HARDCODED = _NYSE_HOLIDAYS_2026 | _NYSE_HOLIDAYS_2027


def is_market_day(dt: "datetime | None" = None) -> bool:
    """True if NYSE is open today (regular session, full or partial).

    Returns False on Sat/Sun and on hardcoded NYSE full-close holidays.
    Prefers pandas_market_calendars if installed; falls back to the
    _NYSE_HOLIDAYS_HARDCODED set above.

    Note: early-close days (day-after-Thanksgiving, Christmas Eve in some
    years, July 3 some years) are STILL trading days and return True.
    """
    dt = dt or _et_now()
    # Saturday=5, Sunday=6
    if dt.weekday() >= 5:
        return False
    # Try pandas_market_calendars (authoritative)
    try:
        import pandas_market_calendars as mcal  # type: ignore
        nyse = mcal.get_calendar("NYSE")
        schedule = nyse.schedule(start_date=dt.date(), end_date=dt.date())
        return len(schedule) > 0
    except ImportError:
        pass
    except Exception as e:
        log.warning("pandas_market_calendars lookup failed (%s) — falling back to hardcoded list", e)
    # Fallback: hardcoded holiday list
    return dt.strftime("%Y-%m-%d") not in _NYSE_HOLIDAYS_HARDCODED


def _market_day_guard(subcommand: str) -> bool:
    """Common early-exit guard for paper-trade subcommands.

    Returns True if the caller should proceed (market is open today),
    False if the caller should early-exit with exit code 0.
    Logs a clear INFO message in both cases.
    """
    if is_market_day():
        return True
    now = _et_now()
    log.info(
        "[MARKET-CLOSED] %s skipped: %s (%s) is not a NYSE trading day "
        "(weekend or holiday). Exiting cleanly.",
        subcommand,
        now.strftime("%Y-%m-%d"),
        now.strftime("%A"),
    )
    return False


# ---------------------------------------------------------------------------
# Credential detection
# ---------------------------------------------------------------------------
def _detect_mode() -> tuple[str, str | None, str | None]:
    """
    Returns (mode, api_key, secret_key).
    mode is 'LIVE_PAPER' if credentials found, else 'SIMULATED'.
    Credentials are sourced from env vars or macOS Keychain.
    Values are NEVER logged — only mode is reported.
    """
    api_key = os.environ.get("ALPACA_PAPER_API_KEY")
    secret_key = os.environ.get("ALPACA_PAPER_SECRET_KEY")

    if api_key and secret_key:
        log.info("Credentials: env vars → mode=LIVE_PAPER")
        return "LIVE_PAPER", api_key, secret_key

    # Fallback: source ~/.config/auto_signup/alpaca.env (chmod 600 file)
    # before falling through to Keychain. This is how launchd invokes the
    # script — plist EnvironmentVariables don't include credentials by design.
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
                log.info("Credentials: %s → mode=LIVE_PAPER", env_file)
                return "LIVE_PAPER", api_key, secret_key
        except Exception as e:
            log.warning("env file read failed: %s", e)

    # Try Keychain
    def _kc(service: str) -> str | None:
        try:
            result = subprocess.run(
                ["security", "find-generic-password", "-s", service, "-w"],
                capture_output=True, text=True, timeout=5
            )
            val = result.stdout.strip()
            return val if val else None
        except Exception:
            return None

    api_key = _kc("alpaca-paper-api-key")
    secret_key = _kc("alpaca-paper-secret-key")

    if api_key and secret_key:
        log.info("Credentials: macOS Keychain → mode=LIVE_PAPER")
        return "LIVE_PAPER", api_key, secret_key

    log.warning("Credentials: NOT FOUND (Keychain+env both missing) → mode=SIMULATED")
    return "SIMULATED", None, None


MODE, _API_KEY, _SECRET_KEY = _detect_mode()


# ---------------------------------------------------------------------------
# Signal loading
# ---------------------------------------------------------------------------
def load_today_signals(today: str | None = None) -> list[dict[str, Any]]:
    """Load today's signal file. Returns [] if missing."""
    today_str = today or date.today().isoformat()
    path = SIGNALS_DIR / f"{today_str}.json"
    if not path.exists():
        log.warning(f"Signal file missing: {path}")
        return []
    with open(path) as f:
        signals = json.load(f)
    log.info(f"Loaded {len(signals)} signals from {path}")
    return signals


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------
def load_state(today: str) -> dict:
    """Load today's trading state (open positions, P&L running total)."""
    path = STATE_DIR / f"{today}_state.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {
        "date": today,
        "positions": {},       # {ticker: {qty, entry_price, order_id, notional}}
        "closed_trades": [],   # list of fill dicts
        "realized_pnl": 0.0,
        "halted": False,       # True if daily loss limit breached
        "mode": MODE,
    }


def save_state(state: dict) -> None:
    today = state["date"]
    path = STATE_DIR / f"{today}_state.json"
    with open(path, "w") as f:
        json.dump(state, f, indent=2, default=str)
    log.debug(f"State saved → {path}")


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------
def compute_position_sizes(signals: list[dict], current_exposure: float) -> list[dict]:
    """
    Equal-weight sizing across all firing signals, capped by:
      - MAX_POSITION_NOTIONAL per ticker
      - remaining headroom under MAX_TOTAL_EXPOSURE
    Returns only tickers that fit within guardrails.
    """
    firing = [s for s in signals if s.get("signal") == 1]
    if not firing:
        return []

    available = max(0.0, MAX_TOTAL_EXPOSURE - current_exposure)
    if available <= 0:
        log.warning(f"No headroom: current_exposure={current_exposure:.0f} >= {MAX_TOTAL_EXPOSURE:.0f}")
        return []

    per_ticker_budget = min(MAX_POSITION_NOTIONAL, available / len(firing))
    sized = []
    for s in firing:
        notional = per_ticker_budget
        if notional < 1.0:
            continue
        s = dict(s)
        s["notional"] = round(notional, 2)
        sized.append(s)
    log.info(f"Sized {len(sized)} positions at ~${per_ticker_budget:.0f}/each "
             f"(available=${available:.0f})")
    return sized


# ---------------------------------------------------------------------------
# LIVE_PAPER: Alpaca helpers (routed through Gabriel alpaca-system wrapper)
# ---------------------------------------------------------------------------
# Cache of (OrderManager, AccountManager, MarketDataManager) so wrapper init
# (config TOML parse + AlpacaClient construction) only happens once per run.
_WRAPPER_MANAGERS: tuple[Any, Any, Any] | None = None


def _alpaca_managers() -> tuple[Any, Any, Any]:
    """Return cached (OrderManager, AccountManager, MarketDataManager).

    The wrapper handles auth, retry, sub-penny rejection, qty/notional caps,
    rate-limit token bucket, and idempotent client_order_id resubmission. The
    workspace `_rl_call` token bucket still pre-throttles every API call as
    a second layer of defense (200-RPM safe cap vs the wrapper's 9000-RPM
    permissive cap) — see `alpaca_rate_limit.py` + report
    `reports/alpaca_streaming_ratelimit_2026-05-18.md`.

    Raises RuntimeError if MODE != 'LIVE_PAPER' (caller bug — SIMULATED paths
    should never reach this).
    """
    global _WRAPPER_MANAGERS
    if _WRAPPER_MANAGERS is not None:
        return _WRAPPER_MANAGERS
    if MODE != "LIVE_PAPER" or not _API_KEY or not _SECRET_KEY:
        raise RuntimeError(
            "_alpaca_managers() called outside LIVE_PAPER mode (no credentials). "
            "SIMULATED paths must not reach this."
        )
    # Lazy-import the wrapper here so SIMULATED mode never triggers it.
    from alpaca_system import (  # type: ignore[import-not-found]
        AccountManager,
        AlpacaClient,
        AlpacaCredentials,
        Guardrails,
        MarketDataManager,
        OrderManager,
        load_config,
    )
    config = load_config(environment="paper")
    creds = AlpacaCredentials(api_key=_API_KEY, secret_key=_SECRET_KEY)
    client = AlpacaClient(config, creds)
    guardrails = Guardrails(config)
    om = OrderManager(client, guardrails)
    am = AccountManager(client)
    md = MarketDataManager(client)
    _WRAPPER_MANAGERS = (om, am, md)
    log.info(
        "Gabriel alpaca-system wrapper initialized "
        f"(env={config.environment}, max_order_qty={config.max_order_qty}, "
        f"max_order_notional={config.max_order_notional:.0f})"
    )
    return _WRAPPER_MANAGERS


def _alpaca_get_latest_price(ticker: str) -> float | None:
    """Fetch the latest trade price via Gabriel wrapper, SIP feed.

    The wrapper's MarketDataManager.get_latest_trades() does not currently
    expose a feed override, so we reach through to the wrapper's shared
    StockHistoricalDataClient (the same client the Manager uses internally
    via `client.market_data`) and explicitly request SIP. This keeps the
    "go through the wrapper for auth + retry" convention while preserving
    SIP entitlement coverage (~16-exchange NBBO vs IEX free tier ~2-3%
    volume). Fails loud if account loses SIP entitlement instead of silent
    IEX degradation.

    Routed through the wrapper's request_with_retry for backoff + 429
    handling, plus _rl_call for the workspace token-bucket pre-throttle.
    """
    try:
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockLatestTradeRequest
        _om, _am, _md = _alpaca_managers()
        # Pull the wrapper's underlying client (same AlpacaClient instance
        # the Manager already uses) — avoids constructing a second SDK
        # client and re-doing auth.
        wrapper_client = _md._client  # type: ignore[attr-defined]
        sdk_data_client = wrapper_client.market_data
        req = StockLatestTradeRequest(symbol_or_symbols=[ticker], feed=DataFeed.SIP)
        trades = _rl_call(
            wrapper_client.request_with_retry,
            sdk_data_client.get_stock_latest_trade,
            req,
        )
        trade = trades.get(ticker.upper()) or trades.get(ticker)
        if trade is None:
            log.warning(f"No latest trade returned for {ticker}")
            return None
        return float(trade.price)
    except Exception as e:
        log.warning(f"Could not fetch Alpaca price for {ticker}: {e}")
        return None


def _make_client_order_id(ticker: str, side: str) -> str:
    """
    Generate a deterministic-prefix client_order_id for idempotent submission.

    Format: pt_<YYYYMMDD>_<TICKER>_<SIDE>_<UUID4-prefix>
    Length cap: 48 chars (Alpaca limit). Truncate UUID hex to fit.

    Network blips can cause submit retries. Setting client_order_id makes
    Alpaca reject duplicates with HTTP 422 instead of double-submitting.

    BUG B fix (2026-05-18, reports/alpaca_best_practices_internet_2026-05-18.md).
    """
    today_compact = date.today().strftime("%Y%m%d")
    short_uuid = uuid.uuid4().hex[:8]
    coid = f"pt_{today_compact}_{ticker}_{side.lower()}_{short_uuid}"
    return coid[:48]  # Alpaca max client_order_id length


def _place_market_order_alpaca(
    ticker: str,
    qty: int | None = None,
    side: str = "buy",
    notional: float | None = None,
) -> dict:
    """Place a market order via Gabriel wrapper's OrderManager. Returns order dict.

    Pass `notional=$X` for fractional-share dollar-amount orders (preferred
    for new buys — supports fractional fills, exactly hits the $500/ticker
    cap). Pass `qty=N` for integer-share orders (required for sells of
    fractional positions that opened with notional).

    Sets client_order_id for idempotent submission (BUG B fix). The wrapper
    handles its own idempotency retry via get_order_by_client_id on
    RetryExhaustedError; we add an outer guard so any 422 / duplicate /
    OrderError surface returns a synthetic dict instead of crashing the
    caller (preserves pre-migration behavior).

    Rate-limited at TWO layers:
      - workspace `_rl_call` token-bucket (200-RPM safe cap, 429-aware)
      - wrapper's internal Guardrails token-bucket + request_with_retry
    Both stack: the workspace layer fires first as pre-throttle.
    """
    try:
        from alpaca_system.errors import (  # type: ignore[import-not-found]
            GuardrailError,
            OrderError,
        )
    except Exception:
        # Defensive — wrapper import shouldn't fail here in LIVE_PAPER mode,
        # but if it does we want a plain Exception class to except below.
        class GuardrailError(Exception):  # type: ignore[no-redef]
            pass

        class OrderError(Exception):  # type: ignore[no-redef]
            pass

    om, _am, _md = _alpaca_managers()
    coid = _make_client_order_id(ticker, side)

    # Build kwargs for OrderManager.place_order — exactly ONE of qty/notional.
    place_kwargs: dict[str, Any] = {
        "symbol": ticker,
        "side": side,
        "order_type": "market",
        "time_in_force": "day",
        "client_order_id": coid,
    }
    if notional is not None:
        place_kwargs["notional"] = round(float(notional), 2)
    else:
        place_kwargs["qty"] = qty

    try:
        order = _rl_call(om.place_order, **place_kwargs)
    except (OrderError, GuardrailError) as e:
        # Wrapper raises OrderError for 400/403/422 (incl. duplicate-COID 422)
        # and GuardrailError for local pre-flight failures (sub-penny price,
        # qty > 1000, notional > $100k, etc.). For duplicate-COID specifically,
        # preserve the existing "idempotent skip" behavior so a network retry
        # at the workspace layer doesn't crash. Any other GuardrailError is a
        # real misconfiguration and should propagate.
        err_str = str(e)
        if (
            "422" in err_str
            or "duplicate" in err_str.lower()
            or "client_order_id" in err_str.lower()
        ):
            log.warning(
                f"Alpaca dup client_order_id for {ticker} ({coid}) — "
                "treating as idempotent skip (order already submitted)."
            )
            return {
                "order_id": "",
                "client_order_id": coid,
                "status": "duplicate_skipped",
                "qty": qty,
                "notional": notional,
            }
        raise
    except Exception as e:
        # Fallback: legacy SDK paths can still bubble a raw APIError that
        # doesn't get wrapped (e.g. through request_with_retry on a 401).
        # Mirror old behavior — log + treat duplicate-string as idempotent.
        err_str = str(e)
        if "422" in err_str or "duplicate" in err_str.lower():
            log.warning(
                f"Alpaca dup client_order_id for {ticker} ({coid}) — "
                "treating as idempotent skip (order already submitted)."
            )
            return {
                "order_id": "",
                "client_order_id": coid,
                "status": "duplicate_skipped",
                "qty": qty,
                "notional": notional,
            }
        raise

    size_repr = f"${notional:.2f}" if notional is not None else f"qty={qty}"
    log.info(
        f"Alpaca order placed: {side.upper()} {ticker} {size_repr} → "
        f"id={order.id} coid={coid}"
    )
    return {
        "order_id": str(order.id),
        "client_order_id": coid,
        "status": str(order.status),
        "qty": qty,
        "notional": notional,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Smart limit-order wrapper (a6ead433 #1: retail-fill bps lift)
# Wraps `_place_market_order_alpaca` so existing callsites can opt in to the
# smart-limit router without changing their call signature. When the router
# isn't available or LIMIT_ORDER_ROUTER!=1, falls through to the market path.
# ─────────────────────────────────────────────────────────────────────────────
def _alpaca_get_quote(ticker: str) -> dict[str, float] | None:
    """Fetch latest NBBO bid/ask via Gabriel wrapper's MarketDataManager."""
    try:
        _om, _am, md = _alpaca_managers()
        for attr in ("get_latest_quote", "get_quote"):
            fn = getattr(md, attr, None)
            if callable(fn):
                q = _rl_call(fn, ticker)
                bid = (
                    getattr(q, "bid_price", None)
                    or getattr(q, "bid", None)
                    or (q.get("bid_price") if isinstance(q, dict) else None)
                    or (q.get("bid") if isinstance(q, dict) else None)
                )
                ask = (
                    getattr(q, "ask_price", None)
                    or getattr(q, "ask", None)
                    or (q.get("ask_price") if isinstance(q, dict) else None)
                    or (q.get("ask") if isinstance(q, dict) else None)
                )
                if bid and ask:
                    return {"bid": float(bid), "ask": float(ask)}
        return None
    except Exception as e:  # noqa: BLE001
        log.debug(f"_alpaca_get_quote {ticker} failed: {e}")
        return None


def _alpaca_place_limit_order(
    *, ticker: str, side: str, qty: int, limit_price: float, tif: str = "day",
) -> dict[str, Any]:
    """Place a LIMIT order via Gabriel wrapper. Mirrors _place_market_order_alpaca."""
    om, _am, _md = _alpaca_managers()
    coid = _make_client_order_id(ticker, side)
    place_kwargs: dict[str, Any] = {
        "symbol": ticker,
        "side": side,
        "order_type": "limit",
        "time_in_force": tif,
        "limit_price": round(float(limit_price), 2),
        "qty": qty,
        "client_order_id": coid,
    }
    order = _rl_call(om.place_order, **place_kwargs)
    return {
        "order_id": str(getattr(order, "id", "")),
        "client_order_id": coid,
        "status": str(getattr(order, "status", "")),
        "qty": qty,
        "limit_price": place_kwargs["limit_price"],
    }


def _alpaca_cancel_order(order_id: str) -> None:
    """Cancel an open order via Gabriel wrapper."""
    if not order_id:
        return
    om, _am, _md = _alpaca_managers()
    fn = getattr(om, "cancel_order", None) or getattr(om, "cancel_order_by_id", None)
    if callable(fn):
        try:
            _rl_call(fn, order_id)
        except Exception as e:  # noqa: BLE001
            log.debug(f"cancel_order {order_id} ignored: {e}")


def _alpaca_get_order(order_id: str) -> dict[str, Any]:
    """Poll order status/fill via Gabriel wrapper."""
    if not order_id:
        return {"status": "unknown", "filled_qty": 0}
    om, _am, _md = _alpaca_managers()
    fn = getattr(om, "get_order", None) or getattr(om, "get_order_by_id", None)
    if not callable(fn):
        return {"status": "unknown", "filled_qty": 0}
    try:
        o = _rl_call(fn, order_id)
        return {
            "status": str(getattr(o, "status", "")),
            "filled_qty": getattr(o, "filled_qty", 0) or 0,
            "filled_avg_price": getattr(o, "filled_avg_price", None),
        }
    except Exception as e:  # noqa: BLE001
        log.debug(f"get_order {order_id} err: {e}")
        return {"status": "unknown", "filled_qty": 0}


def _place_smart_order_alpaca(
    ticker: str,
    qty: int | None = None,
    side: str = "buy",
    notional: float | None = None,
    *,
    signal_strength: Any = None,
    prob: float | None = None,
) -> dict:
    """Route through smart-limit router if enabled; else market order (legacy)."""
    if not (_LIMIT_ROUTER_ENABLED and _LIMIT_ROUTER_AVAILABLE and _route_limit_order is not None):
        return _place_market_order_alpaca(ticker, qty=qty, side=side, notional=notional)
    try:
        return _route_limit_order(  # type: ignore[misc]
            ticker, side, qty=qty, notional=notional,
            signal_strength=signal_strength, prob=prob,
            get_quote_fn=_alpaca_get_quote,
            place_limit_fn=_alpaca_place_limit_order,
            cancel_fn=_alpaca_cancel_order,
            get_order_fn=_alpaca_get_order,
            place_market_fn=lambda t, side, qty=None, notional=None: _place_market_order_alpaca(
                t, qty=qty, side=side, notional=notional,
            ),
        )
    except Exception as e:  # noqa: BLE001
        log.warning(f"smart-router error for {ticker}: {e} — falling back to market")
        return _place_market_order_alpaca(ticker, qty=qty, side=side, notional=notional)


def _get_alpaca_positions() -> dict[str, dict]:
    """Return current open positions via Gabriel wrapper's AccountManager.

    Pos qty is parsed as float so fractional-share positions (from notional
    orders) deserialize cleanly without ValueError.
    """
    positions: dict[str, dict] = {}
    try:
        _om, am, _md = _alpaca_managers()
        for pos in _rl_call(am.get_positions):
            positions[pos.symbol] = {
                "qty": float(pos.qty),
                "market_value": float(pos.market_value),
                "avg_entry_price": float(pos.avg_entry_price),
            }
    except Exception as e:
        log.error(f"Could not fetch Alpaca positions: {e}")
    return positions


def _alpaca_close_all_positions() -> list[dict]:
    """Close all open positions via Gabriel wrapper's AccountManager.

    Returns list of close order results; each `o` is an alpaca-py
    ClosePositionResponse with `.symbol` and `.status` (HTTP code).
    """
    results: list[dict] = []
    try:
        _om, am, _md = _alpaca_managers()
        orders = _rl_call(am.close_all_positions, cancel_orders=True)
        for o in orders:
            results.append({
                "ticker": getattr(o, "symbol", "?"),
                "status": str(getattr(o, "status", "unknown")),
            })
            log.info(f"Alpaca flatten: SELL {getattr(o, 'symbol','?')}")
    except Exception as e:
        log.error(f"Alpaca close_all_positions failed: {e}")
    return results


def _get_alpaca_account() -> dict:
    """Fetch account equity, cash, P&L, and last-equity via wrapper AccountManager.

    `last_equity` is the equity at yesterday's close — Alpaca's authoritative
    baseline for today's intraday P&L. (equity - last_equity) = realized + unrealized
    P&L for the current trading day. For an intraday-only strategy (no overnight
    holds) this is effectively realized P&L during the flat-at-close window.
    """
    try:
        _om, am, _md = _alpaca_managers()
        acct = _rl_call(am.get_account)
        return {
            "equity": float(acct.equity),
            "cash": float(acct.cash),
            "portfolio_value": float(acct.portfolio_value),
            "last_equity": float(getattr(acct, "last_equity", acct.equity)),
        }
    except Exception as e:
        log.error(f"Could not fetch Alpaca account: {e}")
        return {}


def _refresh_realized_pnl_from_alpaca(state: dict) -> float:
    """
    BUG A fix (2026-05-18): pull authoritative day-P&L from Alpaca account API.

    In LIVE_PAPER mode, state["realized_pnl"] previously only updated in
    SIMULATED mode (see cmd_flatten), so daily-loss halts NEVER triggered.
    Account could lose unbounded $$ until the 15:55 ET force-flat.

    Strategy: compute current_pnl = equity - last_equity (Alpaca's previous-close
    baseline). For an intraday-flat strategy this approximates realized P&L
    once positions are closed; during the day it includes unrealized too, which
    is the SAFE direction for halts (treat MTM drawdown as worth halting).

    Returns the refreshed value and updates state["realized_pnl"] in-place.
    Falls back to existing state value on API error (fail-safe: existing halt
    logic continues to run on whatever value we have).
    """
    if MODE != "LIVE_PAPER":
        return state.get("realized_pnl", 0.0)
    acct = _get_alpaca_account()
    if not acct:
        log.warning(
            "Could not refresh realized_pnl from Alpaca — using cached "
            f"value ${state.get('realized_pnl', 0.0):.2f}"
        )
        return state.get("realized_pnl", 0.0)
    equity = acct.get("equity")
    last_equity = acct.get("last_equity")
    if equity is None or last_equity is None or last_equity <= 0:
        log.warning(
            "Alpaca account missing equity/last_equity — using cached "
            f"realized_pnl ${state.get('realized_pnl', 0.0):.2f}"
        )
        return state.get("realized_pnl", 0.0)
    day_pnl = equity - last_equity
    prev_cached = state.get("realized_pnl", 0.0)
    state["realized_pnl"] = day_pnl
    state["alpaca_equity"] = equity
    state["alpaca_last_equity"] = last_equity
    log.info(
        f"[LIVE_PAPER] Refreshed day P&L from Alpaca: "
        f"equity=${equity:.2f} last_equity=${last_equity:.2f} "
        f"day_pnl=${day_pnl:.2f} (was ${prev_cached:.2f})"
    )
    return day_pnl


# ---------------------------------------------------------------------------
# SIMULATED: yfinance helpers
# ---------------------------------------------------------------------------
def _sim_get_latest_price(ticker: str) -> float | None:
    """Fetch latest close from yfinance for simulated mode."""
    try:
        df = yf.download(ticker, period="2d", interval="1d", progress=False, auto_adjust=True)
        if df.empty:
            return None
        return float(df["Close"].iloc[-1])
    except Exception as e:
        log.warning(f"yfinance price fetch failed for {ticker}: {e}")
        return None


# ---------------------------------------------------------------------------
# Persistence-collector wiring helpers (audit gap: feed every detail into XGB/Mythos)
# ---------------------------------------------------------------------------
def _run_persist_collector(script_name: str, extra_args: list[str], timeout: int = 60) -> None:
    """Shell out to a scripts/persist_*.py collector via subprocess.

    Best-effort: never raises into the caller. Logs rc + tail of stderr on
    non-zero. Skips silently (with a single info log) if the target script
    is missing so an older deploy without the collector keeps running.
    """
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        log.info(f"[wiring] {script_name} not found — skipping (back-compat)")
        return
    try:
        r = subprocess.run(
            [sys.executable, str(script_path), *extra_args],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        log.info(f"[wiring] {script_name} called → rc={r.returncode}")
        if r.stderr:
            tail = r.stderr.strip()[-500:]
            if tail:
                if r.returncode != 0:
                    log.warning(f"[wiring] {script_name} stderr tail: {tail}")
                else:
                    log.debug(f"[wiring] {script_name} stderr tail: {tail}")
    except subprocess.TimeoutExpired:
        log.warning(f"[wiring] {script_name} timed out after {timeout}s")
    except Exception as e:
        log.warning(f"[wiring] {script_name} raised: {e}")


# ---------------------------------------------------------------------------
# SUBCOMMAND: startup (09:00 ET)
# ---------------------------------------------------------------------------
def cmd_startup(args) -> int:
    """09:00 ET — load models, fetch overnight data, regenerate signals."""
    today = getattr(args, "date", None) or date.today().isoformat()
    dry_run = bool(getattr(args, "dry_run", False))
    log.info(f"=== STARTUP {today} | mode={MODE} | dry_run={dry_run} ===")

    if dry_run:
        log.info(
            "[DRY-RUN] startup: would load_state, refresh halt flags, run "
            f"live_paper_trade_signals.py, and write state to "
            f"{STATE_DIR / (today + '_state.json')}. Skipping all of the above."
        )
        return 0

    state = load_state(today)
    state["mode"] = MODE

    # 2026-05-18 fix: auto-reset stale halt flags. If state was loaded from a
    # prior-day file (load_state reads STATE_DIR/<today>_state.json so this is
    # only relevant when the same date file persists across an aborted day),
    # OR the halted_date attribute predates today, clear the halt so a new
    # session can start trading. Without this, a one-off halt persists forever.
    if state.get("halted"):
        halted_date = state.get("halted_date")
        if not halted_date or halted_date < today:
            log.warning(
                f"[HALT/RESET] Stale halt cleared "
                f"(halted_date={halted_date!r}, today={today}). Resuming trading."
            )
            state["halted"] = False
            state["soft_halted"] = False
            state.pop("halted_date", None)
        else:
            log.warning(
                f"[HALT/HONOR] Halt is current (halted_date={halted_date}, today={today}). "
                "Skipping signal regen — no trades today."
            )
    save_state(state)

    # Run signal generation script
    signal_script = SCRIPTS_DIR / "live_paper_trade_signals.py"
    if not signal_script.exists():
        log.error(f"Signal script missing: {signal_script}")
        return 1

    log.info("Launching signal generation...")
    result = subprocess.run(
        [sys.executable, str(signal_script)],
        capture_output=False,
        timeout=1800,  # 30min — Mythos SIP fallback fan-out across 359 tickers can hit 10-15min
    )
    if result.returncode != 0:
        log.error(f"Signal generation failed with exit code {result.returncode}")
        log.warning("Falling back to NO TRADES TODAY mode.")
        state["halted"] = True
        state["halted_date"] = today
        state["halted_reason"] = f"signal_gen_failed_rc={result.returncode}"
        save_state(state)
        return 1

    # Verify signals written
    signals_path = SIGNALS_DIR / f"{today}.json"
    if signals_path.exists():
        sigs = json.loads(signals_path.read_text())
        firing = [s for s in sigs if s.get("signal") == 1]
        log.info(f"Startup complete. {len(firing)} signals firing today out of {len(sigs)} tickers.")
        if _EB:
            try:
                _EB.publish_from_anywhere("paper_signal_generated", {
                    "date": today,
                    "firing_count": len(firing),
                    "total_tickers": len(sigs),
                    "tickers": [s["ticker"] for s in firing][:20],
                    "mode": MODE,
                }, source="live_paper_trade")
            except Exception:
                pass
    else:
        log.warning("No signals file produced — NO TRADES TODAY.")
        state["halted"] = True
        state["halted_date"] = today
        state["halted_reason"] = "no_signals_file"
        save_state(state)

    # Wiring: account snapshot at startup phase (audit-gap collector — feeds
    # XGBoost+Mythos training rows). Best-effort, never blocks startup.
    _run_persist_collector(
        "persist_account_snapshots.py",
        ["--phase", "startup"],
        timeout=30,
    )

    return 0


# ---------------------------------------------------------------------------
# SUBCOMMAND: open-trades (09:30 ET)
# ---------------------------------------------------------------------------
def cmd_open_trades(args) -> int:
    """09:30 ET — place paper-trade orders for all firing signals."""
    today = getattr(args, "date", None) or date.today().isoformat()
    dry_run = bool(getattr(args, "dry_run", False))
    log.info(f"=== OPEN-TRADES {today} | mode={MODE} | dry_run={dry_run} ===")
    # Market-day guard: skip on weekends + NYSE holidays (dry-run still proceeds for smoke tests).
    if not dry_run and not _market_day_guard("open-trades"):
        return 0
    _assert_market_window("open-trades", 9, 28, 9, 45)

    state = load_state(today)

    if state.get("halted"):
        log.warning("State is HALTED — no trades today (daily HARD halt or startup failure).")
        return 0

    # BUG A fix (2026-05-18): refresh realized_pnl from Alpaca BEFORE halt check.
    # In LIVE_PAPER mode this pulls equity - last_equity (authoritative day P&L);
    # in SIMULATED mode it's a no-op so the cached state value flows through.
    # Skipped under dry-run: get_account is a live API call we should not make.
    if not dry_run:
        _refresh_realized_pnl_from_alpaca(state)
    else:
        log.info(
            "[DRY-RUN] skipping _refresh_realized_pnl_from_alpaca — would call "
            "AccountManager.get_account() to read equity/last_equity."
        )

    # Two-tier daily-loss halt (2026-05-18, see module docstring).
    realized = state.get("realized_pnl", 0.0)
    if realized <= DAILY_LOSS_HARD_HALT:
        log.warning(
            f"[HALT/HARD] Daily HARD halt reached "
            f"(realized_pnl=${realized:.2f} <= ${DAILY_LOSS_HARD_HALT:.0f}) — "
            "halting day. No new entries; flatten at close."
        )
        state["halted"] = True
        state["halted_date"] = today
        state["halted_reason"] = (
            f"hard_halt_realized_pnl={realized:.2f}<=" f"{DAILY_LOSS_HARD_HALT:.0f}"
        )
        save_state(state)
        return 0
    if realized <= DAILY_LOSS_SOFT_HALT:
        log.warning(
            f"[HALT/SOFT] Daily SOFT halt reached "
            f"(realized_pnl=${realized:.2f} <= ${DAILY_LOSS_SOFT_HALT:.0f}) — "
            "exit-only mode. Blocking NEW entries; existing positions managed/flattened normally."
        )
        state["soft_halted"] = True
        state["halted_date"] = today
        state["halted_reason"] = (
            f"soft_halt_realized_pnl={realized:.2f}<=" f"{DAILY_LOSS_SOFT_HALT:.0f}"
        )
        save_state(state)
        return 0

    signals = load_today_signals(today)
    if not signals:
        log.warning("No signals available — skipping open-trades.")
        return 0

    # ----- Signal-decay exit (Quick-win B, 2026-05-22) -----
    # Per ac0f5aca: re-check current signal probs for already-held positions.
    # If entry signal_prob has decayed by >SIGNAL_DECAY_EXIT_THRESHOLD vs today's
    # signal, sell BEFORE opening new trades so the freed slot/cash recycles
    # into the new batch. Held tickers absent from today's signal file are NOT
    # decayed (no current prob to compare against); they wait for STOP/TP/EOD.
    SIGNAL_DECAY_EXIT_THRESHOLD = 0.20
    current_signals_by_ticker = {
        s["ticker"]: s for s in signals if isinstance(s, dict) and s.get("ticker")
    }
    decay_exits: list[str] = []
    for ticker, pos in list(state["positions"].items()):
        entry_prob = pos.get("signal_prob")
        curr_sig = current_signals_by_ticker.get(ticker)
        if entry_prob is None or curr_sig is None:
            continue
        curr_prob = curr_sig.get("prob")
        if curr_prob is None:
            continue
        try:
            decay = float(entry_prob) - float(curr_prob)
        except (TypeError, ValueError):
            continue
        if decay > SIGNAL_DECAY_EXIT_THRESHOLD:
            decay_exits.append(ticker)
            log.info(
                f"[DECAY-EXIT] {ticker}: entry_prob={entry_prob:.3f} "
                f"curr_prob={curr_prob:.3f} decay={decay:.3f} > "
                f"{SIGNAL_DECAY_EXIT_THRESHOLD:.2f} — queuing sell"
            )
    for t in decay_exits:
        pos = state["positions"][t]
        qty = int(pos.get("qty") or 0)
        if qty <= 0:
            log.warning(f"[DECAY-EXIT] {t}: qty<=0, skipping sell")
            continue
        if dry_run:
            log.info(f"[DRY-RUN][DECAY-EXIT] would sell {qty} {t} (signal_decay)")
        elif MODE == "LIVE_PAPER":
            try:
                _place_smart_order_alpaca(t, qty=qty, side="sell")
            except Exception as _de_err:  # noqa: BLE001
                log.warning(f"[DECAY-EXIT] {t}: sell failed: {_de_err} — skipping")
                continue
        else:
            log.info(f"[SIM][DECAY-EXIT] SELL {qty} {t} (signal_decay)")
        closed = state["positions"].pop(t)
        closed["closed_reason"] = "signal_decay"
        closed["closed_at"] = datetime.now(timezone.utc).isoformat()
        state["closed_trades"].append(closed)
    if decay_exits:
        log.info(
            f"[DECAY-EXIT] freed {len(decay_exits)} slot(s) via signal_decay: "
            f"{decay_exits}"
        )
    # ----- end signal-decay exit -----

    current_exposure = sum(
        p.get("notional", 0) for p in state["positions"].values()
    )
    sized = compute_position_sizes(signals, current_exposure)

    # ----- Quick-win C: sell-weakest-to-fund-strongest swap engine -----
    # (2026-05-22) When exposure is saturated and fresh signals are firing that
    # have NO room to enter, rescore held positions vs the new firing batch and
    # propose up to N flips per cycle (challenger margin > weakest + 0.10 prob).
    # Activation: env POSITION_MANAGER_ENABLED=1 (default OFF for A/B safety).
    # Honors a warmup-grace window after cold restart so we don't sell-storm.
    swap_proposals_executed: list[dict] = []
    if os.environ.get("POSITION_MANAGER_ENABLED", "0") == "1":
        firing_now = [s for s in signals if isinstance(s, dict) and s.get("signal") == 1]
        available_headroom = max(0.0, MAX_TOTAL_EXPOSURE - current_exposure)
        # Trigger swaps only when: (a) firing signals exist, (b) no headroom for
        # them all, (c) we have positions to sell. compute_position_sizes
        # returned `sized` with the available headroom split; if some firing
        # signals were dropped (sized < firing_now), there's swap opportunity.
        sized_tickers = {s.get("ticker") for s in sized}
        firing_dropped = [s for s in firing_now if s.get("ticker") not in sized_tickers]
        if firing_dropped and state.get("positions"):
            try:
                from position_manager import PositionManager  # type: ignore[import-not-found]
                mgr = PositionManager()
                if mgr.in_warmup(state):
                    log.info(
                        "[SWAP] in warmup window (last_restart_at fresh < %ds) — "
                        "skipping swap engine this cycle", mgr.warmup_s,
                    )
                else:
                    holdings_scored = mgr.rescore_holdings(state, signals)
                    # Lazy risk-engine instantiation for swap-IN gating (independent
                    # of the per-order loop's risk_engine below — both will read
                    # the same state).
                    _swap_re = None
                    try:
                        from risk_engine import RiskEngine as _RE  # type: ignore
                        _swap_re = _RE(
                            equity=LIVE_BUDGET_USD,
                            positions=state.get("positions", {}),
                        )
                    except Exception:
                        _swap_re = None
                    proposals = mgr.find_swaps(holdings_scored, firing_dropped, _swap_re)
                    if proposals:
                        log.info(
                            "[SWAP] proposing %d flip(s) — max_per_cycle=%d",
                            len(proposals), mgr.max_swaps_per_cycle,
                        )
                    for prop in proposals[: mgr.max_swaps_per_cycle]:
                        log.info(
                            "[SWAP] %s (eff=%.3f) ⇨ %s (in=%.3f, lift=%.3f) | %s",
                            prop.out_ticker, prop.out_score,
                            prop.in_ticker, prop.in_score, prop.expected_lift,
                            prop.reason,
                        )
                        if dry_run:
                            log.info(
                                "[DRY-RUN][SWAP] would route_flip(out=%s qty=%s "
                                "in=%s notional=$%.2f strength=%s prob=%s)",
                                prop.out_ticker, prop.out_qty,
                                prop.in_ticker, prop.out_notional,
                                prop.in_signal_strength, prop.in_signal_prob,
                            )
                            swap_proposals_executed.append({
                                "out_ticker": prop.out_ticker,
                                "in_ticker": prop.in_ticker,
                                "expected_lift": prop.expected_lift,
                                "reason": prop.reason,
                                "executed": False,
                                "dry_run": True,
                            })
                            continue
                        # LIVE_PAPER / SIMULATED: execute via router.route_flip.
                        try:
                            from limit_order_router import route_flip  # type: ignore
                            flip_res = route_flip(
                                out_ticker=prop.out_ticker,
                                out_qty=prop.out_qty,
                                in_ticker=prop.in_ticker,
                                in_notional=prop.out_notional,
                                in_signal_strength=prop.in_signal_strength,
                                in_signal_prob=prop.in_signal_prob,
                                get_quote_fn=_alpaca_get_quote,
                                place_limit_fn=_alpaca_place_limit_order,
                                cancel_fn=_alpaca_cancel_order,
                                get_order_fn=_alpaca_get_order,
                                place_market_fn=_place_market_order_alpaca,
                            )
                        except Exception as _flip_err:
                            log.error("[SWAP] route_flip failed: %s", _flip_err)
                            flip_res = {"status": "error", "error": str(_flip_err)}
                        if flip_res.get("status") == "complete":
                            # Update state: remove OUT, add IN. Tag reasons for audit.
                            outgoing = state["positions"].pop(prop.out_ticker, {})
                            outgoing["closed_reason"] = "swap_out"
                            outgoing["closed_at"] = datetime.now(timezone.utc).isoformat()
                            outgoing["flip_id"] = flip_res.get("flip_id")
                            state.setdefault("closed_today", []).append(
                                {prop.out_ticker: outgoing}
                            )
                            in_fill = (flip_res.get("in") or {})
                            state["positions"][prop.in_ticker] = {
                                "qty": in_fill.get("qty") or 0,
                                "entry_price": in_fill.get("fill_price"),
                                "order_id": in_fill.get("order_id", ""),
                                "client_order_id": in_fill.get("client_order_id", ""),
                                "notional": (
                                    in_fill.get("notional") or prop.out_notional
                                ),
                                "signal_prob": prop.in_signal_prob,
                                "threshold": prop.in_signal.get("threshold"),
                                "opened_at": datetime.now(timezone.utc).isoformat(),
                                "entered_reason": "swap_in",
                                "flip_id": flip_res.get("flip_id"),
                                "pipeline": prop.in_signal.get("pipeline"),
                                "model_run_dir": prop.in_signal.get("model_run_dir"),
                                "feature_hash": prop.in_signal.get("feature_hash"),
                                "features_used": prop.in_signal.get("features_used"),
                            }
                            swap_proposals_executed.append({
                                "out_ticker": prop.out_ticker,
                                "in_ticker": prop.in_ticker,
                                "expected_lift": prop.expected_lift,
                                "reason": prop.reason,
                                "flip_id": flip_res.get("flip_id"),
                                "executed": True,
                            })
                        else:
                            log.warning(
                                "[SWAP] flip not complete (status=%s) — "
                                "state NOT mutated", flip_res.get("status"),
                            )
                            swap_proposals_executed.append({
                                "out_ticker": prop.out_ticker,
                                "in_ticker": prop.in_ticker,
                                "status": flip_res.get("status"),
                                "executed": False,
                            })
                    if swap_proposals_executed and not dry_run:
                        # Recompute exposure + re-size remaining firing signals.
                        # IN-tickers from swaps are now held; remove them from
                        # `sized` so the normal-entry loop doesn't double-buy.
                        in_tickers_swapped = {
                            x["in_ticker"] for x in swap_proposals_executed
                            if x.get("executed")
                        }
                        sized = [s for s in sized if s.get("ticker") not in in_tickers_swapped]
                        current_exposure = sum(
                            p.get("notional", 0) for p in state["positions"].values()
                        )
            except Exception as _swap_err:
                log.warning("[SWAP] engine raised, skipped: %s", _swap_err)
    # ----- end Quick-win C swap engine -----

    if not sized and not swap_proposals_executed:
        log.info("No positions to open (all filtered by guardrails).")
        return 0
    if not sized:
        log.info(
            "Sized list empty after swaps; %d swap(s) executed this cycle.",
            len(swap_proposals_executed),
        )

    # Wiring: pre-open quotes snapshot for all about-to-be-traded tickers.
    # Captures NBBO + last trade context just before order submission so the
    # XGB/Mythos training row can reconstruct the entry-side micro-environment.
    # Skipped in dry-run (no live data calls).
    if not dry_run:
        pre_open_tickers = sorted({s["ticker"] for s in sized})
        if pre_open_tickers:
            _run_persist_collector(
                "persist_quotes_snapshot.py",
                [
                    "--tickers", ",".join(pre_open_tickers),
                    "--phase", "pre_open",
                ],
                timeout=60,
            )

    orders_placed = []

    # ----- 5-gate pre-trade risk engine (Kelly / Liquidity / Correlation /
    # Concentration / Drawdown). Instantiate once with current equity +
    # positions so every per-ticker check sees consistent state. Refusals
    # are logged to paper_trade/risk_engine_decisions.jsonl. Greedy approval:
    # tickers approved earlier in this batch participate in correlation
    # checks for later candidates. -----
    try:
        from risk_engine import RiskEngine  # type: ignore[import-not-found]
        # SYNTHETIC BUDGET (2026-05-22): use LIVE_BUDGET_USD, NOT Alpaca account
        # equity. Alpaca paper account equity (~$95k) would inflate Kelly cap
        # (5% × $95k = $4,750 vs intended 5% × $2k = $100) and concentration
        # cap (5% × $95k = $4,750 vs $100). Real-equity is still tracked in
        # state['equity'] for reporting / drawdown timeline.
        _equity_for_risk = LIVE_BUDGET_USD
        _alpaca_equity_for_report = float(
            state.get("equity") or state.get("portfolio_value") or 0.0
        )
        risk_engine = RiskEngine(
            equity=_equity_for_risk,
            positions=state.get("positions", {}),
        )
        log.info(
            f"[RISK] engine initialized synthetic_budget=${_equity_for_risk:.0f} "
            f"(alpaca_equity=${_alpaca_equity_for_report:.0f}, ignored for sizing)"
        )
    except Exception as _re_err:
        log.warning(f"[RISK] engine init failed — gates DISABLED: {_re_err}")
        risk_engine = None

    for _i, sig in enumerate(sized):
        # 1s stagger reduces 0.0930 mass-fill latency cost per helper finding
        # (reports/alpaca_risk_sizing_2026-05-18.md §"Stagger the 9:30 AM batch").
        # Sleep BEFORE every iteration after the first so 18 orders spread over ~18s.
        # Skip the sleep in dry-run so smoke tests finish fast (no orders to pace).
        if _i > 0 and BATCH_OPEN_STAGGER_S > 0 and not dry_run:
            time.sleep(BATCH_OPEN_STAGGER_S)
        ticker = sig["ticker"]
        notional = sig["notional"]

        # Get price for qty calculation. In dry-run, skip BOTH live data paths
        # (Alpaca SIP + yfinance — both are external network calls) and use a
        # placeholder price so the loop still exercises sizing/logging logic.
        if dry_run:
            price = float(sig.get("price") or sig.get("ref_price") or 100.0)
            log.info(
                f"[DRY-RUN] using placeholder price for {ticker}: ${price:.2f} "
                "(no live data fetched)"
            )
        elif MODE == "LIVE_PAPER":
            price = _alpaca_get_latest_price(ticker)
        else:
            price = _sim_get_latest_price(ticker)

        if not price or price <= 0:
            log.warning(f"Skipping {ticker}: could not get price.")
            continue

        qty = max(1, int(notional / price))
        actual_notional = qty * price

        if actual_notional > MAX_POSITION_NOTIONAL * 1.05:
            log.warning(f"Skipping {ticker}: notional ${actual_notional:.0f} exceeds cap.")
            continue

        # ----- 5-gate pre-trade risk check -----
        if risk_engine is not None:
            decision = risk_engine.check(
                ticker=ticker, qty=qty, signal=sig, price=price
            )
            if not decision.passed:
                log.warning(
                    f"[RISK] {ticker} REFUSED by gate={decision.gate}: "
                    f"{decision.reason}"
                )
                continue
            if decision.adjusted_qty is not None and decision.adjusted_qty != qty:
                log.info(
                    f"[RISK] {ticker} DOWNSIZED by {decision.gate}: "
                    f"qty {qty}→{decision.adjusted_qty} ({decision.reason})"
                )
                qty = decision.adjusted_qty
                actual_notional = qty * price

        if dry_run:
            # DRY-RUN: log intended action, do NOT submit, do NOT mutate state.
            log.info(
                f"[DRY-RUN] would submit BUY {ticker} qty={qty} "
                f"@ ${price:.2f} (notional ~${actual_notional:.2f}, "
                f"prob={sig.get('prob')}, threshold={sig.get('threshold')})"
            )
        elif MODE == "LIVE_PAPER":
            # Prefer notional (fractional-share) order so we hit the per-ticker
            # $500 budget exactly — see report §9(d). Fall back to qty-mode if
            # the symbol rejects notional (sub-$1, recently-IPO'd, etc.).
            order_result = None
            order_size_notional = float(notional)
            # Signal-strength hint for the smart-limit router (a6ead433 #1).
            # Falls through to a plain market order when LIMIT_ORDER_ROUTER!=1.
            _sig_prob = sig.get("prob") if isinstance(sig, dict) else None
            _sig_strength = sig.get("signal_strength") if isinstance(sig, dict) else None
            try:
                order_result = _place_smart_order_alpaca(
                    ticker, side="buy", notional=order_size_notional,
                    signal_strength=_sig_strength, prob=_sig_prob,
                )
            except Exception as e:
                log.warning(
                    f"Notional order failed for {ticker} (${order_size_notional:.2f}): "
                    f"{e} — falling back to qty={qty}"
                )
                try:
                    order_result = _place_smart_order_alpaca(
                        ticker, qty=qty, side="buy",
                        signal_strength=_sig_strength, prob=_sig_prob,
                    )
                except Exception as e2:
                    log.error(f"qty fallback also failed for {ticker}: {e2}")
                    continue
            state["positions"][ticker] = {
                # qty is recorded as the integer-share estimate; the actual
                # filled qty (which may be fractional) lands in the WS fill log.
                "qty": qty,
                "entry_price": price,
                "order_id": order_result.get("order_id", ""),
                "client_order_id": order_result.get("client_order_id", ""),
                "notional": (
                    order_result.get("notional")
                    if order_result.get("notional") is not None
                    else actual_notional
                ),
                "signal_prob": sig.get("prob", None),
                "threshold": sig.get("threshold", None),
                "opened_at": datetime.now(timezone.utc).isoformat(),
                # Model-version stamp (audit gap #9, 2026-05-18) — enables
                # post-trade join from closed_trade rows back to the exact
                # model run + feature set that fired the signal. All fields
                # default to None if sig dict lacks them (back-compat with
                # signals generated before this stamp was added).
                "pipeline": sig.get("pipeline"),
                "model_run_dir": sig.get("model_run_dir"),
                "feature_hash": sig.get("feature_hash"),
                "features_used": sig.get("features_used"),
            }
        else:
            # SIMULATED: record synthetic position
            order_id = f"SIM-{ticker}-{today}"
            log.info(f"[SIM] BUY {qty} {ticker} @ ${price:.2f} = ${actual_notional:.2f}")
            state["positions"][ticker] = {
                "qty": qty,
                "entry_price": price,
                "order_id": order_id,
                "notional": actual_notional,
                "signal_prob": sig.get("prob", None),
                "threshold": sig.get("threshold", None),
                "opened_at": datetime.now(timezone.utc).isoformat(),
                # Model-version stamp (audit gap #9, 2026-05-18) — mirrors the
                # LIVE_PAPER branch so SIMULATED trades carry the same
                # joinability schema. All fields default to None if sig dict
                # lacks them.
                "pipeline": sig.get("pipeline"),
                "model_run_dir": sig.get("model_run_dir"),
                "feature_hash": sig.get("feature_hash"),
                "features_used": sig.get("features_used"),
            }

        orders_placed.append(ticker)

    if dry_run:
        log.info(f"[DRY-RUN] would place {len(orders_placed)} orders — {orders_placed}")
        log.info("[DRY-RUN] skipping save_state — no state mutation.")
        return 0

    log.info(f"Orders placed: {len(orders_placed)} — {orders_placed}")
    save_state(state)

    # Wiring: account snapshot AFTER opens succeed (audit-gap collector).
    # Captures equity/cash/buying-power immediately after entry submissions
    # so the training row can see the post-open account state.
    _run_persist_collector(
        "persist_account_snapshots.py",
        ["--phase", "open"],
        timeout=30,
    )

    if _EB and orders_placed:
        try:
            _EB.publish_from_anywhere("paper_position_opened", {
                "date": today,
                "tickers": orders_placed,
                "count": len(orders_placed),
                "mode": MODE,
            }, source="live_paper_trade")
        except Exception:
            pass
    return 0


# ---------------------------------------------------------------------------
# SUBCOMMAND: flatten (15:55 ET)
# ---------------------------------------------------------------------------
def cmd_flatten(args) -> int:
    """15:55 ET — close all open positions to avoid overnight risk."""
    today = getattr(args, "date", None) or date.today().isoformat()
    dry_run = bool(getattr(args, "dry_run", False))
    log.info(f"=== FLATTEN {today} | mode={MODE} | dry_run={dry_run} ===")
    # Market-day guard: skip on weekends + NYSE holidays (dry-run still proceeds for smoke tests).
    if not dry_run and not _market_day_guard("flatten"):
        return 0
    _assert_market_window("flatten", 15, 50, 16, 0)

    state = load_state(today)

    if not state["positions"] and not state.get("halted"):
        log.info("No open positions to flatten.")
        return 0

    # Wiring: pre-close quotes snapshot for all currently-open positions.
    # Captures exit-side NBBO context just before flatten so the training row
    # has both entry and exit micro-environment to learn from.
    # Skipped in dry-run (no live data calls).
    if not dry_run:
        positions_tickers = sorted(state.get("positions", {}).keys())
        if positions_tickers:
            _run_persist_collector(
                "persist_quotes_snapshot.py",
                [
                    "--tickers", ",".join(positions_tickers),
                    "--phase", "pre_close",
                ],
                timeout=60,
            )

    closed = []

    if dry_run:
        # DRY-RUN: log what we'd close, do NOT call close_all_positions,
        # do NOT mutate state.
        if MODE == "LIVE_PAPER":
            log.info(
                "[DRY-RUN] would call AccountManager.close_all_positions"
                f"(cancel_orders=True) — {len(state['positions'])} tracked positions: "
                f"{list(state['positions'].keys())}"
            )
        else:
            log.info(
                f"[DRY-RUN] (SIMULATED) would close {len(state['positions'])} "
                f"tracked positions: {list(state['positions'].keys())}"
            )
        log.info("[DRY-RUN] skipping save_state — no state mutation.")
        return 0

    if MODE == "LIVE_PAPER":
        results = _alpaca_close_all_positions()
        for r in results:
            ticker = r.get("ticker", "?")
            if ticker in state["positions"]:
                pos = state["positions"].pop(ticker)
                # exit_price=None is INTENTIONAL here — Alpaca fills async.
                # cmd_ingest reconciles the actual fill from
                # paper_trade/fills/<DATE>.jsonl (written by the WS daemon).
                # `pending_ws_fill=True` flags this record for the reconciler.
                # See report §9(a) — fixes the historical exit_price=None bug.
                state["closed_trades"].append({
                    "ticker": ticker,
                    "qty": pos.get("qty"),
                    "entry_price": pos.get("entry_price"),
                    "exit_price": None,
                    "order_id": pos.get("order_id"),
                    "client_order_id": pos.get("client_order_id", ""),
                    "closed_at": datetime.now(timezone.utc).isoformat(),
                    "mode": "LIVE_PAPER",
                    "pending_ws_fill": True,
                    # Carry model-version stamp through to closed_trade so
                    # post-trade XGBoost/Mythos training-row builder can join
                    # back to the firing model (audit gap #9, 2026-05-18).
                    "pipeline": pos.get("pipeline"),
                    "model_run_dir": pos.get("model_run_dir"),
                    "feature_hash": pos.get("feature_hash"),
                    "features_used": pos.get("features_used"),
                    "signal_prob": pos.get("signal_prob"),
                    "threshold": pos.get("threshold"),
                })
                closed.append(ticker)
        # Also clear any positions Alpaca may have known about
        for r in results:
            ticker = r.get("ticker", "?")
            if ticker not in closed:
                closed.append(ticker)

        # BUG A fix (2026-05-18): refresh realized_pnl from Alpaca post-flatten.
        # Once positions are closed, equity - last_equity is the day's realized P&L.
        # This is the value ingest.py + dashboards + the next-day halt check rely on.
        _refresh_realized_pnl_from_alpaca(state)

    else:
        # SIMULATED: use current yfinance price as exit
        tickers_to_close = list(state["positions"].keys())
        for ticker in tickers_to_close:
            pos = state["positions"].pop(ticker)
            exit_price = _sim_get_latest_price(ticker)
            entry_price = pos.get("entry_price", 0.0)
            qty = pos.get("qty", 0)

            if exit_price:
                pnl = (exit_price - entry_price) * qty
                state["realized_pnl"] = state.get("realized_pnl", 0.0) + pnl
                log.info(
                    f"[SIM] SELL {qty} {ticker} @ ${exit_price:.2f} "
                    f"(entry ${entry_price:.2f}) → P&L ${pnl:.2f}"
                )
            else:
                pnl = 0.0
                log.warning(f"[SIM] Could not get exit price for {ticker} — P&L=0")

            state["closed_trades"].append({
                "ticker": ticker,
                "qty": qty,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "closed_at": datetime.now(timezone.utc).isoformat(),
                "mode": "SIMULATED",
                # Carry model-version stamp through to closed_trade (audit gap #9).
                "pipeline": pos.get("pipeline"),
                "model_run_dir": pos.get("model_run_dir"),
                "feature_hash": pos.get("feature_hash"),
                "features_used": pos.get("features_used"),
                "signal_prob": pos.get("signal_prob"),
                "threshold": pos.get("threshold"),
            })
            closed.append(ticker)

    log.info(f"Flattened {len(closed)} positions: {closed}")
    log.info(f"Running realized P&L: ${state['realized_pnl']:.2f}")
    save_state(state)

    # Wiring: account snapshot AFTER flatten succeeds (audit-gap collector).
    # Captures realized day P&L + post-flatten equity for the training row.
    _run_persist_collector(
        "persist_account_snapshots.py",
        ["--phase", "flatten"],
        timeout=30,
    )

    if _EB and closed:
        try:
            _EB.publish_from_anywhere("paper_position_flattened", {
                "date": today,
                "tickers": closed,
                "count": len(closed),
                "realized_pnl": state.get("realized_pnl", 0.0),
                "mode": MODE,
            }, source="live_paper_trade")
        except Exception:
            pass
    return 0


# ---------------------------------------------------------------------------
# WS-fill reconciliation
# ---------------------------------------------------------------------------
FILLS_DIR = PAPER_DIR / "fills"


def _load_ws_fills(today: str) -> dict[str, dict]:
    """Load today's WS-captured fills as {order_id: best_fill_record}.

    The WS daemon (`live_paper_trade_ws.py` → `alpaca_stream_consumer.py`)
    appends one JSONL row per fill / partial_fill event to
    paper_trade/fills/<DATE>.jsonl. For fully-filled orders, the LAST event
    (status='filled') carries the cumulative qty + filled_avg_price.

    Returns the most-progressed (status='filled' wins; else last seen) record
    per order_id. Missing file → empty dict (so reconciler gracefully degrades).
    """
    path = FILLS_DIR / f"{today}.jsonl"
    if not path.exists():
        return {}
    by_oid: dict[str, dict] = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                oid = rec.get("order_id") or ""
                if not oid:
                    continue
                prior = by_oid.get(oid)
                if prior is None:
                    by_oid[oid] = rec
                    continue
                # Prefer status='filled' over partial; else keep last write
                if rec.get("status") == "filled" and prior.get("status") != "filled":
                    by_oid[oid] = rec
                elif prior.get("status") != "filled":
                    by_oid[oid] = rec
    except Exception as e:
        log.warning(f"_load_ws_fills: read failed for {path}: {e}")
        return {}
    return by_oid


_LIFECYCLE_EVENTS = {"canceled", "rejected", "expired", "done_for_day"}


def _load_ws_fills_by_symbol(today: str) -> dict[str, dict]:
    """Aggregate SELL fills by symbol -> {avg_sell_px, sell_qty}.

    Fix A (2026-05-21): the existing _load_ws_fills() dedups by order_id, but
    closed_trades[].order_id is the BUY order_id (from cmd_open), while the
    fills JSONL contains SELL order_ids (from cmd_flatten). They never match,
    so reconcile_ws_fills() reported `reconciled=0, still_pending=N` for every
    LIVE_PAPER day. This helper aggregates SELL fills by symbol so the
    reconciler can fall back to ticker-match when order_id-match fails.
    """
    path = FILLS_DIR / f"{today}.jsonl"
    if not path.exists():
        return {}
    agg: dict[str, dict] = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                side = str(rec.get("side") or "")
                if "SELL" not in side.upper():
                    continue
                sym = rec.get("symbol") or ""
                if not sym:
                    continue
                try:
                    q = float(rec.get("qty") or 0.0)
                    p = float(rec.get("filled_avg_price") or 0.0)
                except (TypeError, ValueError):
                    continue
                if q <= 0 or p <= 0:
                    continue
                slot = agg.setdefault(sym, {"sell_qty": 0.0, "proceeds": 0.0})
                slot["sell_qty"] += q
                slot["proceeds"] += q * p
    except Exception as e:
        log.warning(f"_load_ws_fills_by_symbol: read failed for {path}: {e}")
        return {}
    for sym, slot in agg.items():
        slot["avg_sell_px"] = slot["proceeds"] / slot["sell_qty"] if slot["sell_qty"] > 0 else 0.0
    return agg


def _load_ws_lifecycle_events(today: str) -> list[dict]:
    """Scan today's fills JSONL for non-fill terminal events.

    Audit gap #3 (2026-05-18) — REJECT/CANCEL/EXPIRED/DONE_FOR_DAY events
    were never persisted into state, so post-trade analysis couldn't tell
    "model fired but order was rejected" from "model fired and filled".

    Returns a list of normalized rows (one per event row in the JSONL).
    Schema:
      {order_id, symbol, side, event, status, timestamp, qty, received_at}

    Missing file or unparseable rows → silently skipped. Distinct from
    _load_ws_fills which dedups on order_id; here we keep all rows because
    one order_id may legitimately have multiple lifecycle transitions
    (e.g. partial_fill → canceled-remainder).
    """
    path = FILLS_DIR / f"{today}.jsonl"
    if not path.exists():
        return []
    rows: list[dict] = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                event = (rec.get("event") or rec.get("status") or "").lower()
                if event not in _LIFECYCLE_EVENTS:
                    continue
                rows.append({
                    "order_id": rec.get("order_id") or "",
                    "symbol": rec.get("symbol") or rec.get("ticker") or "",
                    "side": rec.get("side") or "",
                    "event": event,
                    "status": rec.get("status") or event,
                    "timestamp": rec.get("timestamp") or rec.get("event_time") or "",
                    "qty": rec.get("qty"),
                    "received_at": rec.get("received_at") or rec.get("ts") or "",
                    # Carry execution_id when present so idempotency dedup
                    # can prefer it over the (order_id, event) composite key.
                    "execution_id": rec.get("execution_id") or "",
                })
    except Exception as e:
        log.warning(f"_load_ws_lifecycle_events: read failed for {path}: {e}")
        return []
    return rows


def reconcile_ws_fills(today: str | None = None) -> dict[str, Any]:
    """Merge WS-captured fills into today's state.closed_trades.

    For every `closed_trades` entry with `pending_ws_fill=True`, look up the
    matching order_id in `paper_trade/fills/<today>.jsonl` and write
    `exit_price` + `exit_qty` + `pnl`. Returns a summary dict.

    Also mirrors REJECT/CANCEL/EXPIRED/DONE_FOR_DAY events from the same
    JSONL into `state["lifecycle_events"]` (audit gap #3, 2026-05-18).
    Idempotent — re-runs skip rows already persisted.

    Safe to call multiple times — once a record has `pending_ws_fill=False`
    it is skipped.

    This is the BUG FIX for `exit_price=None` in `cmd_flatten` LIVE_PAPER path.
    """
    today_str = today or date.today().isoformat()
    state = load_state(today_str)
    fills_by_oid = _load_ws_fills(today_str)

    reconciled = 0
    still_pending = 0
    # Fix A (2026-05-21): also aggregate SELL fills by symbol so we can match
    # by ticker when order_id match fails (BUY/SELL order_ids do not coincide).
    sells_by_symbol = _load_ws_fills_by_symbol(today_str)
    if not fills_by_oid and not sells_by_symbol:
        log.warning(
            f"[reconcile] No WS fills found at "
            f"{FILLS_DIR / (today_str + '.jsonl')} — exit_price unchanged"
        )
        # Even with zero fills, a day can have rejected/canceled/expired
        # orders — fall through to the lifecycle-events block below so
        # those still get persisted (audit gap #3, 2026-05-18).
        still_pending = sum(
            1 for ct in state["closed_trades"] if ct.get("pending_ws_fill")
        )
    iter_trades = state["closed_trades"] if (fills_by_oid or sells_by_symbol) else []
    for ct in iter_trades:
        if not ct.get("pending_ws_fill"):
            continue
        oid = ct.get("order_id") or ""
        fill = fills_by_oid.get(oid)
        reconcile_src = "ws_fills"
        # Primary path: order_id match.
        if fill:
            try:
                exit_price = float(fill.get("filled_avg_price") or 0.0)
            except (TypeError, ValueError):
                exit_price = 0.0
            try:
                exit_qty = float(fill.get("qty") or 0.0)
            except (TypeError, ValueError):
                exit_qty = 0.0
        else:
            # Fallback (Fix A 2026-05-21): match by ticker via aggregated SELL fills.
            sym_agg = sells_by_symbol.get(ct.get("ticker") or "")
            if not sym_agg:
                still_pending += 1
                continue
            exit_price = float(sym_agg.get("avg_sell_px") or 0.0)
            exit_qty = float(sym_agg.get("sell_qty") or 0.0)
            reconcile_src = "ws_fills_by_symbol"
        if exit_price <= 0.0:
            still_pending += 1
            continue

        ct["exit_price"] = exit_price
        ct["exit_qty"] = exit_qty
        ct["pending_ws_fill"] = False
        ct["reconciled_from"] = reconcile_src
        ct["reconciled_at"] = datetime.now(timezone.utc).isoformat()

        # Compute pnl if entry_price + qty are present.
        entry_price = ct.get("entry_price")
        # Prefer the actual fill qty; fall back to the open-side qty estimate.
        eff_qty = exit_qty or (
            float(ct.get("qty") or 0.0) if ct.get("qty") is not None else 0.0
        )
        if entry_price is not None and eff_qty > 0:
            try:
                ct["pnl"] = (exit_price - float(entry_price)) * eff_qty
            except (TypeError, ValueError):
                pass
        reconciled += 1
        log.info(
            f"[reconcile] {ct.get('ticker')} order={oid[:12]}... "
            f"exit_px=${exit_price:.4f} qty={exit_qty} pnl=${ct.get('pnl', 0.0):.2f}"
        )

    # Audit gap #3 (2026-05-18) — mirror lifecycle events (reject/cancel/
    # expired/done_for_day) into state["lifecycle_events"]. Idempotent: dedup
    # by execution_id when present, else by (order_id, event, timestamp).
    lifecycle_new = _load_ws_lifecycle_events(today_str)
    if "lifecycle_events" not in state or not isinstance(
        state.get("lifecycle_events"), list
    ):
        state["lifecycle_events"] = []
    seen_keys: set[tuple[str, ...]] = set()
    for prev in state["lifecycle_events"]:
        ex_id = prev.get("execution_id") or ""
        if ex_id:
            seen_keys.add(("ex", ex_id))
        else:
            seen_keys.add((
                "k",
                prev.get("order_id") or "",
                prev.get("event") or "",
                prev.get("timestamp") or "",
            ))
    lifecycle_added = 0
    for row in lifecycle_new:
        ex_id = row.get("execution_id") or ""
        if ex_id:
            key: tuple[str, ...] = ("ex", ex_id)
        else:
            key = (
                "k",
                row.get("order_id") or "",
                row.get("event") or "",
                row.get("timestamp") or "",
            )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        state["lifecycle_events"].append(row)
        lifecycle_added += 1
        log.info(
            f"[reconcile] lifecycle event: {row.get('symbol')} "
            f"event={row.get('event')} order={(row.get('order_id') or '')[:12]}..."
        )

    state["last_reconcile_at"] = datetime.now(timezone.utc).isoformat()
    save_state(state)
    summary = {
        "date": today_str,
        "fills_seen": len(fills_by_oid),
        "reconciled": reconciled,
        "still_pending": still_pending,
        "lifecycle_events_added": lifecycle_added,
        "lifecycle_events_total": len(state["lifecycle_events"]),
    }
    log.info(f"[reconcile] summary: {summary}")
    return summary


# ---------------------------------------------------------------------------
# SUBCOMMAND: ingest (16:30 ET)
# ---------------------------------------------------------------------------
def cmd_ingest(args) -> int:
    """16:30 ET — save fills/P&L, append training data, trigger retrain.

    Step 1: reconcile WS fills (merges paper_trade/fills/<DATE>.jsonl into
    state.closed_trades — fills in exit_price for the LIVE_PAPER pending
    records). This is the BUG FIX for exit_price=None.

    Step 2: hand off to live_paper_trade_ingest.py for training-data append +
    retrain (its existing responsibility).
    """
    today = getattr(args, "date", None) or date.today().isoformat()
    dry_run = bool(getattr(args, "dry_run", False))
    log.info(f"=== INGEST {today} | mode={MODE} | dry_run={dry_run} ===")

    if dry_run:
        # DRY-RUN: log what we'd reconcile + ingest, do NOT mutate state and
        # do NOT spawn the downstream ingest pipeline (which writes outcomes
        # parquet + can trigger a retrain).
        state_path = STATE_DIR / f"{today}_state.json"
        fills_path = FILLS_DIR / f"{today}.jsonl"
        log.info(
            f"[DRY-RUN] would reconcile WS fills from {fills_path} into "
            f"{state_path} (would update closed_trades.exit_price + pnl)."
        )
        log.info(
            f"[DRY-RUN] would invoke {SCRIPTS_DIR / 'live_paper_trade_ingest.py'} "
            "— writes outcomes parquet + may trigger retrain. Skipped."
        )
        return 0

    # Step 1: reconcile WS fills (no-op if SIMULATED or no fills file).
    try:
        summary = reconcile_ws_fills(today)
        log.info(f"[INGEST] reconcile_ws_fills → {summary}")
    except Exception as e:
        log.exception(f"[INGEST] reconcile_ws_fills raised — continuing anyway: {e}")

    # Step 2: existing ingest pipeline.
    ingest_script = SCRIPTS_DIR / "live_paper_trade_ingest.py"
    if not ingest_script.exists():
        log.error(f"Ingest script missing: {ingest_script}")
        return 1

    result = subprocess.run(
        [sys.executable, str(ingest_script)],
        capture_output=False,
        timeout=1800,  # 30 min max for retrain
    )
    if result.returncode != 0:
        log.error(f"Ingest script exited with code {result.returncode}")
        return 1

    log.info("Ingest complete.")
    return 0


# ---------------------------------------------------------------------------
# SUBCOMMAND: manage-stops (every 15min between 10:00-15:45 ET)
# ---------------------------------------------------------------------------
# Quick-win A (per audit ac0f5aca, 2026-05-22): intraday stop-loss + take-profit.
# Window 10:00-15:45 ET; fires every 15min via launchd plist
# `com.zg.paper_trade_manage_stops`. Reads current Alpaca positions, computes
# unrealized_pl_pct vs the avg entry price, and submits SELL via the existing
# smart-order router when thresholds are crossed. Prevents the 10% hard-halt
# ($200 loss in $2k budget) by capping per-position drawdown at -3% and
# locking in winners at +5%.
#
# Thresholds (env-overridable so a careful operator can tune without code edit):
#   STOP_LOSS_PCT     default -0.03  (close on -3% unrealized)
#   TAKE_PROFIT_PCT   default  0.05  (close on +5% unrealized)
#
# Zero risk_engine changes — this lives entirely inside live_paper_trade.py.
# State extensions (forward-compat, additive): per-position
# `current_price`, `unrealized_pnl_pct`, `peak_pnl_pct_today`, `last_rescored_at`
# are written ONLY when manage-stops fires; old readers ignore them.
# ---------------------------------------------------------------------------
STOP_LOSS_PCT = float(os.environ.get("STOP_LOSS_PCT", "-0.03"))
TAKE_PROFIT_PCT = float(os.environ.get("TAKE_PROFIT_PCT", "0.05"))


def _current_price_for(ticker: str) -> float | None:
    """Best-effort latest price. Prefers Alpaca SIP; falls back to yfinance.

    Used by cmd_manage_stops to compute unrealized P&L pct against the
    position's avg entry price. Returns None on any failure — caller must
    handle missing price (skip evaluation, don't close on stale data).
    """
    if MODE == "LIVE_PAPER":
        p = _alpaca_get_latest_price(ticker)
        if p is not None:
            return p
    return _sim_get_latest_price(ticker)


def _cmd_manage_stops_market_day_guard(dry_run: bool) -> bool:
    """Helper for cmd_manage_stops: returns True if caller should proceed."""
    if dry_run:
        return True
    return _market_day_guard("manage-stops")


def cmd_manage_stops(args) -> int:
    """Every 15min 10:00-15:45 ET — close positions hitting SL/TP thresholds.

    Reads current positions (from Alpaca in LIVE_PAPER, from state in SIM),
    computes unrealized P&L % vs avg entry price, and submits SELL via
    `_place_smart_order_alpaca` for any position where:
        unrealized_pnl_pct <= STOP_LOSS_PCT   → close, reason=stop_loss
        unrealized_pnl_pct >= TAKE_PROFIT_PCT → close, reason=take_profit

    No-ops cleanly outside the 10:00-15:45 ET window (warning only —
    matches existing _assert_market_window semantics).

    --dry-run: simulate a position at -3.5% drawdown to verify SL fires, and
    +6% gain to verify TP fires. No live API calls. No state mutation.
    """
    today = getattr(args, "date", None) or date.today().isoformat()
    dry_run = bool(getattr(args, "dry_run", False))
    log.info(
        f"=== MANAGE-STOPS {today} | mode={MODE} | dry_run={dry_run} | "
        f"SL={STOP_LOSS_PCT:.2%} TP={TAKE_PROFIT_PCT:.2%} ==="
    )
    # Market-day guard: skip on weekends + NYSE holidays. manage-stops fires
    # 7 days/week per its plist (no Weekday filter), so this guard is the
    # primary defense against weekend execution.
    if not _cmd_manage_stops_market_day_guard(dry_run):
        return 0
    _assert_market_window("manage-stops", 10, 0, 15, 45)

    # ── DRY-RUN: synthetic positions exercise both branches ────────────────
    if dry_run:
        log.info("[DRY-RUN] simulating two positions to verify SL/TP triggers")
        synthetic = [
            ("SL_TEST", 100.0, 96.5),   # -3.5% → should trigger stop_loss
            ("TP_TEST", 100.0, 106.0),  # +6.0% → should trigger take_profit
            ("HOLD_TEST", 100.0, 101.0),  # +1.0% → should hold
        ]
        triggers = []
        for tkr, entry, current in synthetic:
            pnl_pct = (current - entry) / entry
            if pnl_pct <= STOP_LOSS_PCT:
                reason = "stop_loss"
            elif pnl_pct >= TAKE_PROFIT_PCT:
                reason = "take_profit"
            else:
                reason = None
            log.info(
                f"[DRY-RUN] {tkr}: entry=${entry:.2f} cur=${current:.2f} "
                f"pnl_pct={pnl_pct:+.2%} → {reason or 'HOLD'}"
            )
            if reason:
                triggers.append((tkr, reason))
        log.info(f"[DRY-RUN] would close {len(triggers)} positions: {triggers}")
        if len(triggers) != 2 or set(r for _, r in triggers) != {"stop_loss", "take_profit"}:
            log.error("[DRY-RUN] sanity check FAILED — expected exactly one SL + one TP")
            return 1
        log.info("[DRY-RUN] sanity check PASSED")
        return 0

    state = load_state(today)
    if not state.get("positions"):
        log.info("No open positions — nothing to manage.")
        return 0

    # Pull live positions (LIVE_PAPER) so avg_entry_price + qty are authoritative.
    live_positions: dict[str, dict] = {}
    if MODE == "LIVE_PAPER":
        live_positions = _get_alpaca_positions()

    closed_count = 0
    for ticker, pos in list(state["positions"].items()):
        # Resolve avg entry price: prefer Alpaca's value (handles partial fills),
        # fall back to state-recorded entry_price.
        live = live_positions.get(ticker)
        entry_price = (live.get("avg_entry_price") if live else None) or pos.get("entry_price")
        qty = (live.get("qty") if live else None) or pos.get("qty", 0)
        if not entry_price or not qty:
            log.warning(f"[{ticker}] missing entry_price/qty — skipping")
            continue

        current = _current_price_for(ticker)
        if current is None:
            log.warning(f"[{ticker}] could not fetch current price — skipping")
            continue

        pnl_pct = (current - entry_price) / entry_price
        # Stamp state with rescore metadata (additive, forward-compat).
        pos["current_price"] = current
        pos["unrealized_pnl_pct"] = pnl_pct
        prev_peak = pos.get("peak_pnl_pct_today", pnl_pct)
        pos["peak_pnl_pct_today"] = max(prev_peak, pnl_pct)
        pos["last_rescored_at"] = datetime.now(timezone.utc).isoformat()

        if pnl_pct <= STOP_LOSS_PCT:
            reason = "stop_loss"
        elif pnl_pct >= TAKE_PROFIT_PCT:
            reason = "take_profit"
        else:
            log.info(
                f"[{ticker}] entry=${entry_price:.2f} cur=${current:.2f} "
                f"pnl_pct={pnl_pct:+.2%} → HOLD"
            )
            continue

        log.info(
            f"[{ticker}] entry=${entry_price:.2f} cur=${current:.2f} "
            f"pnl_pct={pnl_pct:+.2%} → CLOSE ({reason})"
        )

        # Submit close order. LIVE_PAPER routes through smart-limit/market router.
        # SIMULATED: synthetic fill at current price.
        if MODE == "LIVE_PAPER":
            try:
                order_result = _place_smart_order_alpaca(
                    ticker, qty=int(qty), side="sell",
                    signal_strength=None, prob=None,
                )
                exit_price = order_result.get("filled_avg_price") or current
                pending_ws = order_result.get("filled_avg_price") is None
            except Exception as e:
                log.error(f"[{ticker}] close order failed: {e} — leaving position open")
                continue
        else:
            order_result = {"order_id": f"SIM-STOP-{ticker}-{today}"}
            exit_price = current
            pending_ws = False
            log.info(f"[SIM] SELL {qty} {ticker} @ ${current:.2f} (reason={reason})")

        pnl = (exit_price - entry_price) * qty
        if MODE == "SIMULATED":
            state["realized_pnl"] = state.get("realized_pnl", 0.0) + pnl

        state["closed_trades"].append({
            "ticker": ticker,
            "qty": qty,
            "entry_price": entry_price,
            "exit_price": exit_price if not pending_ws else None,
            "pnl": pnl if not pending_ws else None,
            "order_id": order_result.get("order_id", ""),
            "client_order_id": order_result.get("client_order_id", ""),
            "closed_at": datetime.now(timezone.utc).isoformat(),
            "closed_reason": reason,
            "mode": MODE,
            "pending_ws_fill": pending_ws,
            # Carry model-version stamp through (audit gap #9).
            "pipeline": pos.get("pipeline"),
            "model_run_dir": pos.get("model_run_dir"),
            "feature_hash": pos.get("feature_hash"),
            "features_used": pos.get("features_used"),
            "signal_prob": pos.get("signal_prob"),
            "threshold": pos.get("threshold"),
        })
        state["positions"].pop(ticker, None)
        closed_count += 1

    # LIVE_PAPER: refresh realized P&L from Alpaca even when nothing closed —
    # gives the next halt-check call a fresh equity baseline.
    if MODE == "LIVE_PAPER":
        _refresh_realized_pnl_from_alpaca(state)

    save_state(state)
    log.info(
        f"manage-stops complete: closed {closed_count} position(s), "
        f"{len(state.get('positions', {}))} still open"
    )
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
SUBCOMMANDS = {
    "startup": cmd_startup,
    "open-trades": cmd_open_trades,
    "flatten": cmd_flatten,
    "ingest": cmd_ingest,
    "manage-stops": cmd_manage_stops,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Live paper-trading orchestration for S&P 500 ML Mastery"
    )
    parser.add_argument(
        "subcommand",
        choices=list(SUBCOMMANDS.keys()),
        help="Which phase to run",
    )
    parser.add_argument(
        "--date",
        default=None,
        help=(
            "Override date (YYYY-MM-DD). Defaults to today. When set to "
            "anything other than today's date, live order submission is "
            "automatically skipped (back-test / replay safety). Used for all "
            "date-keyed path lookups (signals/<date>.json, state/<date>_state.json, "
            "fills/<date>.jsonl)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help=(
            "Log what each subcommand WOULD do but skip all live-impact actions: "
            "Alpaca account check (startup), submit_order (open-trades), "
            "close_all_positions (flatten), outcomes-parquet write (ingest). "
            "Forced ON when --date != today."
        ),
    )
    args = parser.parse_args()

    today = args.date or date.today().isoformat()

    # Safety: auto-enable dry-run when --date is not today's date. Running
    # against a past/future date with live API calls would submit real orders
    # tagged with the wrong date — explicit guard at the top so every
    # subcommand inherits it regardless of which handler the user invoked.
    real_today = date.today().isoformat()
    if args.date and args.date != real_today and not args.dry_run:
        log.warning(
            f"[SAFETY] --date={args.date} != today ({real_today}) — "
            "auto-enabling --dry-run. No live API calls will be made."
        )
        args.dry_run = True

    if args.dry_run:
        log.warning(
            f"[DRY-RUN] dry-run mode active for '{args.subcommand}' — "
            "no live API calls, no state-mutating writes."
        )

    log.info(
        f"live_paper_trade.py {args.subcommand} | date={today} | "
        f"mode={MODE} | dry_run={args.dry_run}"
    )

    try:
        rc = SUBCOMMANDS[args.subcommand](args)
        status = "OK" if rc == 0 else "FAILED"
        log.info(f"Subcommand '{args.subcommand}' finished → {status} (rc={rc})")
        return rc
    except Exception as e:
        log.exception(f"FATAL error in '{args.subcommand}': {e}")
        log.warning(f"Falling back to safe mode: no further actions taken.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
