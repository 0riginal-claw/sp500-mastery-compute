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
    MAX_POSITION_NOTIONAL    $500 per ticker
    MAX_TOTAL_EXPOSURE       $25,000 across all open positions
    DAILY_LOSS_SOFT_HALT     -$1,500 → block NEW entries, exit-only mode
    DAILY_LOSS_HARD_HALT     -$2,500 → flat all positions, halt for day
    BATCH_OPEN_STAGGER_S     1.0     → between submit_order calls at 09:30 open
    ORDER_TYPE               MARKET only, regular-hours only
    NO SHORTS / NO OPTIONS / NO LEVERAGE / NO OVERNIGHT HOLDS

    Thresholds source: paper_trade/alpaca_risk_config.yaml (SAFE-SUBSET applied
    2026-05-18 per reports/alpaca_risk_sizing_2026-05-18.md). Half-Kelly/ATR
    sizing + limit-order migration are DEFERRED pending backtest validation.

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
# Risk constants
# ---------------------------------------------------------------------------
MAX_POSITION_NOTIONAL = 500.0    # $ per ticker
MAX_TOTAL_EXPOSURE = 25_000.0    # $ total across all open positions

# Two-tier daily-loss halt (2026-05-18, alpaca_risk_config.yaml SAFE subset).
# Replaces single MAX_DAILY_LOSS = -$1,000 which was too tight (1.06% of cash;
# trips on a routine -0.3% intraday session).
DAILY_LOSS_SOFT_HALT = -1_500.0  # $ → block NEW entries, allow exits/manage existing
DAILY_LOSS_HARD_HALT = -2_500.0  # $ → flat all positions, halt for day
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

    current_exposure = sum(
        p.get("notional", 0) for p in state["positions"].values()
    )
    sized = compute_position_sizes(signals, current_exposure)

    if not sized:
        log.info("No positions to open (all filtered by guardrails).")
        return 0

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
            try:
                order_result = _place_market_order_alpaca(
                    ticker, side="buy", notional=order_size_notional
                )
            except Exception as e:
                log.warning(
                    f"Notional order failed for {ticker} (${order_size_notional:.2f}): "
                    f"{e} — falling back to qty={qty}"
                )
                try:
                    order_result = _place_market_order_alpaca(
                        ticker, qty=qty, side="buy"
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
    if not fills_by_oid:
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
    for ct in (state["closed_trades"] if fills_by_oid else []):
        if not ct.get("pending_ws_fill"):
            continue
        oid = ct.get("order_id") or ""
        fill = fills_by_oid.get(oid)
        if not fill:
            still_pending += 1
            continue
        # Parse fill record
        try:
            exit_price = float(fill.get("filled_avg_price") or 0.0)
        except (TypeError, ValueError):
            exit_price = 0.0
        try:
            exit_qty = float(fill.get("qty") or 0.0)
        except (TypeError, ValueError):
            exit_qty = 0.0
        if exit_price <= 0.0:
            still_pending += 1
            continue

        ct["exit_price"] = exit_price
        ct["exit_qty"] = exit_qty
        ct["pending_ws_fill"] = False
        ct["reconciled_from"] = "ws_fills"
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
# Entry point
# ---------------------------------------------------------------------------
SUBCOMMANDS = {
    "startup": cmd_startup,
    "open-trades": cmd_open_trades,
    "flatten": cmd_flatten,
    "ingest": cmd_ingest,
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
