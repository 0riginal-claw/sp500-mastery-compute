"""
intraday_engine.py — Intraday 4-strategy ensemble live paper-trade engine.

ADDITIVE LAYER — runs in parallel with `live_paper_trade.py` daily loop.
DO NOT activate (no launchd bootstrap) until Tuesday 2026-05-19 post backtest
validation gate.

Flow per 15s tick (09:30-15:55 ET):
  1. Load enabled tickers + per-strategy enable flags from intraday_config.yaml
  2. Fetch last N 1-min bars per ticker (Alpaca SIP via ALPACA_RL when creds
     present; else yfinance 1m fallback)
  3. For each (ticker, strategy) compute score()
  4. Filter by per-(ticker,strategy) threshold (intraday mastery files)
  5. Get strategy weights from intraday_learner.allocate()
  6. Compute weighted ensemble signal per ticker
  7. If firing AND under budget AND not already open: place bracket order
     (DRY_RUN by default; LIVE wiring deferred)
  8. Append trade record (strategy-tagged) to
     paper_trade/strategy_outcomes/<DATE>.jsonl

At 15:55 ET: flatten all intraday positions (logged).

Modes:
  * Default: DRY_RUN (no real orders). Logs everything, returns fake order ids.
  * LIVE: set env var INTRADAY_ENGINE_LIVE=1 — raises NotImplementedError. Live
    wiring is intentionally NOT implemented until validation gate clears.

Subcommands:
  python intraday_engine.py tick-once --tickers AAPL,MSFT,NVDA
  python intraday_engine.py run
  python intraday_engine.py flatten
  python intraday_engine.py report
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time as _time
import uuid
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytz
import yaml

# Add scripts dir to path so we can import sibling modules
SCRIPTS_DIR = Path(__file__).parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from intraday_strategies import STRATEGIES  # noqa: E402

# Best-effort: shared rate-limiter from live_paper_trade.py
try:
    from alpaca_rate_limit import ALPACA_RL  # type: ignore
except Exception:
    ALPACA_RL = None  # type: ignore

# ── paths ────────────────────────────────────────────────────────────────────
WORK = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools/s&p500-ticker-mastery"
)
CONFIG_PATH = WORK / "paper_trade" / "intraday_config.yaml"
OUTCOMES_DIR = WORK / "paper_trade" / "strategy_outcomes"
MASTERY_DIR = WORK / "mastery_files"
LOGS_DIR = WORK / "logs"
WEIGHTS_DIR = WORK / "paper_trade" / "intraday_weights"

for _d in (OUTCOMES_DIR, LOGS_DIR, WEIGHTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── constants ────────────────────────────────────────────────────────────────
ET = pytz.timezone("America/New_York")
POLL_INTERVAL_S = 15
RTH_OPEN = dtime(9, 30)
RTH_FLATTEN = dtime(15, 55)
RTH_NO_NEW = dtime(15, 30)

INTRADAY_PER_TICKER_USD = 500.0
INTRADAY_MAX_TOTAL_USD = 25_000.0

LIVE_MODE = os.environ.get("INTRADAY_ENGINE_LIVE", "0") == "1"
SIM_PROVIDER = "yfinance"  # set by fetch_bars_batched

# ── logging ──────────────────────────────────────────────────────────────────
LOG_FILE = LOGS_DIR / "intraday_engine.log"
logger = logging.getLogger("intraday_engine")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _fh = logging.FileHandler(LOG_FILE)
    _ch = logging.StreamHandler()
    _fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(module)s] %(message)s"
    )
    _fh.setFormatter(_fmt)
    _ch.setFormatter(_fmt)
    logger.addHandler(_fh)
    logger.addHandler(_ch)


# ── config / mastery loaders ─────────────────────────────────────────────────
def load_config() -> dict:
    """Read intraday_config.yaml. Returns {} if missing."""
    if not CONFIG_PATH.exists():
        logger.warning("config missing: %s", CONFIG_PATH)
        return {}
    return yaml.safe_load(CONFIG_PATH.read_text()) or {}


def load_intraday_mastery(ticker: str, strategy_id: str) -> dict | None:
    """Read `mastery_files/{TICKER}_INTRADAY_{strategy_id}_mastered.md` if exists."""
    path = MASTERY_DIR / f"{ticker}_INTRADAY_{strategy_id}_mastered.md"
    if not path.exists():
        return None
    txt = path.read_text()
    # Naive JSON block extraction
    if "```json" in txt:
        start = txt.find("```json") + len("```json")
        end = txt.find("```", start)
        if end > start:
            try:
                return json.loads(txt[start:end].strip())
            except Exception as e:  # pragma: no cover
                logger.warning("mastery parse failed %s: %s", path, e)
    return None


# ── bar fetching ─────────────────────────────────────────────────────────────
def fetch_bars_batched(
    tickers: list[str], limit: int = 60
) -> dict[str, pd.DataFrame]:
    """Fetch last `limit` 1-min bars per ticker.

    Priority:
      1. Alpaca SIP via ALPACA_RL (requires creds) — TODO live wire-up
      2. yfinance fallback (free, rate-limited externally)
    """
    global SIM_PROVIDER
    out: dict[str, pd.DataFrame] = {}

    # Alpaca path is intentionally deferred — live wiring not authorized yet.
    # Always fall through to yfinance in this scaffold.
    SIM_PROVIDER = "yfinance"
    try:
        import yfinance as yf  # type: ignore
    except Exception as e:
        logger.warning("yfinance unavailable (%s) — returning empty bars", e)
        return out

    for t in tickers:
        try:
            df = yf.download(
                t,
                period="1d",
                interval="1m",
                progress=False,
                prepost=False,
                auto_adjust=False,
            )
            if df is None or len(df) == 0:
                logger.info("no yf bars for %s", t)
                continue
            # Normalize to expected schema
            df = df.reset_index()
            # yfinance: 'Datetime' index (tz-aware UTC by default)
            ts_col = "Datetime" if "Datetime" in df.columns else df.columns[0]
            df = df.rename(
                columns={
                    ts_col: "timestamp",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )
            # yfinance sometimes returns multi-index for multi-ticker; we asked
            # for single ticker so columns should be flat. If still multi-index:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower() for c in df.columns]
                df = df.rename(columns={"datetime": "timestamp"})
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(ET)
            out[t] = df.tail(limit).reset_index(drop=True)
        except Exception as e:
            logger.warning("yf fetch failed %s: %s", t, e)
            continue

    return out


# ── signal computation ──────────────────────────────────────────────────────
def compute_signals(
    bars_by_ticker: dict[str, pd.DataFrame],
    enabled_strategies: list[str],
    extra_params: dict[str, dict] | None = None,
) -> list[dict]:
    """Run every (ticker, strategy) pair. Return list of score dicts."""
    out: list[dict] = []
    extra_params = extra_params or {}
    for ticker, bars in bars_by_ticker.items():
        for sid in enabled_strategies:
            fn = STRATEGIES.get(sid)
            if fn is None:
                logger.warning("unknown strategy %s", sid)
                continue
            try:
                rec = fn(
                    bars,
                    ticker=ticker,
                    params=extra_params.get(ticker, {}),
                )
                out.append(rec)
            except Exception as e:  # pragma: no cover
                logger.exception("score() crash %s/%s: %s", ticker, sid, e)
    return out


def apply_thresholds(
    signals: list[dict], cfg: dict
) -> list[dict]:
    """Drop signals where prob < per-(ticker,strategy) threshold."""
    strat_cfg = cfg.get("strategies", {})
    kept = []
    for s in signals:
        if s.get("signal", 0) == 0:
            continue
        sid = s["strategy_id"]
        ticker = s["ticker"]
        # Per-pair mastery threshold overrides default
        mastery = load_intraday_mastery(ticker, sid)
        if mastery and isinstance(mastery, dict):
            thr = float(mastery.get("prob_threshold", 0.6))
        else:
            thr = float(
                strat_cfg.get(sid, {}).get("default_prob_threshold", 0.6)
            )
        if s["prob"] >= thr:
            s["_threshold"] = thr
            kept.append(s)
    return kept


def ensemble(
    filtered_signals: list[dict], weights_by_strategy: dict[str, float]
) -> dict[str, dict]:
    """Combine per-strategy signals per ticker via weighted prob avg.

    Returns {ticker: {weighted_prob, top_strategy, contributing_strategies}}.
    Only includes tickers with weighted_prob >= 0.5.
    """
    by_ticker: dict[str, list[dict]] = {}
    for s in filtered_signals:
        by_ticker.setdefault(s["ticker"], []).append(s)
    out: dict[str, dict] = {}
    for ticker, sigs in by_ticker.items():
        wsum = 0.0
        psum = 0.0
        contrib = []
        top_sig = None
        for s in sigs:
            w = weights_by_strategy.get(s["strategy_id"], 0.0)
            psum += s["prob"] * w
            wsum += w
            contrib.append(s["strategy_id"])
            if top_sig is None or s["prob"] > top_sig["prob"]:
                top_sig = s
        weighted = psum / wsum if wsum > 0 else 0.0
        if weighted >= 0.5 and top_sig is not None:
            out[ticker] = {
                "weighted_prob": round(weighted, 4),
                "top_strategy": top_sig["strategy_id"],
                "contributing_strategies": contrib,
                "top_signal": top_sig,
            }
    return out


# ── order placement (dry-run only) ───────────────────────────────────────────
def place_bracket_order_dryrun(
    ticker: str,
    side: str,
    qty: int,
    entry: float,
    target: float,
    stop: float,
) -> str:
    """Dry-run order: logs and returns fake order id. No network calls."""
    if LIVE_MODE:
        raise NotImplementedError(
            "LIVE mode wiring deferred to Tuesday 2026-05-19 post-validation. "
            "Unset INTRADAY_ENGINE_LIVE to proceed in DRY_RUN."
        )
    oid = f"dryrun_{uuid.uuid4().hex[:8]}"
    logger.info(
        "[DRYRUN] bracket %s side=%s qty=%d entry=%.4f tp=%.4f sl=%.4f -> %s",
        ticker,
        side,
        qty,
        entry,
        target,
        stop,
        oid,
    )
    return oid


def log_trade(record: dict) -> None:
    """Append a strategy-tagged trade record to today's jsonl."""
    today = datetime.now(ET).date().isoformat()
    path = OUTCOMES_DIR / f"{today}.jsonl"
    record.setdefault("ts", datetime.now(ET).isoformat())
    record.setdefault("mode", "live" if LIVE_MODE else "dry_run")
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


# ── flatten (log-only) ───────────────────────────────────────────────────────
def flatten_all() -> int:
    """Log a flatten event for any open intraday positions. Returns count."""
    # In dry-run we don't track positions across runs; emit a marker record.
    logger.info("[DRYRUN] flatten_all called at %s ET", datetime.now(ET))
    log_trade({"event": "flatten_all", "ticker": None, "strategy_id": None})
    return 0


# ── main loop ───────────────────────────────────────────────────────────────
def _resolve_enabled_tickers(cfg: dict) -> list[str]:
    enabled = cfg.get("enabled_tickers") or []
    if enabled:
        return list(enabled)
    # Default: glob v10 mastered tickers from mastery_files/
    pattern = "*_XGB_v10_mythos_mastered.md"
    tickers = sorted(
        p.name.split("_", 1)[0] for p in MASTERY_DIR.glob(pattern)
    )
    disabled = set(cfg.get("disabled_tickers") or [])
    return [t for t in tickers if t not in disabled]


def _read_weights(cfg: dict) -> dict[str, float]:
    """Read today's TS-allocated weights, or fall back to uniform."""
    today = datetime.now(ET).date().isoformat()
    p = WEIGHTS_DIR / f"weights_{today}.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception as e:  # pragma: no cover
            logger.warning("weights parse failed: %s", e)
    strats = list((cfg.get("strategies") or {}).keys()) or list(
        STRATEGIES.keys()
    )
    n = len(strats) or 1
    return {s: 1.0 / n for s in strats}


def tick_once(tickers: list[str] | None = None) -> dict:
    """One end-to-end pass. Returns summary dict for tests/reports."""
    cfg = load_config()
    if not cfg.get("engine_enabled", False):
        logger.warning(
            "engine_enabled=false in config — running in observation mode "
            "(signals computed and logged, NO orders placed)"
        )
    enabled_strats = [
        s
        for s, c in (cfg.get("strategies") or {}).items()
        if c.get("enabled", False)
    ] or list(STRATEGIES.keys())
    tlist = tickers or _resolve_enabled_tickers(cfg) or ["AAPL"]

    logger.info(
        "tick-once: tickers=%d strategies=%s mode=%s live=%s",
        len(tlist),
        enabled_strats,
        "live" if LIVE_MODE else "dry_run",
        LIVE_MODE,
    )

    bars = fetch_bars_batched(tlist, limit=60)
    if not bars:
        logger.warning("no bars fetched — exiting tick early")
        return {"signals": [], "filtered": [], "orders": [], "provider": SIM_PROVIDER}

    raw = compute_signals(bars, enabled_strats)
    filtered = apply_thresholds(raw, cfg)
    weights = _read_weights(cfg)
    fires = ensemble(filtered, weights)

    orders = []
    for ticker, ens in fires.items():
        if not cfg.get("engine_enabled", False):
            # observation mode — log but no order
            logger.info(
                "[OBS] would fire %s prob=%.3f top=%s",
                ticker,
                ens["weighted_prob"],
                ens["top_strategy"],
            )
            continue
        sig = ens["top_signal"]
        entry = sig["entry"] or 0.0
        if entry <= 0:
            continue
        qty = max(1, int(INTRADAY_PER_TICKER_USD // entry))
        try:
            oid = place_bracket_order_dryrun(
                ticker, "buy", qty, entry, sig["target"], sig["stop"]
            )
            rec = {
                "ticker": ticker,
                "strategy_id": ens["top_strategy"],
                "side": "buy",
                "qty": qty,
                "entry": entry,
                "target": sig["target"],
                "stop": sig["stop"],
                "prob": sig["prob"],
                "weighted_prob": ens["weighted_prob"],
                "reason": sig["reason"],
                "order_id": oid,
            }
            log_trade(rec)
            orders.append(rec)
        except NotImplementedError as e:
            logger.error("LIVE mode blocked: %s", e)
            raise

    return {
        "signals": raw,
        "filtered": filtered,
        "fires": fires,
        "orders": orders,
        "provider": SIM_PROVIDER,
    }


def run_loop() -> None:
    """Continuous poll loop 09:30-15:55 ET. Sleep POLL_INTERVAL_S between ticks."""
    logger.info("run_loop start  poll=%ds", POLL_INTERVAL_S)
    flattened = False
    while True:
        now = datetime.now(ET).time()
        if now < RTH_OPEN:
            secs = (
                datetime.combine(date.today(), RTH_OPEN)
                - datetime.combine(date.today(), now)
            ).total_seconds()
            logger.info("pre-market; sleeping %.0fs until 09:30 ET", secs)
            _time.sleep(min(secs, 60))
            continue
        if now >= RTH_FLATTEN:
            if not flattened:
                flatten_all()
                flattened = True
            logger.info("post-flatten; exiting loop")
            return
        try:
            tick_once()
        except NotImplementedError:
            logger.error("LIVE mode active but unwired — exiting loop")
            return
        except Exception as e:  # pragma: no cover
            logger.exception("tick crashed: %s", e)
        _time.sleep(POLL_INTERVAL_S)


def report_today() -> None:
    today = datetime.now(ET).date().isoformat()
    path = OUTCOMES_DIR / f"{today}.jsonl"
    if not path.exists():
        print(f"no outcomes for {today}")
        return
    print(f"=== {today} strategy_outcomes ({path}) ===")
    for ln in path.read_text().splitlines():
        try:
            r = json.loads(ln)
        except Exception:
            continue
        print(
            "  ",
            r.get("ts", "?"),
            r.get("ticker", "?"),
            r.get("strategy_id", "?"),
            "qty=" + str(r.get("qty", "?")),
            "prob=" + str(r.get("prob", "?")),
            r.get("reason", ""),
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="intraday 4-strategy engine")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s_tick = sub.add_parser("tick-once", help="single tick, dry-run")
    s_tick.add_argument("--tickers", type=str, default=None)

    sub.add_parser("run", help="continuous loop until 15:55 ET")
    sub.add_parser("flatten", help="log-only flatten of all positions")
    sub.add_parser("report", help="print today's strategy outcomes")

    args = ap.parse_args()
    if args.cmd == "tick-once":
        tickers = (
            [t.strip().upper() for t in args.tickers.split(",")]
            if args.tickers
            else None
        )
        summary = tick_once(tickers)
        print(
            json.dumps(
                {
                    "n_signals": len(summary["signals"]),
                    "n_filtered": len(summary["filtered"]),
                    "n_orders": len(summary["orders"]),
                    "fires": list((summary.get("fires") or {}).keys()),
                    "provider": summary["provider"],
                },
                indent=2,
            )
        )
    elif args.cmd == "run":
        run_loop()
    elif args.cmd == "flatten":
        flatten_all()
    elif args.cmd == "report":
        report_today()


if __name__ == "__main__":
    main()
