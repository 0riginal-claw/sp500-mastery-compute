"""
live_paper_trade_ingest.py — After-close ingestion: fills, P&L, training data append.

Run at 16:30 ET. This script:
  1. Loads today's state (positions/fills from live_paper_trade.py state file)
  2. Pulls realized fills/P&L from Alpaca (or uses simulated fills from state)
  3. Saves to paper_trade/daily/{YYYY-MM-DD}/
       fills.csv, pnl.json, signals_vs_outcomes.csv, summary.md
  4. Downloads today's bar for every active ticker → saves to
       paper_trade/incremental_bars/{ticker}/{YYYY-MM-DD}.parquet
  5. On Fridays 18:00 ET (or if --retrain flag passed): triggers rolling
       retrain of ALL mastered tickers. Compares new signals to prior.

SIMULATED mode: uses yfinance closing bar as "realized" fill data.
LIVE_PAPER mode: pulls fills from Alpaca portfolio history API.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import warnings
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

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

for _d in [DAILY_DIR, INCREMENTAL_DIR, LOGS_DIR]:
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
log = logging.getLogger("pt_ingest")

sys.path.insert(0, str(SCRIPTS_DIR))

# ---------------------------------------------------------------------------
# Credential detection (mirrors live_paper_trade.py)
# ---------------------------------------------------------------------------
def _detect_mode() -> tuple[str, str | None, str | None]:
    api_key = os.environ.get("ALPACA_PAPER_API_KEY")
    secret_key = os.environ.get("ALPACA_PAPER_SECRET_KEY")
    if api_key and secret_key:
        return "LIVE_PAPER", api_key, secret_key

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

    api_key = _kc("alpaca-paper-api-key")
    secret_key = _kc("alpaca-paper-secret-key")
    if api_key and secret_key:
        return "LIVE_PAPER", api_key, secret_key

    return "SIMULATED", None, None


MODE, _API_KEY, _SECRET_KEY = _detect_mode()

# ---------------------------------------------------------------------------
# Persistence-collector wiring helper (audit gap: feed every detail into XGB/Mythos)
# ---------------------------------------------------------------------------
def _run_persist_collector(script_name: str, extra_args: list[str], timeout: int = 60) -> None:
    """Shell out to a scripts/persist_*.py collector via subprocess.

    Best-effort: never raises into the caller. Logs rc + tail of stderr on
    non-zero. Skips silently if the target script is missing so an older
    deploy without the collector keeps running.
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
# State loading
# ---------------------------------------------------------------------------
def load_state(today: str) -> dict:
    path = STATE_DIR / f"{today}_state.json"
    if path.exists():
        return json.loads(path.read_text())
    log.warning(f"No state file for {today} — using empty state")
    return {
        "date": today,
        "positions": {},
        "closed_trades": [],
        "realized_pnl": 0.0,
        "halted": False,
        "mode": MODE,
    }


# ---------------------------------------------------------------------------
# Alpaca fill retrieval
# ---------------------------------------------------------------------------
def _get_alpaca_fills_today(today: str) -> list[dict]:
    """Pull today's closed orders from Alpaca portfolio/activities."""
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        client = TradingClient(api_key=_API_KEY, secret_key=_SECRET_KEY, paper=True)
        req = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            after=f"{today}T00:00:00Z",
        )
        orders = client.get_orders(req)
        fills = []
        for o in orders:
            fills.append({
                "ticker": o.symbol,
                "order_id": str(o.id),
                "side": str(o.side.value),
                "qty": float(o.filled_qty or 0),
                "avg_fill_price": float(o.filled_avg_price or 0),
                "status": str(o.status.value),
                "filled_at": str(o.filled_at),
            })
        return fills
    except Exception as e:
        log.error(f"Alpaca fills retrieval failed: {e}")
        return []


def _get_alpaca_portfolio_pnl() -> dict:
    """Pull today's portfolio P&L from Alpaca account."""
    try:
        from alpaca.trading.client import TradingClient
        client = TradingClient(api_key=_API_KEY, secret_key=_SECRET_KEY, paper=True)
        acct = client.get_account()
        return {
            "equity": float(acct.equity),
            "cash": float(acct.cash),
            "portfolio_value": float(acct.portfolio_value),
            "unrealized_pl": float(getattr(acct, "unrealized_pl", 0) or 0),
            "last_equity": float(getattr(acct, "last_equity", acct.equity) or acct.equity),
        }
    except Exception as e:
        log.error(f"Alpaca account P&L failed: {e}")
        return {}


# ---------------------------------------------------------------------------
# Incremental bar download
# ---------------------------------------------------------------------------
def save_incremental_bars(tickers: list[str], today: str) -> int:
    """
    Download today's bar for each ticker and save to
    paper_trade/incremental_bars/{ticker}/{YYYY-MM-DD}.parquet.
    Returns count of successful saves.
    """
    saved = 0
    for ticker in tickers:
        try:
            df = yf.download(ticker, period="5d", interval="1d", progress=False, auto_adjust=True)
            if df.empty:
                log.warning(f"Incremental bar: no data for {ticker}")
                continue

            # Normalize column names for MultiIndex yfinance output
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0).str.lower()
            else:
                df.columns = [c.lower() if isinstance(c, str) else c for c in df.columns]

            df.index = pd.to_datetime(df.index)

            # Filter to today only
            today_dt = pd.Timestamp(today)
            today_bars = df[df.index.date == today_dt.date()]

            if today_bars.empty:
                # Use latest available bar (may be most recent trading day)
                today_bars = df.tail(1)

            ticker_dir = INCREMENTAL_DIR / ticker
            ticker_dir.mkdir(parents=True, exist_ok=True)
            out_path = ticker_dir / f"{today}.parquet"
            today_bars.to_parquet(out_path)
            log.debug(f"Saved incremental bar: {out_path}")
            saved += 1

        except Exception as e:
            log.warning(f"Incremental bar failed for {ticker}: {e}")

    log.info(f"Incremental bars saved: {saved}/{len(tickers)}")
    return saved


# ---------------------------------------------------------------------------
# Signals vs outcomes
# ---------------------------------------------------------------------------
def compute_signals_vs_outcomes(state: dict, today: str) -> pd.DataFrame:
    """
    Build a comparison DataFrame of predicted signals vs realized P&L.
    """
    signals_path = SIGNALS_DIR / f"{today}.json"
    signals = {}
    if signals_path.exists():
        for s in json.loads(signals_path.read_text()):
            signals[s["ticker"]] = s

    rows = []
    for trade in state.get("closed_trades", []):
        ticker = trade.get("ticker", "?")
        sig = signals.get(ticker, {})
        rows.append({
            "ticker": ticker,
            "date": today,
            "predicted_prob": sig.get("prob", None),
            "threshold": sig.get("threshold", None),
            "signal": sig.get("signal", None),
            "pipeline": sig.get("pipeline", None),
            "entry_price": trade.get("entry_price"),
            "exit_price": trade.get("exit_price"),
            "qty": trade.get("qty"),
            "pnl": trade.get("pnl"),
            "mode": trade.get("mode", MODE),
        })

    # Also add tickers that had signals but no trade (above-threshold but blocked by guardrails)
    for ticker, sig in signals.items():
        if not any(r["ticker"] == ticker for r in rows):
            rows.append({
                "ticker": ticker,
                "date": today,
                "predicted_prob": sig.get("prob"),
                "threshold": sig.get("threshold"),
                "signal": sig.get("signal"),
                "pipeline": sig.get("pipeline"),
                "entry_price": None,
                "exit_price": None,
                "qty": 0,
                "pnl": None,
                "mode": "NO_TRADE",
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Friday retrain trigger
# ---------------------------------------------------------------------------
def should_retrain(today: str, force: bool = False) -> bool:
    """Returns True if today is Friday (or force=True)."""
    if force:
        return True
    dt = date.fromisoformat(today)
    return dt.weekday() == 4  # 4 = Friday


def trigger_retrain(tickers: list[str], today: str) -> None:
    """
    Trigger rolling retrain for mastered tickers.
    Calls backtest_xgb_v10.py with --use-mythos-features so the refreshed
    mastery uses the same v10/v10_mythos pipeline that live signals discover.
    Compares new probability threshold vs prior to detect mastery loss.
    """
    if not tickers:
        log.info("Retrain: no tickers to retrain.")
        return

    log.info(f"Friday retrain: triggering for {len(tickers)} tickers...")

    v10_script = SCRIPTS_DIR / "backtest_xgb_v10.py"
    if not v10_script.exists():
        log.warning("backtest_xgb_v10.py not found — skipping retrain")
        return

    # Load prior thresholds (search v10 first, then v8/v7 for legacy comparison)
    prior: dict[str, float] = {}
    for d in [
        WORK / "backtests_xgb_v10",
        WORK / "backtests_xgb_v9",
        WORK / "backtests_xgb_v8",
        WORK / "backtests_xgb_v7",
    ]:
        if not d.exists():
            continue
        for meta_path in d.rglob("run_meta.json"):
            try:
                meta = json.loads(meta_path.read_text())
                t = meta.get("ticker")
                thresh = meta.get("strategy", {}).get("prob_threshold")
                if t and thresh and t not in prior:
                    prior[t] = thresh
            except Exception:
                pass

    mastery_changes = []

    for ticker in tickers[:5]:  # Limit to 5 per session to avoid timeout
        log.info(f"Retrain: {ticker}...")
        try:
            # 2026-05-21: --use-mythos-features removed (Mythos dropped per OC
            # audit rank #2). MYTHOS_DISABLED=1 is the default; the flag was a
            # no-op anyway. Re-add the flag AND `MYTHOS_DISABLED=0` env to
            # restore Mythos training.
            result = subprocess.run(
                [
                    sys.executable,
                    str(v10_script),
                    "--ticker", ticker,
                ],
                capture_output=True, text=True, timeout=900,
            )
            if result.returncode != 0:
                log.warning(f"Retrain failed for {ticker}: {result.stderr[-200:]}")
                continue

            # Check for mastery loss by reading new run_meta from v10 output dir
            new_meta_paths = list((WORK / "backtests_xgb_v10").glob(f"{ticker}_*/run_meta.json"))
            if new_meta_paths:
                latest = max(new_meta_paths, key=lambda p: p.stat().st_mtime)
                new_meta = json.loads(latest.read_text())
                new_thresh = new_meta.get("strategy", {}).get("prob_threshold")
                old_thresh = prior.get(ticker)
                if old_thresh and new_thresh and abs(new_thresh - old_thresh) > 0.05:
                    mastery_changes.append({
                        "ticker": ticker,
                        "old_threshold": old_thresh,
                        "new_threshold": new_thresh,
                        "change": new_thresh - old_thresh,
                    })
                    log.warning(
                        f"Mastery change: {ticker} threshold {old_thresh:.2f} → {new_thresh:.2f}"
                    )

        except subprocess.TimeoutExpired:
            log.warning(f"Retrain timed out for {ticker}")
        except Exception as e:
            log.error(f"Retrain error for {ticker}: {e}")

    if mastery_changes:
        changes_path = PAPER_DIR / f"mastery_changes_{today}.json"
        with open(changes_path, "w") as f:
            json.dump(mastery_changes, f, indent=2)
        log.warning(f"Mastery changes written → {changes_path}")


# ---------------------------------------------------------------------------
# Summary markdown
# ---------------------------------------------------------------------------
def write_summary(today: str, state: dict, fills: list[dict], outcomes_df: pd.DataFrame,
                  bars_saved: int, pnl_data: dict) -> Path:
    """Write daily summary markdown."""
    daily_path = DAILY_DIR / today
    daily_path.mkdir(parents=True, exist_ok=True)
    summary_path = daily_path / "summary.md"

    firing_count = len([r for _, r in outcomes_df.iterrows()
                        if r.get("signal") == 1]) if not outcomes_df.empty else 0
    trade_count = len(state.get("closed_trades", []))

    realized_pnl = state.get("realized_pnl", 0.0)
    pnl_str = f"${realized_pnl:+.2f}"
    if MODE == "LIVE_PAPER" and pnl_data:
        pnl_str += f" (Alpaca equity: ${pnl_data.get('equity', 0):.2f})"

    lines = [
        f"# Paper Trade Summary — {today}",
        "",
        f"**Mode:** {MODE}",
        f"**Signals generated:** {len(outcomes_df)} tickers",
        f"**Firing signals (prob > threshold):** {firing_count}",
        f"**Trades placed:** {trade_count}",
        f"**Realized P&L:** {pnl_str}",
        f"**Halted today:** {'YES' if state.get('halted') else 'NO'}",
        f"**Incremental bars saved:** {bars_saved}",
        "",
        "## Trades",
        "",
    ]

    if trade_count > 0:
        lines.append("| Ticker | Entry | Exit | Qty | P&L |")
        lines.append("|--------|-------|------|-----|-----|")
        for t in state.get("closed_trades", []):
            entry = t.get("entry_price", "N/A")
            exit_ = t.get("exit_price", "N/A")
            pnl = t.get("pnl", "N/A")
            entry_str = f"${float(entry):.2f}" if isinstance(entry, (int, float)) else str(entry)
            exit_str = f"${float(exit_):.2f}" if isinstance(exit_, (int, float)) else str(exit_)
            pnl_str2 = f"${float(pnl):+.2f}" if isinstance(pnl, (int, float)) else str(pnl)
            lines.append(f"| {t.get('ticker','?')} | {entry_str} | {exit_str} | {t.get('qty','?')} | {pnl_str2} |")
    else:
        lines.append("*No trades placed today.*")

    lines.extend([
        "",
        "## All Signals",
        "",
    ])

    if not outcomes_df.empty:
        lines.append("| Ticker | Prob | Threshold | Signal | P&L |")
        lines.append("|--------|------|-----------|--------|-----|")
        for _, row in outcomes_df.sort_values("predicted_prob", ascending=False).iterrows():
            prob_str = f"{row['predicted_prob']:.3f}" if pd.notna(row['predicted_prob']) else "N/A"
            thr_str = f"{row['threshold']:.2f}" if pd.notna(row.get('threshold')) else "N/A"
            sig_str = "BUY" if row.get("signal") == 1 else "NO-TRADE"
            pnl_val = row.get("pnl")
            pnl_val_str = f"${float(pnl_val):+.2f}" if pd.notna(pnl_val) and pnl_val is not None else "—"
            lines.append(
                f"| {row['ticker']} | {prob_str} | {thr_str} | {sig_str} | {pnl_val_str} |"
            )

    lines.extend([
        "",
        f"*Generated at {datetime.now(timezone.utc).isoformat()} UTC*",
    ])

    summary_path.write_text("\n".join(lines))
    log.info(f"Summary written → {summary_path}")
    return summary_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Paper trade ingest: fills + retrain")
    parser.add_argument("--date", default=None, help="Override date YYYY-MM-DD")
    parser.add_argument("--retrain", action="store_true", help="Force retrain even if not Friday")
    args = parser.parse_args()

    today = args.date or date.today().isoformat()
    log.info(f"=== INGEST {today} | mode={MODE} ===")

    state = load_state(today)

    # ----- 1. Fills / P&L -----
    daily_path = DAILY_DIR / today
    daily_path.mkdir(parents=True, exist_ok=True)

    pnl_data: dict = {}

    if MODE == "LIVE_PAPER":
        fills = _get_alpaca_fills_today(today)
        pnl_data = _get_alpaca_portfolio_pnl()
        # Merge fill prices back into state closed_trades where exit_price missing
        fill_lookup = {f["ticker"]: f for f in fills}
        for trade in state.get("closed_trades", []):
            ticker = trade.get("ticker")
            if ticker in fill_lookup and not trade.get("exit_price"):
                fill = fill_lookup[ticker]
                trade["exit_price"] = fill.get("avg_fill_price")
                entry = trade.get("entry_price", 0.0) or 0.0
                exit_ = trade.get("exit_price", 0.0) or 0.0
                trade["pnl"] = (float(exit_) - float(entry)) * int(trade.get("qty", 0))
    else:
        fills = [
            {
                "ticker": t.get("ticker"),
                "side": "sell",
                "qty": t.get("qty"),
                "avg_fill_price": t.get("exit_price"),
                "entry_price": t.get("entry_price"),
                "pnl": t.get("pnl"),
                "mode": "SIMULATED",
            }
            for t in state.get("closed_trades", [])
        ]

    # Realized P&L from trades
    if MODE == "SIMULATED":
        pnl_data["realized_pnl"] = sum(
            float(t.get("pnl") or 0) for t in state.get("closed_trades", [])
        )

    # Save fills CSV
    fills_path = daily_path / "fills.csv"
    pd.DataFrame(fills).to_csv(fills_path, index=False)
    log.info(f"Fills saved → {fills_path} ({len(fills)} rows)")

    # Save P&L JSON
    pnl_path = daily_path / "pnl.json"
    with open(pnl_path, "w") as f:
        json.dump({
            "date": today,
            "mode": MODE,
            "realized_pnl": state.get("realized_pnl", 0.0),
            "trade_count": len(state.get("closed_trades", [])),
            **pnl_data,
        }, f, indent=2)
    log.info(f"P&L saved → {pnl_path}")

    # ----- 2. Signals vs outcomes -----
    outcomes_df = compute_signals_vs_outcomes(state, today)
    outcomes_path = daily_path / "signals_vs_outcomes.csv"
    outcomes_df.to_csv(outcomes_path, index=False)
    log.info(f"Signals vs outcomes → {outcomes_path} ({len(outcomes_df)} rows)")

    # ----- 3. Incremental bars -----
    # All tickers that had signals today (whether they traded or not)
    signals_path = SIGNALS_DIR / f"{today}.json"
    all_signal_tickers: list[str] = []
    if signals_path.exists():
        sigs = json.loads(signals_path.read_text())
        all_signal_tickers = [s["ticker"] for s in sigs]

    bars_saved = save_incremental_bars(all_signal_tickers, today)

    # ----- 3a. Account snapshot + intraday 1Min bars -----
    for stem, args_extra in [
        ("persist_account_snapshots", ["--phase", "close"]),
        ("persist_intraday_bars", ["--date", today]),
    ]:
        script_path = SCRIPTS_DIR / f"{stem}.py"
        if not script_path.exists():
            log.warning(f"[wiring] {stem}.py missing — skipping")
            continue
        try:
            r = subprocess.run(
                [sys.executable, str(script_path)] + args_extra,
                capture_output=True, text=True, timeout=300,
            )
            log.info(f"[wiring] {stem} rc={r.returncode}")
            if r.returncode != 0 and r.stderr:
                log.warning(f"[wiring] {stem} stderr tail: {r.stderr[-500:]}")
        except Exception as e:
            log.warning(f"[wiring] {stem} subprocess failed: {e}")

    # ----- 3b. Mythos embedding refresh -----
    # 2026-05-21: Mythos transformer dropped per OC audit rank #2. The refresh
    # is skipped when MYTHOS_DISABLED=1 (default) to avoid wasting compute on
    # zero-fill embeddings. Re-enable with `export MYTHOS_DISABLED=0` if a
    # future Mythos checkpoint is restored.
    _mythos_disabled_raw = os.environ.get("MYTHOS_DISABLED", "1").strip().lower()
    _mythos_disabled = _mythos_disabled_raw in ("1", "true", "yes", "on")
    refresh_path = SCRIPTS_DIR / "refresh_mythos_embeddings.py"
    if _mythos_disabled:
        log.info(
            "Mythos refresh skipped — MYTHOS_DISABLED=1 (OC audit drop 2026-05-21)."
        )
    elif refresh_path.exists() and all_signal_tickers:
        try:
            subprocess.run(
                [
                    sys.executable,
                    str(refresh_path),
                    "--date", today,
                    "--tickers", ",".join(all_signal_tickers),
                ],
                capture_output=True, timeout=900,
            )
            log.info(f"Mythos refresh dispatched for {len(all_signal_tickers)} tickers")
        except Exception as e:
            log.warning(f"Mythos refresh failed: {e}")
    elif not refresh_path.exists():
        log.warning("refresh_mythos_embeddings.py not found — skipping mythos refresh")

    # ----- 4. Summary markdown -----
    write_summary(today, state, fills, outcomes_df, bars_saved, pnl_data)

    # ----- 5. Retrain if Friday -----
    if should_retrain(today, force=args.retrain):
        log.info(f"Friday retrain triggered (today={today})")
        retrain_tickers = all_signal_tickers or list(state.get("positions", {}).keys())
        trigger_retrain(retrain_tickers, today)
    else:
        log.info(f"Not Friday — skipping full retrain (pass --retrain to force)")

    log.info(f"=== INGEST COMPLETE {today} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
