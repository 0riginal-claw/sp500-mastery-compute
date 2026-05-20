"""
intraday_backtest.py — Replay historical 1-min bars through the 4-strategy ensemble.

NO-LOOKAHEAD: at bar t, score() receives only bars[0..t]. Entry happens at
bars[t+1].open (next-bar open). Exit logic walks subsequent bars; if a bar's
high >= target → exit at target; elif low <= stop → exit at stop (conservative,
SL-first when both touched same bar); elif arming threshold crossed → trailing
stop is set to high - 1*ATR14; elif trail armed and low <= trail_stop → exit;
else force-flat at 15:55 ET (close price).

Usage:
    python intraday_backtest.py --ticker AAPL --days 30
    python intraday_backtest.py --tickers AAPL,MSFT,NVDA --days 30
    python intraday_backtest.py --ticker AAPL --days 30 --output /tmp/aapl.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytz

WORK = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools/s&p500-ticker-mastery"
)
SCRIPTS_DIR = WORK / "scripts"
BACKTESTS_DIR = WORK / "paper_trade" / "intraday_backtests"
BACKTESTS_DIR.mkdir(parents=True, exist_ok=True)

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from intraday_strategies import STRATEGIES, compute_atr14  # noqa: E402

ET = pytz.timezone("America/New_York")
RTH_OPEN = dtime(9, 30)
RTH_CLOSE = dtime(16, 0)
FORCE_FLAT = dtime(15, 55)

SLIPPAGE_BPS = 5
PER_TRADE_NOTIONAL = 500.0

PARQUET_ROOT = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/claudes test/data/timeframes/S&P500 5 Year Historical Data"
    "/Minutes TimeFrames/1Min_merged"
)

logger = logging.getLogger("intraday_backtest")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _ch = logging.StreamHandler()
    _ch.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    logger.addHandler(_ch)


# ── data loading ────────────────────────────────────────────────────────────
def load_bars(ticker: str, days: int) -> tuple[pd.DataFrame, str]:
    """Try local parquet first; fall back to yfinance; else synthetic.

    Returns (bars_df, source_label).
    """
    # 1) parquet
    p = PARQUET_ROOT / f"{ticker}.parquet"
    if p.exists():
        try:
            df = pd.read_parquet(
                p,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(ET)
            df = df.sort_values("timestamp").reset_index(drop=True)
            cutoff = datetime.now(ET) - timedelta(days=days + 5)
            df = df[df["timestamp"] >= cutoff].reset_index(drop=True)
            if len(df) > 0:
                return df, "parquet"
        except Exception as e:  # pragma: no cover
            logger.warning("parquet load failed for %s: %s", ticker, e)

    # 2) yfinance
    try:
        import yfinance as yf  # type: ignore

        # yfinance limits 1m to 7 days per request and total 30 days lookback.
        # Pull in 7-day chunks to maximise window.
        chunks = []
        for week_off in range(0, max(1, math.ceil(days / 7))):
            end = datetime.now(ET) - timedelta(days=week_off * 7)
            start = end - timedelta(days=7)
            try:
                df = yf.download(
                    ticker,
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    interval="1m",
                    progress=False,
                    prepost=False,
                    auto_adjust=False,
                )
            except Exception as e:  # pragma: no cover
                logger.warning("yf chunk failed: %s", e)
                continue
            if df is None or len(df) == 0:
                continue
            df = df.reset_index()
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
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [
                    c[0].lower() if isinstance(c, tuple) else str(c).lower()
                    for c in df.columns
                ]
                df = df.rename(columns={"datetime": "timestamp"})
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(ET)
            chunks.append(df)
        if chunks:
            full = pd.concat(chunks).sort_values("timestamp").drop_duplicates(
                subset=["timestamp"]
            ).reset_index(drop=True)
            return full, "yfinance"
    except Exception as e:
        logger.warning("yfinance unavailable: %s", e)

    # 3) synthetic
    logger.warning("falling back to synthetic random walk for %s", ticker)
    return _synth(ticker, days), "synthetic"


def _synth(ticker: str, days: int) -> pd.DataFrame:
    """Random-walk synthetic for harness testing."""
    rng = np.random.default_rng(seed=hash(ticker) & 0xFFFFFFFF)
    sessions = []
    today = datetime.now(ET).date()
    for d_off in range(days):
        sess_date = today - timedelta(days=d_off + 1)
        if sess_date.weekday() >= 5:  # weekend
            continue
        ts = pd.date_range(
            ET.localize(datetime.combine(sess_date, RTH_OPEN)),
            periods=390,
            freq="1min",
        )
        base = 100 + rng.standard_normal() * 5
        rets = rng.standard_normal(390) * 0.001
        prices = base * np.exp(np.cumsum(rets))
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "open": prices + rng.standard_normal(390) * 0.05,
                "close": prices,
                "volume": rng.integers(5_000, 50_000, 390),
            }
        )
        df["high"] = df[["open", "close"]].max(axis=1) + rng.uniform(0, 0.2, 390)
        df["low"] = df[["open", "close"]].min(axis=1) - rng.uniform(0, 0.2, 390)
        sessions.append(df)
    if not sessions:
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
    return pd.concat(sessions).sort_values("timestamp").reset_index(drop=True)


def _filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    t = df["timestamp"].dt.time
    mask = (t >= RTH_OPEN) & (t < RTH_CLOSE)
    return df.loc[mask].reset_index(drop=True)


# ── simulation core ─────────────────────────────────────────────────────────
def _simulate_session(
    session: pd.DataFrame, strategy_id: str, prev_close: float | None
) -> list[dict]:
    """Walk session bar-by-bar. Return list of trade records (≤1 per strategy)."""
    score_fn = STRATEGIES[strategy_id]
    trades: list[dict] = []
    position: Optional[dict] = None  # {entry, entry_idx, target, stop, trail_arm, trail_stop, armed}

    for i in range(len(session) - 1):
        bar = session.iloc[i]
        next_bar = session.iloc[i + 1]
        cur_time = bar["timestamp"].time()

        # force-flat at 15:55
        if cur_time >= FORCE_FLAT and position is not None:
            exit_price = float(bar["close"]) * (1 - SLIPPAGE_BPS / 10_000)
            trades.append(_close(position, exit_price, bar["timestamp"], "force_flat"))
            position = None
            continue

        # exit logic on the OPEN bar (after entry placed)
        if position is not None:
            high = float(bar["high"])
            low = float(bar["low"])
            # SL first (conservative when both touched)
            if low <= position["stop"]:
                px = position["stop"] * (1 - SLIPPAGE_BPS / 10_000)
                trades.append(_close(position, px, bar["timestamp"], "stop"))
                position = None
                continue
            if high >= position["target"]:
                px = position["target"] * (1 - SLIPPAGE_BPS / 10_000)
                trades.append(_close(position, px, bar["timestamp"], "target"))
                position = None
                continue
            # arm trailing
            if not position["armed"] and high >= position["trail_arm"]:
                position["armed"] = True
                position["trail_stop"] = high - position["atr"]
            if position["armed"]:
                # update trail to follow new highs
                position["trail_stop"] = max(
                    position["trail_stop"], high - position["atr"]
                )
                if low <= position["trail_stop"]:
                    px = position["trail_stop"] * (1 - SLIPPAGE_BPS / 10_000)
                    trades.append(_close(position, px, bar["timestamp"], "trail"))
                    position = None
                    continue

        # one trade per session per strategy
        if position is not None or len(trades) > 0:
            continue

        # call score with bars[0..i]
        try:
            rec = score_fn(
                session.iloc[: i + 1],
                ticker=session.attrs.get("ticker", "UNKNOWN"),
                params={"prev_close": prev_close},
            )
        except Exception as e:  # pragma: no cover
            logger.warning("score crash %s bar %d: %s", strategy_id, i, e)
            continue
        if rec.get("signal", 0) != 1:
            continue

        # entry on next bar open
        entry_px_raw = float(next_bar["open"])
        entry_px = entry_px_raw * (1 + SLIPPAGE_BPS / 10_000)
        atr = float(rec["meta"].get("atr14") or compute_atr14(session.iloc[: i + 1]))
        if atr <= 0:
            continue
        position = {
            "ticker": session.attrs.get("ticker", "UNKNOWN"),
            "strategy_id": strategy_id,
            "entry_time": next_bar["timestamp"],
            "entry_idx": i + 1,
            "entry": entry_px,
            "target": float(rec["target"]),
            "stop": float(rec["stop"]),
            "trail_arm": float(rec["trailing_stop_arm_at"]),
            "atr": atr,
            "armed": False,
            "trail_stop": None,
            "prob": float(rec["prob"]),
            "reason": rec["reason"],
        }

    # close at session end if still open
    if position is not None:
        last_bar = session.iloc[-1]
        exit_px = float(last_bar["close"]) * (1 - SLIPPAGE_BPS / 10_000)
        trades.append(_close(position, exit_px, last_bar["timestamp"], "session_end"))

    return trades


def _close(pos: dict, exit_px: float, exit_ts, reason: str) -> dict:
    pnl_pct = (exit_px - pos["entry"]) / pos["entry"]
    hold_min = float(
        (pd.Timestamp(exit_ts) - pd.Timestamp(pos["entry_time"])).total_seconds() / 60
    )
    return {
        "ticker": pos["ticker"],
        "strategy_id": pos["strategy_id"],
        "entry_time": pd.Timestamp(pos["entry_time"]).isoformat(),
        "exit_time": pd.Timestamp(exit_ts).isoformat(),
        "entry": round(pos["entry"], 4),
        "exit": round(exit_px, 4),
        "target": round(pos["target"], 4),
        "stop": round(pos["stop"], 4),
        "atr": round(pos["atr"], 4),
        "prob": round(pos["prob"], 4),
        "pnl_pct": round(pnl_pct, 6),
        "hold_min": round(hold_min, 1),
        "exit_reason": reason,
    }


# ── aggregation ─────────────────────────────────────────────────────────────
def _aggregate(trades: list[dict]) -> dict:
    if not trades:
        return dict(
            n_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            total_return_pct=0.0,
            sharpe=0.0,
            max_dd_pct=0.0,
            avg_hold_min=0.0,
        )
    p = np.array([t["pnl_pct"] for t in trades])
    wins = int((p > 0).sum())
    n = len(p)
    wr = wins / n
    gp = float(p[p > 0].sum())
    gl = float(-p[p < 0].sum())
    pf = (gp / gl) if gl > 0 else (float("inf") if gp > 0 else 0.0)
    total = float(p.sum())
    sharpe = float(p.mean() / p.std()) if p.std() > 0 else 0.0
    eq = np.cumsum(p)
    dd = float((eq - np.maximum.accumulate(eq)).min())
    hold = float(np.mean([t["hold_min"] for t in trades]))
    return dict(
        n_trades=n,
        win_rate=round(wr, 4),
        profit_factor=round(pf, 4) if pf != float("inf") else 99.99,
        total_return_pct=round(total, 6),
        sharpe=round(sharpe, 4),
        max_dd_pct=round(dd, 6),
        avg_hold_min=round(hold, 1),
    )


def backtest_ticker(ticker: str, days: int) -> dict:
    bars, source = load_bars(ticker, days)
    bars = _filter_rth(bars)
    if bars.empty:
        return dict(
            ticker=ticker,
            window=dict(trading_days=0),
            per_strategy={},
            data_source=source,
            bars_loaded=0,
        )
    bars.attrs["ticker"] = ticker
    bars["session"] = bars["timestamp"].dt.date

    per_strat: dict[str, list[dict]] = {sid: [] for sid in STRATEGIES.keys()}
    sessions = sorted(bars["session"].unique())
    # compute per-session prev_close
    sess_close = {s: float(bars[bars["session"] == s]["close"].iloc[-1]) for s in sessions}

    for idx, s in enumerate(sessions):
        prev_close = sess_close[sessions[idx - 1]] if idx > 0 else None
        sub = bars[bars["session"] == s].reset_index(drop=True)
        sub.attrs["ticker"] = ticker
        if len(sub) < 20:
            continue
        for sid in STRATEGIES.keys():
            sub.attrs["ticker"] = ticker
            trades = _simulate_session(sub, sid, prev_close)
            per_strat[sid].extend(trades)

    # write per-strategy jsonl
    label = f"{ticker}_{sessions[0]}_{sessions[-1]}"
    for sid, trs in per_strat.items():
        out = BACKTESTS_DIR / f"{label}_{sid}.jsonl"
        with out.open("w") as f:
            for t in trs:
                f.write(json.dumps(t) + "\n")

    summary = {sid: _aggregate(trs) for sid, trs in per_strat.items()}
    all_trades = [t for trs in per_strat.values() for t in trs]
    summary_ensemble = _aggregate(all_trades)

    return dict(
        ticker=ticker,
        window=dict(
            start=str(sessions[0]),
            end=str(sessions[-1]),
            trading_days=len(sessions),
        ),
        per_strategy=summary,
        ensemble_naive_avg=summary_ensemble,
        data_source=source,
        bars_loaded=int(len(bars)),
    )


# ── cli ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", type=str, default=None)
    ap.add_argument("--tickers", type=str, default=None)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    if args.tickers:
        tlist = [t.strip().upper() for t in args.tickers.split(",")]
    elif args.ticker:
        tlist = [args.ticker.strip().upper()]
    else:
        ap.error("provide --ticker or --tickers")
        return

    t0 = time.time()
    results = {}
    for t in tlist:
        logger.info("backtesting %s (%d days)", t, args.days)
        results[t] = backtest_ticker(t, args.days)

    elapsed = time.time() - t0
    out_obj = {
        "results": results,
        "elapsed_s": round(elapsed, 2),
        "args": vars(args),
    }
    text = json.dumps(out_obj, indent=2)
    if args.output:
        Path(args.output).write_text(text)
        logger.info("wrote %s", args.output)
    print(text)


if __name__ == "__main__":
    main()
