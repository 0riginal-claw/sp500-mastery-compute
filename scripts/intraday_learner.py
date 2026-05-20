"""
intraday_learner.py — Thompson Sampling strategy weight allocator + weekly retrain.

Reads `paper_trade/strategy_outcomes/*.jsonl` over last 30 days, computes
per-(ticker,strategy) Beta(alpha=wins+1, beta=losses+1) posterior, samples once
per day to set weight allocation for next session. Also derives per-pair
prob_threshold via grid search on rolling Sharpe.

Commands:
  python intraday_learner.py allocate
      → paper_trade/intraday_weights/weights_<DATE>.json

  python intraday_learner.py retrain
      → updates mastery_files/{TICKER}_INTRADAY_{strategy}_mastered.md

  python intraday_learner.py report
      → prints rolling 30d (ticker,strategy) stats table
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

WORK = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools/s&p500-ticker-mastery"
)
OUTCOMES_DIR = WORK / "paper_trade" / "strategy_outcomes"
WEIGHTS_DIR = WORK / "paper_trade" / "intraday_weights"
MASTERY_DIR = WORK / "mastery_files"
LOGS_DIR = WORK / "logs"

for _d in (WEIGHTS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("intraday_learner")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _fh = logging.FileHandler(LOGS_DIR / "intraday_learner.log")
    _ch = logging.StreamHandler()
    _fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(module)s] %(message)s"
    )
    _fh.setFormatter(_fmt)
    _ch.setFormatter(_fmt)
    logger.addHandler(_fh)
    logger.addHandler(_ch)

DEFAULT_STRATS = [
    "orb_15min",
    "vwap_mean_revert",
    "momentum_15min",
    "gap_fade_open",
]


# ── data loading ────────────────────────────────────────────────────────────
def load_outcomes(window_days: int = 30) -> pd.DataFrame:
    """Concat last N days of jsonl outcomes. Returns empty df if none.

    Columns expected per record: ts, ticker, strategy_id, side, qty, entry,
    target, stop, prob, reason, order_id, mode, plus optional pnl_pct
    (added by an ingest step downstream).
    """
    today = datetime.utcnow().date()
    rows = []
    for d_off in range(window_days + 1):
        d = today - timedelta(days=d_off)
        p = OUTCOMES_DIR / f"{d.isoformat()}.jsonl"
        if not p.exists():
            continue
        for ln in p.read_text().splitlines():
            try:
                r = json.loads(ln)
            except Exception:
                continue
            # Skip non-trade events (flatten markers, etc.)
            if r.get("event") or r.get("ticker") is None:
                continue
            r.setdefault("pnl_pct", None)
            rows.append(r)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    return df


def per_pair_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per (ticker, strategy_id) aggregates. Treats null pnl as 0 trades counted.

    Returns columns: ticker, strategy_id, n, wins, losses, win_rate,
    pnl_sum, pf, sharpe, avg_hold_min (approx — uses 0 if no data).
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "strategy_id",
                "n",
                "wins",
                "losses",
                "win_rate",
                "pnl_sum",
                "pf",
                "sharpe",
            ]
        )
    df = df[df["pnl_pct"].notna()].copy() if "pnl_pct" in df.columns else df
    if df.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "strategy_id",
                "n",
                "wins",
                "losses",
                "win_rate",
                "pnl_sum",
                "pf",
                "sharpe",
            ]
        )

    def _agg(g: pd.DataFrame) -> pd.Series:
        p = pd.to_numeric(g["pnl_pct"], errors="coerce").dropna()
        if len(p) == 0:
            return pd.Series(
                dict(n=0, wins=0, losses=0, win_rate=0.0, pnl_sum=0.0, pf=0.0, sharpe=0.0)
            )
        wins = int((p > 0).sum())
        losses = int((p <= 0).sum())
        n = int(len(p))
        wr = wins / n if n else 0.0
        gp = float(p[p > 0].sum())
        gl = float(-p[p < 0].sum())
        pf = (gp / gl) if gl > 0 else float("inf") if gp > 0 else 0.0
        sharpe = float(p.mean() / p.std()) if p.std() > 0 else 0.0
        return pd.Series(
            dict(
                n=n,
                wins=wins,
                losses=losses,
                win_rate=wr,
                pnl_sum=float(p.sum()),
                pf=pf,
                sharpe=sharpe,
            )
        )

    return (
        df.groupby(["ticker", "strategy_id"])
        .apply(_agg)
        .reset_index()
    )


# ── Thompson sampling allocator ─────────────────────────────────────────────
def thompson_allocate(
    stats: pd.DataFrame,
    strategies: list[str] | None = None,
    min_weight: float = 0.10,
    max_weight: float = 0.60,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Sample Beta(wins+1, losses+1) per (ticker, strategy), average across
    tickers, normalize. Clipped to [min_weight, max_weight] before re-norm.
    Uniform prior if stats empty.
    """
    rng = rng or np.random.default_rng()
    strats = strategies or DEFAULT_STRATS

    if stats is None or stats.empty:
        n = len(strats) or 1
        return {s: 1.0 / n for s in strats}

    samples_by_strat: dict[str, list[float]] = defaultdict(list)
    for _, row in stats.iterrows():
        sid = str(row["strategy_id"])
        if sid not in strats:
            continue
        a = float(row.get("wins", 0)) + 1.0
        b = float(row.get("losses", 0)) + 1.0
        samples_by_strat[sid].append(float(rng.beta(a, b)))

    raw = {}
    for s in strats:
        vals = samples_by_strat.get(s, [])
        raw[s] = float(np.mean(vals)) if vals else 0.5  # uniform prior fallback

    # normalize
    total = sum(raw.values()) or 1.0
    w = {k: v / total for k, v in raw.items()}

    # clip + renormalize
    w = {k: float(np.clip(v, min_weight, max_weight)) for k, v in w.items()}
    total = sum(w.values()) or 1.0
    w = {k: v / total for k, v in w.items()}
    return w


# ── threshold retrain ───────────────────────────────────────────────────────
def retrain_threshold(
    ticker: str, strategy_id: str, df: pd.DataFrame
) -> dict[str, Any]:
    """Grid search prob_threshold in [0.3..0.9] maximizing Sharpe."""
    sub = df[
        (df["ticker"] == ticker) & (df["strategy_id"] == strategy_id)
    ].copy()
    sub = sub[sub["pnl_pct"].notna()] if "pnl_pct" in sub.columns else sub
    if len(sub) < 5:
        return dict(prob_threshold=0.6, sharpe=0.0, wr=0.0, pf=0.0, n=int(len(sub)))

    best = dict(prob_threshold=0.6, sharpe=-1e9, wr=0.0, pf=0.0, n=0)
    for thr in np.arange(0.3, 0.91, 0.05):
        m = sub[pd.to_numeric(sub["prob"], errors="coerce") >= thr]
        if len(m) < 3:
            continue
        p = pd.to_numeric(m["pnl_pct"], errors="coerce").dropna()
        if len(p) < 3 or p.std() <= 0:
            continue
        sh = float(p.mean() / p.std())
        if sh > best["sharpe"]:
            wins = int((p > 0).sum())
            losses = int((p <= 0).sum())
            gp = float(p[p > 0].sum())
            gl = float(-p[p < 0].sum())
            pf = (gp / gl) if gl > 0 else float("inf") if gp > 0 else 0.0
            best = dict(
                prob_threshold=round(float(thr), 3),
                sharpe=round(sh, 4),
                wr=round(wins / len(p), 4),
                pf=round(pf, 4) if pf != float("inf") else 99.99,
                n=int(len(p)),
            )
    return best


def write_mastery_file(
    ticker: str, strategy_id: str, params: dict, metrics: dict
) -> Path:
    path = (
        MASTERY_DIR / f"{ticker}_INTRADAY_{strategy_id}_mastered.md"
    )
    today = date.today().isoformat()
    n = metrics.get("n", 0)
    body = (
        f"# {ticker} — INTRADAY {strategy_id} Mastered ({today})\n\n"
        f"## Metrics\n"
        f"- n trades: {n}\n"
        f"- WR: {metrics.get('wr', 0):.4f}\n"
        f"- PF: {metrics.get('pf', 0):.4f}\n"
        f"- Sharpe: {metrics.get('sharpe', 0):.4f}\n\n"
        f"## Pipeline\n"
        f"- intraday_engine.py + strategy `{strategy_id}`\n"
        f"- TP/SL: 2.0/1.0 ATR · trailing arm 1.0 ATR · flatten 15:55 ET\n"
        f"- Window: last 30 days of strategy_outcomes\n\n"
        f"## Result\n```json\n{json.dumps({**params, 'ticker': ticker, 'strategy_id': strategy_id, **metrics}, indent=2)}\n```\n"
    )
    path.write_text(body)
    return path


# ── reporting ───────────────────────────────────────────────────────────────
def print_report() -> None:
    df = load_outcomes(30)
    if df.empty:
        print("no outcomes in last 30 days")
        return
    stats = per_pair_stats(df)
    if stats.empty:
        print("no pnl-attributed trades yet (run intraday_engine_ingest first)")
        return
    print(stats.to_string(index=False))


# ── main ────────────────────────────────────────────────────────────────────
def cmd_allocate(args: argparse.Namespace) -> None:
    df = load_outcomes(30)
    stats = per_pair_stats(df) if not df.empty else pd.DataFrame()
    weights = thompson_allocate(stats)
    today = date.today().isoformat()
    out = WEIGHTS_DIR / f"weights_{today}.json"
    out.write_text(json.dumps(weights, indent=2))
    print(json.dumps({"date": today, "weights": weights, "out": str(out)}, indent=2))


def cmd_retrain(args: argparse.Namespace) -> None:
    df = load_outcomes(30)
    if df.empty:
        print("no outcomes — skipping retrain")
        return
    pairs = df.groupby(["ticker", "strategy_id"]).size().reset_index()
    written = []
    for _, row in pairs.iterrows():
        ticker = str(row["ticker"])
        sid = str(row["strategy_id"])
        m = retrain_threshold(ticker, sid, df)
        p = write_mastery_file(ticker, sid, dict(window_days=30), m)
        written.append(str(p))
    print(json.dumps({"written": written}, indent=2))


def cmd_report(args: argparse.Namespace) -> None:
    print_report()


def main() -> None:
    ap = argparse.ArgumentParser(description="intraday TS allocator + retrain")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("allocate")
    sub.add_parser("retrain")
    sub.add_parser("report")
    args = ap.parse_args()
    if args.cmd == "allocate":
        cmd_allocate(args)
    elif args.cmd == "retrain":
        cmd_retrain(args)
    elif args.cmd == "report":
        cmd_report(args)


if __name__ == "__main__":
    main()
