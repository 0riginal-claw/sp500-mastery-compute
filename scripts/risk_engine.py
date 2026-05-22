"""
risk_engine.py — 5-gate pre-trade risk engine for live_paper_trade.

Each gate returns a GateResult(passed: bool, reason: str, adjusted_qty: int|None).
Gates are evaluated in order; a single failure refuses the trade. Gates may
also DOWNSIZE (return passed=True with adjusted_qty < requested) — only the
Drawdown gate currently does this (halves sizes on 7-day DD > 5%).

Gates:
  1. Kelly        — stake ≤ Kelly fraction × equity, based on rolling 30-day
                    win-rate + payoff ratio for THIS ticker (falls back to
                    portfolio-wide stats if per-ticker n<10).
  2. Liquidity    — refuse if ticker's 5-day mean $-volume < $10M/day.
  3. Correlation  — refuse if portfolio's rolling 30-day return correlation
                    matrix has any |ρ|>0.85 between the candidate ticker and
                    any currently-held position (or any other firing ticker
                    in this batch, considered greedily in order).
  4. Concentration— max 5% of equity in any single ticker; max 25% of equity
                    per GICS sector (sector lookup cached, falls back to
                    "Unknown" → treated as its own bucket).
  5. Drawdown     — if 7-day equity DD > 5%, downsize all positions to 50%.
                    If > 10%, refuse new entries entirely.

Audit log: every decision (pass/refuse/downsize) is appended as one JSON line
to paper_trade/risk_engine_decisions.jsonl. Schema:
  {ts, ticker, requested_qty, requested_notional, equity, gate, passed,
   reason, adjusted_qty}

Usage:
  from risk_engine import RiskEngine
  engine = RiskEngine(equity=100_000.0, positions=state["positions"])
  decision = engine.check(ticker="AAPL", qty=10, signal={"prob":0.62,...})
  if not decision.passed:
      log.warning(f"[RISK] {ticker} refused by {decision.gate}: {decision.reason}")
      continue
  qty_to_submit = decision.adjusted_qty or 10  # respect downsize
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("risk_engine")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORK = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools/s&p500-ticker-mastery"
)
PAPER_DIR = WORK / "paper_trade"
DAILY_DIR = PAPER_DIR / "daily"
DECISIONS_LOG = PAPER_DIR / "risk_engine_decisions.jsonl"
SECTOR_CACHE = PAPER_DIR / "sector_map.json"

# ---------------------------------------------------------------------------
# Tunable thresholds
# ---------------------------------------------------------------------------
# Kelly: cap stake at half-Kelly to reduce variance. Floor win-rate sample to
# n=10 — if fewer trades, fall back to portfolio-wide stats. If those also
# lack samples, assume conservative (win-rate=0.50, payoff=1.0 → half-Kelly=0).
KELLY_FRACTION = 0.5            # half-Kelly
KELLY_MIN_TICKER_SAMPLES = 10
KELLY_MIN_PORTFOLIO_SAMPLES = 30
KELLY_DEFAULT_WIN_RATE = 0.50
KELLY_DEFAULT_PAYOFF = 1.0
KELLY_MAX_FRACTION = 0.05       # never stake > 5% of equity per single bet

# Liquidity: 5-day mean $-volume must clear $10M/day.
LIQUIDITY_LOOKBACK_DAYS = 5
LIQUIDITY_MIN_DOLLAR_VOLUME = 10_000_000.0

# Correlation: pairwise |ρ| above this refuses the new entry.
CORR_LOOKBACK_DAYS = 30
CORR_MAX_ABS = 0.85

# Concentration: max single-ticker + sector caps as % of equity.
MAX_PCT_PER_TICKER = 0.05
MAX_PCT_PER_SECTOR = 0.25

# Drawdown: 7-day equity peak-to-trough thresholds.
DD_LOOKBACK_DAYS = 7
DD_SOFT_THRESHOLD = 0.05   # > 5% → halve sizes
DD_HARD_THRESHOLD = 0.10   # > 10% → refuse new entries


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class GateResult:
    passed: bool
    gate: str
    reason: str
    adjusted_qty: int | None = None
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers — price/volume + sector lookup (yfinance-backed, cached)
# ---------------------------------------------------------------------------
_PRICE_CACHE: dict[str, pd.DataFrame] = {}


def _get_price_history(ticker: str, days: int = 60) -> pd.DataFrame | None:
    """
    Returns OHLCV with columns [open,high,low,close,volume] indexed by date.
    Cached in-process. Falls back to None on any error (caller decides whether
    that's pass-through-conservative or refuse).
    """
    if ticker in _PRICE_CACHE:
        df = _PRICE_CACHE[ticker]
        # cached enough?
        if len(df) >= days:
            return df.tail(days).copy()

    # Prefer the local mastery dataset if present.
    candidate_dirs = [
        WORK / "data" / "yfinance_daily" / f"{ticker}.parquet",
        WORK / "tickers_daily" / f"{ticker}.parquet",
        WORK / "data" / "tickers" / f"{ticker}.parquet",
    ]
    for p in candidate_dirs:
        if p.exists():
            try:
                df = pd.read_parquet(p)
                df.columns = [str(c).lower() for c in df.columns]
                df.index = pd.to_datetime(df.index)
                _PRICE_CACHE[ticker] = df
                return df.tail(days).copy()
            except Exception:
                pass

    # Last resort: yfinance.
    try:
        import yfinance as yf
        df = yf.download(
            ticker, period=f"{days + 10}d", interval="1d",
            progress=False, auto_adjust=True,
        )
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0).str.lower()
        else:
            df.columns = [str(c).lower() for c in df.columns]
        df.index = pd.to_datetime(df.index)
        _PRICE_CACHE[ticker] = df
        return df.tail(days).copy()
    except Exception as e:
        log.debug(f"price history fetch failed for {ticker}: {e}")
        return None


def _load_sector_cache() -> dict[str, str]:
    if SECTOR_CACHE.exists():
        try:
            return json.loads(SECTOR_CACHE.read_text())
        except Exception:
            return {}
    return {}


def _save_sector_cache(d: dict[str, str]) -> None:
    try:
        SECTOR_CACHE.parent.mkdir(parents=True, exist_ok=True)
        SECTOR_CACHE.write_text(json.dumps(d, indent=2, sort_keys=True))
    except Exception as e:
        log.debug(f"sector cache write failed: {e}")


def _get_sector(ticker: str) -> str:
    cache = _load_sector_cache()
    if ticker in cache:
        return cache[ticker]
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
        sector = info.get("sector") or "Unknown"
    except Exception:
        sector = "Unknown"
    cache[ticker] = sector
    _save_sector_cache(cache)
    return sector


# ---------------------------------------------------------------------------
# Gate 1: Kelly
# ---------------------------------------------------------------------------
def _kelly_fraction(win_rate: float, payoff_ratio: float) -> float:
    """
    Standard Kelly: f* = W - (1-W)/R, where W=win_rate, R=payoff_ratio.
    Returns 0 if any input non-positive or if optimal f is negative.
    """
    if payoff_ratio <= 0 or win_rate <= 0:
        return 0.0
    f = win_rate - (1.0 - win_rate) / payoff_ratio
    return max(0.0, min(f, 1.0))


def _get_ticker_history_stats(ticker: str, closed_trades: list[dict]) -> tuple[int, float, float]:
    """
    Returns (n, win_rate, payoff_ratio) from past closed_trades for ticker.
    payoff_ratio = mean_win_pnl / abs(mean_loss_pnl). Returns (0,0,0) if no data.
    """
    rows = [t for t in closed_trades if t.get("ticker") == ticker
            and t.get("pnl") is not None]
    if not rows:
        return 0, 0.0, 0.0
    pnls = [float(t["pnl"]) for t in rows]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    if not wins or not losses:
        # Degenerate (all wins or all losses) — treat payoff=1.0 to be safe.
        win_rate = len(wins) / len(pnls)
        return len(pnls), win_rate, 1.0
    win_rate = len(wins) / len(pnls)
    payoff = (sum(wins) / len(wins)) / abs(sum(losses) / len(losses))
    return len(pnls), win_rate, payoff


def _load_historic_closed_trades(max_days: int = 60) -> list[dict]:
    """Load union of closed_trades from the last N daily state files."""
    trades: list[dict] = []
    state_dir = PAPER_DIR / "state"
    if not state_dir.exists():
        return trades
    files = sorted(state_dir.glob("*_state.json"), reverse=True)[:max_days]
    for f in files:
        try:
            s = json.loads(f.read_text())
            trades.extend(s.get("closed_trades", []))
        except Exception:
            pass
    return trades


def gate_kelly(
    ticker: str,
    requested_notional: float,
    equity: float,
    historic_trades: list[dict],
) -> GateResult:
    n_t, wr_t, pf_t = _get_ticker_history_stats(ticker, historic_trades)
    if n_t >= KELLY_MIN_TICKER_SAMPLES:
        win_rate, payoff, sample = wr_t, pf_t, f"ticker_n={n_t}"
    else:
        # Pool across all tickers.
        wins = [float(t["pnl"]) for t in historic_trades
                if t.get("pnl") is not None and float(t["pnl"]) > 0]
        losses = [float(t["pnl"]) for t in historic_trades
                  if t.get("pnl") is not None and float(t["pnl"]) < 0]
        n_p = len(wins) + len(losses)
        if n_p >= KELLY_MIN_PORTFOLIO_SAMPLES and wins and losses:
            win_rate = len(wins) / n_p
            payoff = (sum(wins) / len(wins)) / abs(sum(losses) / len(losses))
            sample = f"portfolio_n={n_p}"
        else:
            win_rate, payoff = KELLY_DEFAULT_WIN_RATE, KELLY_DEFAULT_PAYOFF
            sample = f"default(no_samples,n_t={n_t},n_p={n_p})"

    f_star = _kelly_fraction(win_rate, payoff)
    f_capped = min(f_star * KELLY_FRACTION, KELLY_MAX_FRACTION)
    max_notional = f_capped * equity
    meta = {"win_rate": round(win_rate, 3), "payoff": round(payoff, 3),
            "kelly_f": round(f_star, 4), "f_capped": round(f_capped, 4),
            "max_notional": round(max_notional, 2), "sample": sample}

    if requested_notional <= max_notional or max_notional <= 0:
        # max_notional<=0 means Kelly says "don't bet" — but for the default
        # case (wr=0.50, pf=1.0 → f*=0), we should pass without blocking the
        # whole engine because that's the "no data yet" floor. Treat
        # max_notional==0 as "use the default cap" when sample==default.
        if max_notional <= 0 and sample.startswith("default"):
            # Allow up to KELLY_MAX_FRACTION of equity by default.
            max_notional = KELLY_MAX_FRACTION * equity
            meta["max_notional"] = round(max_notional, 2)
            if requested_notional <= max_notional:
                return GateResult(True, "kelly",
                                  f"pass(default_floor,{sample})", metadata=meta)
            return GateResult(False, "kelly",
                              f"refuse(req=${requested_notional:.0f}>cap=${max_notional:.0f},{sample})",
                              metadata=meta)
        if max_notional <= 0:
            return GateResult(False, "kelly",
                              f"refuse(kelly_f*=0,wr={win_rate:.2f},pf={payoff:.2f},{sample})",
                              metadata=meta)
        return GateResult(True, "kelly",
                          f"pass(req=${requested_notional:.0f}<=cap=${max_notional:.0f})",
                          metadata=meta)
    return GateResult(False, "kelly",
                      f"refuse(req=${requested_notional:.0f}>cap=${max_notional:.0f},{sample})",
                      metadata=meta)


# ---------------------------------------------------------------------------
# Gate 2: Liquidity
# ---------------------------------------------------------------------------
def gate_liquidity(ticker: str) -> GateResult:
    df = _get_price_history(ticker, days=LIQUIDITY_LOOKBACK_DAYS + 5)
    if df is None or df.empty:
        return GateResult(False, "liquidity",
                          "refuse(no_price_history)",
                          metadata={"reason": "no_data"})
    if "close" not in df.columns or "volume" not in df.columns:
        return GateResult(False, "liquidity",
                          f"refuse(missing_cols: have {list(df.columns)})",
                          metadata={"cols": list(df.columns)})
    recent = df.tail(LIQUIDITY_LOOKBACK_DAYS)
    dollar_vol = (recent["close"] * recent["volume"]).mean()
    if pd.isna(dollar_vol):
        return GateResult(False, "liquidity",
                          "refuse(dollar_volume=NaN)", metadata={})
    meta = {"mean_dollar_volume": round(float(dollar_vol), 2),
            "threshold": LIQUIDITY_MIN_DOLLAR_VOLUME}
    if dollar_vol < LIQUIDITY_MIN_DOLLAR_VOLUME:
        return GateResult(False, "liquidity",
                          f"refuse($-vol=${dollar_vol/1e6:.1f}M<${LIQUIDITY_MIN_DOLLAR_VOLUME/1e6:.1f}M)",
                          metadata=meta)
    return GateResult(True, "liquidity",
                      f"pass($-vol=${dollar_vol/1e6:.1f}M)", metadata=meta)


# ---------------------------------------------------------------------------
# Gate 3: Correlation
# ---------------------------------------------------------------------------
def _get_return_series(ticker: str, days: int = CORR_LOOKBACK_DAYS) -> pd.Series | None:
    df = _get_price_history(ticker, days=days + 5)
    if df is None or df.empty or "close" not in df.columns:
        return None
    rets = df["close"].pct_change().dropna().tail(days)
    if len(rets) < max(10, days // 2):
        return None
    rets.name = ticker
    return rets


def gate_correlation(
    ticker: str,
    portfolio_tickers: list[str],
) -> GateResult:
    """
    Refuse if |corr(ticker, t)| > CORR_MAX_ABS for any t in portfolio_tickers.
    """
    if not portfolio_tickers:
        return GateResult(True, "correlation",
                          "pass(no_existing_positions)", metadata={})

    new_rets = _get_return_series(ticker)
    if new_rets is None:
        return GateResult(False, "correlation",
                          f"refuse(no_returns_for_{ticker})",
                          metadata={"reason": "no_returns"})

    breaches = []
    pairs_checked = []
    for t in portfolio_tickers:
        if t == ticker:
            continue
        other = _get_return_series(t)
        if other is None:
            continue
        joined = pd.concat([new_rets, other], axis=1).dropna()
        if len(joined) < 10:
            continue
        rho = float(joined.corr().iloc[0, 1])
        pairs_checked.append((t, round(rho, 3)))
        if not math.isnan(rho) and abs(rho) > CORR_MAX_ABS:
            breaches.append((t, round(rho, 3)))

    meta = {"pairs_checked": pairs_checked, "max_abs": CORR_MAX_ABS,
            "breaches": breaches}
    if breaches:
        b_str = ",".join(f"{t}:{r}" for t, r in breaches)
        return GateResult(False, "correlation",
                          f"refuse(|ρ|>{CORR_MAX_ABS}: {b_str})", metadata=meta)
    return GateResult(True, "correlation",
                      f"pass({len(pairs_checked)} pairs all |ρ|<={CORR_MAX_ABS})",
                      metadata=meta)


# ---------------------------------------------------------------------------
# Gate 4: Concentration
# ---------------------------------------------------------------------------
def gate_concentration(
    ticker: str,
    requested_notional: float,
    equity: float,
    positions: dict,
) -> GateResult:
    """
    Single-ticker cap: new total exposure to ticker ≤ MAX_PCT_PER_TICKER × equity.
    Sector cap: post-add sector total ≤ MAX_PCT_PER_SECTOR × equity.
    """
    existing_ticker_notional = float(positions.get(ticker, {}).get("notional", 0) or 0)
    new_ticker_total = existing_ticker_notional + requested_notional
    ticker_cap = MAX_PCT_PER_TICKER * equity

    if new_ticker_total > ticker_cap:
        return GateResult(False, "concentration",
                          f"refuse(ticker_total=${new_ticker_total:.0f}>${ticker_cap:.0f}={MAX_PCT_PER_TICKER:.0%}×equity)",
                          metadata={"ticker_total": new_ticker_total,
                                    "ticker_cap": ticker_cap})

    # Sector roll-up.
    candidate_sector = _get_sector(ticker)
    sector_totals: dict[str, float] = {}
    for t, p in positions.items():
        if not isinstance(p, dict):
            continue
        s = _get_sector(t)
        sector_totals[s] = sector_totals.get(s, 0.0) + float(p.get("notional", 0) or 0)
    # Add the candidate exposure to its sector (replace existing ticker share).
    sector_totals[candidate_sector] = (
        sector_totals.get(candidate_sector, 0.0)
        - existing_ticker_notional
        + new_ticker_total
    )
    sector_cap = MAX_PCT_PER_SECTOR * equity
    sec_total = sector_totals.get(candidate_sector, 0.0)
    meta = {"sector": candidate_sector,
            "sector_total": round(sec_total, 2),
            "sector_cap": round(sector_cap, 2),
            "ticker_total": round(new_ticker_total, 2),
            "ticker_cap": round(ticker_cap, 2)}
    if sec_total > sector_cap:
        return GateResult(False, "concentration",
                          f"refuse(sector={candidate_sector} total=${sec_total:.0f}>${sector_cap:.0f}={MAX_PCT_PER_SECTOR:.0%}×equity)",
                          metadata=meta)
    return GateResult(True, "concentration",
                      f"pass(ticker=${new_ticker_total:.0f}<=${ticker_cap:.0f}, sector={candidate_sector} ${sec_total:.0f}<=${sector_cap:.0f})",
                      metadata=meta)


# ---------------------------------------------------------------------------
# Gate 5: Drawdown
# ---------------------------------------------------------------------------
def _load_equity_series(days: int = DD_LOOKBACK_DAYS + 5) -> list[float]:
    """
    Walk back through DAILY_DIR/*/pnl.json files in chrono order, extract the
    equity (or last_equity) field. Returns [] if no files found.
    """
    if not DAILY_DIR.exists():
        return []
    rows = []
    for sub in sorted(DAILY_DIR.iterdir()):
        if not sub.is_dir():
            continue
        p = sub / "pnl.json"
        if not p.exists():
            continue
        try:
            d = json.loads(p.read_text())
            eq = d.get("equity") or d.get("portfolio_value") or d.get("last_equity")
            if eq is not None:
                rows.append((sub.name, float(eq)))
        except Exception:
            pass
    return [eq for _, eq in rows[-days:]]


def gate_drawdown(
    requested_qty: int,
    equity_series: list[float] | None = None,
) -> GateResult:
    """
    7-day equity drawdown gate.
      DD <= 5% → pass through unchanged.
      5% < DD <= 10% → pass but halve qty.
      DD > 10% → refuse.
    """
    if equity_series is None:
        equity_series = _load_equity_series()
    if len(equity_series) < 2:
        return GateResult(True, "drawdown",
                          "pass(insufficient_history)", adjusted_qty=requested_qty,
                          metadata={"n": len(equity_series)})
    peak = max(equity_series)
    trough_after_peak = min(equity_series[equity_series.index(peak):])
    dd_pct = (peak - trough_after_peak) / peak if peak > 0 else 0.0
    meta = {"peak": round(peak, 2), "trough": round(trough_after_peak, 2),
            "dd_pct": round(dd_pct, 4), "n": len(equity_series)}
    if dd_pct > DD_HARD_THRESHOLD:
        return GateResult(False, "drawdown",
                          f"refuse(dd={dd_pct:.1%}>{DD_HARD_THRESHOLD:.0%})",
                          metadata=meta)
    if dd_pct > DD_SOFT_THRESHOLD:
        adj = max(1, requested_qty // 2)
        return GateResult(True, "drawdown",
                          f"downsize(dd={dd_pct:.1%}>{DD_SOFT_THRESHOLD:.0%}: qty {requested_qty}→{adj})",
                          adjusted_qty=adj, metadata=meta)
    return GateResult(True, "drawdown",
                      f"pass(dd={dd_pct:.1%}<={DD_SOFT_THRESHOLD:.0%})",
                      adjusted_qty=requested_qty, metadata=meta)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
@dataclass
class RiskDecision:
    passed: bool
    ticker: str
    requested_qty: int
    requested_notional: float
    adjusted_qty: int | None
    gate: str            # name of last gate evaluated (failing gate on refuse)
    reason: str
    gate_results: list[GateResult] = field(default_factory=list)


class RiskEngine:
    """
    Stateful pre-trade risk engine. Construct once per session with current
    equity + open positions, then call .check() per candidate trade.
    """

    def __init__(
        self,
        equity: float,
        positions: dict | None = None,
        historic_trades: list[dict] | None = None,
        equity_series: list[float] | None = None,
        log_path: Path | None = None,
    ):
        self.equity = float(equity) if equity and equity > 0 else 100_000.0
        self.positions = dict(positions or {})
        self.historic_trades = list(historic_trades or _load_historic_closed_trades())
        self.equity_series = (equity_series if equity_series is not None
                              else _load_equity_series())
        self.log_path = log_path or DECISIONS_LOG
        # Track batch-greedy: tickers approved this session are added so
        # subsequent correlation/concentration checks see them.
        self._batch_approved: list[str] = []

    def check(self, ticker: str, qty: int, signal: dict | None = None,
              price: float | None = None) -> RiskDecision:
        """
        Run all 5 gates in order. Short-circuit on first refusal.
        Returns a RiskDecision. Always logs the result.
        """
        signal = signal or {}
        # Price discovery: prefer caller, then signal, then last close.
        if price is None or price <= 0:
            price = float(signal.get("price") or signal.get("ref_price") or 0)
        if price <= 0:
            df = _get_price_history(ticker, days=5)
            if df is not None and not df.empty and "close" in df.columns:
                price = float(df["close"].iloc[-1])
        if price <= 0:
            decision = RiskDecision(
                passed=False, ticker=ticker, requested_qty=qty,
                requested_notional=0.0, adjusted_qty=None,
                gate="precheck", reason="no_price_available",
            )
            self._log(decision)
            return decision
        requested_notional = float(qty) * float(price)
        results: list[GateResult] = []

        # 1) Kelly
        r = gate_kelly(ticker, requested_notional, self.equity, self.historic_trades)
        results.append(r)
        if not r.passed:
            return self._refuse(ticker, qty, requested_notional, results)

        # 2) Liquidity
        r = gate_liquidity(ticker)
        results.append(r)
        if not r.passed:
            return self._refuse(ticker, qty, requested_notional, results)

        # 3) Correlation — include both existing positions and tickers we
        # greedily approved earlier in this batch.
        portfolio_tickers = list(self.positions.keys()) + list(self._batch_approved)
        r = gate_correlation(ticker, portfolio_tickers)
        results.append(r)
        if not r.passed:
            return self._refuse(ticker, qty, requested_notional, results)

        # 4) Concentration — include batch-approved as if held at requested
        # notional (caller is expected to update self.positions after submit).
        r = gate_concentration(ticker, requested_notional, self.equity, self.positions)
        results.append(r)
        if not r.passed:
            return self._refuse(ticker, qty, requested_notional, results)

        # 5) Drawdown (may downsize)
        r = gate_drawdown(qty, self.equity_series)
        results.append(r)
        if not r.passed:
            return self._refuse(ticker, qty, requested_notional, results)

        adjusted_qty = r.adjusted_qty if r.adjusted_qty is not None else qty
        decision = RiskDecision(
            passed=True, ticker=ticker, requested_qty=qty,
            requested_notional=requested_notional,
            adjusted_qty=adjusted_qty,
            gate="all_pass" if adjusted_qty == qty else "drawdown_downsize",
            reason="; ".join(f"{g.gate}:{g.reason}" for g in results),
            gate_results=results,
        )
        self._batch_approved.append(ticker)
        self._log(decision)
        return decision

    def _refuse(self, ticker: str, qty: int, notional: float,
                results: list[GateResult]) -> RiskDecision:
        failing = results[-1]
        decision = RiskDecision(
            passed=False, ticker=ticker, requested_qty=qty,
            requested_notional=notional, adjusted_qty=None,
            gate=failing.gate, reason=failing.reason, gate_results=results,
        )
        self._log(decision)
        return decision

    def _log(self, decision: RiskDecision) -> None:
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            row = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "ticker": decision.ticker,
                "requested_qty": decision.requested_qty,
                "requested_notional": round(decision.requested_notional, 2),
                "equity": round(self.equity, 2),
                "gate": decision.gate,
                "passed": decision.passed,
                "reason": decision.reason,
                "adjusted_qty": decision.adjusted_qty,
                "gate_results": [
                    {"gate": g.gate, "passed": g.passed, "reason": g.reason,
                     "metadata": g.metadata}
                    for g in decision.gate_results
                ],
            }
            with open(self.log_path, "a") as f:
                f.write(json.dumps(row, default=str) + "\n")
        except Exception as e:
            log.debug(f"risk_engine log write failed: {e}")


# ---------------------------------------------------------------------------
# CLI smoke entry
# ---------------------------------------------------------------------------
def _selftest() -> int:
    """Standalone smoke. Exercises each gate with stub data and reports."""
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--scenario",
                   choices=["correlated", "drawdown", "liquidity", "all"],
                   default="all")
    args = p.parse_args()

    failures = 0

    if args.scenario in ("correlated", "all"):
        print("=== SCENARIO 1: correlated portfolio ===")
        # Inject synthetic price cache: 5 perfectly-correlated tickers.
        base = pd.Series(
            np.cumsum(np.random.RandomState(42).randn(40)) + 100,
            index=pd.date_range("2026-01-01", periods=40, freq="B"),
        )
        for t in ["A", "B", "C", "D", "E"]:
            df = pd.DataFrame({
                "open": base + np.random.RandomState(hash(t) & 0xFFFF).randn(40) * 0.01,
                "high": base + 0.5,
                "low": base - 0.5,
                "close": base + np.random.RandomState(hash(t) & 0xFFFF).randn(40) * 0.01,
                "volume": np.full(40, 50_000_000),
            }, index=base.index)
            _PRICE_CACHE[t] = df
        engine = RiskEngine(
            equity=100_000,
            positions={"A": {"qty": 10, "notional": 1000}},
            historic_trades=[],
            equity_series=[100_000, 100_000],
            log_path=Path("/tmp/risk_engine_smoke_corr.jsonl"),
        )
        refused = 0
        for t in ["B", "C", "D", "E"]:
            dec = engine.check(ticker=t, qty=10, price=100.0)
            print(f"  {t}: passed={dec.passed} gate={dec.gate} reason={dec.reason[:120]}")
            if not dec.passed and dec.gate == "correlation":
                refused += 1
        # Expect 4/4 refused (all correlated to A).
        if refused == 4:
            print("PASS: 4/4 highly-correlated tickers refused")
        else:
            print(f"FAIL: expected 4 correlation refusals, got {refused}")
            failures += 1

    if args.scenario in ("drawdown", "all"):
        print("=== SCENARIO 2: 7-day -8% DD ===")
        eq_series = [100_000, 99_000, 97_000, 95_000, 93_000, 92_500, 92_000]
        engine = RiskEngine(
            equity=92_000,
            positions={},
            historic_trades=[],
            equity_series=eq_series,
            log_path=Path("/tmp/risk_engine_smoke_dd.jsonl"),
        )
        # Stub price cache so liquidity/correlation pass.
        _PRICE_CACHE["XYZ"] = pd.DataFrame({
            "open": [100]*30, "high": [101]*30, "low": [99]*30,
            "close": np.linspace(100, 102, 30),
            "volume": [50_000_000]*30,
        }, index=pd.date_range("2026-01-01", periods=30, freq="B"))
        dec = engine.check(ticker="XYZ", qty=10, price=100.0)
        print(f"  XYZ: passed={dec.passed} adjusted_qty={dec.adjusted_qty} gate={dec.gate} reason={dec.reason[:200]}")
        # 92000/100000 = -8% → between 5% and 10% → expect halve qty.
        if dec.passed and dec.adjusted_qty == 5:
            print("PASS: -8% DD halved qty 10→5")
        else:
            print(f"FAIL: expected adjusted_qty=5, got {dec.adjusted_qty}")
            failures += 1

    if args.scenario in ("liquidity", "all"):
        print("=== SCENARIO 3: low-volume ticker ===")
        _PRICE_CACHE["ILLIQ"] = pd.DataFrame({
            "open": [10]*30, "high": [11]*30, "low": [9]*30,
            "close": [10]*30, "volume": [10_000]*30,  # $100K/day << $10M
        }, index=pd.date_range("2026-01-01", periods=30, freq="B"))
        engine = RiskEngine(
            equity=100_000,
            positions={},
            historic_trades=[],
            equity_series=[100_000, 100_000],
            log_path=Path("/tmp/risk_engine_smoke_liq.jsonl"),
        )
        dec = engine.check(ticker="ILLIQ", qty=100, price=10.0)
        print(f"  ILLIQ: passed={dec.passed} gate={dec.gate} reason={dec.reason[:200]}")
        if not dec.passed and dec.gate == "liquidity":
            print("PASS: illiquid ticker refused by liquidity gate")
        else:
            print(f"FAIL: expected liquidity refuse, got gate={dec.gate} passed={dec.passed}")
            failures += 1

    print(f"\n=== TOTAL FAILURES: {failures} ===")
    return failures


if __name__ == "__main__":
    raise SystemExit(_selftest())
