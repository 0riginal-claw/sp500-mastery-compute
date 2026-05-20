#!/usr/bin/env python3
"""
Fast ORB Backtest — fully numpy-vectorized per session.
Avoids Python bar-by-bar loops by using numpy operations per session.

No-lookahead guarantees preserved:
  - OR built from bars 0..(OR_MINUTES-1) only
  - Rolling volume mean computed with lookback (no future data)
  - Entry at next-bar open after signal
  - Intrabar stop/target via low/high arrays
  - Force-flat at 15:55, no entries after 15:30
  - One trade per session
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pytz

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ET_TZ       = pytz.timezone("America/New_York")
DATA_ROOT   = Path("/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/claudes test/data/timeframes/S&P500 5 Year Historical Data/Minutes TimeFrames/1Min_merged")
RESULTS_ROOT = Path("/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/trading-ticker-mastery/backtests")

OR_MINUTES   = 15
VOLUME_MULT  = 1.5
TARGET_MULT  = 1.5
AVG_VOL_BARS = 20
SLIPPAGE_BPS = 5
FEE_PER_SHARE = 0.0035
NOTIONAL      = 5_000.0

import datetime
RTH_OPEN     = datetime.time(9, 30)
RTH_CLOSE    = datetime.time(16, 0)
ENTRY_CUTOFF = datetime.time(15, 30)
FORCE_EXIT_T = datetime.time(15, 55)

THRESHOLDS = dict(n_min=30, wr_min=0.55, pf_min=1.4, ret_min=0.0, dd_max=5.0, hold_max=360)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_rth(ticker: str, start: str, end: str) -> pd.DataFrame:
    path = DATA_ROOT / f"{ticker}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    # Load only needed columns
    df = pd.read_parquet(path, columns=["timestamp","open","high","low","close","volume"])
    df["ts_utc"] = pd.to_datetime(df["timestamp"], utc=True)
    df["ts_et"]  = df["ts_utc"].dt.tz_convert(ET_TZ)
    # Fast RTH filter using integer hour/minute (avoids slow dt.time comparisons)
    h = df["ts_et"].dt.hour
    m = df["ts_et"].dt.minute
    rth = ((h > 9) | ((h == 9) & (m >= 30))) & (h < 16)
    df = df[rth].copy()
    # Precompute min_of_day before date filter (still fast, fewer rows now)
    df["min_of_day"] = h[rth].values * 60 + m[rth].values
    # Date filter using fast normalize (avoids slow dt.date)
    ts_date = df["ts_et"].dt.normalize()
    start_dt = pd.Timestamp(start, tz=ET_TZ)
    end_dt   = pd.Timestamp(end,   tz=ET_TZ)
    df = df[(ts_date >= start_dt) & (ts_date <= end_dt)].copy()
    df["date_et"] = df["ts_et"].dt.date  # only on filtered (smaller) data
    df = df.sort_values("ts_et").reset_index(drop=True)
    for col in ["open","high","low","close","volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# cutoff thresholds in minutes since midnight
ENTRY_CUTOFF_MIN = 15 * 60 + 30   # 930
FORCE_EXIT_MIN   = 15 * 60 + 55   # 955


# ---------------------------------------------------------------------------
# Per-session vectorized backtest
# ---------------------------------------------------------------------------
def _run_session_fast(opens, highs, lows, closes, vols, min_of_day, ts_strs):
    """
    All arrays are 1-D numpy arrays for one RTH session, sorted ascending.
    Returns a single trade dict or None.
    """
    n = len(opens)
    if n < OR_MINUTES + 2:
        return None

    # Build OR
    or_h = float(np.max(highs[:OR_MINUTES]))
    or_l = float(np.min(lows[:OR_MINUTES]))
    or_r = or_h - or_l
    if or_r <= 0:
        return None

    target = or_h + TARGET_MULT * or_r
    stop   = or_l

    # Pre-compute rolling avg volume for each bar i: mean(vol[max(0,i-20):i])
    # Using numpy cumsum trick — O(n), no per-bar loop
    cum_vol = np.concatenate([[0.0], np.cumsum(vols)])
    # rolling_avg[i] = mean of vols[max(0,i-AVG_VOL_BARS) : i]  (excludes bar i)
    window = AVG_VOL_BARS
    counts = np.minimum(np.arange(n), window).clip(min=1).astype(float)
    start_idx = np.maximum(0, np.arange(n) - window)
    rolling_avg = (cum_vol[np.arange(n)] - cum_vol[start_idx]) / counts

    # Candidate bars: i >= OR_MINUTES, entry cutoff not yet breached
    # close > or_h, volume >= VOLUME_MULT * rolling_avg(i), min_of_day[i] < ENTRY_CUTOFF
    # AND min_of_day[i+1] < ENTRY_CUTOFF (fill bar must also be before cutoff)
    # Restrict to indices where next bar exists
    idx = np.arange(OR_MINUTES, n - 1)

    breakout   = closes[idx] > or_h
    vol_ok     = (rolling_avg[idx] > 0) & (vols[idx] >= VOLUME_MULT * rolling_avg[idx])
    time_ok    = (min_of_day[idx] < ENTRY_CUTOFF_MIN)
    next_ok    = (min_of_day[idx + 1] < ENTRY_CUTOFF_MIN)

    cand = idx[breakout & vol_ok & time_ok & next_ok]
    if len(cand) == 0:
        return None

    signal_idx = int(cand[0])
    entry_idx  = signal_idx + 1

    entry_raw    = opens[entry_idx]
    entry_filled = entry_raw * (1 + SLIPPAGE_BPS / 10_000)
    shares       = max(1.0, NOTIONAL / entry_filled)

    # Exit scan from entry_idx onward: vectorize intrabar stop/target and force-flat
    # Force-flat at 15:55
    force_idx = np.searchsorted(min_of_day[entry_idx:], FORCE_EXIT_MIN, side='left')
    last_scan = int(entry_idx + force_idx) if force_idx < n - entry_idx else n - 1

    # Within [entry_idx, last_scan), check stop and target
    scan = np.arange(entry_idx, last_scan + 1)
    hit_stop_mask   = lows[scan]  <= stop
    hit_target_mask = highs[scan] >= target

    # Force-flat condition
    force_mask = min_of_day[scan] >= FORCE_EXIT_MIN

    # Find first exit
    any_exit = hit_stop_mask | hit_target_mask | force_mask
    first_exit_pos = int(np.argmax(any_exit)) if np.any(any_exit) else len(scan) - 1
    exit_bar_local = scan[first_exit_pos]

    # Determine exit type
    if force_mask[first_exit_pos]:
        exit_raw   = opens[exit_bar_local]
        exit_reason = "force_flat_1555"
        forced     = True
    elif hit_stop_mask[first_exit_pos] and hit_target_mask[first_exit_pos]:
        exit_raw   = stop  # conservative: stop first
        exit_reason = "stop"
        forced     = False
    elif hit_stop_mask[first_exit_pos]:
        exit_raw   = stop
        exit_reason = "stop"
        forced     = False
    else:
        exit_raw   = target
        exit_reason = "target"
        forced     = False

    # Clamp to bar range
    bar_l = lows[exit_bar_local]
    bar_h = highs[exit_bar_local]
    exit_raw    = max(bar_l, min(bar_h, exit_raw))
    exit_filled = exit_raw * (1 - SLIPPAGE_BPS / 10_000)

    gross = (exit_filled - entry_filled) * shares
    fees  = 2 * FEE_PER_SHARE * shares
    net   = gross - fees
    hold  = exit_bar_local - entry_idx

    return {
        "entry_bar_et":    ts_strs[entry_idx],
        "exit_bar_et":     ts_strs[exit_bar_local],
        "direction":       "long",
        "shares":          round(float(shares), 4),
        "entry_price":     round(float(entry_filled), 4),
        "exit_price":      round(float(exit_filled), 4),
        "gross_pnl":       round(float(gross), 4),
        "fees":            round(float(fees), 4),
        "net_pnl":         round(float(net), 4),
        "holding_bars":    int(hold),
        "holding_minutes": int(hold),
        "forced_exit":     forced,
        "exit_reason":     exit_reason,
    }


# ---------------------------------------------------------------------------
# Phase runner
# ---------------------------------------------------------------------------
def run_phase(df: pd.DataFrame) -> List[Dict]:
    trades = []
    ts_strs = df["ts_et"].astype(str).values
    opens   = df["open"].values.astype(np.float64)
    highs   = df["high"].values.astype(np.float64)
    lows    = df["low"].values.astype(np.float64)
    closes  = df["close"].values.astype(np.float64)
    vols    = df["volume"].values.astype(np.float64)
    mods    = df["min_of_day"].values.astype(np.int32)
    dates   = df["date_et"].values

    # Session boundaries — find where date changes
    session_starts = np.where(np.concatenate([[True], dates[1:] != dates[:-1]]))[0]
    session_ends   = np.concatenate([session_starts[1:], [len(df)]])

    for s, e in zip(session_starts, session_ends):
        trade = _run_session_fast(
            opens[s:e], highs[s:e], lows[s:e], closes[s:e],
            vols[s:e], mods[s:e], ts_strs[s:e]
        )
        if trade is not None:
            trades.append(trade)
    return trades


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(trades: list, label: str = "") -> dict:
    if not trades:
        return {
            "label": label, "total_trades": 0, "win_rate": None, "profit_factor": None,
            "avg_holding_minutes": None, "total_net_pnl": 0.0, "total_return_pct": None,
            "max_drawdown_pct": None, "sharpe_daily": None, "avg_win": None,
            "avg_loss": None, "expectancy": None, "avg_trades_per_day": None,
            "forced_exits": 0, "num_trading_days": 0,
        }
    pnls   = np.array([t["net_pnl"] for t in trades])
    wins   = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    gp = float(wins.sum()) if len(wins) else 0.0
    gl = float(abs(losses.sum())) if len(losses) else 0.0
    pf = gp / gl if gl > 0 else float("inf")
    wr = len(wins) / len(pnls)
    equity = np.cumsum(pnls)
    peak   = np.maximum.accumulate(equity)
    dd_pct = float(np.max(peak - equity)) / NOTIONAL * 100
    ret    = float(pnls.sum()) / NOTIONAL * 100
    days   = set(t["entry_bar_et"][:10] for t in trades)
    avg_hold = float(np.mean([t["holding_minutes"] for t in trades]))
    dpnls_d: Dict = {}
    for t in trades:
        d = t["entry_bar_et"][:10]
        dpnls_d[d] = dpnls_d.get(d, 0.0) + t["net_pnl"]
    dp = list(dpnls_d.values())
    sharpe = (np.mean(dp) / (np.std(dp) + 1e-9)) * np.sqrt(252) if len(dp) > 1 else None
    return {
        "label": label, "total_trades": len(trades),
        "win_rate": round(wr, 4), "profit_factor": round(pf, 4),
        "avg_holding_minutes": round(avg_hold, 2),
        "avg_trades_per_day": round(len(trades) / max(len(days), 1), 3),
        "total_net_pnl": round(float(pnls.sum()), 2),
        "total_return_pct": round(ret, 4),
        "max_drawdown_pct": round(dd_pct, 4),
        "avg_win": round(float(wins.mean()), 2) if len(wins) else None,
        "avg_loss": round(float(losses.mean()), 2) if len(losses) else None,
        "expectancy": round(float(pnls.mean()), 2),
        "sharpe_daily": round(float(sharpe), 4) if sharpe is not None else None,
        "forced_exits": sum(1 for t in trades if t["forced_exit"]),
        "num_trading_days": len(days),
    }


# ---------------------------------------------------------------------------
# Mastery
# ---------------------------------------------------------------------------
def passes_mastery(m: dict) -> Tuple[bool, List[str]]:
    fails = []
    n  = m.get("total_trades") or 0
    wr = m.get("win_rate") or 0
    pf = m.get("profit_factor") or 0
    r  = m.get("total_return_pct") or 0
    dd = m.get("max_drawdown_pct") or 0
    h  = m.get("avg_holding_minutes") or 9999
    if n  < 30:    fails.append(f"n={n}<30")
    if wr < 0.55:  fails.append(f"wr={wr:.1%}<55%")
    if pf < 1.4:   fails.append(f"pf={pf:.2f}<1.4")
    if r  <= 0:    fails.append(f"ret={r:.2f}%<=0")
    if dd > 5.0:   fails.append(f"dd=-{dd:.2f}%<-5%")
    if h  > 360:   fails.append(f"hold={h:.0f}m>360m")
    return len(fails) == 0, fails


# ---------------------------------------------------------------------------
# Run + save single ticker
# ---------------------------------------------------------------------------
def run_ticker(
    ticker: str,
    train_start="2021-01-01", train_end="2022-12-31",
    test_start="2023-01-01",  test_end="2024-12-31",
) -> dict:
    overall_start = min(train_start, test_start)
    overall_end   = max(train_end,   test_end)
    full_df = load_rth(ticker, overall_start, overall_end)
    if full_df.empty:
        raise ValueError(f"No RTH data for {ticker}")

    results: dict = {"ticker": ticker, "strategy": "ORBStrategy"}
    for phase, s, e in [("train", train_start, train_end), ("test", test_start, test_end)]:
        phase_df = full_df[
            (full_df["date_et"] >= pd.to_datetime(s).date())
            & (full_df["date_et"] <= pd.to_datetime(e).date())
        ].copy()
        if phase_df.empty:
            results[phase] = {"error": f"no data for {phase}"}
            continue
        trades = run_phase(phase_df)
        m = compute_metrics(trades, f"{ticker}|ORBStrategy|{phase}")
        results[phase] = {"metrics": m, "trades": trades, "period_start": s, "period_end": e}

    results["no_lookahead_audit"] = {
        "entry_execution": "next_bar_open", "future_bars_accessed": False,
        "cross_session_leakage": False, "force_flat_rule": "15:55_ET",
    }

    out_dir = RESULTS_ROOT / ticker / "orb_strategy"
    out_dir.mkdir(parents=True, exist_ok=True)
    slim = {k: v for k, v in results.items() if k not in ("train","test")}
    for ph in ("train","test"):
        if ph in results:
            slim[ph] = {k: v for k, v in results[ph].items() if k != "trades"}
    (out_dir / "result.json").write_text(json.dumps(slim, indent=2, default=str))
    for ph in ("train","test"):
        if ph in results and "trades" in results[ph]:
            (out_dir / f"trades_{ph}.json").write_text(
                json.dumps(results[ph]["trades"], indent=2, default=str)
            )
    return results


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------
TICKERS_AE = """A AAPL ABBV ABNB ABT ACGL ACN ADBE ADI ADM ADP ADSK AEE AEP AES AFL AIG AIZ AJG AKAM
ALB ALGN ALL ALLE AMAT AMCR AMD AME AMGN AMP AMT AMZN ANET AON AOS APA APD APH APO APP APTV
ARE ARES ATO AVB AVGO AVY AWK AXON AXP AZO BA BAC BALL BAX BBY BDX BEN BG BIIB BK BKNG BKR
BLDR BLK BMY BR BRK.B BRO BSX BX BXP C CAG CAH CARR CAT CB CBOE CBRE CCI CCL CDNS CDW CEG CF
CFG CHD CHRW CHTR CI CIEN CINF CL CLX CMCSA CME CMG CMI CMS CNC CNP COF COIN COO COP COR COST
CPAY CPB CPRT CPT CRH CRL CRM CRWD CSCO CSGP CSX CTAS CTRA CTSH CTVA CVNA CVS CVX D DAL DASH
DD DDOG DE DECK DELL DG DGX DHI DHR DIS DLR DLTR DOC DOV DOW DPZ DRI DTE DUK DVA DVN DXCM EA
EBAY ECL ED EFX EG EIX EL ELV EME EMR EOG EPAM EQIX EQR EQT ERIE ES ESS ETN ETR EVRG EW EXC
EXE EXPD EXPE EXR""".split()


def run_batch(tickers=TICKERS_AE):
    summary, passed, failed = [], [], []
    t_global = time.time()
    for idx, ticker in enumerate(tickers, 1):
        print(f"[{idx:3d}/{len(tickers)}] {ticker} ...", end=" ", flush=True)
        t0 = time.time()
        try:
            r  = run_ticker(ticker)
            m  = r.get("test", {}).get("metrics", {})
            ok, fail_r = passes_mastery(m)
            status = "PASS" if ok else "FAIL"
            print(
                f"{status} [{time.time()-t0:.1f}s] "
                f"n={m.get('total_trades',0)} wr={m.get('win_rate',0):.1%} "
                f"pf={m.get('profit_factor',0):.2f} ret={m.get('total_return_pct',0):.1f}% "
                f"dd=-{m.get('max_drawdown_pct',0):.1f}%", flush=True)
            row = dict(ticker=ticker, status=status,
                       n=m.get("total_trades",0), wr=m.get("win_rate",0),
                       pf=m.get("profit_factor",0), ret=m.get("total_return_pct",0),
                       dd=m.get("max_drawdown_pct",0), hold=m.get("avg_holding_minutes",0),
                       sharpe=m.get("sharpe_daily"), fail_reasons=fail_r)
            summary.append(row)
            (passed if ok else failed).append(ticker)
        except Exception as e:
            print(f"ERROR [{time.time()-t0:.1f}s] {e}", flush=True)
            summary.append(dict(ticker=ticker, status="ERROR", n=0, wr=0, pf=0,
                                ret=0, dd=0, hold=0, sharpe=None,
                                fail_reasons=[str(e)], error=str(e)))
            failed.append(ticker)
    print(f"\nBatch done in {time.time()-t_global:.0f}s. PASS={len(passed)} FAIL/ERR={len(failed)}")
    return summary, passed, failed


def write_report(summary, passed, failed):
    rpt_dir = Path("/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/trading-ticker-mastery/reports")
    rpt_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ORB Intraday Backtest — S&P 500 Tickers A-E", "",
        "**Test period:** 2023-01-01 to 2024-12-31  ",
        "**Train period:** 2021-01-01 to 2022-12-31  ",
        "**Strategy:** ORBStrategy (15-min OR, long-only, 1 trade/session, entry next-bar-open)  ",
        "**Mastery thresholds:** n>=30, WR>=55%, PF>=1.4, RET>0%, DD>=-5%, AvgHold<360min  ", "",
        "## Summary",
        f"- Total tickers: {len(summary)}",
        f"- PASS: {len(passed)}",
        f"- FAIL/ERROR: {len(failed)}",
        f"- Pass rate: {len(passed)/max(len(summary),1)*100:.1f}%", "",
    ]
    if passed:
        lines += ["## Passing Tickers", "",
                  "| Ticker | N | WR | PF | RET% | DD% | Hold(m) | Sharpe |",
                  "|--------|---|----|----|------|-----|---------|--------|"]
        for r in summary:
            if r["status"] == "PASS":
                sh = f"{r['sharpe']:.2f}" if r["sharpe"] is not None else "N/A"
                lines.append(f"| {r['ticker']} | {r['n']} | {r['wr']:.1%} | {r['pf']:.2f} | {r['ret']:.2f} | -{r['dd']:.2f} | {r['hold']:.0f} | {sh} |")
        lines.append("")
    lines += ["## All Tickers — Full Results", "",
              "| Ticker | Status | N | WR | PF | RET% | DD% | Hold(m) | Fail Reasons |",
              "|--------|--------|---|----|----|------|-----|---------|--------------|"]
    for r in summary:
        if r["status"] == "ERROR":
            lines.append(f"| {r['ticker']} | ERROR | - | - | - | - | - | - | {r.get('error','')[:60]} |")
        else:
            reasons = "; ".join(r["fail_reasons"]) if r["fail_reasons"] else "-"
            lines.append(f"| {r['ticker']} | {r['status']} | {r['n']} | {r['wr']:.1%} | {r['pf']:.2f} | {r['ret']:.2f} | -{r['dd']:.2f} | {r['hold']:.0f} | {reasons} |")
    lines.append("")
    rpt = rpt_dir / "orb_batch_A_E.md"
    jsn = rpt_dir / "orb_batch_A_E_summary.json"
    rpt.write_text("\n".join(lines))
    jsn.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Report: {rpt}")
    print(f"JSON:   {jsn}")
    return rpt


if __name__ == "__main__":
    tickers = TICKERS_AE
    if len(sys.argv) > 1:
        tickers = sys.argv[1:]
    print(f"ORB fast batch: {len(tickers)} tickers")
    print("=" * 70)
    summary, passed, failed = run_batch(tickers)
    write_report(summary, passed, failed)
