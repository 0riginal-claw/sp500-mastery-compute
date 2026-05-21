"""regate_existing_mastery.py — Apply DSR + PBO gates to historical mastery.

Industry-2026 gate spec (per a1b0553 solver code-pattern D + repo
a91a687 mnemox-ai/deflated-sharpe):

  PROMOTE if ALL of:
    pf       >= 1.5
    sharpe   >= 1.5
    dsr_p    >= 0.95     (Deflated Sharpe Ratio p-value)
    pbo      <= 0.15     (Probability of Backtest Overfitting)
    dd       <  0.20
    n_trades >= 200

  DEMOTE prior-mastered tickers that fail any of the above with
  `demoted_reason` populated.

Input sources (cascaded fallback — uses the first that has data):
  1. state/<TICKER>/mastery.json (sweep history -> populated DSR/PBO inputs)
  2. cache/per_ticker_best.parquet (consolidated best-config table)
  3. cache/mastery_priors.parquet (legacy v4/v10 mastered flag)

Output:
  cache/mastered_dsr.parquet  (columns: ticker, pf, sharpe, dsr_p, pbo, dd,
                                n_trades, promoted, demoted_reason)

Usage:
  python scripts/regate_existing_mastery.py             # full universe
  python scripts/regate_existing_mastery.py JPM COIN BXP NVDA   # subset

"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from lopez_de_prado import (
    deflated_sharpe_ratio,
    probability_backtest_overfitting,
)

# Gate thresholds (industry-2026 spec)
GATE_PF = 1.5
GATE_SHARPE = 1.5
GATE_DSR_P = 0.95
GATE_PBO_MAX = 0.15
GATE_DD_MAX = 0.20
GATE_N_TRADES = 200


def _safe_float(x, default: float = 0.0) -> float:
    try:
        f = float(x)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _extract_returns_matrix_from_history(history: list) -> Optional[np.ndarray]:
    """Pull a (T, N) returns matrix from a mastery.json history list.

    Each history entry is one trial (one sweep config). We need a returns
    SERIES per trial for PBO. If absent, we synthesise a degenerate matrix
    from each entry's summary stats so PBO can run (degraded fidelity).
    """
    if not history:
        return None
    cols = []
    for entry in history:
        rets = entry.get("returns") or entry.get("per_bar_returns") or entry.get("trade_returns")
        if rets and isinstance(rets, list) and len(rets) > 1:
            cols.append(np.asarray(rets, dtype=float))
    if not cols:
        return None
    # Truncate to common length
    T = min(len(c) for c in cols)
    if T < 4:
        return None
    M = np.column_stack([c[:T] for c in cols])
    return M


def _stats_from_history(history: list) -> tuple[float, float]:
    """Compute (skew, kurt) of best-trial trade returns if available.

    Defaults to (0.0, 3.0) (normal) when unavailable.
    """
    if not history:
        return 0.0, 3.0
    best = max(history, key=lambda e: _safe_float(e.get("pf", 0)))
    rets = best.get("returns") or best.get("trade_returns")
    if not rets or len(rets) < 4:
        return 0.0, 3.0
    a = np.asarray(rets, dtype=float)
    a = a[np.isfinite(a)]
    if a.size < 4:
        return 0.0, 3.0
    mu = a.mean()
    sd = a.std(ddof=1)
    if sd <= 0:
        return 0.0, 3.0
    z = (a - mu) / sd
    skew = float((z ** 3).mean())
    kurt = float((z ** 4).mean())  # NOT excess (matches PSR formula)
    return skew, kurt


def regate_ticker(
    ticker: str,
    state_dir: Path,
    per_ticker_best: pd.DataFrame,
) -> dict:
    """Return a single result dict for the ticker."""
    out = {
        "ticker": ticker,
        "pf": 0.0,
        "sharpe": 0.0,
        "dsr_p": 0.0,
        "pbo": 0.5,
        "dd": 1.0,
        "n_trades": 0,
        "skew": 0.0,
        "kurt": 3.0,
        "n_trials": 1,
        "prior_status": "unknown",
        "promoted": False,
        "demoted_reason": None,
        "source": "none",
    }

    # 1. state/<TICKER>/mastery.json (richest source when populated)
    mj = state_dir / ticker / "mastery.json"
    history: list = []
    if mj.exists():
        try:
            data = json.loads(mj.read_text())
            out["pf"] = _safe_float(data.get("best_pf"))
            out["sharpe"] = _safe_float(data.get("best_sharpe"))
            out["dd"] = abs(_safe_float(data.get("best_dd"), default=1.0))
            out["n_trades"] = int(_safe_float(data.get("best_n_trades")))
            history = data.get("history") or []
            out["n_trials"] = max(1, len(history) or int(_safe_float(data.get("n_evals"), 1)))
            out["prior_status"] = "mastered" if data.get("mastered") else "not_mastered"
            out["source"] = "state_mastery_json"
            if history:
                out["skew"], out["kurt"] = _stats_from_history(history)
        except Exception as exc:
            out["demoted_reason"] = f"mastery.json parse fail: {exc}"

    # 2. Fallback / supplement: per_ticker_best.parquet
    pb_row = per_ticker_best[per_ticker_best["ticker"] == ticker]
    if (out["pf"] == 0.0 or out["sharpe"] == 0.0) and not pb_row.empty:
        r = pb_row.iloc[0]
        out["pf"] = max(out["pf"], _safe_float(r.get("pf")))
        if "wr" in r and out["sharpe"] == 0.0:
            # No sharpe column directly — derive from return + n + win-rate as a
            # rough proxy: sharpe ~= mean_ret / std_ret. We use comp_score as a
            # lower bound here when present (Hi comp_score correlates).
            cs = _safe_float(r.get("comp_score"))
            ret = _safe_float(r.get("ret"))
            if ret > 0 and out["n_trades"] == 0:
                out["n_trades"] = int(_safe_float(r.get("n")))
                # crude sharpe proxy assuming annual ret series
                out["sharpe"] = max(out["sharpe"], cs)
        if out["dd"] == 1.0:
            dd = _safe_float(r.get("dd"))
            if dd != 0:
                out["dd"] = abs(dd)
        if out["n_trades"] == 0:
            out["n_trades"] = int(_safe_float(r.get("n")))
        if out["prior_status"] in ("unknown", "not_mastered"):
            status = str(r.get("status", "")).upper()
            out["prior_status"] = "mastered" if status == "MASTERED" else out["prior_status"]
        if out["source"] == "none":
            out["source"] = "per_ticker_best"

    # 3. PBO from history matrix (degrades to neutral 0.5 if no per-bar series)
    M = _extract_returns_matrix_from_history(history)
    if M is not None and M.shape[1] >= 2:
        try:
            out["pbo"] = float(probability_backtest_overfitting(M, n_splits=8))
        except Exception as exc:
            out["demoted_reason"] = (out["demoted_reason"] or "") + f" PBO fail: {exc}"
    # If no PBO data available, leave at 0.5 (neutral). The gate threshold is
    # 0.15, so 0.5 fails PBO -> rightly demotes tickers we can't validate.

    # 4. DSR: needs n_obs (returns series length) — proxy with n_trades if no
    #    explicit return-series length.
    if out["sharpe"] > 0:
        n_obs = max(out["n_trades"], 2)  # at least 2 to avoid div by zero
        try:
            dsr_p, _ = deflated_sharpe_ratio(
                sr_observed=out["sharpe"],
                n_trials=out["n_trials"],
                skew=out["skew"],
                kurt=out["kurt"],
                n=n_obs,
            )
            out["dsr_p"] = float(dsr_p)
        except Exception as exc:
            out["demoted_reason"] = (out["demoted_reason"] or "") + f" DSR fail: {exc}"

    # 5. Apply gate
    reasons: list = []
    if out["pf"] < GATE_PF:
        reasons.append(f"pf={out['pf']:.3f}<{GATE_PF}")
    if out["sharpe"] < GATE_SHARPE:
        reasons.append(f"sharpe={out['sharpe']:.3f}<{GATE_SHARPE}")
    if out["dsr_p"] < GATE_DSR_P:
        reasons.append(f"dsr_p={out['dsr_p']:.3f}<{GATE_DSR_P}")
    if out["pbo"] > GATE_PBO_MAX:
        reasons.append(f"pbo={out['pbo']:.3f}>{GATE_PBO_MAX}")
    if out["dd"] >= GATE_DD_MAX:
        reasons.append(f"dd={out['dd']:.3f}>={GATE_DD_MAX}")
    if out["n_trades"] < GATE_N_TRADES:
        reasons.append(f"n_trades={out['n_trades']}<{GATE_N_TRADES}")

    out["promoted"] = not reasons
    if reasons:
        prefix = "demoted: " if out["prior_status"] == "mastered" else "rejected: "
        out["demoted_reason"] = prefix + "; ".join(reasons)

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Regate existing mastery with DSR+PBO")
    ap.add_argument("tickers", nargs="*", help="Specific tickers (default: all in state/)")
    ap.add_argument("--state-dir", default=str(PROJECT_ROOT / "state"))
    ap.add_argument("--output", default=str(PROJECT_ROOT / "cache" / "mastered_dsr.parquet"))
    ap.add_argument("--print-table", action="store_true")
    args = ap.parse_args()

    state_dir = Path(args.state_dir)
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        tickers = sorted([p.name for p in state_dir.iterdir()
                          if p.is_dir() and (p / "mastery.json").exists()])
    if not tickers:
        print(f"No tickers found under {state_dir}", file=sys.stderr)
        sys.exit(1)

    per_ticker_best_path = PROJECT_ROOT / "cache" / "per_ticker_best.parquet"
    if per_ticker_best_path.exists():
        per_ticker_best = pd.read_parquet(per_ticker_best_path)
    else:
        per_ticker_best = pd.DataFrame(columns=["ticker"])

    rows = []
    for tkr in tickers:
        rows.append(regate_ticker(tkr, state_dir, per_ticker_best))

    df = pd.DataFrame(rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")

    promoted = df[df["promoted"]]
    demoted = df[(~df["promoted"]) & (df["prior_status"] == "mastered")]
    print(f"Promoted: {len(promoted)} / Demoted: {len(demoted)} / Total: {len(df)}")

    if args.print_table:
        cols = ["ticker", "pf", "sharpe", "dsr_p", "pbo", "dd", "n_trades",
                "prior_status", "promoted", "demoted_reason"]
        with pd.option_context("display.max_colwidth", 80, "display.width", 200):
            print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
