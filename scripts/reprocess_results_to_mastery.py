# autosolve_skip: pipeline plumbing, no errors expected — 2026-05-22
# karpathy_checked: walk all result.json; compute Sharpe from trades.csv; call mastery_writer; success=at-least-5-mastered
"""reprocess_results_to_mastery.py — Back-fill mastery.json from existing
result.json + trades.csv pairs.

Root cause being fixed (af6ff6f7):
  1. mastery_writer.update_mastery_per_tf() was NEVER called in production —
     227 result.json files have alpha (PF/WR/return) but mastery.json stays at
     defaults (best_pf=0, mastered=false) for every ticker.
  2. result.json files are missing the `sharpe` field. We compute it here from
     the sibling trades.csv (column `pnl_pct`) so the mastery_writer gate
     (sharpe >= 0.8) can evaluate the candidate.

Walks `backtests/<TICKER>/<STRATEGY>/result.json` (default), computes Sharpe
from `trades.csv`, infers timeframe from the strategy folder name (or
result.json's `timeframe` field), and calls `update_mastery_per_tf` for each
non-skipped row.

Reject reasons (logged to state/mastery_writer/silent_skips.jsonl):
  - missing_trades_csv: trades.csv absent
  - missing_metrics:    profit_factor / n_trades absent / None
  - bad_timeframe:      cannot infer a valid timeframe from path or result
  - sharpe_nan_or_inf:  computed Sharpe non-finite
  - mastery_write_fail: exception inside update_mastery_per_tf

Outputs:
  - state/<TICKER>/mastery.json (updated atomically)
  - state/mastery_writer/silent_skips.jsonl (append-only rejects)
  - prints summary to stdout

Usage:
  python3 reprocess_results_to_mastery.py
  python3 reprocess_results_to_mastery.py --backtests-dir /custom/path
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Allow direct invocation without -m: ensure sibling scripts/ is on sys.path.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from mastery_writer import update_mastery_per_tf, ALL_TIMEFRAMES  # noqa: E402

_PROJECT_ROOT = _SCRIPT_DIR.parent
DEFAULT_BACKTESTS_DIR = _PROJECT_ROOT / "backtests"
DEFAULT_STATE_DIR = _PROJECT_ROOT / "state"
SKIPS_PATH = DEFAULT_STATE_DIR / "mastery_writer" / "silent_skips.jsonl"

# Common name -> TF rules (matches naming seen in backtests/).
_NAME_TF_HINTS = {
    "ML_XGB_v10__1Day": "1Day",
    "v10": "1Day",
    "v10_xsec": "1Day",
    "xgb_v10": "1Day",
    "MEAN_REV": "1Day",
    "VWAP": "1Day",
    "ORB": "5Min",        # legacy plain ORB folder maps to default 5Min
    "ORB__5Min": "5Min",
    "ORB__15Min": "15Min",
    "ORB__30Min": "30Min",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("reprocess")


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log_skip(record: dict) -> None:
    SKIPS_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": _now_utc_iso(), **record}
    with open(SKIPS_PATH, "a") as fp:
        fp.write(json.dumps(record) + "\n")


def _compute_sharpe(trades_csv: Path) -> float | None:
    """Annualized Sharpe from per-trade pnl_pct. 0.0 for empty/std=0."""
    if not trades_csv.exists():
        return None
    try:
        df = pd.read_csv(trades_csv)
    except Exception as exc:
        logger.warning("trades.csv read fail %s: %s", trades_csv, exc)
        return None
    if "pnl_pct" not in df.columns or len(df) < 2:
        return 0.0
    ret = df["pnl_pct"].dropna().astype(float).values
    if len(ret) < 2:
        return 0.0
    std = float(np.std(ret))
    if std <= 0.0:
        return 0.0
    sharpe = float(np.mean(ret) / std * np.sqrt(252.0))
    if not np.isfinite(sharpe):
        return None
    return sharpe


def _infer_timeframe(strategy_dir_name: str, result_json: dict) -> str | None:
    """Map strategy folder name -> TF, falling back to result.json field."""
    # 1. Explicit hint table
    if strategy_dir_name in _NAME_TF_HINTS:
        return _NAME_TF_HINTS[strategy_dir_name]
    # 2. result.json `timeframe` field
    tf = result_json.get("timeframe")
    if tf in ALL_TIMEFRAMES:
        return tf
    # 3. Suffix match on the folder name (e.g. "FOO__15Min")
    for known_tf in ALL_TIMEFRAMES:
        if strategy_dir_name.endswith(f"__{known_tf}") or strategy_dir_name == known_tf:
            return known_tf
    return None


def _process_one(result_path: Path) -> tuple[str, dict]:
    """Process a single result.json. Returns (status, details_dict)."""
    strategy_dir = result_path.parent
    ticker_dir = strategy_dir.parent
    ticker = ticker_dir.name
    strategy_name = strategy_dir.name

    try:
        result = json.loads(result_path.read_text())
    except Exception as exc:
        return "skipped", {
            "ticker": ticker, "strategy": strategy_name,
            "reason": f"result_json_parse_fail: {exc}",
        }

    pf = result.get("profit_factor")
    n_trades = result.get("n_trades")
    if pf is None or n_trades is None:
        return "skipped", {
            "ticker": ticker, "strategy": strategy_name,
            "reason": "missing_metrics",
        }

    tf = _infer_timeframe(strategy_name, result)
    if tf is None:
        return "skipped", {
            "ticker": ticker, "strategy": strategy_name,
            "reason": "bad_timeframe",
        }

    trades_csv = strategy_dir / "trades.csv"
    sharpe = _compute_sharpe(trades_csv)
    if sharpe is None:
        return "skipped", {
            "ticker": ticker, "strategy": strategy_name,
            "reason": "missing_trades_csv" if not trades_csv.exists() else "sharpe_nan_or_inf",
        }

    dd = abs(float(result.get("max_drawdown_pct") or 0.0))
    hp = result.get("hyperparams") or {}  # most results don't carry these

    try:
        update_mastery_per_tf(
            ticker=ticker,
            timeframe=tf,
            strategy=strategy_name,
            hyperparams=hp,
            pf=float(pf), sharpe=float(sharpe), dd=dd,
            n_trades=int(n_trades), n_evals=1,
        )
    except Exception as exc:
        return "skipped", {
            "ticker": ticker, "strategy": strategy_name,
            "reason": f"mastery_write_fail: {exc}",
        }

    return "ok", {
        "ticker": ticker, "strategy": strategy_name,
        "timeframe": tf, "pf": float(pf), "sharpe": float(sharpe),
        "dd": dd, "n_trades": int(n_trades),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backtests-dir", type=Path, default=DEFAULT_BACKTESTS_DIR)
    ap.add_argument("--dry-run", action="store_true",
                    help="walk only — do not write mastery.json")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap number of result.json processed (smoke)")
    args = ap.parse_args()

    if not args.backtests_dir.exists():
        logger.error("backtests dir not found: %s", args.backtests_dir)
        return 2

    result_paths = sorted(args.backtests_dir.glob("*/*/result.json"))
    logger.info("found %d result.json files under %s",
                len(result_paths), args.backtests_dir)
    if args.limit > 0:
        result_paths = result_paths[: args.limit]

    n_ok = 0
    n_skip = 0
    successes: list[dict] = []
    for rp in result_paths:
        if args.dry_run:
            logger.info("DRY %s", rp)
            continue
        status, details = _process_one(rp)
        if status == "ok":
            n_ok += 1
            successes.append(details)
        else:
            n_skip += 1
            _log_skip(details)

    logger.info("DONE. ok=%d skipped=%d (skips logged to %s)",
                n_ok, n_skip, SKIPS_PATH)

    # Post-run mastered count.
    mastered_tickers: list[tuple[str, float, float]] = []
    for mastery_file in DEFAULT_STATE_DIR.glob("*/mastery.json"):
        try:
            m = json.loads(mastery_file.read_text())
        except Exception:
            continue
        if m.get("mastered"):
            mastered_tickers.append(
                (m.get("ticker") or mastery_file.parent.name,
                 float(m.get("best_pf", 0.0)),
                 float(m.get("best_sharpe", 0.0)))
            )

    mastered_tickers.sort(key=lambda x: (x[1], x[2]), reverse=True)
    logger.info("mastered=true count: %d", len(mastered_tickers))
    for t, pf, sh in mastered_tickers[:10]:
        logger.info("  MASTERED %s pf=%.3f sharpe=%.3f", t, pf, sh)

    # Brief summary on stdout for caller.
    print(json.dumps({
        "ok": n_ok,
        "skipped": n_skip,
        "mastered_count": len(mastered_tickers),
        "top10": [{"ticker": t, "pf": pf, "sharpe": sh}
                  for t, pf, sh in mastered_tickers[:10]],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
