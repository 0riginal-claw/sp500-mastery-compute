# autosolve_skip: multi-TF wire — 2026-05-21
"""mastery_writer.py — Forward-compatible writer for state/<TICKER>/mastery.json.

Introduced 2026-05-21 (multi-TF wire) so the sweep rollup can merge per-TF
results into mastery.json without breaking existing readers.

EXTENDED SCHEMA (additive — old readers still work):
    {
      "ticker": "NVDA",
      "best_strategy": "ORB",          # global winner across all (strategy, TF)
      "best_timeframe": "5Min",        # NEW: TF of the global winner
      "best_hyperparams": {...},
      "best_pf": 1.85,
      "best_sharpe": 1.21,
      "best_dd": 0.18,
      "best_n_trades": 312,
      "n_evals": 5400,
      "history": [...],                # legacy flat list (preserved)
      "per_tf_results": {              # NEW: per-TF best summary
        "1Min": {"best_pf": 1.2, "best_strategy": "ORB",
                  "best_hyperparams": {...}, "best_sharpe": 0.9,
                  "best_dd": 0.22, "best_n_trades": 540, "n_evals": 108},
        "5Min": {"best_pf": 1.85, ...},
        ...
      },
      "mastered": true,
      "last_updated": "2026-05-21T16:00:00Z"
    }

Backward compatibility:
  - All keys present in old schema remain.
  - `best_timeframe` defaults to "1Day" when missing (legacy runs).
  - `per_tf_results` defaults to `{}` when missing — readers must handle.
  - mastery_priors_loader / regate_existing_mastery already only read the
    flat keys, so the additive fields are silently ignored.

Usage:
    from mastery_writer import update_mastery_per_tf

    update_mastery_per_tf(
        ticker="NVDA",
        timeframe="5Min",
        strategy="ORB",
        hyperparams={"tp_atr": 1.5, "sl_atr": 1.0, "max_hold": 21},
        pf=1.85, sharpe=1.21, dd=0.18, n_trades=312, n_evals=108,
    )
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Resolve project root from THIS file (no env var dependency).
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
DEFAULT_STATE_DIR = _PROJECT_ROOT / "state"

# Promotion thresholds (mirror mastery_priors_loader.already_mastered).
DEFAULT_MASTER_PF = 1.2
DEFAULT_MASTER_SHARPE = 0.8

ALL_TIMEFRAMES = [
    "1Min", "5Min", "15Min", "30Min", "45Min",
    "1Hour", "4Hour", "8Hour", "12Hour", "1Day",
]


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_json(path: Path, data: dict) -> None:
    """Atomic write via tmp + rename. Drive-FUSE safe."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    os.replace(tmp, path)


def _read_mastery(path: Path) -> dict:
    """Read existing mastery.json or return a fresh template."""
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception as exc:
            logger.warning("mastery_writer: parse fail %s -> reset: %s", path, exc)
    return {
        "ticker": path.parent.name,
        "best_strategy": None,
        "best_timeframe": None,
        "best_hyperparams": {},
        "best_pf": 0.0,
        "best_sharpe": 0.0,
        "best_dd": 1.0,
        "best_n_trades": 0,
        "n_evals": 0,
        "history": [],
        "per_tf_results": {},
        "mastered": False,
        "last_updated": _now_utc_iso(),
    }


def update_mastery_per_tf(
    ticker: str,
    timeframe: str,
    strategy: str,
    hyperparams: dict[str, Any] | None = None,
    pf: float = 0.0,
    sharpe: float = 0.0,
    dd: float = 1.0,
    n_trades: int = 0,
    n_evals: int = 1,
    *,
    state_dir: Path | None = None,
    history_entry: dict | None = None,
    master_pf: float = DEFAULT_MASTER_PF,
    master_sharpe: float = DEFAULT_MASTER_SHARPE,
) -> dict:
    """Merge one (strategy, TF) result into state/<TICKER>/mastery.json.

    Updates two things:
      1. per_tf_results[<TF>]: keeps the best (strategy, hp) seen for THIS TF.
      2. top-level best_*: keeps the global best across all (strategy, TF).

    The global winner is selected by PF (then sharpe as tie-breaker), to
    match mastery_priors_loader.already_mastered's PF/sharpe gate.

    Args:
        ticker: stock symbol, e.g. "NVDA"
        timeframe: one of ALL_TIMEFRAMES, e.g. "5Min"
        strategy: strategy label, e.g. "ORB"
        hyperparams: dict of tunable params used for this run
        pf, sharpe, dd, n_trades, n_evals: result metrics
        state_dir: override default state/ dir (testing)
        history_entry: optional dict appended to mastery['history']
        master_pf, master_sharpe: promotion thresholds for `mastered` flag

    Returns:
        The updated mastery dict (also written to disk).
    """
    if timeframe not in ALL_TIMEFRAMES:
        raise ValueError(
            f"unknown timeframe {timeframe!r}; "
            f"choose from {ALL_TIMEFRAMES}"
        )

    sd = Path(state_dir or DEFAULT_STATE_DIR)
    path = sd / ticker / "mastery.json"
    m = _read_mastery(path)

    # Ensure forward-compat container exists for legacy mastery.json.
    m.setdefault("per_tf_results", {})

    # Build per-TF candidate entry.
    candidate = {
        "best_strategy": strategy,
        "best_timeframe": timeframe,
        "best_hyperparams": hyperparams or {},
        "best_pf": float(pf),
        "best_sharpe": float(sharpe),
        "best_dd": float(dd),
        "best_n_trades": int(n_trades),
        "n_evals": int(n_evals),
        "last_updated": _now_utc_iso(),
    }

    # Update per-TF best: keep this TF's incumbent if its PF is higher.
    cur_tf = m["per_tf_results"].get(timeframe, {})
    cur_pf = float(cur_tf.get("best_pf", 0.0))
    cur_sharpe = float(cur_tf.get("best_sharpe", 0.0))
    if (pf, sharpe) > (cur_pf, cur_sharpe):
        m["per_tf_results"][timeframe] = candidate
    else:
        # Bump n_evals counter even if this run didn't win.
        merged = dict(cur_tf) if cur_tf else dict(candidate)
        merged["n_evals"] = int(cur_tf.get("n_evals", 0)) + int(n_evals)
        # Preserve the previous best snapshot (don't overwrite metrics).
        if cur_tf:
            m["per_tf_results"][timeframe] = merged

    # Update global best across all TFs.
    g_pf = float(m.get("best_pf", 0.0))
    g_sharpe = float(m.get("best_sharpe", 0.0))
    if (pf, sharpe) > (g_pf, g_sharpe):
        m["best_strategy"] = strategy
        m["best_timeframe"] = timeframe
        m["best_hyperparams"] = hyperparams or {}
        m["best_pf"] = float(pf)
        m["best_sharpe"] = float(sharpe)
        m["best_dd"] = float(dd)
        m["best_n_trades"] = int(n_trades)

    # Global n_evals counter (across all TFs).
    m["n_evals"] = int(m.get("n_evals", 0)) + int(n_evals)

    # History append (used by DSR/PBO gate later).
    if history_entry is not None:
        m.setdefault("history", []).append(history_entry)

    # Promotion gate: mastered iff PF >= master_pf AND sharpe >= master_sharpe.
    m["mastered"] = (
        float(m.get("best_pf", 0.0)) >= master_pf
        and float(m.get("best_sharpe", 0.0)) >= master_sharpe
    )
    m["last_updated"] = _now_utc_iso()
    m["ticker"] = ticker  # ensure consistent even if file was renamed

    _atomic_write_json(path, m)
    logger.info(
        "mastery_writer: %s tf=%s strat=%s pf=%.3f sharpe=%.3f -> %s",
        ticker, timeframe, strategy, pf, sharpe, path,
    )
    return m


# CLI: python mastery_writer.py NVDA 5Min ORB --pf 1.85 --sharpe 1.21
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Merge one TF result into mastery.json")
    ap.add_argument("ticker")
    ap.add_argument("timeframe", choices=ALL_TIMEFRAMES)
    ap.add_argument("strategy")
    ap.add_argument("--pf", type=float, default=0.0)
    ap.add_argument("--sharpe", type=float, default=0.0)
    ap.add_argument("--dd", type=float, default=1.0)
    ap.add_argument("--n-trades", type=int, default=0)
    ap.add_argument("--n-evals", type=int, default=1)
    ap.add_argument("--hyperparams-json", default="{}")
    args = ap.parse_args()

    hp = json.loads(args.hyperparams_json)
    m = update_mastery_per_tf(
        ticker=args.ticker,
        timeframe=args.timeframe,
        strategy=args.strategy,
        hyperparams=hp,
        pf=args.pf, sharpe=args.sharpe, dd=args.dd,
        n_trades=args.n_trades, n_evals=args.n_evals,
    )
    print(json.dumps(m, indent=2, default=str))
