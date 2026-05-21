"""
mastery_priors_loader.py — Tuned-config + sim-result priors from prior research.

Reads three priors layers behind a single API so a sweep can bias its
hyperparameter search proximal to known-good configs (Optuna-style) instead
of evaluating the full cartesian product.

Priors sources (combined into one MasteryPrior result):

  1. version_3 winner config:
       My Drive/version_3 - Gabriel/Mastered Tickers - Gabriel/{TICKER}/config.yaml
     Schema (winner case):
       ticker: AAPL
       engine: v3
       status: winner
       winner:
         signal: N_qoq_50
         additions: [[H_ema_uptrend, filter], ...]
         stop: 2.5
         target: 3.5
         wr: 89.66
         pf: 19.91
         n: 29
         max_dd: -1.35
         trades_per_week: 0.143

  2. state/mastery.json (current rolling-best per ticker):
       {ticker, best_strategy, best_hyperparams, best_pf, best_sharpe, best_dd,
        best_n_trades, n_evals, history, mastered, last_updated}

  3. per_ticker_best.parquet (cross-ticker A/B router state).

Usage:
    from mastery_priors_loader import MasteryPriors
    mp = MasteryPriors()
    prior = mp.get_prior_config("AAPL")
    if prior.has_prior:
        # Bias hyperparam sweep around prior.stop / prior.target / prior.signal
        ...

The loader is read-only.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

WORK = Path(
    "/Users/orginal/Library/CloudStorage/"
    "GoogleDrive-zachgladstone@gmail.com/My Drive/"
    "AI-Tools/s&p500-ticker-mastery"
)
STATE_DIR = WORK / "state"
PER_TICKER_BEST = WORK / "cache" / "per_ticker_best.parquet"

DRIVE_ROOT = Path(
    "/Users/orginal/Library/CloudStorage/"
    "GoogleDrive-zachgladstone@gmail.com/My Drive"
)
V3_MASTERED_DIR = DRIVE_ROOT / "version_3 - Gabriel" / "Mastered Tickers - Gabriel"


@dataclass
class MasteryPrior:
    ticker: str
    has_prior: bool = False
    # v3 winner config
    v3_status: Optional[str] = None        # "winner" | "loser" | None
    v3_signal: Optional[str] = None
    v3_additions: list = field(default_factory=list)
    v3_stop: Optional[float] = None        # ATR multiple (e.g. 2.5)
    v3_target: Optional[float] = None      # ATR multiple (e.g. 3.5)
    v3_pf: Optional[float] = None
    v3_wr: Optional[float] = None
    v3_n_trades: Optional[int] = None
    v3_max_dd: Optional[float] = None
    # state/mastery.json (current rolling best)
    state_best_strategy: Optional[str] = None
    state_best_hyperparams: dict = field(default_factory=dict)
    state_best_pf: Optional[float] = None
    state_best_sharpe: Optional[float] = None
    state_best_dd: Optional[float] = None
    state_n_evals: int = 0
    state_mastered: bool = False
    # per_ticker_best.parquet
    ptb_xgb_no_topk_best: Optional[bool] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_yaml_minimal(text: str) -> dict:
    """
    Avoid PyYAML dependency. Parse the small v3 config.yaml subset we care about.

    Schema (loser case is also valid):
       ticker: AAPL
       engine: v3
       status: loser
    Or (winner case):
       ticker: AAPL
       engine: v3
       status: winner
       winner:
         signal: N_qoq_50
         additions:
         - - H_ema_uptrend
           - filter
         stop: 2.5
         ...
    """
    out: dict = {}
    winner: dict = {}
    additions: list = []
    in_winner = False
    in_additions = False
    cur_addition: list = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        stripped = line.strip()
        # top-level (indent 0)
        if indent == 0 and ":" in stripped:
            in_additions = False
            in_winner = stripped.startswith("winner:")
            k, _, v = stripped.partition(":")
            v = v.strip()
            if not in_winner and v:
                out[k.strip()] = _coerce(v)
            continue
        # inside `winner:` block (indent 2)
        if in_winner and indent == 2 and ":" in stripped and not stripped.startswith("-"):
            k, _, v = stripped.partition(":")
            v = v.strip()
            if k.strip() == "additions":
                in_additions = True
                continue
            in_additions = False
            if v:
                winner[k.strip()] = _coerce(v)
            continue
        # additions block list items (indent 2 dash, then indent 4 dash for inner list)
        if in_winner and in_additions:
            if stripped.startswith("- -"):
                if cur_addition:
                    additions.append(cur_addition)
                cur_addition = [_coerce(stripped[3:].strip())]
            elif stripped.startswith("-"):
                cur_addition.append(_coerce(stripped[1:].strip()))
    if cur_addition:
        additions.append(cur_addition)
    if additions:
        winner["additions"] = additions
    if winner:
        out["winner"] = winner
    return out


def _coerce(v: str) -> Any:
    s = v.strip().strip('"').strip("'")
    if s in ("true", "True"): return True
    if s in ("false", "False"): return False
    if s in ("null", "None", "~", ""): return None
    try:
        if "." in s or "e" in s.lower(): return float(s)
        return int(s)
    except ValueError:
        return s


class MasteryPriors:
    """Multi-source priors lookup. Read-only."""

    def __init__(
        self,
        state_dir: Path = STATE_DIR,
        v3_dir: Path = V3_MASTERED_DIR,
        per_ticker_best_path: Path = PER_TICKER_BEST,
    ):
        self.state_dir = Path(state_dir)
        self.v3_dir = Path(v3_dir)
        self.ptb_path = Path(per_ticker_best_path)
        self._ptb_cache: Optional[pd.DataFrame] = None

    # -- v3 winner config (Mastered Tickers - Gabriel/) -------------------

    def _read_v3(self, ticker: str) -> dict:
        d = self.v3_dir / ticker / "config.yaml"
        if not d.exists():
            return {}
        try:
            return _parse_yaml_minimal(d.read_text(encoding="utf-8", errors="ignore"))
        except Exception as exc:
            logger.warning("v3 config parse failed %s: %s", d, exc)
            return {}

    # -- state/mastery.json -----------------------------------------------

    def _read_state(self, ticker: str) -> dict:
        p = self.state_dir / ticker / "mastery.json"
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text())
        except Exception as exc:
            logger.warning("mastery.json parse failed %s: %s", p, exc)
            return {}

    # -- per_ticker_best.parquet ------------------------------------------

    def _load_ptb(self) -> Optional[pd.DataFrame]:
        if self._ptb_cache is None and self.ptb_path.exists():
            try:
                self._ptb_cache = pd.read_parquet(self.ptb_path)
            except Exception as exc:
                logger.warning("per_ticker_best read failed: %s", exc)
                self._ptb_cache = pd.DataFrame()
        return self._ptb_cache

    def _read_ptb(self, ticker: str) -> dict:
        df = self._load_ptb()
        if df is None or df.empty or "ticker" not in df.columns:
            return {}
        row = df[df["ticker"] == ticker]
        if row.empty:
            return {}
        r = row.iloc[0].to_dict()
        out = {}
        if "xgb_no_topk_best" in r:
            v = r["xgb_no_topk_best"]
            out["xgb_no_topk_best"] = bool(v) if pd.notna(v) else None
        return out

    # -- public API -------------------------------------------------------

    def get_prior_config(self, ticker: str) -> MasteryPrior:
        v3 = self._read_v3(ticker)
        st = self._read_state(ticker)
        ptb = self._read_ptb(ticker)

        winner = (v3.get("winner") or {}) if v3 else {}
        prior = MasteryPrior(ticker=ticker)
        prior.v3_status = v3.get("status")
        if winner:
            prior.v3_signal = winner.get("signal")
            prior.v3_additions = winner.get("additions", [])
            prior.v3_stop = _to_float(winner.get("stop"))
            prior.v3_target = _to_float(winner.get("target"))
            prior.v3_pf = _to_float(winner.get("pf"))
            prior.v3_wr = _to_float(winner.get("wr"))
            prior.v3_n_trades = _to_int(winner.get("n"))
            prior.v3_max_dd = _to_float(winner.get("max_dd"))
        if st:
            prior.state_best_strategy = st.get("best_strategy")
            prior.state_best_hyperparams = st.get("best_hyperparams") or {}
            prior.state_best_pf = _to_float(st.get("best_pf"))
            prior.state_best_sharpe = _to_float(st.get("best_sharpe"))
            prior.state_best_dd = _to_float(st.get("best_dd"))
            prior.state_n_evals = int(st.get("n_evals", 0) or 0)
            prior.state_mastered = bool(st.get("mastered", False))
        if ptb:
            prior.ptb_xgb_no_topk_best = ptb.get("xgb_no_topk_best")

        prior.has_prior = bool(
            winner or prior.state_n_evals > 0 or prior.ptb_xgb_no_topk_best is not None
        )
        return prior

    def already_mastered(
        self,
        ticker: str,
        pf_thresh: float = 1.2,
        sharpe_thresh: float = 0.8,
    ) -> bool:
        """True if state/mastery.json says we already cleared the bar."""
        p = self.get_prior_config(ticker)
        if p.state_mastered:
            return True
        if (p.state_best_pf or 0) >= pf_thresh and (p.state_best_sharpe or 0) >= sharpe_thresh:
            return True
        return False

    def coverage_report(self, tickers: list[str]) -> dict:
        v3 = sum(1 for t in tickers if (self.v3_dir / t / "config.yaml").exists())
        st = sum(1 for t in tickers if (self.state_dir / t / "mastery.json").exists())
        any_ = sum(1 for t in tickers if self.get_prior_config(t).has_prior)
        already = sum(1 for t in tickers if self.already_mastered(t))
        return {
            "n_tickers": len(tickers),
            "v3_winner_configs": v3,
            "state_mastery_jsons": st,
            "any_prior": any_,
            "already_mastered": already,
        }


def _to_float(v: Any) -> Optional[float]:
    if v is None: return None
    try: return float(v)
    except Exception: return None


def _to_int(v: Any) -> Optional[int]:
    if v is None: return None
    try: return int(v)
    except Exception: return None


# -- smoke helper -------------------------------------------------------

def _smoke():
    import sys
    mp = MasteryPriors()
    for t in ("AAPL", "NVDA", "ZZZNOTREAL"):
        p = mp.get_prior_config(t)
        print(f"  {t}: has_prior={p.has_prior}  v3_status={p.v3_status}  "
              f"v3_signal={p.v3_signal}  v3_stop={p.v3_stop}  v3_target={p.v3_target}  "
              f"v3_pf={p.v3_pf}  state_evals={p.state_n_evals}  "
              f"already_mastered={mp.already_mastered(t)}")
    rpt = mp.coverage_report(["AAPL", "NVDA", "MSFT", "GOOGL", "ZZZNOTREAL"])
    print("coverage:", rpt)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_smoke())
