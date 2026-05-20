"""
gabriel_priors_features.py — Mastery-formula priors from Gabriel.
Wave A (2026-05-17): folder-only (27 tickers covered).
Wave B (2026-05-17): full-coverage upgrade. Reads from a pre-built parquet
                     (cache/gabriel_priors_full.parquet) that includes BOTH
                     the 27 deep-folder masteries AND the 496 single-file
                     stub masteries → up to ~523 tickers covered.

Parquet source-of-truth (built by scripts/gabriel_priors_harvest.py):
  /My Drive/AI-Tools/s&p500-ticker-mastery/cache/gabriel_priors_full.parquet

Fallback (parquet missing): preserves the original behaviour of parsing
analysis_report.md on-the-fly from
  /My Drive/version_3 - Gabriel/Mastered Tickers - Gabriel/<TICKER>/

Emits 5 .shift(1)-safe scalar priors (broadcast as constants — these
scalars summarise a SEPARATE prior backtest that pre-dates any bar in
the live frame, so there is no leakage):

  - gabriel_champion_pf              float
  - gabriel_champion_wr              float in [0,1]
  - gabriel_champion_n_trades        int
  - gabriel_regime_breakdown_score   float in roughly [0, 1]
  - gabriel_monthly_perf_consistency float in [-3, +3] (negative CV)

All features zero-fill when ticker absent. Idempotent.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GABRIEL_ROOT = Path(
    "/Users/orginal/Library/CloudStorage/"
    "GoogleDrive-zachgladstone@gmail.com/My Drive/"
    "version_3 - Gabriel/Mastered Tickers - Gabriel"
)

GABRIEL_PARQUET = Path(
    "/Users/orginal/Library/CloudStorage/"
    "GoogleDrive-zachgladstone@gmail.com/My Drive/"
    "AI-Tools/s&p500-ticker-mastery/cache/gabriel_priors_full.parquet"
)

GABRIEL_PRIORS_FEATURE_NAMES: list[str] = [
    "gabriel_champion_pf",
    "gabriel_champion_wr",
    "gabriel_champion_n_trades",
    "gabriel_regime_breakdown_score",
    "gabriel_monthly_perf_consistency",
]

# In-process cache
_prior_cache: dict[str, dict] = {}
# Parquet-backed lookup table (ticker -> 5-feature dict). Lazy-loaded once.
_parquet_table: Optional[dict] = None


def _load_parquet_table() -> dict:
    """Load the full-coverage parquet into a {ticker: {feature: value}} dict.
    Returns {} on any failure (preserves legacy folder-scan fallback)."""
    global _parquet_table
    if _parquet_table is not None:
        return _parquet_table
    _parquet_table = {}
    if not GABRIEL_PARQUET.exists():
        logger.info("[gabriel_priors] parquet absent, will fall back to folder scan: %s",
                    GABRIEL_PARQUET)
        return _parquet_table
    try:
        df = pd.read_parquet(GABRIEL_PARQUET)
    except Exception as e:
        logger.warning("[gabriel_priors] parquet read failed (%s); folder scan fallback", e)
        return _parquet_table
    # Map columns from harvester schema → feature names
    col_map = {
        "pf": "gabriel_champion_pf",
        "wr": "gabriel_champion_wr",
        "n_trades": "gabriel_champion_n_trades",
        "regime_breakdown_score": "gabriel_regime_breakdown_score",
        "monthly_perf_consistency": "gabriel_monthly_perf_consistency",
    }
    for _, row in df.iterrows():
        tk = str(row.get("ticker", "")).strip()
        if not tk:
            continue
        _parquet_table[tk] = {
            "gabriel_champion_pf": float(row.get("pf", 0.0) or 0.0),
            "gabriel_champion_wr": float(row.get("wr", 0.0) or 0.0),
            "gabriel_champion_n_trades": int(row.get("n_trades", 0) or 0),
            "gabriel_regime_breakdown_score": float(row.get("regime_breakdown_score", 0.0) or 0.0),
            "gabriel_monthly_perf_consistency": float(row.get("monthly_perf_consistency", 0.0) or 0.0),
        }
    logger.info("[gabriel_priors] parquet loaded: %d tickers", len(_parquet_table))
    return _parquet_table

# Regex for "Win Rate" / "Total Trades" / "Profit Factor" rows in overall table
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+")


def _safe_float(s: str) -> Optional[float]:
    m = _NUM_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group())
    except ValueError:
        return None


def _parse_analysis_md(path: Path) -> dict:
    """Parse analysis_report.md and return a dict with parsed scalars.

    Keys returned (each may be missing if section absent):
      overall_pf, overall_wr (fraction), overall_n_trades
      regime_wr_uptrend, regime_wr_sideways, regime_wr_downtrend
      monthly_total_ret (list of floats)
    """
    out: dict = {}
    if not path.exists():
        return out
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.debug("[gabriel] read %s failed: %s", path, e)
        return out

    # ---- Overall block (markdown table) ----
    # Lines look like: | Total Trades | 2751 |
    for line in text.splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.split("|") if c.strip()]
            if len(cells) >= 2:
                key = cells[0].lower()
                val = cells[1]
                if "total trades" == key and "overall_n_trades" not in out:
                    n = _safe_float(val)
                    if n is not None:
                        out["overall_n_trades"] = int(n)
                elif "win rate" == key and "overall_wr" not in out:
                    wr = _safe_float(val)
                    if wr is not None:
                        # convert percent to fraction
                        out["overall_wr"] = wr / 100.0 if wr > 1.0 else wr
                elif "profit factor" == key and "overall_pf" not in out:
                    pf = _safe_float(val)
                    if pf is not None:
                        out["overall_pf"] = pf

    # ---- Regime WR (uptrend / sideways / downtrend) ----
    # Find best per-regime tagline OR average across all formula rows inside the regime block.
    for regime in ("uptrend", "sideways", "downtrend"):
        section_pat = re.compile(rf"### Regime: {regime}\s*(.*?)(?=\n###|\Z)", re.S | re.I)
        m = section_pat.search(text)
        wrs: list[float] = []
        if m:
            block = m.group(1)
            for line in block.splitlines():
                if line.startswith("|") and "|" in line:
                    cells = [c.strip() for c in line.split("|") if c.strip()]
                    # table data row: Formula | Trades | WR% | TotalRet% | PF
                    if len(cells) >= 3 and cells[0].lower() != "formula" and "---" not in cells[0]:
                        wr_cell = cells[2]
                        wr_val = _safe_float(wr_cell)
                        if wr_val is not None and 0 <= wr_val <= 100:
                            wrs.append(wr_val / 100.0)
        if wrs:
            out[f"regime_wr_{regime}"] = float(np.mean(wrs))

    # ---- Monthly perf table ----
    # "## 7. Monthly Performance"  | Month | Trades | WR% | Total Ret% | Avg Ret% | PF |
    m2 = re.search(r"##\s*\d*\.?\s*Monthly Performance\s*(.*)", text, re.S | re.I)
    if m2:
        block = m2.group(1)
        monthly: list[float] = []
        for line in block.splitlines():
            if line.startswith("|") and "|" in line:
                cells = [c.strip() for c in line.split("|") if c.strip()]
                if len(cells) >= 4 and cells[0].lower() not in ("month",) and "---" not in cells[0]:
                    ret_cell = cells[3]
                    rv = _safe_float(ret_cell)
                    if rv is not None:
                        monthly.append(rv)
        if monthly:
            out["monthly_total_ret"] = monthly

    return out


def _parse_champion_from_full_results(path: Path) -> dict:
    """Optional fallback / corroborator. Reads full_results.csv, picks champion
    row (highest PF in_sample, n >= 10). Returns dict with pf, wr (fraction), n.
    """
    out: dict = {}
    if not path.exists():
        return out
    try:
        df = pd.read_csv(path)
    except Exception as e:
        logger.debug("[gabriel] read full_results %s failed: %s", path, e)
        return out
    if df.empty:
        return out
    # restrict to in_sample if 'phase' present
    if "phase" in df.columns:
        df_is = df[df["phase"].astype(str).str.lower() == "in_sample"]
        if not df_is.empty:
            df = df_is
    if "n" in df.columns:
        df = df[df["n"].fillna(0) >= 10]
    if df.empty or "pf" not in df.columns:
        return out
    df = df.sort_values("pf", ascending=False)
    top = df.iloc[0]
    try:
        out["overall_pf"] = float(top.get("pf"))
    except Exception:
        pass
    wr = top.get("wr")
    if wr is not None and not pd.isna(wr):
        try:
            wr_f = float(wr)
            out["overall_wr"] = wr_f / 100.0 if wr_f > 1.0 else wr_f
        except Exception:
            pass
    n = top.get("n")
    if n is not None and not pd.isna(n):
        try:
            out["overall_n_trades"] = int(n)
        except Exception:
            pass
    return out


def _compute_scalars(parsed: dict) -> dict:
    """Map parsed dict → 5 feature scalars (zero defaults)."""
    pf = float(parsed.get("overall_pf", 0.0))
    wr = float(parsed.get("overall_wr", 0.0))
    n = int(parsed.get("overall_n_trades", 0))
    regimes = [
        parsed.get("regime_wr_uptrend"),
        parsed.get("regime_wr_sideways"),
        parsed.get("regime_wr_downtrend"),
    ]
    regimes_filt = [r for r in regimes if r is not None]
    if len(regimes_filt) == 3:
        mean_r = float(np.mean(regimes_filt))
        std_r = float(np.std(regimes_filt))
        regime_score = mean_r - 0.5 * std_r
    else:
        regime_score = 0.0

    monthly = parsed.get("monthly_total_ret", [])
    if monthly and len(monthly) >= 2:
        m_mean = float(np.mean(monthly))
        m_std = float(np.std(monthly))
        if abs(m_mean) > 1e-6:
            cv = m_std / m_mean
            consistency = float(np.clip(-cv, -3.0, 3.0))
        else:
            consistency = 0.0
    else:
        consistency = 0.0

    return {
        "gabriel_champion_pf": pf,
        "gabriel_champion_wr": wr,
        "gabriel_champion_n_trades": n,
        "gabriel_regime_breakdown_score": regime_score,
        "gabriel_monthly_perf_consistency": consistency,
    }


def _load_priors(ticker: str) -> dict:
    if ticker in _prior_cache:
        return _prior_cache[ticker]
    # ---- Fast path: parquet lookup (covers up to 523 tickers) ----
    table = _load_parquet_table()
    if ticker in table:
        out = dict(table[ticker])
        _prior_cache[ticker] = out
        return out
    # ---- Legacy fallback: per-folder analysis_report.md parse ----
    folder = GABRIEL_ROOT / ticker
    if not folder.exists():
        out = _compute_scalars({})
        _prior_cache[ticker] = out
        return out
    parsed_md = _parse_analysis_md(folder / "analysis_report.md")
    # Corroborate / fill missing scalars from full_results.csv
    if "overall_pf" not in parsed_md or "overall_n_trades" not in parsed_md:
        parsed_csv = _parse_champion_from_full_results(folder / "full_results.csv")
        for k, v in parsed_csv.items():
            parsed_md.setdefault(k, v)
    out = _compute_scalars(parsed_md)
    _prior_cache[ticker] = out
    logger.info(
        "[gabriel_priors] %s: pf=%.2f wr=%.3f n=%d regime=%.3f consistency=%.3f",
        ticker, out["gabriel_champion_pf"], out["gabriel_champion_wr"],
        out["gabriel_champion_n_trades"], out["gabriel_regime_breakdown_score"],
        out["gabriel_monthly_perf_consistency"],
    )
    return out


def _zero_fill(df: pd.DataFrame) -> pd.DataFrame:
    for col in GABRIEL_PRIORS_FEATURE_NAMES:
        if col not in df.columns:
            if col == "gabriel_champion_n_trades":
                df[col] = 0
            else:
                df[col] = 0.0
    return df


def add_gabriel_priors_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Append 5 .shift(1)-safe scalar priors to df (broadcast as constants).
    Idempotent; zero-fill when the ticker folder is missing/empty.
    """
    if df is None or len(df) == 0:
        return df
    if all(c in df.columns for c in GABRIEL_PRIORS_FEATURE_NAMES):
        return df

    # We can serve from EITHER the parquet OR the folder root. Only zero-fill
    # if BOTH sources are unavailable.
    if not GABRIEL_PARQUET.exists() and not GABRIEL_ROOT.exists():
        logger.info("[gabriel_priors] no source available (parquet+root both missing) — zero-filling")
        return _zero_fill(df)

    scalars = _load_priors(ticker)
    for col in GABRIEL_PRIORS_FEATURE_NAMES:
        if col not in df.columns:
            df[col] = scalars[col]
    # Cast count to int
    if "gabriel_champion_n_trades" in df.columns:
        try:
            df["gabriel_champion_n_trades"] = df["gabriel_champion_n_trades"].astype(int)
        except Exception:
            pass
    return df


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    tk = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    idx = pd.date_range(end=pd.Timestamp.utcnow().date(), periods=20, freq="B")
    demo = pd.DataFrame({"close": np.linspace(100, 110, len(idx))}, index=idx)
    out = add_gabriel_priors_features(demo, tk)
    print(out[GABRIEL_PRIORS_FEATURE_NAMES].iloc[0].to_dict())
