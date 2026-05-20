"""
mastery_priors_features.py — Past-test mastery files as priors signal feature.

Provides `add_mastery_priors(df, ticker) -> df` that adds 7 features to a daily
OHLCV+features DataFrame, reading historical mastery-file artifacts that record
each ticker's prior backtest performance.

Features added (all .shift(1)-safe via mtime-gated age column):
  - prior_v4_mastered (0/1)       — has v4/ML or v8 mastery file
  - prior_v4_pf (float)            — Profit Factor parsed from v4 file (0 if none)
  - prior_v10_mastered (0/1)       — has v10 MASTERED file (not FAILED)
  - prior_v10_pf (float)           — PF from v10 mastery file (0 if none)
  - prior_v10_dd (float)           — Max DD% from v10 mastery file (0 if none)
  - prior_cross_section_top10 (0/1) — is ticker in top-10 by composite score
  - prior_mastery_age_days (int)    — days since most-recent mastery file's mtime,
                                       0 when no mastery exists OR when the bar
                                       date precedes the mastery file's mtime
                                       (prevents future-knowledge leakage)

Caching:
  Parsed-once results cached at $SP/cache/mastery_priors.parquet so successive
  imports/tickers don't re-parse the entire directory. Cache invalidates when
  the mastery_files/ directory's mtime advances.

Idempotent: re-importing or re-calling add_mastery_priors(df, ticker) will NOT
overwrite columns the caller has already populated.

Author: 2026-05-17 (priors-as-features wave)
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
WORK = Path(
    "/Users/orginal/Library/CloudStorage/"
    "GoogleDrive-zachgladstone@gmail.com/My Drive/"
    "AI-Tools/s&p500-ticker-mastery"
)
MASTERY_DIR = WORK / "mastery_files"
CACHE_DIR = WORK / "cache"
CACHE_PATH = CACHE_DIR / "mastery_priors.parquet"

MASTERY_PRIORS_FEATURE_NAMES: list[str] = [
    "prior_v4_mastered",
    "prior_v4_pf",
    "prior_v10_mastered",
    "prior_v10_pf",
    "prior_v10_dd",
    "prior_cross_section_top10",
    "prior_mastery_age_days",
]


# ---------------------------------------------------------------------------
# Regex parsers
# ---------------------------------------------------------------------------

# Match PF in forms: "PF: 2.20", "PF=2.20", "PF | 2.20" (markdown table),
# "| PF | 2.20 |", "**Profit factor** | 2.20", "profit_factor": 2.20, etc.
_PF_PATTERNS = [
    re.compile(r"profit[_\s]factor[\"']?\s*[:|=]\s*([0-9]+\.?[0-9]*)", re.IGNORECASE),
    re.compile(r"\bPF\b\s*[:|=]\s*([0-9]+\.?[0-9]*)", re.IGNORECASE),
    re.compile(r"\|\s*PF\s*\|\s*([0-9]+\.?[0-9]*)", re.IGNORECASE),
]

# DD patterns: "DD: -1.08", "DD%: -1.08", "max_drawdown_pct": -0.0108,
# "| DD% | -1.08% |", "Max drawdown % | -1.08%"
_DD_PATTERNS = [
    re.compile(r"max[_\s]drawdown[_\s]pct[\"']?\s*[:|=]\s*(-?[0-9]+\.?[0-9]*)", re.IGNORECASE),
    re.compile(r"\bDD%?\b\s*[:|=]\s*(-?[0-9]+\.?[0-9]*)", re.IGNORECASE),
    re.compile(r"\|\s*DD%?\s*\|\s*(-?[0-9]+\.?[0-9]*)", re.IGNORECASE),
    re.compile(r"max drawdown\s*%?\s*\|\s*(-?[0-9]+\.?[0-9]*)", re.IGNORECASE),
]

# WR patterns: "WR: 0.5926", "WR: 59.26%", "win_rate": 0.5926, "Win rate | 53.33%"
_WR_PATTERNS = [
    re.compile(r"win[_\s]rate[\"']?\s*[:|=]\s*([0-9]+\.?[0-9]*)", re.IGNORECASE),
    re.compile(r"\bWR\b\s*[:|=]\s*([0-9]+\.?[0-9]*)", re.IGNORECASE),
    re.compile(r"\|\s*WR%?\s*\|\s*([0-9]+\.?[0-9]*)", re.IGNORECASE),
]


def _first_match(text: str, patterns: list[re.Pattern]) -> Optional[float]:
    """Return the first numeric capture across `patterns`, or None."""
    for pat in patterns:
        m = pat.search(text)
        if m:
            try:
                return float(m.group(1))
            except (ValueError, IndexError):
                continue
    return None


def _parse_mastery_file(path: Path) -> dict:
    """Parse a single mastery markdown file. Tolerant — missing fields => None."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logger.debug("[mastery_priors] read fail %s: %s", path.name, e)
        return {"pf": None, "dd": None, "wr": None, "status": "READ_ERROR"}

    pf = _first_match(text, _PF_PATTERNS)
    dd = _first_match(text, _DD_PATTERNS)
    wr = _first_match(text, _WR_PATTERNS)

    # Normalize DD: if magnitude > 1 it's likely a percent (-1.08 means -1.08%)
    # so convert to fraction (-0.0108). If already fractional (-0.0082), keep.
    if dd is not None and abs(dd) > 1.0:
        dd = dd / 100.0

    # Normalize WR: if > 1 it's percent (53.33), convert to fraction.
    if wr is not None and wr > 1.0:
        wr = wr / 100.0

    # Status detection: FAILED filename or "FAILED" in body => failed.
    name_upper = path.name.upper()
    if "FAILED" in name_upper:
        status = "FAILED"
    elif "MASTERED" in name_upper or "STATUS: MASTERED" in text.upper():
        status = "MASTERED"
    else:
        status = "UNKNOWN"

    return {"pf": pf, "dd": dd, "wr": wr, "status": status}


# ---------------------------------------------------------------------------
# Mastery-file discovery + cache
# ---------------------------------------------------------------------------


def _classify_file(name: str) -> tuple[str, str]:
    """Return (ticker, kind) where kind is one of 'v4', 'v10', 'v8', 'd1rev', 'other'.

    Filename convention examples:
      - AAPL_ML_mastered.md             -> ('AAPL', 'v4')        [ML = Wave-4 ML]
      - AAPL_XGB_v10_mythos_mastered.md -> ('AAPL', 'v10')
      - AAPL_XGB_v10_mythos_FAILED.md   -> ('AAPL', 'v10_failed')
      - AAPL_XGB_v8_alpha158_mastered.md -> ('AAPL', 'v8')
      - BEN_D1REV_mastered.md           -> ('BEN', 'd1rev')
      - BRK.B_ML_mastered.md            -> ('BRK.B', 'v4')
    """
    # Strip .md
    base = name[:-3] if name.endswith(".md") else name
    upper = base.upper()

    # Ticker = up to the first underscore (preserves "BRK.B")
    if "_" not in base:
        return (base, "other")
    ticker, rest = base.split("_", 1)
    rest_upper = rest.upper()

    if "XGB_V10" in rest_upper or "_V10_" in rest_upper:
        if "FAILED" in rest_upper:
            return (ticker, "v10_failed")
        return (ticker, "v10")
    if "XGB_V8" in rest_upper or "ALPHA158" in rest_upper:
        return (ticker, "v8")
    if rest_upper.startswith("ML_") or rest_upper == "ML_MASTERED":
        return (ticker, "v4")
    if "D1REV" in rest_upper or "D1_REV" in rest_upper:
        return (ticker, "d1rev")
    return (ticker, "other")


def _scan_mastery_dir() -> pd.DataFrame:
    """Walk mastery_files/ and return a per-ticker summary DataFrame.

    Columns:
      ticker, prior_v4_mastered, prior_v4_pf, prior_v10_mastered, prior_v10_pf,
      prior_v10_dd, latest_mtime (UTC), composite_score
    """
    if not MASTERY_DIR.exists():
        logger.warning("[mastery_priors] mastery_files dir not found: %s", MASTERY_DIR)
        return pd.DataFrame(columns=[
            "ticker", "prior_v4_mastered", "prior_v4_pf",
            "prior_v10_mastered", "prior_v10_pf", "prior_v10_dd",
            "latest_mtime", "composite_score",
        ])

    rows: dict[str, dict] = {}
    for fp in sorted(MASTERY_DIR.iterdir()):
        if not fp.is_file() or not fp.name.endswith(".md"):
            continue
        # Skip archived/leaky files
        if ".archived" in fp.name or ".warning" in fp.name:
            continue

        ticker, kind = _classify_file(fp.name)
        if kind == "other":
            continue
        parsed = _parse_mastery_file(fp)
        mtime = datetime.fromtimestamp(fp.stat().st_mtime, tz=timezone.utc)

        rec = rows.setdefault(ticker, {
            "ticker": ticker,
            "prior_v4_mastered": 0,
            "prior_v4_pf": 0.0,
            "prior_v10_mastered": 0,
            "prior_v10_pf": 0.0,
            "prior_v10_dd": 0.0,
            "latest_mtime": None,
        })

        # v4 / v8 / d1rev all count toward "prior_v4_mastered" flag (any prior pass)
        if kind in ("v4", "v8", "d1rev") and parsed["status"] == "MASTERED":
            rec["prior_v4_mastered"] = 1
            if parsed["pf"] is not None and parsed["pf"] > rec["prior_v4_pf"]:
                # Keep best PF across multiple v4/v8 entries for same ticker
                rec["prior_v4_pf"] = float(parsed["pf"])

        if kind == "v10" and parsed["status"] == "MASTERED":
            rec["prior_v10_mastered"] = 1
            if parsed["pf"] is not None:
                rec["prior_v10_pf"] = float(parsed["pf"])
            if parsed["dd"] is not None:
                rec["prior_v10_dd"] = float(parsed["dd"])
        elif kind == "v10_failed":
            # FAILED v10 still records PF/DD (informative) but mastered=0.
            if parsed["pf"] is not None and rec["prior_v10_pf"] == 0.0:
                rec["prior_v10_pf"] = float(parsed["pf"])
            if parsed["dd"] is not None and rec["prior_v10_dd"] == 0.0:
                rec["prior_v10_dd"] = float(parsed["dd"])

        # Track latest mtime across all mastery files for this ticker
        if rec["latest_mtime"] is None or mtime > rec["latest_mtime"]:
            rec["latest_mtime"] = mtime

    if not rows:
        return pd.DataFrame(columns=[
            "ticker", "prior_v4_mastered", "prior_v4_pf",
            "prior_v10_mastered", "prior_v10_pf", "prior_v10_dd",
            "latest_mtime", "composite_score",
        ])

    df = pd.DataFrame(list(rows.values()))

    # Composite score: weight v10 over v4 (newer/harder), PF dominant, DD penalty.
    df["composite_score"] = (
        df["prior_v10_pf"] * 2.0
        + df["prior_v4_pf"] * 1.0
        + df["prior_v10_dd"] * 1.0   # DD is negative, so this is a penalty
        + df["prior_v10_mastered"] * 0.5
        + df["prior_v4_mastered"] * 0.25
    )
    return df


def _cache_valid(cache_path: Path, source_dir: Path) -> bool:
    """Cache valid if parquet exists and is newer than source dir mtime."""
    if not cache_path.exists() or not source_dir.exists():
        return False
    try:
        return cache_path.stat().st_mtime >= source_dir.stat().st_mtime
    except OSError:
        return False


def load_mastery_priors_table(force_refresh: bool = False) -> pd.DataFrame:
    """Return per-ticker priors DataFrame (cached at cache/mastery_priors.parquet).

    Cache key = mastery_files/ directory mtime. Any new/touched file invalidates.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not force_refresh and _cache_valid(CACHE_PATH, MASTERY_DIR):
        try:
            df = pd.read_parquet(CACHE_PATH)
            logger.debug("[mastery_priors] cache HIT: %d rows", len(df))
            return df
        except Exception as e:
            logger.warning("[mastery_priors] cache read failed: %s — regenerating", e)

    df = _scan_mastery_dir()

    # Add cross-section top-10 flag here (computed once, cached)
    df["prior_cross_section_top10"] = 0
    if len(df) >= 10:
        top10_idx = df.nlargest(10, "composite_score").index
        df.loc[top10_idx, "prior_cross_section_top10"] = 1

    try:
        df.to_parquet(CACHE_PATH, index=False)
        logger.info("[mastery_priors] cache wrote %d rows -> %s", len(df), CACHE_PATH)
    except Exception as e:
        logger.warning("[mastery_priors] cache write failed: %s", e)
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def add_mastery_priors(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Add 7 mastery-priors features to df. Idempotent + .shift(1)-safe.

    Args:
        df: A daily DataFrame indexed by date (DatetimeIndex assumed).
        ticker: Stock symbol used to look up the ticker's mastery files.

    Returns:
        df with 7 new columns appended. Existing columns are NOT overwritten.
        Age column is mtime-gated so any bar with date < mastery_file_mtime
        gets age=0 (prevents future-knowledge leakage).
    """
    if df is None or len(df) == 0:
        return df

    # Idempotent guard: if all 7 already present, return unchanged.
    already_present = [c for c in MASTERY_PRIORS_FEATURE_NAMES if c in df.columns]
    if len(already_present) == len(MASTERY_PRIORS_FEATURE_NAMES):
        logger.debug("[mastery_priors] all 7 features already present — skipping")
        return df

    try:
        priors_df = load_mastery_priors_table()
    except Exception as e:
        logger.warning("[mastery_priors] load failed (%s) — zeroing all features", e)
        for col in MASTERY_PRIORS_FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0 if col.endswith(("_mastered", "_top10", "_age_days")) else 0.0
        return df

    # Pull this ticker's row
    rec = priors_df[priors_df["ticker"] == ticker]
    if len(rec) == 0:
        # No mastery file for this ticker — zero everything (still .shift(1)-safe).
        defaults = {
            "prior_v4_mastered": 0,
            "prior_v4_pf": 0.0,
            "prior_v10_mastered": 0,
            "prior_v10_pf": 0.0,
            "prior_v10_dd": 0.0,
            "prior_cross_section_top10": 0,
            "prior_mastery_age_days": 0,
        }
        for col, val in defaults.items():
            if col not in df.columns:
                df[col] = val
        return df

    row = rec.iloc[0]

    # Scalar features (constant across all rows) — these are "static priors";
    # they reflect a single past-test outcome, so they don't leak intra-series.
    # The .shift(1)-safety question is "does the model see this value before the
    # underlying mastery file existed?" — addressed via age gating below.
    static_vals = {
        "prior_v4_mastered": int(row["prior_v4_mastered"]),
        "prior_v4_pf": float(row["prior_v4_pf"]),
        "prior_v10_mastered": int(row["prior_v10_mastered"]),
        "prior_v10_pf": float(row["prior_v10_pf"]),
        "prior_v10_dd": float(row["prior_v10_dd"]),
        "prior_cross_section_top10": int(row.get("prior_cross_section_top10", 0)),
    }
    for col, val in static_vals.items():
        if col not in df.columns:
            df[col] = val

    # Age column: days since mastery file mtime, BUT only for bars on/after
    # the mtime. Bars before the mtime get age=0 (the file didn't exist yet,
    # so the model couldn't have known about it — prevents future leakage).
    if "prior_mastery_age_days" not in df.columns:
        mtime = row["latest_mtime"]
        if mtime is None or not isinstance(df.index, pd.DatetimeIndex):
            df["prior_mastery_age_days"] = 0
        else:
            # Ensure both sides are timezone-naive for comparison
            if mtime.tzinfo is not None:
                mtime_naive = mtime.replace(tzinfo=None)
            else:
                mtime_naive = mtime
            idx = df.index
            if idx.tz is not None:
                idx_naive = idx.tz_convert(None)
            else:
                idx_naive = idx
            # Bars before mtime => age = 0 (file didn't exist yet)
            # Bars on/after mtime => age = (bar_date - mtime).days
            mtime_ts = pd.Timestamp(mtime_naive)
            delta_days = (idx_naive - mtime_ts).days
            age = np.where(delta_days >= 0, delta_days, 0).astype(int)
            df["prior_mastery_age_days"] = age

    return df


# ---------------------------------------------------------------------------
# CLI for manual cache rebuild / sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Mastery-priors feature module.")
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--ticker", default=None, help="Sanity-check one ticker.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = load_mastery_priors_table(force_refresh=args.rebuild_cache)
    print(f"Loaded {len(df)} ticker rows.")
    print(df.head(15).to_string())
    print()
    print(f"v4-mastered: {int(df['prior_v4_mastered'].sum())}")
    print(f"v10-mastered: {int(df['prior_v10_mastered'].sum())}")
    print(f"top-10 by composite_score:")
    print(df.nlargest(10, "composite_score")[
        ["ticker", "prior_v4_pf", "prior_v10_pf", "prior_v10_dd", "composite_score"]
    ].to_string(index=False))

    if args.ticker:
        # Build a tiny DataFrame and inject features for visual inspection
        idx = pd.date_range("2024-01-01", periods=10, freq="B")
        test_df = pd.DataFrame({"close": np.arange(10.0)}, index=idx)
        out = add_mastery_priors(test_df, args.ticker)
        print()
        print(f"Sample feature injection for {args.ticker}:")
        print(out.tail(5).to_string())
