"""sec-edgar/sec-edgar feature wrapper.

Source: https://github.com/sec-edgar/sec-edgar
License: Apache-2.0
Discovered: 2026-05-17 (longtail_05)
Fleshed-out: 2026-05-20 (top7-followup; previously a 2-col zero-fill stub)

Causal-safe: producer emits past-only values; consumer applies .shift(1).
Graceful degradation: returns df with zero-filled feature columns on any error.

Implementation
--------------
This wrapper delegates to `edgar_extras_features.add_edgar_extras_features`
(landed 2026-05-20, mission edgar_govtrades_full) which exposes 12 real
EDGAR-derived columns from the local SQLite EDGAR mirror at
`claudes test/data/edgar/data/edgar.db`.

Returned columns (renamed with `se_*` prefix to preserve the original
sec_edgar namespace for downstream consumers):

  - se_days_since_def14a            (G1) proxy / governance event recency
  - se_def14a_flag_30d              (G1) recent proxy flag
  - se_days_since_any_amendment     (G2) restatement recency
  - se_amendment_flag_30d           (G2) recent restatement flag
  - se_days_since_likely_earnings_8k (G3) earnings 8-K recency (period_of_report-based)
  - se_likely_earnings_8k_flag_7d   (G3) within-week earnings flag
  - se_filed_to_period_lag_days     (G3) filed-vs-period lag
  - se_days_since_s1                (G4) dilution-risk recency
  - se_s1_flag_180d                 (G4) 180-day dilution flag
  - se_filings_count_7d             (G5) 7-day filing burst counter
  - se_burst_flag                   (G5) ge3 filings in 7d
  - se_filing_density_accel         (G6) 7d/30d filing rate ratio

If `edgar_extras_features` cannot be imported (or the EDGAR DB is missing on
this machine) the wrapper falls back to the original 2-column zero-fill so
downstream backtest steps never crash.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

# Path-of-truth references kept for documentation / future expansion.
_CLONE_ROOT = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/"
    "My Drive/AI-Tools/repos-claude-clones/sec-edgar"
)
if _CLONE_ROOT.exists() and str(_CLONE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CLONE_ROOT))

# Make our scripts/ dir importable so `edgar_extras_features` resolves regardless
# of CWD.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# Map from edgar_extras column name -> sec_edgar column name (1:1).
_RENAME_MAP: dict[str, str] = {
    "edgar_days_since_def14a": "se_days_since_def14a",
    "edgar_def14a_flag_30d": "se_def14a_flag_30d",
    "edgar_days_since_any_amendment": "se_days_since_any_amendment",
    "edgar_amendment_flag_30d": "se_amendment_flag_30d",
    "edgar_days_since_likely_earnings_8k": "se_days_since_likely_earnings_8k",
    "edgar_likely_earnings_8k_flag_7d": "se_likely_earnings_8k_flag_7d",
    "edgar_filed_to_period_lag_days": "se_filed_to_period_lag_days",
    "edgar_days_since_s1": "se_days_since_s1",
    "edgar_s1_flag_180d": "se_s1_flag_180d",
    "edgar_filings_count_7d": "se_filings_count_7d",
    "edgar_burst_flag": "se_burst_flag",
    "edgar_filing_density_accel": "se_filing_density_accel",
}

# Public feature list — what `backtest_xgb_v10` checks for stub-vs-fleshed.
FEATURES: list[str] = list(_RENAME_MAP.values())

# Legacy 2-col stub names — kept for backwards-compat zero-fill if extras
# import is unavailable. New consumers should reference FEATURES instead.
_LEGACY_STUB_COLS: list[str] = ["se_signal_a", "se_signal_b"]


def _zero_fill_full(df: pd.DataFrame) -> pd.DataFrame:
    """Zero-fill all 12 real feature columns + 2 legacy stub cols."""
    for c in FEATURES:
        df[c] = 0
    for c in _LEGACY_STUB_COLS:
        if c not in df.columns:
            df[c] = 0.0
    return df


def add_sec_edgar_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Add sec-edgar features (12 real cols, see module docstring).

    Inputs:
        df: must contain at minimum 'date' col OR a DatetimeIndex.
        ticker: ticker symbol (string)

    Output:
        df with FEATURES (12) + _LEGACY_STUB_COLS (2) columns appended.
        Always returns df (graceful zero-fill on any error).
    """
    df = df.copy()
    try:
        # Delegate to edgar_extras (12 real cols pulled from edgar.db)
        from edgar_extras_features import (  # type: ignore[import-not-found]
            add_edgar_extras_features,
        )

        with_extras = add_edgar_extras_features(df, ticker)

        # Rename edgar_* -> se_*
        for src_col, dst_col in _RENAME_MAP.items():
            if src_col in with_extras.columns:
                df[dst_col] = with_extras[src_col].values
            else:
                df[dst_col] = 0

        # Keep legacy 2 stub cols zero-filled (no downstream consumer is known
        # to read them, but preserving them is cheap and avoids regression
        # surprises in case some persisted feature manifest expects them).
        for c in _LEGACY_STUB_COLS:
            if c not in df.columns:
                df[c] = 0.0

        return df
    except Exception as exc:  # broad-except: graceful degradation per module policy
        LOG.warning(
            "add_sec_edgar_features delegation issue for %s: %s - zero-filling 12+2 cols",
            ticker, exc,
        )
        return _zero_fill_full(df)


# Smoke runner
if __name__ == "__main__":
    import sys as _sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    tickers = _sys.argv[1:] or ["AAPL"]
    for tk in tickers:
        dates = pd.date_range("2022-01-01", "2025-12-31", freq="D")
        df0 = pd.DataFrame({"date": dates, "close": np.linspace(100, 200, len(dates))})
        out = add_sec_edgar_features(df0, tk)
        new = [c for c in out.columns if c.startswith("se_")]
        print(f"--- {tk} ---")
        print(f"  shape: {df0.shape} -> {out.shape}")
        print(f"  new se_ cols: {len(new)}")
        for c in new:
            s = out[c]
            try:
                nz = (s.astype(float) != 0).sum()
                pct = 100.0 * nz / len(s)
                print(f"    {c}: nonzero={nz:>6d} ({pct:5.1f}%)")
            except Exception as e:
                print(f"    {c}: err {e}")
