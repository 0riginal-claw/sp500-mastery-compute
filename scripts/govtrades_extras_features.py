"""govtrades_extras_features.py — Unified wrapper for the 3 gov-trades modules
not currently wired into v10.

Wires:
  - photis_govtrades_contracts_features  (gt_contracts_*    — 3 cols)
  - photis_govtrades_lobbying_amt_features (gt_lob_amt_*    — 3 cols)
  - synapse_gov_enhanced_features        (syn_gov_*         — 39 cols, MARKET-WIDE)

Why a wrapper
-------------
Each underlying module expects a slightly different df shape:
  * contracts + lobbying_amt → require DatetimeIndex
  * synapse_gov_enhanced     → requires DatetimeIndex (market-wide, ticker-agnostic)

v10's frame is `date`-column-keyed, not DatetimeIndex-keyed. This wrapper does
the index conversion + idempotency + graceful failure pattern v10 uses
everywhere, then calls the inner modules. Output is a single .shift(1)-safe
merge of all 3 modules' new columns (45 cols total).

Source paths catalogued under
research/edgar_govtrades_full/repolocal_2026-05-20.md.

Wired 2026-05-20 under mission `edgar_govtrades_full` — fills the
"gov-trades unwired extras" gap from the 11-folder catalog audit.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

# Lazy import inner modules — guard so this wrapper still loads if any of the
# 3 inner modules is missing.
try:
    from photis_govtrades_contracts_features import (  # noqa: E402
        add_photis_govtrades_contracts_features,
        FEATURE_NAMES as _CONTRACTS_NAMES,
    )
    _CONTRACTS_OK = True
except Exception as e:
    LOG.warning("[govtrades_extras] contracts module unavailable: %s", e)
    _CONTRACTS_OK = False
    _CONTRACTS_NAMES = [
        "gt_contracts_ttm_usd",
        "gt_contracts_award_count_30d",
        "gt_contracts_qoq_growth",
    ]

try:
    from photis_govtrades_lobbying_amt_features import (  # noqa: E402
        add_photis_govtrades_lobbying_amt_features,
        FEATURE_NAMES as _LOB_NAMES,
    )
    _LOB_OK = True
except Exception as e:
    LOG.warning("[govtrades_extras] lobbying_amt module unavailable: %s", e)
    _LOB_OK = False
    _LOB_NAMES = [
        "gt_lob_amt_30d_usd",
        "gt_lob_amt_qoq_growth",
        "gt_lob_amt_ttm_usd",
    ]

try:
    from synapse_gov_enhanced_features import (  # noqa: E402
        add_synapse_gov_enhanced_features,
        _SIGNAL_COLS as _SYN_SIGNAL_COLS,
        _PREFIX as _SYN_PREFIX,
    )
    _SYN_OK = True
    _SYN_NAMES = [_SYN_PREFIX + c for c in _SYN_SIGNAL_COLS]
except Exception as e:
    LOG.warning("[govtrades_extras] synapse_gov_enhanced unavailable: %s", e)
    _SYN_OK = False
    _SYN_NAMES = []


GOVTRADES_EXTRAS_FEATURE_NAMES: list[str] = list(_CONTRACTS_NAMES) + list(_LOB_NAMES) + list(_SYN_NAMES)


def _zero_fill(df: pd.DataFrame) -> pd.DataFrame:
    for c in GOVTRADES_EXTRAS_FEATURE_NAMES:
        if c not in df.columns:
            df[c] = 0.0
    return df


def _ensure_dt_index(df: pd.DataFrame) -> tuple[pd.DataFrame, Optional[pd.RangeIndex]]:
    """Return df with a DatetimeIndex (from 'date' col if needed) + original index for restore."""
    if isinstance(df.index, pd.DatetimeIndex):
        return df, None
    if "date" in df.columns:
        s = pd.to_datetime(df["date"], errors="coerce")
        if hasattr(s.dt, "tz") and s.dt.tz is not None:
            s = s.dt.tz_convert(None)
        orig_idx = df.index
        out = df.copy()
        out.index = pd.DatetimeIndex(s)
        return out, orig_idx
    return df, None


def add_govtrades_extras_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Add all 45 gov-trades-extras features (3 contracts + 3 lobbying + 39 synapse).

    Required input: df['date'] (datetime64) or df with DatetimeIndex.
    Output: df with GOVTRADES_EXTRAS_FEATURE_NAMES columns appended. Always
    returns df (graceful zero-fill on any error).
    """
    df = df.copy()
    try:
        # Convert to DT-index (inner modules expect it)
        df_dt, orig_idx = _ensure_dt_index(df)

        # Inner: contracts
        if _CONTRACTS_OK:
            try:
                df_dt = add_photis_govtrades_contracts_features(df_dt, ticker=ticker)
            except Exception as e:
                LOG.warning("[govtrades_extras] contracts failed for %s: %s", ticker, e)
                for c in _CONTRACTS_NAMES:
                    if c not in df_dt.columns:
                        df_dt[c] = 0.0
        else:
            for c in _CONTRACTS_NAMES:
                df_dt[c] = 0.0

        # Inner: lobbying_amt
        if _LOB_OK:
            try:
                df_dt = add_photis_govtrades_lobbying_amt_features(df_dt, ticker=ticker)
            except Exception as e:
                LOG.warning("[govtrades_extras] lobbying_amt failed for %s: %s", ticker, e)
                for c in _LOB_NAMES:
                    if c not in df_dt.columns:
                        df_dt[c] = 0.0
        else:
            for c in _LOB_NAMES:
                df_dt[c] = 0.0

        # Inner: synapse (market-wide, doesn't use ticker)
        if _SYN_OK and _SYN_NAMES:
            try:
                df_dt = add_synapse_gov_enhanced_features(df_dt)
                # Synapse fills NaN outside coverage; replace with 0 for tree models
                for c in _SYN_NAMES:
                    if c in df_dt.columns:
                        df_dt[c] = pd.to_numeric(df_dt[c], errors="coerce").fillna(0.0)
            except Exception as e:
                LOG.warning("[govtrades_extras] synapse failed: %s", e)
                for c in _SYN_NAMES:
                    if c not in df_dt.columns:
                        df_dt[c] = 0.0
        else:
            for c in _SYN_NAMES:
                df_dt[c] = 0.0

        # Restore original index if we monkeyed with it
        if orig_idx is not None:
            df_dt.index = orig_idx

        # Copy new cols back to original df
        for c in GOVTRADES_EXTRAS_FEATURE_NAMES:
            if c in df_dt.columns:
                df[c] = df_dt[c].values
            else:
                df[c] = 0.0
        return df
    except Exception as e:
        LOG.warning(
            "[govtrades_extras] add_govtrades_extras_features failed for %s: %s",
            ticker, e,
        )
        return _zero_fill(df)


# WIRE_CANDIDATE marker for the consumer auto-wirer
WIRE_CANDIDATE = True
WIRE_MODULE_NAME = "govtrades_extras_features"


# Smoke runner
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    tickers = sys.argv[1:] or ["AAPL", "XOM"]
    for tk in tickers:
        dates = pd.date_range("2022-01-01", "2025-12-31", freq="D")
        df = pd.DataFrame({"date": dates})
        out = add_govtrades_extras_features(df, tk)
        print(f"--- {tk} ---")
        print(f"  shape: {df.shape} -> {out.shape}")
        print(f"  new cols: {len(GOVTRADES_EXTRAS_FEATURE_NAMES)}")
        print(f"  contracts ok: {_CONTRACTS_OK}  lobbying ok: {_LOB_OK}  synapse ok: {_SYN_OK}")
        # Sample nonzero stats
        nonzero_cols = 0
        for c in GOVTRADES_EXTRAS_FEATURE_NAMES:
            if c not in out.columns:
                continue
            try:
                s = pd.to_numeric(out[c], errors="coerce").fillna(0)
                if (s != 0).any():
                    nonzero_cols += 1
            except Exception:
                pass
        print(f"  nonzero cols (any row): {nonzero_cols}/{len(GOVTRADES_EXTRAS_FEATURE_NAMES)}")
