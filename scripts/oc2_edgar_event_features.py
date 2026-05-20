"""oc2_edgar_event_features.py — EDGAR filing event features.

Extracted from OC-2/phitis/strategies/strategies_edgar_event.py and
OC-2/phitis/scripts/edgar_feature_extraction.py.

Key signals:
  E1: Event avoidance — breakouts during [-3, +2] day window around earnings have
      lower WR. Suppressing them raises quality.
  E2: Post-filing momentum — 1-5 days after earnings 8-K: PEAD (Post-Earnings
      Announcement Drift) creates institutional re-positioning breakouts.
  E3: Quiet period — 30+ days since any filing: TA signals dominate, cleaner breakouts.
  E4: Filing risk score — composite < 0.5 → low event risk → tradeable.
  IR7: Filing proximity boost — within 3 days of any filing: avg PnL $25.87 vs $6.25
       outside window (4x outperformance). Exploit, not avoid.

Usage modes:
  A) Pre-join edgar feature columns into df before calling this function.
     Required columns (if available): in_event_window, days_since_earnings_8k,
     days_since_any_filing, recent_8k_flag, earnings_8k_item_2_02_flag,
     filing_density_30d, disclosure_quiet_period_flag.
  B) Without edgar data — all feature columns are NaN/0 (harmless to model;
     tree models handle missing via NaN splits). The schema is still emitted
     so the feature set is stable across tickers with and without edgar data.

All features are .shift(1)-safe (edgar data must already be lagged before joining,
using filed_at date for alignment, not transaction date).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Expected EDGAR columns (may be pre-joined or absent)
# ---------------------------------------------------------------------------

_EDGAR_COLS = {
    "in_event_window": 0,                  # 1 if [-3,+2] days around earnings/10-Q/10-K
    "days_since_earnings_8k": np.nan,       # days since last earnings 8-K
    "days_since_any_filing": np.nan,        # days since any SEC filing
    "recent_8k_flag": 0,                   # 8-K within 30 days
    "earnings_8k_item_2_02_flag": 0,       # recent earnings 8-K with item 2.02
    "filing_density_30d": 0,              # filing count in last 30 days
    "disclosure_quiet_period_flag": 0,    # no filings in 30 days
}


def add_oc2_edgar_event_features(
    df: pd.DataFrame,
    ticker: str | None = None,
    quiet_days: int = 30,
    momentum_window_min: int = 1,
    momentum_window_max: int = 5,
    proximity_boost_days: int = 3,
    risk_score_threshold: float = 0.5,
    density_cap: float = 10.0,
) -> pd.DataFrame:
    """Add EDGAR filing event features.

    If edgar columns are not in df, output columns are NaN/0 stubs.
    Model receives consistent feature schema regardless of data availability.

    New columns
    -----------
    edgar_in_event_window           float  1 if [-3,+2] days around earnings/10-Q/10-K
    edgar_not_in_event_window       int    1 - in_event_window (avoidance signal)
    edgar_days_since_earnings_8k    float  days since last earnings 8-K (NaN = never)
    edgar_days_since_any_filing     float  days since any SEC filing (NaN = never)
    edgar_post_filing_momentum      int    1 if 1-5 days after earnings 8-K (PEAD window)
    edgar_quiet_period_flag         int    1 if no filing in quiet_days days
    edgar_filing_risk_score         float  composite risk score 0-1:
                                           recent_8k*0.3 + near_earn*0.4 + density_norm*0.3
    edgar_low_risk_flag             int    1 if filing_risk_score < risk_score_threshold
    edgar_filing_proximity_boost    float  1.3 if within proximity_boost_days of any filing,
                                           else 1.0 (position-size hint from IR7)
    edgar_recent_8k_flag            float  pass-through of recent_8k_flag
    edgar_earnings_flag             float  pass-through of earnings_8k_item_2_02_flag
    edgar_filing_density_30d        float  pass-through of filing_density_30d
    """
    df = df.copy()

    # Pull each edgar column, falling back to default if absent
    def _get(col: str) -> pd.Series:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
        return pd.Series(_EDGAR_COLS[col], index=df.index, dtype=float)

    in_ev = _get("in_event_window")
    ds_earn = _get("days_since_earnings_8k")
    ds_any = _get("days_since_any_filing")
    recent_8k = _get("recent_8k_flag").fillna(0)
    near_earn = _get("earnings_8k_item_2_02_flag").fillna(0)
    density = _get("filing_density_30d").fillna(0)
    quiet = _get("disclosure_quiet_period_flag")

    # Pass-throughs
    df["edgar_in_event_window"] = in_ev
    df["edgar_not_in_event_window"] = (1 - in_ev.fillna(0)).clip(0, 1).astype(int)
    df["edgar_days_since_earnings_8k"] = ds_earn
    df["edgar_days_since_any_filing"] = ds_any
    df["edgar_recent_8k_flag"] = recent_8k
    df["edgar_earnings_flag"] = near_earn
    df["edgar_filing_density_30d"] = density

    # Post-filing momentum window: 1-5 days after earnings
    in_momentum = (
        ds_earn.notna()
        & (ds_earn >= momentum_window_min)
        & (ds_earn <= momentum_window_max)
    )
    df["edgar_post_filing_momentum"] = in_momentum.astype(int)

    # Quiet period
    quiet_flag = ds_any.notna() & (ds_any >= quiet_days)
    # If disclosure_quiet_period_flag is present, use it directly; else derive
    if "disclosure_quiet_period_flag" in df.columns:
        df["edgar_quiet_period_flag"] = quiet.fillna(0).astype(int)
    else:
        df["edgar_quiet_period_flag"] = quiet_flag.astype(int)

    # Filing risk score (0-1)
    density_norm = density.clip(0, density_cap) / density_cap
    risk_score = (recent_8k * 0.3 + near_earn * 0.4 + density_norm * 0.3).clip(0, 1)
    df["edgar_filing_risk_score"] = risk_score
    df["edgar_low_risk_flag"] = (risk_score < risk_score_threshold).astype(int)

    # Filing proximity boost (1.3 if within proximity_boost_days of any filing)
    near_filing = ds_any.notna() & (ds_any <= proximity_boost_days)
    df["edgar_filing_proximity_boost"] = np.where(near_filing, 1.3, 1.0)

    return df
