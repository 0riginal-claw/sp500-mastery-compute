"""
daily_integration_features.py
Top-3 features from daily discovery review 2026-05-16.

All features use .shift(1) — no forward-look. Safe to chain into any build_v*_features call.

Features added (7 columns total):
  1. beta_adj_residual_ret_z21      -- idiosyncratic return residual z-scored 21d
     beta_adj_residual_ret_z63      -- same, 63d window
  2. csrs_5d                        -- cross-sector reversion score (5d)
     csrs_10d                       -- cross-sector reversion score (10d)
     csrs_20d                       -- cross-sector reversion score (20d)
  3. earn_contam_gate               -- earnings contamination harmonic gate
     earn_post_rv_gate              -- post-earnings realized-vol gate

Why these three:
  1. Beta-adjusted residual: v7 has beta_spy_60d and corr_spy_60d but NO residual return
     or its z-score. Raw returns on high-beta names (AAPL 1.2β, AVGO 1.3β) are dominated
     by market moves; only the idiosyncratic residual mean-reverts. Expected lift: +PF 0.05-0.12
     for high-beta failing names (AVGO, NCLH, MCHP).

  2. Cross-sector reversion score: v7 has sector_relative_return_5d (raw diff) but NOT
     the beta-rank-weighted formulation. Normalizing by rolling_std_diff_63d and multiplying
     by rank(beta_sector) gives a calibrated "idiosyncratic stretch vs sector gravity" signal.
     Expected lift: closes the gap for PNC-like sector-correlated names. Effort: S.

  3. Earnings contamination harmonic gate: v7 has days_until_earnings (raw int) and
     is_earnings_week (binary) from alpaca_features. Missing: the composite harmonic decay
     that SIMULTANEOUSLY gates pre-earnings (proximity) and post-earnings (gap × ATR) regime.
     XGBoost currently trains on both regimes, diluting RSI weights. This gives the tree a
     single-scalar gate to branch on. Expected lift: +WR ~0.01 for AAPL, TER, GEHC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import zscore as scipy_zscore


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_div(a: pd.Series, b: pd.Series, fill: float = 0.0) -> pd.Series:
    """Element-wise division, filling inf/nan with fill."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = a / b.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan).fillna(fill)


def _rolling_zscore(s: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Rolling z-score of series s."""
    if min_periods is None:
        min_periods = max(10, window // 3)
    mu = s.rolling(window, min_periods=min_periods).mean()
    sig = s.rolling(window, min_periods=min_periods).std()
    return _safe_div(s - mu, sig)


# ---------------------------------------------------------------------------
# Feature 1: Beta-Adjusted SPY Residual Return (z-scored)
# ---------------------------------------------------------------------------

def _add_beta_adj_residual(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute idiosyncratic return residual after removing rolling-60d SPY beta.

    residual_t = ret_t - beta_60d * spy_ret_t
    z21 = zscore(residual, 21d rolling)
    z63 = zscore(residual, 63d rolling)

    Inputs expected in df (all pre-shifted by build_features):
      - 'ret_1d'       : daily log or arithmetic return of ticker
      - 'spy_return_5d': 5d SPY return (from cross_sectional layer, shifted).
                         We reconstruct 1d SPY from this if needed, else fallback.
      - 'beta_spy_60d' : rolling 60d beta vs SPY (cross_sectional layer, shifted).

    If columns are missing we degrade gracefully (feature = 0).
    """
    out_cols = ["beta_adj_residual_ret_z21", "beta_adj_residual_ret_z63"]

    # ------------------------------------------------------------------
    # Determine ticker 1d return
    # ------------------------------------------------------------------
    if "ret_1d" in df.columns:
        tk_ret = df["ret_1d"]
    elif "close" in df.columns:
        tk_ret = np.log(df["close"] / df["close"].shift(1)).shift(1)
    else:
        for col in out_cols:
            df[col] = 0.0
        return df

    # ------------------------------------------------------------------
    # Determine SPY 1d return proxy
    # ------------------------------------------------------------------
    if "spy_return_1d" in df.columns:
        spy_1d = df["spy_return_1d"].shift(1)
    elif "spy_return_5d" in df.columns:
        # approximate 1d from 5d cumulative; not perfect but good enough
        spy_1d = df["spy_return_5d"].diff(1).shift(1) / 5.0
    elif "spy_relative_return_5d" in df.columns and "ret_5d" in df.columns:
        # spy_relative_return_5d = ret_5d_ticker - ret_5d_spy
        # => spy_5d = ret_5d - spy_relative_return_5d
        # => spy_1d_proxy ~ spy_5d / 5
        spy_5d = df["ret_5d"] - df["spy_relative_return_5d"]
        spy_1d = (spy_5d / 5.0).shift(1)
    elif "spy_rel_return_5d" in df.columns and "ret_5d" in df.columns:
        spy_5d = df["ret_5d"] - df["spy_rel_return_5d"]
        spy_1d = (spy_5d / 5.0).shift(1)
    else:
        for col in out_cols:
            df[col] = 0.0
        return df

    # ------------------------------------------------------------------
    # Beta: use existing column or compute on the fly
    # ------------------------------------------------------------------
    if "beta_spy_60d" in df.columns:
        beta = df["beta_spy_60d"].shift(1)  # already safe; extra shift for insurance
    else:
        # Compute rolling beta on raw 1d returns
        cov = tk_ret.rolling(60, min_periods=20).cov(spy_1d)
        var_spy = spy_1d.rolling(60, min_periods=20).var()
        beta = _safe_div(cov, var_spy, fill=1.0)

    # ------------------------------------------------------------------
    # Residual and z-scores
    # ------------------------------------------------------------------
    residual = tk_ret - beta * spy_1d
    df["beta_adj_residual_ret_z21"] = _rolling_zscore(residual, 21).shift(1)
    df["beta_adj_residual_ret_z63"] = _rolling_zscore(residual, 63).shift(1)

    return df


# ---------------------------------------------------------------------------
# Feature 2: Cross-Sector Reversion Score (CSRS)
# ---------------------------------------------------------------------------

def _add_cross_sector_reversion_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sector reversion score at 3 lookbacks.

    csrs_Nd = (ret_Nd_ticker - ret_Nd_sector) / rolling_std_diff_63d
              * rank_percentile(beta_spy_60d, clip to [0.1, 2.0])

    Interpretation: positive = ticker underperformed sector by a lot (relative
    to historical dispersion) for a high-beta name → stronger mean-reversion pull
    back toward sector.

    Inputs expected (shifted already):
      - 'sector_relative_return_5d' (from cross_sectional)
      - 'sector_return_5d'          (from cross_sectional)
      - 'ret_5d' / 'ret_10d' / 'ret_20d' from base features
      - 'beta_spy_60d'              (from cross_sectional)
    """
    out_cols = ["csrs_5d", "csrs_10d", "csrs_20d"]

    # ------------------------------------------------------------------
    # Sector relative returns
    # ------------------------------------------------------------------
    if "sector_relative_return_5d" in df.columns:
        sector_rel_5d = df["sector_relative_return_5d"].shift(1)
    else:
        for col in out_cols:
            df[col] = 0.0
        return df

    # 10d and 20d: approximate from cumulative if not available
    if "ret_10d" in df.columns and "sector_return_5d" in df.columns:
        # sector_return_5d is 5d mean — 10d ~ 2× 5d
        sector_ret_10d = df["sector_return_5d"].shift(1) * 2.0
        rel_10d = df["ret_10d"].shift(1) - sector_ret_10d
    else:
        rel_10d = sector_rel_5d * 2.0  # crude approximation

    if "ret_21d" in df.columns and "sector_return_5d" in df.columns:
        sector_ret_20d = df["sector_return_5d"].shift(1) * 4.0
        rel_20d = df["ret_21d"].shift(1) - sector_ret_20d
    else:
        rel_20d = sector_rel_5d * 4.0

    # ------------------------------------------------------------------
    # Rolling std of the differential (63d) for normalisation
    # ------------------------------------------------------------------
    std_diff_63 = sector_rel_5d.rolling(63, min_periods=21).std()
    std_diff_63 = std_diff_63.replace(0, np.nan)

    # ------------------------------------------------------------------
    # Beta rank: re-scale beta percentile so high-beta = stronger weight
    # beta_spy_60d already shifted; rank among rolling window not possible
    # cross-sectionally, so use sigmoid of beta as a scalar weight
    # ------------------------------------------------------------------
    if "beta_spy_60d" in df.columns:
        beta = df["beta_spy_60d"].clip(0.1, 2.5).shift(1)
        # Normalize: center at 1.0 beta, scale up to ~1.5 for beta=2
        beta_weight = (beta / 1.0).clip(0.5, 2.0)
    else:
        beta_weight = pd.Series(1.0, index=df.index)

    # ------------------------------------------------------------------
    # CSRS = (relative_return / std_diff_63) * beta_weight
    # Positive means ticker *under*performed sector (expects reversion UP)
    # ------------------------------------------------------------------
    df["csrs_5d"]  = (_safe_div(-sector_rel_5d, std_diff_63) * beta_weight).shift(1)
    df["csrs_10d"] = (_safe_div(-rel_10d,        std_diff_63) * beta_weight).shift(1)
    df["csrs_20d"] = (_safe_div(-rel_20d,        std_diff_63) * beta_weight).shift(1)

    # Clip to reasonable range
    for col in out_cols:
        df[col] = df[col].clip(-5, 5)

    return df


# ---------------------------------------------------------------------------
# Feature 3: Earnings Contamination Harmonic Gate
# ---------------------------------------------------------------------------

def _add_earnings_contamination_gate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Harmonic decay gate combining:
      - Pre-earnings proximity decay: 1 / (1 + days_to_next_earn^2) * rv_percentile_20d
      - Post-earnings gap gate:       1 / (1 + days_since_earn^2)  * gap_size_rel_atr

    Unlike the binary is_earnings_week, this gives XGBoost a smooth scalar.
    Near earnings (days=1): gate ≈ 0.5 × rv_pct → XGBoost learns "suppress reversion".
    Far from earnings (days=20): gate ≈ 0.002 → signal passes through cleanly.

    Inputs expected:
      - 'days_until_earnings'      (alpaca_features, shifted)
      - 'days_since_last_earnings' (alpaca_features, shifted)
      - 'atr_14'                   (base features, shifted)
      - 'close', 'open'            for gap computation
    """
    out_cols = ["earn_contam_gate", "earn_post_rv_gate"]

    # ------------------------------------------------------------------
    # Realized volatility percentile (20d) -- used as modulator
    # rv_percentile_20d: 0 = calm, 1 = high vol
    # ------------------------------------------------------------------
    if "atr_14" in df.columns and "close" in df.columns:
        rv_20d = df["close"].pct_change().rolling(20, min_periods=10).std().shift(1)
        rv_pct = rv_20d.rank(pct=True)  # cross-time percentile, not cross-section
    else:
        rv_pct = pd.Series(0.5, index=df.index)

    # ------------------------------------------------------------------
    # Pre-earnings harmonic decay
    # ------------------------------------------------------------------
    if "days_until_earnings" in df.columns:
        d_to = df["days_until_earnings"].shift(1).clip(0, 90).fillna(90)
        pre_gate = 1.0 / (1.0 + d_to ** 2) * rv_pct
    else:
        pre_gate = pd.Series(0.0, index=df.index)

    # ------------------------------------------------------------------
    # Post-earnings gap × ATR gate
    # ------------------------------------------------------------------
    if "days_since_last_earnings" in df.columns and "atr_14" in df.columns \
            and "close" in df.columns and "open" in df.columns:
        d_since = df["days_since_last_earnings"].shift(1).clip(0, 90).fillna(90)
        gap = (df["open"].shift(1) - df["close"].shift(2)).abs()
        atr = df["atr_14"].shift(1).replace(0, np.nan)
        gap_rel_atr = _safe_div(gap, atr, fill=0.0).clip(0, 5)
        post_gate = 1.0 / (1.0 + d_since ** 2) * gap_rel_atr
    else:
        post_gate = pd.Series(0.0, index=df.index)

    # ------------------------------------------------------------------
    # Combined gate (both gates active near earnings, both decay away)
    # ------------------------------------------------------------------
    df["earn_contam_gate"]   = (pre_gate + post_gate).clip(0, 1).shift(1)
    df["earn_post_rv_gate"]  = post_gate.shift(1).clip(0, 1)

    return df


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def add_daily_integration_features(
    df: pd.DataFrame,
    ticker: str | None = None,
) -> pd.DataFrame:
    """
    Top-3 features from daily discovery review 2026-05-16. .shift(1) safe.

    Adds 7 columns:
      beta_adj_residual_ret_z21   -- idiosyncratic residual z (21d)
      beta_adj_residual_ret_z63   -- idiosyncratic residual z (63d)
      csrs_5d                     -- cross-sector reversion score 5d
      csrs_10d                    -- cross-sector reversion score 10d
      csrs_20d                    -- cross-sector reversion score 20d
      earn_contam_gate            -- earnings contamination harmonic gate
      earn_post_rv_gate           -- post-earnings gap regime gate

    Parameters
    ----------
    df     : DataFrame with at minimum 'close', 'open', 'atr_14', and ideally
             the cross-sectional and alpaca feature columns.
    ticker : optional ticker string (used for logging only).

    Returns
    -------
    df with 7 new feature columns appended (in-place modification on copy).
    """
    df = df.copy()

    try:
        df = _add_beta_adj_residual(df)
    except Exception as exc:
        print(f"  [daily_integration] beta_adj_residual failed ({ticker}): {exc}")
        df["beta_adj_residual_ret_z21"] = 0.0
        df["beta_adj_residual_ret_z63"] = 0.0

    try:
        df = _add_cross_sector_reversion_score(df)
    except Exception as exc:
        print(f"  [daily_integration] csrs failed ({ticker}): {exc}")
        df["csrs_5d"] = 0.0
        df["csrs_10d"] = 0.0
        df["csrs_20d"] = 0.0

    try:
        df = _add_earnings_contamination_gate(df)
    except Exception as exc:
        print(f"  [daily_integration] earn_gate failed ({ticker}): {exc}")
        df["earn_contam_gate"]  = 0.0
        df["earn_post_rv_gate"] = 0.0

    return df


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from pathlib import Path

    SCRIPTS = Path(__file__).parent
    sys.path.insert(0, str(SCRIPTS))

    try:
        import backtest_ml as bml
        import alt_data_features as adf
        import alpaca_features as alf
        import cross_sectional_features as csf

        ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
        print(f"Smoke test: {ticker}")
        d = bml.load_daily(ticker)
        f = bml.build_features(d)

        # Add alpaca (earnings) features
        try:
            f = alf.add_alpaca_features(f, ticker)
            print(f"  +alpaca: {f.shape[1]}")
        except Exception as e:
            print(f"  [warn] alpaca: {e}")

        # Add cross-sectional (sector, beta)
        try:
            agg = csf.precompute_universe_aggregates()
            f = csf.add_cross_sectional_features(f, ticker, agg)
            print(f"  +cross_sectional: {f.shape[1]}")
        except Exception as e:
            print(f"  [warn] csf: {e}")

        before = f.shape[1]
        f = add_daily_integration_features(f, ticker)
        added = f.shape[1] - before
        print(f"  +daily_integration: added {added} cols → total {f.shape[1]}")

        new_cols = [
            "beta_adj_residual_ret_z21", "beta_adj_residual_ret_z63",
            "csrs_5d", "csrs_10d", "csrs_20d",
            "earn_contam_gate", "earn_post_rv_gate",
        ]
        print("\nNew column stats:")
        print(f[new_cols].describe().round(4))
        nonzero = {c: (f[c] != 0).sum() for c in new_cols}
        print("\nNon-zero rows:", nonzero)

    except Exception as exc:
        print(f"Smoke test failed: {exc}")
        raise
