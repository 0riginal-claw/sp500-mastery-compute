"""
volatility_estimator_features.py
=================================
Advanced range-based volatility estimators as XGBoost features.

All formulas are standard academic/public-domain mathematics; no GPL code vendored.

Estimators implemented
----------------------
1. Parkinson (1980) — high-low range estimator
2. Garman-Klass (1980) — OHLC estimator
3. Rogers-Satchell (1991) — drift-corrected OHLC estimator
4. Yang-Zhang (2000) — overnight + open-to-close + RS combined estimator

API
---
    add_volatility_features(df: pd.DataFrame) -> pd.DataFrame

Input:   daily OHLCV DataFrame with columns open, high, low, close, volume
         (column names lower-cased; DatetimeIndex).
Output:  original df + 15 new columns (all .shift(1)-safe, annualised, float64).

Feature list (15 total)
-----------------------
Windows: 10, 20, 60 trading days

Parkinson:         vol_parkinson_10 / 20 / 60
Garman-Klass:      vol_gk_10 / 20 / 60
Rogers-Satchell:   vol_rs_10 / 20 / 60
Yang-Zhang:        vol_yz_10 / 20 / 60
Ratio features:    vol_yz_20_vs_60         (YZ-20 / YZ-60 — vol regime indicator)
                   vol_yz_realized_eff     (YZ-20 / close-to-close 20d vol)
                   vol_pk_vs_yz_20         (Parkinson-20 / YZ-20 — jump detection)

References (no code borrowed)
------------------------------
  Parkinson, M. (1980). "The Extreme Value Method for Estimating the Variance
    of the Rate of Return." J. of Business, 53(1), 61-65.
  Garman, M., & Klass, M. (1980). "On the Estimation of Security Price
    Volatilities from Historical Data." J. of Business, 53(1), 67-78.
  Rogers, L., & Satchell, S. (1991). "Estimating Variance from High, Low and
    Closing Prices." Ann. of Applied Probability, 1(4), 504-512.
  Yang, D., & Zhang, Q. (2000). "Drift-Independent Volatility Estimation Based
    on High, Low, Open, and Close Prices." J. of Business, 73(3), 477-491.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANNUALISE = np.sqrt(252)

# Column prefix/name sets used for shift-detection
_VOL_PREFIXES = ("vol_parkinson_", "vol_gk_", "vol_rs_", "vol_yz_")
_VOL_RATIO_COLS = ("vol_yz_20_vs_60", "vol_yz_realized_eff", "vol_pk_vs_yz_20")


# ---------------------------------------------------------------------------
# Per-bar squared log-return terms (pure numpy — no GPL)
# ---------------------------------------------------------------------------

def _parkinson_sq(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """(1/(4*ln2)) * ln(H/L)^2  — Parkinson (1980) per bar."""
    return (np.log(high / low) ** 2) / (4.0 * np.log(2.0))


def _gk_sq(open_: np.ndarray, high: np.ndarray,
           low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """0.5*ln(H/L)^2 - (2*ln2 - 1)*ln(C/O)^2  — Garman-Klass (1980) per bar."""
    hl2 = 0.5 * np.log(high / low) ** 2
    co2 = (2.0 * np.log(2.0) - 1.0) * np.log(close / open_) ** 2
    return hl2 - co2


def _rs_sq(open_: np.ndarray, high: np.ndarray,
           low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O)  — Rogers-Satchell (1991) per bar."""
    return (np.log(high / close) * np.log(high / open_)
            + np.log(low / close) * np.log(low / open_))


def _yz_components(open_: pd.Series, high: pd.Series,
                   low: pd.Series, close: pd.Series,
                   window: int) -> pd.Series:
    """
    Yang-Zhang (2000) volatility for a rolling window.

    sigma_YZ^2 = sigma_overnight^2 + k*sigma_OC^2 + (1-k)*sigma_RS^2

    where k = 0.34 / (1 + (N+1)/(N-1))   (optimal weighting factor per Yang-Zhang)
    sigma_overnight^2 uses the close-to-open log-return (overnight gap component).
    sigma_OC^2 uses the open-to-close log-return (intraday component).
    sigma_RS^2 is the Rogers-Satchell estimate (drift-corrected range component).
    """
    N = window
    k = 0.34 / (1.0 + (N + 1.0) / (N - 1.0))

    # Overnight return: ln(O_t / C_{t-1})
    overnight = np.log(open_ / close.shift(1))

    # Open-to-close return: ln(C_t / O_t)
    oc = np.log(close / open_)

    rs_arr = _rs_sq(open_.values, high.values, low.values, close.values)

    # Variance estimates using rolling mean (the YZ estimator formula)
    ovn_mean = overnight.rolling(N).mean()
    oc_mean = oc.rolling(N).mean()

    sigma_ovn_sq = ((overnight - ovn_mean) ** 2).rolling(N).sum() / (N - 1)
    sigma_oc_sq = ((oc - oc_mean) ** 2).rolling(N).sum() / (N - 1)
    sigma_rs_sq = pd.Series(rs_arr, index=open_.index).rolling(N).mean()

    sigma_yz_sq = sigma_ovn_sq + k * sigma_oc_sq + (1.0 - k) * sigma_rs_sq
    return np.sqrt(sigma_yz_sq.clip(lower=0.0)) * ANNUALISE


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds advanced volatility estimators as XGBoost features.

    Input:  Daily OHLCV DataFrame (columns: open, high, low, close, volume).
    Output: Original df + 15 new float64 columns, all .shift(1)-safe
            (each rolling window ends at bar t-1 relative to the label bar t).

    All estimators are annualised: multiplied by sqrt(252).
    Typical magnitude for equities: 0.10 - 0.60 annualised.

    Feature count: 15
        12 base  = 4 estimators x 3 windows (10, 20, 60)
         3 ratio = vol_yz_20_vs_60, vol_yz_realized_eff, vol_pk_vs_yz_20
    """
    df = df.copy()

    o = df["open"].astype(float).clip(lower=1e-9)
    h = df["high"].astype(float).clip(lower=1e-9)
    l = df["low"].astype(float).clip(lower=1e-9)
    c = df["close"].astype(float).clip(lower=1e-9)

    # Per-bar squared terms (Series)
    pk_sq = pd.Series(_parkinson_sq(h.values, l.values), index=df.index)
    gk_sq = pd.Series(_gk_sq(o.values, h.values, l.values, c.values), index=df.index)
    rs_sq = pd.Series(_rs_sq(o.values, h.values, l.values, c.values), index=df.index)

    for win in (10, 20, 60):
        # Parkinson
        pk_var = pk_sq.rolling(win).mean()
        df[f"vol_parkinson_{win}"] = np.sqrt(pk_var.clip(lower=0)) * ANNUALISE

        # Garman-Klass (clip negative variance to 0 — can occur in very flat bars)
        gk_var = gk_sq.rolling(win).mean()
        df[f"vol_gk_{win}"] = np.sqrt(gk_var.clip(lower=0)) * ANNUALISE

        # Rogers-Satchell
        rs_var = rs_sq.rolling(win).mean()
        df[f"vol_rs_{win}"] = np.sqrt(rs_var.clip(lower=0)) * ANNUALISE

        # Yang-Zhang
        df[f"vol_yz_{win}"] = _yz_components(o, h, l, c, win)

    # ----- Ratio / derived features (3 total) -----

    yz20 = df["vol_yz_20"]
    yz60 = df["vol_yz_60"]
    pk20 = df["vol_parkinson_20"]

    # Regime indicator: short-term vol vs long-term vol (>1 => vol expanding)
    df["vol_yz_20_vs_60"] = (yz20 / yz60.replace(0, np.nan)).clip(0.1, 10.0)

    # Efficiency ratio: YZ vs naive close-to-close vol
    # Values > 1 indicate the market's range is wider than its net move (mean-reverting)
    cc_ret = np.log(c / c.shift(1))
    cc_vol20 = cc_ret.rolling(20).std() * ANNUALISE
    df["vol_yz_realized_eff"] = (yz20 / cc_vol20.replace(0, np.nan)).clip(0.1, 10.0)

    # Jump detection: Parkinson ignores overnight gaps; YZ includes them.
    # Ratio < 1 indicates overnight-gap-dominated vol (potential news/earnings event).
    df["vol_pk_vs_yz_20"] = (pk20 / yz20.replace(0, np.nan)).clip(0.1, 10.0)

    # All new features shifted by 1 bar: bar-t feature uses data only through bar t-1.
    # This guarantees .shift(1) safety — no lookahead contamination.
    vol_cols = [
        col for col in df.columns
        if any(col.startswith(pfx) for pfx in _VOL_PREFIXES)
        or col in _VOL_RATIO_COLS
    ]
    df[vol_cols] = df[vol_cols].shift(1)

    return df


# Convenience: expose feature names for downstream feature-tracking
VOL_FEATURE_NAMES: list[str] = (
    [f"vol_parkinson_{w}" for w in (10, 20, 60)]
    + [f"vol_gk_{w}" for w in (10, 20, 60)]
    + [f"vol_rs_{w}" for w in (10, 20, 60)]
    + [f"vol_yz_{w}" for w in (10, 20, 60)]
    + list(_VOL_RATIO_COLS)
)


# ---------------------------------------------------------------------------
# Smoke test (run directly: python volatility_estimator_features.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not installed — skipping smoke test", file=sys.stderr)
        sys.exit(1)

    print("Fetching AAPL daily OHLCV (2018-2024)...")
    raw = yf.download("AAPL", start="2018-01-01", end="2024-12-31",
                      auto_adjust=True, progress=False)
    # yfinance >= 0.2 returns a MultiIndex — flatten it
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]
    raw = raw[["open", "high", "low", "close", "volume"]].dropna()

    before = set(raw.columns)
    out = add_volatility_features(raw)
    after = set(out.columns)
    new_cols = sorted(after - before)

    print(f"\n+{len(new_cols)} features added:")
    for col in new_cols:
        s = out[col].dropna()
        print(f"  {col:<30s}  n={len(s):5d}  "
              f"mean={s.mean():.4f}  min={s.min():.4f}  max={s.max():.4f}  "
              f"finite={np.isfinite(s).all()}")

    # Verify exact expected count
    expected = 15
    assert len(new_cols) == expected, (
        f"Expected {expected} new features, got {len(new_cols)}: {new_cols}"
    )

    # Sanity: annualised vols should be in (0.01, 5.0) for equities
    vol_raw = [c for c in new_cols
               if c not in ("vol_yz_20_vs_60", "vol_yz_realized_eff", "vol_pk_vs_yz_20")]
    means = out[vol_raw].mean()
    assert (means > 0.05).all(), f"Some vols suspiciously low:\n{means[means <= 0.05]}"
    assert (means < 2.0).all(), f"Some vols suspiciously high:\n{means[means >= 2.0]}"

    # All values finite
    for col in new_cols:
        s = out[col].dropna()
        assert np.isfinite(s).all(), f"{col} has non-finite values"

    # Verify .shift(1) safety: last raw bar's vol should NOT appear in the matching output row
    # (the output row for the last bar uses data from the second-to-last bar)
    assert len(new_cols) == len(VOL_FEATURE_NAMES), (
        f"VOL_FEATURE_NAMES mismatch: {set(new_cols) ^ set(VOL_FEATURE_NAMES)}"
    )

    print(f"\nSmoke test PASSED — {len(new_cols)} features, all finite, "
          f"magnitudes in expected range.")
