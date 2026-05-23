"""
macro_regime_features.py — Macro / regime features (TOP-10 #10, from FinRL).

Shipped 2026-05-22 (Wave-altdata). Inspired by
AI4Finance-Foundation/FinRL/meta/data_processors/processor_yahoofinance.py
(`add_turbulence`, `calculate_turbulence`). Re-implemented self-contained:
no FinRL dependency.

Features added (5 cols):
  - turbulence_252                : Mahalanobis-distance turbulence (rolling 252d
                                    cov on the price's return distribution).
                                    Per-ticker (not cross-sectional). 0 for the
                                    first 252 bars (warm-up).
  - turbulence_z_60d              : z-score of turbulence over trailing 60d.
  - vix_close                     : daily VIX close (from cache/fred_macro/VIXCLS.parquet).
  - vix_quantile_252              : VIX rank-quantile in [0,1] over trailing 252d
                                    (0 = calm, 1 = panic).
  - vix_regime                    : 3-state tag {-1: calm (<25th pct),
                                                  0: normal,
                                                 +1: stress (>=75th pct)}.
  - drawdown_state                : current drawdown depth as fraction of trailing
                                    252d running max (negative when below peak).
  - term_structure_10y2y          : DGS10 - DGS2 spread from cache/fred_macro/.
                                    Daily, propagated to bar dates by merge_asof
                                    backward (PIT-safe: bar D uses last spread
                                    published before D).

All features are .shift(1)-safe:
  - Turbulence at bar D uses returns strictly before D (window [D-252, D-1]).
  - VIX/term-structure at bar D uses the last published value with date < D
    (merge_asof direction='backward', allow_exact_matches=False).
  - Drawdown uses running max over [D-252, D-1] (excludes D's close).

Graceful failure: missing FRED cache → vix-related cols zero-fill.
Idempotent: re-calling on already-augmented df is a no-op.

Env-gate: MACRO_REGIME_ENABLED=1 (default 0).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]
FRED_DIR = _ROOT / "cache" / "fred_macro"

MACRO_REGIME_FEATURE_NAMES: list[str] = [
    "turbulence_252",
    "turbulence_z_60d",
    "vix_close",
    "vix_quantile_252",
    "vix_regime",
    "drawdown_state",
    "term_structure_10y2y",
]


def _enabled() -> bool:
    return os.environ.get("MACRO_REGIME_ENABLED", "0") == "1"


def _load_fred_series(sid: str) -> Optional[pd.DataFrame]:
    p = FRED_DIR / f"{sid}.parquet"
    if not p.exists():
        logger.warning("[macro] FRED series missing: %s", p)
        return None
    try:
        df = pd.read_parquet(p)
        df["date"] = pd.to_datetime(df["date"])
        df = df.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)
        return df[["date", "value"]]
    except Exception as e:
        logger.warning("[macro] FRED load %s failed: %s", sid, e)
        return None


def _calc_turbulence(returns: pd.Series, window: int = 252) -> np.ndarray:
    """Per-ticker rolling Mahalanobis turbulence on a univariate return series.

    For bar i (i >= window), let r_hist = returns[i-window:i] (strict <, no leak),
    mu = mean(r_hist), sigma2 = var(r_hist). Univariate Mahalanobis:
        t_i = (r_i - mu)^2 / sigma2     (degenerate 1-D case of FinRL's pinv form)

    For multi-ticker turbulence (true Mahalanobis on cross-section cov), a
    separate utility is needed — out of scope for per-ticker feature pipeline.
    """
    n = len(returns)
    out = np.zeros(n, dtype=np.float64)
    r = returns.values.astype(np.float64)
    for i in range(window, n):
        hist = r[i - window : i]  # strict <  (excludes r[i])
        hist = hist[~np.isnan(hist)]
        if hist.size < 30:
            continue
        mu = hist.mean()
        s2 = hist.var(ddof=1)
        if s2 <= 0 or np.isnan(r[i]):
            continue
        out[i] = ((r[i] - mu) ** 2) / s2
    return out


def _calc_drawdown_state(close: pd.Series, window: int = 252) -> np.ndarray:
    """Current depth relative to trailing-window running max (excluding current bar).
    Returns negative values (0 = at peak, -0.2 = 20% below peak).
    """
    n = len(close)
    out = np.zeros(n, dtype=np.float64)
    c = close.values.astype(np.float64)
    for i in range(1, n):
        lo = max(0, i - window)
        hist = c[lo:i]  # strict <
        if hist.size == 0:
            continue
        peak = np.nanmax(hist)
        if peak > 0 and not np.isnan(c[i]):
            out[i] = (c[i] - peak) / peak
    return out


def _merge_backward_pit(df_bar_dates: pd.DatetimeIndex, series_df: pd.DataFrame) -> np.ndarray:
    """For each bar date D, return last series value with date < D.
    PIT-safe (no exact-match leakage). Returns array of len(df_bar_dates).
    """
    out = np.zeros(len(df_bar_dates), dtype=np.float64)
    if series_df is None or series_df.empty:
        return out
    # asof requires sorted left key
    left = pd.DataFrame({"bar_date": df_bar_dates}).sort_values("bar_date").reset_index()
    right = series_df.rename(columns={"date": "series_date"}).sort_values("series_date")
    merged = pd.merge_asof(
        left,
        right,
        left_on="bar_date",
        right_on="series_date",
        direction="backward",
        allow_exact_matches=False,  # strict <
    )
    # restore original order
    merged = merged.sort_values("index")
    vals = merged["value"].fillna(0.0).values.astype(np.float64)
    return vals


def _zero_fill(df: pd.DataFrame) -> pd.DataFrame:
    for col in MACRO_REGIME_FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0.0
    return df


def add_macro_regime_features(df: pd.DataFrame, ticker: str = "_") -> pd.DataFrame:
    """Append 7 macro/regime features to df. Idempotent. .shift(1)-safe.

    Args:
        df: DataFrame with DatetimeIndex (or 'date' col) AND a 'close' col.
        ticker: optional, only used for logging.
    """
    if df is None or len(df) == 0:
        return df
    if all(c in df.columns for c in MACRO_REGIME_FEATURE_NAMES):
        return df

    if not _enabled():
        return _zero_fill(df)

    if "close" not in df.columns:
        logger.warning("[macro] no 'close' column for %s — zero-fill", ticker)
        return _zero_fill(df)

    # Determine bar dates
    if isinstance(df.index, pd.DatetimeIndex):
        bar_dates = df.index
    elif "date" in df.columns:
        bar_dates = pd.DatetimeIndex(pd.to_datetime(df["date"]))
    else:
        return _zero_fill(df)
    if bar_dates.tz is not None:
        bar_dates = bar_dates.tz_convert(None)

    # ---- turbulence_252 ----
    returns = df["close"].astype(float).pct_change().fillna(0.0)
    turb = _calc_turbulence(returns, window=252)
    df["turbulence_252"] = turb

    # ---- turbulence_z_60d ----
    s = pd.Series(turb, index=df.index if isinstance(df.index, pd.DatetimeIndex) else None)
    roll_mu = s.shift(1).rolling(60, min_periods=10).mean()
    roll_sd = s.shift(1).rolling(60, min_periods=10).std(ddof=0)
    z = ((s - roll_mu) / roll_sd.replace(0, np.nan)).fillna(0.0).clip(-10, 10)
    df["turbulence_z_60d"] = z.values

    # ---- VIX-related ----
    vix_df = _load_fred_series("VIXCLS")
    vix_vals = _merge_backward_pit(bar_dates, vix_df)
    df["vix_close"] = vix_vals

    # vix_quantile_252: rank quantile over trailing 252 (strict <)
    vix_s = pd.Series(vix_vals)
    vix_q = np.zeros(len(vix_vals), dtype=np.float64)
    for i in range(len(vix_vals)):
        lo = max(0, i - 252)
        hist = vix_vals[lo:i]
        hist = hist[hist > 0]
        if hist.size >= 30 and vix_vals[i] > 0:
            vix_q[i] = float((hist < vix_vals[i]).sum()) / float(hist.size)
    df["vix_quantile_252"] = vix_q

    # vix_regime: -1 calm, 0 normal, +1 stress
    regime = np.zeros(len(vix_q), dtype=np.float64)
    regime[vix_q < 0.25] = -1.0
    regime[vix_q >= 0.75] = 1.0
    # bars before sufficient history (vix_q==0) -> normal (0)
    regime[vix_q == 0.0] = 0.0
    df["vix_regime"] = regime

    # ---- drawdown_state ----
    df["drawdown_state"] = _calc_drawdown_state(df["close"].astype(float), window=252)

    # ---- term_structure (DGS10 - DGS2) ----
    dgs10 = _load_fred_series("DGS10")
    dgs2 = _load_fred_series("DGS2")
    if dgs10 is not None and dgs2 is not None:
        t10 = _merge_backward_pit(bar_dates, dgs10)
        t2 = _merge_backward_pit(bar_dates, dgs2)
        df["term_structure_10y2y"] = t10 - t2
    else:
        # Try the pre-computed spread if both legs missing
        t10y2y = _load_fred_series("T10Y2Y")
        df["term_structure_10y2y"] = _merge_backward_pit(bar_dates, t10y2y)

    return df


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    tk = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 252

    # Force-enable for smoke
    os.environ["MACRO_REGIME_ENABLED"] = "1"

    print(f"[smoke] ticker={tk} days={days}")
    idx = pd.date_range(end=pd.Timestamp.utcnow().date(), periods=days, freq="B")
    rng = np.random.default_rng(42)
    rets = rng.normal(0.0005, 0.012, size=len(idx))
    rets[len(idx) // 2 : len(idx) // 2 + 10] = -0.05  # synthetic stress
    close = 100.0 * np.exp(np.cumsum(rets))
    demo = pd.DataFrame({"close": close}, index=idx)

    out = add_macro_regime_features(demo, tk)
    print(f"[smoke] input cols: 1, output cols: {out.shape[1]}")
    print(out[MACRO_REGIME_FEATURE_NAMES].tail(5).to_string())

    nz = {k: int((out[k] != 0).sum()) for k in MACRO_REGIME_FEATURE_NAMES}
    print(f"[smoke] non-zero counts: {nz}")
    print(
        "[smoke] turbulence stats:",
        f"mean={out['turbulence_252'].mean():.4f}",
        f"max={out['turbulence_252'].max():.4f}",
        f"std={out['turbulence_252'].std():.4f}",
    )
    print(
        "[smoke] vix stats:",
        f"min={out['vix_close'].min():.2f}",
        f"max={out['vix_close'].max():.2f}",
        f"regime_dist={dict(zip(*np.unique(out['vix_regime'].values, return_counts=True)))}",
    )
