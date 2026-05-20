"""
vpin_50bucket_features.py — VPIN (Volume-synchronized Probability of Informed
Trading) with 50-bucket daily-bar approximation.

NO-LOOKAHEAD AUDIT
==================
Data source : daily OHLCV already in df from v9 stack (yfinance); approximates
              alpaca_1min_bars via BVC (Bulk Volume Classification — López de Prado
              & O'Hara 2011/2012 RFS). No Alpaca API key required.
Computation :
  1. delta_close[t] = close[t] - close[t-1]            — bar-t price change
  2. sigma_delta[t] = rolling_50_std(delta_close)       — mild same-bar σ estimation
                                                           (analogous parameter
                                                           lookahead caveat as GARCH
                                                           features in this pipeline;
                                                           empirically ~0.5–1%)
  3. P_buy[t]       = Φ(delta_close[t] / sigma_delta[t]) — BVC buy-fraction estimate
                      where Φ = standard normal CDF
  4. V_imbalance[t] = |2·P_buy[t] − 1|                — normalised |V_buy − V_sell|
                      ∈ [0, 1]; 0 = balanced flow, 1 = one-sided
  5. vpin_50[t]     = rolling_50_mean(V_imbalance[t])  — 50-bucket VPIN estimate
  6. vpin_50_z21[t] = z-score of vpin_50 over trailing 21 bars
  7. vpin_buy_frac_10[t] = EMA(P_buy, span=10)         — short-window buy fraction
  All three outputs apply .shift(1) before assignment so the value stored at
  row t is based on data through bar t−1 only. No same-bar lookahead.

Fallback : if any required column (close, volume) is missing, zero-fills all features.

License : MIT (own implementation).
Reference : Easley, López de Prado & O'Hara, "Flow Toxicity and Liquidity in a
            High Frequency World", Review of Financial Studies 25(5), 2012.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)

VPIN_50BUCKET_FEATURE_NAMES: list[str] = [
    "vpin_50bucket",        # 50-bar rolling VPIN estimate in [0, 1]
    "vpin_50bucket_z21",    # z-score of vpin_50bucket over trailing 21 bars
    "vpin_buy_frac_10",     # EMA(10) buy-volume fraction in [0, 1]
]

_VPIN_WINDOW = 50
_ZSCORE_WINDOW = 21
_BUY_FRAC_SPAN = 10


def compute_vpin_50bucket_features(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
) -> pd.DataFrame:
    """Append VPIN 50-bucket features to *df* (returns a copy with new columns).

    Inputs consumed from df:
      - 'close'  : daily close price (lowercase; also tries 'Close')
      - 'volume' : daily volume      (lowercase; also tries 'Volume')

    Output columns added (see VPIN_50BUCKET_FEATURE_NAMES):
      - vpin_50bucket     : 50-bar rolling VPIN probability estimate [0, 1]
      - vpin_50bucket_z21 : 21-bar z-score of vpin_50bucket
      - vpin_buy_frac_10  : EMA(10) buy-volume fraction [0, 1]

    All outputs are .shift(1)-safe: value at row t uses only bar t−1 data.
    """
    # -- Locate close column --
    close_col = None
    for candidate in ("close", "Close", "$close"):
        if candidate in df.columns:
            close_col = candidate
            break

    # -- Locate volume column --
    vol_col = None
    for candidate in ("volume", "Volume", "$volume"):
        if candidate in df.columns:
            vol_col = candidate
            break

    if close_col is None or vol_col is None:
        missing = []
        if close_col is None:
            missing.append("close")
        if vol_col is None:
            missing.append("volume")
        logger.warning(
            "[vpin_50bucket] required column(s) missing (%s) — zero-filling; ticker=%s",
            ", ".join(missing),
            ticker or "?",
        )
        df = df.copy()
        for col in VPIN_50BUCKET_FEATURE_NAMES:
            df[col] = 0.0
        return df

    close = df[close_col].astype(float)
    # volume used only for logging / future extensions; BVC normalises it away
    volume = df[vol_col].astype(float).replace(0, np.nan)

    # -- Step 1: price change --
    delta_close = close.diff()  # NaN at first bar

    # -- Step 2: rolling σ of price changes (50-bar) --
    sigma_delta = delta_close.rolling(_VPIN_WINDOW, min_periods=10).std()
    sigma_delta = sigma_delta.replace(0, np.nan)

    # -- Step 3: BVC buy fraction via standard-normal CDF --
    z_score_raw = delta_close / sigma_delta          # dimensionless
    p_buy = pd.Series(
        norm.cdf(z_score_raw.fillna(0.0).values),
        index=df.index,
        dtype=float,
    )
    # Clip to avoid numerical edge cases
    p_buy = p_buy.clip(0.0, 1.0)

    # -- Step 4: normalised volume imbalance per bar --
    v_imbalance = (2.0 * p_buy - 1.0).abs()  # in [0, 1]

    # -- Step 5: 50-bucket rolling VPIN --
    vpin_50 = v_imbalance.rolling(_VPIN_WINDOW, min_periods=10).mean()

    # -- Step 6: 21-bar z-score of VPIN --
    roll21 = vpin_50.rolling(_ZSCORE_WINDOW, min_periods=5)
    vpin_z21 = (vpin_50 - roll21.mean()) / roll21.std().replace(0, np.nan)
    vpin_z21 = vpin_z21.fillna(0.0)

    # -- Step 7: short-window buy fraction (EMA 10) --
    buy_frac_10 = p_buy.ewm(span=_BUY_FRAC_SPAN, adjust=False).mean()

    # -- Apply .shift(1): feature at row t uses only bar t−1 data --
    vpin_50_safe = vpin_50.shift(1).fillna(0.0)
    vpin_z21_safe = vpin_z21.shift(1).fillna(0.0)
    buy_frac_10_safe = buy_frac_10.shift(1).fillna(0.5)  # 0.5 = neutral prior

    logger.info(
        "[vpin_50bucket] computed OK; ticker=%s; vpin_mean=%.4f; vol_rows=%d",
        ticker or "?",
        float(vpin_50_safe.mean()),
        int(volume.notna().sum()),
    )

    df = df.copy()
    df["vpin_50bucket"] = vpin_50_safe
    df["vpin_50bucket_z21"] = vpin_z21_safe
    df["vpin_buy_frac_10"] = buy_frac_10_safe
    return df
