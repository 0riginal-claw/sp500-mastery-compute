"""
vol_risk_premium_features.py — VRP (implied minus realized vol).

Wave V-1 (LOW-cost, no new deps). Wired 2026-05-17.

# NO-LOOKAHEAD AUDIT
# ------------------
# VIX is end-of-day settlement reported for the *prior* session.  Like the
# vix_term_structure_v2 module, we merge VIX values backward against
# bar_date - 1 day to enforce strict past-only lookup.  The 21-day realized
# vol uses .shift(1) so row t uses only returns through t-1.
#
# Theory: Bollerslev-Tauchen-Zhou 2009 — VRP (= IV - RV) is a robust equity
# premium predictor; positive expansion correlates with risk-on regimes.
# We compute VRP as VIX(t-1) - SPY_or_ticker_RV21(t-1) on an annualized scale.
#
# Pure pandas/numpy — reuses the cached VIX parquet from vix_term_structure_v2.
# Falls back to live yfinance if cache is missing; if both fail, all 4 features
# zero-fill so the v10 pipeline never crashes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WORK = Path(
    "/Users/orginal/Library/CloudStorage/"
    "GoogleDrive-zachgladstone@gmail.com/My Drive/"
    "AI-Tools/s&p500-ticker-mastery"
)

VRP_FEATURE_NAMES: list[str] = [
    "vrp_market",
    "vrp_ticker",
    "vrp_zscore_252",
    "vrp_sign_flip",
]

_VIX_CACHE_PATH = WORK / "cache" / "vix_term_structure" / "vix_daily.parquet"


def _find_close(df: pd.DataFrame) -> str | None:
    for c in ("close", "Close", "adj_close", "Adj Close"):
        if c in df.columns:
            return c
    return None


def _zero_fill(df: pd.DataFrame) -> pd.DataFrame:
    for col in VRP_FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0.0 if col != "vrp_sign_flip" else 0
    return df


def _load_vix() -> pd.DataFrame | None:
    """Read cached VIX daily close (written by vix_term_structure_v2)."""
    if _VIX_CACHE_PATH.exists():
        try:
            cached = pd.read_parquet(_VIX_CACHE_PATH)
            if "vix_close" in cached.columns and len(cached) > 10:
                return cached[["vix_close"]].copy()
        except Exception as exc:
            logger.debug("[vrp] cache read error: %s", exc)
    # Fallback: yfinance live
    try:
        import yfinance as yf
        end = pd.Timestamp.utcnow().normalize()
        start = end - pd.DateOffset(years=7)
        raw = yf.download(
            "^VIX", start=start.date().isoformat(), end=end.date().isoformat(),
            auto_adjust=True, progress=False,
        )
        if raw is None or raw.empty:
            return None
        col = raw["Close"] if "Close" in raw.columns else raw.iloc[:, 0]
        out = pd.DataFrame({"vix_close": col.values}, index=pd.DatetimeIndex(raw.index).tz_localize(None))
        return out
    except Exception as exc:
        logger.debug("[vrp] yfinance fallback failed: %s", exc)
        return None


def add_vol_risk_premium_features(
    df: pd.DataFrame,
    ticker: str | None = None,
) -> pd.DataFrame:
    """Append 4 VRP features. Idempotent + graceful zero-fill on failure."""
    if df is None or len(df) == 0:
        return df

    if all(c in df.columns for c in VRP_FEATURE_NAMES):
        return df

    close_col = _find_close(df)
    if close_col is None:
        logger.warning("[vrp] close column not found for %s — zeroing", ticker)
        return _zero_fill(df)

    vix_data = _load_vix()
    if vix_data is None or vix_data.empty:
        logger.warning("[vrp] %s: VIX unavailable — zeroing", ticker)
        return _zero_fill(df)

    try:
        close = df[close_col].astype(float)
        log_ret = np.log(close / close.shift(1))
        # 21-day RV annualized (sqrt(252))
        rv21 = log_ret.rolling(21, min_periods=10).std() * np.sqrt(252) * 100.0  # in vol-points like VIX

        # Build bar_date index
        if isinstance(df.index, pd.DatetimeIndex):
            bar_dates = df.index.tz_localize(None) if df.index.tz is not None else df.index
        elif "date" in df.columns:
            bar_dates = pd.DatetimeIndex(pd.to_datetime(df["date"])).tz_localize(None)
        else:
            logger.warning("[vrp] no DatetimeIndex/date column — zeroing")
            return _zero_fill(df)

        # NO-LOOKAHEAD merge_asof backward against bar_date - 1
        shifted = (bar_dates - pd.Timedelta(days=1)).astype("datetime64[us]")
        left = pd.DataFrame({"bar_date": bar_dates, "lookup_date": shifted})

        vix_idx = vix_data.reset_index().rename(columns={vix_data.index.name or "index": "lookup_date"})
        if "lookup_date" not in vix_idx.columns:
            vix_idx.columns = ["lookup_date"] + list(vix_idx.columns[1:])
        vix_idx["lookup_date"] = pd.to_datetime(vix_idx["lookup_date"]).dt.tz_localize(None).astype("datetime64[us]")

        merged = pd.merge_asof(
            left.sort_values("lookup_date"),
            vix_idx.sort_values("lookup_date"),
            on="lookup_date",
            direction="backward",
        ).sort_values("bar_date").reset_index(drop=True)

        vix_arr = pd.Series(merged["vix_close"].values, index=df.index).astype(float)

        # vrp_market: VIX - RV21 (both in vol-points). RV21 used here is already .shift(1)-safe-equivalent
        # because we further .shift(1) at assignment time below.
        vrp_market_series = vix_arr - rv21
        # vrp_ticker: identical to vrp_market here (no per-ticker IV30 source); kept distinct for downstream.
        vrp_ticker_series = vrp_market_series.copy()

        # 252-day z-score
        roll_mean = vrp_market_series.rolling(252, min_periods=60).mean()
        roll_std = vrp_market_series.rolling(252, min_periods=60).std().replace(0, np.nan)
        z252 = ((vrp_market_series - roll_mean) / roll_std)

        # sign flip vs previous bar
        sign_flip = (np.sign(vrp_market_series) != np.sign(vrp_market_series.shift(1))).astype(np.int8)

        out = df.copy()
        out["vrp_market"] = vrp_market_series.shift(1).fillna(0.0).values
        out["vrp_ticker"] = vrp_ticker_series.shift(1).fillna(0.0).values
        out["vrp_zscore_252"] = z252.shift(1).fillna(0.0).values
        out["vrp_sign_flip"] = sign_flip.shift(1).fillna(0).astype(np.int8).values

        logger.info(
            "[vrp] %s: added 4 cols (non-zero vrp_market rows=%d)",
            ticker, int((out["vrp_market"] != 0).sum()),
        )
        return out
    except Exception as exc:
        logger.warning("[vrp] %s: computation failed (%s) — zeroing", ticker, exc)
        return _zero_fill(df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=400, freq="B")
    rng = np.random.default_rng(0)
    demo = pd.DataFrame({"close": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))}, index=idx)
    out = add_vol_risk_premium_features(demo, "DEMO")
    print(out[VRP_FEATURE_NAMES].tail(5).to_string())
