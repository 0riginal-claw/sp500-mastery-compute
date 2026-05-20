"""
vol_of_vix_features.py — Realized vol of VIX (systemic-instability proxy).

Wave V-1 (LOW-cost, no new deps). Wired 2026-05-17.

# NO-LOOKAHEAD AUDIT
# ------------------
# Reuses cached VIX parquet from vix_term_structure_v2 module.
# VIX close = end-of-day settlement of prior session, so we merge_asof
# BACKWARD against bar_date - 1 day for strict past-only alignment.
# Realized-vol windows are computed on the merged-but-shifted VIX series, and
# we additionally .shift(1) before assignment.
#
# Features:
#   vix_realized_vol_21    — 21-bar stdev of log(VIX), annualized
#   vix_vol_zscore_252     — 252-day z-score of vix_realized_vol_21
#   vix_vol_spike_indicator — int8, 1 when z > 2
#
# Pure pandas/numpy.
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

VVIX_FEATURE_NAMES: list[str] = [
    "vix_realized_vol_21",
    "vix_vol_zscore_252",
    "vix_vol_spike_indicator",
]

_VIX_CACHE_PATH = WORK / "cache" / "vix_term_structure" / "vix_daily.parquet"


def _zero_fill(df: pd.DataFrame) -> pd.DataFrame:
    for col in VVIX_FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0.0 if col != "vix_vol_spike_indicator" else np.int8(0)
    return df


def _load_vix() -> pd.DataFrame | None:
    if _VIX_CACHE_PATH.exists():
        try:
            cached = pd.read_parquet(_VIX_CACHE_PATH)
            if "vix_close" in cached.columns and len(cached) > 10:
                return cached[["vix_close"]].copy()
        except Exception as exc:
            logger.debug("[vol_of_vix] cache read error: %s", exc)
    try:
        import yfinance as yf
        end = pd.Timestamp.utcnow().normalize()
        start = end - pd.DateOffset(years=7)
        raw = yf.download("^VIX", start=start.date().isoformat(),
                          end=end.date().isoformat(),
                          auto_adjust=True, progress=False)
        if raw is None or raw.empty:
            return None
        col = raw["Close"] if "Close" in raw.columns else raw.iloc[:, 0]
        return pd.DataFrame({"vix_close": col.values},
                            index=pd.DatetimeIndex(raw.index).tz_localize(None))
    except Exception as exc:
        logger.debug("[vol_of_vix] yfinance fallback failed: %s", exc)
        return None


def add_vol_of_vix_features(
    df: pd.DataFrame,
    ticker: str | None = None,
) -> pd.DataFrame:
    """Append 3 vol-of-VIX features. Idempotent + graceful."""
    if df is None or len(df) == 0:
        return df

    if all(c in df.columns for c in VVIX_FEATURE_NAMES):
        return df

    vix_data = _load_vix()
    if vix_data is None or vix_data.empty:
        logger.warning("[vol_of_vix] %s: VIX unavailable — zeroing", ticker)
        return _zero_fill(df)

    try:
        # Compute vol-of-VIX on the VIX-native daily series first
        vix_close = vix_data["vix_close"].astype(float).sort_index()
        vix_logret = np.log(vix_close / vix_close.shift(1))
        rv21 = vix_logret.rolling(21, min_periods=10).std() * np.sqrt(252)

        roll_mean = rv21.rolling(252, min_periods=60).mean()
        roll_std = rv21.rolling(252, min_periods=60).std().replace(0, np.nan)
        z252 = ((rv21 - roll_mean) / roll_std).fillna(0.0)
        spike = (z252 > 2.0).astype(np.int8)

        vix_features = pd.DataFrame({
            "vix_realized_vol_21": rv21.values,
            "vix_vol_zscore_252": z252.values,
            "vix_vol_spike_indicator": spike.values,
        }, index=vix_close.index)

        # NO-LOOKAHEAD merge_asof BACKWARD against bar_date - 1
        if isinstance(df.index, pd.DatetimeIndex):
            bar_dates = df.index.tz_localize(None) if df.index.tz is not None else df.index
        elif "date" in df.columns:
            bar_dates = pd.DatetimeIndex(pd.to_datetime(df["date"])).tz_localize(None)
        else:
            logger.warning("[vol_of_vix] %s: no DatetimeIndex — zeroing", ticker)
            return _zero_fill(df)

        shifted = (bar_dates - pd.Timedelta(days=1)).astype("datetime64[us]")
        left = pd.DataFrame({"bar_date": bar_dates, "lookup_date": shifted})
        right = vix_features.reset_index().rename(columns={vix_features.index.name or "index": "lookup_date"})
        if "lookup_date" not in right.columns:
            right.columns = ["lookup_date"] + list(right.columns[1:])
        right["lookup_date"] = pd.to_datetime(right["lookup_date"]).dt.tz_localize(None).astype("datetime64[us]")

        merged = pd.merge_asof(
            left.sort_values("lookup_date"),
            right.sort_values("lookup_date"),
            on="lookup_date",
            direction="backward",
        ).sort_values("bar_date").reset_index(drop=True)

        out = df.copy()
        # Reset to df.index then .shift(1) for double safety
        rv_s = pd.Series(merged["vix_realized_vol_21"].values, index=df.index).shift(1).fillna(0.0)
        z_s = pd.Series(merged["vix_vol_zscore_252"].values, index=df.index).shift(1).fillna(0.0)
        sp_s = pd.Series(merged["vix_vol_spike_indicator"].values, index=df.index).shift(1).fillna(0)

        out["vix_realized_vol_21"] = rv_s.values
        out["vix_vol_zscore_252"] = z_s.values
        out["vix_vol_spike_indicator"] = sp_s.astype(np.int8).values

        logger.info(
            "[vol_of_vix] %s: added 3 cols (spike rows=%d)",
            ticker, int(out["vix_vol_spike_indicator"].sum()),
        )
        return out
    except Exception as exc:
        logger.warning("[vol_of_vix] %s: computation failed (%s) — zeroing", ticker, exc)
        return _zero_fill(df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=300, freq="B")
    rng = np.random.default_rng(0)
    demo = pd.DataFrame({"close": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))}, index=idx)
    out = add_vol_of_vix_features(demo, "DEMO")
    print(out[VVIX_FEATURE_NAMES].tail(5).to_string())
