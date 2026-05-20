"""
cycle060_features.py — Wrapper for cycle060 OPTIONS features (Wave Cycle, 2026-05-17).

Adds 3 daily features beyond what `options_flow_features.py` (v10 Wave A) already
provides. v10's existing options_flow module gives:
    put_call_volume_ratio, iv_vs_rv_divergence, unusual_options_activity_flag.

This wrapper extends the stack with three OI/delta-driven features that the
original cycle060 engine computes from the Alpaca OPRA snapshot stream:
    - put_call_oi_ratio        : total put OI / total call OI (5-day smoothed)
    - volume_to_oi_ratio        : (total call+put volume) / (total OI), 5-day smoothed
    - net_delta_exposure_z21    : 21-day z-score of net dealer delta exposure proxy
                                  (net_delta = sum(call_oi * delta_call) - sum(put_oi * delta_put))

Data source priority (all .shift(1)-safe — bar D uses snapshots strictly < D):
  1. $SP/cache/options_snapshots/<TICKER>.parquet (the same cache options_flow uses).
     Required extra columns:  total_call_oi, total_put_oi, net_delta_exposure.
     If these columns are absent the wrapper zero-fills.
  2. If the snapshot cache has no OI columns, falls back to a graceful zero-fill
     (rather than burning paid API calls on each call).

This file does NOT modify the original cycle060 engine
(`research/active/cycle060_options_features/options_features.py`); it is a v10-side
wrapper that consumes the SAME snapshot parquet cache the options_flow module
populates today. The cycle060 engine remains the authoritative source for
recomputing those snapshots from Alpaca + Quiver.

Idempotent: re-running on an already-augmented df is a no-op.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WORK = Path(
    "/Users/orginal/Library/CloudStorage/"
    "GoogleDrive-zachgladstone@gmail.com/My Drive/"
    "AI-Tools/s&p500-ticker-mastery"
)
SNAPSHOT_DIR = WORK / "cache" / "options_snapshots"

CYCLE060_FEATURE_NAMES: list[str] = [
    "put_call_oi_ratio",
    "volume_to_oi_ratio",
    "net_delta_exposure_z21",
]

_OI_WIN = 5      # rolling smoothing window for OI ratios
_DELTA_WIN = 21  # z-score window for net delta exposure


def _load_snapshots(ticker: str) -> pd.DataFrame:
    """Load same parquet that options_flow_features.py reads; tolerate missing
    OI / delta columns by returning whatever is present.
    """
    path = SNAPSHOT_DIR / f"{ticker}.parquet"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path)
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"]).dt.tz_localize(None)
        df = df.sort_values("snapshot_date").reset_index(drop=True)
        return df
    except Exception as exc:  # noqa: BLE001
        logger.warning("[cycle060] snapshot read %s failed: %s", ticker, exc)
        return pd.DataFrame()


def _zero_fill(df: pd.DataFrame) -> pd.DataFrame:
    for col in CYCLE060_FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0.0
    return df


def add_cycle060_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Append 3 cycle060 OI/delta features to df (idempotent, .shift(1)-safe)."""
    if df is None or len(df) == 0:
        return df

    # idempotent
    if all(c in df.columns for c in CYCLE060_FEATURE_NAMES):
        return df

    snaps = _load_snapshots(ticker)
    has_oi = (
        not snaps.empty
        and "total_call_oi" in snaps.columns
        and "total_put_oi" in snaps.columns
    )
    has_delta = not snaps.empty and "net_delta_exposure" in snaps.columns

    if not has_oi and not has_delta:
        return _zero_fill(df)

    # ---- compute snapshot-side rolling features (daily index) ----
    snaps_d = snaps.copy()
    snaps_d["snapshot_date"] = (
        pd.to_datetime(snaps_d["snapshot_date"]).dt.tz_localize(None).dt.normalize()
    )
    snaps_d = snaps_d.set_index("snapshot_date").sort_index()

    if has_oi:
        call_oi = pd.to_numeric(snaps_d["total_call_oi"], errors="coerce").astype(float)
        put_oi = pd.to_numeric(snaps_d["total_put_oi"], errors="coerce").astype(float)
        pc_oi_raw = (put_oi / call_oi.replace(0, np.nan)).clip(0.0, 10.0).fillna(0.0)
        snaps_d["put_call_oi_ratio"] = pc_oi_raw.rolling(_OI_WIN, min_periods=1).mean()
        if "total_call_vol" in snaps_d.columns and "total_put_vol" in snaps_d.columns:
            tot_vol = (
                snaps_d["total_call_vol"].astype(float)
                + snaps_d["total_put_vol"].astype(float)
            )
            tot_oi = call_oi + put_oi
            vol_oi_raw = (tot_vol / tot_oi.replace(0, np.nan)).clip(0.0, 20.0).fillna(0.0)
            snaps_d["volume_to_oi_ratio"] = vol_oi_raw.rolling(_OI_WIN, min_periods=1).mean()
        else:
            snaps_d["volume_to_oi_ratio"] = 0.0
    else:
        snaps_d["put_call_oi_ratio"] = 0.0
        snaps_d["volume_to_oi_ratio"] = 0.0

    if has_delta:
        nd = pd.to_numeric(snaps_d["net_delta_exposure"], errors="coerce").astype(float)
        mu = nd.rolling(_DELTA_WIN, min_periods=5).mean()
        sd = nd.rolling(_DELTA_WIN, min_periods=5).std()
        snaps_d["net_delta_exposure_z21"] = ((nd - mu) / sd.replace(0, np.nan)).fillna(0.0)
    else:
        snaps_d["net_delta_exposure_z21"] = 0.0

    # ---- merge_asof onto bar dates (strict prior-only) ----
    if isinstance(df.index, pd.DatetimeIndex):
        bar_dates = df.index
    elif "date" in df.columns:
        bar_dates = pd.DatetimeIndex(pd.to_datetime(df["date"]))
    else:
        return _zero_fill(df)
    if bar_dates.tz is not None:
        bar_dates = bar_dates.tz_convert(None)

    bar_df = pd.DataFrame(
        {"bar_date": pd.to_datetime(bar_dates.normalize()).astype("datetime64[ns]")}
    ).reset_index(drop=True)
    bar_df["__pos"] = range(len(bar_df))
    bar_sorted = bar_df.sort_values("bar_date").reset_index(drop=True)

    right = snaps_d.reset_index().rename(columns={"snapshot_date": "bar_date"})
    right["bar_date"] = pd.to_datetime(right["bar_date"]).astype("datetime64[ns]")
    right = right[
        ["bar_date", "put_call_oi_ratio", "volume_to_oi_ratio", "net_delta_exposure_z21"]
    ].sort_values("bar_date").reset_index(drop=True)

    merged = pd.merge_asof(
        bar_sorted,
        right,
        on="bar_date",
        direction="backward",
        allow_exact_matches=False,
    )
    merged = merged.sort_values("__pos").reset_index(drop=True)

    if "put_call_oi_ratio" not in df.columns:
        df["put_call_oi_ratio"] = merged["put_call_oi_ratio"].fillna(0.0).astype(float).values
    if "volume_to_oi_ratio" not in df.columns:
        df["volume_to_oi_ratio"] = merged["volume_to_oi_ratio"].fillna(0.0).astype(float).values
    if "net_delta_exposure_z21" not in df.columns:
        df["net_delta_exposure_z21"] = (
            merged["net_delta_exposure_z21"].fillna(0.0).astype(float).values
        )
    return df


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    tk = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    idx = pd.date_range(end=pd.Timestamp.utcnow().date(), periods=80, freq="B")
    demo = pd.DataFrame({"close": np.linspace(100, 110, len(idx))}, index=idx)
    out = add_cycle060_features(demo, tk)
    print(f"In cols: 1  Out cols: {out.shape[1]}")
    print(out[CYCLE060_FEATURE_NAMES].tail(3).to_string())
