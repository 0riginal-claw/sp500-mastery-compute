"""
options_flow_features.py — Options-flow features for v10 (Wave A, 2026-05-17).

Adds 3 daily features derived from yfinance options chain + a simple IV
realized-vol proxy. All features are .shift(1)-safe (the value on bar D is
computed from options snapshots whose snapshot_date < D, i.e. they could only
have been observed BEFORE the open of bar D).

Features added:
  - put_call_volume_ratio          : trailing 5d mean of (puts_volume / calls_volume)
                                     clipped to [0, 5]; 1.0 means perfectly neutral.
  - iv_vs_rv_divergence            : (atm_iv - realized_vol_21d) (annualized fraction).
                                     Positive = options pricing in MORE vol than realized
                                     (often bearish / pre-event), negative = complacent.
  - unusual_options_activity_flag  : 1 when total option volume >= 2x its trailing
                                     20-day mean AND p/c ratio sits in the outer
                                     deciles ({<0.5 OR >2.0}), else 0.

Data sources (zero paid APIs, license-friendly = Drive owner):
  1. Cached snapshots at $SP/cache/options_snapshots/<TICKER>.parquet (if present).
     Each snapshot row: snapshot_date, total_call_vol, total_put_vol, atm_iv.
  2. Fallback live yfinance.Ticker(symbol).option_chain() for the FIRST 3 expiries
     when cache is missing — only triggered once per process per ticker; result
     is point-in-time-stamped TODAY (i.e. it informs only bars >= TODAY+1).
  3. Realized vol uses df['log_ret_1d'] if present, else log diff of df['close'].

Graceful failure:
  - Missing cache + yfinance unavailable     -> all 3 cols zero-filled.
  - Cache exists but ticker absent           -> all 3 cols zero-filled.
  - Some snapshot rows missing IV            -> IV-dependent feat falls back to 0.0
                                                 for those bars only (other 2 still computed).

Idempotent: re-calling on already-augmented df is a no-op.

Author: 2026-05-17 (Wave A — Drive-harvest feature wiring).
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

OPTIONS_FLOW_FEATURE_NAMES: list[str] = [
    "put_call_volume_ratio",
    "iv_vs_rv_divergence",
    "unusual_options_activity_flag",
]

# Rolling window for P/C smoothing
_PC_WIN = 5
# Window for "unusual activity" baseline
_UNUSUAL_WIN = 20
# IV - RV uses 21d realized vol as comparison
_RV_WIN = 21
# Hard cap on raw P/C ratio (avoid div-by-zero blowups)
_PC_CAP = 5.0


# ---------------------------------------------------------------------------
# Snapshot loader (cache-first; yfinance fallback)
# ---------------------------------------------------------------------------


def _load_snapshots(ticker: str) -> pd.DataFrame:
    """Return DataFrame with columns: snapshot_date, total_call_vol, total_put_vol,
    atm_iv. Empty DataFrame if no data is available.

    Snapshot cache schema (parquet):
      snapshot_date  datetime64[ns]  (tz-naive)
      total_call_vol float64
      total_put_vol  float64
      atm_iv         float64   (annualized; may be NaN)
    """
    path = SNAPSHOT_DIR / f"{ticker}.parquet"
    if path.exists():
        try:
            df = pd.read_parquet(path)
            # normalize types
            df["snapshot_date"] = pd.to_datetime(df["snapshot_date"]).dt.tz_localize(None)
            for c in ("total_call_vol", "total_put_vol", "atm_iv"):
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["snapshot_date"]).copy()
            df = df.sort_values("snapshot_date").reset_index(drop=True)
            logger.debug("[options_flow] cache HIT %s: %d rows", ticker, len(df))
            return df
        except Exception as e:
            logger.warning("[options_flow] cache read %s failed: %s", ticker, e)

    # ---- yfinance fallback (one-shot snapshot dated TODAY) ----
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        expiries = tk.options[:3] if tk.options else []
        if not expiries:
            return pd.DataFrame(columns=["snapshot_date", "total_call_vol", "total_put_vol", "atm_iv"])
        call_vol = 0.0
        put_vol = 0.0
        ivs: list[float] = []
        spot = None
        try:
            hist = tk.history(period="2d")
            if len(hist) > 0:
                spot = float(hist["Close"].iloc[-1])
        except Exception:
            pass
        for exp in expiries:
            try:
                ch = tk.option_chain(exp)
                calls = ch.calls
                puts = ch.puts
                call_vol += float(calls["volume"].fillna(0).sum())
                put_vol += float(puts["volume"].fillna(0).sum())
                if spot is not None:
                    # ATM = closest strike to spot, average call+put IV
                    for side in (calls, puts):
                        side["dist"] = (side["strike"] - spot).abs()
                        side_atm = side.nsmallest(1, "dist")
                        if not side_atm.empty:
                            iv = float(side_atm["impliedVolatility"].iloc[0])
                            if 0 < iv < 5:
                                ivs.append(iv)
            except Exception:
                continue
        atm_iv = float(np.mean(ivs)) if ivs else np.nan
        today = pd.Timestamp.utcnow().tz_localize(None).normalize()
        out = pd.DataFrame(
            [{"snapshot_date": today, "total_call_vol": call_vol, "total_put_vol": put_vol, "atm_iv": atm_iv}]
        )
        # best-effort persist for next call
        try:
            SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
            out.to_parquet(path, index=False)
        except Exception as e:
            logger.debug("[options_flow] cache write %s skipped: %s", ticker, e)
        logger.info("[options_flow] yfinance snapshot %s: calls=%.0f puts=%.0f iv=%.3f", ticker, call_vol, put_vol, atm_iv)
        return out
    except Exception as e:
        logger.debug("[options_flow] yfinance fallback failed %s: %s", ticker, e)
        return pd.DataFrame(columns=["snapshot_date", "total_call_vol", "total_put_vol", "atm_iv"])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _zero_fill(df: pd.DataFrame) -> pd.DataFrame:
    for col in OPTIONS_FLOW_FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0.0 if col != "unusual_options_activity_flag" else 0
    return df


def add_options_flow_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Append 3 .shift(1)-safe options-flow features to df (idempotent).

    Args:
        df: DataFrame with DatetimeIndex OR a 'date' column. Should also have
            'close' for realized-vol fallback (or 'log_ret_1d').
        ticker: Symbol used to look up options snapshots.

    Returns:
        df with 3 new columns appended.
    """
    if df is None or len(df) == 0:
        return df

    # idempotent
    present = [c for c in OPTIONS_FLOW_FEATURE_NAMES if c in df.columns]
    if len(present) == len(OPTIONS_FLOW_FEATURE_NAMES):
        return df

    snaps = _load_snapshots(ticker)
    if snaps.empty:
        return _zero_fill(df)

    # Get bar dates as tz-naive DatetimeIndex
    if isinstance(df.index, pd.DatetimeIndex):
        bar_dates = df.index
    elif "date" in df.columns:
        bar_dates = pd.DatetimeIndex(pd.to_datetime(df["date"]))
    else:
        return _zero_fill(df)
    if bar_dates.tz is not None:
        bar_dates = bar_dates.tz_convert(None)

    # Build snapshot daily series — fill missing days forward inside the snapshot
    snaps_d = snaps.copy()
    snaps_d["snapshot_date"] = pd.to_datetime(snaps_d["snapshot_date"]).dt.tz_localize(None).dt.normalize()
    snaps_d = snaps_d.set_index("snapshot_date").sort_index()
    snaps_d["pc_raw"] = (
        snaps_d["total_put_vol"].astype(float)
        / snaps_d["total_call_vol"].replace(0, np.nan).astype(float)
    )
    snaps_d["pc_raw"] = snaps_d["pc_raw"].fillna(0.0).clip(0.0, _PC_CAP)
    snaps_d["total_vol"] = snaps_d["total_call_vol"].astype(float) + snaps_d["total_put_vol"].astype(float)

    # Trailing 5d mean of P/C (only past snapshots count; min_periods=1)
    snaps_d["pc_smooth"] = snaps_d["pc_raw"].rolling(_PC_WIN, min_periods=1).mean()
    # Trailing 20d mean of total volume (baseline for 'unusual' flag)
    snaps_d["total_vol_mean20"] = snaps_d["total_vol"].rolling(_UNUSUAL_WIN, min_periods=3).mean()
    snaps_d["unusual_flag"] = (
        (snaps_d["total_vol"] >= 2.0 * snaps_d["total_vol_mean20"])
        & ((snaps_d["pc_raw"] < 0.5) | (snaps_d["pc_raw"] > 2.0))
    ).astype(int)

    # ---- Realized vol on bar frame ----
    if "log_ret_1d" in df.columns:
        rets = df["log_ret_1d"].astype(float)
    elif "close" in df.columns:
        rets = np.log(df["close"].astype(float)).diff()
    else:
        rets = pd.Series(0.0, index=df.index)
    # 21d annualized realized vol (252 bar yr)
    rv = rets.rolling(_RV_WIN, min_periods=5).std() * np.sqrt(252.0)
    rv = rv.fillna(0.0)

    # Build a daily DataFrame keyed by bar_dates with .shift(1)-safe lookup of
    # snapshot values. We use merge_asof direction='backward' allow_exact_matches=False
    # so each bar at date D sees only snapshots with snapshot_date < D.
    # Force both sides to datetime64[ns] to avoid us/ns dtype-mismatch crashes
    # when yfinance returns Timestamp.utcnow() (microsecond precision).
    bar_df = pd.DataFrame(
        {"bar_date": pd.to_datetime(bar_dates.normalize()).astype("datetime64[ns]")}
    ).reset_index(drop=True)
    bar_df["__pos"] = range(len(bar_df))
    bar_sorted = bar_df.sort_values("bar_date").reset_index(drop=True)

    right = snaps_d.reset_index().rename(columns={"snapshot_date": "bar_date"})
    right["bar_date"] = pd.to_datetime(right["bar_date"]).astype("datetime64[ns]")
    right = right[["bar_date", "pc_smooth", "atm_iv", "unusual_flag"]].sort_values("bar_date").reset_index(drop=True)

    merged = pd.merge_asof(
        bar_sorted,
        right,
        on="bar_date",
        direction="backward",
        allow_exact_matches=False,
    )
    merged = merged.sort_values("__pos").reset_index(drop=True)

    pc_smooth = merged["pc_smooth"].fillna(0.0).clip(0.0, _PC_CAP).astype(float).values
    atm_iv = merged["atm_iv"].astype(float).values
    unusual = merged["unusual_flag"].fillna(0).astype(int).values

    iv_minus_rv = np.where(np.isnan(atm_iv), 0.0, atm_iv - rv.values)

    if "put_call_volume_ratio" not in df.columns:
        df["put_call_volume_ratio"] = pc_smooth
    if "iv_vs_rv_divergence" not in df.columns:
        df["iv_vs_rv_divergence"] = iv_minus_rv
    if "unusual_options_activity_flag" not in df.columns:
        df["unusual_options_activity_flag"] = unusual

    return df


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    tk = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    idx = pd.date_range(end=pd.Timestamp.utcnow().date(), periods=100, freq="B")
    demo = pd.DataFrame({"close": np.linspace(100, 110, len(idx))}, index=idx)
    out = add_options_flow_features(demo, tk)
    print(f"Input cols: 1 Output cols: {out.shape[1]}")
    print(out[OPTIONS_FLOW_FEATURE_NAMES].tail(3).to_string())
