"""
cycle058_features.py — Wrapper for cycle058 MARKET CONTEXT features (Wave Cycle, 2026-05-17).

v10's existing `macro_features.py` covers DAILY SPY / sector ETF / VIX features
(spy_return_5d/21d, vix_close, sector_return_5d, etc.). cycle058 adds two
distinct slices on top of that:

  1. PER-DAY SPY-INTRADAY features (computed from 5-min SPY bars, locked at
     session start so they describe the regime entering today's session). The
     cycle058 source `mc_intraday_features.py` exposes a per-(date, bar_idx)
     cache and 5 binary gates. For DAILY feature consumption, this wrapper
     computes ONE row per session-day:
         - mc_spy_intra_cum_ret_eod   : SPY cumulative ret from open at close
         - mc_spy_intra_above_or30h_eod : SPY closed above its OR30 high (binary)
         - mc_spy_intra_below_or30l_eod : SPY closed below its OR30 low (binary)
       All three are shifted by 1 trading day so bar D consumes only D-1 SPY.

  2. TICKER-vs-SECTOR relative strength (5d), which the daily MC engine
     computes but `macro_features.py` does NOT (macro_features has SPY-relative,
     not sector-relative):
         - mc_rs_sector_5d          : ticker_5d_ret − sector_etf_5d_ret
         - mc_rs_sector_5d_sign     : sign of the above (-1 / 0 / +1)

Total: 5 features. All .shift(1)-safe.

Data sources (NO new paid API):
  - SPY 1-min bars: same _load_1min() pattern v10 uses (alpaca cache → claudes
    test fallback → zero-fill if absent).
  - Sector ETF map: best-effort static dict (covers top sectors via the v10
    `macro_features` SECTOR_PREFIXES set). If ticker isn't mapped, falls back
    to SPY for the RS calc (so the feature collapses to spy-relative).
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CYCLE058_FEATURE_NAMES: list[str] = [
    "mc_spy_intra_cum_ret_eod",
    "mc_spy_intra_above_or30h_eod",
    "mc_spy_intra_below_or30l_eod",
    "mc_rs_sector_5d",
    "mc_rs_sector_5d_sign",
]

_ET_TZ = "America/New_York"

_CACHE_ALPACA = (
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools/s&p500-ticker-mastery/cache/alpaca_features"
)
_CACHE_CLAUDES = (
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/claudes test/data/timeframes/S&P500 5 Year Historical Data"
    "/Minutes TimeFrames/1Min_merged"
)
_DAY_CACHE_CLAUDES = (
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/claudes test/data/timeframes/S&P500 5 Year Historical Data"
    "/Day TimeFrames/1Day"
)
_SPY_INTRA_CACHE = (
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/claudes test/research/active/cycle058_market_context/_spy_5min.csv"
)

# Minimal sector map (the dominant slice — extended fallbacks via macro_features)
_SECTOR_ETF = {
    # finance
    "JPM": "XLF", "BAC": "XLF", "WFC": "XLF", "C": "XLF", "GS": "XLF", "MS": "XLF", "SCHW": "XLF",
    # tech
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AVGO": "XLK", "CSCO": "XLK", "ORCL": "XLK",
    "ADBE": "XLK", "CRM": "XLK", "INTC": "XLK", "AMD": "XLK",
    # energy
    "XOM": "XLE", "CVX": "XLE", "COP": "XLE", "SLB": "XLE", "EOG": "XLE",
    # health
    "JNJ": "XLV", "UNH": "XLV", "PFE": "XLV", "MRK": "XLV", "ABBV": "XLV", "LLY": "XLV",
    # industrials
    "GE": "XLI", "CAT": "XLI", "HON": "XLI", "UPS": "XLI", "RTX": "XLI",
    # consumer discretionary
    "AMZN": "XLY", "TSLA": "XLY", "HD": "XLY", "MCD": "XLY", "NKE": "XLY",
    # staples
    "PG": "XLP", "KO": "XLP", "PEP": "XLP", "WMT": "XLP", "COST": "XLP",
    # utilities
    "NEE": "XLU", "DUK": "XLU", "SO": "XLU",
    # comm services
    "GOOG": "XLC", "GOOGL": "XLC", "META": "XLC", "DIS": "XLC", "NFLX": "XLC",
    # materials
    "LIN": "XLB", "FCX": "XLB", "NEM": "XLB",
    # real estate
    "AMT": "XLRE", "PLD": "XLRE", "SPG": "XLRE",
}


def _sector_for(ticker: str) -> str:
    return _SECTOR_ETF.get(ticker.upper(), "SPY")


def _load_1min(ticker: str) -> pd.DataFrame:
    """Load 1-min bars ET-tz-aware; fallback chain alpaca → claudes-test."""
    for root, suffix in (
        (_CACHE_ALPACA, f"{ticker}_1min.parquet"),
        (_CACHE_CLAUDES, f"{ticker}.parquet"),
    ):
        path = os.path.join(root, suffix)
        if not os.path.exists(path):
            continue
        try:
            raw = pd.read_parquet(path)
            if "timestamp" in raw.columns:
                raw = raw.set_index("timestamp")
            raw = raw.sort_index()
            if raw.index.tz is None:
                raw.index = raw.index.tz_localize("UTC")
            raw.index = raw.index.tz_convert(_ET_TZ)
            return raw
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"cycle058_features: load failed {path!r}: {exc}")
            continue
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


def _load_daily(ticker: str) -> pd.DataFrame:
    """Load daily bars. Used for sector ETF 5d ret. Returns DataFrame with
    DatetimeIndex (tz-naive date) and 'close' column. Falls back to zero if
    no parquet exists.
    """
    path = os.path.join(_DAY_CACHE_CLAUDES, f"{ticker}.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        raw = pd.read_parquet(path)
        if "timestamp" in raw.columns:
            raw["date"] = pd.to_datetime(raw["timestamp"]).dt.tz_localize(None).dt.normalize()
            raw = raw.set_index("date")
        elif isinstance(raw.index, pd.DatetimeIndex):
            raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()
        return raw.sort_index()[["close"]].astype(float)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"cycle058_features: daily load failed {path!r}: {exc}")
        return pd.DataFrame()


def _zero_fill(df: pd.DataFrame) -> pd.DataFrame:
    for col in CYCLE058_FEATURE_NAMES:
        if col not in df.columns:
            if col in (
                "mc_spy_intra_above_or30h_eod",
                "mc_spy_intra_below_or30l_eod",
                "mc_rs_sector_5d_sign",
            ):
                df[col] = 0
            else:
                df[col] = 0.0
    return df


def _spy_intra_daily() -> pd.DataFrame:
    """Build per-day SPY-intraday aggregates. Returns DataFrame indexed by date
    with columns: cum_ret_eod, above_or30h, below_or30l. Tries 1-min SPY cache,
    falls back to _spy_5min.csv. Empty DataFrame if neither is available.
    """
    bars = _load_1min("SPY")
    rows: list[dict] = []
    if not bars.empty:
        bars = bars.copy()
        bars["hm"] = bars.index.hour * 60 + bars.index.minute
        bars = bars[(bars["hm"] >= 9 * 60 + 30) & (bars["hm"] < 16 * 60)]
        if bars.empty:
            pass
        else:
            bars["date"] = bars.index.date
            or_end_m = 9 * 60 + 30 + 30  # 30-min OR
            for d, g in bars.groupby("date", sort=True):
                if len(g) < 30:
                    continue
                g = g.sort_index()
                open_px = float(g["open"].iloc[0])
                close_px = float(g["close"].iloc[-1])
                or_mask = g["hm"] < or_end_m
                if not or_mask.any():
                    continue
                or_h = float(g.loc[or_mask, "high"].max())
                or_l = float(g.loc[or_mask, "low"].min())
                rows.append(
                    {
                        "date": pd.Timestamp(d),
                        "cum_ret_eod": (close_px - open_px) / max(open_px, 1e-9),
                        "above_or30h": int(close_px > or_h),
                        "below_or30l": int(close_px < or_l),
                    }
                )
    elif os.path.exists(_SPY_INTRA_CACHE):
        try:
            m5 = pd.read_csv(_SPY_INTRA_CACHE)
            if "date" in m5.columns:
                m5["date"] = pd.to_datetime(m5["date"]).dt.tz_localize(None).dt.normalize()
            else:
                return pd.DataFrame()
            for d, g in m5.sort_values(["date", "bar_idx_in_day"]).groupby("date"):
                if len(g) < 6:
                    continue
                open_px = float(g["open"].iloc[0])
                close_px = float(g["close"].iloc[-1])
                or_h = float(g["high"].iloc[:6].max())
                or_l = float(g["low"].iloc[:6].min())
                rows.append(
                    {
                        "date": pd.Timestamp(d),
                        "cum_ret_eod": (close_px - open_px) / max(open_px, 1e-9),
                        "above_or30h": int(close_px > or_h),
                        "below_or30l": int(close_px < or_l),
                    }
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[cycle058] _spy_5min.csv parse failed: %s", exc)
            return pd.DataFrame()

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).set_index("date").sort_index()
    return out


def add_cycle058_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Append 5 cycle058 market-context features to df. Idempotent, .shift(1)-safe."""
    if df is None or len(df) == 0:
        return df
    if all(c in df.columns for c in CYCLE058_FEATURE_NAMES):
        return df

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

    # ---- SPY-intraday daily aggregate, merge_asof (strictly prior bar) ----
    spy = _spy_intra_daily()
    if not spy.empty:
        right = spy.reset_index().rename(columns={"date": "bar_date"})
        right["bar_date"] = pd.to_datetime(right["bar_date"]).astype("datetime64[ns]")
        right = right[["bar_date", "cum_ret_eod", "above_or30h", "below_or30l"]].sort_values(
            "bar_date"
        ).reset_index(drop=True)
        merged_spy = pd.merge_asof(
            bar_sorted,
            right,
            on="bar_date",
            direction="backward",
            allow_exact_matches=False,
        )
        merged_spy = merged_spy.sort_values("__pos").reset_index(drop=True)
        spy_cum = merged_spy["cum_ret_eod"].fillna(0.0).astype(float).values
        spy_above = merged_spy["above_or30h"].fillna(0).astype(int).values
        spy_below = merged_spy["below_or30l"].fillna(0).astype(int).values
    else:
        spy_cum = np.zeros(len(df), dtype=float)
        spy_above = np.zeros(len(df), dtype=int)
        spy_below = np.zeros(len(df), dtype=int)

    # ---- Relative strength vs sector ETF (daily 5d ret) ----
    sec_etf = _sector_for(ticker)
    sec_daily = _load_daily(sec_etf)
    if not sec_daily.empty and "close" in df.columns:
        sec_close = sec_daily["close"]
        sec_5d = sec_close.pct_change(5)
        # Align onto bar_dates via merge_asof (prior-only)
        sec_df = pd.DataFrame(
            {"bar_date": pd.to_datetime(sec_5d.index).astype("datetime64[ns]"),
             "sec_5d": sec_5d.values}
        ).sort_values("bar_date").reset_index(drop=True)
        merged_sec = pd.merge_asof(
            bar_sorted,
            sec_df,
            on="bar_date",
            direction="backward",
            allow_exact_matches=False,
        )
        merged_sec = merged_sec.sort_values("__pos").reset_index(drop=True)
        sec_ret5_aligned = merged_sec["sec_5d"].fillna(0.0).astype(float).values

        # Ticker's 5d ret on its own close (shift 1 → strictly prior)
        tk_close = pd.to_numeric(df["close"], errors="coerce").astype(float)
        tk_ret5 = tk_close.pct_change(5).shift(1).fillna(0.0).values
        rs5 = tk_ret5 - sec_ret5_aligned
        rs5_sign = np.sign(rs5).astype("int8")
    else:
        rs5 = np.zeros(len(df), dtype=float)
        rs5_sign = np.zeros(len(df), dtype=int)

    if "mc_spy_intra_cum_ret_eod" not in df.columns:
        df["mc_spy_intra_cum_ret_eod"] = spy_cum
    if "mc_spy_intra_above_or30h_eod" not in df.columns:
        df["mc_spy_intra_above_or30h_eod"] = spy_above
    if "mc_spy_intra_below_or30l_eod" not in df.columns:
        df["mc_spy_intra_below_or30l_eod"] = spy_below
    if "mc_rs_sector_5d" not in df.columns:
        df["mc_rs_sector_5d"] = rs5
    if "mc_rs_sector_5d_sign" not in df.columns:
        df["mc_rs_sector_5d_sign"] = rs5_sign.astype(int)
    return df


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    tk = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    idx = pd.date_range(end=pd.Timestamp.utcnow().date(), periods=80, freq="B")
    rng = np.random.default_rng(11)
    close = 100 + np.cumsum(rng.normal(0.1, 1.5, len(idx)))
    demo = pd.DataFrame({"close": close}, index=idx)
    out = add_cycle058_features(demo, tk)
    print(f"In cols: 1  Out cols: {out.shape[1]}")
    print(out[CYCLE058_FEATURE_NAMES].tail(5).to_string())
