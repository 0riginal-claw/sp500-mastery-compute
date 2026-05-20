"""
auction_features.py — Wave M-1 #12: Opening + Closing auction features.

Approach
--------
Opening auction is approximated by the first 09:30 ET 1-min bar (regular
session open print); closing auction is the last 15:59 ET 1-min bar (closing
print).  Auction returns + auction volume share capture institutional flow
that doesn't appear in continuous trading.  Closing-auction directional
agreement with the session is documented (Hu-Pan-Wang 2018).

Features added (6)
------------------
  open_auction_ret              — opening bar return (open->close inside the bar)
  close_auction_ret             — closing bar return (open->close inside the bar)
  open_auction_vol_share        — open bar volume / total session volume
  close_auction_vol_share       — close bar volume / total session volume
  auction_imbalance_ratio       — close_auction_vol / max(open_auction_vol, eps)
  close_auction_dir_vs_session  — 1 if close_auction_ret has same sign as session ret

All outputs .shift(1)-safe — features for row t use only day t-1 intraday.

License : MIT (own impl). Refs: Hu/Pan/Wang 2018; standard market microstructure.
"""

from __future__ import annotations

import logging
import os
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

AUCTION_FEATURE_NAMES: list[str] = [
    "open_auction_ret",
    "close_auction_ret",
    "open_auction_vol_share",
    "close_auction_vol_share",
    "auction_imbalance_ratio",
    "close_auction_dir_vs_session",
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


def _load_1min(ticker: str) -> pd.DataFrame:
    """Shared 1-min loader; mirrors vpin_features._load_1min."""
    for root, suffix in (
        (_CACHE_ALPACA, f"{ticker}_1min.parquet"),
        (_CACHE_CLAUDES, f"{ticker}.parquet"),
    ):
        path = os.path.join(root, suffix)
        if os.path.exists(path):
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
                warnings.warn(f"auction: failed to load {path!r}: {exc}")
                continue
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


def _auction_one_day(day_bars: pd.DataFrame) -> tuple:
    """Return 6-tuple of auction features for a single trading day."""
    nan_row = (np.nan,) * 6
    if len(day_bars) < 30:
        return nan_row

    # Identify opening bar (09:30) and closing bar (15:59).
    try:
        idx = day_bars.index
        open_mask = (idx.hour == 9) & (idx.minute == 30)
        close_mask = (idx.hour == 15) & (idx.minute == 59)
        open_row = day_bars.loc[open_mask]
        close_row = day_bars.loc[close_mask]
    except Exception:  # noqa: BLE001
        return nan_row
    if open_row.empty:
        open_row = day_bars.iloc[[0]]
    if close_row.empty:
        close_row = day_bars.iloc[[-1]]

    open_bar = open_row.iloc[0]
    close_bar = close_row.iloc[0]
    open_o, open_c = float(open_bar["open"]), float(open_bar["close"])
    close_o, close_c = float(close_bar["open"]), float(close_bar["close"])
    open_v = float(max(open_bar["volume"], 0.0))
    close_v = float(max(close_bar["volume"], 0.0))

    if open_o <= 0 or close_o <= 0:
        return nan_row

    open_auction_ret = (open_c - open_o) / open_o
    close_auction_ret = (close_c - close_o) / close_o
    total_vol = float(day_bars["volume"].astype(float).clip(lower=0).sum())
    if total_vol <= 0:
        return nan_row
    open_vol_share = open_v / total_vol
    close_vol_share = close_v / total_vol
    imb_ratio = close_v / max(open_v, 1e-9)

    session_o = float(day_bars["open"].iloc[0])
    session_c = float(day_bars["close"].iloc[-1])
    session_ret = (session_c - session_o) / session_o if session_o > 0 else 0.0
    same_sign = 1.0 if (
        np.sign(close_auction_ret) == np.sign(session_ret) and session_ret != 0.0
    ) else 0.0

    return (
        float(open_auction_ret),
        float(close_auction_ret),
        float(open_vol_share),
        float(close_vol_share),
        float(imb_ratio),
        float(same_sign),
    )


def add_auction_features(
    df_daily: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    """Append 6 auction features to df_daily. Zero-fills if cache missing."""
    df = df_daily.copy()
    for c in AUCTION_FEATURE_NAMES:
        if c not in df.columns:
            df[c] = 0.0

    bars = _load_1min(ticker)
    if bars.empty or "close" not in bars.columns or "volume" not in bars.columns:
        logger.warning(
            "[auction] no 1-min bars for %s — zero-filling", ticker
        )
        return df

    try:
        rth = bars.between_time("09:30", "15:59")
    except Exception:  # noqa: BLE001
        rth = bars
    if rth.empty:
        return df

    rows: list[tuple] = []
    by_day = rth.groupby(rth.index.normalize().date)
    for d, day_bars in by_day:
        out = _auction_one_day(day_bars)
        rows.append((pd.Timestamp(d),) + out)
    if not rows:
        return df

    cols = ["date"] + AUCTION_FEATURE_NAMES
    adf = pd.DataFrame(rows, columns=cols).set_index("date").sort_index()
    feats = adf[AUCTION_FEATURE_NAMES].shift(1).fillna(0.0)

    if isinstance(df.index, pd.DatetimeIndex):
        join_idx = pd.DatetimeIndex(df.index.normalize())
    elif "date" in df.columns:
        join_idx = pd.DatetimeIndex(pd.to_datetime(df["date"]).dt.normalize())
    else:
        logger.warning(
            "[auction] cannot align to df index (no date col) — zeroing"
        )
        return df
    for col in AUCTION_FEATURE_NAMES:
        df[col] = feats[col].reindex(join_idx).values
        df[col] = df[col].fillna(0.0).astype(float)
    logger.info(
        "[auction] %s: open_auction_ret_mean=%.6f, close_auction_ret_mean=%.6f over %d days",
        ticker,
        float(feats["open_auction_ret"].mean()),
        float(feats["close_auction_ret"].mean()),
        len(feats),
    )
    return df
