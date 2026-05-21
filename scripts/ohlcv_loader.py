"""
ohlcv_loader.py — Unified OHLCV loader for backtest pipelines.
2026-05-21: Cache B (Alpaca 5yr multi-TF) wired as primary source.

Source priority (load() method):
    1. Cache B GOLD  : version_3 - Gabriel/Gabriel_Alpaca TimeFrames/
                       Partitioned by month: <TF-dir>/<TICKER>/YYYY-MM.parquet
                       Schema: [timestamp(UTC), open, high, low, close, volume, trade_count, vwap]
                       Coverage: 502 tickers x 10 TFs (1Min...1Day), 2021-04 -> 2026-04
    2. Fallback yf 5yr: s&p500-ticker-mastery/cache/yfinance_5yr/<TICKER>.parquet
                       Schema: [date, open, high, low, close, volume, ticker]
                       Coverage: 509 tickers, daily only.
    3. Network yfinance: yfinance.Ticker(...).history(period="5y") (worst case)

Returns DataFrame normalized to:
    columns: ['open', 'high', 'low', 'close', 'volume']
    index  : tz-naive DatetimeIndex named 'timestamp' (matches load_daily() shape
             produced by the legacy 1-min-parquet path in backtest_ml.py)

Timeframe directory map:
    1Day    -> "Day TimeFrames/1Day"
    1Hour   -> "Hour TimeFrames/1Hour"
    4Hour   -> "Hour TimeFrames/4Hour"
    8Hour   -> "Hour TimeFrames/8Hour"
    12Hour  -> "Hour TimeFrames/12Hour"
    1Min    -> "Minutes TimeFrames/1Min"
    5Min    -> "Minutes TimeFrames/5Min"
    15Min   -> "Minutes TimeFrames/15Min"
    30Min   -> "Minutes TimeFrames/30Min"
    45Min   -> "Minutes TimeFrames/45Min"
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

DRIVE_ROOT = Path(
    "/Users/orginal/Library/CloudStorage/"
    "GoogleDrive-zachgladstone@gmail.com/My Drive"
)

# Cache B GOLD root (Alpaca 5yr multi-TF; 306K parquet, 502 tickers, 10 TFs).
DEFAULT_CACHE_B_ROOT = DRIVE_ROOT / "version_3 - Gabriel" / "Gabriel_Alpaca TimeFrames"

# Fallback: yfinance 5yr daily cache (509 tickers).
DEFAULT_YF5Y_ROOT = (
    DRIVE_ROOT / "AI-Tools" / "s&p500-ticker-mastery" / "cache" / "yfinance_5yr"
)

# Per-call tmp cache to avoid re-reading 60+ parquet files within a run.
_RAM_CACHE: dict[tuple[str, str], pd.DataFrame] = {}

TIMEFRAME_DIR_MAP = {
    "1Day": ("Day TimeFrames", "1Day"),
    "1Hour": ("Hour TimeFrames", "1Hour"),
    "4Hour": ("Hour TimeFrames", "4Hour"),
    "8Hour": ("Hour TimeFrames", "8Hour"),
    "12Hour": ("Hour TimeFrames", "12Hour"),
    "1Min": ("Minutes TimeFrames", "1Min"),
    "5Min": ("Minutes TimeFrames", "5Min"),
    "15Min": ("Minutes TimeFrames", "15Min"),
    "30Min": ("Minutes TimeFrames", "30Min"),
    "45Min": ("Minutes TimeFrames", "45Min"),
}


class OhlcvLoader:
    """Multi-source OHLCV loader. Cache B GOLD primary, yf5y fallback, yfinance net last."""

    def __init__(
        self,
        cache_b_root: Optional[Path] = None,
        fallback_yfinance_5yr_root: Optional[Path] = None,
        allow_network_fallback: bool = True,
    ):
        self.cache_b_root = Path(cache_b_root or DEFAULT_CACHE_B_ROOT)
        self.yf5y_root = Path(fallback_yfinance_5yr_root or DEFAULT_YF5Y_ROOT)
        self.allow_network = allow_network_fallback

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load(
        self,
        ticker: str,
        timeframe: str = "1Day",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load OHLCV bars for ticker @ timeframe with optional date slice.

        Args:
            ticker: stock symbol e.g. "AAPL"
            timeframe: one of TIMEFRAME_DIR_MAP keys (default "1Day")
            start, end: ISO dates (YYYY-MM-DD); inclusive on both ends.

        Returns:
            DataFrame[open,high,low,close,volume] with tz-naive DatetimeIndex.
        """
        key = (ticker, timeframe)
        if key in _RAM_CACHE:
            df = _RAM_CACHE[key]
        else:
            df = self._try_cache_b(ticker, timeframe)
            if df is None and timeframe == "1Day":
                df = self._try_yf5y(ticker)
            if df is None and self.allow_network and timeframe == "1Day":
                df = self._try_yf_network(ticker)
            if df is None:
                raise RuntimeError(
                    f"OhlcvLoader: no data found for {ticker} @ {timeframe} "
                    f"(tried Cache B, yf5y, network={self.allow_network})"
                )
            _RAM_CACHE[key] = df

        if start is not None:
            df = df.loc[df.index >= pd.Timestamp(start)]
        if end is not None:
            df = df.loc[df.index <= pd.Timestamp(end)]
        return df

    # ------------------------------------------------------------------
    # Source 1: Cache B GOLD (Alpaca 5yr multi-TF)
    # ------------------------------------------------------------------
    def _try_cache_b(self, ticker: str, timeframe: str) -> Optional[pd.DataFrame]:
        if timeframe not in TIMEFRAME_DIR_MAP:
            logger.warning("ohlcv_loader: unknown timeframe %r (skip Cache B)", timeframe)
            return None
        parent_dir, tf_dir = TIMEFRAME_DIR_MAP[timeframe]
        ticker_dir = self.cache_b_root / parent_dir / tf_dir / ticker
        if not ticker_dir.exists():
            return None
        parquets = sorted(ticker_dir.glob("*.parquet"))
        if not parquets:
            return None
        dfs = []
        for p in parquets:
            try:
                dfs.append(pd.read_parquet(p))
            except Exception as e:
                logger.warning("ohlcv_loader: failed reading %s: %s", p, e)
        if not dfs:
            return None
        df = pd.concat(dfs, ignore_index=True)
        return self._normalize_cache_b(df)

    @staticmethod
    def _normalize_cache_b(df: pd.DataFrame) -> pd.DataFrame:
        """Cache B -> standard shape:
        cols [timestamp(UTC), open, high, low, close, volume, trade_count, vwap]
        ->   index DatetimeIndex(tz-naive, 'timestamp'), cols [open,high,low,close,volume]
        """
        df = df.copy()
        if "timestamp" not in df.columns:
            raise ValueError("Cache B parquet missing 'timestamp' column")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
        # tz-naive to match load_daily() legacy shape
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        # Keep only the 5 canonical OHLCV cols (preserve trade_count/vwap as
        # extras if downstream wants them, but core wired path is 5-col).
        keep = ["open", "high", "low", "close", "volume"]
        for c in keep:
            if c not in df.columns:
                raise ValueError(f"Cache B parquet missing {c!r} column")
        df = df[keep].astype("float64")
        df = df.dropna(subset=["open", "high", "low", "close"])
        # Drop duplicate timestamps (monthly-partition boundaries / edge cases)
        df = df[~df.index.duplicated(keep="first")]
        return df

    # ------------------------------------------------------------------
    # Source 2: yfinance_5yr local parquet fallback (daily only)
    # ------------------------------------------------------------------
    def _try_yf5y(self, ticker: str) -> Optional[pd.DataFrame]:
        path = self.yf5y_root / f"{ticker}.parquet"
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            logger.warning("ohlcv_loader: failed reading yf5y %s: %s", path, e)
            return None
        return self._normalize_yf5y(df)

    @staticmethod
    def _normalize_yf5y(df: pd.DataFrame) -> pd.DataFrame:
        """yf5y -> standard shape:
        cols [date, open, high, low, close, volume, ticker]
        ->   index DatetimeIndex(tz-naive, 'timestamp'), cols [open,high,low,close,volume]
        """
        df = df.copy()
        if "date" not in df.columns:
            raise ValueError("yf5y parquet missing 'date' column")
        df["timestamp"] = pd.to_datetime(df["date"])
        df = df.set_index("timestamp").sort_index()
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)
        keep = ["open", "high", "low", "close", "volume"]
        df = df[keep].astype("float64")
        df = df.dropna(subset=["open", "high", "low", "close"])
        df = df[~df.index.duplicated(keep="first")]
        return df

    # ------------------------------------------------------------------
    # Source 3: yfinance network (last-resort)
    # ------------------------------------------------------------------
    def _try_yf_network(self, ticker: str) -> Optional[pd.DataFrame]:
        try:
            import yfinance as yf
        except ImportError:
            return None
        try:
            raw = yf.Ticker(ticker).history(period="5y", auto_adjust=True)
        except Exception as e:
            logger.warning("ohlcv_loader: yfinance net fetch failed for %s: %s", ticker, e)
            return None
        if raw is None or raw.empty:
            return None
        raw = raw.copy()
        raw.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in raw.columns]
        if getattr(raw.index, "tz", None) is not None:
            raw.index = raw.index.tz_localize(None)
        raw.index.name = "timestamp"
        keep = ["open", "high", "low", "close", "volume"]
        df = raw[keep].astype("float64")
        df = df.dropna(subset=["open", "high", "low", "close"])
        return df


# Module-level default instance for convenience.
_DEFAULT_LOADER = None


def get_default_loader() -> OhlcvLoader:
    global _DEFAULT_LOADER
    if _DEFAULT_LOADER is None:
        _DEFAULT_LOADER = OhlcvLoader()
    return _DEFAULT_LOADER


def load_ohlcv(
    ticker: str,
    timeframe: str = "1Day",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Module-level convenience: get_default_loader().load(...)."""
    return get_default_loader().load(ticker, timeframe, start, end)


# CLI smoke test: python ohlcv_loader.py AAPL 1Day
if __name__ == "__main__":
    sym = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    tf = sys.argv[2] if len(sys.argv) > 2 else "1Day"
    df = load_ohlcv(sym, tf)
    print(f"{sym} {tf}: shape={df.shape}")
    print(f"  range: {df.index.min()} -> {df.index.max()}")
    print(f"  cols : {df.columns.tolist()}")
    print(df.head(2))
    print("...")
    print(df.tail(2))
