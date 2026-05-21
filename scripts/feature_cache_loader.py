"""
feature_cache_loader.py — Multi-source precomputed-feature lookup.

Wires three feature cache layers behind a single API so sweeps + backtests can
skip recomputation when a cache hit exists.

Cache priority (best -> last-resort):
  1. v10 full cache:   AI-Tools/s&p500-ticker-mastery/cache/features/{TICKER}_v10_full_{hash}.parquet
                       (1492 cols, daily bars, current pipeline output)
  2. version_3 cache:  My Drive/version_3 - Gabriel/research_cycle_001/precomputed/{TICKER}.parquet
                       (20 cols, minute bars, indicators + filings/gov flags)
  3. None              -> caller must recompute via build_features.

Cache metadata for v10:
  Each .parquet has a sibling .json containing
    {ticker, start, end, feature_set, version, created_at, n_rows, n_cols}
  Files are hash-named per feature-config; we pick newest by created_at.

Usage:
    from feature_cache_loader import FeatureCache
    fc = FeatureCache()
    df, meta = fc.get_features("AAPL", version="v10_full")
    if df is None:
        # Recompute via build_features()
        ...

The loader is read-only. It never writes to the cache (that responsibility lives
in backtest_xgb_v10.py post-build).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

WORK = Path(
    "/Users/orginal/Library/CloudStorage/"
    "GoogleDrive-zachgladstone@gmail.com/My Drive/"
    "AI-Tools/s&p500-ticker-mastery"
)
V10_CACHE_DIR = WORK / "cache" / "features"

DRIVE_ROOT = Path(
    "/Users/orginal/Library/CloudStorage/"
    "GoogleDrive-zachgladstone@gmail.com/My Drive"
)
V3_PRECOMPUTED_DIR = DRIVE_ROOT / "version_3 - Gabriel" / "research_cycle_001" / "precomputed"


@dataclass
class CacheHit:
    df: pd.DataFrame
    source: str           # "v10_full" | "v3_precomputed"
    path: str
    n_rows: int
    n_cols: int
    feature_set: str
    created_at: Optional[str] = None


class FeatureCache:
    """Multi-source feature cache. Read-only."""

    def __init__(
        self,
        v10_dir: Path = V10_CACHE_DIR,
        v3_dir: Path = V3_PRECOMPUTED_DIR,
    ):
        self.v10_dir = Path(v10_dir)
        self.v3_dir = Path(v3_dir)

    # -- v10 cache --------------------------------------------------------

    def _latest_v10_match(self, ticker: str, version: str) -> Optional[Path]:
        """Return newest cache parquet for ticker+version (by JSON created_at)."""
        if not self.v10_dir.exists():
            return None
        pattern = f"{ticker}_{version}_*.parquet"
        candidates = sorted(self.v10_dir.glob(pattern))
        if not candidates:
            return None
        # Sort by sibling-json created_at if present, else by file mtime.
        def keyer(p: Path):
            sib = p.with_suffix(".json")
            if sib.exists():
                try:
                    meta = json.loads(sib.read_text())
                    return meta.get("created_at", "") or p.stat().st_mtime
                except Exception:
                    pass
            return p.stat().st_mtime
        candidates.sort(key=keyer, reverse=True)
        return candidates[0]

    def _load_v10(self, ticker: str, version: str) -> Optional[CacheHit]:
        p = self._latest_v10_match(ticker, version)
        if p is None:
            return None
        try:
            df = pd.read_parquet(p)
        except Exception as exc:
            logger.warning("v10 cache read failed %s: %s", p, exc)
            return None
        meta_path = p.with_suffix(".json")
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                pass
        return CacheHit(
            df=df,
            source="v10_full",
            path=str(p),
            n_rows=len(df),
            n_cols=len(df.columns),
            feature_set=meta.get("feature_set", version),
            created_at=meta.get("created_at"),
        )

    # -- v3 precomputed ---------------------------------------------------

    def _load_v3(self, ticker: str) -> Optional[CacheHit]:
        p = self.v3_dir / f"{ticker}.parquet"
        if not p.exists():
            return None
        try:
            df = pd.read_parquet(p)
        except Exception as exc:
            logger.warning("v3 precomputed read failed %s: %s", p, exc)
            return None
        # v3 stores `timestamp` as a column, not index; promote it for downstream parity
        if "timestamp" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index("timestamp")
        return CacheHit(
            df=df,
            source="v3_precomputed",
            path=str(p),
            n_rows=len(df),
            n_cols=len(df.columns),
            feature_set="v3_minute_indicators",
            created_at=None,
        )

    # -- public API -------------------------------------------------------

    def get_features(
        self,
        ticker: str,
        version: str = "v10_full",
        allow_v3_fallback: bool = True,
    ) -> Tuple[Optional[pd.DataFrame], Optional[CacheHit]]:
        """
        Return (df, hit) for the freshest cache. (None, None) if no cache.

        Parameters
        ----------
        ticker : e.g. "AAPL"
        version : v10 cache feature_set string. Default "v10_full".
        allow_v3_fallback : if v10 misses, try v3 precomputed (different schema!).

        Notes
        -----
        v3 and v10 schemas are NOT interchangeable (20 cols minute-bars vs
        1492 cols daily-bars). Callers that need v10-schema features should
        check `hit.source == "v10_full"` before using.
        """
        hit = self._load_v10(ticker, version)
        if hit is not None:
            return hit.df, hit
        if allow_v3_fallback:
            hit = self._load_v3(ticker)
            if hit is not None:
                return hit.df, hit
        return None, None

    def has_v10_cache(self, ticker: str, version: str = "v10_full") -> bool:
        return self._latest_v10_match(ticker, version) is not None

    def has_v3_precomputed(self, ticker: str) -> bool:
        return (self.v3_dir / f"{ticker}.parquet").exists()

    def coverage_report(self, tickers: list[str], version: str = "v10_full") -> dict:
        """Cheap coverage scan for a ticker list. No DataFrame loads."""
        v10_hit = sum(1 for t in tickers if self.has_v10_cache(t, version))
        v3_hit = sum(1 for t in tickers if self.has_v3_precomputed(t))
        any_hit = sum(
            1 for t in tickers
            if self.has_v10_cache(t, version) or self.has_v3_precomputed(t)
        )
        return {
            "n_tickers": len(tickers),
            "v10_hits": v10_hit,
            "v3_hits": v3_hit,
            "any_hits": any_hit,
            "miss": len(tickers) - any_hit,
        }


# -- smoke helper -------------------------------------------------------

def _smoke():
    """Quick correctness check. Run: python feature_cache_loader.py"""
    import sys
    fc = FeatureCache()
    for t in ("AAPL", "NVDA", "ZZZNOTREAL"):
        df, hit = fc.get_features(t)
        if df is None:
            print(f"  {t}: MISS")
            continue
        print(f"  {t}: src={hit.source} shape={df.shape} feature_set={hit.feature_set}")
    rpt = fc.coverage_report(["AAPL", "NVDA", "MSFT", "GOOGL", "ZZZNOTREAL"])
    print("coverage:", rpt)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_smoke())
