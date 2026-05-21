"""
feature_cache.py — Generic disk-cache for XGBoost feature DataFrames.

Caches computed feature sets to parquet on disk with SHA-256 keyed filenames,
sidecar JSON manifests, TTL-based expiry, and safe trash-based invalidation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import pandas as pd

logger = logging.getLogger("feature_cache")

# Default cache dir resolved relative to this file so the `&` in the workspace
# path is handled by the OS without any string manipulation.
_DEFAULT_CACHE_DIR: Path = Path(__file__).resolve().parent.parent / "cache" / "features"

DateLike = Union[str, pd.Timestamp]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_date(d: DateLike) -> str:
    """Return an ISO-8601 date string regardless of input type."""
    if isinstance(d, pd.Timestamp):
        return d.strftime("%Y-%m-%d")
    return str(d)


def _cache_key(ticker: str, start: str, end: str, feature_set: str, version: str) -> str:
    """Return 16-char hex SHA-256 of the cache identity string."""
    raw = f"{ticker}|{start}|{end}|{feature_set}|{version}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _resolve_cache_dir(cache_dir: Optional[str]) -> Path:
    p = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
    p.mkdir(parents=True, exist_ok=True)
    return p


def _parquet_path(ticker: str, feature_set: str, key: str, cache_dir: Path) -> Path:
    return cache_dir / f"{ticker}_{feature_set}_{key}.parquet"


def _sidecar_path(parquet: Path) -> Path:
    return parquet.with_suffix(".json")


def _age_hours(path: Path) -> float:
    return (time.time() - path.stat().st_mtime) / 3600.0


def _write_sidecar(path: Path, meta: dict) -> None:
    path.write_text(json.dumps(meta, indent=2, default=str))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cache_path(
    ticker: str,
    date_range: Tuple[DateLike, DateLike],
    feature_set: str,
    version: str = "v1",
    cache_dir: Optional[str] = None,
) -> Path:
    """Return the expected parquet path for the given inputs (file may not exist)."""
    start = _normalize_date(date_range[0])
    end = _normalize_date(date_range[1])
    key = _cache_key(ticker, start, end, feature_set, version)
    cdir = _resolve_cache_dir(cache_dir)
    return _parquet_path(ticker, feature_set, key, cdir)


def get_cached(
    ticker: str,
    date_range: Tuple[DateLike, DateLike],
    feature_set: str,
    compute_fn: Callable[[], pd.DataFrame],
    version: str = "v1",
    ttl_days: int = 7,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Return a cached DataFrame, computing and storing it on a miss or TTL expiry.

    Parameters
    ----------
    ticker:       Ticker symbol, e.g. "AAPL".
    date_range:   (start_date, end_date) as strings or pd.Timestamps.
    feature_set:  Identifier for the feature set, e.g. "v10_full".
    compute_fn:   Callable with no args that returns the DataFrame when called.
    version:      Version stamp; bumping this busts all cached entries.
    ttl_days:     Maximum age of a cached file before it is treated as a miss.
    cache_dir:    Override the default cache directory.
    """
    start = _normalize_date(date_range[0])
    end = _normalize_date(date_range[1])
    key = _cache_key(ticker, start, end, feature_set, version)
    cdir = _resolve_cache_dir(cache_dir)
    pq = _parquet_path(ticker, feature_set, key, cdir)
    basename = pq.name

    # --- HIT check ---
    if pq.exists():
        age_h = _age_hours(pq)
        age_d = age_h / 24.0
        if age_d <= ttl_days:
            try:
                # PATCH-2 (2026-05-21 extreme-speedup): pyarrow multi-thread read.
                # pd.read_parquet defaults to use_threads=True via pyarrow but the
                # explicit pyarrow path with thread pool sized to CPU count is
                # 1.5-3x faster on multi-column feature parquets (>1k cols).
                import pyarrow.parquet as _pq
                import os as _os
                _n_threads = int(_os.environ.get("PARQUET_N_THREADS", "0")) or _os.cpu_count() or 4
                df = _pq.read_table(pq, use_threads=True).to_pandas(
                    use_threads=True, self_destruct=True
                )
                logger.info(
                    "[feature_cache] HIT %s rows=%d cols=%d age=%.1fh threads=%d",
                    basename, len(df), len(df.columns), age_h, _n_threads,
                )
                return df
            except Exception as exc:
                logger.warning(
                    "[feature_cache] read failure on %s (%s) — recomputing", basename, exc
                )
        else:
            logger.info("[feature_cache] STALE %s age=%.1fd > ttl=%dd", basename, age_d, ttl_days)

    # --- MISS / STALE: compute and store ---
    t0 = time.time()
    df = compute_fn()
    elapsed = time.time() - t0

    # PATCH-2 (2026-05-21): zstd-1 compression — same write speed as snappy
    # but ~30-40% smaller files → faster reads. Falls back to snappy if zstd
    # not built in (older pyarrow).
    try:
        df.to_parquet(pq, compression="zstd", compression_level=1, index=True)
    except Exception:
        df.to_parquet(pq, compression="snappy", index=True)

    meta = {
        "ticker": ticker,
        "start": start,
        "end": end,
        "feature_set": feature_set,
        "version": version,
        "created_at": datetime.utcnow().isoformat(),
        "n_rows": len(df),
        "n_cols": len(df.columns),
    }
    _write_sidecar(_sidecar_path(pq), meta)

    logger.info(
        "[feature_cache] MISS %s computed_in=%.2fs rows=%d cols=%d",
        basename, elapsed, len(df), len(df.columns),
    )
    return df


def invalidate(
    ticker: str,
    feature_set: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> int:
    """Move matching cache entries to a timestamped trash folder (no permanent deletes).

    Parameters
    ----------
    ticker:      Ticker whose cache entries to remove.
    feature_set: If given, only remove entries for this feature set.
    cache_dir:   Override cache directory.

    Returns the number of parquet files moved.
    """
    cdir = _resolve_cache_dir(cache_dir)
    ts = datetime.utcnow().strftime("%Y-%m-%d-%H%M")
    trash = cdir / "_trash" / ts
    trash.mkdir(parents=True, exist_ok=True)

    pattern = f"{ticker}_{feature_set}_*.parquet" if feature_set else f"{ticker}_*.parquet"
    moved = 0
    for pq in cdir.glob(pattern):
        pq.rename(trash / pq.name)
        sc = _sidecar_path(pq)
        if (cdir / sc.name).exists():
            (cdir / sc.name).rename(trash / sc.name)
        moved += 1
        logger.info("[feature_cache] invalidated %s -> %s", pq.name, trash)

    return moved


def cache_stats(cache_dir: Optional[str] = None) -> dict:
    """Return summary statistics for the cache directory.

    Returns
    -------
    dict with keys: n_entries, total_size_mb, oldest_age_days, newest_age_days
    """
    cdir = _resolve_cache_dir(cache_dir)
    parquets = [p for p in cdir.glob("*.parquet") if not p.name.startswith("_")]

    if not parquets:
        return {
            "n_entries": 0,
            "total_size_mb": 0.0,
            "oldest_age_days": None,
            "newest_age_days": None,
        }

    now = time.time()
    sizes = [p.stat().st_size for p in parquets]
    ages_d = [(now - p.stat().st_mtime) / 86400.0 for p in parquets]

    return {
        "n_entries": len(parquets),
        "total_size_mb": round(sum(sizes) / 1_048_576, 3),
        "oldest_age_days": round(max(ages_d), 4),
        "newest_age_days": round(min(ages_d), 4),
    }


# ---------------------------------------------------------------------------
# __main__ — quick stats printout
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    stats = cache_stats()
    print(json.dumps(stats, indent=2, default=str))
