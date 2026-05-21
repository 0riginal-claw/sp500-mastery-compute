"""
sector_aggregate_features.py — sector-pooled cross-sectional features.

Per OC audit #4: pool same features across all tickers in same GICS sector,
expose sector-level signal as input to per-ticker xgBoost.

NO-LOOKAHEAD AUDIT (2026-05-21)
================================
Data sources consumed:
  - state/per-ticker cached parquet at cache/features/<TKR>_v10_full_*.parquet
    (precomputed feature parquets — each row indexed by the SAME timestamp the
    feature describes; feature engineering inside those parquets already does
    its own .shift(1) where necessary).
  - cache/universe_agg_ticker_sectors.parquet — static GICS sector mapping
    (no per-date variation; safe to read in full).

Computation:
  1. Read sector mapping (502 tickers × 12 sectors).
  2. For the current ticker, identify its sector S.
  3. For each of the CORE_FEATURES (20 well-known technical features), load the
     same column from every OTHER ticker in sector S at THE SAME bar t (strict
     same-time-bar pool — no future bars used).
  4. Compute three quantities per (bar, feature):
        - sector_<f>_mean   — mean across sector members (excluding the ticker)
        - sector_<f>_std    — population std across sector members
        - ticker_vs_sector_<f>_z  — (ticker[f] - sector mean) / (sector std + eps)
  5. .shift(1) the entire output frame so feature[t] only uses bars ≤ t-1.

Strict NO LOOKAHEAD guarantees:
  - Only same-time-bar values are pooled — never any forward bar.
  - The final .shift(1) ensures feature for bar t uses only data up to t-1.
  - The self-ticker is excluded from the sector mean/std to avoid trivial leakage.
  - Cached sector parquets are read-only — no in-place mutation of training data.

Output (60 columns = 20 features × 3 stats):
  For each f in CORE_FEATURES (n=20):
    sector_<f>_mean, sector_<f>_std, ticker_vs_sector_<f>_z

Fallback behavior:
  - Missing sector mapping → all-zero output.
  - Missing sector cache parquets → all-zero output with warn.
  - Too few sector members (< 2) → mean=ticker_value, std=0, z=0.

Data source: same v10 feature cache parquets already produced by the pipeline.
License:    internal.
"""
from __future__ import annotations

import logging
import os
import glob
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
_SP_ROOT = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/"
    "My Drive/AI-Tools/s&p500-ticker-mastery"
)
_SECTOR_MAP_PATH = _SP_ROOT / "cache" / "universe_agg_ticker_sectors.parquet"
_FEATURE_CACHE_DIR = _SP_ROOT / "cache" / "features"

# Top-20 high-importance core features (xgBoost importance findings).
# All exist in standard v10 cached parquets.
CORE_FEATURES: list[str] = [
    "ret_1d", "ret_5d", "ret_21d", "ret_63d",
    "rsi_14", "rsi_21",
    "atr_14", "atr_pct",
    "adx_14", "cci_20",
    "bb_pct", "bb_width",
    "vol_ratio",
    "ema_5_gt_ema_20",
    "macd_hist",
    "ema_20", "ema_50", "ema_200",
    "close", "vol_sma_20",
]

SECTOR_FEATURE_NAMES: list[str] = []
for _f in CORE_FEATURES:
    SECTOR_FEATURE_NAMES.extend([
        f"sector_{_f}_mean",
        f"sector_{_f}_std",
        f"ticker_vs_sector_{_f}_z",
    ])
SECTOR_FEATURE_COUNT: int = len(SECTOR_FEATURE_NAMES)  # = 60

_EPS = 1e-9


def _load_sector_mapping() -> Optional[pd.Series]:
    """Return Series index=ticker, value=sector, or None on failure."""
    if not _SECTOR_MAP_PATH.exists():
        logger.warning("[sector_agg] sector mapping not found: %s", _SECTOR_MAP_PATH)
        return None
    try:
        df = pd.read_parquet(_SECTOR_MAP_PATH)
        return df.iloc[:, 0]  # first (and only) col = sector
    except (OSError, ValueError) as exc:
        logger.warning("[sector_agg] sector mapping load failed: %s", exc)
        return None


def _newest_cache_parquet(ticker: str) -> Optional[Path]:
    """Pick the newest v10_full cache parquet for *ticker* (or None)."""
    pattern = str(_FEATURE_CACHE_DIR / f"{ticker}_v10_full_*.parquet")
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return Path(files[0])


def _load_sector_member_frame(ticker: str, cols_wanted: list[str]) -> Optional[pd.DataFrame]:
    """Load a sector member's cached features (only the columns we need)."""
    parq = _newest_cache_parquet(ticker)
    if parq is None:
        return None
    try:
        df = pd.read_parquet(parq, columns=[c for c in cols_wanted if c])
        return df
    except Exception as exc:  # noqa: BLE001  pyarrow may raise non-ValueError
        logger.debug("[sector_agg] could not load %s: %s", parq.name, exc)
        return None


def _zero_output(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with all SECTOR_FEATURE_NAMES set to 0.0."""
    for col in SECTOR_FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0.0
    return df


def compute_sector_aggregate_features(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
) -> pd.DataFrame:
    """Append sector-pooled cross-sectional features (60 cols) to *df*.

    For each of CORE_FEATURES (n=20):
      - sector_<f>_mean         : mean across sector members at same bar
      - sector_<f>_std          : population std across sector members at same bar
      - ticker_vs_sector_<f>_z  : (this ticker's value - sector mean) / (std + eps)

    All outputs .shift(1)-safe (final .shift(1) before assignment).

    Args:
        df:     Per-ticker feature frame with datetime-indexed rows and at least
                the CORE_FEATURES columns present (missing ones are skipped).
        ticker: Stock symbol used to look up GICS sector + exclude self from pool.

    Returns:
        df with up to 60 new columns appended.
    """
    df = df.copy()

    if ticker is None:
        logger.warning("[sector_agg] ticker not provided — zero-filling 60 cols")
        return _zero_output(df)

    sector_map = _load_sector_mapping()
    if sector_map is None or ticker not in sector_map.index:
        logger.warning("[sector_agg] no sector mapping for %s — zero-filling", ticker)
        return _zero_output(df)

    sector = sector_map.loc[ticker]
    peers = sector_map[sector_map == sector].index.tolist()
    peers = [p for p in peers if p != ticker]

    if len(peers) < 2:
        logger.warning(
            "[sector_agg] sector=%s for %s has %d peers — degenerate; zero-filling",
            sector, ticker, len(peers),
        )
        return _zero_output(df)

    # Which CORE_FEATURES are actually present in this ticker's frame?
    available_feats = [f for f in CORE_FEATURES if f in df.columns]
    if not available_feats:
        logger.warning("[sector_agg] none of CORE_FEATURES present in %s frame", ticker)
        return _zero_output(df)

    # ---- Step 1: stack each peer's columns into a 3-D dict ----
    # per_feat[f] -> DataFrame with index=date, columns=peer_ticker
    per_feat: dict[str, pd.DataFrame] = {f: pd.DataFrame(index=df.index) for f in available_feats}
    n_loaded = 0
    for peer in peers:
        peer_df = _load_sector_member_frame(peer, available_feats)
        if peer_df is None or peer_df.empty:
            continue
        n_loaded += 1
        for f in available_feats:
            if f in peer_df.columns:
                per_feat[f][peer] = peer_df[f]

    if n_loaded < 2:
        logger.warning(
            "[sector_agg] only %d peer cache files loaded for sector=%s — zero-filling",
            n_loaded, sector,
        )
        return _zero_output(df)

    # ---- Step 2: per-bar mean / std + ticker z-score ----
    sector_blocks: dict[str, pd.Series] = {}
    for f in available_feats:
        peers_frame = per_feat[f]
        # Numeric coerce + drop fully-empty rows (each row's sector pop = ≥2 peers)
        peers_frame = peers_frame.apply(pd.to_numeric, errors="coerce")
        mean = peers_frame.mean(axis=1, skipna=True)
        std  = peers_frame.std(axis=1, skipna=True, ddof=0)
        # Reindex onto df index in case some peer parquets had wider/narrower index
        mean = mean.reindex(df.index)
        std  = std.reindex(df.index)
        ticker_series = pd.to_numeric(df[f], errors="coerce")
        z = (ticker_series - mean) / (std.replace(0.0, np.nan) + _EPS)

        sector_blocks[f"sector_{f}_mean"] = mean.fillna(0.0)
        sector_blocks[f"sector_{f}_std"]  = std.fillna(0.0)
        sector_blocks[f"ticker_vs_sector_{f}_z"] = z.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # ---- Step 3: assemble + .shift(1) for strict no-lookahead ----
    out = pd.DataFrame(sector_blocks, index=df.index).shift(1).fillna(0.0)

    # Add any missing columns (features that weren't in df) as zeros for stable schema
    for col in SECTOR_FEATURE_NAMES:
        if col not in out.columns:
            out[col] = 0.0
    out = out[SECTOR_FEATURE_NAMES]  # canonical order

    # Append
    for col in SECTOR_FEATURE_NAMES:
        df[col] = out[col].values.astype(float)

    logger.info(
        "[sector_agg] %s sector=%s peers=%d feats_used=%d cols_added=%d",
        ticker, sector, n_loaded, len(available_feats), SECTOR_FEATURE_COUNT,
    )
    return df


# -----------------------------------------------------------------------------
# Smoke entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    tkr = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
    pq = _newest_cache_parquet(tkr)
    if pq is None:
        print(f"no cache for {tkr}")
        sys.exit(1)
    feat = pd.read_parquet(pq)
    n_before = feat.shape[1]
    feat = compute_sector_aggregate_features(feat, ticker=tkr)
    n_after = feat.shape[1]
    print(f"{tkr}: {n_before} -> {n_after} cols (+{n_after - n_before})")
    print("Sample new cols (last 5 bars):")
    print(feat[SECTOR_FEATURE_NAMES[:6]].tail())
