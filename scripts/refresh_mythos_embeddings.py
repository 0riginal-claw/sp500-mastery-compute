"""
refresh_mythos_embeddings.py — Live-loop daily refresh of 256-dim OpenMythos
embeddings for the tickers that produced signals on a given session.

Invoked by live_paper_trade_ingest.py after save_incremental_bars(). Writes a
parquet per session date to:

    paper_trade/mythos_embeddings/<YYYY-MM-DD>.parquet

Schema (one row per ticker):
    ticker             : str
    date               : str (YYYY-MM-DD)
    emb_0 .. emb_255   : float32  (256 columns)
    checkpoint_path    : str       (env MYTHOS_CHECKPOINT_PATH or 'fallback_zero')
    fallback_flag      : bool      (True if compute_mythos_embedding returned zeros)

The script is idempotent — re-running for the same --date overwrites the parquet.
If mythos_features is unimportable (missing module / torch), the run logs a
warning and exits 0 so the upstream ingest is never blocked.

Usage:
    python scripts/refresh_mythos_embeddings.py \
        --date 2026-05-18 \
        --tickers AAPL,MSFT,NVDA
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORK = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools/s&p500-ticker-mastery"
)
SCRIPTS_DIR = WORK / "scripts"
PAPER_DIR = WORK / "paper_trade"
EMB_DIR = PAPER_DIR / "mythos_embeddings"
EMB_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCRIPTS_DIR))

EMBEDDING_DIM = 256

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] mythos_refresh — %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("mythos_refresh")


# ---------------------------------------------------------------------------
# Mythos import (graceful)
# ---------------------------------------------------------------------------
def _try_import_mythos():
    """Try to import compute_mythos_embedding; return None on failure."""
    try:
        from mythos_features import compute_mythos_embedding  # type: ignore
        return compute_mythos_embedding
    except Exception as e:
        log.warning(f"mythos_features unavailable ({e}) — using zero-vec fallback")
        return None


# ---------------------------------------------------------------------------
# Pandas / pyarrow import (graceful so smoke can complete even if missing)
# ---------------------------------------------------------------------------
def _try_import_pandas():
    try:
        import pandas as pd  # noqa: F401
        return pd
    except Exception as e:
        log.error(f"pandas unavailable: {e} — cannot write parquet")
        return None


def _checkpoint_label() -> str:
    """Return checkpoint path label for the schema."""
    return os.environ.get("MYTHOS_CHECKPOINT_PATH") or "fallback_zero"


# ---------------------------------------------------------------------------
# Per-ticker compute
# ---------------------------------------------------------------------------
def _compute_one(compute_fn, ticker: str, date: str) -> tuple[np.ndarray, bool]:
    """
    Compute embedding for one ticker. Returns (vec, fallback_flag).
    Wrapped so a single ticker failure doesn't kill the loop.
    """
    if compute_fn is None:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32), True

    try:
        emb = compute_fn(ticker, end_date=date)
        if emb is None:
            log.warning(f"{ticker}: compute_mythos_embedding returned None — fallback")
            return np.zeros(EMBEDDING_DIM, dtype=np.float32), True

        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        if emb.shape != (EMBEDDING_DIM,):
            log.warning(
                f"{ticker}: unexpected shape {emb.shape} (expected ({EMBEDDING_DIM},)) — fallback"
            )
            return np.zeros(EMBEDDING_DIM, dtype=np.float32), True

        # Heuristic: an all-zero embedding means the function bailed out
        # (missing parquet, missing checkpoint, etc.). Mark as fallback.
        is_fallback = bool(np.all(emb == 0))
        return emb, is_fallback

    except Exception as e:
        log.warning(f"{ticker}: compute failed ({e}) — fallback")
        return np.zeros(EMBEDDING_DIM, dtype=np.float32), True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refresh OpenMythos 256-dim embeddings for live tickers."
    )
    parser.add_argument("--date", required=True, help="Session date YYYY-MM-DD")
    parser.add_argument(
        "--tickers",
        required=True,
        help="Comma-separated tickers, e.g. AAPL,MSFT,NVDA",
    )
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        log.warning("No tickers passed — exiting 0")
        return 0

    log.info(f"Refreshing mythos embeddings for {len(tickers)} tickers @ {args.date}")

    pd = _try_import_pandas()
    if pd is None:
        return 0  # Non-fatal: upstream ingest must still complete.

    compute_fn = _try_import_mythos()
    checkpoint = _checkpoint_label()

    rows = []
    success = 0
    fallbacks = 0

    for i, ticker in enumerate(tickers, 1):
        log.info(f"[{i}/{len(tickers)}] {ticker} ...")
        emb, is_fallback = _compute_one(compute_fn, ticker, args.date)
        if is_fallback:
            fallbacks += 1
        else:
            success += 1

        row: dict = {
            "ticker": ticker,
            "date": args.date,
            "checkpoint_path": checkpoint,
            "fallback_flag": is_fallback,
        }
        for j in range(EMBEDDING_DIM):
            row[f"emb_{j}"] = float(emb[j])
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = EMB_DIR / f"{args.date}.parquet"

    # Idempotent overwrite
    try:
        df.to_parquet(out_path, index=False)
        log.info(
            f"Wrote {out_path} ({len(rows)} rows, "
            f"{success} real, {fallbacks} fallback, checkpoint={checkpoint})"
        )
    except Exception as e:
        log.error(f"Parquet write failed: {e}")
        # Fallback to CSV so data isn't lost
        csv_path = out_path.with_suffix(".csv")
        try:
            df.to_csv(csv_path, index=False)
            log.warning(f"Wrote CSV fallback: {csv_path}")
        except Exception as e2:
            log.error(f"CSV fallback also failed: {e2}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
