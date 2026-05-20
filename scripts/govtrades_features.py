"""
govtrades_features.py — Congress / lobbying features for v10 (Wave A, 2026-05-17).

Reads SQLite at:
  /My Drive/Ph0tis/Gov-Trades/data/govtrades.db

Tables used:
  - congress_trades(ticker, transaction_date, report_date, transaction_type, amount_min, ...)
  - lobbying(ticker, date, amount, registrant, ...)

All features are .shift(1)-safe: bar at date D only sees rows whose
`transaction_date < D` (for trades) or `date < D` (for lobbying).

Features added:
  - congress_trade_density_5d       : count of congress trades for ticker in trailing 5d.
  - congress_buy_sell_ratio_5d      : (n_buys - n_sells) / max(n_total, 1) in trailing 5d,
                                      so range is in [-1, +1] (1 = all buys, -1 = all sells).
  - lobbying_filing_count_30d       : count of distinct lobbying records for ticker
                                      in trailing 30 calendar days.

To avoid SQLite WAL contention on the Drive-backed DB, we copy it once per
process to /tmp/govtrades_wave_a.db. Subsequent reads hit the local copy.

Idempotent: re-calling on already-augmented df is a no-op.
Graceful failure: missing DB / missing tables / empty rows → all 3 cols 0.
"""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GOVTRADES_DB_DRIVE = (
    "/Users/orginal/Library/CloudStorage/"
    "GoogleDrive-zachgladstone@gmail.com/My Drive/"
    "Ph0tis/Gov-Trades/data/govtrades.db"
)
GOVTRADES_DB_LOCAL = "/tmp/govtrades_wave_a.db"

GOVTRADES_FEATURE_NAMES: list[str] = [
    "congress_trade_density_5d",
    "congress_buy_sell_ratio_5d",
    "lobbying_filing_count_30d",
]

# Module-level cache: ticker -> (trade_dates: np.ndarray ordinal, trade_signs: np.ndarray +1/-1/0)
_ct_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
# ticker -> lobbying date ordinals array
_lob_cache: dict[str, np.ndarray] = {}


def _ensure_local_db() -> bool:
    """Copy govtrades.db to /tmp once per process using SQLite's online backup
    API. Returns True if local copy exists and is queryable.

    Falls back to shutil.copy2 + WAL/SHM sidecar copy if backup() fails.
    A raw shutil.copy2 of a WAL-mode DB without its sidecars can produce a
    "database disk image is malformed" error on read, so we attempt the safer
    path first.
    """
    if os.path.exists(GOVTRADES_DB_LOCAL):
        # Verify it's queryable; if not, force re-copy.
        try:
            with sqlite3.connect(GOVTRADES_DB_LOCAL, timeout=10.0) as con:
                con.execute("SELECT COUNT(*) FROM congress_trades").fetchone()
            return True
        except Exception:
            try:
                os.remove(GOVTRADES_DB_LOCAL)
            except OSError:
                pass

    if not os.path.exists(GOVTRADES_DB_DRIVE):
        logger.warning("[govtrades] source DB missing: %s", GOVTRADES_DB_DRIVE)
        return False

    # --- Try SQLite online backup API (resilient to WAL) ---
    try:
        src = sqlite3.connect(f"file:{GOVTRADES_DB_DRIVE}?mode=ro", uri=True, timeout=30.0)
        dst = sqlite3.connect(GOVTRADES_DB_LOCAL)
        with dst:
            src.backup(dst)
        src.close()
        dst.close()
        # Sanity probe
        with sqlite3.connect(GOVTRADES_DB_LOCAL, timeout=10.0) as con:
            con.execute("SELECT COUNT(*) FROM congress_trades").fetchone()
        logger.info("[govtrades] sqlite-backup -> %s OK", GOVTRADES_DB_LOCAL)
        return True
    except Exception as e:
        logger.warning("[govtrades] sqlite-backup failed (%s) — trying shutil+wal", e)

    # --- Fallback: copy DB + WAL + SHM sidecars (best effort) ---
    try:
        shutil.copy2(GOVTRADES_DB_DRIVE, GOVTRADES_DB_LOCAL)
        for sfx in ("-wal", "-shm"):
            src_side = GOVTRADES_DB_DRIVE + sfx
            if os.path.exists(src_side):
                try:
                    shutil.copy2(src_side, GOVTRADES_DB_LOCAL + sfx)
                except Exception as ee:
                    logger.debug("[govtrades] sidecar %s copy skipped: %s", sfx, ee)
        with sqlite3.connect(GOVTRADES_DB_LOCAL, timeout=10.0) as con:
            con.execute("SELECT COUNT(*) FROM congress_trades").fetchone()
        logger.info("[govtrades] shutil-copy+wal -> %s OK", GOVTRADES_DB_LOCAL)
        return True
    except Exception as e:
        logger.warning("[govtrades] all copy strategies failed: %s", e)
        # Wipe broken /tmp file
        try:
            os.remove(GOVTRADES_DB_LOCAL)
        except OSError:
            pass
        return False


def _classify_tx(tx_type: str) -> int:
    """Map QuiverQuant transaction_type to +1 (buy) / -1 (sell) / 0 (other)."""
    if not isinstance(tx_type, str):
        return 0
    t = tx_type.lower()
    if "purchase" in t or "buy" in t or "acquire" in t:
        return 1
    if "sale" in t or "sell" in t or "dispos" in t:
        return -1
    return 0


def _load_congress(ticker: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (sorted_date_ordinals, signs) for the ticker's congress trades.
    transaction_date used (point-in-time legislator activity date).
    """
    if ticker in _ct_cache:
        return _ct_cache[ticker]
    if not _ensure_local_db():
        _ct_cache[ticker] = (np.array([], dtype=np.int64), np.array([], dtype=np.int8))
        return _ct_cache[ticker]
    try:
        with sqlite3.connect(GOVTRADES_DB_LOCAL, timeout=30.0) as con:
            df = pd.read_sql_query(
                "SELECT transaction_date, transaction_type FROM congress_trades WHERE ticker = ?",
                con,
                params=(ticker,),
            )
    except Exception as e:
        logger.warning("[govtrades] congress query failed %s: %s", ticker, e)
        _ct_cache[ticker] = (np.array([], dtype=np.int64), np.array([], dtype=np.int8))
        return _ct_cache[ticker]
    if df.empty:
        _ct_cache[ticker] = (np.array([], dtype=np.int64), np.array([], dtype=np.int8))
        return _ct_cache[ticker]
    df["date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df["ord"] = df["date"].dt.normalize().map(lambda d: d.toordinal()).astype(np.int64)
    df["sign"] = df["transaction_type"].apply(_classify_tx).astype(np.int8)
    df = df.sort_values("ord").reset_index(drop=True)
    _ct_cache[ticker] = (df["ord"].values, df["sign"].values)
    logger.info("[govtrades] congress %s: %d rows", ticker, len(df))
    return _ct_cache[ticker]


def _load_lobbying(ticker: str) -> np.ndarray:
    """Return sorted ordinals of lobbying record dates for the ticker."""
    if ticker in _lob_cache:
        return _lob_cache[ticker]
    if not _ensure_local_db():
        _lob_cache[ticker] = np.array([], dtype=np.int64)
        return _lob_cache[ticker]
    try:
        with sqlite3.connect(GOVTRADES_DB_LOCAL, timeout=30.0) as con:
            df = pd.read_sql_query(
                "SELECT date FROM lobbying WHERE ticker = ?", con, params=(ticker,)
            )
    except Exception as e:
        logger.warning("[govtrades] lobbying query failed %s: %s", ticker, e)
        _lob_cache[ticker] = np.array([], dtype=np.int64)
        return _lob_cache[ticker]
    if df.empty:
        _lob_cache[ticker] = np.array([], dtype=np.int64)
        return _lob_cache[ticker]
    df["d"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["d"]).copy()
    ords = np.sort(df["d"].dt.normalize().map(lambda d: d.toordinal()).astype(np.int64).values)
    _lob_cache[ticker] = ords
    logger.info("[govtrades] lobbying %s: %d rows", ticker, len(ords))
    return ords


def _zero_fill(df: pd.DataFrame) -> pd.DataFrame:
    for col in GOVTRADES_FEATURE_NAMES:
        if col not in df.columns:
            if col == "congress_buy_sell_ratio_5d":
                df[col] = 0.0
            elif col.endswith("_5d") or col.endswith("_30d"):
                df[col] = 0
            else:
                df[col] = 0.0
    return df


def add_govtrades_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Append 3 .shift(1)-safe Congress/lobbying features to df. Idempotent.

    Args:
        df: DataFrame with DatetimeIndex or a 'date' column.
        ticker: Symbol used to query SQLite tables.
    """
    if df is None or len(df) == 0:
        return df
    if all(c in df.columns for c in GOVTRADES_FEATURE_NAMES):
        return df

    if isinstance(df.index, pd.DatetimeIndex):
        bar_dates = df.index
    elif "date" in df.columns:
        bar_dates = pd.DatetimeIndex(pd.to_datetime(df["date"]))
    else:
        return _zero_fill(df)
    if bar_dates.tz is not None:
        bar_dates = bar_dates.tz_convert(None)

    bar_ords = np.array(
        [d.toordinal() for d in bar_dates.normalize().to_pydatetime()], dtype=np.int64
    )

    ct_ords, ct_signs = _load_congress(ticker)
    lob_ords = _load_lobbying(ticker)

    # ---- congress_trade_density_5d (strict <) ----
    # For each bar_ord d, count events with ord in (d-5, d) — strict-less-than.
    density = np.zeros(len(bar_ords), dtype=np.int64)
    bs_ratio = np.zeros(len(bar_ords), dtype=np.float64)
    if ct_ords.size > 0:
        for i, d in enumerate(bar_ords):
            lo = d - 5
            # strict less than d  ⇒  ord <= d-1
            hi = d - 1
            i_lo = np.searchsorted(ct_ords, lo, side="right")  # ord > lo
            i_hi = np.searchsorted(ct_ords, hi, side="right")  # ord > hi+? -> use right then -1
            # count in (lo, d)  ==  (lo, hi]  ==  ord in {lo+1, ..., hi}
            j_lo = np.searchsorted(ct_ords, lo, side="right")
            j_hi = np.searchsorted(ct_ords, hi, side="right")
            n = j_hi - j_lo
            density[i] = int(n)
            if n > 0:
                signs = ct_signs[j_lo:j_hi]
                # buys = +1, sells = -1, other = 0
                bs_ratio[i] = float(signs.sum()) / float(n)

    # ---- lobbying_filing_count_30d (strict <) ----
    lob_count = np.zeros(len(bar_ords), dtype=np.int64)
    if lob_ords.size > 0:
        for i, d in enumerate(bar_ords):
            lo = d - 30
            hi = d - 1
            j_lo = np.searchsorted(lob_ords, lo, side="right")
            j_hi = np.searchsorted(lob_ords, hi, side="right")
            lob_count[i] = int(j_hi - j_lo)

    if "congress_trade_density_5d" not in df.columns:
        df["congress_trade_density_5d"] = density
    if "congress_buy_sell_ratio_5d" not in df.columns:
        df["congress_buy_sell_ratio_5d"] = bs_ratio
    if "lobbying_filing_count_30d" not in df.columns:
        df["lobbying_filing_count_30d"] = lob_count

    return df


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    tk = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    idx = pd.date_range(end=pd.Timestamp.utcnow().date(), periods=200, freq="B")
    demo = pd.DataFrame({"close": np.linspace(100, 110, len(idx))}, index=idx)
    out = add_govtrades_features(demo, tk)
    print(f"Input cols: 1 Output cols: {out.shape[1]}")
    print(out[GOVTRADES_FEATURE_NAMES].tail(5).to_string())
    print(
        "stats:",
        out["congress_trade_density_5d"].sum(),
        out["congress_buy_sell_ratio_5d"].abs().mean(),
        out["lobbying_filing_count_30d"].sum(),
    )
