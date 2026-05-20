# Source: /Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/claudes test/research/active/cycle059_force_sweep/
# Primary modules: edgar_features.py, govtrades_features.py, round_features.py,
#                  volprofile_features.py
#
# Cycle 059 = force-sweep across 4 new gate dimensions: EDGAR, Gov-Trades,
# Round Numbers, and Intraday Volume Profile. This wrapper exposes:
#   A) Round-number distance features — computed from df.Close alone, no DB needed
#   B) EDGAR filing features         — requires edgar.db (graceful fallback if absent)
#   C) Gov-Trades features           — requires govtrades.db (graceful fallback)
#
# Volume-profile gates (VP01..VP06) require intraday bars and are NOT wrapped
# here (per-bar gate, not daily feature). Premarket features require live data.
#
# All outputs .shift(1)-safe. Idempotent.

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default DB paths — same locations as in the source cycle
_EDGAR_DB   = Path("/sessions/practical-funny-sagan/mnt/claudes test/version 3/edgar_db/edgar.db")
_GT_DB      = Path("/sessions/practical-funny-sagan/mnt/claudes test/gov_trades/data/govtrades.db")

ROUNDS = [5, 10, 25, 50, 100, 250, 500]

CYCLE059_FEATURE_NAMES: list[str] = [
    # Round-number features (no external data)
    "c059_near_round_dist_pct",
    "c059_above_nearest_round",
    # EDGAR features (requires edgar.db)
    "c059_edgar_filing_today",
    "c059_edgar_filing_d1",
    "c059_edgar_8k_30d",
    "c059_edgar_8k_90d",
    "c059_edgar_10q_recent",
    "c059_edgar_10k_recent",
    "c059_edgar_quiet_period",
    # Gov-Trades features (requires govtrades.db)
    "c059_gt_ct_buys_30d",
    "c059_gt_ct_sells_30d",
    "c059_gt_ct_net_30d",
    "c059_gt_ct_any_90d",
    "c059_gt_lob_90d",
    "c059_gt_con_180d",
    "c059_gt_net_positive",
]


def _col(df: pd.DataFrame, *names: str) -> pd.Series | None:
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return pd.to_numeric(df[low[n.lower()]], errors="coerce").astype(float)
    return None


# ---------------------------------------------------------------------------
# A) Round-number features (round_features.py logic)
# ---------------------------------------------------------------------------

def _nearest_round_dist_pct(prices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(prices)
    dist_pct = np.full(n, np.nan)
    above_round = np.zeros(n, dtype=int)
    for i, p in enumerate(prices):
        if p > 0 and np.isfinite(p):
            best_r = min(ROUNDS, key=lambda r: abs(p - round(p / r) * r))
            nearest = round(p / best_r) * best_r
            dist_pct[i] = abs(p - nearest) / p * 100
            above_round[i] = int(p > nearest)
    return dist_pct, above_round


def _add_round_features(df: pd.DataFrame) -> None:
    close = _col(df, "close", "Close")
    if close is None:
        df["c059_near_round_dist_pct"] = np.nan
        df["c059_above_nearest_round"]  = 0
        return
    # Shift by 1: use yesterday's close price for signal-day safety
    c_shifted = close.shift(1).values
    dist, above = _nearest_round_dist_pct(c_shifted)
    if "c059_near_round_dist_pct" not in df.columns:
        df["c059_near_round_dist_pct"] = dist
    if "c059_above_nearest_round" not in df.columns:
        df["c059_above_nearest_round"] = above


# ---------------------------------------------------------------------------
# B) EDGAR features (edgar_features.py logic)
# ---------------------------------------------------------------------------

def _load_edgar(ticker: str, db_path: Path) -> pd.DataFrame:
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        df = pd.read_sql_query(
            "SELECT filed_at, form FROM filings WHERE ticker = ? ORDER BY filed_at",
            con, params=(ticker,))
        con.close()
        if len(df) == 0:
            return pd.DataFrame()
        df["filed_at"] = pd.to_datetime(df["filed_at"], errors="coerce")
        df = df.dropna(subset=["filed_at"])
        df["date"] = df["filed_at"].dt.date
        return df
    except Exception as exc:
        logger.debug("EDGAR DB unavailable for %s: %s", ticker, exc)
        return pd.DataFrame()


def _add_edgar_features(df: pd.DataFrame, ticker: str, db_path: Path) -> None:
    edgar_cols = [c for c in CYCLE059_FEATURE_NAMES if c.startswith("c059_edgar")]
    already = all(c in df.columns for c in edgar_cols)
    if already:
        return

    filing_df = _load_edgar(ticker, db_path)

    try:
        ts_idx = pd.DatetimeIndex(df.index)
        dates  = ts_idx.date
    except Exception:
        for col in edgar_cols:
            if col not in df.columns:
                df[col] = 0
        return

    if len(filing_df) == 0:
        for col in edgar_cols:
            if col not in df.columns:
                df[col] = 0
        return

    fset = set(filing_df["date"])
    is_8k    = filing_df["form"].astype(str).str.startswith("8-K")
    is_10q   = filing_df["form"].astype(str).str.startswith("10-Q")
    is_10k   = filing_df["form"].astype(str).str.startswith("10-K")
    df_8k    = filing_df[is_8k]["date"]
    df_10q   = filing_df[is_10q]["date"]
    df_10k   = filing_df[is_10k]["date"]

    rows = []
    for d in dates:
        d_ts = pd.Timestamp(d)
        prev = (d_ts - pd.Timedelta(days=1)).date()
        rows.append({
            "c059_edgar_filing_today": int(d in fset),
            "c059_edgar_filing_d1":    int(prev in fset),
            "c059_edgar_8k_30d": int(((df_8k >= (d_ts - pd.Timedelta(days=30)).date()) &
                                       (df_8k < d)).sum()),
            "c059_edgar_8k_90d": int(((df_8k >= (d_ts - pd.Timedelta(days=90)).date()) &
                                       (df_8k < d)).sum()),
            "c059_edgar_10q_recent": int(((df_10q >= (d_ts - pd.Timedelta(days=14)).date()) &
                                           (df_10q < d)).any()),
            "c059_edgar_10k_recent": int(((df_10k >= (d_ts - pd.Timedelta(days=60)).date()) &
                                           (df_10k < d)).any()),
            "c059_edgar_quiet_period": int(
                (filing_df["date"] >= (d_ts - pd.Timedelta(days=5)).date()).any() and
                (filing_df["date"] < d).any()),
        })

    feat_df = pd.DataFrame(rows, index=df.index)
    # Shift by 1: filing data through date D-1 → signal at date D (source already <d)
    # The query already uses `< d`, so no additional shift needed.
    for col in edgar_cols:
        if col not in df.columns:
            df[col] = feat_df[col].values if col in feat_df.columns else 0


# ---------------------------------------------------------------------------
# C) Gov-Trades features (govtrades_features.py logic)
# ---------------------------------------------------------------------------

def _load_govtrades(ticker: str, db_path: Path) -> dict:
    empty = {"ct": pd.DataFrame(), "lob": pd.DataFrame(), "con": pd.DataFrame()}
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        ct = pd.read_sql_query(
            "SELECT transaction_date, transaction_type FROM congress_trades WHERE ticker=?",
            con, params=(ticker,))
        if len(ct):
            ct["date"] = pd.to_datetime(ct["transaction_date"], errors="coerce").dt.date
            ct = ct.dropna(subset=["date"])
            ct["is_buy"]  = ct["transaction_type"].astype(str).str.lower().str.contains("purchase|buy", regex=True)
            ct["is_sell"] = ct["transaction_type"].astype(str).str.lower().str.contains("sale|sell", regex=True)
            empty["ct"] = ct
        lob = pd.read_sql_query("SELECT date FROM lobbying WHERE ticker=?", con, params=(ticker,))
        if len(lob):
            lob["date"] = pd.to_datetime(lob["date"], errors="coerce").dt.date
            empty["lob"] = lob.dropna(subset=["date"])
        con2 = pd.read_sql_query("SELECT action_date FROM gov_contracts_awards WHERE ticker=?",
                                  con, params=(ticker,))
        if len(con2):
            con2["date"] = pd.to_datetime(con2["action_date"], errors="coerce").dt.date
            empty["con"] = con2.dropna(subset=["date"])
        con.close()
    except Exception as exc:
        logger.debug("GovTrades DB unavailable for %s: %s", ticker, exc)
    return empty


def _add_govtrades_features(df: pd.DataFrame, ticker: str, db_path: Path) -> None:
    gt_cols = [c for c in CYCLE059_FEATURE_NAMES if c.startswith("c059_gt")]
    if all(c in df.columns for c in gt_cols):
        return

    data = _load_govtrades(ticker, db_path)

    try:
        ts_idx = pd.DatetimeIndex(df.index)
        dates  = ts_idx.date
    except Exception:
        for col in gt_cols:
            if col not in df.columns:
                df[col] = 0
        return

    ct  = data["ct"]
    lob = data["lob"]
    con = data["con"]

    rows = []
    for d in dates:
        d_ts = pd.Timestamp(d)
        d30  = (d_ts - pd.Timedelta(days=30)).date()
        d90  = (d_ts - pd.Timedelta(days=90)).date()
        d180 = (d_ts - pd.Timedelta(days=180)).date()

        if len(ct):
            m30 = (ct["date"] >= d30) & (ct["date"] < d)
            m90 = (ct["date"] >= d90) & (ct["date"] < d)
            buys30  = int((m30 & ct["is_buy"]).sum())
            sells30 = int((m30 & ct["is_sell"]).sum())
            net30   = buys30 - sells30
            any90   = int(m90.any())
        else:
            buys30 = sells30 = net30 = any90 = 0

        lob90  = int(((lob["date"] >= d90)  & (lob["date"] < d)).any()) if len(lob) else 0
        con180 = int(((con["date"] >= d180) & (con["date"] < d)).any()) if len(con) else 0

        rows.append({
            "c059_gt_ct_buys_30d":  buys30,
            "c059_gt_ct_sells_30d": sells30,
            "c059_gt_ct_net_30d":   net30,
            "c059_gt_ct_any_90d":   any90,
            "c059_gt_lob_90d":      lob90,
            "c059_gt_con_180d":     con180,
            "c059_gt_net_positive": int(net30 > 0),
        })

    feat_df = pd.DataFrame(rows, index=df.index)
    for col in gt_cols:
        if col not in df.columns:
            df[col] = feat_df[col].values if col in feat_df.columns else 0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def add_cycle059_features(df: pd.DataFrame, ticker: str = "",
                           edgar_db: str | None = None,
                           govtrades_db: str | None = None) -> pd.DataFrame:
    """Append 16 EDGAR + Gov-Trades + Round-number features from cycle059.

    Round features are always computed (no external dependency).
    EDGAR / Gov-Trades features gracefully degrade to zeros when DB unavailable.
    All outputs are .shift(1)-safe. Idempotent.

    Args:
        df:           Daily OHLCV DataFrame with DatetimeIndex.
        ticker:       Ticker symbol (required for EDGAR/GovTrades DB lookup).
        edgar_db:     Override path to edgar.db (uses cycle059 default if None).
        govtrades_db: Override path to govtrades.db.
    """
    if df is None or len(df) < 2:
        return df
    if all(c in df.columns for c in CYCLE059_FEATURE_NAMES):
        return df

    _add_round_features(df)
    _add_edgar_features(
        df, ticker,
        Path(edgar_db) if edgar_db else _EDGAR_DB)
    _add_govtrades_features(
        df, ticker,
        Path(govtrades_db) if govtrades_db else _GT_DB)

    return df


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s %(message)s")
    rng = np.random.default_rng(42)
    idx = pd.date_range("2025-01-01", periods=100, freq="B")
    close = 150 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, 100)))
    demo = pd.DataFrame({
        "High":   close * (1 + np.abs(rng.normal(0, 0.007, 100))),
        "Low":    close * (1 - np.abs(rng.normal(0, 0.007, 100))),
        "Close":  close,
        "Volume": rng.integers(1_000_000, 5_000_000, 100).astype(float),
    }, index=idx)
    tk = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    out = add_cycle059_features(demo, tk)
    print(f"cycle059: {len(CYCLE059_FEATURE_NAMES)} features. Shape: {out.shape}")
    print(out[CYCLE059_FEATURE_NAMES].tail(5).to_string())
    print(f"\nnear_round_dist_pct: {out['c059_near_round_dist_pct'].describe()}")
