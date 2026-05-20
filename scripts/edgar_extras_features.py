"""edgar_extras_features.py — EDGAR features NOT covered by hist_data_edgar_features.

Source: claudes test/data/edgar/data/edgar.db (SQLite, 57,066 filings,
500 tickers, 2020-01-02 to 2026-04-24).

Why this exists
---------------
`hist_data_edgar_features.py` already extracts 9 features from edgar.db:
days_since_any/8k/10q/10k, filing_flag_7d/30d, eightk_flag_7d, filings_count_90d,
has_10k_this_year.

The gap-analysis (research/edgar_govtrades_full/gap_analysis_2026-05-20.md)
identified the following data IN edgar.db but NOT in any feature module:

  G1. DEF 14A (proxy statements) — 3,289 rows. Governance / compensation events.
      No existing feature flags proxy filings.
  G2. Amendments (8-K/A, 10-K/A, 10-Q/A) — 962 rows. Restatements, often
      followed by abnormal moves. No existing feature flags amendments.
  G3. Period-of-report mismatch — earnings 8-K typically files within 2-5 days
      of fiscal period end. Current edgar features ignore `period_of_report`.
      Diff (filed_at - period_of_report) is a strong "earnings 8-K vs other
      8-K" proxy because earnings releases lag period end by <10 days while
      other 8-Ks (exec change, M&A, agreement) have no period.
  G4. S-1 / S-1/A (IPO / secondary registration) — 48 rows. Dilution risk
      signal — usually preceded by stock-price elevation, followed by
      reset.
  G5. Filing burst — count of filings in trailing 7 days. Spikes precede
      large moves more reliably than 30/90d densities (which average out
      bursts).

This module adds 12 features (all .shift(1)-safe via the same merge_asof
backward / searchsorted side='left' patterns used in hist_data_edgar_features).

Wired 2026-05-20 under mission `edgar_govtrades_full` — fills the
"raw EDGAR extras" gap from the 11-folder catalog audit.
"""
from __future__ import annotations

import logging
import os
import sqlite3
from typing import Optional

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

EDGAR_DB_PATH = (
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/"
    "My Drive/claudes test/data/edgar/data/edgar.db"
)

EDGAR_EXTRAS_FEATURE_NAMES: list[str] = [
    # G1 — DEF 14A
    "edgar_days_since_def14a",
    "edgar_def14a_flag_30d",
    # G2 — amendments
    "edgar_days_since_any_amendment",
    "edgar_amendment_flag_30d",
    # G3 — earnings 8-K proxy via period-of-report lag
    "edgar_days_since_likely_earnings_8k",
    "edgar_likely_earnings_8k_flag_7d",
    "edgar_filed_to_period_lag_days",
    # G4 — S-1 dilution risk
    "edgar_days_since_s1",
    "edgar_s1_flag_180d",
    # G5 — filing burst (7-day)
    "edgar_filings_count_7d",
    "edgar_burst_flag",
    # G6 — filing density acceleration (7d / 30d ratio)
    "edgar_filing_density_accel",
]

_CAP_DAYS = 9999
_FILINGS_CACHE: dict[str, pd.DataFrame] = {}


def _zero_fill(df: pd.DataFrame) -> pd.DataFrame:
    for c in EDGAR_EXTRAS_FEATURE_NAMES:
        df[c] = 0
    return df


def _load_filings(ticker: str) -> Optional[pd.DataFrame]:
    """Load all filings (with period_of_report) for a ticker from edgar.db."""
    if ticker in _FILINGS_CACHE:
        return _FILINGS_CACHE[ticker]
    if not os.path.exists(EDGAR_DB_PATH):
        LOG.warning("[edgar_extras] DB not found at %s", EDGAR_DB_PATH)
        _FILINGS_CACHE[ticker] = pd.DataFrame(
            columns=["ticker", "form", "filed_at", "period_of_report"]
        )
        return _FILINGS_CACHE[ticker]
    try:
        with sqlite3.connect(f"file:{EDGAR_DB_PATH}?mode=ro", uri=True) as conn:
            q = (
                "SELECT ticker, form, filed_at, period_of_report FROM filings "
                "WHERE ticker = ? ORDER BY filed_at"
            )
            f = pd.read_sql_query(q, conn, params=(ticker,))
        if f.empty:
            _FILINGS_CACHE[ticker] = f
            return f
        f["filed_at"] = pd.to_datetime(
            f["filed_at"], utc=True, errors="coerce"
        ).dt.tz_convert(None)
        f["period_of_report"] = pd.to_datetime(
            f["period_of_report"], errors="coerce"
        )
        f = f.dropna(subset=["filed_at"]).reset_index(drop=True)
        _FILINGS_CACHE[ticker] = f
        return f
    except Exception as e:
        LOG.warning("[edgar_extras] DB read failed for %s: %s", ticker, e)
        _FILINGS_CACHE[ticker] = pd.DataFrame(
            columns=["ticker", "form", "filed_at", "period_of_report"]
        )
        return _FILINGS_CACHE[ticker]


def _resolve_dates(df: pd.DataFrame) -> Optional[pd.Series]:
    """Resolve the bar-date series from df['date'] or DatetimeIndex; None on failure."""
    try:
        if "date" in df.columns:
            s = pd.to_datetime(df["date"], errors="coerce")
            if hasattr(s.dt, "tz") and s.dt.tz is not None:
                s = s.dt.tz_convert(None)
            return s
        if isinstance(df.index, pd.DatetimeIndex):
            idx = df.index.tz_convert(None) if df.index.tz is not None else df.index
            return pd.Series(idx, index=df.index)
    except Exception as e:
        LOG.warning("[edgar_extras] date resolution failed: %s", e)
    return None


def _days_since(bars_df: pd.DataFrame, filings_df: pd.DataFrame) -> pd.Series:
    """merge_asof backward, strict-prior. Returns 9999 if no prior."""
    if filings_df.empty:
        return pd.Series([_CAP_DAYS] * len(bars_df), dtype="int64")
    m = pd.merge_asof(
        bars_df[["date"]],
        filings_df.rename(columns={"filed_at": "_f"})[["_f"]].sort_values("_f"),
        left_on="date",
        right_on="_f",
        direction="backward",
        allow_exact_matches=False,
    )
    d = (m["date"] - m["_f"]).dt.days
    return d.fillna(_CAP_DAYS).clip(upper=_CAP_DAYS).astype("int64")


def _count_in_window(
    bars_dates_np: np.ndarray,
    filing_dates_np: np.ndarray,
    days: int,
) -> np.ndarray:
    """Count strictly-prior filings in trailing window of `days` calendar days."""
    if filing_dates_np.size == 0:
        return np.zeros(len(bars_dates_np), dtype="int32")
    win_lo = bars_dates_np - np.timedelta64(days, "D")
    hi_idx = np.searchsorted(filing_dates_np, bars_dates_np, side="left")
    lo_idx = np.searchsorted(filing_dates_np, win_lo, side="left")
    return (hi_idx - lo_idx).astype("int32")


def add_edgar_extras_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Add 12 EDGAR extras features (DEF 14A, amendments, likely-earnings 8-K,
    S-1, burst, accel).

    Required input: df['date'] (datetime64) or df with DatetimeIndex.
    Output: df with EDGAR_EXTRAS_FEATURE_NAMES columns appended. Always returns
    df (graceful zero-fill on any error). Producer emits past-only values.
    """
    df = df.copy()
    try:
        dates = _resolve_dates(df)
        if dates is None or dates.isna().all():
            LOG.warning("[edgar_extras] %s: no usable 'date' col; zero-filling", ticker)
            return _zero_fill(df)

        filings = _load_filings(ticker)
        if filings is None or filings.empty:
            return _zero_fill(df)

        form_upper = filings["form"].astype(str).str.upper()
        # G1 DEF 14A
        f_def14a = (
            filings.loc[form_upper.str.startswith("DEF 14A"), ["filed_at"]]
            .sort_values("filed_at")
            .reset_index(drop=True)
        )
        # G2 amendments — anything with /A suffix
        f_amend = (
            filings.loc[form_upper.str.contains(r"/A$", regex=True), ["filed_at"]]
            .sort_values("filed_at")
            .reset_index(drop=True)
        )
        # G3 likely earnings 8-K: 8-K with period_of_report within 10 days of filed_at
        f_8k = filings.loc[form_upper == "8-K"].copy()
        if not f_8k.empty:
            f_8k["_lag"] = (f_8k["filed_at"] - f_8k["period_of_report"]).dt.days
            # earnings 8-Ks lag period end by 0-10 days; non-earnings have NaN
            # or much larger lags (or report immediate events with lag=0 on
            # a non-fiscal date).
            mask_earn = (f_8k["_lag"] >= 0) & (f_8k["_lag"] <= 10) & (f_8k["period_of_report"].notna())
            f_earn8k = (
                f_8k.loc[mask_earn, ["filed_at"]]
                .sort_values("filed_at")
                .reset_index(drop=True)
            )
        else:
            f_earn8k = pd.DataFrame(columns=["filed_at"])
        # G4 S-1
        f_s1 = (
            filings.loc[form_upper.str.startswith("S-1"), ["filed_at"]]
            .sort_values("filed_at")
            .reset_index(drop=True)
        )
        # G5 any filing (for burst + accel)
        f_any = (
            filings[["filed_at"]].sort_values("filed_at").reset_index(drop=True)
        )

        # Build bars (preserve orig idx)
        bars = (
            pd.DataFrame({"date": dates.values})
            .reset_index(drop=False)
            .rename(columns={"index": "_orig_idx"})
        )
        bars["date"] = pd.to_datetime(bars["date"])
        bars_sorted = bars.sort_values("date").reset_index(drop=True)

        # --- G1 DEF 14A ---
        bars_sorted["edgar_days_since_def14a"] = _days_since(bars_sorted, f_def14a)
        bars_sorted["edgar_def14a_flag_30d"] = (
            bars_sorted["edgar_days_since_def14a"] <= 30
        ).astype("int8")

        # --- G2 amendments ---
        bars_sorted["edgar_days_since_any_amendment"] = _days_since(bars_sorted, f_amend)
        bars_sorted["edgar_amendment_flag_30d"] = (
            bars_sorted["edgar_days_since_any_amendment"] <= 30
        ).astype("int8")

        # --- G3 likely earnings 8-K ---
        bars_sorted["edgar_days_since_likely_earnings_8k"] = _days_since(
            bars_sorted, f_earn8k
        )
        bars_sorted["edgar_likely_earnings_8k_flag_7d"] = (
            bars_sorted["edgar_days_since_likely_earnings_8k"] <= 7
        ).astype("int8")
        # Lag at the most recent 8-K (whatever form) — proxy for filing freshness
        # We compute lag of the most recent earnings-like 8-K observed strictly prior.
        if not f_earn8k.empty:
            # Re-derive: for each bar, find prior earn8k, lookup lag
            f8k_for_join = f_8k.loc[(f_8k["_lag"] >= 0) & (f_8k["_lag"] <= 10) & (f_8k["period_of_report"].notna())][
                ["filed_at", "_lag"]
            ].sort_values("filed_at").reset_index(drop=True)
            m = pd.merge_asof(
                bars_sorted[["date"]],
                f8k_for_join.rename(columns={"filed_at": "_f"}),
                left_on="date",
                right_on="_f",
                direction="backward",
                allow_exact_matches=False,
            )
            bars_sorted["edgar_filed_to_period_lag_days"] = (
                m["_lag"].fillna(_CAP_DAYS).clip(upper=_CAP_DAYS).astype("int64")
            )
        else:
            bars_sorted["edgar_filed_to_period_lag_days"] = _CAP_DAYS

        # --- G4 S-1 ---
        bars_sorted["edgar_days_since_s1"] = _days_since(bars_sorted, f_s1)
        bars_sorted["edgar_s1_flag_180d"] = (
            bars_sorted["edgar_days_since_s1"] <= 180
        ).astype("int8")

        # --- G5 filing burst (7d) ---
        any_dates_np = (
            f_any["filed_at"].values.astype("datetime64[ns]")
            if not f_any.empty
            else np.array([], dtype="datetime64[ns]")
        )
        bars_dates_np = bars_sorted["date"].values.astype("datetime64[ns]")
        c7 = _count_in_window(bars_dates_np, any_dates_np, 7)
        c30 = _count_in_window(bars_dates_np, any_dates_np, 30)
        bars_sorted["edgar_filings_count_7d"] = c7
        bars_sorted["edgar_burst_flag"] = (c7 >= 3).astype("int8")

        # --- G6 accel = (7d_rate) / (30d_rate) ---
        # rate = count / window_days; ratio with safe-div
        rate7 = c7 / 7.0
        rate30 = np.where(c30 > 0, c30 / 30.0, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            accel = np.where(rate30 > 0, rate7 / rate30, 0.0)
        accel = np.nan_to_num(accel, nan=0.0, posinf=10.0, neginf=0.0)
        bars_sorted["edgar_filing_density_accel"] = accel.astype("float32")

        # Restore original order
        bars_back = bars_sorted.sort_values("_orig_idx").reset_index(drop=True)
        for c in EDGAR_EXTRAS_FEATURE_NAMES:
            df[c] = bars_back[c].values
        return df
    except Exception as e:
        LOG.warning("[edgar_extras] add_edgar_extras_features failed for %s: %s", ticker, e)
        return _zero_fill(df)


# WIRE_CANDIDATE marker for the consumer auto-wirer
WIRE_CANDIDATE = True
WIRE_MODULE_NAME = "edgar_extras_features"


# Smoke runner — `python edgar_extras_features.py AAPL XOM`
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    tickers = sys.argv[1:] or ["AAPL", "XOM"]
    for tk in tickers:
        dates = pd.date_range("2022-01-01", "2025-12-31", freq="D")
        df = pd.DataFrame({"date": dates})
        out = add_edgar_extras_features(df, tk)
        print(f"--- {tk} ---")
        print(f"  shape: {df.shape} -> {out.shape}")
        print(f"  new cols: {len(EDGAR_EXTRAS_FEATURE_NAMES)}")
        # non-zero columns
        for c in EDGAR_EXTRAS_FEATURE_NAMES:
            s = out[c]
            try:
                pos = (s != 0).sum()
                pct = 100.0 * pos / len(s)
                print(f"    {c}: nonzero={pos:>6d} ({pct:5.1f}%)  min={s.min()}  max={s.max()}")
            except Exception as e:
                print(f"    {c}: err {e}")
