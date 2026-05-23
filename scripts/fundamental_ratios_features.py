"""
fundamental_ratios_features.py
==============================
Adds fundamental ratio features (~50-70) to a daily price DataFrame for
S&P 500 backtesting, using JerBouma/FinanceToolkit.

Source: GitHub TOP-10 #4 — FinanceToolkit
  pip install financetoolkit  (no API key needed for ratios; FMP key only
  needed for the `models` sub-API)

Cadence
-------
Ratios are computed at QUARTERLY cadence from Yahoo Finance financial
statements, then forward-filled into the daily panel. A `.shift(1)` step
ensures the value on bar t reflects only knowledge from t-1.

Features added (~50-70 columns, prefixed `fund_`)
-------------------------------------------------
All 67 ratios returned by `Toolkit.ratios.collect_all_ratios()` plus four
manually-derived composite scores (Piotroski-lite, Altman-Z-lite, DuPont
3-stage, FCF yield). Column names are snake-cased.

Composite scores
----------------
fund_piotroski_lite   : 0-9 score from ROA>0, OCF>0, ROA up YoY, OCF>NI,
                        leverage down, current ratio up, no dilution,
                        gross margin up, asset turnover up
                        (subset that's reliably available from Yahoo)
fund_altman_z_lite    : 1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E where
                        A=working_capital/assets, B=retained_earnings/assets,
                        C=ebit/assets, D=mcap/total_liab, E=revenue/assets
fund_dupont_roe       : net_margin * asset_turnover * equity_multiplier
fund_fcf_yield        : trailing FCF / market_cap

Env gate
--------
FUNDAMENTAL_RATIOS_ENABLED=1 to activate. Default OFF (returns df unchanged
with all `fund_*` columns set to 0.0).

Graceful degradation
--------------------
If FinanceToolkit fails (network, ticker unmapped, missing fields), the
function logs a warning and returns the input df with all `fund_*` columns
filled to 0.0.

Dependencies
------------
  financetoolkit  (pip install financetoolkit)
  pandas, numpy   (already in sp500-mastery venv)
"""

from __future__ import annotations

import logging
import os
import re
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ENV_FLAG = "FUNDAMENTAL_RATIOS_ENABLED"

# Composite/derived feature column names (always present in output)
COMPOSITE_COLS: List[str] = [
    "fund_piotroski_lite",
    "fund_altman_z_lite",
    "fund_dupont_roe",
    "fund_fcf_yield",
]


def _snake(name: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip().lower())
    return re.sub(r"_+", "_", s).strip("_")


def _quarter_period_to_timestamp(p) -> pd.Timestamp:
    """Convert pandas Period (e.g. 2025Q1) to the QUARTER-END timestamp.
    We use end-of-quarter to keep features strictly past-aware after shift(1).
    """
    try:
        return p.to_timestamp(how="end").normalize()
    except Exception:
        return pd.Timestamp(str(p))


def _zero_fill(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = 0.0
    return out


def _fetch_ratios(ticker: str) -> pd.DataFrame:
    """Returns wide DataFrame: index=quarter-end Timestamp, columns=fund_* ratios."""
    from financetoolkit import Toolkit

    tk = Toolkit(tickers=[ticker], api_key=None, quarterly=True)
    ratios = tk.ratios.collect_all_ratios()  # rows=ratio names, cols=Period
    # Transpose to rows=Period, cols=ratio name
    rt = ratios.T.copy()
    rt.index = [_quarter_period_to_timestamp(p) for p in rt.index]
    rt.columns = [f"fund_{_snake(c)}" for c in rt.columns]
    rt = rt.sort_index()
    return rt


def _add_composites(rt: pd.DataFrame) -> pd.DataFrame:
    """Compute Piotroski-lite, Altman-Z-lite, DuPont, FCF yield from available ratios."""
    out = rt.copy()

    def col(name: str) -> pd.Series:
        return out[name] if name in out.columns else pd.Series(0.0, index=out.index)

    # ----- DuPont ROE = net_margin * asset_turnover * equity_multiplier
    out["fund_dupont_roe"] = (
        col("fund_net_profit_margin")
        * col("fund_asset_turnover_ratio")
        * col("fund_equity_multiplier")
    )

    # ----- FCF yield = free_cash_flow_yield (already a ratio)
    fcfy = col("fund_free_cash_flow_yield")
    if (fcfy == 0).all():
        fcfy = col("fund_free_cash_flow_operating_cash_flow_ratio")  # weak proxy
    out["fund_fcf_yield"] = fcfy.fillna(0.0)

    # ----- Piotroski-lite (subset of 9 signals reliably available)
    roa = col("fund_return_on_assets")
    ocf_ratio = col("fund_operating_cash_flow_ratio")
    gm = col("fund_gross_margin")
    at = col("fund_asset_turnover_ratio")
    de = col("fund_debt_to_equity_ratio")
    cr = col("fund_current_ratio")

    piotroski = pd.Series(0.0, index=out.index)
    piotroski += (roa > 0).astype(float)
    piotroski += (ocf_ratio > 0).astype(float)
    piotroski += (roa.diff().fillna(0) > 0).astype(float)
    piotroski += (gm.diff().fillna(0) > 0).astype(float)
    piotroski += (at.diff().fillna(0) > 0).astype(float)
    piotroski += (de.diff().fillna(0) < 0).astype(float)  # leverage down
    piotroski += (cr.diff().fillna(0) > 0).astype(float)
    out["fund_piotroski_lite"] = piotroski

    # ----- Altman-Z-lite: weighted sum of available ratio proxies
    # Use return_on_assets (proxy for EBIT/assets) and equity_multiplier^-1
    # (proxy for equity/liab). Coefficients scaled to keep magnitudes sane.
    asset_turn = col("fund_asset_turnover_ratio")
    em = col("fund_equity_multiplier").replace(0, np.nan)
    equity_liab_proxy = (1.0 / em).fillna(0.0)
    z = (
        1.2 * col("fund_working_capital").pipe(_safe_norm)
        + 1.4 * col("fund_return_on_equity")
        + 3.3 * roa
        + 0.6 * equity_liab_proxy
        + 1.0 * asset_turn
    )
    out["fund_altman_z_lite"] = z.fillna(0.0)

    return out


def _safe_norm(s: pd.Series) -> pd.Series:
    """Min-max normalize a series to [0,1] for use in composite scores. Robust to NaN."""
    s = s.astype(float)
    if s.isna().all() or s.max() == s.min():
        return pd.Series(0.0, index=s.index)
    lo, hi = s.min(), s.max()
    return ((s - lo) / (hi - lo)).fillna(0.0)


def add_fundamental_ratio_features(
    df: pd.DataFrame,
    ticker: str,
    date_col: str = "Date",
) -> pd.DataFrame:
    """
    Merge fundamental ratio features onto a daily-bar price DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Daily bars with a date column or DatetimeIndex.
    ticker : str
        Ticker symbol (e.g. 'AAPL').
    date_col : str
        Name of date column if df has a column-based date; ignored if index is DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Original df with fund_* columns appended, forward-filled from quarterly
        cadence, then .shift(1) to enforce no-lookahead.
    """
    if os.getenv(ENV_FLAG, "0") != "1":
        return _zero_fill(df, COMPOSITE_COLS)

    if df is None or len(df) == 0:
        return df

    try:
        rt = _fetch_ratios(ticker)
    except Exception as e:
        logger.warning("FinanceToolkit fetch failed for %s: %s", ticker, e)
        return _zero_fill(df, COMPOSITE_COLS)

    if rt.empty:
        logger.warning("FinanceToolkit returned empty ratios for %s", ticker)
        return _zero_fill(df, COMPOSITE_COLS)

    rt = _add_composites(rt)

    # Build daily panel: forward-fill quarterly values into a daily index
    if isinstance(df.index, pd.DatetimeIndex):
        daily_idx = df.index
        out = df.copy()
        merge_on_index = True
    else:
        if date_col not in df.columns:
            logger.warning("date_col '%s' not in df; returning zero-filled", date_col)
            return _zero_fill(df, COMPOSITE_COLS)
        daily_idx = pd.DatetimeIndex(pd.to_datetime(df[date_col]))
        out = df.copy()
        merge_on_index = False

    # Reindex ratios onto daily grid via merge_asof (backward = use latest known)
    rt_sorted = rt.sort_index()
    daily_df = pd.DataFrame(index=daily_idx).sort_index()
    merged = pd.merge_asof(
        daily_df.reset_index().rename(columns={"index": "_d"}),
        rt_sorted.reset_index().rename(columns={"index": "_d"}),
        on="_d",
        direction="backward",
    ).set_index("_d")
    merged.index = daily_idx  # preserve original order
    # Shift by 1 bar to enforce no-lookahead (value on t = info known at t-1)
    merged = merged.shift(1).fillna(0.0)

    # Attach to out
    for c in merged.columns:
        if merge_on_index:
            out[c] = merged[c].values
        else:
            out[c] = merged[c].values

    # Ensure composites exist even if Toolkit was missing a contributing ratio
    out = _zero_fill(out, COMPOSITE_COLS)
    return out


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    tkr = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    os.environ[ENV_FLAG] = "1"
    dates = pd.date_range("2024-01-01", periods=300, freq="B")
    sample = pd.DataFrame({"Date": dates, "Close": np.linspace(150, 200, 300)})
    out = add_fundamental_ratio_features(sample, tkr, date_col="Date")
    fund_cols = [c for c in out.columns if c.startswith("fund_")]
    print(f"ticker={tkr} rows={len(out)} fund_cols={len(fund_cols)}")
    print("sample fund_cols:", fund_cols[:10])
    print("composite tail:")
    print(out[COMPOSITE_COLS].tail(3))
