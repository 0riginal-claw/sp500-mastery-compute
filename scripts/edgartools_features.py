"""
Wrapper: edgartools (dgunning/edgartools) — SEC EDGAR insider trade signals.
Fetches Form 4 filings for a ticker and computes buy/sell pressure signals
that can be joined to an OHLCV dataframe.

Note: requires network access to SEC EDGAR. Results cached locally.
If edgartools not installed, attempts pip install; if offline, returns zeros.
"""
import sys
import os
import subprocess
from pathlib import Path
from datetime import timedelta

import pandas as pd

EDGAR_CLONE = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools/repos-claude-clones/edgartools"
)


def _ensure_edgartools():
    try:
        import edgar  # noqa
        return True
    except ImportError:
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "edgartools", "-q"],
                check=True, timeout=60,
            )
            return True
        except Exception:
            return False


def _fetch_form4_transactions(ticker: str, days_lookback: int = 365) -> pd.DataFrame:
    """
    Pull Form 4 transactions for ticker from EDGAR.
    Returns DataFrame with columns: date, shares, value, is_purchase.
    """
    from edgar import Company, set_identity
    set_identity("research@example.com")  # required by SEC fair-access policy
    company = Company(ticker)
    filings = company.get_filings(form="4")
    if filings is None or len(filings) == 0:
        return pd.DataFrame(columns=["date", "shares", "value", "is_purchase"])

    rows = []
    cutoff = pd.Timestamp.today() - timedelta(days=days_lookback)
    for filing in filings[:50]:  # cap at 50 most recent
        try:
            ownership = filing.obj()
            tx_df = ownership.transactions
            if tx_df is None or len(tx_df) == 0:
                continue
            for _, row in tx_df.iterrows():
                dt = pd.to_datetime(row.get("transactionDate", None), errors="coerce")
                if pd.isna(dt) or dt < cutoff:
                    continue
                shares = float(row.get("transactionShares", 0) or 0)
                price = float(row.get("transactionPricePerShare", 0) or 0)
                code = str(row.get("transactionCode", ""))
                is_purchase = code in ("P", "M", "G", "A")
                rows.append({"date": dt.date(), "shares": abs(shares), "value": abs(shares * price), "is_purchase": is_purchase})
        except Exception:
            continue

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["date", "shares", "value", "is_purchase"])


def add_edgartools_features(
    df: pd.DataFrame,
    ticker: str = None,
    date_col: str = None,
    window_days: int = 30,
    days_lookback: int = 365,
) -> pd.DataFrame:
    """
    Add SEC Form 4 insider-trade signal columns to OHLCV dataframe.

    Adds columns:
      edgar_insider_buy_count_{w}d   — # buy transactions in trailing window
      edgar_insider_sell_count_{w}d  — # sell transactions in trailing window
      edgar_insider_net_shares_{w}d  — net shares (buys - sells) in trailing window
      edgar_insider_buy_pressure_{w}d — buy_value / (buy_value + sell_value + 1)

    ticker: stock symbol (e.g. "AAPL"). If None, all insider columns = 0.
    date_col: column with dates; if None, uses df.index.
    window_days: rolling window in calendar days.
    days_lookback: how far back to pull Form 4 filings.
    """
    df = df.copy()
    w = window_days
    cols = [
        f"edgar_insider_buy_count_{w}d",
        f"edgar_insider_sell_count_{w}d",
        f"edgar_insider_net_shares_{w}d",
        f"edgar_insider_buy_pressure_{w}d",
    ]

    # default to zeros
    for c in cols:
        df[c] = 0.0

    if ticker is None:
        return df

    if not _ensure_edgartools():
        return df

    try:
        tx = _fetch_form4_transactions(ticker, days_lookback=days_lookback)
        if tx.empty:
            return df

        tx["date"] = pd.to_datetime(tx["date"])
        tx_buys = tx[tx["is_purchase"]]
        tx_sells = tx[~tx["is_purchase"]]

        # get dates from df
        if date_col:
            dates = pd.to_datetime(df[date_col])
        else:
            dates = pd.to_datetime(df.index)

        buy_cnt, sell_cnt, net_shares, buy_pres = [], [], [], []
        delta = timedelta(days=w)
        for dt in dates:
            mask_b = (tx_buys["date"] >= dt - delta) & (tx_buys["date"] <= dt)
            mask_s = (tx_sells["date"] >= dt - delta) & (tx_sells["date"] <= dt)
            bc = int(mask_b.sum())
            sc = int(mask_s.sum())
            ns = float(tx_buys.loc[mask_b, "shares"].sum()) - float(tx_sells.loc[mask_s, "shares"].sum())
            bv = float(tx_buys.loc[mask_b, "value"].sum())
            sv = float(tx_sells.loc[mask_s, "value"].sum())
            bp = bv / (bv + sv + 1.0)
            buy_cnt.append(bc)
            sell_cnt.append(sc)
            net_shares.append(ns)
            buy_pres.append(bp)

        df[f"edgar_insider_buy_count_{w}d"] = buy_cnt
        df[f"edgar_insider_sell_count_{w}d"] = sell_cnt
        df[f"edgar_insider_net_shares_{w}d"] = net_shares
        df[f"edgar_insider_buy_pressure_{w}d"] = buy_pres
    except Exception as e:
        print(f"[edgartools_features] fetch failed: {e}. Returning zero columns.")

    return df


if __name__ == "__main__":
    import numpy as np
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    rng = np.random.default_rng(7)
    test_df = pd.DataFrame(
        {"open": 100, "high": 105, "low": 95, "close": 100 + rng.standard_normal(10).cumsum(), "volume": 1e6},
        index=dates,
    )
    # dry run without network — ticker=None returns zeros
    out = add_edgartools_features(test_df, ticker=None)
    assert "edgar_insider_buy_count_30d" in out.columns
    print(out[["close", "edgar_insider_buy_count_30d", "edgar_insider_buy_pressure_30d"]])
    print("edgartools_features smoke test PASS")
