"""hist_data_mythos_deltas_features — Mythos backtest delta features (static broadcast).

Source: AI-Tools/reports/mythos_xgboost_integration/per_ticker_summaries/mythos_cmp_*.json
Wired: 2026-05-17.  Fills the mythos curriculum prior gap from the D6 audit.

Emits 6 per-ticker STATIC features (same value every row, no shift needed):

    mythos_has_summary          1.0 if a mythos_cmp_TICKER.json exists, else 0.0
    mythos_delta_win_rate       deltas.win_rate  (mythos - baseline; negative = hurt)
    mythos_delta_profit_factor  deltas.profit_factor
    mythos_delta_total_return   deltas.total_return_pct
    mythos_baseline_profit_factor  baseline.profit_factor
    mythos_improved_flag        1.0 if delta_pf > 0 AND delta_wr > 0, else 0.0

CAVEAT: Only 7 of ~500 S&P tickers have summaries (AAPL, BXP, COIN, JPM, NVDA, TPL,
XOM).  All other tickers receive zero-filled features with mythos_has_summary=0.
Treat mythos_has_summary=0 as "no signal" — do not interpret zeros as negative.

The deltas summarise backtest outcomes FROM hindsight (curriculum mastery files) and
are intentionally static: they are a PRIOR about how mythos embedding affected this
ticker's backtest, not a per-bar signal.  No lookahead risk within a live bar sequence.

Never raises — any parse/IO failure silently zero-fills all 6 columns.
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Optional

import pandas as pd

LOG = logging.getLogger(__name__)

MYTHOS_SUMMARIES_DIR = (
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools/reports/mythos_xgboost_integration/per_ticker_summaries"
)

MYTHOS_DELTA_FEATURE_NAMES: list[str] = [
    "mythos_has_summary",
    "mythos_delta_win_rate",
    "mythos_delta_profit_factor",
    "mythos_delta_total_return",
    "mythos_baseline_profit_factor",
    "mythos_improved_flag",
]

# Module-level cache: populated once on first call, reused thereafter.
_SUMMARY_CACHE: Optional[dict[str, dict[str, float]]] = None


def _zero_fill(df: pd.DataFrame) -> pd.DataFrame:
    for c in MYTHOS_DELTA_FEATURE_NAMES:
        df[c] = 0.0
    return df


def _load_summaries() -> dict[str, dict[str, float]]:
    global _SUMMARY_CACHE
    if _SUMMARY_CACHE is not None:
        return _SUMMARY_CACHE
    out: dict[str, dict[str, float]] = {}
    try:
        d = pathlib.Path(MYTHOS_SUMMARIES_DIR)
        if not d.exists():
            LOG.warning("mythos_deltas: summaries dir missing: %s", d)
            _SUMMARY_CACHE = out
            return out
        for p in d.glob("mythos_cmp_*.json"):
            try:
                with p.open() as fh:
                    j = json.load(fh)
                tk = str(j.get("ticker", "")).strip().upper()
                if not tk:
                    tk = p.stem.replace("mythos_cmp_", "").upper()
                deltas = j.get("deltas", {})
                baseline = j.get("baseline", {})
                d_wr = float(deltas.get("win_rate", 0.0))
                d_pf = float(deltas.get("profit_factor", 0.0))
                d_ret = float(deltas.get("total_return_pct", 0.0))
                base_pf = float(baseline.get("profit_factor", 0.0))
                improved = 1.0 if (d_pf > 0.0 and d_wr > 0.0) else 0.0
                out[tk] = {
                    "mythos_has_summary": 1.0,
                    "mythos_delta_win_rate": d_wr,
                    "mythos_delta_profit_factor": d_pf,
                    "mythos_delta_total_return": d_ret,
                    "mythos_baseline_profit_factor": base_pf,
                    "mythos_improved_flag": improved,
                }
            except Exception as e:
                LOG.warning("mythos_deltas: parse failed for %s: %s", p.name, e)
    except Exception as e:
        LOG.warning("mythos_deltas: scan failed: %s", e)
    _SUMMARY_CACHE = out
    return out


def add_mythos_deltas_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Broadcast 6 static mythos-delta features across all rows for *ticker*.

    Returns a copy of *df* with the 6 MYTHOS_DELTA_FEATURE_NAMES columns added
    (or overwritten if already present).  Never raises.
    """
    df = df.copy()
    try:
        summaries = _load_summaries()
        tk = str(ticker).strip().upper()
        row = summaries.get(tk)
        if row is None:
            return _zero_fill(df)
        for c in MYTHOS_DELTA_FEATURE_NAMES:
            df[c] = row[c]
        return df
    except Exception as e:
        LOG.warning("mythos_deltas: add_mythos_deltas_features failed (%s): %s", ticker, e)
        return _zero_fill(df)


# ---------------------------------------------------------------------------
# Self-test — run with:  python hist_data_mythos_deltas_features.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG)

    # Build a 3-row dummy DataFrame
    base_df = pd.DataFrame({"close": [100.0, 101.0, 102.0]})

    # --- Test 1: AAPL (should have non-zero features) ---
    out_aapl = add_mythos_deltas_features(base_df, "AAPL")
    assert "mythos_has_summary" in out_aapl.columns, "missing column"
    assert out_aapl["mythos_has_summary"].iloc[0] == 1.0, "AAPL should have summary"
    # Values from mythos_cmp_AAPL.json
    assert abs(out_aapl["mythos_delta_win_rate"].iloc[0] - (-0.0678733032)) < 1e-6, (
        f"wr delta mismatch: {out_aapl['mythos_delta_win_rate'].iloc[0]}"
    )
    assert abs(out_aapl["mythos_delta_profit_factor"].iloc[0] - (-0.5726812344)) < 1e-6, (
        f"pf delta mismatch: {out_aapl['mythos_delta_profit_factor'].iloc[0]}"
    )
    # delta_pf < 0 → improved_flag should be 0
    assert out_aapl["mythos_improved_flag"].iloc[0] == 0.0, "AAPL improved_flag should be 0"
    # All 3 rows identical (static broadcast)
    for c in MYTHOS_DELTA_FEATURE_NAMES:
        assert out_aapl[c].nunique() == 1, f"non-static column {c}"
    print("PASS AAPL — 6 features non-zero, static broadcast, improved_flag=0")

    # --- Test 2: MSFT (not in 7-ticker set) ---
    out_msft = add_mythos_deltas_features(base_df, "MSFT")
    assert out_msft["mythos_has_summary"].iloc[0] == 0.0, "MSFT should have no summary"
    for c in MYTHOS_DELTA_FEATURE_NAMES:
        assert out_msft[c].iloc[0] == 0.0, f"MSFT {c} should be 0"
    print("PASS MSFT — all 6 features zero, mythos_has_summary=0")

    # --- Test 3: idempotency — calling twice yields same result ---
    out_aapl2 = add_mythos_deltas_features(base_df, "AAPL")
    for c in MYTHOS_DELTA_FEATURE_NAMES:
        assert (out_aapl[c] == out_aapl2[c]).all(), f"idempotency fail on {c}"
    print("PASS idempotency")

    print("\nAll self-tests PASSED")
    sys.exit(0)
