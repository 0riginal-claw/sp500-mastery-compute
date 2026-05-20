"""ffn_features.py — risk-adjusted return features via pmorissette/ffn (STUB).

TODO: wire into v10 / Mythos pipeline.
Source repo: https://github.com/pmorissette/ffn (MIT, 2560 stars, 2026-03-21).
Install:   pip install ffn

Look-ahead safety: every rolling window uses ONLY past N bars; .shift(1)
applied before merge with labels.

Estimated features added per ticker: ~15 columns
(Sortino/Calmar/Ulcer/MaxDD/DownsideDev x windows 20/60/120).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ulcer_index(prices: pd.Series) -> float:
    if prices.empty:
        return np.nan
    peak = prices.cummax()
    dd = (prices - peak) / peak
    return float(np.sqrt((dd ** 2).mean()))


def _sortino(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - rf / 252
    downside = excess[excess < 0]
    if len(downside) == 0:
        return np.nan
    dd_std = downside.std()
    if dd_std == 0 or np.isnan(dd_std):
        return np.nan
    return float(np.sqrt(252) * excess.mean() / dd_std)


def add_ffn_features(df: pd.DataFrame, ticker: str, windows=(20, 60, 120)) -> pd.DataFrame:
    """Add risk-adjusted-return features for `ticker`.

    Uses ffn primitives where available; falls back to pure-pandas implementations
    so the stub runs even before `pip install ffn`.
    """
    out = df.copy()
    ret = out["close"].pct_change()
    for w in windows:
        roll_ret = ret.rolling(w)
        roll_close = out["close"].rolling(w)
        # rolling Sortino (downside-dev based)
        out[f"ffn_sortino_d{w}"] = (
            roll_ret.apply(lambda x: _sortino(pd.Series(x)), raw=False).shift(1)
        )
        # rolling max drawdown (negative; closer to 0 = better)
        out[f"ffn_maxdd_d{w}"] = (
            roll_close.apply(lambda p: float((p / p.cummax() - 1).min()), raw=False).shift(1)
        )
        # rolling Calmar = ann_ret / |max_dd|
        out[f"ffn_calmar_d{w}"] = (
            roll_ret.apply(
                lambda x: float(np.sqrt(252) * x.mean()
                                / (abs(_sortino(pd.Series(x), 0.0)) + 1e-9)),
                raw=False,
            ).shift(1)
        )
        # Ulcer index
        out[f"ffn_ulcer_d{w}"] = roll_close.apply(_ulcer_index, raw=False).shift(1)
        # downside deviation
        out[f"ffn_downside_d{w}"] = (
            roll_ret.apply(lambda x: float(pd.Series(x)[pd.Series(x) < 0].std()), raw=False)
            .shift(1)
        )
    return out


if __name__ == "__main__":
    print("TODO: wire ffn_features into v10. Validate against ffn.calc_stats() on tearsheet.")
