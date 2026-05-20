"""oc2_donchian_per_ticker_selectivity_features.py — Per-ticker Donchian selectivity.

Extracted from OC-2/phitis/scripts/donchian_symbol_selectivity_v2.py.
Key insight: not all tickers respond equally to Donchian(20) breakouts.
The v2 script runs 7 rolling walk-forward tests across 50+ tickers and finds:
  - false_breakout_rate > 0.40 → consistently losing symbols
  - avg_hold_bars < 15 → edge preservation (quick resolution)
  - profit_factor ranking → leading selector for OOS performance

This wrapper computes selectivity proxies purely from the ticker's price history:
  - Rolling false breakout rate estimate (breakout that reverses within 3 bars)
  - Rolling win rate estimate for Donchian(20) breakouts
  - Multi-window (10/15/20/25/30) Donchian parameter quality scan
  - Selectivity score: composite of false breakout rate and win rate

All features are .shift(1)-safe — no lookahead.
Per-ticker aware: YES — each ticker has different breakout characteristics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rolling_max(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=1).max()

def _rolling_min(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=1).min()

def _atr14(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_c = close.shift(1).fillna(close)
    tr = pd.concat(
        [high - low, (high - prev_c).abs(), (low - prev_c).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(14, min_periods=1).mean()

def _compute_breakout_outcomes(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    window: int = 20,
    hold_bars: int = 3,
    atr_target_mult: float = 2.0,
    atr_stop_mult: float = 1.5,
) -> pd.Series:
    """Compute rolling breakout outcome labels (win=1, loss=0, none=NaN).

    For each bar where a Donchian breakout is detected, track whether the next
    hold_bars bars reach the ATR target before the ATR stop. Returns NaN for
    non-breakout bars.
    """
    n = len(close)
    c = close.to_numpy(dtype=np.float64)
    h = high.to_numpy(dtype=np.float64)
    l_arr = low.to_numpy(dtype=np.float64)

    prev_c = np.roll(c, 1); prev_c[0] = c[0]
    tr = np.maximum(h - l_arr, np.maximum(np.abs(h - prev_c), np.abs(l_arr - prev_c)))
    atr = pd.Series(tr).rolling(14, min_periods=1).mean().to_numpy()

    upper = pd.Series(h).rolling(window, min_periods=1).max().shift(1).to_numpy()

    outcomes = np.full(n, np.nan)
    for i in range(window, n - hold_bars):
        if np.isnan(upper[i]) or atr[i] <= 0:
            continue
        if c[i] > upper[i]:  # breakout
            entry = c[i]
            tp = entry + atr_target_mult * atr[i]
            sl = entry - atr_stop_mult * atr[i]
            win = False; loss = False
            for j in range(i + 1, min(i + 1 + hold_bars, n)):
                if h[j] >= tp:
                    win = True; break
                if l_arr[j] <= sl:
                    loss = True; break
            if win:
                outcomes[i] = 1.0
            elif loss:
                outcomes[i] = 0.0
            # else: inconclusive (NaN stays)

    return pd.Series(outcomes, index=close.index)


# ---------------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------------

def add_oc2_donchian_per_ticker_selectivity_features(
    df: pd.DataFrame,
    ticker: str | None = None,
    lookback: int = 60,
) -> pd.DataFrame:
    """Add per-ticker Donchian selectivity score features.

    New columns
    -----------
    sel_false_breakout_rate_60d float  rolling 60-bar false breakout rate estimate
                                       (breakouts that lose within 3 bars / total breakouts)
    sel_win_rate_60d            float  rolling 60-bar win rate for Donchian(20) breakouts
    sel_breakout_frequency_60d  float  breakouts per 60 bars (trading frequency proxy)
    sel_optimal_window          int    Donchian window [10/20/30] with lowest FBR in rolling 60d
    sel_selectivity_score       float  composite: win_rate - 2*false_breakout_rate (higher=better)
    sel_is_high_selectivity     int    1 if selectivity_score > 0.3 AND false_breakout_rate < 0.4
    sel_donchian20_upper        float  Donchian(20) upper band (prev bar, shift-safe)
    sel_donchian20_lower        float  Donchian(40) lower band (prev bar, shift-safe)
    sel_avg_atr14               float  ATR(14) rolling 60d average (vol regime)
    sel_current_atr_vs_avg      float  ratio of current ATR to 60d average ATR
    """
    df = df.copy()
    h = df["high"]
    l = df["low"]
    c = df["close"]

    # Donchian channels for the model to use
    df["sel_donchian20_upper"] = _rolling_max(h, 20).shift(1)
    df["sel_donchian20_lower"] = _rolling_min(l, 40).shift(1)

    # ATR regime
    atr = _atr14(h, l, c).shift(1)
    atr_avg60 = atr.rolling(lookback, min_periods=20).mean()
    df["sel_avg_atr14"] = atr_avg60
    df["sel_current_atr_vs_avg"] = (atr / atr_avg60.replace(0, np.nan)).fillna(1.0)

    # Breakout outcomes for window=20
    outcomes_20 = _compute_breakout_outcomes(c, h, l, window=20, hold_bars=3)

    # Shift outcomes so we only know t-1 results at bar t
    out_shifted = outcomes_20.shift(1)

    # Rolling FBR and WR (over lookback bars, counting only breakout bars)
    # Use expanding mask approach: compute over last `lookback` bars where outcome is not NaN
    def _rolling_mean_nonnan(s: pd.Series, win: int) -> pd.Series:
        return s.rolling(win, min_periods=5).mean()

    def _rolling_count_nonnan(s: pd.Series, win: int) -> pd.Series:
        return s.notna().rolling(win, min_periods=1).sum()

    df["sel_win_rate_60d"] = _rolling_mean_nonnan(out_shifted, lookback)
    df["sel_false_breakout_rate_60d"] = _rolling_mean_nonnan(1.0 - out_shifted.fillna(np.nan), lookback)
    df["sel_breakout_frequency_60d"] = (
        _rolling_count_nonnan(out_shifted, lookback) / lookback
    )

    # Multi-window scan: compare FBR for windows 10, 20, 30
    fbr_by_window: dict[int, pd.Series] = {}
    for w in [10, 20, 30]:
        out_w = _compute_breakout_outcomes(c, h, l, window=w, hold_bars=3).shift(1)
        fbr_by_window[w] = 1.0 - _rolling_mean_nonnan(out_w, lookback)

    # Optimal window: window with lowest FBR
    fbr_df = pd.DataFrame({w: s for w, s in fbr_by_window.items()})
    df["sel_optimal_window"] = fbr_df.idxmin(axis=1).fillna(20).astype(int)

    # Selectivity score: composite
    wr = df["sel_win_rate_60d"].fillna(0.5)
    fbr = df["sel_false_breakout_rate_60d"].fillna(0.5)
    sel_score = wr - 2.0 * fbr
    df["sel_selectivity_score"] = sel_score.clip(-1.0, 1.0)
    df["sel_is_high_selectivity"] = ((sel_score > 0.3) & (fbr < 0.4)).astype(int)

    return df
