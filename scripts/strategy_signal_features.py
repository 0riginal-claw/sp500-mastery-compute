"""
strategy_signal_features.py — Two feature modules for the SP500 ML pipeline.

Module 1 — add_strategy_signal_features
    Converts the three rule-based strategies (D1_REV, D2_MOM, D3_GOLD) into
    binary/numeric ML features.  Input df must have build_features() already
    applied (rsi_14, ema_20, ema_50, ema_200, ret_21d, vol_sma_20, atr_14).

Module 2 — add_five_filter_stack
    Implements the 5-filter stack (vol surge, ATR expansion, VWAP alignment,
    Donchian breakout, 3-TF trend) cited in OC-2 research as producing
    80-85% WR when all five align.  Daily-OHLCV proxies only.

All computed series are shifted by 1 before assignment so there is zero
lookahead — the feature at index t reflects information available at the
close of bar t-1.

Dependencies: pandas, numpy (both already present in the sp500-mastery venv).
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helper: safe rolling percentile rank (returns Series in [0,1])
# ---------------------------------------------------------------------------

def _pct_rank(s: pd.Series, window: int, min_periods: int = 60) -> pd.Series:
    """Rolling percentile rank of s within the trailing `window` bars."""
    return s.rolling(window, min_periods=min_periods).rank(pct=True)


# ---------------------------------------------------------------------------
# Module 1 — Strategy signal features
# ---------------------------------------------------------------------------

def add_strategy_signal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds rule-based strategy signals as binary/numeric features.

    Input: daily df with build_features() already applied.
    Required columns: rsi_14, ema_20, ema_50, ema_200, ret_21d,
                      vol_sma_20, atr_14, close, volume.

    Returns df with new columns appended.  All .shift(1) safe —
    no bar-t close/volume data leaks into the features.

    New columns
    -----------
    d1_rev_signal       int   RSI(14) < 30
    d2_mom_signal       int   21d return in top-30% of rolling 252d history
    d3_gold_signal      int   close > ema50 AND ema50 > ema200 AND close < ema20
    d1_d2_agree         int   both d1 and d2 firing
    d1_d3_agree         int   both d1 and d3 firing
    d2_d3_agree         int   both d2 and d3 firing
    n_strategies_firing int   count of {d1, d2, d3} simultaneously firing
    days_since_d1       float bars elapsed since last d1_rev_signal (NaN if never)
    days_since_d2       float bars elapsed since last d2_mom_signal (NaN if never)
    """
    df = df.copy()

    # ---- pre-shifted series (all reflect bar t-1 close prices) -------------
    # build_features already shifts rsi_14, ema_20, ema_50, ema_200
    # so we can use them directly without an extra shift.
    rsi = df['rsi_14']                    # already shifted inside build_features
    ema20 = df['ema_20']
    ema50 = df['ema_50']
    ema200 = df['ema_200']
    ret21 = df['ret_21d']                 # already shifted
    close_prev = df['close'].shift(1)     # yesterday close for D3 GOLD check

    # D1_REV — RSI(14) < 30 on the prior-day close
    df['d1_rev_signal'] = (rsi < 30).astype(int)

    # D2_MOM — 21d return in top-30% of own 252-day history
    #   ret_21d is already shifted; we rank within a 252-bar rolling window,
    #   then shift one more time so rank at t is computed from bar t-1 data.
    ret21_rank = _pct_rank(df['close'].pct_change(21), window=252).shift(1)
    df['d2_mom_signal'] = (ret21_rank > 0.70).astype(int)

    # D3_GOLD — golden-cross pullback: close > ema50 > ema200, close < ema20
    #   All ema columns are already shifted; close_prev is bar t-1 close.
    df['d3_gold_signal'] = (
        (close_prev > ema50) &
        (ema50 > ema200) &
        (close_prev < ema20) &
        ema200.notna()
    ).astype(int)

    # ---- agreement features ------------------------------------------------
    d1 = df['d1_rev_signal']
    d2 = df['d2_mom_signal']
    d3 = df['d3_gold_signal']

    df['d1_d2_agree'] = (d1 & d2).astype(int)
    df['d1_d3_agree'] = (d1 & d3).astype(int)
    df['d2_d3_agree'] = (d2 & d3).astype(int)
    df['n_strategies_firing'] = d1 + d2 + d3

    # ---- recency features: bars elapsed since last signal ------------------
    def _days_since(signal_col: pd.Series) -> pd.Series:
        """Returns bars elapsed since last True; NaN if signal has never fired."""
        arr = signal_col.values.astype(float)          # 0/1/NaN, numpy array
        idx = np.arange(len(arr), dtype=float)
        last = np.where(arr == 1, idx, np.nan)
        # forward-fill the last-seen position
        mask = np.isnan(last)
        for i in range(1, len(last)):
            if mask[i]:
                last[i] = last[i - 1]
        result = np.where(np.isnan(last), np.nan, idx - last)
        return pd.Series(result, index=signal_col.index, dtype=float)

    df['days_since_d1'] = _days_since(d1)
    df['days_since_d2'] = _days_since(d2)

    return df


# ---------------------------------------------------------------------------
# Module 2 — 5-filter stack
# ---------------------------------------------------------------------------

def add_five_filter_stack(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the 5-filter stack producing 80-85% WR when all fire (OC-2 research).

    Input: daily OHLCV df with build_features() applied.
    Required columns: close, high, low, volume, atr_14.

    Filters
    -------
    F1 — Volume surge: volume / 20d rolling average
    F2 — ATR expansion: ATR(14) / 60d rolling ATR mean
    F3 — VWAP alignment proxy: close above vol-weighted avg of last 21 days
    F4 — Donchian breakout (ORB proxy): close > rolling-20d high
    F5 — 3-TF trend alignment: daily AND weekly bullish simultaneously

    All features are shifted 1 bar so bar-t features use only bar t-1 data.

    New columns
    -----------
    f1_vol_above_1_5x   int  volume > 1.5x 20d avg
    f1_vol_above_2x     int  volume > 2.0x 20d avg
    f2_atr_above_1x     int  ATR > 1.0x 60d ATR mean
    f2_atr_above_1_5x   int  ATR > 1.5x 60d ATR mean
    f3_above_vwap_proxy int  close > 21d vol-weighted avg price
    f4_breakout_today   int  close > rolling-20d high (computed on prev bar)
    f5_three_tf_rule    int  close > ema20 AND ret_5d > 0 AND ret_21d > 0
    f5_strict_all5      int  all five primary filters firing simultaneously
    n_filters_passing   int  count of {f1_1.5x, f2_1x, f3, f4, f5} firing
    filter_score_wtd    float weighted score (vol=0.2, atr=0.2, vwap=0.25,
                                              brk=0.2, 3tf=0.15)
    """
    df = df.copy()

    c = df['close']
    h = df['high']
    l = df['low']
    v = df['volume']

    # ---- F1: Volume surge --------------------------------------------------
    vol_ma20 = v.rolling(20, min_periods=10).mean()
    vol_ratio = v / vol_ma20.replace(0, np.nan)

    df['f1_vol_above_1_5x'] = (vol_ratio > 1.5).shift(1).astype(int)
    df['f1_vol_above_2x']   = (vol_ratio > 2.0).shift(1).astype(int)

    # ---- F2: ATR expansion -------------------------------------------------
    # atr_14 is already shifted once in build_features; we need the raw ATR
    # here to compute a rolling mean, so we recompute it without shift.
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr14_raw = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    atr_ma60 = atr14_raw.rolling(60, min_periods=30).mean()
    atr_ratio = atr14_raw / atr_ma60.replace(0, np.nan)

    df['f2_atr_above_1x']   = (atr_ratio > 1.0).shift(1).astype(int)
    df['f2_atr_above_1_5x'] = (atr_ratio > 1.5).shift(1).astype(int)

    # ---- F3: VWAP alignment proxy (21-day vol-weighted price) --------------
    # typical_price * volume sum / volume sum over 21 bars
    tp = (h + l + c) / 3
    tp_x_v = tp * v
    vwap21 = tp_x_v.rolling(21, min_periods=10).sum() / v.rolling(21, min_periods=10).sum()

    df['f3_above_vwap_proxy'] = (c > vwap21).shift(1).astype(int)

    # ---- F4: Donchian breakout (ORB proxy) ---------------------------------
    # close > highest close of the prior 20 bars (excludes current bar)
    roll20_high = c.shift(1).rolling(20, min_periods=10).max()
    df['f4_breakout_today'] = (c.shift(1) > roll20_high.shift(1)).astype(int)

    # ---- F5: 3-TF trend alignment ------------------------------------------
    # Daily: close > ema20 (already shifted in build_features)
    # Weekly proxy: ret_5d > 0 (already shifted)
    # Monthly proxy: ret_21d > 0 (already shifted)
    ema20 = df['ema_20']           # already shifted
    ret5  = df['ret_5d']           # already shifted
    ret21 = df['ret_21d']          # already shifted

    df['f5_three_tf_rule'] = (
        (c.shift(1) > ema20) &
        (ret5 > 0) &
        (ret21 > 0)
    ).astype(int)

    # ---- Composite features ------------------------------------------------
    f1 = df['f1_vol_above_1_5x']
    f2 = df['f2_atr_above_1x']
    f3 = df['f3_above_vwap_proxy']
    f4 = df['f4_breakout_today']
    f5 = df['f5_three_tf_rule']

    # Strict: all five simultaneously
    df['f5_strict_all5'] = (f1 & f2 & f3 & f4 & f5).astype(int)

    # Count and weighted score
    df['n_filters_passing'] = f1 + f2 + f3 + f4 + f5

    df['filter_score_wtd'] = (
        f1.astype(float) * 0.20 +
        f2.astype(float) * 0.20 +
        f3.astype(float) * 0.25 +
        f4.astype(float) * 0.20 +
        f5.astype(float) * 0.15
    )

    return df


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    sys.path.insert(
        0,
        '/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com'
        '/My Drive/AI-Tools/s&p500-ticker-mastery/scripts'
    )
    import backtest_ml as bml

    print('Loading AAPL daily data ...')
    df = bml.load_daily('AAPL')
    df = bml.build_features(df)
    print(f'Base df: {len(df)} rows, {len(df.columns)} cols')

    out = add_strategy_signal_features(df.copy())
    new1 = [c for c in out.columns if c not in df.columns]

    out2 = add_five_filter_stack(out.copy())
    new2 = [c for c in out2.columns if c not in out.columns]

    print(f'\nstrategy_signals: +{len(new1)}, five_filter: +{len(new2)}')
    print(f'Total new: {len(new1) + len(new2)}')
    print()

    for col in (new1 + new2)[:30]:
        if pd.api.types.is_numeric_dtype(out2[col]):
            nz = (out2[col].notna() & (out2[col] != 0)).mean()
            print(f'  {col}: {nz*100:.0f}% non-zero')
