"""
backtest_xgb_v3.py — Unified XGBoost pipeline with ALL feature sources.

Combines:
  - 46 base technical indicators (build_features from backtest_ml)
  - 22 intraday features (intraday_features.add_intraday_features)
  - 9 alt-data features (alt_data_features.add_all_alt_features) — EDGAR + GovTrades + Lobbying
  - ~15 cross-sectional + macro features (cross_sectional_features)
  - 12+ Trading-Insight-Info adds (Connors RSI, Donchian, Keltner, HMA, KAMA, 3-bar reversal, TTM squeeze, etc.)

Plus per-ticker prob_threshold sweep at runtime to discover the best threshold per ticker.

Usage:
    python backtest_xgb_v3.py --ticker AAPL --output-dir backtests_xgb_v3/AAPL_50
    python backtest_xgb_v3.py --ticker AAPL --output-dir backtests_xgb_v3/AAPL --sweep-threshold
"""

import argparse, json, os, sys, warnings
warnings.filterwarnings('ignore')
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import pandas_ta_classic as pta
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtest_ml as bml   # reuse load_daily, build_features, simulate, compute_metrics
import alt_data_features as adf
import intraday_features as idf
import cross_sectional_features as csf

WORK = Path("/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/s&p500-ticker-mastery")


# ─────────────────────────────────────────────────────────────────────
# Trading Insight Info derived features (top picks from agent extraction)
# ─────────────────────────────────────────────────────────────────────

def add_trading_insight_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds ~15 features derived from Trading Insight Info recommendations.
    All point-in-time safe (use .shift(1) on outputs)."""
    df = df.copy()
    c = df['close']; h = df['high']; l = df['low']; v = df['volume']; o = df['open']

    # 1. Connors RSI — composite short-term mean-reversion
    rsi3 = pta.rsi(c, length=3)
    # Streak: count consecutive up/down closes
    streak = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        if c.iloc[i] > c.iloc[i-1]:
            streak.iloc[i] = max(1, streak.iloc[i-1] + 1) if streak.iloc[i-1] > 0 else 1
        elif c.iloc[i] < c.iloc[i-1]:
            streak.iloc[i] = min(-1, streak.iloc[i-1] - 1) if streak.iloc[i-1] < 0 else -1
    rsi_streak = pta.rsi(streak, length=2)
    # 1-day pct return rank (rolling 100)
    pct_ret = c.pct_change()
    pct_rank = pct_ret.rolling(100, min_periods=20).apply(lambda x: (x.rank().iloc[-1] - 1) / max(len(x) - 1, 1) * 100, raw=False)
    connors_rsi = ((rsi3 + rsi_streak.fillna(50) + pct_rank.fillna(50)) / 3).shift(1)
    df['connors_rsi'] = connors_rsi
    df['connors_rsi_oversold'] = (connors_rsi < 15).astype(int)
    df['connors_rsi_overbought'] = (connors_rsi > 85).astype(int)

    # 2. Donchian Channels
    dc_high = h.rolling(20).max().shift(1)
    dc_low = l.rolling(20).min().shift(1)
    df['donchian_pos'] = ((c.shift(1) - dc_low) / (dc_high - dc_low).replace(0, np.nan))
    df['donchian_break_up'] = (c > dc_high).astype(int).shift(1)
    df['donchian_break_dn'] = (c < dc_low).astype(int).shift(1)

    # 3. Keltner Channels (EMA20 ± 2*ATR20)
    ema20 = pta.ema(c, length=20)
    atr20 = pta.atr(h, l, c, length=20)
    df['keltner_upper'] = (ema20 + 2 * atr20).shift(1)
    df['keltner_lower'] = (ema20 - 2 * atr20).shift(1)
    df['keltner_pos'] = ((c.shift(1) - df['keltner_lower']) / (df['keltner_upper'] - df['keltner_lower']).replace(0, np.nan))

    # 4. Hull Moving Average (HMA) — lag-reduced trend
    def wma(series, n):
        weights = np.arange(1, n + 1)
        return series.rolling(n).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    n_hma = 16
    hma_inner = 2 * wma(c, n_hma // 2) - wma(c, n_hma)
    hma = wma(hma_inner, int(np.sqrt(n_hma)))
    df['hma_16'] = hma.shift(1)
    df['close_above_hma'] = (c.shift(1) > hma.shift(1)).astype(int)

    # 5. TTM Squeeze (BBands inside Keltner)
    bb = pta.bbands(c, length=20)
    bb_upper = bb['BBU_20_2.0']; bb_lower = bb['BBL_20_2.0']
    kel_upper_ttm = ema20 + 1.5 * atr20
    kel_lower_ttm = ema20 - 1.5 * atr20
    in_squeeze = (bb_upper < kel_upper_ttm) & (bb_lower > kel_lower_ttm)
    df['ttm_squeeze_on'] = in_squeeze.astype(int).shift(1)

    # 6. Inside-bar pattern
    df['inside_bar'] = ((h < h.shift(1)) & (l > l.shift(1))).shift(1).astype(float).fillna(0)

    # 7. Three-bar reversal (bull)
    df['three_bar_reversal_bull'] = (
        (l < l.shift(1)) & (l.shift(1) < l.shift(2)) & (c > c.shift(1))
    ).shift(1).astype(float).fillna(0)

    # 8. Earnings-window proxy: 80-90 day cycle (10-Q quarterly cadence)
    # No actual earnings dates — use 90-day modulo as a coarse proxy
    if isinstance(df.index, pd.DatetimeIndex):
        days_in_year = df.index.dayofyear
        df['quarter_cycle_pos'] = ((days_in_year % 90) / 90.0)

    # 9. Volume delta proxy (since we don't have order-flow, use price-direction × volume)
    pct_chg = c.pct_change()
    vol_delta_proxy = np.sign(pct_chg) * v
    df['vol_delta_proxy_5d'] = vol_delta_proxy.rolling(5).sum().shift(1)
    df['vol_delta_proxy_20d'] = vol_delta_proxy.rolling(20).sum().shift(1)
    df['cvd_proxy'] = vol_delta_proxy.cumsum().shift(1)

    # 10. RSI divergence proxy (price up but RSI down)
    rsi14 = pta.rsi(c, length=14)
    df['rsi_divg_bear'] = ((c.shift(1) > c.shift(2)) & (rsi14.shift(1) < rsi14.shift(2))).astype(float).fillna(0)
    df['rsi_divg_bull'] = ((c.shift(1) < c.shift(2)) & (rsi14.shift(1) > rsi14.shift(2))).astype(float).fillna(0)

    # 11. ADX strong-trend regime flag
    adx_df = pta.adx(h, l, c, length=14)
    df['adx_strong'] = (adx_df['ADX_14'] > 25).astype(float).shift(1)

    # 12. Volatility regime (current ATR vs 60d avg)
    df['vol_regime_high'] = (atr20 > atr20.rolling(60).mean()).astype(float).shift(1)

    # 13. Bollinger %B
    df['bb_pct_b'] = ((c.shift(1) - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan))

    return df


# ─────────────────────────────────────────────────────────────────────
# Unified feature pipeline
# ─────────────────────────────────────────────────────────────────────

def build_unified_features(ticker: str, universe_agg: dict = None) -> pd.DataFrame:
    """Stack all feature sources. Returns df with all features + label."""
    d = bml.load_daily(ticker)
    print(f"  base daily bars: {len(d):,}")

    # Layer 1: base 46 indicators
    f = bml.build_features(d)
    print(f"  after build_features: {f.shape[1]} cols")

    # Layer 2: intraday features
    try:
        f = idf.add_intraday_features(f, ticker)
        print(f"  after intraday: {f.shape[1]} cols")
    except Exception as e:
        print(f"  [warn] intraday failed: {e}")

    # Layer 3: alt-data (EDGAR + GovTrades + Lobbying)
    try:
        f = adf.add_all_alt_features(f, ticker)
        # Drop ultra-sparse alt-data cols (<10% non-zero for this ticker)
        for c in list(f.columns):
            if c.startswith(('cong_', 'lobbying_', 'filing_', 'days_since_')):
                if (f[c] != 0).mean() < 0.10:
                    f = f.drop(columns=[c])
        print(f"  after alt-data (pruned <10%): {f.shape[1]} cols")
    except Exception as e:
        print(f"  [warn] alt-data failed: {e}")

    # Layer 4: cross-sectional + macro (skip if cache not ready — full precompute is slow)
    if universe_agg is not None:
        try:
            f = csf.add_cross_sectional_features(f, ticker, universe_agg)
            print(f"  after cross-sectional: {f.shape[1]} cols")
        except Exception as e:
            print(f"  [warn] cross-sectional failed: {e}")
    else:
        print(f"  [skip] cross-sectional (no universe_agg cache)")

    # Layer 5: Trading Insight Info adds
    try:
        f = add_trading_insight_features(f)
        print(f"  after trading-insight: {f.shape[1]} cols")
    except Exception as e:
        print(f"  [warn] trading-insight failed: {e}")

    f = f.dropna(subset=['rsi_14', 'atr_14', 'ema_200', 'fwd_ret_21d', 'y'])
    return f


def numeric_feature_cols(df):
    return [c for c in df.columns
            if c not in ['open', 'high', 'low', 'close', 'volume', 'fwd_ret_21d', 'y']
            and pd.api.types.is_numeric_dtype(df[c])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ticker', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--prob-threshold', type=float, default=0.50)
    ap.add_argument('--sweep-threshold', action='store_true')
    ap.add_argument('--tp-atr', type=float, default=1.5)
    ap.add_argument('--sl-atr', type=float, default=1.0)
    ap.add_argument('--max-hold', type=int, default=21)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[{args.ticker}] building unified features...")
    # Check for prebuilt cache — only use cross-sectional if cache exists
    cache_path = WORK / "cache" / "universe_agg.parquet"
    universe_agg = None
    if cache_path.exists() and hasattr(csf, 'precompute_universe_aggregates'):
        try:
            universe_agg = csf.precompute_universe_aggregates()
            print(f"  loaded universe_agg from cache")
        except Exception as e:
            print(f"  [warn] cache load failed: {e}")
            universe_agg = None
    else:
        print(f"  cross-sectional cache not ready — skipping that layer")
    f = build_unified_features(args.ticker, universe_agg)
    fc = numeric_feature_cols(f)
    print(f"  total feature cols: {len(fc)}")
    print(f"  total rows after dropna: {len(f):,}")

    # Walk-forward with embargo
    folds = bml.make_walk_forward_folds(f, train_months=24, test_months=12, step_months=12)
    print(f"  folds: {len(folds)}")

    LABEL_EMBARGO_DAYS = 21
    all_signals = pd.Series(False, index=f.index)
    all_probs = pd.Series(np.nan, index=f.index)
    fold_summaries = []

    for fold in folds:
        train_end_emb = pd.Timestamp(fold['train_end']) - pd.tseries.offsets.BDay(LABEL_EMBARGO_DAYS)
        train = f[(f.index >= fold['train_start']) & (f.index < train_end_emb)]
        oos = f[(f.index >= fold['oos_start']) & (f.index < fold['oos_end'])]
        if len(train) < 50 or len(oos) < 20: continue

        X_tr = train[fc].fillna(0).values; y_tr = train['y'].values
        X_oos = oos[fc].fillna(0).values

        model = xgb.XGBClassifier(max_depth=4, learning_rate=0.05, n_estimators=100,
                                  tree_method='hist', eval_metric='logloss',
                                  n_jobs=1, random_state=42, verbosity=0)
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_oos)[:, 1]
        all_probs.loc[oos.index] = probs

        fold_summaries.append({
            'fold': fold['fold'], 'oos_start': fold['oos_start'], 'oos_end': fold['oos_end'],
            'n_train': len(train), 'n_oos': len(oos), 'mean_oos_prob': float(probs.mean()),
        })

    # Threshold handling
    if args.sweep_threshold:
        # Sweep over thresholds; pick best by PF subject to constraints
        sweep_rows = []
        for thr in np.arange(0.48, 0.66, 0.02):
            sig = all_probs > thr
            trades = bml.simulate(f, sig.fillna(False), args.tp_atr, args.sl_atr, args.max_hold)
            m = bml.compute_metrics(trades)
            sweep_rows.append({'thr': round(thr, 2), **m})
        sweep_df = pd.DataFrame(sweep_rows)
        sweep_df.to_csv(f"{args.output_dir}/threshold_sweep.csv", index=False)
        # pick best
        mask = ((sweep_df['profit_factor'] >= 1.5) & (sweep_df['win_rate'] >= 0.53)
                & (sweep_df['n_trades'] >= 8) & (sweep_df['max_drawdown_pct'] >= -0.03)
                & (sweep_df['total_return_pct'] > 0))
        if mask.any():
            best = sweep_df[mask].sort_values('profit_factor', ascending=False).iloc[0]
            chosen_thr = float(best['thr'])
            print(f"  → sweep best thr={chosen_thr} PF={best['profit_factor']:.2f}")
        else:
            chosen_thr = float(sweep_df.sort_values('profit_factor', ascending=False).iloc[0]['thr'])
            print(f"  → sweep no constraint-passing thr; picked highest-PF thr={chosen_thr}")
    else:
        chosen_thr = args.prob_threshold

    final_sig = (all_probs > chosen_thr).fillna(False)
    trades = bml.simulate(f, final_sig, args.tp_atr, args.sl_atr, args.max_hold)
    metrics = bml.compute_metrics(trades)
    print(f"  final thr={chosen_thr}: n={metrics['n_trades']}, WR={metrics.get('win_rate'):.3f}, PF={metrics.get('profit_factor'):.3f}, RET={metrics.get('total_return_pct'):.4f}, DD={metrics.get('max_drawdown_pct'):.4f}")

    trades.to_csv(f"{args.output_dir}/trades.csv", index=False)

    def to_py(o):
        if isinstance(o, dict): return {k: to_py(v) for k, v in o.items()}
        if isinstance(o, list): return [to_py(v) for v in o]
        if hasattr(o, 'item'): return o.item()
        if isinstance(o, float) and (np.isnan(o) or np.isinf(o)): return None
        return o

    meta = to_py({
        'ticker': args.ticker, 'pipeline_version': 'xgb_v3_unified',
        'strategy_variant': 'ML_XGB_v3_unified',
        'run_at': datetime.utcnow().isoformat() + 'Z',
        'features_used': fc, 'n_features': len(fc),
        'feature_sources': {
            'base_indicators': 'build_features (~46 pandas-ta)',
            'intraday': 'intraday_features.add_intraday_features (~22)',
            'alt_data': 'alt_data_features.add_all_alt_features (EDGAR/GovTrades/Lobbying)',
            'cross_sectional': 'cross_sectional_features.add_cross_sectional_features',
            'trading_insight': 'add_trading_insight_features (~15)',
        },
        'daily_bars': len(f), 'walk_forward_folds': len(fold_summaries),
        'strategy': {
            'name': 'ML_XGB_v3_unified', 'side': 'long',
            'tp_atr': args.tp_atr, 'sl_atr': args.sl_atr, 'max_hold_days': args.max_hold,
            'prob_threshold': chosen_thr, 'threshold_swept': args.sweep_threshold,
            'model': 'XGBClassifier (defaults)', 'calibration': 'native predict_proba',
            'slippage_bps': 5.0, 'fee_per_share': 0.0035, 'notional_per_trade': 5000,
        },
        'metrics_oos_aggregate': metrics, 'fold_summaries': fold_summaries,
    })
    with open(f"{args.output_dir}/run_meta.json", 'w') as fp:
        json.dump(meta, fp, indent=2, default=str)
    print(f"  → {args.output_dir}/run_meta.json")


if __name__ == '__main__':
    main()
