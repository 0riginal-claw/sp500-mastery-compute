"""
backtest_xgb_v7.py — MAXIMAL-PLUS XGBoost pipeline.

All available feature layers (v6 stack + 2 new modules):
  - v5 base 587 features (technical, intraday, alt-data, trading-insight parts 1-4, v3 hand-coded)
  - cross-sectional 17 features (cache-loaded fast)
  - macro 40 features (yfinance no-key)
  - strategy signal 10 features + 5-filter stack 15 features
  - google trends 7 features
  - insider Form 4 8 features
  - multi-timeframe 15 features: H1/H4/M5/M15 + cross-TF (NEW v7)
  - news sentiment 8 features: VADER scores on yfinance news (NEW v7)
  - advanced volatility estimators 15 features: Parkinson/GK/RS/YZ x 3 windows
    + 3 ratio features (vol_yz_20_vs_60, vol_yz_realized_eff, vol_pk_vs_yz_20)

Total: ~715+ features. Same scout-prune-refit pattern as v5/v6 (top-K=50).
"""
import argparse, json, os, sys, warnings
warnings.filterwarnings('ignore')
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtest_ml as bml
import alt_data_features as adf
import intraday_features as idf

# v3 trading insight (13 hand-coded)
from backtest_xgb_v3 import add_trading_insight_features

# v5/v4 trading insight parts 1-4 (~496 features)
try:
    from trading_insight_features_part1 import add_features_part1
except Exception as e:
    print(f"[warn] part1: {e}", file=sys.stderr); add_features_part1 = None
try:
    from trading_insight_features_part2 import add_features_part2
except Exception as e:
    print(f"[warn] part2: {e}", file=sys.stderr); add_features_part2 = None
try:
    from trading_insight_features_part3 import add_features_part3
except Exception as e:
    print(f"[warn] part3: {e}", file=sys.stderr); add_features_part3 = None
try:
    from trading_insight_features_part4 import add_features_part4
except Exception as e:
    print(f"[warn] part4: {e}", file=sys.stderr); add_features_part4 = None

# v6 modules
try: import cross_sectional_features as csf
except Exception as e:
    print(f"[warn] csf: {e}", file=sys.stderr); csf = None
try: import macro_features as macf
except Exception as e:
    print(f"[warn] macf: {e}", file=sys.stderr); macf = None
try:
    from strategy_signal_features import add_strategy_signal_features, add_five_filter_stack
except Exception as e:
    print(f"[warn] strategy_signal: {e}", file=sys.stderr)
    add_strategy_signal_features = None; add_five_filter_stack = None
try: import google_trends_features as gtf
except Exception as e:
    print(f"[warn] gtf: {e}", file=sys.stderr); gtf = None
try: import insider_form4_features as f4f
except Exception as e:
    print(f"[warn] f4f: {e}", file=sys.stderr); f4f = None

# v7 NEW modules
try: import multi_timeframe_features as mtff
except Exception as e:
    print(f"[warn] mtff: {e}", file=sys.stderr); mtff = None
try: import news_sentiment_features as nsf
except Exception as e:
    print(f"[warn] nsf: {e}", file=sys.stderr); nsf = None

# v7 alt-data extension: EDGAR v2 + gov contracts + lobbying issues (12 features)
try: import alt_data_features_v2 as adf2
except Exception as e:
    print(f"[warn] adf2: {e}", file=sys.stderr); adf2 = None

# v7 patch: advanced range-based volatility estimators (15 features)
try:
    from volatility_estimator_features import add_volatility_features as _add_vol_features
except Exception as e:
    print(f"[warn] vol-estimators: {e}", file=sys.stderr); _add_vol_features = None

WORK = Path("/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/s&p500-ticker-mastery")
LABEL_EMBARGO_DAYS = 21


def build_v7_features(ticker: str, universe_agg: dict = None) -> pd.DataFrame:
    d = bml.load_daily(ticker)
    print(f"  [v7] base daily: {len(d):,}")
    f = bml.build_features(d)
    print(f"    +build_features: {f.shape[1]}")

    # Standard layers (inherited from v6)
    try: f = idf.add_intraday_features(f, ticker); print(f"    +intraday: {f.shape[1]}")
    except Exception as e: print(f"    [w] intraday: {e}")
    try:
        f = adf.add_all_alt_features(f, ticker)
        for c in list(f.columns):
            if c.startswith(('cong_','lobbying_','filing_','days_since_')):
                if (f[c]!=0).mean() < 0.10: f = f.drop(columns=[c])
        print(f"    +alt-data (pruned): {f.shape[1]}")
    except Exception as e: print(f"    [w] alt-data: {e}")
    if universe_agg is not None and csf is not None:
        try: f = csf.add_cross_sectional_features(f, ticker, universe_agg); print(f"    +cross-sectional: {f.shape[1]}")
        except Exception as e: print(f"    [w] csf: {e}")
    try: f = add_trading_insight_features(f); print(f"    +trading-insight v3: {f.shape[1]}")
    except Exception as e: print(f"    [w] insight-v3: {e}")
    for name, fn in [('part1',add_features_part1),('part2',add_features_part2),
                     ('part3',add_features_part3),('part4',add_features_part4)]:
        if fn is None: continue
        try: f = fn(f); print(f"    +insight-{name}: {f.shape[1]}")
        except Exception as e: print(f"    [w] {name}: {e}")

    # v6 NEW layers
    if macf is not None:
        try: f = macf.add_macro_features(f); print(f"    +macro: {f.shape[1]}")
        except Exception as e: print(f"    [w] macro: {e}")
    if add_strategy_signal_features is not None:
        try: f = add_strategy_signal_features(f); print(f"    +strategy-signal: {f.shape[1]}")
        except Exception as e: print(f"    [w] strat-sig: {e}")
    if add_five_filter_stack is not None:
        try: f = add_five_filter_stack(f); print(f"    +5-filter: {f.shape[1]}")
        except Exception as e: print(f"    [w] 5-filter: {e}")
    if gtf is not None:
        try: f = gtf.add_google_trends_features(f, ticker); print(f"    +gtrends: {f.shape[1]}")
        except Exception as e: print(f"    [w] gtrends: {e}")
    if f4f is not None:
        try: f = f4f.add_insider_form4_features(f, ticker); print(f"    +form4: {f.shape[1]}")
        except Exception as e: print(f"    [w] form4: {e}")

    # v7 NEW layers
    if mtff is not None:
        try: f = mtff.add_multi_timeframe_features(f, ticker); print(f"    +multi-timeframe: {f.shape[1]}")
        except Exception as e: print(f"    [w] mtff: {e}")
    if nsf is not None:
        try: f = nsf.add_news_sentiment_features(f, ticker); print(f"    +news-sentiment: {f.shape[1]}")
        except Exception as e: print(f"    [w] nsf: {e}")

    # alt-data v2: EDGAR v2 + gov contracts + lobbying issues (12 new features)
    if adf2 is not None:
        try: f = adf2.add_all_v2_features(f, ticker); print(f"    +alt-data-v2: {f.shape[1]}")
        except Exception as e: print(f"    [w] adf2: {e}")

    # v7 patch: advanced range-based volatility estimators (15 features)
    if _add_vol_features is not None:
        try: f = _add_vol_features(f); print(f"    +vol-estimators (P/GK/RS/YZ x3 + 3 ratios): {f.shape[1]}")
        except Exception as e: print(f"    [w] vol-estimators: {e}")

    f = f.loc[:, ~f.columns.duplicated()]
    f = f.dropna(subset=['rsi_14','atr_14','ema_200','fwd_ret_21d','y'])
    return f


def numeric_cols(df):
    return [c for c in df.columns if c not in ['open','high','low','close','volume','fwd_ret_21d','y']
            and pd.api.types.is_numeric_dtype(df[c])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ticker', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--prob-threshold', type=float, default=0.50)
    ap.add_argument('--sweep-threshold', action='store_true')
    ap.add_argument('--top-k', type=int, default=50)
    ap.add_argument('--tp-atr', type=float, default=1.5)
    ap.add_argument('--sl-atr', type=float, default=1.0)
    ap.add_argument('--max-hold', type=int, default=21)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    universe_agg = None
    cache_path = WORK / "cache" / "universe_agg_manifest.json"
    if cache_path.exists() and csf is not None:
        try: universe_agg = csf.precompute_universe_aggregates()
        except Exception as e: print(f"  [csf] cache load failed: {e}")

    f = build_v7_features(args.ticker, universe_agg)
    fc = numeric_cols(f)
    print(f"  TOTAL features: {len(fc)}; rows: {len(f)}")

    folds = bml.make_walk_forward_folds(f, train_months=24, test_months=12, step_months=12)
    print(f"  folds: {len(folds)}")
    all_probs = pd.Series(np.nan, index=f.index)
    fold_summaries = []
    fold_top_features = []

    for fold in folds:
        train_end_emb = pd.Timestamp(fold['train_end']) - pd.tseries.offsets.BDay(LABEL_EMBARGO_DAYS)
        train = f[(f.index >= fold['train_start']) & (f.index < train_end_emb)]
        oos = f[(f.index >= fold['oos_start']) & (f.index < fold['oos_end'])]
        if len(train) < 50 or len(oos) < 20: continue

        # Scout model — get top-K features by gain
        X_tr_all = train[fc].fillna(0).values; y_tr = train['y'].values
        X_oos_all = oos[fc].fillna(0).values
        scout = xgb.XGBClassifier(max_depth=3, learning_rate=0.05, n_estimators=50,
                                  tree_method='hist', eval_metric='logloss',
                                  n_jobs=1, random_state=42, verbosity=0)
        scout.fit(X_tr_all, y_tr)
        importances = list(zip(fc, scout.feature_importances_))
        importances.sort(key=lambda x: -x[1])
        top_features = [c for c, imp in importances[:args.top_k] if imp > 0]
        if len(top_features) < 10:
            top_features = [c for c, _ in importances[:args.top_k]]
        fold_top_features.append({'fold': fold['fold'], 'top_features': top_features[:30]})

        # Final model on pruned features
        X_tr = train[top_features].fillna(0).values
        X_oos = oos[top_features].fillna(0).values
        final = xgb.XGBClassifier(max_depth=4, learning_rate=0.05, n_estimators=100,
                                  tree_method='hist', eval_metric='logloss',
                                  n_jobs=1, random_state=42, verbosity=0)
        final.fit(X_tr, y_tr)
        probs = final.predict_proba(X_oos)[:, 1]
        all_probs.loc[oos.index] = probs
        fold_summaries.append({'fold': fold['fold'], 'n_train': len(train), 'n_oos': len(oos),
                               'n_top_features': len(top_features),
                               'mean_oos_prob': float(probs.mean())})

    # Threshold handling
    if args.sweep_threshold:
        rows = []
        for thr in np.arange(0.46, 0.70, 0.02):
            sig = all_probs > thr
            trades = bml.simulate(f, sig.fillna(False), args.tp_atr, args.sl_atr, args.max_hold)
            mm = bml.compute_metrics(trades)
            rows.append({'thr': round(thr, 2), **mm})
        sdf = pd.DataFrame(rows); sdf.to_csv(f"{args.output_dir}/threshold_sweep.csv", index=False)
        mask = ((sdf['profit_factor']>=1.5)&(sdf['win_rate']>=0.53)&(sdf['n_trades']>=8)
                &(sdf['max_drawdown_pct']>=-0.03)&(sdf['total_return_pct']>0))
        if mask.any():
            chosen_thr = float(sdf[mask].sort_values('profit_factor', ascending=False).iloc[0]['thr'])
        else:
            chosen_thr = float(sdf.sort_values('profit_factor', ascending=False).iloc[0]['thr'])
        print(f"  -> chose thr={chosen_thr}")
    else:
        chosen_thr = args.prob_threshold

    final_sig = (all_probs > chosen_thr).fillna(False)
    trades = bml.simulate(f, final_sig, args.tp_atr, args.sl_atr, args.max_hold)
    metrics = bml.compute_metrics(trades)
    print(f"  FINAL thr={chosen_thr}: n={metrics['n_trades']}, WR={metrics.get('win_rate',0):.3f}, PF={metrics.get('profit_factor',0):.3f}, RET={metrics.get('total_return_pct',0):.4f}, DD={metrics.get('max_drawdown_pct',0):.4f}")
    trades.to_csv(f"{args.output_dir}/trades.csv", index=False)

    def to_py(o):
        if isinstance(o, dict): return {k: to_py(v) for k,v in o.items()}
        if isinstance(o, list): return [to_py(v) for v in o]
        if hasattr(o, 'item'): return o.item()
        if isinstance(o, float) and (np.isnan(o) or np.isinf(o)): return None
        return o

    meta = to_py({
        'ticker': args.ticker, 'pipeline_version': 'xgb_v7_maximal_plus',
        'strategy_variant': 'ML_XGB_v7_maximal_plus',
        'run_at': datetime.utcnow().isoformat()+'Z',
        'features_total': len(fc), 'top_k': args.top_k, 'rows': len(f),
        'feature_sources': {
            'base+intraday+alt+insight_v3+parts1-4': '~587',
            'cross_sectional': '17 (if cache)',
            'macro_yfinance': '40',
            'strategy_signal+five_filter': '~25',
            'google_trends': '7',
            'insider_form4': '8',
            'multi_timeframe_h1_h4_m5_m15': '15 (NEW v7)',
            'news_sentiment_vader': '8 (NEW v7)',
        },
        'fold_top_features': fold_top_features,
        'walk_forward_folds': len(fold_summaries),
        'strategy': {'name':'ML_XGB_v7_maximal_plus', 'side':'long',
                     'tp_atr':args.tp_atr, 'sl_atr':args.sl_atr, 'max_hold_days':args.max_hold,
                     'prob_threshold':chosen_thr, 'threshold_swept':args.sweep_threshold,
                     'model':'XGBClassifier (scout-prune-refit top-K)',
                     'slippage_bps':5.0, 'fee_per_share':0.0035, 'notional_per_trade':5000},
        'metrics_oos_aggregate': metrics, 'fold_summaries': fold_summaries,
    })
    with open(f"{args.output_dir}/run_meta.json", 'w') as fp:
        json.dump(meta, fp, indent=2, default=str)


if __name__ == '__main__':
    main()
