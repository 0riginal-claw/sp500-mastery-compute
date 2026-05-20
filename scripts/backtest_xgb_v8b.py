"""
backtest_xgb_v8b.py — DFS-Plus XGBoost pipeline (fork of v8 Alpha158-Plus).

All available feature layers (v8 stack + Featuretools DFS module):
  - v8 base ~870+ features (v7 full stack + Qlib Alpha158 158 features)
  - Featuretools DFS 60 features: pandas-native emulation of DFS primitives
    (mean/std/sum/skew/max/min/count_above_mean x windows[5,10,20,63],
     cum_sum/cum_mean/diff/percentile, depth-2 interactions, OHLCV cross-features)
    Note: featuretools 1.31.0 is installed; DFS is implemented natively due to
    woodwork 0.31.0 + pandas 3.x accessor-caching incompatibility.
    Module: scripts/featuretools_dfs_features.py

Total: ~930+ features. Same scout-prune-refit pattern (top-K=50).

DO NOT modify v8. This is a clean fork. Key differences from v8:
  1. build_v8b_features() calls build_v8_features() then adds DFS layer.
  2. pipeline_version / strategy_variant strings updated to v8b.
  3. DFS feature importances tracked per fold.
"""
import argparse, json, os, sys, warnings
warnings.filterwarnings('ignore')
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import v8 feature builder (non-destructive -- v8.py is untouched)
from backtest_xgb_v8 import build_v8_features
from backtest_xgb_v7 import numeric_cols
import backtest_ml as bml

# Cross-sectional cache (reuse from v7/v8)
try:
    import cross_sectional_features as csf
except Exception as _e:
    print(f"[warn] csf: {_e}", file=sys.stderr)
    csf = None

# v8b NEW module: Featuretools DFS features
try:
    from featuretools_dfs_features import add_dfs_features, dfs_feature_names
    DFS_AVAILABLE = True
except Exception as _e:
    print(f"[warn] featuretools_dfs_features: {_e}", file=sys.stderr)
    DFS_AVAILABLE = False

WORK = Path("/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/s&p500-ticker-mastery")
LABEL_EMBARGO_DAYS = 21

# DFS feature prefix for importance tracking
DFS_FEAT_NAMES = dfs_feature_names() if DFS_AVAILABLE else []


def build_v8b_features(ticker: str, universe_agg: dict = None) -> pd.DataFrame:
    """
    Build v8b feature set: full v8 stack (v7 + Alpha158) + Featuretools DFS layer.
    universe_agg is the cross-sectional precomputed aggregates dict.
    """
    # Full v8 stack (unmodified: v7 + Alpha158)
    f = build_v8_features(ticker, universe_agg)
    print(f"  [v8b] after v8 stack: {f.shape[1]} cols")

    # v8b NEW layer: Featuretools DFS features
    if DFS_AVAILABLE:
        before = f.shape[1]
        try:
            f = add_dfs_features(f, ticker=ticker)
            added = f.shape[1] - before
            print(f"    +featuretools-dfs (v8b): +{added} cols -> {f.shape[1]}")
        except Exception as e:
            print(f"    [w] dfs: {e}")
    else:
        print("    [w] featuretools_dfs_features not available -- running v8 feature set only")

    f = f.loc[:, ~f.columns.duplicated()]
    f = f.dropna(subset=['rsi_14', 'atr_14', 'ema_200', 'fwd_ret_21d', 'y'])
    return f


def main():
    ap = argparse.ArgumentParser(description="XGBoost v8b -- with Featuretools DFS features")
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

    # Load cross-sectional cache
    universe_agg = None
    cache_path = WORK / "cache" / "universe_agg_manifest.json"
    if cache_path.exists() and csf is not None:
        try:
            universe_agg = csf.precompute_universe_aggregates()
        except Exception as e:
            print(f"  [csf] cache load failed: {e}")

    # Build feature set
    f = build_v8b_features(args.ticker, universe_agg)
    fc = numeric_cols(f)
    print(f"  TOTAL features: {len(fc)}; rows: {len(f)}")

    # Walk-forward folds
    folds = bml.make_walk_forward_folds(f, train_months=24, test_months=12, step_months=12)
    print(f"  folds: {len(folds)}")
    all_probs = pd.Series(np.nan, index=f.index)
    fold_summaries = []
    fold_top_features = []
    fold_dfs_importances = []  # track v8b-specific DFS feature importances

    dfs_feat_set = set(DFS_FEAT_NAMES)

    for fold in folds:
        train_end_emb = pd.Timestamp(fold['train_end']) - pd.tseries.offsets.BDay(LABEL_EMBARGO_DAYS)
        train = f[(f.index >= fold['train_start']) & (f.index < train_end_emb)]
        oos = f[(f.index >= fold['oos_start']) & (f.index < fold['oos_end'])]
        if len(train) < 50 or len(oos) < 20:
            continue

        # Scout model -- get top-K features by gain
        X_tr_all = train[fc].fillna(0).values
        y_tr = train['y'].values
        X_oos_all = oos[fc].fillna(0).values
        scout = xgb.XGBClassifier(
            max_depth=3, learning_rate=0.05, n_estimators=50,
            tree_method='hist', eval_metric='logloss',
            n_jobs=1, random_state=42, verbosity=0
        )
        scout.fit(X_tr_all, y_tr)

        importances = list(zip(fc, scout.feature_importances_))
        importances.sort(key=lambda x: -x[1])
        top_features = [c for c, imp in importances[:args.top_k] if imp > 0]
        if len(top_features) < 10:
            top_features = [c for c, _ in importances[:args.top_k]]
        fold_top_features.append({'fold': fold['fold'], 'top_features': top_features[:30]})

        # Track DFS feature importances specifically
        imp_dict = dict(importances)
        dfs_imps = {
            feat: float(imp_dict.get(feat, 0.0))
            for feat in fc
            if feat.startswith('dfs_') and imp_dict.get(feat, 0.0) > 0
        }
        top_dfs = sorted(dfs_imps.items(), key=lambda x: -x[1])[:10]
        fold_dfs_importances.append({
            'fold': fold['fold'],
            'dfs_in_top50': sum(1 for feat in top_features if feat.startswith('dfs_')),
            'top_dfs': dict(top_dfs),
        })

        # Final model on pruned features
        X_tr = train[top_features].fillna(0).values
        X_oos = oos[top_features].fillna(0).values
        final = xgb.XGBClassifier(
            max_depth=4, learning_rate=0.05, n_estimators=100,
            tree_method='hist', eval_metric='logloss',
            n_jobs=1, random_state=42, verbosity=0
        )
        final.fit(X_tr, y_tr)
        probs = final.predict_proba(X_oos)[:, 1]
        all_probs.loc[oos.index] = probs
        fold_summaries.append({
            'fold': fold['fold'],
            'n_train': len(train),
            'n_oos': len(oos),
            'n_top_features': len(top_features),
            'mean_oos_prob': float(probs.mean()),
        })

    # Threshold sweep or fixed
    if args.sweep_threshold:
        rows = []
        for thr in np.arange(0.46, 0.70, 0.02):
            sig = all_probs > thr
            trades = bml.simulate(f, sig.fillna(False), args.tp_atr, args.sl_atr, args.max_hold)
            mm = bml.compute_metrics(trades)
            rows.append({'thr': round(thr, 2), **mm})
        sdf = pd.DataFrame(rows)
        sdf.to_csv(f"{args.output_dir}/threshold_sweep.csv", index=False)
        mask = (
            (sdf['profit_factor'] >= 1.5)
            & (sdf['win_rate'] >= 0.53)
            & (sdf['n_trades'] >= 8)
            & (sdf['max_drawdown_pct'] >= -0.03)
            & (sdf['total_return_pct'] > 0)
        )
        if mask.any():
            chosen_thr = float(
                sdf[mask].sort_values('profit_factor', ascending=False).iloc[0]['thr']
            )
        else:
            chosen_thr = float(
                sdf.sort_values('profit_factor', ascending=False).iloc[0]['thr']
            )
        print(f"  -> chose thr={chosen_thr}")
    else:
        chosen_thr = args.prob_threshold
        sdf = None

    # Final simulation
    final_sig = (all_probs > chosen_thr).fillna(False)
    trades = bml.simulate(f, final_sig, args.tp_atr, args.sl_atr, args.max_hold)
    metrics = bml.compute_metrics(trades)
    print(
        f"  FINAL thr={chosen_thr}: n={metrics['n_trades']}, "
        f"WR={metrics.get('win_rate',0):.3f}, PF={metrics.get('profit_factor',0):.3f}, "
        f"RET={metrics.get('total_return_pct',0):.4f}, DD={metrics.get('max_drawdown_pct',0):.4f}"
    )
    trades.to_csv(f"{args.output_dir}/trades.csv", index=False)
    if sdf is not None:
        sdf.to_csv(f"{args.output_dir}/threshold_sweep.csv", index=False)

    def to_py(o):
        if isinstance(o, dict): return {k: to_py(v) for k, v in o.items()}
        if isinstance(o, list): return [to_py(v) for v in o]
        if hasattr(o, 'item'): return o.item()
        if isinstance(o, float) and (np.isnan(o) or np.isinf(o)): return None
        return o

    meta = to_py({
        'ticker': args.ticker,
        'pipeline_version': 'xgb_v8b_dfs',
        'strategy_variant': 'ML_XGB_v8b_featuretools_dfs',
        'run_at': datetime.utcnow().isoformat() + 'Z',
        'features_total': len(fc),
        'top_k': args.top_k,
        'rows': len(f),
        'feature_sources': {
            'base+intraday+alt+insight_v3+parts1-4': '~587',
            'cross_sectional': '17 (if cache)',
            'macro_yfinance': '40',
            'strategy_signal+five_filter': '~25',
            'google_trends': '7',
            'insider_form4': '8',
            'multi_timeframe_h1_h4_m5_m15': '15 (v7)',
            'news_sentiment_vader': '8 (v7)',
            'vol_estimators': '14 (v7)',
            'qlib_alpha158_pandas_port': '158 (v8)',
            'featuretools_dfs_pandas_native': '60 (NEW v8b)',
        },
        'dfs_feature_count': len([c for c in fc if c.startswith('dfs_')]),
        'dfs_available': DFS_AVAILABLE,
        'fold_top_features': fold_top_features,
        'fold_dfs_importances': fold_dfs_importances,
        'walk_forward_folds': len(fold_summaries),
        'strategy': {
            'name': 'ML_XGB_v8b_featuretools_dfs',
            'side': 'long',
            'tp_atr': args.tp_atr,
            'sl_atr': args.sl_atr,
            'max_hold_days': args.max_hold,
            'prob_threshold': chosen_thr,
            'threshold_swept': args.sweep_threshold,
            'model': 'XGBClassifier (scout-prune-refit top-K)',
            'slippage_bps': 5.0,
            'fee_per_share': 0.0035,
            'notional_per_trade': 5000,
        },
        'metrics_oos_aggregate': metrics,
        'fold_summaries': fold_summaries,
    })
    with open(f"{args.output_dir}/run_meta.json", 'w') as fp:
        json.dump(meta, fp, indent=2, default=str)


if __name__ == '__main__':
    main()
