"""
backtest_xgb_v8.py — Alpha158-Plus XGBoost pipeline (fork of v7).

All available feature layers (v7 stack + Qlib Alpha158 module):
  - v7 base ~715+ features (technical, intraday, alt-data, trading-insight parts 1-4,
    cross-sectional, macro, strategy-signal, 5-filter, google-trends, form4,
    multi-timeframe, news-sentiment, vol-estimators)
  - Qlib Alpha158 158 features: pure-pandas port of microsoft/qlib Alpha158DSL
    (kbar 9, price 4, rolling 30 operators x 5 windows = 150 + price)
    Source: qlib/contrib/data/loader.py Alpha158DL (MIT License)
    Module: scripts/qlib_alpha158_features.py

Total: ~870+ features. Same scout-prune-refit pattern as v5/v6/v7 (top-K=50).

DO NOT modify v7. This is a clean fork. Key differences from v7:
  1. build_v8_features() calls build_v7_features() then adds alpha158 layer.
  2. pipeline_version / strategy_variant strings updated to v8.
  3. Alpha158 feature importances tracked per fold.
"""
import argparse, json, os, sys, warnings
warnings.filterwarnings('ignore')
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import v7 feature builder (non-destructive -- v7.py is untouched)
from backtest_xgb_v7 import build_v7_features, numeric_cols
import backtest_ml as bml

# Cross-sectional cache (reuse from v7)
try:
    import cross_sectional_features as csf
except Exception as _e:
    print(f"[warn] csf: {_e}", file=sys.stderr)
    csf = None

# v8 NEW module: Qlib Alpha158 (pure pandas, no qlib import)
try:
    from qlib_alpha158_features import add_alpha158_features, alpha158_feature_names
    ALPHA158_AVAILABLE = True
except Exception as _e:
    print(f"[warn] qlib_alpha158_features: {_e}", file=sys.stderr)
    ALPHA158_AVAILABLE = False

# v8 patch: closeable gaps (yfinance fundamentals + FINRA short + analyst)
try:
    from closeable_gaps_features import add_closeable_gap_features, closeable_gap_feature_names
    CLOSEABLE_GAPS_AVAILABLE = True
except Exception as _e:
    print(f"[warn] closeable_gaps_features: {_e}", file=sys.stderr)
    CLOSEABLE_GAPS_AVAILABLE = False

# WORK is only used for the local cross-sectional cache lookup.
# On GH Actions it won't exist; the cache load will simply be skipped.
_WORK_DEFAULT = (
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools/s&p500-ticker-mastery"
)
WORK = Path(os.environ.get("BACKTEST_WORK_DIR", _WORK_DEFAULT))
LABEL_EMBARGO_DAYS = 21

# Known Alpha158 feature names for importance tracking
ALPHA158_FEAT_NAMES = alpha158_feature_names() if ALPHA158_AVAILABLE else []
CLOSEABLE_GAP_FEAT_NAMES = closeable_gap_feature_names() if CLOSEABLE_GAPS_AVAILABLE else []


def build_v8_features(ticker: str, universe_agg: dict = None) -> pd.DataFrame:
    """
    Build v8 feature set: full v7 stack + Qlib Alpha158 layer.
    universe_agg is the cross-sectional precomputed aggregates dict.
    """
    # Full v7 stack (unmodified)
    f = build_v7_features(ticker, universe_agg)
    print(f"  [v8] after v7 stack: {f.shape[1]} cols")

    # v8 NEW layer: Alpha158
    if ALPHA158_AVAILABLE:
        before = f.shape[1]
        try:
            f = add_alpha158_features(f)
            added = f.shape[1] - before
            print(f"    +alpha158 (v8): +{added} cols -> {f.shape[1]}")
        except Exception as e:
            print(f"    [w] alpha158: {e}")
    else:
        print("    [w] qlib_alpha158_features not available -- running v7 feature set only")

    # v8 patch: closeable gaps (fundamentals + short interest + analyst)
    if CLOSEABLE_GAPS_AVAILABLE:
        before = f.shape[1]
        try:
            f = add_closeable_gap_features(f, ticker)
            added = f.shape[1] - before
            print(f"    +closeable-gaps (v8 patch): +{added} cols -> {f.shape[1]}")
        except Exception as e:
            print(f"    [w] closeable-gaps: {e}")

    f = f.loc[:, ~f.columns.duplicated()]
    f = f.dropna(subset=['rsi_14', 'atr_14', 'ema_200', 'fwd_ret_21d', 'y'])
    return f


def main():
    ap = argparse.ArgumentParser(description="XGBoost v8 -- with Qlib Alpha158 features")
    ap.add_argument('--ticker', required=True)
    # Accept both --output-dir (local convention) and --out-dir (GH Actions workflow).
    ap.add_argument('--output-dir', '--out-dir', dest='output_dir', required=True)
    # --strategy and --job-id are passed by the workflow but not used by the model;
    # accept them so argparse does not error out.
    ap.add_argument('--strategy', default='')
    ap.add_argument('--job-id', default='')
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
    f = build_v8_features(args.ticker, universe_agg)
    fc = numeric_cols(f)
    print(f"  TOTAL features: {len(fc)}; rows: {len(f)}")

    # Walk-forward folds
    folds = bml.make_walk_forward_folds(f, train_months=24, test_months=12, step_months=12)
    print(f"  folds: {len(folds)}")
    all_probs = pd.Series(np.nan, index=f.index)
    fold_summaries = []
    fold_top_features = []
    fold_alpha158_importances = []  # track v8-specific feature importances

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

        # Track alpha158 new feature importances specifically
        imp_dict = dict(importances)
        a158_imps = {
            feat: float(imp_dict.get(feat, 0.0))
            for feat in ALPHA158_FEAT_NAMES
            if feat in fc and imp_dict.get(feat, 0.0) > 0
        }
        # Top 10 alpha158 features by importance this fold
        top_a158 = sorted(a158_imps.items(), key=lambda x: -x[1])[:10]
        a158_set = set(ALPHA158_FEAT_NAMES)
        fold_alpha158_importances.append({
            'fold': fold['fold'],
            'alpha158_in_top50': sum(1 for feat in top_features if feat in a158_set),
            'top_alpha158': dict(top_a158),
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
        'strategy': args.strategy,          # passed by GH Actions workflow
        'job_id': args.job_id,              # passed by GH Actions workflow
        'data_source': os.environ.get('BACKTEST_DATA_SOURCE', 'local_parquet'),
        'pipeline_version': 'xgb_v8_alpha158',
        'strategy_variant': 'ML_XGB_v8_alpha158',
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
            'qlib_alpha158_pandas_port': '158 (NEW v8)',
            'closeable_gaps_yfinance_finra': '18 (v8 patch)',
        },
        'alpha158_feature_count': len(ALPHA158_FEAT_NAMES),
        'alpha158_available': ALPHA158_AVAILABLE,
        'closeable_gap_feature_count': len(CLOSEABLE_GAP_FEAT_NAMES),
        'closeable_gaps_available': CLOSEABLE_GAPS_AVAILABLE,
        'fold_top_features': fold_top_features,
        'fold_alpha158_importances': fold_alpha158_importances,
        'walk_forward_folds': len(fold_summaries),
        'model_params': {
            'name': 'ML_XGB_v8_alpha158',
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

    # Also write result.json in the format the rollup job expects.
    # The workflow's inline Python already handles this if run_meta.json exists,
    # but writing it explicitly here ensures the dispatcher polling logic works.
    result = {
        'ticker': args.ticker,
        'strategy': args.strategy,
        'job_id': args.job_id,
        'cloud': 'github_actions',
        'status': 'completed',
        'returncode': 0,
        'completed_at': datetime.utcnow().isoformat() + 'Z',
        'result': {
            'n_trades': metrics['n_trades'],
            'win_rate': metrics.get('win_rate'),
            'profit_factor': metrics.get('profit_factor'),
            'total_return_pct': metrics.get('total_return_pct'),
            'max_drawdown_pct': metrics.get('max_drawdown_pct'),
            'prob_threshold': chosen_thr,
            'rows': len(f),
            'features_total': len(fc),
            'data_source': os.environ.get('BACKTEST_DATA_SOURCE', 'local_parquet'),
        },
    }
    result = to_py(result)
    with open(f"{args.output_dir}/result.json", 'w') as fp:
        json.dump(result, fp, indent=2, default=str)


if __name__ == '__main__':
    main()
