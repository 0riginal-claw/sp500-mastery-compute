"""
backtest_xgb_v4.py — MAXIMAL XGBoost pipeline.

ALL feature layers stacked:
  - Base 46 indicators (build_features)
  - 22 intraday features
  - 9 alt-data features (EDGAR + GovTrades + Lobbying, /tmp-copied)
  - 17 cross-sectional features (cached or skipped if cache absent)
  - Trading Insight Info parts 1-4 (~496 features: 116+141+151+88)

Total expected feature count: ~600+

Plus per-ticker prob_threshold sweep.
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
try:
    import cross_sectional_features as csf
except Exception:
    csf = None

# Trading Insight Info parts 1-4
try:
    from trading_insight_features_part1 import add_features_part1
except Exception as e:
    print(f"[warn] part1 import failed: {e}", file=sys.stderr)
    add_features_part1 = None
try:
    from trading_insight_features_part2 import add_features_part2
except Exception as e:
    print(f"[warn] part2 import failed: {e}", file=sys.stderr)
    add_features_part2 = None
try:
    from trading_insight_features_part3 import add_features_part3
except Exception as e:
    print(f"[warn] part3 import failed: {e}", file=sys.stderr)
    add_features_part3 = None
try:
    from trading_insight_features_part4 import add_features_part4
except Exception as e:
    print(f"[warn] part4 import failed: {e}", file=sys.stderr)
    add_features_part4 = None

# Also include v3's hand-coded trading-insight features (the original 13)
from backtest_xgb_v3 import add_trading_insight_features

WORK = Path("/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/s&p500-ticker-mastery")
LABEL_EMBARGO_DAYS = 21


def build_v4_features(ticker: str, universe_agg: dict = None) -> pd.DataFrame:
    d = bml.load_daily(ticker)
    print(f"  [v4 build] base daily bars: {len(d):,}")
    f = bml.build_features(d)
    print(f"    after build_features: {f.shape[1]}")
    try:
        f = idf.add_intraday_features(f, ticker)
        print(f"    after intraday: {f.shape[1]}")
    except Exception as e: print(f"    [warn] intraday: {e}")
    try:
        f = adf.add_all_alt_features(f, ticker)
        for c in list(f.columns):
            if c.startswith(('cong_', 'lobbying_', 'filing_', 'days_since_')):
                if (f[c] != 0).mean() < 0.10: f = f.drop(columns=[c])
        print(f"    after alt-data (pruned): {f.shape[1]}")
    except Exception as e: print(f"    [warn] alt-data: {e}")
    if universe_agg is not None and csf is not None:
        try:
            f = csf.add_cross_sectional_features(f, ticker, universe_agg)
            print(f"    after cross-sectional: {f.shape[1]}")
        except Exception as e: print(f"    [warn] cross-sectional: {e}")
    try:
        f = add_trading_insight_features(f)
        print(f"    after trading-insight v3: {f.shape[1]}")
    except Exception as e: print(f"    [warn] trading-insight v3: {e}")
    # Parts 1-4
    for name, fn in [('part1', add_features_part1), ('part2', add_features_part2),
                     ('part3', add_features_part3), ('part4', add_features_part4)]:
        if fn is None:
            print(f"    [skip] trading-insight {name}: not imported")
            continue
        try:
            f = fn(f)
            print(f"    after trading-insight {name}: {f.shape[1]}")
        except Exception as e:
            print(f"    [warn] trading-insight {name}: {type(e).__name__}: {str(e)[:200]}")
    # Drop dupe columns just in case
    f = f.loc[:, ~f.columns.duplicated()]
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

    print(f"[{args.ticker}] building v4 unified features (all 496+ trading-insight features)...")
    cache_path = WORK / "cache" / "universe_agg.parquet"
    universe_agg = None
    if cache_path.exists() and csf is not None:
        try: universe_agg = csf.precompute_universe_aggregates()
        except Exception as e: print(f"  cache load failed: {e}")
    f = build_v4_features(args.ticker, universe_agg)
    fc = numeric_feature_cols(f)
    print(f"  TOTAL features: {len(fc)}; rows: {len(f)}")

    folds = bml.make_walk_forward_folds(f, train_months=24, test_months=12, step_months=12)
    print(f"  folds: {len(folds)}")
    all_probs = pd.Series(np.nan, index=f.index)
    fold_summaries = []
    for fold in folds:
        train_end_emb = pd.Timestamp(fold['train_end']) - pd.tseries.offsets.BDay(LABEL_EMBARGO_DAYS)
        train = f[(f.index >= fold['train_start']) & (f.index < train_end_emb)]
        oos = f[(f.index >= fold['oos_start']) & (f.index < fold['oos_end'])]
        if len(train) < 50 or len(oos) < 20: continue
        X_tr = train[fc].fillna(0).values; y_tr = train['y'].values
        X_oos = oos[fc].fillna(0).values
        m = xgb.XGBClassifier(max_depth=4, learning_rate=0.05, n_estimators=100,
                              tree_method='hist', eval_metric='logloss',
                              n_jobs=1, random_state=42, verbosity=0)
        m.fit(X_tr, y_tr)
        probs = m.predict_proba(X_oos)[:, 1]
        all_probs.loc[oos.index] = probs
        fold_summaries.append({'fold': fold['fold'], 'oos_start': fold['oos_start'],
                               'n_train': len(train), 'n_oos': len(oos),
                               'mean_oos_prob': float(probs.mean())})

    if args.sweep_threshold:
        sweep_rows = []
        for thr in np.arange(0.46, 0.70, 0.02):
            sig = all_probs > thr
            trades = bml.simulate(f, sig.fillna(False), args.tp_atr, args.sl_atr, args.max_hold)
            mm = bml.compute_metrics(trades)
            sweep_rows.append({'thr': round(thr, 2), **mm})
        sweep_df = pd.DataFrame(sweep_rows)
        sweep_df.to_csv(f"{args.output_dir}/threshold_sweep.csv", index=False)
        mask = ((sweep_df['profit_factor'] >= 1.5) & (sweep_df['win_rate'] >= 0.53)
                & (sweep_df['n_trades'] >= 8) & (sweep_df['max_drawdown_pct'] >= -0.03)
                & (sweep_df['total_return_pct'] > 0))
        if mask.any():
            best = sweep_df[mask].sort_values('profit_factor', ascending=False).iloc[0]
            chosen_thr = float(best['thr'])
        else:
            chosen_thr = float(sweep_df.sort_values('profit_factor', ascending=False).iloc[0]['thr'])
        print(f"  → sweep chose thr={chosen_thr}")
    else:
        chosen_thr = args.prob_threshold

    final_sig = (all_probs > chosen_thr).fillna(False)
    trades = bml.simulate(f, final_sig, args.tp_atr, args.sl_atr, args.max_hold)
    metrics = bml.compute_metrics(trades)
    print(f"  final thr={chosen_thr}: n={metrics['n_trades']}, WR={metrics.get('win_rate', 0):.3f}, PF={metrics.get('profit_factor', 0):.3f}, RET={metrics.get('total_return_pct', 0):.4f}, DD={metrics.get('max_drawdown_pct', 0):.4f}")
    trades.to_csv(f"{args.output_dir}/trades.csv", index=False)

    def to_py(o):
        if isinstance(o, dict): return {k: to_py(v) for k, v in o.items()}
        if isinstance(o, list): return [to_py(v) for v in o]
        if hasattr(o, 'item'): return o.item()
        if isinstance(o, float) and (np.isnan(o) or np.isinf(o)): return None
        return o

    meta = to_py({
        'ticker': args.ticker, 'pipeline_version': 'xgb_v4_maximal',
        'strategy_variant': 'ML_XGB_v4_maximal',
        'run_at': datetime.utcnow().isoformat() + 'Z',
        'features_used': fc, 'n_features': len(fc),
        'feature_sources': {
            'base_indicators': '~46',
            'intraday': '22',
            'alt_data_pruned': 'variable',
            'cross_sectional': '17 if cache else skipped',
            'trading_insight_v3': '13',
            'trading_insight_part1': '116',
            'trading_insight_part2': '141',
            'trading_insight_part3': '151',
            'trading_insight_part4': '88',
        },
        'daily_bars': len(f), 'walk_forward_folds': len(fold_summaries),
        'strategy': {
            'name': 'ML_XGB_v4_maximal', 'side': 'long',
            'tp_atr': args.tp_atr, 'sl_atr': args.sl_atr, 'max_hold_days': args.max_hold,
            'prob_threshold': chosen_thr, 'threshold_swept': args.sweep_threshold,
            'model': 'XGBClassifier (defaults)', 'calibration': 'native predict_proba',
            'slippage_bps': 5.0, 'fee_per_share': 0.0035, 'notional_per_trade': 5000,
        },
        'metrics_oos_aggregate': metrics, 'fold_summaries': fold_summaries,
    })
    with open(f"{args.output_dir}/run_meta.json", 'w') as fp:
        json.dump(meta, fp, indent=2, default=str)


if __name__ == '__main__':
    main()
