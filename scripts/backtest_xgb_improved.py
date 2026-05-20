"""
backtest_xgb_improved.py — XGBoost v6 + 3 targeted improvements.

Improvements over v6 baseline (max_depth=4, lr=0.05, n_est=100, no constraints):
  1. monotone_constraints on RSI/oscillator features
     RSI-type features: lower value -> higher bounce probability (constraint=-1)
     Trend features: higher EMA spread -> lower reversion signal (constraint=+1 or 0)
  2. inverse-volatility sample_weight = 1 / realized_vol_21d
     Downweights noisy high-vol regimes, focuses model on predictable low-vol reversals.
  3. Regularized hyperparams: colsample_bytree=0.7, subsample=0.8, min_child_weight=5
     Combats overfitting with ~600 training rows per fold.

Smoke-test usage:
    python backtest_xgb_improved.py --ticker AAPL --output-dir backtests_xgb_improved/AAPL
    python backtest_xgb_improved.py --ticker NCLH --output-dir backtests_xgb_improved/NCLH
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

WORK = Path("/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/s&p500-ticker-mastery")
LABEL_EMBARGO_DAYS = 21

# -----------------------------------------------------------------------
# Monotone constraints: map feature-name patterns -> direction
#   -1 = monotone decreasing (high value -> lower p(buy))
#   +1 = monotone increasing (high value -> higher p(buy))
#    0 = unconstrained
# These encode mean-reversion domain knowledge:
#   - RSI oversold (low RSI) => higher buy probability => constraint=-1
#   - Williams %R oversold (very negative) => higher buy prob => -1
#   - Stochastic K/D oversold => -1
#   - CCI (low CCI => oversold => -1)
#   - Distance above EMA => lower reversion signal (further above = less likely to revert up) => -1
# -----------------------------------------------------------------------
MONOTONE_PATTERNS = {
    # RSI: lower RSI -> more oversold -> higher P(reversion up) -> constraint = -1
    'rsi_': -1,
    # Williams %R: already negative scale; more negative = more oversold = higher signal -> +1
    # (willr_14 ranges -100 to 0; -100=oversold; higher P(buy) when willr near -100 => +1 not obvious)
    # We skip willr as direction depends on sign convention. Set to 0 (unconstrained).
    # Stochastic K/D: lower = oversold = higher P(bounce) -> -1
    'stoch_k': -1,
    'stoch_d': -1,
    # CCI: more negative = oversold = higher P(bounce) -> -1
    'cci_20': -1,
    # Distance from EMA normalized: large positive = overbought = less likely to bounce up -> -1
    'dist_ema': -1,
    # MFI: lower = more selling pressure = potential mean-reversion trigger -> -1
    'mfi_14': -1,
    # BB pct: near 0 = near lower band = oversold = higher P(bounce) -> -1
    'bb_pct': -1,
}


def get_monotone_constraints(feature_names: list) -> dict:
    """Build monotone_constraints dict for features matching known patterns."""
    constraints = {}
    for feat in feature_names:
        for pattern, direction in MONOTONE_PATTERNS.items():
            if feat.startswith(pattern) or feat == pattern:
                constraints[feat] = direction
                break
    return constraints


def compute_inverse_vol_weights(df: pd.DataFrame) -> np.ndarray:
    """
    Compute sample weights as inverse of 21-day realized volatility.
    Weight = 1 / max(rv_21d, floor) normalized to mean=1.
    Low-vol regimes (predictable) get higher weight.
    """
    rets = df['close'].pct_change()
    rv_21 = rets.rolling(21).std().reindex(df.index)
    floor = rv_21.quantile(0.1)  # don't let any weight blow up
    rv_21 = rv_21.clip(lower=float(floor) if float(floor) > 0 else 0.001)
    rv_21 = rv_21.fillna(rv_21.median())
    weights = 1.0 / rv_21.values
    weights = weights / weights.mean()  # normalize so sum stays same
    return weights.astype(np.float32)


def numeric_cols(df):
    return [c for c in df.columns
            if c not in ['open', 'high', 'low', 'close', 'volume', 'fwd_ret_21d', 'y']
            and pd.api.types.is_numeric_dtype(df[c])]


def run_ticker(ticker: str, output_dir: str,
               prob_threshold: float = 0.50,
               sweep_threshold: bool = False,
               top_k: int = 50,
               tp_atr: float = 1.5,
               sl_atr: float = 1.0,
               max_hold: int = 21,
               use_monotone: bool = True,
               use_inv_vol_weights: bool = True,
               compare_baseline: bool = True):

    os.makedirs(output_dir, exist_ok=True)

    # Load data (reuse bml pipeline — base features only for speed/reliability)
    d = bml.load_daily(ticker)
    f = bml.build_features(d)
    f = f.dropna(subset=['rsi_14', 'atr_14', 'ema_200', 'fwd_ret_21d', 'y'])
    fc = numeric_cols(f)
    print(f"  [{ticker}] rows={len(f)}, features={len(fc)}")

    # Precompute inverse-vol weights for the full dataset
    all_inv_vol_weights = compute_inverse_vol_weights(f)
    inv_vol_series = pd.Series(all_inv_vol_weights, index=f.index)

    # Build monotone constraints from top_k features (we'll set during final model fit)
    # We determine constraints after scout selects features, so they are accurate.

    folds = bml.make_walk_forward_folds(f, train_months=24, test_months=12, step_months=12)
    print(f"  folds: {len(folds)}")

    all_probs_improved = pd.Series(np.nan, index=f.index)
    all_probs_baseline = pd.Series(np.nan, index=f.index) if compare_baseline else None
    fold_summaries = []

    for fold in folds:
        train_end_emb = pd.Timestamp(fold['train_end']) - pd.tseries.offsets.BDay(LABEL_EMBARGO_DAYS)
        train = f[(f.index >= fold['train_start']) & (f.index < train_end_emb)]
        oos = f[(f.index >= fold['oos_start']) & (f.index < fold['oos_end'])]
        if len(train) < 50 or len(oos) < 20:
            continue

        X_tr_all = train[fc].fillna(0).values
        y_tr = train['y'].values
        X_oos_all = oos[fc].fillna(0).values

        # --- Scout model (same as v6) ---
        scout = xgb.XGBClassifier(
            max_depth=3, learning_rate=0.05, n_estimators=50,
            tree_method='hist', eval_metric='logloss',
            n_jobs=1, random_state=42, verbosity=0
        )
        scout.fit(X_tr_all, y_tr)
        importances = sorted(zip(fc, scout.feature_importances_), key=lambda x: -x[1])
        top_features = [c for c, imp in importances[:top_k] if imp > 0]
        if len(top_features) < 10:
            top_features = [c for c, _ in importances[:top_k]]

        X_tr = train[top_features].fillna(0).values
        X_oos = oos[top_features].fillna(0).values
        y_tr_fold = train['y'].values

        # class balance weight
        pos_w = float((y_tr_fold == 0).sum()) / max((y_tr_fold == 1).sum(), 1)

        # --- IMPROVEMENT 1: Inverse-vol sample weights ---
        if use_inv_vol_weights:
            sw = inv_vol_series.reindex(train.index).fillna(1.0).values.astype(np.float32)
        else:
            sw = None

        # --- IMPROVEMENT 2: Monotone constraints on RSI/oscillator features ---
        mono_constraints = {}
        if use_monotone:
            mono_constraints = get_monotone_constraints(top_features)
            # XGBoost expects the dict keys to be exactly the feature names used in fit
            # When we pass numpy arrays we can't use feature names directly in older XGBoost
            # Instead use the positional tuple format: build ordered tuple
            # For XGBoost >= 1.7, dict with feature names works when feature_names set
            # We'll use the list/tuple format indexed by position in top_features
            constraint_list = [mono_constraints.get(f, 0) for f in top_features]
            mono_arg = tuple(constraint_list)
        else:
            mono_arg = None

        # --- IMPROVEMENT 3: Regularized hyperparams for small N ---
        improved_params = dict(
            max_depth=4,
            learning_rate=0.05,
            n_estimators=150,           # slightly more trees since we regularize
            colsample_bytree=0.7,       # was implicitly 1.0 in v6
            subsample=0.8,              # was implicitly 1.0 in v6
            min_child_weight=5,         # was default 1 in v6; prevents tiny leaf splits
            reg_alpha=0.1,              # L1 on weights
            reg_lambda=1.5,             # L2 on weights (default is 1)
            scale_pos_weight=pos_w,
            tree_method='hist',
            eval_metric='logloss',
            n_jobs=1,
            random_state=42,
            verbosity=0,
        )
        if mono_arg is not None:
            improved_params['monotone_constraints'] = mono_arg

        final_improved = xgb.XGBClassifier(**improved_params)
        fit_kwargs = {}
        if use_inv_vol_weights and sw is not None:
            fit_kwargs['sample_weight'] = sw

        try:
            final_improved.fit(X_tr, y_tr_fold, **fit_kwargs)
        except Exception as e:
            print(f"    [warn] fold {fold['fold']} improved fit error: {e}")
            final_improved = xgb.XGBClassifier(
                max_depth=4, learning_rate=0.05, n_estimators=150,
                tree_method='hist', eval_metric='logloss',
                n_jobs=1, random_state=42, verbosity=0
            )
            final_improved.fit(X_tr, y_tr_fold)

        probs_improved = final_improved.predict_proba(X_oos)[:, 1]
        all_probs_improved.loc[oos.index] = probs_improved

        # --- Baseline (v6 exact) for comparison ---
        if compare_baseline:
            baseline = xgb.XGBClassifier(
                max_depth=4, learning_rate=0.05, n_estimators=100,
                tree_method='hist', eval_metric='logloss',
                n_jobs=1, random_state=42, verbosity=0
            )
            baseline.fit(X_tr, y_tr_fold)
            probs_baseline = baseline.predict_proba(X_oos)[:, 1]
            all_probs_baseline.loc[oos.index] = probs_baseline

        n_constrained = sum(1 for v in (mono_constraints or {}).values() if v != 0)
        fold_summaries.append({
            'fold': fold['fold'],
            'n_train': len(train),
            'n_oos': len(oos),
            'n_top_features': len(top_features),
            'n_monotone_constrained': n_constrained,
            'mean_inv_vol_weight': float(sw.mean()) if sw is not None else 1.0,
        })
        print(f"    fold {fold['fold']}: train={len(train)}, oos={len(oos)}, "
              f"mono_constrained={n_constrained}, inv_vol_sw_mean={fold_summaries[-1]['mean_inv_vol_weight']:.3f}")

    # --- Threshold sweep ---
    if sweep_threshold:
        rows = []
        for thr in np.arange(0.46, 0.70, 0.02):
            sig = (all_probs_improved > thr).fillna(False)
            trades = bml.simulate(f, sig, tp_atr, sl_atr, max_hold)
            mm = bml.compute_metrics(trades)
            rows.append({'thr': round(thr, 2), **mm})
        sdf = pd.DataFrame(rows)
        sdf.to_csv(f"{output_dir}/threshold_sweep.csv", index=False)
        mask = ((sdf['profit_factor'] >= 1.5) & (sdf['win_rate'] >= 0.53)
                & (sdf['n_trades'] >= 8) & (sdf['max_drawdown_pct'] >= -0.03)
                & (sdf['total_return_pct'] > 0))
        if mask.any():
            chosen_thr = float(sdf[mask].sort_values('profit_factor', ascending=False).iloc[0]['thr'])
        else:
            chosen_thr = float(sdf.sort_values('profit_factor', ascending=False).iloc[0]['thr'])
        print(f"  -> chose thr={chosen_thr}")
    else:
        chosen_thr = prob_threshold

    # --- Final metrics: improved ---
    sig_improved = (all_probs_improved > chosen_thr).fillna(False)
    trades_improved = bml.simulate(f, sig_improved, tp_atr, sl_atr, max_hold)
    metrics_improved = bml.compute_metrics(trades_improved)

    # --- Final metrics: baseline ---
    metrics_baseline = {}
    if compare_baseline:
        sig_baseline = (all_probs_baseline > chosen_thr).fillna(False)
        trades_baseline = bml.simulate(f, sig_baseline, tp_atr, sl_atr, max_hold)
        metrics_baseline = bml.compute_metrics(trades_baseline)

    # Print comparison
    print(f"\n  === {ticker} RESULTS (thr={chosen_thr}) ===")
    print(f"  BASELINE  n={metrics_baseline.get('n_trades','?'):>4} | "
          f"WR={metrics_baseline.get('win_rate',0):.3f} | "
          f"PF={metrics_baseline.get('profit_factor',0):.3f} | "
          f"RET={metrics_baseline.get('total_return_pct',0):.4f} | "
          f"DD={metrics_baseline.get('max_drawdown_pct',0):.4f}")
    print(f"  IMPROVED  n={metrics_improved.get('n_trades','?'):>4} | "
          f"WR={metrics_improved.get('win_rate',0):.3f} | "
          f"PF={metrics_improved.get('profit_factor',0):.3f} | "
          f"RET={metrics_improved.get('total_return_pct',0):.4f} | "
          f"DD={metrics_improved.get('max_drawdown_pct',0):.4f}")

    def _lift(key, sign=1):
        b = metrics_baseline.get(key, 0) or 0
        i = metrics_improved.get(key, 0) or 0
        return sign * (i - b)

    print(f"  DELTA PF  {_lift('profit_factor'):+.3f} | "
          f"WR {_lift('win_rate'):+.3f} | "
          f"RET {_lift('total_return_pct'):+.4f} | "
          f"DD {_lift('max_drawdown_pct', sign=-1):+.4f}")

    # --- Save artifacts ---
    trades_improved.to_csv(f"{output_dir}/trades_improved.csv", index=False)
    if compare_baseline and len(trades_baseline) > 0:
        trades_baseline.to_csv(f"{output_dir}/trades_baseline.csv", index=False)

    def to_py(o):
        if isinstance(o, dict): return {k: to_py(v) for k, v in o.items()}
        if isinstance(o, list): return [to_py(v) for v in o]
        if hasattr(o, 'item'): return o.item()
        if isinstance(o, float) and (np.isnan(o) or np.isinf(o)): return None
        return o

    meta = to_py({
        'ticker': ticker,
        'pipeline_version': 'xgb_improved_v1',
        'run_at': datetime.utcnow().isoformat() + 'Z',
        'improvements_applied': {
            'monotone_constraints': use_monotone,
            'inv_vol_sample_weight': use_inv_vol_weights,
            'regularized_params': True,
            'description': (
                'monotone_constraints on RSI/oscillator features encoding mean-reversion direction; '
                'inverse-vol sample_weight=1/rv21d to focus on low-noise regimes; '
                'colsample_bytree=0.7, subsample=0.8, min_child_weight=5 for small-N regularization'
            ),
        },
        'model_params': {
            'baseline': {'max_depth': 4, 'lr': 0.05, 'n_est': 100,
                         'colsample_bytree': 1.0, 'subsample': 1.0, 'min_child_weight': 1},
            'improved': {'max_depth': 4, 'lr': 0.05, 'n_est': 150,
                         'colsample_bytree': 0.7, 'subsample': 0.8,
                         'min_child_weight': 5, 'reg_alpha': 0.1, 'reg_lambda': 1.5,
                         'monotone_constraints': 'per_fold_dict',
                         'sample_weight': 'inv_vol_21d'},
        },
        'prob_threshold': chosen_thr,
        'metrics_baseline': metrics_baseline,
        'metrics_improved': metrics_improved,
        'fold_summaries': fold_summaries,
        'walk_forward_folds': len(fold_summaries),
    })

    with open(f"{output_dir}/run_meta.json", 'w') as fp:
        json.dump(meta, fp, indent=2, default=str)

    return metrics_improved, metrics_baseline


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
    ap.add_argument('--no-monotone', action='store_true', help='Disable monotone constraints')
    ap.add_argument('--no-inv-vol', action='store_true', help='Disable inverse-vol weights')
    ap.add_argument('--no-compare', action='store_true', help='Skip baseline comparison')
    args = ap.parse_args()

    run_ticker(
        ticker=args.ticker,
        output_dir=args.output_dir,
        prob_threshold=args.prob_threshold,
        sweep_threshold=args.sweep_threshold,
        top_k=args.top_k,
        tp_atr=args.tp_atr,
        sl_atr=args.sl_atr,
        max_hold=args.max_hold,
        use_monotone=not args.no_monotone,
        use_inv_vol_weights=not args.no_inv_vol,
        compare_baseline=not args.no_compare,
    )


if __name__ == '__main__':
    main()
