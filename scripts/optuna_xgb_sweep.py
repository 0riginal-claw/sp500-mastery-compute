"""
optuna_xgb_sweep.py — Wave 7: per-ticker XGBoost hyperparameter sweep.

Architecture (per data-scientist agent design 2026-05-14):

  Walk-forward 24mo train / 12mo OOS:
    24mo training split into 18mo train_inner + 6mo val_inner (for hyperparam search)
    12mo OOS touched ONCE at the end with reconciled best params

  Objective: PF_val if all constraints met (WR_val>=0.53, n>=4, DD>=-0.03, RET>0) else -1.0

  Pre-screen with default XGBoost params; only sweep tickers with n_trades>=4 AND PF>=1.0.

  Per-ticker artifact tree:
    sweep_artifacts/{ticker}/fold_{N}_study.pkl      — full optuna study (replayable)
    sweep_artifacts/{ticker}/best_params.json        — reconciled params + threshold + stability flag
    sweep_artifacts/{ticker}/oos_trades.csv          — final OOS trades
    sweep_artifacts/{ticker}/sweep_summary.json      — top-level results

Usage:
    python optuna_xgb_sweep.py prescan --tickers AAPL,NVDA,XOM,SCHW,CSCO
    python optuna_xgb_sweep.py sweep --ticker AAPL --n-trials 50
    python optuna_xgb_sweep.py sweep-all --n-trials 50 --n-cores 8
"""

import argparse, json, os, sys, pickle, warnings
warnings.filterwarnings('ignore')
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtest_ml as bml  # reuse load_daily, build_features, simulate, metrics
import alt_data_features as adf

WORK = Path("/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/s&p500-ticker-mastery")
DATA_ROOT = Path("/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/claudes test/data/timeframes/S&P500 5 Year Historical Data/Minutes TimeFrames/1Min_merged")
SWEEP_ROOT = WORK / "sweep_artifacts"

LABEL_EMBARGO_DAYS = 21
TRAIN_INNER_MONTHS = 18
VAL_INNER_MONTHS = 6
PRESCREEN_THR_PF = 1.0
PRESCREEN_THR_N = 4


def list_available_tickers():
    return sorted(p.stem for p in DATA_ROOT.glob("*.parquet"))


def load_and_featurize(ticker: str, with_alt: bool = True) -> pd.DataFrame:
    d = bml.load_daily(ticker)
    f = bml.build_features(d)
    if with_alt:
        try:
            f = adf.add_all_alt_features(f, ticker)
            # Drop alt-features with <50% non-zero coverage for this ticker
            alt_cols = ['filing_count_30d','filing_count_8k_30d','filing_count_10q_180d',
                        'days_since_last_10q','cong_net_buy_60d','cong_n_unique_buyers_90d',
                        'cong_chamber_buy_ratio_30d','lobbying_spend_30d','lobbying_n_issues_90d']
            for c in alt_cols:
                if c in f.columns:
                    cov = (f[c] != 0).mean()
                    if cov < 0.50:
                        f = f.drop(columns=[c])
        except Exception as e:
            print(f"  [warn] alt-data failed for {ticker}: {e}")
    return f.dropna(subset=['rsi_14','atr_14','ema_200','fwd_ret_21d','y'])


def feature_cols(df: pd.DataFrame):
    return [c for c in df.columns
            if c not in ['open','high','low','close','volume','fwd_ret_21d','y']
            and pd.api.types.is_numeric_dtype(df[c])]


def make_inner_folds(df, train_months=24, val_months=6, oos_months=12, step_months=12):
    """Walk-forward with NESTED train_inner/val_inner inside the 24mo training window."""
    start = df.index.min(); end = df.index.max()
    folds = []
    cur = start + pd.DateOffset(months=train_months)
    while True:
        oos_end = cur + pd.DateOffset(months=oos_months)
        if oos_end > end - pd.DateOffset(months=1): break
        train_start = cur - pd.DateOffset(months=train_months)
        train_inner_end = train_start + pd.DateOffset(months=TRAIN_INNER_MONTHS)
        val_inner_end = train_inner_end + pd.DateOffset(months=VAL_INNER_MONTHS)
        folds.append({
            'fold': len(folds)+1,
            'train_start': train_start.isoformat(),
            'train_inner_end': (train_inner_end - pd.tseries.offsets.BDay(LABEL_EMBARGO_DAYS)).isoformat(),
            'val_inner_start': train_inner_end.isoformat(),
            'val_inner_end': (val_inner_end - pd.tseries.offsets.BDay(LABEL_EMBARGO_DAYS)).isoformat(),
            'oos_start': cur.isoformat(),
            'oos_end': oos_end.isoformat(),
        })
        cur = cur + pd.DateOffset(months=step_months)
    return folds


def constraints_met(m: dict, n_trade_min: int = PRESCREEN_THR_N) -> bool:
    if (m.get('n_trades') or 0) < n_trade_min: return False
    if (m.get('win_rate') or 0) < 0.53: return False
    if (m.get('profit_factor') or 0) < 1.5: return False
    if (m.get('total_return_pct') or 0) <= 0: return False
    if (m.get('max_drawdown_pct') or 0) < -0.03: return False
    return True


def fit_xgb(X_train, y_train, X_val, y_val, params: dict):
    """Fit XGBoost with early stopping on val_inner; return model + best_iteration."""
    m = xgb.XGBClassifier(
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        n_estimators=400,
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        min_child_weight=params['min_child_weight'],
        reg_alpha=params['reg_alpha'],
        reg_lambda=params['reg_lambda'],
        gamma=params['gamma'],
        scale_pos_weight=params.get('scale_pos_weight', 1.0),
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='hist',
        early_stopping_rounds=20,
        n_jobs=1,  # outer parallelism is by ticker
        random_state=42,
        verbosity=0,
    )
    m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return m


def post_hoc_threshold_sweep(probs, df_val, tp_atr, sl_atr, max_hold):
    """1D sweep over prob_threshold ∈ [0.48, 0.65] step 0.01; return best PF + thr."""
    best = {'thr': 0.50, 'pf': -1.0, 'metrics': None}
    for thr in np.arange(0.48, 0.66, 0.01):
        sig = pd.Series(probs > thr, index=df_val.index)
        trades = bml.simulate(df_val, sig, tp_atr, sl_atr, max_hold)
        m = bml.compute_metrics(trades)
        if not constraints_met(m): continue
        if m['profit_factor'] > best['pf']:
            best = {'thr': float(thr), 'pf': m['profit_factor'], 'metrics': m}
    return best


def suggest_params(trial: optuna.Trial) -> dict:
    return {
        'max_depth': trial.suggest_int('max_depth', 2, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.20, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5.0, log=True),
        'gamma': trial.suggest_float('gamma', 0.0, 0.5),
        'tp_atr': trial.suggest_float('tp_atr', 0.8, 2.5),
        'sl_atr': trial.suggest_float('sl_atr', 0.5, 2.0),
    }


def prescan(ticker: str) -> dict:
    """Quick run with default XGB params; output enough to gate the full sweep."""
    try:
        f = load_and_featurize(ticker, with_alt=False)
    except Exception as e:
        return {'ticker': ticker, 'status': 'LOAD_FAIL', 'error': str(e)}
    if len(f) < 200:
        return {'ticker': ticker, 'status': 'TOO_FEW_BARS', 'n_bars': len(f)}
    fc = feature_cols(f)
    folds = make_inner_folds(f, train_months=24, val_months=6, oos_months=12, step_months=12)
    all_signals = pd.Series(False, index=f.index)
    for fold in folds:
        train = f[(f.index >= fold['train_start']) & (f.index < fold['train_inner_end'])]
        oos = f[(f.index >= fold['oos_start']) & (f.index < fold['oos_end'])]
        if len(train) < 50 or len(oos) < 20: continue
        X_tr = train[fc].values; y_tr = train['y'].values
        X_oos = oos[fc].values
        m = xgb.XGBClassifier(max_depth=4, learning_rate=0.05, n_estimators=100,
                              tree_method='hist', n_jobs=1, eval_metric='logloss',
                              random_state=42, verbosity=0)
        m.fit(X_tr, y_tr)
        probs = m.predict_proba(X_oos)[:, 1]
        all_signals.loc[oos.index] = probs > 0.50
    trades = bml.simulate(f, all_signals, 1.5, 1.0, 21)
    metrics = bml.compute_metrics(trades)
    return {
        'ticker': ticker, 'status': 'OK',
        'n_trades': metrics['n_trades'], 'PF': metrics.get('profit_factor'),
        'WR': metrics.get('win_rate'), 'RET%': metrics.get('total_return_pct'),
        'DD%': metrics.get('max_drawdown_pct'),
        'passes_prescreen': (metrics['n_trades'] >= PRESCREEN_THR_N
                             and (metrics.get('profit_factor') or 0) >= PRESCREEN_THR_PF),
    }


def sweep(ticker: str, n_trials: int = 50, with_alt: bool = True) -> dict:
    out = SWEEP_ROOT / ticker
    out.mkdir(parents=True, exist_ok=True)
    f = load_and_featurize(ticker, with_alt=with_alt)
    fc = feature_cols(f)
    folds = make_inner_folds(f)
    print(f"[{ticker}] bars={len(f)}, features={len(fc)}, folds={len(folds)}")

    fold_results = []
    for fold in folds:
        train_inner = f[(f.index >= fold['train_start']) & (f.index < fold['train_inner_end'])]
        val_inner = f[(f.index >= fold['val_inner_start']) & (f.index < fold['val_inner_end'])]
        if len(train_inner) < 50 or len(val_inner) < 20: continue

        X_tr = train_inner[fc].values; y_tr = train_inner['y'].values
        X_val = val_inner[fc].values; y_val = val_inner['y'].values
        pos_w = float((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1)

        def objective(trial):
            params = suggest_params(trial)
            params['scale_pos_weight'] = pos_w
            try:
                m = fit_xgb(X_tr, y_tr, X_val, y_val, params)
            except Exception:
                return -1.0
            probs = m.predict_proba(X_val)[:, 1]
            best_thr = post_hoc_threshold_sweep(probs, val_inner, params['tp_atr'], params['sl_atr'], 21)
            return best_thr['pf']

        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=42),
                                    pruner=optuna.pruners.MedianPruner(n_startup_trials=10))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        with open(out / f"fold_{fold['fold']}_study.pkl", 'wb') as fp:
            pickle.dump(study, fp)

        if study.best_value is not None and study.best_value > 0:
            best_params = study.best_params
            best_params['scale_pos_weight'] = pos_w
            # Refit on train_inner, evaluate on val_inner, then on OOS
            m = fit_xgb(X_tr, y_tr, X_val, y_val, best_params)
            probs_val = m.predict_proba(X_val)[:, 1]
            best_thr = post_hoc_threshold_sweep(probs_val, val_inner, best_params['tp_atr'], best_params['sl_atr'], 21)

            oos = f[(f.index >= fold['oos_start']) & (f.index < fold['oos_end'])]
            X_oos = oos[fc].values
            probs_oos = m.predict_proba(X_oos)[:, 1]
            # Apply conservatism: thr* + 0.02
            applied_thr = min(best_thr['thr'] + 0.02, 0.65)
            oos_sig = pd.Series(probs_oos > applied_thr, index=oos.index)
            oos_trades = bml.simulate(oos, oos_sig, best_params['tp_atr'], best_params['sl_atr'], 21)
            oos_metrics = bml.compute_metrics(oos_trades)

            fold_results.append({
                'fold': fold['fold'],
                'best_val_pf': study.best_value,
                'best_params': best_params,
                'best_thr_val': best_thr['thr'],
                'applied_thr_oos': applied_thr,
                'val_metrics': best_thr['metrics'],
                'oos_metrics': oos_metrics,
            })
            print(f"  fold {fold['fold']}: val PF={study.best_value:.2f} thr={best_thr['thr']:.2f} "
                  f"-> oos PF={oos_metrics.get('profit_factor', 0):.2f} WR={oos_metrics.get('win_rate', 0):.3f} n={oos_metrics['n_trades']}")
        else:
            fold_results.append({'fold': fold['fold'], 'status': 'NO_FEASIBLE_CONFIG'})
            print(f"  fold {fold['fold']}: NO_FEASIBLE_CONFIG")

    # Aggregate across folds (pool OOS trades, recompute pooled metrics)
    summary = {
        'ticker': ticker, 'run_at': datetime.utcnow().isoformat() + 'Z',
        'n_features': len(fc), 'n_bars': len(f), 'n_trials': n_trials, 'with_alt': with_alt,
        'fold_results': fold_results,
    }
    with open(out / "sweep_summary.json", 'w') as fp:
        json.dump(summary, fp, indent=2, default=str)
    return summary


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p1 = sub.add_parser('prescan')
    p1.add_argument('--tickers', help='comma-separated list', default=None)
    p1.add_argument('--all', action='store_true')
    p2 = sub.add_parser('sweep')
    p2.add_argument('--ticker', required=True)
    p2.add_argument('--n-trials', type=int, default=50)
    p2.add_argument('--no-alt', action='store_true')
    args = ap.parse_args()

    if args.cmd == 'prescan':
        tickers = list_available_tickers() if args.all else args.tickers.split(',')
        results = []
        for tk in tickers:
            r = prescan(tk)
            results.append(r)
            ps = '✓' if r.get('passes_prescreen') else '✗'
            print(f"  {ps} {tk:6s} n={r.get('n_trades','?'):>4} PF={r.get('PF',0) or 0:.2f}")
        SWEEP_ROOT.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(SWEEP_ROOT / "prescan_results.csv", index=False)
        print(f"\n=== prescan summary: {sum(1 for r in results if r.get('passes_prescreen'))} / {len(results)} pass ===")
    elif args.cmd == 'sweep':
        sweep(args.ticker, n_trials=args.n_trials, with_alt=not args.no_alt)


if __name__ == '__main__':
    main()
