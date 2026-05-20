"""
backtest_xgb_v5_pruned.py — XGBoost pipeline with per-fold feature pruning.

Builds the same 587-feature set as v4, then inside each WF fold:
  1. Fits a cheap scout model (max_depth=3, n_estimators=50) on train_inner.
  2. Ranks features by gain importance, keeps top-K (default 50).
  3. Refits the production model on only those top-K features.

This recovers signal density lost when going from v3 (99 features) to v4 (587 features).

Outputs: backtests_xgb_v5/{ticker}_v5/run_meta.json  (includes top_k_features list)
"""
import argparse
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtest_ml as bml
import alt_data_features as adf
import intraday_features as idf

try:
    import cross_sectional_features as csf
except Exception:
    csf = None

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

from backtest_xgb_v3 import add_trading_insight_features

WORK = Path("/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/s&p500-ticker-mastery")
LABEL_EMBARGO_DAYS = 21

# --- Scout hyper-params (cheap) ---
SCOUT_MAX_DEPTH = 3
SCOUT_N_EST = 50

# --- Production model hyper-params (same as v4) ---
PROD_MAX_DEPTH = 4
PROD_LEARNING_RATE = 0.05
PROD_N_EST = 100

# --- Default top-K to keep ---
DEFAULT_TOP_K = 50


# ---------------------------------------------------------------------------
# Feature construction — identical to v4 build_v4_features
# ---------------------------------------------------------------------------

def build_v4_features(ticker: str, universe_agg: dict = None) -> pd.DataFrame:
    d = bml.load_daily(ticker)
    print(f"  [v5 build] base daily bars: {len(d):,}")
    f = bml.build_features(d)
    print(f"    after build_features: {f.shape[1]}")
    try:
        f = idf.add_intraday_features(f, ticker)
        print(f"    after intraday: {f.shape[1]}")
    except Exception as e:
        print(f"    [warn] intraday: {e}")
    try:
        f = adf.add_all_alt_features(f, ticker)
        for c in list(f.columns):
            if c.startswith(("cong_", "lobbying_", "filing_", "days_since_")):
                if (f[c] != 0).mean() < 0.10:
                    f = f.drop(columns=[c])
        print(f"    after alt-data (pruned): {f.shape[1]}")
    except Exception as e:
        print(f"    [warn] alt-data: {e}")
    if universe_agg is not None and csf is not None:
        try:
            f = csf.add_cross_sectional_features(f, ticker, universe_agg)
            print(f"    after cross-sectional: {f.shape[1]}")
        except Exception as e:
            print(f"    [warn] cross-sectional: {e}")
    try:
        f = add_trading_insight_features(f)
        print(f"    after trading-insight v3: {f.shape[1]}")
    except Exception as e:
        print(f"    [warn] trading-insight v3: {e}")
    for name, fn in [
        ("part1", add_features_part1),
        ("part2", add_features_part2),
        ("part3", add_features_part3),
        ("part4", add_features_part4),
    ]:
        if fn is None:
            print(f"    [skip] trading-insight {name}: not imported")
            continue
        try:
            f = fn(f)
            print(f"    after trading-insight {name}: {f.shape[1]}")
        except Exception as e:
            print(f"    [warn] trading-insight {name}: {type(e).__name__}: {str(e)[:200]}")
    f = f.loc[:, ~f.columns.duplicated()]
    f = f.dropna(subset=["rsi_14", "atr_14", "ema_200", "fwd_ret_21d", "y"])
    return f


def numeric_feature_cols(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c not in ["open", "high", "low", "close", "volume", "fwd_ret_21d", "y"]
        and pd.api.types.is_numeric_dtype(df[c])
    ]


# ---------------------------------------------------------------------------
# Per-fold feature selection via scout model
# ---------------------------------------------------------------------------

def select_top_k_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    top_k: int,
) -> list[str]:
    """Fit a cheap scout XGBoost and return the top-K features by gain importance."""
    scout = xgb.XGBClassifier(
        max_depth=SCOUT_MAX_DEPTH,
        n_estimators=SCOUT_N_EST,
        tree_method="hist",
        eval_metric="logloss",
        n_jobs=1,
        random_state=42,
        verbosity=0,
    )
    scout.fit(X_train, y_train)
    importances = scout.feature_importances_  # gain by default in XGBClassifier
    ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    top_names = [name for name, _ in ranked[:top_k]]
    return top_names


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--prob-threshold", type=float, default=0.50)
    ap.add_argument("--sweep-threshold", action="store_true")
    ap.add_argument("--tp-atr", type=float, default=1.5)
    ap.add_argument("--sl-atr", type=float, default=1.0)
    ap.add_argument("--max-hold", type=int, default=21)
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                    help="Number of top features to keep per fold (default: 50)")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[{args.ticker}] building v5 features (v4 feature set, top-K={args.top_k} pruning)...")
    universe_agg = None
    cache_path = WORK / "cache" / "universe_agg.parquet"
    if cache_path.exists() and csf is not None:
        try:
            universe_agg = csf.precompute_universe_aggregates()
        except Exception as e:
            print(f"  cache load failed: {e}")

    f = build_v4_features(args.ticker, universe_agg)
    all_fc = numeric_feature_cols(f)
    print(f"  TOTAL features before pruning: {len(all_fc)}; rows: {len(f)}")

    folds = bml.make_walk_forward_folds(f, train_months=24, test_months=12, step_months=12)
    print(f"  folds: {len(folds)}")

    all_probs = pd.Series(np.nan, index=f.index)
    fold_summaries = []
    fold_top_features: dict[int, list[str]] = {}

    for fold in folds:
        train_end_emb = pd.Timestamp(fold["train_end"]) - pd.tseries.offsets.BDay(LABEL_EMBARGO_DAYS)
        train = f[(f.index >= fold["train_start"]) & (f.index < train_end_emb)]
        oos = f[(f.index >= fold["oos_start"]) & (f.index < fold["oos_end"])]
        if len(train) < 50 or len(oos) < 20:
            print(f"  fold {fold['fold']}: skipped (train={len(train)}, oos={len(oos)})")
            continue

        X_tr_full = train[all_fc].fillna(0).values
        y_tr = train["y"].values

        # --- Step 1: scout → select top-K features ---
        top_k_actual = min(args.top_k, len(all_fc))
        top_features = select_top_k_features(X_tr_full, y_tr, all_fc, top_k_actual)
        fold_top_features[fold["fold"]] = top_features

        # --- Step 2: refit production model on top-K only ---
        X_tr_pruned = train[top_features].fillna(0).values
        X_oos_pruned = oos[top_features].fillna(0).values

        prod = xgb.XGBClassifier(
            max_depth=PROD_MAX_DEPTH,
            learning_rate=PROD_LEARNING_RATE,
            n_estimators=PROD_N_EST,
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=1,
            random_state=42,
            verbosity=0,
        )
        prod.fit(X_tr_pruned, y_tr)
        probs = prod.predict_proba(X_oos_pruned)[:, 1]
        all_probs.loc[oos.index] = probs

        fold_summaries.append({
            "fold": fold["fold"],
            "oos_start": fold["oos_start"],
            "n_train": len(train),
            "n_oos": len(oos),
            "mean_oos_prob": float(probs.mean()),
            "top_features_used": top_features,
        })
        print(
            f"  fold {fold['fold']}: train={len(train)}, oos={len(oos)}, "
            f"top_k_used={len(top_features)}, mean_prob={probs.mean():.3f}"
        )

    # Aggregate top features across all folds (frequency count)
    from collections import Counter
    feature_vote_counter: Counter = Counter()
    for features_list in fold_top_features.values():
        feature_vote_counter.update(features_list)
    # top_k_features: features that appeared in the most folds
    top_k_features_global = [f_name for f_name, _ in feature_vote_counter.most_common(args.top_k)]

    # --- Threshold sweep ---
    if args.sweep_threshold:
        sweep_rows = []
        for thr in np.arange(0.46, 0.70, 0.02):
            sig = all_probs > thr
            trades = bml.simulate(f, sig.fillna(False), args.tp_atr, args.sl_atr, args.max_hold)
            mm = bml.compute_metrics(trades)
            sweep_rows.append({"thr": round(thr, 2), **mm})
        sweep_df = pd.DataFrame(sweep_rows)
        sweep_df.to_csv(f"{args.output_dir}/threshold_sweep.csv", index=False)
        mask = (
            (sweep_df["profit_factor"] >= 1.5)
            & (sweep_df["win_rate"] >= 0.53)
            & (sweep_df["n_trades"] >= 8)
            & (sweep_df["max_drawdown_pct"] >= -0.03)
            & (sweep_df["total_return_pct"] > 0)
        )
        if mask.any():
            best = sweep_df[mask].sort_values("profit_factor", ascending=False).iloc[0]
            chosen_thr = float(best["thr"])
        else:
            chosen_thr = float(sweep_df.sort_values("profit_factor", ascending=False).iloc[0]["thr"])
        print(f"  → sweep chose thr={chosen_thr}")
    else:
        chosen_thr = args.prob_threshold

    final_sig = (all_probs > chosen_thr).fillna(False)
    trades = bml.simulate(f, final_sig, args.tp_atr, args.sl_atr, args.max_hold)
    metrics = bml.compute_metrics(trades)
    print(
        f"  final thr={chosen_thr}: n={metrics['n_trades']}, "
        f"WR={metrics.get('win_rate', 0):.3f}, PF={metrics.get('profit_factor', 0):.3f}, "
        f"RET={metrics.get('total_return_pct', 0):.4f}, DD={metrics.get('max_drawdown_pct', 0):.4f}"
    )
    trades.to_csv(f"{args.output_dir}/trades.csv", index=False)

    def to_py(o):
        if isinstance(o, dict):
            return {k: to_py(v) for k, v in o.items()}
        if isinstance(o, list):
            return [to_py(v) for v in o]
        if hasattr(o, "item"):
            return o.item()
        if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
            return None
        return o

    # Classify top features by group
    def classify_feature(name: str) -> str:
        if any(name.startswith(p) for p in ("cong_", "lobbying_", "filing_", "days_since_", "edgar_", "gov_")):
            return "alt_data"
        if any(name.startswith(p) for p in ("intra_", "gap_", "overnight_", "range_", "vwap_")):
            return "intraday"
        if any(name.startswith(p) for p in ("ti_p1_", "ti_p2_", "ti_p3_", "ti_p4_", "part1_", "part2_", "part3_", "part4_")):
            return "trading_insight"
        if name in (
            "rsi_14", "rsi_7", "rsi_21", "macd_line", "macd_signal", "macd_hist",
            "bb_pct", "bb_width", "atr_14", "ema_9", "ema_21", "ema_50", "ema_200",
            "volume_ratio", "adx_14", "obv_slope", "cci_20", "stoch_k", "stoch_d",
        ):
            return "base"
        return "base_or_insight"

    feature_group_counts: dict[str, int] = {}
    for fname in top_k_features_global:
        grp = classify_feature(fname)
        feature_group_counts[grp] = feature_group_counts.get(grp, 0) + 1

    meta = to_py({
        "ticker": args.ticker,
        "pipeline_version": "xgb_v5_pruned",
        "strategy_variant": "ML_XGB_v5_pruned",
        "run_at": datetime.utcnow().isoformat() + "Z",
        "top_k": args.top_k,
        "n_features_before_pruning": len(all_fc),
        "top_k_features": top_k_features_global,
        "top_k_feature_group_counts": feature_group_counts,
        "feature_sources": {
            "base_indicators": "~46",
            "intraday": "22",
            "alt_data_pruned": "variable",
            "cross_sectional": "17 if cache else skipped",
            "trading_insight_v3": "13",
            "trading_insight_part1": "116",
            "trading_insight_part2": "141",
            "trading_insight_part3": "151",
            "trading_insight_part4": "88",
        },
        "daily_bars": len(f),
        "walk_forward_folds": len(fold_summaries),
        "strategy": {
            "name": "ML_XGB_v5_pruned",
            "side": "long",
            "tp_atr": args.tp_atr,
            "sl_atr": args.sl_atr,
            "max_hold_days": args.max_hold,
            "prob_threshold": chosen_thr,
            "threshold_swept": args.sweep_threshold,
            "model": f"XGBClassifier scout(d=3,n=50) + prod(d=4,lr=0.05,n=100)",
            "calibration": "native predict_proba",
            "slippage_bps": 5.0,
            "fee_per_share": 0.0035,
            "notional_per_trade": 5000,
        },
        "metrics_oos_aggregate": metrics,
        "fold_summaries": fold_summaries,
    })
    with open(f"{args.output_dir}/run_meta.json", "w") as fp:
        json.dump(meta, fp, indent=2, default=str)
    print(f"  [v5] saved run_meta.json → {args.output_dir}")


if __name__ == "__main__":
    main()
