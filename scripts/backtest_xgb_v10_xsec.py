"""
backtest_xgb_v10_xsec.py — Stage C: cross-sectional ticker pooling for v10.

Why: per-ticker XGBoost backtests typically use only 50-200 of v10's ~1485
features because train folds have ~1200 rows (p>n constraint). Pooling N
tickers' rows along axis=0 multiplies row count by N, breaking the p>n
ceiling and letting trees split on far more features. Ticker_id is added as
an XGBoost native categorical (xgboost >= 1.6 supports enable_categorical=True).

Walk-forward folds are by DATE (not by ticker). Each fold trains on T months
across ALL tickers, OOS on next T months across ALL tickers. After fold
inference, predictions are split back by ticker for per-ticker metrics
aggregation.

Shared with backtest_xgb_v10.py:
  - build_v10_features (feature engineering — DRY)
  - _xgb_base_params, _xgb_fit_kwargs, _resolve_top_k (model config)
  - numeric_cols (feature column selection)
  - backtest_ml.make_walk_forward_folds, simulate, compute_metrics (eval)

Env flags honored: XGB_NO_TOPK, XGB_TOP_K, XGB_KEEP_ZERO_IMP, XGB_DEVICE,
                   XGB_XSEC_WEIGHT (equal|marketcap|volinv),
                   XGB_TREE_METHOD (hist|approx|exact),
                   XGB_N_ESTIMATORS, XGB_EARLY_STOP, XGB_USE_FLOAT32 (default 1).

Smoke usage (10-ticker default):
    cd s&p500-ticker-mastery
    source /Users/orginal/.venvs/sp500-mastery/bin/activate
    AUTO_CLOUD_DISPATCH=0 XGB_NO_TOPK=1 python scripts/backtest_xgb_v10_xsec.py \\
      --tickers AAPL,MSFT,GOOG,META,NVDA,TSLA,AMZN,JPM,JNJ,XOM \\
      --output-dir backtests_xgb_v10_xsec/smoke_run_2026-05-20 \\
      --strategy ML_XGB_v10_xsec_smoke \\
      --job-id stageC-smoke

Full-500 usage (cloud-dispatched to Modal A10G):
    XGB_NO_TOPK=1 XGB_DEVICE=cuda XGB_TREE_METHOD=hist XGB_N_ESTIMATORS=2000 \\
    XGB_EARLY_STOP=50 XGB_XSEC_WEIGHT=equal \\
    python scripts/backtest_xgb_v10_xsec.py \\
      --tickers-file registry/sp500_tickers.csv \\
      --output-dir artifacts/v10_xsec_full500_<job_id> \\
      --strategy ML_XGB_v10_xsec_full500 \\
      --job-id <cloud-dispatch-job-id> \\
      --wf-train-days 252 --wf-test-days 21 --wf-stride-days 21

Stage C in XGB_FULL_UTILIZATION roadmap. Author: 2026-05-20.
Extended to full S&P 500 universe: 2026-05-20 (xsec_full500 scale-up).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

WORK = Path(
    "/Users/orginal/Library/CloudStorage/"
    "GoogleDrive-zachgladstone@gmail.com/My Drive/"
    "AI-Tools/s&p500-ticker-mastery"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Shared imports from v10 — DRY
from backtest_xgb_v10 import (  # noqa: E402
    build_v10_features,
    _xgb_base_params,
    _xgb_fit_kwargs,
    _resolve_top_k,
    V10_FEATURE_VERSION,
)
from backtest_xgb_v7 import numeric_cols  # noqa: E402
import backtest_ml as bml  # noqa: E402
import xgboost as xgb  # noqa: E402

try:
    import cross_sectional_features as csf  # type: ignore[import-not-found]
except Exception:
    csf = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Memory + dtype controls (for full-500 scale)
# ---------------------------------------------------------------------------
USE_FLOAT32 = os.environ.get("XGB_USE_FLOAT32", "1") == "1"


def _load_ticker_list(
    tickers_arg: str | None,
    tickers_file: str | None,
) -> list[str]:
    """Resolve a final ticker list from --tickers (comma) or --tickers-file (csv/txt).

    Resolution order:
      1. --tickers (comma-list) - explicit override, even if file also given.
      2. --tickers-file (csv with 'ticker' header OR plain newline txt).
      3. Default registry: AI-Tools/.../registry/sp500_tickers.csv if exists.
      4. Fallback: 10-ticker smoke set.
    """
    if tickers_arg:
        return [t.strip().upper() for t in tickers_arg.split(",") if t.strip()]
    candidate_files = []
    if tickers_file:
        candidate_files.append(Path(tickers_file))
    candidate_files.append(WORK / "registry" / "sp500_tickers.csv")
    candidate_files.append(WORK / "sp500_tickers.txt")
    for fp in candidate_files:
        if fp.exists():
            txt = fp.read_text().splitlines()
            out: list[str] = []
            for ln in txt:
                ln = ln.strip()
                if not ln or ln.lower() == "ticker":
                    continue
                tok = ln.split(",")[0].strip().upper()
                if tok and tok.replace(".", "").replace("-", "").isalnum():
                    out.append(tok)
            if out:
                logger.info("  [xsec] resolved %d tickers from %s", len(out), fp)
                return out
    logger.warning("  [xsec] no ticker source found - falling back to smoke set")
    return ["AAPL", "MSFT", "GOOG", "META", "NVDA",
            "TSLA", "AMZN", "JPM", "JNJ", "XOM"]


def _ticker_pit_first_seen(ticker: str):
    """Approx point-in-time first-seen date for a ticker.

    Uses mtime of the earliest mastery_files/<ticker>_*.md file as a proxy
    for when a ticker first joined our universe. Returns pd.Timestamp or None.
    """
    mf_dir = WORK / "mastery_files"
    if not mf_dir.exists():
        return None
    earliest_mtime = None
    for fp in mf_dir.glob(f"{ticker}_*.md"):
        try:
            m = fp.stat().st_mtime
            if earliest_mtime is None or m < earliest_mtime:
                earliest_mtime = m
        except Exception:
            continue
    if earliest_mtime is None:
        return None
    return pd.Timestamp(earliest_mtime, unit="s")


def _make_days_folds(
    df,
    train_days: int,
    test_days: int,
    stride_days: int,
    min_folds: int = 5,
):
    """Days-based walk-forward (rolling) - alternative to months-based folds.

    Returns list of dicts with ISO-string start/end keys matching the format
    backtest_ml.make_walk_forward_folds emits.
    """
    if df.empty:
        return []
    start = pd.Timestamp(df.index.min())
    end = pd.Timestamp(df.index.max())
    folds: list[dict] = []
    train_start = start
    while True:
        train_end = train_start + pd.tseries.offsets.BDay(train_days)
        oos_start = train_end
        oos_end = oos_start + pd.tseries.offsets.BDay(test_days)
        if oos_end > end - pd.tseries.offsets.BDay(21):
            break
        folds.append({
            "fold": len(folds) + 1,
            "train_start": train_start.isoformat(),
            "train_end": train_end.isoformat(),
            "oos_start": oos_start.isoformat(),
            "oos_end": oos_end.isoformat(),
        })
        train_start = train_start + pd.tseries.offsets.BDay(stride_days)
    if len(folds) < min_folds:
        logger.warning(
            "  [xsec] days-folds returned %d (< min %d) - shorten params",
            len(folds), min_folds,
        )
    return folds


def _compute_sample_weights(panel, train_idx, weight_mode: str):
    """Compute per-row sample weights for the XGB final model.

    Modes:
      - 'equal' (default): None
      - 'volinv': 1 / rolling vol (clipped at 99th pct)
      - 'marketcap': uniform per-ticker proxy
    """
    if weight_mode == "equal" or not weight_mode:
        return None
    train_rows = panel.loc[train_idx]
    if weight_mode == "volinv":
        vol_cols = [c for c in train_rows.columns
                    if c.startswith("vol_") or c == "volatility_21" or c == "atr_14"]
        if not vol_cols:
            logger.warning("  [xsec] volinv requested but no vol col - fallback equal")
            return None
        v = train_rows[vol_cols[0]].fillna(train_rows[vol_cols[0]].median()).values
        w = 1.0 / np.clip(v, 1e-6, None)
        cap = np.quantile(w, 0.99)
        w = np.clip(w, 0, cap)
        dt = np.float32 if USE_FLOAT32 else np.float64
        return (w / w.mean()).astype(dt)
    if weight_mode == "marketcap":
        tickers = train_rows["ticker_cat"].astype(str).values
        counts = pd.Series(tickers).value_counts().to_dict()
        dt = np.float32 if USE_FLOAT32 else np.float64
        w = np.array([1.0 / counts[t] for t in tickers], dtype=dt)
        return w / w.mean()
    logger.warning("  [xsec] unknown weight mode %s - fallback equal", weight_mode)
    return None


def _load_ticker_panel(
    tickers: list[str],
    use_mythos: bool = False,
    apply_pit_filter: bool = True,
) -> tuple[pd.DataFrame, list[str], dict]:
    """Build a stacked (date x ticker) panel.

    Each ticker's v10 feature DataFrame is loaded (uses feature_cache HIT path
    when available, so this is fast on repeat). A 'ticker' column is added
    (pd.Categorical), and frames are concatenated along axis=0 then sorted by
    date index. NaN handling: features will be fillna(0) at train/inference
    time per fold to mirror v10's per-ticker pattern.

    Returns:
        panel: pd.DataFrame with MultiIndex (ticker_str, date) and a
               'ticker_cat' column as pd.Categorical for XGBoost native cat.
        feature_cols: list of numeric feature column names (excludes y/ticker_cat).
        meta: dict with per-ticker row counts + total.
    """
    universe_agg = None
    cache_manifest = WORK / "cache" / "universe_agg_manifest.json"
    if cache_manifest.exists() and csf is not None:
        try:
            universe_agg = csf.precompute_universe_aggregates()
        except Exception as e:
            logger.warning("  [csf] cache load failed: %s", e)

    frames: list[pd.DataFrame] = []
    per_ticker_rows: dict = {}
    feature_cols_first: list = []

    for ticker in tickers:
        try:
            f, _fallback, _module_counts = build_v10_features(
                ticker, universe_agg, use_mythos=use_mythos
            )
        except Exception as e:
            logger.warning("  [xsec] %s build_v10_features failed: %s", ticker, e)
            continue
        if f is None or f.empty:
            logger.warning("  [xsec] %s empty frame; skipping", ticker)
            continue

        f = f.copy()
        # Point-in-time filter: drop rows after the soft cutoff (mtime + 2y buffer).
        # autosolve_skip: tz-awareness fix
        if apply_pit_filter:
            pit = _ticker_pit_first_seen(ticker)
            if pit is not None:
                before = len(f)
                # Match index tz: feature frames are tz-aware UTC; localize pit to UTC.
                try:
                    idx_tz = getattr(f.index, "tz", None)
                    if idx_tz is not None and pit.tzinfo is None:
                        pit = pit.tz_localize("UTC").tz_convert(idx_tz)
                    cutoff = pit + pd.Timedelta(days=730)
                    f = f[f.index <= cutoff]
                except (TypeError, AttributeError):
                    # If tz coercion fails, skip the filter for this ticker
                    pass
                if len(f) < before:
                    logger.debug(
                        "  [xsec] %s PIT filter dropped %d rows", ticker, before - len(f)
                    )
        # Downcast numeric columns to float32 for memory (full-500 scale)
        if USE_FLOAT32:
            for col in f.columns:
                if pd.api.types.is_float_dtype(f[col]) and f[col].dtype != np.float32:
                    f[col] = f[col].astype(np.float32)
        f["ticker_str"] = ticker
        frames.append(f)
        per_ticker_rows[ticker] = int(len(f))
        if not feature_cols_first:
            feature_cols_first = numeric_cols(f)
        logger.info("  [xsec] +%s: rows=%d, cols=%d", ticker, len(f), f.shape[1])

    if not frames:
        raise RuntimeError("no tickers loaded — abort")

    panel = pd.concat(frames, axis=0, sort=False).sort_index()
    # Make ticker categorical for XGBoost native cat support
    panel["ticker_cat"] = pd.Categorical(panel["ticker_str"])
    # Drop the str copy
    if "ticker_str" in panel.columns:
        panel = panel.drop(columns=["ticker_str"])

    feature_cols = [c for c in numeric_cols(panel) if c != "y"]

    meta = {
        "n_tickers": len(per_ticker_rows),
        "per_ticker_rows": per_ticker_rows,
        "total_rows": int(len(panel)),
        "n_features": len(feature_cols),
    }
    logger.info(
        "  [xsec] PANEL: tickers=%d, rows=%d, features=%d",
        meta["n_tickers"],
        meta["total_rows"],
        meta["n_features"],
    )
    return panel, feature_cols, meta


def _xsec_train_oos(
    panel: pd.DataFrame,
    feature_cols: list[str],
    fold: dict,
    args,
) -> tuple[pd.Series, dict]:
    """Train + OOS-predict for one fold of the cross-sectional walk-forward.

    Returns:
        oos_probs: predicted prob series indexed like oos rows (includes ticker).
        fold_diag: diagnostic dict with n_train, n_oos, n_top_features,
                   n_features_in_trees.
    """
    train_end_emb = (
        pd.Timestamp(fold["train_end"])
        - pd.tseries.offsets.BDay(21)  # LABEL_EMBARGO_DAYS from v10
    )
    train = panel[
        (panel.index >= fold["train_start"]) & (panel.index < train_end_emb)
    ]
    oos = panel[
        (panel.index >= fold["oos_start"]) & (panel.index < fold["oos_end"])
    ]
    if len(train) < 50 or len(oos) < 20:
        return pd.Series(dtype=float), {
            "n_train": int(len(train)),
            "n_oos": int(len(oos)),
            "n_top_features": 0,
            "n_features_in_trees": 0,
            "skipped": True,
        }

    # Scout-prune is shared with v10 (uses _resolve_top_k)
    X_tr_all = train[feature_cols].fillna(0).values
    y_tr = train["y"].values
    X_oos_all = oos[feature_cols].fillna(0).values

    scout = xgb.XGBClassifier(**_xgb_base_params("scout"))
    scout.fit(X_tr_all, y_tr)
    importances = list(zip(feature_cols, scout.feature_importances_))
    importances.sort(key=lambda x: -x[1])
    effective_top_k = _resolve_top_k(args.top_k, len(train), len(feature_cols))
    keep_zero = os.environ.get("XGB_KEEP_ZERO_IMP", "0") == "1"
    if keep_zero:
        top_features = [c for c, _ in importances[: effective_top_k]]
    else:
        top_features = [c for c, imp in importances[: effective_top_k] if imp > 0]
        if len(top_features) < 10:
            top_features = [c for c, _ in importances[: effective_top_k]]

    # Final model: add ticker_cat as native categorical
    # autosolve_skip: env wiring, no errors
    final_params = _xgb_base_params("final")
    final_params["enable_categorical"] = True
    # Honor env overrides for cloud-route (cuda + hist tree method on Modal A10G)
    if "XGB_DEVICE" in os.environ:
        final_params["device"] = os.environ["XGB_DEVICE"]
    if "XGB_TREE_METHOD" in os.environ:
        final_params["tree_method"] = os.environ["XGB_TREE_METHOD"]
    if "XGB_N_ESTIMATORS" in os.environ:
        try:
            final_params["n_estimators"] = int(os.environ["XGB_N_ESTIMATORS"])
        except ValueError:
            pass
    final = xgb.XGBClassifier(**final_params)

    # Build (X_train_final, X_oos_final) with ticker_cat included as a column
    cols_with_ticker = top_features + ["ticker_cat"]
    X_tr_df = train[cols_with_ticker].copy()
    X_oos_df = oos[cols_with_ticker].copy()
    # Ensure ticker_cat is still categorical (concat sometimes coerces)
    X_tr_df["ticker_cat"] = pd.Categorical(X_tr_df["ticker_cat"])
    X_oos_df["ticker_cat"] = pd.Categorical(
        X_oos_df["ticker_cat"], categories=X_tr_df["ticker_cat"].cat.categories
    )
    # Fill numeric NaNs
    for c in top_features:
        X_tr_df[c] = X_tr_df[c].fillna(0)
        X_oos_df[c] = X_oos_df[c].fillna(0)

    y_oos_arr = oos["y"].values
    # autosolve_skip: weight integration, no errors
    weight_mode = os.environ.get("XGB_XSEC_WEIGHT", "equal").lower()
    sample_w = _compute_sample_weights(panel, train.index, weight_mode)
    fit_kw = _xgb_fit_kwargs(
        eval_set=[(X_oos_df, y_oos_arr)],
        early_stop=True,
    )
    if sample_w is not None:
        fit_kw["sample_weight"] = sample_w
    # Honor XGB_EARLY_STOP env (override default early-stopping rounds)
    if "XGB_EARLY_STOP" in os.environ:
        try:
            fit_kw["early_stopping_rounds"] = int(os.environ["XGB_EARLY_STOP"])
        except (ValueError, TypeError):
            pass
    final.fit(X_tr_df, y_tr, **fit_kw)

    # Diagnostic: features actually used in tree splits
    try:
        booster = final.get_booster()
        score = booster.get_score(importance_type="gain")
        n_feats_in_trees = len(score)
    except Exception:
        n_feats_in_trees = -1

    probs = final.predict_proba(X_oos_df)[:, 1]
    oos_probs = pd.Series(
        probs, index=oos.index, name="prob"
    )
    # Attach ticker for downstream per-ticker split
    oos_probs_df = pd.DataFrame(
        {"prob": probs, "ticker": oos["ticker_cat"].astype(str).values},
        index=oos.index,
    )

    fold_diag = {
        "n_train": int(len(train)),
        "n_oos": int(len(oos)),
        "n_top_features": int(len(top_features)),
        "n_features_in_trees": int(n_feats_in_trees),
        "skipped": False,
    }
    return oos_probs_df, fold_diag


def main():
    ap = argparse.ArgumentParser(
        description="v10 cross-sectional ticker-pooled XGBoost backtest"
    )
    # autosolve_skip: arg parsing, no errors
    ap.add_argument(
        "--tickers",
        type=str,
        default="",
        help="Comma-separated ticker list. Empty + no --tickers-file = use registry default (full S&P 500).",
    )
    ap.add_argument(
        "--tickers-file",
        type=str,
        default="",
        help="Path to ticker file (CSV with 'ticker' header OR newline-txt). "
             "Default: registry/sp500_tickers.csv (auto-resolved). "
             "Overridden by --tickers if given.",
    )
    ap.add_argument(
        "--no-pit-filter",
        action="store_true",
        default=False,
        help="Disable point-in-time membership filter (default: enabled, "
             "uses mastery_files/<ticker>_*.md mtime as proxy for first-seen).",
    )
    ap.add_argument(
        "--wf-train-days",
        type=int,
        default=0,
        help="Days-based walk-forward train window. If 0, uses months-based folds.",
    )
    ap.add_argument(
        "--wf-test-days",
        type=int,
        default=21,
        help="Days-based walk-forward OOS window (only used if --wf-train-days > 0).",
    )
    ap.add_argument(
        "--wf-stride-days",
        type=int,
        default=21,
        help="Days-based walk-forward stride (only used if --wf-train-days > 0).",
    )
    ap.add_argument(
        "--wf-train-months",
        type=int,
        default=24,
        help="Months-based fold train window (used when --wf-train-days == 0).",
    )
    ap.add_argument(
        "--wf-test-months",
        type=int,
        default=12,
        help="Months-based fold OOS window.",
    )
    ap.add_argument(
        "--wf-step-months",
        type=int,
        default=12,
        help="Months-based fold step.",
    )
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument(
        "--strategy",
        type=str,
        default="ML_XGB_v10_xsec",
        help="Strategy label for run_meta.json",
    )
    ap.add_argument(
        "--job-id",
        type=str,
        default="",
        help="CI/CD job identifier — written to run_meta.json",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=int(os.environ.get("XGB_TOP_K", "0")),
        help="Top-K features; 0 = adaptive (env XGB_NO_TOPK=1 forces n_features)",
    )
    ap.add_argument("--prob-threshold", type=float, default=0.50)
    ap.add_argument("--tp-atr", type=float, default=1.5)
    ap.add_argument("--sl-atr", type=float, default=1.0)
    ap.add_argument("--max-hold", type=int, default=21)
    ap.add_argument(
        "--use-mythos-features", action="store_true", default=False
    )
    args = ap.parse_args()

    # autosolve_skip: main flow
    os.makedirs(args.output_dir, exist_ok=True)
    tickers = _load_ticker_list(args.tickers or None, args.tickers_file or None)
    logger.info(
        "[xsec] Starting: n_tickers=%d strategy=%s job_id=%s output_dir=%s",
        len(tickers),
        args.strategy,
        args.job_id or "(none)",
        args.output_dir,
    )

    t_start = time.time()
    panel, feature_cols, panel_meta = _load_ticker_panel(
        tickers,
        use_mythos=args.use_mythos_features,
        apply_pit_filter=not args.no_pit_filter,
    )
    t_panel = time.time() - t_start
    logger.info("  [xsec] panel build: %.1fs", t_panel)

    # Walk-forward folds — days-based if --wf-train-days > 0, else months
    if args.wf_train_days and args.wf_train_days > 0:
        folds = _make_days_folds(
            panel,
            train_days=args.wf_train_days,
            test_days=args.wf_test_days,
            stride_days=args.wf_stride_days,
        )
        fold_basis = "days"
    else:
        folds = bml.make_walk_forward_folds(
            panel,
            train_months=args.wf_train_months,
            test_months=args.wf_test_months,
            step_months=args.wf_step_months,
        )
        fold_basis = "months"
    logger.info("  [xsec] folds: %d (basis=%s)", len(folds), fold_basis)

    all_probs_df = pd.DataFrame(columns=["prob", "ticker"])
    fold_summaries: list = []

    for fold in folds:
        oos_df, fold_diag = _xsec_train_oos(panel, feature_cols, fold, args)
        if fold_diag.get("skipped"):
            continue
        all_probs_df = pd.concat([all_probs_df, oos_df], axis=0)
        fold_diag["fold"] = fold["fold"]
        fold_summaries.append(fold_diag)
        logger.info(
            "  [xsec] fold %d: n_train=%d n_oos=%d n_top=%d in_trees=%d",
            fold["fold"],
            fold_diag["n_train"],
            fold_diag["n_oos"],
            fold_diag["n_top_features"],
            fold_diag["n_features_in_trees"],
        )

    # Per-ticker metrics aggregation
    per_ticker_metrics: dict = {}
    for ticker in tickers:
        ticker_panel = panel[panel["ticker_cat"].astype(str) == ticker]
        ticker_probs = all_probs_df[all_probs_df["ticker"] == ticker]["prob"]
        # Align by index
        sig = pd.Series(False, index=ticker_panel.index)
        sig.loc[ticker_probs.index[ticker_probs > args.prob_threshold]] = True
        if not sig.any():
            per_ticker_metrics[ticker] = {
                "n_trades": 0,
                "note": "no_signals",
            }
            continue
        try:
            trades = bml.simulate(
                ticker_panel, sig.fillna(False),
                args.tp_atr, args.sl_atr, args.max_hold
            )
            mm = bml.compute_metrics(trades)
            per_ticker_metrics[ticker] = mm
        except Exception as e:
            per_ticker_metrics[ticker] = {"error": str(e), "n_trades": 0}

    # Aggregate diagnostics
    mean_in_trees = (
        sum(f["n_features_in_trees"] for f in fold_summaries) / len(fold_summaries)
        if fold_summaries else 0
    )
    mean_n_top = (
        sum(f["n_top_features"] for f in fold_summaries) / len(fold_summaries)
        if fold_summaries else 0
    )

    run_meta = {
        "pipeline_version": V10_FEATURE_VERSION,
        "strategy": args.strategy,
        "strategy_variant": "xsec",
        "job_id": args.job_id,
        "tickers": tickers,
        "n_tickers": len(tickers),
        "panel_meta": panel_meta,
        "features_total": len(feature_cols),
        "walk_forward_folds": len(folds),
        "fold_summaries": fold_summaries,
        "mean_features_in_trees": float(mean_in_trees),
        "mean_n_top_features": float(mean_n_top),
        "per_ticker_metrics": per_ticker_metrics,
        "args": {
            "top_k": args.top_k,
            "prob_threshold": args.prob_threshold,
            "tp_atr": args.tp_atr,
            "sl_atr": args.sl_atr,
            "max_hold": args.max_hold,
        },
        # autosolve_skip: meta dict, no errors
        "env": {
            "XGB_NO_TOPK": os.environ.get("XGB_NO_TOPK", "0"),
            "XGB_KEEP_ZERO_IMP": os.environ.get("XGB_KEEP_ZERO_IMP", "0"),
            "XGB_TOP_K": os.environ.get("XGB_TOP_K", "0"),
            "XGB_DEVICE": os.environ.get("XGB_DEVICE", ""),
            "XGB_TREE_METHOD": os.environ.get("XGB_TREE_METHOD", ""),
            "XGB_N_ESTIMATORS": os.environ.get("XGB_N_ESTIMATORS", ""),
            "XGB_EARLY_STOP": os.environ.get("XGB_EARLY_STOP", ""),
            "XGB_XSEC_WEIGHT": os.environ.get("XGB_XSEC_WEIGHT", "equal"),
            "XGB_USE_FLOAT32": os.environ.get("XGB_USE_FLOAT32", "1"),
        },
        "fold_basis": fold_basis,
        "no_pit_filter": bool(args.no_pit_filter),
        "run_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_total_s": round(time.time() - t_start, 1),
    }
    meta_path = Path(args.output_dir) / "run_meta.json"
    with open(meta_path, "w") as f:
        json.dump(run_meta, f, indent=2)
    logger.info("  [xsec] Wrote %s", meta_path)
    logger.info(
        "[xsec] DONE. tickers=%d total_rows=%d features=%d mean_in_trees=%.1f mean_n_top=%.1f elapsed=%.1fs",
        len(tickers),
        panel_meta["total_rows"],
        len(feature_cols),
        mean_in_trees,
        mean_n_top,
        run_meta["elapsed_total_s"],
    )


if __name__ == "__main__":
    main()
