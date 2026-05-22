"""
live_paper_trade_signals.py — Generate today's trading signals from v7/v8/v9/v10 XGBoost models.

For each mastered ticker, this script:
  1. Discovers the best available model
     (v10 > v9_mythos > v8_alpha158 > v8_daily > v7; tie → newest run_at)
  2. Fetches overnight/latest OHLCV bar data via yfinance
  3. Builds the same feature set used at training time
  4. Generates a calibrated probability and compares to that ticker's threshold
  5. Writes output to:
       paper_trade/signals/{YYYY-MM-DD}.json

Output format per ticker:
  {
    "ticker": "AAPL",
    "date": "2026-05-16",
    "prob": 0.73,
    "threshold": 0.68,
    "signal": 1,           # 1 = BUY, 0 = NO TRADE
    "position_size": 100.0, # notional in USD (capped at MAX_POSITION_NOTIONAL=$100=5% of $2k budget; informational only — live_paper_trade.compute_position_sizes does the actual sizing and may downsize further)
    "pipeline": "xgb_v8_alpha158",
    "model_run_dir": "...",
    "features_used": 50
  }

Mode note: This script does NOT require Alpaca credentials. It runs entirely
on yfinance data to produce probability estimates from the trained XGBoost models.

IMPORTANT: Models are NOT retrained here. We use the most recent refit model
from the walk-forward fold, which is the final fold's refit_model.pkl (if saved)
or re-run inference using the top-K feature set from run_meta.json.
Since the v7/v8 scripts do not persist pkl files by default, this script
re-runs a FAST inference pass — builds features on the latest N bars and
applies the model trained on the final fold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORK = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools/s&p500-ticker-mastery"
)
SCRIPTS_DIR = WORK / "scripts"
PAPER_DIR = WORK / "paper_trade"
SIGNALS_DIR = PAPER_DIR / "signals"
FEATURES_DIR = PAPER_DIR / "features"
LOGS_DIR = WORK / "logs"
MASTERY_DIR = WORK / "mastery_files"

SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Feature-row persistence (data_persistence_audit gap §2.2)
# ---------------------------------------------------------------------------
def _persist_feature_row(
    ticker: str,
    feature_row: "pd.DataFrame",
    *,
    prob: float,
    threshold: float,
    signal: int,
    pipeline: str,
    model_run_dir: str,
    feature_hash: str | None,
    signal_ts_utc: str,
    today_iso: str,
) -> None:
    """
    Persist the EXACT feature row passed to model.predict_proba() to:
        paper_trade/features/<DATE>/<TICKER>.parquet

    One file per ticker per day; overwrites on rerun (signal generation is
    idempotent per day). Wrapped in try/except — a parquet failure never
    crashes signal generation.
    """
    try:
        # Guard: live without pyarrow/fastparquet must not blow up.
        try:
            import pyarrow  # noqa: F401
        except Exception:
            try:
                import fastparquet  # noqa: F401
            except Exception:
                log.warning(
                    f"{ticker}: parquet engine unavailable (no pyarrow/fastparquet); "
                    f"skipping feature-row persistence"
                )
                return

        if feature_row is None or len(feature_row) == 0:
            log.warning(f"{ticker}: feature_row empty/None — skipping persistence")
            return

        # Single-row frame — keep as-is, then attach metadata columns.
        row = feature_row.copy()
        if len(row) > 1:
            row = row.iloc[[-1]]

        # Compute feature_hash from sorted column names if not provided
        # (matches the convention in run_meta.json for v10 persisted models).
        fhash = feature_hash
        if not fhash:
            cols_sorted = sorted(str(c) for c in row.columns)
            fhash = hashlib.sha256(
                ",".join(cols_sorted).encode("utf-8")
            ).hexdigest()[:16]

        row["prob"] = float(prob)
        row["threshold"] = float(threshold)
        row["signal"] = int(signal)
        row["pipeline"] = str(pipeline)
        row["model_run_dir"] = str(model_run_dir)
        row["feature_hash"] = str(fhash)
        row["signal_ts_utc"] = str(signal_ts_utc)
        row["ticker"] = str(ticker)
        row["date"] = str(today_iso)

        # Coerce object/mixed-dtype columns to string so parquet writer never
        # fails on Timestamp/Period/Decimal/etc. (dtype guard).
        for c in row.columns:
            if row[c].dtype == "object":
                try:
                    row[c] = row[c].astype(float)
                except Exception:
                    row[c] = row[c].astype(str)

        out_dir = FEATURES_DIR / today_iso
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{ticker}.parquet"
        row.to_parquet(out_path, index=True)
        log.info(
            f"{ticker}: feature-row persisted → {out_path} "
            f"({row.shape[1]} cols incl. metadata)"
        )
    except Exception as exc:  # pylint: disable=broad-except
        log.warning(f"{ticker}: feature-row persistence failed — {exc}")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FILE = LOGS_DIR / "paper_trade.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.FileHandler(str(LOG_FILE)),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("pt_signals")

# Make sure our scripts are importable
sys.path.insert(0, str(SCRIPTS_DIR))

# Cap (2026-05-22): 20% of $2k synthetic budget. Mirrors live_paper_trade.py
# constant. This is INFORMATIONAL ONLY — actual sizing is done by
# live_paper_trade.compute_position_sizes (which honours env LIVE_BUDGET_USD
# / LIVE_MAX_POSITION_USD overrides). Reads the env at import time so signal
# files written under a non-default budget reflect the active cap.
MAX_POSITION_NOTIONAL = float(
    os.environ.get("LIVE_MAX_POSITION_USD",
                   str(float(os.environ.get("LIVE_BUDGET_USD", "2000")) * 0.05))
)
LABEL_EMBARGO_DAYS = 21

# ---------------------------------------------------------------------------
# Phase A feature-gap close (2026-05-18): wire OHLCV-only feature modules
# from backtest_xgb_v10.py into the live signal builder. Goal: 50 → ~300 cols.
# Each import is best-effort; missing modules are skipped (fail-soft).
# ---------------------------------------------------------------------------
_PHASE_A_MODULES: list[tuple[str, str, str]] = [
    # (label, module name, function name)
    ("alpha158",        "qlib_alpha158_features",            "add_alpha158_features"),
    ("dfs",             "featuretools_dfs_features",         "add_dfs_features"),
    ("vol_estimators",  "volatility_estimator_features",     "add_volatility_features"),
    ("ti_part1",        "trading_insight_features_part1",    "add_features_part1"),
    ("ti_part2",        "trading_insight_features_part2",    "add_features_part2"),
    ("ti_part3",        "trading_insight_features_part3",    "add_features_part3"),
    ("ti_part4",        "trading_insight_features_part4",    "add_features_part4"),
    ("squeeze",         "bollinger_keltner_squeeze_features","add_bollinger_keltner_squeeze_features"),
]

_PHASE_A_FNS: dict[str, Any] = {}
for _label, _mod, _fn in _PHASE_A_MODULES:
    try:
        _m = __import__(_mod)
        _PHASE_A_FNS[_label] = getattr(_m, _fn)
    except Exception as _exc:
        log.warning(f"phase_a: could not import {_mod}.{_fn} — {_exc}")
log.info(
    f"phase_a: feature modules ready = {list(_PHASE_A_FNS.keys())} "
    f"({len(_PHASE_A_FNS)}/{len(_PHASE_A_MODULES)})"
)


def _extend_features_phase_a(df50: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
    """
    Expand the 50-feature basic frame with OHLCV-only modules from v10 backtest.
    Each module is wrapped in try/except — failure is logged and the partial
    frame is returned (never raises).
    """
    df = df50
    start_cols = df.shape[1]
    per_module_added: dict[str, int] = {}

    for label, fn in _PHASE_A_FNS.items():
        before = df.shape[1]
        try:
            extended = fn(df)
            # Some modules return None on degenerate input — skip in that case.
            if extended is None or not isinstance(extended, pd.DataFrame):
                log.warning(f"{ticker}: phase_a:{label} returned non-DataFrame, skipping")
                continue
            df = extended
            per_module_added[label] = df.shape[1] - before
        except Exception as exc:
            log.warning(
                f"{ticker}: phase_a:{label} failed ({exc}) — keeping prior frame"
            )
            continue

    total_added = df.shape[1] - start_cols
    log.info(
        f"{ticker}: phase_a expansion {start_cols} → {df.shape[1]} cols "
        f"(+{total_added}) breakdown={per_module_added}"
    )
    return df

# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------
def discover_models() -> dict[str, dict]:
    """
    Scan v7 / v8 / v9 / v10 backtest directories to find the best available
    model per ticker. Returns {ticker: {dir, meta, priority, run_at}}.

    Priority (highest first):
      xgb_v10 (mythos / daily_integration / featuretools) > xgb_v9_mythos
        > xgb_v8_alpha158 > xgb_v8_daily_integration > xgb_v7_maximal_plus

    Tie-breaker within a priority: most recent run_at wins.
    """
    priority_map = {
        # v10 — current backtest standard (xgb_v10 + optional mythos)
        "xgb_v10": 6,
        "xgb_v10_mythos": 6,
        # v9 — mythos integration
        "xgb_v9_mythos": 5,
        "xgb_v9": 4,
        # v8 — alpha158 / daily integration
        "xgb_v8_alpha158": 3,
        "xgb_v8_daily_integration": 2,
        # v7 — legacy fallback
        "xgb_v7_maximal_plus": 1,
    }

    models: dict[str, dict] = {}
    # NOTE: v10 runs live at backtests/<TICKER>/xgb_v10/run_meta.json
    # (NOT backtests_xgb_v10/). Scan the unified backtests/ root first.
    search_dirs = [
        WORK / "backtests",
        WORK / "backtests_xgb_v9",
        WORK / "backtests_xgb_v8",
        WORK / "backtests_xgb_v7",
    ]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for meta_path in search_dir.rglob("run_meta.json"):
            try:
                meta = json.loads(meta_path.read_text())
                ticker = meta.get("ticker")
                pipeline = meta.get("pipeline_version", "")
                priority = priority_map.get(pipeline, 0)
                run_at = meta.get("run_at", "")

                if not ticker or priority == 0:
                    continue

                existing = models.get(ticker)
                # Promote if strictly higher priority, OR same priority but newer run.
                if (
                    existing is None
                    or existing["priority"] < priority
                    or (existing["priority"] == priority and existing.get("run_at", "") < run_at)
                ):
                    models[ticker] = {
                        "dir": meta_path.parent,
                        "meta": meta,
                        "priority": priority,
                        "pipeline": pipeline,
                        "run_at": run_at,
                    }
            except Exception as e:
                log.debug(f"Could not parse {meta_path}: {e}")

    # Summary by pipeline for visibility (helps catch BACKTEST != LIVE drift).
    by_pipeline: dict[str, int] = {}
    for info in models.values():
        by_pipeline[info["pipeline"]] = by_pipeline.get(info["pipeline"], 0) + 1
    log.info(
        f"Model discovery: {len(models)} tickers found across v7/v8/v9/v10 runs "
        f"— by pipeline: {by_pipeline}"
    )
    return models


# ---------------------------------------------------------------------------
# Feature building (lightweight inference version)
# ---------------------------------------------------------------------------
def _build_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a minimal but robust feature set for inference.
    Uses pandas-ta style indicators computed entirely from OHLCV.
    These approximate the core of the v7/v8 feature set.
    No lookahead: all indicators use .shift(1) on inputs.
    """
    f = df.copy()

    # Price history (shifted by 1 to avoid lookahead)
    f["open_lag1"] = f["open"].shift(1)
    f["high_lag1"] = f["high"].shift(1)
    f["low_lag1"] = f["low"].shift(1)
    f["close_lag1"] = f["close"].shift(1)
    f["volume_lag1"] = f["volume"].shift(1)

    c = f["close"].shift(1)  # no-lookahead close

    # Returns
    for w in [1, 2, 5, 10, 21, 63]:
        f[f"ret_{w}d"] = c.pct_change(w)

    # RSI
    for w in [7, 14, 21, 28]:
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(w).mean()
        loss = (-delta.clip(upper=0)).rolling(w).mean()
        rs = gain / (loss + 1e-9)
        f[f"rsi_{w}"] = 100 - (100 / (1 + rs))

    # EMA
    for w in [5, 10, 20, 50, 200]:
        f[f"ema_{w}"] = c.ewm(span=w, adjust=False).mean()
    f["ema_cross_5_20"] = (f["ema_5"] - f["ema_20"]) / (f["ema_20"].abs() + 1e-9)
    f["ema_cross_20_50"] = (f["ema_20"] - f["ema_50"]) / (f["ema_50"].abs() + 1e-9)
    f["ema_cross_50_200"] = (f["ema_50"] - f["ema_200"]) / (f["ema_200"].abs() + 1e-9)
    f["price_vs_ema200"] = (c - f["ema_200"]) / (f["ema_200"].abs() + 1e-9)

    # ATR
    hl = f["high"].shift(1) - f["low"].shift(1)
    hc = (f["high"].shift(1) - f["close"].shift(2)).abs()
    lc = (f["low"].shift(1) - f["close"].shift(2)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    for w in [14, 28]:
        f[f"atr_{w}"] = tr.rolling(w).mean()
    f["atr_ratio_14_28"] = f["atr_14"] / (f["atr_28"] + 1e-9)

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    f["macd"] = ema12 - ema26
    f["macd_signal"] = f["macd"].ewm(span=9, adjust=False).mean()
    f["macd_hist"] = f["macd"] - f["macd_signal"]

    # Bollinger Bands
    for w in [20]:
        mid = c.rolling(w).mean()
        std = c.rolling(w).std()
        f[f"bb_upper_{w}"] = mid + 2 * std
        f[f"bb_lower_{w}"] = mid - 2 * std
        f[f"bb_width_{w}"] = (f[f"bb_upper_{w}"] - f[f"bb_lower_{w}"]) / (mid.abs() + 1e-9)
        f[f"bb_pct_{w}"] = (c - f[f"bb_lower_{w}"]) / (f[f"bb_upper_{w}"] - f[f"bb_lower_{w}"] + 1e-9)

    # Volume features
    v = f["volume"].shift(1)
    f["vol_sma_20"] = v.rolling(20).mean()
    f["vol_ratio_20"] = v / (f["vol_sma_20"] + 1e-9)
    f["vol_sma_50"] = v.rolling(50).mean()

    # OBV approximation
    direction = np.sign(c.diff())
    f["obv"] = (direction * v).cumsum()
    f["obv_ema_20"] = f["obv"].ewm(span=20, adjust=False).mean()

    # Volatility
    for w in [10, 20, 60]:
        f[f"vol_{w}d"] = c.pct_change().rolling(w).std() * np.sqrt(252)

    # Day-of-week, month seasonality
    f["day_of_week"] = pd.to_datetime(f.index).dayofweek
    f["month"] = pd.to_datetime(f.index).month
    f["month_sin"] = np.sin(2 * np.pi * f["month"] / 12)
    f["month_cos"] = np.cos(2 * np.pi * f["month"] / 12)

    # Drawdown from rolling high
    rolling_max = c.rolling(60).max()
    f["feat_drawdown_60"] = (c - rolling_max) / (rolling_max.abs() + 1e-9)

    # Momentum
    for w in [5, 10, 21]:
        f[f"feat_momentum_{w}"] = c / (c.shift(w) + 1e-9) - 1

    return f


def build_inference_features(ticker: str, meta: dict) -> pd.DataFrame | None:
    """
    Download recent history for ticker and build features for inference.
    Returns a DataFrame with the last row ready for prediction.
    """
    try:
        df_raw = yf.download(ticker, period="300d", interval="1d", progress=False, auto_adjust=True)
        if df_raw.empty or len(df_raw) < 60:
            log.warning(f"{ticker}: insufficient data ({len(df_raw)} bars)")
            return None

        # Normalize column names
        df_raw.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df_raw.columns]
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = df_raw.columns.get_level_values(0).str.lower()

        df_raw.index = pd.to_datetime(df_raw.index)

        f = _build_basic_features(df_raw)

        # Phase A (2026-05-18): wire additional OHLCV-only feature modules from
        # the v10 backtest stack (alpha158, DFS, vol estimators, trading insight
        # parts 1-4, squeeze). Fail-soft: any module that errors is skipped.
        try:
            f = _extend_features_phase_a(f, ticker=ticker)
        except Exception as exc:
            log.warning(f"{ticker}: phase_a aggregator raised — keeping basic 50 ({exc})")

        f = f.replace([np.inf, -np.inf], np.nan)
        return f

    except Exception as e:
        log.warning(f"{ticker}: feature build failed — {e}")
        return None


# ---------------------------------------------------------------------------
# XGBoost inference
# ---------------------------------------------------------------------------
def _try_load_persisted_model(model_info: dict, features_df: pd.DataFrame) -> dict | None:
    """
    Attempt to use a v10/v8/v9 `refit_model.pkl` saved next to run_meta.json.

    Returns dict {prob, feature_count, model_path, feature_alignment} on success,
    or None if no usable persisted model exists (caller falls back to on-the-fly
    refit). Logs a WARN with the reason on any mismatch / failure.

    Feature alignment policy:
      - If >=80% of the saved model's feature_cols are present in features_df,
        proceed with the loaded model (zero-fill missing).
      - Otherwise WARN and return None so caller falls back.
    """
    pkl_path = Path(model_info["dir"]) / "refit_model.pkl"
    if not pkl_path.exists():
        return None
    try:
        import joblib  # local import, not all envs need it

        blob = joblib.load(pkl_path)
        model = blob.get("model")
        pmeta = blob.get("meta", {})
        feature_cols = pmeta.get("feature_cols", []) or []
        if model is None or not feature_cols:
            log.warning(
                f"{model_info['meta'].get('ticker','?')}: refit_model.pkl present "
                f"but model/feature_cols missing — falling back"
            )
            return None

        inf_row = features_df.iloc[[-1]]
        available = [c for c in feature_cols if c in inf_row.columns]
        missing = [c for c in feature_cols if c not in inf_row.columns]
        coverage = len(available) / max(len(feature_cols), 1)

        if coverage < 0.80:
            log.warning(
                f"{pmeta.get('ticker','?')}: persisted-model coverage "
                f"{coverage:.1%} ({len(available)}/{len(feature_cols)}) below 80%. "
                f"Missing examples: {missing[:5]}... — falling back to on-the-fly refit"
            )
            return None

        # Zero-fill missing columns to keep column order/shape.
        X = pd.DataFrame(index=inf_row.index)
        for c in feature_cols:
            X[c] = inf_row[c].values if c in inf_row.columns else 0.0
        X = X.fillna(0.0)

        prob = float(model.predict_proba(X.values)[0, 1])
        return {
            "prob": prob,
            "feature_count": len(feature_cols),
            "feature_alignment_pct": round(coverage * 100, 1),
            "model_path": str(pkl_path),
            "feature_hash": pmeta.get("feature_hash"),
            "missing_n": len(missing),
        }
    except Exception as exc:  # pylint: disable=broad-except
        log.warning(
            f"{model_info['meta'].get('ticker','?')}: refit_model.pkl load failed: "
            f"{exc} — falling back to on-the-fly refit"
        )
        return None


def run_inference(ticker: str, model_info: dict, features_df: pd.DataFrame) -> dict | None:
    """
    Predict probability for the most recent bar.

    Preferred path: load `refit_model.pkl` next to run_meta.json (persisted by
    backtest_xgb_v10.py from 2026-05-18 on) and predict directly. This avoids
    on-the-fly refit and matches BACKTEST vs LIVE.

    Fallback path: scout-prune-refit on the lightweight feature set (the prior
    behaviour). This fires whenever no pkl exists, the pkl's feature columns
    can't be aligned with live's basic feature builder (typical for v10 since
    live cannot reproduce all 969 features), or load fails.
    """
    meta = model_info["meta"]
    # Schema drift: v7/v8 use meta["strategy"]["prob_threshold"]; v10 uses a
    # flat meta["strategy"]="ORB" + nested meta["strategy_params"]["prob_threshold"].
    # Probe both shapes so threshold lookup works for every pipeline version.
    _strat = meta.get("strategy", {})
    _params = meta.get("strategy_params", {})
    if isinstance(_strat, dict):
        threshold = _strat.get("prob_threshold", _params.get("prob_threshold", 0.6))
    else:
        threshold = _params.get("prob_threshold", 0.6)
    top_k = meta.get("top_k", 50)
    pipeline = model_info["pipeline"]

    # ----- Preferred path: persisted model -----
    loaded = _try_load_persisted_model(model_info, features_df)
    if loaded is not None:
        prob = loaded["prob"]
        signal = 1 if prob >= threshold else 0
        log.info(
            f"{ticker}: [persisted] prob={prob:.3f} threshold={threshold} "
            f"signal={signal} features_used={loaded['feature_count']} "
            f"alignment={loaded['feature_alignment_pct']}% pipeline={pipeline}"
        )
        _today_iso = date.today().isoformat()
        _signal_ts_utc = datetime.utcnow().isoformat()
        # Persist EXACT inference row (same one passed to predict_proba inside
        # _try_load_persisted_model — reconstruct from features_df last bar,
        # restricted to the persisted model's feature_cols which is what got
        # zero-filled and scored).
        _persisted_row = features_df.iloc[[-1]]
        _persist_feature_row(
            ticker=ticker,
            feature_row=_persisted_row,
            prob=prob,
            threshold=threshold,
            signal=signal,
            pipeline=pipeline,
            model_run_dir=str(model_info["dir"]),
            feature_hash=loaded.get("feature_hash"),
            signal_ts_utc=_signal_ts_utc,
            today_iso=_today_iso,
        )
        return {
            "ticker": ticker,
            "date": _today_iso,
            "prob": round(prob, 4),
            "threshold": threshold,
            "signal": signal,
            "position_size": MAX_POSITION_NOTIONAL if signal == 1 else 0.0,
            "pipeline": pipeline,
            "model_run_dir": str(model_info["dir"]),
            "features_used": loaded["feature_count"],
            "inference_mode": "persisted_pkl",
            "feature_alignment_pct": loaded["feature_alignment_pct"],
            "feature_hash": loaded.get("feature_hash"),
            "generated_at": _signal_ts_utc,
        }

    # ----- Fallback path: on-the-fly scout-prune-refit -----
    try:
        import xgboost as xgb
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.calibration import CalibratedClassifierCV

        # Build label: 21-day forward return > 0 (matches training target)
        df = features_df.copy()
        df["fwd_ret"] = df["close"].pct_change(LABEL_EMBARGO_DAYS).shift(-LABEL_EMBARGO_DAYS)
        df["y"] = (df["fwd_ret"] > 0).astype(int)

        # Drop rows with NaN label or key features
        df = df.dropna(subset=["y", "rsi_14", "atr_14", "ema_200"])

        # Need at least 100 rows for a meaningful model
        if len(df) < 100:
            log.warning(f"{ticker}: only {len(df)} labeled rows, skipping inference")
            return None

        # Feature columns (exclude OHLCV and derived columns used for labels)
        exclude = {"open", "high", "low", "close", "volume", "y", "fwd_ret",
                   "open_lag1", "high_lag1", "low_lag1", "close_lag1", "volume_lag1"}
        feature_cols = [c for c in df.columns if c not in exclude]
        feature_cols = [c for c in feature_cols if df[c].notna().sum() > 50]

        # All-but-last row for training, last row for inference
        # Use time-series split: last 60 days = OOS, rest = train
        train_df = df.iloc[:-60].dropna(subset=feature_cols)
        inference_row = df.iloc[[-1]]

        if len(train_df) < 80:
            log.warning(f"{ticker}: insufficient training rows after split ({len(train_df)})")
            return None

        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df["y"]

        # Scout model to find important features
        scout = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            use_label_encoder=False,
            verbosity=0,
            random_state=42,
        )
        scout.fit(X_train, y_train)

        # Select top-K features
        importances = pd.Series(
            scout.feature_importances_, index=feature_cols
        ).sort_values(ascending=False)
        top_features = importances.head(min(top_k, len(feature_cols))).index.tolist()

        X_top = train_df[top_features].fillna(0)

        # Refit model on top features
        refit = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            use_label_encoder=False,
            verbosity=0,
            random_state=42,
        )
        refit.fit(X_top, y_train)

        # Predict on inference row
        X_inf = inference_row[top_features].fillna(0)
        if X_inf.isnull().all(axis=None):
            log.warning(f"{ticker}: inference row is all-NaN")
            return None

        prob = float(refit.predict_proba(X_inf)[0, 1])
        signal = 1 if prob >= threshold else 0

        log.info(
            f"{ticker}: prob={prob:.3f} threshold={threshold} signal={signal} "
            f"features_used={len(top_features)} pipeline={pipeline}"
        )

        _today_iso = date.today().isoformat()
        _signal_ts_utc = datetime.utcnow().isoformat()
        # X_inf is the EXACT single-row frame passed to refit.predict_proba().
        _persist_feature_row(
            ticker=ticker,
            feature_row=X_inf,
            prob=prob,
            threshold=threshold,
            signal=signal,
            pipeline=pipeline,
            model_run_dir=str(model_info["dir"]),
            feature_hash=None,  # compute via sorted col-name sha256
            signal_ts_utc=_signal_ts_utc,
            today_iso=_today_iso,
        )

        return {
            "ticker": ticker,
            "date": _today_iso,
            "prob": round(prob, 4),
            "threshold": threshold,
            "signal": signal,
            "position_size": MAX_POSITION_NOTIONAL if signal == 1 else 0.0,
            "pipeline": pipeline,
            "model_run_dir": str(model_info["dir"]),
            "features_used": len(top_features),
            "inference_mode": "on_the_fly_refit",
            "generated_at": _signal_ts_utc,
        }

    except Exception as e:
        log.error(f"{ticker}: inference failed — {e}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI argument parser. Supports optional --refresh-tickers subset mode.

    --refresh-tickers <CSV|@FILE>
        Run inference ONLY for the listed tickers (comma-separated, or @file
        with one ticker per line). New per-ticker results are merged INTO the
        existing {today}.json — non-listed tickers in the existing file are
        preserved. Use case (Quick-win B, 2026-05-22): re-score held positions
        mid-day for the signal-decay-exit check in cmd_open_trades without
        running the full universe pass.
    """
    p = argparse.ArgumentParser(description="Generate paper-trade signals.")
    p.add_argument(
        "--refresh-tickers",
        type=str,
        default=None,
        metavar="CSV|@FILE",
        help=(
            "Refresh only the listed tickers (comma-sep or @path/to/list.txt). "
            "Merges INTO existing today.json instead of overwriting."
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help=(
            "Bypass the market-day guard (run signal generation even on "
            "weekends / NYSE holidays). Default: skip on non-trading days."
        ),
    )
    return p.parse_args(argv)


def _resolve_refresh_tickers(arg: str | None) -> set[str] | None:
    """Resolve --refresh-tickers arg to a set of uppercase tickers, or None."""
    if not arg:
        return None
    raw: str
    if arg.startswith("@"):
        path = Path(arg[1:])
        if not path.exists():
            raise FileNotFoundError(f"--refresh-tickers file not found: {path}")
        raw = path.read_text()
        tokens = re.split(r"[,\s]+", raw)
    else:
        tokens = arg.split(",")
    tickers = {t.strip().upper() for t in tokens if t.strip()}
    return tickers or None


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    refresh_set = _resolve_refresh_tickers(args.refresh_tickers)

    today = date.today().isoformat()
    log.info(f"=== SIGNAL GENERATION {today} ===")

    # Market-day guard (defense-in-depth): skip on weekends + NYSE holidays.
    # Signal gen is a 10-15 min compute over 359 tickers — wasted on closed days.
    # --force flag bypasses the guard (for manual testing or stale-data refresh).
    if not getattr(args, "force", False):
        try:
            # Import the canonical helper from live_paper_trade (sibling script).
            from pathlib import Path
            import sys as _sys
            _scripts_dir = Path(__file__).parent
            if str(_scripts_dir) not in _sys.path:
                _sys.path.insert(0, str(_scripts_dir))
            from live_paper_trade import is_market_day  # type: ignore
            from datetime import datetime as _dt
            import pytz as _pytz
            _now = _dt.now(_pytz.timezone("America/New_York"))
            if not is_market_day(_now):
                log.info(
                    "[MARKET-CLOSED] signal-gen skipped: %s (%s) is not a NYSE "
                    "trading day (weekend or holiday). Use --force to bypass.",
                    _now.strftime("%Y-%m-%d"),
                    _now.strftime("%A"),
                )
                return 0
        except Exception as e:
            log.warning("Market-day guard import failed (%s) — proceeding without guard", e)

    if refresh_set is not None:
        log.info(
            f"[REFRESH-MODE] limiting to {len(refresh_set)} ticker(s): "
            f"{sorted(refresh_set)}"
        )

    models = discover_models()
    if not models:
        log.error("No models found in backtests_xgb_v7 or backtests_xgb_v8")
        return 1

    # In refresh mode, filter discovered models to the requested subset.
    if refresh_set is not None:
        missing = sorted(refresh_set - set(models.keys()))
        if missing:
            log.warning(
                f"[REFRESH-MODE] requested tickers without models: {missing}"
            )
        models = {t: m for t, m in models.items() if t in refresh_set}
        if not models:
            log.error("[REFRESH-MODE] no requested tickers have models — abort")
            return 1

    results = []
    errors = []

    for ticker, model_info in sorted(models.items()):
        try:
            log.info(f"Processing {ticker} ({model_info['pipeline']})...")
            features_df = build_inference_features(ticker, model_info["meta"])
            if features_df is None:
                errors.append(ticker)
                continue

            result = run_inference(ticker, model_info, features_df)
            if result is None:
                errors.append(ticker)
                continue

            results.append(result)

        except Exception as e:
            log.error(f"Unexpected error for {ticker}: {e}")
            errors.append(ticker)
            continue

    firing = [r for r in results if r["signal"] == 1]
    no_trade = [r for r in results if r["signal"] == 0]

    log.info(
        f"Signal generation complete: "
        f"{len(firing)} FIRING, {len(no_trade)} NO-TRADE, {len(errors)} ERRORS"
    )

    if errors:
        log.warning(f"Errors on: {errors}")

    # Write output file. In refresh mode, MERGE into existing file: refreshed
    # tickers overwrite, untouched ones stay. In full mode, overwrite entirely.
    output_path = SIGNALS_DIR / f"{today}.json"
    if refresh_set is not None and output_path.exists():
        try:
            with open(output_path) as f:
                existing = json.load(f)
        except Exception as _ex:
            log.warning(
                f"[REFRESH-MODE] could not load existing {output_path}: {_ex} — "
                "writing refreshed subset only"
            )
            existing = []
        by_ticker = {r["ticker"]: r for r in existing if isinstance(r, dict) and r.get("ticker")}
        for r in results:
            by_ticker[r["ticker"]] = r
        merged = list(by_ticker.values())
        with open(output_path, "w") as f:
            json.dump(merged, f, indent=2)
        log.info(
            f"Signals MERGED → {output_path} "
            f"({len(results)} refreshed, {len(merged)} total)"
        )
    else:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"Signals written → {output_path} ({len(results)} tickers)")

    # Print summary of firing tickers
    if firing:
        log.info("FIRING signals:")
        for r in sorted(firing, key=lambda x: x["prob"], reverse=True):
            log.info(f"  {r['ticker']}: prob={r['prob']:.3f} (threshold={r['threshold']})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
