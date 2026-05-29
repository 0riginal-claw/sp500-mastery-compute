"""seed_generator.py — Bayesian seed proposer for per-ticker championship search.

R5 task #71: fits `g(ticker_metadata, regime) -> seed_proposal` over the posterior
archive (~1,200 hypotheses tested across ~166 tickers) so championship_search can
move from brute-force enumeration of 5 seeds × 24 perturbations toward a smaller,
better-targeted candidate set.

Two heads share the same training set + feature engineering:

  CLF — multinomial LogisticRegression over the 5 (or 6) parent_seed labels,
        features = (sector_oh, mcap_bucket, ADV_bucket, vol_decile, beta_bucket).
        Interpretable, useful when posterior data is sparse.

  REG — LGBMRegressor predicting holdout_sharpe (HoldSR) from
        (ticker_metadata + variant params). More flexible; used to rank candidates.

Training set is assembled from `data/posteriors/<TICKER>.json` rows (each row has
parent_seed_id, perturb_params, and a result.holdout_sharpe). Ticker metadata is
joined from `lab.championship_metadata.enrich_metadata` (cached so we don't pay
the per-call cost 1200 times) + the cached universe_vol_distribution.json.

Public API:

  propose_seeds(ticker, n=5)            -> List[hypothesis_dict]
       Returns N best-predicted hypothesis dicts for `ticker` ranked by predicted
       HoldSR. Each dict passes `lab.knowledge.indicators.validate_test_unit`.
       Falls back to the canonical Mission-12 seed grid when (a) no model is
       fit yet, (b) the ticker has no usable metadata, or (c) the predicted
       ranking is degenerate (constant predictions).

  update_posterior(ticker, sap_id, observed_outcome) -> None
       Append a new (ticker, sap, outcome) observation to the rolling training
       set. Triggers an automatic re-fit when >=50 new observations have arrived
       since the last fit. Safe to call from championship_search's existing
       per-variant update_posterior path.

  model_info() -> dict
       Returns {"model_class", "training_set_size", "holdout_logloss",
                "holdout_spearman", "last_fit_utc", "seed_label_counts", ...}

  fit(force=False) -> dict
       (Re-)train both models from the on-disk posterior corpus. Returns the
       same dict shape as model_info().

Storage:

  data/seed_generator/training_set_<utc>.parquet   — assembled training rows
  data/seed_generator/posterior_<utc>.json         — model state + metrics
  data/seed_generator/_latest.json                 — pointer to most recent posterior

Models are serialised to per-fit pickles under data/seed_generator/models/<utc>/.

Honesty notes:
  * Posterior coverage is ~166 tickers × ~10 variants = ~1,660 rows; outcome
    distribution is heavily skewed toward HoldSR < 0 (most variants die). Sample
    weighting is applied: positive-HoldSR rows get higher weight so the classifier
    doesn't collapse onto the majority "this seed didn't work either" label.
  * The CLF predicts the BEST parent_seed for a ticker (1 row per ticker, label =
    argmax-HoldSR seed). The REG predicts continuous HoldSR per (ticker, variant).
  * Held-out evaluation uses stratified-by-ticker split (entire tickers go into
    holdout, not random rows) — prevents the model from memorising per-ticker
    quirks via within-ticker leakage.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

DRIVE_BASE = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive"
)
SP500_MASTERY = DRIVE_BASE / "AI-Tools/s&p500-ticker-mastery"
POSTERIOR_DIR = SP500_MASTERY / "data/posteriors"
SEEDGEN_DIR = SP500_MASTERY / "data/seed_generator"
SEEDGEN_MODELS_DIR = SEEDGEN_DIR / "models"
SEEDGEN_LATEST = SEEDGEN_DIR / "_latest.json"

REFIT_TRIGGER_N = 50            # re-fit after this many new observations
MIN_TRAINING_ROWS = 50           # below this, propose_seeds falls back to canonical grid

# Canonical seed labels (kept in sync with championship_search._SEED_TEMPLATES).
# GOV_AWARE_v2 is a separate label from GOV_AWARE so the classifier can
# distinguish boolean vs numeric alt-data variants. CATALYST_CONFLUENCE and
# CROSS_SYMBOL_REGIME were added by later R5 tasks and are now part of the
# active 8-seed pool that variant_generator walks; including them in SEED_LABELS
# lets the seed_generator propose them when posterior data warrants it.
SEED_LABELS = (
    "PURE_TECH",
    "ORB_MORNING",
    "VWAP_MTF",
    "GOV_AWARE",
    "GOV_AWARE_v2",
    "HYBRID_REGIME",
    "CATALYST_CONFLUENCE",
    "CROSS_SYMBOL_REGIME",
)

# Canonical perturb defaults — what the seed generator emits when it doesn't
# have enough data to pick a winning perturb combo. These match
# championship_search._stratified_perturb_samples()'s canonical[0].
DEFAULT_PERTURB = {
    "adx_thresh": 15, "ema_fast": 5, "ema_slow": 13, "donch": 10, "atr_mult": 1.0,
}

# Cross-asset gate variants (must match championship_search to keep variants
# valid when seed_generator emits them).
CROSS_ASSET_GATES = [
    "TRUE",
    "vix_term_struct > 1.0",
    "vix_term_struct < 1.0",
    "sector_rs_rank <= 3",
    "hyg_lqd_ratio > 0",
    "abs_spy_beta_60d > 0.5",
]


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering helpers
# ─────────────────────────────────────────────────────────────────────────────


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        fx = float(x)
        if np.isnan(fx) or np.isinf(fx):
            return None
        return fx
    except (TypeError, ValueError):
        return None


def _mcap_bucket(mcap: Optional[float]) -> str:
    """Coarse log-binned market cap bucket."""
    if mcap is None or mcap <= 0:
        return "unknown"
    # $B thresholds: micro <2, small 2-10, mid 10-50, large 50-200, mega 200+
    b = mcap / 1e9
    if b < 2: return "micro"
    if b < 10: return "small"
    if b < 50: return "mid"
    if b < 200: return "large"
    return "mega"


def _adv_bucket(adv: Optional[float]) -> str:
    """Coarse log-binned ADV bucket (shares per day)."""
    if adv is None or adv <= 0:
        return "unknown"
    if adv < 5e5: return "thin"
    if adv < 2e6: return "low"
    if adv < 1e7: return "mid"
    if adv < 5e7: return "high"
    return "huge"


def _beta_bucket(beta: Optional[float]) -> str:
    if beta is None:
        return "unknown"
    if beta < 0.5: return "low"
    if beta < 0.9: return "below_mkt"
    if beta < 1.1: return "near_mkt"
    if beta < 1.5: return "above_mkt"
    return "high"


def _vol_decile_bucket(d: Any) -> str:
    """Realized-vol decile (1=low, 10=high), or 'unknown'."""
    if d is None:
        return "unknown"
    try:
        di = int(d)
        if 1 <= di <= 10:
            return f"d{di}"
    except (TypeError, ValueError):
        pass
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Metadata loader (cached for the duration of a fit; on cache miss falls back
# to the universe_vol_distribution + sector map directly, avoiding the heavier
# yfinance load path used by championship_metadata.enrich_metadata when called
# 1,200 times in a hot loop).
# ─────────────────────────────────────────────────────────────────────────────


_META_CACHE: Dict[str, Dict[str, Any]] = {}
_VOL_DIST_CACHE: Optional[dict] = None


def _vol_dist() -> dict:
    global _VOL_DIST_CACHE
    if _VOL_DIST_CACHE is not None:
        return _VOL_DIST_CACHE
    p = SP500_MASTERY / "data/universe_vol_distribution.json"
    try:
        _VOL_DIST_CACHE = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        _VOL_DIST_CACHE = {}
    return _VOL_DIST_CACHE


def _decile_from_vol_dist(ticker: str) -> Optional[int]:
    """Look up ticker's vol decile straight from the cached distribution."""
    vd = _vol_dist()
    vols = vd.get("vols") or {}
    if not isinstance(vols, dict):
        return None
    v = vols.get(ticker.upper())
    fv = _safe_float(v)
    if fv is None:
        return None
    # Build decile bins on demand
    all_vols = sorted(_safe_float(x) for x in vols.values() if _safe_float(x) is not None)
    if not all_vols:
        return None
    # 10 deciles
    n = len(all_vols)
    rank = sum(1 for x in all_vols if x <= fv)
    decile = max(1, min(10, int(np.ceil(rank / n * 10))))
    return decile


def _ticker_metadata(ticker: str) -> Dict[str, Any]:
    """Return raw numeric metadata dict for `ticker`. Cached in-process.

    Tries `championship_metadata.enrich_metadata` first; on failure (most likely
    because the daily OHLC parquet is missing) falls back to vol-distribution +
    sector map only. Missing fields are returned as None — downstream feature
    encoders bucket None into 'unknown'.
    """
    tk = ticker.upper()
    if tk in _META_CACHE:
        return _META_CACHE[tk]
    meta: Dict[str, Any] = {
        "sector": None, "mcap": None, "adv_20d": None,
        "vol_decile": None, "beta": None,
    }
    try:
        import championship_metadata as _cm  # type: ignore
        m = _cm.enrich_metadata(tk, vol_dist=_vol_dist(), formatted=False)
        meta.update({
            "sector": m.get("sector"),
            "mcap": _safe_float(m.get("mcap")),
            "adv_20d": _safe_float(m.get("adv_20d")),
            "vol_decile": m.get("vol_decile"),
            "beta": _safe_float(m.get("beta")),
        })
    except Exception:
        # Fallback: sector from CSV + vol decile from cached distribution
        try:
            import championship_metadata as _cm  # type: ignore
            sectors = _cm.load_sector_map()
            meta["sector"] = sectors.get(tk)
        except Exception:
            pass
        meta["vol_decile"] = _decile_from_vol_dist(tk)
    _META_CACHE[tk] = meta
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# Training set assembly
# ─────────────────────────────────────────────────────────────────────────────


def _iter_posterior_files() -> List[Path]:
    if not POSTERIOR_DIR.exists():
        return []
    return sorted(POSTERIOR_DIR.glob("*.json"))


def assemble_training_set(verbose: bool = True) -> "Any":
    """Walk posteriors, join with metadata, return a pandas DataFrame.

    One row per (ticker × variant). Columns:
      ticker, sap_id, parent_seed_id, adx_thresh, ema_fast, ema_slow,
      donch, atr_mult, holdout_sharpe, win_rate, n_trades, pbo, dsr_prob,
      status_label, sector, mcap, adv_20d, vol_decile, beta
    """
    import pandas as pd
    rows: List[Dict[str, Any]] = []
    files = _iter_posterior_files()
    if verbose:
        print(f"[seed_generator] walking {len(files)} posterior files…", flush=True)
    for p in files:
        try:
            j = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        ticker = j.get("ticker") or p.stem.upper()
        meta = _ticker_metadata(ticker)
        history = j.get("history") or []
        for h in history:
            r = h.get("result") or {}
            params = h.get("perturb_params") or {}
            seed = h.get("parent_seed_id")
            if not seed:
                continue
            rows.append({
                "ticker": ticker,
                "sap_id": h.get("sap_id"),
                "parent_seed_id": seed,
                # Variant params (numeric, NaN-tolerant)
                "adx_thresh": _safe_float(params.get("adx_thresh")),
                "ema_fast": _safe_float(params.get("ema_fast")),
                "ema_slow": _safe_float(params.get("ema_slow")),
                "donch": _safe_float(params.get("donch")),
                "atr_mult": _safe_float(params.get("atr_mult")),
                # Outcome
                "holdout_sharpe": _safe_float(r.get("holdout_sharpe")),
                "full_sharpe": _safe_float(r.get("full_sharpe")),
                "win_rate": _safe_float(r.get("win_rate")),
                "n_trades": _safe_float(r.get("n_trades")),
                "pbo": _safe_float(r.get("pbo")),
                "dsr_prob": _safe_float(r.get("dsr_prob")),
                "status_label": h.get("status"),  # locked / survived / died
                # Metadata
                "sector": meta.get("sector"),
                "mcap": meta.get("mcap"),
                "adv_20d": meta.get("adv_20d"),
                "vol_decile": meta.get("vol_decile"),
                "beta": meta.get("beta"),
                # Derived buckets (computed up-front so downstream consumers
                # don't have to)
                "mcap_bucket": _mcap_bucket(meta.get("mcap")),
                "adv_bucket": _adv_bucket(meta.get("adv_20d")),
                "beta_bucket": _beta_bucket(meta.get("beta")),
                "vol_bucket": _vol_decile_bucket(meta.get("vol_decile")),
            })
    df = pd.DataFrame(rows)
    if verbose:
        print(f"[seed_generator] assembled {len(df)} rows from {len(files)} files; "
              f"unique tickers: {df['ticker'].nunique() if len(df) else 0}",
              flush=True)
    return df


def _verdict_from_csv() -> Dict[str, str]:
    """Join in REAL/SUSPECT/ARTIFACT verdicts from permutation_test CSVs when present.

    Returns: {sap_id: verdict}. Empty dict if neither CSV exists.
    """
    import csv
    out: Dict[str, str] = {}
    csvs = [
        DRIVE_BASE / "AI-Tools/reports/permutation_test_2000perm_2026-05-29.csv",
        DRIVE_BASE / "AI-Tools/reports/permutation_test_2026-05-29.csv",
    ]
    for p in csvs:
        if not p.exists():
            continue
        try:
            with p.open(encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    sap = (row.get("sap_id") or "").strip()
                    v = (row.get("verdict") or "").strip()
                    if sap and v and sap not in out:
                        out[sap] = v
        except OSError:
            continue
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Feature encoder (shared between CLF + REG)
# ─────────────────────────────────────────────────────────────────────────────

# Stable feature column order. One-hot for the categorical buckets; numeric
# columns passed through with NaN→0 fill (after a missing-indicator column).
_TICKER_CAT_COLS = ("sector", "mcap_bucket", "adv_bucket", "beta_bucket", "vol_bucket")
_VARIANT_NUM_COLS = ("adx_thresh", "ema_fast", "ema_slow", "donch", "atr_mult")


def _build_feature_frame(df_in, with_variant_params: bool):
    """Return (X_df, feature_names). Pure pandas, no sklearn pipeline so the
    fit artifacts are easy to inspect."""
    import pandas as pd
    parts = []
    for c in _TICKER_CAT_COLS:
        s = df_in[c].fillna("unknown").astype(str)
        oh = pd.get_dummies(s, prefix=c, dtype=float)
        parts.append(oh)
    if with_variant_params:
        for c in _VARIANT_NUM_COLS:
            v = df_in[c].astype(float)
            parts.append(v.fillna(v.median() if v.notna().any() else 0.0).to_frame(c))
            parts.append(v.isna().astype(float).to_frame(f"{c}_missing"))
    X = pd.concat(parts, axis=1)
    return X, list(X.columns)


# ─────────────────────────────────────────────────────────────────────────────
# Fit
# ─────────────────────────────────────────────────────────────────────────────


def _stratified_ticker_split(df, holdout_frac: float = 0.20, seed: int = 17):
    """Split rows by TICKER (not row) — entire tickers go into holdout. Returns
    (train_idx, test_idx) as numpy arrays."""
    rng = np.random.default_rng(seed)
    tickers = sorted(df["ticker"].dropna().unique().tolist())
    rng.shuffle(tickers)
    n_holdout = max(1, int(len(tickers) * holdout_frac))
    holdout_tickers = set(tickers[:n_holdout])
    train_mask = ~df["ticker"].isin(holdout_tickers)
    return df.index[train_mask].to_numpy(), df.index[~train_mask].to_numpy()


def _sample_weights(df) -> np.ndarray:
    """Weight rows so positive-HoldSR rows get more attention. Reduces label
    collapse onto majority 'this variant died' outcomes.

    Formula:
      base weight = 1.0
      +1.0 if HoldSR > 0
      +2.0 if HoldSR > 1.0
      +3.0 if HoldSR > 1.5  (these are the survivors we actually care about)
    """
    hs = df["holdout_sharpe"].fillna(-99.0).astype(float).to_numpy()
    w = np.ones_like(hs)
    w[hs > 0.0] += 1.0
    w[hs > 1.0] += 2.0
    w[hs > 1.5] += 3.0
    return w


def _fit_clf(df, train_idx, test_idx) -> Tuple[Any, Dict[str, Any]]:
    """Fit multinomial logistic on per-ticker argmax-HoldSR best-seed labels."""
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss

    # Aggregate to one row per ticker: label = parent_seed with max HoldSR.
    # If a ticker has no positive-HoldSR variant, label = the least-bad seed.
    train_df = df.loc[train_idx]
    test_df = df.loc[test_idx]

    def _per_ticker_best(d):
        # For each ticker pick the variant with the highest holdout_sharpe.
        # Tickers with all-NaN HoldSR get dropped (idxmax returns NaN there).
        # Ties → first seen.
        d2 = d.dropna(subset=["holdout_sharpe"])
        if len(d2) == 0:
            return d2
        best_idx = d2.groupby("ticker")["holdout_sharpe"].idxmax()
        best_idx = best_idx.dropna()
        return d2.loc[best_idx]

    train_best = _per_ticker_best(train_df)
    test_best = _per_ticker_best(test_df)

    # Keep only tickers that hit one of the known labels
    train_best = train_best[train_best["parent_seed_id"].isin(SEED_LABELS)]
    test_best = test_best[test_best["parent_seed_id"].isin(SEED_LABELS)]

    Xtr, feat_names = _build_feature_frame(train_best, with_variant_params=False)
    Xte, _ = _build_feature_frame(test_best, with_variant_params=False)
    Xte = Xte.reindex(columns=feat_names, fill_value=0.0)

    ytr = train_best["parent_seed_id"].astype(str).to_numpy()
    yte = test_best["parent_seed_id"].astype(str).to_numpy()

    # sklearn 1.8 removed the multi_class kwarg — it's auto-detected from y now.
    # Try the new signature first; if that fails (older sklearn), use the legacy
    # multi_class="multinomial" form.
    try:
        clf = LogisticRegression(
            solver="lbfgs",
            max_iter=500,
            C=1.0,
            class_weight="balanced",
        )
        clf.fit(Xtr.to_numpy(), ytr)
    except TypeError:
        clf = LogisticRegression(
            multi_class="multinomial",  # type: ignore[arg-type]
            solver="lbfgs",
            max_iter=500,
            C=1.0,
            class_weight="balanced",
        )
        clf.fit(Xtr.to_numpy(), ytr)

    metrics: Dict[str, Any] = {
        "train_rows": int(len(Xtr)),
        "test_rows": int(len(Xte)),
        "labels": list(clf.classes_),
    }
    if len(Xte) > 0 and len(set(yte)) >= 2:
        proba = clf.predict_proba(Xte.to_numpy())
        try:
            ll = log_loss(yte, proba, labels=list(clf.classes_))
        except ValueError as e:
            ll = float("nan")
            metrics["logloss_error"] = str(e)
        metrics["holdout_logloss"] = float(ll)
        # Top-1 accuracy
        pred = clf.classes_[np.argmax(proba, axis=1)]
        metrics["holdout_top1_accuracy"] = float(np.mean(pred == yte))
    else:
        metrics["holdout_logloss"] = float("nan")
        metrics["holdout_top1_accuracy"] = float("nan")

    artifact = {
        "model": clf,
        "feature_names": feat_names,
        "labels": list(clf.classes_),
    }
    return artifact, metrics


def _fit_reg(df, train_idx, test_idx) -> Tuple[Any, Dict[str, Any]]:
    """Fit LGBMRegressor on holdout_sharpe with seed + variant + metadata feats."""
    import pandas as pd
    from sklearn.metrics import mean_squared_error
    from scipy.stats import spearmanr

    train_df = df.loc[train_idx].copy()
    test_df = df.loc[test_idx].copy()

    # Filter to rows with a usable target
    train_df = train_df.dropna(subset=["holdout_sharpe"])
    test_df = test_df.dropna(subset=["holdout_sharpe"])

    # One-hot the parent_seed alongside the metadata buckets + variant numerics
    Xtr_meta, feat_names = _build_feature_frame(train_df, with_variant_params=True)
    Xte_meta, _ = _build_feature_frame(test_df, with_variant_params=True)

    seed_tr_oh = pd.get_dummies(train_df["parent_seed_id"].astype(str),
                                prefix="seed", dtype=float)
    seed_te_oh = pd.get_dummies(test_df["parent_seed_id"].astype(str),
                                prefix="seed", dtype=float)
    seed_te_oh = seed_te_oh.reindex(columns=seed_tr_oh.columns, fill_value=0.0)

    Xtr = pd.concat([Xtr_meta.reset_index(drop=True),
                     seed_tr_oh.reset_index(drop=True)], axis=1)
    Xte = pd.concat([Xte_meta.reset_index(drop=True),
                     seed_te_oh.reset_index(drop=True)], axis=1)
    feat_names = list(Xtr.columns)
    Xte = Xte.reindex(columns=feat_names, fill_value=0.0)

    ytr = train_df["holdout_sharpe"].astype(float).to_numpy()
    yte = test_df["holdout_sharpe"].astype(float).to_numpy()

    weights = _sample_weights(train_df)

    try:
        import lightgbm as lgb
        reg = lgb.LGBMRegressor(
            n_estimators=400,
            learning_rate=0.04,
            num_leaves=31,
            min_child_samples=8,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=17,
            n_jobs=1,
            verbose=-1,
        )
        reg.fit(Xtr.to_numpy(), ytr, sample_weight=weights)
        model_name = "LGBMRegressor"
    except Exception as e:
        print(f"[seed_generator] lightgbm fit failed ({e}); falling back to "
              f"GradientBoostingRegressor", flush=True)
        from sklearn.ensemble import GradientBoostingRegressor
        reg = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=17,
        )
        reg.fit(Xtr.to_numpy(), ytr, sample_weight=weights)
        model_name = "GradientBoostingRegressor"

    metrics: Dict[str, Any] = {
        "train_rows": int(len(Xtr)),
        "test_rows": int(len(Xte)),
        "model_name": model_name,
    }
    if len(Xte) >= 5:
        # Pass a DataFrame so LightGBM aligns by feature names (silences the
        # "X does not have valid feature names" warning).
        pred = reg.predict(Xte)
        rmse = float(np.sqrt(mean_squared_error(yte, pred)))
        try:
            rho, _p = spearmanr(yte, pred)
            metrics["holdout_spearman"] = float(rho) if rho == rho else float("nan")
            metrics["holdout_spearman_p"] = float(_p) if _p == _p else float("nan")
        except Exception:
            metrics["holdout_spearman"] = float("nan")
        metrics["holdout_rmse"] = rmse
    else:
        metrics["holdout_spearman"] = float("nan")
        metrics["holdout_rmse"] = float("nan")

    artifact = {
        "model": reg,
        "feature_names": feat_names,
        "model_name": model_name,
    }
    return artifact, metrics


def fit(force: bool = False, verbose: bool = True) -> Dict[str, Any]:
    """Assemble training set + fit both heads. Persists everything under
    `data/seed_generator/`. Returns a model_info()-shaped dict.

    `force` is currently a no-op (always refits) but reserved for incremental-fit
    semantics later.
    """
    _ = force  # unused for now
    SEEDGEN_DIR.mkdir(parents=True, exist_ok=True)
    SEEDGEN_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = assemble_training_set(verbose=verbose)
    if len(df) == 0:
        info = {
            "status": "no_data",
            "training_set_size": 0,
            "last_fit_utc": None,
        }
        return info

    # Join in permutation verdicts (best-effort)
    verdicts = _verdict_from_csv()
    df["perm_verdict"] = df["sap_id"].map(lambda s: verdicts.get(str(s), None))

    # Persist training set (parquet preferred; CSV fallback)
    utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    parquet_path = SEEDGEN_DIR / f"training_set_{utc}.parquet"
    # Stable name too (matches the brief's expected output)
    parquet_stable = SEEDGEN_DIR / "training_set_2026-05-29.parquet"
    try:
        df.to_parquet(parquet_path, index=False)
        df.to_parquet(parquet_stable, index=False)
        train_path = str(parquet_path)
    except Exception as e:
        print(f"[seed_generator] parquet write failed ({e}); using CSV", flush=True)
        train_path = str(parquet_path.with_suffix(".csv"))
        df.to_csv(train_path, index=False)

    # Split + fit
    train_idx, test_idx = _stratified_ticker_split(df, holdout_frac=0.20)
    if verbose:
        print(f"[seed_generator] split: train={len(train_idx)} rows / "
              f"{df.loc[train_idx, 'ticker'].nunique()} tickers, "
              f"holdout={len(test_idx)} rows / "
              f"{df.loc[test_idx, 'ticker'].nunique()} tickers", flush=True)

    clf_art, clf_metrics = _fit_clf(df, train_idx, test_idx)
    reg_art, reg_metrics = _fit_reg(df, train_idx, test_idx)

    # Pick "best" model on a normalised score: -logloss (CLF) vs spearman (REG).
    # In practice both heads are USED at inference time (the CLF picks WHICH
    # seeds to consider for the ticker, the REG ranks them). So "best" here is
    # informational only.
    best_head = "regressor" if (reg_metrics.get("holdout_spearman") == reg_metrics.get("holdout_spearman")  # not NaN
                                 and reg_metrics["holdout_spearman"] > 0.0) else "classifier"

    # Persist artifacts
    model_dir = SEEDGEN_MODELS_DIR / utc
    model_dir.mkdir(parents=True, exist_ok=True)
    with (model_dir / "clf.pkl").open("wb") as fh:
        pickle.dump(clf_art, fh)
    with (model_dir / "reg.pkl").open("wb") as fh:
        pickle.dump(reg_art, fh)

    info: Dict[str, Any] = {
        "status": "fit_ok",
        "last_fit_utc": utc,
        "training_set_size": int(len(df)),
        "training_set_path": train_path,
        "training_set_stable_path": str(parquet_stable),
        "n_tickers": int(df["ticker"].nunique()),
        "n_seed_labels": int(df["parent_seed_id"].nunique()),
        "seed_label_counts": (
            df["parent_seed_id"].value_counts().to_dict()
        ),
        "outcome_dist": {
            "n_pos_holdsr": int((df["holdout_sharpe"].astype(float) > 0).sum()),
            "n_holdsr_gt_1": int((df["holdout_sharpe"].astype(float) > 1.0).sum()),
            "n_holdsr_gt_1p5": int((df["holdout_sharpe"].astype(float) > 1.5).sum()),
            "median_holdsr": float(df["holdout_sharpe"].astype(float).median()) if df["holdout_sharpe"].notna().any() else None,
        },
        "clf_metrics": clf_metrics,
        "reg_metrics": reg_metrics,
        "best_head": best_head,
        "model_dir": str(model_dir),
        "clf_model_class": "LogisticRegression(multinomial, balanced)",
        "reg_model_class": reg_metrics.get("model_name", "unknown"),
    }

    # Posterior JSON
    posterior_path = SEEDGEN_DIR / f"posterior_{utc}.json"
    posterior_path.write_text(json.dumps(
        {k: v for k, v in info.items()},
        indent=2, default=str,
    ))
    # Pointer
    SEEDGEN_LATEST.write_text(json.dumps({
        "last_fit_utc": utc,
        "model_dir": str(model_dir),
        "posterior_path": str(posterior_path),
        "training_set_path": train_path,
    }, indent=2))
    info["posterior_path"] = str(posterior_path)
    if verbose:
        print(f"[seed_generator] fit complete; logloss={clf_metrics.get('holdout_logloss')}, "
              f"spearman={reg_metrics.get('holdout_spearman')}", flush=True)
    return info


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────


_LOADED: Dict[str, Any] = {"clf": None, "reg": None, "info": None}


def _load_latest() -> Dict[str, Any]:
    """Load the most recent fit artifacts into the module-level cache. Returns
    the info dict. Returns {"status": "no_model"} if nothing is fit yet."""
    if _LOADED["info"] is not None:
        return _LOADED["info"]
    if not SEEDGEN_LATEST.exists():
        return {"status": "no_model"}
    try:
        latest = json.loads(SEEDGEN_LATEST.read_text())
    except (OSError, json.JSONDecodeError):
        return {"status": "no_model"}
    model_dir = Path(latest.get("model_dir", ""))
    posterior_path = Path(latest.get("posterior_path", ""))
    if not model_dir.exists():
        return {"status": "no_model"}
    try:
        with (model_dir / "clf.pkl").open("rb") as fh:
            _LOADED["clf"] = pickle.load(fh)
        with (model_dir / "reg.pkl").open("rb") as fh:
            _LOADED["reg"] = pickle.load(fh)
    except Exception as e:
        return {"status": "load_failed", "error": str(e)}
    if posterior_path.exists():
        try:
            _LOADED["info"] = json.loads(posterior_path.read_text())
        except (OSError, json.JSONDecodeError):
            _LOADED["info"] = {"status": "info_unreadable"}
    else:
        _LOADED["info"] = {"status": "info_missing"}
    return _LOADED["info"]


def model_info() -> Dict[str, Any]:
    info = _load_latest()
    out = dict(info) if isinstance(info, dict) else {"status": "unknown"}
    if out.get("status") not in ("no_model", "load_failed", None):
        # Surface the headline fields the brief asked for
        out.setdefault("model_class",
                       f"{out.get('clf_model_class', '?')} + {out.get('reg_model_class', '?')}")
        out.setdefault("holdout_logloss",
                       (out.get("clf_metrics") or {}).get("holdout_logloss"))
        out.setdefault("holdout_spearman",
                       (out.get("reg_metrics") or {}).get("holdout_spearman"))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Inference: propose_seeds
# ─────────────────────────────────────────────────────────────────────────────

# Canonical perturbation pool to score per seed when ranking. Same shape as
# championship_search._stratified_perturb_samples canonical[*]. Each tuple is a
# distinct (adx_thresh, ema_fast, ema_slow, donch, atr_mult).
_PERTURB_POOL = [
    {"adx_thresh": 15, "ema_fast": 5,  "ema_slow": 13, "donch": 10, "atr_mult": 1.0},
    {"adx_thresh": 20, "ema_fast": 9,  "ema_slow": 21, "donch": 20, "atr_mult": 1.5},
    {"adx_thresh": 22, "ema_fast": 13, "ema_slow": 34, "donch": 20, "atr_mult": 1.5},
    {"adx_thresh": 25, "ema_fast": 9,  "ema_slow": 21, "donch": 40, "atr_mult": 2.0},
    {"adx_thresh": 20, "ema_fast": 21, "ema_slow": 55, "donch": 20, "atr_mult": 2.0},
    {"adx_thresh": 15, "ema_fast": 9,  "ema_slow": 21, "donch": 10, "atr_mult": 1.0},
]


def _candidate_rows(ticker: str):
    """Build a candidate DataFrame: every (seed × perturb) combo for this ticker."""
    import pandas as pd
    meta = _ticker_metadata(ticker)
    rows = []
    for seed in SEED_LABELS:
        for p in _PERTURB_POOL:
            rows.append({
                "ticker": ticker.upper(),
                "parent_seed_id": seed,
                "adx_thresh": p["adx_thresh"],
                "ema_fast": p["ema_fast"],
                "ema_slow": p["ema_slow"],
                "donch": p["donch"],
                "atr_mult": p["atr_mult"],
                "sector": meta.get("sector"),
                "mcap_bucket": _mcap_bucket(meta.get("mcap")),
                "adv_bucket": _adv_bucket(meta.get("adv_20d")),
                "beta_bucket": _beta_bucket(meta.get("beta")),
                "vol_bucket": _vol_decile_bucket(meta.get("vol_decile")),
            })
    return pd.DataFrame(rows)


def _score_candidates(cand_df) -> "Any":
    """Apply the regressor to score every candidate row's predicted HoldSR."""
    import pandas as pd
    info = _load_latest()
    if info.get("status") in ("no_model", "load_failed", None):
        return None
    reg_art = _LOADED.get("reg")
    if reg_art is None:
        return None

    X_meta, _ = _build_feature_frame(cand_df, with_variant_params=True)
    # Add seed one-hot
    seed_oh = pd.get_dummies(cand_df["parent_seed_id"].astype(str),
                             prefix="seed", dtype=float)
    X = pd.concat([X_meta.reset_index(drop=True),
                   seed_oh.reset_index(drop=True)], axis=1)
    feat_names = reg_art["feature_names"]
    X = X.reindex(columns=feat_names, fill_value=0.0)

    try:
        pred = reg_art["model"].predict(X)
    except Exception as e:
        print(f"[seed_generator] regressor predict failed: {e}", flush=True)
        return None
    out = cand_df.copy()
    out["predicted_holdsr"] = pred
    return out


def _hypothesis_for_seed(ticker: str, seed: str, perturb: Dict[str, Any],
                         seq: int = 1) -> Dict[str, Any]:
    """Render a runnable hypothesis dict from (ticker, seed, perturb).

    Imports championship_search lazily so we reuse the canonical templates.
    Falls back to a built-in copy of the PURE_TECH template if the import fails.
    """
    try:
        import championship_search as _cs  # type: ignore
        templates = {t["seed_id"]: t for t in _cs._SEED_TEMPLATES}
        tf_stack_by_seed = _cs._TF_STACK_BY_SEED
        xa_pool = _cs._CROSS_ASSET_GATE_VARIANTS
        _fmt = _cs._format_template
        _validate = _cs.validate_test_unit
    except Exception:
        # Minimal embedded fallback so propose_seeds() can still run in isolation
        templates = {
            "PURE_TECH": {
                "seed_id": "PURE_TECH",
                "side": "long",
                "regime_gate": "1d.ADX(14) > {adx_thresh}",
                "bias_filter": "1d.EMA({ema_fast}) > 1d.EMA({ema_slow})",
                "trigger": "Close > VWAP AND Close > Donchian_UP({donch}) AND 15min.RSI(14) > 50",
                "confirmation": "Volume > 1.2 * SMA(Volume, 20)",
                "timing": "RSI(14) > 50",
                "exit": "{atr_mult} * ATR(14) trailing stop",
                "no_trade": "1d.ChopIdx(14) > 62",
                "data_sources": ["embedded_fallback_template"],
            },
        }
        tf_stack_by_seed = {"PURE_TECH": ["5min", "15min", "1d"]}
        xa_pool = CROSS_ASSET_GATES
        def _fmt(tmpl, p):
            out = {}
            for k, v in tmpl.items():
                if isinstance(v, str):
                    try:
                        out[k] = v.format(**p)
                    except (KeyError, IndexError):
                        out[k] = v
                else:
                    out[k] = v
            return out
        from knowledge.indicators import validate_test_unit as _validate

    tmpl = templates.get(seed) or templates["PURE_TECH"]
    rendered = _fmt(tmpl, perturb)
    tf_stack = list(tf_stack_by_seed.get(seed, ["1d"]))
    seen: List[str] = []
    for t in tf_stack:
        if t not in seen:
            seen.append(t)
    tf_stack = seen
    primary_tf = tf_stack[0] if tf_stack else "1d"
    # Cross-asset gate: pick idx 0 (noop) by default for proposed seeds. The
    # downstream variant_generator will swap in real XA-gates when it fans these
    # out for actual testing — but the proposed seed itself stays back-compat.
    xa_idx = 0
    cross_asset_gate = xa_pool[xa_idx]

    sap_id = f"SAP-{ticker.upper()}-PROPOSED-{seq:03d}"
    hyp = {
        "id": sap_id,
        "name": f"{seed} @ ADX>{perturb['adx_thresh']} EMA({perturb['ema_fast']}/{perturb['ema_slow']}) "
                f"Donch{perturb['donch']} {perturb['atr_mult']}xATR TF={'>'.join(tf_stack)} (proposed)",
        "thesis": (
            f"Bayesian-proposed seed for {ticker.upper()} from "
            f"seed_generator.propose_seeds (parent seed {seed}, "
            f"perturb={perturb})."
        ),
        "parent_seed_id": seed,
        "perturb_params": dict(perturb),
        "regime_gate": rendered.get("regime_gate", "TRUE"),
        "bias_filter": rendered.get("bias_filter", "TRUE"),
        "trigger": rendered.get("trigger", "FALSE"),
        "confirmation": rendered.get("confirmation", "TRUE"),
        "timing": rendered.get("timing", "TRUE"),
        "exit": rendered.get("exit", "FALSE"),
        "no_trade": rendered.get("no_trade", "FALSE"),
        "side": rendered.get("side", "long"),
        "cross_asset_gate": cross_asset_gate,
        "cost": "5bps_per_side",
        "universe": f"single_ticker:{ticker.upper()}",
        "timeframe": primary_tf,
        "timeframe_stack": tf_stack,
        "data_sources": list(tmpl.get("data_sources", [])),
        "source": "seed_generator",
    }
    if "child_hypotheses" in rendered:
        hyp["child_hypotheses"] = rendered["child_hypotheses"]
    if "alt_data_overlay" in rendered:
        hyp["alt_data_overlay"] = rendered["alt_data_overlay"]
    gate = _validate(hyp)
    if not gate.get("ok"):
        # Should not happen for canonical templates, but degrade visibly rather
        # than silently emitting an invalid hypothesis.
        hyp["_validate_failed"] = gate.get("reason")
    return hyp


def _fallback_propose(ticker: str, n: int) -> List[Dict[str, Any]]:
    """Canonical Mission-12 seed grid, used when no model is fit yet."""
    out: List[Dict[str, Any]] = []
    seeds_in_order = [s for s in SEED_LABELS if s != "GOV_AWARE_v2"][:5]
    # Reorder so PURE_TECH and VWAP_MTF come first (empirically the top survivors
    # in the championship roll-up — see reports/championship_roll_up_2026-05-29.md).
    priority_seeds = ["VWAP_MTF", "PURE_TECH", "HYBRID_REGIME", "GOV_AWARE", "ORB_MORNING"]
    seed_ord: List[str] = []
    for s in priority_seeds:
        if s in seeds_in_order and s not in seed_ord:
            seed_ord.append(s)
    for s in seeds_in_order:
        if s not in seed_ord:
            seed_ord.append(s)

    seq = 0
    for s in seed_ord:
        if len(out) >= n:
            break
        seq += 1
        out.append(_hypothesis_for_seed(ticker, s, DEFAULT_PERTURB, seq=seq))
    while len(out) < n:
        seq += 1
        # Cycle through perturb pool with PURE_TECH
        p = _PERTURB_POOL[(seq - 1) % len(_PERTURB_POOL)]
        out.append(_hypothesis_for_seed(ticker, "PURE_TECH", p, seq=seq))
    return out[:n]


def propose_seeds(ticker: str, n: int = 5) -> List[Dict[str, Any]]:
    """Return N hypothesis dicts ranked by predicted HoldSR. Validated."""
    info = _load_latest()
    if info.get("status") in ("no_model", "load_failed", None):
        return _fallback_propose(ticker, n)

    cand = _candidate_rows(ticker)
    scored = _score_candidates(cand)
    if scored is None or len(scored) == 0:
        return _fallback_propose(ticker, n)

    # Sort descending by predicted HoldSR
    scored = scored.sort_values("predicted_holdsr", ascending=False).reset_index(drop=True)

    # Detect degenerate predictions (all identical) → fall back
    if scored["predicted_holdsr"].nunique() <= 1:
        return _fallback_propose(ticker, n)

    # Walk top rows, taking at most 2 from any single parent_seed_id so the N
    # proposals are diverse (don't all collapse to "VWAP_MTF wins everything").
    out: List[Dict[str, Any]] = []
    per_seed_count: Dict[str, int] = {}
    seq = 0
    for _i, row in scored.iterrows():
        if len(out) >= n:
            break
        seed = str(row["parent_seed_id"])
        if per_seed_count.get(seed, 0) >= 2:
            continue
        seq += 1
        hyp = _hypothesis_for_seed(
            ticker, seed,
            {k: row[k] for k in _VARIANT_NUM_COLS},
            seq=seq,
        )
        hyp["predicted_holdsr"] = float(row["predicted_holdsr"])
        out.append(hyp)
        per_seed_count[seed] = per_seed_count.get(seed, 0) + 1

    # Fill remaining slots with fallback if diversity-throttled too aggressively
    while len(out) < n:
        seq += 1
        out.append(_hypothesis_for_seed(ticker,
                                        "PURE_TECH",
                                        _PERTURB_POOL[(seq - 1) % len(_PERTURB_POOL)],
                                        seq=seq))
    return out[:n]


# ─────────────────────────────────────────────────────────────────────────────
# update_posterior
# ─────────────────────────────────────────────────────────────────────────────

_NEW_OBS_SINCE_FIT_PATH = SEEDGEN_DIR / "new_obs_counter.json"


def _bump_new_obs() -> int:
    SEEDGEN_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    if _NEW_OBS_SINCE_FIT_PATH.exists():
        try:
            n = int(json.loads(_NEW_OBS_SINCE_FIT_PATH.read_text()).get("n", 0))
        except (OSError, json.JSONDecodeError, ValueError):
            n = 0
    n += 1
    try:
        _NEW_OBS_SINCE_FIT_PATH.write_text(json.dumps({"n": n}))
    except OSError:
        pass
    return n


def _reset_new_obs():
    try:
        _NEW_OBS_SINCE_FIT_PATH.write_text(json.dumps({"n": 0}))
    except OSError:
        pass


def update_posterior(ticker: str, sap_id: str,
                     observed_outcome: Dict[str, Any]) -> None:
    """Append a new observation to the rolling training set.

    Does NOT touch the underlying `data/posteriors/<T>.json` — that file is owned
    by championship_search.update_posterior. We just track the COUNT of new
    obs and trigger an automatic re-fit when REFIT_TRIGGER_N is crossed.

    `observed_outcome` is a result dict shaped like a championship_search row
    (must contain at least `parent_seed_id` and `holdout_sharpe`). We log it
    to an audit JSONL for traceability.
    """
    SEEDGEN_DIR.mkdir(parents=True, exist_ok=True)
    audit = SEEDGEN_DIR / "obs_audit.jsonl"
    rec = {
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ticker": ticker.upper(),
        "sap_id": sap_id,
        "parent_seed_id": observed_outcome.get("parent_seed_id"),
        "holdout_sharpe": _safe_float(observed_outcome.get("holdout_sharpe")),
        "status": observed_outcome.get("status"),
    }
    try:
        with audit.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
    except OSError as e:
        print(f"[seed_generator] obs audit write failed: {e}", flush=True)

    n = _bump_new_obs()
    if n >= REFIT_TRIGGER_N:
        try:
            fit(verbose=False)
            _reset_new_obs()
        except Exception as e:
            print(f"[seed_generator] auto re-fit failed: {e}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _cli():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "info", "propose", "smoke"])
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()
    if args.cmd == "fit":
        info = fit(verbose=True)
        print(json.dumps({k: v for k, v in info.items()
                          if k not in ("seed_label_counts",)},
                         indent=2, default=str))
    elif args.cmd == "info":
        print(json.dumps(model_info(), indent=2, default=str))
    elif args.cmd == "propose":
        props = propose_seeds(args.ticker, args.n)
        print(json.dumps(props, indent=2, default=str))
    elif args.cmd == "smoke":
        # End-to-end: fit + propose 5 for AAPL + validate each
        from knowledge.indicators import validate_test_unit
        info = fit(verbose=True)
        print("fit_info:", json.dumps({k: v for k, v in info.items()
                                       if k not in ("seed_label_counts",)},
                                       indent=2, default=str))
        props = propose_seeds(args.ticker, args.n)
        print(f"\n=== {args.n} proposals for {args.ticker} ===")
        for i, p in enumerate(props, 1):
            gate = validate_test_unit(p)
            print(f"\n[{i}] {p.get('id')}  seed={p.get('parent_seed_id')}  "
                  f"pred_HoldSR={p.get('predicted_holdsr', 'n/a')}  "
                  f"gate_ok={gate.get('ok')}")
            print(f"    perturb={p.get('perturb_params')}")
            print(f"    trigger={p.get('trigger')[:80] if p.get('trigger') else ''}")


if __name__ == "__main__":
    _cli()
