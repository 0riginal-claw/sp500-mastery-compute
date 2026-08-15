# XGBoost full-feature utilization roadmap

**Author:** 2026-05-20 / patch session
**Target:** raise XGBoost feature utilization from ~4% (50 of 1485) toward 60%+ (>=800 features splitting trees) without OOS metric regression beyond +/-5%.

## Problem

`backtest_xgb_v10.py` builds ~1485-1503 features per ticker but the model only sees 50 of them. The scout pre-filter + small fold size (1200 rows) + top-K cap eliminated 96% of engineered features before the final XGBoost ever saw them.

## Stage A — env-gated top-K bypass + capacity bump (LANDED 2026-05-20, OPT-IN)

**Trigger:** `XGB_NO_TOPK=1`

**Changes in `scripts/backtest_xgb_v10.py`:**
- `_resolve_top_k()` returns `n_features` when `XGB_NO_TOPK=1`.
- `_xgb_base_params("final"|"persist")` bumps capacity when `XGB_NO_TOPK=1`:
  - `n_estimators`: 100 -> 500
  - `max_depth`: 4 -> 6
  - `max_leaves`: 31 -> 63
  - `reg_alpha`: 0.01 -> 0.1  (stronger L1 sparsity)
  - `reg_lambda`: 1.0 -> 2.0  (stronger L2 smoothing)
- `_xgb_fit_kwargs()` bumps EarlyStopping rounds 10 -> 20.
- Scout `imp>0` filter retained by default (acts as natural feature regularization). `XGB_KEEP_ZERO_IMP=1` removes that filter (true full bypass — observed to over-fit small folds).

**Usage:**
```bash
AUTO_CLOUD_DISPATCH=0 XGB_NO_TOPK=1 python scripts/backtest_xgb_v10.py \
  --ticker AAPL --output-dir backtests_xgb_v10/AAPL_notopk --strategy ML_XGB_v10_notopk
```

**Smoke results** (4 tickers, baseline vs Stage A, see `logs/auto_solve/feature_full_util_repolocal_2026-05-20.md`):

| ticker | baseline PF | Stage A PF | delta | baseline WR | Stage A WR | features_in_trees baseline | features_in_trees Stage A |
|---|---|---|---|---|---|---|---|
| AAPL | 1.376 | 2.351 | +71% | 0.480 | 0.614 | 29 | 83 |
| MSFT | 1.622 | 1.050 | -35% (REGRESSED) | 0.542 | 0.425 | 70 | 85 |
| XOM  | 0.848 | 0.971 | +15% | 0.359 | 0.400 | 14 | 38 |
| SPY  | n/a (no run_meta produced) | n/a | n/a | n/a | n/a | n/a | n/a |

**Verdict:**
- features_in_trees averaged ~70 across Stage A runs — far below the >=800 target.
- Per-ticker behavior is mixed (AAPL +71%, MSFT -35%). MSFT regression (>5%) means Stage A is NOT a global default win.
- Stage A is therefore exposed as an **opt-in env flag**, not a default. Per-ticker mastery sweep can decide based on its prior best.
- The ~70-feature mean tells us XGBoost early stopping + tree budget naturally restrict utilization no matter how many features pass through. The scout pre-filter at ~150 features is closer to the post-pruning sweet spot.

## Stage B — stacked submodel ensemble (NOT YET IMPLEMENTED)

**Trigger:** `XGB_STACKED=1`  (placeholder — code not written)

**Design:**
- Identify feature groups by name prefix patterns: `alpha158_*`, `dfs_*`, `gtrends_*`, `edgar_*`, `gov_trades_*`, `microstructure_*`, `vol_*`, `macro_*`, `alpaca_*`.
- Train a per-group XGBClassifier on each group's ~100-200 features (no p>n problem within a group).
- Use walk-forward OOF predictions from each submodel as 8-12 meta-features.
- Final prediction = meta-XGB on submodel preds.
- Aggregate `features_used_in_trees` across submodels = sum.

**Deferred reason:** Stage A's mixed result (1 ticker regressed) suggests we should validate Stage C first (cross-sectional pool dramatically changes the n>>p balance) before adding submodel complexity.

## Stage C — cross-sectional ticker pooling (BUILT, NOT YET SMOKED 2026-05-20)

**Script:** `scripts/backtest_xgb_v10_xsec.py`

**Trigger:** new entry-point CLI (no env flag — pass `--tickers` list)

**Design:**
- Pool N tickers' rows along axis=0. With 10 tickers x ~1200 rows = ~12,000 rows; with 500 tickers x ~1200 = ~600,000 rows.
- Add `ticker_cat` column as `pd.Categorical`. Use XGBoost `enable_categorical=True` (xgboost >= 1.6 native cat — no one-hot blow-up).
- Walk-forward by DATE on the panel. Each fold trains on T months across ALL tickers, OOS on next T months across ALL tickers.
- After fold inference, split predictions back by ticker for per-ticker `bml.simulate` -> per-ticker metrics.
- Shares `_xgb_base_params`, `_xgb_fit_kwargs`, `_resolve_top_k` with v10 (so `XGB_NO_TOPK=1` works in Stage C too).

**Smoke usage:**
```bash
AUTO_CLOUD_DISPATCH=0 XGB_NO_TOPK=1 python scripts/backtest_xgb_v10_xsec.py \
  --tickers AAPL,MSFT,GOOG,META,NVDA,TSLA,AMZN,JPM,JNJ,XOM \
  --output-dir backtests_xgb_v10_xsec/smoke_2026-05-20 \
  --strategy ML_XGB_v10_xsec_smoke
```

**Why Stage C has the best shot:**
With 10 tickers x 1200 rows = 12k rows >> 1485 features, the p>>n problem flips. With 500 tickers = 600k rows, XGBoost trees have so much data that they will split on far more features even without any cap bypass. This is the **architecturally correct** fix vs Stage A's "remove the cap and hope for the best".

## Stage D — cloud-routed 500-ticker re-mastery sweep (BLOCKED ON STAGES A/C SMOKE)

**Trigger:** bulk enqueue via `cloud_dispatch.enqueue_job(...)`

**Design (when ready):**
```python
from cloud_dispatch import enqueue_job
for ticker in SP500_TICKERS:
    enqueue_job(
        ticker=ticker,
        strategy="ML_XGB_v10_fullutil",
        script="backtest_xgb_v10.py",
        env={"XGB_NO_TOPK": "1"},  # or "0" depending on per-ticker prior best
    )
```

Tail `logs/cloud_dispatch.log` for the first 10 completions before bulk-enqueuing the remaining 490.

**Blocked because:** Stage A smoke showed mixed per-ticker results — a global `XGB_NO_TOPK=1` sweep would hurt tickers like MSFT. Need either Stage C results (cross-sectional may be globally better) OR per-ticker A/B router using `per_ticker_best.parquet`.

## Backup + rollback

- Pre-edit backup: `s&p500-ticker-mastery/backups/backtest_xgb_v10-pre-fullutil-2026-05-20/backtest_xgb_v10.py`
- Rollback: `cp s&p500-ticker-mastery/backups/backtest_xgb_v10-pre-fullutil-2026-05-20/backtest_xgb_v10.py s&p500-ticker-mastery/scripts/backtest_xgb_v10.py`
- Stage A changes are **opt-in** — leaving env vars unset = identical behavior to pre-patch. No revert needed for the default code path.

## Env flag matrix

| flag | default | effect |
|---|---|---|
| `XGB_NO_TOPK` | `0` | Set `1` to bypass top-K cap (use n_features) + bump model capacity. |
| `XGB_KEEP_ZERO_IMP` | `0` | Set `1` to also keep zero-scout-importance features (true bypass; may overfit). |
| `XGB_TOP_K` | `0` (adaptive) | Override top-K explicitly (per-feature mastery). |
| `XGB_DEVICE` | `cpu` | `cuda` to use GPU. |
| `XGB_INTERACTION_CONSTRAINTS` | `0` | Tree-split semantic group constraints (pre-existing). |
| `XGB_MONOTONIC` | `0` | Sign-of-effect monotonicity (pre-existing). |
| `XGB_STACKED` | (not impl) | Stage B placeholder. |

## Conclusion + next actions

- Stage A: opt-in only, per-ticker variable. Helps when ticker has signal in many features (AAPL/XOM), hurts when signal is concentrated (MSFT).
- Stage B: deferred until C results known.
- Stage C: script built; needs smoke + comparison vs per-ticker v10 baseline.
- Stage D: blocked on a global-win signal from A or C.
- Real `features_in_trees >= 800` requires either Stage C (n>>p) OR a much larger tree budget (5000+ rounds) — both should be tested.
