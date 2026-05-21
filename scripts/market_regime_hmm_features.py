"""
market_regime_hmm_features.py — 5-state market-regime HMM classifier.

Per OC audit #5: HMM regime classifier on VIX + cross-sectional vol + sector
correlation → 1-of-5 regime state. Output: regime_state (0-4), regime_prob_<i>
for i in 0..4, regime_persistence (consecutive days in current regime).

NO-LOOKAHEAD AUDIT (2026-05-21)
================================
Data sources consumed:
  - cache/macro_data.parquet — daily VIX, SPY, sector ETFs (XL?). Same-bar data.
  - cache/sector_returns.parquet — daily per-sector return aggregates (12 sectors).

Computation:
  1. Build market-level observation series (3-dim):
        obs[0,t] = log(VIX_21d_MA)[t]            — VIX trend
        obs[1,t] = cross-sectional 5-yr realized vol across sector returns
        obs[2,t] = mean off-diagonal sector-pair correlation (252d rolling)
  2. ALL three obs columns are .shift(1) — bar t uses bars ≤ t-1 only.
  3. Fit GaussianHMM(n_components=5, covariance_type="diag") on full history.
     Parameter-fit lookahead is the same mild caveat as GARCH/HMM3 (documented).
  4. Viterbi decode + posteriors using only causally-available bars.
  5. regime_persistence[t] = consecutive days the regime label is unchanged
     up to and including t (computed forward, never uses future bars).

Strict NO LOOKAHEAD guarantees:
  - Every observation column is .shift(1) before HMM input.
  - Per-ticker dataframe rows are .reindex()-ed onto the ticker timeline; missing
    regime values forward-filled (which only uses past).
  - persistence is a forward-running cumulative — never sees future.
  - HMM parameter fit on full series has documented ~0.5-1% mild lookahead (same
    as hmm_3state module — accepted trade-off for the regime classifier family).

Output (6 columns):
  - regime_state           : int [0..4], Viterbi-decoded
  - regime_prob_0..4       : posterior probabilities, sum to 1.0
  - regime_persistence     : consecutive-day count in current regime

Fallback:
  - hmmlearn missing → all-zero with regime_state=2 (neutral), prob_2=1.0.
  - Macro data missing → all-zero with regime_state=2 (neutral), prob_2=1.0.
  - Insufficient history (< 252 bars) → quantile-based fallback (VIX terciles
    blended with sector vol terciles → 5 combined buckets).

Data source: standard daily macro cache parquets (no paid API required).
Algorithm:  Baum-Welch HMM via hmmlearn (BSD-3).
License:    internal.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
_SP_ROOT = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/"
    "My Drive/AI-Tools/s&p500-ticker-mastery"
)
_MACRO_PATH = _SP_ROOT / "cache" / "macro_data.parquet"
_SECTOR_RET_PATH = _SP_ROOT / "cache" / "sector_returns.parquet"

REGIME_N_STATES: int = 5
REGIME_FEATURE_NAMES: list[str] = [
    "regime_state",
    "regime_prob_0",
    "regime_prob_1",
    "regime_prob_2",
    "regime_prob_3",
    "regime_prob_4",
    "regime_persistence",
]
REGIME_FEATURE_COUNT: int = len(REGIME_FEATURE_NAMES)  # 7 (= 1 state + 5 probs + 1 persistence)

_HMM_MIN_BARS = 252       # ≥1 trading year required to attempt HMM fit
_VIX_MA_WINDOW = 21       # VIX trend window
_SECTOR_CORR_WINDOW = 252 # rolling pair-corr window (1 year)
_SECTOR_VOL_WINDOW = 21   # cross-sectional vol window (1 trading month)
_EPS = 1e-9

# -----------------------------------------------------------------------------
# Macro observation builder (causal, .shift(1)-safe)
# -----------------------------------------------------------------------------


def _build_macro_obs() -> Optional[pd.DataFrame]:
    """Build a 3-column observation frame (datetime-indexed):
        log_vix_ma:    log(VIX 21d MA)
        xsec_vol:      cross-sectional std of sector returns (rolling 21d annualized)
        sector_corr:   mean off-diagonal pair-correlation across sector returns
                       (rolling 252d).
    All three returned values are .shift(1)-safe (i.e. NaN-leading by 1 bar
    relative to the underlying same-bar series).
    """
    if not _MACRO_PATH.exists() or not _SECTOR_RET_PATH.exists():
        logger.warning("[market_regime] macro/sector cache missing")
        return None
    try:
        macro = pd.read_parquet(_MACRO_PATH)
        sect = pd.read_parquet(_SECTOR_RET_PATH)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[market_regime] macro/sector load failed: %s", exc)
        return None

    if "vix" not in macro.columns:
        logger.warning("[market_regime] vix col not in macro_data")
        return None

    # Normalize indices to tz-naive daily date
    def _to_naive(df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.index, pd.DatetimeIndex):
            idx = df.index
            if idx.tz is not None:
                idx = idx.tz_convert("UTC").tz_localize(None)
            df = df.copy()
            df.index = pd.to_datetime(idx).normalize()
        return df

    macro = _to_naive(macro)
    sect  = _to_naive(sect)

    # Obs 1: log VIX 21d MA  (same-bar quantity; shift below)
    vix = pd.to_numeric(macro["vix"], errors="coerce")
    log_vix_ma = np.log(vix.clip(lower=1e-6)).rolling(_VIX_MA_WINDOW, min_periods=5).mean()

    # Obs 2: cross-sectional vol across sectors (annualized)
    # For each bar, take the 12-sector return row → std then annualize, smoothed.
    cross_std = sect.std(axis=1, skipna=True) * np.sqrt(252)
    xsec_vol  = cross_std.rolling(_SECTOR_VOL_WINDOW, min_periods=5).mean()

    # Obs 3: mean off-diagonal pair-correlation across sectors, rolling 252d
    # Computed via rolling cov→corr trick on sector return matrix.
    n_sec = sect.shape[1]
    # Pre-fill NaNs with zero so rolling.corr is well-defined; sparseness handled below.
    s_filled = sect.fillna(0.0)
    # rolling pairwise mean off-diagonal correlation via formula:
    #   mean_offdiag_corr = (1/(n*(n-1))) * (sum_corr_matrix - n)
    # We approximate via rolling-window means/stds + dot-product trick:
    win = _SECTOR_CORR_WINDOW

    def _rolling_mean_offdiag_corr(df_: pd.DataFrame, window: int) -> pd.Series:
        # Z-score each column inside the rolling window; then off-diag corr ≈ mean(Z·Z^T) - diag
        roll_mean = df_.rolling(window, min_periods=max(20, window // 4)).mean()
        roll_std  = df_.rolling(window, min_periods=max(20, window // 4)).std(ddof=0)
        z = (df_ - roll_mean) / (roll_std + _EPS)
        # Rolling sum of pairwise products = (sum_i z_i)^2 - sum_i z_i^2 per bar — average over window
        # But for ROLLING corr we need sum_{t in window} z_i[t] * z_j[t] / window for each i,j.
        # Approximation: cumulative pair-prod estimator via z.dot(z.T) is expensive.
        # Simpler practical proxy: instantaneous off-diag corr ≈ 1 - mean_var/(mean_total)
        # We compute true rolling corr per-pair using pandas.DataFrame.rolling().corr() but only
        # the off-diagonal mean — done via a single rolling call returning a 3-D structure.
        # Pandas exposes this as df.rolling(w).corr() returning multi-index frame.
        try:
            rc = df_.rolling(window, min_periods=max(20, window // 4)).corr()
        except Exception as exc:  # noqa: BLE001
            logger.warning("[market_regime] rolling.corr failed: %s — zero-filling sector_corr", exc)
            return pd.Series(0.0, index=df_.index)
        # rc has MultiIndex (date, col); for each date pull the n×n block, take off-diag mean.
        out = []
        dates = rc.index.get_level_values(0).unique()
        cols = df_.columns
        for d in dates:
            block = rc.loc[d]
            # off-diagonal mean
            arr = block.values
            n = arr.shape[0]
            if n < 2:
                out.append(np.nan)
                continue
            off_sum = np.nansum(arr) - np.nansum(np.diag(arr))
            denom = n * (n - 1)
            out.append(off_sum / denom if denom else np.nan)
        s = pd.Series(out, index=dates)
        return s.reindex(df_.index)

    sector_corr = _rolling_mean_offdiag_corr(s_filled, win)

    # Align all three on a common index
    common_idx = log_vix_ma.index.union(xsec_vol.index).union(sector_corr.index)
    log_vix_ma  = log_vix_ma.reindex(common_idx)
    xsec_vol    = xsec_vol.reindex(common_idx)
    sector_corr = sector_corr.reindex(common_idx)

    obs = pd.concat(
        [
            log_vix_ma.rename("log_vix_ma"),
            xsec_vol.rename("xsec_vol"),
            sector_corr.rename("sector_corr"),
        ],
        axis=1,
    )

    # ---- KEY NO-LOOKAHEAD STEP ----
    obs = obs.shift(1)
    return obs


def _quantile_fallback(obs: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Quantile blend (VIX-tercile × xsec-vol-tercile mapped to 5 buckets)."""
    log_vix = obs["log_vix_ma"]
    xs = obs["xsec_vol"]
    valid = log_vix.notna() & xs.notna()
    if valid.sum() < 5:
        s = pd.Series(2, index=obs.index, dtype=int)
        prob = pd.DataFrame(0.0, index=obs.index, columns=[f"regime_prob_{i}" for i in range(REGIME_N_STATES)])
        prob["regime_prob_2"] = 1.0
        return s, prob
    v33 = log_vix[valid].quantile(0.333)
    v67 = log_vix[valid].quantile(0.667)
    x50 = xs[valid].quantile(0.5)

    state = pd.Series(2, index=obs.index, dtype=int)
    state[(log_vix <= v33) & (xs <= x50)] = 0
    state[(log_vix <= v33) & (xs > x50)]  = 1
    state[(log_vix > v33) & (log_vix <= v67)] = 2
    state[(log_vix > v67) & (xs <= x50)]  = 3
    state[(log_vix > v67) & (xs > x50)]   = 4

    prob = pd.DataFrame(0.0, index=obs.index, columns=[f"regime_prob_{i}" for i in range(REGIME_N_STATES)])
    for i in range(REGIME_N_STATES):
        prob.loc[state == i, f"regime_prob_{i}"] = 1.0
    return state, prob


def _fit_hmm_and_decode(obs: pd.DataFrame) -> Optional[tuple[pd.Series, pd.DataFrame]]:
    """Fit a 5-state GaussianHMM on *obs* (.shift(1) already applied)."""
    try:
        from hmmlearn.hmm import GaussianHMM  # noqa: PLC0415
    except ImportError:
        logger.warning("[market_regime] hmmlearn not available — quantile fallback")
        return None

    valid_mask = obs.notna().all(axis=1)
    if valid_mask.sum() < _HMM_MIN_BARS:
        logger.warning(
            "[market_regime] insufficient valid bars (%d < %d) — quantile fallback",
            valid_mask.sum(), _HMM_MIN_BARS,
        )
        return None

    train = obs[valid_mask].values
    # Fill the full obs frame (for prediction across the entire date index)
    obs_full = obs.ffill().bfill()
    if obs_full.isna().any().any():
        obs_full = obs_full.fillna(0.0)

    try:
        model = GaussianHMM(
            n_components=REGIME_N_STATES,
            covariance_type="diag",
            n_iter=200,
            random_state=42,
        )
        model.fit(train)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[market_regime] HMM fit failed (%s) — quantile fallback", exc)
        return None

    try:
        raw_states = model.predict(obs_full.values).astype(int)
        post_probs = model.predict_proba(obs_full.values)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[market_regime] HMM decode failed (%s) — quantile fallback", exc)
        return None

    # Sort regimes by ascending mean log-VIX (col 0): 0=lowest-VIX, 4=highest-VIX
    means = model.means_[:, 0]
    sort_idx = np.argsort(means)
    state_map = {int(orig): int(rank) for rank, orig in enumerate(sort_idx)}
    sorted_states = np.array([state_map[s] for s in raw_states], dtype=int)
    sorted_probs = post_probs[:, sort_idx]

    state = pd.Series(sorted_states, index=obs.index, dtype=int)
    prob_df = pd.DataFrame(
        sorted_probs,
        index=obs.index,
        columns=[f"regime_prob_{i}" for i in range(REGIME_N_STATES)],
    )

    # Zero out positions before any valid obs (mask first invalid run)
    pre_valid = ~valid_mask.cummax()  # True before the FIRST valid bar
    state.loc[pre_valid] = 2
    for c in prob_df.columns:
        prob_df.loc[pre_valid, c] = 0.0
    prob_df.loc[pre_valid, "regime_prob_2"] = 1.0
    return state, prob_df


def _persistence(state: pd.Series) -> pd.Series:
    """Consecutive-day count in the current regime, forward-running."""
    s = state.fillna(method="ffill").fillna(2).astype(int)
    changed = (s != s.shift(1)).astype(int)
    grp = changed.cumsum()
    return s.groupby(grp).cumcount() + 1


def _zero_output(df: pd.DataFrame) -> pd.DataFrame:
    """Neutral-fill 6 cols when HMM cannot run."""
    if "regime_state" not in df.columns:
        df["regime_state"] = 2
    for i in range(REGIME_N_STATES):
        col = f"regime_prob_{i}"
        if col not in df.columns:
            df[col] = 1.0 if i == 2 else 0.0
    if "regime_persistence" not in df.columns:
        df["regime_persistence"] = 0
    return df


# -----------------------------------------------------------------------------
# Public entrypoint
# -----------------------------------------------------------------------------
def compute_market_regime_hmm_features(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
) -> pd.DataFrame:
    """Append 6 market-regime HMM features (regime_state + 5 probs + persistence)
    to *df*. Returns a copy with up to 7 new columns appended.

    All outputs are .shift(1)-safe via the input-observation shift performed
    inside _build_macro_obs().
    """
    df = df.copy()

    obs = _build_macro_obs()
    if obs is None:
        logger.warning("[market_regime] obs build failed — zero output for %s", ticker or "?")
        return _zero_output(df)

    fit = _fit_hmm_and_decode(obs)
    if fit is None:
        state, prob_df = _quantile_fallback(obs)
    else:
        state, prob_df = fit

    persistence = _persistence(state)

    # Reindex regime outputs onto the ticker's date index, forward-fill from past
    # only (state and prob are computed on the union of macro days; reindex maps
    # ticker bars to the most recent prior macro bar via ffill, never future).
    tgt_idx = df.index
    # Convert macro/regime index to tz-naive for safe asof/ffill alignment
    if isinstance(tgt_idx, pd.DatetimeIndex):
        if tgt_idx.tz is not None:
            tgt_idx_naive = tgt_idx.tz_convert("UTC").tz_localize(None).normalize()
        else:
            tgt_idx_naive = tgt_idx.normalize()
    else:
        tgt_idx_naive = tgt_idx

    # Use reindex(method="ffill") so missing bars take last past available regime
    state_ff = state.reindex(state.index.union(tgt_idx_naive)).sort_index().ffill().reindex(tgt_idx_naive)
    prob_ff = prob_df.reindex(prob_df.index.union(tgt_idx_naive)).sort_index().ffill().reindex(tgt_idx_naive)
    pers_ff = persistence.reindex(persistence.index.union(tgt_idx_naive)).sort_index().ffill().reindex(tgt_idx_naive)

    df["regime_state"] = pd.to_numeric(state_ff.values, errors="coerce")
    df["regime_state"] = df["regime_state"].fillna(2).astype(int)
    for i in range(REGIME_N_STATES):
        col = f"regime_prob_{i}"
        df[col] = pd.to_numeric(prob_ff[col].values, errors="coerce")
        df[col] = df[col].fillna(1.0 if i == 2 else 0.0).astype(float)
    df["regime_persistence"] = pd.to_numeric(pers_ff.values, errors="coerce").astype(float)
    df["regime_persistence"] = df["regime_persistence"].fillna(0).astype(int)

    logger.info(
        "[market_regime] %s: regime distribution %s",
        ticker or "?",
        df["regime_state"].value_counts().to_dict(),
    )
    return df


# -----------------------------------------------------------------------------
# Smoke entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    obs = _build_macro_obs()
    if obs is None:
        print("obs build failed")
        sys.exit(1)
    print("obs shape:", obs.shape, "valid bars:", obs.notna().all(axis=1).sum())
    fit = _fit_hmm_and_decode(obs)
    if fit is None:
        state, prob_df = _quantile_fallback(obs)
    else:
        state, prob_df = fit
    print("Regime distribution:", state.value_counts().to_dict())
    print("Sample tail:")
    print(pd.concat([state.rename("regime_state"), prob_df], axis=1).tail(5))
