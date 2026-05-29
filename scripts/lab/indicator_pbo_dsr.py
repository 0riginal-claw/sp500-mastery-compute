"""indicator_pbo_dsr.py — Probability of Backtest Overfitting (CSCV) + Deflated Sharpe Ratio.

References
----------
- Bailey, Borwein, Lopez de Prado, Zhu (2017) "The Probability of Backtest Overfitting"
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253
- Bailey & Lopez de Prado (2014) "The Deflated Sharpe Ratio"

Pure-NumPy implementations. No external deps beyond numpy + scipy.stats.

PBO method (CSCV):
  Given an (T, N) matrix M where M[t, n] is the per-period return of strategy/config n at time t,
  split T evenly into S non-overlapping chunks. For each combination of S/2 chunks (the "training
  set" J) and the complementary S/2 chunks (the "test set" J_bar):
    - rank configs by IS performance on J
    - identify best config in IS; record its relative OOS rank on J_bar
    - logit = ln(omega_n* / (1 - omega_n*))
  PBO = fraction of combinations where the IS-best config is in the bottom half OOS.

DSR method:
  DSR = Phi( (SR - SR0) / sigma_SR )
  where SR0 = E[max SR_i over N trials] ≈ sqrt(V[SR]) * ((1 - gamma) Phi^-1(1 - 1/N) + gamma Phi^-1(1 - 1/(N*e)))
  and sigma_SR = sqrt((1 - skew*SR + (kurt-1)/4 * SR^2) / (T - 1))
  gamma ≈ 0.5772156649 (Euler-Mascheroni).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Tuple

import numpy as np
from scipy import stats

EULER_MASCHERONI = 0.5772156649015329


# ---------------------------------------------------------------------------
# Combinatorially Symmetric Cross-Validation → PBO
# ---------------------------------------------------------------------------


def _sharpe(rets: np.ndarray, ann: float = 1.0) -> np.ndarray:
    """Per-column annualized Sharpe ratio. NaN-safe.

    rets shape: (T, N). Returns shape (N,).
    """
    mu = np.nanmean(rets, axis=0)
    sd = np.nanstd(rets, axis=0, ddof=1)
    out = np.full_like(mu, np.nan)
    mask = sd > 0
    out[mask] = (mu[mask] / sd[mask]) * np.sqrt(ann)
    return out


def cscv_pbo(returns_matrix: np.ndarray, s_chunks: int = 16) -> dict:
    """Compute PBO via Combinatorially Symmetric Cross-Validation.

    Parameters
    ----------
    returns_matrix : (T, N) float array. Each column n is a per-period return series of one
        parameter configuration. NaN allowed (cells with no signal).
    s_chunks : int. Must be even. 16 per Bailey et al.

    Returns
    -------
    dict with keys: pbo, logits (np.ndarray), n_combos, n_configs, n_periods.
    """
    if s_chunks % 2 != 0:
        raise ValueError("s_chunks must be even")
    M = np.asarray(returns_matrix, dtype=np.float64)
    T, N = M.shape
    if T < s_chunks:
        raise ValueError(f"need T>=s_chunks, got T={T}, s={s_chunks}")
    if N < 2:
        raise ValueError(f"need >=2 configs, got N={N}")

    # Trim T so it divides evenly
    chunk_size = T // s_chunks
    T_used = chunk_size * s_chunks
    M = M[:T_used]
    chunks = M.reshape(s_chunks, chunk_size, N)  # (S, T/S, N)

    # All choose(S, S/2) combinations
    half = s_chunks // 2
    all_idx = list(range(s_chunks))
    logits = []
    losses = 0
    total = 0
    for tr_idx in combinations(all_idx, half):
        tr_set = set(tr_idx)
        te_idx = tuple(i for i in all_idx if i not in tr_set)
        J = chunks[list(tr_idx)].reshape(-1, N)        # (T_train, N)
        J_bar = chunks[list(te_idx)].reshape(-1, N)    # (T_test, N)
        is_sr = _sharpe(J)
        oos_sr = _sharpe(J_bar)
        # Pick IS-best (highest in-sample Sharpe)
        if np.all(np.isnan(is_sr)):
            continue
        n_star = int(np.nanargmax(is_sr))
        # Rank OOS — fraction of configs strictly worse than n_star OOS
        # omega = OOS_rank(n_star) / (N+1), in (0, 1)
        oos_clean = np.where(np.isnan(oos_sr), -np.inf, oos_sr)
        rank = (oos_clean < oos_clean[n_star]).sum() + 0.5 * (oos_clean == oos_clean[n_star]).sum()
        omega = rank / N
        # Clip away exact 0/1 for finite logit
        omega = float(np.clip(omega, 1e-6, 1 - 1e-6))
        logit = math.log(omega / (1 - omega))
        logits.append(logit)
        if logit < 0:
            losses += 1
        total += 1

    pbo = losses / total if total else float("nan")
    return {
        "pbo": pbo,
        "logits": np.array(logits, dtype=np.float64),
        "n_combos": total,
        "n_configs": int(N),
        "n_periods": int(T_used),
        "s_chunks": int(s_chunks),
    }


# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio
# ---------------------------------------------------------------------------


def deflated_sharpe(
    sharpe_observed: float,
    rets: np.ndarray,
    n_trials: int,
    variance_of_sharpes: float | None = None,
) -> dict:
    """Compute Deflated Sharpe Ratio probability.

    Parameters
    ----------
    sharpe_observed : observed annualized Sharpe of the selected strategy
    rets : 1-D return series of the selected strategy (for skew/kurtosis)
    n_trials : N — number of strategies/parameter configs tried
    variance_of_sharpes : V[{SR_i}] across trials; if None, uses 1.0 as a conservative prior

    Returns
    -------
    dict: dsr_prob (probability that true SR > 0), sr0, sigma_sr, skew, excess_kurt, T_used
    """
    r = np.asarray(rets, dtype=np.float64)
    r = r[~np.isnan(r)]
    T = r.size
    if T < 30:
        return {"dsr_prob": float("nan"), "reason": f"T={T} too small"}

    skew = float(stats.skew(r, bias=False))
    excess_kurt = float(stats.kurtosis(r, fisher=True, bias=False))  # excess kurtosis (=kurt-3 normal=0)

    # SR0: expected max of N draws from N(0,V)
    V = 1.0 if variance_of_sharpes is None else max(float(variance_of_sharpes), 1e-12)
    sqrtV = math.sqrt(V)
    N = max(int(n_trials), 1)
    if N > 1:
        z1 = stats.norm.ppf(1.0 - 1.0 / N)
        z2 = stats.norm.ppf(1.0 - 1.0 / (N * math.e))
        sr0 = sqrtV * ((1 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2)
    else:
        sr0 = 0.0

    # sigma_SR: standard error of observed SR (Mertens 2002)
    # Using kurt = excess + 3 (Bailey & LdP use kurt - 1 = excess + 2)
    SR = float(sharpe_observed)
    sigma_sr = math.sqrt(max(1e-12, (1.0 - skew * SR + (excess_kurt + 2.0) / 4.0 * SR * SR) / (T - 1)))

    # DSR: P(true SR > SR0)
    z = (SR - sr0) / sigma_sr if sigma_sr > 0 else float("inf")
    dsr_prob = float(stats.norm.cdf(z))

    return {
        "dsr_prob": dsr_prob,
        "sr_observed": SR,
        "sr0_threshold": sr0,
        "sigma_sr": sigma_sr,
        "skew": skew,
        "excess_kurt": excess_kurt,
        "n_trials": N,
        "variance_of_sharpes": V,
        "T_used": int(T),
        "z_score": z,
    }


# ---------------------------------------------------------------------------
# Walk-forward fold generator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fold:
    train_start: int
    train_end: int  # exclusive
    test_start: int
    test_end: int  # exclusive
    embargo: int = 0


def rolling_walkforward_folds(
    n_obs: int,
    n_folds: int = 12,
    train_frac: float = 0.7,
    embargo_frac: float = 0.02,
) -> list[Fold]:
    """Generate rolling walk-forward folds.

    Each fold: train = `train_frac` of the rolling window; test = remainder.
    Window slides forward by test-size each fold so test sets are disjoint.

    Note: the brief says "12 folds, 1yr train, 3mo test". For 5min bars over ~5 years, that is
    ~19500 train / ~4875 test bars per fold. We parametrize so callers can tune.
    """
    if n_obs < n_folds * 30:
        raise ValueError(f"need n_obs >= {n_folds*30}, got {n_obs}")
    # Sliding window: fold i starts at i * test_size
    # window_size satisfies window * n_folds_overlap = n_obs roughly. Simpler: anchor first fold
    # at 0, fix test size = n_obs / (n_folds + train_frac/(1-train_frac))... too fiddly.
    # Practical: fix train=80%, test=20% of `window`; window = n_obs / n_folds * (1/(1-train_frac))
    test_size = max(30, n_obs // (n_folds + int(train_frac / max(1e-3, 1 - train_frac))))
    window = int(test_size / max(1e-3, 1 - train_frac))
    train_size = window - test_size
    embargo = max(0, int(embargo_frac * n_obs))
    folds = []
    for i in range(n_folds):
        start = i * test_size
        tr_s = start
        tr_e = start + train_size
        te_s = tr_e + embargo
        te_e = te_s + test_size
        if te_e > n_obs:
            break
        folds.append(Fold(tr_s, tr_e, te_s, te_e, embargo))
    return folds


def walk_forward_efficiency(in_sample_sharpe: float, out_sample_sharpe: float) -> float:
    """WFE = OOS Sharpe / IS Sharpe. >50% per Phase 2 exit criteria."""
    if in_sample_sharpe <= 0:
        return 0.0
    return out_sample_sharpe / in_sample_sharpe


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    # Synthetic: 5 noisy configs with true zero edge, one with mild positive edge
    T, N = 1024, 10
    mu = np.zeros(N)
    mu[3] = 0.02  # one truly slightly positive config
    rets = rng.normal(loc=mu, scale=1.0, size=(T, N))
    pbo_res = cscv_pbo(rets, s_chunks=16)
    print(f"PBO test: pbo={pbo_res['pbo']:.3f} combos={pbo_res['n_combos']}")

    # DSR
    selected = rets[:, 3]
    sr_selected = float(np.mean(selected) / np.std(selected, ddof=1) * np.sqrt(252))
    dsr = deflated_sharpe(sr_selected, selected, n_trials=N, variance_of_sharpes=np.var(_sharpe(rets) * np.sqrt(252)))
    print(f"DSR test: sr={sr_selected:.3f} sr0={dsr['sr0_threshold']:.3f} dsr_prob={dsr['dsr_prob']:.3f}")

    folds = rolling_walkforward_folds(50000, n_folds=12)
    print(f"Folds: {len(folds)} sample={folds[0]}")
