"""pbo.py — Probability of Backtest Overfitting (PBO) via CSCV.

Reference: Bailey, Borwein, López de Prado, Zhu, "The Probability of
Backtest Overfitting", J. Comput. Finance (2014). The Combinatorially
Symmetric Cross-Validation (CSCV) procedure:

    1. Stack the per-trial return time series into a matrix M of shape
       (T, N) where T = bars/days, N = trial configurations.
    2. Partition rows into S equal slices.
    3. For each of the C(S, S/2) combinations of S/2 slices forming the
       in-sample (IS), the remaining slices form OOS.
    4. For each split: pick the trial j* that maximises the IS Sharpe.
       Compute that trial's OOS rank (relative rank in [0, 1] of its OOS
       performance among all N trials).
    5. Define the logit: w = rank / (1 - rank) on (0, 1). PBO is the
       fraction of splits where logit(w) <= 0, i.e. the best-IS trial
       had below-median OOS performance.

PBO ~ 0   => no overfitting (best-IS = best-OOS reliably).
PBO ~ 0.5 => pure noise (selection is random).
PBO > 0.5 => actively anti-selected (rare but seen with extreme curve-fits).

Public API
----------
probability_backtest_overfitting(matrix_returns, n_splits=16) -> float
    matrix_returns : 2D array-like, shape (T, N).
    n_splits       : S in Bailey 2014 (must be even, default 16 ->
                     C(16, 8) = 12,870 splits).
"""
from __future__ import annotations

import math
from itertools import combinations
from typing import List, Sequence, Union

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _HAS_NUMPY = False


def _sharpe(col: Sequence[float]) -> float:
    """Annualised-agnostic Sharpe: mean / std on the raw return series.

    PBO is rank-invariant in the Sharpe constant, so we omit annualisation.
    """
    if _HAS_NUMPY:
        a = np.asarray(col, dtype=float)
        a = a[~np.isnan(a)]
        if a.size < 2:
            return 0.0
        sd = a.std(ddof=1)
        if sd <= 0:
            return 0.0
        return float(a.mean() / sd)
    # Pure-Python fallback
    xs = [x for x in col if x == x]  # nan filter
    if len(xs) < 2:
        return 0.0
    mu = sum(xs) / len(xs)
    var = sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)
    if var <= 0:
        return 0.0
    return mu / math.sqrt(var)


def probability_backtest_overfitting(
    matrix_returns: Union[Sequence[Sequence[float]], "np.ndarray"],
    n_splits: int = 16,
) -> float:
    """Combinatorially Symmetric Cross-Validation PBO.

    Parameters
    ----------
    matrix_returns : array-like (T, N)
        Each column is one trial's per-bar return series.
    n_splits : int, even, default 16
        S in Bailey 2014. C(S, S/2) combos drive runtime.

    Returns
    -------
    pbo : float in [0, 1]
        Fraction of CSCV splits where the best-IS trial under-performed
        the OOS median.
    """
    if n_splits % 2:
        raise ValueError(f"n_splits must be even, got {n_splits}")
    if not _HAS_NUMPY:
        raise RuntimeError("PBO requires numpy")

    M = np.asarray(matrix_returns, dtype=float)
    if M.ndim != 2:
        raise ValueError(f"matrix_returns must be 2D, got shape {M.shape}")
    T, N = M.shape
    if N < 2:
        # Single trial -> no selection -> PBO is undefined; return 0.5
        # (neutral) to be honest about it.
        return 0.5
    if T < n_splits:
        # Not enough rows for the requested split count; fall back to
        # the largest feasible even S.
        n_splits = max(2, (T // 2) * 2)
        if n_splits < 2:
            return 0.5

    # 1. Partition rows into n_splits contiguous slices
    slice_idx: List[np.ndarray] = np.array_split(np.arange(T), n_splits)
    half = n_splits // 2

    logits: List[float] = []
    for is_slices in combinations(range(n_splits), half):
        is_rows = np.concatenate([slice_idx[i] for i in is_slices])
        oos_rows = np.concatenate(
            [slice_idx[i] for i in range(n_splits) if i not in set(is_slices)]
        )
        is_M = M[is_rows]
        oos_M = M[oos_rows]

        # 2. Per-trial Sharpe in IS + OOS
        is_sr = np.array([_sharpe(is_M[:, j]) for j in range(N)])
        oos_sr = np.array([_sharpe(oos_M[:, j]) for j in range(N)])

        if not np.isfinite(is_sr).any() or not np.isfinite(oos_sr).any():
            continue

        # 3. Pick best-IS trial
        # nan-safe argmax
        is_sr_safe = np.where(np.isnan(is_sr), -np.inf, is_sr)
        j_star = int(np.argmax(is_sr_safe))

        # 4. OOS rank of j_star (fractional rank in [0, 1])
        oos_safe = np.where(np.isnan(oos_sr), -np.inf, oos_sr)
        # rank: count how many trials had strictly lower OOS Sharpe
        order = np.argsort(oos_safe)
        rank_pos = int(np.where(order == j_star)[0][0])
        # fractional rank in (0, 1) — avoid 0 and 1 for the logit
        w_bar = (rank_pos + 1) / (N + 1)

        # 5. Logit
        if 0 < w_bar < 1:
            logit = math.log(w_bar / (1.0 - w_bar))
            logits.append(logit)

    if not logits:
        return 0.5  # neutral when no splits succeeded

    # PBO = fraction of splits where best-IS landed below OOS median
    pbo = float(sum(1 for x in logits if x <= 0) / len(logits))
    return pbo


# Smoke / sanity self-test -------------------------------------------------

if __name__ == "__main__":
    if not _HAS_NUMPY:
        raise SystemExit("numpy required for smoke test")
    rng = np.random.default_rng(42)

    # 1. Pure-noise matrix: 200 bars x 50 trials, all N(0, 1).
    #    Selection is random, expect PBO ~ 0.5.
    noise = rng.normal(0, 1, size=(200, 50))
    pbo_noise = probability_backtest_overfitting(noise, n_splits=8)
    print(f"[smoke] PBO(pure-noise 200x50, S=8) = {pbo_noise:.3f} (expect ~0.5)")
    assert 0.3 <= pbo_noise <= 0.7, f"PBO noise out of band: {pbo_noise}"

    # 2. One genuinely-skilled trial in column 0 (drift), rest noise.
    #    Expect PBO < 0.2 (the skilled trial dominates both IS and OOS).
    skill = noise.copy()
    skill[:, 0] += 0.5  # large positive drift
    pbo_skill = probability_backtest_overfitting(skill, n_splits=8)
    print(f"[smoke] PBO(1 skilled + 49 noise, S=8) = {pbo_skill:.3f} (expect < 0.2)")
    assert pbo_skill < 0.2, f"PBO skill out of band: {pbo_skill}"

    # 3. Monotone-increasing trial in column 0 (curve-fit to first half).
    #    Curve-fit: high in IS, mean-zero in OOS. Expect HIGH PBO.
    overfit = rng.normal(0, 1, size=(200, 50))
    # Trial 0 is fit to the first half only: pump returns 0..99, zero after
    overfit[:100, 0] += 0.8
    overfit[100:, 0] += 0.0
    # With contiguous slicing + IS = first-half, this trial dominates IS
    # and is mediocre OOS -> PBO should be elevated vs noise baseline.
    pbo_overfit = probability_backtest_overfitting(overfit, n_splits=8)
    print(f"[smoke] PBO(curve-fit-first-half, S=8) = {pbo_overfit:.3f} (expect > PBO_noise)")

    print("[smoke] pbo.py PASS")
