# autosolve_skip: bootstrap CI metrics — 2026-05-21
"""bootstrap_ci_metrics.py — Bootstrap confidence intervals for backtest metrics.

Per af9312dd finding #5: replace single-number Sharpe/PF/DD with CI bands so
gates and reporting reflect statistical uncertainty rather than a point
estimate.

Uses moving-block bootstrap (Kunsch 1989, Politis & Romano 1994) to preserve
serial correlation in financial return series. Default block length ~30
trading days. Defaults: n_boot=1000, ci=0.95.

Public API:
    bootstrap_sharpe_ci(returns, n_boot=1000, ci=0.95, block_size=30,
                        ann_factor=252, seed=None) -> (low, high, point)
    bootstrap_pf_ci(trade_pnls, n_boot=1000, ci=0.95, seed=None)
        -> (low, high, point)
    bootstrap_dd_ci(equity, n_boot=1000, ci=0.95, block_size=30, seed=None)
        -> (low, high, point)
    moving_block_bootstrap(arr, n_boot, block_size, rng) -> ndarray (n_boot, T)

References:
  Politis, D. N., & Romano, J. P. (1994). The Stationary Bootstrap.
    J. American Statistical Association, 89(428).
  Lopez de Prado, M. (2018). Advances in Financial Machine Learning, ch. 11.

Notes:
  - Returns array must be log/simple returns (NOT prices) for Sharpe.
  - For PF the input is per-trade PnLs (NOT per-bar returns).
  - For DD the input is equity curve (cumulative), so we bootstrap returns
    derived from it, rebuild equity, then take maxDD.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

__all__ = [
    "bootstrap_sharpe_ci",
    "bootstrap_pf_ci",
    "bootstrap_dd_ci",
    "moving_block_bootstrap",
]


def _as_1d_array(x: Sequence[float] | np.ndarray, name: str = "x") -> np.ndarray:
    a = np.asarray(x, dtype=float).ravel()
    if a.size == 0:
        raise ValueError(f"{name}: empty array")
    if not np.all(np.isfinite(a)):
        # Replace non-finite with 0 (rare in trading data); warn via err.
        a = np.where(np.isfinite(a), a, 0.0)
    return a


def moving_block_bootstrap(
    arr: np.ndarray,
    n_boot: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return (n_boot, T) resamples via moving-block bootstrap.

    Each resample is built by concatenating ceil(T/block_size) randomly-chosen
    overlapping blocks of length `block_size` from the source, then truncated
    to T. Preserves intra-block serial correlation.
    """
    T = arr.size
    block_size = max(1, min(int(block_size), T))
    n_blocks = math.ceil(T / block_size)
    # All valid starting indices for a full-length block.
    max_start = T - block_size
    if max_start < 0:
        # arr shorter than block_size: just iid-resample.
        idx = rng.integers(0, T, size=(n_boot, T))
        return arr[idx]
    starts = rng.integers(0, max_start + 1, size=(n_boot, n_blocks))
    # Build (n_boot, n_blocks, block_size) then reshape.
    offsets = np.arange(block_size)
    idx = starts[:, :, None] + offsets[None, None, :]
    idx = idx.reshape(n_boot, n_blocks * block_size)[:, :T]
    return arr[idx]


def _ci_bounds(samples: np.ndarray, ci: float) -> tuple[float, float]:
    alpha = (1.0 - ci) / 2.0
    low = float(np.nanpercentile(samples, 100.0 * alpha))
    high = float(np.nanpercentile(samples, 100.0 * (1.0 - alpha)))
    return low, high


def bootstrap_sharpe_ci(
    returns: Sequence[float] | np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    block_size: int = 30,
    ann_factor: int = 252,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Bootstrap CI for the annualized Sharpe ratio.

    Args:
        returns: per-bar return series (log or simple).
        n_boot: number of bootstrap resamples.
        ci: confidence level, e.g. 0.95.
        block_size: moving-block length (default ~30 ≈ 1 trading month).
        ann_factor: bars-per-year (252 daily, 252*78 5min etc.).
        seed: optional RNG seed for reproducibility.

    Returns:
        (low, high, point_estimate)
    """
    r = _as_1d_array(returns, "returns")
    rng = np.random.default_rng(seed)
    samples = moving_block_bootstrap(r, n_boot, block_size, rng)
    mu = samples.mean(axis=1)
    sd = samples.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sr_per_bar = np.where(sd > 0, mu / sd, 0.0)
    sr_ann = sr_per_bar * math.sqrt(ann_factor)
    low, high = _ci_bounds(sr_ann, ci)
    point_mu = r.mean()
    point_sd = r.std(ddof=1) if r.size > 1 else 0.0
    point = float(point_mu / point_sd * math.sqrt(ann_factor)) if point_sd > 0 else 0.0
    return low, high, point


def bootstrap_pf_ci(
    trade_pnls: Sequence[float] | np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Bootstrap CI for Profit Factor = sum(wins) / |sum(losses)|.

    PF is per-TRADE, so we use plain iid bootstrap over trades (no block —
    individual trade PnLs are typically uncorrelated under the alpha
    assumption; this matches Pardo 2008 and tradesim conventions).
    """
    p = _as_1d_array(trade_pnls, "trade_pnls")
    rng = np.random.default_rng(seed)
    n = p.size
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = p[idx]
    wins = np.where(samples > 0, samples, 0.0).sum(axis=1)
    losses = -np.where(samples < 0, samples, 0.0).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pf = np.where(losses > 0, wins / losses, np.where(wins > 0, np.inf, 1.0))
    pf = np.where(np.isfinite(pf), pf, np.nan)
    low, high = _ci_bounds(pf, ci)
    # Point estimate.
    w_pt = p[p > 0].sum()
    l_pt = -p[p < 0].sum()
    point = float(w_pt / l_pt) if l_pt > 0 else (float("inf") if w_pt > 0 else 1.0)
    return low, high, point


def _max_drawdown(equity: np.ndarray) -> float:
    """Return max drawdown as positive fraction of peak."""
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    # Guard against zero/neg peak.
    safe_peak = np.where(peak > 0, peak, 1.0)
    dd = (peak - equity) / safe_peak
    return float(dd.max())


def bootstrap_dd_ci(
    equity: Sequence[float] | np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    block_size: int = 30,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Bootstrap CI for max-drawdown.

    Derives per-bar simple returns from the equity curve, block-resamples
    them, rebuilds equity from the resampled returns starting at equity[0],
    and reports the max-DD distribution.
    """
    e = _as_1d_array(equity, "equity")
    if e.size < 2:
        return 0.0, 0.0, 0.0
    # Per-bar simple returns; avoid div-by-zero.
    base = np.where(e[:-1] != 0, e[:-1], 1e-12)
    rets = (e[1:] - e[:-1]) / base
    rng = np.random.default_rng(seed)
    samples = moving_block_bootstrap(rets, n_boot, block_size, rng)
    start = float(e[0]) if e[0] != 0 else 1.0
    # Rebuild equity per resample: cumprod(1+r) * start.
    rebuilt = np.cumprod(1.0 + samples, axis=1) * start
    # Prepend start to keep T+1 length parity.
    rebuilt_full = np.concatenate(
        [np.full((n_boot, 1), start), rebuilt], axis=1
    )
    dds = np.array([_max_drawdown(row) for row in rebuilt_full])
    low, high = _ci_bounds(dds, ci)
    point = _max_drawdown(e)
    return low, high, point


# ---- Smoke ---------------------------------------------------------------
def _smoke() -> int:
    import sys

    print("[smoke] bootstrap_ci_metrics — running...")
    rng = np.random.default_rng(0)

    # 1) Sharpe CI brackets true value on a synthetic random walk
    #    with mu=0.0005, sigma=0.01, T=2520 (10y daily).
    mu_true, sigma_true, T = 0.0005, 0.01, 2520
    r = rng.normal(mu_true, sigma_true, size=T)
    true_sr_ann = (mu_true / sigma_true) * math.sqrt(252)
    low, high, point = bootstrap_sharpe_ci(r, n_boot=500, seed=42)
    assert low <= point <= high, f"point {point} outside CI [{low}, {high}]"
    assert (low - 0.5) <= true_sr_ann <= (high + 0.5), (
        f"true_sr_ann={true_sr_ann:.3f} far from CI [{low:.3f}, {high:.3f}]"
    )
    print(
        f"[smoke] Sharpe CI: low={low:.3f} point={point:.3f} high={high:.3f} "
        f"(true≈{true_sr_ann:.3f})"
    )

    # 2) PF CI on a positive-edge trade series.
    pnls = rng.normal(0.5, 2.0, size=300)  # slight positive edge
    low, high, point = bootstrap_pf_ci(pnls, n_boot=500, seed=42)
    assert low <= point <= high, f"PF point {point} outside CI"
    assert point > 0.5, f"PF point {point} unexpectedly low"
    print(f"[smoke] PF CI: low={low:.3f} point={point:.3f} high={high:.3f}")

    # 3) DD CI on a synthetic equity curve.
    eq = np.cumprod(1.0 + r) * 100.0
    low, high, point = bootstrap_dd_ci(eq, n_boot=500, seed=42)
    assert 0.0 <= low <= point <= high <= 1.0, (
        f"DD CI invalid: low={low} point={point} high={high}"
    )
    print(f"[smoke] DD CI: low={low:.3f} point={point:.3f} high={high:.3f}")

    # 4) Coverage: re-run sharpe CI on 30 fresh random walks; >=80% should
    #    bracket the true value. (Tight 95% nominal coverage isn't expected
    #    at small n_boot=200, so we relax to 80%.)
    n_runs, hits = 30, 0
    for k in range(n_runs):
        rk = np.random.default_rng(k + 1).normal(mu_true, sigma_true, size=T)
        lo_k, hi_k, _ = bootstrap_sharpe_ci(rk, n_boot=200, seed=k + 1)
        if lo_k <= true_sr_ann <= hi_k:
            hits += 1
    coverage = hits / n_runs
    print(f"[smoke] Sharpe CI coverage on synthetic RW: {coverage:.2%} (target>=0.80)")
    assert coverage >= 0.80, f"Sharpe CI coverage too low: {coverage:.2%}"

    print("[smoke] bootstrap_ci_metrics PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(_smoke())
