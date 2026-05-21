"""deflated_sharpe.py — DSR + PSR + MinTRL primitives.

Reference: Bailey & López de Prado, "The Deflated Sharpe Ratio: Correcting
for Selection Bias, Backtest Overfitting, and Non-Normality"
SSRN 2460551 (2014), J. Portfolio Management.

Closed-form. No external deps beyond `math` (numpy used only for vector
inputs in helpers). Pure-Python so it runs anywhere a sweep runs.

Public API
----------
probabilistic_sharpe_ratio(sr_observed, skew, kurt, n) -> float
    P[true SR > 0 | observed SR=sr_observed], one-tailed (Bailey/LdP 2012).

deflated_sharpe_ratio(sr_observed, n_trials, skew, kurt, n, var_sr_trials=None,
                       sr_benchmark=None) -> (dsr_p, min_trl)
    PSR re-evaluated against an expected-maximum-SR null, accounting for
    multiple testing across n_trials independent backtests. Returns the
    p-value (probability the observed SR exceeds the selection-biased null)
    plus the Minimum Track Record Length needed for the observed SR to
    clear that null at the 95% confidence level.

min_track_record_length(sr_observed, sr_benchmark, skew, kurt, conf=0.95)
    -> float (in same time units as `n`).

expected_max_sr(n_trials, var_sr_trials) -> float
    Closed-form expectation of the maximum SR drawn from `n_trials`
    independent N(0, var_sr_trials) trial SRs. Uses the Euler-Mascheroni
    approximation per Bailey 2014 eq.(10).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

# Standard normal helpers --------------------------------------------------

_SQRT2 = math.sqrt(2.0)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF (Abramowitz approximation via erf)."""
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def _norm_ppf(p: float) -> float:
    """Standard normal inverse CDF (Acklam's algorithm, 1e-9 accurate).

    Avoids the scipy dep. Adequate for the gating use-case (we just need
    monotone CDF/quantile for hypothesis testing).
    """
    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"_norm_ppf: p out of (0,1): {p}")
    # Coefficients
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q / \
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
            ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)


# Euler-Mascheroni constant ------------------------------------------------
_EULER_MASCHERONI = 0.5772156649015329


# Core PSR -----------------------------------------------------------------

def probabilistic_sharpe_ratio(
    sr_observed: float,
    skew: float,
    kurt: float,
    n: int,
    sr_benchmark: float = 0.0,
) -> float:
    """Probabilistic Sharpe Ratio (Bailey & López de Prado 2012).

    PSR(sr*) = P[SR_true > sr_benchmark | SR_observed = sr_observed]
             = Phi( (sr_observed - sr_benchmark) * sqrt(n-1) /
                    sqrt(1 - skew*sr_observed + (kurt-1)/4 * sr_observed**2) )

    Parameters
    ----------
    sr_observed : float
        Observed Sharpe ratio (same time units as `n`, typically annual or
        per-bar; consistency with `n` is what matters).
    skew : float
        Sample skewness of returns. Use 0 for normal returns.
    kurt : float
        Sample kurtosis of returns (NOT excess; i.e. 3 for a normal).
    n : int
        Number of return observations.
    sr_benchmark : float, default 0.0
        Benchmark SR to beat.

    Returns
    -------
    psr : float in [0, 1]
    """
    if n < 2:
        return 0.0
    denom_sq = 1.0 - skew * sr_observed + (kurt - 1.0) / 4.0 * sr_observed * sr_observed
    if denom_sq <= 0:
        # Pathological inputs (e.g. extreme skew/kurt combo). Conservative:
        # return 0 (cannot prove SR>benchmark).
        return 0.0
    z = (sr_observed - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(denom_sq)
    return _norm_cdf(z)


# Expected max SR + DSR ----------------------------------------------------

def expected_max_sr(n_trials: int, var_sr_trials: float) -> float:
    """Closed-form E[max SR] over `n_trials` IID N(0, var_sr_trials) trials.

    Per Bailey & López de Prado 2014 eq.(10):
        E[max] ≈ sqrt(var) * ( (1-gamma) * Φ⁻¹(1 - 1/N) +
                               gamma     * Φ⁻¹(1 - 1/(N*e)) )
    where gamma is Euler-Mascheroni, N is n_trials, e is exp(1).
    """
    if n_trials <= 1:
        return 0.0
    if var_sr_trials <= 0:
        return 0.0
    sd = math.sqrt(var_sr_trials)
    p1 = 1.0 - 1.0 / n_trials
    p2 = 1.0 - 1.0 / (n_trials * math.e)
    # Guard against p2 -> 1
    p1 = min(p1, 1.0 - 1e-12)
    p2 = min(p2, 1.0 - 1e-12)
    g = _EULER_MASCHERONI
    return sd * ((1 - g) * _norm_ppf(p1) + g * _norm_ppf(p2))


def deflated_sharpe_ratio(
    sr_observed: float,
    n_trials: int,
    skew: float,
    kurt: float,
    n: int,
    var_sr_trials: Optional[float] = None,
    sr_benchmark: Optional[float] = None,
) -> Tuple[float, float]:
    """Deflated Sharpe Ratio (Bailey & López de Prado 2014).

    DSR = PSR evaluated against an inflated benchmark =
          E[max_{k=1..N_trials} SR_k] under H0 (no skill).

    Parameters
    ----------
    sr_observed : float
        Sharpe of the candidate strategy.
    n_trials : int
        Number of independent backtest configurations tried during selection.
    skew, kurt : float
        Sample skew + kurtosis of the candidate strategy's returns.
    n : int
        Length of the candidate strategy's return series.
    var_sr_trials : float, optional
        Variance of the SR estimates across the N_trials configurations.
        If omitted, defaults to 1/(n-1) (the asymptotic variance of a
        single SR estimator under the null SR=0) — this is the conservative
        Bailey 2014 default when per-trial SRs are not available.
    sr_benchmark : float, optional
        Override the deflated benchmark directly (skips the expected-max
        computation).

    Returns
    -------
    dsr_p : float in [0, 1]
        Probability the strategy's true SR exceeds the selection-biased null.
    min_trl : float
        Minimum Track Record Length (same time units as `n`) for the
        observed SR to clear the deflated benchmark at 95% confidence.
    """
    if n < 2:
        return 0.0, float("inf")
    if sr_benchmark is None:
        if var_sr_trials is None:
            # Conservative default per Bailey 2014 footnote: under H0:SR=0,
            # asymptotic Var[SR_hat] = 1/(n-1). At N_trials configs, the
            # SR variance ACROSS trials is empirically larger, so this is
            # a lower-bound deflation (gate is conservative, not liberal).
            var_sr_trials = 1.0 / (n - 1)
        sr_benchmark = expected_max_sr(n_trials, var_sr_trials)
    dsr_p = probabilistic_sharpe_ratio(sr_observed, skew, kurt, n, sr_benchmark)
    min_trl = min_track_record_length(sr_observed, sr_benchmark, skew, kurt, conf=0.95)
    return dsr_p, min_trl


def min_track_record_length(
    sr_observed: float,
    sr_benchmark: float,
    skew: float,
    kurt: float,
    conf: float = 0.95,
) -> float:
    """Minimum Track Record Length (Bailey & LdP 2012 eq.(8)).

    n* = 1 + (1 - skew*sr + (kurt-1)/4 * sr^2) * (z_conf / (sr - sr_bench))^2

    Returns +inf if sr_observed <= sr_benchmark (gate can never clear).
    """
    if sr_observed <= sr_benchmark:
        return float("inf")
    z = _norm_ppf(conf)
    denom_sq = 1.0 - skew * sr_observed + (kurt - 1.0) / 4.0 * sr_observed * sr_observed
    if denom_sq <= 0:
        return float("inf")
    diff = sr_observed - sr_benchmark
    return 1.0 + denom_sq * (z / diff) ** 2


# Smoke / sanity self-test -------------------------------------------------

if __name__ == "__main__":
    # NB. Bailey/LdP convention: `sr_observed` is in the SAME time units as
    # `n`. If `n` is in DAYS, then `sr_observed` is the daily SR. To convert
    # an annualised SR to daily: sr_daily = sr_annual / sqrt(252).
    SR_ANNUAL_TO_DAILY = 1.0 / math.sqrt(252)

    # 1. Plain PSR sanity (annual sr=1.2 -> daily ~0.0756, normal, 252 days)
    sr_d = 1.2 * SR_ANNUAL_TO_DAILY
    psr = probabilistic_sharpe_ratio(sr_d, skew=0.0, kurt=3.0, n=252)
    print(f"[smoke] PSR(sr_ann=1.2, normal, n=252) = {psr:.4f} (expect > 0.5)")
    assert psr > 0.5, "PSR sanity failed"

    # 2. DSR spec smoke: annual sr=1.2, N_trials=100, normal, n_obs=252.
    #    With N=100 trials the deflated benchmark inflates substantially;
    #    a daily-equivalent annual SR=1.2 is borderline and should fail.
    dsr_p, min_trl = deflated_sharpe_ratio(
        sr_observed=sr_d, n_trials=100, skew=0.0, kurt=3.0, n=252)
    print(f"[smoke] DSR(sr_ann=1.2, N=100, normal, n=252) = ({dsr_p:.4f}, MinTRL={min_trl:.1f}d)")
    assert dsr_p < 0.5, f"DSR sanity failed: expected <0.5, got {dsr_p}"

    # 3. DSR clears when annual SR is high (=3.0) + few trials + lots of bars
    sr_high = 3.0 * SR_ANNUAL_TO_DAILY
    dsr_p2, _ = deflated_sharpe_ratio(
        sr_observed=sr_high, n_trials=10, skew=0.0, kurt=3.0, n=1000)
    print(f"[smoke] DSR(sr_ann=3.0, N=10,  normal, n=1000) = {dsr_p2:.4f} (expect > 0.95)")
    assert dsr_p2 > 0.95, "DSR high-skill sanity failed"

    # 4. Heavy tails crush PSR vs normal at same SR
    psr_fat = probabilistic_sharpe_ratio(sr_d, skew=-1.0, kurt=6.0, n=252)
    print(f"[smoke] PSR(sr_ann=1.2, skew=-1, kurt=6, n=252) = {psr_fat:.4f} (< normal PSR)")
    assert psr_fat < psr, "Heavy-tail penalty not applied"

    # 5. expected_max_sr monotone in N_trials
    em10 = expected_max_sr(10, 1.0 / 251)
    em1000 = expected_max_sr(1000, 1.0 / 251)
    print(f"[smoke] E[max SR] N=10 -> {em10:.4f},  N=1000 -> {em1000:.4f}  (monotone up)")
    assert em1000 > em10, "expected_max_sr not monotone in N"

    print("[smoke] deflated_sharpe.py PASS")
