# autosolve_skip: exec-aware metrics — 2026-05-21
"""exec_aware_metrics.py — Subtract realistic execution costs from backtest metrics.

Per af9312dd finding #5: backtest Sharpe/PF/DD without execution cost is
optimistic. This module subtracts slippage + commission + bid-ask spread
from each trade and recomputes the headline metrics on the NET stream.

Defaults reflect 2026 Alpaca Markets paper-trade economics:
  - commission_bps  = 0.0   (Alpaca: $0 stock equities, $0.0035/share options)
  - slippage_bps    = 5.0   (market-order; 1 tick on $100 stock = ~1 bp,
                             5 bps conservative for liquid S&P 500 names)
  - spread_bps      = 2.0   (half-spread, since fills are typically at mid
                             but worst-case marketable on one side)

Public API:
    apply_costs_to_trades(trade_pnls, trade_notionals, slippage_bps=5.0,
                          commission_bps=0.0, spread_bps=2.0,
                          fixed_per_trade=0.0) -> ndarray
    apply_costs_to_returns(returns, turnover=None, slippage_bps=5.0,
                           commission_bps=0.0, spread_bps=2.0) -> ndarray
    net_metrics(trade_pnls, trade_notionals, returns, equity, ann_factor=252,
                **cost_kwargs) -> dict
        with keys gross/net for sharpe, pf, dd, expectancy.

The cost model assumes:
  total_cost_per_trade = notional * (slippage_bps + spread_bps) / 10_000
                       + notional * commission_bps / 10_000
                       + fixed_per_trade
That cost is subtracted from each trade PnL. For per-bar returns we use
optional turnover (fraction of capital traded that bar); if turnover is None
we assume entry/exit on every bar (worst case) - exposed via parameter.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

__all__ = [
    "apply_costs_to_trades",
    "apply_costs_to_returns",
    "net_metrics",
    "DEFAULT_SLIPPAGE_BPS",
    "DEFAULT_COMMISSION_BPS",
    "DEFAULT_SPREAD_BPS",
]

DEFAULT_SLIPPAGE_BPS = 5.0
DEFAULT_COMMISSION_BPS = 0.0   # Alpaca free-tier stock equities
DEFAULT_SPREAD_BPS = 2.0


def _as_arr(x, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=float).ravel()
    if a.size == 0:
        raise ValueError(f"{name}: empty array")
    return a


def apply_costs_to_trades(
    trade_pnls: Sequence[float] | np.ndarray,
    trade_notionals: Sequence[float] | np.ndarray | None = None,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    commission_bps: float = DEFAULT_COMMISSION_BPS,
    spread_bps: float = DEFAULT_SPREAD_BPS,
    fixed_per_trade: float = 0.0,
) -> np.ndarray:
    """Subtract per-trade execution cost from each PnL.

    If `trade_notionals` is None, we use |trade_pnls| as a fallback proxy
    (this overstates cost for small-PnL trades on large positions; pass real
    notionals when available).
    """
    p = _as_arr(trade_pnls, "trade_pnls")
    if trade_notionals is None:
        # Heuristic: assume average position implied by |PnL| / 0.01
        # (i.e. ~1% trade return on average). Better than nothing; the
        # caller is encouraged to pass actual notionals.
        notional = np.abs(p) / 0.01
    else:
        notional = _as_arr(trade_notionals, "trade_notionals")
        if notional.size != p.size:
            raise ValueError(
                f"trade_notionals size {notional.size} != trade_pnls size {p.size}"
            )
    bps_total = float(slippage_bps + commission_bps + spread_bps) / 10_000.0
    cost = notional * bps_total + float(fixed_per_trade)
    return p - cost


def apply_costs_to_returns(
    returns: Sequence[float] | np.ndarray,
    turnover: Sequence[float] | np.ndarray | None = None,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    commission_bps: float = DEFAULT_COMMISSION_BPS,
    spread_bps: float = DEFAULT_SPREAD_BPS,
) -> np.ndarray:
    """Subtract per-bar execution cost (as fraction of equity) from returns.

    cost_bar = turnover_bar * (slippage + commission + spread) / 10000

    If `turnover` is None, we treat every bar as a full round-trip (1.0
    turnover) — pessimistic upper bound on costs.
    """
    r = _as_arr(returns, "returns")
    if turnover is None:
        t = np.ones_like(r)
    else:
        t = _as_arr(turnover, "turnover")
        if t.size != r.size:
            raise ValueError(f"turnover size {t.size} != returns size {r.size}")
    bps_total = float(slippage_bps + commission_bps + spread_bps) / 10_000.0
    return r - t * bps_total


def _sharpe_ann(returns: np.ndarray, ann_factor: int) -> float:
    if returns.size < 2:
        return 0.0
    mu = returns.mean()
    sd = returns.std(ddof=1)
    if sd <= 0:
        return 0.0
    return float(mu / sd * math.sqrt(ann_factor))


def _profit_factor(pnls: np.ndarray) -> float:
    wins = pnls[pnls > 0].sum()
    losses = -pnls[pnls < 0].sum()
    if losses > 0:
        return float(wins / losses)
    return float("inf") if wins > 0 else 1.0


def _max_dd_from_equity(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    safe_peak = np.where(peak > 0, peak, 1.0)
    return float(((peak - equity) / safe_peak).max())


def _expectancy(pnls: np.ndarray) -> float:
    return float(pnls.mean()) if pnls.size > 0 else 0.0


def net_metrics(
    trade_pnls: Sequence[float] | np.ndarray,
    trade_notionals: Sequence[float] | np.ndarray | None = None,
    returns: Sequence[float] | np.ndarray | None = None,
    equity: Sequence[float] | np.ndarray | None = None,
    ann_factor: int = 252,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    commission_bps: float = DEFAULT_COMMISSION_BPS,
    spread_bps: float = DEFAULT_SPREAD_BPS,
    fixed_per_trade: float = 0.0,
    turnover: Sequence[float] | np.ndarray | None = None,
) -> dict:
    """Return gross + net headline metrics with delta.

    Required: trade_pnls.
    Optional: trade_notionals (for accurate cost), returns (for Sharpe),
              equity (for DD), turnover (per-bar fraction traded).

    Returns dict with keys:
        gross_pf, net_pf, pf_delta
        gross_sharpe, net_sharpe, sharpe_delta
        gross_dd, net_dd, dd_delta
        gross_expectancy, net_expectancy, expectancy_delta
        cost_bps_per_trade  (total round-trip bps assumed)
    """
    p_gross = _as_arr(trade_pnls, "trade_pnls")
    p_net = apply_costs_to_trades(
        p_gross,
        trade_notionals=trade_notionals,
        slippage_bps=slippage_bps,
        commission_bps=commission_bps,
        spread_bps=spread_bps,
        fixed_per_trade=fixed_per_trade,
    )

    gross_pf = _profit_factor(p_gross)
    net_pf = _profit_factor(p_net)
    gross_exp = _expectancy(p_gross)
    net_exp = _expectancy(p_net)

    if returns is not None:
        r_gross = _as_arr(returns, "returns")
        r_net = apply_costs_to_returns(
            r_gross,
            turnover=turnover,
            slippage_bps=slippage_bps,
            commission_bps=commission_bps,
            spread_bps=spread_bps,
        )
        gross_sharpe = _sharpe_ann(r_gross, ann_factor)
        net_sharpe = _sharpe_ann(r_net, ann_factor)
    else:
        gross_sharpe = net_sharpe = float("nan")

    if equity is not None:
        eq = _as_arr(equity, "equity")
        gross_dd = _max_dd_from_equity(eq)
        # Net equity = gross equity * cumulative-cost-drag factor.
        # Quick approximation: rebuild equity from net per-bar returns if
        # available, else scale equity by total bps over horizon.
        if returns is not None:
            r_net_arr = apply_costs_to_returns(
                _as_arr(returns, "returns"),
                turnover=turnover,
                slippage_bps=slippage_bps,
                commission_bps=commission_bps,
                spread_bps=spread_bps,
            )
            start = float(eq[0]) if eq[0] != 0 else 1.0
            net_eq = np.concatenate(
                [[start], start * np.cumprod(1.0 + r_net_arr)]
            )[: eq.size]
            net_dd = _max_dd_from_equity(net_eq)
        else:
            # Lump-sum approximation if no return stream available.
            n_bars = eq.size
            total_drag = (slippage_bps + commission_bps + spread_bps) * n_bars / 10_000.0
            net_eq = eq * (1.0 - min(total_drag, 0.99))
            net_dd = _max_dd_from_equity(net_eq)
    else:
        gross_dd = net_dd = float("nan")

    return {
        "gross_pf": gross_pf,
        "net_pf": net_pf,
        "pf_delta": net_pf - gross_pf if math.isfinite(gross_pf) and math.isfinite(net_pf) else float("nan"),
        "gross_sharpe": gross_sharpe,
        "net_sharpe": net_sharpe,
        "sharpe_delta": net_sharpe - gross_sharpe if math.isfinite(gross_sharpe) else float("nan"),
        "gross_dd": gross_dd,
        "net_dd": net_dd,
        "dd_delta": net_dd - gross_dd if math.isfinite(gross_dd) else float("nan"),
        "gross_expectancy": gross_exp,
        "net_expectancy": net_exp,
        "expectancy_delta": net_exp - gross_exp,
        "cost_bps_per_trade": float(slippage_bps + commission_bps + spread_bps),
    }


# ---- Smoke ---------------------------------------------------------------
def _smoke() -> int:
    print("[smoke] exec_aware_metrics — running...")
    rng = np.random.default_rng(7)

    # 1) Trade PnLs with positive edge; costs should reduce PF.
    pnls = rng.normal(0.5, 2.0, size=400)
    notionals = np.full_like(pnls, 100.0)  # $100 per trade
    net = apply_costs_to_trades(pnls, notionals, slippage_bps=5, spread_bps=2)
    assert (net < pnls).all(), "costs did not reduce PnLs"
    gross_pf = _profit_factor(pnls)
    net_pf = _profit_factor(net)
    assert net_pf < gross_pf, f"net PF {net_pf} not < gross PF {gross_pf}"
    print(f"[smoke] PF gross={gross_pf:.3f} -> net={net_pf:.3f} (drag={gross_pf - net_pf:.3f})")

    # 2) Per-bar returns; net Sharpe should be lower than gross.
    r = rng.normal(0.0005, 0.01, size=1000)
    r_net = apply_costs_to_returns(r, turnover=np.full_like(r, 0.1))
    gross_sr = _sharpe_ann(r, 252)
    net_sr = _sharpe_ann(r_net, 252)
    assert net_sr < gross_sr, f"net Sharpe {net_sr} not < gross {gross_sr}"
    print(f"[smoke] Sharpe gross={gross_sr:.3f} -> net={net_sr:.3f}")

    # 3) net_metrics aggregator.
    eq = np.cumprod(1.0 + r) * 100.0
    out = net_metrics(
        trade_pnls=pnls,
        trade_notionals=notionals,
        returns=r,
        equity=eq,
        turnover=np.full_like(r, 0.1),
    )
    assert out["net_pf"] < out["gross_pf"]
    assert out["net_sharpe"] < out["gross_sharpe"]
    assert out["net_dd"] >= out["gross_dd"] - 1e-9  # cost can only widen DD
    assert out["net_expectancy"] < out["gross_expectancy"]
    print("[smoke] net_metrics aggregator OK:")
    for k, v in out.items():
        if isinstance(v, float):
            print(f"          {k:>20s} = {v:.4f}")

    print("[smoke] exec_aware_metrics PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(_smoke())
