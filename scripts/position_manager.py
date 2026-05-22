"""
position_manager.py — Sell-weakest-to-fund-strongest opportunity-cost swap engine.

Quick-win C (2026-05-22). Cross-track design (a02ec159 internet + a903bab2
github + ac0f5aca local) consolidated:

  - Edge-Differential Threshold: challenger margin > weakest + 0.10 (10ppt
    cushion vs round-trip costs ~3bps × 2 = 6bps, expressed in prob-space).
  - Anti-disposition: rescore holdings via
        effective_score = max(live_signal, decayed_alpha)
    so a winner whose live signal has cooled does not get cut prematurely.
  - Warmup-grace: skip swaps for `warmup_s` after a cold restart so we don't
    sell-storm on partial state.
  - max_swaps_per_cycle: cap how many flips fire per `find_swaps` call.
  - Time-decay of entry score:
        remaining_alpha = entry_score * exp(-decay_rate * t_held_hours)
    with decay_rate=0.08 (so half-life ~8.7h — held positions retain meaningful
    alpha through one full intraday session).

Invariants:
  - PURE. Reads nothing from disk, makes no network calls. Order execution
    is the caller's responsibility (cmd_open_trades wires it to the router).
  - NO import from live_paper_trade (avoid circular import). All inputs are
    plain dicts / lists; outputs are plain dataclasses.
  - Paper-trade only. NEVER moves real money.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("position_manager")


# ─────────────────────────────────────────────────────────────────────────────
# Defaults (overridable via PositionManager ctor or env in caller)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_DECAY_RATE = 0.08       # per hour. Half-life ~8.7h.
DEFAULT_SWAP_MARGIN = 0.10      # challenger must beat weakest by ≥ 10ppt.
DEFAULT_WARMUP_S = 300          # 5 min cold-restart grace.
DEFAULT_MAX_SWAPS_PER_CYCLE = 2 # bound total churn per open-trades cycle.


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PositionScore:
    """Rescore output for one currently-held ticker."""
    ticker: str
    entry_score: float
    live_signal: float          # current prob/score for THIS ticker (or 0 if no fresh signal)
    decayed_alpha: float        # entry_score * exp(-decay_rate * t_held_hours)
    effective_score: float      # max(live_signal, decayed_alpha) — anti-disposition
    t_held_hours: float
    notional: float
    qty: float


@dataclass
class SwapProposal:
    """A proposed sell-weakest / buy-strongest swap pair."""
    out_ticker: str
    in_ticker: str
    out_score: float
    in_score: float
    expected_lift: float        # in_score - out_score (prob-space ppt)
    out_notional: float
    out_qty: float
    in_signal_strength: str | None = None
    in_signal_prob: float | None = None
    in_signal: dict = field(default_factory=dict)   # raw firing dict (for router)
    reason: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_float(x: Any, default: float = 0.0) -> float:
    """Defensive float coercion (state values come from JSON — may be str/None)."""
    if x is None:
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _parse_iso_ts(ts: str | None) -> float | None:
    """Parse an ISO-8601 timestamp → unix seconds. None on failure."""
    if not ts:
        return None
    try:
        from datetime import datetime
        # Accept trailing 'Z' as UTC.
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts).timestamp()
    except Exception:
        return None


def _signal_score(sig: dict) -> float:
    """Pull prob from a signal dict (prob preferred; signal_strength → 0/1 fallback)."""
    p = sig.get("prob")
    if p is not None:
        return _to_float(p)
    s = sig.get("signal_strength")
    if isinstance(s, str):
        if s.upper() in ("HIGH", "STRONG", "TOP_DECILE"):
            return 0.90
        if s.upper() in ("NORMAL", "MEDIUM", "BASE"):
            return 0.70
    return _to_float(sig.get("signal"), 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# PositionManager
# ─────────────────────────────────────────────────────────────────────────────

class PositionManager:
    """Opportunity-cost swap engine for the paper-trade open-trades cycle.

    Usage (called from cmd_open_trades when exposure is saturated):

        mgr = PositionManager()
        scored = mgr.rescore_holdings(state, signals_today)
        swaps = mgr.find_swaps(scored, firing_candidates, risk_engine)
        for swap in swaps[:mgr.max_swaps_per_cycle]:
            ...execute via router.route_flip(...)
    """

    def __init__(
        self,
        decay_rate: float = DEFAULT_DECAY_RATE,
        swap_margin: float = DEFAULT_SWAP_MARGIN,
        warmup_s: int = DEFAULT_WARMUP_S,
        max_swaps_per_cycle: int = DEFAULT_MAX_SWAPS_PER_CYCLE,
        now_fn: Any = time.time,
    ) -> None:
        self.decay_rate = float(decay_rate)
        self.swap_margin = float(swap_margin)
        self.warmup_s = int(warmup_s)
        self.max_swaps_per_cycle = int(max_swaps_per_cycle)
        self._now_fn = now_fn

    # ─── time-decay primitive ───────────────────────────────────────────────

    def remaining_alpha(self, entry_score: float, t_held_hours: float) -> float:
        """Exponential decay of entry-time conviction."""
        if entry_score <= 0 or t_held_hours <= 0:
            return max(0.0, float(entry_score))
        return float(entry_score) * math.exp(-self.decay_rate * float(t_held_hours))

    # ─── anti-disposition rescore ───────────────────────────────────────────

    def effective_score(self, position: dict, current_signal: dict | None) -> float:
        """max(live_signal, decayed_alpha) so winners with cooled signals are protected."""
        entry_score = _to_float(position.get("signal_prob"))
        opened_at = _parse_iso_ts(position.get("opened_at"))
        if opened_at is not None:
            t_held_hours = max(0.0, (self._now_fn() - opened_at) / 3600.0)
        else:
            t_held_hours = 0.0
        decayed = self.remaining_alpha(entry_score, t_held_hours)
        live = _signal_score(current_signal) if current_signal else 0.0
        return max(decayed, live)

    # ─── rescore all held positions ─────────────────────────────────────────

    def rescore_holdings(
        self,
        state: dict,
        signals_today: list[dict],
    ) -> list[PositionScore]:
        """Score every held position. Ranked weakest→strongest is the caller's job
        if needed (we return unsorted; find_swaps sorts internally)."""
        positions = state.get("positions") or {}
        # Build a lookup of fresh signals by ticker.
        sig_by_ticker: dict[str, dict] = {}
        for s in (signals_today or []):
            t = s.get("ticker")
            if t:
                sig_by_ticker[t] = s

        scored: list[PositionScore] = []
        for ticker, pos in positions.items():
            entry_score = _to_float(pos.get("signal_prob"))
            opened_at = _parse_iso_ts(pos.get("opened_at"))
            t_held_hours = 0.0
            if opened_at is not None:
                t_held_hours = max(0.0, (self._now_fn() - opened_at) / 3600.0)
            decayed = self.remaining_alpha(entry_score, t_held_hours)
            cur_sig = sig_by_ticker.get(ticker)
            live = _signal_score(cur_sig) if cur_sig else 0.0
            eff = max(decayed, live)
            scored.append(
                PositionScore(
                    ticker=ticker,
                    entry_score=entry_score,
                    live_signal=live,
                    decayed_alpha=decayed,
                    effective_score=eff,
                    t_held_hours=t_held_hours,
                    notional=_to_float(pos.get("notional")),
                    qty=_to_float(pos.get("qty")),
                )
            )
        return scored

    # ─── propose swaps ──────────────────────────────────────────────────────

    def find_swaps(
        self,
        holdings_scored: list[PositionScore],
        candidates_firing: list[dict],
        risk_engine: Any | None = None,
    ) -> list[SwapProposal]:
        """Pair weakest-held with strongest-fresh-not-held; emit SwapProposal when
        challenger.score > weakest.score + swap_margin.

        Greedy O(N+M log M): sort holdings ASC, sort fresh DESC, pair from the ends.
        Skip candidates already held (no self-flip). Honor max_swaps_per_cycle.
        If risk_engine supplied, gate the IN side through risk_engine.check(...) —
        a refusal kills that pair (try next candidate).
        """
        if not holdings_scored or not candidates_firing:
            return []

        held_tickers = {p.ticker for p in holdings_scored}
        fresh: list[dict] = []
        for sig in candidates_firing:
            t = sig.get("ticker")
            if not t or t in held_tickers:
                continue  # already held — not a swap candidate
            if int(sig.get("signal", 0)) != 1:
                continue  # only firing signals
            fresh.append(sig)
        if not fresh:
            return []

        # Sort weakest first; strongest first.
        weak = sorted(holdings_scored, key=lambda p: p.effective_score)
        strong = sorted(fresh, key=lambda s: _signal_score(s), reverse=True)

        proposals: list[SwapProposal] = []
        used_out: set[str] = set()
        used_in: set[str] = set()

        si = 0  # strong index
        for w in weak:
            if len(proposals) >= self.max_swaps_per_cycle:
                break
            if w.ticker in used_out:
                continue
            # Find the next unconsumed challenger that beats this weakest by margin.
            while si < len(strong):
                cand = strong[si]
                cand_t = cand.get("ticker")
                cand_score = _signal_score(cand)
                if cand_t in used_in:
                    si += 1
                    continue
                # Margin check — challenger MUST beat the weakest by swap_margin.
                if cand_score < w.effective_score + self.swap_margin:
                    # Strongest remaining doesn't clear the bar → nothing else will.
                    return proposals
                # Optional risk-engine gate on the IN side.
                if risk_engine is not None:
                    try:
                        approx_qty = max(1, int(w.notional / max(cand.get("price") or 1.0, 1.0)))
                        decision = risk_engine.check(
                            ticker=cand_t,
                            qty=approx_qty,
                            signal=cand,
                            price=_to_float(cand.get("price"), 1.0) or 1.0,
                        )
                        if not decision.passed:
                            log.info(
                                "[SWAP] IN-side risk refusal: %s gate=%s reason=%s",
                                cand_t, getattr(decision, "gate", "?"),
                                getattr(decision, "reason", "?"),
                            )
                            si += 1
                            continue
                    except Exception as e:
                        log.warning("[SWAP] risk_engine.check raised on %s: %s", cand_t, e)
                # Accept pair.
                proposals.append(
                    SwapProposal(
                        out_ticker=w.ticker,
                        in_ticker=str(cand_t),
                        out_score=w.effective_score,
                        in_score=cand_score,
                        expected_lift=cand_score - w.effective_score,
                        out_notional=w.notional,
                        out_qty=w.qty,
                        in_signal_strength=cand.get("signal_strength"),
                        in_signal_prob=cand.get("prob"),
                        in_signal=dict(cand),
                        reason=(
                            f"in={cand_score:.3f} > out={w.effective_score:.3f} "
                            f"+ margin={self.swap_margin:.2f}"
                        ),
                    )
                )
                used_out.add(w.ticker)
                used_in.add(str(cand_t))
                si += 1
                break
        return proposals

    # ─── warmup gate ────────────────────────────────────────────────────────

    def in_warmup(self, state: dict) -> bool:
        """True if we're within `warmup_s` of the last cold restart — skip swaps."""
        last = _to_float(state.get("last_restart_at"), 0.0)
        if last <= 0:
            return False
        return (self._now_fn() - last) < self.warmup_s


__all__ = [
    "PositionManager",
    "PositionScore",
    "SwapProposal",
    "DEFAULT_DECAY_RATE",
    "DEFAULT_SWAP_MARGIN",
    "DEFAULT_WARMUP_S",
    "DEFAULT_MAX_SWAPS_PER_CYCLE",
]
