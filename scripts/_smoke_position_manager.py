"""
_smoke_position_manager.py — smoke + integration test for Quick-win C.

Scenarios:
  1) 5 held positions (probs at entry: 0.92, 0.85, 0.78, 0.65, 0.60) +
     3 fresh signals (probs: 0.95, 0.88, 0.71).
     Expected: 2 swaps fire — 0.95 swaps OUT 0.60, 0.88 swaps OUT 0.65.
     0.71 stays out (margin too thin vs 0.78: 0.71 < 0.78 + 0.10).

  2) Warmup-grace: same state, but last_restart_at set 60s ago.
     Expected: in_warmup() == True; no swaps fire.

  3) max_swaps_per_cycle cap: lots of cheap challengers; only 2 fire.

  4) route_flip mock smoke: SELL succeeds, BUY succeeds → status=complete,
     audit JSONL has 2 lines (one per leg) with shared flip_id prefix.

Runs without touching real broker / network. Prints PASS/FAIL summary at exit.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Ensure we can import sibling modules without venv activation.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from position_manager import PositionManager, SwapProposal  # noqa: E402
from limit_order_router import route_flip  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _iso(dt):
    return dt.astimezone(timezone.utc).isoformat()


def _mock_state(probs_held=None, hours_held=2.0, last_restart_at=0):
    """Build a state dict with N held positions, each opened `hours_held` ago."""
    probs_held = probs_held if probs_held is not None else [0.92, 0.85, 0.78, 0.65, 0.60]
    opened = datetime.now(timezone.utc) - timedelta(hours=hours_held)
    positions = {}
    for i, p in enumerate(probs_held):
        ticker = f"H{i}"
        positions[ticker] = {
            "qty": 10,
            "entry_price": 100.0,
            "order_id": f"mock-{ticker}",
            "client_order_id": f"coid-{ticker}",
            "notional": 100.0,
            "signal_prob": p,
            "threshold": 0.55,
            "opened_at": _iso(opened),
        }
    return {
        "positions": positions,
        "last_restart_at": last_restart_at,
    }


def _mock_signals(probs_fresh=None):
    """Build firing-signal dicts for fresh challengers (not in held set)."""
    probs_fresh = probs_fresh if probs_fresh is not None else [0.95, 0.88, 0.71]
    out = []
    for i, p in enumerate(probs_fresh):
        out.append({
            "ticker": f"F{i}",
            "prob": p,
            "signal": 1,
            "signal_strength": "HIGH" if p >= 0.85 else "NORMAL",
            "threshold": 0.55,
            "price": 50.0,
        })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 1: canonical 5-held + 3-fresh
# ─────────────────────────────────────────────────────────────────────────────

def scenario_1():
    print("\n=== Scenario 1: 5 held vs 3 fresh ===")
    state = _mock_state()
    signals = _mock_signals()
    mgr = PositionManager(swap_margin=0.10)

    scored = mgr.rescore_holdings(state, signals)
    for s in sorted(scored, key=lambda x: x.effective_score):
        print(
            f"  held {s.ticker}: entry={s.entry_score:.3f} "
            f"decayed={s.decayed_alpha:.3f} live={s.live_signal:.3f} "
            f"eff={s.effective_score:.3f} t_held={s.t_held_hours:.2f}h"
        )

    swaps = mgr.find_swaps(scored, signals, risk_engine=None)
    print(f"  → proposed {len(swaps)} swap(s):")
    for sw in swaps:
        print(
            f"    OUT {sw.out_ticker} (eff={sw.out_score:.3f}) ⇨ "
            f"IN {sw.in_ticker} (in={sw.in_score:.3f}, "
            f"lift={sw.expected_lift:.3f}) | {sw.reason}"
        )

    # Expected: 2 swaps. 0.95 ⇨ H4 (eff~0.60), 0.88 ⇨ H3 (eff~0.65).
    # 0.71 stays out because margin too thin vs next-weakest H2 (eff~0.78):
    # 0.71 < 0.78 + 0.10.
    ok = len(swaps) == 2
    if ok:
        outs = {sw.out_ticker for sw in swaps}
        ins = {sw.in_ticker for sw in swaps}
        # exp weakest two: H4 (0.60) and H3 (0.65); strongest two: F0 (0.95), F1 (0.88)
        ok = outs == {"H3", "H4"} and ins == {"F0", "F1"}
    return ok, f"scenario_1: {len(swaps)} swap(s), outs={[sw.out_ticker for sw in swaps]}, ins={[sw.in_ticker for sw in swaps]}"


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 2: warmup-grace
# ─────────────────────────────────────────────────────────────────────────────

def scenario_2():
    print("\n=== Scenario 2: warmup-grace (last_restart_at fresh) ===")
    state = _mock_state(last_restart_at=time.time() - 60)  # 60s ago
    mgr = PositionManager(warmup_s=300)
    in_w = mgr.in_warmup(state)
    print(f"  in_warmup() = {in_w}  (expect True)")

    # And after warmup expires:
    state2 = _mock_state(last_restart_at=time.time() - 600)  # 10min ago
    in_w2 = mgr.in_warmup(state2)
    print(f"  in_warmup() after 10min = {in_w2}  (expect False)")

    return (in_w is True and in_w2 is False), "scenario_2: warmup gate honored"


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 3: max_swaps_per_cycle cap
# ─────────────────────────────────────────────────────────────────────────────

def scenario_3():
    print("\n=== Scenario 3: max_swaps_per_cycle=2 cap ===")
    # 5 weak held + 5 strong fresh — all would qualify, but cap at 2.
    state = _mock_state(probs_held=[0.30, 0.31, 0.32, 0.33, 0.34])
    signals = _mock_signals(probs_fresh=[0.99, 0.98, 0.97, 0.96, 0.95])
    mgr = PositionManager(swap_margin=0.10, max_swaps_per_cycle=2)
    scored = mgr.rescore_holdings(state, signals)
    swaps = mgr.find_swaps(scored, signals)
    print(f"  → proposed {len(swaps)} (expect 2)")
    return len(swaps) == 2, f"scenario_3: capped at {len(swaps)}"


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 4: route_flip mock smoke (SELL + BUY both fill, shared flip_id)
# ─────────────────────────────────────────────────────────────────────────────

class _MockClock:
    def __init__(self):
        self.t = 0.0
    def now(self):
        return self.t
    def sleep(self, s):
        self.t += s


def scenario_4():
    print("\n=== Scenario 4: route_flip mock smoke (SELL+BUY both fill) ===")
    clock = _MockClock()

    # SELL fills immediately on first poll. BUY fills on second poll.
    fills_log = Path(tempfile.mkstemp(suffix=".jsonl")[1])
    place_calls = []
    cancel_calls = []

    def mock_quote(ticker):
        return {"bid": 99.5, "ask": 100.5}

    _state = {"poll_count": 0}
    def mock_place_limit(ticker, side, qty, limit_price, tif="day", client_order_id=None):
        oid = f"ord-{ticker}-{side}-{clock.t:.2f}"
        place_calls.append({
            "ticker": ticker, "side": side, "qty": qty,
            "limit_price": limit_price, "coid": client_order_id,
        })
        return {"order_id": oid, "client_order_id": client_order_id or oid}

    def mock_get_order(order_id):
        _state["poll_count"] += 1
        # Every limit fills on its first poll for the smoke (status filled).
        return {
            "order_id": order_id,
            "status": "filled",
            "filled_qty": "10",
            "filled_avg_price": "100.0",
        }

    def mock_cancel(order_id):
        cancel_calls.append(order_id)

    def mock_place_market(ticker, side=None, qty=None, notional=None, client_order_id=None):
        return {
            "order_id": f"mkt-{ticker}",
            "client_order_id": client_order_id or f"mkt-{ticker}",
            "status": "filled",
            "qty": qty,
            "notional": notional,
            "fill_price": 100.0,
        }

    res = route_flip(
        out_ticker="H4",
        out_qty=10,
        in_ticker="F0",
        in_notional=100.0,
        in_signal_strength="HIGH",
        in_signal_prob=0.95,
        get_quote_fn=mock_quote,
        place_limit_fn=mock_place_limit,
        cancel_fn=mock_cancel,
        get_order_fn=mock_get_order,
        place_market_fn=mock_place_market,
        tif_seconds=5,
        fills_log_path=fills_log,
        now_fn=clock.now,
        sleep_fn=clock.sleep,
    )

    print(f"  flip status: {res.get('status')}  flip_id={res.get('flip_id')}")
    out_st = (res.get("out") or {}).get("status")
    in_st = (res.get("in") or {}).get("status")
    print(f"  OUT status={out_st}  IN status={in_st}")
    print(f"  place_calls: {len(place_calls)} (expect 2 — one SELL + one BUY)")

    # Inspect audit log.
    lines = [ln for ln in fills_log.read_text().splitlines() if ln.strip()]
    print(f"  audit log lines: {len(lines)} (expect 2)")
    ok_status = res.get("status") == "complete"
    ok_calls = len(place_calls) == 2
    ok_log = len(lines) == 2
    # Both coids should share the flip prefix.
    ok_coid = all(
        (c.get("coid") or "").startswith(f"pt_flip_{res.get('flip_id')}")
        for c in place_calls
    )
    fills_log.unlink(missing_ok=True)
    return (ok_status and ok_calls and ok_log and ok_coid), (
        f"scenario_4: status={res.get('status')}, "
        f"place_calls={len(place_calls)}, audit_lines={len(lines)}, "
        f"coid_prefix_ok={ok_coid}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    results = []
    for fn in (scenario_1, scenario_2, scenario_3, scenario_4):
        try:
            ok, msg = fn()
        except Exception as e:
            ok, msg = False, f"{fn.__name__}: RAISED {type(e).__name__}: {e}"
        results.append((fn.__name__, ok, msg))

    print("\n=== SUMMARY ===")
    all_ok = True
    for name, ok, msg in results:
        marker = "PASS" if ok else "FAIL"
        print(f"  [{marker}] {name}: {msg}")
        if not ok:
            all_ok = False

    # Dump machine-readable JSON for proof-of-work.
    out_path = Path(__file__).resolve().parents[2] / "research" / "qw_C_swap_2026-05-22" / "smoke_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([
        {"scenario": n, "passed": ok, "detail": msg} for n, ok, msg in results
    ], indent=2) + "\n")
    print(f"\nResults JSON: {out_path}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
