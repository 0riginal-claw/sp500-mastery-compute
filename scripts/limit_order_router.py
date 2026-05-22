"""
limit_order_router.py — Smart limit-order routing layer for paper-trade execution.

Motivation (from a6ead433 #1 finding):
  Retail limit-order fill rate ~65% vs HFT <3% → ~10-20bps "free" per trade if
  we cross the spread less than the market-order path does. This module routes
  buy/sell orders through either a marketable limit (cross spread by 1¢) for
  high-conviction signals, or a passive mid-spread limit for normal signals,
  and escalates to a market order after a 60s TIF if unfilled.

Design (Slice 1):
  - Pure routing function `route_order(ticker, side, qty, signal_strength, ...)`
  - Strategy:
      signal_strength HIGH (top decile, prob >= 0.85)   → marketable limit (+1c)
      signal_strength NORMAL                            → passive limit at mid
  - Time-in-force: 60s. If still open after 60s → cancel + escalate to market.
  - All fills logged to paper_trade/limit_order_fills.jsonl with:
      ts, ticker, side, qty/notional, route_type, limit_price, mid_at_submit,
      fill_price, fill_qty, status, escalated, latency_s, bps_vs_mid_at_submit
  - Drop-in wrapper: caller passes a `place_market_fn` callable used as the
    market-fallback path so the router is decoupled from the wrapper SDK and
    fully unit-testable (mocked SDK in smoke test).

Caller contract:
  router_result = route_order(
      ticker, side, qty=N | notional=N, signal_strength="HIGH"|"NORMAL",
      get_quote_fn=<callable -> {"bid": float, "ask": float}>,
      place_limit_fn=<callable(symbol, side, qty, limit_price, tif="day") -> order_dict>,
      cancel_fn=<callable(order_id) -> None>,
      get_order_fn=<callable(order_id) -> order_dict>,  # for poll/status
      place_market_fn=<callable(ticker, side, qty=, notional=) -> order_dict>,  # fallback
      tif_seconds=60,
  )

Return dict shape mirrors the existing _place_market_order_alpaca return:
  {
    "order_id": str,
    "client_order_id": str,
    "status": str,        # "filled" | "partial" | "escalated_market" | "error"
    "qty": int | None,
    "notional": float | None,
    # router-specific:
    "route_type": "marketable_limit" | "passive_limit" | "market_escalation",
    "limit_price": float | None,
    "fill_price": float | None,
    "mid_at_submit": float | None,
    "bps_vs_mid": float | None,
    "escalated": bool,
    "latency_s": float,
  }

Paper-trade safety: NEVER reads/writes real money. All fills mocked or routed
through the Alpaca paper endpoint (URL injected by the caller).
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("limit_order_router")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_TIF_SECONDS = 60
POLL_INTERVAL_S = 1.0
SIGNAL_STRENGTH_HIGH_PROB = 0.85  # top-decile heuristic; tune via measurement
CROSS_PENNY_USD = 0.01  # marketable-limit aggression: cross spread by 1c

# Fills log location — under the mastery paper_trade/ tree, per workspace rules.
_DEFAULT_FILLS_LOG = (
    Path(__file__).resolve().parent.parent / "paper_trade" / "limit_order_fills.jsonl"
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _classify_strength(signal_strength: Any, prob: float | None = None) -> str:
    """Map varying inputs to HIGH | NORMAL. Defensive against missing fields."""
    if isinstance(signal_strength, str):
        s = signal_strength.upper()
        if s in ("HIGH", "STRONG", "TOP_DECILE"):
            return "HIGH"
        if s in ("NORMAL", "MEDIUM", "BASE"):
            return "NORMAL"
    if prob is not None and prob >= SIGNAL_STRENGTH_HIGH_PROB:
        return "HIGH"
    return "NORMAL"


def _round_penny(price: float) -> float:
    """Round to 1¢ to satisfy Alpaca sub-penny pricing guardrail."""
    return round(price + 1e-9, 2)


def _compute_limit_price(
    strength: str,
    side: str,
    bid: float,
    ask: float,
) -> tuple[float, float, str]:
    """Return (limit_price, mid_at_submit, route_type) given quote + strength."""
    mid = (bid + ask) / 2.0
    side_l = side.lower()
    if strength == "HIGH":
        # Marketable limit — cross the spread by 1c to maximize fill probability
        # while capping worst-case slippage. For BUY → ask+1c; for SELL → bid-1c.
        if side_l == "buy":
            return _round_penny(ask + CROSS_PENNY_USD), mid, "marketable_limit"
        return _round_penny(max(bid - CROSS_PENNY_USD, 0.01)), mid, "marketable_limit"
    # NORMAL — passive limit at mid (or slightly more passive)
    return _round_penny(mid), mid, "passive_limit"


def _append_fill_log(record: dict[str, Any], log_path: Path | None = None) -> None:
    """Atomic-append a JSON line. Creates parent dir if missing."""
    path = log_path or _DEFAULT_FILLS_LOG
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception as e:  # noqa: BLE001
        log.warning("limit_order_router: failed to append fill log %s: %s", path, e)


def _bps_vs_mid(fill_price: float | None, mid: float | None, side: str) -> float | None:
    """Positive bps = price improvement vs mid (paid less on buy / received more on sell)."""
    if fill_price is None or mid is None or mid <= 0:
        return None
    sign = -1.0 if side.lower() == "buy" else 1.0
    return sign * (fill_price - mid) / mid * 1e4


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def route_order(  # noqa: PLR0913
    ticker: str,
    side: str,
    qty: int | None = None,
    notional: float | None = None,
    signal_strength: Any = None,
    prob: float | None = None,
    *,
    get_quote_fn: Callable[[str], dict[str, float] | None],
    place_limit_fn: Callable[..., dict[str, Any]],
    cancel_fn: Callable[[str], None],
    get_order_fn: Callable[[str], dict[str, Any]],
    place_market_fn: Callable[..., dict[str, Any]],
    tif_seconds: int = DEFAULT_TIF_SECONDS,
    fills_log_path: Path | None = None,
    now_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Route a buy/sell order via limit-first, market-fallback strategy.

    Caller injects all SDK callables → decouples from alpaca-py + makes testable.
    See module docstring for full contract.
    """
    side_l = side.lower()
    if side_l not in ("buy", "sell"):
        raise ValueError(f"route_order: invalid side {side!r}")
    if qty is None and notional is None:
        raise ValueError("route_order: must pass exactly one of qty/notional")

    started = now_fn()
    strength = _classify_strength(signal_strength, prob)

    # 1) Fetch quote. On failure → degrade to market (caller still gets a fill).
    quote = None
    try:
        quote = get_quote_fn(ticker)
    except Exception as e:  # noqa: BLE001
        log.warning("limit_order_router: quote fetch failed for %s: %s", ticker, e)
    if not quote or quote.get("bid") in (None, 0) or quote.get("ask") in (None, 0):
        log.info("limit_order_router: no quote for %s → market fallback", ticker)
        return _execute_market_fallback(
            ticker, side_l, qty, notional, place_market_fn, started, now_fn,
            fills_log_path, route_type="market_no_quote", strength=strength,
        )

    bid, ask = float(quote["bid"]), float(quote["ask"])
    limit_price, mid, route_type = _compute_limit_price(strength, side_l, bid, ask)

    # 2) Submit limit order. Wrap qty/notional — Alpaca limit orders require qty.
    submit_qty = qty
    if submit_qty is None and notional is not None and ask > 0:
        # Convert notional → integer share qty for limit orders (no fractional limits).
        submit_qty = max(1, int(float(notional) / ask))

    try:
        limit_order = place_limit_fn(
            ticker=ticker,
            side=side_l,
            qty=submit_qty,
            limit_price=limit_price,
            tif="day",
        )
    except Exception as e:  # noqa: BLE001
        log.warning(
            "limit_order_router: limit submit failed for %s: %s → market fallback",
            ticker, e,
        )
        return _execute_market_fallback(
            ticker, side_l, qty, notional, place_market_fn, started, now_fn,
            fills_log_path, route_type="market_limit_submit_failed",
            strength=strength, mid_at_submit=mid,
        )

    order_id = str(limit_order.get("order_id") or limit_order.get("id") or "")
    coid = str(limit_order.get("client_order_id") or "")

    # 3) Poll until filled or TIF expires.
    deadline = started + tif_seconds
    fill_price: float | None = None
    fill_qty: float | None = None
    final_status = "open"
    while now_fn() < deadline:
        sleep_fn(POLL_INTERVAL_S)
        try:
            cur = get_order_fn(order_id)
        except Exception as e:  # noqa: BLE001
            log.debug("limit_order_router: poll error %s: %s", order_id, e)
            continue
        status = str(cur.get("status", "")).lower()
        filled = cur.get("filled_qty") or cur.get("filled") or 0
        try:
            filled_f = float(filled)
        except Exception:  # noqa: BLE001
            filled_f = 0.0
        if filled_f > 0:
            fill_qty = filled_f
            try:
                fill_price = float(cur.get("filled_avg_price") or cur.get("fill_price") or 0) or None
            except Exception:  # noqa: BLE001
                fill_price = None
        if status in ("filled", "done_for_day"):
            final_status = "filled"
            break
        if status in ("canceled", "cancelled", "rejected", "expired"):
            final_status = status
            break

    # 4) If still open or partial after TIF → cancel + escalate to market.
    escalated = False
    if final_status not in ("filled",):
        try:
            cancel_fn(order_id)
        except Exception as e:  # noqa: BLE001
            log.debug("limit_order_router: cancel %s noop/err: %s", order_id, e)

        # Compute remaining quantity to fill via market.
        remaining_qty: int | None = None
        if submit_qty is not None:
            try:
                remaining_qty = max(0, int(submit_qty - (fill_qty or 0)))
            except Exception:  # noqa: BLE001
                remaining_qty = submit_qty

        if (remaining_qty is None and notional is not None) or (remaining_qty and remaining_qty > 0):
            try:
                market_result = place_market_fn(
                    ticker, side=side_l, qty=remaining_qty, notional=notional if remaining_qty is None else None,
                )
                escalated = True
                # If we got a market fill, overwrite the fill_price (effective avg).
                if market_result and market_result.get("fill_price"):
                    try:
                        fill_price = float(market_result["fill_price"])  # type: ignore[arg-type]
                    except Exception:  # noqa: BLE001
                        pass
                if market_result and market_result.get("status"):
                    final_status = f"escalated_market:{market_result['status']}"
                else:
                    final_status = "escalated_market"
                if not order_id:
                    order_id = str(market_result.get("order_id", ""))
                if not coid:
                    coid = str(market_result.get("client_order_id", ""))
            except Exception as e:  # noqa: BLE001
                log.error("limit_order_router: market escalation failed for %s: %s", ticker, e)
                final_status = "error_market_escalation"

    latency = now_fn() - started
    bps = _bps_vs_mid(fill_price, mid, side_l)

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker,
        "side": side_l,
        "qty": qty,
        "notional": notional,
        "signal_strength": strength,
        "route_type": route_type,
        "limit_price": limit_price,
        "mid_at_submit": mid,
        "bid_at_submit": bid,
        "ask_at_submit": ask,
        "fill_price": fill_price,
        "fill_qty": fill_qty,
        "status": final_status,
        "escalated": escalated,
        "latency_s": round(latency, 3),
        "bps_vs_mid": bps,
        "order_id": order_id,
        "client_order_id": coid,
    }
    _append_fill_log(record, fills_log_path)

    return {
        "order_id": order_id,
        "client_order_id": coid,
        "status": final_status,
        "qty": qty,
        "notional": notional,
        "route_type": route_type,
        "limit_price": limit_price,
        "fill_price": fill_price,
        "mid_at_submit": mid,
        "bps_vs_mid": bps,
        "escalated": escalated,
        "latency_s": round(latency, 3),
    }


def _execute_market_fallback(  # noqa: PLR0913
    ticker: str,
    side: str,
    qty: int | None,
    notional: float | None,
    place_market_fn: Callable[..., dict[str, Any]],
    started: float,
    now_fn: Callable[[], float],
    fills_log_path: Path | None,
    *,
    route_type: str,
    strength: str,
    mid_at_submit: float | None = None,
) -> dict[str, Any]:
    """Degrade-to-market path used when limit submission is impossible."""
    try:
        result = place_market_fn(ticker, side=side, qty=qty, notional=notional)
    except Exception as e:  # noqa: BLE001
        log.error("limit_order_router: market fallback also failed for %s: %s", ticker, e)
        result = {
            "order_id": "",
            "client_order_id": "",
            "status": "error",
            "qty": qty,
            "notional": notional,
        }
    latency = now_fn() - started
    fill_price = None
    try:
        if result.get("fill_price") is not None:
            fill_price = float(result["fill_price"])  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001
        pass
    bps = _bps_vs_mid(fill_price, mid_at_submit, side)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker,
        "side": side,
        "qty": qty,
        "notional": notional,
        "signal_strength": strength,
        "route_type": route_type,
        "limit_price": None,
        "mid_at_submit": mid_at_submit,
        "fill_price": fill_price,
        "status": result.get("status", "unknown"),
        "escalated": True,
        "latency_s": round(latency, 3),
        "bps_vs_mid": bps,
        "order_id": result.get("order_id", ""),
        "client_order_id": result.get("client_order_id", ""),
    }
    _append_fill_log(record, fills_log_path)
    return {
        **result,
        "route_type": route_type,
        "limit_price": None,
        "fill_price": fill_price,
        "mid_at_submit": mid_at_submit,
        "bps_vs_mid": bps,
        "escalated": True,
        "latency_s": round(latency, 3),
    }


__all__ = ["route_order", "_classify_strength", "_compute_limit_price"]
