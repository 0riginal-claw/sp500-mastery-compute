"""example_hypotheses.py — Five worked hypotheses for the validator refactor.

Each conforms to `lab.knowledge.indicators.strategy_hypothesis_template()` so the gate at
`validate_test_unit()` accepts it. Each cites its data sources explicitly.

Roles supported by the role parser (see hypothesis_runner._RoleParser):
  • OHLC / Volume: Close, High, Low, Open, Volume
  • Moving averages: SMA(n), EMA(n)
  • Volatility bands: BB.upper(n,k), BB.lower(n,k), BB.mid(n,k), BB.pctb(n,k), ATR(n)
  • Trend / regime: ADX(n), ChopIdx(n), Supertrend(n,mult)
  • Channels: Donchian_UP(n), Donchian_DN(n)
  • Momentum / oscillator: RSI(n), StochK/StochD, Williams(n), CCI(n), MFI(n), ConnorsRSI(p1,p2,p3),
    Fisher(n), MACD(f,s,sg), MACD_hist
  • Volume / conviction: OBV, CMF(n), VolumeExpansion(n,mult), VWAP
  • Logic: AND, OR, NOT, comparisons (>,<,>=,<=,==,!=), arithmetic

Roles whose VALUES come from outside the indicator math (alt-data flags) are expressed as
boolean indicator-like calls registered at runtime via a side-table — e.g. SAP-002 uses
`InsiderForm4_LT5d` which the runner resolves through `lab.knowledge.edgar.get_form4(ticker)`
on demand. Those hypothesis roles fall through to the alt-data hook in hypothesis_runner.
"""

from __future__ import annotations

from typing import Dict, List


# ─────────────────────────────────────────────────────────────────────────────
# SAP-001 — Control: properly-wrapped Donchian-20 breakout (the prior-best OHLCV)
# ─────────────────────────────────────────────────────────────────────────────
SAP_001 = {
    "id": "SAP-001",
    "name": "Donchian-20 UP breakout, ADX-gated, EMA-biased, Volume-confirmed",
    "thesis": (
        "The prior best standalone signal (Donchian-20 UP) gave 0.547 WR on AAPL and 0.753 "
        "on the baseline cohort — but as a standalone trigger with no regime gate, bias filter, "
        "or volume confirmation. This hypothesis wraps the same trigger in the full confluence "
        "stack to test whether confluence raises WR / lowers PBO."
    ),
    # ── Roles ──
    "regime_gate": "ADX(14) > 25",
    "bias_filter": "EMA(9) > EMA(21)",
    "trigger": "Close > Donchian_UP(20)",
    "confirmation": "Volume > 1.5 * SMA(Volume, 20)",
    "timing": "RSI(14) > 50",
    "exit": "1.5 * ATR(14) trailing stop",
    "no_trade": "ChopIdx(14) > 62",
    "side": "long",
    # ── Costs / universe ──
    "cost": "5bps_per_side",
    "universe": "SP500_502",
    "timeframe": "1d",  # default daily (5min FUSE blocked per slice #6 deviations)
    # ── Data sources (must cite) ──
    "data_sources": [
        "yfinance_daily_5yr_cache (lab.indicator_hardening_runner.DRIVE_OHLC_DAILY)",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# SAP-002 — Alt-data trigger: Form 4 insider buy + classical confluence
# ─────────────────────────────────────────────────────────────────────────────
SAP_002 = {
    "id": "SAP-002",
    "name": "Form 4 insider buy >$1M in last 5d + ADX regime + EMA bias + Volume confirm",
    "thesis": (
        "Insider open-market buys above a dollar threshold inside a 5-day window are a "
        "documented bullish catalyst. The hypothesis tests whether that alt-data trigger, "
        "when wrapped in ADX-gated regime + EMA bias + Volume confirmation, exceeds the "
        "Donchian baseline on WR while keeping PBO < 0.15."
    ),
    "regime_gate": "ADX(14) > 20",
    "bias_filter": "EMA(9) > EMA(21)",
    # Alt-data overlay (task #40, 2026-05-28): InsiderForm4_LT5d_GT1M now resolves through
    # hypothesis_runner._AltDataResolver against lab.knowledge.edgar (Form 4). When the
    # EDGAR Form 4 backfill is incomplete (current status — coverage() lists Form 4 as
    # ``forms_partial_backfill_pending``), the resolver returns False for every bar with
    # a diagnostic note, so the trigger degrades to never firing rather than silently
    # falling back to Donchian. The legacy ``alt_data_overlay`` side-table below is kept
    # for backward compat (task #41 orchestrator may still read it).
    "trigger": "Close > Donchian_UP(20) AND InsiderForm4_LT5d_GT1M",
    "alt_data_overlay": {
        "trigger_extra": {
            "name": "InsiderForm4_LT5d_GT1M",
            "loader": "lab.knowledge.edgar.get_form4",
            "params": {"min_value_usd": 1_000_000, "window_days": 5},
            "kind": "boolean_and",
        }
    },
    "confirmation": "Volume > 1.2 * SMA(Volume, 20)",
    "timing": "RSI(14) > 45",
    "exit": "2.0 * ATR(14) trailing stop",
    "no_trade": "ChopIdx(14) > 65",
    "side": "long",
    "cost": "5bps_per_side",
    "universe": "SP500_502",
    "timeframe": "1d",
    "data_sources": [
        "yfinance_daily_5yr_cache",
        "edgar (lab.knowledge.edgar.get_form4) — Form 4 insider transactions, "
        "/My Drive/Ph0tis/Edgar/data/index/edgar.db",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# SAP-003 — Catalyst stack: Congress trade + 8-K + ADX regime + ATR trail
# ─────────────────────────────────────────────────────────────────────────────
SAP_003 = {
    "id": "SAP-003",
    "name": "Congress buy (last 30d) + 8-K material event (last 5d) + ADX>20 + ATR trail",
    "thesis": (
        "Stacking two government-information catalysts (congressional buy in last 30 days + "
        "8-K filed in last 5 days) inside an ADX-gated regime should produce higher hit-rates "
        "than either signal alone. Tests whether multi-catalyst stacks beat pure OHLCV."
    ),
    "regime_gate": "ADX(14) > 20",
    "bias_filter": "EMA(20) > EMA(50)",
    # Alt-data tokens (task #40): CongressBuy_LT30d uses disclosure-date filtering
    # (report_date if present; transaction_date + 45d otherwise) so the join is
    # strictly post-disclosure. 8K_LT5d uses filed_at (post-publication).
    "trigger": "Close > Donchian_UP(20) AND CongressBuy_LT30d AND 8K_LT5d",
    "alt_data_overlay": {
        "trigger_extra": [
            {
                "name": "CongressBuy_LT30d",
                "loader": "lab.knowledge.govtrades.get_congress_trades",
                "params": {"side": "buy", "window_days": 30},
                "kind": "boolean_and",
            },
            {
                "name": "8K_LT5d",
                "loader": "lab.knowledge.edgar.get_8k",
                "params": {"window_days": 5},
                "kind": "boolean_and",
            },
        ]
    },
    "confirmation": "Volume > 1.5 * SMA(Volume, 20)",
    "timing": "RSI(14) > 50",
    "exit": "1.5 * ATR(14) trailing stop",
    "no_trade": "ChopIdx(14) > 62",
    "side": "long",
    "cost": "5bps_per_side",
    "universe": "SP500_502",
    "timeframe": "1d",
    "data_sources": [
        "yfinance_daily_5yr_cache",
        "govtrades (lab.knowledge.govtrades.get_congress_trades) — QuiverQuant Hobbyist",
        "edgar (lab.knowledge.edgar.get_8k) — 8-K material events",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# SAP-004 — Microstructure: Off-exchange (dark-pool) anomaly + RSI<70 + Volume Expansion
# ─────────────────────────────────────────────────────────────────────────────
SAP_004 = {
    "id": "SAP-004",
    "name": "Off-exchange/Dark-Pool anomaly + RSI<70 + Volume Expansion",
    "thesis": (
        "When the Dark Pool Index (DPI) prints above its 90th percentile over the trailing "
        "lookback window, institutional accumulation is likely under way before price action "
        "confirms it. Pair with a non-overbought RSI gate and a clear Volume Expansion to "
        "filter for the moment the move begins. Off-exchange data is sourced from QuiverQuant's "
        "new offexchange table (added this session)."
    ),
    "regime_gate": "ADX(14) > 18",
    "bias_filter": "EMA(9) > EMA(21)",
    # Alt-data token DPI_GT_P90 resolves through _AltDataResolver against the
    # offexchange table. DPI is forward-filled into bar-time using strictly prior
    # event dates (no same-day lookahead), then a 5d rolling 90th-percentile gate
    # is applied. (window_trading_days is fixed at 5 in the token; to use a 90d
    # window the SAP can switch to a custom token like DPI_GT_P90 — see resolver
    # docstring for the supported grammar.)
    "trigger": "Close > SMA(20) AND DPI_GT_P90",
    "alt_data_overlay": {
        "trigger_extra": {
            "name": "DPI_GT_P90",
            "loader": "lab.knowledge.govtrades.get_offexchange",
            "params": {"percentile_window_days": 90, "min_percentile": 90},
            "kind": "boolean_and",
        }
    },
    "confirmation": "VolumeExpansion(20, 1.5)",
    "timing": "RSI(14) < 70",
    "exit": "2.0 * ATR(14) trailing stop",
    "no_trade": "ChopIdx(14) > 60",
    "side": "long",
    "cost": "5bps_per_side",
    "universe": "SP500_502",
    "timeframe": "1d",
    "data_sources": [
        "yfinance_daily_5yr_cache",
        "govtrades.get_offexchange — Dark Pool Index (otc_short/otc_total) per QuiverQuant",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# SAP-005 — Regime-switch composite: trend regime vs. range regime → different sub-strategies
# ─────────────────────────────────────────────────────────────────────────────
SAP_005 = {
    "id": "SAP-005",
    "name": "Regime-switch: ADX>25 → Donchian breakout; ADX<20 → Connors RSI mean-rev; 20-25 → flat",
    "thesis": (
        "ADX cleanly separates trend from range regimes. A single static strategy under-performs "
        "in whichever regime it doesn't fit. This composite swaps between a Donchian-20 trend-"
        "follower (ADX>25) and a Connors-RSI mean-reverter (ADX<20), staying flat in the "
        "transitional band 20-25. Tests whether explicit regime routing improves portfolio-"
        "level WR / Sharpe vs. either child standalone."
    ),
    # Composite hypothesis uses child_hypotheses list — the runner handles regime routing
    "regime_gate": "TRUE",  # parent gate is permissive; children carry their own gates
    "bias_filter": "TRUE",
    "trigger": "FALSE",     # parent's own trigger is unused — child triggers fire
    "confirmation": "TRUE",
    "timing": "TRUE",
    "exit": "FALSE",
    "no_trade": "ChopIdx(14) > 70",
    "cost": "5bps_per_side",
    "universe": "SP500_502",
    "timeframe": "1d",
    "data_sources": [
        "yfinance_daily_5yr_cache",
    ],
    "child_hypotheses": [
        {
            "regime": "ADX(14) > 25",
            "hypothesis": {
                "id": "SAP-005a",
                "regime_gate": "ADX(14) > 25",
                "bias_filter": "EMA(9) > EMA(21)",
                "trigger": "Close > Donchian_UP(20)",
                "confirmation": "Volume > 1.2 * SMA(Volume, 20)",
                "timing": "RSI(14) > 50",
                "exit": "1.5 * ATR(14) trailing stop",
                "no_trade": "FALSE",
                "side": "long",
                "cost": "5bps", "universe": "SP500_502", "timeframe": "1d",
                "data_sources": ["yfinance_daily_5yr_cache"],
            },
        },
        {
            "regime": "ADX(14) < 20",
            "hypothesis": {
                "id": "SAP-005b",
                "regime_gate": "ADX(14) < 20",
                "bias_filter": "TRUE",
                "trigger": "ConnorsRSI(3, 2, 100) < 15",
                "confirmation": "Close < BB.lower(20, 2.0)",
                "timing": "TRUE",
                "exit": "ConnorsRSI(3, 2, 100) > 70",
                "no_trade": "FALSE",
                "side": "long",
                "cost": "5bps", "universe": "SP500_502", "timeframe": "1d",
                "data_sources": ["yfinance_daily_5yr_cache"],
            },
        },
    ],
}


HYPOTHESES: List[Dict] = [SAP_001, SAP_002, SAP_003, SAP_004, SAP_005]
HYPOTHESES_BY_ID: Dict[str, Dict] = {h["id"]: h for h in HYPOTHESES}


if __name__ == "__main__":
    import json
    from knowledge.indicators import validate_test_unit

    print("Hypothesis gate-validation report:")
    for h in HYPOTHESES:
        r = validate_test_unit(h)
        print(f"  {h['id']:8s}  ok={r['ok']}  reason={r['reason']!r}")
        if not r["ok"]:
            print(f"            missing_roles={r['missing_roles']}")
    print("\nFull SAP-001:")
    print(json.dumps(SAP_001, indent=2))
