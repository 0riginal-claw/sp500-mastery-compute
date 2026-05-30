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


# ─────────────────────────────────────────────────────────────────────────────
# CHAMP-002 templates (added 2026-05-29, R5 task #69) — alt-data NUMERIC as
# PRIMARY trigger (not boolean gate). Per chairman verdict: First Principles R5
# + Executor R5 + Expansionist R5 converged on "numeric alt-data signals are
# the next investigational layer; treat as primary entries with classical
# OHLCV as confirmation, not the inverse."
# ─────────────────────────────────────────────────────────────────────────────


# SAP-CAT-CC — CATALYST_CONFLUENCE
#   Alt-data event window (Form 4 insider cluster + 8-K pulse + news velocity)
#   becomes the PRIMARY trigger; OHLCV (ADX regime gate + EMA bias + volume
#   confirmation + 1d VWAP timing) supports it. ATR trailing-stop exit.
#   Anchor TF: 5min entry, 15min/1h/1d as higher-TF gates.
SAP_CAT_CC = {
    "id": "CHAMP-002-CATALYST_CONFLUENCE",
    "name": "Catalyst confluence — F4 cluster + 8-K + news velocity as PRIMARY trigger",
    "thesis": (
        "Numeric alt-data catalysts (Form 4 insider cluster score, 8-K pulse, "
        "news velocity Z) are leading indicators of subsequent OHLCV moves. "
        "Treat them as the PRIMARY trigger (not gate), with classical "
        "confluence (ADX regime, EMA bias, volume) as confirmation. Tests "
        "whether catalyst-as-entry beats catalyst-as-filter, which is the "
        "Mission 12 GOV_AWARE v1 + v2 default mode."
    ),
    "regime_gate": "form4_insider_cluster_score > 60 AND ADX(14) > 18",
    "bias_filter": "EMA(9) > EMA(21)",
    # PRIMARY trigger = alt-data event (numeric tokens). Either an 8-K pulse
    # OR a news velocity surge fires entry.
    "trigger": "8k_pulse >= 1 OR news_velocity_zscore > 1.5",
    # Classical confirmation: volume expansion + 1h not-overbought
    "confirmation": "Volume > 1.3 * SMA(Volume, 20) AND 1h.RSI(14) < 70",
    # Timing: above daily VWAP — bar must be net positive intraday in higher TF
    "timing": "1d.Close > 1d.VWAP",
    "exit": "1.5 * ATR(14) trailing stop",
    "no_trade": "ChopIdx(14) > 65",
    "side": "long",
    "cost": "5bps_per_side",
    "universe": "CHAMP-002 cohort (top-3 ADV × 4 sectors)",
    "timeframe": "5min",
    "timeframe_stack": ["15min", "1h", "1d"],
    "data_sources": [
        "alpaca_5yr local 5Min cache (/Volumes/ZG-2TB/zg/cache/alpaca_5yr/5Min)",
        "yfinance_daily_5yr cache (1d.VWAP + ADX + EMA + Chop)",
        "indicator_compute_altdata.form4_insider_cluster_score (edgar Form 4)",
        "indicator_compute_altdata.eight_k_pulse (edgar 8-K, 5d count)",
        "indicator_compute_altdata.news_velocity_zscore (news 7d-vs-90d Z)",
        "15min/1h — resampled from 5min on the fly",
    ],
}


# SAP-CSR — CROSS_SYMBOL_REGIME
#   Cross-asset macro state (vix_term_structure + hyg_lqd_ratio) defines the
#   regime; sector_rs_rank picks the ticker; spy_beta picks the market
#   exposure. Donchian-20 break is the entry trigger.
SAP_CSR = {
    "id": "CHAMP-002-CROSS_SYMBOL_REGIME",
    "name": "Cross-symbol regime — VIX-TS contango + sector top-3 + spy-beta>0.5",
    "thesis": (
        "Cross-asset macro state is a stronger gate than ticker-local "
        "indicators in regimes where the market regime dominates ticker "
        "idiosyncrasy. Filter for risk-on cross-asset (vix_term_struct > 1, "
        "hyg_lqd_ratio rising), then pick tickers leading their sector "
        "(sector_rs_rank <= 3) with non-trivial market exposure (spy_beta > "
        "0.5). Entry on a daily Donchian-20 break with 1.5× volume expansion."
    ),
    "regime_gate": "vix_term_struct > 1.0 AND hyg_lqd_ratio > 0",
    "bias_filter": "sector_rs_rank <= 3",
    "trigger": "1d.Close > 1d.Donchian_UP(20)",
    "confirmation": "Volume > 1.5 * SMA(Volume, 20)",
    "timing": "spy_beta_60d > 0.5",
    "exit": "2.0 * ATR(14) trailing stop",
    "no_trade": "sector_rs_rank > 6",
    "side": "long",
    "cost": "5bps_per_side",
    "universe": "CHAMP-002 cohort (top-3 ADV × 4 sectors)",
    "timeframe": "5min",
    "timeframe_stack": ["15min", "1h", "1d"],
    "data_sources": [
        "alpaca_5yr local 5Min cache",
        "yfinance_daily_5yr cache (1d.Donchian + ATR + Volume baseline)",
        "indicator_compute_xsym.vix_term_structure (^VIX + ^VXV)",
        "indicator_compute_xsym.hyg_lqd_ratio (HYG + LQD)",
        "indicator_compute_xsym.sector_rs_rank (sector ETFs)",
        "indicator_compute_xsym.spy_beta_60d (60d OLS vs SPY)",
        "15min/1h — resampled from 5min on the fly",
    ],
}


# SAP-GAv2N — GOV_AWARE_v2_numeric template (also referenced as a seed in
# championship_search._SEED_TEMPLATES under seed_id 'GOV_AWARE_v2'). Re-exported
# here as a static example so validate_test_unit can sanity-check it.
SAP_GAV2N = {
    "id": "CHAMP-002-GOV_AWARE_v2_numeric",
    "name": "GOV_AWARE v2 (numeric) — F4 cluster OR congress lead-lag + news/dark-pool",
    "thesis": (
        "Numeric alt-data tokens (F4 cluster, congress lead-lag, news velocity, "
        "dark-pool divergence Z) compose via thresholds and degrade NaN→0 "
        "where the source is missing, vs. v1's boolean tokens which silently "
        "return False. Test whether the numeric overlay beats the boolean."
    ),
    "regime_gate": "ADX(14) > 18 AND 8k_pulse < 3",
    "bias_filter": "EMA(9) > EMA(21)",
    "trigger": "form4_insider_cluster_score > 50 OR congress_lead_lag < 15",
    "confirmation": "news_velocity_zscore > 1.5 OR dark_pool_divergence_z < -1.5",
    "timing": "RSI(14) > 45",
    "exit": "1.5 * ATR(14) trailing stop",
    "no_trade": "ADX(14) < 15",
    "side": "long",
    "cost": "5bps_per_side",
    "universe": "CHAMP-002 cohort (top-3 ADV × 4 sectors)",
    "timeframe": "1d",
    "timeframe_stack": ["1d"],
    "data_sources": [
        "yfinance_daily_5yr cache (OHLCV anchor)",
        "indicator_compute_altdata.form4_insider_cluster_score (edgar Form 4)",
        "indicator_compute_altdata.congress_lead_lag (govtrades)",
        "indicator_compute_altdata.news_velocity_zscore",
        "indicator_compute_altdata.dark_pool_divergence_z (FINRA offexchange)",
        "indicator_compute_altdata.eight_k_pulse (edgar 8-K, 5d count)",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# CHAMP-003B template (added 2026-05-30, R5/R6 task #83) — CONFLUENCE_v1 seed
# Expansionist R6's fix to the CHAMP-002 conjunction trap. OHLCV breakout is
# the trigger; alt-data is the OR'd confirmation. Pre-reg:
# AI-Tools/reports/champ_003b_confluence_pre_registration_2026-05-29.md
# (sha256: 031563e0f14f56877c775ebc8262bb10a4a34344d4822977f9beeb0c240dc2a8)
# ─────────────────────────────────────────────────────────────────────────────


SAP_CONFLUENCE_V1 = {
    "id": "CHAMP-003B-CONFLUENCE_v1",
    "name": "CONFLUENCE v1 — OHLCV breakout + OR'd alt-data confirmation",
    "thesis": (
        "OHLCV breakout (Donchian_UP) is the trigger; alt-data confluence (OR over "
        "form4_insider_cluster_score, news_velocity_zscore, dark_pool_divergence_z, "
        "8k_pulse) is the confirmation. Reverses the CHAMP-002 AND'd 5-gate trap "
        "that produced n_trades=0 for 67% of the cohort. Alt-data ASSISTS the "
        "strategy, it does not gate it (per feedback_indicators_are_assistance)."
    ),
    "regime_gate": "ADX(14) > 20",
    "bias_filter": "EMA(9) > EMA(21)",
    "trigger": "Close > Donchian_UP(20)",
    # OR-confluence over 4 alt-data sources — any one being hot is sufficient
    "confirmation": (
        "form4_insider_cluster_score > 30 "
        "OR news_velocity_zscore > 1.5 "
        "OR dark_pool_divergence_z < -1.5 "
        "OR 8k_pulse >= 1"
    ),
    "timing": "RSI(14) > 40 AND RSI(14) < 70",
    "exit": "1.5 * ATR(14) trailing stop",
    "no_trade": "ChopIdx(14) > 62",
    "side": "long",
    "cost": "5bps_per_side",
    "universe": "CHAMP-003B cohort (12 sector-diverse tickers, identical to CHAMP-002 attempt4)",
    "timeframe": "1d",
    "timeframe_stack": ["1d"],
    "data_sources": [
        "yfinance_daily_5yr cache (daily OHLCV anchor)",
        "indicator_compute_altdata.form4_insider_cluster_score (edgar Form 4)",
        "indicator_compute_altdata.news_velocity_zscore (news 7d-vs-90d Z)",
        "indicator_compute_altdata.dark_pool_divergence_z (FINRA offexchange)",
        "indicator_compute_altdata.eight_k_pulse (edgar 8-K, 5d count)",
    ],
}


HYPOTHESES: List[Dict] = [SAP_001, SAP_002, SAP_003, SAP_004, SAP_005,
                          SAP_CAT_CC, SAP_CSR, SAP_GAV2N, SAP_CONFLUENCE_V1]
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
