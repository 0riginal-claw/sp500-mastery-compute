"""championship_search.py — Per-ticker championship search loop.

Locked methodology (2026-05-29): the unit of mastery is the TICKER, not the cohort.
Each of 502 tickers gets its own perfected strategy, found via a per-ticker variant
search seeded from Mission 12's 5-agent grid (PURE_TECH, ORB_MORNING, VWAP_MTF,
GOV_AWARE, HYBRID_REGIME).

Wraps `lab.hypothesis_runner.run_hypothesis_for_ticker`. Does NOT modify it.

Public API:
  search_championship(ticker, n_variants=24, timeframe='1d') -> dict
      Run N hypothesis variants on the ticker, return the best by holdout Sharpe.

  variant_generator(ticker, n) -> Iterable[dict]
      Yield N strategy hypotheses for the ticker, seeded from the Mission 12 5-agent
      grid + perturbations. Each variant is a valid `strategy_hypothesis` dict that
      passes `lab.knowledge.indicators.validate_test_unit`.

  update_posterior(ticker, result) -> None
      Append result to data/posteriors/<TICKER>.json. Posterior tracks: locked,
      survived, died SAPs + cumulative best.

  write_championship_file(ticker, best_result) -> str
      Write CHAMPIONSHIP_FORMULA.md per Mission 12 23-item spec. Returns path.

Cohort-level results become a robustness diagnostic, not the primary report. After
per-ticker championships are written, an orchestrator may optionally report cohort-
level summary stats — but the primary output IS the championship files.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))  # for knowledge.*

# Reuse the runner + gate. DO NOT modify hypothesis_runner.py — task #40 owns that file.
import hypothesis_runner as _hr  # noqa: E402
import indicator_hardening_runner as _ihr  # noqa: E402
from knowledge.indicators import validate_test_unit  # noqa: E402
from knowledge.intraday_research import agent_assignment_plan, hard_rules  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Storage paths
# ─────────────────────────────────────────────────────────────────────────────
DRIVE_BASE = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive"
)
POSTERIOR_DIR = DRIVE_BASE / "AI-Tools/s&p500-ticker-mastery/data/posteriors"
TECH0_MASTERED = DRIVE_BASE / "Tech0/Data Master/universe/mastered"  # FUSE-blind likely
LAB_CHAMP_MIRROR = DRIVE_BASE / "AI-Tools/research-lab/data_inventory/championships"
LOCAL_CHAMP_MIRROR = Path("/Volumes/ZG-2TB/zg/championship_mirror")


# ─────────────────────────────────────────────────────────────────────────────
# Variant generator — 5 seeds × perturbations
# ─────────────────────────────────────────────────────────────────────────────

# Per-seed multi-TF stacks (task #53, 2026-05-29). The original Mission 12 spec
# called for 3-TF stacks per seed (1d bias + 15min structure + 5min entry, etc.)
# — task #41 flattened these to single-TF because the 5min FUSE path was hostile
# under load. Task #53 wires the multi-TF data path (via the local
# /Volumes/ZG-2TB/zg/cache/alpaca_5yr mirror + on-the-fly resampling from 5min)
# and restores the 3-TF stacks.
#
# Conventions:
#   * The FIRST entry is the PRIMARY (entry/exit) timeframe. Trades fire here.
#   * Higher TFs are regime gates / bias filters / trend confirmation, accessed
#     via the ``<TF>.<TOKEN>`` cross-TF role-expression syntax (see
#     hypothesis_runner._rewrite_tf_prefixes).
#   * 1min is currently unavailable on the local cache (only 5Min + 1Day are
#     mirrored). ORB_MORNING substitutes "first 5min bar of session" for "1min
#     break of opening range" — honest degradation, documented in the template's
#     data_sources field.
_TF_STACK_BY_SEED = {
    "PURE_TECH":     ["5min", "15min", "1d"],
    "ORB_MORNING":   ["5min", "5min", "1d"],   # 1min unavailable — see note
    "VWAP_MTF":      ["5min", "15min", "1h"],
    "GOV_AWARE":     ["5min", "1h",    "1d"],
    # GOV_AWARE_v2 (task #56) — numeric alt-data tokens. Runs on daily TF
    # because numeric alt-data series (Form 4 cluster score, congress lead-lag,
    # news velocity Z) are bar-timestamp-keyed and disclose at day-level
    # granularity. Multi-TF for v2 is deferred to a follow-up task.
    "GOV_AWARE_v2":  ["1d"],
    # CHAMP-002 seeds (task #69) — alt-data NUMERIC as PRIMARY trigger. Both
    # use the 5min entry + higher TFs as gates pattern (Mission 12 intent),
    # with the cohort being the CHAMP-002 12-ticker pre-registered set.
    "CATALYST_CONFLUENCE":  ["5min", "15min", "1h", "1d"],
    "CROSS_SYMBOL_REGIME":  ["5min", "15min", "1h", "1d"],
    "HYBRID_REGIME": ["5min", "1h",    "1d"],
}

# Cross-asset gate variants (task #57). One per perturbation index; the
# variant_generator picks one from this list per seed variant. The TRUE entry
# is the default-on noop (=> no extra filter, same as legacy behavior).
_CROSS_ASSET_GATE_VARIANTS: List[str] = [
    "TRUE",                              # 0 — noop (back-compat)
    "vix_term_struct > 1.0",             # 1 — contango (risk-on)
    "vix_term_struct < 1.0",             # 2 — backwardation (risk-off)
    "sector_rs_rank <= 3",               # 3 — ticker in top 3 of its sector
    "hyg_lqd_ratio > 0",                 # 4 — credit risk-on
    "abs_spy_beta_60d > 0.5",            # 5 — non-trivial market exposure
]

# 5 seed templates derived from Mission 12's agent_assignment_plan('A').
# Each template encodes the agent's archetype as a runnable hypothesis dict.
# Role expressions support the cross-TF ``<TF>.<TOKEN>`` syntax — higher TFs
# in each seed's stack are used for bias/regime gates so the seed actually
# behaves as a multi-timeframe strategy (Mission 12 intent).
_SEED_TEMPLATES = [
    {
        "seed_id": "PURE_TECH",
        "theory": (
            "Pure technical confluence — 1D trend bias (EMA stack) + 15min higher-low "
            "structure (RSI stays above 50 + Close > 15min Donchian mid) + 5min entry "
            "on VWAP reclaim. Three-TF Mission 12 design."
        ),
        "side": "long",
        # 1D regime: ADX above thresh on the daily TF. Forward-filled onto 5min bars.
        "regime_gate": "1d.ADX(14) > {adx_thresh}",
        # 1D bias: daily EMA trend stack must agree.
        "bias_filter": "1d.EMA({ema_fast}) > 1d.EMA({ema_slow})",
        # 15min structure: close above mid-Donchian AND RSI above 50 (higher-low proxy).
        # 5min entry trigger: VWAP reclaim AND close > 5min Donchian.
        "trigger": "Close > VWAP AND Close > Donchian_UP({donch}) AND 15min.RSI(14) > 50",
        "confirmation": "Volume > 1.2 * SMA(Volume, 20)",
        "timing": "RSI(14) > 50",
        "exit": "{atr_mult} * ATR(14) trailing stop",
        "no_trade": "1d.ChopIdx(14) > 62",
        "data_sources": [
            "alpaca_5yr local 5Min cache (/Volumes/ZG-2TB/zg/cache/alpaca_5yr/5Min)",
            "yfinance_daily_5yr cache (lab.indicator_hardening_runner.DRIVE_OHLC_DAILY)",
            "15min — resampled from 5min on the fly (hypothesis_runner._resample_ohlcv_from_5min)",
        ],
    },
    {
        "seed_id": "ORB_MORNING",
        "theory": (
            "Opening Range Breakout — 1D direction bias + 5min ORB (first bar of session "
            "high/low) + 5min trigger on break of opening range. "
            "1min not available in local cache; uses 5min first-bar of session as ORB "
            "proxy (honest degradation, documented)."
        ),
        "side": "long",
        # 1D regime: daily ADX above thresh.
        "regime_gate": "1d.ADX(14) > {adx_thresh}",
        # 1D bias: daily EMA stack — direction filter.
        "bias_filter": "1d.EMA({ema_fast}) > 1d.EMA({ema_slow})",
        # 5min entry trigger: above VWAP (intraday bias) AND Donchian break (ORB proxy).
        "trigger": "Close > VWAP AND Close > Donchian_UP({donch})",
        "confirmation": "VolumeExpansion(20, 1.5)",
        "timing": "RSI(14) > 50",
        "exit": "{atr_mult} * ATR(14) trailing stop",
        "no_trade": "1d.ChopIdx(14) > 60",
        "data_sources": [
            "alpaca_5yr local 5Min cache (/Volumes/ZG-2TB/zg/cache/alpaca_5yr/5Min)",
            "yfinance_daily_5yr cache (1D direction bias)",
            "1Min: NOT LOADED — local cache has 5Min + 1Day only; ORB-1min trigger "
            "approximated by 5min first-bar break + VWAP confluence. Adding 1Min "
            "directly is future work (FUSE blocks under load on this Mac).",
        ],
    },
    {
        "seed_id": "VWAP_MTF",
        "theory": (
            "VWAP-anchored multi-timeframe continuation — 1H trend (Close vs 1H VWAP "
            "+ EMA stack), 15min pullback setup (RSI dip below 60, Close holds VWAP), "
            "5min entry on continuation."
        ),
        "side": "long",
        # 1H trend bias: 1H VWAP + EMA stack.
        "regime_gate": "1h.Close > 1h.VWAP",
        "bias_filter": "1h.EMA({ema_fast}) > 1h.EMA({ema_slow})",
        # 15min setup: pullback then continuation (RSI in setup band).
        # 5min entry: Close > 5min VWAP + 5min EMA(fast).
        "trigger": "Close > VWAP AND Close > EMA({ema_fast}) AND 15min.RSI(14) > 45",
        "confirmation": "Volume > 1.0 * SMA(Volume, 20)",
        "timing": "RSI(14) > 45 AND RSI(14) < 75",
        "exit": "{atr_mult} * ATR(14) trailing stop",
        "no_trade": "1h.ChopIdx(14) > 65",
        "data_sources": [
            "alpaca_5yr local 5Min cache",
            "1H + 15min — resampled from 5min on the fly",
        ],
    },
    {
        "seed_id": "GOV_AWARE",
        "theory": (
            "Catalyst-confirmed momentum — alt-data event (Form 4 / Congress / 8-K) "
            "AND 1H confluence (above 1H VWAP + EMA stack) AND 5min entry trigger. "
            "When the alt-data overlay's EDGAR/Govtrades backfill is incomplete, "
            "tokens resolve to False and the variant naturally never fires "
            "(no silent degradation to pure-tech)."
        ),
        "side": "long",
        # 1D regime: daily ADX above thresh (long-trend qualifier).
        "regime_gate": "1d.ADX(14) > {adx_thresh}",
        # 1H confluence: above 1H VWAP + EMA stack (intraday bias filter).
        "bias_filter": "1h.Close > 1h.VWAP AND 1h.EMA({ema_fast}) > 1h.EMA({ema_slow})",
        # 5min entry: Close > 5min Donchian AND alt-data event window present.
        # InsiderForm4_LT30d is the canonical "Form 4 buy in last 30d" token — alt
        # resolver returns False until the EDGAR Form 4 backfill lands.
        "trigger": "Close > Donchian_UP({donch}) AND InsiderForm4_LT30d",
        "alt_data_overlay": {
            "trigger_extra": [
                {
                    "name": "InsiderForm4_LT30d",
                    "loader": "lab.knowledge.edgar.get_form4",
                    "params": {"min_value_usd": 500_000, "window_days": 30},
                    "kind": "boolean_and",
                },
            ]
        },
        "confirmation": "VolumeExpansion(20, 1.5)",
        "timing": "RSI(14) > 45",
        "exit": "{atr_mult} * ATR(14) trailing stop",
        "no_trade": "1d.ADX(14) < 15",
        "data_sources": [
            "alpaca_5yr local 5Min cache",
            "yfinance_daily_5yr cache (1D regime gate)",
            "1H — resampled from 5min on the fly",
            "edgar (lab.knowledge.edgar.get_form4) — Form 4 insider transactions "
            "[backfill pending — overlay token resolves to False until landed]",
        ],
    },
    {
        "seed_id": "GOV_AWARE_v2",
        "theory": (
            "Catalyst-confirmed momentum (NUMERIC alt-data, task #56) — Form 4 insider "
            "cluster score + congress lead-lag trigger, news velocity Z + dark-pool "
            "divergence Z confirmation, 8K pulse / ADX regime gate. Numeric tokens "
            "compare via thresholds and degrade NaN→0 (no signal) where the source "
            "is missing, vs. v1's boolean tokens which silently return False."
        ),
        "side": "long",
        # Daily TF only — numeric alt-data series are bar-keyed at daily granularity.
        # ADX gate is the OHLCV anchor; 8k_pulse < 3 keeps the regime from firing
        # during catalyst-overload chop (e.g. earnings + 10-K + 8-K in the same week).
        "regime_gate": "ADX(14) > {adx_thresh} AND 8k_pulse < 3",
        "bias_filter": "EMA({ema_fast}) > EMA({ema_slow})",
        # Numeric trigger: ANY insider cluster OR recent congress disclosure.
        # form4_insider_cluster_score > 50 = meaningful insider conviction
        # congress_lead_lag < 15 = disclosure within last 15 days
        "trigger": "form4_insider_cluster_score > 50 OR congress_lead_lag < 15",
        # Numeric confirmation: news flow surge OR institutional positioning shift.
        # news_velocity_zscore > 1.5 = catalyst-grade news flow
        # dark_pool_divergence_z < -1.5 = dark-pool short-volume z-score collapse
        "confirmation": "news_velocity_zscore > 1.5 OR dark_pool_divergence_z < -1.5",
        "timing": "RSI(14) > 45",
        "exit": "{atr_mult} * ATR(14) trailing stop",
        "no_trade": "ADX(14) < 15",
        "data_sources": [
            "yfinance_daily_5yr cache (OHLCV anchor)",
            "indicator_compute_altdata.form4_insider_cluster_score (edgar Form 4)",
            "indicator_compute_altdata.congress_lead_lag (govtrades congress disclosures)",
            "indicator_compute_altdata.news_velocity_zscore (news 7d-vs-90d Z)",
            "indicator_compute_altdata.dark_pool_divergence_z (FINRA offexchange)",
            "indicator_compute_altdata.eight_k_pulse (edgar 8-K, 5d count)",
        ],
    },
    # ─── CHAMP-002 seeds (task #69, 2026-05-29) ────────────────────────────
    # Alt-data NUMERIC tokens become the PRIMARY trigger (not boolean gate).
    # See AI-Tools/reports/champ_002_pre_registration_2026-05-29.md for the
    # locked thresholds + 12-ticker cohort that this seed targets.
    {
        "seed_id": "CATALYST_CONFLUENCE",
        "theory": (
            "Numeric alt-data catalysts (Form 4 insider cluster, 8-K pulse, "
            "news velocity Z) are leading indicators. Treat them as PRIMARY "
            "entry trigger; OHLCV (ADX regime, EMA bias, volume confirm, "
            "daily VWAP timing) is confirmation. CHAMP-002 investigational."
        ),
        "side": "long",
        # 1D regime gate: F4 insider cluster + daily ADX both must agree
        "regime_gate": "form4_insider_cluster_score > 60 AND 1d.ADX(14) > {adx_thresh}",
        # 1D bias filter: daily EMA stack
        "bias_filter": "1d.EMA({ema_fast}) > 1d.EMA({ema_slow})",
        # PRIMARY trigger = alt-data numeric token threshold (NOT OHLCV)
        "trigger": "8k_pulse >= 1 OR news_velocity_zscore > 1.5",
        # Confirmation: volume expansion at the entry TF + 1h not-overbought
        "confirmation": "Volume > 1.3 * SMA(Volume, 20) AND 1h.RSI(14) < 70",
        # Timing: above daily VWAP — net-positive trading day
        "timing": "1d.Close > 1d.VWAP",
        "exit": "{atr_mult} * ATR(14) trailing stop",
        "no_trade": "1d.ChopIdx(14) > 65",
        "data_sources": [
            "alpaca_5yr local 5Min cache (/Volumes/ZG-2TB/zg/cache/alpaca_5yr/5Min)",
            "yfinance_daily_5yr cache (1d.VWAP + ADX + EMA + Chop + Volume baseline)",
            "indicator_compute_altdata.form4_insider_cluster_score (edgar Form 4)",
            "indicator_compute_altdata.eight_k_pulse (edgar 8-K, 5d count)",
            "indicator_compute_altdata.news_velocity_zscore (news 7d-vs-90d Z)",
            "15min/1h — resampled from 5min on the fly",
        ],
    },
    {
        "seed_id": "CROSS_SYMBOL_REGIME",
        "theory": (
            "Cross-asset macro state (vix_term_structure + hyg_lqd_ratio) "
            "defines the regime; sector_rs_rank picks the ticker; spy_beta_60d "
            "picks market exposure. Donchian-20 break is the entry trigger. "
            "Tests whether macro-cross-asset gating beats ticker-local gates."
        ),
        "side": "long",
        # Cross-asset regime: risk-on (vix-TS in contango + HYG/LQD rising)
        "regime_gate": "vix_term_struct > 1.0 AND hyg_lqd_ratio > 0",
        # Bias: ticker is leading its sector (top-3 by RS rank)
        "bias_filter": "sector_rs_rank <= 3",
        # Entry: daily Donchian-20 break (slow-trend follower)
        "trigger": "1d.Close > 1d.Donchian_UP({donch})",
        # Confirmation: volume expansion
        "confirmation": "Volume > 1.5 * SMA(Volume, 20)",
        # Timing: non-trivial market exposure
        "timing": "spy_beta_60d > 0.5",
        "exit": "{atr_mult} * ATR(14) trailing stop",
        "no_trade": "sector_rs_rank > 6",
        "data_sources": [
            "alpaca_5yr local 5Min cache",
            "yfinance_daily_5yr cache (1d.Donchian + ATR + Volume baseline)",
            "indicator_compute_xsym.vix_term_structure (^VIX + ^VXV)",
            "indicator_compute_xsym.hyg_lqd_ratio (HYG + LQD)",
            "indicator_compute_xsym.sector_rs_rank (sector ETFs)",
            "indicator_compute_xsym.spy_beta_60d (60d OLS vs SPY)",
            "15min/1h — resampled from 5min on the fly",
        ],
    },
    {
        "seed_id": "HYBRID_REGIME",
        "theory": (
            "Regime-switching MTF: 1D ADX regime selects child strategy — trend child "
            "(1H BB setup + 5min Donchian entry) when 1d.ADX > thresh, range child "
            "(5min Connors-RSI + BB.lower mean-revert) when 1d.ADX < 20."
        ),
        "side": "long",
        "regime_gate": "TRUE",
        "bias_filter": "TRUE",
        "trigger": "FALSE",  # parent unused; children fire
        "confirmation": "TRUE",
        "timing": "TRUE",
        "exit": "FALSE",
        "no_trade": "1d.ChopIdx(14) > 70",
        "data_sources": [
            "alpaca_5yr local 5Min cache",
            "yfinance_daily_5yr cache (1D regime selector)",
            "1H — resampled from 5min on the fly",
        ],
        "child_hypotheses": [
            {
                # Trend child: 1D ADX confirms trend, 1H BB(2) setup, 5min Donchian
                # break entry. Higher TFs as gates; 5min as the entry timeframe.
                "regime": "1d.ADX(14) > {adx_thresh}",
                "hypothesis": {
                    "id": "_trend",
                    "regime_gate": "1d.ADX(14) > {adx_thresh}",
                    "bias_filter": "1h.Close > 1h.BB.mid(20, 2.0) AND 1h.EMA({ema_fast}) > 1h.EMA({ema_slow})",
                    "trigger": "Close > Donchian_UP({donch}) AND Close > VWAP",
                    "confirmation": "Volume > 1.2 * SMA(Volume, 20)",
                    "timing": "RSI(14) > 50",
                    "exit": "{atr_mult} * ATR(14) trailing stop",
                    "no_trade": "FALSE",
                    "side": "long",
                    "cost": "5bps_per_side",
                    "universe": "SP500_502",
                    "timeframe": "5min",
                    "data_sources": ["alpaca_5yr 5Min", "yfinance daily", "1H resampled"],
                },
            },
            {
                # Range child: 1D ADX low → mean-revert on 5min via Connors RSI +
                # BB.lower setup. No higher-TF gates (range regimes act locally).
                "regime": "1d.ADX(14) < 20",
                "hypothesis": {
                    "id": "_range",
                    "regime_gate": "1d.ADX(14) < 20",
                    "bias_filter": "TRUE",
                    "trigger": "ConnorsRSI(3, 2, 100) < 15",
                    "confirmation": "Close < BB.lower(20, 2.0)",
                    "timing": "TRUE",
                    "exit": "ConnorsRSI(3, 2, 100) > 70",
                    "no_trade": "FALSE",
                    "side": "long",
                    "cost": "5bps_per_side",
                    "universe": "SP500_502",
                    "timeframe": "5min",
                    "data_sources": ["alpaca_5yr 5Min", "yfinance daily"],
                },
            },
        ],
    },
]


# Perturbation grid. Cartesian product is large; we sample a stratified subset per seed.
# Defaults give 5 seeds * 5 perturbations = 25 variants (close to default n=24).
_PERTURB_GRID = {
    "adx_thresh": [15, 20, 22, 25],
    "ema_fast": [5, 9, 13, 21],
    "ema_slow": [13, 21, 34, 55],
    "donch": [10, 20, 40],
    "atr_mult": [1.0, 1.5, 2.0],
}


def _format_template(tmpl: dict, p: dict) -> dict:
    """Render `{key}` placeholders inside string role expressions using params dict.

    Recurses into child_hypotheses. Leaves non-string fields alone.
    """
    out: Dict[str, Any] = {}
    for k, v in tmpl.items():
        if isinstance(v, str):
            try:
                out[k] = v.format(**p)
            except (KeyError, IndexError):
                out[k] = v
        elif isinstance(v, list) and k == "child_hypotheses":
            new_children = []
            for c in v:
                new_c = {}
                for ck, cv in c.items():
                    if isinstance(cv, str):
                        try:
                            new_c[ck] = cv.format(**p)
                        except (KeyError, IndexError):
                            new_c[ck] = cv
                    elif isinstance(cv, dict):
                        new_c[ck] = _format_template(cv, p)
                    else:
                        new_c[ck] = cv
                new_children.append(new_c)
            out[k] = new_children
        else:
            out[k] = v
    return out


def _sane_param_combo(p: dict) -> bool:
    """Reject obviously broken perturb combos (e.g. EMA fast >= EMA slow)."""
    if p.get("ema_fast", 0) >= p.get("ema_slow", 1):
        return False
    return True


def _stratified_perturb_samples(n_per_seed: int, priors: Optional[dict] = None) -> List[dict]:
    """Yield n_per_seed perturbation dicts, sampled deterministically across the grid.

    If `priors` is provided (from a prior posterior's `next_priors`), bias the first
    sample toward the prior's preferred axis values. This makes search iterations
    converge toward the posterior's best region.
    """
    base_combos = []
    # Canonical 5 corners: low-ADX/fast-EMA, mid, default, slow-EMA, high-ADX/wide-trail
    canonical = [
        {"adx_thresh": 15, "ema_fast": 5,  "ema_slow": 13, "donch": 10, "atr_mult": 1.0},
        {"adx_thresh": 20, "ema_fast": 9,  "ema_slow": 21, "donch": 20, "atr_mult": 1.5},
        {"adx_thresh": 22, "ema_fast": 13, "ema_slow": 34, "donch": 20, "atr_mult": 1.5},
        {"adx_thresh": 25, "ema_fast": 9,  "ema_slow": 21, "donch": 40, "atr_mult": 2.0},
        {"adx_thresh": 20, "ema_fast": 21, "ema_slow": 55, "donch": 20, "atr_mult": 2.0},
    ]
    # If priors present, prepend a prior-biased combo
    if priors:
        prior_combo = canonical[1].copy()  # default mid
        for k in ("adx_thresh", "ema_fast", "ema_slow", "donch", "atr_mult"):
            v = priors.get(k)
            if isinstance(v, (int, float)) and v in _PERTURB_GRID[k]:
                prior_combo[k] = v
        if _sane_param_combo(prior_combo):
            base_combos.append(prior_combo)
    for c in canonical:
        if _sane_param_combo(c) and c not in base_combos:
            base_combos.append(c)
    # If user asked for more than canonical, extend by walking the grid
    if n_per_seed > len(base_combos):
        extra_needed = n_per_seed - len(base_combos)
        for adx in _PERTURB_GRID["adx_thresh"]:
            for ema_f, ema_s in zip(_PERTURB_GRID["ema_fast"], _PERTURB_GRID["ema_slow"]):
                for d in _PERTURB_GRID["donch"]:
                    for a in _PERTURB_GRID["atr_mult"]:
                        c = {"adx_thresh": adx, "ema_fast": ema_f, "ema_slow": ema_s,
                             "donch": d, "atr_mult": a}
                        if c in base_combos:
                            continue
                        if not _sane_param_combo(c):
                            continue
                        base_combos.append(c)
                        extra_needed -= 1
                        if extra_needed <= 0:
                            break
                    if extra_needed <= 0: break
                if extra_needed <= 0: break
            if extra_needed <= 0: break
    return base_combos[:n_per_seed]


def variant_generator(
    ticker: str,
    n: int,
    priors: Optional[dict] = None,
    seeds: Optional[List[str]] = None,
) -> Iterable[dict]:
    """Yield up to N strategy hypotheses for the ticker.

    Layout: divide N across active seed templates as evenly as possible. For each
    seed, apply stratified perturbations. Each yielded variant has a unique SAP ID
    `SAP-<TICKER>-<NNN>` and a `parent_seed_id` field for traceability.

    If `seeds` is given (list of seed_id strings), the generator restricts the
    search to those seeds only — used by CHAMP-002 to run just the
    CATALYST_CONFLUENCE + CROSS_SYMBOL_REGIME + GOV_AWARE_v2 trio without the
    Mission 12 legacy 5-agent grid.

    Variants that fail `validate_test_unit` are skipped (with a warning printed) —
    the generator may yield FEWER than N if many variants fail validation. In
    practice all template-rendered variants pass the gate since required roles are
    always present.
    """
    active_seeds = _SEED_TEMPLATES
    if seeds:
        wanted = {s.strip() for s in seeds if s.strip()}
        active_seeds = [s for s in _SEED_TEMPLATES if s["seed_id"] in wanted]
        if not active_seeds:
            print(f"  [variant_generator] no seeds matched {seeds!r}; "
                  f"available: {[s['seed_id'] for s in _SEED_TEMPLATES]}", flush=True)
            return
    n_seeds = len(active_seeds)
    base_per = max(1, n // n_seeds)
    remainder = n - base_per * n_seeds
    counts = [base_per + (1 if i < remainder else 0) for i in range(n_seeds)]

    seq = 0
    for seed, cnt in zip(active_seeds, counts):
        if cnt <= 0:
            continue
        # Resolve the TF stack for this seed. Stacks come from _TF_STACK_BY_SEED;
        # the primary TF is the first entry (entry/exit fires here), the rest are
        # higher TFs available via the cross-TF role-expression syntax.
        tf_stack = list(_TF_STACK_BY_SEED.get(seed["seed_id"], ["1d"]))
        # Dedupe while preserving order (e.g. ORB_MORNING has 5min twice).
        seen = []
        for t in tf_stack:
            if t not in seen:
                seen.append(t)
        tf_stack = seen
        primary_tf = tf_stack[0]
        for perturb_idx, p in enumerate(_stratified_perturb_samples(cnt, priors=priors)):
            seq += 1
            tmpl = _format_template(seed, p)
            sap_id = f"SAP-{ticker.upper()}-{seq:03d}"
            # Cross-asset gate (task #57): assign one perturbation per variant by
            # stepping through _CROSS_ASSET_GATE_VARIANTS. Variant 0 is the legacy
            # noop (TRUE) — preserves back-compat for the first variant per seed.
            # GOV_AWARE_v2 (numeric) skips the noop and starts at idx 1 so EVERY
            # v2 variant exercises an actual cross-asset filter — matches the task
            # spec "≥1 variant per seed uses a cross_asset_gate".
            if seed["seed_id"] == "GOV_AWARE_v2":
                xa_idx = 1 + (perturb_idx % (len(_CROSS_ASSET_GATE_VARIANTS) - 1))
            else:
                xa_idx = perturb_idx % len(_CROSS_ASSET_GATE_VARIANTS)
            cross_asset_gate = _CROSS_ASSET_GATE_VARIANTS[xa_idx]
            variant = {
                "id": sap_id,
                "name": f"{seed['seed_id']} @ ADX>{p['adx_thresh']} EMA({p['ema_fast']}/{p['ema_slow']}) "
                        f"Donch{p['donch']} {p['atr_mult']}xATR  TF={'>'.join(tf_stack)} XA={xa_idx}",
                "thesis": (
                    f"Per-ticker championship variant for {ticker} seeded from "
                    f"Mission 12 agent grid '{seed['seed_id']}' "
                    f"(MTF stack: {' / '.join(tf_stack)}, XA-gate: {cross_asset_gate}): "
                    f"{seed['theory']}"
                ),
                "parent_seed_id": seed["seed_id"],
                "perturb_params": p,
                # Roles (rendered from template)
                "regime_gate": tmpl.get("regime_gate", "TRUE"),
                "bias_filter": tmpl.get("bias_filter", "TRUE"),
                "trigger": tmpl.get("trigger", "FALSE"),
                "confirmation": tmpl.get("confirmation", "TRUE"),
                "timing": tmpl.get("timing", "TRUE"),
                "exit": tmpl.get("exit", "FALSE"),
                "no_trade": tmpl.get("no_trade", "FALSE"),
                "side": tmpl.get("side", "long"),
                # Cross-asset regime gate (task #57). Evaluated by hypothesis_runner
                # via _XSymResolver and AND'd onto regime_gate. TRUE = noop.
                "cross_asset_gate": cross_asset_gate,
                # Required gate fields
                "cost": "5bps_per_side",
                "universe": f"single_ticker:{ticker.upper()}",
                # Multi-TF (task #53): primary is the entry TF; the stack is
                # propagated through hypothesis_runner.run_hypothesis_for_ticker.
                "timeframe": primary_tf,
                "timeframe_stack": tf_stack,
                "data_sources": list(seed.get("data_sources", [])),
            }
            # Carry child_hypotheses / alt_data_overlay through if present
            if "child_hypotheses" in tmpl:
                variant["child_hypotheses"] = tmpl["child_hypotheses"]
            if "alt_data_overlay" in tmpl:
                variant["alt_data_overlay"] = tmpl["alt_data_overlay"]
            # Gate-validate
            gate = validate_test_unit(variant)
            if not gate["ok"]:
                print(f"  [variant_generator] {sap_id} FAILED gate: {gate['reason']}", flush=True)
                continue
            yield variant


# ─────────────────────────────────────────────────────────────────────────────
# Posterior
# ─────────────────────────────────────────────────────────────────────────────


def _posterior_path(ticker: str) -> Path:
    POSTERIOR_DIR.mkdir(parents=True, exist_ok=True)
    return POSTERIOR_DIR / f"{ticker.upper()}.json"


def _load_posterior(ticker: str) -> dict:
    p = _posterior_path(ticker)
    if not p.exists():
        return {
            "ticker": ticker.upper(),
            "history": [],
            "current_best": None,
            "current_best_score": None,
            "search_iterations": 0,
            "next_priors": {},
        }
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as e:
        print(f"  [posterior] couldn't load {p}: {e} — starting fresh", flush=True)
        return {
            "ticker": ticker.upper(), "history": [],
            "current_best": None, "current_best_score": None,
            "search_iterations": 0, "next_priors": {},
        }


def _score_for_posterior(result: dict) -> float:
    """Combined score used to rank variants. Holdout Sharpe if usable, else WR.

    Returns -inf if the result is no_data / errored / has no trades.
    """
    if result.get("status") != "ok":
        return float("-inf")
    if (result.get("n_trades") or 0) <= 0:
        return float("-inf")
    hs = result.get("holdout_sharpe")
    if hs is not None and isinstance(hs, (int, float)) and not np.isnan(hs):
        return float(hs)
    wr = result.get("win_rate")
    if wr is not None and isinstance(wr, (int, float)) and not np.isnan(wr):
        # Wrap WR in [-1, +1] so it's comparable in fallback only
        return float(wr - 0.5)
    return float("-inf")


def _classify_status(result: dict) -> str:
    """Per-variant status label: locked / survived / died.

    locked  — passes Mission 12 promotion thresholds (PBO<0.15, DSR>0.5, WR>=0.50)
    survived — produced trades, didn't meet locked threshold, but didn't blow up
    died    — no trades, NaN sharpe, or zero edge
    """
    if result.get("status") != "ok":
        return "died"
    if (result.get("n_trades") or 0) == 0:
        return "died"
    wr = result.get("win_rate")
    pbo = result.get("pbo")
    dsr = result.get("dsr_prob")
    hs = result.get("holdout_sharpe")
    if (wr is not None and not np.isnan(wr) and wr >= 0.50
            and pbo is not None and not np.isnan(pbo) and pbo < 0.15
            and dsr is not None and not np.isnan(dsr) and dsr > 0.50
            and hs is not None and not np.isnan(hs) and hs > 0):
        return "locked"
    # Survived if at least produced trades and didn't fully blow up
    if (hs is not None and not np.isnan(hs)) or (wr is not None and not np.isnan(wr)):
        return "survived"
    return "died"


def _derive_next_priors(history: List[dict]) -> dict:
    """Look at locked/survived entries and produce a `next_priors` dict to bias the
    next search iteration toward the best perturbation axis values.
    """
    locked_or_survived = [
        h for h in history
        if h.get("status") in ("locked", "survived") and h.get("perturb_params")
    ]
    if not locked_or_survived:
        return {}
    # Pick the single highest-scoring entry's perturb params as the prior
    best = max(
        locked_or_survived,
        key=lambda h: h.get("score") if h.get("score") is not None else float("-inf"),
    )
    return dict(best.get("perturb_params", {}))


def update_posterior(ticker: str, result: dict) -> None:
    """Append result to data/posteriors/<TICKER>.json. Atomic write."""
    post = _load_posterior(ticker)
    score = _score_for_posterior(result)
    status = _classify_status(result)
    entry = {
        "sap_id": result.get("hypothesis_id") or result.get("sap_id") or "SAP-?",
        "parent_seed_id": result.get("parent_seed_id"),
        "perturb_params": result.get("perturb_params"),
        "status": status,
        "score": None if score == float("-inf") else score,
        "result": {
            k: result.get(k) for k in (
                "n_obs", "n_trades", "win_rate", "full_sharpe",
                "is_sharpe_median", "oos_sharpe_median", "wfe",
                "pbo", "dsr_prob", "holdout_sharpe",
            )
        },
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    post["history"].append(entry)
    post["search_iterations"] = len(post["history"])
    # Update current_best
    best_score = post.get("current_best_score")
    if best_score is None or (entry["score"] is not None and entry["score"] > best_score):
        post["current_best"] = entry["sap_id"]
        post["current_best_score"] = entry["score"]
    post["next_priors"] = _derive_next_priors(post["history"])
    # Atomic write
    p = _posterior_path(ticker)
    tmp = p.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(post, indent=2, default=str))
        tmp.replace(p)
    except OSError as e:
        print(f"  [posterior] write failed for {ticker}: {e}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Championship file writer — Mission 12's 23-item spec
# ─────────────────────────────────────────────────────────────────────────────


def _fmt(v: Any, default: str = "n/a") -> str:
    if v is None:
        return default
    if isinstance(v, float):
        if np.isnan(v):
            return default
        return f"{v:.4f}"
    return str(v)


def _ticker_metadata(ticker: str) -> Dict[str, str]:
    """Look up sector / market-cap / ADV / vol-decile / beta for the championship file.

    Delegates to `lab.championship_metadata.enrich_metadata`, which pulls:
      sector     ← S&P GICS Sector (canonical from sp500_constituents-detailed.csv)
      mcap       ← shares_outstanding × last_close
      adv_20d    ← mean(volume[-20:]) from yfinance_5yr daily cache
      vol_decile ← annualized realized vol → decile within 502-universe distribution
      beta       ← 60-day OLS regression of ticker returns vs SPY returns

    Falls back to placeholders if the enricher import fails (e.g. running in an
    isolated context without the lab module). The championship file STILL gets
    written either way — the values just degrade to 'unknown_pending_metadata_backfill'
    so the downstream backfill pass can patch them in later.
    """
    try:
        from . import championship_metadata as _cm  # type: ignore
    except ImportError:
        try:
            import championship_metadata as _cm  # type: ignore
        except ImportError as e:
            print(f"  [championship] enricher unavailable ({e}) — placeholders inserted",
                  flush=True)
            return {
                "sector": "unknown_pending_metadata_backfill",
                "mcap": "unknown_pending_metadata_backfill",
                "adv": "unknown_pending_metadata_backfill",
                "vol_decile": "unknown_pending_metadata_backfill",
                "beta": "unknown_pending_metadata_backfill",
            }
    try:
        m = _cm.enrich_metadata(ticker, formatted=True)
        return {
            "sector": m.get("sector") or "unknown",
            "mcap": m.get("mcap") or "unknown",
            "adv": m.get("adv_20d") or "unknown",
            "vol_decile": m.get("vol_decile") or "unknown",
            "beta": m.get("beta") or "unknown",
        }
    except Exception as e:
        # Honest degradation: enricher had a runtime error, mark for backfill.
        print(f"  [championship] enrich_metadata({ticker}) raised {type(e).__name__}: {e}",
              flush=True)
        return {
            "sector": "unknown_pending_metadata_backfill",
            "mcap": "unknown_pending_metadata_backfill",
            "adv": "unknown_pending_metadata_backfill",
            "vol_decile": "unknown_pending_metadata_backfill",
            "beta": "unknown_pending_metadata_backfill",
        }


def write_championship_file(ticker: str, best_result: dict) -> List[str]:
    """Write CHAMPIONSHIP_FORMULA.md per Mission 12 23-item spec to ALL viable
    destinations (Tech0 best-effort, /Volumes mirror, Drive mirror).

    Returns list of paths written.
    """
    ticker_u = ticker.upper()
    hyp = best_result.get("hypothesis") or {}
    res = best_result.get("result") or best_result  # result may be flat
    meta = _ticker_metadata(ticker_u)
    utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    md = []
    md.append(f"# {ticker_u} CHAMPIONSHIP_FORMULA")
    md.append("")
    md.append(f"_Locked-at: {utc}_  ")
    md.append(f"_Generated by: lab.championship_search (per-ticker mastery, locked methodology 2026-05-29)_  ")
    md.append("")
    md.append("## 23-item Mission 12 spec")
    md.append("")
    md.append(f"1. **Ticker**: {ticker_u}")
    md.append(f"2. **Sector**: {meta['sector']}")
    md.append(f"3. **Market Cap**: {meta['mcap']}")
    md.append(f"4. **ADV (Avg Daily Volume)**: {meta['adv']}")
    md.append(f"5. **Realized Vol decile**: {meta['vol_decile']}")
    md.append(f"6. **Beta**: {meta['beta']}")
    md.append(f"7. **Hypothesis SAP-ID**: {hyp.get('id', 'SAP-?')}  (parent seed: {hyp.get('parent_seed_id', '?')})")
    md.append(f"8. **Regime Gate**: `{hyp.get('regime_gate', 'n/a')}`")
    md.append(f"9. **Bias Filter**: `{hyp.get('bias_filter', 'n/a')}`")
    md.append(f"10. **Trigger**: `{hyp.get('trigger', 'n/a')}`")
    md.append(f"11. **Confirmation**: `{hyp.get('confirmation', 'n/a')}`")
    md.append(f"12. **Timing**: `{hyp.get('timing', 'n/a')}`")
    md.append(f"13. **Exit Rule**: `{hyp.get('exit', 'n/a')}`")
    md.append(f"14. **No-Trade Rule**: `{hyp.get('no_trade', 'n/a')}`")
    md.append(f"15. **Cost Model**: {hyp.get('cost', 'n/a')}")
    md.append(f"16. **Timeframe**: {hyp.get('timeframe', 'n/a')}")
    md.append("17. **Data Sources**:")
    for ds in hyp.get("data_sources", []):
        md.append(f"    - {ds}")
    md.append(f"18. **Universe constraint**: {hyp.get('universe', 'n/a')}")
    md.append(f"19. **Walk-Forward Efficiency**: {_fmt(res.get('wfe'))}")
    md.append(f"20. **PBO**: {_fmt(res.get('pbo'))}")
    md.append(f"21. **DSR**: {_fmt(res.get('dsr_prob'))}")
    md.append(f"22. **Holdout Sharpe**: {_fmt(res.get('holdout_sharpe'))}")
    md.append(f"23. **Locked-at UTC**: {utc}")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Supplemental — full result row")
    md.append("")
    md.append("```json")
    md.append(json.dumps({
        "n_obs": res.get("n_obs"),
        "n_trades": res.get("n_trades"),
        "win_rate": res.get("win_rate"),
        "full_sharpe": res.get("full_sharpe"),
        "is_sharpe_median": res.get("is_sharpe_median"),
        "oos_sharpe_median": res.get("oos_sharpe_median"),
        "wfe": res.get("wfe"),
        "pbo": res.get("pbo"),
        "dsr_prob": res.get("dsr_prob"),
        "holdout_sharpe": res.get("holdout_sharpe"),
        "insample_sharpe_pre_holdout": res.get("insample_sharpe_pre_holdout"),
        "stability": res.get("stability"),
        "perturb_params": hyp.get("perturb_params"),
    }, indent=2, default=str))
    md.append("```")
    md.append("")
    md.append("## Mission 12 hard rules (applied)")
    md.append("")
    for i, rule in enumerate(hard_rules(), 1):
        md.append(f"{i}. {rule}")
    md.append("")
    md.append(f"## Notes")
    md.append("")
    md.append(f"- Search seeded from Mission 12 5-agent grid (PURE_TECH, ORB_MORNING, "
              f"VWAP_MTF, GOV_AWARE, HYBRID_REGIME).")
    md.append(f"- This file represents the BEST variant of {best_result.get('n_variants_tested', '?')} "
              f"per-ticker variants tested.")
    md.append(f"- Cohort-level validation is a robustness diagnostic, not the primary "
              f"report — the unit of mastery is THIS ticker.")
    if any(str(v).startswith("unknown_pending_metadata_backfill") for v in meta.values()):
        md.append(f"- Metadata items 2-6 marked 'unknown_pending_metadata_backfill' will be "
                  f"filled by a separate metadata enrichment pass — run "
                  f"`python -m lab.championship_metadata --tickers {ticker_u} --backfill`.")
    else:
        md.append(f"- Metadata items 2-6 enriched at write time by "
                  f"`lab.championship_metadata.enrich_metadata` (sector from S&P GICS, "
                  f"mcap = shares_outstanding × last_close, ADV from 20-day yfinance "
                  f"volume, vol decile within 502-universe, beta vs SPY).")
    body = "\n".join(md) + "\n"

    paths_written: List[str] = []
    # Three target destinations:
    #   1. Tech0 (canonical per spec; FUSE-blind on this Mac → best-effort)
    #   2. /Volumes/ZG-2TB local mirror (always writable)
    #   3. Drive lab mirror (always writable)
    targets = [
        TECH0_MASTERED / ticker_u / "CHAMPIONSHIP_FORMULA.md",
        LOCAL_CHAMP_MIRROR / ticker_u / "CHAMPIONSHIP_FORMULA.md",
        LAB_CHAMP_MIRROR / ticker_u / "CHAMPIONSHIP_FORMULA.md",
    ]
    for t in targets:
        try:
            t.parent.mkdir(parents=True, exist_ok=True)
            t.write_text(body, encoding="utf-8")
            paths_written.append(str(t))
        except OSError as e:
            # Tech0 will commonly fail under FUSE — that's expected and documented
            print(f"  [championship] couldn't write {t}: {e}", flush=True)
    return paths_written


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────


def search_championship(
    ticker: str,
    n_variants: int = 24,
    timeframe: str = "1d",
    n_folds: int = 12,
    seeds: Optional[List[str]] = None,
) -> dict:
    """Run N hypothesis variants on the ticker, return the best by holdout Sharpe.

    For each variant:
      1. Call hypothesis_runner.run_hypothesis_for_ticker(hyp, ticker, ...).
      2. Score the result; classify locked/survived/died.
      3. Append to per-ticker posterior.

    After all variants: pick best by score, write CHAMPIONSHIP_FORMULA.md.

    Returns:
      {
        "ticker": str, "n_variants_tested": int, "n_variants_failed_gate": int,
        "best_sap_id": str, "best_score": float,
        "best_result": {...full row + hypothesis dict...},
        "championship_files": [paths],
        "posterior_path": str,
        "all_results": [...] (truncated dicts for diagnostic),
        "elapsed_sec": float,
      }
    """
    t0 = time.time()

    # SIV pre-flight gate (Council R3+R4): SOFT warning only — don't break dispatch.
    # `lab.knowledge.siv.is_promotion_allowed()` returns (False, reason) until the
    # SIV checklist is fully PRESENT (kill-criterion + DSMB + blinding + holdout
    # separation all complete). Champions written below MUST NOT be promoted to
    # paper trading while this gate is closed.
    try:
        from .knowledge import siv as _siv  # type: ignore
        _allowed, _siv_reason = _siv.is_promotion_allowed()
        if not _allowed:
            print(f"  [SIV-GATE] promotion-to-paper BLOCKED: {_siv_reason}", flush=True)
        else:
            print(f"  [SIV-GATE] {_siv_reason}", flush=True)
    except Exception as _siv_e:  # pragma: no cover — defensive: gate must never crash search
        print(f"  [SIV-GATE] check failed ({type(_siv_e).__name__}: {_siv_e}); "
              f"treating as CLOSED (no promotion).", flush=True)

    _ihr.set_timeframe(timeframe)
    bars_per_year = _ihr._state["bars_per_year"]

    # Load priors from any existing posterior — so re-running search converges
    post = _load_posterior(ticker)
    priors = post.get("next_priors") or None

    variants = list(variant_generator(ticker, n_variants, priors=priors, seeds=seeds))
    print(
        f"\n=== championship_search({ticker}) — {len(variants)} variants "
        f"(timeframe={timeframe}, priors={'yes' if priors else 'none'}) ===",
        flush=True,
    )

    all_results: List[dict] = []
    failed_gate = n_variants - len(variants)  # generator already skipped these

    for idx, hyp in enumerate(variants, 1):
        sap_id = hyp.get("id", f"SAP-{ticker}-{idx:03d}")
        try:
            row = _hr.run_hypothesis_for_ticker(hyp, ticker, bars_per_year, n_folds=n_folds)
        except Exception as e:
            traceback.print_exc()
            row = {"ticker": ticker, "status": f"error:{type(e).__name__}", "error_msg": str(e)}
        row["hypothesis_id"] = sap_id
        row["parent_seed_id"] = hyp.get("parent_seed_id")
        row["perturb_params"] = hyp.get("perturb_params")
        # Stash hypothesis dict alongside row so championship writer can use it
        row["hypothesis"] = hyp
        update_posterior(ticker, row)
        all_results.append(row)
        score = _score_for_posterior(row)
        status_lbl = _classify_status(row)
        if row.get("status") == "ok":
            print(
                f"  [{idx:>2}/{len(variants)}] {sap_id} ({hyp.get('parent_seed_id', '?')}): "
                f"WR={_fmt(row.get('win_rate'))} N={row.get('n_trades', 0):>4d} "
                f"HoldSR={_fmt(row.get('holdout_sharpe'))} "
                f"PBO={_fmt(row.get('pbo'))} DSR={_fmt(row.get('dsr_prob'))} "
                f"→ {status_lbl} (score={_fmt(score if score != float('-inf') else None)})",
                flush=True,
            )
        else:
            print(f"  [{idx:>2}/{len(variants)}] {sap_id}: {row.get('status')}", flush=True)

    # Pick best
    scored = [(r, _score_for_posterior(r)) for r in all_results]
    scored = [(r, s) for r, s in scored if s != float("-inf")]
    if not scored:
        print(f"  No viable variant for {ticker} — all died.", flush=True)
        return {
            "ticker": ticker.upper(),
            "n_variants_tested": len(all_results),
            "n_variants_failed_gate": failed_gate,
            "best_sap_id": None,
            "best_score": None,
            "best_result": None,
            "championship_files": [],
            "posterior_path": str(_posterior_path(ticker)),
            "all_results": [_summarize(r) for r in all_results],
            "elapsed_sec": time.time() - t0,
        }
    best_row, best_score = max(scored, key=lambda t: t[1])
    best_payload = {
        "hypothesis": best_row.get("hypothesis"),
        "result": {k: v for k, v in best_row.items() if k != "hypothesis"},
        "n_variants_tested": len(all_results),
    }
    paths = write_championship_file(ticker, best_payload)
    return {
        "ticker": ticker.upper(),
        "n_variants_tested": len(all_results),
        "n_variants_failed_gate": failed_gate,
        "best_sap_id": best_row.get("hypothesis_id"),
        "best_score": best_score,
        "best_result": {k: best_row.get(k) for k in (
            "hypothesis_id", "parent_seed_id", "perturb_params",
            "win_rate", "n_trades", "full_sharpe", "holdout_sharpe",
            "pbo", "dsr_prob", "wfe",
        )},
        "championship_files": paths,
        "posterior_path": str(_posterior_path(ticker)),
        "all_results": [_summarize(r) for r in all_results],
        "elapsed_sec": time.time() - t0,
    }


def _summarize(r: dict) -> dict:
    """Trim a full result row to a diagnostic-only summary (drop hypothesis dict, etc)."""
    return {
        "sap_id": r.get("hypothesis_id"),
        "parent_seed_id": r.get("parent_seed_id"),
        "status": r.get("status"),
        "win_rate": r.get("win_rate"),
        "n_trades": r.get("n_trades"),
        "holdout_sharpe": r.get("holdout_sharpe"),
        "pbo": r.get("pbo"),
        "dsr_prob": r.get("dsr_prob"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=["A", "AAPL", "NVDA"],
                    help="Tickers to search. Default: A (Mission 12 bootstrap), AAPL, NVDA.")
    ap.add_argument("--n-variants", type=int, default=10)
    ap.add_argument("--timeframe", default="1d", choices=["1d", "5min"])
    ap.add_argument("--n-folds", type=int, default=12)
    ap.add_argument("--seeds", nargs="+", default=None,
                    help="Optional seed_id allowlist (e.g. CATALYST_CONFLUENCE "
                         "CROSS_SYMBOL_REGIME GOV_AWARE_v2). Default: all seeds.")
    args = ap.parse_args()

    summary = {}
    for tk in args.tickers:
        res = search_championship(tk, n_variants=args.n_variants,
                                  timeframe=args.timeframe, n_folds=args.n_folds,
                                  seeds=args.seeds)
        summary[tk] = {
            "best_sap_id": res["best_sap_id"],
            "best_score": res["best_score"],
            "n_variants_tested": res["n_variants_tested"],
            "championship_files": res["championship_files"],
            "elapsed_sec": res["elapsed_sec"],
            "best_result": res.get("best_result"),
        }

    print("\n\n=== CHAMPIONSHIP SEARCH SUMMARY ===")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
