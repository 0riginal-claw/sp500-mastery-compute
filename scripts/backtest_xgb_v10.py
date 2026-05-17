"""
backtest_xgb_v10.py — Alpha158-Plus + OpenMythos + 3 missing feature modules.

Feature stack (v9 + three previously-missing modules):
  - v9 base ~1126+ features (v8 + optional 256-dim Mythos layer)
  - [NEW] daily_integration_features — 7 features:
        beta_adj_residual_ret_z21/z63, csrs_5d/10d/20d,
        earn_contam_gate, earn_post_rv_gate
  - [NEW] alpaca_features — up to 13 features:
        earnings proximity (4), dividend/ex-div (4), splits (2), metadata (3)
  - [NEW] featuretools_dfs_features — up to 60 depth-2 interaction features
        (DFS primitives: mean/std/sum/skew/max/min + depth-2 cross-products)

Total: ~870 (v8 base) + 256 (Mythos, optional) + 7 (daily_int) + 13 (alpaca)
       + ~60 (dfs) = ~1,010–1,266+ features depending on Mythos checkpoint.

Key differences from v9:
  1. build_v10_features() calls build_v9_features() then adds the three
     missing modules in sequence.
  2. Three new import blocks (each gracefully degraded on import failure).
  3. --job-id CLI arg restored (was in v8, dropped in v9; GH Actions require it).
  4. run_meta.json extended with v10 section, feature counts per module, job_id.
  5. pipeline_version / strategy_variant updated to v10.

Usage:
    # Standard run (no Mythos):
    python backtest_xgb_v10.py --ticker AAPL --strategy ORB \\
        --job-id smoke-001 --out-dir /tmp/aapl_v10

    # With Mythos embeddings:
    MYTHOS_CHECKPOINT_PATH=/path/to/mythos_financial_v0.pt \\
    python backtest_xgb_v10.py --ticker AAPL --strategy ORB \\
        --job-id run-001 --out-dir /tmp/aapl_v10_mythos --use-mythos-features

DO NOT modify v9. This is a clean fork.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

WORK = Path(
    "/Users/orginal/Library/CloudStorage/"
    "GoogleDrive-zachgladstone@gmail.com/My Drive/"
    "AI-Tools/s&p500-ticker-mastery"
)
LABEL_EMBARGO_DAYS = 21

V10_FEATURE_VERSION = "v10.6.8"  # 2026-05-17 — Wave EFD1: senate_efd_options_disclosure_count_30d_replay wired (+1 Senate STOCK Act options-disclosure 30d rolling count, QuiverQuant free API); prior: v10.6.7 Wave FEC1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import v9 feature builder (non-destructive — v9.py is untouched)
# ---------------------------------------------------------------------------

from backtest_xgb_v9 import build_v9_features, MYTHOS_FEAT_NAMES, MYTHOS_FEATURE_DIM  # noqa: E402
from backtest_xgb_v7 import numeric_cols  # noqa: E402
import backtest_ml as bml  # noqa: E402
import xgboost as xgb  # noqa: E402
from feature_cache import get_cached  # noqa: E402

# Cross-sectional cache (unchanged from v9)
try:
    import cross_sectional_features as csf
except Exception as _e:
    logger.warning("csf unavailable: %s", _e)
    csf = None

# ---------------------------------------------------------------------------
# Helper A: daily_integration_features
# ---------------------------------------------------------------------------

try:
    from daily_integration_features import add_daily_integration_features  # noqa: E402
    DAILY_INT_AVAILABLE = True
    logger.info("[v10] daily_integration_features loaded OK")
except Exception as _daily_err:
    logger.warning(
        "[v10] daily_integration_features not importable: %s — 7 features zeroed", _daily_err
    )
    DAILY_INT_AVAILABLE = False

    def add_daily_integration_features(  # type: ignore[misc]
        df: pd.DataFrame,
        ticker: Optional[str] = None,
    ) -> pd.DataFrame:
        """Stub: fill all 7 daily_integration cols with 0."""
        for col in [
            "beta_adj_residual_ret_z21",
            "beta_adj_residual_ret_z63",
            "csrs_5d",
            "csrs_10d",
            "csrs_20d",
            "earn_contam_gate",
            "earn_post_rv_gate",
        ]:
            df[col] = 0.0
        return df


DAILY_INT_FEATURE_NAMES: list[str] = [
    "beta_adj_residual_ret_z21",
    "beta_adj_residual_ret_z63",
    "csrs_5d",
    "csrs_10d",
    "csrs_20d",
    "earn_contam_gate",
    "earn_post_rv_gate",
]

# ---------------------------------------------------------------------------
# Helper B: alpaca_features
# ---------------------------------------------------------------------------

try:
    from alpaca_features import add_alpaca_features  # noqa: E402
    ALPACA_AVAILABLE = True
    logger.info("[v10] alpaca_features loaded OK")
except Exception as _alp_err:
    logger.warning(
        "[v10] alpaca_features not importable: %s — 13 features zeroed", _alp_err
    )
    ALPACA_AVAILABLE = False

    def add_alpaca_features(  # type: ignore[misc]
        daily_df: pd.DataFrame,
        ticker: str,
    ) -> pd.DataFrame:
        """Stub: fill all 13 alpaca cols with 0."""
        for col in [
            "days_until_earnings",
            "is_earnings_week",
            "earnings_surprise_last",
            "days_since_last_earnings",
            "ex_div_proximity",
            "days_since_last_exdiv",
            "div_yield_trailing",
            "dividend_growth_yoy",
            "days_since_last_split",
            "is_post_split_60d",
            "log_market_cap",
            "short_interest_pct",
            "sector_encoded",
        ]:
            daily_df[col] = 0.0
        return daily_df


ALPACA_FEATURE_NAMES: list[str] = [
    "days_until_earnings",
    "is_earnings_week",
    "earnings_surprise_last",
    "days_since_last_earnings",
    "ex_div_proximity",
    "days_since_last_exdiv",
    "div_yield_trailing",
    "dividend_growth_yoy",
    "days_since_last_split",
    "is_post_split_60d",
    "log_market_cap",
    "short_interest_pct",
    "sector_encoded",
]

# ---------------------------------------------------------------------------
# Helper C: featuretools_dfs_features
# ---------------------------------------------------------------------------

try:
    from featuretools_dfs_features import add_dfs_features, dfs_feature_names  # noqa: E402
    DFS_AVAILABLE = True
    logger.info("[v10] featuretools_dfs_features loaded OK")
except Exception as _dfs_err:
    logger.warning(
        "[v10] featuretools_dfs_features not importable: %s — DFS features skipped", _dfs_err
    )
    DFS_AVAILABLE = False

    def add_dfs_features(  # type: ignore[misc]
        df: pd.DataFrame,
        ticker: Optional[str] = None,
    ) -> pd.DataFrame:
        """Stub: return df unchanged when featuretools module is unavailable."""
        return df

    def dfs_feature_names() -> list[str]:  # type: ignore[misc]
        return []


# ---------------------------------------------------------------------------
# Helper D: insider_form4_features (SEC Form 4 insider-disclosure trades)
# Wired 2026-05-17 as the "gov-trades" module — fills feature-module gap.
# ---------------------------------------------------------------------------

try:
    from insider_form4_features import add_insider_form4_features  # noqa: E402
    INSIDER_FORM4_AVAILABLE = True
    logger.info("[v10] insider_form4_features loaded OK")
except Exception as _ins_err:
    logger.warning(
        "[v10] insider_form4_features not importable: %s — 8 features zeroed", _ins_err
    )
    INSIDER_FORM4_AVAILABLE = False

    def add_insider_form4_features(  # type: ignore[misc]
        daily_df: pd.DataFrame,
        ticker: str,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Stub: fill all 8 insider_form4 cols with 0."""
        for col in [
            "insider_buy_count_30d",
            "insider_sell_count_30d",
            "insider_net_buy_count_60d",
            "insider_cluster_buy_flag",
            "insider_cluster_sell_flag",
            "days_since_last_insider_buy",
            "days_since_last_insider_sell",
            "insider_buy_dollar_amount_60d_log",
        ]:
            daily_df[col] = 0.0
        return daily_df


INSIDER_FORM4_FEATURE_NAMES: list[str] = [
    "insider_buy_count_30d",
    "insider_sell_count_30d",
    "insider_net_buy_count_60d",
    "insider_cluster_buy_flag",
    "insider_cluster_sell_flag",
    "days_since_last_insider_buy",
    "days_since_last_insider_sell",
    "insider_buy_dollar_amount_60d_log",
]


# ---------------------------------------------------------------------------
# Helper E: mastery_priors_features (past-test mastery files as priors)
# Wired 2026-05-17 — reads 311 v4 + 7 v10 mastery markdown artifacts and
# emits 7 per-ticker priors features (mastered flags, PF, DD, top-10 flag,
# .shift(1)-safe mtime-gated age). See $SP/scripts/mastery_priors_features.py.
# ---------------------------------------------------------------------------

try:
    from mastery_priors_features import (  # noqa: E402
        add_mastery_priors,
        MASTERY_PRIORS_FEATURE_NAMES,
    )
    MASTERY_PRIORS_AVAILABLE = True
    logger.info("[v10] mastery_priors_features loaded OK")
except Exception as _mp_err:
    logger.warning(
        "[v10] mastery_priors_features not importable: %s — 7 features zeroed", _mp_err
    )
    MASTERY_PRIORS_AVAILABLE = False

    MASTERY_PRIORS_FEATURE_NAMES: list[str] = [  # type: ignore[no-redef]
        "prior_v4_mastered",
        "prior_v4_pf",
        "prior_v10_mastered",
        "prior_v10_pf",
        "prior_v10_dd",
        "prior_cross_section_top10",
        "prior_mastery_age_days",
    ]

    def add_mastery_priors(  # type: ignore[misc]
        df: pd.DataFrame,
        ticker: str,
    ) -> pd.DataFrame:
        """Stub: fill all 7 mastery_priors cols with 0 when module unavailable."""
        for col in MASTERY_PRIORS_FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0
        return df


# ---------------------------------------------------------------------------
# Helper F: paper_trade_outcome_features (live paper-trade feedback features)
# Wired 2026-05-17 — reads $SP/paper_trade/state/*_state.json closed_trades[]
# and emits 7 .shift(1)-safe rolling outcome features (win-rate, PF, count,
# last-outcome-sign, avg holding, signal-to-fill lag, current drawdown).
# Zero-fills when paper_trade/ tree empty or ticker has no closed trades yet.
# See $SP/scripts/paper_trade_outcome_features.py.
# ---------------------------------------------------------------------------

try:
    from paper_trade_outcome_features import (  # noqa: E402
        add_paper_trade_outcome_features,
        PT_FEATURE_NAMES,
    )
    PT_OUTCOMES_AVAILABLE = True
    logger.info("[v10] paper_trade_outcome_features loaded OK")
except Exception as _pt_err:
    logger.warning(
        "[v10] paper_trade_outcome_features not importable: %s — 7 features zeroed",
        _pt_err,
    )
    PT_OUTCOMES_AVAILABLE = False

    PT_FEATURE_NAMES: list[str] = [  # type: ignore[no-redef]
        "paper_trade_win_rate_30d",
        "paper_trade_pf_30d",
        "paper_trade_count_30d",
        "paper_trade_last_outcome_sign",
        "paper_trade_avg_holding_days",
        "paper_trade_signal_to_fill_lag_min",
        "paper_trade_in_drawdown_pct",
    ]

    def add_paper_trade_outcome_features(  # type: ignore[misc]
        df: pd.DataFrame,
        ticker: str,
    ) -> pd.DataFrame:
        """Stub: fill all 7 paper-trade outcome cols with 0."""
        for col in PT_FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0 if col in (
                    "paper_trade_count_30d", "paper_trade_last_outcome_sign"
                ) else 0.0
        return df


# ---------------------------------------------------------------------------
# Helper G/H/I: GitHub treasure-hunt features (wave 11, 2026-05-17)
#   G: stumpy   — matrix-profile motif/discord (~6 features × windows)
#   H: ffn      — Sortino/Calmar/Ulcer/Downside (~15 features × windows)
#   I: pandas-ta-classic — non-TA-Lib indicators (Aberration/KVO/KST/PVO/Vortex/Squeeze/RVI/NVI/PVI/TSI/etc., ~30 feats)
# All graceful-fail. .shift(1)-safe (rolling windows on prior bars).
# ---------------------------------------------------------------------------
try:
    from stumpy_features import add_stumpy_features  # noqa: E402
    STUMPY_AVAILABLE = True
    logger.info("[v10] stumpy_features loaded OK")
except Exception as _st_err:
    logger.warning("[v10] stumpy_features not importable: %s — features zeroed", _st_err)
    STUMPY_AVAILABLE = False
    def add_stumpy_features(df, ticker, windows=(10, 20, 60)):  # type: ignore[misc]
        return df

try:
    from ffn_features import add_ffn_features  # noqa: E402
    FFN_AVAILABLE = True
    logger.info("[v10] ffn_features loaded OK")
except Exception as _ffn_err:
    logger.warning("[v10] ffn_features not importable: %s — features zeroed", _ffn_err)
    FFN_AVAILABLE = False
    def add_ffn_features(df, ticker, windows=(20, 60, 120)):  # type: ignore[misc]
        return df

try:
    from pandas_ta_classic_features import add_pandas_ta_classic_features  # noqa: E402
    PTC_AVAILABLE = True
    logger.info("[v10] pandas_ta_classic_features loaded OK")
except Exception as _ptc_err:
    logger.warning("[v10] pandas_ta_classic_features not importable: %s — features zeroed", _ptc_err)
    PTC_AVAILABLE = False
    def add_pandas_ta_classic_features(df, ticker):  # type: ignore[misc]
        return df


# ---------------------------------------------------------------------------
# Helper J / K / L / M: Wave A — 12 new features (2026-05-17)
#   J: options_flow      — put_call_ratio + iv_vs_rv + unusual flag (3)
#   K: govtrades         — congress density/buy-sell ratio + lobbying count (3)
#   L: time_of_day       — bucket 0-4 (1)
#   M: gabriel_priors    — champion PF/WR/N + regime/monthly priors (5)
# All graceful-fail. All .shift(1)-safe.
# ---------------------------------------------------------------------------
try:
    from options_flow_features import (  # noqa: E402
        add_options_flow_features,
        OPTIONS_FLOW_FEATURE_NAMES,
    )
    OPTIONS_FLOW_AVAILABLE = True
    logger.info("[v10] options_flow_features loaded OK")
except Exception as _of_err:
    logger.warning("[v10] options_flow_features not importable: %s — 3 features zeroed", _of_err)
    OPTIONS_FLOW_AVAILABLE = False
    OPTIONS_FLOW_FEATURE_NAMES = [
        "put_call_volume_ratio",
        "iv_vs_rv_divergence",
        "unusual_options_activity_flag",
    ]
    def add_options_flow_features(df, ticker):  # type: ignore[misc]
        for c in OPTIONS_FLOW_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0 if c != "unusual_options_activity_flag" else 0
        return df

try:
    from govtrades_features import (  # noqa: E402
        add_govtrades_features,
        GOVTRADES_FEATURE_NAMES,
    )
    GOVTRADES_AVAILABLE = True
    logger.info("[v10] govtrades_features loaded OK")
except Exception as _gt_err:
    logger.warning("[v10] govtrades_features not importable: %s — 3 features zeroed", _gt_err)
    GOVTRADES_AVAILABLE = False
    GOVTRADES_FEATURE_NAMES = [
        "congress_trade_density_5d",
        "congress_buy_sell_ratio_5d",
        "lobbying_filing_count_30d",
    ]
    def add_govtrades_features(df, ticker):  # type: ignore[misc]
        for c in GOVTRADES_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0 if c != "congress_buy_sell_ratio_5d" else 0.0
        return df

try:
    from time_of_day_features import (  # noqa: E402
        add_time_of_day_features,
        TOD_FEATURE_NAMES,
    )
    TOD_AVAILABLE = True
    logger.info("[v10] time_of_day_features loaded OK")
except Exception as _tod_err:
    logger.warning("[v10] time_of_day_features not importable: %s — 1 feature zeroed", _tod_err)
    TOD_AVAILABLE = False
    TOD_FEATURE_NAMES = ["time_of_day_bucket"]
    def add_time_of_day_features(df, ticker):  # type: ignore[misc]
        if "time_of_day_bucket" not in df.columns:
            df["time_of_day_bucket"] = 2  # default mid-day fallback
        return df

try:
    from gabriel_priors_features import (  # noqa: E402
        add_gabriel_priors_features,
        GABRIEL_PRIORS_FEATURE_NAMES,
    )
    GABRIEL_PRIORS_AVAILABLE = True
    logger.info("[v10] gabriel_priors_features loaded OK")
except Exception as _gp_err:
    logger.warning("[v10] gabriel_priors_features not importable: %s — 5 features zeroed", _gp_err)
    GABRIEL_PRIORS_AVAILABLE = False
    GABRIEL_PRIORS_FEATURE_NAMES = [
        "gabriel_champion_pf",
        "gabriel_champion_wr",
        "gabriel_champion_n_trades",
        "gabriel_regime_breakdown_score",
        "gabriel_monthly_perf_consistency",
    ]
    def add_gabriel_priors_features(df, ticker):  # type: ignore[misc]
        for c in GABRIEL_PRIORS_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0 if c == "gabriel_champion_n_trades" else 0.0
        return df

try:
    from vix_term_structure_v2_features import (  # noqa: E402
        compute_vix_term_structure_v2_features,
        VIX_TS_FEATURE_NAMES,
    )
    VIX_TS_AVAILABLE = True
    logger.info("[v10] vix_term_structure_v2_features loaded OK")
except Exception as _vts_err:
    logger.warning("[v10] vix_term_structure_v2_features not importable: %s — 3 features zeroed", _vts_err)
    VIX_TS_AVAILABLE = False
    VIX_TS_FEATURE_NAMES = ["vix9d_vix_ratio", "vix_term_inverted", "vix9d_vix_ratio_z10"]
    def compute_vix_term_structure_v2_features(df):  # type: ignore[misc]
        for c in VIX_TS_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0 if c != "vix_term_inverted" else 0
        return df


# ---------------------------------------------------------------------------
# Helper N: garch_11_cond_vol (GARCH(1,1) conditional volatility, 3 features)
# Wired 2026-05-17. Uses arch package (BSD-3, Kevin Sheppard) + yfinance close.
# .shift(1)-safe: all outputs use prior-bar conditional variance only.
# ---------------------------------------------------------------------------
try:
    from garch_11_cond_vol_features import (  # noqa: E402
        compute_garch_11_cond_vol_features,
        GARCH11_FEATURE_NAMES,
    )
    GARCH11_AVAILABLE = True
    logger.info("[v10] garch_11_cond_vol_features loaded OK")
except Exception as _garch_err:
    logger.warning(
        "[v10] garch_11_cond_vol_features not importable: %s — 3 features zeroed", _garch_err
    )
    GARCH11_AVAILABLE = False
    GARCH11_FEATURE_NAMES = [
        "garch11_cond_vol_1d",
        "garch11_cond_vol_z21",
        "garch11_persistence",
    ]

    def compute_garch_11_cond_vol_features(df, ticker=None):  # type: ignore[misc]
        for c in GARCH11_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0
        return df


# ---------------------------------------------------------------------------
# Helper O: egarch_11_leverage (EGARCH(1,1) leverage-effect vol, 3 features)
# Wired 2026-05-17. Uses arch package (BSD-3) + yfinance close.
# .shift(1)-safe: all outputs use prior-bar conditional variance only.
# ---------------------------------------------------------------------------
try:
    from egarch_11_leverage_features import (  # noqa: E402
        compute_egarch_11_leverage_features,
        EGARCH11_LEV_FEATURE_NAMES,
    )
    EGARCH11_LEV_AVAILABLE = True
    logger.info("[v10] egarch_11_leverage_features loaded OK")
except Exception as _egarch_err:
    logger.warning(
        "[v10] egarch_11_leverage_features not importable: %s — 3 features zeroed", _egarch_err
    )
    EGARCH11_LEV_AVAILABLE = False
    EGARCH11_LEV_FEATURE_NAMES = [
        "egarch11_lev_cond_vol_1d",
        "egarch11_lev_effect",
        "egarch11_lev_vol_z21",
    ]

    def compute_egarch_11_leverage_features(df, ticker=None):  # type: ignore[misc]
        for c in EGARCH11_LEV_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0
        return df


# ---------------------------------------------------------------------------
# Helper P: vpin_50bucket (VPIN 50-bucket BVC approximation, 3 features)
# Wired 2026-05-17. Uses daily OHLCV (close + volume) from v9 stack;
# approximates alpaca_1min_bars via BVC (López de Prado/O'Hara 2012 RFS).
# .shift(1)-safe: all outputs shift by 1 bar before assignment.
# ---------------------------------------------------------------------------
try:
    from vpin_50bucket_features import (  # noqa: E402
        compute_vpin_50bucket_features,
        VPIN_50BUCKET_FEATURE_NAMES,
    )
    VPIN_50BUCKET_AVAILABLE = True
    logger.info("[v10] vpin_50bucket_features loaded OK")
except Exception as _vpin_err:
    logger.warning(
        "[v10] vpin_50bucket_features not importable: %s — 3 features zeroed", _vpin_err
    )
    VPIN_50BUCKET_AVAILABLE = False
    VPIN_50BUCKET_FEATURE_NAMES = [
        "vpin_50bucket",
        "vpin_50bucket_z21",
        "vpin_buy_frac_10",
    ]

    def compute_vpin_50bucket_features(df, ticker=None):  # type: ignore[misc]
        for c in VPIN_50BUCKET_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0
        return df


# ---------------------------------------------------------------------------
# Helper Q: kyles_lambda_intraday (Kyle 1985 λ via BVC approximation, 3 features)
# Wired 2026-05-17. Approximates alpaca_1min_bars via daily BVC signed-volume;
# estimates price-impact coefficient (λ) using rolling OLS over 20-day window.
# .shift(1)-safe: both delta_price and signed_vol are lagged before regression.
# ---------------------------------------------------------------------------
try:
    from kyles_lambda_intraday_features import (  # noqa: E402
        compute_kyles_lambda_intraday_features,
        KYLES_LAMBDA_FEATURE_NAMES,
    )
    KYLES_LAMBDA_AVAILABLE = True
    logger.info("[v10] kyles_lambda_intraday_features loaded OK")
except Exception as _kl_err:
    logger.warning(
        "[v10] kyles_lambda_intraday_features not importable: %s — 3 features zeroed", _kl_err
    )
    KYLES_LAMBDA_AVAILABLE = False
    KYLES_LAMBDA_FEATURE_NAMES = [
        "kyles_lambda",
        "kyles_lambda_z21",
        "kyles_lambda_trend",
    ]

    def compute_kyles_lambda_intraday_features(df, ticker=None):  # type: ignore[misc]
        for c in KYLES_LAMBDA_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0
        return df


# ---------------------------------------------------------------------------
# Helper W: vpin_features (TRUE 1-min VPIN, 5 features) — Wave M-1 #1
# Wired 2026-05-17. Uses Alpaca 1-min cache via _load_1min(); falls back to
# claudes-test 1Min_merged. .shift(1)-safe inside the module.
# ---------------------------------------------------------------------------
try:
    from vpin_features import (  # noqa: E402
        add_vpin_features,
        VPIN_FEATURE_NAMES,
    )
    VPIN_INTRADAY_AVAILABLE = True
    logger.info("[v10] vpin_features (intraday) loaded OK")
except Exception as _vpin_intra_err:
    logger.warning(
        "[v10] vpin_features (intraday) not importable: %s — 5 features zeroed",
        _vpin_intra_err,
    )
    VPIN_INTRADAY_AVAILABLE = False
    VPIN_FEATURE_NAMES = [
        "vpin_eod",
        "vpin_max_today",
        "vpin_zscore_60d",
        "vpin_above_p95",
        "vpin_buy_frac_eod",
    ]

    def add_vpin_features(df, ticker=None):  # type: ignore[misc]
        for c in VPIN_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0
        return df


# ---------------------------------------------------------------------------
# Helper X: tick_imbalance_features (Lee-Ready, 5 features) — Wave M-1 #6
# Wired 2026-05-17. Uses Alpaca 1-min cache via _load_1min(); falls back to
# claudes-test 1Min_merged. .shift(1)-safe inside the module.
# ---------------------------------------------------------------------------
try:
    from tick_imbalance_features import (  # noqa: E402
        add_tick_imbalance_features,
        TICK_IMBALANCE_FEATURE_NAMES,
    )
    TICK_IMBALANCE_AVAILABLE = True
    logger.info("[v10] tick_imbalance_features loaded OK")
except Exception as _ti_err:
    logger.warning(
        "[v10] tick_imbalance_features not importable: %s — 5 features zeroed",
        _ti_err,
    )
    TICK_IMBALANCE_AVAILABLE = False
    TICK_IMBALANCE_FEATURE_NAMES = [
        "tick_imb_eod",
        "tick_imb_first_hour",
        "tick_imb_last_hour",
        "tick_imb_5d_avg",
        "tick_imb_first_vs_last_hour_diff",
    ]

    def add_tick_imbalance_features(df, ticker=None):  # type: ignore[misc]
        for c in TICK_IMBALANCE_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0
        return df


# ---------------------------------------------------------------------------
# Helper Y: volume_profile_features (POC / VA / shape, 6 features) — Wave M-1 #11
# Wired 2026-05-17. Uses Alpaca 1-min cache via _load_1min(); falls back to
# claudes-test 1Min_merged. .shift(1)-safe inside the module.
# ---------------------------------------------------------------------------
try:
    from volume_profile_features import (  # noqa: E402
        add_volume_profile_features,
        VOLUME_PROFILE_FEATURE_NAMES,
    )
    VOLUME_PROFILE_AVAILABLE = True
    logger.info("[v10] volume_profile_features loaded OK")
except Exception as _vp_err:
    logger.warning(
        "[v10] volume_profile_features not importable: %s — 6 features zeroed",
        _vp_err,
    )
    VOLUME_PROFILE_AVAILABLE = False
    VOLUME_PROFILE_FEATURE_NAMES = [
        "vp_poc_price",
        "vp_close_minus_poc_atr",
        "vp_va_high",
        "vp_va_low",
        "vp_close_inside_va_indicator",
        "vp_profile_shape",
    ]

    def add_volume_profile_features(df, ticker=None):  # type: ignore[misc]
        for c in VOLUME_PROFILE_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0
        return df


# ---------------------------------------------------------------------------
# Helper Z: auction_features (open/close auction, 6 features) — Wave M-1 #12
# Wired 2026-05-17. Uses Alpaca 1-min cache via _load_1min(); falls back to
# claudes-test 1Min_merged. .shift(1)-safe inside the module.
# ---------------------------------------------------------------------------
try:
    from auction_features import (  # noqa: E402
        add_auction_features,
        AUCTION_FEATURE_NAMES,
    )
    AUCTION_AVAILABLE = True
    logger.info("[v10] auction_features loaded OK")
except Exception as _auc_err:
    logger.warning(
        "[v10] auction_features not importable: %s — 6 features zeroed",
        _auc_err,
    )
    AUCTION_AVAILABLE = False
    AUCTION_FEATURE_NAMES = [
        "open_auction_ret",
        "close_auction_ret",
        "open_auction_vol_share",
        "close_auction_vol_share",
        "auction_imbalance_ratio",
        "close_auction_dir_vs_session",
    ]

    def add_auction_features(df, ticker=None):  # type: ignore[misc]
        for c in AUCTION_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0
        return df


# ---------------------------------------------------------------------------
# Wave V-1 vol/regime low-cost 7-pack (no new deps). Wired 2026-05-17.
# All modules pure pandas/numpy/scipy and .shift(1)-safe internally.
# ---------------------------------------------------------------------------

# Helper VA: vol_of_vol_features (3 features) — candidate #7
try:
    from vol_of_vol_features import (  # noqa: E402
        add_vol_of_vol_features,
        VOL_OF_VOL_FEATURE_NAMES,
    )
    VOL_OF_VOL_AVAILABLE = True
    logger.info("[v10] vol_of_vol_features loaded OK")
except Exception as _vov_err:
    logger.warning("[v10] vol_of_vol_features not importable: %s — 3 features zeroed", _vov_err)
    VOL_OF_VOL_AVAILABLE = False
    VOL_OF_VOL_FEATURE_NAMES = ["vov_20_20", "vov_60_60", "vov_zscore_252"]

    def add_vol_of_vol_features(df, ticker=None):  # type: ignore[misc]
        for c in VOL_OF_VOL_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0
        return df


# Helper VB: vol_risk_premium_features (4 features) — candidate #12
try:
    from vol_risk_premium_features import (  # noqa: E402
        add_vol_risk_premium_features,
        VRP_FEATURE_NAMES,
    )
    VRP_AVAILABLE = True
    logger.info("[v10] vol_risk_premium_features loaded OK")
except Exception as _vrp_err:
    logger.warning("[v10] vol_risk_premium_features not importable: %s — 4 features zeroed", _vrp_err)
    VRP_AVAILABLE = False
    VRP_FEATURE_NAMES = ["vrp_market", "vrp_ticker", "vrp_zscore_252", "vrp_sign_flip"]

    def add_vol_risk_premium_features(df, ticker=None):  # type: ignore[misc]
        for c in VRP_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0 if c != "vrp_sign_flip" else 0
        return df


# Helper VC: vol_target_sizing_features (2 features) — candidate #13
# Depends on garch11_cond_vol_1d (already wired at Step 15); has RV fallback.
try:
    from vol_target_sizing_features import (  # noqa: E402
        add_vol_target_sizing_features,
        VOL_TARGET_FEATURE_NAMES,
    )
    VOL_TARGET_AVAILABLE = True
    logger.info("[v10] vol_target_sizing_features loaded OK")
except Exception as _vt_err:
    logger.warning("[v10] vol_target_sizing_features not importable: %s — 2 features neutral", _vt_err)
    VOL_TARGET_AVAILABLE = False
    VOL_TARGET_FEATURE_NAMES = ["vol_target_ratio", "vol_target_clipped_5x"]

    def add_vol_target_sizing_features(df, ticker=None):  # type: ignore[misc]
        for c in VOL_TARGET_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 1.0
        return df


# Helper VD: vol_breakout_nr_features (5 features) — candidate #14
try:
    from vol_breakout_nr_features import (  # noqa: E402
        add_vol_breakout_nr_features,
        VOL_BREAKOUT_FEATURE_NAMES,
    )
    VOL_BREAKOUT_AVAILABLE = True
    logger.info("[v10] vol_breakout_nr_features loaded OK")
except Exception as _nr_err:
    logger.warning("[v10] vol_breakout_nr_features not importable: %s — 5 features zeroed", _nr_err)
    VOL_BREAKOUT_AVAILABLE = False
    VOL_BREAKOUT_FEATURE_NAMES = [
        "nr4_indicator", "nr7_indicator", "wr7_indicator",
        "days_since_nr7", "range_pct_of_atr20",
    ]

    def add_vol_breakout_nr_features(df, ticker=None):  # type: ignore[misc]
        for c in VOL_BREAKOUT_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0 if c != "range_pct_of_atr20" else 1.0
        return df


# Helper VE: bollinger_keltner_squeeze_features (4 features) — candidate #15
try:
    from bollinger_keltner_squeeze_features import (  # noqa: E402
        add_bollinger_keltner_squeeze_features,
        SQUEEZE_FEATURE_NAMES,
    )
    SQUEEZE_AVAILABLE = True
    logger.info("[v10] bollinger_keltner_squeeze_features loaded OK")
except Exception as _sq_err:
    logger.warning("[v10] bollinger_keltner_squeeze_features not importable: %s — 4 features zeroed", _sq_err)
    SQUEEZE_AVAILABLE = False
    SQUEEZE_FEATURE_NAMES = [
        "squeeze_on_indicator", "days_in_squeeze",
        "squeeze_release_indicator", "squeeze_momentum_proxy",
    ]

    def add_bollinger_keltner_squeeze_features(df, ticker=None):  # type: ignore[misc]
        for c in SQUEEZE_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0
        return df


# Helper VF: vol_of_vix_features (3 features) — candidate #18
try:
    from vol_of_vix_features import (  # noqa: E402
        add_vol_of_vix_features,
        VVIX_FEATURE_NAMES,
    )
    VOL_OF_VIX_AVAILABLE = True
    logger.info("[v10] vol_of_vix_features loaded OK")
except Exception as _vvx_err:
    logger.warning("[v10] vol_of_vix_features not importable: %s — 3 features zeroed", _vvx_err)
    VOL_OF_VIX_AVAILABLE = False
    VVIX_FEATURE_NAMES = ["vix_realized_vol_21", "vix_vol_zscore_252", "vix_vol_spike_indicator"]

    def add_vol_of_vix_features(df, ticker=None):  # type: ignore[misc]
        for c in VVIX_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0
        return df


# Helper VG: rv_term_structure_features (4 features) — candidate #19
try:
    from rv_term_structure_features import (  # noqa: E402
        add_rv_term_structure_features,
        RV_TERM_FEATURE_NAMES,
    )
    RV_TERM_AVAILABLE = True
    logger.info("[v10] rv_term_structure_features loaded OK")
except Exception as _rvt_err:
    logger.warning("[v10] rv_term_structure_features not importable: %s — 4 features zeroed", _rvt_err)
    RV_TERM_AVAILABLE = False
    RV_TERM_FEATURE_NAMES = [
        "rv5_over_rv21", "rv5_over_rv63",
        "rv_slope_252z", "rv_backwardation_indicator",
    ]

    def add_rv_term_structure_features(df, ticker=None):  # type: ignore[misc]
        for c in RV_TERM_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 1.0 if c in ("rv5_over_rv21", "rv5_over_rv63") else 0.0
        return df


# ---------------------------------------------------------------------------
# Helper VH: amihud_illiquidity_ratio (Amihud 2002 illiquidity ratio, 5 features)
# Wired 2026-05-17. Uses yfinance_daily_OHLCV (close + volume); no extra API.
# .shift(1)-safe: raw ratio computed same-bar then shifted before rolling stats.
# ---------------------------------------------------------------------------
try:
    from amihud_illiquidity_ratio_features import (  # noqa: E402
        compute_amihud_illiquidity_ratio_features,
        AMIHUD_FEATURE_NAMES,
    )
    AMIHUD_AVAILABLE = True
    logger.info("[v10] amihud_illiquidity_ratio_features loaded OK")
except Exception as _amihud_err:
    logger.warning(
        "[v10] amihud_illiquidity_ratio_features not importable: %s — 5 features zeroed",
        _amihud_err,
    )
    AMIHUD_AVAILABLE = False
    AMIHUD_FEATURE_NAMES = [
        "amihud_illiq",
        "amihud_illiq_z21",
        "amihud_illiq_trend",
        "amihud_illiq_ma5",
        "amihud_illiq_spike",
    ]

    def compute_amihud_illiquidity_ratio_features(df, ticker=None):  # type: ignore[misc]
        for c in AMIHUD_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0
        return df


# ---------------------------------------------------------------------------
# Helper VI: rolls_effective_spread (Roll 1984 JFE, 3 features) — Wave H-1
# Wired 2026-05-17. Uses Alpaca 1-min cache; falls back to claudes-test 1Min_merged.
# .shift(1)-safe: all outputs assigned via prior-day shift before returning.
# ---------------------------------------------------------------------------
try:
    from rolls_effective_spread_features import (  # noqa: E402
        compute_rolls_effective_spread_features,
        ROLLS_EFFECTIVE_SPREAD_FEATURE_NAMES,
    )
    ROLLS_SPREAD_AVAILABLE = True
    logger.info("[v10] rolls_effective_spread_features loaded OK")
except Exception as _rolls_err:
    logger.warning(
        "[v10] rolls_effective_spread_features not importable: %s — 3 features zeroed",
        _rolls_err,
    )
    ROLLS_SPREAD_AVAILABLE = False
    ROLLS_EFFECTIVE_SPREAD_FEATURE_NAMES = [
        "rolls_spread_eod",
        "rolls_spread_z21",
        "rolls_spread_rel",
    ]

    def compute_rolls_effective_spread_features(df, ticker=None):  # type: ignore[misc]
        for c in ROLLS_EFFECTIVE_SPREAD_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0
        return df


# ---------------------------------------------------------------------------
# Helper AA: cycle051_features (multi-TF SR daily pivots, 5 features) — Wave Cycle
# Wired 2026-05-17. Source: claudes test/research/archive/cycle051_multi_tf_features_2026-05-03.
# .shift(1)-safe: prior-session HLC drives daily pivot levels.
# ---------------------------------------------------------------------------
try:
    from cycle051_features import (  # noqa: E402
        add_cycle051_features,
        CYCLE051_FEATURE_NAMES,
    )
    CYCLE051_AVAILABLE = True
    logger.info("[v10] cycle051_features loaded OK")
except Exception as _c051_err:
    logger.warning(
        "[v10] cycle051_features not importable: %s — 5 features zeroed", _c051_err
    )
    CYCLE051_AVAILABLE = False
    CYCLE051_FEATURE_NAMES = [
        "sr_1day_pp", "sr_1day_r1", "sr_1day_s1",
        "sr_dist_1day_pp_pct", "sr_above_1day_pp",
    ]

    def add_cycle051_features(df, ticker=None):  # type: ignore[misc]
        for c in CYCLE051_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0 if c == "sr_above_1day_pp" else 0.0
        return df


# ---------------------------------------------------------------------------
# Helper BB: cycle055_features (volatility-gate daily proxies, 5 features) — Wave Cycle
# Wired 2026-05-17. Source: claudes test/research/archive/cycle055_volatility_gates_2026-05-05.
# .shift(1)-safe: all rolling stats shifted 1 bar before assignment.
# ---------------------------------------------------------------------------
try:
    from cycle055_features import (  # noqa: E402
        add_cycle055_features,
        CYCLE055_FEATURE_NAMES,
    )
    CYCLE055_AVAILABLE = True
    logger.info("[v10] cycle055_features loaded OK")
except Exception as _c055_err:
    logger.warning(
        "[v10] cycle055_features not importable: %s — 5 features zeroed", _c055_err
    )
    CYCLE055_AVAILABLE = False
    CYCLE055_FEATURE_NAMES = [
        "vg_atr_pct_14", "vg_range_5d_pct", "vg_vol_regime",
        "vg_in_normal_regime", "vg_rvol_floor_ok",
    ]

    def add_cycle055_features(df, ticker=None):  # type: ignore[misc]
        for c in CYCLE055_FEATURE_NAMES:
            if c not in df.columns:
                if c == "vg_vol_regime":
                    df[c] = 1
                elif c in ("vg_in_normal_regime", "vg_rvol_floor_ok"):
                    df[c] = 0
                else:
                    df[c] = 0.0
        return df


# ---------------------------------------------------------------------------
# Helper CC: cycle058_features (SPY-intra + sector RS, 5 features) — Wave Cycle
# Wired 2026-05-17. Source: claudes test/research/active/cycle058_market_context.
# .shift(1)-safe: merge_asof direction=backward, allow_exact_matches=False.
# ---------------------------------------------------------------------------
try:
    from cycle058_features import (  # noqa: E402
        add_cycle058_features,
        CYCLE058_FEATURE_NAMES,
    )
    CYCLE058_AVAILABLE = True
    logger.info("[v10] cycle058_features loaded OK")
except Exception as _c058_err:
    logger.warning(
        "[v10] cycle058_features not importable: %s — 5 features zeroed", _c058_err
    )
    CYCLE058_AVAILABLE = False
    CYCLE058_FEATURE_NAMES = [
        "mc_spy_intra_cum_ret_eod", "mc_spy_intra_above_or30h_eod",
        "mc_spy_intra_below_or30l_eod", "mc_rs_sector_5d", "mc_rs_sector_5d_sign",
    ]

    def add_cycle058_features(df, ticker=None):  # type: ignore[misc]
        for c in CYCLE058_FEATURE_NAMES:
            if c not in df.columns:
                if c in ("mc_spy_intra_above_or30h_eod",
                         "mc_spy_intra_below_or30l_eod",
                         "mc_rs_sector_5d_sign"):
                    df[c] = 0
                else:
                    df[c] = 0.0
        return df


# ---------------------------------------------------------------------------
# Helper DD: cycle060_features (OI / volume-to-OI / net-delta-z, 3 features) — Wave Cycle
# Wired 2026-05-17. Source: claudes test/research/active/cycle060_options_features.
# .shift(1)-safe: snapshots merged backward + allow_exact_matches=False.
# Note: requires options_snapshots parquet cache populated; otherwise zero-fills.
# ---------------------------------------------------------------------------
try:
    from cycle060_features import (  # noqa: E402
        add_cycle060_features,
        CYCLE060_FEATURE_NAMES,
    )
    CYCLE060_AVAILABLE = True
    logger.info("[v10] cycle060_features loaded OK")
except Exception as _c060_err:
    logger.warning(
        "[v10] cycle060_features not importable: %s — 3 features zeroed", _c060_err
    )
    CYCLE060_AVAILABLE = False
    CYCLE060_FEATURE_NAMES = [
        "put_call_oi_ratio", "volume_to_oi_ratio", "net_delta_exposure_z21",
    ]

    def add_cycle060_features(df, ticker=None):  # type: ignore[misc]
        for c in CYCLE060_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0
        return df


# ---------------------------------------------------------------------------
# Helper EE: cycle061_features (time-of-day daily aggregates, 4 features) — Wave Cycle
# Wired 2026-05-17. Source: claudes test/research/active/cycle061_time_of_day.
# .shift(1)-safe: merge_asof direction=backward + allow_exact_matches=False.
# ---------------------------------------------------------------------------
try:
    from cycle061_features import (  # noqa: E402
        add_cycle061_features,
        CYCLE061_FEATURE_NAMES,
    )
    CYCLE061_AVAILABLE = True
    logger.info("[v10] cycle061_features loaded OK")
except Exception as _c061_err:
    logger.warning(
        "[v10] cycle061_features not importable: %s — 4 features zeroed", _c061_err
    )
    CYCLE061_AVAILABLE = False
    CYCLE061_FEATURE_NAMES = [
        "tod_OR_break_up_rate_5d", "tod_OR_break_down_rate_5d",
        "tod_morning_volume_share", "tod_power_hour_volume_share",
    ]

    def add_cycle061_features(df, ticker=None):  # type: ignore[misc]
        for c in CYCLE061_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0
        return df


# ---------------------------------------------------------------------------
# Helper GG: add_finance_database_features (FinanceDatabase metadata, 4 features)
# Wired 2026-05-17. Source: github:JerBouma/FinanceDatabase (MIT, no paid API).
# Features: fdb_sector, fdb_industry, fdb_market_cap, fdb_exchange (static metadata).
# .shift(1)-safe: static ticker metadata — no intra-bar or future quantity referenced.
# ---------------------------------------------------------------------------
try:
    from add_finance_database_features_features import (  # noqa: E402
        compute_add_finance_database_features_features,
        FDB_FEATURE_NAMES,
        FDB_FEATURE_COUNT,
    )
    FDB_AVAILABLE = True
    logger.info("[v10] add_finance_database_features loaded OK")
except Exception as _fdb_err:
    logger.warning(
        "[v10] add_finance_database_features not importable: %s — 4 features zeroed",
        _fdb_err,
    )
    FDB_AVAILABLE = False
    FDB_FEATURE_COUNT = 4
    FDB_FEATURE_NAMES: list[str] = [  # type: ignore[no-redef]
        "fdb_sector",
        "fdb_industry",
        "fdb_market_cap",
        "fdb_exchange",
    ]

    def compute_add_finance_database_features_features(  # type: ignore[misc]
        df: pd.DataFrame,
        ticker: Optional[str] = None,
    ) -> pd.DataFrame:
        """Stub: zero-fill all 4 fdb cols when module unavailable."""
        for col in FDB_FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0
        return df


# ---------------------------------------------------------------------------
# Helper FG: mlforecast_features (mlforecast-style rolling/EWM/expanding lag
# features, 11 cols: mlf_close_roll5_mean/std, mlf_close_roll21_mean,
# mlf_returns_roll5_mean/std, mlf_volume_roll5_mean/std, mlf_close_ewm_alpha02,
# mlf_returns_expanding_mean, mlf_hl_range_roll5_mean, mlf_close_roll21_max_ratio).
# Source: github:Nixtla/mlforecast (Apache-2.0). Pure-pandas; no paid API.
# shift(1)-safe: all inputs pre-shifted 1 bar inside the module. Wired 2026-05-17.
# ---------------------------------------------------------------------------
try:
    from mlforecast_features_features import (  # noqa: E402
        compute_mlforecast_features_features,
        MLFORECAST_FEATURE_NAMES,
        MLFORECAST_FEATURE_COUNT,
    )
    MLF_AVAILABLE = True
    logger.info("[v10] mlforecast_features loaded OK")
except Exception as _mlf_err:
    logger.warning(
        "[v10] mlforecast_features not importable: %s — 11 features zeroed",
        _mlf_err,
    )
    MLF_AVAILABLE = False
    MLFORECAST_FEATURE_COUNT = 11
    MLFORECAST_FEATURE_NAMES: list[str] = [  # type: ignore[no-redef]
        "mlf_close_roll5_mean",
        "mlf_close_roll5_std",
        "mlf_close_roll21_mean",
        "mlf_returns_roll5_mean",
        "mlf_returns_roll5_std",
        "mlf_volume_roll5_mean",
        "mlf_volume_roll5_std",
        "mlf_close_ewm_alpha02",
        "mlf_returns_expanding_mean",
        "mlf_hl_range_roll5_mean",
        "mlf_close_roll21_max_ratio",
    ]

    def compute_mlforecast_features_features(  # type: ignore[misc]
        df: pd.DataFrame,
        ticker: Optional[str] = None,
    ) -> pd.DataFrame:
        """Stub: zero-fill all 11 mlf_ cols when module unavailable."""
        for col in MLFORECAST_FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0.0
        return df


# ---------------------------------------------------------------------------
# Helper FH: neuralforecast_features (NeuralForecast-inspired decomposition,
# 5 features: nf_trend_slope_21d/63d, nf_fourier_sin/cos_annual,
# nf_residual_vol_21d). Wired 2026-05-17.
# Source: github:Nixtla/neuralforecast (Apache-2.0). Pure pandas/numpy; no paid API.
# shift(1)-safe: all price inputs pre-shifted 1 bar inside the module.
# ---------------------------------------------------------------------------
try:
    from neuralforecast_features_features import (  # noqa: E402
        compute_neuralforecast_features_features,
        NEURALFORECAST_FEATURE_NAMES,
        NEURALFORECAST_FEATURE_COUNT,
    )
    NF_AVAILABLE = True
    logger.info("[v10] neuralforecast_features loaded OK")
except Exception as _nf_err:
    logger.warning(
        "[v10] neuralforecast_features not importable: %s — 5 features zeroed",
        _nf_err,
    )
    NF_AVAILABLE = False
    NEURALFORECAST_FEATURE_COUNT = 5
    NEURALFORECAST_FEATURE_NAMES: list[str] = [  # type: ignore[no-redef]
        "nf_trend_slope_21d",
        "nf_trend_slope_63d",
        "nf_fourier_sin_annual",
        "nf_fourier_cos_annual",
        "nf_residual_vol_21d",
    ]

    def compute_neuralforecast_features_features(  # type: ignore[misc]
        df: pd.DataFrame,
        ticker: Optional[str] = None,
    ) -> pd.DataFrame:
        """Stub: zero-fill all 5 nf_ cols when module unavailable."""
        for col in NEURALFORECAST_FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0.0
        return df


# ---------------------------------------------------------------------------
# Helper GH: worldquant_alpha101_replay (WorldQuant Alpha#101 candle-direction
# z-score, 1 feature: wq101_replay_alpha101_z21). Wired 2026-05-17.
# Source: github:lvlh2/alpha101 (MIT). Pure pandas/numpy; no paid API.
# shift(1)-safe: all price inputs pre-shifted 1 bar inside the module.
# ---------------------------------------------------------------------------
try:
    from worldquant_alpha101_replay_20260517t224845z_features import (  # noqa: E402
        compute_worldquant_alpha101_replay_20260517t224845z_features,
        WQ101_REPLAY_FEATURE_NAMES,
        WQ101_REPLAY_FEATURE_COUNT,
    )
    WQ101_REPLAY_AVAILABLE = True
    logger.info("[v10] worldquant_alpha101_replay loaded OK")
except Exception as _wq101_err:
    logger.warning(
        "[v10] worldquant_alpha101_replay not importable: %s — 1 feature zeroed",
        _wq101_err,
    )
    WQ101_REPLAY_AVAILABLE = False
    WQ101_REPLAY_FEATURE_COUNT = 1
    WQ101_REPLAY_FEATURE_NAMES: list[str] = ["wq101_replay_alpha101_z21"]  # type: ignore[no-redef]

    def compute_worldquant_alpha101_replay_20260517t224845z_features(  # type: ignore[misc]
        df: pd.DataFrame,
        ticker: Optional[str] = None,
    ) -> pd.DataFrame:
        """Stub: zero-fill wq101_replay col when module unavailable."""
        if "wq101_replay_alpha101_z21" not in df.columns:
            df["wq101_replay_alpha101_z21"] = 0.0
        return df


# ---------------------------------------------------------------------------
# Helper GJ: alpha101_ts_safe_subset_replay (STHSF/alpha101 Alpha#6 z-score,
# 1 feature: a101_ts_wq6_z21). Wired 2026-05-17.
# Source: github:STHSF/alpha101 (MIT, no paid API). Pure pandas/numpy.
# shift(1)-safe: open and volume both shifted 1 bar before rolling correlation.
# ---------------------------------------------------------------------------
try:
    from alpha101_ts_safe_subset_replay_20260517t224845z_features import (  # noqa: E402
        compute_alpha101_ts_safe_subset_replay_20260517t224845z_features,
        ALPHA101_TS_SAFE_FEATURE_NAMES,
        ALPHA101_TS_SAFE_FEATURE_COUNT,
    )
    ALPHA101_TS_SAFE_AVAILABLE = True
    logger.info("[v10] alpha101_ts_safe_subset_replay loaded OK")
except Exception as _a101ts_err:
    logger.warning(
        "[v10] alpha101_ts_safe_subset_replay not importable: %s — 1 feature zeroed",
        _a101ts_err,
    )
    ALPHA101_TS_SAFE_AVAILABLE = False
    ALPHA101_TS_SAFE_FEATURE_COUNT = 1
    ALPHA101_TS_SAFE_FEATURE_NAMES: list[str] = ["a101_ts_wq6_z21"]  # type: ignore[no-redef]

    def compute_alpha101_ts_safe_subset_replay_20260517t224845z_features(  # type: ignore[misc]
        df: pd.DataFrame,
        ticker: Optional[str] = None,
    ) -> pd.DataFrame:
        """Stub: zero-fill a101_ts col when module unavailable."""
        if "a101_ts_wq6_z21" not in df.columns:
            df["a101_ts_wq6_z21"] = 0.0
        return df


# ---------------------------------------------------------------------------
# Helper FF: add_alpha_features_core_features (WorldQuant-style alpha101 port,
# 30 features: afc_alpha001..afc_alpha030). Wired 2026-05-17.
# Source: github:GiovanniPioDelvecchio/alpha_features_core (MIT license).
# .shift(1)-safe: all outputs use rolling windows over prior-completed bars only.
# ---------------------------------------------------------------------------
try:
    from add_alpha_features_core_features_features import (  # noqa: E402
        compute_add_alpha_features_core_features_features,
        AFC_CORE_FEATURE_NAMES,
        AFC_CORE_FEATURE_COUNT,
    )
    AFC_CORE_AVAILABLE = True
    logger.info("[v10] add_alpha_features_core_features loaded OK")
except Exception as _afc_err:
    logger.warning(
        "[v10] add_alpha_features_core_features not importable: %s — 30 features zeroed",
        _afc_err,
    )
    AFC_CORE_AVAILABLE = False
    AFC_CORE_FEATURE_COUNT = 30
    AFC_CORE_FEATURE_NAMES: list[str] = [f"afc_alpha{i:03d}" for i in range(1, 31)]  # type: ignore[no-redef]

    def compute_add_alpha_features_core_features_features(  # type: ignore[misc]
        df: pd.DataFrame,
        ticker: Optional[str] = None,
    ) -> pd.DataFrame:
        """Stub: zero-fill all 30 afc_alpha cols when module unavailable."""
        for col in AFC_CORE_FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0.0
        return df


# ---------------------------------------------------------------------------
# Helper GI: hist_data_mythos_deltas_features (Mythos curriculum prior, 6 features)
# Wired 2026-05-17. Source: AI-Tools/reports/mythos_xgboost_integration/per_ticker_summaries/
# Static per-ticker broadcast — same value all rows, no shift(1) needed.
# CAVEAT: 7/500 S&P tickers have summaries (AAPL, BXP, COIN, JPM, NVDA, TPL, XOM);
#         remainder zero-fills with mythos_has_summary=0.
# ---------------------------------------------------------------------------
try:
    from hist_data_mythos_deltas_features import (  # noqa: E402
        add_mythos_deltas_features,
        MYTHOS_DELTA_FEATURE_NAMES,
    )
    MYTHOS_DELTAS_AVAILABLE = True
    logger.info("[v10] hist_data_mythos_deltas_features loaded OK")
except Exception as _mythos_deltas_err:
    logger.warning(
        "[v10] hist_data_mythos_deltas_features not importable: %s — 6 features zeroed",
        _mythos_deltas_err,
    )
    MYTHOS_DELTAS_AVAILABLE = False
    MYTHOS_DELTA_FEATURE_NAMES = [  # type: ignore[assignment]
        "mythos_has_summary",
        "mythos_delta_win_rate",
        "mythos_delta_profit_factor",
        "mythos_delta_total_return",
        "mythos_baseline_profit_factor",
        "mythos_improved_flag",
    ]

    def add_mythos_deltas_features(df, ticker=None):  # type: ignore[misc]
        for c in MYTHOS_DELTA_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0.0
        return df


# ---------------------------------------------------------------------------
# Helper GJ: hist_data_edgar_features (EDGAR filing recency/density, 9 features)
# Wired 2026-05-17. Source: claudes test/data/edgar/data/edgar.db (SQLite,
# 57,066 filings, 500 tickers, 2020-01-02 → 2026-04-24).
# Per-bar dynamic features — .shift(1)-safe via merge_asof(direction='backward',
# allow_exact_matches=False) + searchsorted side='left' (strict-prior boundary).
# Distinct from sec_edgar_features.py (repo-binding stub, unused) — this module
# reads the local indexed EDGAR DB directly.
# ---------------------------------------------------------------------------
try:
    from hist_data_edgar_features import (  # noqa: E402
        add_edgar_features,
        EDGAR_FEATURE_NAMES,
    )
    EDGAR_DB_AVAILABLE = True
    logger.info("[v10] hist_data_edgar_features loaded OK")
except Exception as _edgar_err:
    logger.warning(
        "[v10] hist_data_edgar_features not importable: %s — 9 features zeroed",
        _edgar_err,
    )
    EDGAR_DB_AVAILABLE = False
    EDGAR_FEATURE_NAMES = [  # type: ignore[assignment]
        "edgar_days_since_any_filing",
        "edgar_days_since_8k",
        "edgar_days_since_10q",
        "edgar_days_since_10k",
        "edgar_filing_flag_7d",
        "edgar_filing_flag_30d",
        "edgar_eightk_flag_7d",
        "edgar_filings_count_90d",
        "edgar_has_10k_this_year",
    ]

    def add_edgar_features(df, ticker=None):  # type: ignore[misc]
        for c in EDGAR_FEATURE_NAMES:
            if c not in df.columns:
                df[c] = 0
        return df


# ---------------------------------------------------------------------------
# Helper GK: ceo_personal_donation_flag_political_replay (FEC Schedule A
# individual-contributions employer-search signal, 1 feature). Wired 2026-05-17.
# Source: api.fec.gov/v1/schedules/schedule_a/ (public, no paid API key).
# .shift(1)-safe: rolling 90-day count uses only contribution_receipt_date <
# bar_date (strict-prior searchsorted boundary). See module for full audit.
# ---------------------------------------------------------------------------
try:
    from ceo_personal_donation_flag_political_replay_20260517t225454z_features import (  # noqa: E402
        compute_ceo_personal_donation_flag_political_replay_20260517t225454z_features,
        CEO_DONATION_FEATURE_NAMES,
        CEO_DONATION_FEATURE_COUNT,
    )
    CEO_DONATION_AVAILABLE = True
    logger.info("[v10] ceo_personal_donation_flag_political_replay loaded OK")
except Exception as _ceo_don_err:
    logger.warning(
        "[v10] ceo_personal_donation_flag_political_replay not importable: %s — 1 feature zeroed",
        _ceo_don_err,
    )
    CEO_DONATION_AVAILABLE = False
    CEO_DONATION_FEATURE_COUNT = 1
    CEO_DONATION_FEATURE_NAMES: list[str] = ["fec_donation_flag_90d"]  # type: ignore[no-redef]

    def compute_ceo_personal_donation_flag_political_replay_20260517t225454z_features(  # type: ignore[misc]
        df: pd.DataFrame,
        ticker: Optional[str] = None,
    ) -> pd.DataFrame:
        """Stub: zero-fill fec_donation_flag_90d when module unavailable."""
        if "fec_donation_flag_90d" not in df.columns:
            df["fec_donation_flag_90d"] = 0.0
        return df


# ---------------------------------------------------------------------------
# Helper GL: senate_efd_options_disclosure_count_30d_replay (Senate STOCK Act
# options-transaction rolling count, 1 feature). Wired 2026-05-17.
# Source: QuiverQuant free congressional-trading API (no paid API key).
# .shift(1)-safe: rolling 30d count uses only disclosures whose public
# report_date < bar_date (strict-prior searchsorted boundary).
# ---------------------------------------------------------------------------
try:
    from senate_efd_options_disclosure_count_30d_replay_20260517t225454z_features import (  # noqa: E402
        compute_senate_efd_options_disclosure_count_30d_replay_20260517t225454z_features,
        SENATE_EFD_OPTIONS_FEATURE_NAMES,
        SENATE_EFD_OPTIONS_FEATURE_COUNT,
    )
    SENATE_EFD_OPTIONS_AVAILABLE = True
    logger.info("[v10] senate_efd_options_disclosure_count_30d_replay loaded OK")
except Exception as _sefd_err:
    logger.warning(
        "[v10] senate_efd_options_disclosure_count_30d_replay not importable: %s — 1 feature zeroed",
        _sefd_err,
    )
    SENATE_EFD_OPTIONS_AVAILABLE = False
    SENATE_EFD_OPTIONS_FEATURE_COUNT = 1
    SENATE_EFD_OPTIONS_FEATURE_NAMES: list[str] = ["senate_efd_options_count_30d"]  # type: ignore[no-redef]

    def compute_senate_efd_options_disclosure_count_30d_replay_20260517t225454z_features(  # type: ignore[misc]
        df: pd.DataFrame,
        ticker: Optional[str] = None,
    ) -> pd.DataFrame:
        """Stub: zero-fill senate_efd_options_count_30d when module unavailable."""
        if "senate_efd_options_count_30d" not in df.columns:
            df["senate_efd_options_count_30d"] = 0.0
        return df


# ---------------------------------------------------------------------------
# v10 feature builder
# ---------------------------------------------------------------------------


def _build_v10_features_impl(
    ticker: str,
    universe_agg: Optional[dict] = None,
    use_mythos: bool = False,
) -> tuple[pd.DataFrame, int, dict]:
    """Build v10 feature set: full v9 stack + 5 newly-wired modules.

    Pipeline order:
      1. build_v9_features()      — v8 base + optional 256-dim Mythos
      2. add_alpaca_features()    — earnings/div/split/metadata (13 cols)
         [BEFORE daily_integration so earn_contam_gate has access to
         days_until_earnings / days_since_last_earnings from alpaca layer]
      2.5 add_insider_form4_features() — SEC Form 4 insider disclosures
         (8 cols: buy/sell counts, cluster flags, days-since, $-amount).
         Wired 2026-05-17 — was the "gov-trades" module gap.
      3. add_daily_integration_features() — 7 composite cols
      4. add_dfs_features()       — up to 60 DFS depth-2 interaction cols
      5. add_mastery_priors()     — 7 per-ticker priors features parsed from
         $SP/mastery_files/*.md (v4 + v10 mastery markdown artifacts).
         .shift(1)-safe via mtime-gated age column. Wired 2026-05-17.
      6. add_paper_trade_outcome_features() — 7 features computed from CLOSED
         paper trades (win-rate, PF, count, last-outcome-sign, avg holding,
         signal-to-fill lag, current drawdown). Rolling 30 calendar days,
         .shift(1)-safe via strict trade_date < bar_date merge_asof.
         Source: $SP/paper_trade/state/*_state.json. Wired 2026-05-17.

    Args:
        ticker: Stock symbol.
        universe_agg: Cross-sectional precomputed aggregates dict (from csf).
        use_mythos: If True, append 256 Mythos embedding features.

    Returns:
        Tuple of:
          - pd.DataFrame with all features + target column 'y'.
          - int fallback_rows (Mythos zero-embedding rows; 0 when Mythos off).
          - dict module_feature_counts with per-module feature counts.
    """
    # ---- Step 1: Full v9 stack ----
    f, mythos_fallback_rows = build_v9_features(ticker, universe_agg, use_mythos=use_mythos)
    after_v9 = f.shape[1]
    logger.info("  [v10] after v9 stack: %d cols", after_v9)

    # ---- Step 2: Alpaca features (must precede daily_integration) ----
    before_alp = f.shape[1]
    try:
        f = add_alpaca_features(f, ticker)
    except Exception as exc:
        logger.warning("  [v10] alpaca_features call failed (%s): %s — zeroing", ticker, exc)
        for col in ALPACA_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_alpaca = f.shape[1] - before_alp
    logger.info("  [v10] +alpaca: +%d cols -> %d total", added_alpaca, f.shape[1])

    # ---- Step 2.5: Insider Form 4 features (idempotent guard) ----
    # NOTE 2026-05-17: insider_form4_features is ALREADY wired into the v7→v9
    # chain (build_v7_features imports f4f and calls add_insider_form4_features
    # before returning). The 8 insider cols thus arrive inside the v9_base
    # count. We retain Helper-D import (for explicit dependency tracking) and
    # only re-fill missing cols defensively — never overwrite an existing
    # value because v9 may already have populated them.
    before_ins = f.shape[1]
    missing_insider = [c for c in INSIDER_FORM4_FEATURE_NAMES if c not in f.columns]
    if missing_insider:
        try:
            f = add_insider_form4_features(f, ticker)
            logger.info(
                "  [v10] insider_form4 cols missing from v9 (%d) — backfilled via Helper D",
                len(missing_insider),
            )
        except Exception as exc:
            logger.warning(
                "  [v10] insider_form4_features fallback call failed (%s): %s — zeroing missing cols",
                ticker, exc,
            )
            for col in missing_insider:
                if col not in f.columns:
                    f[col] = 0.0
    added_insider = f.shape[1] - before_ins
    logger.info(
        "  [v10] insider_form4 check: +%d new cols (already-present: %d) -> %d total",
        added_insider, 8 - len(missing_insider), f.shape[1],
    )

    # ---- Step 3: Daily integration features ----
    before_di = f.shape[1]
    try:
        f = add_daily_integration_features(f, ticker)
    except Exception as exc:
        logger.warning(
            "  [v10] daily_integration_features call failed (%s): %s — zeroing", ticker, exc
        )
        for col in DAILY_INT_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_daily_int = f.shape[1] - before_di
    logger.info("  [v10] +daily_integration: +%d cols -> %d total", added_daily_int, f.shape[1])

    # ---- Step 4: DFS features ----
    before_dfs = f.shape[1]
    try:
        f = add_dfs_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] add_dfs_features call failed (%s): %s — skipping", ticker, exc)
    added_dfs = f.shape[1] - before_dfs
    logger.info("  [v10] +dfs: +%d cols -> %d total", added_dfs, f.shape[1])

    # ---- Step 5: Mastery priors (past-test artifact priors, 7 features) ----
    # Wired 2026-05-17. Reads $SP/mastery_files/*.md (311 v4 + 7 v10) and emits
    # per-ticker prior-mastered flags, PF, DD, top-10 flag, .shift(1)-safe age.
    before_mp = f.shape[1]
    try:
        f = add_mastery_priors(f, ticker)
    except Exception as exc:
        logger.warning(
            "  [v10] add_mastery_priors call failed (%s): %s — zeroing", ticker, exc
        )
        for col in MASTERY_PRIORS_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0
    added_mastery_priors = f.shape[1] - before_mp
    logger.info(
        "  [v10] +mastery_priors: +%d cols -> %d total", added_mastery_priors, f.shape[1]
    )

    # ---- Step 6: Paper-trade outcome features (live-feedback loop) ----
    # Reads $SP/paper_trade/state/*_state.json closed_trades[] and adds 7
    # rolling 30d outcome features. Zero-fills gracefully when no paper trades
    # exist yet for the ticker. .shift(1)-safe via strict-less-than merge_asof.
    before_pt = f.shape[1]
    try:
        f = add_paper_trade_outcome_features(f, ticker)
    except Exception as exc:
        logger.warning(
            "  [v10] add_paper_trade_outcome_features call failed (%s): %s — zeroing",
            ticker, exc,
        )
        for col in PT_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0 if col in (
                    "paper_trade_count_30d", "paper_trade_last_outcome_sign"
                ) else 0.0
    added_paper_trade_outcomes = f.shape[1] - before_pt
    logger.info(
        "  [v10] +paper_trade_outcomes: +%d cols -> %d total",
        added_paper_trade_outcomes, f.shape[1],
    )

    # ---- Step 7: Stumpy (matrix-profile motif/discord) ----
    before_stumpy = f.shape[1]
    try:
        f = add_stumpy_features(f, ticker)
    except Exception as exc:
        logger.warning("  [v10] stumpy call failed (%s): %s — skipping", ticker, exc)
    added_stumpy = f.shape[1] - before_stumpy
    logger.info("  [v10] +stumpy: +%d cols -> %d total", added_stumpy, f.shape[1])

    # ---- Step 8: FFN (Sortino/Calmar/Ulcer/Downside risk metrics) ----
    before_ffn = f.shape[1]
    try:
        f = add_ffn_features(f, ticker)
    except Exception as exc:
        logger.warning("  [v10] ffn call failed (%s): %s — skipping", ticker, exc)
    added_ffn = f.shape[1] - before_ffn
    logger.info("  [v10] +ffn: +%d cols -> %d total", added_ffn, f.shape[1])

    # ---- Step 9: pandas-ta-classic (non-TA-Lib indicators) ----
    before_ptc = f.shape[1]
    try:
        f = add_pandas_ta_classic_features(f, ticker)
    except Exception as exc:
        logger.warning("  [v10] pandas_ta_classic call failed (%s): %s — skipping", ticker, exc)
    added_ptc = f.shape[1] - before_ptc
    logger.info("  [v10] +pandas_ta_classic: +%d cols -> %d total", added_ptc, f.shape[1])

    # ---- Step 10: options_flow (Wave A, 3 features) ----
    before_of = f.shape[1]
    try:
        f = add_options_flow_features(f, ticker)
    except Exception as exc:
        logger.warning("  [v10] options_flow call failed (%s): %s — zeroing", ticker, exc)
        for col in OPTIONS_FLOW_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0 if col != "unusual_options_activity_flag" else 0
    added_options_flow = f.shape[1] - before_of
    logger.info("  [v10] +options_flow: +%d cols -> %d total", added_options_flow, f.shape[1])

    # ---- Step 11: govtrades (Wave A, 3 features) ----
    before_gt = f.shape[1]
    try:
        f = add_govtrades_features(f, ticker)
    except Exception as exc:
        logger.warning("  [v10] govtrades call failed (%s): %s — zeroing", ticker, exc)
        for col in GOVTRADES_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0 if col != "congress_buy_sell_ratio_5d" else 0.0
    added_govtrades = f.shape[1] - before_gt
    logger.info("  [v10] +govtrades: +%d cols -> %d total", added_govtrades, f.shape[1])

    # ---- Step 12: time_of_day (Wave A, 1 feature) ----
    before_tod = f.shape[1]
    try:
        f = add_time_of_day_features(f, ticker)
    except Exception as exc:
        logger.warning("  [v10] time_of_day call failed (%s): %s — zeroing", ticker, exc)
        if "time_of_day_bucket" not in f.columns:
            f["time_of_day_bucket"] = 2
    added_tod = f.shape[1] - before_tod
    logger.info("  [v10] +time_of_day: +%d cols -> %d total", added_tod, f.shape[1])

    # ---- Step 13: gabriel_priors (Wave A, 5 features) ----
    before_gp = f.shape[1]
    try:
        f = add_gabriel_priors_features(f, ticker)
    except Exception as exc:
        logger.warning("  [v10] gabriel_priors call failed (%s): %s — zeroing", ticker, exc)
        for col in GABRIEL_PRIORS_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0 if col == "gabriel_champion_n_trades" else 0.0
    added_gabriel = f.shape[1] - before_gp
    logger.info("  [v10] +gabriel_priors: +%d cols -> %d total", added_gabriel, f.shape[1])

    # ---- Step 14: vix_term_structure_v2 (3 features: ratio, inverted, z10) ----
    before_vts = f.shape[1]
    try:
        f = compute_vix_term_structure_v2_features(f)
    except Exception as exc:
        logger.warning("  [v10] vix_term_structure_v2 call failed (%s): %s — zeroing", ticker, exc)
        for col in VIX_TS_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0 if col != "vix_term_inverted" else 0
    added_vts = f.shape[1] - before_vts
    logger.info("  [v10] +vix_term_structure_v2: +%d cols -> %d total", added_vts, f.shape[1])

    # ---- Step 15: garch_11_cond_vol (GARCH(1,1) cond vol, 3 features) ----
    before_garch = f.shape[1]
    try:
        f = compute_garch_11_cond_vol_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] garch_11_cond_vol call failed (%s): %s — zeroing", ticker, exc)
        for col in GARCH11_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_garch = f.shape[1] - before_garch
    logger.info("  [v10] +garch_11_cond_vol: +%d cols -> %d total", added_garch, f.shape[1])

    # ---- Step 16: egarch_11_leverage (EGARCH(1,1) leverage-effect vol, 3 features) ----
    before_egarch_lev = f.shape[1]
    try:
        f = compute_egarch_11_leverage_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] egarch_11_leverage call failed (%s): %s — zeroing", ticker, exc)
        for col in EGARCH11_LEV_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_egarch_lev = f.shape[1] - before_egarch_lev
    logger.info("  [v10] +egarch_11_leverage: +%d cols -> %d total", added_egarch_lev, f.shape[1])

    # ---- Step 17: vpin_50bucket (VPIN 50-bucket BVC approximation, 3 features) ----
    before_vpin = f.shape[1]
    try:
        f = compute_vpin_50bucket_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] vpin_50bucket call failed (%s): %s — zeroing", ticker, exc)
        for col in VPIN_50BUCKET_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_vpin = f.shape[1] - before_vpin
    logger.info("  [v10] +vpin_50bucket: +%d cols -> %d total", added_vpin, f.shape[1])

    # ---- Step 18: kyles_lambda_intraday (Kyle 1985 λ, BVC approx, 3 features) ----
    before_kl = f.shape[1]
    try:
        f = compute_kyles_lambda_intraday_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] kyles_lambda_intraday call failed (%s): %s — zeroing", ticker, exc)
        for col in KYLES_LAMBDA_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_kyles_lambda = f.shape[1] - before_kl
    logger.info("  [v10] +kyles_lambda_intraday: +%d cols -> %d total", added_kyles_lambda, f.shape[1])

    # ---- Step 19: vpin_features (TRUE 1-min VPIN, 5 features) — Wave M-1 #1 ----
    before_vpin_intra = f.shape[1]
    try:
        f = add_vpin_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] vpin_features (intraday) call failed (%s): %s — zeroing", ticker, exc)
        for col in VPIN_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_vpin_intra = f.shape[1] - before_vpin_intra
    logger.info("  [v10] +vpin_intraday: +%d cols -> %d total", added_vpin_intra, f.shape[1])

    # ---- Step 20: tick_imbalance_features (Lee-Ready, 5 features) — Wave M-1 #6 ----
    before_ti = f.shape[1]
    try:
        f = add_tick_imbalance_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] tick_imbalance_features call failed (%s): %s — zeroing", ticker, exc)
        for col in TICK_IMBALANCE_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_tick_imbalance = f.shape[1] - before_ti
    logger.info("  [v10] +tick_imbalance: +%d cols -> %d total", added_tick_imbalance, f.shape[1])

    # ---- Step 21: volume_profile_features (POC/VA/shape, 6 features) — Wave M-1 #11 ----
    before_vp = f.shape[1]
    try:
        f = add_volume_profile_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] volume_profile_features call failed (%s): %s — zeroing", ticker, exc)
        for col in VOLUME_PROFILE_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_volume_profile = f.shape[1] - before_vp
    logger.info("  [v10] +volume_profile: +%d cols -> %d total", added_volume_profile, f.shape[1])

    # ---- Step 22: auction_features (open/close auction, 6 features) — Wave M-1 #12 ----
    before_auc = f.shape[1]
    try:
        f = add_auction_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] auction_features call failed (%s): %s — zeroing", ticker, exc)
        for col in AUCTION_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_auction = f.shape[1] - before_auc
    logger.info("  [v10] +auction: +%d cols -> %d total", added_auction, f.shape[1])

    # ---- Step 23: vol_of_vol (3 features) — Wave V-1 #7 ----
    before_vov = f.shape[1]
    try:
        f = add_vol_of_vol_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] vol_of_vol call failed (%s): %s — zeroing", ticker, exc)
        for col in VOL_OF_VOL_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_vov = f.shape[1] - before_vov
    logger.info("  [v10] +vol_of_vol: +%d cols -> %d total", added_vov, f.shape[1])

    # ---- Step 24: vol_risk_premium (4 features) — Wave V-1 #12 ----
    before_vrp = f.shape[1]
    try:
        f = add_vol_risk_premium_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] vol_risk_premium call failed (%s): %s — zeroing", ticker, exc)
        for col in VRP_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0 if col != "vrp_sign_flip" else 0
    added_vrp = f.shape[1] - before_vrp
    logger.info("  [v10] +vol_risk_premium: +%d cols -> %d total", added_vrp, f.shape[1])

    # ---- Step 25: vol_target_sizing (2 features) — Wave V-1 #13 ----
    # MUST run AFTER Step 15 (garch_11_cond_vol) so vol_target_ratio can consume it.
    before_vt = f.shape[1]
    try:
        f = add_vol_target_sizing_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] vol_target_sizing call failed (%s): %s — neutral fill", ticker, exc)
        for col in VOL_TARGET_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 1.0
    added_vt = f.shape[1] - before_vt
    logger.info("  [v10] +vol_target_sizing: +%d cols -> %d total", added_vt, f.shape[1])

    # ---- Step 26: vol_breakout_nr (5 features) — Wave V-1 #14 ----
    before_nr = f.shape[1]
    try:
        f = add_vol_breakout_nr_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] vol_breakout_nr call failed (%s): %s — zeroing", ticker, exc)
        for col in VOL_BREAKOUT_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0 if col != "range_pct_of_atr20" else 1.0
    added_nr = f.shape[1] - before_nr
    logger.info("  [v10] +vol_breakout_nr: +%d cols -> %d total", added_nr, f.shape[1])

    # ---- Step 27: bollinger_keltner_squeeze (4 features) — Wave V-1 #15 ----
    before_sq = f.shape[1]
    try:
        f = add_bollinger_keltner_squeeze_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] bb_kc_squeeze call failed (%s): %s — zeroing", ticker, exc)
        for col in SQUEEZE_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_sq = f.shape[1] - before_sq
    logger.info("  [v10] +bb_kc_squeeze: +%d cols -> %d total", added_sq, f.shape[1])

    # ---- Step 28: vol_of_vix (3 features) — Wave V-1 #18 ----
    before_vvx = f.shape[1]
    try:
        f = add_vol_of_vix_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] vol_of_vix call failed (%s): %s — zeroing", ticker, exc)
        for col in VVIX_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_vvx = f.shape[1] - before_vvx
    logger.info("  [v10] +vol_of_vix: +%d cols -> %d total", added_vvx, f.shape[1])

    # ---- Step 29: rv_term_structure (4 features) — Wave V-1 #19 ----
    before_rvt = f.shape[1]
    try:
        f = add_rv_term_structure_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] rv_term_structure call failed (%s): %s — zeroing", ticker, exc)
        for col in RV_TERM_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 1.0 if col in ("rv5_over_rv21", "rv5_over_rv63") else 0.0
    added_rvt = f.shape[1] - before_rvt
    logger.info("  [v10] +rv_term_structure: +%d cols -> %d total", added_rvt, f.shape[1])

    # ---- Step 30: amihud_illiquidity_ratio (Amihud 2002, 5 features) — Wave G ----
    before_amihud = f.shape[1]
    try:
        f = compute_amihud_illiquidity_ratio_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] amihud_illiq call failed (%s): %s — zeroing", ticker, exc)
        for col in AMIHUD_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_amihud = f.shape[1] - before_amihud
    logger.info("  [v10] +amihud_illiq: +%d cols -> %d total", added_amihud, f.shape[1])

    # ---- Step 31: rolls_effective_spread (Roll 1984 JFE, 3 features) — Wave H-1 ----
    before_rolls = f.shape[1]
    try:
        f = compute_rolls_effective_spread_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] rolls_effective_spread call failed (%s): %s — zeroing", ticker, exc)
        for col in ROLLS_EFFECTIVE_SPREAD_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_rolls = f.shape[1] - before_rolls
    logger.info("  [v10] +rolls_effective_spread: +%d cols -> %d total", added_rolls, f.shape[1])

    # ---- Step 32: cycle051_features (daily-pivot SR, 5 features) — Wave Cycle ----
    before_c051 = f.shape[1]
    try:
        f = add_cycle051_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] cycle051 call failed (%s): %s — zeroing", ticker, exc)
        for col in CYCLE051_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0 if col == "sr_above_1day_pp" else 0.0
    added_c051 = f.shape[1] - before_c051
    logger.info("  [v10] +cycle051: +%d cols -> %d total", added_c051, f.shape[1])

    # ---- Step 33: cycle055_features (vol-gate daily proxies, 5 features) — Wave Cycle ----
    before_c055 = f.shape[1]
    try:
        f = add_cycle055_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] cycle055 call failed (%s): %s — zeroing", ticker, exc)
        for col in CYCLE055_FEATURE_NAMES:
            if col not in f.columns:
                if col == "vg_vol_regime":
                    f[col] = 1
                elif col in ("vg_in_normal_regime", "vg_rvol_floor_ok"):
                    f[col] = 0
                else:
                    f[col] = 0.0
    added_c055 = f.shape[1] - before_c055
    logger.info("  [v10] +cycle055: +%d cols -> %d total", added_c055, f.shape[1])

    # ---- Step 34: cycle058_features (SPY-intra + sector RS, 5 features) — Wave Cycle ----
    before_c058 = f.shape[1]
    try:
        f = add_cycle058_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] cycle058 call failed (%s): %s — zeroing", ticker, exc)
        for col in CYCLE058_FEATURE_NAMES:
            if col not in f.columns:
                if col in ("mc_spy_intra_above_or30h_eod",
                           "mc_spy_intra_below_or30l_eod",
                           "mc_rs_sector_5d_sign"):
                    f[col] = 0
                else:
                    f[col] = 0.0
    added_c058 = f.shape[1] - before_c058
    logger.info("  [v10] +cycle058: +%d cols -> %d total", added_c058, f.shape[1])

    # ---- Step 35: cycle060_features (OI / vol-to-OI / net-delta-z, 3 features) — Wave Cycle ----
    before_c060 = f.shape[1]
    try:
        f = add_cycle060_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] cycle060 call failed (%s): %s — zeroing", ticker, exc)
        for col in CYCLE060_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_c060 = f.shape[1] - before_c060
    logger.info("  [v10] +cycle060: +%d cols -> %d total", added_c060, f.shape[1])

    # ---- Step 36: cycle061_features (TOD daily aggregates, 4 features) — Wave Cycle ----
    before_c061 = f.shape[1]
    try:
        f = add_cycle061_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] cycle061 call failed (%s): %s — zeroing", ticker, exc)
        for col in CYCLE061_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_c061 = f.shape[1] - before_c061
    logger.info("  [v10] +cycle061: +%d cols -> %d total", added_c061, f.shape[1])

    # ---- Step 37: add_alpha_features_core_features (WorldQuant alpha101 port, 30 features) ----
    before_afc = f.shape[1]
    try:
        f = compute_add_alpha_features_core_features_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] afc_core call failed (%s): %s — zeroing", ticker, exc)
        for col in AFC_CORE_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_afc = f.shape[1] - before_afc
    logger.info("  [v10] +alpha_features_core: +%d cols -> %d total", added_afc, f.shape[1])

    # ---- Step 38: add_finance_database_features (FinanceDatabase metadata, 4 features) — Wave FDB ----
    before_fdb = f.shape[1]
    try:
        f = compute_add_finance_database_features_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] finance_database call failed (%s): %s — zeroing", ticker, exc)
        for col in FDB_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0
    added_fdb = f.shape[1] - before_fdb
    logger.info("  [v10] +finance_database: +%d cols -> %d total", added_fdb, f.shape[1])

    # ---- Step 39: mlforecast_features (rolling/EWM/expanding lag, 11 features) — Wave MLF ----
    before_mlf = f.shape[1]
    try:
        f = compute_mlforecast_features_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] mlforecast_features call failed (%s): %s — zeroing", ticker, exc)
        for col in MLFORECAST_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_mlf = f.shape[1] - before_mlf
    logger.info("  [v10] +mlforecast_features: +%d cols -> %d total", added_mlf, f.shape[1])

    # ---- Step 40: neuralforecast_features (NBEATS/NHITS decomp, 5 features) — Wave NF ----
    before_nf = f.shape[1]
    try:
        f = compute_neuralforecast_features_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] neuralforecast_features call failed (%s): %s — zeroing", ticker, exc)
        for col in NEURALFORECAST_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_nf = f.shape[1] - before_nf
    logger.info("  [v10] +neuralforecast_features: +%d cols -> %d total", added_nf, f.shape[1])

    # ---- Step 41: worldquant_alpha101_replay (Alpha#101 z-score, 1 feature) — Wave WQ101 ----
    before_wq101 = f.shape[1]
    try:
        f = compute_worldquant_alpha101_replay_20260517t224845z_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] wq101_replay call failed (%s): %s — zeroing", ticker, exc)
        if "wq101_replay_alpha101_z21" not in f.columns:
            f["wq101_replay_alpha101_z21"] = 0.0
    added_wq101 = f.shape[1] - before_wq101
    logger.info("  [v10] +wq101_replay: +%d cols -> %d total", added_wq101, f.shape[1])

    # ---- Step 42: mythos_deltas (Mythos curriculum prior, 6 static features) — Wave D6 ----
    before_mythos_deltas = f.shape[1]
    try:
        f = add_mythos_deltas_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] mythos_deltas call failed (%s): %s — zeroing", ticker, exc)
        for col in MYTHOS_DELTA_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0.0
    added_mythos_deltas = f.shape[1] - before_mythos_deltas
    logger.info("  [v10] +mythos_deltas: +%d cols -> %d total", added_mythos_deltas, f.shape[1])

    # ---- Step 43: alpha101_ts_safe_subset_replay (Alpha#6 z-score, 1 feature) — Wave TS101 ----
    before_a101ts = f.shape[1]
    try:
        f = compute_alpha101_ts_safe_subset_replay_20260517t224845z_features(f, ticker=ticker)
    except Exception as exc:
        logger.warning("  [v10] alpha101_ts_safe call failed (%s): %s — zeroing", ticker, exc)
        if "a101_ts_wq6_z21" not in f.columns:
            f["a101_ts_wq6_z21"] = 0.0
    added_a101ts = f.shape[1] - before_a101ts
    logger.info("  [v10] +alpha101_ts_safe_subset: +%d cols -> %d total", added_a101ts, f.shape[1])

    # ---- Step 44: edgar (EDGAR filing recency/density, 9 dynamic features) — Wave B1 ----
    # Source: claudes test/data/edgar/data/edgar.db (500 tickers, 2020-2026).
    # .shift(1)-safe via merge_asof(direction='backward', allow_exact_matches=False).
    before_edgar = f.shape[1]
    try:
        f = add_edgar_features(f, ticker)
    except Exception as exc:
        logger.warning("  [v10] edgar call failed (%s): %s — zeroing", ticker, exc)
        for col in EDGAR_FEATURE_NAMES:
            if col not in f.columns:
                f[col] = 0
    added_edgar = f.shape[1] - before_edgar
    logger.info("  [v10] +edgar: +%d cols -> %d total", added_edgar, f.shape[1])

    # ---- Step 45: ceo_personal_donation_flag_political_replay (FEC, 1 feature) — Wave FEC1 ----
    # Source: api.fec.gov Schedule A employer-search. Public, no paid key.
    # .shift(1)-safe via strict receipt_date < bar_date searchsorted boundary.
    before_ceo_don = f.shape[1]
    try:
        f = compute_ceo_personal_donation_flag_political_replay_20260517t225454z_features(
            f, ticker=ticker
        )
    except Exception as exc:
        logger.warning(
            "  [v10] ceo_personal_donation call failed (%s): %s — zeroing", ticker, exc
        )
        if "fec_donation_flag_90d" not in f.columns:
            f["fec_donation_flag_90d"] = 0.0
    added_ceo_don = f.shape[1] - before_ceo_don
    logger.info(
        "  [v10] +ceo_personal_donation: +%d cols -> %d total", added_ceo_don, f.shape[1]
    )

    # ---- Step 46: senate_efd_options_disclosure_count_30d_replay (1 feature) — Wave EFD1 ----
    # Source: QuiverQuant free congressional-trading API. No paid key.
    # .shift(1)-safe via strict report_date < bar_date searchsorted boundary.
    before_sefd = f.shape[1]
    try:
        f = compute_senate_efd_options_disclosure_count_30d_replay_20260517t225454z_features(
            f, ticker=ticker
        )
    except Exception as exc:
        logger.warning(
            "  [v10] senate_efd_options call failed (%s): %s — zeroing", ticker, exc
        )
        if "senate_efd_options_count_30d" not in f.columns:
            f["senate_efd_options_count_30d"] = 0.0
    added_sefd = f.shape[1] - before_sefd
    logger.info(
        "  [v10] +senate_efd_options: +%d cols -> %d total", added_sefd, f.shape[1]
    )

    # ---- Dedup + dropna on critical columns ----
    f = f.loc[:, ~f.columns.duplicated()]
    f = f.dropna(subset=["rsi_14", "atr_14", "ema_200", "fwd_ret_21d", "y"])

    module_feature_counts = {
        "v9_base": after_v9,
        "alpaca_added": added_alpaca,
        "insider_form4_added": added_insider,
        "daily_integration_added": added_daily_int,
        "dfs_added": added_dfs,
        "mastery_priors_added": added_mastery_priors,
        "paper_trade_outcomes_added": added_paper_trade_outcomes,
        "stumpy_added": added_stumpy,
        "ffn_added": added_ffn,
        "pandas_ta_classic_added": added_ptc,
        # Wave A (2026-05-17)
        "options_flow_added": added_options_flow,
        "govtrades_added": added_govtrades,
        "time_of_day_added": added_tod,
        "gabriel_priors_added": added_gabriel,
        # Wave B (2026-05-17)
        "vix_term_structure_v2_added": added_vts,
        # Wave C (2026-05-17)
        "garch_11_cond_vol_added": added_garch,
        # Wave D (2026-05-17)
        "egarch_11_leverage_added": added_egarch_lev,
        # Wave E (2026-05-17)
        "vpin_50bucket_added": added_vpin,
        # Wave F (2026-05-17)
        "kyles_lambda_intraday_added": added_kyles_lambda,
        # Wave M-1 microstructure top-4 (2026-05-17)
        "vpin_intraday_added": added_vpin_intra,
        "tick_imbalance_added": added_tick_imbalance,
        "volume_profile_added": added_volume_profile,
        "auction_added": added_auction,
        # Wave V-1 vol/regime low-cost 7-pack (2026-05-17)
        "vol_of_vol_added": added_vov,
        "vol_risk_premium_added": added_vrp,
        "vol_target_sizing_added": added_vt,
        "vol_breakout_nr_added": added_nr,
        "bollinger_keltner_squeeze_added": added_sq,
        "vol_of_vix_added": added_vvx,
        "rv_term_structure_added": added_rvt,
        # Wave G (2026-05-17)
        "amihud_illiq_added": added_amihud,
        # Wave H-1 (2026-05-17)
        "rolls_effective_spread_added": added_rolls,
        # Wave Cycle (2026-05-17) — 5 cycle engines from claudes test
        "cycle051_added": added_c051,
        "cycle055_added": added_c055,
        "cycle058_added": added_c058,
        "cycle060_added": added_c060,
        "cycle061_added": added_c061,
        # Wave AFC (2026-05-17) — WorldQuant alpha101 port via alpha_features_core
        "alpha_features_core_added": added_afc,
        # Wave FDB (2026-05-17) — FinanceDatabase metadata (sector/industry/market_cap/exchange)
        "finance_database_added": added_fdb,
        # Wave MLF (2026-05-17) — mlforecast-style rolling/EWM/expanding lag features (11 cols)
        "mlforecast_features_added": added_mlf,
        # Wave NF (2026-05-17) — NeuralForecast decomp features (5 cols)
        "neuralforecast_features_added": added_nf,
        # Wave WQ101 (2026-05-17) — WorldQuant Alpha#101 z-score (1 col)
        "worldquant_alpha101_replay_added": added_wq101,
        # Wave D6 (2026-05-17) — Mythos curriculum prior static features (6 cols)
        "mythos_deltas_added": added_mythos_deltas,
        # Wave TS101 (2026-05-17) — STHSF/alpha101 Alpha#6 z-score (1 col)
        "alpha101_ts_safe_subset_added": added_a101ts,
        # Wave B1 (2026-05-17) — EDGAR filing recency/density (9 cols)
        "edgar_added": added_edgar,
        # Wave FEC1 (2026-05-17) — FEC employer donation rolling flag (1 col)
        "ceo_personal_donation_flag_added": added_ceo_don,
        # Wave EFD1 (2026-05-17) — Senate STOCK Act options-disclosure 30d count (1 col)
        "senate_efd_options_disclosure_count_30d_added": added_sefd,
        "total_after_dedup_dropna": f.shape[1],
    }
    return f, mythos_fallback_rows, module_feature_counts


def build_v10_features(
    ticker: str,
    universe_agg: Optional[dict] = None,
    use_mythos: bool = False,
    start_date=None,
    end_date=None,
) -> tuple[pd.DataFrame, int, dict]:
    """Cached wrapper around _build_v10_features_impl.

    Adds a disk-based parquet cache keyed on (ticker, start_date, end_date,
    feature_set, V10_FEATURE_VERSION) via feature_cache.get_cached().

    start_date / end_date are optional and used only for cache-key purposes
    (the impl pulls all available history regardless). Pass them when the
    caller knows the date window so that different windows get separate cache
    entries.

    On a cache HIT, returns (cached_df, 0, {"from_cache": True}).
    On a cache MISS, returns the full tuple from _build_v10_features_impl.
    """
    _start = str(start_date) if start_date is not None else "all"
    _end = str(end_date) if end_date is not None else "all"

    # Mutable container to capture impl results when compute_fn is invoked
    _meta: list = []

    def _compute() -> pd.DataFrame:
        df, fallback_rows, module_counts = _build_v10_features_impl(
            ticker, universe_agg, use_mythos=use_mythos
        )
        _meta.extend([fallback_rows, module_counts])
        return df

    cached_df = get_cached(
        ticker=ticker,
        date_range=(_start, _end),
        feature_set="v10_full",
        compute_fn=_compute,
        version=V10_FEATURE_VERSION,
        ttl_days=7,
    )

    if _meta:
        # Cache miss: _compute() ran and populated real metadata
        return cached_df, _meta[0], _meta[1]
    # Cache hit: _compute() was skipped; return sentinel metadata
    return cached_df, 0, {"from_cache": True}


# ---------------------------------------------------------------------------
# CLI entry point — Helper D: --job-id restored
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "XGBoost v10 — v9 stack + daily_integration + alpaca + featuretools DFS"
        )
    )
    ap.add_argument("--ticker", required=True, help="Stock symbol e.g. AAPL")
    # Support both --output-dir (v9 style) and --out-dir (GH Actions / v8 style)
    _od = ap.add_mutually_exclusive_group(required=True)
    _od.add_argument("--output-dir", dest="output_dir", help="Directory for output files")
    _od.add_argument("--out-dir", dest="output_dir", help="Alias for --output-dir (GH Actions)")
    # --job-id: restored from v8; required by GH Actions workflow
    ap.add_argument(
        "--job-id",
        default="",
        help="CI/CD job identifier (e.g. smoke-001). Written to run_meta.json.",
    )
    ap.add_argument(
        "--use-mythos-features",
        action="store_true",
        default=False,
        help="Append 256-dim OpenMythos embeddings to the feature matrix",
    )
    ap.add_argument("--prob-threshold", type=float, default=0.50)
    ap.add_argument("--sweep-threshold", action="store_true")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--tp-atr", type=float, default=1.5)
    ap.add_argument("--sl-atr", type=float, default=1.0)
    ap.add_argument("--max-hold", type=int, default=21)
    ap.add_argument("--strategy", default="default", help="Strategy label for metadata")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(
        "[v10] Starting: ticker=%s strategy=%s job_id=%s output_dir=%s",
        args.ticker,
        args.strategy,
        args.job_id or "(none)",
        args.output_dir,
    )

    # ---- Cross-sectional cache ----
    universe_agg = None
    cache_path = WORK / "cache" / "universe_agg_manifest.json"
    if cache_path.exists() and csf is not None:
        try:
            universe_agg = csf.precompute_universe_aggregates()
        except Exception as e:
            logger.warning("  [csf] cache load failed: %s", e)

    # ---- Build v10 feature set ----
    f, mythos_fallback_rows, module_counts = build_v10_features(
        args.ticker,
        universe_agg,
        use_mythos=args.use_mythos_features,
    )
    fc = numeric_cols(f)
    logger.info(
        "  TOTAL features: %d; rows: %d; mythos_fallback_rows: %d",
        len(fc),
        len(f),
        mythos_fallback_rows,
    )
    logger.info("  Module counts: %s", module_counts)

    # Resolve checkpoint path for metadata
    mythos_checkpoint_path = os.environ.get("MYTHOS_CHECKPOINT_PATH", "")
    if not mythos_checkpoint_path:
        mythos_checkpoint_path = str(
            Path(WORK).parent / "checkpoints" / "mythos_financial_v0.pt"
        )

    # ---- Walk-forward folds ----
    folds = bml.make_walk_forward_folds(f, train_months=24, test_months=12, step_months=12)
    logger.info("  folds: %d", len(folds))
    all_probs = pd.Series(np.nan, index=f.index)
    fold_summaries = []
    fold_top_features = []
    fold_mythos_importances = []

    mythos_feat_set = set(MYTHOS_FEAT_NAMES)

    for fold in folds:
        train_end_emb = (
            pd.Timestamp(fold["train_end"])
            - pd.tseries.offsets.BDay(LABEL_EMBARGO_DAYS)
        )
        train = f[
            (f.index >= fold["train_start"]) & (f.index < train_end_emb)
        ]
        oos = f[
            (f.index >= fold["oos_start"]) & (f.index < fold["oos_end"])
        ]
        if len(train) < 50 or len(oos) < 20:
            continue

        # Scout model — get top-K features by gain
        X_tr_all = train[fc].fillna(0).values
        y_tr = train["y"].values
        X_oos_all = oos[fc].fillna(0).values

        scout = xgb.XGBClassifier(
            max_depth=3,
            learning_rate=0.05,
            n_estimators=50,
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=1,
            random_state=42,
            verbosity=0,
        )
        scout.fit(X_tr_all, y_tr)

        importances = list(zip(fc, scout.feature_importances_))
        importances.sort(key=lambda x: -x[1])
        top_features = [c for c, imp in importances[: args.top_k] if imp > 0]
        if len(top_features) < 10:
            top_features = [c for c, _ in importances[: args.top_k]]
        fold_top_features.append({"fold": fold["fold"], "top_features": top_features[:30]})

        # Track Mythos feature importances specifically
        imp_dict = dict(importances)
        mythos_imps = {
            feat: float(imp_dict.get(feat, 0.0))
            for feat in MYTHOS_FEAT_NAMES
            if feat in fc and imp_dict.get(feat, 0.0) > 0
        }
        top_mythos = sorted(mythos_imps.items(), key=lambda x: -x[1])[:10]
        fold_mythos_importances.append(
            {
                "fold": fold["fold"],
                "mythos_in_top50": sum(
                    1 for feat in top_features if feat in mythos_feat_set
                ),
                "top_mythos": dict(top_mythos),
            }
        )

        # Final model on pruned features
        X_tr = train[top_features].fillna(0).values
        X_oos = oos[top_features].fillna(0).values
        final = xgb.XGBClassifier(
            max_depth=4,
            learning_rate=0.05,
            n_estimators=100,
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=1,
            random_state=42,
            verbosity=0,
        )
        final.fit(X_tr, y_tr)
        probs = final.predict_proba(X_oos)[:, 1]
        all_probs.loc[oos.index] = probs
        fold_summaries.append(
            {
                "fold": fold["fold"],
                "n_train": len(train),
                "n_oos": len(oos),
                "n_top_features": len(top_features),
                "mean_oos_prob": float(probs.mean()),
            }
        )

    # ---- Threshold sweep or fixed ----
    sdf = None
    if args.sweep_threshold:
        rows = []
        for thr in np.arange(0.46, 0.70, 0.02):
            sig = all_probs > thr
            trades = bml.simulate(
                f, sig.fillna(False), args.tp_atr, args.sl_atr, args.max_hold
            )
            mm = bml.compute_metrics(trades)
            rows.append({"thr": round(thr, 2), **mm})
        sdf = pd.DataFrame(rows)
        sdf.to_csv(f"{args.output_dir}/threshold_sweep.csv", index=False)
        mask = (
            (sdf["profit_factor"] >= 1.5)
            & (sdf["win_rate"] >= 0.53)
            & (sdf["n_trades"] >= 8)
            & (sdf["max_drawdown_pct"] >= -0.03)
            & (sdf["total_return_pct"] > 0)
        )
        if mask.any():
            chosen_thr = float(
                sdf[mask]
                .sort_values("profit_factor", ascending=False)
                .iloc[0]["thr"]
            )
        else:
            chosen_thr = float(
                sdf.sort_values("profit_factor", ascending=False).iloc[0]["thr"]
            )
        logger.info("  -> chose thr=%.2f", chosen_thr)
    else:
        chosen_thr = args.prob_threshold

    # ---- Final simulation ----
    final_sig = (all_probs > chosen_thr).fillna(False)
    trades = bml.simulate(f, final_sig, args.tp_atr, args.sl_atr, args.max_hold)
    metrics = bml.compute_metrics(trades)
    logger.info(
        "  FINAL thr=%.2f: n=%d, WR=%.3f, PF=%.3f, RET=%.4f, DD=%.4f",
        chosen_thr,
        metrics["n_trades"],
        metrics.get("win_rate", 0),
        metrics.get("profit_factor", 0),
        metrics.get("total_return_pct", 0),
        metrics.get("max_drawdown_pct", 0),
    )

    trades.to_csv(f"{args.output_dir}/trades.csv", index=False)
    if sdf is not None:
        sdf.to_csv(f"{args.output_dir}/threshold_sweep.csv", index=False)

    # ---- Serialization helper ----
    def to_py(obj):
        if isinstance(obj, dict):
            return {k: to_py(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_py(v) for v in obj]
        if hasattr(obj, "item"):
            return obj.item()
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        return obj

    # ---- run_meta.json ----
    meta = to_py(
        {
            "ticker": args.ticker,
            "job_id": args.job_id,
            "pipeline_version": "xgb_v10",
            "strategy_variant": "ML_XGB_v10",
            "strategy": args.strategy,
            "run_at": datetime.utcnow().isoformat() + "Z",
            "features_total": len(fc),
            "top_k": args.top_k,
            "rows": len(f),
            # v10 module availability
            "v10_modules": {
                "daily_integration_features": DAILY_INT_AVAILABLE,
                "alpaca_features": ALPACA_AVAILABLE,
                "featuretools_dfs_features": DFS_AVAILABLE,
                # Wave A (2026-05-17)
                "options_flow_features": OPTIONS_FLOW_AVAILABLE,
                "govtrades_features": GOVTRADES_AVAILABLE,
                "time_of_day_features": TOD_AVAILABLE,
                "gabriel_priors_features": GABRIEL_PRIORS_AVAILABLE,
            },
            # per-module feature counts
            "module_feature_counts": module_counts,
            # Mythos-specific metadata (inherited from v9)
            "use_mythos_features": args.use_mythos_features,
            "mythos_checkpoint_path": mythos_checkpoint_path,
            "mythos_feature_count": MYTHOS_FEATURE_DIM if args.use_mythos_features else 0,
            "mythos_fallback_rows": mythos_fallback_rows,
            "mythos_fallback_pct": (
                round(mythos_fallback_rows / max(len(f), 1), 4)
                if args.use_mythos_features
                else None
            ),
            "feature_sources": {
                "base+intraday+alt+insight_v3+parts1-4": "~587",
                "cross_sectional": "17 (if cache)",
                "macro_yfinance": "40",
                "strategy_signal+five_filter": "~25",
                "google_trends": "7",
                "insider_form4": "8",
                "multi_timeframe_h1_h4_m5_m15": "15 (v7)",
                "news_sentiment_vader": "8 (v7)",
                "vol_estimators": "14 (v7)",
                "qlib_alpha158_pandas_port": "158 (v8)",
                "closeable_gaps_yfinance_finra": "18 (v8 patch)",
                "openmythos_256dim_embedding": (
                    "256 (v9)" if args.use_mythos_features else "disabled"
                ),
                "alpaca_yfinance_earnings_div_splits": "13 (v10 NEW)",
                "daily_integration_beta_residual_csrs_earn_gate": "7 (v10 NEW)",
                "featuretools_dfs_depth2_interactions": "~60 (v10 NEW)",
                "options_flow_pc_ratio_iv_unusual": "3 (v10.4 Wave A)",
                "govtrades_congress_density_buy_sell_lobby": "3 (v10.4 Wave A)",
                "time_of_day_bucket": "1 (v10.4 Wave A)",
                "gabriel_priors_pf_wr_n_regime_consistency": "5 (v10.4 Wave A)",
            },
            "walk_forward_folds": len(fold_summaries),
            "strategy_params": {
                "name": "ML_XGB_v10",
                "side": "long",
                "tp_atr": args.tp_atr,
                "sl_atr": args.sl_atr,
                "max_hold_days": args.max_hold,
                "prob_threshold": chosen_thr,
                "threshold_swept": args.sweep_threshold,
                "model": "XGBClassifier (scout-prune-refit top-K)",
                "slippage_bps": 5.0,
                "fee_per_share": 0.0035,
                "notional_per_trade": 5000,
            },
            "metrics_oos_aggregate": metrics,
            "fold_summaries": fold_summaries,
            "fold_top_features": fold_top_features,
            "fold_mythos_importances": fold_mythos_importances,
        }
    )

    meta_path = f"{args.output_dir}/run_meta.json"
    with open(meta_path, "w") as fp:
        json.dump(meta, fp, indent=2, default=str)
    logger.info("  Wrote %s", meta_path)

    # Write lightweight result.json (smoke-test contract)
    result = {
        "ticker": args.ticker,
        "job_id": args.job_id,
        "pipeline_version": "xgb_v10",
        "features_total": len(fc),
        "rows": len(f),
        "module_feature_counts": module_counts,
        "n_trades": metrics["n_trades"],
        "win_rate": metrics.get("win_rate"),
        "profit_factor": metrics.get("profit_factor"),
        "total_return_pct": metrics.get("total_return_pct"),
        "max_drawdown_pct": metrics.get("max_drawdown_pct"),
        "status": "ok",
    }
    result_path = f"{args.output_dir}/result.json"
    with open(result_path, "w") as fp:
        json.dump(to_py(result), fp, indent=2, default=str)
    logger.info("  Wrote %s", result_path)
    logger.info("[v10] DONE. features=%d rows=%d", len(fc), len(f))


if __name__ == "__main__":
    main()
