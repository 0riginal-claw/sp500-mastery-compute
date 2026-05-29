"""hypothesis_runner.py — 6-step validation pipeline on a STRATEGY HYPOTHESIS, not a
single indicator.

Locked methodology (2026-05-29): the unit of validation is a multi-indicator hypothesis dict
with explicit roles: regime_gate × bias_filter × trigger × confirmation × timing × exit ×
no_trade × cost × universe × timeframe. See lab.knowledge.indicators.methodology_principle().

This runner accepts such a hypothesis dict, walks the per-bar state machine, computes a returns
series, then runs the same six diagnostics as indicator_hardening_runner (walk-forward, PBO via
CSCV, DSR, parameter stability via ±10% perturbation, final holdout). The PBO/DSR universe is
the hypothesis's PARAMETER perturbations, not 11 separate indicators.

Public API:
  run_hypothesis(hypothesis: dict, holdout_after: str = "2025-01-01") -> dict
  evaluate_hypothesis(bars, hypothesis) -> np.ndarray  # signal per bar in {-1,0,+1}

The role parser is intentionally minimal — see `_RoleParser` docstring for the grammar.

Alt-data overlay (added 2026-05-28, task #40): role expressions may include alt-data event-
window tokens (e.g. ``InsiderForm4_LT5d``, ``CongressBuy_LT30d``, ``8K_LT5d``, ``DPI_GT_P90``,
``DarkPoolZ_LT_neg1p5``, ``NewsEvent_LT3d``, ``NewsCount_GT_P75_30d``). These resolve through
``_AltDataResolver`` (this file) against ``lab.knowledge.{edgar, govtrades, news}``. All
resolutions strictly enforce ``event_timestamp <= bar_timestamp`` (no look-ahead). The resolver
caches one fetch per (ticker, source) and one resolved series per (ticker, token).
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))  # for knowledge.* package

from indicator_compute import (  # noqa: E402
    _atr, _ema, _rolling_max, _rolling_min, _rolling_std, _rsi, _sma,
    bollinger, cci, connors_rsi, fisher_transform, keltner, macd, mfi, obv,
    stochastic, supertrend, williams_r,
)
from indicator_pbo_dsr import (  # noqa: E402
    cscv_pbo, deflated_sharpe, rolling_walkforward_folds, walk_forward_efficiency,
)
from knowledge.indicators import validate_test_unit  # noqa: E402

# Reuse loader from sibling runner for OHLC fetch and timeframe state
import indicator_hardening_runner as _ihr  # noqa: E402

ArrayDict = Dict[str, np.ndarray]

COST_PER_SIDE = 5e-4  # 5 bps per side; matches indicator_hardening_runner

# Persisted output roots (mirror of indicator_hardening_runner)
RESULTS_LOCAL = Path("/Volumes/ZG-2TB/zg/hyp_validate/results")
DRIVE_RESULTS = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools/s&p500-ticker-mastery/data/hypothesis_validation"
)

# ============================================================================
# Multi-timeframe data loading (added 2026-05-29, task #53).
#
# Mission 12 seeds (PURE_TECH, ORB_MORNING, VWAP_MTF, GOV_AWARE, HYBRID_REGIME)
# were originally designed as multi-timeframe (1D thesis + 15min structure +
# 5min entry, etc.) but the task #41 dispatch flattened them to single-TF. This
# section wires the multi-TF data path:
#
#   * Local alpaca_5yr cache at /Volumes/ZG-2TB/zg/cache/alpaca_5yr/{1Day,5Min}
#     (502 tickers, ~100MB each, fast — pyarrow read in ~1s vs FUSE >60s).
#   * 15min and 1H are RESAMPLED from 5min on-the-fly (cached per call).
#     The alpaca Drive tree DOES contain 15Min/1Hour parquet, but FUSE under
#     load (load_avg 21+ at refactor time) reads them in >60s — resampling
#     from local 5min costs <100ms.
#   * 1d uses the existing yfinance_daily_5yr cache (already wired via
#     ``indicator_hardening_runner.DRIVE_OHLC_DAILY``).
#
# Honest gap (documented in the GOV_AWARE/ORB_MORNING data_sources field):
# 1Min data is NOT used in this pass. The 1Min Drive tree is huge and FUSE-
# blind under load; resampling from 5min gives no extra signal. The ORB
# trigger therefore uses the first 5min bar of the session as a proxy for
# the "1min break of opening range" trigger. When 1Min is needed (e.g. live
# trading), the loader can be extended to read 1Min directly — left as
# future work.
#
# Alignment policy (no lookahead, MANDATORY):
#   higher-TF value on a lower-TF bar at timestamp T uses ONLY the higher-TF
#   bar whose close timestamp t_higher <= T - 1_higher_bar. This means
#   "yesterday's daily close on today's 5min bars" — today's daily bar is
#   still incomplete during today's session and using it would be lookahead.
# ============================================================================

# Local OHLC cache roots (fast SSD, populated by prior backfills).
_ALPACA_5YR_LOCAL = Path("/Volumes/ZG-2TB/zg/cache/alpaca_5yr")  # has 1Day + 5Min
# Drive minute/hour fallback (slow on FUSE; only used when local is missing).
_ALPACA_5YR_DRIVE = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/version_3 - Gabriel/Gabriel_Alpaca TimeFrames"
)
# Canonical TF aliases — lowercase keys map to a single normalized name.
_TF_ALIASES = {
    "1d": "1d", "1day": "1d", "day": "1d", "d": "1d", "daily": "1d",
    "5min": "5min", "5m": "5min", "5": "5min",
    "15min": "15min", "15m": "15min",
    "1h": "1h", "1hour": "1h", "h": "1h", "60min": "1h",
    "1min": "1min", "1m": "1min",  # not loaded in this pass — see module docstring
}

# Resample rules for downsampling from 5min.
_RESAMPLE_FROM_5MIN = {
    "15min": "15min",   # pandas resample rule
    "1h": "1h",
}


def _normalize_tf(tf: str) -> str:
    """Normalize a timeframe string. Raises if unknown."""
    if tf is None:
        return "1d"
    k = str(tf).strip().lower()
    if k not in _TF_ALIASES:
        raise ValueError(f"unknown timeframe {tf!r}; supported: {sorted(set(_TF_ALIASES.values()))}")
    return _TF_ALIASES[k]


def _load_alpaca_5min_concat(ticker: str) -> Optional["pd.DataFrame"]:
    """Concat all monthly 5min parquet files for a ticker into one DataFrame.

    Returns a DataFrame with columns [timestamp(tz-naive UTC), open, high, low,
    close, volume]. Schema mirrors the alpaca alpaca_5yr cache (timestamp tz-aware
    UTC — stripped to tz-naive for downstream join-friendliness).

    Returns None if no parquets exist.
    """
    base_local = _ALPACA_5YR_LOCAL / "5Min" / ticker.upper()
    base_drive = _ALPACA_5YR_DRIVE / "Minutes TimeFrames" / "5Min" / ticker.upper()
    base = base_local if base_local.is_dir() else base_drive
    if not base.is_dir():
        return None
    parquets = sorted(base.glob("*.parquet"))
    if not parquets:
        return None
    frames = []
    for p in parquets:
        try:
            frames.append(pd.read_parquet(p))
        except OSError:
            continue
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    # Schema: timestamp may be tz-aware UTC or in an index.
    if "timestamp" not in df.columns:
        # Some parquets place ts in the index — recover.
        if df.index.name == "timestamp":
            df = df.reset_index()
        else:
            return None
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df["timestamp"] = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
    # Restrict to RTH-only window (14:30-21:00 UTC ≈ 9:30-16:00 ET) to keep
    # cross-TF joins meaningful — pre/post-market vol is light and confuses VWAP.
    # Note: this also drops the 09:30 first bar in DST shoulder months; we keep
    # it intentionally because the ORB proxy uses session open.
    h = df["timestamp"].dt.hour
    m = df["timestamp"].dt.minute
    in_rth = ((h > 14) | ((h == 14) & (m >= 30))) & (h < 21)
    df = df.loc[in_rth].reset_index(drop=True)
    needed = {"open", "high", "low", "close", "volume"}
    if not needed.issubset(df.columns):
        return None
    df = df.dropna(subset=list(needed))
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def _resample_ohlcv_from_5min(df_5min: "pd.DataFrame", rule: str) -> "pd.DataFrame":
    """Resample a 5min OHLCV DataFrame to a coarser TF.

    rule: pandas resample rule, e.g. '15min' or '1h'.

    OHLC aggregation: open=first, high=max, low=min, close=last, volume=sum.
    Returns a DataFrame with the same column layout as the 5min source.
    """
    if df_5min is None or df_5min.empty:
        return df_5min
    g = df_5min.set_index("timestamp")
    out = g.resample(rule, label="right", closed="right").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna(subset=["close"]).reset_index()
    return out


def _load_bars_by_tf(ticker: str, tfs: List[str]) -> Tuple[
    Dict[str, ArrayDict], Dict[str, "pd.DatetimeIndex"], Dict[str, str]
]:
    """Load OHLCV bars for each timeframe in ``tfs``.

    Returns:
      bars_by_tf: {tf_canonical: {open, high, low, close, volume} as np.ndarray}
      ts_by_tf:   {tf_canonical: pd.DatetimeIndex of bar timestamps (tz-naive UTC)}
      notes:      {tf_canonical: human-readable provenance note}

    Tfs not loaded (e.g. 1min which is currently skipped on this Mac) are simply
    omitted from the returned dicts — callers must check membership before use.
    """
    bars_by_tf: Dict[str, ArrayDict] = {}
    ts_by_tf: Dict[str, "pd.DatetimeIndex"] = {}
    notes: Dict[str, str] = {}
    # Canonicalize + dedupe
    tfs_norm = []
    for t in tfs:
        try:
            n = _normalize_tf(t)
        except ValueError:
            continue
        if n not in tfs_norm:
            tfs_norm.append(n)

    # Always load 5min first if any sub-day TF is requested (needed for resample).
    needs_5min = any(t in ("5min", "15min", "1h") for t in tfs_norm)
    df_5min = None
    if needs_5min:
        df_5min = _load_alpaca_5min_concat(ticker)
        if df_5min is not None and len(df_5min) >= 200:
            notes["5min"] = f"alpaca_5yr local 5Min cache ({len(df_5min)} bars)"
        else:
            df_5min = None  # treat too-small as missing

    for tf in tfs_norm:
        if tf == "1d":
            # Use existing daily loader from indicator_hardening_runner; it has
            # the standard yfinance_daily_5yr cache layout. Also need timestamps —
            # _load_bar_timestamps does that lazily.
            saved_tf = _ihr._state["timeframe"]
            try:
                _ihr.set_timeframe("1d")
                bars = _ihr.load_ohlc(ticker)
                ts = _load_bar_timestamps(ticker)
            finally:
                _ihr.set_timeframe(saved_tf)
            if bars is not None and ts is not None and len(ts) == len(bars["close"]):
                bars_by_tf[tf] = bars
                ts_by_tf[tf] = pd.DatetimeIndex(pd.to_datetime(ts).tz_localize(None) if getattr(ts, "tz", None) else pd.to_datetime(ts))
                notes[tf] = f"yfinance_daily_5yr cache ({len(bars['close'])} bars)"
            continue
        if tf == "5min":
            if df_5min is None:
                continue
            bars_by_tf[tf] = {k: df_5min[k].to_numpy(dtype=np.float64)
                              for k in ("open", "high", "low", "close", "volume")}
            ts_by_tf[tf] = pd.DatetimeIndex(df_5min["timestamp"].to_numpy())
            continue
        if tf in ("15min", "1h"):
            if df_5min is None:
                continue
            rule = _RESAMPLE_FROM_5MIN[tf]
            df_r = _resample_ohlcv_from_5min(df_5min, rule)
            if df_r is None or len(df_r) < 100:
                continue
            bars_by_tf[tf] = {k: df_r[k].to_numpy(dtype=np.float64)
                              for k in ("open", "high", "low", "close", "volume")}
            ts_by_tf[tf] = pd.DatetimeIndex(df_r["timestamp"].to_numpy())
            notes[tf] = f"resampled from 5min ({len(df_r)} bars, rule={rule})"
            continue
        # 1min not loaded in this pass (documented in module docstring).
        notes[tf] = "NOT LOADED — see module docstring §multi-timeframe"
    return bars_by_tf, ts_by_tf, notes


def _align_higher_tf_to_lower(
    values_high: np.ndarray,
    ts_high: "pd.DatetimeIndex",
    ts_low: "pd.DatetimeIndex",
) -> np.ndarray:
    """Forward-fill a higher-TF series onto a lower-TF timestamp grid.

    MANDATORY no-lookahead: the value at low-TF bar t_low is the higher-TF value
    from the most recent higher-TF bar whose close time is STRICTLY BEFORE t_low.
    For daily-on-5min: the 1d bar with date == 2026-05-28 has close_ts treated
    as end-of-day; on the next 5min bar (e.g. 2026-05-29 14:30) the alignment
    correctly picks up the 2026-05-28 value. On 2026-05-28 14:30 the alignment
    picks up 2026-05-27 (yesterday).

    Implementation: searchsorted with side='left' then subtract 1 gives the
    "last bar strictly before". Values before the first higher-TF bar are NaN.
    """
    ts_high_arr = ts_high.to_numpy()
    ts_low_arr = ts_low.to_numpy()
    # Search left so values_high[idx-1] is the most recent bar STRICTLY before
    # the lower-TF timestamp. For daily bars whose timestamp is conventionally
    # midnight or end-of-day, this is the right call: on a 5min bar at
    # 2026-05-29 14:30, the 2026-05-29 daily bar (if it exists at all in the
    # cache) is incomplete and would be lookahead. We always look one back.
    idx = np.searchsorted(ts_high_arr, ts_low_arr, side="left") - 1
    out = np.full(len(ts_low_arr), np.nan, dtype=np.float64)
    valid = idx >= 0
    out[valid] = values_high[idx[valid]]
    return out


# Cross-TF prefix regex. Matches `<TF>.<TOKEN>` at IDENT boundaries; TF is one
# of the canonical aliases. The replacement is a TF-suffixed bareword which the
# parser dispatches via the multi-TF indicator table.
#
# Examples of accepted prefixes:
#   1d.ADX(14)         → __TF1D_ADX(14)
#   1D.Close           → __TF1D_Close
#   5min.VWAP          → __TF5MIN_VWAP
#   15MIN.Donchian_UP(20) → __TF15MIN_Donchian_UP(20)
#   1H.RSI(14)         → __TF1H_RSI(14)
#
# Tokens already starting with __TF (already rewritten) are left alone.
_TF_PREFIX_REGEX = re.compile(
    r"(?<![A-Za-z0-9_])(1d|1D|5min|5MIN|5Min|15min|15MIN|15Min|1h|1H|1Hour|1Min|1min)\.([A-Za-z_][A-Za-z0-9_]*)",
)


def _rewrite_tf_prefixes(expr: str) -> str:
    """Pre-process a role expression: rewrite ``<TF>.<TOKEN>`` to ``__TF<NORM>_<TOKEN>``.

    Backward compat: expressions without TF prefixes pass through unchanged.
    """
    if not isinstance(expr, str) or "." not in expr:
        return expr

    def repl(m):
        tf_raw, tok = m.group(1), m.group(2)
        try:
            tf_norm = _normalize_tf(tf_raw)
        except ValueError:
            return m.group(0)
        # Canonical uppercase suffix for the rewritten ident
        tf_suffix = tf_norm.upper().replace("MIN", "MIN").replace("H", "H")
        return f"__TF{tf_suffix}_{tok}"

    return _TF_PREFIX_REGEX.sub(repl, expr)


# ============================================================================
# Role expression parser  --  grammar (informal):
#
#   expr      := or_expr
#   or_expr   := and_expr ( "OR" and_expr )*
#   and_expr  := cmp_expr ( "AND" cmp_expr )*
#   cmp_expr  := add_expr ( ( ">" | "<" | ">=" | "<=" | "==" | "!=" ) add_expr )?
#   add_expr  := mul_expr ( ( "+" | "-" ) mul_expr )*
#   mul_expr  := unary    ( ( "*" | "x" | "×" | "/" ) unary )*
#   unary     := ( "-" | "NOT" ) unary | atom
#   atom      := NUMBER | indicator_call | bareword | "(" expr ")"
#   indicator_call := IDENT ( "(" args ")" )? ( "." attr )?
#
# Supported indicator tokens (case-sensitive on first letter except for the keywords below):
#   Close, High, Low, Open, Volume        — raw OHLCV
#   SMA(n), EMA(n), ATR(n), RSI(n)        — value series
#   ADX(n)                                 — Wilder's ADX
#   Donchian_UP(n), Donchian_DN(n)         — channel highs/lows (close vs prior n-bar extreme)
#   BB.upper(n,k), BB.lower(n,k), BB.mid(n,k), BB.pctb(n,k)
#   VWAP                                  — running vwap
#   ChopIdx(n) / ChoppinessIndex(n)
#   VolumeExpansion(n,mult)               — boolean: v > mult * SMA(volume, n)
#   StochK(n,d,s), StochD(n,d,s)
#   Williams(n)                           — Williams %R
#   CCI(n), MFI(n), CMF(n)
#   ConnorsRSI(p1,p2,p3)
#   Supertrend(n,mult)                    — returns ±1 trend
#
# Keywords (case-insensitive): AND, OR, NOT, TRUE, FALSE
# Comparisons return float arrays of 0.0/1.0 so they can compose.
# Boolean context: any non-zero value is truthy.
# ============================================================================


# NB: ALT_IDENT comes before NUMBER so that ``8K_LT5d`` tokenizes as one IDENT instead of
# NUMBER(8) + IDENT(K_LT5d). It must NOT match plain numeric literals — anchored with a
# trailing alpha char (e.g. ``8K``) to disambiguate.
_TOK_REGEX = re.compile(
    r"\s*(?:"
    r"(?P<ALT_IDENT>\d+[A-Za-z][A-Za-z0-9_]*)"
    r"|(?P<NUMBER>\d+(?:\.\d+)?)"
    r"|(?P<OP>>=|<=|==|!=|>|<|&&|\|\||AND|OR|NOT|and|or|not)"
    r"|(?P<IDENT>[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?P<DOT>\.)"
    r"|(?P<LP>\()"
    r"|(?P<RP>\))"
    r"|(?P<COMMA>,)"
    r"|(?P<PLUS>\+)"
    r"|(?P<MINUS>-)"
    r"|(?P<MUL>[*×x])"
    r"|(?P<DIV>/)"
    r")"
)


@dataclass
class _Tok:
    kind: str
    val: str


def _tokenize(expr: str) -> List[_Tok]:
    out: List[_Tok] = []
    i = 0
    s = expr.strip()
    while i < len(s):
        m = _TOK_REGEX.match(s, i)
        if not m:
            raise ValueError(f"role parser: cannot tokenize at: {s[i:i+24]!r}")
        i = m.end()
        for k, v in m.groupdict().items():
            if v is not None:
                # Normalize OP tokens
                if k == "OP":
                    up = v.upper().replace("&&", "AND").replace("||", "OR")
                    out.append(_Tok("OP", up))
                elif k == "ALT_IDENT":
                    # Alt-data tokens that start with a digit (e.g. ``8K_LT5d``) are
                    # lexed as ALT_IDENT but should flow through the IDENT path so the
                    # parser routes them to the alt-data resolver.
                    out.append(_Tok("IDENT", v))
                else:
                    out.append(_Tok(k, v))
                break
    return out


def _bcast(x: np.ndarray, n: int) -> np.ndarray:
    """Broadcast scalar → length-n array of float64."""
    if isinstance(x, np.ndarray) and x.shape == (n,):
        return x
    if np.isscalar(x):
        return np.full(n, float(x), dtype=np.float64)
    raise ValueError(f"cannot broadcast {type(x)} to length {n}")


def _wilder_adx(bars: ArrayDict, period: int = 14) -> np.ndarray:
    """Wilder's ADX. Returns a float series, NaN until enough data."""
    h, l, c = bars["high"], bars["low"], bars["close"]
    n = len(c)
    up = np.zeros(n)
    dn = np.zeros(n)
    up[1:] = h[1:] - h[:-1]
    dn[1:] = l[:-1] - l[1:]
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    from indicator_compute import _true_range, _wilder
    tr = _true_range(bars)
    atr = _wilder(tr, period)
    pdi = 100.0 * _wilder(plus_dm, period) / np.where(atr > 0, atr, 1.0)
    mdi = 100.0 * _wilder(minus_dm, period) / np.where(atr > 0, atr, 1.0)
    dx = 100.0 * np.abs(pdi - mdi) / np.where((pdi + mdi) > 0, pdi + mdi, 1.0)
    adx = _wilder(dx, period)
    adx[:period * 2] = np.nan
    return adx


def _chop_index(bars: ArrayDict, period: int = 14) -> np.ndarray:
    """Choppiness Index (Dreiss). 0..100, higher = choppy (range), lower = trending."""
    from indicator_compute import _true_range
    h, l = bars["high"], bars["low"]
    n = len(h)
    tr = _true_range(bars)
    atr_sum = np.full(n, np.nan)
    hh = _rolling_max(h, period)
    ll = _rolling_min(l, period)
    csum = np.cumsum(np.insert(tr, 0, 0.0))
    if n >= period:
        atr_sum[period - 1:] = csum[period:] - csum[:-period]
    out = np.full(n, np.nan)
    denom = hh - ll
    log_p = np.log10(period)
    mask = (denom > 0) & (atr_sum > 0) & ~np.isnan(atr_sum)
    out[mask] = 100.0 * np.log10(atr_sum[mask] / denom[mask]) / log_p
    return out


def _running_vwap(bars: ArrayDict) -> np.ndarray:
    """Cumulative VWAP from start of series (no session reset — adequate for daily and for 5min over short windows)."""
    h, l, c, v = bars["high"], bars["low"], bars["close"], bars["volume"]
    tp = (h + l + c) / 3.0
    cum_pv = np.cumsum(tp * v)
    cum_v = np.cumsum(v)
    out = np.full_like(c, np.nan, dtype=np.float64)
    mask = cum_v > 0
    out[mask] = cum_pv[mask] / cum_v[mask]
    return out


def _donchian_up(bars: ArrayDict, period: int = 20) -> np.ndarray:
    """Returns prior n-bar HH (so close > this == breakout)."""
    h = bars["high"]
    hh = _rolling_max(h, period)
    # Shift by 1 to compare against prior-bar window
    out = np.full_like(hh, np.nan)
    out[1:] = hh[:-1]
    return out


def _donchian_dn(bars: ArrayDict, period: int = 20) -> np.ndarray:
    l = bars["low"]
    ll = _rolling_min(l, period)
    out = np.full_like(ll, np.nan)
    out[1:] = ll[:-1]
    return out


def _cmf(bars: ArrayDict, period: int = 21) -> np.ndarray:
    """Chaikin Money Flow."""
    h, l, c, v = bars["high"], bars["low"], bars["close"], bars["volume"]
    denom = h - l
    mfm = np.where(denom > 0, ((c - l) - (h - c)) / np.where(denom > 0, denom, 1.0), 0.0)
    mfv = mfm * v
    num = _sma(mfv, period) * period
    den = _sma(v, period) * period
    out = np.full_like(c, np.nan)
    mask = den > 0
    out[mask] = num[mask] / den[mask]
    return out


# ============================================================================
# _AltDataResolver — wires lab.knowledge.{edgar, govtrades, news} into the role
# parser as per-bar boolean event-window flags.
#
# Strict no-lookahead discipline (timestamp column per source):
#   Form 4         → use ``accepted_at`` if present in row, else ``filed_at``
#                    (~2-business-day disclosure lag is baked into filed_at)
#   Congress trades → use ``report_date`` (disclosure date, typically 30-45d
#                    after transaction). Falls back to transaction_date +
#                    LOOKBACK_CONGRESS_DISCLOSURE_DAYS (=45) if report_date NaT.
#   8-K / 10-K /
#   DEF 14A / S-1  → use ``filed_at``
#   Off-exchange   → ``date`` is the as-of trading date (T+0 publication). DPI
#                    Z-score for bar at T uses rows with date < T (strictly
#                    prior, so the rolling stats avoid the bar's own value).
#   News           → ``published_utc`` (already the live publication ts).
#
# Token grammar (case-insensitive):
#   InsiderForm4_LT5d                      Form 4 buys, any, in 5 trading days
#   InsiderForm4_LT5d_GT1M                 Form 4 buys total > $1M in 5 trading days
#   InsiderClusterBuy_5d_2plus             ≥2 distinct insider buyers in 5 trading days
#   CongressBuy_LT30d                      Congress Buy disclosed in 30 calendar days
#   8K_LT5d                                8-K filing in 5 trading days
#   DPI_GT_P90 / DPI_LT_P10                DPI rolling Z-score (5d window) > 90th /< 10th pct
#   DarkPoolZ_LT_neg1p5 / DarkPoolZ_GT_1p5 DPI Z-score thresholds (5d rolling)
#   NewsEvent_LT3d                         Any news article in 3 calendar days
#   NewsCount_GT_P75_30d                   News count over 30d > 75th pct of 1y rolling
# ============================================================================


# 2TB tier paths used when the canonical knowledge wrappers throw FileNotFoundError
# (the lab.knowledge.govtrades wrapper points at /My Drive/Ph0tis/... which is missing).
_ALT_DATA_GOVTRADES_DB_CANDIDATES = (
    "/Volumes/ZG-2TB/zg/govtrades/data/govtrades.db",
    "/Users/orginal/.zg/govtrades/data/govtrades.db",
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/"
    "My Drive/Ph0tis/Gov-Trades/data/govtrades.db",
)


def _resolve_govtrades_db_path() -> Optional[str]:
    """Multi-tier path resolver mirroring news._db_path()."""
    import os as _os
    env = _os.environ.get("GOVTRADES_DB")
    if env and _os.path.exists(env):
        return env
    for p in _ALT_DATA_GOVTRADES_DB_CANDIDATES:
        if _os.path.exists(p):
            return p
    return None


# Token parsing helpers
_TOKEN_FORM4         = re.compile(r"^InsiderForm4_LT(\d+)d(?:_GT(\d+)([MK]))?$", re.IGNORECASE)
_TOKEN_FORM4_CLUSTER = re.compile(r"^InsiderClusterBuy_(\d+)d_(\d+)plus$", re.IGNORECASE)
_TOKEN_CONGRESS      = re.compile(r"^CongressBuy_LT(\d+)d$", re.IGNORECASE)
_TOKEN_8K            = re.compile(r"^8K_LT(\d+)d$", re.IGNORECASE)
_TOKEN_DPI_PCT       = re.compile(r"^DPI_(GT|LT)_P(\d+)$", re.IGNORECASE)
_TOKEN_DARKPOOLZ     = re.compile(r"^DarkPoolZ_(LT|GT)_(neg)?(\d+)(?:p(\d+))?$", re.IGNORECASE)
_TOKEN_NEWS_EVENT    = re.compile(r"^NewsEvent_LT(\d+)d$", re.IGNORECASE)
_TOKEN_NEWS_COUNT    = re.compile(r"^NewsCount_GT_P(\d+)_(\d+)d$", re.IGNORECASE)


def _looks_like_alt_data_token(s: str) -> bool:
    """Cheap pre-filter so the role parser only goes to the resolver for plausible tokens."""
    if not isinstance(s, str) or not s:
        return False
    return any(p.match(s) for p in (
        _TOKEN_FORM4, _TOKEN_FORM4_CLUSTER, _TOKEN_CONGRESS, _TOKEN_8K,
        _TOKEN_DPI_PCT, _TOKEN_DARKPOOLZ, _TOKEN_NEWS_EVENT, _TOKEN_NEWS_COUNT,
    ))


class _AltDataResolver:
    """Per-bar resolver for alt-data event-window tokens.

    Args:
        ticker:        symbol the resolver answers for.
        bar_timestamps: np.ndarray of pd.Timestamp (one per bar). MUST be tz-naive UTC-equivalent.
        lookback_pad_days: extra calendar days fetched before the first bar to satisfy long
            lookback windows (90 for DPI percentile, 365 for news count percentile).

    Methods:
        resolve(token) -> np.ndarray[bool] of length len(bar_timestamps).
        diagnostics()  -> dict explaining what data the resolver found per source.
    """

    def __init__(self, ticker: str, bar_timestamps: "pd.DatetimeIndex",
                 lookback_pad_days: int = 400):
        self.ticker = ticker.upper()
        self.bar_ts = pd.to_datetime(pd.Series(bar_timestamps)).reset_index(drop=True)
        self.n = len(self.bar_ts)
        self.lookback_pad_days = int(lookback_pad_days)
        self._series_cache: Dict[str, np.ndarray] = {}
        self._source_cache: Dict[str, Any] = {}
        self._diagnostics: Dict[str, Any] = {"ticker": self.ticker, "n_bars": self.n}

    # ---- public ----
    def resolve(self, token: str) -> np.ndarray:
        """Return a length-n bool array for the token. Unknown tokens raise ValueError.

        Caches per (resolver_instance, token).
        """
        if token in self._series_cache:
            return self._series_cache[token]
        try:
            arr = self._dispatch(token)
        except _AltDataUnknownToken:
            raise
        except Exception as e:  # pragma: no cover - data infra is brittle
            self._diagnostics.setdefault("errors", []).append({"token": token, "err": str(e)})
            arr = np.zeros(self.n, dtype=bool)
        if arr.dtype != bool:
            arr = arr.astype(bool)
        if arr.shape != (self.n,):
            raise ValueError(
                f"_AltDataResolver: token {token!r} returned shape {arr.shape}, expected ({self.n},)"
            )
        self._series_cache[token] = arr
        return arr

    def diagnostics(self) -> Dict[str, Any]:
        return dict(self._diagnostics)

    # ---- dispatch ----
    def _dispatch(self, token: str) -> np.ndarray:
        m = _TOKEN_FORM4.match(token)
        if m:
            window = int(m.group(1))
            thresh_amt = m.group(2)
            thresh_unit = m.group(3)
            min_value = None
            if thresh_amt and thresh_unit:
                mult = 1_000_000 if thresh_unit.upper() == "M" else 1_000
                min_value = int(thresh_amt) * mult
            return self._resolve_form4(window_trading_days=window, min_value_usd=min_value)
        m = _TOKEN_FORM4_CLUSTER.match(token)
        if m:
            return self._resolve_form4_cluster(
                window_trading_days=int(m.group(1)),
                min_distinct_buyers=int(m.group(2)),
            )
        m = _TOKEN_CONGRESS.match(token)
        if m:
            return self._resolve_congress_buy(window_cal_days=int(m.group(1)))
        m = _TOKEN_8K.match(token)
        if m:
            return self._resolve_8k(window_trading_days=int(m.group(1)))
        m = _TOKEN_DPI_PCT.match(token)
        if m:
            side, pct = m.group(1).upper(), int(m.group(2))
            return self._resolve_dpi_percentile(side=side, percentile=pct,
                                                window_trading_days=5)
        m = _TOKEN_DARKPOOLZ.match(token)
        if m:
            side = m.group(1).upper()
            neg = m.group(2) is not None
            whole = int(m.group(3))
            frac = m.group(4)
            thresh = whole + (int(frac) / (10 ** len(frac))) if frac else whole
            if neg:
                thresh = -thresh
            return self._resolve_dpi_zscore(side=side, threshold=float(thresh),
                                            window_trading_days=5)
        m = _TOKEN_NEWS_EVENT.match(token)
        if m:
            return self._resolve_news_event(window_cal_days=int(m.group(1)))
        m = _TOKEN_NEWS_COUNT.match(token)
        if m:
            return self._resolve_news_count_percentile(
                percentile=int(m.group(1)),
                window_cal_days=int(m.group(2)),
            )
        raise _AltDataUnknownToken(token)

    # ---- source loaders (cached) ----
    def _load_edgar_form(self, form_code: str) -> "pd.DataFrame":
        """Pull a single form type out of EDGAR. ``form_code`` is the value stored
        in the ``form`` column (e.g. ``'8-K'``, ``'4'``)."""
        key = f"edgar:{form_code}"
        if key in self._source_cache:
            return self._source_cache[key]
        # Resolve via wrapper if it works; otherwise direct SQLite over the EDGAR db.
        rows: List[Dict[str, Any]] = []
        try:
            from knowledge import edgar as _edgar  # type: ignore
            # The wrapper's `form=` kwarg is buggy (calls EdgarCache.get_filings with
            # `form=` instead of `form_type=`). Use the underlying class directly.
            try:
                from edgar_cache_loader import EdgarCache  # type: ignore
                rows = EdgarCache.get_filings(self.ticker, form_type=form_code)
            except Exception:
                # Last-ditch: use the wrapper anyway and let it raise.
                rows = _edgar.get_filings(self.ticker, form=form_code) or []
        except Exception as e:
            self._diagnostics.setdefault("source_errors", {})[key] = str(e)
            rows = []
        if not rows:
            df = pd.DataFrame(columns=["form", "filed_at", "accession_number"])
        else:
            df = pd.DataFrame(rows)
        if not df.empty and "filed_at" in df.columns:
            df["filed_at"] = pd.to_datetime(df["filed_at"], errors="coerce")
            df = df.dropna(subset=["filed_at"]).sort_values("filed_at")
        self._source_cache[key] = df
        self._diagnostics.setdefault("source_rows", {})[key] = int(len(df))
        return df

    def _load_congress(self) -> "pd.DataFrame":
        key = "govtrades:congress"
        if key in self._source_cache:
            return self._source_cache[key]
        df = pd.DataFrame()
        wrapper_err: Optional[str] = None
        try:
            from knowledge import govtrades as _gt  # type: ignore
            df = _gt.get_congress_trades(self.ticker)
        except Exception as e:
            wrapper_err = repr(e)
            df = pd.DataFrame()
        # The govtrades wrapper catches FileNotFoundError internally (returns empty DF +
        # WARN log) when the canonical /My Drive/Ph0tis/... path is missing. Detect that
        # path-missing case and try the 2TB hot tier directly.
        if df is None or df.empty:
            p = _resolve_govtrades_db_path()
            self._diagnostics.setdefault("source_errors", {})[key] = (
                f"wrapper returned empty (err={wrapper_err}); trying direct sqlite at {p}"
            )
            if p:
                try:
                    import sqlite3
                    with sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=10.0) as con:
                        df = pd.read_sql_query(
                            "SELECT transaction_date, report_date, transaction_type, "
                            "amount_min, representative FROM congress_trades WHERE ticker = ?",
                            con, params=(self.ticker,),
                            parse_dates=["transaction_date", "report_date"],
                        )
                except Exception as e2:
                    self._diagnostics["source_errors"][key + ":sqlite"] = str(e2)
                    df = pd.DataFrame()
        if df is None:
            df = pd.DataFrame()
        if not df.empty:
            # Build effective disclosure-aware date: prefer report_date; else transaction_date + 45d
            tx = pd.to_datetime(df.get("transaction_date"), errors="coerce")
            rd = pd.to_datetime(df.get("report_date"), errors="coerce")
            disclosure = rd.where(rd.notna(), tx + pd.Timedelta(days=45))
            df = df.assign(_disclosure_date=disclosure)
            df = df.dropna(subset=["_disclosure_date"]).sort_values("_disclosure_date")
        self._source_cache[key] = df
        self._diagnostics.setdefault("source_rows", {})[key] = int(len(df))
        return df

    def _load_offexchange(self) -> "pd.DataFrame":
        key = "govtrades:offexchange"
        if key in self._source_cache:
            return self._source_cache[key]
        df = pd.DataFrame()
        wrapper_err: Optional[str] = None
        try:
            from knowledge import govtrades as _gt  # type: ignore
            df = _gt.get_offexchange(self.ticker)
        except Exception as e:
            wrapper_err = repr(e)
            df = pd.DataFrame()
        # Same fallback as _load_congress — the wrapper catches the missing-DB error and
        # returns an empty DF, so we have to retry via direct SQLite on the 2TB hot tier.
        if df is None or df.empty:
            p = _resolve_govtrades_db_path()
            self._diagnostics.setdefault("source_errors", {})[key] = (
                f"wrapper returned empty (err={wrapper_err}); trying direct sqlite at {p}"
            )
            if p:
                try:
                    import sqlite3
                    with sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=10.0) as con:
                        df = pd.read_sql_query(
                            "SELECT date, otc_short, otc_total, dpi FROM offexchange "
                            "WHERE ticker = ? ORDER BY date",
                            con, params=(self.ticker,),
                            parse_dates=["date"],
                        )
                except Exception as e2:
                    self._diagnostics["source_errors"][key + ":sqlite"] = str(e2)
                    df = pd.DataFrame()
        if df is None:
            df = pd.DataFrame()
        if not df.empty:
            df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        self._source_cache[key] = df
        self._diagnostics.setdefault("source_rows", {})[key] = int(len(df))
        return df

    def _load_news(self) -> "pd.DataFrame":
        key = "news"
        if key in self._source_cache:
            return self._source_cache[key]
        df = pd.DataFrame()
        try:
            from knowledge import news as _news  # type: ignore
            # Bound the fetch to bar range plus pad to keep memory sane.
            if self.n > 0:
                start = (self.bar_ts.iloc[0] - pd.Timedelta(days=self.lookback_pad_days)).strftime("%Y-%m-%d")
                end = (self.bar_ts.iloc[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                df = _news.get_news(self.ticker, start=start, end=end)
            else:
                df = _news.get_news(self.ticker)
        except Exception as e:
            self._diagnostics.setdefault("source_errors", {})[key] = str(e)
            df = pd.DataFrame()
        if df is None:
            df = pd.DataFrame()
        if not df.empty and "published_utc" in df.columns:
            df["published_utc"] = pd.to_datetime(df["published_utc"], errors="coerce", utc=True)
            # Strip tz so it compares apples-to-apples with naive bar timestamps
            df["published_utc"] = df["published_utc"].dt.tz_localize(None)
            df = df.dropna(subset=["published_utc"]).sort_values("published_utc").reset_index(drop=True)
        self._source_cache[key] = df
        self._diagnostics.setdefault("source_rows", {})[key] = int(len(df))
        return df

    # ---- core resolvers ----
    @staticmethod
    def _searchsorted_count(sorted_ts: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """For each bar, count events in (lo[i], hi[i]] using two binary searches.

        sorted_ts assumed strictly sorted ascending. Returns counts per bar.
        """
        if sorted_ts.size == 0:
            return np.zeros(lo.shape[0], dtype=np.int64)
        hi_idx = np.searchsorted(sorted_ts, hi, side="right")
        lo_idx = np.searchsorted(sorted_ts, lo, side="right")
        return (hi_idx - lo_idx).astype(np.int64)

    def _resolve_form4(self, window_trading_days: int,
                       min_value_usd: Optional[int]) -> np.ndarray:
        """Form 4 buys in trailing N trading days. EDGAR DB currently holds 0 Form 4
        rows (backfill pending — see edgar.coverage()) so this returns all False with a
        diagnostic note. When the backfill lands, the body below will pick up automatically."""
        df = self._load_edgar_form("4")
        self._diagnostics.setdefault("token_notes", {})["InsiderForm4"] = {
            "rows_available": int(len(df)),
            "note": "Form 4 backfill pending (edgar.coverage() lists Form 4 as partial)" if df.empty else "",
        }
        if df.empty:
            return np.zeros(self.n, dtype=bool)
        # If we did have rows, we'd filter to ``transaction_type == 'P'`` (open-market buy)
        # and aggregate by ``filed_at`` (which is post-disclosure, no lookahead).
        ts = df["filed_at"].dt.tz_localize(None).to_numpy()
        # Convert window in trading days to calendar days approx (×1.45) for the timestamp join.
        # We could be more precise by using `bar_ts[i-window]` but trading-day-window vs calendar
        # is unambiguous here because every event row is timestamped.
        window_cal = int(window_trading_days * 1.45) + 1
        lo = (self.bar_ts - pd.Timedelta(days=window_cal)).to_numpy()
        hi = self.bar_ts.to_numpy()
        counts = self._searchsorted_count(ts, lo, hi)
        if min_value_usd is None:
            return counts > 0
        # When amounts are available, sum amounts in window per bar.
        if "value_usd" not in df.columns:
            # No amount column — fall back to "any buy" since we can't enforce $ threshold.
            return counts > 0
        # Slower path: per-bar sum (only reached when rows present)
        values = df["value_usd"].to_numpy(dtype=np.float64)
        out = np.zeros(self.n, dtype=bool)
        hi_idx = np.searchsorted(ts, hi, side="right")
        lo_idx = np.searchsorted(ts, lo, side="right")
        cum = np.concatenate([[0.0], np.cumsum(values)])
        sums = cum[hi_idx] - cum[lo_idx]
        out[:] = sums >= float(min_value_usd)
        return out

    def _resolve_form4_cluster(self, window_trading_days: int,
                               min_distinct_buyers: int) -> np.ndarray:
        df = self._load_edgar_form("4")
        self._diagnostics.setdefault("token_notes", {})["InsiderClusterBuy"] = {
            "rows_available": int(len(df)),
            "note": "Form 4 backfill pending" if df.empty else "",
        }
        # See _resolve_form4 — same Form 4 availability constraint.
        return np.zeros(self.n, dtype=bool)

    def _resolve_congress_buy(self, window_cal_days: int) -> np.ndarray:
        df = self._load_congress()
        if df.empty:
            return np.zeros(self.n, dtype=bool)
        buys = df[df["transaction_type"].astype(str).str.lower().str.startswith("purchase", na=False)]
        if buys.empty:
            return np.zeros(self.n, dtype=bool)
        ts = buys["_disclosure_date"].to_numpy()
        lo = (self.bar_ts - pd.Timedelta(days=window_cal_days)).to_numpy()
        hi = self.bar_ts.to_numpy()
        counts = self._searchsorted_count(np.sort(ts), lo, hi)
        return counts > 0

    def _resolve_8k(self, window_trading_days: int) -> np.ndarray:
        df = self._load_edgar_form("8-K")
        if df.empty:
            return np.zeros(self.n, dtype=bool)
        ts = df["filed_at"].dt.tz_localize(None).to_numpy()
        # Convert trading days to calendar days; 8-K filings are continuous-time anyway.
        window_cal = int(window_trading_days * 1.45) + 1
        lo = (self.bar_ts - pd.Timedelta(days=window_cal)).to_numpy()
        hi = self.bar_ts.to_numpy()
        counts = self._searchsorted_count(ts, lo, hi)
        return counts > 0

    def _build_dpi_aligned_series(self) -> Optional[np.ndarray]:
        """Forward-fill DPI to bar timestamps using ``date <= bar_ts`` strictly (no lookahead).

        Returns float series of length n with NaN where no prior DPI is available.
        """
        df = self._load_offexchange()
        if df.empty or "dpi" not in df.columns:
            return None
        ev_ts = df["date"].to_numpy()
        ev_val = df["dpi"].to_numpy(dtype=np.float64)
        bar_arr = self.bar_ts.to_numpy()
        # For strict "<= bar_ts" + 1-day publication-lag safety, use side="right"
        # then subtract 1 (so events with date == bar_ts are excluded — i.e. only
        # prior trading sessions count). This guarantees no use of the same-day DPI
        # for a same-day signal, which is the conservative choice.
        idx = np.searchsorted(ev_ts, bar_arr, side="right") - 1
        out = np.full(self.n, np.nan)
        valid = idx >= 0
        out[valid] = ev_val[idx[valid]]
        return out

    def _resolve_dpi_percentile(self, side: str, percentile: int,
                                window_trading_days: int) -> np.ndarray:
        dpi_series = self._build_dpi_aligned_series()
        if dpi_series is None or np.all(np.isnan(dpi_series)):
            return np.zeros(self.n, dtype=bool)
        # Compute rolling window percentile of strictly prior values
        s = pd.Series(dpi_series)
        # window of N trading days, computed over the bar-aligned series so it's
        # naturally on the trading clock; min_periods=N to avoid early-NaN noise.
        win = window_trading_days
        # Use shift(1) so the percentile excludes the current bar (no lookahead on
        # the bar's own DPI value, even though DPI is event-time).
        shifted = s.shift(1)
        thresh = shifted.rolling(win, min_periods=max(2, win // 2)).quantile(percentile / 100.0)
        if side == "GT":
            out = (s > thresh)
        else:
            out = (s < thresh)
        out = out.fillna(False).to_numpy().astype(bool)
        return out

    def _resolve_dpi_zscore(self, side: str, threshold: float,
                            window_trading_days: int) -> np.ndarray:
        dpi_series = self._build_dpi_aligned_series()
        if dpi_series is None or np.all(np.isnan(dpi_series)):
            return np.zeros(self.n, dtype=bool)
        s = pd.Series(dpi_series)
        shifted = s.shift(1)
        mu = shifted.rolling(window_trading_days, min_periods=max(2, window_trading_days // 2)).mean()
        sd = shifted.rolling(window_trading_days, min_periods=max(2, window_trading_days // 2)).std(ddof=1)
        z = (s - mu) / sd.where(sd > 0)
        if side == "GT":
            out = (z > threshold)
        else:
            out = (z < threshold)
        return out.fillna(False).to_numpy().astype(bool)

    def _resolve_news_event(self, window_cal_days: int) -> np.ndarray:
        df = self._load_news()
        if df.empty:
            return np.zeros(self.n, dtype=bool)
        ts = df["published_utc"].to_numpy()
        lo = (self.bar_ts - pd.Timedelta(days=window_cal_days)).to_numpy()
        hi = self.bar_ts.to_numpy()
        counts = self._searchsorted_count(ts, lo, hi)
        return counts > 0

    def _resolve_news_count_percentile(self, percentile: int, window_cal_days: int) -> np.ndarray:
        df = self._load_news()
        if df.empty:
            return np.zeros(self.n, dtype=bool)
        ts = df["published_utc"].to_numpy()
        lo = (self.bar_ts - pd.Timedelta(days=window_cal_days)).to_numpy()
        hi = self.bar_ts.to_numpy()
        counts = self._searchsorted_count(ts, lo, hi).astype(np.float64)
        # Rolling 1-year percentile of counts, shifted by 1 bar to avoid lookahead.
        s = pd.Series(counts)
        thresh = s.shift(1).rolling(252, min_periods=30).quantile(percentile / 100.0)
        out = (s > thresh).fillna(False).to_numpy().astype(bool)
        return out


class _AltDataUnknownToken(Exception):
    """Raised when a bareword isn't a recognized alt-data token. The role parser converts
    this to a normal 'unknown indicator' error."""


def _hypothesis_uses_alt_data(h: dict) -> bool:
    """Cheap check: does any role string contain a plausible alt-data token?

    Scans role expressions for IDENT-shaped runs (incl. ALT_IDENT like ``8K_LT5d``) and
    matches against the alt-data token regexes. Recurses into child_hypotheses.
    """
    ident_re = re.compile(r"\b\d*[A-Za-z][A-Za-z0-9_]+\b")
    role_keys = ("regime_gate", "bias_filter", "trigger", "confirmation",
                 "timing", "exit", "no_trade")
    for k in role_keys:
        v = h.get(k)
        if not isinstance(v, str):
            continue
        for m in ident_re.findall(v):
            if _looks_like_alt_data_token(m):
                return True
    if isinstance(h.get("child_hypotheses"), list):
        for c in h["child_hypotheses"]:
            if _hypothesis_uses_alt_data(c.get("hypothesis", {})):
                return True
            reg = c.get("regime")
            if isinstance(reg, str):
                for m in ident_re.findall(reg):
                    if _looks_like_alt_data_token(m):
                        return True
    return False


class _RoleParser:
    """Recursive-descent parser that turns a role-expression string into a per-bar numpy series.

    Comparison ops return float 0.0/1.0 arrays (so a `gate` is just truthy mask).
    Arithmetic + indicator calls return float series.

    Alt-data hook: if a bareword identifier (e.g. ``InsiderForm4_LT5d``) isn't in the
    indicator table AND an ``alt_data_resolver`` was passed at construction, the parser
    calls ``alt_data_resolver.resolve(token)`` and returns its bool series as a float
    0/1 array. This lets the same parser drive OHLCV indicators and alt-data event flags
    in the same expression (e.g. ``Close > Donchian_UP(20) AND CongressBuy_LT30d``).

    Multi-timeframe hook (task #53, 2026-05-29): bareword identifiers prefixed with
    ``__TF<NORM>_`` (rewritten from the user-facing ``<TF>.<TOKEN>`` syntax — see
    ``_rewrite_tf_prefixes``) are resolved against the corresponding higher-TF bars in
    ``bars_by_tf`` and aligned forward (no-lookahead) to the primary TF's timestamps.
    The legacy single-TF constructor signature is preserved for backward compat: when
    ``bars_by_tf`` is None, the parser auto-wraps ``bars`` as ``{primary_tf: bars}``.
    """

    def __init__(
        self,
        bars: ArrayDict,
        alt_data_resolver: Optional["_AltDataResolver"] = None,
        *,
        bars_by_tf: Optional[Dict[str, ArrayDict]] = None,
        ts_by_tf: Optional[Dict[str, "pd.DatetimeIndex"]] = None,
        primary_tf: str = "1d",
        altdata_numeric_resolver: Optional["_AltDataNumericResolver"] = None,
        xsym_resolver: Optional["_XSymResolver"] = None,
    ):
        self.bars = bars
        self.n = len(bars["close"])
        self.alt_data_resolver = alt_data_resolver
        # Task #56 (GOV_AWARE numeric) + #57 (cross-asset gates): additional resolvers
        # that return NUMERIC pd.Series (not bool). Resolved by lowercase identifier
        # match; falls through to the normal "unknown identifier" error if neither
        # resolver knows the name. Attached lazily by callers — None means feature off.
        self.altdata_numeric_resolver = altdata_numeric_resolver
        self.xsym_resolver = xsym_resolver
        # Multi-TF state. Default: wrap single-TF input as a one-entry map so the
        # cross-TF lookup path can ALWAYS go through bars_by_tf — keeps the code
        # uniform without bifurcating eval logic.
        self.primary_tf = _normalize_tf(primary_tf) if primary_tf else "1d"
        if bars_by_tf is None:
            self.bars_by_tf = {self.primary_tf: bars}
            self.ts_by_tf = ts_by_tf or {}
        else:
            self.bars_by_tf = dict(bars_by_tf)
            # The primary entry MUST be the same dict the legacy `bars` points at,
            # so legacy indicator calls (Close, ADX, etc.) operate on the right grid.
            self.bars_by_tf.setdefault(self.primary_tf, bars)
            self.ts_by_tf = dict(ts_by_tf or {})
        # Cache for cross-TF aligned indicator arrays: {(tf, token, args_key) → ndarray}.
        # Keyed so that ``1d.ADX(14)`` evaluated multiple times in the same role expr
        # (or across roles within one evaluate_hypothesis call) re-uses the same array.
        self._xtf_cache: Dict[Tuple[str, str, str], np.ndarray] = {}
        # Indicator dispatch table (name -> callable producing a np.ndarray of length n)
        self.indicators: Dict[str, Callable[..., np.ndarray]] = {
            "close": lambda: bars["close"],
            "high": lambda: bars["high"],
            "low": lambda: bars["low"],
            "open": lambda: bars["open"],
            "volume": lambda: bars["volume"],
            "sma": lambda n, *, src=None: _sma(self._resolve_src(src), int(n)),
            "ema": lambda n, *, src=None: _ema(self._resolve_src(src), int(n)),
            "atr": lambda n=14: _atr(bars, int(n)),
            "rsi": lambda n=14: _rsi(bars["close"], int(n)),
            "adx": lambda n=14: _wilder_adx(bars, int(n)),
            "donchian_up": lambda n=20: _donchian_up(bars, int(n)),
            "donchian_dn": lambda n=20: _donchian_dn(bars, int(n)),
            "vwap": lambda: _running_vwap(bars),
            "chopidx": lambda n=14: _chop_index(bars, int(n)),
            "choppinessindex": lambda n=14: _chop_index(bars, int(n)),
            "stochk": lambda k=14, d=3, s=3: stochastic(bars, int(k), int(d), int(s))["k"],
            "stochd": lambda k=14, d=3, s=3: stochastic(bars, int(k), int(d), int(s))["d"],
            "williams": lambda n=14: williams_r(bars, int(n)),
            "cci": lambda n=20: cci(bars, int(n)),
            "mfi": lambda n=14: mfi(bars, int(n)),
            "cmf": lambda n=21: _cmf(bars, int(n)),
            "connorsrsi": lambda r=3, s=2, k=100: connors_rsi(bars, int(r), int(s), int(k)),
            "supertrend": lambda n=10, m=3.0: supertrend(bars, int(n), float(m))["trend"].astype(np.float64),
            "fisher": lambda n=10: fisher_transform(bars, int(n)),
            "macd": lambda f=12, s=26, sg=9: macd(bars, int(f), int(s), int(sg))["macd"],
            "macd_hist": lambda f=12, s=26, sg=9: macd(bars, int(f), int(s), int(sg))["hist"],
            # Composite booleans
            "volumeexpansion": lambda n=20, mult=1.5: (
                bars["volume"] > float(mult) * _sma(bars["volume"], int(n))
            ).astype(np.float64),
            # BB attribute accessor handled separately via dotted forms
            "bb": lambda n=20, k=2.0: bollinger(bars, int(n), float(k)),
            "obv": lambda: obv(bars),
            # Trivial truthy constants
            "true": lambda: np.ones(self.n),
            "false": lambda: np.zeros(self.n),
        }

    def _resolve_src(self, src) -> np.ndarray:
        if src is None:
            return self.bars["close"]
        if isinstance(src, np.ndarray):
            return src
        if isinstance(src, str):
            key = src.lower()
            if key in ("close", "high", "low", "open", "volume"):
                return self.bars[key]
        raise ValueError(f"unknown SMA/EMA src: {src!r}")

    # --------- public ---------
    def evaluate(self, expr: str) -> np.ndarray:
        if expr is None:
            return np.ones(self.n)
        if isinstance(expr, (int, float)):
            return np.full(self.n, float(expr))
        if isinstance(expr, bool):
            return np.ones(self.n) if expr else np.zeros(self.n)
        if not isinstance(expr, str) or not expr.strip():
            return np.ones(self.n)
        # Multi-TF prefix rewrite: ``1d.ADX(14)`` → ``__TF1D_ADX(14)`` so it tokenizes
        # as a single bareword the parser can dispatch to the cross-TF resolver.
        # This is a no-op if the expression contains no TF prefixes.
        rewritten = _rewrite_tf_prefixes(expr)
        self.toks = _tokenize(rewritten)
        self.pos = 0
        result = self._parse_or()
        if self.pos != len(self.toks):
            raise ValueError(f"role parser: trailing tokens at pos {self.pos} expr={expr!r}")
        if np.isscalar(result):
            return np.full(self.n, float(result))
        return _bcast(result, self.n)

    # --------- recursive descent ---------
    def _peek(self) -> Optional[_Tok]:
        return self.toks[self.pos] if self.pos < len(self.toks) else None

    def _eat(self) -> _Tok:
        t = self.toks[self.pos]
        self.pos += 1
        return t

    def _accept(self, kind: str, val: Optional[str] = None) -> Optional[_Tok]:
        t = self._peek()
        if t is None or t.kind != kind:
            return None
        if val is not None and t.val != val:
            return None
        self.pos += 1
        return t

    def _expect(self, kind: str, val: Optional[str] = None) -> _Tok:
        t = self._accept(kind, val)
        if t is None:
            got = self._peek()
            raise ValueError(f"role parser: expected {kind}{f'={val!r}' if val else ''}, got {got}")
        return t

    def _parse_or(self):
        left = self._parse_and()
        while self._accept("OP", "OR"):
            right = self._parse_and()
            left = ((_bcast(left, self.n) != 0) | (_bcast(right, self.n) != 0)).astype(np.float64)
        return left

    def _parse_and(self):
        left = self._parse_cmp()
        while self._accept("OP", "AND"):
            right = self._parse_cmp()
            left = ((_bcast(left, self.n) != 0) & (_bcast(right, self.n) != 0)).astype(np.float64)
        return left

    def _parse_cmp(self):
        left = self._parse_add()
        t = self._peek()
        if t and t.kind == "OP" and t.val in (">", "<", ">=", "<=", "==", "!="):
            self._eat()
            right = self._parse_add()
            L = _bcast(left, self.n)
            R = _bcast(right, self.n)
            op = t.val
            with np.errstate(invalid="ignore"):
                if op == ">":  res = L > R
                elif op == "<":  res = L < R
                elif op == ">=": res = L >= R
                elif op == "<=": res = L <= R
                elif op == "==": res = L == R
                else:            res = L != R
            out = res.astype(np.float64)
            # Where either side is NaN, comparison fails → 0
            nan_mask = np.isnan(L) | np.isnan(R)
            out[nan_mask] = 0.0
            return out
        return left

    def _parse_add(self):
        left = self._parse_mul()
        while True:
            if self._accept("PLUS"):
                right = self._parse_mul()
                left = _bcast(left, self.n) + _bcast(right, self.n)
            elif self._accept("MINUS"):
                right = self._parse_mul()
                left = _bcast(left, self.n) - _bcast(right, self.n)
            else:
                return left

    def _parse_mul(self):
        left = self._parse_unary()
        while True:
            if self._accept("MUL"):
                right = self._parse_unary()
                left = _bcast(left, self.n) * _bcast(right, self.n)
            elif self._accept("DIV"):
                right = self._parse_unary()
                R = _bcast(right, self.n)
                out = np.full(self.n, np.nan)
                mask = R != 0
                out[mask] = _bcast(left, self.n)[mask] / R[mask]
                left = out
            else:
                return left

    def _parse_unary(self):
        if self._accept("MINUS"):
            inner = self._parse_unary()
            return -_bcast(inner, self.n)
        if self._accept("OP", "NOT"):
            inner = self._parse_unary()
            return (_bcast(inner, self.n) == 0).astype(np.float64)
        return self._parse_atom()

    def _parse_atom(self):
        t = self._peek()
        if t is None:
            raise ValueError("role parser: unexpected end of expression")
        if t.kind == "NUMBER":
            self._eat()
            return float(t.val)
        if t.kind == "LP":
            self._eat()
            inner = self._parse_or()
            self._expect("RP")
            return inner
        if t.kind == "IDENT":
            return self._parse_indicator_call()
        raise ValueError(f"role parser: unexpected token {t}")

    # --------- cross-TF helpers ---------
    @staticmethod
    def _parse_xtf_ident(ident: str) -> Optional[Tuple[str, str]]:
        """If ident is ``__TF<NORM>_<TOKEN>``, return (tf_canonical, token).
        Else return None.

        Canonical normalization (uppercase suffix → lowercase canonical):
          __TF1D_X       → ("1d", "X")
          __TF5MIN_X     → ("5min", "X")
          __TF15MIN_X    → ("15min", "X")
          __TF1H_X       → ("1h", "X")
          __TF1MIN_X     → ("1min", "X")
        """
        if not ident.startswith("__TF"):
            return None
        rest = ident[len("__TF"):]
        # Greedy match on the known TF suffixes, longest first to avoid 1MIN→1M ambig.
        for tf_upper, tf_canon in (
            ("15MIN", "15min"),
            ("5MIN",  "5min"),
            ("1MIN",  "1min"),
            ("1H",    "1h"),
            ("1D",    "1d"),
        ):
            if rest.startswith(tf_upper + "_"):
                return tf_canon, rest[len(tf_upper) + 1:]
        return None

    def _eval_cross_tf_token(
        self, tf: str, token: str, args: List[float], kwargs: Dict[str, Any],
        dotted_attr: Optional[str] = None,
    ) -> np.ndarray:
        """Compute ``token(*args, **kwargs)`` on the higher-TF bars, then align
        forward (no-lookahead) to the primary TF's timestamps.

        ``dotted_attr`` is set for BB.upper / BB.lower / BB.mid / BB.pctb forms.

        Caches the aligned result per (tf, token, args_key) so repeated references
        in the same role-expression set don't recompute.
        """
        tf = _normalize_tf(tf)
        # Same-TF reference (e.g. 5min.Close inside a 5min-primary role) is just
        # the regular indicator on the primary bars — no alignment needed.
        if tf == self.primary_tf:
            return self._eval_indicator_on_bars(self.bars, token, args, kwargs, dotted_attr=dotted_attr)
        if tf not in self.bars_by_tf:
            # Higher TF not loaded. Honest degradation: emit NaN so any downstream
            # comparison evaluates False (per _parse_cmp's NaN-handling), and the
            # signal effectively never fires. Better than silently substituting
            # the primary-TF value (which would be a lookahead-flavored bug).
            return np.full(self.n, np.nan, dtype=np.float64)
        args_key = f"{token}|{args}|{kwargs}|{dotted_attr}"
        cache_key = (tf, token, args_key)
        if cache_key in self._xtf_cache:
            return self._xtf_cache[cache_key]
        bars_high = self.bars_by_tf[tf]
        ts_high = self.ts_by_tf.get(tf)
        ts_primary = self.ts_by_tf.get(self.primary_tf)
        # Compute the indicator on the higher-TF bars.
        try:
            arr_high = self._eval_indicator_on_bars(
                bars_high, token, args, kwargs, dotted_attr=dotted_attr,
            )
        except ValueError:
            # Token isn't in the indicator table for the higher TF; surface NaN.
            self._xtf_cache[cache_key] = np.full(self.n, np.nan, dtype=np.float64)
            return self._xtf_cache[cache_key]
        if ts_high is None or ts_primary is None or len(ts_primary) != self.n:
            # Without aligned timestamps we cannot do no-lookahead forward fill;
            # the legacy single-TF code path already lacked timestamps for the
            # primary TF when called from non-multi-TF entrypoints. In that case
            # the safe fallback is to emit NaN (signal never fires) and log via
            # the result row. The multi-TF entrypoint always supplies both.
            self._xtf_cache[cache_key] = np.full(self.n, np.nan, dtype=np.float64)
            return self._xtf_cache[cache_key]
        aligned = _align_higher_tf_to_lower(arr_high, ts_high, ts_primary)
        self._xtf_cache[cache_key] = aligned
        return aligned

    def _eval_indicator_on_bars(
        self, bars: ArrayDict, token: str, args: List[float], kwargs: Dict[str, Any],
        dotted_attr: Optional[str] = None,
    ) -> np.ndarray:
        """Evaluate ``token(*args, **kwargs)[.dotted_attr]`` on a specific bars dict.

        Reuses the indicator table by building a one-off mini-_RoleParser bound to
        the alternate bars. (Lighter than refactoring every callable to take a
        ``bars`` arg; we sidestep by spinning up a transient parser instance.)
        """
        # Create a transient parser bound to the alternate bars. We DO NOT pass
        # an alt_data_resolver — alt-data is bar-timestamp-keyed and aligns at
        # the resolver layer; cross-TF alt-data tokens are out of scope for v1.
        transient = _RoleParser(bars, alt_data_resolver=None, primary_tf=self.primary_tf)
        key = token.lower()
        if dotted_attr is not None:
            if key != "bb":
                raise ValueError(f"role parser: dotted attr only supported on BB, got {token!r}.{dotted_attr!r}")
            res = transient.indicators["bb"](*args, **kwargs)
            if dotted_attr not in res:
                raise ValueError(f"role parser: BB has no attribute {dotted_attr!r}")
            return res[dotted_attr]
        if key not in transient.indicators:
            raise ValueError(f"role parser: unknown cross-TF indicator {token!r}")
        out = transient.indicators[key](*args, **kwargs)
        if isinstance(out, dict):
            raise ValueError(f"role parser: cross-TF {token}(...) returned dict; use {token}.attr form")
        return out

    def _parse_indicator_call(self):
        ident = self._eat().val
        # Cross-TF dispatch: ``__TF<NORM>_<TOKEN>`` was rewritten upstream from
        # the user-facing ``<TF>.<TOKEN>`` syntax. Strip the prefix, dispatch to
        # the higher-TF bars, and forward-fill onto the primary TF's grid.
        xtf = self._parse_xtf_ident(ident)
        if xtf is not None:
            tf_canon, raw_token = xtf
            # Handle BB.upper-style dotted attribute on cross-TF too:
            # ``1d.BB.upper(20, 2)`` rewrites to ``__TF1D_BB.upper(20, 2)``
            # which leaves the dotted attr accessible at parse time.
            dotted_attr = None
            if raw_token.lower() == "bb" and self._accept("DOT"):
                attr_tok = self._expect("IDENT")
                dotted_attr = attr_tok.val.lower()
            args, kwargs = self._parse_call_args_if_any()
            return self._eval_cross_tf_token(tf_canon, raw_token, args, kwargs, dotted_attr=dotted_attr)
        key = ident.lower()
        # Special boolean keywords already in indicators table
        # Handle "BB.upper(20, 2)" style dotted attribute calls
        if key == "bb" and self._accept("DOT"):
            attr_tok = self._expect("IDENT")
            attr = attr_tok.val.lower()
            args, kwargs = self._parse_call_args_if_any()
            res = self.indicators["bb"](*args, **kwargs)
            if attr not in res:
                raise ValueError(f"role parser: BB has no attribute {attr!r}")
            return res[attr]
        if key not in self.indicators:
            # Task #56: NUMERIC alt-data tokens (form4_insider_cluster_score etc.) — these
            # carry NO call args and resolve to a per-bar float series via the numeric
            # resolver. Check BEFORE bool alt-data so numeric takes priority when both
            # resolvers see the name. Returns NaN-filled series if source missing —
            # downstream comparisons handle NaN by yielding 0 (no signal).
            if (self.altdata_numeric_resolver is not None
                    and self.altdata_numeric_resolver.knows(key)):
                if self._peek() and self._peek().kind == "LP":
                    # Allow but ignore empty parens for consistency with how indicator
                    # tokens look in DSL — `form4_insider_cluster_score()` is fine.
                    _args, _kwargs = self._parse_call_args_if_any()
                    if _args or _kwargs:
                        raise ValueError(
                            f"role parser: numeric alt-data token {ident!r} takes no call args"
                        )
                arr = self.altdata_numeric_resolver.resolve(key)
                return arr.astype(np.float64)
            # Task #57: cross-asset (cross-symbol) tokens (vix_term_struct, sector_rs_rank,
            # hyg_lqd_ratio, spy_beta_60d, sector_rotation_rank, vix_multiplied_atr,
            # spy_correlation, dxy_delta, abs_spy_beta_60d). Empty-paren also allowed.
            if (self.xsym_resolver is not None
                    and self.xsym_resolver.knows(key)):
                if self._peek() and self._peek().kind == "LP":
                    _args, _kwargs = self._parse_call_args_if_any()
                    if _args or _kwargs:
                        raise ValueError(
                            f"role parser: cross-asset token {ident!r} takes no call args"
                        )
                arr = self.xsym_resolver.resolve(key)
                return arr.astype(np.float64)
            # Alt-data fallback: a bareword like ``InsiderForm4_LT5d`` resolves to a per-bar
            # bool series via the alt-data resolver if one was attached.
            if self.alt_data_resolver is not None and _looks_like_alt_data_token(ident):
                # Alt-data tokens carry params in the NAME (e.g. ``_LT5d``) so they MUST NOT
                # be followed by ``(args)``. If they are, surface a clear error.
                if self._peek() and self._peek().kind == "LP":
                    raise ValueError(
                        f"role parser: alt-data token {ident!r} takes no call args"
                    )
                try:
                    arr = self.alt_data_resolver.resolve(ident)
                    return arr.astype(np.float64)
                except _AltDataUnknownToken:
                    pass  # fall through to the generic error below
            raise ValueError(
                f"role parser: unknown indicator/identifier {ident!r}. "
                f"Available: {sorted(self.indicators.keys())}"
            )
        args, kwargs = self._parse_call_args_if_any()
        fn = self.indicators[key]
        try:
            out = fn(*args, **kwargs)
        except TypeError as e:
            raise ValueError(f"role parser: bad args to {ident}({args},{kwargs}): {e}")
        if isinstance(out, dict):
            raise ValueError(
                f"role parser: {ident}(...) returned dict; use attribute access like {ident}.upper(...)")
        return out

    def _parse_call_args_if_any(self) -> Tuple[List[float], Dict[str, Any]]:
        if not self._accept("LP"):
            return [], {}
        args: List[float] = []
        kwargs: Dict[str, Any] = {}
        if self._accept("RP"):
            return args, kwargs
        while True:
            # Numeric positional args only (no kwargs in current grammar — keep it simple)
            sign = 1.0
            if self._accept("MINUS"):
                sign = -1.0
            t = self._peek()
            if t and t.kind == "NUMBER":
                self._eat()
                args.append(sign * float(t.val))
            else:
                # Allow bare-ident like Close to pass as `src` (for SMA(close, 20) style)
                if t and t.kind == "IDENT":
                    self._eat()
                    # If it's a known data source name, pass as a `src` kwarg
                    if t.val.lower() in ("close", "high", "low", "open", "volume"):
                        kwargs["src"] = t.val.lower()
                    else:
                        raise ValueError(f"role parser: only numeric or src args supported, got {t}")
                else:
                    raise ValueError(f"role parser: expected arg, got {t}")
            if self._accept("COMMA"):
                continue
            self._expect("RP")
            return args, kwargs


# ============================================================================
# Hypothesis state machine
# ============================================================================


def _bool_mask(arr: np.ndarray) -> np.ndarray:
    """Treat NaN as False, non-zero as True."""
    out = np.where(np.isnan(arr), 0.0, arr) != 0
    return out


def _numeric_param_paths(d: dict, prefix: str = "") -> List[Tuple[str, float]]:
    """Walk a hypothesis dict and find numeric literals embedded in role strings to perturb."""
    out: List[Tuple[str, float]] = []
    for k, v in d.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out.append((f"{prefix}{k}", float(v)))
    return out


def _perturb_numeric_literals_in_string(expr: str, factor: float) -> str:
    """Find numeric literals in a role expression and scale them by factor.

    Intentionally crude: matches integer/float literals NOT immediately preceded by an alpha
    char (so 'SMA(20)' becomes 'SMA(22)' at factor 1.1, but '5min' inside a string stays).
    """
    if not isinstance(expr, str):
        return expr

    def repl(m):
        v = float(m.group(0))
        new = v * factor
        # Preserve integer-ness when the original had no decimal
        if "." not in m.group(0):
            new_int = int(round(new))
            return str(new_int if new_int > 0 else 1)
        return f"{new:.4g}"

    return re.sub(r"(?<![A-Za-z_])\d+(?:\.\d+)?", repl, expr)


def _bars_per_year_for_tf(tf: str) -> int:
    """Annualization factor (bars/year) for a given canonical timeframe.

    Used to scale Sharpe; numbers reflect ~6.5h RTH sessions × 252 trading days.
    """
    tf = _normalize_tf(tf)
    return {
        "1d":    252,
        "1h":    252 * 7,   # ~7 hourly bars per RTH session (rounded)
        "15min": 252 * 26,
        "5min":  252 * 78,
        "1min":  252 * 390,
    }.get(tf, 252)


def _load_bar_timestamps(ticker: str) -> Optional["pd.DatetimeIndex"]:
    """Re-read the OHLC parquet for the active timeframe and pull out the timestamp column.

    Mirrors the path-resolution logic of ``indicator_hardening_runner.load_ohlc`` but only
    fetches the timestamp column. Returns None when no parquet is available.
    """
    tf = _ihr._state["timeframe"]
    if tf == "5min":
        p_local = _ihr.OHLC_DIR / f"{ticker}_5min.parquet"
        p_drive = _ihr.DRIVE_OHLC_5MIN / f"{ticker}_5min.parquet"
    else:
        p_local = None
        p_drive = _ihr.DRIVE_OHLC_DAILY / f"{ticker}.parquet"
    path = (p_local if (p_local is not None and p_local.exists() and p_local.stat().st_size > 1000)
            else p_drive)
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None
    needed = {"open", "high", "low", "close", "volume"}
    # Mirror the renaming in load_ohlc so dropna sees normalized columns
    rename = {}
    for canon in ("open", "high", "low", "close", "volume", "date", "Date", "DATE", "timestamp", "Timestamp"):
        if canon in df.columns:
            rename[canon] = canon.lower() if canon.lower() in ("open", "high", "low", "close", "volume", "date", "timestamp") else canon
    df = df.rename(columns=rename)
    # Same dropna+min-bars contract as load_ohlc so the timestamp array stays aligned
    if not needed.issubset(df.columns):
        return None
    df = df.dropna(subset=list(needed))
    if len(df) < _ihr._state["min_bars"]:
        return None
    ts_col = None
    for cand in ("date", "timestamp"):
        if cand in df.columns:
            ts_col = cand
            break
    if ts_col is None and not isinstance(df.index, pd.RangeIndex):
        return pd.to_datetime(df.index)
    if ts_col is None:
        return None
    return pd.to_datetime(df[ts_col].to_numpy())


def evaluate_hypothesis(bars: ArrayDict, hypothesis: dict,
                        alt_data_resolver: Optional[_AltDataResolver] = None,
                        *,
                        bars_by_tf: Optional[Dict[str, ArrayDict]] = None,
                        ts_by_tf: Optional[Dict[str, "pd.DatetimeIndex"]] = None,
                        primary_tf: str = "1d",
                        altdata_numeric_resolver: Optional["_AltDataNumericResolver"] = None,
                        xsym_resolver: Optional["_XSymResolver"] = None) -> np.ndarray:
    """Walk the per-bar state machine for a hypothesis.

    Returns a position series in {-1, 0, +1}, length n.

    Multi-TF (task #53): when ``bars_by_tf`` is provided, cross-TF tokens of the
    form ``<TF>.<TOKEN>`` in any role expression resolve against the corresponding
    bars in ``bars_by_tf``, forward-filled (no-lookahead) to the primary TF's
    timestamp grid. ``bars`` MUST be ``bars_by_tf[primary_tf]`` — the position
    series fires on the primary timeframe; higher TFs are regime/bias gates only.

    State machine rules:
      • regime_gate False → flat for this bar (no entry, but exits on existing position
        still fire to be safe).
      • bias_filter sets `bias`: +1 if rule is true and trigger is breakout/long-flavor,
        else -1 when paired with bearish trigger. Since the grammar doesn't distinguish,
        we treat bias_filter as a long-only gate when True, short-only gate when its
        negation is supplied. Simplification: if bias_filter is True → only longs are
        allowed; if False → only shorts. For a long-only hypothesis the bias is True.
      • trigger True → arm an entry for next bar (no look-ahead).
      • confirmation must also be True on the same bar as trigger.
      • timing must also be True.
      • Entry happens at next bar's open (we use close-to-close returns + shift so
        the position takes effect bar+1).
      • exit True while in a position → flat at next bar.
      • no_trade True → force flat immediately.

    For simplicity and to avoid silent direction errors, this routine treats the
    hypothesis as LONG-ONLY unless the hypothesis dict has key "side"="short", in
    which case shorts are produced. Mixed long/short hypotheses (e.g. regime-switch)
    can supply a `child_hypotheses` list — each child evaluated and OR-merged by
    regime mask.
    """
    n = len(bars["close"])
    # Regime-switch composite: list of {regime: <expr>, hypothesis: {...}}
    if isinstance(hypothesis.get("child_hypotheses"), list):
        parser = _RoleParser(
            bars, alt_data_resolver=alt_data_resolver,
            bars_by_tf=bars_by_tf, ts_by_tf=ts_by_tf, primary_tf=primary_tf,
            altdata_numeric_resolver=altdata_numeric_resolver,
            xsym_resolver=xsym_resolver,
        )
        pos = np.zeros(n, dtype=np.int8)
        for child in hypothesis["child_hypotheses"]:
            mask = _bool_mask(parser.evaluate(child.get("regime", "TRUE")))
            sub_pos = evaluate_hypothesis(
                bars, child["hypothesis"],
                alt_data_resolver=alt_data_resolver,
                bars_by_tf=bars_by_tf, ts_by_tf=ts_by_tf, primary_tf=primary_tf,
                altdata_numeric_resolver=altdata_numeric_resolver,
                xsym_resolver=xsym_resolver,
            )
            # Apply mask: child position only counts during its regime
            pos = np.where(mask & (pos == 0), sub_pos, pos).astype(np.int8)
        # no_trade override (parent-level)
        if hypothesis.get("no_trade"):
            no_trade_mask = _bool_mask(parser.evaluate(hypothesis["no_trade"]))
            pos[no_trade_mask] = 0
        # Cross-asset gate (task #57, parent-level): force flat where xsym gate False
        if hypothesis.get("cross_asset_gate"):
            xa_mask = _bool_mask(parser.evaluate(hypothesis["cross_asset_gate"]))
            pos[~xa_mask] = 0
        return pos

    parser = _RoleParser(
        bars, alt_data_resolver=alt_data_resolver,
        bars_by_tf=bars_by_tf, ts_by_tf=ts_by_tf, primary_tf=primary_tf,
        altdata_numeric_resolver=altdata_numeric_resolver,
        xsym_resolver=xsym_resolver,
    )
    side = -1 if str(hypothesis.get("side", "long")).lower() == "short" else 1
    gate = _bool_mask(parser.evaluate(hypothesis.get("regime_gate", "TRUE")))
    # Cross-asset gate (task #57): an additional AND'd filter ON TOP of regime_gate.
    # Defaults to TRUE so legacy v1 hypotheses without the field are unaffected.
    cross_asset_gate = _bool_mask(parser.evaluate(hypothesis.get("cross_asset_gate", "TRUE")))
    gate = gate & cross_asset_gate
    bias = _bool_mask(parser.evaluate(hypothesis.get("bias_filter", "TRUE")))
    trig = _bool_mask(parser.evaluate(hypothesis.get("trigger", "FALSE")))
    conf = _bool_mask(parser.evaluate(hypothesis.get("confirmation", "TRUE")))
    timing = _bool_mask(parser.evaluate(hypothesis.get("timing", "TRUE")))
    no_trade = _bool_mask(parser.evaluate(hypothesis.get("no_trade", "FALSE")))
    # exit: a single boolean expression evaluated per bar (price-based exits like ATR stop
    # are represented as truthy conditions — the runner handles the actual stop calc only
    # if exit string is exactly "ATR_STOP" or contains "trailing"; for simplicity we let
    # the role parser evaluate the exit string directly and check for trail keyword)
    exit_expr = hypothesis.get("exit", "FALSE")
    exit_arr = _bool_mask(parser.evaluate(exit_expr)) if not _is_trailing_stop(exit_expr) else None
    trailing_mult, trailing_period = _parse_trailing_stop(exit_expr) if _is_trailing_stop(exit_expr) else (None, None)

    pos = np.zeros(n, dtype=np.int8)
    if trailing_mult is not None:
        from indicator_compute import _atr as _atr_fn
        atr = _atr_fn(bars, trailing_period or 14)
    else:
        atr = None
    state = 0  # 0=flat, side=long/short
    entry_price = 0.0
    trail_anchor = 0.0  # highest-since-entry for long, lowest for short
    close = bars["close"]
    for i in range(n):
        # no_trade override
        if no_trade[i]:
            state = 0
        # If in a position, check exit conditions
        if state != 0:
            if atr is not None and not np.isnan(atr[i]):
                if state == 1:
                    trail_anchor = max(trail_anchor, close[i])
                    if close[i] <= trail_anchor - trailing_mult * atr[i]:
                        state = 0
                else:
                    trail_anchor = min(trail_anchor, close[i])
                    if close[i] >= trail_anchor + trailing_mult * atr[i]:
                        state = 0
            elif exit_arr is not None and exit_arr[i]:
                state = 0
        # If flat, check for entry
        if state == 0 and gate[i] and bias[i] and trig[i] and conf[i] and timing[i] and not no_trade[i]:
            state = side
            entry_price = close[i]
            trail_anchor = entry_price
        pos[i] = state
    return pos


def _is_trailing_stop(expr) -> bool:
    if not isinstance(expr, str):
        return False
    e = expr.lower()
    return "trailing" in e or "atr_stop" in e or ("atr" in e and ("trail" in e or "stop" in e))


def _parse_trailing_stop(expr: str) -> Tuple[float, int]:
    """Pick a multiplier (default 1.5) and ATR period (default 14) out of an exit string."""
    m_mult = re.search(r"(\d+(?:\.\d+)?)\s*[*×x]\s*ATR", expr, re.IGNORECASE)
    mult = float(m_mult.group(1)) if m_mult else 1.5
    m_period = re.search(r"ATR\s*\(\s*(\d+)\s*\)", expr, re.IGNORECASE)
    period = int(m_period.group(1)) if m_period else 14
    return mult, period


# ============================================================================
# 6-step pipeline on a hypothesis
# ============================================================================


def returns_from_position(bars: ArrayDict, pos: np.ndarray) -> np.ndarray:
    close = bars["close"]
    log_ret = np.zeros_like(close)
    log_ret[1:] = np.log(close[1:] / close[:-1])
    p = np.zeros_like(close, dtype=np.float64)
    p[1:] = pos[:-1].astype(np.float64)  # shift forward → no look-ahead
    dpos = np.zeros_like(close)
    dpos[1:] = np.abs(p[1:] - p[:-1])
    return p * log_ret - COST_PER_SIDE * dpos


def win_rate_from_position(bars: ArrayDict, pos: np.ndarray) -> Tuple[float, int]:
    close = bars["close"]
    p = np.zeros_like(close, dtype=np.int8)
    p[1:] = pos[:-1]
    trades = []
    i = 0
    n = len(close)
    while i < n:
        if p[i] == 0:
            i += 1; continue
        side = p[i]
        entry = close[i]
        j = i
        while j < n and p[j] == side:
            j += 1
        exit_ = close[min(j - 1, n - 1)]
        ret = side * (exit_ / entry - 1) - 2 * COST_PER_SIDE
        trades.append(ret)
        i = j
    if not trades:
        return float("nan"), 0
    arr = np.asarray(trades)
    return float((arr > 0).mean()), int(arr.size)


def annualized_sharpe(rets: np.ndarray, bars_per_year: int = 252) -> float:
    rets = rets[~np.isnan(rets)]
    if rets.size < 50 or np.std(rets, ddof=1) == 0:
        return float("nan")
    return float(np.mean(rets) / np.std(rets, ddof=1) * np.sqrt(bars_per_year))


def _generate_perturbed_hypotheses(hypothesis: dict, perturb_pcts=(-10, 0, 10)) -> List[dict]:
    """Make ±10% variants by scaling every numeric literal in every role expression.

    Returns a list of hypothesis dicts (length = len(perturb_pcts)). The 0% entry is the
    original hypothesis (deep-copied).
    """
    out = []
    for pct in perturb_pcts:
        factor = 1.0 + pct / 100.0
        new = copy.deepcopy(hypothesis)
        for role in ("regime_gate", "bias_filter", "trigger", "confirmation",
                     "timing", "exit", "no_trade"):
            if isinstance(new.get(role), str):
                new[role] = _perturb_numeric_literals_in_string(new[role], factor)
        # child_hypotheses (regime-switch)
        if isinstance(new.get("child_hypotheses"), list):
            new["child_hypotheses"] = [
                {
                    "regime": _perturb_numeric_literals_in_string(c.get("regime", "TRUE"), factor),
                    "hypothesis": {
                        **{k: (_perturb_numeric_literals_in_string(v, factor) if isinstance(v, str) and k in
                               ("regime_gate", "bias_filter", "trigger", "confirmation", "timing", "exit", "no_trade")
                               else v)
                           for k, v in c["hypothesis"].items()}
                    },
                }
                for c in new["child_hypotheses"]
            ]
        out.append(new)
    return out


def run_hypothesis_for_ticker(
    hypothesis: dict, ticker: str, bars_per_year: int, n_folds: int = 12,
) -> dict:
    """Run a hypothesis on one ticker. Mirrors run_indicator_for_ticker but for a hypothesis.

    Multi-TF behavior (task #53, 2026-05-29):
      * If the hypothesis has a ``timeframe_stack`` (list of TF strings, e.g.
        ``["5min", "15min", "1d"]``) the runner loads bars for each TF and wires
        them into the parser so cross-TF tokens like ``1d.ADX(14)`` resolve.
        The first entry is the primary (entry/exit) TF.
      * If only ``timeframe`` is set (legacy single-TF), the runner falls back to
        the single-TF path — fully back-compatible with task #41 dispatch.
    """
    # Pull the TF stack from the hypothesis. Back-compat: when only `timeframe`
    # is set, treat it as a single-element stack.
    tf_stack_raw = hypothesis.get("timeframe_stack")
    primary_tf_raw = hypothesis.get("timeframe", "1d")
    if isinstance(tf_stack_raw, (list, tuple)) and tf_stack_raw:
        tf_stack = [str(t) for t in tf_stack_raw]
        primary_tf = _normalize_tf(tf_stack[0])
    else:
        tf_stack = [str(primary_tf_raw)]
        primary_tf = _normalize_tf(primary_tf_raw)

    bars_by_tf: Optional[Dict[str, ArrayDict]] = None
    ts_by_tf: Optional[Dict[str, "pd.DatetimeIndex"]] = None
    load_notes: Dict[str, str] = {}

    if len(tf_stack) > 1:
        # Multi-TF path: load each timeframe's bars.
        bars_by_tf, ts_by_tf, load_notes = _load_bars_by_tf(ticker, tf_stack)
        if primary_tf not in bars_by_tf:
            return {
                "ticker": ticker, "status": "no_data",
                "load_notes": load_notes,
                "missing_primary_tf": primary_tf,
            }
        bars = bars_by_tf[primary_tf]
        # Adjust bars_per_year for the primary TF so Sharpe / fold sizing is right.
        bars_per_year = _bars_per_year_for_tf(primary_tf)
    else:
        # Legacy single-TF path: use indicator_hardening_runner's loader so the
        # existing daily cache + min_bars contract are preserved.
        saved_tf = _ihr._state["timeframe"]
        try:
            _ihr.set_timeframe(primary_tf if primary_tf in ("1d", "5min") else "1d")
            bars = _ihr.load_ohlc(ticker)
            bars_per_year = _ihr._state["bars_per_year"]
        finally:
            _ihr.set_timeframe(saved_tf)
        if bars is None:
            return {"ticker": ticker, "status": "no_data"}
        # Lazy: skip building ts_by_tf — alt_resolver and cross-TF are
        # gated below.

    # Alt-data resolver: only build it if the hypothesis actually references alt-data tokens.
    # Building the resolver does the expensive parquet re-read for the timestamp column —
    # skip it for SAP-001 / SAP-005 etc.
    alt_resolver = None
    altdata_numeric_resolver = None
    xsym_resolver = None
    # Bar timestamps are needed for any of the 3 resolvers — compute once.
    bar_ts_obj = None
    if (_hypothesis_uses_alt_data(hypothesis)
            or _hypothesis_uses_altdata_numeric(hypothesis)
            or _hypothesis_uses_xsym(hypothesis)):
        if ts_by_tf is not None and primary_tf in ts_by_tf:
            bar_ts_obj = ts_by_tf[primary_tf]
        else:
            bar_ts_obj = _load_bar_timestamps(ticker)
    if _hypothesis_uses_alt_data(hypothesis):
        if bar_ts_obj is not None and len(bar_ts_obj) == len(bars["close"]):
            alt_resolver = _AltDataResolver(ticker, bar_ts_obj)
        # else: tokens degrade to all-False (no signal)
    if _hypothesis_uses_altdata_numeric(hypothesis):
        if bar_ts_obj is not None and len(bar_ts_obj) == len(bars["close"]):
            try:
                altdata_numeric_resolver = _AltDataNumericResolver(ticker, bar_ts_obj)
            except Exception as e:
                print(f"  [altdata_numeric] init failed for {ticker}: {e}", flush=True)
        # else: tokens degrade to all-NaN (comparison yields 0 → no signal)
    if _hypothesis_uses_xsym(hypothesis):
        if bar_ts_obj is not None and len(bar_ts_obj) == len(bars["close"]):
            try:
                xsym_resolver = _XSymResolver(ticker, bar_ts_obj)
            except Exception as e:
                print(f"  [xsym] init failed for {ticker}: {e}", flush=True)

    pos = evaluate_hypothesis(
        bars, hypothesis, alt_data_resolver=alt_resolver,
        bars_by_tf=bars_by_tf, ts_by_tf=ts_by_tf, primary_tf=primary_tf,
        altdata_numeric_resolver=altdata_numeric_resolver,
        xsym_resolver=xsym_resolver,
    )
    rets = returns_from_position(bars, pos)
    full_sharpe = annualized_sharpe(rets, bars_per_year)
    wr, n_trades = win_rate_from_position(bars, pos)

    # Step 1: walk-forward folds
    n_obs = len(rets)
    eff_folds = n_folds if n_obs >= n_folds * 60 else max(4, min(n_folds, n_obs // 60))
    is_sharpes, oos_sharpes = [], []
    try:
        folds = rolling_walkforward_folds(n_obs, n_folds=eff_folds, train_frac=0.8, embargo_frac=0.005)
        for f in folds:
            is_sharpes.append(annualized_sharpe(rets[f.train_start:f.train_end], bars_per_year))
            oos_sharpes.append(annualized_sharpe(rets[f.test_start:f.test_end], bars_per_year))
    except Exception:
        folds = []
    is_med = float(np.nanmedian(is_sharpes)) if is_sharpes else float("nan")
    oos_med = float(np.nanmedian(oos_sharpes)) if oos_sharpes else float("nan")
    wfe = walk_forward_efficiency(is_med, oos_med)

    # Step 3: PBO over hypothesis perturbations (parameter grid = ±10% perturb)
    variants = _generate_perturbed_hypotheses(hypothesis)
    M = np.zeros((n_obs, len(variants)), dtype=np.float64)
    for j, v in enumerate(variants):
        try:
            # Re-use the same resolver across variants — perturbing role numerics
            # doesn't change the alt-data joins (which key off token names, not literals).
            # Multi-TF args also propagate so cross-TF tokens stay consistent
            # across perturbed-variant PBO evaluation.
            p_j = evaluate_hypothesis(
                bars, v, alt_data_resolver=alt_resolver,
                bars_by_tf=bars_by_tf, ts_by_tf=ts_by_tf, primary_tf=primary_tf,
                altdata_numeric_resolver=altdata_numeric_resolver,
                xsym_resolver=xsym_resolver,
            )
            M[:, j] = returns_from_position(bars, p_j)
        except Exception as e:
            M[:, j] = np.nan
    try:
        pbo_res = cscv_pbo(M, s_chunks=16) if n_obs >= 16 and M.shape[1] >= 2 else {"pbo": float("nan")}
    except Exception as e:
        pbo_res = {"pbo": float("nan"), "error": str(e)}

    # Step 4: DSR
    config_sharpes = np.array([annualized_sharpe(M[:, j], bars_per_year) for j in range(M.shape[1])])
    var_sr = float(np.nanvar(config_sharpes)) if np.sum(~np.isnan(config_sharpes)) > 1 else 1.0
    dsr_res = deflated_sharpe(full_sharpe, rets, n_trials=max(len(variants), 10), variance_of_sharpes=var_sr)

    # Step 5: Stability
    if np.sum(~np.isnan(config_sharpes)) > 1:
        mean_sr = float(np.nanmean(config_sharpes))
        std_sr = float(np.nanstd(config_sharpes))
        stab_cv = (std_sr / abs(mean_sr)) if abs(mean_sr) > 1e-6 else float("inf")
    else:
        mean_sr, std_sr, stab_cv = float("nan"), float("nan"), float("nan")
    stability = {
        "perturb_pcts": [-10, 0, 10],
        "grid_sharpes": config_sharpes.tolist(),
        "mean_sharpe": mean_sr,
        "std_sharpe": std_sr,
        "cv": stab_cv,
        "stable": (stab_cv < 0.5) if not np.isnan(stab_cv) else False,
    }

    # Step 6: Final holdout — last 10% of bars
    cut = int(0.9 * n_obs)
    holdout_sharpe = annualized_sharpe(rets[cut:], bars_per_year)
    insample_sharpe_pre = annualized_sharpe(rets[:cut], bars_per_year)

    return {
        "ticker": ticker,
        "status": "ok",
        "n_obs": int(n_obs),
        "n_trades": n_trades,
        "win_rate": wr,
        "full_sharpe": full_sharpe,
        "is_sharpe_median": is_med,
        "oos_sharpe_median": oos_med,
        "wfe": wfe,
        "is_sharpes": is_sharpes,
        "oos_sharpes": oos_sharpes,
        "pbo": pbo_res.get("pbo"),
        "pbo_n_combos": pbo_res.get("n_combos"),
        "dsr_prob": dsr_res.get("dsr_prob"),
        "dsr_sr0": dsr_res.get("sr0_threshold"),
        "stability": stability,
        "holdout_sharpe": holdout_sharpe,
        "insample_sharpe_pre_holdout": insample_sharpe_pre,
        "timeframe_stack": tf_stack if len(tf_stack) > 1 else None,
        "primary_tf": primary_tf,
        "multi_tf_load_notes": load_notes if load_notes else None,
    }


def run_hypothesis(
    hypothesis: dict,
    tickers: Optional[List[str]] = None,
    timeframe: str = "1d",
    holdout_after: str = "2025-01-01",
    n_folds: int = 12,
    utc_tag: Optional[str] = None,
    write_results: bool = True,
) -> dict:
    """Run 6-step validation on a strategy hypothesis. Returns aggregated dict.

    Mirrors lab.indicator_hardening_runner.run_one_indicator but the unit is the hypothesis,
    not a single indicator. Per-ticker results are stored, then aggregated across the cohort.

    Returns:
      {
        "hypothesis_id": str,
        "wfa": {is_med, oos_med, wfe_mean, ...},
        "pbo": float,
        "dsr": float,
        "stability": {cv_mean, stable_frac, ...},
        "holdout": {holdout_sharpe_mean, ...},
        "status": "TESTED_PRELIMINARY" | "TESTED_MULTIPLE_TICKERS" | "REJECTED" | "NO_DATA",
        "cohort": [per-ticker dicts...]
      }
    """
    # Validate input
    gate = validate_test_unit(hypothesis)
    if not gate["ok"]:
        return {
            "hypothesis_id": hypothesis.get("id", "<unknown>"),
            "status": "REJECTED_BY_GATE",
            "reason": gate["reason"],
            "missing_roles": gate["missing_roles"],
        }

    _ihr.set_timeframe(timeframe)
    bars_per_year = _ihr._state["bars_per_year"]
    if tickers is None:
        tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
                   "META", "TSLA", "JPM", "XOM", "JNJ"]
    sap_id = hypothesis.get("id", "SAP-NULL")
    utc_tag = utc_tag or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    t0 = time.time()
    cohort_rows = []
    for tk in tickers:
        try:
            row = run_hypothesis_for_ticker(hypothesis, tk, bars_per_year, n_folds=n_folds)
            row["ticker"] = tk
            cohort_rows.append(row)
            if row["status"] == "ok":
                print(
                    f"  {tk}: WR={row['win_rate']:.3f} N={row['n_trades']:>4d} "
                    f"WFE={row['wfe']:+.2f} PBO={row['pbo']:.3f} DSR={row['dsr_prob']:.3f}",
                    flush=True,
                )
            else:
                print(f"  {tk}: {row['status']}", flush=True)
        except Exception as e:
            traceback.print_exc()
            cohort_rows.append({"ticker": tk, "status": f"error:{e}"})

    valid = [r for r in cohort_rows if r.get("status") == "ok"]
    if not valid:
        agg = {
            "hypothesis_id": sap_id,
            "wfa": {}, "pbo": float("nan"), "dsr": float("nan"),
            "stability": {}, "holdout": {},
            "status": "NO_DATA",
            "cohort": cohort_rows,
        }
        return agg

    pbo_mean = float(np.nanmean([r["pbo"] for r in valid if r["pbo"] is not None]))
    dsr_mean = float(np.nanmean([r["dsr_prob"] for r in valid if r["dsr_prob"] is not None]))
    wr_mean = float(np.nanmean([r["win_rate"] for r in valid if r["win_rate"] is not None and not np.isnan(r["win_rate"])]))
    wfe_mean = float(np.nanmean([r["wfe"] for r in valid]))
    holdout_mean = float(np.nanmean([r["holdout_sharpe"] for r in valid if r["holdout_sharpe"] is not None and not np.isnan(r["holdout_sharpe"])]))
    stab_cv_mean = float(np.nanmean([r["stability"]["cv"] for r in valid if r.get("stability")]))
    stable_frac = float(np.mean([1.0 if r["stability"].get("stable") else 0.0 for r in valid if r.get("stability")]))
    n_trades_total = int(np.sum([r["n_trades"] for r in valid]))

    # Promotion rule (same as Phase 2): PBO < 0.15 AND DSR > 0.95 AND mean WR >= 0.50
    if pbo_mean < 0.15 and dsr_mean > 0.95 and wr_mean >= 0.50:
        status = "TESTED_MULTIPLE_TICKERS"
    elif (pbo_mean > 0.5) or (dsr_mean < 0.5) or (wr_mean < 0.45):
        status = "REJECTED"
    else:
        status = "TESTED_PRELIMINARY"

    agg = {
        "hypothesis_id": sap_id,
        "wfa": {
            "wfe_mean": wfe_mean,
            "is_sharpe_med_per_ticker": [r["is_sharpe_median"] for r in valid],
            "oos_sharpe_med_per_ticker": [r["oos_sharpe_median"] for r in valid],
        },
        "pbo": pbo_mean,
        "dsr": dsr_mean,
        "stability": {
            "cv_mean": stab_cv_mean,
            "stable_frac": stable_frac,
        },
        "holdout": {
            "holdout_sharpe_mean": holdout_mean,
            "holdout_after": holdout_after,
        },
        "status": status,
        "n_tickers_attempted": len(tickers),
        "n_tickers_ok": len(valid),
        "wr_mean": wr_mean,
        "n_trades_total": n_trades_total,
        "elapsed_sec": time.time() - t0,
        "timeframe": timeframe,
        "cohort": cohort_rows,
    }

    if write_results:
        for base in (RESULTS_LOCAL / sap_id / utc_tag, DRIVE_RESULTS / sap_id / utc_tag):
            try:
                base.mkdir(parents=True, exist_ok=True)
                with open(base / "summary.json", "w") as f:
                    json.dump(
                        {**agg, "hypothesis": hypothesis, "utc": utc_tag},
                        f, indent=2, default=str,
                    )
                # Cohort CSV
                csv_rows = []
                for r in cohort_rows:
                    csv_rows.append({
                        "ticker": r.get("ticker"),
                        "status": r.get("status"),
                        "n_obs": r.get("n_obs"),
                        "n_trades": r.get("n_trades"),
                        "win_rate": r.get("win_rate"),
                        "full_sharpe": r.get("full_sharpe"),
                        "is_sharpe_med": r.get("is_sharpe_median"),
                        "oos_sharpe_med": r.get("oos_sharpe_median"),
                        "wfe": r.get("wfe"),
                        "pbo": r.get("pbo"),
                        "dsr_prob": r.get("dsr_prob"),
                        "holdout_sharpe": r.get("holdout_sharpe"),
                    })
                pd.DataFrame(csv_rows).to_csv(base / "cohort.csv", index=False)
            except OSError as e:
                print(f"  [persist] failed at {base}: {e}", flush=True)
    return agg


# =============================================================================
# Task #56 (GOV_AWARE numeric upgrade) + #57 (cross-asset regime gates)
# 2026-05-29 — appended at file bottom to avoid collision with task #53's
# multi-TF refactor (primary_tf/bars_by_tf). Surgical hooks above only inject
# new keyword args to existing constructors/functions; the heavy lifting lives
# here. Both classes are lazy-loaded — only built if the hypothesis uses the
# tokens. NaN-safe degradation: missing source → NaN series → comparison → 0.
# =============================================================================


# ---- Token sets (lowercase, identifier-form — match what _RoleParser sees) ----

# Numeric alt-data tokens dispatched against lab.indicator_compute_altdata.
# The role parser will route these to _AltDataNumericResolver.resolve(key).
_ALTDATA_NUMERIC_TOKENS = frozenset({
    "form4_insider_cluster_score",
    "congress_lead_lag",
    "news_velocity_zscore",
    "dark_pool_divergence_z",
    "lobbying_intensity",
    "gov_contract_inflow",
    "8k_pulse",
    "eight_k_pulse",
})

# Cross-asset (cross-symbol) tokens dispatched against lab.indicator_compute_xsym.
# Compound tokens like ``vix_term_struct``, ``sector_rs_rank``, ``hyg_lqd_ratio``,
# ``spy_beta_60d``, ``abs_spy_beta_60d`` are normalized in _XSymResolver.resolve().
_XSYM_TOKENS = frozenset({
    "vix_term_struct", "vix_term_structure",
    "sector_rs_rank", "sector_relative_strength",
    "hyg_lqd_ratio",
    "spy_beta_60d", "spy_beta",
    "abs_spy_beta_60d",
    "vix_multiplied_atr",
    "spy_correlation",
    "sector_rotation_rank",
    "dxy_delta",
})


def _hypothesis_uses_altdata_numeric(h: dict) -> bool:
    """Cheap text scan: does any role expression reference a numeric alt-data token?
    Recurses into child_hypotheses + cross_asset_gate."""
    if not isinstance(h, dict):
        return False
    role_keys = ("regime_gate", "bias_filter", "trigger", "confirmation",
                 "timing", "exit", "no_trade", "cross_asset_gate")
    ident_re = re.compile(r"\b\d*[A-Za-z][A-Za-z0-9_]+\b")
    for k in role_keys:
        v = h.get(k)
        if not isinstance(v, str):
            continue
        for m in ident_re.findall(v):
            if m.lower() in _ALTDATA_NUMERIC_TOKENS:
                return True
    if isinstance(h.get("child_hypotheses"), list):
        for c in h["child_hypotheses"]:
            if _hypothesis_uses_altdata_numeric(c.get("hypothesis", {})):
                return True
            reg = c.get("regime")
            if isinstance(reg, str):
                for m in ident_re.findall(reg):
                    if m.lower() in _ALTDATA_NUMERIC_TOKENS:
                        return True
    return False


def _hypothesis_uses_xsym(h: dict) -> bool:
    """Cheap text scan: does any role expression reference a cross-asset token?
    Recurses into child_hypotheses + cross_asset_gate."""
    if not isinstance(h, dict):
        return False
    role_keys = ("regime_gate", "bias_filter", "trigger", "confirmation",
                 "timing", "exit", "no_trade", "cross_asset_gate")
    ident_re = re.compile(r"\b[A-Za-z][A-Za-z0-9_]+\b")
    for k in role_keys:
        v = h.get(k)
        if not isinstance(v, str):
            continue
        for m in ident_re.findall(v):
            if m.lower() in _XSYM_TOKENS:
                return True
    if isinstance(h.get("child_hypotheses"), list):
        for c in h["child_hypotheses"]:
            if _hypothesis_uses_xsym(c.get("hypothesis", {})):
                return True
            reg = c.get("regime")
            if isinstance(reg, str):
                for m in ident_re.findall(reg):
                    if m.lower() in _XSYM_TOKENS:
                        return True
    return False


class _AltDataNumericResolver:
    """Per-bar numeric resolver for alt-data tokens (task #56).

    Wraps lab.indicator_compute_altdata's continuous-value functions and exposes
    them as ``resolve(token: str) -> np.ndarray[float]`` of length n_bars.

    Differs from _AltDataResolver (which returns bool event flags) — this one
    returns continuous scores so the role parser can do ``token > threshold``
    style numeric comparisons.

    Per-token cache: same instance never recomputes the same numeric series.
    """

    def __init__(self, ticker: str, bar_timestamps: "pd.DatetimeIndex"):
        self.ticker = ticker.upper()
        # Convert tz-aware → naive (the altdata functions tolerate either but
        # normalize internally).
        ts = pd.DatetimeIndex(bar_timestamps)
        if ts.tz is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        self.bar_ts = ts
        self.n = len(ts)
        self._series_cache: Dict[str, np.ndarray] = {}
        self._diagnostics: Dict[str, Any] = {"ticker": self.ticker, "n_bars": self.n}
        # Lazy-import the altdata module (avoids cycle when running smoke tests).
        try:
            import indicator_compute_altdata as _altmod  # type: ignore
        except ImportError:
            try:
                from lab import indicator_compute_altdata as _altmod  # type: ignore
            except ImportError as e:
                self._altmod = None
                self._diagnostics["import_error"] = repr(e)
                return
        self._altmod = _altmod

    def knows(self, token: str) -> bool:
        return token.lower() in _ALTDATA_NUMERIC_TOKENS

    def resolve(self, token: str) -> np.ndarray:
        """Return length-n float array. NaN where source is missing."""
        key = token.lower()
        if key in self._series_cache:
            return self._series_cache[key]
        if self._altmod is None:
            arr = np.full(self.n, np.nan, dtype=np.float64)
            self._series_cache[key] = arr
            return arr
        # Map identifier → function name in indicator_compute_altdata.
        # ``8k_pulse`` is the user-facing token; the function is ``eight_k_pulse``
        # (Python identifier can't start with a digit).
        fn_name = {
            "8k_pulse": "eight_k_pulse",
        }.get(key, key)
        fn = getattr(self._altmod, fn_name, None)
        if fn is None:
            arr = np.full(self.n, np.nan, dtype=np.float64)
            self._diagnostics.setdefault("unknown_fn", []).append(token)
            self._series_cache[key] = arr
            return arr
        try:
            ser = fn(self.ticker, self.bar_ts)
        except Exception as e:  # noqa: BLE001
            self._diagnostics.setdefault("errors", []).append({"token": token, "err": repr(e)})
            arr = np.full(self.n, np.nan, dtype=np.float64)
            self._series_cache[key] = arr
            return arr
        # Coerce to plain ndarray, length-n. If lengths mismatch, align by
        # reindex on the bar timestamps (defensive — usually returns already-aligned).
        if isinstance(ser, pd.Series):
            if len(ser) != self.n:
                try:
                    ser = ser.reindex(self.bar_ts, method="ffill")
                except Exception:
                    pass
            arr = ser.to_numpy(dtype=np.float64, na_value=np.nan)
        else:
            arr = np.asarray(ser, dtype=np.float64)
        if arr.shape != (self.n,):
            # Last-resort: pad or truncate with NaN to keep parser invariant.
            fixed = np.full(self.n, np.nan, dtype=np.float64)
            m = min(arr.shape[0], self.n)
            fixed[:m] = arr[:m]
            arr = fixed
        self._series_cache[key] = arr
        return arr

    def diagnostics(self) -> Dict[str, Any]:
        return dict(self._diagnostics)


class _XSymResolver:
    """Per-bar numeric resolver for cross-asset tokens (task #57).

    Wraps lab.indicator_compute_xsym's cross-symbol functions and exposes them
    as ``resolve(token: str) -> np.ndarray[float]`` of length n_bars, aligned to
    the target ticker's bar timestamps.

    Token aliases (user-facing ↔ xsym function):
        vix_term_struct      → vix_term_structure(target_symbol=ticker)
        sector_rs_rank       → sector_rotation_rank(symbol_sector=<sector ETF>, target_symbol=ticker)
        hyg_lqd_ratio        → hyg_lqd_ratio(target_symbol=ticker)
        spy_beta_60d         → spy_beta(target=ticker, n=60)
        abs_spy_beta_60d     → abs(spy_beta(..., n=60))
        vix_multiplied_atr   → vix_multiplied_atr(target=ticker)
        spy_correlation      → spy_correlation(target=ticker)
        sector_rotation_rank → sector_rotation_rank(...)
        dxy_delta            → dxy_delta(target_symbol=ticker)
        sector_relative_strength → sector_relative_strength(target=ticker, sector_etf=<inferred>)

    Missing reference symbols (e.g. VIX not in parquet store) → NaN series.
    """

    # Best-effort static map ticker → sector ETF. Falls back to XLK if unknown.
    # The championship_metadata.enrich_metadata returns the canonical sector for
    # known S&P 500 tickers; we use a small map here to avoid a circular dep.
    _DEFAULT_SECTOR_ETF = "XLK"

    def __init__(self, ticker: str, bar_timestamps: "pd.DatetimeIndex"):
        self.ticker = ticker.upper()
        ts = pd.DatetimeIndex(bar_timestamps)
        if ts.tz is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        self.bar_ts = ts
        self.n = len(ts)
        self._series_cache: Dict[str, np.ndarray] = {}
        self._bars_by_symbol: Dict[str, "pd.DataFrame"] = {}
        self._diagnostics: Dict[str, Any] = {"ticker": self.ticker, "n_bars": self.n}
        try:
            import indicator_compute_xsym as _xsymmod  # type: ignore
        except ImportError:
            try:
                from lab import indicator_compute_xsym as _xsymmod  # type: ignore
            except ImportError as e:
                self._xsymmod = None
                self._diagnostics["import_error"] = repr(e)
                return
        self._xsymmod = _xsymmod

    def knows(self, token: str) -> bool:
        return token.lower() in _XSYM_TOKENS

    def _sector_for_ticker(self) -> str:
        """Cheap sector lookup: try championship_metadata.enrich_metadata, fall back to XLK."""
        try:
            try:
                import championship_metadata as _cm  # type: ignore
            except ImportError:
                from lab import championship_metadata as _cm  # type: ignore
            meta = _cm.enrich_metadata(self.ticker, formatted=False)
            sector = (meta or {}).get("sector")
            # Map GICS sector → sector ETF (best-effort)
            sector_map = {
                "Information Technology": "XLK",
                "Technology": "XLK",
                "Financials": "XLF",
                "Energy": "XLE",
                "Health Care": "XLV",
                "Healthcare": "XLV",
                "Utilities": "XLU",
                "Consumer Discretionary": "XLY",
                "Consumer Staples": "XLP",
                "Industrials": "XLI",
                "Materials": "XLB",
                "Real Estate": "XLRE",
                "Communication Services": "XLC",
            }
            if sector in sector_map:
                return sector_map[sector]
        except Exception:
            pass
        return self._DEFAULT_SECTOR_ETF

    def _align_to_bars(self, ser: "pd.Series") -> np.ndarray:
        """Reindex an xsym output series onto the ticker's bar timestamps (ffill).
        Returns length-n float64 array.
        """
        if ser is None or not isinstance(ser, pd.Series):
            return np.full(self.n, np.nan, dtype=np.float64)
        # Normalize index tz
        idx = ser.index
        if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
            ser = ser.copy()
            ser.index = idx.tz_convert("UTC").tz_localize(None)
        try:
            aligned = ser.reindex(self.bar_ts, method="ffill")
        except Exception:
            aligned = ser.reindex(self.bar_ts)
        arr = aligned.to_numpy(dtype=np.float64, na_value=np.nan)
        if arr.shape != (self.n,):
            fixed = np.full(self.n, np.nan, dtype=np.float64)
            m = min(arr.shape[0], self.n)
            fixed[:m] = arr[:m]
            arr = fixed
        return arr

    def resolve(self, token: str) -> np.ndarray:
        key = token.lower()
        if key in self._series_cache:
            return self._series_cache[key]
        if self._xsymmod is None:
            arr = np.full(self.n, np.nan, dtype=np.float64)
            self._series_cache[key] = arr
            return arr
        try:
            ser = self._compute(key)
        except Exception as e:  # noqa: BLE001
            self._diagnostics.setdefault("errors", []).append({"token": token, "err": repr(e)})
            ser = None
        if ser is None:
            arr = np.full(self.n, np.nan, dtype=np.float64)
            self._series_cache[key] = arr
            return arr
        arr = self._align_to_bars(ser)
        # Handle special abs() wrapper
        if key == "abs_spy_beta_60d":
            arr = np.abs(arr)
        self._series_cache[key] = arr
        return arr

    def _compute(self, key: str) -> Optional["pd.Series"]:
        """Dispatch the lowercase token to the xsym function. Returns a pd.Series
        (NOT yet aligned to bar_ts; _align_to_bars does that)."""
        m = self._xsymmod
        bbs = self._bars_by_symbol  # reused across calls so each ref-symbol is loaded once
        if key in ("vix_term_struct", "vix_term_structure"):
            return m.vix_term_structure(bbs, target_symbol=self.ticker)
        if key == "hyg_lqd_ratio":
            return m.hyg_lqd_ratio(bbs, target_symbol=self.ticker)
        if key in ("spy_beta_60d", "spy_beta", "abs_spy_beta_60d"):
            return m.spy_beta(bbs, self.ticker, n=60)
        if key == "vix_multiplied_atr":
            return m.vix_multiplied_atr(bbs, self.ticker)
        if key == "spy_correlation":
            return m.spy_correlation(bbs, self.ticker)
        if key == "dxy_delta":
            return m.dxy_delta(bbs, target_symbol=self.ticker)
        if key in ("sector_rs_rank", "sector_rotation_rank"):
            sector_etf = self._sector_for_ticker()
            return m.sector_rotation_rank(bbs, symbol_sector=sector_etf,
                                           target_symbol=self.ticker)
        if key == "sector_relative_strength":
            sector_etf = self._sector_for_ticker()
            return m.sector_relative_strength(bbs, self.ticker, sector_etf=sector_etf)
        return None

    def diagnostics(self) -> Dict[str, Any]:
        return dict(self._diagnostics)


# =============================================================================
# End task #56 + #57 additions
# =============================================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hypothesis-id", default="SAP-001",
                    help="Look up the hypothesis by id in example_hypotheses.HYPOTHESES")
    ap.add_argument("--tickers", nargs="+",
                    default=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
                             "META", "TSLA", "JPM", "XOM", "JNJ"])
    ap.add_argument("--timeframe", default="1d", choices=["1d", "5min"])
    ap.add_argument("--n-folds", type=int, default=12)
    ap.add_argument("--holdout-after", default="2025-01-01")
    args = ap.parse_args()

    from example_hypotheses import HYPOTHESES_BY_ID
    if args.hypothesis_id not in HYPOTHESES_BY_ID:
        print(f"Unknown hypothesis id {args.hypothesis_id}. Available: {list(HYPOTHESES_BY_ID)}")
        return 2
    hyp = HYPOTHESES_BY_ID[args.hypothesis_id]
    print(f"\n=== {args.hypothesis_id} === tickers={args.tickers}", flush=True)
    res = run_hypothesis(
        hyp, tickers=args.tickers, timeframe=args.timeframe,
        n_folds=args.n_folds, holdout_after=args.holdout_after,
    )
    print("\n=== RESULT ===")
    print(json.dumps(
        {k: v for k, v in res.items() if k != "cohort"},
        indent=2, default=str,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
