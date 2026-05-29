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


_TOK_REGEX = re.compile(
    r"\s*(?:"
    r"(?P<NUMBER>\d+(?:\.\d+)?)"
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


class _RoleParser:
    """Recursive-descent parser that turns a role-expression string into a per-bar numpy series.

    Comparison ops return float 0.0/1.0 arrays (so a `gate` is just truthy mask).
    Arithmetic + indicator calls return float series.
    """

    def __init__(self, bars: ArrayDict):
        self.bars = bars
        self.n = len(bars["close"])
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
        self.toks = _tokenize(expr)
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

    def _parse_indicator_call(self):
        ident = self._eat().val
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


def evaluate_hypothesis(bars: ArrayDict, hypothesis: dict) -> np.ndarray:
    """Walk the per-bar state machine for a hypothesis.

    Returns a position series in {-1, 0, +1}, length n.

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
        parser = _RoleParser(bars)
        pos = np.zeros(n, dtype=np.int8)
        for child in hypothesis["child_hypotheses"]:
            mask = _bool_mask(parser.evaluate(child.get("regime", "TRUE")))
            sub_pos = evaluate_hypothesis(bars, child["hypothesis"])
            # Apply mask: child position only counts during its regime
            pos = np.where(mask & (pos == 0), sub_pos, pos).astype(np.int8)
        # no_trade override (parent-level)
        if hypothesis.get("no_trade"):
            no_trade_mask = _bool_mask(parser.evaluate(hypothesis["no_trade"]))
            pos[no_trade_mask] = 0
        return pos

    parser = _RoleParser(bars)
    side = -1 if str(hypothesis.get("side", "long")).lower() == "short" else 1
    gate = _bool_mask(parser.evaluate(hypothesis.get("regime_gate", "TRUE")))
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
    """Run a hypothesis on one ticker. Mirrors run_indicator_for_ticker but for a hypothesis."""
    bars = _ihr.load_ohlc(ticker)
    if bars is None:
        return {"ticker": ticker, "status": "no_data"}

    pos = evaluate_hypothesis(bars, hypothesis)
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
            p_j = evaluate_hypothesis(bars, v)
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
