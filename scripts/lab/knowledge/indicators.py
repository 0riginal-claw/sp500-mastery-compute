"""
Indicator-catalog knowledge — 107 indicators, redundancy clusters, recommended settings.

Loads from `Tech0/Data Master/BackTests & Data/indicator_mastery_index.csv` (38 curated rows)
and `indicator_manifest.json` (full 107). Falls back to embedded constants from the
insights playbook when the live CSV isn't reachable via Drive FUSE.

Top funcs:
  all_indicators()              — list of all curated rows as dicts
  by_status(status)             — filter to rows matching a status (e.g. 'TESTED_MULTIPLE_TICKERS')
  by_id(id)                     — single row by indicator_id
  with_recorded_win_rate()      — only rows with numeric win_rate
  primary_for_breakouts()       — flagged PRIMARY or HIGH for breakouts
  redundancy_clusters()         — dict of cluster_name → members  (from overfit research)
  is_redundant(a, b)            — True if same redundancy cluster
  recommended_settings(tf)      — default params per timeframe (5min S&P)
  banned_combinations()         — explicit math-equivalent bans (Williams %R = StochK etc.)
  tested_multiple_tickers()     — the small validated set
"""
from __future__ import annotations

import csv
import io
import json
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

DRIVE_BASE = "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive"
INDEX_CSV = f"{DRIVE_BASE}/Tech0/Data Master/BackTests & Data/indicator_mastery_index.csv"
MANIFEST_JSON = f"{DRIVE_BASE}/Tech0/Data Master/BackTests & Data/indicator_manifest.json"
LAB_MIRROR_DIR = f"{DRIVE_BASE}/AI-Tools/research-lab/data_inventory"


def _read_csv_safe(path: str) -> Optional[List[Dict[str, str]]]:
    """Drive FUSE may not see Tech0; return None on ENOENT, raise on other errors."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        return None


def _read_json_safe(path: str) -> Optional[Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


# Embedded fallback — distilled from the playbook so callers always get something useful
# even when Drive FUSE can't see Tech0 (a known intermittent issue this Mac).
_EMBEDDED_WIN_RATES = {
    1: ("Donchian Channel (20) UP", 0.5474, "AAPL · also 0.753 on baseline cohort"),
    2: ("Donchian Channel (20) DN", 0.4725, "AAPL only"),
    3: ("Donchian Channel (40) UP", 0.772, "optimized_cycle001"),
    4: ("Volume Expansion", 0.5678, "ir4 test"),
    6: ("RSI(14)", 0.5292, "AAPL-neutral"),
    17: ("ADX(14)", 0.5835, "ir5 test"),
    18: ("CMF(21)", 0.5862, "ir6 test"),
}

_EMBEDDED_PRIMARY_BREAKOUT_IDS = [1, 3, 4, 10, 16, 17, 27]  # Donchian UP/40, VolExp, VWAP, ChopIdx, ADX, OpeningRange

_REDUNDANCY_CLUSTERS = {
    "momentum_oscillators": {
        "members": ["RSI", "Stochastic", "Williams %R", "CCI", "MFI"],
        "correlation_range": "0.70-0.95",
        "winner": "RSI(14)",
        "reason": "Smoothest; most research; already validated"
    },
    "trend_ma_family": {
        "members": ["SMA", "EMA", "DEMA", "TEMA", "MACD zero-cross"],
        "correlation_range": "0.90-1.00",
        "winner": "EMA(9)/EMA(21) pair",
        "reason": "Faster than SMA; MACD zero-cross is mathematically identical to EMA cross"
    },
    "volatility_bands": {
        "members": ["Bollinger Bands", "Keltner Channels", "Donchian Channels"],
        "correlation_range": "~0.80",
        "winner": "context-dependent",
        "reason": "BB for mean-reversion · Donchian for breakouts · Keltner only for TTM Squeeze setup"
    },
    "directional_volume": {
        "members": ["OBV", "Chaikin MF", "Accumulation/Distribution"],
        "correlation_range": "~0.90",
        "winner": "OBV",
        "reason": "Simplest; longest track record"
    },
    "pure_volume": {
        "members": ["Volume SMA", "Volume EMA", "VWAP"],
        "correlation_range": "least correlated",
        "winner": "VWAP",
        "reason": "Price-weighted; daily reset; institutional benchmark"
    }
}

_BANNED_COMBOS = [
    {
        "name": "Williams %R + Fast Stochastic %K",
        "rule": "Williams %R = (StochK × -1) - 100",
        "verdict": "Mathematically identical, never use both"
    },
    {
        "name": "MACD zero-line cross + EMA(12)/EMA(26) cross",
        "rule": "MACD = 0 ⟺ EMA(12) = EMA(26)",
        "verdict": "Same event, two names — pick one"
    }
]

_RECOMMENDED_SETTINGS_5MIN_SP500 = {
    "CMF": {"period": 21, "threshold": 0.05, "scalp_variant": "period 10-14"},
    "Keltner Channels": {"ema": 20, "atr_mult": 1.5, "fast_variant": "10 EMA / 1.0 ATR"},
    "ADX": {"period": 14, "threshold": 20, "scalp_variant": "period 7-10"},
    "Aroon": {"period": 25, "intraday_variant": "14-20"},
    "Parabolic SAR": {"step": 0.02, "max": 0.20, "scalp_variant": "0.03/0.30"},
    "CCI": {"period": 20, "normal_thresholds": (-100, 100), "high_vol_thresholds": (-200, 200)},
    "Williams %R": {"period": 14, "scalp_variant": "period 10 with -90/-10 thresholds"},
    "MACD Histogram": {"raschke": (5, 13, 1), "scalp": (3, 10, 1), "smoother": (8, 17, 9)},
    "OBV": {"params": "none — apply 20-period SMA overlay for trend"},
    "Elder Ray": {"ema": 13, "scalp_variant": "EMA 8-10"}
}


@lru_cache(maxsize=1)
def all_indicators() -> List[Dict[str, str]]:
    """Return all curated rows from indicator_mastery_index.csv (typically 38)."""
    rows = _read_csv_safe(INDEX_CSV)
    if rows is None:
        # Drive FUSE blind to Tech0 — try lab mirror
        for fname in Path(LAB_MIRROR_DIR).rglob("indicator_mastery_index.csv"):
            rows = _read_csv_safe(str(fname))
            if rows:
                break
    return rows or []


@lru_cache(maxsize=1)
def manifest_full() -> List[Dict[str, Any]]:
    """Return full 107-indicator manifest (every indicator the engine has implementations for)."""
    return _read_json_safe(MANIFEST_JSON) or []


def by_status(status: str) -> List[Dict[str, str]]:
    """Filter curated rows to a specific status (e.g. 'TESTED_MULTIPLE_TICKERS')."""
    return [r for r in all_indicators() if r.get("status") == status]


def by_id(indicator_id: int) -> Optional[Dict[str, str]]:
    """Single row by indicator_id (1-based, matches the CSV)."""
    sid = str(indicator_id)
    for r in all_indicators():
        if r.get("indicator_id") == sid:
            return r
    return None


def with_recorded_win_rate() -> List[Dict[str, Any]]:
    """
    All indicators with a numeric win_rate. Falls back to embedded constants if Drive is
    unreachable. Each record: indicator_id, name, win_rate, context.
    """
    rows = all_indicators()
    if rows:
        out: List[Dict[str, Any]] = []
        for r in rows:
            wr = r.get("win_rate", "")
            if wr and wr not in ("N/A", "NOT_RECORDED", "NEEDS_DATA"):
                try:
                    # Some cells are like "0.5474(AAPL)/0.753(baseline)" — keep raw string
                    out.append({
                        "indicator_id": int(r["indicator_id"]),
                        "name": r.get("indicator_name", ""),
                        "win_rate_raw": wr,
                        "total_signals": r.get("total_signals", ""),
                        "status": r.get("status", ""),
                    })
                except (KeyError, ValueError):
                    continue
        if out:
            return out
    # fallback
    return [
        {"indicator_id": k, "name": v[0], "win_rate_raw": str(v[1]), "context": v[2]}
        for k, v in sorted(_EMBEDDED_WIN_RATES.items())
    ]


def primary_for_breakouts() -> List[Dict[str, str]]:
    """All indicators flagged PRIMARY / HIGH / CONFIRMED for 'improves_breakouts'."""
    rows = all_indicators()
    if rows:
        return [r for r in rows
                if r.get("improves_breakouts") in ("PRIMARY", "HIGH", "CONFIRMED")]
    # fallback to embedded ids
    return [{"indicator_id": str(i), "indicator_name": f"id-{i}",
             "improves_breakouts": "PRIMARY/HIGH (embedded)"}
            for i in _EMBEDDED_PRIMARY_BREAKOUT_IDS]


def tested_multiple_tickers() -> List[Dict[str, str]]:
    """The small validated set — tested beyond AAPL."""
    return by_status("TESTED_MULTIPLE_TICKERS")


def redundancy_clusters() -> Dict[str, Dict[str, Any]]:
    """
    Correlation-based redundancy clusters from the overfit research.
    Pick at most ONE per cluster when generating new candidates.
    """
    return dict(_REDUNDANCY_CLUSTERS)


def is_redundant(a: str, b: str) -> bool:
    """True if a and b live in the same redundancy cluster."""
    a_l, b_l = a.lower(), b.lower()
    for cluster in _REDUNDANCY_CLUSTERS.values():
        members_l = [m.lower() for m in cluster["members"]]
        # token-substring match (e.g. 'RSI' inside 'RSI(14)')
        a_in = any(m in a_l or a_l in m for m in members_l)
        b_in = any(m in b_l or b_l in m for m in members_l)
        if a_in and b_in:
            return True
    return False


def banned_combinations() -> List[Dict[str, str]]:
    """Mathematically equivalent pairs to never use together."""
    return list(_BANNED_COMBOS)


def recommended_settings(timeframe: str = "5min") -> Dict[str, Dict[str, Any]]:
    """Default parameters per indicator for a given timeframe. Currently only 5min S&P 500."""
    if timeframe != "5min":
        return {}
    return dict(_RECOMMENDED_SETTINGS_5MIN_SP500)


def _clear_cache():
    all_indicators.cache_clear()
    manifest_full.cache_clear()


# Quick CLI: `python -m lab.knowledge.indicators`
if __name__ == "__main__":
    print(f"Curated catalog rows: {len(all_indicators())}")
    print(f"Full manifest indicators: {len(manifest_full())}")
    print(f"With recorded win_rate: {len(with_recorded_win_rate())}")
    print(f"PRIMARY/HIGH for breakouts: {len(primary_for_breakouts())}")
    print(f"Redundancy clusters: {list(redundancy_clusters().keys())}")
    print(f"Banned combinations: {[b['name'] for b in banned_combinations()]}")
