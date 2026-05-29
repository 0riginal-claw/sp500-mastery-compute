"""
Multi-timeframe knowledge — 11 timeframes, confluence partners, noise levels.

Loads from `Tech0/Data Master/BackTests & Data/_multi_timeframe_research/timeframe_table.csv`
(produced by Mission 13). Falls back to embedded skeleton if the live CSV isn't reachable
or isn't generated yet (background sub-agent in flight at module-creation time).

Top funcs:
  all_timeframes()              — list of all 11 timeframe records
  row(tf)                       — single timeframe record
  confluence_partners(tf)       — list of 2-3 partner timeframes
  noise_level(tf)               — 'high' / 'medium' / 'low'
  suggested_indicators(tf)      — list of indicator_ids to overlay on this timeframe
  tier_for(tf)                  — 'bias' / 'structure' / 'confirmation' / 'execution'
  workflow_checklist()          — ordered top-down workflow
"""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

DRIVE_BASE = "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive"
TABLE_CSV = f"{DRIVE_BASE}/Tech0/Data Master/BackTests & Data/_multi_timeframe_research/timeframe_table.csv"
LAB_MIRROR = f"{DRIVE_BASE}/AI-Tools/research-lab/data_inventory/multi_timeframe_research/timeframe_table.csv"

TIMEFRAMES = ["1min", "5min", "10min", "15min", "30min", "45min", "1hr", "4hr", "8hr", "12hr", "1day"]

# Tier mapping (from the spec's three-tier mental model)
_TIER_MAP = {
    "1day": "bias", "12hr": "bias", "8hr": "bias",
    "4hr": "structure", "1hr": "structure",
    "45min": "confirmation", "30min": "confirmation", "15min": "confirmation",
    "10min": "execution", "5min": "execution",
    "1min": "execution"
}

# Embedded skeleton — used when timeframe_table.csv isn't yet generated.
# Confluence partner choices follow the workflow:
#   execution timeframes pair with confirmation;
#   confirmation timeframes pair with structure;
#   structure timeframes pair with bias.
_EMBEDDED_SKELETON: Dict[str, Dict[str, Any]] = {
    "1min": {"main_purpose": "Execution + immediate order flow",
              "noise_level": "high",
              "confluence_partners": ["5min", "15min"]},
    "5min": {"main_purpose": "Short-term intraday structure + entry timing",
              "noise_level": "high",
              "confluence_partners": ["1min", "15min", "1hr"]},
    "10min": {"main_purpose": "Bridge between fast execution and cleaner structure",
               "noise_level": "medium",
               "confluence_partners": ["5min", "30min"]},
    "15min": {"main_purpose": "Major intraday structure + trend confirmation",
               "noise_level": "medium",
               "confluence_partners": ["5min", "1hr"]},
    "30min": {"main_purpose": "Stronger intraday bias + larger candle structure",
               "noise_level": "medium",
               "confluence_partners": ["15min", "1hr"]},
    "45min": {"main_purpose": "Intermediate intraday-to-swing context",
               "noise_level": "medium",
               "confluence_partners": ["30min", "1hr"]},
    "1hr": {"main_purpose": "Primary intraday directional bias",
              "noise_level": "low",
              "confluence_partners": ["15min", "4hr"]},
    "4hr": {"main_purpose": "Macro intraday/swing context + dominant trend",
              "noise_level": "low",
              "confluence_partners": ["1hr", "1day"]},
    "8hr": {"main_purpose": "Broader market regime + multi-session structure",
              "noise_level": "low",
              "confluence_partners": ["4hr", "1day"]},
    "12hr": {"main_purpose": "High-level trend & condition filter",
               "noise_level": "low",
               "confluence_partners": ["8hr", "1day"]},
    "1day": {"main_purpose": "Strongest context + overall trend",
              "noise_level": "low",
              "confluence_partners": ["4hr", "1hr"]},
}

_WORKFLOW = [
    ("1day", "What's the dominant trend? Above or below 20/50/200 SMA?"),
    ("12hr/8hr", "Is the market in a regime (trend / range / transition)?"),
    ("4hr/1hr", "Where are the major S/R + supply/demand zones? What's the intraday bias?"),
    ("45min/30min/15min", "Is intraday structure confirming the higher-tf bias? BOS, retest, HL/LH?"),
    ("10min/5min", "Is the setup forming cleanly? VWAP reaction, volume confirmation, candle close?"),
    ("1min", "Execute. Entry trigger, stop placement, R:R check.")
]


def _read_table_csv() -> Optional[List[Dict[str, str]]]:
    """Try the Tech0 path first, then the lab mirror."""
    for p in (TABLE_CSV, LAB_MIRROR):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return list(csv.DictReader(f))
        except FileNotFoundError:
            continue
    return None


@lru_cache(maxsize=1)
def _load() -> Dict[str, Dict[str, Any]]:
    rows = _read_table_csv()
    if not rows:
        return dict(_EMBEDDED_SKELETON)
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        tf = r.get("timeframe", "").strip()
        if not tf:
            continue
        partners = r.get("confluence_partners", "")
        ind_ids = r.get("suggested_indicator_ids", "")
        out[tf] = {
            "main_purpose": r.get("main_purpose", ""),
            "info_provided": r.get("info_provided", ""),
            "how_it_improves_trades": r.get("how_it_improves_trades", ""),
            "mistakes_if_ignored": r.get("mistakes_if_ignored", ""),
            "best_for_entry": r.get("best_for_entry", ""),
            "best_for_stop": r.get("best_for_stop", ""),
            "best_for_target": r.get("best_for_target", ""),
            "best_for_bias": r.get("best_for_bias", ""),
            "confluence_partners": [p.strip() for p in partners.split(";") if p.strip()],
            "noise_level": r.get("noise_level", ""),
            "suggested_indicator_ids": [
                int(x) for x in ind_ids.split(";") if x.strip().isdigit()
            ],
        }
    return out


def all_timeframes() -> List[Dict[str, Any]]:
    """Return all 11 timeframe records (in canonical order)."""
    data = _load()
    return [{"timeframe": tf, **data.get(tf, {})} for tf in TIMEFRAMES]


def row(tf: str) -> Optional[Dict[str, Any]]:
    """Single timeframe record. tf must be one of TIMEFRAMES."""
    data = _load()
    if tf not in data:
        return None
    return {"timeframe": tf, **data[tf]}


def confluence_partners(tf: str) -> List[str]:
    """2-3 timeframes that pair best with `tf`."""
    r = row(tf)
    return list(r.get("confluence_partners", [])) if r else []


def noise_level(tf: str) -> str:
    """'high' / 'medium' / 'low'."""
    r = row(tf)
    return r.get("noise_level", "") if r else ""


def suggested_indicators(tf: str) -> List[int]:
    """List of indicator_ids (from indicator_mastery_index.csv) to overlay on this timeframe."""
    r = row(tf)
    return list(r.get("suggested_indicator_ids", [])) if r else []


def tier_for(tf: str) -> str:
    """'bias' / 'structure' / 'confirmation' / 'execution' per the three-tier model."""
    return _TIER_MAP.get(tf, "")


def timeframes_by_tier(tier: str) -> List[str]:
    """All timeframes in a given tier."""
    return [tf for tf, t in _TIER_MAP.items() if t == tier]


def workflow_checklist() -> List[Dict[str, str]]:
    """Top-down workflow: 1day → 12hr/8hr → 4hr/1hr → ... → 1min."""
    return [{"step": s, "question": q} for s, q in _WORKFLOW]


def _clear_cache():
    _load.cache_clear()


if __name__ == "__main__":
    print(f"Timeframes loaded: {len(all_timeframes())}")
    print(f"Tiers: {sorted(set(_TIER_MAP.values()))}")
    print(f"Workflow steps: {len(workflow_checklist())}")
    for tf in TIMEFRAMES:
        partners = confluence_partners(tf)
        tier = tier_for(tf)
        print(f"  {tf:6s} [{tier:13s}]  partners={partners}")
