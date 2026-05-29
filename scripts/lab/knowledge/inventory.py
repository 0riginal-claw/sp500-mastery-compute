"""
Inventory knowledge — mastered ticker list, active cycle, Tech0 top-level structure.

Loads from:
  - `AI-Tools/research-lab/data_inventory/mastered_tickers_20260528.txt`
  - `Tech0/Data Master/universe/active_stack.json`
  - `Tech0/_master_inventory_20260528.md`

Top funcs:
  mastered()                    — list of 502 mastered S&P 500 tickers
  is_mastered(ticker)           — bool
  active_cycle()                — e.g. 'cycle059_force_sweep'
  active_stack_path()           — relative path within Tech0 to the active stack CSV
  tech0_top_level()             — high-level summary of Tech0 structure
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

DRIVE_BASE = "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive"
TICKERS_TXT_LAB = f"{DRIVE_BASE}/AI-Tools/research-lab/data_inventory/mastered_tickers_20260528.txt"
TICKERS_TXT_TECH0 = f"{DRIVE_BASE}/Tech0/_mastered_tickers_20260528.txt"
ACTIVE_STACK_JSON = f"{DRIVE_BASE}/Tech0/Data Master/universe/active_stack.json"

# Embedded fallback — captured at module-write time (2026-05-28)
_FALLBACK_ACTIVE = {
    "cycle_n": 59,
    "cycle_dir": "research/active/cycle059_force_sweep",
    "stack_path": "research/active/cycle059_force_sweep/_active_stack_C059_ALL_FINAL.csv",
    "set_at": "2026-05-05T19:00:00",
    "set_by": "Phase 2 migration",
    "comment": "Phase 2 complete: all live-system paths migrated to new layout."
}


def _try_read(path: str) -> Optional[str]:
    try:
        return Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


@lru_cache(maxsize=1)
def mastered() -> List[str]:
    """Sorted list of 502 mastered S&P 500 tickers (with 3 meta entries stripped)."""
    txt = _try_read(TICKERS_TXT_LAB) or _try_read(TICKERS_TXT_TECH0)
    if not txt:
        return []
    return sorted(
        line.strip() for line in txt.splitlines()
        if line.strip() and not line.strip().startswith("_")
    )


def is_mastered(ticker: str) -> bool:
    """Is this ticker in the mastered set?"""
    return ticker.upper() in set(mastered())


@lru_cache(maxsize=1)
def _load_active() -> Dict[str, str]:
    txt = _try_read(ACTIVE_STACK_JSON)
    if not txt:
        return dict(_FALLBACK_ACTIVE)
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return dict(_FALLBACK_ACTIVE)


def active_cycle() -> str:
    """Active strategy research cycle name (e.g. 'cycle059_force_sweep')."""
    d = _load_active()
    cycle_dir = d.get("cycle_dir", "")
    return cycle_dir.rsplit("/", 1)[-1] if cycle_dir else f"cycle{d.get('cycle_n', '?'):03d}"


def active_stack_path() -> str:
    """Relative path within Tech0 to the active stack CSV."""
    return _load_active().get("stack_path", "")


def tech0_top_level() -> Dict[str, str]:
    """Top-level summary of Tech0 folder structure (from this session's inventory)."""
    return {
        "Research Master": "Edgar/, GovTrades/, research/, strategy_intelligence_system/",
        "Data Master": "BackTests & Data/, universe/, data/",
        "Code Master": "system/{trader, broker, portfolio, runtime-gate, agents}, plugins/, bin/, scripts/, tools/",
        "Ops Master": "operations/, state/",
        "Archive Master": "archive_dead/, gpt4all_project/",
        "Docs Master": "docs/, prompts/",
        "Photis": "placeholder (Gabriel content still in Ph0tis/)",
        "root_docs_count": "22 (INDEX.md, paths.toml, AGENTS.md, CLAUDE.md, ...)",
        "path_inventory_txt_files": "12 (ALL_PATHS.txt 151 MB, version_3-Gabriel.txt 44 MB, AI-Tools.txt 43 MB, ...)",
    }


def _clear_cache():
    mastered.cache_clear()
    _load_active.cache_clear()


if __name__ == "__main__":
    m = mastered()
    print(f"Mastered tickers: {len(m)}")
    if m:
        print(f"  head: {m[:10]}")
        print(f"  AAPL mastered? {is_mastered('AAPL')}")
        print(f"  ZZZ  mastered? {is_mastered('ZZZ')}")
    print(f"Active cycle: {active_cycle()}")
    print(f"Stack path: {active_stack_path()}")
