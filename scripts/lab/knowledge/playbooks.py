"""
Playbooks knowledge — pointer index to the markdown research artifacts.

Each playbook has a short slug; lookup returns the absolute Drive path. Designed for
prompt / RAG injection: the autonomous generator can fetch the full text on demand
rather than always carrying all 50+ KB in context.

Top funcs:
  all_playbooks()               — list of all playbook records
  path(slug)                    — absolute Drive path for a playbook slug
  read(slug)                    — full text content (use sparingly — they're 10-50 KB)
  search(query)                 — case-insensitive substring match across slugs + descriptions
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

DRIVE_BASE = "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive"

_PLAYBOOKS: List[Dict[str, str]] = [
    {
        "slug": "indicators_methodology",
        "name": "Indicators Methodology Playbook (LOCKED PRINCIPLE)",
        "path": f"{DRIVE_BASE}/AI-Tools/reports/indicators_methodology_playbook_2026-05-29.md",
        "size_estimate": "7 KB",
        "description": "User-locked hard rule: indicators ASSIST strategies; they are informational inputs, never the sole signal. Unit of validation is a strategy hypothesis dict (regime_gate × bias_filter × trigger × confirmation × timing × exit × cost × universe × timeframe), NOT a single indicator. Includes the 6 informational axes (trend / momentum / vol band / volume / mean-rev / structure), the strategy_hypothesis_template, and the reframe of the 2026-05-28 standalone validation result. Strongest hypotheses combine alt-data as trigger with OHLCV indicators as confluence.",
        "use_when": "Writing any backtest brief; designing a strategy hypothesis; validating an indicator; refactoring a validator API; pre-registering a SAP. ALWAYS read before writing 'test indicator X' in any brief.",
        "structured_form": "lab.knowledge.indicators (methodology_principle / informational_axes / strategy_hypothesis_template / validate_test_unit)",
        "authority": "user-set hard rule"
    },
    {
        "slug": "indicator_insights",
        "name": "Indicator Insights Playbook",
        "path": f"{DRIVE_BASE}/AI-Tools/reports/indicator_insights_playbook_2026-05-28.md",
        "size_estimate": "16 KB",
        "description": "Distilled lessons from the 107-indicator catalog. What worked + why + how to replicate; what failed + why; 5-min S&P settings; redundancy clusters; how to use going forward without changing existing code.",
        "use_when": "Generating new indicator candidates, deciding which to test next, biasing search toward proven patterns."
    },
    {
        "slug": "indicator_hardening_plan",
        "name": "Indicator Hardening Plan (heavyweight)",
        "path": f"{DRIVE_BASE}/AI-Tools/reports/indicator_hardening_plan_2026-05-28.md",
        "size_estimate": "13 KB",
        "description": "6-phase plan: Phase 0 fix infrastructure bugs (synthetic OU data ban, slippage 1→5 bps, R:R inversion, EMA cross, open-range), Phase 1 redundancy prune, Phase 2 re-validate winners with PBO/DSR/walk-forward, Phase 3 test gaps, Phase 4 broaden to S&P 500, Phase 5 regime-switching ensemble.",
        "use_when": "Designing a validation pipeline; running serious backtests; need PBO/DSR/walk-forward methodology."
    },
    {
        "slug": "candle_structure_signals",
        "name": "Candle Structure Intraday Signals",
        "path": f"{DRIVE_BASE}/AI-Tools/research-lab/data_inventory/candle_structure_extract/candle_signals.md",
        "size_estimate": "10 KB",
        "description": "14 intraday candle-structure signals (items 4-17 from Candle Structure in Trading.pdf) + 10-question pre-trade checklist. Verbatim wording preserved.",
        "use_when": "Building intraday entry/exit rules; checking confirmation layers; generating candle-pattern features.",
        "structured_form": "lab.knowledge.candle_signals (Python module)"
    },
    {
        "slug": "candle_structure_action_plan",
        "name": "Candle Structure Action Plan (ML upgrade)",
        "path": f"{DRIVE_BASE}/AI-Tools/research-lab/data_inventory/candle_structure_extract/action_plan.md",
        "size_estimate": "2 KB",
        "description": "5-step plan: (1) Add baseline models (logistic/RF/LightGBM/CatBoost/ELO), (2) Upgrade ELO (Glicko/TrueSkill/opponent-adj/time-decayed), (3) Improve probability testing (Brier, log loss, calibration), (4) Add walk-forward, (5) Add a decision layer (edge/EV/threshold/skip-low-conviction).",
        "use_when": "Planning ML model upgrades; designing the decision layer; setting up calibration testing."
    },
    {
        "slug": "candle_structure_gap_analysis",
        "name": "Candle Structure Gap Analysis vs 107-indicator catalog",
        "path": f"{DRIVE_BASE}/AI-Tools/research-lab/data_inventory/candle_structure_extract/gap_analysis.md",
        "size_estimate": "14 KB",
        "description": "Cross-reference of 221 PDF items (sections C+D+E) against the 38-row indicator_mastery_index. 19 already covered, 202 new. Suggested next actions + reframes.",
        "use_when": "Deciding which new indicators to register / queue for backtest; understanding what's net new."
    },
    {
        "slug": "candle_structure_indicator_catalog",
        "name": "Candle Structure PDF Indicator Catalog",
        "path": f"{DRIVE_BASE}/AI-Tools/research-lab/data_inventory/candle_structure_extract/pdf_indicator_catalog.csv",
        "size_estimate": "32 KB",
        "description": "221 rows: sections C (71 indicators/tools), D (75 advanced charting/order flow), E (75 ML/feature eng/evaluation). Each row has section, item_number, name, description, category, already_in_mastery_index flag.",
        "use_when": "Need machine-readable inventory of every tool/concept in the candle-structure PDF; cross-referencing against existing catalog."
    },
    {
        "slug": "multi_timeframe_breakdown",
        "name": "Multi-Timeframe Trading Breakdown (11 timeframes)",
        "path": f"{DRIVE_BASE}/AI-Tools/research-lab/data_inventory/multi_timeframe_research/multi_timeframe_breakdown.md",
        "size_estimate": "≥80 KB (in flight)",
        "description": "Per-timeframe deep breakdown: best-for, signal vocabulary, 6+ setups each, mistakes, what it's bad for, indicator overlay suggestions. Plus combination workflows + 10 multi-tf alignment examples.",
        "use_when": "Generating multi-timeframe strategies; deciding confluence partners; biasing setup choice by timeframe.",
        "structured_form": "lab.knowledge.multi_timeframe (Python module)",
        "status": "in_flight_subagent"
    },
    {
        "slug": "multi_timeframe_table",
        "name": "Multi-Timeframe Comparison Table",
        "path": f"{DRIVE_BASE}/AI-Tools/research-lab/data_inventory/multi_timeframe_research/timeframe_table.csv",
        "size_estimate": "~3 KB (in flight)",
        "description": "11 rows × 12 columns: timeframe, main_purpose, info_provided, how_it_improves_trades, mistakes_if_ignored, best_for_entry/stop/target/bias, confluence_partners, noise_level, suggested_indicator_ids.",
        "use_when": "Programmatic timeframe selection; one-line lookups."
    },
    {
        "slug": "master_inventory",
        "name": "Tech0 Master Inventory",
        "path": f"{DRIVE_BASE}/AI-Tools/research-lab/data_inventory/master_inventory_20260528.md",
        "size_estimate": "15 KB",
        "description": "Top-level Tech0 structure: 7 Master folders, 12 path-inventory .txt files, 22 root docs. Code Master runtime files, BackTests & Data highlights, known data-quality gaps (13+ files named 'strategy.md').",
        "use_when": "Need to find a file in Tech0; understanding the codebase layout; onboarding new sub-agents."
    },
    {
        "slug": "mac_disk_audit",
        "name": "Mac Disk Audit",
        "path": f"{DRIVE_BASE}/AI-Tools/reports/mac_disk_audit_2026-05-28.md",
        "size_estimate": "~10 KB",
        "description": "Mac SSD breakdown by tier: Library/Application Support, .venvs, .rustup, Drive cache (81 GB!), etc. Storage migration plan to 2TB external drive.",
        "use_when": "Infrastructure questions; understanding where files live on disk."
    },
]


def all_playbooks() -> List[Dict[str, str]]:
    """List of all playbook records (slug, name, path, description, etc.)."""
    return [dict(p) for p in _PLAYBOOKS]


def path(slug: str) -> Optional[str]:
    """Absolute Drive path for a playbook slug."""
    for p in _PLAYBOOKS:
        if p["slug"] == slug:
            return p["path"]
    return None


def read(slug: str) -> Optional[str]:
    """Full text content of a playbook. Use sparingly (10-50 KB)."""
    p = path(slug)
    if p is None:
        return None
    try:
        return Path(p).read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def search(query: str) -> List[Dict[str, str]]:
    """Case-insensitive substring match across slugs + names + descriptions."""
    q = query.lower()
    return [
        dict(p) for p in _PLAYBOOKS
        if q in p["slug"].lower()
        or q in p["name"].lower()
        or q in p.get("description", "").lower()
        or q in p.get("use_when", "").lower()
    ]


def _clear_cache():
    pass


if __name__ == "__main__":
    print(f"Playbooks: {len(all_playbooks())}")
    for p in all_playbooks():
        status = p.get("status", "ok")
        print(f"  {p['slug']:35s}  [{status}]  {p['size_estimate']:10s}")
