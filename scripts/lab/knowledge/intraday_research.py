"""
Intraday research bootstrap knowledge — Phase 1 first deliverable (Mission 12 output).

Loads from `claudes test/intraday_research/` (lab mirror at
`research-lab/data_inventory/intraday_research/`). Contains the bootstrap ticker, spec
captures, mastery map, and the 5-agent assignment grid for ticker A.

Top funcs:
  selected_ticker()             — 'A' (Agilent Technologies)
  ticker_counts()               — {mastered, universe, unmastered}
  bootstrap_doc_path(ticker)    — path to bootstrap.md for a ticker
  read_bootstrap(ticker)        — full markdown content
  agent_assignment_plan(ticker) — list of N sub-agents to spawn (next step)
  key_findings()                — list of 8 audit findings + open questions
  spec_section(name)            — verbatim section from spec_captured.md
  hard_rules()                  — non-negotiable rules from the spec
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

DRIVE_BASE = "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive"
WORK = f"{DRIVE_BASE}/claudes test/intraday_research"
LAB_MIRROR = f"{DRIVE_BASE}/AI-Tools/research-lab/data_inventory/intraday_research"

_FILES = {
    "spec_outline": "spec_outline.md",
    "spec_captured": "spec_captured.md",
    "spec_raw": "spec_raw.txt",
    "spec_phase2_raw": "spec_phase2_raw.txt",
    "tickers_universe": "tickers_universe.txt",
    "tickers_mastered": "tickers_mastered.txt",
    "tickers_unmastered": "tickers_unmastered.txt",
    "mastery_map": "mastery_map.md",
}


def _resolve(slug: str) -> Optional[Path]:
    fname = _FILES.get(slug)
    if not fname:
        return None
    for base in (WORK, LAB_MIRROR):
        p = Path(base) / fname
        if p.exists():
            return p
    return None


def _resolve_ticker_doc(ticker: str, fname: str = "bootstrap.md") -> Optional[Path]:
    t = ticker.upper()
    for base in (WORK, LAB_MIRROR):
        p = Path(base) / t / fname
        if p.exists():
            return p
    return None


@lru_cache(maxsize=1)
def selected_ticker() -> Optional[str]:
    """The first unmastered ticker chosen for Phase 1 bootstrap. 'A' (Agilent Technologies)."""
    p = _resolve("tickers_unmastered")
    if not p:
        return "A"  # embedded fallback per mission 12 result
    lines = [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return lines[0] if lines else None


def _count_lines(slug: str) -> int:
    p = _resolve(slug)
    if not p:
        return 0
    return sum(1 for l in p.read_text(encoding="utf-8").splitlines() if l.strip())


@lru_cache(maxsize=1)
def ticker_counts() -> Dict[str, int]:
    """Counts from the three ticker text files."""
    return {
        "universe": _count_lines("tickers_universe"),
        "mastered_loose": _count_lines("tickers_mastered"),
        "unmastered_strict_spec": _count_lines("tickers_unmastered"),
    }


def bootstrap_doc_path(ticker: str) -> Optional[str]:
    """Absolute path to bootstrap.md for a ticker (typically just 'A' for now)."""
    p = _resolve_ticker_doc(ticker, "bootstrap.md")
    return str(p) if p else None


def read_bootstrap(ticker: str) -> Optional[str]:
    """Full text of bootstrap.md for a ticker."""
    p = _resolve_ticker_doc(ticker, "bootstrap.md")
    return p.read_text(encoding="utf-8") if p else None


# Embedded — captured from Mission 12 agent's report (always available even if Drive blind)
_KEY_FINDINGS = [
    "Historical data path in original brief is stale ('Ph0tis/Photis - Gabriels Version/...'); "
    "real live path is Tech0/Data Master/data/timeframes/1_Alpaca TimeFrames/S&P500 5 Year Historical Data",
    "Loose-mastered (folder exists in mastered/) = 502; ALL 502 tickers have a folder. "
    "First unmastered ticker only exists under STRICT-SPEC audit (real CHAMPIONSHIP_FORMULA.md required).",
    "Strict-spec audit reveals broad mastery gaps: AAPL has ~20/23 items; ACGL ~12/23; "
    "A/ABBV/ABNB/ABT have only a pointer .md, NO championship file. A is first alphabetical strict-fail.",
    "Open question: does mastered/_results/<T>/ count toward the 23-item requirement, or must "
    "everything sit in the ticker's own folder?",
    "10MIN timeframe is missing from on-disk data. Spec rule 10 lists it — derive from 1Min if needed.",
    "PDF is 256 pages, but only ~28 pages have content (1-14 Phase 1, 237-256 Phase 2). "
    "Pages 15-236 are blank ChatGPT-export filler.",
    "Data ends 2026-04-20; today is 2026-05-28 → ~5-week gap. Re-fetch recent 5 weeks before any "
    "live paper deployment.",
    "FUSE-blindness on this Mac: Tech0/ and Photis paths invisible. All enumeration via MCP. "
    "Recommendation: agents should run on Modal/cloud worker that natively sees Tech0/ rather than "
    "streaming via MCP base64.",
    "Sub-agent compute budget: 5-agent × 262-week walk-forward on Opus 4.7 XHIGH is the heaviest "
    "compute item ever attempted. Recommend auto_cloud_dispatcher: heavy backtesting on "
    "Modal/GitHub Actions, only weekly review + theory revision on Claude.",
]


def key_findings() -> List[str]:
    """8+ key findings from the bootstrap audit (always available)."""
    return list(_KEY_FINDINGS)


_HARD_RULES = [
    "Intraday only. All trades open after session start and close ≥5 min before close. No overnight/swing.",
    "No lookahead. No future candles. No EOD data before EOD. No filings before real filing timestamp. No news before release.",
    "One ticker at a time.",
    "Weekly walk-forward.",
    "No copy-pasting strategies across tickers — each ticker's champion formula must be justified for that ticker.",
    "A ticker is MASTERED only if it has a complete 23-item championship file (per spec).",
    "Failed cycles are NOT final — re-run with new combinations (different EDGAR usage, gov-trades, indicators, timeframes, entry/exit/no-trade logic).",
]


def hard_rules() -> List[str]:
    """The 7 hard rules from the spec — apply to every downstream sub-agent."""
    return list(_HARD_RULES)


def spec_section(name: str) -> Optional[str]:
    """
    Pull a section from spec_captured.md by header-substring match.
    Common section names: 'Phase 1 trading rules', 'weekly cycle', 'weekly report', 'championship', etc.
    """
    p = _resolve("spec_captured")
    if not p:
        return None
    text = p.read_text(encoding="utf-8")
    name_l = name.lower()
    # find the matching ## header
    pattern = re.compile(r"^##\s+(.+)$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    for i, m in enumerate(matches):
        if name_l in m.group(1).lower():
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            return text[start:end].strip()
    return None


# Embedded — the 5-agent grid proposed in Mission 12's bootstrap.md for ticker A
_AGENT_GRID_A = [
    {
        "agent_id": "A1_PURE_TECH",
        "theory": "Pure technical confluence — no fundamentals or alt-data",
        "timeframes": "1D thesis + 15MIN structure + 5MIN entry",
        "edgar_use": "none",
        "gov_use": "none",
        "indicator_set": "Donchian(20/40) UP, ATR, VWAP, Chopping Index",
        "strategy_archetype": "Donchian breakout with regime filter",
        "entry_logic": "1D bullish + 15MIN HL + 5MIN VWAP reclaim + ChopIdx < 38",
        "exit_logic": "1.5 ATR trailing stop or 2R target",
        "no_trade_filter": "ChopIdx >= 62 or volume < 1.5x avg",
    },
    {
        "agent_id": "A2_ORB_MORNING",
        "theory": "Opening Range Breakout (AAPL-template)",
        "timeframes": "1D bias + 5MIN ORB + 1MIN trigger",
        "edgar_use": "none",
        "gov_use": "none",
        "indicator_set": "Opening Range, ATR, Volume Expansion, VWAP",
        "strategy_archetype": "First-6-bar opening range with volume confirmation",
        "entry_logic": "Break of OR high + volume > 1.5x avg + above VWAP",
        "exit_logic": "Stop below OR low; target = 1x OR-size projected",
        "no_trade_filter": "Time > 10:30 ET or gap > 2 ATR",
    },
    {
        "agent_id": "A3_VWAP_MTF",
        "theory": "VWAP-based multi-timeframe mean reversion + continuation",
        "timeframes": "1H trend + 15MIN setup + 5MIN entry",
        "edgar_use": "none",
        "gov_use": "none",
        "indicator_set": "VWAP, VWAP StdDev bands, ATR, RSI(14)",
        "strategy_archetype": "VWAP pullback long in 1H uptrend",
        "entry_logic": "1H above VWAP + pullback to 5MIN VWAP + RSI(14) > 40 + close above VWAP",
        "exit_logic": "Stop below wick low; target = VWAP +1σ",
        "no_trade_filter": "1H below VWAP or RSI < 30 (extreme oversold)",
    },
    {
        "agent_id": "A4_GOV_AWARE",
        "theory": "Catalyst-driven: insider trading + congress trades as edge",
        "timeframes": "EDGAR/Gov-Trades event + 1H bias + 5MIN entry",
        "edgar_use": "Form 4 insider transactions (lookback 30d)",
        "gov_use": "congress_trades + lobbying (lookback 60d)",
        "indicator_set": "ADX(14), Volume Expansion, ATR",
        "strategy_archetype": "Catalyst-confirmed momentum",
        "entry_logic": "Form 4 buy in last 30d OR positive congress trade + ADX > 20 + volume > 1.5x avg",
        "exit_logic": "Hold to EOD or 2R; trailing 1 ATR after 1R",
        "no_trade_filter": "ADX < 15 or no catalyst in window",
    },
    {
        "agent_id": "A5_HYBRID_REGIME",
        "theory": "Regime-switching: trend (Donchian+ADX) vs range (RSI(2)+BB) per ADX threshold",
        "timeframes": "1D regime + 1H setup + 5MIN entry",
        "edgar_use": "none",
        "gov_use": "none",
        "indicator_set": "ADX(14), Donchian(20), RSI(2), Bollinger Bands, ATR",
        "strategy_archetype": "Conditional model: ADX > 25 → trend; ADX < 20 → range; 20-25 → reduce size",
        "entry_logic": "ADX > 25: Donchian breakout; ADX < 20: RSI(2) < 5 at lower BB",
        "exit_logic": "ATR-based stop; target depends on regime (1.5x ATR trend; midpoint range)",
        "no_trade_filter": "ADX 20-25 buffer zone: skip or reduce size 50%",
    },
]


def agent_assignment_plan(ticker: str = "A") -> List[Dict[str, str]]:
    """5-agent assignment grid for ticker. Currently hard-coded for A; future tickers TBD."""
    if ticker.upper() == "A":
        return [dict(a) for a in _AGENT_GRID_A]
    return []


def workspace_path() -> str:
    """Working directory for this research cycle."""
    return WORK


def lab_mirror_path() -> str:
    """Lab-accessible mirror path."""
    return LAB_MIRROR


def _clear_cache():
    selected_ticker.cache_clear()
    ticker_counts.cache_clear()


if __name__ == "__main__":
    print(f"Selected ticker: {selected_ticker()}")
    print(f"Counts: {ticker_counts()}")
    print(f"Bootstrap doc for A: {bootstrap_doc_path('A')}")
    print(f"\nHard rules: {len(hard_rules())}")
    print(f"Key findings: {len(key_findings())}")
    print(f"Agents proposed for A: {len(agent_assignment_plan('A'))}")
    for a in agent_assignment_plan("A"):
        print(f"  {a['agent_id']:18s}  {a['theory'][:80]}")
