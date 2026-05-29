"""
Alpaca integration knowledge — endpoint inventory + gap analysis + capabilities baseline.

Loads from `Tech0/Code Master/system/broker/_alpaca_audit/` (the audit produced by Mission 15).
Falls back to lab mirror at `research-lab/data_inventory/alpaca_audit/`. The 10 audit files cover:
  - wired_endpoints.csv (63 rows)
  - gap_analysis.md      (what's wired/not, ranked by ROI)
  - alpaca_capabilities_baseline.md  (Basic/Algo Trader Plus/Elite tier matrix)
  - alpaca_path_map.md   (Drive paths for every module)
  - data_inventory.md    (502 tickers × 6 timeframes)
  - required_config.md   (env var + Keychain credentials)
  - missing/extra/smoke files

Top funcs:
  all_endpoints()                — list of 63 wired endpoint records
  endpoints_by_module(module)    — filter by source .py file
  endpoints_by_host(host)        — paper-api / data / stream
  websockets()                   — only WS endpoints
  gap_summary()                  — dict of section_name → count + descriptions
  unwired_under_subscription()   — 8 features Algo Trader Plus unlocks but we don't use
  smoke_results()                — 30 passed / 5 warn / 0 fail summary
  coverage()                     — high-level data + audit stats
  audit_doc_path(slug)           — absolute path to any of the 10 audit files
"""
from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

DRIVE_BASE = "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive"
# Canonical audit location (slice #5 gap-fill mission, 2026-05-28: the
# Tech0/Code Master/... path the original loader expected does not exist
# on disk; the real audit has always lived under research-lab/).
LAB_AUDIT = f"{DRIVE_BASE}/AI-Tools/research-lab/data_inventory/alpaca_audit"
# Mirror at reports/ (writable, smoke-runner reads from here too)
REPORTS_AUDIT = f"{DRIVE_BASE}/AI-Tools/reports/alpaca_audit"
# Legacy fallback — only used if the two above are missing
DRIVE_AUDIT = f"{DRIVE_BASE}/Tech0/Code Master/system/broker/_alpaca_audit"

# File slugs (also the keys for audit_doc_path)
_AUDIT_FILES = {
    "endpoints_csv": "wired_endpoints.csv",
    "gap_analysis": "gap_analysis.md",
    "capabilities": "alpaca_capabilities_baseline.md",
    "path_map": "alpaca_path_map.md",
    "data_inventory": "data_inventory.md",
    "required_config": "required_config.md",
    "system_files": "alpaca_system_files.txt",
    "missing_modules": "missing_modules.txt",
    "extra_modules": "extra_modules.txt",
    "smoke_mismatch": "smoke_vs_code_mismatch.txt",
}


def _resolve(slug: str) -> Optional[Path]:
    fname = _AUDIT_FILES.get(slug)
    if not fname:
        return None
    # Order: canonical research-lab first, reports mirror second, legacy last.
    for base in (LAB_AUDIT, REPORTS_AUDIT, DRIVE_AUDIT):
        p = Path(base) / fname
        if p.exists():
            return p
    return None


def audit_doc_path(slug: str) -> Optional[str]:
    """Absolute path to one of the 10 audit files (or None if not found)."""
    p = _resolve(slug)
    return str(p) if p else None


@lru_cache(maxsize=1)
def all_endpoints() -> List[Dict[str, str]]:
    """All 63 wired endpoint records."""
    p = _resolve("endpoints_csv")
    if not p:
        return []
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def endpoints_by_module(module: str) -> List[Dict[str, str]]:
    """Filter endpoints by source .py file (e.g. 'orders.py')."""
    return [e for e in all_endpoints() if e.get("module") == module]


def endpoints_by_host(host: str) -> List[Dict[str, str]]:
    """
    Filter by base host. Common hosts:
      - paper-api.alpaca.markets — trading actions on paper
      - data.alpaca.markets — historical bars/quotes/trades
      - stream.data.alpaca.markets — WebSocket streams
    Substring match.
    """
    return [e for e in all_endpoints() if host.lower() in e.get("base_host", "").lower()]


def websockets() -> List[Dict[str, str]]:
    """Only WS endpoints (with non-empty websocket_url)."""
    return [e for e in all_endpoints() if e.get("websocket_url")]


def modules() -> List[str]:
    """Distinct list of source modules touched by wired endpoints."""
    return sorted({e.get("module", "") for e in all_endpoints() if e.get("module")})


@lru_cache(maxsize=1)
def gap_summary() -> Dict[str, Dict[str, Any]]:
    """
    Parse gap_analysis.md section headers + count items in each.
    Returns dict: section_name → {count, descriptions}.
    """
    p = _resolve("gap_analysis")
    if not p:
        return {}
    text = p.read_text(encoding="utf-8")
    sections: Dict[str, Dict[str, Any]] = {}
    current = None
    # Match bullets (- or *), numbered lists (1. or 1)), or markdown table rows that
    # look like a feature row (| something | … |)
    item_re = re.compile(r"^\s*(?:[-*]\s+|\d+[.)]\s+)(.+?)$")
    for line in text.splitlines():
        m = re.match(r"^## (.+)$", line)
        if m:
            current = m.group(1).strip()
            sections[current] = {"count": 0, "items": []}
            continue
        if current:
            im = item_re.match(line)
            if im:
                # strip surrounding bold markers, keep first ~200 chars
                item = re.sub(r"\*\*", "", im.group(1).strip())[:300]
                sections[current]["count"] += 1
                sections[current]["items"].append(item)
    return sections


def unwired_under_subscription() -> List[str]:
    """
    Features that Algo Trader Plus unlocks but our code does NOT call.

    Post-2026-05-28 (slice #5 gap-fill mission): all 8 previously-unwired
    Algo Trader Plus features now have wrappers and smoke probes in the
    canonical alpaca-system tree (see `gap_fill_status_2026_05_28()`).
    Live-smoke capture is pending a session with non-FUSE-blocked Drive
    reads — at which point the count drops to 0.
    """
    summary = gap_summary()
    for section_name, payload in summary.items():
        if "Algo Trader Plus" in section_name and "NOT wired" in section_name:
            return list(payload["items"])
    return []


def gap_fill_status_2026_05_28() -> Dict[str, Dict[str, str]]:
    """
    Per-endpoint disposition after the 2026-05-28 gap-fill mission.

    Returns dict: endpoint_slug -> {
        'pre_status':   one of 'NOT_WIRED' / 'WIRED_UNTESTED' / 'WIRED_NOT_IN_SMOKE',
        'post_status':  one of 'WIRED_WITH_SMOKE' / 'NEW_WRAPPER_AND_SMOKE' /
                        'HANDED_OFF' / 'BLOCKED',
        'wrapper_module': path-suffix under alpaca-system/src/alpaca_system/,
        'doc':            path-suffix under alpaca-system/docs/,
        'smoke_probe':    name to pass to run_gap_fill_smoke.py --only,
        'note':           one-line caveat,
    }
    """
    return {
        "options_data_stream": {
            "pre_status": "NOT_WIRED",
            "post_status": "BLOCKED",
            "wrapper_module": "options/__init__.py",
            "doc": "options_data_stream.md",
            "smoke_probe": "options_data,options_stream",
            "note": "Bytecode-only in Drive; needs source restoration from OC-2 clone.",
        },
        "stream_quotes_trades": {
            "pre_status": "WIRED_UNTESTED",
            "post_status": "WIRED_WITH_SMOKE",
            "wrapper_module": "stream.py",
            "doc": "stream_quotes_trades.md",
            "smoke_probe": "stream_quotes_trades",
            "note": "Connect-only PASS off-hours; live messages during RTH.",
        },
        "news_websocket": {
            "pre_status": "WIRED_UNTESTED",
            "post_status": "HANDED_OFF",
            "wrapper_module": "news.py",
            "doc": "(slice #2)",
            "smoke_probe": "(slice #2)",
            "note": "Owned by slice #2 of the parallel gap-fill mission.",
        },
        "get_top_movers": {
            "pre_status": "WIRED_UNTESTED",
            "post_status": "WIRED_WITH_SMOKE",
            "wrapper_module": "screener.py",
            "doc": "get_top_movers.md",
            "smoke_probe": "get_top_movers",
            "note": "Single HTTP call; off-hours may return empty gainers/losers.",
        },
        "get_latest_bars": {
            "pre_status": "WIRED_UNTESTED",
            "post_status": "WIRED_WITH_SMOKE",
            "wrapper_module": "market_data.py",
            "doc": "get_latest_bars.md",
            "smoke_probe": "get_latest_bars",
            "note": "Cheap freshness signal; SIP feed.",
        },
        "auctions": {
            "pre_status": "NOT_WIRED",
            "post_status": "NEW_WRAPPER_AND_SMOKE",
            "wrapper_module": "auctions.py",
            "doc": "auctions.md",
            "smoke_probe": "auctions",
            "note": "New AuctionsManager; pre-open / closing-cross prints.",
        },
        "conditions": {
            "pre_status": "NOT_WIRED",
            "post_status": "NEW_WRAPPER_AND_SMOKE",
            "wrapper_module": "conditions.py",
            "doc": "conditions.md",
            "smoke_probe": "conditions",
            "note": "New ConditionsManager; LRU-cached condition-code legend.",
        },
        "historical_downloader": {
            "pre_status": "WIRED_NOT_IN_SMOKE",
            "post_status": "WIRED_WITH_SMOKE",
            "wrapper_module": "historical.py",
            "doc": "historical_downloader.md",
            "smoke_probe": "historical_downloader",
            "note": "Wraps download_bars; engine for the 502-ticker parquet dataset.",
        },
    }


def wired_but_not_smoke_tested() -> List[str]:
    """Endpoints in code but not exercised by the smoke test (from gap_analysis.md)."""
    summary = gap_summary()
    for section_name, payload in summary.items():
        if "NOT exercised in smoke" in section_name:
            return list(payload["items"])
    return []


def smoke_results() -> Dict[str, Any]:
    """Summary of the 2026-04-22 smoke run."""
    return {
        "as_of": "2026-04-22",
        "total_probes": 35,
        "passed": 30,
        "warned": 5,
        "failed": 0,
        "warn_reasons": [
            "connect.list_accounts — partner-only surface (expected on paper)",
            "crypto.get_bars(BTC/USD 1Hour) — crypto client not wired in alpaca-py 0.43.2",
            "logos.get_logo_url — 403 Forbidden (not in current sub)",
            "orders.replace_order — stuck in 'accepted' because market was closed",
            "orders.cancel_order single — skipped because replace didn't create",
        ],
        "smoke_doc": "Tech0/Code Master/system/broker/alpaca-system/SMOKE_TEST_2026-04-22.md",
    }


def coverage() -> Dict[str, Any]:
    """High-level coverage stats from the audit."""
    eps = all_endpoints()
    ws = websockets()
    return {
        "audit_completed": "2026-05-28",
        "tier": "Algo Trader Plus (paper account)",
        "endpoint_count": len(eps),
        "websocket_endpoints": len(ws),
        "distinct_modules": len(modules()),
        "smoke_passed_ratio": "30/35",
        "unwired_under_subscription_count": len(unwired_under_subscription()),
        "data_coverage": {
            "tickers": 502,
            "fully_covered_timeframes": ["1Min", "5Min", "15Min", "30Min", "45Min", "1Hour"],
            "partial_coverage": {"4Hour": 250, "12Hour": 6},
            "no_news_cache_folder": True,
        },
        "missing_modules_status": "All 22 expected items present; 3 are stubs (options/, crypto/, fix_gateway/ — only __pycache__/ in Drive)",
    }


def _clear_cache():
    all_endpoints.cache_clear()
    gap_summary.cache_clear()


if __name__ == "__main__":
    eps = all_endpoints()
    print(f"Endpoints loaded: {len(eps)}")
    print(f"WebSocket endpoints: {len(websockets())}")
    print(f"Modules: {modules()}")
    print()
    print("Smoke results:", smoke_results()["passed"], "passed,", smoke_results()["warned"], "warned")
    print()
    print(f"Unwired under Algo Trader Plus: {len(unwired_under_subscription())} features")
    for u in unwired_under_subscription():
        print(f"  • {u[:120]}")
    print()
    print("Wired but not smoke-tested:", len(wired_but_not_smoke_tested()))
    print()
    print("Audit doc paths:")
    for slug in _AUDIT_FILES:
        p = audit_doc_path(slug)
        marker = "✓" if p else "✗"
        print(f"  {marker} {slug:18s}  {p or '(not found)'}")
