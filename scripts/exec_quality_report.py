"""
exec_quality_report.py — Daily execution-quality report for the limit-order
router (a6ead433 #1 lift measurement).

Inputs:
  - paper_trade/limit_order_fills.jsonl    (router fills since LIMIT_ORDER_ROUTER=1)
  - paper_trade/fills/<date>/*.jsonl       (baseline market fills from WS consumer)

Outputs:
  - paper_trade/reports/exec_quality/<date>.json
  - stdout: human summary table

Metrics:
  - avg_fill_vs_mid_bps        # router only (paid vs mid at submit)
  - fill_rate_pct              # router: not_escalated / total
  - escalation_rate_pct        # router: escalated to market / total
  - avg_latency_s              # router only
  - lift_vs_baseline_bps       # router avg - baseline avg (where baseline=0bps by defn)
  - per_route_type             # split by marketable_limit / passive_limit
  - per_signal_strength        # HIGH vs NORMAL

CLI:
  python exec_quality_report.py [--date YYYY-MM-DD] [--lookback-days 4]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from datetime import date as dt_date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

WORK = Path(
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/AI-Tools/s&p500-ticker-mastery"
)
PAPER = WORK / "paper_trade"
ROUTER_LOG = PAPER / "limit_order_fills.jsonl"
REPORT_DIR = PAPER / "reports" / "exec_quality"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:  # noqa: BLE001
                continue
    return out


def _parse_ts(s: Any) -> datetime | None:
    if not s or not isinstance(s, str):
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:  # noqa: BLE001
        return None


def _filter_by_date(rows: list[dict], target: dt_date) -> list[dict]:
    out = []
    for r in rows:
        ts = _parse_ts(r.get("ts"))
        if ts and ts.astimezone(timezone.utc).date() == target:
            out.append(r)
    return out


def _safe_mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return float(statistics.mean(xs))


def compute_metrics(router_rows: list[dict]) -> dict[str, Any]:
    total = len(router_rows)
    if total == 0:
        return {
            "total": 0, "fill_rate_pct": None, "escalation_rate_pct": None,
            "avg_fill_vs_mid_bps": None, "avg_latency_s": None,
            "per_route_type": {}, "per_signal_strength": {},
        }
    escalated = sum(1 for r in router_rows if r.get("escalated"))
    not_esc = total - escalated
    bps = _safe_mean([r.get("bps_vs_mid") for r in router_rows if not r.get("escalated")])
    latency = _safe_mean([r.get("latency_s") for r in router_rows])

    per_route: dict[str, dict] = defaultdict(lambda: {"n": 0, "bps_sum": 0.0, "bps_n": 0})
    per_strength: dict[str, dict] = defaultdict(lambda: {"n": 0, "bps_sum": 0.0, "bps_n": 0})
    for r in router_rows:
        rt = r.get("route_type") or "unknown"
        per_route[rt]["n"] += 1
        if isinstance(r.get("bps_vs_mid"), (int, float)):
            per_route[rt]["bps_sum"] += float(r["bps_vs_mid"])
            per_route[rt]["bps_n"] += 1
        st = (r.get("signal_strength") or "NORMAL").upper()
        per_strength[st]["n"] += 1
        if isinstance(r.get("bps_vs_mid"), (int, float)):
            per_strength[st]["bps_sum"] += float(r["bps_vs_mid"])
            per_strength[st]["bps_n"] += 1

    return {
        "total": total,
        "filled_no_escalation": not_esc,
        "escalated_to_market": escalated,
        "fill_rate_pct": round(not_esc / total * 100, 2),
        "escalation_rate_pct": round(escalated / total * 100, 2),
        "avg_fill_vs_mid_bps": round(bps, 2) if bps is not None else None,
        "avg_latency_s": round(latency, 3) if latency is not None else None,
        "per_route_type": {
            k: {
                "n": v["n"],
                "avg_bps": round(v["bps_sum"] / v["bps_n"], 2) if v["bps_n"] else None,
            }
            for k, v in per_route.items()
        },
        "per_signal_strength": {
            k: {
                "n": v["n"],
                "avg_bps": round(v["bps_sum"] / v["bps_n"], 2) if v["bps_n"] else None,
            }
            for k, v in per_strength.items()
        },
    }


def render_summary(target: dt_date, metrics: dict[str, Any]) -> str:
    lines = [
        f"== Execution quality report — {target.isoformat()} ==",
        f"Router-routed orders:  {metrics['total']}",
    ]
    if metrics["total"]:
        lines += [
            f"Fill rate (no esc):    {metrics['fill_rate_pct']}%",
            f"Escalation rate:       {metrics['escalation_rate_pct']}%",
            f"Avg fill-vs-mid bps:   {metrics['avg_fill_vs_mid_bps']} (target 10-20bps)",
            f"Avg latency:           {metrics['avg_latency_s']}s",
            "",
            "By route type:",
        ]
        for rt, v in metrics["per_route_type"].items():
            lines.append(f"  {rt:24s} n={v['n']:>4}  avg_bps={v['avg_bps']}")
        lines.append("\nBy signal strength:")
        for st, v in metrics["per_signal_strength"].items():
            lines.append(f"  {st:24s} n={v['n']:>4}  avg_bps={v['avg_bps']}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default: today UTC)")
    ap.add_argument("--lookback-days", type=int, default=4,
                    help="Baseline window for market-order comparison (informational)")
    ap.add_argument("--write", action="store_true", default=True,
                    help="Write JSON report to disk")
    args = ap.parse_args()

    target = (
        dt_date.fromisoformat(args.date)
        if args.date else datetime.now(timezone.utc).date()
    )

    all_rows = _load_jsonl(ROUTER_LOG)
    today_rows = _filter_by_date(all_rows, target)
    metrics = compute_metrics(today_rows)
    metrics["date"] = target.isoformat()
    metrics["baseline_lookback_days"] = args.lookback_days
    metrics["baseline_avg_bps_assumption"] = 0.0  # market orders = 0bps by defn

    if args.write:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        out = REPORT_DIR / f"{target.isoformat()}.json"
        out.write_text(json.dumps(metrics, indent=2, default=str))
        print(f"Wrote {out}")

    print()
    print(render_summary(target, metrics))
    return 0


if __name__ == "__main__":
    sys.exit(main())
