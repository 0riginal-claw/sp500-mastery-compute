"""
progress_dashboard.py — 24/7 progress tracker.

Outputs:
  - dashboard/dashboard.md  — human-readable
  - dashboard/state.json    — machine-readable
  - dashboard/history.jsonl — append-only log of all snapshots

Verifies completion by checking actual artifacts on disk (not task-list claims).
Run every 10 min via LaunchAgent OR on-demand.
"""
import json, glob, time, sys
from datetime import datetime, timezone
from pathlib import Path

# ── event bus (best-effort) ──────────────────────────────────────────────────
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from event_bus import EventBus as _EventBus
    _EB = _EventBus
except Exception:
    _EB = None  # type: ignore[assignment]

WORK = Path("/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/s&p500-ticker-mastery")
DASH = WORK / "dashboard"


def collect_state() -> dict:
    s = {'ts': datetime.now(timezone.utc).isoformat()}

    # Mastery files (verified by actual file on disk)
    mfiles = list((WORK / "mastery_files").glob("*mastered*.md")) if (WORK / "mastery_files").exists() else []
    mastery_tickers = sorted({p.stem.split("_")[0] for p in mfiles})
    s['mastered'] = {
        'count': len(mastery_tickers),
        'pct_sp500': 100 * len(mastery_tickers) / 502,
        'tickers': mastery_tickers,
    }

    # Backtests by pipeline
    for v in ['ml', 'xgb', 'ml_sweep', 'xgb_v3', 'xgb_v4', 'daily', 'ml_v3']:
        run_dir = WORK / f"backtests_{v}"
        if run_dir.exists():
            metas = list(run_dir.glob("*/run_meta.json"))
            mastered_in_v = 0
            for p in metas:
                try:
                    with open(p) as f: mm = json.load(f).get('metrics_oos_aggregate', {})
                    if ((mm.get('profit_factor') or 0) >= 1.5 and (mm.get('win_rate') or 0) >= 0.53
                        and (mm.get('total_return_pct') or 0) > 0 and (mm.get('max_drawdown_pct') or 0) >= -0.03
                        and (mm.get('n_trades') or 0) >= 8):
                        mastered_in_v += 1
                except Exception:
                    pass
            s[f'pipeline_{v}'] = {'runs': len(metas), 'mastered': mastered_in_v}

    # Feature discovery state
    discovery_dir = WORK / "feature_discovery"
    if discovery_dir.exists():
        reports = sorted((discovery_dir / "reports").glob("*.md")) if (discovery_dir / "reports").exists() else []
        queue_path = discovery_dir / "inbox" / "queue.json"
        queue_size = 0
        if queue_path.exists():
            try: queue_size = len(json.loads(queue_path.read_text()))
            except Exception: pass
        s['discovery'] = {
            'reports_count': len(reports),
            'last_report': reports[-1].name if reports else None,
            'queue_size': queue_size,
        }

    # Overseer state
    overseer_dir = WORK / "overseer"
    if overseer_dir.exists():
        history = sorted((overseer_dir / "history").glob("*.json")) if (overseer_dir / "history").exists() else []
        s['overseer'] = {
            'cycles_run': len(history),
            'last_cycle': history[-1].name if history else None,
            'has_recommendations': (overseer_dir / "recommendations.json").exists(),
        }

    # External repos
    ext = Path("/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/external-repos")
    if ext.exists():
        s['external_repos'] = sorted(d.name for d in ext.iterdir() if d.is_dir())

    # Scripts inventory
    scripts_dir = WORK / "scripts"
    if scripts_dir.exists():
        s['scripts'] = sorted(p.name for p in scripts_dir.glob("*.py"))

    # Claude usage pacing state (written by usage_pacing_daemon.py)
    pacing_path = DASH / "pacing_state.json"
    if pacing_path.exists():
        try:
            s['pacing'] = json.loads(pacing_path.read_text())
        except Exception as e:
            s['pacing'] = {'error': f'failed to read pacing_state.json: {e}'}

    return s


def _pacing_regime_emoji(regime: str) -> str:
    return {
        'under': '🟢',
        'on': '🟡',
        'on_pace': '🟡',
        'over': '🟠',
        'emergency': '🔴',
    }.get(str(regime).lower(), '⚪')


def _to_pct(v) -> float:
    """Normalize a pacing fraction or whole-percent value to a percent (0-100+).

    usage_pacing_daemon.py writes week_used_pct / week_elapsed_pct as
    fractions where 1.0 = 100% of the weekly limit (so values can exceed 1.0
    when over-quota — e.g. 1.59 == 159% used). Older or alternate sources may
    write whole percents already. Heuristic:
      - value <= 5 → treat as fraction, multiply by 100
      - value  > 5 → treat as already a percent
    (Real fractions almost never exceed ~3.0, and a true "percent" reading at
    or below 5% is rare enough that scaling it up is acceptable noise.)
    """
    try:
        fv = float(v)
    except Exception:
        return 0.0
    return fv * 100.0 if fv <= 5.0 else fv


def _pacing_bar(used_pct: float, elapsed_pct: float, width: int = 20) -> str:
    """Tiny ASCII bar comparing used (#) vs elapsed (|) over 0-100%."""
    try:
        u = max(0.0, min(100.0, _to_pct(used_pct)))
        e = max(0.0, min(100.0, _to_pct(elapsed_pct)))
    except Exception:
        return '[?]'
    u_cells = int(round(u / 100.0 * width))
    e_cells = int(round(e / 100.0 * width))
    cells = []
    for i in range(width):
        if i < u_cells and i == e_cells:
            cells.append('X')  # overlap right at the elapsed mark
        elif i < u_cells:
            cells.append('#')
        elif i == e_cells:
            cells.append('|')
        else:
            cells.append('-')
    return '[' + ''.join(cells) + f'] used={u:.0f}%  elapsed={e:.0f}%'


def render_md(s: dict) -> str:
    md = f"# S&P 500 Mastery — Progress Dashboard\n\n"
    md += f"_Snapshot at {s['ts']}_\n\n"
    md += f"## 🎯 Mastery: **{s['mastered']['count']}/502 ({s['mastered']['pct_sp500']:.1f}%)**\n\n"

    # Claude usage pacing (from dashboard/pacing_state.json)
    if 'pacing' in s:
        p = s['pacing']
        if 'error' in p:
            md += f"## Claude usage pacing\n\n_⚠️ {p['error']}_\n\n"
        else:
            regime = p.get('regime') or p.get('regime_name') or 'unknown'
            emoji = _pacing_regime_emoji(regime)
            used_pct = p.get('week_used_pct')
            elapsed_pct = p.get('week_elapsed_pct')
            pace_ratio = p.get('pace_ratio')
            rec_model = p.get('recommended_model_default') or p.get('recommended_model') or 'n/a'
            hours_until_reset = p.get('hours_until_reset')
            override = p.get('override') or p.get('manual_override')
            md += "## Claude usage pacing\n\n"
            md += f"- Regime: {emoji} **{regime}**"
            if override:
                md += f"  _(override: `{override}`)_"
            md += "\n"
            if used_pct is not None:
                try:
                    md += f"- Week used: **{_to_pct(used_pct):.1f}%**\n"
                except Exception:
                    md += f"- Week used: {used_pct}\n"
            if elapsed_pct is not None:
                try:
                    md += f"- Week elapsed: **{_to_pct(elapsed_pct):.1f}%**\n"
                except Exception:
                    md += f"- Week elapsed: {elapsed_pct}\n"
            if pace_ratio is not None:
                try:
                    md += f"- Pace ratio (used/elapsed): **{float(pace_ratio):.2f}**\n"
                except Exception:
                    md += f"- Pace ratio: {pace_ratio}\n"
            md += f"- Recommended model default: **{rec_model}**\n"
            if hours_until_reset is not None:
                try:
                    md += f"- Hours until weekly reset: **{float(hours_until_reset):.1f}h**\n"
                except Exception:
                    md += f"- Hours until weekly reset: {hours_until_reset}\n"
            if used_pct is not None and elapsed_pct is not None:
                try:
                    md += f"\n```\n{_pacing_bar(float(used_pct), float(elapsed_pct))}\n```\n"
                except Exception:
                    pass
            if 'updated_at' in p:
                md += f"\n_pacing snapshot: {p['updated_at']}_\n"
            md += "\n"
    else:
        md += "## Claude usage pacing\n\n_pacing_state.json not yet written by usage_pacing_daemon.py_\n\n"

    md += "## Pipeline runs\n\n| Pipeline | Runs | Mastered |\n|---|---|---|\n"
    for k in [k for k in s.keys() if k.startswith('pipeline_')]:
        v = s[k]
        md += f"| {k.replace('pipeline_','')} | {v['runs']} | {v['mastered']} |\n"

    if 'discovery' in s:
        md += f"\n## Discovery daemon\n- Reports: {s['discovery']['reports_count']}\n- Queue size: {s['discovery']['queue_size']}\n- Last report: {s['discovery']['last_report']}\n"

    if 'overseer' in s:
        md += f"\n## Overseer\n- Cycles: {s['overseer']['cycles_run']}\n- Last: {s['overseer']['last_cycle']}\n- Recommendations file: {s['overseer']['has_recommendations']}\n"

    if 'external_repos' in s:
        md += f"\n## External repos cloned ({len(s['external_repos'])})\n"
        for r in s['external_repos']: md += f"- {r}\n"

    if 'scripts' in s:
        md += f"\n## Pipeline scripts ({len(s['scripts'])})\n"
        for sc in s['scripts']: md += f"- {sc}\n"

    md += f"\n## Mastered tickers ({len(s['mastered']['tickers'])})\n"
    md += ", ".join(s['mastered']['tickers'])
    md += "\n"
    return md


def main():
    DASH.mkdir(parents=True, exist_ok=True)

    # Read previous count for change detection
    _prev_count: int = -1
    _state_path = DASH / "state.json"
    if _state_path.exists():
        try:
            _prev_count = json.loads(_state_path.read_text()).get("mastered", {}).get("count", -1)
        except Exception:
            pass

    state = collect_state()
    (DASH / "state.json").write_text(json.dumps(state, indent=2, default=str))
    (DASH / "dashboard.md").write_text(render_md(state))
    # Append to history
    hist_path = DASH / "history.jsonl"
    with open(hist_path, 'a') as f:
        f.write(json.dumps({'ts': state['ts'], 'mastered_count': state['mastered']['count'],
                            'pipeline_runs': {k: v['runs'] for k, v in state.items() if k.startswith('pipeline_')}}) + '\n')
    print(f"dashboard updated: {state['mastered']['count']}/502 mastered ({state['mastered']['pct_sp500']:.1f}%)")

    # Publish event if mastery count changed
    new_count = state["mastered"]["count"]
    if _EB and new_count != _prev_count:
        try:
            _EB.publish_from_anywhere("mastery_count_changed", {
                "prev_count": _prev_count,
                "new_count": new_count,
                "delta": new_count - _prev_count if _prev_count >= 0 else None,
                "pct_sp500": state["mastered"]["pct_sp500"],
            }, source="progress_dashboard")
        except Exception:
            pass


if __name__ == '__main__':
    main()
