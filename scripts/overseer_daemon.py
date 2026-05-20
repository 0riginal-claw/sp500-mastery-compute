"""
overseer_daemon.py — 24/7 CONSTANT PROACTIVE agent.

Runs every 15 min via macOS LaunchAgent. Each cycle:
  1. Snapshot current mission state (mastery count, failing tickers, recent reports)
  2. Call DeepSeek (via OpenClaw) with "what should we do NOW" prompt
  3. Write recommendations to overseer/recommendations.json
  4. Write digest to overseer/last_digest.md
  5. Log everything to logs/overseer.log

Designed to be the ALWAYS-ON ideation loop. Main Claude thread reads
overseer/recommendations.json on each wakeup and executes top items.
"""
import json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path

# ── event bus (best-effort; never crash overseer if import fails) ──
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from event_bus import EventBus as _EventBus
    _EB = _EventBus
except Exception:
    _EB = None  # type: ignore[assignment]

WORK = Path("/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/s&p500-ticker-mastery")
OVERSEER_ROOT = WORK / "overseer"
LOG_PATH = Path("/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/logs/overseer.log")
sys.path.insert(0, str(WORK / "scripts"))
from deepseek_direct import call_deepseek_direct  # noqa: E402


def log(msg: str):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    line = f"[{ts}] {msg}\n"
    with open(LOG_PATH, 'a') as f: f.write(line)
    print(line, end='', flush=True)


def get_state() -> dict:
    """Snapshot of mission state."""
    mastery = sorted({p.stem.split("_")[0] for p in (WORK / "mastery_files").glob("*mastered*.md")}) if (WORK / "mastery_files").exists() else []

    # Recent discovery reports
    reports_dir = WORK / "feature_discovery" / "reports"
    recent_reports = []
    if reports_dir.exists():
        for p in sorted(reports_dir.glob("*.md"))[-5:]:
            recent_reports.append({'path': p.name, 'ts': p.stat().st_mtime})

    # Recent OOS results
    v4_count = len(list((WORK / "backtests_xgb_v4").glob("*/run_meta.json"))) if (WORK / "backtests_xgb_v4").exists() else 0
    v3_count = len(list((WORK / "backtests_xgb_v3").glob("*/run_meta.json"))) if (WORK / "backtests_xgb_v3").exists() else 0

    # Failing tickers (closest to mastery)
    failing_closest = []
    import glob
    for p in glob.glob(str(WORK / "backtests_xgb_v3/*/run_meta.json"))[:300]:
        tk = Path(p).parent.name.replace('_v3', '')
        if tk in mastery: continue
        try:
            with open(p) as fp: m = json.load(fp).get('metrics_oos_aggregate', {})
            wr = m.get('win_rate') or 0; pf = m.get('profit_factor') or 0
            failing_closest.append({'ticker': tk, 'pf': pf, 'wr': wr,
                                    'gap': max(0, 1.5 - pf) + max(0, 0.53 - wr)})
        except: pass
    failing_closest.sort(key=lambda x: x['gap'])

    return {
        'mastered_count': len(mastery),
        'mastered_sample_top10': mastery[:10],
        'mastered_all': mastery,
        'v3_runs': v3_count,
        'v4_runs': v4_count,
        'recent_reports': recent_reports,
        'failing_closest_top10': failing_closest[:10],
        'ts': datetime.now(timezone.utc).isoformat(),
    }


def _run_openclaw(prompt: str, timeout: int, thinking: str, session_id: str) -> str:
    """Direct DeepSeek API call. Returns response text or ''.

    The `thinking` and `session_id` params are kept for backwards-compatibility
    with call_deepseek() but are no longer forwarded.
    Replaces openclaw subprocess (80-90s) with direct urllib call (~2s).
    """
    try:
        return call_deepseek_direct(prompt, timeout=timeout + 5, max_tokens=512, temperature=0.3)
    except RuntimeError as e:
        log(f"DeepSeek call failed: {e}")
        return ""
    except Exception as e:
        log(f"DeepSeek call failed: {e}")
        return ""


RETRY_PROMPT_TEMPLATE = """Quant overseer. {mastered}/502 mastered. Failing: {tickers}.
JSON only: {{"top_actions":["...","...","..."],"tickers_to_focus":["...","...","...","...","..."],"verdict":"one line"}}"""


def call_deepseek(prompt: str, timeout: int = 25, state: dict | None = None) -> str:
    """Call DeepSeek via OpenClaw with retry on empty. Returns extracted text or ''.

    Primary call: --thinking low, timeout 25s (fits in 1-min cron window).
    Retry (once): shorter prompt, --thinking low, same timeout.
    """
    session_id = f"overseer-{int(time.time())}"

    log(f"DeepSeek primary call (thinking=low, timeout={timeout}s)...")
    result = _run_openclaw(prompt, timeout=timeout, thinking="low", session_id=session_id)

    if result:
        return result

    # Retry with minimal prompt and thinking=low
    log("DeepSeek primary empty — retrying with reduced prompt and thinking=low...")
    if state:
        tickers = [t['ticker'] for t in state.get('failing_closest_top10', [])[:5]]
        retry_prompt = RETRY_PROMPT_TEMPLATE.format(
            mastered=state.get('mastered_count', '?'),
            tickers=', '.join(tickers),
        )
    else:
        retry_prompt = 'Reply with JSON: {"top_actions":["run more backtests"],"tickers_to_focus":[],"verdict":"continue current plan"}'

    retry_session_id = f"overseer-retry-{int(time.time())}"
    result = _run_openclaw(retry_prompt, timeout=timeout, thinking="low", session_id=retry_session_id)
    if result:
        log("DeepSeek retry succeeded.")
    else:
        log("DeepSeek retry also empty — skip ideation this cycle.")
    return result


def main():
    log("=== overseer cycle start ===")
    OVERSEER_ROOT.mkdir(parents=True, exist_ok=True)
    state = get_state()
    log(f"state: mastered={state['mastered_count']}/502, v3_runs={state['v3_runs']}, v4_runs={state['v4_runs']}")

    failing_tickers = [t['ticker'] for t in state['failing_closest_top10']]
    # Keep prompt under 500 chars for fast inference in 1-min window
    prompt = (
        f"Quant overseer. {state['mastered_count']}/502 mastered. "
        f"v3={state['v3_runs']} v4={state['v4_runs']}. "
        f"Near-mastery tickers: {failing_tickers[:5]}. "
        "JSON only (no extra text): "
        '{"top_actions":["a1","a2","a3"],"tickers_to_focus":["t1","t2","t3","t4","t5"],"verdict":"one sentence"}'
    )
    assert len(prompt) < 600, f"prompt too long: {len(prompt)}"

    log("calling DeepSeek (thinking=low, timeout=25)...")
    response = call_deepseek(prompt, timeout=25, state=state)
    if response:
        log(f"DeepSeek returned {len(response)} chars")
    else:
        log("DeepSeek empty — skip this cycle's ideation")
        response = ""

    # Try to extract JSON from response
    import re
    rec = {}
    m = re.search(r'\{[\s\S]*"verdict"[\s\S]*?\}', response)
    if m:
        try: rec = json.loads(m.group(0))
        except Exception as e: log(f"JSON parse failed: {e}")

    # Write recommendations
    ts = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')
    rec_full = {
        'ts': datetime.now(timezone.utc).isoformat(),
        'state': state,
        'deepseek_raw': response[:5000],  # cap
        'recommendations': rec,
    }
    (OVERSEER_ROOT / "recommendations.json").write_text(json.dumps(rec_full, indent=2, default=str))
    # Also archive
    archive_dir = OVERSEER_ROOT / "history"
    archive_dir.mkdir(exist_ok=True)
    (archive_dir / f"{ts}.json").write_text(json.dumps(rec_full, indent=2, default=str))

    # Write digest markdown
    digest = f"""# Overseer digest — {ts}

**Mastery:** {state['mastered_count']}/502 ({100*state['mastered_count']/502:.1f}%)
**v3 runs:** {state['v3_runs']} | **v4 runs:** {state['v4_runs']}

## Top failing tickers (closest to mastery)
"""
    for t in state['failing_closest_top10']:
        digest += f"- {t['ticker']}: PF={t['pf']:.2f}, WR={t['wr']:.2%}, gap={t['gap']:.3f}\n"

    digest += f"\n## DeepSeek recommendations\n\n"
    if rec:
        digest += "```json\n" + json.dumps(rec, indent=2, default=str)[:3000] + "\n```\n"
    digest += f"\n## Raw response excerpt\n\n```\n{response[:2500]}\n```\n"

    (OVERSEER_ROOT / "last_digest.md").write_text(digest)
    (archive_dir / f"{ts}.md").write_text(digest)
    log(f"=== overseer cycle complete — wrote digest + recommendations ===\n")
    if _EB:
        try:
            _EB.publish_from_anywhere("overseer_cycle_complete", {
                "mastered_count": state["mastered_count"],
                "v3_runs": state["v3_runs"],
                "verdict": rec.get("verdict", "")[:200],
            }, source="overseer_daemon")
        except Exception:
            pass


if __name__ == '__main__':
    try: main()
    except Exception as e:
        log(f"FATAL: {type(e).__name__}: {e}")
        import traceback; log(traceback.format_exc())
        sys.exit(1)
