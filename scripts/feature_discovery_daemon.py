"""
feature_discovery_daemon.py — runs 24/7 via macOS LaunchAgent (every 6h).

Each cycle:
  1. Search GitHub for new/updated trading/ML feature repos
  2. Query DeepSeek (via OpenClaw) for new feature ideas tied to current mastery state
  3. Scan SSRN/arxiv RSS for quant finance papers
  4. Catalog new ideas not yet in the pipeline
  5. Write timestamped markdown report + JSON inbox queue

Outputs:
  feature_discovery/reports/{YYYY-MM-DD-HHMM}.md  — human-readable digest
  feature_discovery/inbox/queue.json              — prioritized items to integrate
  feature_discovery/state.json                    — cursor state (last GH timestamp, last seen IDs)

Designed to run as a LaunchAgent at ~/Library/LaunchAgents/com.zg.feature_discovery.plist
"""
import concurrent.futures
import json, os, subprocess, sys, time, urllib.request, urllib.parse
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ── event bus (best-effort) ──────────────────────────────────────────────────
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from event_bus import EventBus as _EventBus
    _EB = _EventBus
except Exception:
    _EB = None  # type: ignore[assignment]

# ── wire_candidate emitter (structured output for consumer daemon) ───────────
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from wire_candidate import (
        emit as _wire_emit,
        parse_markdown_blocks as _wire_parse,
        WIRE_CANDIDATE_PROMPT_SUFFIX as _WIRE_SUFFIX,
    )
    _WIRE = True
except Exception:
    _WIRE = False
    _WIRE_SUFFIX = ""

WORK = Path("/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/s&p500-ticker-mastery")
DISCOVERY_ROOT = WORK / "feature_discovery"
REPORTS_DIR = DISCOVERY_ROOT / "reports"
INBOX_DIR = DISCOVERY_ROOT / "inbox"
STATE_PATH = DISCOVERY_ROOT / "state.json"
LOG_PATH = Path("/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/logs/feature_discovery.log")
OPENCLAW = Path("/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/bin/openclaw-gdrive")

# ─────────────────────────────────────────────────────────────────────
# State management — track what we've already seen so we don't repeat
# ─────────────────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_PATH.exists():
        try: return json.loads(STATE_PATH.read_text())
        except Exception: pass
    return {
        'last_run': None,
        'github_seen_repos': [],
        'arxiv_seen_ids': [],
        'ssrn_seen_ids': [],
        'deepseek_query_count': 0,
        'mastered_count_history': [],
    }


def save_state(s: dict):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(s, indent=2, default=str))


def log(msg: str):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    line = f"[{ts}] {msg}\n"
    with open(LOG_PATH, 'a') as f: f.write(line)
    print(line, end='', flush=True)


# ─────────────────────────────────────────────────────────────────────
# Current mastery snapshot — feeds into DeepSeek prompt for context
# ─────────────────────────────────────────────────────────────────────

def get_mastery_snapshot() -> dict:
    mastery_dir = WORK / "mastery_files"
    files = list(mastery_dir.glob("*mastered*.md")) if mastery_dir.exists() else []
    tickers = sorted({p.stem.split("_")[0] for p in files})
    return {
        'count': len(tickers),
        'tickers': tickers,
        'tickers_sample_top10': tickers[:10] if tickers else [],
        'total_sp500_tested': 502,
    }


def get_v3_failing_tickers() -> list:
    """Tickers that have been v3-tested but did NOT master — these are the 'still failing' pool."""
    import glob
    try:
        rows = []
        for p in glob.glob(str(WORK / "backtests_xgb_v3/*/run_meta.json")):
            tk = Path(p).parent.name.replace('_v3', '')
            with open(p) as f: meta = json.load(f); m = meta.get('metrics_oos_aggregate', {})
            wr = m.get('win_rate') or 0; pf = m.get('profit_factor') or 0
            ret = m.get('total_return_pct') or 0; n = m.get('n_trades', 0) or 0
            dd = m.get('max_drawdown_pct') or 0
            mastered = (pf >= 1.5 and wr >= 0.53 and ret > 0 and dd >= -0.03 and n >= 8)
            if not mastered:
                rows.append({'ticker': tk, 'PF': pf, 'WR': wr, 'n': n, 'gap_pf': max(0, 1.5 - pf), 'gap_wr': max(0, 0.53 - wr)})
        rows.sort(key=lambda r: r['gap_pf'] + r['gap_wr'])
        return [r['ticker'] for r in rows[:30]]  # closest 30 to mastery
    except Exception as e:
        log(f"failing tickers query failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────
# GitHub search via public REST API
# ─────────────────────────────────────────────────────────────────────

def github_search(query: str, sort: str = "updated", per_page: int = 30) -> list:
    """Public unauthenticated GH search (rate-limited 10/min). For our cadence (every 6h) this is fine."""
    url = f"https://api.github.com/search/repositories?q={urllib.parse.quote(query)}&sort={sort}&per_page={per_page}"
    try:
        req = urllib.request.Request(url, headers={'Accept': 'application/vnd.github+json',
                                                   'User-Agent': 'feature-discovery-daemon/1.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return data.get('items', [])
    except Exception as e:
        log(f"GH search failed for '{query[:40]}': {e}")
        return []


# ─────────────────────────────────────────────────────────────────────
# DeepSeek synthesis call (via OpenClaw — uses internet)
# ─────────────────────────────────────────────────────────────────────

def deepseek_synthesize(prompt: str, timeout: int = 600) -> str:
    """Call DeepSeek via OpenClaw. Returns assistant message or '' on failure.

    Uses `capability model run` (not `agent`) — one-shot inference, no session.
    Response JSON: {"ok": true, "outputs": [{"text": "..."}]}
    """
    try:
        r = subprocess.run([
            str(OPENCLAW), "capability", "model", "run",
            "--local", "--model", "deepseek/deepseek-v4-flash",
            "--json", "--prompt", prompt,
        ], capture_output=True, text=True, timeout=timeout + 30)
        if r.returncode != 0:
            log(f"DeepSeek returncode={r.returncode}: {r.stderr[:300]}")
            return ""
        stdout = r.stdout.strip()
        if not stdout:
            return ""
        # Parse JSON envelope: {"ok": true, "outputs": [{"text": "..."}]}
        try:
            d = json.loads(stdout)
            # Primary path: outputs[0].text
            outputs = d.get("outputs") or []
            if outputs and isinstance(outputs, list):
                text = outputs[0].get("text", "")
                if text:
                    return text.strip()
            # Fallback: common top-level keys
            for k in ("response", "text", "content", "message", "result"):
                if k in d and d[k]:
                    return str(d[k]).strip()
            return stdout[-5000:]
        except json.JSONDecodeError:
            return stdout[-5000:]
    except Exception as e:
        log(f"DeepSeek call failed: {e}")
        return ""


# ─────────────────────────────────────────────────────────────────────
# Parallel DeepSeek burst — 8 angles per cycle (2026-05-17)
# ─────────────────────────────────────────────────────────────────────

PARALLEL_DEEPSEEK_ANGLES = [
    "factor models",
    "alternative data sources",
    "microstructure patterns",
    "regime detection methods",
    "sentiment / news signals",
    "options flow signals",
    "congressional + insider trades",
    "intraday VWAP / order-flow features",
]
PARALLEL_DEEPSEEK_TIMEOUT = 90  # per angle (DeepSeek can be slow on big prompts)
PARALLEL_DEEPSEEK_MAX_WORKERS = 8  # ThreadPoolExecutor cap

# Use the direct API caller for parallel burst (urllib, ~2s/call) instead of
# the OpenClaw subprocess (~80s/call) — much higher throughput.
sys.path.insert(0, str(WORK / "scripts"))
try:
    from deepseek_direct import call_deepseek_direct as _ds_direct  # type: ignore[import]
    _HAS_DIRECT = True
except Exception:
    _HAS_DIRECT = False
    _ds_direct = None  # type: ignore[assignment]


def _angle_prompt(angle: str, mastery_count: int, failing: list) -> str:
    suffix = _WIRE_SUFFIX if _WIRE else ""
    return (
        f"You are a quant feature engineering analyst. S&P 500 daily mean-reversion XGBoost "
        f"pipeline, {mastery_count}/502 mastered, 722 features (v7/v8). "
        f"Closest-to-mastery tickers (top 5): {failing[:5]}.\n\n"
        f"FOCUS AREA: {angle}\n\n"
        f"Give 3 SPECIFIC new feature ideas in this area. Each must include "
        f"computation recipe + why it would help + estimated PF impact.\n\n"
        f"{suffix}"
    )


def _single_angle(angle: str, mastery_count: int, failing: list) -> dict:
    t0 = time.monotonic()
    if _HAS_DIRECT:
        try:
            text = _ds_direct(
                _angle_prompt(angle, mastery_count, failing),
                timeout=PARALLEL_DEEPSEEK_TIMEOUT, max_tokens=800, temperature=0.4
            )
            return {"ok": bool(text), "angle": angle, "text": text or "", "elapsed_s": round(time.monotonic() - t0, 2)}
        except Exception as exc:
            return {"ok": False, "angle": angle, "text": "", "error": str(exc), "elapsed_s": round(time.monotonic() - t0, 2)}
    # Fallback to openclaw subprocess
    txt = deepseek_synthesize(_angle_prompt(angle, mastery_count, failing), timeout=PARALLEL_DEEPSEEK_TIMEOUT)
    return {"ok": bool(txt), "angle": angle, "text": txt, "elapsed_s": round(time.monotonic() - t0, 2)}


def run_parallel_deepseek_burst(mastery_count: int, failing: list, cycle_ts: str) -> dict:
    """Fire 8 parallel DeepSeek angle-queries and emit WIRE_CANDIDATEs.

    Returns: {"calls": N, "ok": k, "wire_emitted": m, "total_elapsed_s": float}.
    """
    burst_t0 = time.monotonic()
    log(f"[BURST] starting parallel DeepSeek burst — {len(PARALLEL_DEEPSEEK_ANGLES)} angles")
    results: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL_DEEPSEEK_MAX_WORKERS) as ex:
        futs = {ex.submit(_single_angle, a, mastery_count, failing): a for a in PARALLEL_DEEPSEEK_ANGLES}
        for fut in concurrent.futures.as_completed(futs, timeout=PARALLEL_DEEPSEEK_TIMEOUT + 30):
            try:
                results.append(fut.result())
            except Exception as exc:
                results.append({"ok": False, "angle": futs[fut], "text": "", "error": str(exc)})

    wire_emitted = 0
    ok = 0
    if _WIRE:
        all_cands: list = []
        for r in results:
            if not r.get("ok") or not r.get("text"):
                continue
            ok += 1
            try:
                cands = _wire_parse(r["text"], discovered_by="feature_discovery")
                all_cands.extend(cands)
            except Exception as exc:
                log(f"[BURST] parse failed for angle {r['angle']}: {exc}")
        if all_cands:
            md_path = REPORTS_DIR / f"wire_candidates_deepseek_burst_{cycle_ts[:10]}.md"
            try:
                res = _wire_emit(all_cands, discovered_by="feature_discovery", write_md=True, write_jsonl=True, md_path=md_path)
                wire_emitted = res["emitted"]
                log(f"[BURST] emitted {wire_emitted} → md={res['md_path']}")
            except Exception as exc:
                log(f"[BURST] emit failed: {exc}")
    else:
        ok = sum(1 for r in results if r.get("ok"))

    total = time.monotonic() - burst_t0
    summary = {
        "calls": len(results),
        "ok": ok,
        "errors": len(results) - ok,
        "wire_emitted": wire_emitted,
        "total_elapsed_s": round(total, 2),
        "backend": "deepseek_direct" if _HAS_DIRECT else "openclaw_subprocess",
    }
    log(f"[BURST] complete: {summary}")
    return summary


# ─────────────────────────────────────────────────────────────────────
# Main discovery cycle
# ─────────────────────────────────────────────────────────────────────

def main():
    log("=== feature discovery daemon — cycle start ===")
    state = load_state()
    mastery = get_mastery_snapshot()
    failing = get_v3_failing_tickers()
    log(f"mastered: {mastery['count']}/502 ; sample failing-close: {failing[:5]}")
    state['mastered_count_history'].append({
        'ts': datetime.now(timezone.utc).isoformat(),
        'count': mastery['count'],
    })
    # Trim history to last 50 entries
    state['mastered_count_history'] = state['mastered_count_history'][-50:]

    cycle_ts = datetime.now(timezone.utc).strftime('%Y-%m-%d-%H%M')
    queue_items = []

    # ── 1. GitHub recon — skip 4 of every 5 cycles (rate-limit safety) ──
    # At 5-min cadence: effective GH search every 25 min (10/hr cycles, GH 2/hr).
    state['gh_cycle_counter'] = state.get('gh_cycle_counter', 0) + 1
    do_gh_search = (state['gh_cycle_counter'] % 5 == 1)
    log(f"GH search this cycle: {do_gh_search} (counter={state['gh_cycle_counter']})")

    gh_queries = [
        "xgboost trading mean-reversion python",
        "stock features sentiment xgboost",
        "options flow scraper python",
        "insider transactions SEC form 4 parser",
        "VWAP intraday features python",
        "market microstructure features python",
        "FRED economic data python",
        "pandas-ta indicators 2024",
        "vectorbt walk-forward 2024",
        "earnings calendar python 2024",
    ]
    gh_new = []
    seen_gh = set(state['github_seen_repos'])
    if do_gh_search:
        for q in gh_queries:
            items = github_search(q, sort="updated", per_page=10)
            for it in items:
                full_name = it.get('full_name', '')
                stars = it.get('stargazers_count', 0)
                updated = it.get('updated_at', '')
                if full_name and full_name not in seen_gh and stars >= 5:
                    gh_new.append({
                        'full_name': full_name,
                        'stars': stars,
                        'description': (it.get('description') or '')[:200],
                        'updated_at': updated,
                        'query': q,
                        'url': it.get('html_url'),
                        'license': (it.get('license') or {}).get('spdx_id'),
                    })
                    seen_gh.add(full_name)
            time.sleep(8)  # respect 10/min unauthenticated rate
        state['github_seen_repos'] = sorted(seen_gh)[-500:]  # cap to last 500
    log(f"GH: {len(gh_new)} new repos discovered")

    # ── 2. DeepSeek synthesis ────────────────────────────────────────
    ds_prompt = f"""You are a quant feature engineering analyst. We have a S&P 500 daily mean-reversion XGBoost pipeline with {mastery['count']}/502 tickers MASTERED.

CURRENT FEATURE STACK ({mastery['count']}/502 mastered):
- 53 base technical indicators (RSI, EMA, MACD, BBands, ADX, CCI, MFI, Stoch, Aroon, Vortex, ATR, OBV, returns, volume ratio)
- 22 intraday features (opening range, VWAP, time-of-day, gap)
- ~9 alt-data features (EDGAR filings, congressional trades, lobbying)
- 13 trading-insight features (Connors RSI, Donchian, Keltner, Hull MA, TTM squeeze, RSI divergence)
- Per-ticker prob_threshold sweep

CLOSEST-TO-MASTERY FAILING TICKERS (top 5 of 30): {failing[:5]}

NEW REPOS DISCOVERED SINCE LAST RUN: {[r['full_name'] + ' (' + str(r['stars']) + '*)' for r in gh_new[:5]]}

Give me 5 SPECIFIC new feature ideas (~80 words each) that would most likely unlock additional mastery. Each idea must include:
1. Feature name + computation recipe (one-line pseudocode)
2. Why this would help (tied to specific market regime or behavior)
3. Estimated impact (high/med/low) on PF lift
4. Implementation cost (low/med/high)

Be ruthless about specificity — no platitudes. End with JSON: {{"queue": [{{...}}, ...]}} containing the 5 items with keys: name, recipe, why, impact, cost.
""" + (_WIRE_SUFFIX if _WIRE else "")
    log("calling DeepSeek for feature synthesis...")
    ds_response = deepseek_synthesize(ds_prompt, timeout=600)

    # ── 2b. PARALLEL DEEPSEEK BURST (2026-05-17) ──
    # Fire 8 angle-specific DeepSeek queries concurrently — direct API
    # (urllib, ~2s/call), wires the responses through WIRE_CANDIDATE parser
    # so output flows into the standard consumer pipeline.
    burst_summary = {"calls": 0, "ok": 0, "wire_emitted": 0, "errors": 0}
    try:
        burst_summary = run_parallel_deepseek_burst(mastery['count'], failing, cycle_ts)
    except Exception as exc:
        log(f"[BURST] uncaught error: {exc}")
    wire_emitted_count = 0
    if ds_response:
        log(f"DeepSeek returned {len(ds_response)} chars")
        # Try to extract JSON queue from response
        import re
        match = re.search(r'\{\s*"queue"\s*:\s*\[.*?\]\s*\}', ds_response, re.DOTALL)
        if match:
            try:
                ds_items = json.loads(match.group(0))['queue']
                for item in ds_items:
                    queue_items.append({**item, 'source': 'deepseek', 'discovered_at': cycle_ts})
            except Exception as e:
                log(f"DeepSeek JSON parse failed: {e}")
        # Parse structured WIRE_CANDIDATE blocks (additive — does not replace JSON path).
        if _WIRE:
            try:
                wire_cands = _wire_parse(ds_response, discovered_by="feature_discovery")
                if wire_cands:
                    res = _wire_emit(wire_cands, discovered_by="feature_discovery")
                    wire_emitted_count = res["emitted"]
                    log(f"WIRE: emitted {wire_emitted_count} candidates → md={res['md_path']} jsonl={res['jsonl_path']}")
            except Exception as e:
                log(f"WIRE: emit failed: {e}")
        state['deepseek_query_count'] += 1
    else:
        log("DeepSeek returned empty — skipping this cycle's synthesis")

    # Also emit GH discoveries as wire_candidates (structured, low-fidelity — needs human review)
    if _WIRE and gh_new:
        gh_wire = []
        for r in gh_new[:10]:
            gh_wire.append({
                'feature_name': re.sub(r'[^a-z0-9_]', '_', r['full_name'].lower()).strip('_'),
                'description': r['description'][:160] or f"GitHub repo {r['full_name']}",
                'data_source': r['url'],
                'data_source_license': (r.get('license') or 'UNKNOWN'),
                'features_added': 1,
                'shift_1_safe': 'unclear',
                'integration_cost': 'MED',
                'requires_paid_api': 'no',
                'requires_human_review': 'yes',
                'expected_lift_pct': 'unknown',
                'citations': [r['url']],
                'discovered_at': cycle_ts,
            })
        try:
            res = _wire_emit(gh_wire, discovered_by="feature_discovery")
            wire_emitted_count += res["emitted"]
            log(f"WIRE (GH): emitted {res['emitted']} candidates")
        except Exception as e:
            log(f"WIRE (GH): emit failed: {e}")

    # ── 3. arxiv quant-finance RSS scan ──────────────────────────────
    arxiv_new = []
    try:
        arxiv_url = "http://export.arxiv.org/api/query?search_query=cat:q-fin.ST+OR+cat:q-fin.TR&sortBy=lastUpdatedDate&sortOrder=descending&max_results=20"
        req = urllib.request.Request(arxiv_url, headers={'User-Agent': 'feature-discovery-daemon/1.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            xml_text = resp.read().decode('utf-8', errors='replace')
        # Lightweight parse — extract <id>, <title>, <updated>
        import re
        entries = re.findall(r'<entry>(.*?)</entry>', xml_text, re.DOTALL)
        seen_arxiv = set(state.get('arxiv_seen_ids', []))
        for entry in entries[:20]:
            entry_id = (re.search(r'<id>(.*?)</id>', entry) or [None, ''])[1] if re.search(r'<id>(.*?)</id>', entry) else ''
            arxiv_id_match = re.search(r'<id>(.*?)</id>', entry)
            arxiv_id = arxiv_id_match.group(1).strip() if arxiv_id_match else ''
            title_match = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
            title = title_match.group(1).strip() if title_match else ''
            updated_match = re.search(r'<updated>(.*?)</updated>', entry)
            updated = updated_match.group(1).strip() if updated_match else ''
            if arxiv_id and arxiv_id not in seen_arxiv:
                arxiv_new.append({'id': arxiv_id, 'title': title[:200], 'updated': updated})
                seen_arxiv.add(arxiv_id)
        state['arxiv_seen_ids'] = sorted(seen_arxiv)[-300:]
        log(f"arxiv: {len(arxiv_new)} new papers")
    except Exception as e:
        log(f"arxiv scan failed: {e}")

    # ── 4. Compose report ────────────────────────────────────────────
    report_md = f"""# Feature Discovery Report — {cycle_ts}

**Mastery state:** {mastery['count']}/502 ({100*mastery['count']/502:.1f}%)
**Closest-to-mastery failing (top 5):** {', '.join(failing[:5])}
**Cycle index:** {len(state['mastered_count_history'])}

## 1. New GitHub repos ({len(gh_new)})
"""
    for r in gh_new[:15]:
        report_md += f"- **{r['full_name']}** ({r['stars']}⭐, {r['license'] or '?'}) — {r['description']}\n"
        report_md += f"  - url: {r['url']} | query: `{r['query']}`\n"

    report_md += f"\n## 2. DeepSeek-synthesized feature ideas ({len(queue_items)})\n\n"
    if ds_response:
        report_md += "### Full DeepSeek response\n\n"
        report_md += "```\n" + ds_response[:5000] + "\n```\n\n"

    report_md += f"\n## 3. arxiv q-fin papers ({len(arxiv_new)})\n"
    for p in arxiv_new[:10]:
        report_md += f"- [{p['updated'][:10]}] {p['title']}\n  - {p['id']}\n"

    report_md += f"\n---\n_Generated by feature_discovery_daemon.py at {datetime.now(timezone.utc).isoformat()}_\n"

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"{cycle_ts}.md"
    report_path.write_text(report_md)
    log(f"wrote report: {report_path}")

    # ── 5. Update inbox queue ─────────────────────────────────────────
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    queue_path = INBOX_DIR / "queue.json"
    existing_queue = []
    if queue_path.exists():
        try: existing_queue = json.loads(queue_path.read_text())
        except Exception: pass
    # Add GH discoveries to queue
    for r in gh_new[:10]:
        queue_items.append({
            'name': r['full_name'],
            'recipe': f"clone {r['url']} and read README + key source files",
            'why': r['description'],
            'impact': 'unknown — investigate',
            'cost': 'med',
            'source': 'github',
            'discovered_at': cycle_ts,
            'stars': r['stars'],
            'license': r['license'],
        })
    existing_queue.extend(queue_items)
    existing_queue = existing_queue[-200:]  # cap
    queue_path.write_text(json.dumps(existing_queue, indent=2, default=str))
    log(f"queue: {len(existing_queue)} items total (+{len(queue_items)} this cycle)")

    state['last_run'] = datetime.now(timezone.utc).isoformat()
    save_state(state)
    if _EB:
        try:
            _EB.publish_from_anywhere("discovery_report_written", {
                "report_path": str(report_path),
                "queue_items_added": len(queue_items),
                "gh_new": len(gh_new),
                "arxiv_new": len(arxiv_new),
                "wire_candidates_emitted": wire_emitted_count,
            }, source="feature_discovery_daemon")
        except Exception:
            pass
    log(f"=== cycle complete — next run in 6h ===\n")


if __name__ == '__main__':
    try: main()
    except Exception as e:
        log(f"FATAL: {type(e).__name__}: {e}")
        import traceback; log(traceback.format_exc())
        sys.exit(1)
