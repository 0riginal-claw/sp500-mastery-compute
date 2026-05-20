"""
run_wave.py — orchestrate a full wave: N tickers × M strategies → mastery files + aggregate report.

Dispatch modes:
  --dispatch-mode cloud  (default) Enqueue each (ticker, variant) via cloud_dispatch.enqueue_job,
                         poll for run_meta.json before aggregating.
  --dispatch-mode local  Legacy subprocess.run behavior (original).

Usage:
    python run_wave.py --wave 1 --tickers AAPL,MSFT,FTNT --variants 1A,1B
                       [--dispatch-mode {cloud,local}] [--dry-run]
"""

import argparse, json, os, subprocess, sys, time
from pathlib import Path

WORK = "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/s&p500-ticker-mastery"
PY = "/Users/orginal/.venvs/sp500-mastery/bin/python"

_CLOUD_DISPATCH_ROOT = Path(WORK) / "scripts"
CLOUD_POLL_INTERVAL = 15
CLOUD_JOB_TIMEOUT = 3600


def _try_import_cloud_dispatch():
    try:
        if str(_CLOUD_DISPATCH_ROOT) not in sys.path:
            sys.path.insert(0, str(_CLOUD_DISPATCH_ROOT))
        import cloud_dispatch  # type: ignore[import]
        return cloud_dispatch
    except ImportError as exc:
        print(f"[cloud_dispatch] import failed ({exc}) — falling back to local mode", file=sys.stderr)
        return None

def run_one(ticker, variant, output_dir):
    cmd = [PY, f"{WORK}/scripts/backtest_v2.py",
           "--ticker", ticker, "--strategy", variant, "--output-dir", output_dir]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return {'ticker': ticker, 'variant': variant, 'output_dir': output_dir,
            'stdout': r.stdout, 'stderr': r.stderr, 'returncode': r.returncode}

def aggregate(wave_n, variant, results):
    rows = []
    for r in results:
        if r['returncode'] != 0:
            rows.append({'ticker': r['ticker'], 'variant': r['variant'], 'status': 'ERROR', 'stderr': r['stderr'][:200]})
            continue
        meta_path = f"{r['output_dir']}/run_meta.json"
        if not os.path.exists(meta_path):
            rows.append({'ticker': r['ticker'], 'variant': r['variant'], 'status': 'NO_META'})
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        m = meta.get('metrics_oos_aggregate', {})
        rows.append({
            'ticker': r['ticker'],
            'variant': r['variant'],
            'n_trades': m.get('n_trades', 0),
            'win_rate': m.get('win_rate'),
            'profit_factor': m.get('profit_factor'),
            'total_return_pct': m.get('total_return_pct'),
            'max_drawdown_pct': m.get('max_drawdown_pct'),
            'sl_exits_pct': m.get('sl_exits_pct'),
            'tp_exits_pct': m.get('tp_exits_pct'),
            'eod_exits_pct': m.get('eod_exits_pct'),
            'time_exits_pct': m.get('time_exits_pct'),
            'signal_count': meta.get('signal_count', 0),
        })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wave', type=int, required=True)
    ap.add_argument('--tickers', required=True, help='comma-separated list')
    ap.add_argument('--variants', required=True, help='comma-separated list of strategy variant codes')
    ap.add_argument('--dispatch-mode', choices=['local', 'cloud'], default='cloud',
                    help="'cloud' enqueues via cloud_dispatch (default); 'local' uses subprocess directly.")
    ap.add_argument('--dry-run', action='store_true',
                    help='Log what would be enqueued without dispatching (cloud mode only).')
    args = ap.parse_args()

    tickers = args.tickers.split(',')
    variants = args.variants.split(',')

    base = f"{WORK}/backtests_wave{args.wave}"
    os.makedirs(base, exist_ok=True)

    all_results = []

    if args.dispatch_mode == 'cloud':
        # Cloud dispatch mode
        cd = _try_import_cloud_dispatch()
        if cd is None:
            print("[warn] cloud_dispatch unavailable — using local fallback", file=sys.stderr)
            args.dispatch_mode = 'local'
        else:
            jobs = [(ticker, variant) for variant in variants for ticker in tickers]
            job_map: dict[str, tuple[str, str, str]] = {}  # job_id -> (ticker, variant, out)

            for ticker, variant in jobs:
                out = f"{base}/{variant}/{ticker}"
                os.makedirs(out, exist_ok=True)
                if args.dry_run:
                    print(f"[DRY-RUN] Would enqueue: ticker={ticker} strategy={variant} script=scripts/backtest_v2.py")
                    continue
                try:
                    job_id = cd.enqueue_job(ticker=ticker, strategy=variant, script="scripts/backtest_v2.py")
                    job_map[job_id] = (ticker, variant, out)
                    print(f"[cloud] Enqueued {ticker}/{variant} job_id={job_id}", flush=True)
                except Exception as exc:
                    print(f"[cloud] Enqueue failed {ticker}/{variant}: {exc} — running locally", file=sys.stderr)
                    r = run_one(ticker, variant, out)
                    all_results.append(r)

            if not args.dry_run:
                # Poll for run_meta.json
                pending = dict(job_map)
                deadline = time.time() + CLOUD_JOB_TIMEOUT
                while pending and time.time() < deadline:
                    done_ids = []
                    for job_id, (ticker, variant, out) in list(pending.items()):
                        meta_path = f"{out}/run_meta.json"
                        if os.path.exists(meta_path):
                            print(f"[poll] {ticker}/{variant} result ready", flush=True)
                            all_results.append({'ticker': ticker, 'variant': variant, 'output_dir': out,
                                                'stdout': '', 'stderr': '', 'returncode': 0})
                            done_ids.append(job_id)
                            continue
                        try:
                            status_info = cd.check_status(job_id)
                            if status_info.get("status") == "failed":
                                print(f"[poll] {ticker}/{variant} cloud job failed", file=sys.stderr)
                                all_results.append({'ticker': ticker, 'variant': variant, 'output_dir': out,
                                                    'stdout': '', 'stderr': 'cloud_failed', 'returncode': 1})
                                done_ids.append(job_id)
                        except Exception:
                            pass
                    for jid in done_ids:
                        pending.pop(jid, None)
                    if pending:
                        time.sleep(CLOUD_POLL_INTERVAL)
                if pending:
                    print(f"[warn] {len(pending)} jobs timed out", file=sys.stderr)

    if args.dispatch_mode == 'local':
        for variant in variants:
            for ticker in tickers:
                out = f"{base}/{variant}/{ticker}"
                os.makedirs(out, exist_ok=True)
                print(f"[wave{args.wave}/{variant}/{ticker}] running...")
                r = run_one(ticker, variant, out)
                if r['returncode'] != 0:
                    print(f"  ERROR: {r['stderr'][:200]}")
                else:
                    for line in r['stdout'].splitlines():
                        if 'OOS:' in line:
                            print(f"  {line.strip()}")
                            break
                all_results.append(r)

    # aggregate
    print(f"\n=== aggregate wave {args.wave} ===")
    rows = aggregate(args.wave, None, all_results)
    import pandas as pd
    summary_df = pd.DataFrame(rows)
    summary_path = f"{WORK}/reports/wave_{args.wave}_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nwrote summary: {summary_path}")
    print(summary_df.to_string(index=False))

    # generate mastery files
    print(f"\n=== generate mastery files ===")
    for r in all_results:
        if r['returncode'] != 0: continue
        t = r['ticker']; v = r['variant']
        out_md = f"{WORK}/tickers/{t}_{v}.md"
        cmd = [PY, f"{WORK}/scripts/generate_mastery_file.py",
               "--ticker", t, "--backtest-dir", r['output_dir'], "--output", out_md]
        rr = subprocess.run(cmd, capture_output=True, text=True)
        if rr.returncode == 0:
            print(f"  → {out_md}")
        else:
            print(f"  ERROR ({t}/{v}): {rr.stderr[:200]}")

if __name__ == '__main__':
    main()
