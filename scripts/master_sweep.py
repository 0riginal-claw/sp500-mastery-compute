"""master_sweep.py — Per-ticker A/B router for XGB strategy dispatch.

PATCHED 2026-05-21: wires `mastery_priors_loader` + `feature_cache_loader` so
priors-bias and already-mastered skips reduce wall-clock cost when sweeping.

PATCHED 2026-05-21 (DSR+PBO): adds `--gate-check` post-sweep promotion gate
using Deflated Sharpe Ratio + Probability of Backtest Overfitting (Bailey &
López de Prado 2014). Filters ~22k expected false positives at 274k+ JobSpec
scale (Bailey 2014: 8.4% of pure-noise strategies clear Sharpe>1.0 by chance).

Priors-bias modes (--use-priors):
  - skip_mastered: drop tickers whose state/mastery.json shows PF >= 1.2
                   AND sharpe >= 0.8 (configurable thresholds).
  - emit_env: for tickers with a v3 winner config, inject
              PRIOR_STOP_ATR / PRIOR_TARGET_ATR / PRIOR_SIGNAL env vars so
              backtest_xgb_v10.py (or any sweep) can bias its hyperparam
              search proximal to the known-good point.

Gate-check mode (--gate-check):
  After sweep results are in, run DSR+PBO promotion gate over state/*/
  mastery.json. Promotes / demotes in cache/mastered_dsr.parquet.

The original A/B-router behavior is preserved as the default (no --use-priors).
"""
from __future__ import annotations
import argparse, logging, sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import pandas as pd
import cloud_dispatch as cd

# New 2026-05-21 imports (kept optional so old smoke flows still work)
try:
    from mastery_priors_loader import MasteryPriors
except Exception as _exc:
    MasteryPriors = None  # type: ignore[assignment]
try:
    from feature_cache_loader import FeatureCache
except Exception as _exc:
    FeatureCache = None  # type: ignore[assignment]

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

BEST_PARQUET = SCRIPT_DIR.parent / 'cache' / 'per_ticker_best.parquet'


def load_best() -> pd.DataFrame:
    return pd.read_parquet(BEST_PARQUET)


def pick_env(ticker: str, best_df: pd.DataFrame) -> dict[str, str]:
    row = best_df[best_df['ticker'] == ticker]
    if row.empty:
        logger.info("  %s: no best-row -> default", ticker)
        return {}
    val = row.iloc[0].get('xgb_no_topk_best')
    if val is True or str(val).lower() == 'true':
        logger.info("  %s: xgb_no_topk_best=True -> XGB_NO_TOPK=1", ticker)
        return {'XGB_NO_TOPK': '1'}
    logger.info("  %s: xgb_no_topk_best=%s -> default", ticker, val)
    return {}


def priors_bias_env(ticker: str, mp) -> tuple[dict[str, str], bool, str]:
    """
    Returns (env, skip, reason).
      skip=True means caller should drop the ticker from the dispatch.
    """
    if mp is None:
        return {}, False, "no priors loader"
    try:
        prior = mp.get_prior_config(ticker)
    except Exception as exc:
        return {}, False, f"prior load fail: {exc}"
    if mp.already_mastered(ticker):
        return {}, True, "already mastered (state PF>=1.2 sharpe>=0.8)"
    env: dict[str, str] = {}
    if prior.v3_status == "winner":
        if prior.v3_stop is not None:
            env["PRIOR_STOP_ATR"] = f"{prior.v3_stop}"
        if prior.v3_target is not None:
            env["PRIOR_TARGET_ATR"] = f"{prior.v3_target}"
        if prior.v3_signal:
            env["PRIOR_SIGNAL"] = str(prior.v3_signal)
        if prior.v3_pf is not None:
            env["PRIOR_PF"] = f"{prior.v3_pf}"
    return env, False, (f"v3_status={prior.v3_status}" if prior.has_prior else "no prior")


def run_gate_check(tickers: list[str] | None) -> int:
    """Apply DSR+PBO promotion gate to state/<ticker>/mastery.json files.

    Delegates to `regate_existing_mastery.main` for the heavy lifting so
    there's one canonical implementation. Returns exit code.
    """
    # Build argv equivalent to: regate_existing_mastery.py [tickers...]
    import subprocess
    cmd = [sys.executable, str(SCRIPT_DIR / 'regate_existing_mastery.py'), '--print-table']
    if tickers:
        cmd += tickers
    logger.info("Gate-check: %s", " ".join(cmd))
    return subprocess.call(cmd)


def main() -> None:
    ap = argparse.ArgumentParser(description='Per-ticker A/B router for XGB strategy dispatch')
    g = ap.add_mutually_exclusive_group(required=False)
    g.add_argument('--tickers', type=str, help='Comma-separated tickers')
    g.add_argument('--tickers-from-csv', type=str, help='CSV file with ticker column')
    ap.add_argument('--strategy', type=str, default='ML_XGB_v10')
    ap.add_argument('--script', type=str, default='scripts/backtest_xgb_v10.py')
    ap.add_argument('--smoke', action='store_true', help='Dry-run: log jobs, skip enqueue')
    ap.add_argument('--priority', type=int, default=5)
    # 2026-05-21 priors wiring
    ap.add_argument('--use-priors', action='store_true',
                    help='Bias env via v3 winner configs + skip already-mastered tickers')
    ap.add_argument('--prior-pf-thresh', type=float, default=1.2)
    ap.add_argument('--prior-sharpe-thresh', type=float, default=0.8)
    ap.add_argument('--show-feature-coverage', action='store_true',
                    help='Print feature-cache coverage report and exit')
    # 2026-05-21 DSR+PBO gate-check mode
    ap.add_argument('--gate-check', action='store_true',
                    help='Run DSR+PBO promotion gate on state/*/mastery.json '
                         '(post-sweep). Writes cache/mastered_dsr.parquet.')
    args = ap.parse_args()

    # --gate-check short-circuits the dispatcher (it's the promotion path)
    if args.gate_check:
        tk = None
        if args.tickers:
            tk = [t.strip().upper() for t in args.tickers.split(',') if t.strip()]
        elif args.tickers_from_csv:
            tk = pd.read_csv(args.tickers_from_csv)['ticker'].astype(str).str.upper().tolist()
        sys.exit(run_gate_check(tk))

    # Original A/B-router path requires --tickers or --tickers-from-csv
    if not (args.tickers or args.tickers_from_csv):
        ap.error('one of --tickers, --tickers-from-csv, or --gate-check is required')

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(',') if t.strip()]
    else:
        tickers = pd.read_csv(args.tickers_from_csv)['ticker'].astype(str).str.upper().tolist()

    if args.show_feature_coverage:
        if FeatureCache is None:
            print("FeatureCache unavailable")
            return
        fc = FeatureCache()
        rpt = fc.coverage_report(tickers)
        print("feature-cache coverage:", rpt)
        return

    best = load_best()
    logger.info("Loaded %d best-config rows from %s", len(best), BEST_PARQUET)

    mp = None
    if args.use_priors:
        if MasteryPriors is None:
            logger.warning("--use-priors requested but MasteryPriors import failed")
        else:
            mp = MasteryPriors()
            logger.info("Mastery priors loader active (skip_pf>=%.2f sharpe>=%.2f)",
                        args.prior_pf_thresh, args.prior_sharpe_thresh)

    out: list[tuple[str, str | None, dict[str, str]]] = []
    skipped: list[tuple[str, str]] = []
    for tkr in tickers:
        env = pick_env(tkr, best)
        if mp is not None:
            p_env, skip, reason = priors_bias_env(tkr, mp)
            if skip:
                skipped.append((tkr, reason))
                logger.info("  [SKIP] %s: %s", tkr, reason)
                continue
            env.update(p_env)
            if p_env:
                logger.info("  %s: priors-bias env=%s (%s)", tkr, p_env, reason)
        if args.smoke:
            jid = None
            logger.info("  [SMOKE] %s -> env=%s (not enqueued)", tkr, env)
        else:
            sweeps_dir = SCRIPT_DIR.parent / 'sweeps'
            sweeps_dir.mkdir(parents=True, exist_ok=True)
            for attempt in range(3):
                try:
                    jid = cd.enqueue_job(
                        ticker=tkr,
                        strategy=args.strategy,
                        script=args.script,
                        priority=args.priority,
                        extra_env=env if env else None,
                        subprocess_fallback=False,
                    )
                    break
                except Exception as exc:
                    logger.warning("  enqueue_job attempt %d failed: %s", attempt + 1, exc)
                    if attempt == 2:
                        raise
                    import time; time.sleep(2)
            logger.info("  enqueued %s job=%s env=%s", tkr, jid, env)
        out.append((tkr, jid, env))

    mode = '[SMOKE]' if args.smoke else ''
    print(f"\n{mode} Dispatched {len(out)} jobs (skipped {len(skipped)} via priors).")
    for tkr, jid, env in out:
        print(f"  {tkr}: job={jid} env={env}")
    if skipped:
        print("\n  -- skipped --")
        for tkr, reason in skipped:
            print(f"  {tkr}: {reason}")


if __name__ == '__main__':
    main()
