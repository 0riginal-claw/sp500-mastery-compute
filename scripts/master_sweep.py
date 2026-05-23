# autosolve_skip: multi-TF wire — 2026-05-21
"""master_sweep.py — Per-ticker A/B router for XGB strategy dispatch.

PATCHED 2026-05-21: wires `mastery_priors_loader` + `feature_cache_loader` so
priors-bias and already-mastered skips reduce wall-clock cost when sweeping.

PATCHED 2026-05-21 (DSR+PBO): adds `--gate-check` post-sweep promotion gate
using Deflated Sharpe Ratio + Probability of Backtest Overfitting (Bailey &
López de Prado 2014). Filters ~22k expected false positives at 274k+ JobSpec
scale (Bailey 2014: 8.4% of pure-noise strategies clear Sharpe>1.0 by chance).

PATCHED 2026-05-21 (multi-TF): adds `--timeframes` (comma-sep, default
"1Day" for backward compat) and `--strategy-tf-filter` to enumerate the
(ticker × strategy × TF) cross-product. Each generated job carries
BACKTEST_TIMEFRAME in extra_env so backtest_xgb_v10.py routes the loader
to the correct Cache B TF.

  Strategy/TF compatibility matrix (used to skip incoherent combos):
    ORB         -> 1Min/5Min/15Min/30Min
    VWAP        -> 1Min/5Min/15Min/30Min/45Min/1Hour
    v10/default -> 1Hour/1Day
    momentum    -> 1Hour/4Hour/8Hour/12Hour/1Day
    catalyst    -> 1Day
    mean_revert -> 5Min/15Min/30Min

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

# 2026-05-21 multi-TF wire — strategy/timeframe compatibility matrix.
# Maps strategy label -> set of timeframes where the strategy makes sense.
# Used by --strategy-tf-filter to skip incoherent (strategy, TF) cells.
# Mirror of the warn-only check in backtest_xgb_v10.main().
STRATEGY_TF_COMPAT: dict[str, set[str]] = {
    "ORB":         {"1Min", "5Min", "10Min", "15Min", "30Min"},   # 2026-05-22: 10Min added (bridge TF)
    "VWAP":        {"1Min", "5Min", "10Min", "15Min", "30Min", "45Min", "1Hour"},   # 2026-05-22: 10Min
    "v10":         {"1Hour", "1Day"},
    "ML_XGB_v10":  {"1Hour", "1Day"},   # alias used by default --strategy
    "default":     {"1Day"},             # plain "default" sweeps stay daily
    "momentum":    {"1Hour", "4Hour", "8Hour", "12Hour", "1Day"},
    "catalyst":    {"1Day"},
    "mean_revert": {"5Min", "10Min", "15Min", "30Min"},   # 2026-05-22: 10Min added
}

ALL_TIMEFRAMES = [
    "1Min", "5Min", "10Min", "15Min", "30Min", "45Min",   # 2026-05-22: 10Min added
    "1Hour", "4Hour", "8Hour", "12Hour", "1Day",
]


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
    # 2026-05-21 multi-TF wire
    ap.add_argument('--timeframes', type=str, default='1Day',
                    help='Comma-separated Cache B timeframes to sweep across '
                         '(default "1Day" for back-compat). Use "all" for the '
                         '10-TF set: ' + ",".join(ALL_TIMEFRAMES))
    ap.add_argument('--strategy-tf-filter', action='store_true',
                    help='Skip (strategy, TF) combos outside STRATEGY_TF_COMPAT')
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

    # 2026-05-21 multi-TF wire — resolve --timeframes (csv or "all").
    if args.timeframes.strip().lower() == 'all':
        tf_list = list(ALL_TIMEFRAMES)
    else:
        tf_list = [t.strip() for t in args.timeframes.split(',') if t.strip()]
    # validate
    bad_tfs = [t for t in tf_list if t not in ALL_TIMEFRAMES]
    if bad_tfs:
        ap.error(f"unknown timeframes: {bad_tfs}; choose from {ALL_TIMEFRAMES}")
    logger.info("Sweeping %d timeframe(s): %s", len(tf_list), tf_list)

    # Strategy/TF compat filter — drop incoherent combos.
    if args.strategy_tf_filter:
        allowed = STRATEGY_TF_COMPAT.get(args.strategy)
        if allowed is None:
            logger.warning(
                "  --strategy-tf-filter on but strategy=%s not in compat map; "
                "all TFs will run", args.strategy)
            allowed_tfs = tf_list
        else:
            allowed_tfs = [t for t in tf_list if t in allowed]
            dropped = sorted(set(tf_list) - set(allowed_tfs))
            if dropped:
                logger.info("  filter: dropped TFs %s for strategy=%s",
                            dropped, args.strategy)
        tf_list_for_strategy = allowed_tfs
    else:
        tf_list_for_strategy = tf_list

    out: list[tuple[str, str, str | None, dict[str, str]]] = []
    skipped: list[tuple[str, str]] = []
    for tkr in tickers:
        env_base = pick_env(tkr, best)
        if mp is not None:
            p_env, skip, reason = priors_bias_env(tkr, mp)
            if skip:
                skipped.append((tkr, reason))
                logger.info("  [SKIP] %s: %s", tkr, reason)
                continue
            env_base.update(p_env)
            if p_env:
                logger.info("  %s: priors-bias env=%s (%s)", tkr, p_env, reason)

        # Per-TF expansion: one job per (ticker, strategy, TF).
        for tf in tf_list_for_strategy:
            env = dict(env_base)
            env['BACKTEST_TIMEFRAME'] = tf
            if args.smoke:
                jid = None
                logger.info("  [SMOKE] %s tf=%s -> env=%s (not enqueued)",
                            tkr, tf, env)
            else:
                sweeps_dir = SCRIPT_DIR.parent / 'sweeps'
                sweeps_dir.mkdir(parents=True, exist_ok=True)
                # Tag strategy label with TF so sweeps/dispatched.jsonl
                # and downstream rollups can bin by (strategy, TF) without
                # extra parsing. Format: "<strategy>__<tf>" preserves the
                # original strategy as a prefix for grep-ability.
                strategy_tagged = f"{args.strategy}__{tf}"
                for attempt in range(3):
                    try:
                        jid = cd.enqueue_job(
                            ticker=tkr,
                            strategy=strategy_tagged,
                            script=args.script,
                            priority=args.priority,
                            extra_env=env if env else None,
                            subprocess_fallback=False,
                        )
                        break
                    except Exception as exc:
                        logger.warning(
                            "  enqueue_job attempt %d failed: %s",
                            attempt + 1, exc)
                        if attempt == 2:
                            raise
                        import time; time.sleep(2)
                logger.info("  enqueued %s tf=%s job=%s env=%s",
                            tkr, tf, jid, env)
            out.append((tkr, tf, jid, env))

    mode = '[SMOKE]' if args.smoke else ''
    print(f"\n{mode} Dispatched {len(out)} jobs across {len(tf_list_for_strategy)} TF(s) "
          f"(skipped {len(skipped)} via priors).")
    for tkr, tf, jid, env in out:
        print(f"  {tkr} tf={tf}: job={jid} env={env}")
    if skipped:
        print("\n  -- skipped --")
        for tkr, reason in skipped:
            print(f"  {tkr}: {reason}")


if __name__ == '__main__':
    main()
