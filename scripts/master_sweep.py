"""master_sweep.py — Per-ticker A/B router for XGB strategy dispatch."""
from __future__ import annotations
import argparse, logging, sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import pandas as pd
import cloud_dispatch as cd

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

BEST_PARQUET = SCRIPT_DIR.parent / 'cache' / 'per_ticker_best.parquet'


def load_best() -> pd.DataFrame:
    return pd.read_parquet(BEST_PARQUET)


def pick_env(ticker: str, best_df: pd.DataFrame) -> dict[str, str]:
    row = best_df[best_df['ticker'] == ticker]
    if row.empty:
        logger.info("  %s: no best-row → default", ticker)
        return {}
    val = row.iloc[0].get('xgb_no_topk_best')
    if val is True or str(val).lower() == 'true':
        logger.info("  %s: xgb_no_topk_best=True → XGB_NO_TOPK=1", ticker)
        return {'XGB_NO_TOPK': '1'}
    logger.info("  %s: xgb_no_topk_best=%s → default", ticker, val)
    return {}


def main() -> None:
    ap = argparse.ArgumentParser(description='Per-ticker A/B router for XGB strategy dispatch')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--tickers', type=str, help='Comma-separated tickers')
    g.add_argument('--tickers-from-csv', type=str, help='CSV file with ticker column')
    ap.add_argument('--strategy', type=str, default='ML_XGB_v10')
    ap.add_argument('--script', type=str, default='scripts/backtest_xgb_v10.py')
    ap.add_argument('--smoke', action='store_true', help='Dry-run: log jobs, skip enqueue')
    ap.add_argument('--priority', type=int, default=5)
    args = ap.parse_args()

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(',') if t.strip()]
    else:
        tickers = pd.read_csv(args.tickers_from_csv)['ticker'].astype(str).str.upper().tolist()

    best = load_best()
    logger.info("Loaded %d best-config rows from %s", len(best), BEST_PARQUET)

    out: list[tuple[str, str | None, dict[str, str]]] = []
    for tkr in tickers:
        env = pick_env(tkr, best)
        if args.smoke:
            jid = None
            logger.info("  [SMOKE] %s → env=%s (not enqueued)", tkr, env)
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
    print(f"\n{mode} Dispatched {len(out)} jobs.")
    for tkr, jid, env in out:
        print(f"  {tkr}: job={jid} env={env}")


if __name__ == '__main__':
    main()
