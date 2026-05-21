# ZGC Daily Backtest Sweep

End-to-end workflow that runs the full S&P 500 ticker mastery backtest on
GitHub Actions (free, unlimited minutes on public repos) once per day and
publishes a leaderboard.

## Overview

| Workflow | Purpose | Trigger |
|---|---|---|
| `.github/workflows/sweep.yml` | Single / few-ticker dispatcher-driven sweep | `workflow_dispatch` |
| `.github/workflows/xsec_matrix.yml` | Full-S&P-500 cross-sectional XGBoost (sharded) | `workflow_dispatch` |
| `.github/workflows/zgc_backtest_sweep.yml` | **Daily scheduled** per-ticker leaderboard sweep | `workflow_dispatch` + cron `0 4 * * *` |

The `zgc_backtest_sweep.yml` workflow is the leaderboard-driven daily sweep
this doc covers.

## Run cadence

- **Cron:** `0 4 * * *` (04:00 UTC daily — about 3 hours after US market close
  at 20:00 EST / 01:00 UTC, leaving buffer for yfinance EOD data to propagate)
- **Manual:** `workflow_dispatch` from the Actions tab or via `gh workflow run`
- **Concurrency:** `max-parallel: 20` (GitHub Free tier cap)
- **Per-shard timeout:** 60 minutes
- **Default sharding:** 25 tickers per matrix cell -> 20 shards for 500 tickers,
  one strategy. Multiple strategies multiply the shard count linearly.

## Inputs (workflow_dispatch)

| Input | Default | Notes |
|---|---|---|
| `strategies` | `ORB` | Comma-separated list of strategy labels (forwarded to backtest as `--strategy`). The v8 model is strategy-agnostic so this is primarily an output-dir / leaderboard label. |
| `max_tickers` | `500` | Cap universe. Use `10` for a smoke test. |
| `tickers_file` | `registry/sp500_tickers.csv` | Source ticker list. First row may be a header. |
| `shard_size` | `25` | Tickers per matrix cell. Smaller -> more parallel jobs, faster wall-clock. Cap = 256 matrix jobs/run total. |
| `open_pr` | `true` | If `true`, posts leaderboard to the most-recent open PR. If `false`, only commits artifact. |
| `extra_env_json` | `{}` | JSON object of per-job env overrides, e.g. `{"XGB_NO_TOPK":"1"}`. |

## How to add new tickers

Append rows to `registry/sp500_tickers.csv` (first column = ticker symbol).
Commit and push. The next scheduled run will pick them up automatically.

```bash
echo "NEWTICKER" >> registry/sp500_tickers.csv
git add registry/sp500_tickers.csv
git commit -m "chore(registry): add NEWTICKER"
git push
```

## How to add new strategies

Either:

1. Pass the strategy label via `strategies` input on a manual dispatch:
   ```
   gh workflow run zgc-backtest-sweep -f strategies="ORB,VWAP,MOMO"
   ```
2. For a strategy that requires different model logic, fork
   `scripts/backtest_xgb_v8.py` into `scripts/backtest_xgb_v8_<strategy>.py`
   and add a branch in the `Run shard backtest` step of the workflow that
   selects the script based on `${STRATEGY}`. Keep the argparse interface
   identical (`--ticker --strategy --out-dir --job-id`) so the rollup
   continues to work unchanged.

## How to read the leaderboard PR comment

After every run, the rollup job posts a comment on the most-recently-updated
open PR with:

- **Header:** run label (UTC timestamp), run ID link, completed / failed / total counts
- **High-quality section:** tickers where PF > 1.2 AND Sharpe > 0.8 (the
  thresholds your strategy filter recipe asks for). Top 30 shown.
- **Top 20 by Profit Factor:** ranked by `pf` field (descending)
- **Top 20 by Sharpe:** ranked by `sharpe` field (descending)

Columns: Ticker, Strategy, PF, Sharpe, WR, n, DD.

If no open PR exists, the comment is skipped — the leaderboard JSON is still
committed to `sweeps/leaderboard_latest.json`.

## How to download per-ticker JSON

Each shard uploads a workflow artifact named `shard-<STRATEGY>_s<NNN>`. The
artifact contains:

```
shard_results/<SHARD_ID>/
  <TICKER1>/
    result.json
    stdout.log
  <TICKER2>/
    result.json
    stdout.log
  ...
  shard_summary.json
```

Download via:

```bash
gh run download <RUN_ID> --repo 0riginal-claw/sp500-mastery-compute \
                        --pattern 'shard-*' \
                        --dir /tmp/zgc_sweep_<RUN_ID>
```

Or for a specific shard:

```bash
gh run download <RUN_ID> --name shard-ORB_s003 --dir /tmp/shard_s003
```

Artifacts retain for 14 days.

## How to filter for PF > 1.2 AND Sharpe > 0.8

Two routes — one from the leaderboard JSON, one from raw per-ticker JSONs.

### From leaderboard_latest.json (preferred — already pre-filtered)

```bash
curl -s https://raw.githubusercontent.com/0riginal-claw/sp500-mastery-compute/main/sweeps/leaderboard_latest.json \
  | jq '.high_quality | map({ticker, strategy, pf, sharpe, wr, n, dd})'
```

### From the full all_rows array (custom thresholds)

```bash
curl -s https://raw.githubusercontent.com/0riginal-claw/sp500-mastery-compute/main/sweeps/leaderboard_latest.json \
  | jq '.all_rows | map(select(.pf > 1.5 and .sharpe > 1.0 and .n > 30))'
```

### Locally from downloaded shard artifacts

```bash
cat /tmp/zgc_sweep_<RUN_ID>/shard-*/shard_results/*/*/result.json \
  | jq -s 'map(select(.profit_factor > 1.2 and .sharpe > 0.8))'
```

(Field names in raw `result.json` are whatever `backtest_xgb_v8.py` emits;
the leaderboard normalizes them to lower-case `pf`/`sharpe`/`wr`/`n`/`dd`.)

## Quota & cost

- **Public repo:** Actions minutes are **unlimited**.
- **Per-job:** ~5–25 minutes wall depending on ticker (yfinance fetch + v8
  feature build + walk-forward XGBoost).
- **Daily cost:** at 500 tickers x ~15 min avg / 20 parallel = ~6.25 hours
  wall clock per run, ~125 runner-hours/day. Free under the public-repo rules.
- **Failure budget:** `fail-fast: false` and per-job `timeout-minutes: 60`
  guard the wall clock. Shards that crash do not block the rollup.

## Smoke test

```bash
# Tiny sanity run (3 tickers, default ORB strategy, no PR comment)
gh workflow run zgc-backtest-sweep \
  --ref feat/zgc-backtest-sweep \
  -f max_tickers=3 \
  -f shard_size=3 \
  -f open_pr=false
```

Then watch:

```bash
gh run watch --repo 0riginal-claw/sp500-mastery-compute
```

## Failure modes & troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Shard times out at 60min | yfinance throttling on some tickers, or feature-build hot loop | Reduce `shard_size`, or run only the failing tickers via `sweep.yml` single-job mode |
| Many tickers report `status=failed` | Upstream yfinance EOD not yet published when cron ran | Re-dispatch manually a few hours later, or move cron later |
| PR comment not posted | No open PR exists | Either open a PR, set `open_pr=false` to skip, or check workflow logs |
| Push race in rollup | 20 shards all racing the leaderboard commit | The workflow already retries with `git pull --rebase --autostash --strategy-option=theirs origin main` up to 8 attempts |
| Matrix > 256 jobs | Too many strategies x tickers / shard_size | Increase `shard_size` (e.g. 50 = 10 shards/strategy) or run strategies on separate dispatches |

## Revert procedure

Disable the cron without deleting the workflow:

```yaml
# In .github/workflows/zgc_backtest_sweep.yml, comment out:
# on:
#   schedule:
#     - cron: '0 4 * * *'
```

Commit and push. The `workflow_dispatch` trigger remains for manual runs.

To fully remove:

```bash
git rm .github/workflows/zgc_backtest_sweep.yml
git commit -m "chore: remove zgc backtest sweep"
git push
```

Backup of the original (if you ever modify and want to revert):
`AI-Tools/backups/gh-actions-backtest-sweep-2026-05-21/`.

## Related

- `sweep.yml` — single-job dispatcher path (used by `multi_cloud_dispatcher.py`)
- `xsec_matrix.yml` — cross-sectional XGBoost sharded over the full universe
- `scripts/backtest_xgb_v8.py` — the model invoked by every shard
- `registry/sp500_tickers.csv` — universe source
