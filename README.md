# sp500-mastery-compute

Compute worker for the S&P 500 ticker-mastery backtest sweep.

This repository is the **remote-compute mirror** of the local research project
at `My Drive/AI-Tools/s&p500-ticker-mastery/`. It contains only the code needed
to run a single backtest job inside a GitHub Actions runner. Source data
(parquet caches, mastery files, prior backtests) is **not** mirrored here.

## What this repo does

The `multi_cloud_dispatcher.py` script (run locally on the user's Mac) submits
backtest jobs to this repository via the GitHub Actions API. Each job triggers
the workflow at `.github/workflows/sweep.yml`, which:

1. Boots an `ubuntu-latest` runner.
2. Installs the dependencies from `requirements.txt`.
3. Runs `scripts/backtest_xgb_v8.py --ticker <T> --strategy <S>`.
4. Uploads the `result.json` artifact and commits it back to `backtests/<T>/<S>/`.

The local dispatcher polls the repo for committed `result.json` files to detect
completion.

## Required GitHub Actions secrets

| Secret | Purpose |
|---|---|
| `BACKTEST_GITHUB_TOKEN` (optional) | PAT with `contents: write` for the rollup job's commit-back step. Falls back to `github.token` if absent. |

## Manual workflow dispatch (smoke test)

From the GitHub UI:
**Actions → xgb-sweep → Run workflow** with `ticker=AAPL`, `strategy=ORB`.

Or via API (replace `<token>`):

```bash
curl -X POST \
  -H "Authorization: Bearer <token>" \
  -H "Accept: application/vnd.github+json" \
  https://api.github.com/repos/0riginal-claw/sp500-mastery-compute/actions/workflows/sweep.yml/dispatches \
  -d '{"ref":"main","inputs":{"ticker":"AAPL","strategy":"ORB"}}'
```

## Known limitations

`scripts/backtest_xgb_v8.py` currently hard-codes the local Google Drive path
for cached parquet input. On a fresh runner with no cache it will exit non-zero
and write a `status: failed` `result.json`. The workflow itself still completes
end-to-end — this validates the plumbing. Adapting the backtest script to fetch
its own data (e.g. from a bucket or yfinance) is the next step.

## Safety

This repo is private. No `.env`, secrets, tokens, credentials, or local
state files are committed (see `.gitignore`).
