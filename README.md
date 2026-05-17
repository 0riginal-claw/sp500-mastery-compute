# sp500-mastery-compute

S&P 500 XGBoost backtest sweep — runs on GitHub Actions ubuntu-latest runners.

Triggered remotely via `workflow_dispatch` from `multi_cloud_dispatcher.py`.
Results commit back to `backtests/<ticker>/<strategy>/result.json`.

See `.github/workflows/sweep.yml` for inputs.
