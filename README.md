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

---

## Run a ZG Chain Node

This repo also hosts the **ZG Chain validator** — a tiny Proof-of-Stake daemon
(~600 lines of stdlib-only Python) that anchors AI-agent state to an on-chain
Merkle log. Validators earn **1 ZGC per accepted block** (~288 blocks/day).
Early validators (first 10 nodes) get a 5x reward multiplier; nodes 11-50 get
2x. Full reward schedule + design notes in
[`docs/RUN_A_NODE.md`](docs/RUN_A_NODE.md).

Pick one of the four boot paths below — they all boot the same daemon
([`scripts/zg_chain_node.py`](scripts/zg_chain_node.py)).

### Way 1: one-line curl (60 seconds)

```bash
curl -fsSL https://raw.githubusercontent.com/0riginal-claw/sp500-mastery-compute/main/scripts/zg_chain_join.sh \
  | bash -s -- \
      --bootnode https://seed1.zgc.run \
      --node-id "$(hostname)" \
      --stake 1.0
```

Requirements: Python 3.8+, outbound HTTPS. No port-forwarding needed.

### Way 2: Docker (30 seconds)

```bash
docker run -d --name zg-node \
  -e NODE_ID="$(hostname)" \
  -e PEERS="https://seed1.zgc.run,https://seed2.zgc.run" \
  -p 9933:9933 \
  -v zg-state:/data \
  ghcr.io/0riginal-claw/zg-chain-node:latest
```

Or via compose:

```bash
curl -fsSL https://raw.githubusercontent.com/0riginal-claw/sp500-mastery-compute/main/docker-compose.yml -o docker-compose.yml
docker compose up -d
```

### Way 3: GitHub Actions (free, zero maintenance)

Fork this repo OR drop
[`.github/workflows/zg_validator.yml`](.github/workflows/zg_validator.yml)
into any of **your** public repos. Every public repo on GitHub gets 2000
free runner minutes/month — enough for one validator on a 15-min cron with
margin to spare.

Set repo secrets:

| Secret | Purpose |
|---|---|
| `ZG_VALIDATOR_ADDR` (optional) | ZG address to credit block rewards (default: auto-derived) |
| `ZG_BOOTNODES` (optional) | Comma-separated peer URLs (default: public seeds) |

Then **Actions → zg-validator → Run workflow** for the first manual smoke,
or wait for the next cron tick.

### Way 4: Replit or Glitch (browser-only)

Fork the template at [`templates/replit/`](templates/replit/) or
[`templates/glitch/`](templates/glitch/) and click Run.

### Verify

```bash
curl -sX POST http://localhost:9933/health | jq .
curl -sX POST http://localhost:9933/head   | jq .
```

### Auto-discovery

New nodes find peers via (1) the `--bootnode` flag, (2) the seed list
[`state/seed_peers.json`](state/seed_peers.json) in this repo, (3) gossip
from any reachable peer's `/peers` endpoint. Want your seed listed publicly?
Open a PR adding your `/health`-responsive URL to `state/seed_peers.json`.

Full operator guide: [`docs/RUN_A_NODE.md`](docs/RUN_A_NODE.md).
