# Multi-Cloud XGBoost Sweep — Setup Guide

One-page reference. Full architecture: `reports/distributed_compute_options.md`.

---

## Table of contents

1. [Step 1 — GitHub Actions adapter](#step-1--github-actions-adapter)
2. [Step 2 — Modal adapter](#step-2--modal-adapter)
3. [Step 3 — Run the dispatcher](#step-3--run-the-dispatcher)
4. [Step 4 — Add jobs to the queue](#step-4--add-jobs-to-the-queue)
5. [Step 5 — mac_local adapter (Phase 3 — active by default)](#step-5--mac_local-adapter-phase-3--active-by-default)
6. [Step 6 — Enable Phase 2 adapters (optional)](#step-6--enable-phase-2-adapters-optional)
7. [Step 7 — Bacalhau adapter (Phase 4)](#step-7--bacalhau-adapter-phase-4)
8. [Step 8 — circleci_oss adapter](#step-8--circleci_oss-adapter)
9. [Step 9 — drone_ci adapter (self-hosted Drone CI)](#step-9--drone_ci-adapter-self-hosted-drone-ci)

Supporting sections: [Quota safety](#quota-safety) · [Files created by this system](#files-created-by-this-system) · [Cost summary](#cost-summary)

---

## What this system does

Reads `sweeps/queue.txt` (one `<script> <ticker> <strategy>` job per line), picks
the cloud with the most remaining free quota, submits the job via that cloud's
adapter, tracks quota consumption in `sweeps/cloud_usage.json`, and polls
`backtests/<ticker>/<strategy>/result.json` for completion. Stops sending jobs to
any cloud that has consumed 80% of its monthly quota.

Phase 1 adapters (production-ready): **GitHub Actions**, **Modal**.
Phase 2 stubs (wired, not yet live): Oracle A1, GCP, AWS, Render, Railway, Fly.
Phase 3 adapter (live, last-resort): **mac_local** — runs jobs on this Mac with
hard CPU/RAM/load/worker caps. Selected only after all remote clouds are full.

---

## Step 1 — GitHub Actions adapter

### 1a. Make your repo public (recommended)

Public repos get **unlimited** GitHub Actions minutes. Private repos cap at
2,000 min/month on the Free plan (~4,000 jobs at 0.5 min each).

Settings > Danger Zone > Change visibility > Make public

### 1b. Create a PAT

GitHub > Settings > Developer settings > Personal access tokens > Fine-grained tokens

Permissions needed:
- `Actions: Write` (to trigger workflow_dispatch)
- `Contents: Write` (for the rollup job to commit results back)

Token scope: just the sp500-ticker-mastery repo.

Save the token value. You will never see it again.

### 1c. Add repo secrets

Repo > Settings > Secrets and variables > Actions > New repository secret:

| Secret name | Value |
|---|---|
| `BACKTEST_GITHUB_TOKEN` | PAT from step 1b |

### 1d. Set environment variables for the dispatcher

```bash
export GITHUB_TOKEN="ghp_..."          # PAT from step 1b
export GITHUB_OWNER="youruser"         # GitHub username or org
export GITHUB_REPO="sp500-ticker-mastery"
export GITHUB_WORKFLOW_ID="sweep.yml"
export GITHUB_BRANCH="main"
export GITHUB_IS_PUBLIC="true"         # set "false" if private
```

Add these to `~/.zshrc` or a `.env` file loaded by the dispatcher.

### 1e. Verify the workflow file is on the default branch

The file must be at `.github/workflows/sweep.yml` on the branch you
set in `GITHUB_BRANCH`. Push it if needed:

```bash
git add .github/workflows/sweep.yml
git commit -m "feat: add xgb-sweep workflow"
git push
```

### 1f. Test with a single job

```bash
# Via dispatcher dry-run (no real API call)
python scripts/multi_cloud_dispatcher.py --dry-run

# Via GitHub CLI (real dispatch, one job)
gh workflow run sweep.yml \
  --field ticker=AAPL \
  --field strategy=ORB \
  --field script=scripts/backtest_xgb_v8.py
```

---

## Step 2 — Modal adapter

### 2a. Sign up

https://modal.com — email signup, no credit card required.

Starter plan: $30 free credit per month. Auto-pauses when credit hits $0.
No surprise charges possible.

### 2b. Install Modal SDK and authenticate

```bash
pip install modal
modal token new    # opens browser, saves token to ~/.modal.toml
```

This writes `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` to `~/.modal.toml`.
The dispatcher also reads them from environment variables if you prefer:

```bash
export MODAL_TOKEN_ID="ak-..."
export MODAL_TOKEN_SECRET="as-..."
```

### 2c. Deploy the worker function

```bash
modal deploy scripts/modal_worker.py
```

This uploads the project code as a Modal Mount and registers `run_backtest`
as a remote function. Takes ~60 seconds on first deploy (image build).
Subsequent deploys are incremental.

### 2d. Test with a single job

```bash
modal run scripts/modal_worker.py \
  --ticker AAPL \
  --strategy ORB \
  --script scripts/backtest_xgb_v8.py
```

You should see JSON output and a `result.json` written to `backtests/AAPL/ORB/`.

### 2e. Create a Modal Secret (optional but recommended)

If your backtest scripts need API keys at runtime:

```bash
modal secret create sp500-secrets \
  ALPACA_KEY=... \
  ALPACA_SECRET=... \
  POLYGON_KEY=...
```

The worker references `sp500-secrets` automatically (see `modal_worker.py`).

---

## Step 3 — Run the dispatcher

### Dry-run (inspect routing decisions, no real submissions)

```bash
python scripts/multi_cloud_dispatcher.py --dry-run
```

### Single pass (submit all pending jobs, then exit)

```bash
python scripts/multi_cloud_dispatcher.py
```

### Daemon mode (runs forever, re-polls queue every 30 s)

```bash
python scripts/multi_cloud_dispatcher.py --daemon
```

### Simulate 50 mock jobs (test quota logic, no API calls)

```bash
python scripts/multi_cloud_dispatcher.py --simulate 50
```

### Reset quota counters at month rollover

```bash
python scripts/multi_cloud_dispatcher.py --reset-usage
```

---

## Step 4 — Add jobs to the queue

Each line in `sweeps/queue.txt`:
```
scripts/backtest_xgb_v8.py AAPL ORB
scripts/backtest_xgb_v8.py AAPL VWAP
```

Queue is re-read on every dispatch pass. You can append jobs while the daemon
is running — they will be picked up on the next poll.

---

## Step 5 — mac_local adapter (Phase 3 — active by default)

The Mac-local adapter is **enabled by default** in `cloud_usage.json` but is
always the last cloud selected.  It provides an overflow valve when every
remote cloud is at capacity.

### Hard safety caps (load-avg-937 prevention)

All four checks must pass before a job is spawned locally:

| Cap field | Default | What it measures |
|---|---|---|
| `max_cpu_pct` | 60 % | `psutil.cpu_percent(interval=0.5)` |
| `max_mem_pct` | 70 % | `psutil.virtual_memory().percent` |
| `max_load` | 8.0 | `psutil.getloadavg()[0]` (1-minute) |
| `max_workers` | 4 | Active backtest subprocesses in this session |

If any cap is exceeded the adapter raises `MacCapExceeded` and the job is
deferred — no retry storm, no runaway process trees.

Change caps in `sweeps/cloud_usage.json` under the `mac_local` key.  Do not
raise `max_load` above the physical core count (8) or `max_workers` above 4
without profiling per-job memory consumption first.

### Routing — Mac is always last

```
github_actions → modal → oracle_a1 → (other phase-2 stubs) → mac_local
```

The dispatcher calls `pick_cloud(prefer_remote=True)` which tries every enabled
remote cloud before falling back to `mac_local`.  For jobs the caller marks as
small (`prefer_remote=False`), Mac enters normal headroom ranking.

### Simulating Mac fallback

```bash
# Mac gets only a few jobs when remotes have capacity
python scripts/multi_cloud_dispatcher.py --simulate 100

# Mac absorbs all overflow when remotes are saturated
python scripts/multi_cloud_dispatcher.py --simulate 100 --sim-throttle-remote
```

### Per-job output

Each Mac job writes stdout/stderr to `logs/mac_<job_id>.log`.
The submission receipt includes `pid` and `log` path for debugging.

---

## Step 6 — Enable Phase 2 adapters (optional)

Edit `sweeps/cloud_usage.json` and set `"enabled": true` for any cloud, then
supply the corresponding environment variables:

| Cloud | Enable field | Environment variables needed |
|---|---|---|
| Oracle A1 | `oracle_a1.enabled` | `ORACLE_A1_HOST`, `ORACLE_A1_SSH_KEY` |
| GCP | `google_cloud.enabled` | `GCP_INSTANCE_IP`, `GCP_SSH_KEY` |
| AWS | `aws_free.enabled` | `AWS_INSTANCE_IP`, `AWS_SSH_KEY` |
| Render | `render.enabled` | `RENDER_API_KEY`, `RENDER_SERVICE_ID` |
| Railway | `railway.enabled` | `RAILWAY_API_TOKEN`, `RAILWAY_PROJECT_ID` |
| Fly.io | `fly_io.enabled` | `FLY_API_TOKEN`, `FLY_APP_NAME` |

Phase 2 adapter implementations are stubs — they log a dry-run message until
you wire up the real API calls in `multi_cloud_dispatcher.py`.

---

## Quota safety

The dispatcher stops sending new jobs to any cloud once it has consumed
`safety_margin_pct`% of its monthly quota (default: 80%).

To see current headroom for all clouds:

```bash
python3 - <<'EOF'
import json
from pathlib import Path
data = json.loads(Path("sweeps/cloud_usage.json").read_text())
for k, v in data.items():
    if k.startswith("_") or not v.get("enabled"):
        continue
    model = v.get("billing_model", "?")
    if model == "minutes":
        used = v.get("used_min_this_month", 0)
        quota = v.get("quota_min", 0)
        print(f"{k:20s}  {used:.0f}/{quota} min  ({used/quota*100:.1f}%)")
    elif model == "credit_usd":
        used = v.get("used_credit_this_month", 0)
        quota = v.get("quota_credit", 0)
        print(f"{k:20s}  ${used:.2f}/${quota:.2f}  ({used/quota*100:.1f}%)")
    else:
        inf = v.get("in_flight_jobs", 0)
        mx  = v.get("max_concurrent", v.get("max_concurrent_jobs", v.get("max_concurrent_containers", 1)))
        print(f"{k:20s}  {inf}/{mx} in-flight")
EOF
```

---

## Files created by this system

| File | Purpose |
|---|---|
| `scripts/multi_cloud_dispatcher.py` | Main dispatcher daemon |
| `scripts/modal_worker.py` | Modal remote function |
| `.github/workflows/sweep.yml` | GitHub Actions matrix workflow |
| `sweeps/queue.txt` | Job queue (one job per line) |
| `sweeps/cloud_usage.json` | Quota tracker (read/written by dispatcher) |
| `sweeps/last_run_summary.json` | Written by GitHub Actions rollup job |
| `backtests/<ticker>/<strategy>/result.json` | Per-job output (polled by dispatcher) |
| `logs/dispatcher.log` | Dispatcher run log |
| `logs/dispatch_YYYYMMDD_HHMMSS.json` | Per-pass submission audit log |

---

## Cost summary

| Cloud | Monthly free budget | Our sweep cost (2,510 jobs) | Sweeps/month free |
|---|---|---|---|
| GitHub Actions (public) | Unlimited | $0.00 | Unlimited |
| GitHub Actions (private) | 2,000 min | ~1,255 min (~63%) | ~1.5 full sweeps |
| Modal Starter | $30 credit | ~$0.50 | ~60 sweeps |
| Oracle Ampere A1 | Always free | $0.00 | Unlimited |
| CircleCI OSS | 400k credits ≈ 40k min | ~1,255 min (~3%) | ~30 full sweeps |

**Public repo + Modal = effectively unlimited sweeps at zero cost.**

---

## Step 7 — Bacalhau adapter (Phase 4)

Bacalhau (https://bacalhau.org) is a public, permissionless, Docker-native
decentralised compute network. It is free to use with no wallet or account
required. The dispatcher caps at 20 concurrent jobs as a courtesy limit.

### 7a. Install the Bacalhau CLI

```bash
curl -sL https://get.bacalhau.org/install.sh | bash
```

Verify: `bacalhau version`

The adapter checks for the CLI on PATH at import time. If it is not found,
`enabled=false` is enforced regardless of the JSON config value, and a
warning is logged with the install hint.

### 7b. Build the backtest Docker image

The Dockerfile at `scripts/bacalhau_backtest.Dockerfile` packages
`backtest_xgb_v8.py` and its dependencies into an image that accepts
`TICKER` and `STRATEGY` environment variables and writes
`/output/result.json`.

```bash
docker build \
  -t ghcr.io/<owner>/sp500-backtest:latest \
  -f scripts/bacalhau_backtest.Dockerfile .
```

Replace `<owner>` with your GitHub username or organisation.

### 7c. Publish the image to GitHub Container Registry (GHCR)

```bash
# Authenticate (requires a PAT with packages:write scope)
echo $GHCR_PAT | docker login ghcr.io -u <owner> --password-stdin

# Push
docker push ghcr.io/<owner>/sp500-backtest:latest
```

Make the package public in the GitHub UI:
`github.com/<owner>?tab=packages` -> select the package -> Package settings ->
Change visibility -> Public. This is required for Bacalhau network nodes to
pull the image without authentication.

### 7d. Update cloud_usage.json

Set the correct image reference and enable the adapter:

```json
"bacalhau": {
  "enabled": true,
  "docker_image": "ghcr.io/<owner>/sp500-backtest:latest"
}
```

The `bacalhau_cli_path` field defaults to `"bacalhau"` (resolved via PATH).
Override it if your binary is at a non-standard location.

### 7e. Test with a dry-run

```bash
# Confirm bacalhau adapter is recognised, no real submissions
python scripts/multi_cloud_dispatcher.py --simulate 30 --dry-run
```

Expected output includes a line like:

```
[DRY-RUN] bacalhau: would submit ticker=... strategy=... image=...
```

### 7f. Quota and safety notes

| Parameter | Default | Notes |
|---|---|---|
| `max_concurrent` | 20 | Soft politeness cap on the public network |
| `max_job_seconds` | 600 | Background poll thread timeout per job |
| CLI required | yes | Adapter raises RuntimeError if CLI not on PATH |
| Cost | Free | Public network; no billing |

---

## Step 8 — circleci_oss adapter

CircleCI offers an Open Source plan that grants **400,000 free credits per month**
to approved public repositories.  At the `docker/small@1x` executor rate
(~10 credits/min) this is equivalent to **~40,000 runner-minutes per month** —
enough for roughly 13,000–40,000 backtest jobs depending on per-job runtime.

### 8a. Apply for the OSS plan

1. Your repository must be **public** on GitHub.
2. Visit https://circleci.com/open-source/ and click "Apply for OSS credits".
3. Fill out the form with your GitHub org/repo.  Approval typically takes a few
   business days.
4. Once approved, the credits appear automatically on your CircleCI plan page.

### 8b. Create a Personal API Token

CircleCI v2 API requires a Personal API Token (not a project token) to trigger
pipelines programmatically.

1. Log in to https://app.circleci.com
2. Go to **User Settings** (top-right menu) > **Personal API Tokens**
3. Click **Create New Token**, give it a name (e.g. `dispatcher`), copy the value.

You will not be able to see the token again — save it immediately.

### 8c. Set the environment variable

```bash
export CIRCLECI_TOKEN="your-token-here"
```

Add this to `~/.zshrc` or your `.env` file so the dispatcher can find it.

Optionally also set org/repo to avoid editing cloud_usage.json:

```bash
export CIRCLECI_ORG="youruser"       # GitHub username or org
export CIRCLECI_REPO="sp500-ticker-mastery"
```

### 8d. Set project environment variables in CircleCI

The `.circleci/config.yml` pipeline commits results back to the repo using a
GitHub PAT stored as a CircleCI project env var.

1. In CircleCI: **Project Settings** > **Environment Variables** > **Add Variable**

| Name | Value |
|---|---|
| `GH_TOKEN` | GitHub PAT with `Contents: Write` scope (same PAT used for GITHUB_TOKEN) |

### 8e. Add the pipeline config to your repo

Copy `scripts/circleci_oss_config.yml` to `.circleci/config.yml` in your
repository root and push to `main`:

```bash
cp scripts/circleci_oss_config.yml .circleci/config.yml
# Adjust the script invocation inside config.yml (orb_fast.py → your entry point)
git add .circleci/config.yml
git commit -m "feat: add CircleCI OSS parameterised pipeline"
git push
```

Verify CircleCI picks it up: **Projects** > your repo > **Pipelines** — you
should see the pipeline appear (it will show "No workflows ran" until triggered).

### 8f. Enable the adapter

Edit `sweeps/cloud_usage.json` — find the `circleci_oss` block and set:

```json
"enabled": true,
"circleci_org": "youruser",
"circleci_repo": "sp500-ticker-mastery"
```

Leave `used_min_this_month: 0` — the dispatcher updates it automatically.

### 8g. Test with a dry-run

```bash
# No real API call; logs the URL and payload that would be sent
python scripts/multi_cloud_dispatcher.py --dry-run

# Simulate 30 jobs to verify circleci_oss is recognized in quota routing
python scripts/multi_cloud_dispatcher.py --simulate 30 --dry-run
```

Expected output includes a line like:
```
INFO dispatcher — Cloud headroom — remote: {'circleci_oss': '100.0%', ...}
```

### 8h. Monthly reset behavior

The dispatcher tracks consumed minutes in `used_min_this_month`.  CircleCI
resets credits on the **1st of each calendar month** (`month_reset_day: 1`
in the config block).  Reset the dispatcher counter at the same time:

```bash
python scripts/multi_cloud_dispatcher.py --reset-usage
```

Or add a cron job:
```bash
# Run at 00:05 on the 1st of every month
5 0 1 * * cd /path/to/project && python scripts/multi_cloud_dispatcher.py --reset-usage
```

### 8i. Sample `.circleci/config.yml`

The full annotated template is at `scripts/circleci_oss_config.yml`.
Key points:

- **Executor**: `docker/small@1x` (`cimg/python:3.11`) — cheapest executor,
  1 vCPU / 2 GB RAM, ~10 credits/min on OSS plan.
- **Pipeline parameters**: `ticker`, `strategy`, `job_id` — injected by the
  dispatcher on every trigger.
- **Script invocation**: `python scripts/orb_fast.py --ticker ... --strategy ...`
  — adjust to whichever entry-point you use; it must write output to
  `backtests/<ticker>/<strategy>/result.json`.
- **Result delivery** (Option A, default): commit `result.json` back to `main`
  using `GH_TOKEN`.  The dispatcher's `poll_result()` detects it on the next
  `git pull` / filesystem check.
- **Result delivery** (Option B, commented out): store as a CircleCI artifact;
  requires adapting `poll_result()` to download via the CircleCI artifacts API.

### 8j. Concurrency cap

The OSS plan has a **parallelism limit** (typically 30 concurrent jobs).
The `max_concurrent: 30` field in `cloud_usage.json` enforces this in the
dispatcher — no more than 30 jobs will be in-flight at once.

Adjust downward if you observe queueing delays or receive concurrency-limit
errors from the CircleCI API.

---

## Step 9 — drone_ci adapter (self-hosted Drone CI)

### Overview

Drone CI is an open-source, container-native CI system
(https://www.drone.io/enterprise/opensource/).  You self-host the server;
agents ("runners") are stateless Docker containers that friends register
against your server.  There is **no monthly minute quota** — the only
bottleneck is the number of concurrent agent slots friends have donated.

Billing model in `cloud_usage.json`: `concurrent_cap`.
Default `max_concurrent`: 8 (tune to match total donated agent capacity).

### 9a. Environment variables

```bash
export DRONE_SERVER="https://drone.example.com"   # base URL of your Drone server
export DRONE_TOKEN="your-drone-user-token"         # Account > Token in Drone UI
```

The env-var names can be overridden in `cloud_usage.json` without a code change:

```json
"drone_server_url_env": "DRONE_SERVER",
"drone_token_env":      "DRONE_TOKEN"
```

If `DRONE_SERVER` is not set the adapter automatically falls back to dry-run
mode — no network calls are ever made with missing credentials.

### 9b. Enable the adapter

In `sweeps/cloud_usage.json`, update the `drone_ci` block:

```json
"drone_ci": {
  "enabled": true,
  "drone_repo": "youruser/sp500-ticker-mastery",
  "drone_pipeline_branch": "main",
  "max_concurrent": 8
}
```

Set `max_concurrent` to the sum of all `DRONE_RUNNER_CAPACITY` values across
every registered agent.

### 9c. Repo pipeline template

Copy `scripts/drone_ci_pipeline.yml` to `.drone.yml` at the root of your
backtest repo.  The pipeline has three steps:

| Step | Image | What it does |
|---|---|---|
| `setup` | `python:3.11-slim` | `pip install -r requirements.txt` |
| `run-backtest` | `python:3.11-slim` | Runs `$SCRIPT --ticker $TICKER --strategy $STRATEGY`, writes `result.json` |
| `upload-result-rclone` | `rclone/rclone` | Copies `result.json` to GCS/S3 so the dispatcher can poll it |

An alternative upload step using `gh release upload` is included as
commented-out YAML.

The pipeline `trigger.event` is set to `custom` so it fires only on explicit
API triggers from the dispatcher — not on every push.

### 9d. Register a friend's agent

On each friend's machine (requires Docker):

```bash
docker run --detach \
  --env=DRONE_RPC_PROTO=https \
  --env=DRONE_RPC_HOST=drone.example.com \
  --env=DRONE_RPC_SECRET=<shared-rpc-secret> \
  --env=DRONE_RUNNER_CAPACITY=2 \
  --env=DRONE_RUNNER_NAME=$(hostname) \
  --volume=/var/run/docker.sock:/var/run/docker.sock \
  --restart=always \
  --name=drone-runner \
  drone/drone-runner-docker:1
```

- `DRONE_RPC_SECRET` — must match the `DRONE_RPC_SECRET` env var on the server.
- `DRONE_RUNNER_CAPACITY=2` — set to `(CPU cores / 2)` to leave headroom.
- `--restart=always` — agent reconnects automatically after reboots.

### 9e. Test the adapter

```bash
# Dry-run — logs the would-be API call, no real network request
python scripts/multi_cloud_dispatcher.py --dry-run

# Simulate 30 jobs — drone_ci appears in headroom table (enabled=false, skipped)
python scripts/multi_cloud_dispatcher.py --simulate 30 --dry-run
```

When `enabled=false` the dispatcher logs `drone_ci` in the usage table with
100% headroom available (8/8 slots free) but routes no jobs there.
Set `enabled=true` to activate.

### 9f. Verify builds once live

```bash
# Via Drone CLI (install: https://docs.drone.io/cli/install/)
drone build ls youruser/sp500-ticker-mastery

# Or open the Drone UI:
# https://drone.example.com/youruser/sp500-ticker-mastery
```
