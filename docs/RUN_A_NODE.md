# Run a ZG Chain Node — 4 ways, pick one

Run a validator on the **ZG Merkle chain**. Validators earn **1 ZGC per accepted block** (~288 blocks/day → ~8,640 ZGC/month at full uptime). New chain — early validators get a 2x to 5x reward multiplier (see [Rewards](#rewards)).

This document is the canonical operator guide. All four paths boot the same daemon (`scripts/zg_chain_node.py`) — they differ only in where the daemon runs.

---

## Way 1 — One-liner (60 seconds, any Linux / macOS / WSL2 box)

```bash
curl -fsSL https://raw.githubusercontent.com/0riginal-claw/sp500-mastery-compute/main/scripts/zg_chain_join.sh \
  | bash -s -- \
      --bootnode https://seed1.zgc.run \
      --node-id "$(hostname)" \
      --stake 1.0
```

What it does: drops the daemon + state into `~/.zg-chain/`, registers a validator address, kicks off the daemon, prints your reward address.

Requirements: Python 3.8+, outbound HTTPS. No port-forwarding needed (lite mode).

---

## Way 2 — Docker (30 seconds, any Docker host)

```bash
docker run -d --name zg-node \
  -e NODE_ID="$(hostname)" \
  -e PEERS="https://seed1.zgc.run,https://seed2.zgc.run" \
  -e VALIDATOR_ADDR="zg1auto" \
  -p 9933:9933 \
  -v zg-state:/data \
  ghcr.io/0riginal-claw/zg-chain-node:latest
```

Or with `docker compose`:

```bash
curl -fsSL https://raw.githubusercontent.com/0riginal-claw/sp500-mastery-compute/main/docker-compose.yml -o docker-compose.yml
docker compose up -d
docker compose logs -f zg-node
```

Verify health: `curl -sX POST http://localhost:9933/health | jq .`

---

## Way 3 — GitHub Actions (free, zero maintenance)

Fork this repo or add the workflow to **any of your public repos**. GitHub gives every public repo 2000 free minutes/month of `ubuntu-latest` runners; the workflow consumes about ~96 min/day at the default 15-minute cron — well under quota with margin.

1. Add `.github/workflows/zg_validator.yml` (copy from this repo).
2. (Optional) Set repo secrets:
   - `ZG_VALIDATOR_ADDR` — your ZG address (e.g. `zg1myreward...`)
   - `ZG_BOOTNODES` — comma-separated peer URLs (defaults to public seeds)
3. Enable Actions on the repo: **Settings → Actions → General → Allow all actions**.
4. Trigger a manual run: **Actions → zg-validator → Run workflow**.

Every cron tick (every 15 min) boots a 2-minute validator, syncs the chain, submits a heartbeat block, uploads `/tmp/zg-head.json` as a run artifact. Your rewards accrue to `ZG_VALIDATOR_ADDR`.

---

## Way 4 — Replit or Glitch (browser-only, free always-on)

### Replit

1. Fork the template at `templates/replit/` in this repo.
2. Click **Run**.
3. (Optional) Add secret `ZG_VALIDATOR_ADDR` under **Secrets**.
4. Keep the Repl awake with [UptimeRobot](https://uptimerobot.com/) pinging your Repl URL every 5 minutes (free).

### Glitch

1. Remix the template at `templates/glitch/`.
2. Set environment variables in `.env`:
   ```
   PEERS=https://seed1.zgc.run,https://seed2.zgc.run
   ZG_VALIDATOR_ADDR=zg1your_addr
   ```
3. Glitch boots `node index.js`, which spawns the Python daemon and proxies a healthcheck on the public port so Glitch's idle-detector keeps the project alive.

---

## Requirements

| | Minimum |
|--|--|
| OS | Linux / macOS / WSL2 / any Docker host |
| Python | 3.8+ (Way 1) — not needed for Way 2 |
| RAM | 256 MB |
| Disk | 1 GB |
| Network | outbound HTTPS (no port-forward needed for lite mode) |
| Uptime | >= 99% (10 missed blocks → 24h cooldown, no slashing) |

---

## Rewards

```
1 ZGC per accepted block. ~288 blocks/day.

Distribution:
  70% — validator that proposes the block
  20% — referrer of that validator (if any)
  10% — foundation pool (airdrops + bounties)

Early-adopter bonus:
  Nodes   1-10:  5x   (5 ZGC/block)
  Nodes  11-50:  2x
  Nodes  51-100: 1x
  Nodes   100+:  0.5x

Onboarding airdrops:
  GitHub Actions fork that runs successfully for 7 days: 50 ZGC
  Docker pull that runs for 7 days:                       10 ZGC
  Replit / Glitch fork that runs for 7 days:              25 ZGC
  Referrer of a new validator:                            10% of their earnings forever
```

---

## P2P Peer Discovery (auto-replication)

Every node:

1. **Inbound peer registration** — exposes `POST /peers` and `POST /register_validator`. New nodes call this on a known peer to introduce themselves.
2. **Gossip** — every `--sync-secs` (default 15s), the daemon polls each known peer's `/head` and `/peers`, merging new peer URLs into the local peer set. Cap at 256 peers; eviction by oldest-last-seen.
3. **Seed list** — on boot, if no peers are configured, the daemon resolves the DNS TXT record `_zg-seeds.zgc.run` for a seed list. Falls back to a baked-in constant (`SEED_PEERS` in `zg_chain_node.py`) if DNS is blocked.
4. **GitHub seed manifest** — `https://raw.githubusercontent.com/0riginal-claw/sp500-mastery-compute/main/state/seed_peers.json` is fetched on first boot and merged. Anyone running a node can submit a PR to add themselves to the list.

This means a single bootstrap URL is enough — the network discovers itself from there.

---

## Verify your validator

```bash
# Local node head
curl -sX POST http://localhost:9933/head | jq .

# Your balance
curl -sX POST http://localhost:9933/balance -d '{"addr":"zg1your_addr"}' | jq .

# Network top-100
curl -s https://zgc.run/top100 | head
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `cannot import merkle_chain.py` | Make sure `scripts/merkle_chain.py` lives next to `zg_chain_node.py`. |
| `Address already in use :9933` | Another node is running. Stop it (`pkill -f zg_chain_node`) or pick a different `--port`. |
| `peer unreachable` for all seeds | Outbound HTTPS might be blocked. Try `--peers http://...` instead of HTTPS, or run the Docker image with `--network host`. |
| GH Actions workflow disabled after 60 days inactivity | Push any commit to re-enable. The workflow self-pings via cron so this only happens for true-zero-traffic forks. |

---

## Source

- Daemon: [`scripts/zg_chain_node.py`](../scripts/zg_chain_node.py)
- Block format: [`scripts/merkle_chain.py`](../scripts/merkle_chain.py)
- Installer: [`scripts/zg_chain_join.sh`](../scripts/zg_chain_join.sh)
- Issues / questions: open one at https://github.com/0riginal-claw/sp500-mastery-compute/issues
