# Cloud Signup Runbook — Manual-Form Providers

**Use this when a provider needs human signup (cat C/D in feasibility report).**
**Goal:** complete each signup in 30-90 seconds; capture token to standard path so downstream automation picks it up.

**Source:** `AI-Tools/reports/free_compute_signup_automation_2026-05-17.md`

---

## Universal preflight (once, ~30s)

```bash
# Use a stable, work-isolated email + a password manager entry per provider
export SIGNUP_EMAIL="${SIGNUP_EMAIL:-orginal_clawdbot@yahoo.com}"
mkdir -p "$HOME/AI-Tools/logs/auto_signup"
```

Open password manager. Generate a unique 24-char password per provider. Save under: `cloud-signup/<provider>`.

---

## 1. Kaggle (Cat C — 5 min)

**What you get:** 30h T4 GPU/wk, 20h TPU/wk, free notebooks.

1. Visit https://www.kaggle.com/account/login → "Register".
2. Email signup (use `$SIGNUP_EMAIL`), confirm email link.
3. After login → Profile → Settings → "Phone Verification". Enter phone, paste SMS code.
4. Settings → API → "Create New Token". Downloads `kaggle.json`.
5. Move + secure:
   ```bash
   mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   pip install --user kaggle
   kaggle competitions list  # smoke test
   ```

**Token path:** `~/.kaggle/kaggle.json`

---

## 2. GitLab CI (Cat C — 4 min if BYO-runner; ~6 min if CC)

**What you get:** 400 CI/CD min/mo on shared runners (with CC) OR unlimited on your own runner.

**Path A — bring-your-own-runner (NO CC needed):**

1. Visit https://gitlab.com/users/sign_up. Email + password. Confirm email.
2. Skip phone/CC prompts (decline shared runners).
3. Top-right avatar → "Edit profile" → "Access Tokens" → name "ci-dispatch", scope `api,read_repository,write_repository`. Copy token.
4. Install your own runner on the Mac:
   ```bash
   brew install gitlab-runner
   gitlab-runner register  # paste token + URL https://gitlab.com
   ```
5. Save token:
   ```bash
   echo "export GITLAB_TOKEN='<paste>'" >> ~/AI-Tools/secrets/cloud_tokens.env
   ```

**Path B — shared runners (need CC):**

3'. When prompted, enter CC details. GitLab places $1 hold, immediately releases.

**Token path:** `$GITLAB_TOKEN` env var (loaded from `~/AI-Tools/secrets/cloud_tokens.env`)

---

## 3. Koyeb (Cat C — 4 min)

**What you get:** 1 free web service (1vCPU/512MB) + 5h Postgres.

1. Visit https://app.koyeb.com/auth/signup.
2. Sign in with GitHub (preferred — auto-bypasses CAPTCHA + CC).
3. Create new organization "freetier-personal".
4. If CC requested anyway: Profile → Billing → enter CC. $1 verify hold.
5. Settings → API → Create token. Copy.

**Token path:** `echo "export KOYEB_TOKEN='<paste>'" >> ~/AI-Tools/secrets/cloud_tokens.env`

---

## 4. Northflank (Cat C — 5 min)

**What you get:** 2 services + 1 DB + 2 cron jobs, always-on (no idle sleep).

1. Visit https://app.northflank.com/signup. Email + password. Confirm email.
2. Add CC ($1 verify, no charge). Required for free tier.
3. Create project → Settings → API → New token.
4. Save:
   ```bash
   echo "export NORTHFLANK_TOKEN='<paste>'" >> ~/AI-Tools/secrets/cloud_tokens.env
   ```

---

## 5. Oracle A1 ARM (Cat D — 20 min, most friction)

**What you get:** 4 OCPU + 24 GB RAM ARM Ampere always-free + 200 GB storage. **Biggest free compute on the list.**

1. Visit https://www.oracle.com/cloud/free/. Click "Start for free".
2. Email + verify. Country, address, phone (real SMS), CC (real, $1 hold, no charge).
3. Wait for "account ready" email (5-30 minutes).
4. Login → Compute → Create Instance.
5. **Critical:** Select `VM.Standard.A1.Flex` shape (the ARM Ampere, NOT x86) and choose home region's ADs to find capacity. **Capacity is constrained — may need 2-3 retries on different ADs.**
6. Generate API key:
   ```bash
   mkdir -p ~/.oci && openssl genrsa -out ~/.oci/oci_api_key.pem 2048
   chmod 600 ~/.oci/oci_api_key.pem
   openssl rsa -pubout -in ~/.oci/oci_api_key.pem -out ~/.oci/oci_api_key_public.pem
   ```
7. Upload `oci_api_key_public.pem` in OCI Console → Profile → API Keys. Copy fingerprint.
8. Create `~/.oci/config`:
   ```ini
   [DEFAULT]
   user=<your-user-OCID>
   fingerprint=<your-fingerprint>
   tenancy=<your-tenancy-OCID>
   region=us-ashburn-1
   key_file=~/.oci/oci_api_key.pem
   ```
9. Install + smoke test:
   ```bash
   brew install oci-cli
   oci os ns get  # should print your tenancy namespace
   ```

**Token path:** `~/.oci/config` + `~/.oci/oci_api_key.pem`

---

## 6. SageMaker Studio Lab (Cat D — 1-5 business days)

**What you get:** Free T4 GPU (4h/session) or 8h CPU/session, 15GB persistent storage. NO AWS account needed.

1. Visit https://studiolab.sagemaker.aws/requestAccount. Submit email + use case ("ML research").
2. **Wait 1-5 business days for approval email.**
3. **Within 7 days of approval**, click registration link in approval email.
4. Set username + password. Verify email again.
5. Login → "Get Free Account". Choose CPU or GPU. Click "Start runtime". Open JupyterLab.

**Token path:** (no CLI token by default — interactive notebook only) — `$SAGEMAKER_SL_USER` env var documents the username for reference.

**Skip the queue:** ask Anthropic / AWS partner / educational org for a referral code.

---

## 7. Paperspace (Cat C — 5 min, now DigitalOcean)

**What you get:** Free Gradient notebook with M4000 GPU (8GB).

1. Visit https://console.paperspace.com/signup (redirects to DigitalOcean).
2. Email + password OR GitHub OAuth.
3. Verify email.
4. Gradient → Notebooks → Create → Free tier M4000.
5. Profile → API Keys → New Key.
6. Save:
   ```bash
   echo "export PAPERSPACE_API_KEY='<paste>'" >> ~/AI-Tools/secrets/cloud_tokens.env
   ```

---

## Token Aggregation

After running these manual signups, run:

```bash
# load all tokens into current shell
source ~/AI-Tools/secrets/cloud_tokens.env

# verify multi_cloud_dispatcher picks them up
python "$HOME/AI-Tools/s&p500-ticker-mastery/scripts/multi_cloud_dispatcher.py" --inventory
```

Expected output: each provider shows `READY` with the captured token, not `DEFERRED — no signup`.

---

## Safety Notes

- **NEVER paste tokens into chat, search queries, or third-party tools.**
- **NEVER commit `~/AI-Tools/secrets/cloud_tokens.env`** — confirm it's in `.gitignore`.
- **If a token leaks:** revoke immediately in the provider's settings UI, then regenerate.
- **If a provider re-prompts for phone/CC on existing account:** likely a security trigger — don't re-add card; instead, contact support.

---

## Source

Generated by auto-signup research subagent 2026-05-17. See:
- `AI-Tools/reports/free_compute_signup_automation_2026-05-17.md`

---

## Appendix — Cat A/B Providers (added 2026-05-17)

The providers below are EITHER (a) no-signup-required OR (b) email-only with no CC/phone friction. Most have a CLI device-flow that supplants web signup entirely.

### 8. Bacalhau (Cat A — NO SIGNUP)

**What you get:** Free access to public demo distributed compute network.

```bash
curl -sL https://get.bacalhau.org/install.sh | bash
bacalhau version  # verify install
bacalhau docker run ubuntu echo "hello"  # smoke test against public network
```

**Token path:** None — public network requires no credentials.

### 9. Vercel (Cat A — CLI device flow)

**What you get:** Hobby plan — 100 GB bandwidth, 100 GB-h functions, unlimited static.

```bash
npm i -g vercel
vercel login  # opens browser device-code flow; choose email or GitHub/GitLab/Bitbucket OAuth
# Token is auto-saved to ~/.local/share/com.vercel.cli/auth.json (macOS path varies)
```

**Token path:** `~/.local/share/com.vercel.cli/auth.json` (managed by CLI; no manual handling)

### 10. Lightning AI (Cat B — email/OAuth, no CC)

**What you get:** 1 free 4-CPU Studio + monthly GPU credit allotment, 10 GB storage.

1. Visit https://lightning.ai/sign-up. Choose Google/GitHub OAuth (fastest) or email+password.
2. CLI:
   ```bash
   pip install lightning
   lightning login  # browser OAuth callback
   ```

**Token path:** `~/.lightning/credentials.json` (CLI-managed)

### 11. GitHub Codespaces (Cat B — bundled with GitHub)

**What you get:** 120 core-hours/month free (Free plan), 15 GB storage.

1. If no GitHub account: https://github.com/signup (username/email/password + phone OTP).
2. Codespaces is auto-enabled — no separate signup.
3. CLI: `gh auth login` (device flow).

**Token path:** managed by `gh` CLI; or `$GH_TOKEN` env var

### 12. Hugging Face Spaces (Cat B — email-only)

**What you get:** CPU Basic (2 vCPU, 16 GB RAM) FREE unlimited (sleeps when idle).

1. Visit https://huggingface.co/join. Email + username + password.
2. Settings → Access Tokens → "New token" (scope: `write`).
3. CLI:
   ```bash
   pip install huggingface_hub
   huggingface-cli login  # paste token
   ```

**Token path:** `~/.cache/huggingface/token`

### 13. Codeberg (Cat B — email-only)

**What you get:** Forgejo Actions CI runners (limited, community-funded).

1. Visit https://codeberg.org/user/sign_up. Username + email + password.
2. Verify email (~30s).
3. Settings → Applications → "Generate New Token".

**Token path:** `echo "export CODEBERG_TOKEN='<paste>'" >> ~/AI-Tools/secrets/cloud_tokens.env`

---

## Updated Priority Order (capacity-weighted, by signup friction)

| Rank | Provider | Why first | Cat |
|------|----------|-----------|-----|
| 1 | Bacalhau | Zero signup | A |
| 2 | Vercel | CLI device-flow | A |
| 3 | HuggingFace Spaces | Free unlimited CPU | B |
| 4 | Lightning AI | Real GPU credits | B |
| 5 | GitHub Codespaces | 120 core-h/mo | B |
| 6 | GitLab CI | 400 CI-min/mo (own runner path = no CC) | B/C |
| 7 | Paperspace | Free M4000/T4 GPU | C |
| 8 | Northflank | 2 always-on services | C |
| 9 | Koyeb | 1 always-on + 5h Postgres | C |
| 10 | Kaggle | 30h T4/wk (best free GPU capacity) | C |
| 11 | Codeberg | CI runners limited | B |
| 12 | SageMaker Studio Lab | Waitlist gates everything | D |
| 13 | Oracle A1 | Biggest spec but biggest friction | D |
