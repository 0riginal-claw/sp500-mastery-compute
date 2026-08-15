# Master Free-Tier Compute Signup Runbook
_Paste-and-go. ~5 min per service. Last updated 2026-05-16._

---

## TL;DR Priority Order

1. **Firebase Functions (Spark)** — No CC, fastest signup (Google login → project created in 60 sec), 2M invocations/month. _Catch: Spark blocks outbound to non-Google APIs._
2. **IBM Code Engine Lite** — No CC, fast IBMid signup (~2 min), generous vCPU-sec/GB-sec quota, scale-to-zero containers. _Catch: cold-start on idle._
3. **Northflank (Developer Sandbox)** — CC required (no charge), always-on Docker containers, 2 services + 1 DB. _Catch: CC wall at signup._
4. **Oracle Ampere A1 (Always Free)** — CC required + longest signup (~10–25 min total) + capacity lottery, BUT the most powerful free tier on the planet (4 OCPU / 24 GB RAM, 10 TB egress, perpetual). _Catch: "Out of Host Capacity" is common._

---

## Comparison Matrix

| Service | CC Required | Setup Time | Always-On? | Free Limit Headline | Best For |
|---|---|---|---|---|---|
| Firebase Functions | No | ~2 min | No (event-driven) | 2M invocations, 400K GB-sec, 5 GB egress/mo | Webhooks, HTTP APIs, event handlers |
| IBM Code Engine | No | ~3 min | No (scale-to-zero) | 100K vCPU-sec, 200K GB-sec, 500 build-min/mo | Containerized apps, light APIs |
| Northflank | Yes (no charge) | ~5 min | Yes | 2 services, 2 cron jobs, 1 DB, unlimited builds | Always-on APIs, Docker apps, hobby backends |
| Oracle A1 | Yes ($1 hold) | 10–25 min | Yes | 4 OCPU / 24 GB RAM, 200 GB storage, 10 TB egress | Long-running services, ARM workloads, full VMs |

---

## Quick Decision Tree

```
Need always-on long-running VM with SSH access?
  → Oracle Ampere A1 (most powerful; budget 20 min + CC)

Need always-on Docker container, CC is fine?
  → Northflank (2 services, instant deploys from Git)

Need stateless HTTP functions, NO credit card?
  → Firebase Functions (fastest) OR IBM Code Engine (containers)
    → Outbound calls to non-Google APIs? → IBM Code Engine
    → Google-only integrations / webhooks? → Firebase Functions
```

---

## Service Runbooks

---

## 1. Firebase Functions (Spark Plan) — No CC

**Signup URL:** https://console.firebase.google.com

### Signup Steps (≤2 min)
1. Visit https://console.firebase.google.com
2. Click **Add project** → enter project name (e.g., `my-functions-app`)
3. Skip Google Analytics for speed
4. Click **Create project** — defaults to free Spark plan
5. Wait ~60 sec for initialization → done

### CLI Install
```bash
npm install -g firebase-tools
firebase --version  # verify
```

### First Deploy
```bash
cd my-project
firebase login          # opens browser auth

firebase init functions
# Prompts: select project, language (JavaScript), ESLint (your choice), npm install (Yes)
```

Edit `functions/index.js`:
```javascript
const functions = require("firebase-functions");

exports.helloWorld = functions.https.onRequest((request, response) => {
  response.send("Hello from Firebase Cloud Functions!");
});
```

```bash
firebase deploy --only functions
# Output: ✓ functions[helloWorld] → https://us-central1-<project-id>.cloudfunctions.net/helloWorld
```

### Verify
```bash
curl https://us-central1-<project-id>.cloudfunctions.net/helloWorld
# Expected: "Hello from Firebase Cloud Functions!"
```

### Free Tier Limits
- 2,000,000 invocations/month
- 400,000 GB-seconds/month
- 200,000 CPU-seconds/month
- 5 GB outbound/month
- 60 sec timeout (1st gen only on Spark)

### Gotchas
- **Spark blocks outbound to non-Google services** (no 3rd-party APIs, no payment processors without Blaze upgrade)
- **2nd gen runtime requires Blaze** (pay-as-you-go); Spark = 1st gen only
- **Cold starts:** 1–5 sec on Spark (shared resources)
- **Overage = function disabled** until next billing cycle; no auto-upgrade

### Sources
- https://firebase.google.com/pricing — Spark plan limits confirmed 2026-05-16
- https://firebase.google.com/docs/projects/billing/firebase-pricing-plans — 2026-05-16
- https://firebase.google.com/docs/functions/get-started — 2026-05-16
- https://www.npmjs.com/package/firebase-tools — 2026-05-16

---

## 2. IBM Code Engine Lite — No CC

**Signup URL:** https://cloud.ibm.com/registration

### Signup Steps (≤3 min)
1. Go to https://cloud.ibm.com/registration
2. Enter email + password (or SSO)
3. Verify email (check inbox)
4. Choose **Lite** account when prompted
5. Accept terms → account active immediately

### CLI Install
```bash
# Linux / macOS one-liner
curl -fsSL https://clis.cloud.ibm.com/install/linux | bash && ibmcloud plugin install code-engine

# macOS Homebrew alternative
brew install ibm-cloud-cli && ibmcloud plugin install code-engine

ibmcloud ce --version  # verify
```

### First Deploy
```bash
# Login (opens browser)
ibmcloud login -r us-south

# Create project
ibmcloud ce project create --name myproject
ibmcloud ce project select --name myproject

# Deploy hello-world
ibmcloud ce app create --name hello-app --image icr.io/codeengine/hello --port 8080

# Get URL
ibmcloud ce app get --name hello-app --output url
```

### Verify
```bash
curl https://<app-url>
# Expected: "Hello from IBM Cloud Code Engine!"
```

### Free Tier Limits
- 100,000 vCPU-seconds/month (~1 day continuous 1-vCPU)
- 200,000 GB-seconds/month (~23 hr continuous 256 MB app)
- Unlimited requests (pay only for compute)
- 500 build-minutes/month

### Gotchas
- **Scale-to-zero = cold starts** on first request after idle (~few seconds)
- **Deploy in us-south** for best free-tier availability; other regions may vary
- **500 build-min/month** is tight if building containers often — use pre-built images to conserve
- Upgrading to Pay-as-you-go (CC required) unlocks $200 promo credit

### Sources
- https://www.ibm.com/products/cloud/free — 2026-05-16
- https://cloud.ibm.com/docs/codeengine?topic=codeengine-pricing — 2026-05-16
- https://cloud.ibm.com/docs/codeengine?topic=codeengine-install-cli — 2026-05-16
- https://cloud.ibm.com/docs/codeengine?topic=codeengine-getting-started — 2026-05-16
- https://cloud.ibm.com/docs/codeengine?topic=codeengine-regions — 2026-05-16

---

## 3. Northflank (Developer Sandbox) — CC Required, No Charge

**Signup URL:** https://app.northflank.com/signup

### Signup Steps (≤5 min)
1. Visit https://app.northflank.com/signup
2. Sign up with email + password (or GitHub/GitLab OAuth)
3. Confirm email
4. Create team name (e.g., `my-team`)
5. Choose **Developer Sandbox** (free tier)
6. Enter credit card (required; no charge unless you upgrade)
7. Accept terms → account active

### CLI Install
```bash
npm install -g @northflank/cli
northflank --version  # verify
```

### First Deploy

**Option A: From GitHub (recommended)**
```bash
northflank login
# Follow prompts to create API token in web UI, paste into CLI

northflank create deploymentService
# Prompts: project name, service name, GitHub repo, build/start commands, port
# Auto-deploys on git push thereafter:
git push origin main
```

**Option B: Quick Docker test**
```bash
northflank create deploymentService \
  --name "hello" \
  --image "nginx:latest" \
  --port 80
```

### Verify
```bash
northflank get deploymentService --name my-api
# Check app.northflank.com → service → Domain for public URL
# Status should show "Running"
```

### Free Tier Limits
- 2 always-on container services
- 2 cron jobs
- 1 database
- Unlimited build minutes (shared builders)
- Northflank subdomain only (no custom domains)

### Gotchas
- **CC required at signup** — biggest friction point for no-card users
- **Services do NOT sleep** — free tier is always-on (good); monitor quota
- **2 service limit** — delete one to add another
- **Long builds may queue** on shared builders
- **Custom domains** require paid plan; free tier = `*.app.northflank.com`

### Sources
- https://northflank.com/pricing — 2026-05-16
- https://northflank.com/docs/v1/application/billing/pricing-on-northflank — 2026-05-16
- https://www.npmjs.com/package/@northflank/cli — 2026-05-16
- https://northflank.com/docs/v1/application/getting-started/build-and-deploy-your-code — 2026-05-16

---

## 4. Oracle Ampere A1 (Always Free) — CC Required

**Signup URL:** https://signup.oraclecloud.com/

### Signup Steps (~10–25 min total)
1. Open https://signup.oraclecloud.com/ — choose **Cloud Account** (not Java Cloud, not ATP trial)
2. Enter email → click **Next**
3. Fill personal info (name, company, country, phone) → Oracle sends SMS code
4. Enter SMS code → click **Verify**
5. Enter credit card (identity verification; $1 hold released in 3–5 days; no actual charge)
6. **Set tenant name** (becomes `https://<tenant>.oraclecloud.com` — cannot change later)
7. **Choose home region** — CRITICAL: pick a low-contention region for A1 availability:
   - `eu-frankfurt-1` (recommended)
   - `ap-singapore-1`
   - `ap-sydney-1`
   - Avoid `us-ashburn-1` / `us-phoenix-1` (high contention)
8. Accept terms → **Create my account** → wait for activation email (~5–15 min)

### CLI Install
```bash
# macOS
brew install oci-cli

# Python (cross-platform)
pip install oci-cli

oci --version  # verify
```

Configure credentials:
```bash
oci setup config
# Needs: Tenancy OCID, User OCID (both in OCI Console > Admin)
# Generates API signing key (add public key to OCI Console > My Profile > API Keys)
```

### First Deploy: Provision A1 Instance

**Via Console (simplest):**
1. Log in → **Compute > Instances** → **Create Instance**
2. Name: `arm-vm-1`
3. Image: `Canonical Ubuntu 22.04` or `Oracle Linux 8.x`
4. Shape: `VM.Standard.A1.Flex` (Always Free shape)
5. OCPU: 2–4 (total across all instances ≤ 4)
6. Memory: proportional (1 OCPU ≈ 6 GB)
7. VCN: use default; enable **Public IP**; add your SSH public key
8. Click **Create** → boots in 1–2 min

**Via OCI CLI:**
```bash
TENANCY_OCID="ocid1.tenancy.oc1..aaaaa..."      # Console > Admin > Tenancy
COMPARTMENT_OCID="ocid1.compartment.oc1..aaaa..." # root = same as tenancy
SUBNET_OCID="ocid1.subnet.oc1.region..aaaa..."   # Console > Networking > VCN
IMAGE_OCID="ocid1.image.oc1.region..aaaa..."     # Ubuntu 22.04 for your region
SSH_PUBLIC_KEY="ssh-rsa AAAA... your@email.com"

oci compute instance launch \
  --compartment-id "$COMPARTMENT_OCID" \
  --availability-domain "AD-1" \
  --shape "VM.Standard.A1.Flex" \
  --shape-config '{"ocpus": 2, "memory_in_gbs": 12}' \
  --image-id "$IMAGE_OCID" \
  --subnet-id "$SUBNET_OCID" \
  --ssh-authorized-keys-file /path/to/id_rsa.pub \
  --display-name "arm-vm-1" \
  --assign-public-ip true
```

### Verify
```bash
export PUBLIC_IP="1.2.3.4"   # Console > Compute > Instances > instance > Primary VNIC

ssh ubuntu@$PUBLIC_IP          # Ubuntu
# or
ssh opc@$PUBLIC_IP             # Oracle Linux

uname -m   # → aarch64 (ARM confirmed)
free -h    # check RAM
nproc      # check CPU count
```

### Free Tier Limits
- 4 OCPUs + 24 GB RAM (split across up to 4 VMs)
- 200 GB block storage
- 10 TB/month egress
- 2x AMD E2.1.Micro instances (bonus)
- Perpetual — no expiration

### Gotchas
- **"Out of Host Capacity"** is common in popular regions — retry or switch region; use [hitrov/oci-arm-host-capacity](https://github.com/hitrov/oci-arm-host-capacity) for automated retries
- **Home region is permanent** — if you pick a bad region (always full), you may need a new account
- **Idle reclaim:** instances idle >7 consecutive days may be terminated by Oracle — keep warm with a cron job or occasional SSH
- **$1 authorization hold** on CC — not a charge; released in 3–5 days
- **Virtual/prepaid/PIN-debit cards not accepted** — must be a real credit/debit card functioning as credit
- **Never upgrade then downgrade** — Oracle may suspend the account; stick to Always Free shapes

### Sources
- https://www.oracle.com/cloud/free/ — Always Free overview, 2026-05-16
- https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm — exact limits, 2026-05-16
- https://www.oracle.com/cloud/free/faq/ — CC requirements, 2026-05-16
- https://docs.oracle.com/en-us/iaas/Content/GSG/Tasks/signingup_topic-Sign_Up_for_Free_Oracle_Cloud_Promotion.htm — signup steps, 2026-05-16
- https://fullmetalbrackets.com/blog/oci-free-tier-breakdown — practical breakdown, 2026-05-16
- https://medium.com/@me69oshan/get-always-free-vm-instance-in-oracle-cloud-and-solve-out-of-host-capacity-issue-the-easy-way-88babae4eae5 — capacity workaround, 2026-05-16
- https://github.com/hitrov/oci-arm-host-capacity — automated retry tool, 2026-05-16

---

## Combined Gotchas Across All 4

- [ ] **CC hold vs. charge:** Oracle places a $1 auth hold (released 3–5 days). Northflank requires CC but charges nothing on free tier. Always verify your statement shows $0 net.
- [ ] **Pick less-popular Oracle region:** `eu-frankfurt-1`, `ap-singapore-1`, or `ap-sydney-1` have far better A1 availability than US regions.
- [ ] **Firebase Spark blocks non-Google outbound:** If your function calls a 3rd-party API (Stripe, Twilio, etc.) you need Blaze (pay-as-you-go). Plan before you build.
- [ ] **IBM scale-to-zero cold starts:** First request after idle takes a few extra seconds. Not suitable for latency-sensitive always-on workloads without a keep-warm pinger.
- [ ] **Oracle idle reclaim:** Instances idle >7 days may be reclaimed. Add a cron job or periodic SSH to keep alive.
- [ ] **Northflank 2-service limit:** Free tier hard-caps at 2 running services. Plan your deployments; delete before adding.
- [ ] **IBM 500 build-min/month:** Use pre-built images (`icr.io/*`) to avoid burning build quota. Source-to-image builds eat minutes fast.
- [ ] **Oracle tenant name is permanent:** Choose something generic (`myname` or org name) — it becomes part of your login URL forever.
- [ ] **Firebase 2nd gen requires Blaze:** If you need longer timeouts (>60 sec), more memory, or 2nd gen features, Spark won't cut it.
- [ ] **Virtual/prepaid cards rejected by Oracle:** Must be a real credit or credit-functioning debit card.

---

## Sources

| URL | Note | Accessed |
|---|---|---|
| https://firebase.google.com/pricing | Spark plan free limits | 2026-05-16 |
| https://firebase.google.com/docs/projects/billing/firebase-pricing-plans | Billing plan details | 2026-05-16 |
| https://firebase.google.com/docs/functions/get-started | Functions quickstart | 2026-05-16 |
| https://www.npmjs.com/package/firebase-tools | Firebase CLI | 2026-05-16 |
| https://www.ibm.com/products/cloud/free | IBM Cloud free tier overview | 2026-05-16 |
| https://cloud.ibm.com/docs/codeengine?topic=codeengine-pricing | Code Engine pricing | 2026-05-16 |
| https://cloud.ibm.com/docs/codeengine?topic=codeengine-install-cli | CLI install | 2026-05-16 |
| https://cloud.ibm.com/docs/codeengine?topic=codeengine-getting-started | Getting started | 2026-05-16 |
| https://cloud.ibm.com/docs/codeengine?topic=codeengine-regions | Region availability | 2026-05-16 |
| https://www.srvrlss.io/provider/ibm-cloud-engine/ | IBM Code Engine 2026 review | 2026-05-16 |
| https://northflank.com/pricing | Northflank pricing | 2026-05-16 |
| https://northflank.com/docs/v1/application/billing/pricing-on-northflank | Northflank billing docs | 2026-05-16 |
| https://www.npmjs.com/package/@northflank/cli | Northflank CLI | 2026-05-16 |
| https://northflank.com/docs/v1/application/getting-started/build-and-deploy-your-code | Northflank deploy guide | 2026-05-16 |
| https://www.oracle.com/cloud/free/ | Oracle Always Free overview | 2026-05-16 |
| https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm | Always Free exact limits | 2026-05-16 |
| https://www.oracle.com/cloud/free/faq/ | CC & signup FAQ | 2026-05-16 |
| https://docs.oracle.com/en-us/iaas/Content/GSG/Tasks/signingup_topic-Sign_Up_for_Free_Oracle_Cloud_Promotion.htm | Signup guide | 2026-05-16 |
| https://fullmetalbrackets.com/blog/oci-free-tier-breakdown | Practical A1 breakdown | 2026-05-16 |
| https://medium.com/@me69oshan/get-always-free-vm-instance-in-oracle-cloud-and-solve-out-of-host-capacity-issue-the-easy-way-88babae4eae5 | Capacity workaround guide | 2026-05-16 |
| https://github.com/hitrov/oci-arm-host-capacity | Automated A1 capacity retry | 2026-05-16 |
