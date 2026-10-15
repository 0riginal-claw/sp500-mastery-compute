# Alpaca Paper Credentials — macOS Keychain Setup

One-time setup that stores the Alpaca paper-trading API key + secret in the
macOS login Keychain with an ACL that pre-authorizes the `sp500-mastery`
venv `python3` to read them **without a GUI prompt**.

This is required so cron / `launchd` jobs that fire at Monday 09:30 ET (and
any other unattended trading-window job) don't stall on an interactive
Keychain pop-up.

## Run once

Export the values into the environment, then invoke the helper. Do **not**
type the values inline on the command line where they'll be persisted in
shell history — prefix the env vars or use a leading space, depending on
your shell's `HISTCONTROL`.

```bash
ALPACA_API_KEY="PKXXXXXXXXXXXXXXXXXX" \
ALPACA_SECRET_KEY="YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY" \
bash "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/scripts/setup_alpaca_keychain.sh"
```

Replace the placeholder values with the real paper-trading key + secret
from <https://app.alpaca.markets/paper/dashboard/overview>.

The script will:

1. Validate that both env vars are non-empty.
2. Warn (non-fatal) if the venv python or `/usr/bin/security` aren't found.
3. Call `security add-generic-password -U` twice (idempotent — re-running
   updates the stored value rather than erroring).
4. Pre-authorize `/Users/orginal/.venvs/sp500-mastery/bin/python3` and
   `/usr/bin/security` via `-T` ACL flags so silent reads are allowed.
5. Verify both entries exist via `security find-generic-password`.

## Verify

```bash
security find-generic-password -s alpaca-paper-api-key
security find-generic-password -s alpaca-paper-secret-key
```

Both commands should print metadata (service, account, class) without
prompting. To print the **actual stored value** add `-w`:

```bash
security find-generic-password -s alpaca-paper-api-key -w
```

The first time a non-pre-authorized process (e.g. an arbitrary shell)
calls the above with `-w`, the Keychain *may* prompt. Calls from the
pre-authorized venv python should never prompt.

## Read from Python

```python
import subprocess

def _kc(service: str) -> str:
    return subprocess.check_output(
        ["security", "find-generic-password", "-s", service, "-w"],
        text=True,
    ).strip()

api_key = _kc("alpaca-paper-api-key")
secret  = _kc("alpaca-paper-secret-key")
```

## Rotate / remove

If you ever need to delete or replace the stored credentials:

```bash
security delete-generic-password -s alpaca-paper-api-key
security delete-generic-password -s alpaca-paper-secret-key
```

Then re-run the setup script with the new values.

## Adding more authorized apps

To grant another binary silent read access (e.g. a second venv or a
compiled CLI), add another `-T <path>` flag to both `security
add-generic-password` calls in `scripts/setup_alpaca_keychain.sh` and
re-run. The `-U` flag makes the update idempotent.

## Notes

- The script reads credentials from the **environment**, never an
  interactive prompt — this avoids capturing the secret in transcripts.
- `security add-generic-password` accepts the value via `-w "<value>"`;
  macOS scrubs the `security` binary's argv from `ps` output, but the
  script also avoids ever echoing the value.
- Service names (`alpaca-paper-api-key`, `alpaca-paper-secret-key`) are
  the contract between this helper and the Python reader — change them
  in lockstep or you'll get `errSecItemNotFound`.
- The account name is `$USER` (your macOS login user).
