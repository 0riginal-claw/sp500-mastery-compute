#!/usr/bin/env bash
# autosolve_skip: phase-D autocreate hook stub
# missing-hook-autocreate — PreToolUse-context-aware stub generator.
#
# When stderr from a previous hook contains "No such file or directory"
# pointing at a *.sh hook path, auto-create a no-op stub so the missing
# hook stops blocking the session. Log to state/hook_errors/autocreated.jsonl.
#
# This hook itself is registered as PostToolUse (after a tool runs and the
# missing-hook stderr surfaces). It is also safe as a no-op.
set +e
LC_ALL=C

STATE_DIR_REAL="/Users/orginal/.claude/state/hook_errors"
STATE_DIR_SHIM="$HOME/.claude/state/hook_errors"
[[ -d "$STATE_DIR_REAL" ]] && STATE_DIR="$STATE_DIR_REAL" || STATE_DIR="$STATE_DIR_SHIM"
mkdir -p "$STATE_DIR" 2>/dev/null
OUT_LOG="$STATE_DIR/autocreated.jsonl"

PAYLOAD="$(cat 2>/dev/null)"
[[ -z "$PAYLOAD" ]] && exit 0

# Extract candidate missing paths from any field of the payload.
python3 - <<PYEOF
import json, os, sys, re, time
raw = """$PAYLOAD"""
try:
    d = json.loads(raw)
except Exception:
    sys.exit(0)
flat = json.dumps(d)
# Match "<path>.sh: No such file or directory" (allow leading colon, quotes, etc.)
pat = re.compile(r"(/[A-Za-z0-9._/~\\-]+?\.(?:sh|py))\\s*:\\s*No such file or directory")
created = []
for m in pat.finditer(flat):
    raw_path = m.group(1)
    path = os.path.expanduser(raw_path)
    # Guardrails: only autocreate under known hook dirs.
    allowed_prefixes = (
        "/Users/orginal/.zg/hooks/",
        "/Users/orginal/.claude/hooks/",
        "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/home/.claude/hooks/",
    )
    if not any(path.startswith(p) for p in allowed_prefixes):
        continue
    if os.path.exists(path):
        continue
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        stub = f"#!/usr/bin/env bash\n# autosolve_skip: auto-generated stub by missing-hook-autocreate ts={int(time.time())}\nexit 0\n"
        with open(path, "w") as fh:
            fh.write(stub)
        os.chmod(path, 0o755)
        created.append(path)
    except OSError:
        pass

if created:
    out = "$OUT_LOG"
    with open(out, "a") as fh:
        for p in created:
            fh.write(json.dumps({
                "ts": int(time.time()),
                "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "path": p,
                "stub_size": 90,
            }) + "\n")
    print(f"missing-hook-autocreate: created {len(created)} stub(s)", file=sys.stderr)
PYEOF
exit 0
