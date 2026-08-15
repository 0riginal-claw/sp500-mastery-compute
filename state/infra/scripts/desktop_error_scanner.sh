#!/usr/bin/env bash
# autosolve_skip: phase-C scanner
# desktop_error_scanner.sh — Phase C
#
# Every 60s scan ~/Desktop/*.txt for lines matching "error|Failed|BLOCKED".
# For each new finding, write state/hook_errors/desktop_<sanitised-name>.json
# (atomic). The SessionStart Phase E injector then picks it up alongside the
# main hook-error findings.
set -uo pipefail
LC_ALL=C

DESKTOP_DIR="${DESKTOP_DIR:-/Users/orginal/Desktop}"
STATE_DIR="${DESKTOP_ERR_STATE_DIR:-/Users/orginal/.claude/state/hook_errors}"
CYCLE_SEC="${DESKTOP_ERR_CYCLE_SEC:-60}"
mkdir -p "$STATE_DIR"
CURSOR_FILE="$STATE_DIR/_desktop_cursors.json"

scan_once() {
  python3 - <<'PYEOF'
import os, sys, json, re, glob, time
DESKTOP = os.environ.get("DESKTOP_DIR", "/Users/orginal/Desktop")
STATE_DIR = os.environ.get("DESKTOP_ERR_STATE_DIR", "/Users/orginal/.claude/state/hook_errors")
CUR = os.path.join(STATE_DIR, "_desktop_cursors.json")

try:
    cursors = json.load(open(CUR))
except Exception:
    cursors = {}

pat = re.compile(r"\b(error|errors|Failed|BLOCKED|Traceback|Exception)\b", re.IGNORECASE)
n_files = n_findings = 0
for fp in sorted(glob.glob(os.path.join(DESKTOP, "*.txt"))):
    n_files += 1
    try:
        size = os.path.getsize(fp)
    except OSError:
        continue
    cur = int(cursors.get(fp, 0))
    if size < cur: cur = 0
    if size == cur: continue
    try:
        with open(fp, "rb") as fh:
            fh.seek(cur)
            chunk = fh.read(size - cur).decode("utf-8", errors="replace")
    except OSError:
        continue
    matches = []
    for line in chunk.splitlines():
        if pat.search(line):
            matches.append(line.strip()[:400])
    cursors[fp] = size
    if matches:
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(fp))
        out = os.path.join(STATE_DIR, f"desktop_{safe}.json")
        # Atomic write
        tmp = out + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({
                "ts": int(time.time()),
                "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "path": fp,
                "matches": matches[-20:],
                "n_matches": len(matches),
            }, fh)
        os.replace(tmp, out)
        # Also append to the today.jsonl so Phase E injector sees them
        date = time.strftime("%Y-%m-%d", time.gmtime())
        today = os.path.join(STATE_DIR, f"{date}.jsonl")
        with open(today, "a") as fh:
            for m in matches[-10:]:
                fh.write(json.dumps({
                    "ts": int(time.time()),
                    "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "session_id": "desktop",
                    "project_dir": os.path.basename(fp),
                    "phase": "Desktop",
                    "tool": "?",
                    "snippet": m[:400],
                    "raw_line_offset": -1,
                }) + "\n")
        n_findings += len(matches)

# Persist cursors atomically
tmp = CUR + ".tmp"
with open(tmp, "w") as fh:
    json.dump(cursors, fh)
os.replace(tmp, CUR)

# Heartbeat
hb = os.path.join(STATE_DIR, "desktop_heartbeat.json")
tmp = hb + ".tmp"
with open(tmp, "w") as fh:
    json.dump({
        "ts": int(time.time()),
        "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pid": os.getpid(),
        "files_scanned": n_files,
        "new_findings": n_findings,
    }, fh)
os.replace(tmp, hb)
print(json.dumps({"files_scanned": n_files, "findings": n_findings}))
PYEOF
}

if [[ "${1:-}" == "--once" ]]; then
    scan_once
    exit 0
fi

# Daemon mode
while true; do
    scan_once >/dev/null 2>&1
    sleep "$CYCLE_SEC"
done
