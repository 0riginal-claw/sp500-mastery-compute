"""Build sweeps/queue.txt for backtest_xgb_v10.py across all SP500 tickers."""
import os, re, json, shutil
from pathlib import Path

SP_ROOT = Path("/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/AI-Tools/s&p500-ticker-mastery")

# 1. Universe from dispatched.jsonl
tickers_all = set()
with open(SP_ROOT / "sweeps" / "dispatched.jsonl") as f:
    for line in f:
        try:
            j = json.loads(line)
            if "ticker" in j:
                tickers_all.add(j["ticker"])
        except Exception:
            pass
all_tickers = sorted(tickers_all)
print(f"Universe: {len(all_tickers)} tickers from dispatched.jsonl")

# 2. Save sp500_tickers.txt
sp500_path = SP_ROOT / "sp500_tickers.txt"
if sp500_path.exists():
    shutil.copy(sp500_path, str(sp500_path) + ".bak")
    print("Backed up existing sp500_tickers.txt → .bak")
sp500_path.write_text("\n".join(all_tickers) + "\n", encoding="utf-8")
print(f"Wrote {len(all_tickers)} tickers → {sp500_path}")

# 3. v4-mastered set
mask = re.compile(r"_(?:ML|XGB_v\d+\w*|D1REV)_mastered\.md$")
mastered_v4 = set()
for fn in os.listdir(SP_ROOT / "mastery_files"):
    if fn.endswith("_mastered.md"):
        mastered_v4.add(mask.sub("", fn))
print(f"v4 mastered: {len(mastered_v4)} tickers")

# 4. Priority sets
P1_seeds = {"TPL", "JPM", "BXP", "NVDA"}
P2_seeds = {"AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "BRK.B", "JPM", "V"}

def assign_priority(ticker):
    if ticker in P1_seeds:
        return 1
    if ticker in P2_seeds:
        return 2
    if ticker in mastered_v4:
        return 3
    return 4

# 5. Build ordered queue
queue_entries = []
priorities_map = {}
for t in all_tickers:
    p = assign_priority(t)
    queue_entries.append((p, t))
    priorities_map[t] = p
queue_entries.sort(key=lambda x: (x[0], x[1]))

# 6. Write queue.txt
queue_path = SP_ROOT / "sweeps" / "queue.txt"
lines = []
current_p = None
for p, ticker in queue_entries:
    if p != current_p:
        lines.append(f"# -- Priority {p} --")
        current_p = p
    lines.append(f"scripts/backtest_xgb_v10.py {ticker} ORB")
queue_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
total_jobs = sum(1 for l in lines if not l.startswith("#"))
print(f"Wrote {total_jobs} jobs → {queue_path}")

# 7. Write state/queue_priorities.json
state_dir = SP_ROOT / "state"
state_dir.mkdir(exist_ok=True)
prio_counts = {1: 0, 2: 0, 3: 0, 4: 0}
for t, p in priorities_map.items():
    prio_counts[p] += 1

priority_output = {
    "generated": "2026-05-16",
    "total_tickers": len(all_tickers),
    "total_jobs": total_jobs,
    "script": "scripts/backtest_xgb_v10.py",
    "strategy": "ORB",
    "source": "dispatched.jsonl (509 unique tickers)",
    "priority_counts": {f"P{k}": v for k, v in prio_counts.items()},
    "priority_rules": {
        "P1": "TPL, JPM, BXP, NVDA — session-preservation top performers",
        "P2": "High-mcap SPY-weight: AAPL MSFT GOOGL AMZN NVDA META TSLA BRK.B JPM V",
        "P3": f"v4-mastered ({len(mastered_v4)} tickers re-validate on v10)",
        "P4": "Never-mastered (fill gaps)"
    },
    "per_ticker": priorities_map
}
(state_dir / "queue_priorities.json").write_text(json.dumps(priority_output, indent=2), encoding="utf-8")
print(f"Wrote priority metadata → {state_dir / 'queue_priorities.json'}")

# 8. Summary
for p in (1, 2, 3, 4):
    print(f"  P{p}: {prio_counts[p]} tickers")

# 9. Verification sample
non_comment = [l for l in queue_path.read_text().splitlines() if l and not l.startswith("#")]
print("\nFirst 5 jobs:")
for l in non_comment[:5]:
    print(" ", l)
print("Last 5 jobs:")
for l in non_comment[-5:]:
    print(" ", l)

# 10. Parseable check: simulate dispatcher parse
errors = 0
for l in non_comment:
    parts = l.strip().split()
    if len(parts) not in (3, 4):
        print(f"BAD LINE: {l!r}")
        errors += 1
print(f"\nParse check: {len(non_comment)} lines, {errors} errors")
