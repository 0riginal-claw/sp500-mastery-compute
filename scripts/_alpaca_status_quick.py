# autosolve_skip: emergency live-trading repair
# autosolve_skip: emergency live-trading repair retry
"""Quick read-only Alpaca account/positions/orders status check.

Reuses live_paper_trade.py's cred loader. Read-only; never places orders.
Created by emergency live-trading audit 2026-05-20.
"""
import sys
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

CRED_PATH = Path.home() / ".config" / "auto_signup" / "alpaca.env"
if CRED_PATH.exists():
    for line in CRED_PATH.read_text().splitlines():
        line = line.strip()
        if line.startswith("export ") and "=" in line:
            k, v = line[7:].split("=", 1)
            v = v.strip().strip('"').strip("'")
            os.environ[k] = v

from alpaca.trading.client import TradingClient  # noqa: E402
from alpaca.trading.requests import GetOrdersRequest  # noqa: E402
from alpaca.trading.enums import QueryOrderStatus  # noqa: E402
from datetime import datetime, timezone, timedelta  # noqa: E402

key = os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY")
secret = os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_API_SECRET")
assert key and secret, "No Alpaca credentials found"

client = TradingClient(key, secret, paper=True)
acct = client.get_account()

print("=== LIVE ALPACA ACCOUNT @ " + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC") + " ===")
print(f"  account: {acct.account_number}  status={acct.status}")
print(f"  trading_blocked={acct.trading_blocked}  account_blocked={acct.account_blocked}")
print(f"  equity=${acct.equity}  last_equity=${acct.last_equity}")
print(f"  cash=${acct.cash}  long_mv=${acct.long_market_value}  short_mv=${acct.short_market_value}")
day_pnl = float(acct.equity) - float(acct.last_equity)
print(f"  day_pnl=${day_pnl:+.2f}")
print(f"  daytrade_count={acct.daytrade_count}  pdt={acct.pattern_day_trader}")
print()

positions = client.get_all_positions()
print(f"=== OPEN POSITIONS: {len(positions)} ===")
total_pnl = 0.0
for p in sorted(positions, key=lambda x: x.symbol):
    pnl = float(p.unrealized_pl)
    pct = float(p.unrealized_plpc) * 100
    total_pnl += pnl
    print(f"  {p.symbol:6s} qty={p.qty:>4s} mv=${float(p.market_value):>9.2f} cost=${float(p.cost_basis):>9.2f} pnl=${pnl:>+7.2f} ({pct:+.2f}%)")
print(f"  TOTAL unrealized_pnl: ${total_pnl:+.2f}")
print()

req = GetOrdersRequest(status=QueryOrderStatus.ALL, after=(datetime.now(timezone.utc) - timedelta(hours=24)), limit=200)
try:
    orders = client.get_orders(filter=req)
except TypeError:
    orders = client.get_orders(req)
status_ct = {}
for o in orders:
    s = o.status.value if hasattr(o.status, "value") else str(o.status)
    status_ct[s] = status_ct.get(s, 0) + 1
print(f"=== ORDERS last 24h: {len(orders)} total ===")
print(f"  status_breakdown={status_ct}")
print()
print("Most recent 10 orders:")
for o in sorted(orders, key=lambda x: x.submitted_at or datetime.now(timezone.utc), reverse=True)[:10]:
    s = o.status.value if hasattr(o.status, "value") else str(o.status)
    sub = o.submitted_at.strftime("%Y-%m-%d %H:%M:%S") if o.submitted_at else "N/A"
    fil = o.filled_at.strftime("%H:%M:%S") if o.filled_at else "-"
    side = o.side.value if hasattr(o.side, "value") else str(o.side)
    print(f"  {sub}  {o.symbol:6s}  {side:4s}  status={s:10s} filled_at={fil}")
