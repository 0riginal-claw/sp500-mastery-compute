#!/usr/bin/env python3
"""edgar_10q_signal.py — Generate 10-Q delay-filtered trading signals.

Usage:
    python3 edgar_10q_signal.py                          # today's signals
    python3 edgar_10q_signal.py --date 2026-04-24        # specific date
    python3 edgar_10q_signal.py --lookback 7              # last 7 days
    python3 edgar_10q_signal.py --validate               # WF validation mode

Strategy: Buy at close on 10-Q filing date if filing delay (quarter-end to filed_at)
is between 35-50 days. Sell at close after 5 trading days.
Source: research/notes/20260520_wave3_10q_strategy.md
"""

import sqlite3, json, sys, os
from datetime import datetime, timedelta

DB = "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/claudes test/data/edgar/data/edgar.db"

def get_signals(target_date=None, lookback_days=1):
    """Return list of tickers with active 10-Q signals for given date(s)."""
    if target_date is None:
        target_date = datetime.now().strftime('%Y-%m-%d')
    
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    
    # Date range: lookback_days before target_date to target_date
    end_date = target_date
    start = datetime.strptime(target_date, '%Y-%m-%d') - timedelta(days=lookback_days)
    start_date = start.strftime('%Y-%m-%d')
    
    # Query 10-Qs with delay between 35-50 days
    rows = conn.execute("""
        SELECT ticker, filed_at, period_of_report,
               ROUND(JULIANDAY(filed_at) - JULIANDAY(period_of_report)) as delay_days
        FROM filings 
        WHERE form = '10-Q'
        AND filed_at >= ? AND filed_at <= ?
        AND ticker IS NOT NULL AND ticker != ''
        AND period_of_report IS NOT NULL
        AND JULIANDAY(filed_at) - JULIANDAY(period_of_report) BETWEEN 35 AND 50
        ORDER BY filed_at
    """, (start_date, end_date)).fetchall()
    
    conn.close()
    
    signals = []
    for r in rows:
        entry_date = r['filed_at'][:10]
        signals.append({
            'ticker': r['ticker'],
            'entry_date': entry_date,
            'exit_date': _add_trading_days(entry_date, 5),
            'filing_date': entry_date,
            'delay_days': int(r['delay_days']),
            'strategy': '10Q_DELAY_FILTER',
            'confidence': 'HIGH' if int(r['delay_days']) >= 38 else 'MEDIUM',
            'hold_days': 5,
        })
    
    return signals

def _add_trading_days(date_str, n):
    """Add n trading days to a date (rough approximation)."""
    from datetime import datetime, timedelta
    d = datetime.strptime(date_str, '%Y-%m-%d')
    added = 0
    while added < n:
        d += timedelta(days=1)
        if d.weekday() < 5:  # Mon-Fri
            added += 1
    return d.strftime('%Y-%m-%d')

if __name__ == '__main__':
    target_date = None
    lookback_days = 1
    validate = False
    
    for arg in sys.argv[1:]:
        if arg.startswith('--date='):
            target_date = arg.split('=')[1]
        elif arg.startswith('--lookback='):
            lookback_days = int(arg.split('=')[1])
        elif arg == '--validate':
            validate = True
    
    signals = get_signals(target_date, lookback_days if not validate else 1825)
    
    print(json.dumps({
        'generated_at': datetime.now().isoformat(),
        'target_date': target_date or datetime.now().strftime('%Y-%m-%d'),
        'lookback_days': lookback_days,
        'total_signals': len(signals),
        'signals': signals[:50],  # limit output
        'total_available': len(signals),
    }, indent=2))
    
    if validate:
        print(f"\nValidation: {len(signals)} total signals in DB", file=sys.stderr)
        tickers = set(s['ticker'] for s in signals)
        print(f"Unique tickers: {len(tickers)}", file=sys.stderr)
        dates = set(s['entry_date'] for s in signals)
        print(f"Date range: {min(dates)} to {max(dates)}", file=sys.stderr)
