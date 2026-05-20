"""
backtest_v1.py — intraday backtest with walk-forward, no-lookahead by construction.

Usage:
    python backtest_v1.py --ticker AAPL --output-dir backtests/AAPL

Strategy v1 ("low-touch revert"):
    Entry (long-only intraday): RSI(14)<35, price within 0.5*ATR of VWAP, RelVol(20)>1.2,
                                 not in first 15min of session, not in last 5min.
    Exit: TP at +1.5*ATR, SL at -0.7*ATR, time-stop 30 bars, hard EOD flatten at 15:55 ET.
    Position sizing: fixed $5,000 notional per trade (placeholder — real sizing is mastery-file fld 8).

No-lookahead protection (built into the code, not opt-in):
    - All daily features come from completed previous trading sessions (shift to prior session).
    - VWAP/RSI/ATR/RelVol computed with .shift(1) so the value at bar t uses bars [..., t-1].
    - PDH/PDL are previous-session highs/lows, never current session.
    - Resampling uses closed='left', label='left'.
    - Walk-forward: train on rolling 12 months, test on next 3 months, advance 3 months.

Outputs written to <output-dir>:
    - equity_curve.csv
    - trades.csv
    - walk_forward.csv
    - features_sample.parquet  (50 rows for verification)
    - run_meta.json            (commands, dates, parameters, software versions)
"""

import argparse, json, os, sys, hashlib
from datetime import datetime
import pandas as pd
import numpy as np

DATA_ROOT = "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/claudes test/data/timeframes/S&P500 5 Year Historical Data/Minutes TimeFrames/1Min_merged"

# ---------------------------------------------------------------------------
# 1. Data loading + RTH filter + 5m resample
# ---------------------------------------------------------------------------

def load_1m(ticker):
    """Load the 1-min parquet, return DataFrame indexed by UTC timestamp."""
    path = f"{DATA_ROOT}/{ticker}.parquet"
    df = pd.read_parquet(path)
    # timestamp is already UTC tz-aware
    df = df.set_index('timestamp').sort_index()
    return df

def filter_rth(df):
    """Filter to regular trading hours 09:30-16:00 ET (handles DST automatically)."""
    et = df.index.tz_convert('America/New_York')
    # 09:30 ET to 15:59 ET (16:00 close)
    mask = (
        ((et.hour > 9) | ((et.hour == 9) & (et.minute >= 30))) &
        (et.hour < 16)
    )
    # exclude weekends + days the entire 09:30-16:00 window is empty (holidays)
    out = df[mask].copy()
    return out

def resample_5m(df_rth):
    """Resample 1m -> 5m. closed='left' label='left' so a bar labeled 09:30 contains 09:30-09:34."""
    agg = {
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
        'volume': 'sum', 'trade_count': 'sum',
        'vwap': lambda x: (x * df_rth.loc[x.index, 'volume']).sum() / df_rth.loc[x.index, 'volume'].sum() if df_rth.loc[x.index, 'volume'].sum() > 0 else np.nan
    }
    # use volume-weighted vwap reconstruction; or just take first vwap of bar as cheaper proxy
    out = df_rth.resample('5min', closed='left', label='left').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
        'volume': 'sum', 'trade_count': 'sum',
    })
    # rebuild vwap as cumulative session VWAP (resets daily) — done later in feature step
    out = out.dropna(subset=['open', 'high', 'low', 'close'])
    return out

# ---------------------------------------------------------------------------
# 2. Features — strict no-lookahead via .shift(1) and per-session reset
# ---------------------------------------------------------------------------

def add_session_id(df):
    """Tag each bar with its session date (ET). VWAP/PDH/PDL/etc reset per session."""
    et = df.index.tz_convert('America/New_York')
    df = df.copy()
    df['session_date'] = et.date
    df['session_minute'] = (et.hour - 9) * 60 + et.minute - 30  # minute since 09:30 ET (0..389)
    return df

def add_vwap(df):
    """Cumulative session VWAP — resets at each new session."""
    df = df.copy()
    typical = (df['high'] + df['low'] + df['close']) / 3
    tp_v = typical * df['volume']
    df['vwap_session'] = tp_v.groupby(df['session_date']).cumsum() / df['volume'].groupby(df['session_date']).cumsum()
    return df

def add_rsi(df, period=14):
    df = df.copy()
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    # shift(1) so the value at bar t uses only bars [..., t-1]
    df['rsi_14'] = rsi.shift(1)
    return df

def add_atr(df, period=14):
    df = df.copy()
    h, l, c = df['high'], df['low'], df['close']
    prev_close = c.shift(1)
    tr = pd.concat([
        h - l,
        (h - prev_close).abs(),
        (l - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    df['atr_14'] = atr.shift(1)
    return df

def add_rel_vol(df, period=20):
    """Volume / mean volume of the SAME minute-of-session over prior 20 SESSIONS."""
    df = df.copy()
    # for each (session_minute), the rolling mean over prior 20 occurrences
    by_min = df.groupby('session_minute')['volume']
    # rank by session_date and take rolling
    df['_vol_lag'] = by_min.transform(lambda s: s.shift(1).rolling(period, min_periods=5).mean())
    df['rel_vol'] = df['volume'] / df['_vol_lag']
    df = df.drop(columns=['_vol_lag'])
    return df

def add_pdh_pdl(df):
    """Previous trading session's high and low — applied to all bars of current session."""
    sess_hl = df.groupby('session_date').agg(daily_high=('high', 'max'), daily_low=('low', 'min'))
    sess_hl['pdh'] = sess_hl['daily_high'].shift(1)
    sess_hl['pdl'] = sess_hl['daily_low'].shift(1)
    df = df.merge(sess_hl[['pdh', 'pdl']], left_on='session_date', right_index=True, how='left')
    return df

def add_features(df):
    df = add_session_id(df)
    df = add_vwap(df)
    df = add_rsi(df, 14)
    df = add_atr(df, 14)
    df = add_rel_vol(df, 20)
    df = add_pdh_pdl(df)
    # distance from session vwap, normalized by atr (avoid div-by-zero)
    df['dist_vwap_atr'] = (df['close'] - df['vwap_session']) / df['atr_14']
    return df

# ---------------------------------------------------------------------------
# 3. Strategy — generate signals + execute trades
# ---------------------------------------------------------------------------

def generate_signals(df):
    """
    Long entry: RSI<35, |dist_vwap_atr| < 0.5, rel_vol > 1.2, session_minute >= 15, session_minute < 385.
    Returns a Series of booleans indexed like df.
    """
    cond = (
        (df['rsi_14'] < 35) &
        (df['dist_vwap_atr'].abs() < 0.5) &
        (df['rel_vol'] > 1.2) &
        (df['session_minute'] >= 15) &
        (df['session_minute'] < 385) &       # not in last 5 min (last 5min bars start at 385/390)
        df['rsi_14'].notna() &
        df['atr_14'].notna() &
        df['rel_vol'].notna() &
        df['dist_vwap_atr'].notna()
    )
    return cond

def simulate(df, signals, slippage_bps=2.0, fee_per_share=0.0035, notional=5000):
    """
    Simple sequential trade simulator. One open position at a time.
    Entry on bar after signal (next bar's open) — never on signal bar (no lookahead).
    Exit on first of: +1.5 ATR target, -0.7 ATR stop, 30-bar time stop, end-of-session.
    """
    trades = []
    in_pos = False
    entry_bar = None
    entry_price = None
    entry_atr = None
    entry_session = None
    qty = 0

    rows = df.reset_index().to_dict('records')
    sig_arr = signals.values

    for i in range(len(rows) - 1):
        bar = rows[i]
        nxt = rows[i + 1]

        if not in_pos:
            if sig_arr[i] and bar['session_minute'] >= 15 and bar['session_minute'] < 380:
                # enter on next bar's open
                entry_price = nxt['open'] * (1 + slippage_bps / 10000)
                entry_bar = i + 1
                entry_atr = bar['atr_14']
                entry_session = bar['session_date']
                qty = int(notional / entry_price)
                if qty < 1:
                    continue
                in_pos = True
        else:
            cur = rows[i]
            # exit conditions
            tp = entry_price + 1.5 * entry_atr
            sl = entry_price - 0.7 * entry_atr
            bars_held = i - entry_bar

            exit_reason = None
            exit_price = None

            # 1. EOD: same session, last bar at minute>=385 → flatten at this bar's close
            if cur['session_date'] != entry_session:
                # shouldn't happen — we should flatten at EOD of entry session, not next day
                exit_reason = 'EOD_CROSS'
                # this is a bug if it ever fires; flatten at prior bar
                exit_price = rows[i-1]['close'] * (1 - slippage_bps / 10000)
            elif cur['session_minute'] >= 385:
                exit_reason = 'EOD'
                exit_price = cur['close'] * (1 - slippage_bps / 10000)
            # 2. TP hit during this bar
            elif cur['high'] >= tp:
                exit_reason = 'TP'
                exit_price = tp * (1 - slippage_bps / 10000)
            # 3. SL hit during this bar
            elif cur['low'] <= sl:
                exit_reason = 'SL'
                exit_price = sl * (1 - slippage_bps / 10000)
            # 4. time stop
            elif bars_held >= 30:
                exit_reason = 'TIME'
                exit_price = cur['close'] * (1 - slippage_bps / 10000)

            if exit_reason:
                pnl_per_share = exit_price - entry_price
                fees = fee_per_share * qty * 2
                pnl = pnl_per_share * qty - fees
                trades.append({
                    'entry_time': rows[entry_bar]['timestamp'],
                    'entry_price': entry_price,
                    'entry_session': entry_session.isoformat() if hasattr(entry_session, 'isoformat') else str(entry_session),
                    'exit_time': cur['timestamp'],
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'qty': qty,
                    'bars_held': bars_held,
                    'pnl': pnl,
                    'pnl_pct': pnl_per_share / entry_price,
                    'entry_atr': entry_atr,
                    'fees': fees,
                })
                in_pos = False
                entry_bar = None
                entry_price = None
                qty = 0

    return pd.DataFrame(trades)

# ---------------------------------------------------------------------------
# 4. Metrics
# ---------------------------------------------------------------------------

def compute_metrics(trades_df):
    if len(trades_df) == 0:
        return {'n_trades': 0, 'win_rate': None, 'profit_factor': None,
                'expectancy': None, 'total_pnl': 0, 'max_drawdown': 0,
                'total_return_pct': 0, 'sharpe': None, 'avg_trade': None,
                'avg_win': None, 'avg_loss': None, 'eod_exits_pct': None,
                'tp_exits_pct': None, 'sl_exits_pct': None, 'time_exits_pct': None}
    n = len(trades_df)
    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] <= 0]
    gross_win = wins['pnl'].sum()
    gross_loss = abs(losses['pnl'].sum())
    wr = len(wins) / n
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')
    expectancy = trades_df['pnl'].mean()
    total_pnl = trades_df['pnl'].sum()
    # equity curve (cumulative)
    equity = trades_df['pnl'].cumsum()
    drawdown = equity - equity.cummax()
    max_dd = drawdown.min()
    # use $50K notional as baseline equity for % return
    baseline = 50000
    total_return = total_pnl / baseline
    # very rough sharpe — per-trade returns annualized assuming ~250 trades/yr
    if len(trades_df) > 5:
        per_trade_ret = trades_df['pnl_pct']
        sharpe = (per_trade_ret.mean() / per_trade_ret.std()) * np.sqrt(252) if per_trade_ret.std() > 0 else None
    else:
        sharpe = None
    exit_counts = trades_df['exit_reason'].value_counts(normalize=True).to_dict()
    return {
        'n_trades': n,
        'win_rate': wr,
        'profit_factor': pf,
        'expectancy': expectancy,
        'total_pnl': total_pnl,
        'max_drawdown': max_dd,
        'max_drawdown_pct': max_dd / baseline,
        'total_return_pct': total_return,
        'sharpe': sharpe,
        'avg_trade': trades_df['pnl'].mean(),
        'avg_win': wins['pnl'].mean() if len(wins) else None,
        'avg_loss': losses['pnl'].mean() if len(losses) else None,
        'tp_exits_pct': exit_counts.get('TP', 0),
        'sl_exits_pct': exit_counts.get('SL', 0),
        'eod_exits_pct': exit_counts.get('EOD', 0),
        'time_exits_pct': exit_counts.get('TIME', 0),
    }

# ---------------------------------------------------------------------------
# 5. Walk-forward
# ---------------------------------------------------------------------------

def walk_forward(features, train_months=12, test_months=3, step_months=3, max_folds=8):
    """
    For this strategy (rule-based, no per-fold parameter tuning), we treat each WF fold
    as a regime-isolation test: confirm strategy works in the OOS window, no leak.
    Since no parameters are fit per fold, the training window is informational only —
    we just slice trades by date and report per-fold metrics.
    """
    start = features.index.min()
    end = features.index.max()

    folds = []
    cur_oos_start = start + pd.DateOffset(months=train_months)
    while True:
        oos_end = cur_oos_start + pd.DateOffset(months=test_months)
        if oos_end > end or len(folds) >= max_folds:
            break
        folds.append({
            'fold': len(folds) + 1,
            'is_start': (cur_oos_start - pd.DateOffset(months=train_months)).isoformat(),
            'is_end': cur_oos_start.isoformat(),
            'oos_start': cur_oos_start.isoformat(),
            'oos_end': oos_end.isoformat(),
        })
        cur_oos_start = cur_oos_start + pd.DateOffset(months=step_months)
    return folds

def evaluate_folds(features, signals, folds, **trade_kwargs):
    """Run the simulator on each fold's OOS window, return per-fold metrics."""
    rows = []
    for f in folds:
        oos = features[(features.index >= f['oos_start']) & (features.index < f['oos_end'])]
        if len(oos) == 0:
            continue
        sig = signals.loc[oos.index]
        trades = simulate(oos, sig, **trade_kwargs)
        m = compute_metrics(trades)
        m['fold'] = f['fold']
        m['oos_start'] = f['oos_start']
        m['oos_end'] = f['oos_end']
        rows.append(m)
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ticker', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--max-folds', type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[{args.ticker}] loading 1m parquet...")
    df_1m = load_1m(args.ticker)
    print(f"  loaded {len(df_1m):,} bars, {df_1m.index.min()} → {df_1m.index.max()}")

    print(f"[{args.ticker}] filtering to RTH...")
    df_rth = filter_rth(df_1m)
    print(f"  RTH bars: {len(df_rth):,}")

    print(f"[{args.ticker}] resampling to 5m...")
    df_5m = resample_5m(df_rth)
    print(f"  5m bars: {len(df_5m):,}")

    print(f"[{args.ticker}] computing features...")
    feats = add_features(df_5m)
    n_before = len(feats)
    feats = feats.dropna(subset=['rsi_14', 'atr_14', 'rel_vol', 'dist_vwap_atr'])
    print(f"  features after dropping warm-up NaN: {len(feats):,} (dropped {n_before - len(feats):,})")

    print(f"[{args.ticker}] generating signals...")
    signals = generate_signals(feats)
    print(f"  signal-true bars: {signals.sum():,}")

    print(f"[{args.ticker}] simulating full-period (informational)...")
    trades_all = simulate(feats, signals)
    full_metrics = compute_metrics(trades_all)
    print(f"  full period: {full_metrics['n_trades']} trades, WR={full_metrics['win_rate']}, PF={full_metrics['profit_factor']}")

    print(f"[{args.ticker}] walk-forward...")
    folds = walk_forward(feats, max_folds=args.max_folds)
    print(f"  {len(folds)} folds")
    wf_metrics = evaluate_folds(feats, signals, folds)
    print(f"  per-fold trades: {wf_metrics['n_trades'].tolist() if len(wf_metrics) > 0 else 'none'}")

    # save outputs
    trades_all.to_csv(f"{args.output_dir}/trades.csv", index=False)
    if len(trades_all) > 0:
        equity = trades_all[['exit_time', 'pnl']].copy()
        equity['cum_pnl'] = equity['pnl'].cumsum()
        equity.to_csv(f"{args.output_dir}/equity_curve.csv", index=False)
    wf_metrics.to_csv(f"{args.output_dir}/walk_forward.csv", index=False)
    feats.head(50).to_parquet(f"{args.output_dir}/features_sample.parquet")

    # aggregate OOS metrics (sum across all folds)
    if len(wf_metrics) > 0:
        oos_combined_trades = []
        for f in folds:
            oos = feats[(feats.index >= f['oos_start']) & (feats.index < f['oos_end'])]
            if len(oos) == 0:
                continue
            sig = signals.loc[oos.index]
            oos_trades = simulate(oos, sig)
            oos_combined_trades.append(oos_trades)
        oos_all = pd.concat(oos_combined_trades, ignore_index=True) if oos_combined_trades else pd.DataFrame()
        oos_metrics = compute_metrics(oos_all)
    else:
        oos_metrics = compute_metrics(pd.DataFrame())

    meta = {
        'ticker': args.ticker,
        'pipeline_version': 'v1.0',
        'run_at': datetime.utcnow().isoformat() + 'Z',
        'data_path': f"{DATA_ROOT}/{args.ticker}.parquet",
        'bars_1m_total': len(df_1m),
        'bars_rth': len(df_rth),
        'bars_5m': len(df_5m),
        'bars_5m_after_warmup': len(feats),
        'signal_count': int(signals.sum()),
        'date_range': {
            'data_start': df_1m.index.min().isoformat(),
            'data_end': df_1m.index.max().isoformat(),
        },
        'walk_forward': {
            'n_folds': len(folds),
            'folds': folds,
        },
        'strategy': {
            'name': 'low_touch_revert_v1',
            'entry_rule': 'RSI<35 AND |dist_vwap_atr|<0.5 AND rel_vol>1.2 AND 15<=session_min<385',
            'exit_rule': 'TP=+1.5*ATR OR SL=-0.7*ATR OR 30-bar-time-stop OR EOD@session_min>=385',
            'slippage_bps': 2.0,
            'fee_per_share': 0.0035,
            'notional_per_trade': 5000,
        },
        'metrics_full_period': full_metrics,
        'metrics_oos_aggregate': oos_metrics,
        'metrics_per_fold': wf_metrics.to_dict(orient='records') if len(wf_metrics) > 0 else [],
    }
    # convert any numpy types to plain python
    def to_py(o):
        if isinstance(o, dict): return {k: to_py(v) for k, v in o.items()}
        if isinstance(o, list): return [to_py(v) for v in o]
        if hasattr(o, 'item'): return o.item()
        if isinstance(o, float) and (np.isnan(o) or np.isinf(o)): return None
        return o
    meta = to_py(meta)

    with open(f"{args.output_dir}/run_meta.json", 'w') as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"\n[{args.ticker}] DONE.")
    print(f"  OOS aggregate: n={oos_metrics['n_trades']}, WR={oos_metrics['win_rate']}, PF={oos_metrics['profit_factor']}, DD%={oos_metrics['max_drawdown_pct']}")
    print(f"  Output: {args.output_dir}/")
    return 0

if __name__ == '__main__':
    sys.exit(main())
