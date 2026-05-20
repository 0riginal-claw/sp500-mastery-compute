"""
backtest_v2.py — Wave 1 pipeline, two strategy variants informed by DeepSeek pilot critique.

Changes from v1:
    - --strategy {1A, 1B} selector
    - n_vwap_bars stability gate (DeepSeek finding #4)
    - Inverted SL/TP for reversion: TP=0.7 ATR, SL=1.0 ATR (DeepSeek finding #3)
    - Save BOTH a 50-row single-session sample AND a 500-row multi-session sample for validator

Usage:
    python backtest_v2.py --ticker AAPL --strategy 1A --output-dir backtests_wave1/1A/AAPL
"""

import argparse, json, os, sys
from datetime import datetime
import pandas as pd
import numpy as np

DATA_ROOT = "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/claudes test/data/timeframes/S&P500 5 Year Historical Data/Minutes TimeFrames/1Min_merged"

def load_1m(ticker):
    df = pd.read_parquet(f"{DATA_ROOT}/{ticker}.parquet")
    return df.set_index('timestamp').sort_index()

def filter_rth(df):
    et = df.index.tz_convert('America/New_York')
    mask = (
        ((et.hour > 9) | ((et.hour == 9) & (et.minute >= 30))) &
        (et.hour < 16)
    )
    return df[mask].copy()

def resample_5m(df_rth):
    return df_rth.resample('5min', closed='left', label='left').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
        'volume': 'sum', 'trade_count': 'sum',
    }).dropna(subset=['open', 'high', 'low', 'close'])

def add_session_id(df):
    et = df.index.tz_convert('America/New_York')
    df = df.copy()
    df['session_date'] = et.date
    df['session_minute'] = (et.hour - 9) * 60 + et.minute - 30
    return df

def add_vwap(df):
    """Cumulative session VWAP + a counter of bars-into-session (for stability gate)."""
    df = df.copy()
    typical = (df['high'] + df['low'] + df['close']) / 3
    tp_v = typical * df['volume']
    cum_tpv = tp_v.groupby(df['session_date']).cumsum()
    cum_v = df['volume'].groupby(df['session_date']).cumsum()
    df['vwap_session'] = cum_tpv / cum_v
    # n_vwap_bars = how many bars contributed to the running VWAP at THIS bar
    df['n_vwap_bars'] = df.groupby('session_date').cumcount() + 1
    return df

def add_rsi(df, period=14):
    df = df.copy()
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['rsi_14'] = (100 - 100 / (1 + rs)).shift(1)
    return df

def add_atr(df, period=14):
    df = df.copy()
    h, l, c = df['high'], df['low'], df['close']
    prev_close = c.shift(1)
    tr = pd.concat([h - l, (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(axis=1)
    df['atr_14'] = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean().shift(1)
    return df

def add_rel_vol(df, period=20):
    df = df.copy()
    by_min = df.groupby('session_minute')['volume']
    df['_vol_lag'] = by_min.transform(lambda s: s.shift(1).rolling(period, min_periods=5).mean())
    df['rel_vol'] = df['volume'] / df['_vol_lag']
    return df.drop(columns=['_vol_lag'])

def add_pdh_pdl(df):
    sess_hl = df.groupby('session_date').agg(daily_high=('high', 'max'), daily_low=('low', 'min'))
    sess_hl['pdh'] = sess_hl['daily_high'].shift(1)
    sess_hl['pdl'] = sess_hl['daily_low'].shift(1)
    return df.merge(sess_hl[['pdh', 'pdl']], left_on='session_date', right_index=True, how='left')

def add_features(df):
    df = add_session_id(df)
    df = add_vwap(df)
    df = add_rsi(df, 14)
    df = add_atr(df, 14)
    df = add_rel_vol(df, 20)
    df = add_pdh_pdl(df)
    df['dist_vwap_atr'] = (df['close'] - df['vwap_session']) / df['atr_14']
    return df

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def signals_1A(df):
    """Variant 1A: Drop VWAP gate, add n_vwap_bars>=10, no dist_vwap requirement."""
    return (
        (df['rsi_14'] < 35) &
        (df['rel_vol'] > 1.2) &
        (df['n_vwap_bars'] >= 10) &
        (df['session_minute'] >= 15) &
        (df['session_minute'] < 385) &
        df['rsi_14'].notna() &
        df['atr_14'].notna() &
        df['rel_vol'].notna()
    )

def signals_1B(df):
    """Variant 1B: Require close < VWAP (true below-VWAP oversold)."""
    return (
        (df['rsi_14'] < 35) &
        (df['close'] < df['vwap_session']) &
        (df['rel_vol'] > 1.2) &
        (df['n_vwap_bars'] >= 10) &
        (df['session_minute'] >= 15) &
        (df['session_minute'] < 385) &
        df['rsi_14'].notna() &
        df['atr_14'].notna() &
        df['rel_vol'].notna()
    )

STRATEGY_MAP = {
    '1A': signals_1A,
    '1B': signals_1B,
}

# ---------------------------------------------------------------------------
# Simulator with reversion-appropriate exits
# ---------------------------------------------------------------------------

def simulate(df, signals, tp_atr=0.7, sl_atr=1.0, time_stop=30, slippage_bps=2.0, fee_per_share=0.0035, notional=5000):
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
            tp = entry_price + tp_atr * entry_atr
            sl = entry_price - sl_atr * entry_atr
            bars_held = i - entry_bar

            exit_reason, exit_price = None, None

            if cur['session_date'] != entry_session:
                exit_reason, exit_price = 'EOD_CROSS', rows[i-1]['close'] * (1 - slippage_bps / 10000)
            elif cur['session_minute'] >= 385:
                exit_reason, exit_price = 'EOD', cur['close'] * (1 - slippage_bps / 10000)
            elif cur['high'] >= tp:
                exit_reason, exit_price = 'TP', tp * (1 - slippage_bps / 10000)
            elif cur['low'] <= sl:
                exit_reason, exit_price = 'SL', sl * (1 - slippage_bps / 10000)
            elif bars_held >= time_stop:
                exit_reason, exit_price = 'TIME', cur['close'] * (1 - slippage_bps / 10000)

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

def compute_metrics(trades_df):
    if len(trades_df) == 0:
        return {'n_trades': 0, 'win_rate': None, 'profit_factor': None,
                'expectancy': None, 'total_pnl': 0, 'max_drawdown': 0,
                'max_drawdown_pct': 0, 'total_return_pct': 0, 'sharpe': None,
                'avg_trade': None, 'avg_win': None, 'avg_loss': None,
                'tp_exits_pct': None, 'sl_exits_pct': None, 'eod_exits_pct': None, 'time_exits_pct': None}
    n = len(trades_df)
    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] <= 0]
    gross_win = wins['pnl'].sum()
    gross_loss = abs(losses['pnl'].sum())
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')
    expectancy = trades_df['pnl'].mean()
    total_pnl = trades_df['pnl'].sum()
    equity = trades_df['pnl'].cumsum()
    drawdown = equity - equity.cummax()
    max_dd = drawdown.min()
    baseline = 50000
    sharpe = (trades_df['pnl_pct'].mean() / trades_df['pnl_pct'].std()) * np.sqrt(252) if len(trades_df) > 5 and trades_df['pnl_pct'].std() > 0 else None
    exit_counts = trades_df['exit_reason'].value_counts(normalize=True).to_dict()
    return {
        'n_trades': n,
        'win_rate': len(wins) / n,
        'profit_factor': pf,
        'expectancy': expectancy,
        'total_pnl': total_pnl,
        'max_drawdown': max_dd,
        'max_drawdown_pct': max_dd / baseline,
        'total_return_pct': total_pnl / baseline,
        'sharpe': sharpe,
        'avg_trade': trades_df['pnl'].mean(),
        'avg_win': wins['pnl'].mean() if len(wins) else None,
        'avg_loss': losses['pnl'].mean() if len(losses) else None,
        'tp_exits_pct': exit_counts.get('TP', 0),
        'sl_exits_pct': exit_counts.get('SL', 0),
        'eod_exits_pct': exit_counts.get('EOD', 0),
        'time_exits_pct': exit_counts.get('TIME', 0),
    }

def walk_forward(features, train_months=12, test_months=3, step_months=3, max_folds=8):
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

def to_py(o):
    if isinstance(o, dict): return {k: to_py(v) for k, v in o.items()}
    if isinstance(o, list): return [to_py(v) for v in o]
    if hasattr(o, 'item'): return o.item()
    if isinstance(o, float) and (np.isnan(o) or np.isinf(o)): return None
    return o

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ticker', required=True)
    ap.add_argument('--strategy', required=True, choices=list(STRATEGY_MAP.keys()))
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--max-folds', type=int, default=8)
    ap.add_argument('--tp-atr', type=float, default=0.7)
    ap.add_argument('--sl-atr', type=float, default=1.0)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[{args.ticker}/{args.strategy}] loading...")
    df_1m = load_1m(args.ticker)
    df_rth = filter_rth(df_1m)
    df_5m = resample_5m(df_rth)
    feats = add_features(df_5m)
    feats = feats.dropna(subset=['rsi_14', 'atr_14', 'rel_vol'])
    print(f"  features: {len(feats):,}")

    signals = STRATEGY_MAP[args.strategy](feats)
    print(f"  signals: {signals.sum():,}")

    folds = walk_forward(feats, max_folds=args.max_folds)
    fold_metrics = []
    for f in folds:
        oos = feats[(feats.index >= f['oos_start']) & (feats.index < f['oos_end'])]
        if len(oos) == 0: continue
        sig = signals.loc[oos.index]
        trades = simulate(oos, sig, tp_atr=args.tp_atr, sl_atr=args.sl_atr)
        m = compute_metrics(trades)
        m['fold'] = f['fold']
        m['oos_start'] = f['oos_start']
        m['oos_end'] = f['oos_end']
        fold_metrics.append(m)

    # OOS aggregate
    all_oos_trades = []
    for f in folds:
        oos = feats[(feats.index >= f['oos_start']) & (feats.index < f['oos_end'])]
        if len(oos) == 0: continue
        sig = signals.loc[oos.index]
        all_oos_trades.append(simulate(oos, sig, tp_atr=args.tp_atr, sl_atr=args.sl_atr))
    oos_combined = pd.concat(all_oos_trades, ignore_index=True) if all_oos_trades else pd.DataFrame()
    oos_metrics = compute_metrics(oos_combined)

    oos_combined.to_csv(f"{args.output_dir}/trades.csv", index=False)
    if len(oos_combined) > 0:
        equity = oos_combined[['exit_time', 'pnl']].copy()
        equity['cum_pnl'] = equity['pnl'].cumsum()
        equity.to_csv(f"{args.output_dir}/equity_curve.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(f"{args.output_dir}/walk_forward.csv", index=False)
    feats.head(50).to_parquet(f"{args.output_dir}/features_sample.parquet")
    # multi-session sample for upgraded validator: 500 rows spanning ~5 sessions
    feats.iloc[:500].to_parquet(f"{args.output_dir}/features_sample_multi_session.parquet")

    meta = {
        'ticker': args.ticker,
        'pipeline_version': 'v2.0',
        'strategy_variant': args.strategy,
        'run_at': datetime.utcnow().isoformat() + 'Z',
        'data_path': f"{DATA_ROOT}/{args.ticker}.parquet",
        'bars_1m_total': len(df_1m),
        'bars_rth': len(df_rth),
        'bars_5m': len(df_5m),
        'bars_5m_after_warmup': len(feats),
        'signal_count': int(signals.sum()),
        'date_range': {'data_start': df_1m.index.min().isoformat(), 'data_end': df_1m.index.max().isoformat()},
        'walk_forward': {'n_folds': len(folds), 'folds': folds},
        'strategy': {
            'variant': args.strategy,
            'name': f"low_touch_revert_{args.strategy}",
            'tp_atr': args.tp_atr,
            'sl_atr': args.sl_atr,
            'time_stop_bars': 30,
            'slippage_bps': 2.0,
            'fee_per_share': 0.0035,
            'notional_per_trade': 5000,
        },
        'metrics_oos_aggregate': oos_metrics,
        'metrics_per_fold': fold_metrics,
    }
    meta = to_py(meta)
    with open(f"{args.output_dir}/run_meta.json", 'w') as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"  OOS: n={oos_metrics['n_trades']}, WR={oos_metrics['win_rate']}, PF={oos_metrics['profit_factor']}, RET%={oos_metrics['total_return_pct']}")

if __name__ == '__main__':
    main()
