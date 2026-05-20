"""
backtest_v3.py — Wave 2 redesign with momentum + short support.

New vs v2:
    - Strategy M1: momentum continuation (DeepSeek's #1 recommendation)
    - Strategy 1A_short: same as 1A entry but short — tests if there's a tradeable negative drift
    - --side {long,short} flag for any strategy
    - VWAP-cross dynamic exit (for M1)

Usage:
    python backtest_v3.py --ticker AAPL --strategy M1 --side long --output-dir backtests_wave2/M1/AAPL
    python backtest_v3.py --ticker AAPL --strategy 1A --side short --output-dir backtests_wave2/S1/AAPL
"""

import argparse, json, os, sys
from datetime import datetime
import pandas as pd
import numpy as np

DATA_ROOT = "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/claudes test/data/timeframes/S&P500 5 Year Historical Data/Minutes TimeFrames/1Min_merged"

def load_1m(t): return pd.read_parquet(f"{DATA_ROOT}/{t}.parquet").set_index('timestamp').sort_index()

def filter_rth(df):
    et = df.index.tz_convert('America/New_York')
    return df[((et.hour > 9) | ((et.hour == 9) & (et.minute >= 30))) & (et.hour < 16)].copy()

def resample_5m(df):
    return df.resample('5min', closed='left', label='left').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum', 'trade_count': 'sum',
    }).dropna(subset=['open', 'high', 'low', 'close'])

def add_features(df):
    et = df.index.tz_convert('America/New_York')
    df = df.copy()
    df['session_date'] = et.date
    df['session_minute'] = (et.hour - 9) * 60 + et.minute - 30
    # VWAP session-cumulative
    typical = (df['high'] + df['low'] + df['close']) / 3
    cum_tpv = (typical * df['volume']).groupby(df['session_date']).cumsum()
    cum_v = df['volume'].groupby(df['session_date']).cumsum()
    df['vwap_session'] = cum_tpv / cum_v
    df['n_vwap_bars'] = df.groupby('session_date').cumcount() + 1
    # RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['rsi_14'] = (100 - 100/(1+rs)).shift(1)
    # ATR
    h, l, c = df['high'], df['low'], df['close']
    pc = c.shift(1)
    tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    df['atr_14'] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean().shift(1)
    # rel_vol by minute-of-session over prior 20 sessions
    by_min = df.groupby('session_minute')['volume']
    df['rel_vol'] = df['volume'] / by_min.transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
    # 3-bar return (shifted so bar t value = return computed up to bar t-1)
    df['ret_3bar'] = df['close'].pct_change(3).shift(1)
    return df

def signals_1A(df):
    return ((df['rsi_14'] < 35) & (df['rel_vol'] > 1.2) & (df['n_vwap_bars'] >= 10) &
            (df['session_minute'] >= 15) & (df['session_minute'] < 385) &
            df['rsi_14'].notna() & df['atr_14'].notna() & df['rel_vol'].notna())

def signals_M1(df):
    """DeepSeek's #1: momentum continuation. 3-bar return > +0.3%, close > vwap, RSI>55, vol>1.0."""
    return ((df['ret_3bar'] > 0.003) &
            (df['close'] > df['vwap_session']) &
            (df['rsi_14'] > 55) &
            (df['rel_vol'] > 1.0) &
            (df['n_vwap_bars'] >= 10) &
            (df['session_minute'] >= 30) & (df['session_minute'] < 375) &
            df['rsi_14'].notna() & df['atr_14'].notna() & df['rel_vol'].notna() & df['ret_3bar'].notna())

STRATEGY_MAP = {'1A': signals_1A, 'M1': signals_M1}

def simulate(df, signals, side, tp_atr, sl_atr, time_stop, vwap_cross_exit=False,
             slippage_bps=2.0, fee_per_share=0.0035, notional=5000):
    """Simulator supporting long or short with optional VWAP-cross dynamic exit (for M1)."""
    trades = []
    in_pos = False
    entry_bar = None; entry_price = None; entry_atr = None; entry_session = None; qty = 0

    rows = df.reset_index().to_dict('records')
    sig = signals.values
    sign = 1 if side == 'long' else -1  # +1 long, -1 short

    for i in range(len(rows) - 1):
        bar = rows[i]; nxt = rows[i+1]
        if not in_pos:
            if sig[i] and bar['session_minute'] >= 15 and bar['session_minute'] < 380:
                slip = slippage_bps/10000
                # for long: pay ask, slightly above open; for short: hit bid, slightly below open
                entry_price = nxt['open'] * (1 + sign*slip)
                entry_bar = i+1; entry_atr = bar['atr_14']; entry_session = bar['session_date']
                qty = int(notional / entry_price)
                if qty < 1: continue
                in_pos = True
        else:
            cur = rows[i]
            # for long: TP above entry, SL below; for short: TP below entry, SL above
            tp = entry_price + sign * tp_atr * entry_atr
            sl = entry_price - sign * sl_atr * entry_atr
            bars_held = i - entry_bar
            exit_reason, exit_price = None, None

            if cur['session_date'] != entry_session:
                exit_reason, exit_price = 'EOD_CROSS', rows[i-1]['close'] * (1 - sign*slippage_bps/10000)
            elif cur['session_minute'] >= 385:
                exit_reason, exit_price = 'EOD', cur['close'] * (1 - sign*slippage_bps/10000)
            elif sign == 1:  # long
                if cur['high'] >= tp:
                    exit_reason, exit_price = 'TP', tp * (1 - slippage_bps/10000)
                elif cur['low'] <= sl:
                    exit_reason, exit_price = 'SL', sl * (1 - slippage_bps/10000)
            else:  # short
                if cur['low'] <= tp:
                    exit_reason, exit_price = 'TP', tp * (1 + slippage_bps/10000)
                elif cur['high'] >= sl:
                    exit_reason, exit_price = 'SL', sl * (1 + slippage_bps/10000)
            # VWAP-cross dynamic exit (M1 only)
            if not exit_reason and vwap_cross_exit:
                if sign == 1 and cur['close'] < cur['vwap_session']:
                    exit_reason, exit_price = 'VWAP_CROSS', nxt['open'] * (1 - slippage_bps/10000)
                elif sign == -1 and cur['close'] > cur['vwap_session']:
                    exit_reason, exit_price = 'VWAP_CROSS', nxt['open'] * (1 + slippage_bps/10000)
            # time stop
            if not exit_reason and bars_held >= time_stop:
                exit_reason, exit_price = 'TIME', cur['close'] * (1 - sign*slippage_bps/10000)

            if exit_reason:
                # PnL: long = (exit - entry) * qty; short = (entry - exit) * qty
                pnl_per_share = (exit_price - entry_price) * sign
                fees = fee_per_share * qty * 2
                pnl = pnl_per_share * qty - fees
                trades.append({
                    'entry_time': rows[entry_bar]['timestamp'], 'entry_price': entry_price,
                    'entry_session': str(entry_session),
                    'exit_time': cur['timestamp'], 'exit_price': exit_price, 'exit_reason': exit_reason,
                    'qty': qty, 'bars_held': bars_held, 'pnl': pnl,
                    'pnl_pct': pnl_per_share/entry_price, 'entry_atr': entry_atr, 'fees': fees, 'side': side,
                })
                in_pos = False; entry_bar = entry_price = None; qty = 0
    return pd.DataFrame(trades)

def metrics(t):
    if len(t) == 0:
        return {'n_trades': 0, 'win_rate': None, 'profit_factor': None, 'expectancy': None,
                'total_return_pct': 0, 'max_drawdown_pct': 0, 'sharpe': None,
                'sl_exits_pct': 0, 'tp_exits_pct': 0, 'eod_exits_pct': 0, 'time_exits_pct': 0, 'vwap_cross_exits_pct': 0}
    n = len(t); w = t[t['pnl']>0]; l = t[t['pnl']<=0]
    gw = w['pnl'].sum(); gl = abs(l['pnl'].sum())
    return {'n_trades': n, 'win_rate': len(w)/n,
            'profit_factor': gw/gl if gl>0 else float('inf'),
            'expectancy': t['pnl'].mean(), 'total_return_pct': t['pnl'].sum()/50000,
            'max_drawdown_pct': (t['pnl'].cumsum()-t['pnl'].cumsum().cummax()).min()/50000,
            'sharpe': (t['pnl_pct'].mean()/t['pnl_pct'].std())*np.sqrt(252) if len(t)>5 and t['pnl_pct'].std()>0 else None,
            **{k+'_pct': t['exit_reason'].value_counts(normalize=True).get(k,0) for k in ['TP','SL','EOD','TIME','VWAP_CROSS']}}

def walk_forward(df, train=12, test=3, step=3, max_folds=8):
    start, end = df.index.min(), df.index.max()
    folds = []; cur = start + pd.DateOffset(months=train)
    while True:
        ee = cur + pd.DateOffset(months=test)
        if ee > end or len(folds) >= max_folds: break
        folds.append({'fold': len(folds)+1, 'oos_start': cur.isoformat(), 'oos_end': ee.isoformat()})
        cur = cur + pd.DateOffset(months=step)
    return folds

def to_py(o):
    if isinstance(o, dict): return {k: to_py(v) for k,v in o.items()}
    if isinstance(o, list): return [to_py(v) for v in o]
    if hasattr(o, 'item'): return o.item()
    if isinstance(o, float) and (np.isnan(o) or np.isinf(o)): return None
    return o

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ticker', required=True)
    ap.add_argument('--strategy', required=True, choices=list(STRATEGY_MAP.keys()))
    ap.add_argument('--side', default='long', choices=['long', 'short'])
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--tp-atr', type=float, default=1.0)
    ap.add_argument('--sl-atr', type=float, default=1.0)
    ap.add_argument('--time-stop', type=int, default=30)
    ap.add_argument('--vwap-cross-exit', action='store_true')
    ap.add_argument('--max-folds', type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[{args.ticker}/{args.strategy}/{args.side}] loading...")
    d = filter_rth(load_1m(args.ticker))
    f5 = add_features(resample_5m(d))
    f5 = f5.dropna(subset=['rsi_14','atr_14','rel_vol'])
    print(f"  features: {len(f5):,}")
    sigs = STRATEGY_MAP[args.strategy](f5)
    print(f"  signals: {sigs.sum():,}")

    folds = walk_forward(f5, max_folds=args.max_folds)
    fold_m = []; all_oos = []
    for fold in folds:
        oos = f5[(f5.index >= fold['oos_start']) & (f5.index < fold['oos_end'])]
        if len(oos) == 0: continue
        s = sigs.loc[oos.index]
        tr = simulate(oos, s, args.side, args.tp_atr, args.sl_atr, args.time_stop, vwap_cross_exit=args.vwap_cross_exit)
        m = metrics(tr); m['fold'] = fold['fold']
        fold_m.append(m); all_oos.append(tr)
    oos_combined = pd.concat(all_oos, ignore_index=True) if all_oos else pd.DataFrame()
    om = metrics(oos_combined)

    oos_combined.to_csv(f"{args.output_dir}/trades.csv", index=False)
    pd.DataFrame(fold_m).to_csv(f"{args.output_dir}/walk_forward.csv", index=False)
    f5.iloc[:500].to_parquet(f"{args.output_dir}/features_sample_multi_session.parquet")

    meta = to_py({
        'ticker': args.ticker, 'pipeline_version': 'v3.0',
        'strategy_variant': f"{args.strategy}_{args.side}",
        'run_at': datetime.utcnow().isoformat()+'Z',
        'bars_5m_after_warmup': len(f5), 'signal_count': int(sigs.sum()),
        'strategy': {'name': args.strategy, 'side': args.side, 'tp_atr': args.tp_atr, 'sl_atr': args.sl_atr,
                     'time_stop_bars': args.time_stop, 'vwap_cross_exit': args.vwap_cross_exit,
                     'slippage_bps': 2.0, 'fee_per_share': 0.0035, 'notional_per_trade': 5000},
        'metrics_oos_aggregate': om, 'metrics_per_fold': fold_m, 'walk_forward': {'folds': folds},
    })
    with open(f"{args.output_dir}/run_meta.json",'w') as fp: json.dump(meta, fp, indent=2, default=str)
    print(f"  OOS: n={om['n_trades']}, WR={om['win_rate']}, PF={om['profit_factor']}, RET%={om['total_return_pct']}")

if __name__ == '__main__': main()
