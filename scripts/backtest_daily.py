"""
backtest_daily.py — Daily-frequency backtest (DeepSeek's strategic pivot).

Strategies tested:
    D1_REV: Daily mean-reversion. Entry: RSI(14) < 30 at daily close. Exit: 21 days OR TP=1.5 ATR / SL=1.0 ATR.
    D2_MOM: Daily momentum. Entry: 21-day return in top 30% of own history. Exit: 21 days OR TP=1.5 ATR / SL=1.0 ATR.
    D3_GOLD: Golden cross pullback. Entry: close > SMA(50) AND close > SMA(200) AND close < SMA(20). Exit: same.

Same WF methodology, just daily bars from RESAMPLING the 1-min parquet.

Usage:
    python backtest_daily.py --ticker AAPL --strategy D1_REV --output-dir backtests_daily/D1_REV/AAPL
"""

import argparse, json, os, sys
from datetime import datetime
import pandas as pd
import numpy as np

DATA_ROOT = "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com/My Drive/claudes test/data/timeframes/S&P500 5 Year Historical Data/Minutes TimeFrames/1Min_merged"

def load_1m_to_daily(ticker):
    df = pd.read_parquet(f"{DATA_ROOT}/{ticker}.parquet").set_index('timestamp').sort_index()
    # filter to RTH (avoid pre-market/after-hours skewing daily OHLCV)
    et = df.index.tz_convert('America/New_York')
    rth = df[((et.hour > 9) | ((et.hour == 9) & (et.minute >= 30))) & (et.hour < 16)].copy()
    # resample to daily: OHLCV from session bars only
    daily = rth.resample('1D', closed='left', label='left').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
        'volume': 'sum', 'trade_count': 'sum',
    }).dropna(subset=['open', 'high', 'low', 'close'])
    return daily

def add_daily_features(df):
    df = df.copy()
    # RSI(14) daily
    delta = df['close'].diff()
    gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['rsi_14'] = (100 - 100/(1+rs)).shift(1)
    # ATR(14) daily
    h, l, c = df['high'], df['low'], df['close']
    pc = c.shift(1)
    tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    df['atr_14'] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean().shift(1)
    # SMAs
    for n in [5, 10, 20, 50, 200]:
        df[f'sma_{n}'] = df['close'].rolling(n).mean().shift(1)
    # 21-day return percentile (rolling lookback for percentile calc)
    df['ret_21d'] = df['close'].pct_change(21).shift(1)
    # 21-day volume avg
    df['vol_avg_21'] = df['volume'].rolling(21).mean().shift(1)
    return df

def signals_D1_REV(df):
    """Daily RSI(14)<30 mean-reversion. DeepSeek's pivot pick."""
    return ((df['rsi_14'] < 30) &
            df['rsi_14'].notna() & df['atr_14'].notna())

def signals_D2_MOM(df):
    """Daily momentum: top 30% 21-day return + above-avg volume."""
    # use trailing 252-day percentile rank to define "top 30%"
    rank = df['ret_21d'].rolling(252, min_periods=60).rank(pct=True).shift(1)
    return ((rank > 0.70) &
            (df['volume'] > df['vol_avg_21']) &
            df['rsi_14'].notna() & df['atr_14'].notna() & rank.notna())

def signals_D3_GOLD(df):
    """Golden cross with pullback: close > SMA50 > SMA200 AND close < SMA20 (long-term up, short-term pulled back)."""
    return ((df['close'] > df['sma_50']) &
            (df['sma_50'] > df['sma_200']) &
            (df['close'] < df['sma_20']) &
            df['sma_200'].notna() & df['atr_14'].notna())

STRATEGY_MAP = {'D1_REV': signals_D1_REV, 'D2_MOM': signals_D2_MOM, 'D3_GOLD': signals_D3_GOLD}

def simulate_daily(df, signals, side, tp_atr, sl_atr, max_hold_days, slippage_bps=5.0, fee_per_share=0.0035, notional=5000):
    """Daily simulator. One position at a time per ticker. Entry on next day's open."""
    trades = []
    in_pos = False
    entry_bar = entry_price = entry_atr = None; qty = 0
    rows = df.reset_index().to_dict('records')
    sig = signals.values
    sign = 1 if side == 'long' else -1

    for i in range(len(rows) - 1):
        bar = rows[i]; nxt = rows[i+1]
        if not in_pos:
            if sig[i]:
                slip = slippage_bps / 10000
                entry_price = nxt['open'] * (1 + sign * slip)
                entry_bar = i + 1
                entry_atr = bar['atr_14']
                qty = int(notional / entry_price)
                if qty < 1: continue
                in_pos = True
        else:
            cur = rows[i]
            tp = entry_price + sign * tp_atr * entry_atr
            sl = entry_price - sign * sl_atr * entry_atr
            bars_held = i - entry_bar
            exit_reason, exit_price = None, None

            if sign == 1:
                if cur['high'] >= tp:
                    exit_reason, exit_price = 'TP', tp * (1 - slippage_bps/10000)
                elif cur['low'] <= sl:
                    exit_reason, exit_price = 'SL', sl * (1 - slippage_bps/10000)
            else:
                if cur['low'] <= tp:
                    exit_reason, exit_price = 'TP', tp * (1 + slippage_bps/10000)
                elif cur['high'] >= sl:
                    exit_reason, exit_price = 'SL', sl * (1 + slippage_bps/10000)

            if not exit_reason and bars_held >= max_hold_days:
                exit_reason, exit_price = 'TIME', cur['close'] * (1 - sign*slippage_bps/10000)

            if exit_reason:
                pnl_per_share = (exit_price - entry_price) * sign
                fees = fee_per_share * qty * 2
                pnl = pnl_per_share * qty - fees
                trades.append({
                    'entry_time': rows[entry_bar]['timestamp'], 'entry_price': entry_price,
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
                'tp_exits_pct': 0, 'sl_exits_pct': 0, 'time_exits_pct': 0}
    n = len(t); w = t[t['pnl']>0]; l = t[t['pnl']<=0]
    gw = w['pnl'].sum(); gl = abs(l['pnl'].sum())
    ec = t.copy()
    ec['cum_pnl'] = t['pnl'].cumsum()
    return {'n_trades': n, 'win_rate': len(w)/n,
            'profit_factor': gw/gl if gl>0 else float('inf'),
            'expectancy': t['pnl'].mean(), 'total_return_pct': t['pnl'].sum()/50000,
            'max_drawdown_pct': (ec['cum_pnl']-ec['cum_pnl'].cummax()).min()/50000,
            'sharpe': (t['pnl_pct'].mean()/t['pnl_pct'].std())*np.sqrt(12) if len(t)>5 and t['pnl_pct'].std()>0 else None,
            'tp_exits_pct': (t['exit_reason']=='TP').mean(), 'sl_exits_pct': (t['exit_reason']=='SL').mean(),
            'time_exits_pct': (t['exit_reason']=='TIME').mean()}

def walk_forward(df, train_months=24, test_months=12, step_months=12, max_folds=3):
    start = df.index.min(); end = df.index.max()
    folds = []; cur = start + pd.DateOffset(months=train_months)
    while True:
        ee = cur + pd.DateOffset(months=test_months)
        if ee > end or len(folds) >= max_folds: break
        folds.append({'fold': len(folds)+1, 'oos_start': cur.isoformat(), 'oos_end': ee.isoformat()})
        cur = cur + pd.DateOffset(months=step_months)
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
    ap.add_argument('--side', default='long', choices=['long','short'])
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--tp-atr', type=float, default=1.5)
    ap.add_argument('--sl-atr', type=float, default=1.0)
    ap.add_argument('--max-hold', type=int, default=21)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[{args.ticker}/{args.strategy}/{args.side}] daily resample...")
    d = load_1m_to_daily(args.ticker)
    print(f"  daily bars: {len(d):,}")
    f = add_daily_features(d).dropna(subset=['rsi_14','atr_14'])
    print(f"  features after warmup: {len(f):,}")

    sigs = STRATEGY_MAP[args.strategy](f)
    print(f"  signals: {sigs.sum():,}")

    folds = walk_forward(f, max_folds=8)
    all_oos = []
    for fold in folds:
        oos = f[(f.index >= fold['oos_start']) & (f.index < fold['oos_end'])]
        if len(oos) == 0: continue
        s = sigs.loc[oos.index]
        tr = simulate_daily(oos, s, args.side, args.tp_atr, args.sl_atr, args.max_hold)
        all_oos.append(tr)
    oos_combined = pd.concat(all_oos, ignore_index=True) if all_oos else pd.DataFrame()
    om = metrics(oos_combined)

    oos_combined.to_csv(f"{args.output_dir}/trades.csv", index=False)
    meta = to_py({
        'ticker': args.ticker, 'pipeline_version': 'daily_v1',
        'strategy_variant': f"{args.strategy}_{args.side}",
        'run_at': datetime.utcnow().isoformat()+'Z',
        'daily_bars': len(d), 'features_after_warmup': len(f), 'signal_count': int(sigs.sum()),
        'strategy': {'name': args.strategy, 'side': args.side, 'tp_atr': args.tp_atr, 'sl_atr': args.sl_atr,
                     'max_hold_days': args.max_hold, 'slippage_bps': 5.0, 'fee_per_share': 0.0035},
        'metrics_oos_aggregate': om,
        'walk_forward': {'folds': folds, 'train_months': 24, 'test_months': 12, 'step_months': 12},
    })
    with open(f"{args.output_dir}/run_meta.json",'w') as fp: json.dump(meta, fp, indent=2, default=str)
    print(f"  OOS: n={om['n_trades']}, WR={om['win_rate']}, PF={om['profit_factor']}, RET%={om['total_return_pct']}")

if __name__ == '__main__': main()
