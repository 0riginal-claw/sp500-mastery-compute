"""
backtest_ml.py — Wave 4 ML-augmented daily strategy.

v2 fixes (2026-05-14 code-review):
  - Label-boundary embargo: drop last 21 train bars per fold so forward-21d labels can't peek into OOS
  - OBV-EMA leak (line 88) fixed: compute EMA on raw OBV, single shift(1) at end
  - dist_ema_20_atr dead code removed

v3 fixes (2026-05-16 cloud-compat):
  - BACKTEST_DATA_SOURCE env var gates data loading strategy:
      local_parquet  (default) — reads 1-min parquet from DATA_ROOT (local Drive)
      yfinance_daily            — fetches 5y daily OHLCV from yfinance; no Drive needed
  - DATA_ROOT now also reads from BACKTEST_DATA_ROOT env var so CI can override path
  - pandas_ta_classic import made soft so yfinance path works without it

Approach (per DeepSeek + Trading Insight Info §3):
    1. Compute ~60 daily features per ticker via pandas-ta-classic
    2. Train HistGradientBoosting classifier to predict P(next-21d return > 0)
    3. Walk-forward: 2-yr train / 1-yr OOS / 1-yr step
    4. Calibrate probabilities with CalibratedClassifierCV (Platt)
    5. Threshold-tune: enter only when calibrated p > threshold
    6. Apply expected-value layer: skip if EV < friction

Goal: produce trades on tickers where D1_REV (RSI<30) didn't fire enough times.

Usage (local):
    python backtest_ml.py --ticker AAPL --output-dir backtests_ml/AAPL

Usage (CI / GH Actions — no Drive):
    BACKTEST_DATA_SOURCE=yfinance_daily python backtest_ml.py --ticker AAPL --output-dir /tmp/out
"""

import argparse, json, os, sys, warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit

# pandas_ta_classic is only required for the local parquet path (build_features).
# Import it softly so the yfinance_daily path can run without it installed.
try:
    import pandas_ta_classic as pta
    _PTA_AVAILABLE = True
except ImportError:
    pta = None
    _PTA_AVAILABLE = False

# ---------------------------------------------------------------------------
# Data source configuration — controlled by BACKTEST_DATA_SOURCE env var.
#   local_parquet   (default): read from local 1-min parquet files on Drive
#   yfinance_daily           : fetch 5-year daily OHLCV from yfinance (no Drive needed)
# ---------------------------------------------------------------------------
_DATA_SOURCE = os.environ.get("BACKTEST_DATA_SOURCE", "local_parquet").strip().lower()

DATA_ROOT = os.environ.get(
    "BACKTEST_DATA_ROOT",
    "/Users/orginal/Library/CloudStorage/GoogleDrive-zachgladstone@gmail.com"
    "/My Drive/claudes test/data/timeframes/S&P500 5 Year Historical Data"
    "/Minutes TimeFrames/1Min_merged",
)

# Cache dir for yfinance downloads so re-runs within the same job are fast.
_YF_CACHE_DIR = os.environ.get("BACKTEST_YF_CACHE", "/tmp/data_cache")


def _load_daily_yfinance(ticker: str) -> pd.DataFrame:
    """Fetch 5 years of daily OHLCV from yfinance; cache to /tmp/data_cache/<ticker>.parquet.

    Returns a DataFrame with columns [open, high, low, close, volume] and a
    tz-naive DatetimeIndex of business days (mirrors the shape produced by
    load_daily() from the 1-min parquet path).
    """
    import yfinance as yf

    os.makedirs(_YF_CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(_YF_CACHE_DIR, f"{ticker}_daily.parquet")

    if os.path.exists(cache_file):
        print(f"  [yf] loading cached daily bars: {cache_file}", file=sys.stderr)
        df = pd.read_parquet(cache_file)
    else:
        print(f"  [yf] downloading 5y daily bars for {ticker}...", file=sys.stderr)
        raw = yf.download(ticker, period="5y", interval="1d", auto_adjust=True, progress=False)
        if raw.empty:
            raise RuntimeError(f"yfinance returned no data for {ticker}")
        # Normalise column names to lowercase
        raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in raw.columns]
        # Keep only OHLCV; drop timezone to match local-parquet behaviour
        df = raw[["open", "high", "low", "close", "volume"]].copy()
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index.name = "timestamp"
        df = df.dropna(subset=["open", "high", "low", "close"])
        df.to_parquet(cache_file)
        print(f"  [yf] cached {len(df)} daily bars -> {cache_file}", file=sys.stderr)

    return df


def load_daily(ticker: str) -> pd.DataFrame:
    """Load daily OHLCV bars for ticker.

    Routes to yfinance or local parquet based on BACKTEST_DATA_SOURCE env var.
    """
    if _DATA_SOURCE == "yfinance_daily":
        return _load_daily_yfinance(ticker)

    # Default: local 1-min parquet -> resample to daily RTH bars
    df = pd.read_parquet(f"{DATA_ROOT}/{ticker}.parquet").set_index("timestamp").sort_index()
    et = df.index.tz_convert("America/New_York")
    rth = df[((et.hour > 9) | ((et.hour == 9) & (et.minute >= 30))) & (et.hour < 16)].copy()
    daily = rth.resample("1D", closed="left", label="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum",
    }).dropna(subset=["open", "high", "low", "close"])
    return daily

def _ema_pandas(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average via pandas ewm (no external dep)."""
    return series.ewm(span=span, adjust=False).mean()


def _rsi_pandas(series: pd.Series, length: int = 14) -> pd.Series:
    """Wilder RSI via pure pandas."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def _atr_pandas(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """Average True Range via pure pandas."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def _build_features_pandas_fallback(df: pd.DataFrame) -> pd.DataFrame:
    """Pure-pandas feature builder used when pandas_ta_classic is not available.

    Produces a subset of features (~30) that covers all pipeline-required columns
    (rsi_14, atr_14, ema_200, fwd_ret_21d, y) plus enough signal features to
    run XGBoost folds without crashing.  No look-ahead: every feature uses
    shift(1) except the forward label.
    """
    df = df.copy()
    c = df["close"]; h = df["high"]; l = df["low"]; v = df["volume"]

    # RSI variants
    for n in [7, 14, 21, 28]:
        df[f"rsi_{n}"] = _rsi_pandas(c, n).shift(1)

    # EMAs
    for n in [5, 10, 20, 50, 200]:
        df[f"ema_{n}"] = _ema_pandas(c, n).shift(1)

    # ATR
    df["atr_14"] = _atr_pandas(h, l, c, 14).shift(1)
    df["atr_28"] = _atr_pandas(h, l, c, 28).shift(1)

    # MACD (12/26/9 EMA)
    ema12 = _ema_pandas(c, 12); ema26 = _ema_pandas(c, 26)
    macd_line = ema12 - ema26
    signal_line = _ema_pandas(macd_line, 9)
    df["macd"] = macd_line.shift(1)
    df["macd_signal"] = signal_line.shift(1)
    df["macd_hist"] = (macd_line - signal_line).shift(1)

    # Bollinger Bands (20, 2)
    sma20 = c.rolling(20).mean(); std20 = c.rolling(20).std()
    bb_upper = sma20 + 2 * std20; bb_lower = sma20 - 2 * std20
    df["bb_upper"] = bb_upper.shift(1)
    df["bb_lower"] = bb_lower.shift(1)
    df["bb_width"] = (bb_upper - bb_lower).shift(1)
    bb_range = (bb_upper - bb_lower).replace(0, float("nan"))
    df["bb_pct"] = ((c - bb_lower) / bb_range).shift(1)

    # Volume features
    df["vol_sma_20"] = v.rolling(20).mean().shift(1)
    df["vol_ratio"] = (v / v.rolling(20).mean()).shift(1)

    # Returns
    for n in [1, 3, 5, 10, 21, 63]:
        df[f"ret_{n}d"] = c.pct_change(n).shift(1)

    # Distance from EMA20 / ATR
    df["dist_ema_20_atr"] = (c.shift(1) - df["ema_20"]) / df["atr_14"]
    df["atr_pct"] = df["atr_14"] / c.shift(1)

    # SMA crossover flags
    df["ema_5_gt_ema_20"] = (df["ema_5"] > df["ema_20"]).astype(int)
    df["ema_20_gt_ema_50"] = (df["ema_20"] > df["ema_50"]).astype(int)
    df["close_gt_ema_50"] = (c.shift(1) > df["ema_50"]).astype(int)

    # Daily range normalised by ATR
    df["daily_range_atr"] = ((h - l) / df["atr_14"]).shift(1)

    # OBV (simple running sum direction)
    direction = pd.Series(0.0, index=c.index)
    direction[c > c.shift(1)] = 1.0
    direction[c < c.shift(1)] = -1.0
    _obv_raw = (direction * v).cumsum()
    df["obv"] = _obv_raw.shift(1)
    df["obv_ema_20"] = _ema_pandas(_obv_raw, 20).shift(1)

    # Forward label (no shift — looks 21 bars ahead from bar t)
    df["fwd_ret_21d"] = c.pct_change(21).shift(-21)
    df["y"] = (df["fwd_ret_21d"] > 0).astype(int)

    return df


def build_features(df):
    """~60 features from pandas-ta-classic + custom. All shift(1) for no-lookahead.

    When BACKTEST_DATA_SOURCE=yfinance_daily and pandas_ta_classic is not installed,
    only the hand-coded features (returns, EMA, ATR, vol ratios) are computed.
    All pipeline-required columns (rsi_14, atr_14, ema_200, fwd_ret_21d, y) are
    always produced via pure-pandas fallbacks so the pipeline does not crash.
    """
    if not _PTA_AVAILABLE:
        return _build_features_pandas_fallback(df)

    df = df.copy()
    c = df['close']; h = df['high']; l = df['low']; v = df['volume']
    o = df['open']

    # RSI variants
    for n in [7, 14, 21, 28]:
        df[f'rsi_{n}'] = pta.rsi(c, length=n).shift(1)

    # EMAs
    for n in [5, 10, 20, 50, 200]:
        df[f'ema_{n}'] = pta.ema(c, length=n).shift(1)

    # ATR
    df['atr_14'] = pta.atr(h, l, c, length=14).shift(1)
    df['atr_28'] = pta.atr(h, l, c, length=28).shift(1)

    # MACD
    macd = pta.macd(c)
    df['macd'] = macd['MACD_12_26_9'].shift(1)
    df['macd_signal'] = macd['MACDs_12_26_9'].shift(1)
    df['macd_hist'] = macd['MACDh_12_26_9'].shift(1)

    # Bollinger Bands
    bb = pta.bbands(c, length=20)
    df['bb_upper'] = bb['BBU_20_2.0'].shift(1)
    df['bb_lower'] = bb['BBL_20_2.0'].shift(1)
    df['bb_width'] = (bb['BBU_20_2.0'] - bb['BBL_20_2.0']).shift(1)
    df['bb_pct'] = ((c - bb['BBL_20_2.0']) / (bb['BBU_20_2.0'] - bb['BBL_20_2.0'])).shift(1)

    # ADX
    adx = pta.adx(h, l, c, length=14)
    df['adx_14'] = adx['ADX_14'].shift(1)
    df['dmp_14'] = adx['DMP_14'].shift(1)
    df['dmn_14'] = adx['DMN_14'].shift(1)

    # CCI
    df['cci_20'] = pta.cci(h, l, c, length=20).shift(1)

    # MFI
    df['mfi_14'] = pta.mfi(h, l, c, v, length=14).shift(1)

    # Williams %R
    df['willr_14'] = pta.willr(h, l, c, length=14).shift(1)

    # OBV — compute raw, then EMA on raw, then shift once to avoid leak
    _obv_raw = pta.obv(c, v)
    df['obv'] = _obv_raw.shift(1)
    df['obv_ema_20'] = pta.ema(_obv_raw, length=20).shift(1)

    # Stochastic
    stoch = pta.stoch(h, l, c)
    df['stoch_k'] = stoch['STOCHk_14_3_3'].shift(1)
    df['stoch_d'] = stoch['STOCHd_14_3_3'].shift(1)

    # Aroon
    aroon = pta.aroon(h, l, length=14)
    df['aroon_up'] = aroon['AROONU_14'].shift(1)
    df['aroon_down'] = aroon['AROOND_14'].shift(1)

    # Vortex
    vtx = pta.vortex(h, l, c, length=14)
    df['vtx_pos'] = vtx['VTXP_14'].shift(1)
    df['vtx_neg'] = vtx['VTXM_14'].shift(1)

    # Volume-based
    df['vol_sma_20'] = v.rolling(20).mean().shift(1)
    df['vol_ratio'] = (v / v.rolling(20).mean()).shift(1)

    # Returns
    for n in [1, 3, 5, 10, 21, 63]:
        df[f'ret_{n}d'] = c.pct_change(n).shift(1)

    # Distance from EMA20 normalized by ATR — uses already-shifted ema_20 and atr_14
    df['dist_ema_20_atr'] = (df['close'].shift(1) - df['ema_20']) / df['atr_14']

    # Volatility regimes
    df['atr_pct'] = (df['atr_14'] / c.shift(1))

    # SMA crossover features
    df['ema_5_gt_ema_20'] = (df['ema_5'] > df['ema_20']).astype(int)
    df['ema_20_gt_ema_50'] = (df['ema_20'] > df['ema_50']).astype(int)
    df['close_gt_ema_50'] = (c.shift(1) > df['ema_50']).astype(int)

    # Range
    df['daily_range_atr'] = ((h - l) / df['atr_14']).shift(1)

    # Target: forward 21-day return > 0
    df['fwd_ret_21d'] = c.pct_change(21).shift(-21)  # at bar t, looking forward 21 bars
    df['y'] = (df['fwd_ret_21d'] > 0).astype(int)

    return df

FEATURE_COLS = None  # set in build_features post-call

def make_walk_forward_folds(df, train_months=24, test_months=12, step_months=12):
    start = df.index.min(); end = df.index.max()
    folds = []
    cur = start + pd.DateOffset(months=train_months)
    while True:
        ee = cur + pd.DateOffset(months=test_months)
        if ee > end - pd.DateOffset(months=1): break  # leave room for the 21-day fwd target
        folds.append({'fold': len(folds)+1,
                      'train_start': (cur - pd.DateOffset(months=train_months)).isoformat(),
                      'train_end': cur.isoformat(),
                      'oos_start': cur.isoformat(),
                      'oos_end': ee.isoformat()})
        cur = cur + pd.DateOffset(months=step_months)
    return folds

def simulate(df, signals, tp_atr, sl_atr, max_hold, side='long', slippage_bps=5.0, fee_per_share=0.0035, notional=5000):
    trades = []
    in_pos = False
    entry_bar = entry_price = entry_atr = None; qty = 0
    rows = df.reset_index().to_dict('records')
    sig = signals.reindex(df.index).fillna(False).values
    sign = 1 if side == 'long' else -1

    for i in range(len(rows) - 1):
        if not in_pos:
            if sig[i]:
                nxt = rows[i+1]
                bar = rows[i]
                if pd.isna(bar.get('atr_14')): continue
                slip = slippage_bps / 10000
                entry_price = nxt['open'] * (1 + sign*slip)
                entry_bar = i + 1; entry_atr = bar['atr_14']
                qty = int(notional / entry_price)
                if qty < 1: continue
                in_pos = True
        else:
            cur = rows[i]
            tp = entry_price + sign * tp_atr * entry_atr
            sl = entry_price - sign * sl_atr * entry_atr
            bars_held = i - entry_bar
            er, ep = None, None
            if sign == 1:
                if cur['high'] >= tp: er, ep = 'TP', tp * (1 - slippage_bps/10000)
                elif cur['low'] <= sl: er, ep = 'SL', sl * (1 - slippage_bps/10000)
            if not er and bars_held >= max_hold:
                er, ep = 'TIME', cur['close'] * (1 - sign*slippage_bps/10000)
            if er:
                pnl_per_share = (ep - entry_price) * sign
                fees = fee_per_share * qty * 2
                trades.append({
                    'entry_time': rows[entry_bar]['timestamp'], 'entry_price': entry_price,
                    'exit_time': cur['timestamp'], 'exit_price': ep, 'exit_reason': er,
                    'qty': qty, 'bars_held': bars_held, 'pnl': pnl_per_share * qty - fees,
                    'pnl_pct': pnl_per_share/entry_price,
                })
                in_pos = False; entry_bar = entry_price = None; qty = 0
    return pd.DataFrame(trades)

def compute_metrics(t):
    if len(t) == 0:
        return {'n_trades': 0, 'win_rate': None, 'profit_factor': None, 'expectancy': None,
                'total_return_pct': 0, 'max_drawdown_pct': 0}
    n = len(t); w = t[t['pnl']>0]; l = t[t['pnl']<=0]
    gw = w['pnl'].sum(); gl = abs(l['pnl'].sum())
    return {'n_trades': n, 'win_rate': len(w)/n,
            'profit_factor': gw/gl if gl>0 else float('inf'),
            'expectancy': t['pnl'].mean(), 'total_return_pct': t['pnl'].sum()/50000,
            'max_drawdown_pct': (t['pnl'].cumsum()-t['pnl'].cumsum().cummax()).min()/50000}

def to_py(o):
    if isinstance(o, dict): return {k: to_py(v) for k,v in o.items()}
    if isinstance(o, list): return [to_py(v) for v in o]
    if hasattr(o, 'item'): return o.item()
    if isinstance(o, float) and (np.isnan(o) or np.isinf(o)): return None
    return o

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ticker', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--prob-threshold', type=float, default=0.55)
    ap.add_argument('--tp-atr', type=float, default=1.5)
    ap.add_argument('--sl-atr', type=float, default=1.0)
    ap.add_argument('--max-hold', type=int, default=21)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[{args.ticker}] loading + features...")
    d = load_daily(args.ticker)
    f = build_features(d).dropna(subset=['rsi_14','atr_14','ema_200','fwd_ret_21d','y'])
    feature_cols = [c for c in f.columns
                    if c not in ['open','high','low','close','volume','fwd_ret_21d','y']
                    and pd.api.types.is_numeric_dtype(f[c])]
    print(f"  bars: {len(f):,}; features: {len(feature_cols)}")

    folds = make_walk_forward_folds(f, train_months=24, test_months=12, step_months=12)
    print(f"  folds: {len(folds)}")

    all_oos_signals = pd.Series(False, index=f.index)
    fold_summaries = []

    # Embargo: forward-21d labels mean the last 21 train bars peek into OOS. Drop them.
    LABEL_EMBARGO_DAYS = 21
    for fold in folds:
        train_end_embargoed = pd.Timestamp(fold['train_end']) - pd.tseries.offsets.BDay(LABEL_EMBARGO_DAYS)
        train = f[(f.index >= fold['train_start']) & (f.index < train_end_embargoed)]
        oos = f[(f.index >= fold['oos_start']) & (f.index < fold['oos_end'])]
        if len(train) < 50 or len(oos) < 20:
            continue
        X_tr = train[feature_cols].values; y_tr = train['y'].values
        X_oos = oos[feature_cols].values

        base = HistGradientBoostingClassifier(max_iter=100, max_depth=4, learning_rate=0.05, random_state=42)
        # calibrated wrapper
        cal = CalibratedClassifierCV(base, cv=TimeSeriesSplit(n_splits=3), method='sigmoid')
        cal.fit(X_tr, y_tr)
        oos_probs = cal.predict_proba(X_oos)[:, 1]

        signals_oos = pd.Series(oos_probs > args.prob_threshold, index=oos.index)
        all_oos_signals.loc[oos.index] = signals_oos.values
        fold_summaries.append({
            'fold': fold['fold'], 'oos_start': fold['oos_start'], 'oos_end': fold['oos_end'],
            'n_train': len(train), 'n_oos': len(oos),
            'signals_above_threshold': int(signals_oos.sum()),
            'mean_oos_prob': float(oos_probs.mean()),
        })

    # Combine signal across all OOS, run simulator on combined OOS
    oos_full = f[all_oos_signals]
    print(f"  OOS signal-true rows: {all_oos_signals.sum()}")
    trades = simulate(f, all_oos_signals, args.tp_atr, args.sl_atr, args.max_hold)
    m = compute_metrics(trades)
    print(f"  trades: {m['n_trades']}, WR={m['win_rate']}, PF={m['profit_factor']}, RET%={m['total_return_pct']}")

    trades.to_csv(f"{args.output_dir}/trades.csv", index=False)
    pd.DataFrame(fold_summaries).to_csv(f"{args.output_dir}/fold_summaries.csv", index=False)
    meta = to_py({
        'ticker': args.ticker, 'pipeline_version': 'ml_v1',
        'strategy_variant': 'ML_HGB_21d',
        'run_at': datetime.utcnow().isoformat() + 'Z',
        'features_used': feature_cols, 'n_features': len(feature_cols),
        'daily_bars': len(d), 'features_after_warmup': len(f),
        'walk_forward': {'folds': folds, 'train_months': 24, 'test_months': 12, 'step_months': 12},
        'strategy': {'name': 'ML_HGB_21d', 'side': 'long', 'tp_atr': args.tp_atr, 'sl_atr': args.sl_atr,
                     'max_hold_days': args.max_hold, 'prob_threshold': args.prob_threshold,
                     'model': 'HistGradientBoostingClassifier', 'calibration': 'CalibratedClassifierCV (sigmoid)',
                     'slippage_bps': 5.0, 'fee_per_share': 0.0035, 'notional_per_trade': 5000},
        'metrics_oos_aggregate': m, 'fold_summaries': fold_summaries,
    })
    with open(f"{args.output_dir}/run_meta.json", 'w') as fp:
        json.dump(meta, fp, indent=2, default=str)

if __name__ == '__main__':
    main()
